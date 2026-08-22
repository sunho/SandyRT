#include "MidIRMaterializer.h"

#include <bit>
#include <functional>
#include <memory>

namespace sandy::ir::mid_ir {

namespace {

Result<int64_t> infer_paged_tensor_grow_dim(const std::vector<int64_t>& dims) {
    int64_t growDim = -1;
    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] != core::Shape::kDynamic)
            continue;
        if (growDim >= 0)
            return make_error("PagedTensor input shape must have exactly one dynamic grow dimension");
        growDim = static_cast<int64_t>(i);
    }
    if (growDim < 0) {
        if (dims.empty())
            return make_error("PagedTensor input shape must have rank >= 1");
        return 0;
    }
    return growDim;
}

Result<core::DType> dtype_from_string(const std::string& dtype) {
    if (dtype == "f32") return core::DType::F32;
    if (dtype == "f16") return core::DType::F16;
    if (dtype == "bf16") return core::DType::BF16;
    if (dtype == "i32") return core::DType::I32;
    if (dtype == "i64") return core::DType::I64;
    if (dtype == "u8") return core::DType::U8;
    return make_error("unknown dtype '" + dtype + "'");
}

Result<ValueType> value_type_from_high_tensor_type(
        const high_ir::TensorType& type) {
    if (type.dtype.empty())
        return make_error("typed tensor requires dtype");
    auto dtype = dtype_from_string(type.dtype);
    if (!dtype)
        return make_error(dtype.error());
    if (type.kind == high_ir::TensorKind::PagedTensor) {
        if (type.pageSize <= 0)
            return make_error("PagedTensor requires positive page_size");
        if (!std::has_single_bit(static_cast<uint64_t>(type.pageSize)))
            return make_error("PagedTensor page_size must be a power of two");
        auto growDim = infer_paged_tensor_grow_dim(type.dims);
        if (!growDim)
            return make_error(growDim.error());
        return ValueType::paged_tensor(
            core::Shape(type.dims),
            dtype.take(),
            growDim.take(),
            type.pageSize);
    }
    return ValueType::tensor(core::Shape(type.dims), dtype.take());
}

} // namespace

MidIRMaterializer::MidIRMaterializer()
    : lowering_(BuiltinLowering::createDefault()) {
    register_all_ops();
}

Result<std::unique_ptr<Graph>> MidIRMaterializer::materialize(
        const high_ir::Graph& graph,
        const weight::Weights& weights,
        const MaterializeOptions& options) {
    auto mid_graph = std::make_unique<Graph>();
    Builder builder(*mid_graph);

    std::unordered_map<int, Value*> value_map;
    std::unordered_map<int, std::vector<Value*>> tuple_map;
    int64_t nextInputIndex = 0;

    auto createInputFromType =
        [&](int64_t inputIndex,
            int64_t tupleElement,
            const high_ir::TensorType& type) -> Result<Value*> {
            auto valueType = value_type_from_high_tensor_type(type);
            if (!valueType)
                return make_error(valueType.error());
            auto vt = valueType.take();
            if (vt.kind == ValueKind::PagedTensor) {
                return builder.createPagedTensorInput(
                    inputIndex,
                    vt.shape,
                    vt.dtype,
                    vt.growDim,
                    vt.pageSize,
                    tupleElement);
            }
            return builder.createInput(inputIndex, vt.shape, vt.dtype, tupleElement);
        };

    std::function<Result<std::vector<Value*>>(const high_ir::Value*)>
    expandTuple = [&](const high_ir::Value* value) -> Result<std::vector<Value*>> {
        auto tupleIt = tuple_map.find(value->id);
        if (tupleIt != tuple_map.end())
            return tupleIt->second;
        if (!value->def)
            return make_error("tensor tuple value has no defining op");
        const auto& def = *value->def;
        if (def.kind == high_ir::Op::TensorTupleCreate) {
            std::vector<Value*> elements;
            elements.reserve(def.operands.size());
            for (auto* operand : def.operands)
                elements.push_back(value_map.at(operand->id));
            return elements;
        }
        if (def.kind == high_ir::Op::TensorTupleAppend) {
            auto base = expandTuple(def.operands[0]);
            if (!base)
                return make_error(base.error());
            auto elements = base.take();
            elements.push_back(value_map.at(def.operands[1]->id));
            return elements;
        }
        return make_error("cannot expand tensor tuple");
    };

    for (auto& op : graph.ops()) {
        switch (op.kind) {
            case high_ir::Op::Input: {
                Value* v = nullptr;
                if (op.inputKind == high_ir::InputKind::TensorTuple) {
                    std::vector<Value*> elements;
                    elements.reserve(op.inputTensorTupleElements.size());
                    for (size_t i = 0; i < op.inputTensorTupleElements.size(); i++) {
                        auto element = createInputFromType(
                            nextInputIndex,
                            static_cast<int64_t>(i),
                            op.inputTensorTupleElements[i]);
                        if (!element)
                            return make_error("input '" + op.inputName + "': " + element.error());
                        elements.push_back(element.take());
                    }
                    tuple_map[op.results[0]->id] = std::move(elements);
                    nextInputIndex++;
                    break;
                } else if (op.inputKind == high_ir::InputKind::PagedTensor) {
                    core::DType dtype = core::DType::F32;
                    if (!op.inputPagedTensorDType.empty()) {
                        auto parsed = dtype_from_string(op.inputPagedTensorDType);
                        if (!parsed)
                            return make_error(parsed.error());
                        dtype = parsed.take();
                    } else {
                        auto it = options.input_tensor_descs.find(op.inputName);
                        if (it == options.input_tensor_descs.end())
                            return make_error("no shape provided for input '" + op.inputName + "'");
                        dtype = it->second.dtype;
                    }
                    auto growDim = infer_paged_tensor_grow_dim(op.inputPagedTensorDims);
                    if (!growDim)
                        return make_error("input '" + op.inputName + "': " + growDim.error());
                    v = builder.createPagedTensorInput(
                        nextInputIndex++,
                        core::Shape(op.inputPagedTensorDims),
                        dtype,
                        growDim.take(),
                        op.inputPagedTensorPageSize);
                } else {
                    if (!op.inputTensorDType.empty()) {
                        auto dtype = dtype_from_string(op.inputTensorDType);
                        if (!dtype)
                            return make_error(dtype.error());
                        v = builder.createInput(
                            nextInputIndex++,
                            core::Shape(op.inputTensorDims),
                            dtype.take());
                    } else {
                        auto it = options.input_tensor_descs.find(op.inputName);
                        if (it == options.input_tensor_descs.end())
                            return make_error("no shape provided for input '" + op.inputName + "'");
                        v = builder.createInput(nextInputIndex++, it->second.shape, it->second.dtype);
                    }
                }
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::Weight: {
                auto tensor = weights.get_tensor(op.weightName);
                if (!tensor)
                    return make_error("weight not found: " + op.weightName);
                const auto& desc = tensor->desc();
                auto* v = builder.createWeight(op.weightName, desc.shape, desc.dtype);
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::TensorTupleCreate: {
                std::vector<Value*> elements;
                elements.reserve(op.operands.size());
                for (auto* operand : op.operands)
                    elements.push_back(value_map.at(operand->id));
                auto* v = builder.createTensorTupleCreate(elements);
                value_map[op.results[0]->id] = v;
                tuple_map[op.results[0]->id] = std::move(elements);
                break;
            }
            case high_ir::Op::TensorTupleAppend: {
                auto elements = expandTuple(op.results[0]);
                if (!elements)
                    return make_error(elements.error());
                tuple_map[op.results[0]->id] = elements.take();
                auto* v = builder.createTensorTupleCreate(tuple_map[op.results[0]->id]);
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::TensorTupleGet: {
                auto elements = expandTuple(op.operands[0]);
                if (!elements)
                    return make_error(elements.error());
                auto expanded = elements.take();
                if (op.tupleIndex < 0 ||
                    static_cast<size_t>(op.tupleIndex) >= expanded.size()) {
                    return make_error("tensor tuple index out of range");
                }
                value_map[op.results[0]->id] =
                    expanded[static_cast<size_t>(op.tupleIndex)];
                break;
            }
            case high_ir::Op::Builtin: {
                std::vector<Value*> operands;
                for (auto* hv : op.operands)
                    operands.push_back(value_map.at(hv->id));

                AttrMap attrs;
                for (auto& a : op.attrs) {
                    switch (a.type) {
                        case high_ir::Type::Int:
                            attrs[a.name] = AttrValue::make_int(a.intVal);
                            break;
                        case high_ir::Type::Float:
                            attrs[a.name] = AttrValue::make_float(a.floatVal);
                            break;
                        case high_ir::Type::DType:
                            attrs[a.name] = AttrValue::make_dtype(a.dtypeVal);
                            break;
                        case high_ir::Type::String:
                            attrs[a.name] = AttrValue::make_string(a.strVal);
                            break;
                        case high_ir::Type::IntList:
                            attrs[a.name] = AttrValue::make_int_list(a.intListVal);
                            break;
                        default:
                            break;
                    }
                }

                auto* fn = lowering_.lookup(op.name);
                if (!fn)
                    return make_error("no lowering for builtin '" + op.name + "'");

                auto results = (*fn)(builder, operands, attrs, static_cast<int>(op.results.size()));
                if (!results)
                    return make_error(results.error());
                auto midResults = results.take();
                if (midResults.size() != op.results.size()) {
                    return make_error("lowering for builtin '" + op.name +
                                      "' returned wrong number of results");
                }
                for (size_t i = 0; i < midResults.size(); i++)
                    value_map[op.results[i]->id] = midResults[i];
                break;
            }
            case high_ir::Op::IntConst: {
                auto* v = builder.createUntypedConstantF32(static_cast<float>(op.intVal));
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::FloatConst: {
                auto* v = builder.createUntypedConstantF32(static_cast<float>(op.floatVal));
                value_map[op.results[0]->id] = v;
                break;
            }
            case high_ir::Op::StringConst:
                return make_error("unexpected const op in materialization");
        }
    }

    std::vector<Value*> outputs;
    for (auto* hv : graph.outputs()) {
        if (hv->type == high_ir::Type::TensorTuple &&
            value_map.find(hv->id) == value_map.end()) {
            auto elements = expandTuple(hv);
            if (!elements)
                return make_error(elements.error());
            auto expanded = elements.take();
            auto* tuple = builder.createTensorTupleCreate(expanded);
            value_map[hv->id] = tuple;
        }
        outputs.push_back(value_map.at(hv->id));
    }
    builder.setOutputs(outputs);

    return mid_graph;
}

} // namespace sandy::ir::mid_ir
