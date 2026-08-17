#include "Compiler.h"
#include "HighIR.h"
#include "TensorBuffer.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

class TestTensorBuffer final : public sandy::core::TensorBuffer {
public:
    explicit TestTensorBuffer(sandy::core::TensorDesc desc)
        : TensorBuffer(std::move(desc)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

class TestWeights final : public sandy::weight::Weights {
public:
    void add(sandy::core::TensorDesc desc) {
        auto name = desc.name;
        tensors_[name] = std::make_shared<TestTensorBuffer>(std::move(desc));
    }

    std::vector<sandy::core::TensorDesc> descriptors() const override {
        std::vector<sandy::core::TensorDesc> result;
        for (const auto& [_, tensor] : tensors_) {
            result.push_back(tensor->desc());
        }
        return result;
    }

    std::shared_ptr<sandy::core::TensorBuffer> get_tensor(
            const std::string& name) const override {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return nullptr;
        return it->second;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<TestTensorBuffer>> tensors_;
};

fs::path makeTempDir() {
    std::error_code ec;
    fs::path base = fs::temp_directory_path(ec);
    EXPECT_FALSE(ec) << ec.message();

    auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path dir = base / ("sandy_import_test_" + std::to_string(id));
    fs::create_directories(dir / "nested", ec);
    EXPECT_FALSE(ec) << ec.message();
    return dir;
}

void writeFile(const fs::path& path, const std::string& contents) {
    std::ofstream file(path);
    ASSERT_TRUE(file.good()) << path;
    file << contents;
    ASSERT_TRUE(file.good()) << path;
}

} // namespace

TEST(CompilerTest, LoadSandyGoImportsRelativeFile) {
    fs::path dir = makeTempDir();

    writeFile(dir / "nested" / "layers.sandy.go", R"(
func imported(x Tensor) Tensor {
    return __relu(x)
}
)");

    writeFile(dir / "main.sandy.go", R"(
import "nested/layers.sandy.go"

func main(x Tensor) Tensor {
    return imported(x)
}
)");

    sandy::Compiler compiler;
    auto graph = compiler.load_sandygo((dir / "main.sandy.go").string());

    ASSERT_EQ(graph.outputs().size(), 1u);
    ASSERT_NE(graph.outputs()[0], nullptr);
    ASSERT_NE(graph.outputs()[0]->def, nullptr);
    EXPECT_EQ(graph.outputs()[0]->def->kind, sandy::ir::high_ir::Op::Builtin);
    EXPECT_EQ(graph.outputs()[0]->def->name, "relu");

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, PagedTensorMainParamsMaterializeToMidIRInputs) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(k PagedTensor[[2, -1, 128], page_size=16], v PagedTensor[[2, -1, 128], page_size=16]) Tensor {
    return k
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    ASSERT_EQ(highGraph.outputs().size(), 1u);
    ASSERT_NE(highGraph.outputs()[0], nullptr);
    ASSERT_NE(highGraph.outputs()[0]->def, nullptr);
    EXPECT_EQ(highGraph.outputs()[0]->def->kind, sandy::ir::high_ir::Op::Input);
    EXPECT_EQ(highGraph.outputs()[0]->def->inputKind, sandy::ir::high_ir::InputKind::PagedTensor);
    EXPECT_EQ(highGraph.outputs()[0]->def->inputPagedTensorDims,
              (std::vector<int64_t>{2, -1, 128}));
    EXPECT_EQ(highGraph.outputs()[0]->def->inputPagedTensorPageSize, 16);

    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["k"] = sandy::core::TensorDesc(
        sandy::core::Shape({}), sandy::core::DType::BF16);
    options.input_tensor_descs["v"] = sandy::core::TensorDesc(
        sandy::core::Shape({}), sandy::core::DType::BF16);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    auto* cache = midGraph->outputs()[0];
    ASSERT_NE(cache, nullptr);
    ASSERT_NE(cache->def, nullptr);
    EXPECT_EQ(cache->def->kind, sandy::ir::mid_ir::OpKind::PagedTensorInput);
    EXPECT_EQ(cache->shape, sandy::core::Shape({2, -1, 128}));
    EXPECT_EQ(cache->dtype, sandy::core::DType::BF16);
    EXPECT_EQ(cache->def->attrs.at("index").intVal, 0);
    EXPECT_EQ(cache->def->attrs.at("grow_dim").intVal, 1);
    EXPECT_EQ(cache->def->attrs.at("page_size").intVal, 16);
    EXPECT_EQ(cache->def->attrs.at("dims").intListVal,
              (std::vector<int64_t>{2, -1, 128}));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, FixedPagedTensorTupleInputMaterializesToElementInputsAndTupleOutput) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(k [2]PagedTensor[[128], bf16, page_size=16]) []Tensor {
    var out []Tensor
    out = append(out, k[0])
    out = append(out, k[1])
    return out
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    auto result = compiler.materialize_mid_ir(highGraph, weights, {});
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    auto* tuple = midGraph->outputs()[0];
    ASSERT_NE(tuple, nullptr);
    ASSERT_NE(tuple->def, nullptr);
    EXPECT_EQ(tuple->kind, sandy::ir::mid_ir::ValueKind::TensorTuple);
    EXPECT_EQ(tuple->def->kind, sandy::ir::mid_ir::OpKind::TensorTupleCreate);
    ASSERT_EQ(tuple->def->operands.size(), 2u);

    for (size_t i = 0; i < tuple->def->operands.size(); i++) {
        auto* element = tuple->def->operands[i];
        ASSERT_NE(element->def, nullptr);
        EXPECT_EQ(element->def->kind, sandy::ir::mid_ir::OpKind::PagedTensorInput);
        EXPECT_EQ(element->shape, sandy::core::Shape({128}));
        EXPECT_EQ(element->dtype, sandy::core::DType::BF16);
        EXPECT_EQ(element->def->attrs.at("index").intVal, 0);
        EXPECT_EQ(element->def->attrs.at("tuple_element").intVal,
                  static_cast<int64_t>(i));
        EXPECT_EQ(element->def->attrs.at("page_size").intVal, 16);
    }

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, FixedTensorTupleInputIndexMaterializesToElementInput) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(k [2]Tensor[[128], bf16]) Tensor {
    return k[0]
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    auto result = compiler.materialize_mid_ir(highGraph, weights, {});
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    auto* output = midGraph->outputs()[0];
    ASSERT_NE(output, nullptr);
    ASSERT_NE(output->def, nullptr);
    EXPECT_EQ(output->def->kind, sandy::ir::mid_ir::OpKind::Input);
    EXPECT_EQ(output->shape, sandy::core::Shape({128}));
    EXPECT_EQ(output->dtype, sandy::core::DType::BF16);
    EXPECT_EQ(output->def->attrs.at("index").intVal, 0);
    EXPECT_EQ(output->def->attrs.at("tuple_element").intVal, 0);

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, RMSNormBuiltinMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Tensor) Tensor {
    return __rms_norm(x, @norm.weight)
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    weights.add(sandy::core::TensorDesc(
        "norm.weight", sandy::core::Shape({3}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::RMSNorm);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, RMSNormBuiltinWithoutScaleMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Tensor) Tensor {
    return __rms_norm(x)
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::RMSNorm);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));
    ASSERT_EQ(midGraph->outputs()[0]->def->operands.size(), 1u);

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, LayerNormBuiltinMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Tensor) Tensor {
    return __layer_norm(x, @ln.weight, @ln.bias, epsilon=0.00001)
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    weights.add(sandy::core::TensorDesc(
        "ln.weight", sandy::core::Shape({3}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc(
        "ln.bias", sandy::core::Shape({3}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 4, 3}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::LayerNorm);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 4, 3}));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, AddMulSqrtBuiltinsMaterializeToMidIROps) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Tensor, y Tensor, z Tensor) Tensor {
    return __sqrt(__mul(__add(x, y), z))
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    options.input_tensor_descs["y"] = sandy::core::TensorDesc(
        sandy::core::Shape({3}), sandy::core::DType::F32);
    options.input_tensor_descs["z"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 1}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    auto* entry = midGraph->entry();
    ASSERT_GE(entry->ops.size(), 3u);
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_EQ(entry->ops[static_cast<size_t>(i)]->kind, sandy::ir::mid_ir::OpKind::Input);
        EXPECT_EQ(entry->ops[static_cast<size_t>(i)]->attrs.at("index").intVal, i);
        EXPECT_EQ(entry->ops[static_cast<size_t>(i)]->attrs.count("name"), 0u);
    }

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Sqrt);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, GeluBuiltinLowersThroughConstantsAndTanh) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Tensor) Tensor {
    return __gelu(x)
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Mul);

    bool sawConstant = false;
    bool sawTanh = false;
    for (auto* op : midGraph->entry()->ops) {
        sawConstant = sawConstant || op->kind == sandy::ir::mid_ir::OpKind::Constant;
        sawTanh = sawTanh || op->kind == sandy::ir::mid_ir::OpKind::Tanh;
    }
    EXPECT_TRUE(sawConstant);
    EXPECT_TRUE(sawTanh);

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, SoftcapBuiltinLowersThroughConstantsAndTanh) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto softcap = highGraph.addBuiltin(
        "softcap",
        {x},
        {sandy::ir::high_ir::Attr::fromFloat("cap", 30.0)},
        1);
    highGraph.setOutputs({softcap[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Mul);

    bool sawConstant = false;
    bool sawTanh = false;
    for (auto* op : midGraph->entry()->ops) {
        sawConstant = sawConstant || op->kind == sandy::ir::mid_ir::OpKind::Constant;
        sawTanh = sawTanh || op->kind == sandy::ir::mid_ir::OpKind::Tanh;
    }
    EXPECT_TRUE(sawConstant);
    EXPECT_TRUE(sawTanh);
}

TEST(CompilerTest, SoftcapBuiltinAcceptsPositionalConstantCap) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* cap = highGraph.addFloatConst(30.0);
    auto softcap = highGraph.addBuiltin("softcap", {x, cap}, {}, 1);
    highGraph.setOutputs({softcap[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::BF16);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));
    EXPECT_EQ(midGraph->outputs()[0]->dtype, sandy::core::DType::BF16);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Mul);
}

TEST(CompilerTest, GemmaStyleMatMulWithTransposeMaterializesToMidIROps) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Tensor) Tensor {
    return __matmul(x, __transpose(@embed_tokens.weight))
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    weights.add(sandy::core::TensorDesc(
        "embed_tokens.weight", sandy::core::Shape({5, 3}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::MatMul);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 5}));
    ASSERT_TRUE(midGraph->outputs()[0]->def->attrs.contains("transpose_rhs"));
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("transpose_rhs").intVal, 1);

    int transposeCount = 0;
    for (auto* op : midGraph->entry()->ops) {
        if (op->kind == sandy::ir::mid_ir::OpKind::Transpose)
            transposeCount++;
    }
    EXPECT_EQ(transposeCount, 0);

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, EmbeddingBuiltinMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(input_ids Tensor) Tensor {
    return __embedding(input_ids, @embed_tokens.weight)
}
)");

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo((dir / "main.sandy.go").string());

    TestWeights weights;
    weights.add(sandy::core::TensorDesc(
        "embed_tokens.weight", sandy::core::Shape({5, 3}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["input_ids"] = sandy::core::TensorDesc(
        sandy::core::Shape({2}), sandy::core::DType::I32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Embedding);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, ProgrammaticReshapePermuteMaterializesToMidIROps) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto reshaped = highGraph.addBuiltin(
        "reshape",
        {x},
        {sandy::ir::high_ir::Attr::fromIntList("shape", {1, 2, 3, 2})},
        1);
    auto permuted = highGraph.addBuiltin(
        "permute",
        {reshaped[0]},
        {sandy::ir::high_ir::Attr::fromIntList("dims", {0, 2, 1, 3})},
        1);
    highGraph.setOutputs({permuted[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({1, 2, 6}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Permute);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({1, 3, 2, 2}));
    ASSERT_EQ(midGraph->outputs()[0]->def->operands.size(), 1u);
    EXPECT_EQ(midGraph->outputs()[0]->def->operands[0]->def->kind,
              sandy::ir::mid_ir::OpKind::Reshape);
}

TEST(CompilerTest, ProgrammaticSlidingQueryKeyScoreMaterializesToMidIROp) {
    sandy::ir::high_ir::Graph highGraph;
    auto* q = highGraph.addInput("q");
    auto* k = highGraph.addInput("k");
    auto score = highGraph.addBuiltin(
        "sliding_query_key_score",
        {q, k},
        {sandy::ir::high_ir::Attr::fromInt("window", 2)},
        1);
    highGraph.setOutputs({score[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["q"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 4, 3, 8}), sandy::core::DType::F32);
    options.input_tensor_descs["k"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 2, 5, 8}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind,
              sandy::ir::mid_ir::OpKind::SlidingQueryKeyScore);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 4, 3, 5}));
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("window").intVal, 2);
}

TEST(CompilerTest, ProgrammaticSoftmaxMaterializesToMidIROp) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto softmax = highGraph.addBuiltin(
        "softmax",
        {x},
        {sandy::ir::high_ir::Attr::fromInt("dim", -1)},
        1);
    highGraph.setOutputs({softmax[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 4, 3, 5}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Softmax);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 4, 3, 5}));
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("dim").intVal, -1);
}

TEST(CompilerTest, ProgrammaticRoPEMaterializesToMidIROpWithTheta) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto rope = highGraph.addBuiltin(
        "rope",
        {x},
        {sandy::ir::high_ir::Attr::fromFloat("rope_theta", 10000.0)},
        1);
    highGraph.setOutputs({rope[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, 4, 6}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::RoPE);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3, 4, 6}));
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("rope_theta").floatVal, 10000.0);
}

TEST(CompilerTest, ProgrammaticRoPEMaterializesRotaryDimAttr) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto rope = highGraph.addBuiltin(
        "rope",
        {x},
        {
            sandy::ir::high_ir::Attr::fromFloat("rope_theta", 1000000.0),
            sandy::ir::high_ir::Attr::fromInt("rotary_dim", 128),
        },
        1);
    highGraph.setOutputs({rope[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 8, 16, 512}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::RoPE);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 8, 16, 512}));
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("rope_theta").floatVal, 1000000.0);
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("rotary_dim").intVal, 128);
}

TEST(CompilerTest, ProgrammaticRoPEMaterializesRuntimePositionIdsOperand) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* positionIds = highGraph.addInput("position_ids");
    auto rope = highGraph.addBuiltin(
        "rope",
        {x, positionIds},
        {
            sandy::ir::high_ir::Attr::fromFloat("rope_theta", 10000.0),
            sandy::ir::high_ir::Attr::fromInt("split_half", 1),
        },
        1);
    highGraph.setOutputs({rope[0]});

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({1, 8, 1, 256}), sandy::core::DType::F32);
    options.input_tensor_descs["position_ids"] = sandy::core::TensorDesc(
        sandy::core::Shape({1}), sandy::core::DType::I64);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::RoPE);
    EXPECT_EQ(midGraph->outputs()[0]->def->operands.size(), 2u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({1, 8, 1, 256}));
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("rope_theta").floatVal, 10000.0);
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("split_half").intVal, 1);
}

TEST(CompilerTest, ProgrammaticAttentionMaterializesBatchedQKVToSingleMidIROp) {
    sandy::ir::high_ir::Graph highGraph;
    auto* q = highGraph.addInput("q");
    auto* k = highGraph.addInput("k");
    auto* v = highGraph.addInput("v");
    auto results = highGraph.addBuiltin(
        "attention",
        {q, k, v},
        {
            sandy::ir::high_ir::Attr::fromInt("window", 2),
            sandy::ir::high_ir::Attr::fromFloat("scale", 1.0),
        },
        1);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["q"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 8, 3, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["k"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["v"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 8, 3, 4}));
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Attention);
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("window").intVal, 2);
    EXPECT_EQ(midGraph->outputs()[0]->def->attrs.at("scale").floatVal, 1.0);
}

TEST(CompilerTest, ProgrammaticAttentionMaterializesPositionOffsetsOperand) {
    sandy::ir::high_ir::Graph highGraph;
    auto* q = highGraph.addInput("q");
    auto* k = highGraph.addInput("k");
    auto* v = highGraph.addInput("v");
    auto* positionOffsets = highGraph.addInput("position_offsets");
    auto results = highGraph.addBuiltin(
        "attention",
        {q, k, v, positionOffsets},
        {
            sandy::ir::high_ir::Attr::fromInt("window", 2),
        },
        1);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["q"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 8, 1, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["k"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["v"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 2, 5, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["position_offsets"] = sandy::core::TensorDesc(
        sandy::core::Shape({2}), sandy::core::DType::I64);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Attention);
    ASSERT_EQ(midGraph->outputs()[0]->def->operands.size(), 4u);
    EXPECT_EQ(midGraph->outputs()[0]->def->operands[3]->shape, sandy::core::Shape({2}));
}

TEST(CompilerTest, KVAttentionBuiltinIsRemoved) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto results = highGraph.addBuiltin(
        "kv_attention",
        {x},
        {},
        1);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({1}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("kv_attention"), std::string::npos);
}
