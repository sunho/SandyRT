#include "Compiler.h"
#include "CpuDevice.h"
#include "Engine.h"
#include "Interpreter.h"
#include "Lexer.h"
#include "Parser.h"
#include "SafeTensorWeights.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

float read_bf16(std::span<const uint8_t> data, size_t index) {
    sandy::core::BFloat16 value = sandy::core::bfloat16_from_bits(0);
    std::memcpy(&value, data.data() + index * sizeof(sandy::core::BFloat16), sizeof(value));
    return sandy::core::bfloat16_to_float(value);
}

float read_float_value(std::span<const uint8_t> data, sandy::core::DType dtype, size_t index) {
    switch (dtype) {
        case sandy::core::DType::F32:
            return read_f32(data, index);
        case sandy::core::DType::BF16:
            return read_bf16(data, index);
        default:
            return 0.0f;
    }
}

void add_tensors_to_map(const sandy::weight::Weights& tensors,
                        sandy::engine::TensorMap& map) {
    for (const auto& desc : tensors.descriptors()) {
        auto tensor = tensors.get_tensor(desc.name);
        if (!tensor) {
            fprintf(stderr, "tensor listed in descriptors but missing: %s\n", desc.name.c_str());
            abort();
        }
        map[desc.name] = tensor;
    }
}

void add_inputs_to_positional_map(const sandy::weight::Weights& tensors,
                                  const sandy::ir::high_ir::Graph& graph,
                                  std::vector<sandy::engine::TensorBufferPtr>& inputs) {
    for (const auto& op : graph.ops()) {
        if (op.kind != sandy::ir::high_ir::Op::Input)
            continue;
        auto tensor = tensors.get_tensor(op.inputName);
        if (!tensor) {
            fprintf(stderr, "input tensor missing: %s\n", op.inputName.c_str());
            abort();
        }
        inputs.push_back(tensor);
    }
}

void indent(int depth) {
    for (int i = 0; i < depth; i++) std::cout << "  ";
}

void dump_type(const sandy::sandygo::TypeExpr& t) {
    if (t.kind == sandy::sandygo::TypeExpr::Slice) {
        std::cout << "[]" << t.name;
    } else {
        std::cout << t.name;
    }
}

void dump_expr(const sandy::sandygo::Expr& e, int depth) {
    indent(depth);
    switch (e.kind) {
        case sandy::sandygo::Expr::Ident:
            std::cout << "Ident(" << e.sval << ")\n";
            break;
        case sandy::sandygo::Expr::IntLit:
            std::cout << "Int(" << e.ival << ")\n";
            break;
        case sandy::sandygo::Expr::FloatLit:
            std::cout << "Float(" << e.fval << ")\n";
            break;
        case sandy::sandygo::Expr::StringLit:
            std::cout << "String(\"" << e.sval << "\")\n";
            break;
        case sandy::sandygo::Expr::WeightLit:
            std::cout << "Weight(@" << e.sval << ")\n";
            break;
        case sandy::sandygo::Expr::IntListLit:
            std::cout << "IntList[";
            for (size_t i = 0; i < e.intListVal.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << e.intListVal[i];
            }
            std::cout << "]\n";
            break;
        case sandy::sandygo::Expr::Binary:
            std::cout << "Binary(" << e.op << ")\n";
            dump_expr(*e.left, depth + 1);
            dump_expr(*e.right, depth + 1);
            break;
        case sandy::sandygo::Expr::Unary:
            std::cout << "Unary(" << e.op << ")\n";
            dump_expr(*e.left, depth + 1);
            break;
        case sandy::sandygo::Expr::Call:
            std::cout << "Call\n";
            indent(depth + 1);
            std::cout << "callee:\n";
            dump_expr(*e.left, depth + 2);
            if (!e.args.empty()) {
                indent(depth + 1);
                std::cout << "args:\n";
                for (auto& arg : e.args) dump_expr(*arg, depth + 2);
            }
            if (!e.namedArgs.empty()) {
                indent(depth + 1);
                std::cout << "named:\n";
                for (auto& named : e.namedArgs) {
                    indent(depth + 2);
                    std::cout << named.name << " =\n";
                    dump_expr(*named.value, depth + 3);
                }
            }
            break;
        case sandy::sandygo::Expr::Index:
            std::cout << "Index\n";
            dump_expr(*e.left, depth + 1);
            indent(depth + 1);
            std::cout << "[\n";
            dump_expr(*e.right, depth + 2);
            indent(depth + 1);
            std::cout << "]\n";
            break;
    }
}

void dump_stmt(const sandy::sandygo::Stmt& s, int depth);

void dump_block(const std::vector<sandy::sandygo::StmtPtr>& stmts, int depth) {
    for (auto& stmt : stmts) dump_stmt(*stmt, depth);
}

void dump_stmt(const sandy::sandygo::Stmt& s, int depth) {
    indent(depth);
    switch (s.kind) {
        case sandy::sandygo::Stmt::Assign:
            std::cout << (s.isDecl ? "ShortVarDecl\n" : "Assign\n");
            indent(depth + 1);
            std::cout << "targets:\n";
            for (auto& target : s.targets) dump_expr(*target, depth + 2);
            indent(depth + 1);
            std::cout << "value:\n";
            dump_expr(*s.value, depth + 2);
            break;
        case sandy::sandygo::Stmt::VarDecl:
            std::cout << "VarDecl " << s.name << " ";
            dump_type(s.type);
            std::cout << "\n";
            break;
        case sandy::sandygo::Stmt::Return:
            std::cout << "Return\n";
            for (auto& value : s.values) dump_expr(*value, depth + 1);
            break;
        case sandy::sandygo::Stmt::For:
            std::cout << "For " << s.name << " :=\n";
            indent(depth + 1);
            std::cout << "iter:\n";
            dump_expr(*s.iterExpr, depth + 2);
            indent(depth + 1);
            std::cout << "body:\n";
            dump_block(s.stmts, depth + 2);
            break;
        case sandy::sandygo::Stmt::If:
            std::cout << "If\n";
            indent(depth + 1);
            std::cout << "cond:\n";
            dump_expr(*s.cond, depth + 2);
            indent(depth + 1);
            std::cout << "then:\n";
            dump_block(s.stmts, depth + 2);
            if (!s.elseStmts.empty()) {
                indent(depth + 1);
                std::cout << "else:\n";
                dump_block(s.elseStmts, depth + 2);
            }
            break;
        case sandy::sandygo::Stmt::WeightScope:
            std::cout << "WeightScope\n";
            indent(depth + 1);
            std::cout << "scope:\n";
            dump_expr(*s.scopeExpr, depth + 2);
            indent(depth + 1);
            std::cout << "body:\n";
            dump_block(s.stmts, depth + 2);
            break;
        case sandy::sandygo::Stmt::ExprStmt:
            std::cout << "ExprStmt\n";
            dump_expr(*s.expr, depth + 1);
            break;
    }
}

void dump_program(const sandy::sandygo::Program& program) {
    for (const auto& fn : program.funcs) {
        std::cout << "Func " << fn.name << "(";
        for (size_t i = 0; i < fn.params.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << fn.params[i].name << " ";
            dump_type(fn.params[i].type);
        }
        std::cout << ")";
        if (!fn.returnTypes.empty()) {
            std::cout << " -> (";
            for (size_t i = 0; i < fn.returnTypes.size(); i++) {
                if (i > 0) std::cout << ", ";
                dump_type(fn.returnTypes[i]);
            }
            std::cout << ")";
        }
        std::cout << "\n";
        dump_block(fn.body, 1);
        std::cout << "\n";
    }
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream file(path);
    if (!file) return false;
    std::ostringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return true;
}

struct ProfileStat {
    int64_t count = 0;
    double totalMs = 0.0;
    double maxMs = 0.0;
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_tensor(const std::string& name, sandy::core::TensorBuffer& buffer) {
    auto accessResult = buffer.access();
    if (!accessResult) {
        fprintf(stderr, "cannot access output buffer %s: %s\n", name.c_str(), accessResult.error().c_str());
        return;
    }
    auto access = accessResult.take();
    const auto& desc = access.desc();
    printf("%s: %s%s\n",
           name.c_str(),
           sandy::core::dtype_name(desc.dtype),
           desc.shape.str().c_str());

    if (desc.dtype != sandy::core::DType::F32 && desc.dtype != sandy::core::DType::BF16) {
        printf("  raw bytes: %zu\n", access.data().size());
        return;
    }

    auto numel = desc.shape.numel();
    if (numel < 0) {
        printf("  dynamic output shape\n");
        return;
    }

    if (numel <= 64) {
        printf("  [");
        for (int64_t i = 0; i < numel; i++) {
            if (i > 0) printf(", ");
            printf("%.6g", read_float_value(access.data(), desc.dtype, static_cast<size_t>(i)));
        }
        printf("]\n");
    } else {
        printf("  raw bytes: %zu\n", access.data().size());
    }

    if (numel > 0) {
        int rank = desc.shape.rank();
        int64_t vocab = rank > 0 ? desc.shape.dim(rank - 1) : numel;
        if (vocab <= 0) {
            printf("  dynamic output shape\n");
            return;
        }
        int64_t rows = numel / vocab;
        for (int64_t row = 0; row < rows; row++) {
            constexpr int64_t kTopK = 5;
            int64_t topIds[kTopK];
            float topValues[kTopK];
            for (int64_t i = 0; i < kTopK; i++) {
                topIds[i] = -1;
                topValues[i] = -std::numeric_limits<float>::infinity();
            }

            size_t rowBase = static_cast<size_t>(row * vocab);
            for (int64_t i = 0; i < vocab; i++) {
                float value = read_float_value(
                    access.data(), desc.dtype, rowBase + static_cast<size_t>(i));
                for (int64_t slot = 0; slot < kTopK; slot++) {
                    if (value > topValues[slot]) {
                        for (int64_t move = kTopK - 1; move > slot; move--) {
                            topIds[move] = topIds[move - 1];
                            topValues[move] = topValues[move - 1];
                        }
                        topIds[slot] = i;
                        topValues[slot] = value;
                        break;
                    }
                }
            }
            printf("  argmax[%lld]: %lld (%.6g)\n",
                   static_cast<long long>(row),
                   static_cast<long long>(topIds[0]),
                   topValues[0]);
            printf("  top5[%lld]:", static_cast<long long>(row));
            for (int64_t i = 0; i < kTopK; i++) {
                printf(" %lld(%.6g)",
                       static_cast<long long>(topIds[i]),
                       topValues[i]);
            }
            printf("\n");
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    bool instrument = false;
    int arg = 1;
    if (argc >= 2 && std::strcmp(argv[1], "--instrument") == 0) {
        instrument = true;
        arg = 2;
    }
    if (argc - arg != 3) {
        fprintf(stderr, "usage: cpu_runner [--instrument] <program.sandy.go> <weights.safetensors> <inputs.safetensors>\n");
        return 1;
    }

    const char* programPath = argv[arg];
    const char* weightsPath = argv[arg + 1];
    const char* inputsPath = argv[arg + 2];
    auto totalStart = Clock::now();
    auto stageStart = totalStart;
    auto printStage = [&](const char* name) {
        if (!instrument)
            return;
        auto now = Clock::now();
        printf("[stage] %s time_ms=%.3f\n", name, elapsed_ms(stageStart, now));
        stageStart = now;
    };

    printf("[1/8] reading sandy go: %s\n", programPath);
    std::string source;
    if (!read_file(programPath, source)) {
        fprintf(stderr, "cannot open %s\n", programPath);
        return 1;
    }
    printStage("read_sandygo");

    printf("[2/8] lexing/parsing sandy go\n");
    sandy::sandygo::Lexer lexer(source);
    auto tokens = lexer.tokenize();
    if (lexer.hasError()) {
        fprintf(stderr, "lexer error: %s\n", lexer.errorMessage().c_str());
        return 1;
    }

    sandy::sandygo::Parser parser(std::move(tokens));
    auto program = parser.parse();
    if (parser.hasError()) {
        fprintf(stderr, "parser error: %s\n", parser.errorMessage().c_str());
        return 1;
    }
    printStage("parse_sandygo");

    printf("sandy go ast:\n");
    dump_program(program);
    std::cout << std::flush;
    printStage("dump_sandygo_ast");

    printf("[3/8] lowering sandy go ast -> high ir\n");
    sandy::ir::high_ir::Graph highGraph;
    sandy::sandygo::Interpreter interp(program, highGraph);
    interp.interpret();
    printStage("lower_high_ir");
    printf("high ir:\n");
    highGraph.dump();
    printStage("dump_high_ir");

    sandy::Compiler compiler;

    printf("[4/8] loading weights: %s\n", weightsPath);
    auto weights = sandy::weight::EagerSafeTensorWeights::load(weightsPath);
    printStage("load_weights");

    printf("[5/8] loading inputs: %s\n", inputsPath);
    auto inputs = sandy::weight::EagerSafeTensorWeights::load(inputsPath);
    printStage("load_inputs");

    sandy::ir::mid_ir::MaterializeOptions options;
    for (const auto& desc : inputs->descriptors())
        options.input_tensor_descs[desc.name] = desc;
    printStage("prepare_materialize_options");

    printf("[6/8] materializing mid ir\n");
    auto midResult = compiler.materialize_mid_ir(highGraph, *weights, options);
    if (!midResult) {
        fprintf(stderr, "materialize error: %s\n", midResult.error().c_str());
        return 1;
    }
    auto midGraph = midResult.take();
    printStage("materialize_mid_ir");
    midGraph->dump();
    printStage("dump_mid_ir");

    printf("[7/8] compiling cpu invocation plan\n");
    std::vector<std::unique_ptr<sandy::engine::Device>> devices;
    devices.push_back(std::make_unique<sandy::engine::CpuDevice>());
    sandy::engine::Engine engine(std::move(devices));
    auto graphResult = engine.compile(*midGraph);
    if (!graphResult) {
        fprintf(stderr, "compile error: %s\n", graphResult.error().c_str());
        return 1;
    }
    auto compiledGraph = graphResult.take();
    printStage("compile_kernel_ir_graph");

    std::vector<sandy::engine::TensorBufferPtr> inputBuffers;
    sandy::engine::TensorMap weightMap;
    add_inputs_to_positional_map(*inputs, highGraph, inputBuffers);
    add_tensors_to_map(*weights, weightMap);
    printStage("prepare_runtime_maps");

    printf("[8/8] KernelIR graph runs\n");
    sandy::engine::EngineRunOptions runOptions;
    std::unordered_map<int, ProfileStat> profileStats;
    int64_t profileKernelCount = 0;
    double profileTotalMs = 0.0;
    if (instrument) {
        runOptions.profileKernel = [&](const sandy::engine::InvocProfileEvent& event) {
            profileKernelCount++;
            profileTotalMs += event.elapsedMs;
            auto key = static_cast<int>(event.opKind);
            auto& stat = profileStats[key];
            stat.count++;
            stat.totalMs += event.elapsedMs;
            if (event.elapsedMs > stat.maxMs)
                stat.maxMs = event.elapsedMs;
            printf("[profile] op_index=%zu op_id=%u device=%u kind=%s inputs=%zu outputs=%zu time_ms=%.3f\n",
                   event.opIndex,
                   event.op,
                   event.device,
                   sandy::ir::kernel_ir::op_kind_name(event.opKind),
                   event.inputCount,
                   event.outputCount,
                   event.elapsedMs);
        };
    }

    auto engineRunStart = Clock::now();
    auto runResult = engine.run(
        *compiledGraph,
        inputBuffers,
        weightMap,
        instrument ? &runOptions : nullptr);
    auto engineRunEnd = Clock::now();
    if (!runResult) {
        fprintf(stderr, "run error: %s\n", runResult.error().c_str());
        return 1;
    }
    printStage("engine_run");
    if (instrument) {
        printf("[stage] engine_run_wall time_ms=%.3f\n", elapsed_ms(engineRunStart, engineRunEnd));
        printf("[profile] total kernels=%lld time_ms=%.3f\n",
               static_cast<long long>(profileKernelCount),
               profileTotalMs);
        printf("[profile] by op:\n");
        for (int kind = 0; kind < static_cast<int>(sandy::ir::mid_ir::OpKind::NUM_KINDS); kind++) {
            auto it = profileStats.find(kind);
            if (it == profileStats.end())
                continue;
            auto opKind = static_cast<sandy::ir::mid_ir::OpKind>(kind);
            const auto& stat = it->second;
            double avgMs = stat.count == 0 ? 0.0 : stat.totalMs / static_cast<double>(stat.count);
            printf("  %s count=%lld total_ms=%.3f avg_ms=%.3f max_ms=%.3f\n",
                   sandy::ir::mid_ir::op_kind_name(opKind),
                   static_cast<long long>(stat.count),
                   stat.totalMs,
                   avgMs,
                   stat.maxMs);
        }
    }

    auto outputs = runResult.take();
    for (size_t index = 0; index < outputs.size(); index++) {
        auto& buffer = outputs[index];
        auto name = "output" + std::to_string(index);
        if (!buffer) {
            fprintf(stderr, "null output buffer: %s\n", name.c_str());
            return 1;
        }
        print_tensor(name, *buffer);
    }
    printStage("print_outputs");
    if (instrument)
        printf("[stage] total_runner time_ms=%.3f\n", elapsed_ms(totalStart, Clock::now()));

    return 0;
}
