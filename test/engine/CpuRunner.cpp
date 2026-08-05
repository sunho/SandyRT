#include "Compiler.h"
#include "CpuInterpreterBackend.h"
#include "Engine.h"
#include "Interpreter.h"
#include "Lexer.h"
#include "Parser.h"
#include "SafeTensorWeights.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
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

void print_tensor(const std::string& name,
                  const sandy::engine::backend::BackendBuffer& buffer) {
    const auto& desc = buffer.desc();
    printf("%s: %s%s\n",
           name.c_str(),
           sandy::core::dtype_name(desc.dtype),
           desc.shape.str().c_str());

    if (desc.dtype != sandy::core::DType::F32) {
        printf("  raw bytes: %zu\n", buffer.data().size());
        return;
    }

    auto numel = desc.shape.numel();
    if (numel < 0) {
        printf("  dynamic output shape\n");
        return;
    }

    printf("  [");
    for (int64_t i = 0; i < numel; i++) {
        if (i > 0) printf(", ");
        printf("%.6g", read_f32(buffer.data(), static_cast<size_t>(i)));
    }
    printf("]\n");

    if (numel > 0) {
        int64_t best = 0;
        float bestValue = read_f32(buffer.data(), 0);
        for (int64_t i = 1; i < numel; i++) {
            float value = read_f32(buffer.data(), static_cast<size_t>(i));
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        printf("  argmax: %lld (%.6g)\n", static_cast<long long>(best), bestValue);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: cpu_runner <program.sandy.go> <weights.safetensors> <inputs.safetensors>\n");
        return 1;
    }

    printf("[1/8] reading sandy go: %s\n", argv[1]);
    std::string source;
    if (!read_file(argv[1], source)) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

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

    printf("sandy go ast:\n");
    dump_program(program);
    std::cout << std::flush;

    printf("[3/8] lowering sandy go ast -> high ir\n");
    sandy::ir::high_ir::Graph highGraph;
    sandy::sandygo::Interpreter interp(program, highGraph);
    interp.interpret();
    printf("high ir:\n");
    highGraph.dump();

    sandy::Compiler compiler;

    printf("[4/8] loading weights: %s\n", argv[2]);
    auto weights = sandy::weight::EagerSafeTensorWeights::load(argv[2]);

    printf("[5/8] loading inputs: %s\n", argv[3]);
    auto inputs = sandy::weight::EagerSafeTensorWeights::load(argv[3]);

    sandy::ir::mid_ir::MaterializeOptions options;
    for (const auto& desc : inputs->descriptors())
        options.input_tensor_descs[desc.name] = desc;

    printf("[6/8] materializing mid ir\n");
    auto midResult = compiler.materialize_mid_ir(highGraph, *weights, options);
    if (!midResult) {
        fprintf(stderr, "materialize error: %s\n", midResult.error().c_str());
        return 1;
    }
    auto midGraph = midResult.take();
    midGraph->dump();

    printf("[7/8] creating cpu backend plan\n");
    sandy::engine::Engine engine(
        std::make_unique<sandy::engine::backend::CpuInterpreterBackend>());
    auto planResult = engine.create_plan(*midGraph);
    if (!planResult) {
        fprintf(stderr, "plan error: %s\n", planResult.error().c_str());
        return 1;
    }
    auto plan = planResult.take();

    sandy::engine::TensorMap inputMap;
    sandy::engine::TensorMap weightMap;
    add_tensors_to_map(*inputs, inputMap);
    add_tensors_to_map(*weights, weightMap);

    printf("[8/8] backend mid ir runs\n");
    auto runResult = engine.run(*plan, inputMap, weightMap);
    if (!runResult) {
        fprintf(stderr, "run error: %s\n", runResult.error().c_str());
        return 1;
    }

    auto outputs = runResult.take();
    for (const auto& [name, buffer] : outputs) {
        if (!buffer) {
            fprintf(stderr, "null output buffer: %s\n", name.c_str());
            return 1;
        }
        print_tensor(name, *buffer);
    }

    return 0;
}
