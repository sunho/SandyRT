#include "Compiler.h"
#include "MidIRMaterializer.h"
#include "Lexer.h"
#include "Parser.h"
#include "Interpreter.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>

namespace sandy {

ir::high_ir::Graph Compiler::load_sandygo(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        abort();
    }
    std::ostringstream ss;
    ss << file.rdbuf();

    sandygo::Lexer lexer(ss.str());
    auto tokens = lexer.tokenize();
    if (lexer.hasError()) {
        fprintf(stderr, "lexer error: %s\n", lexer.errorMessage().c_str());
        abort();
    }

    sandygo::Parser parser(std::move(tokens));
    auto program = parser.parse();
    if (parser.hasError()) {
        fprintf(stderr, "parser error: %s\n", parser.errorMessage().c_str());
        abort();
    }

    ir::high_ir::Graph graph;
    sandygo::Interpreter interp(program, graph);
    interp.interpret();
    return graph;
}

Result<std::unique_ptr<ir::mid_ir::Graph>> Compiler::materialize_mid_ir(
        const ir::high_ir::Graph& graph,
        const weight::Weights& weights,
        const ir::mid_ir::MaterializeOptions& options) {
    ir::mid_ir::MidIRMaterializer materializer;
    return materializer.materialize(graph, weights, options);
}

} // namespace sandy
