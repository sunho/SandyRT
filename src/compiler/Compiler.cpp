#include "Compiler.h"
#include "Lexer.h"
#include "Parser.h"
#include "Interpreter.h"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace sandy {

high_ir::Graph Compiler::load_sandygo(const std::string& path) {
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

    high_ir::Graph graph;
    sandygo::Interpreter interp(program, graph);
    interp.interpret();
    return graph;
}

mid_ir::Graph Compiler::materialize_mid_ir(const high_ir::Graph& graph,
                                           const weight::Weights& weights) {
    // TODO: implement
    return mid_ir::Graph{};
}

} // namespace sandy
