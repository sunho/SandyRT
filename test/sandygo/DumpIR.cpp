#include "Lexer.h"
#include "Parser.h"
#include "Interpreter.h"
#include "HighIR.h"

#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: dump_ir <file.sandy.go>\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();

    sandy::sandygo::Lexer lexer(source);
    auto tokens = lexer.tokenize();
    if (lexer.hasError()) {
        std::cerr << "lexer error: " << lexer.errorMessage() << "\n";
        return 1;
    }

    sandy::sandygo::Parser parser(std::move(tokens));
    auto program = parser.parse();
    if (parser.hasError()) {
        std::cerr << "parser error: " << parser.errorMessage() << "\n";
        return 1;
    }

    sandy::ir::high_ir::Graph graph;
    sandy::sandygo::Interpreter interp(program, graph);
    interp.interpret();
    graph.dump();
    return 0;
}
