#include "Compiler.h"
#include "MidIRMaterializer.h"
#include "Lexer.h"
#include "Parser.h"
#include "Interpreter.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace sandy {

namespace {

namespace fs = std::filesystem;

[[noreturn]] void fatal(const std::string& msg) {
    fprintf(stderr, "%s\n", msg.c_str());
    abort();
}

std::string pathKey(const fs::path& path) {
    std::error_code ec;
    fs::path abs = fs::absolute(path, ec);
    if (ec) {
        fatal("cannot resolve " + path.string() + ": " + ec.message());
    }
    return abs.lexically_normal().string();
}

sandygo::Program parseFile(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        fatal("cannot open " + path.string());
    }

    std::ostringstream ss;
    ss << file.rdbuf();

    sandygo::Lexer lexer(ss.str());
    auto tokens = lexer.tokenize();
    if (lexer.hasError()) {
        fatal("lexer error in " + path.string() + ": " + lexer.errorMessage());
    }

    sandygo::Parser parser(std::move(tokens));
    auto program = parser.parse();
    if (parser.hasError()) {
        fatal("parser error in " + path.string() + ": " + parser.errorMessage());
    }

    return program;
}

class SandyGoLoader {
public:
    sandygo::Program load(const fs::path& path) {
        sandygo::Program program;
        loadInto(path, program);
        return program;
    }

private:
    void loadInto(const fs::path& path, sandygo::Program& merged) {
        std::string key = pathKey(path);
        if (loaded_.contains(key)) return;
        if (visiting_.contains(key)) {
            fatal("import cycle involving " + key);
        }

        visiting_.insert(key);
        sandygo::Program program = parseFile(key);
        fs::path baseDir = fs::path(key).parent_path();

        for (const auto& import : program.imports) {
            fs::path importPath(import.path);
            if (importPath.is_relative()) {
                importPath = baseDir / importPath;
            }
            loadInto(importPath, merged);
        }

        for (auto& func : program.funcs) {
            auto [it, inserted] = funcOrigins_.emplace(func.name, key);
            if (!inserted) {
                fatal("duplicate function '" + func.name + "' in " + key +
                      " (already defined in " + it->second + ")");
            }
            merged.funcs.push_back(std::move(func));
        }

        visiting_.erase(key);
        loaded_.insert(key);
    }

    std::unordered_set<std::string> visiting_;
    std::unordered_set<std::string> loaded_;
    std::unordered_map<std::string, std::string> funcOrigins_;
};

} // namespace

ir::high_ir::Graph Compiler::load_sandygo(const std::string& path) {
    SandyGoLoader loader;
    sandygo::Program program = loader.load(path);

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
