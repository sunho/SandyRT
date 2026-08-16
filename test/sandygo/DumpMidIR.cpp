#include "Compiler.h"
#include "TensorBuffer.h"
#include "Weight.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <span>
#include <sstream>
#include <string>
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
        for (const auto& [_, tensor] : tensors_)
            result.push_back(tensor->desc());
        return result;
    }

    std::shared_ptr<sandy::core::TensorBuffer> get_tensor(
            const std::string& name) const override {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            return nullptr;
        return it->second;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<TestTensorBuffer>> tensors_;
};

sandy::core::DType parseDType(const std::string& text) {
    if (text == "f32") return sandy::core::DType::F32;
    if (text == "f16") return sandy::core::DType::F16;
    if (text == "bf16") return sandy::core::DType::BF16;
    if (text == "i32") return sandy::core::DType::I32;
    if (text == "i64") return sandy::core::DType::I64;
    if (text == "u8") return sandy::core::DType::U8;
    std::cerr << "unknown dtype in test directive: " << text << "\n";
    std::exit(1);
}

std::vector<int64_t> parseShape(const std::string& text) {
    std::vector<int64_t> dims;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        auto first = part.find_first_not_of(" \t");
        if (first == std::string::npos)
            continue;
        auto last = part.find_last_not_of(" \t");
        auto token = part.substr(first, last - first + 1);
        if (token == "?")
            dims.push_back(sandy::core::Shape::kDynamic);
        else
            dims.push_back(std::stoll(token));
    }
    return dims;
}

void parseTestDirectives(
        const fs::path& path,
        TestWeights& weights,
        sandy::ir::mid_ir::MaterializeOptions& options) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }

    std::regex directive(
        R"(^\s*//\s*(INPUT|WEIGHT):\s+(\S+)\s+(\S+)\s+\[([^\]]*)\]\s*$)");
    std::string line;
    while (std::getline(file, line)) {
        std::smatch match;
        if (!std::regex_match(line, match, directive))
            continue;

        std::string kind = match[1];
        std::string name = match[2];
        auto dtype = parseDType(match[3]);
        sandy::core::Shape shape(parseShape(match[4]));

        if (kind == "INPUT") {
            options.input_tensor_descs[name] =
                sandy::core::TensorDesc(shape, dtype);
        } else {
            weights.add(sandy::core::TensorDesc(name, shape, dtype));
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: dump_mid_ir <file.sandy.go>\n";
        return 1;
    }

    fs::path path(argv[1]);
    TestWeights weights;
    sandy::ir::mid_ir::MaterializeOptions options;
    parseTestDirectives(path, weights, options);

    sandy::Compiler compiler;
    auto highGraph = compiler.load_sandygo(path.string());
    auto midGraph = compiler.materialize_mid_ir(highGraph, weights, options);
    if (!midGraph) {
        std::cerr << "materialize error: " << midGraph.error() << "\n";
        return 1;
    }

    (*midGraph)->dump();
    return 0;
}
