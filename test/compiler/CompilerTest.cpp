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
func imported(x Node) Node {
    return __relu(x)
}
)");

    writeFile(dir / "main.sandy.go", R"(
import "nested/layers.sandy.go"

func main(x Node) Node {
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

TEST(CompilerTest, RMSNormBuiltinMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Node) Node {
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

TEST(CompilerTest, AddMulSqrtBuiltinsMaterializeToMidIROps) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Node, y Node, z Node) Node {
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

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    ASSERT_NE(midGraph->outputs()[0], nullptr);
    ASSERT_NE(midGraph->outputs()[0]->def, nullptr);
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::Sqrt);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3}));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(CompilerTest, GemmaStyleMatMulWithTransposeMaterializesToMidIROps) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Node) Node {
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

    std::error_code ec;
    fs::remove_all(dir, ec);
}
