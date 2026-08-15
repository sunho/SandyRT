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

TEST(CompilerTest, LayerNormBuiltinMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(x Node) Node {
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
func main(x Node) Node {
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

TEST(CompilerTest, EmbeddingBuiltinMaterializesToMidIROp) {
    fs::path dir = makeTempDir();
    writeFile(dir / "main.sandy.go", R"(
func main(input_ids Node) Node {
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

TEST(CompilerTest, ProgrammaticKVAttentionMaterializesBatchedHKeyValue) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* qWeight = highGraph.addWeight("q.weight");
    auto* kWeight = highGraph.addWeight("k.weight");
    auto* vWeight = highGraph.addWeight("v.weight");
    auto* oWeight = highGraph.addWeight("o.weight");
    auto results = highGraph.addBuiltin(
        "kv_attention",
        {x, qWeight, kWeight, vWeight, oWeight},
        {
            sandy::ir::high_ir::Attr::fromInt("heads", 2),
            sandy::ir::high_ir::Attr::fromInt("kv_heads", 1),
            sandy::ir::high_ir::Attr::fromInt("head_dim", 2),
            sandy::ir::high_ir::Attr::fromInt("window", 2),
        },
        3);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    weights.add(sandy::core::TensorDesc("q.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("k.weight", sandy::core::Shape({2, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("v.weight", sandy::core::Shape({2, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("o.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 3u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3, 4}));
    EXPECT_EQ(midGraph->outputs()[1]->shape, sandy::core::Shape({2, 1, 3, 2}));
    EXPECT_EQ(midGraph->outputs()[2]->shape, sandy::core::Shape({2, 1, 3, 2}));
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::MatMul);
    EXPECT_EQ(midGraph->outputs()[1]->def->kind, sandy::ir::mid_ir::OpKind::Permute);
    EXPECT_EQ(midGraph->outputs()[2]->def->kind, sandy::ir::mid_ir::OpKind::Permute);
}

TEST(CompilerTest, KVAttentionAppliesRoPEWhenThetaIsProvided) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* qWeight = highGraph.addWeight("q.weight");
    auto* kWeight = highGraph.addWeight("k.weight");
    auto* vWeight = highGraph.addWeight("v.weight");
    auto* oWeight = highGraph.addWeight("o.weight");
    auto results = highGraph.addBuiltin(
        "kv_attention",
        {x, qWeight, kWeight, vWeight, oWeight},
        {
            sandy::ir::high_ir::Attr::fromInt("heads", 2),
            sandy::ir::high_ir::Attr::fromInt("kv_heads", 1),
            sandy::ir::high_ir::Attr::fromInt("head_dim", 2),
            sandy::ir::high_ir::Attr::fromInt("window", 2),
            sandy::ir::high_ir::Attr::fromFloat("rope_theta", 10000.0),
        },
        3);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    weights.add(sandy::core::TensorDesc("q.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("k.weight", sandy::core::Shape({2, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("v.weight", sandy::core::Shape({2, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("o.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 3u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3, 4}));
    EXPECT_EQ(midGraph->outputs()[1]->shape, sandy::core::Shape({2, 1, 3, 2}));
    EXPECT_EQ(midGraph->outputs()[2]->shape, sandy::core::Shape({2, 1, 3, 2}));
    EXPECT_EQ(midGraph->outputs()[1]->def->kind, sandy::ir::mid_ir::OpKind::RoPE);
    EXPECT_EQ(midGraph->outputs()[1]->def->attrs.at("rope_theta").floatVal, 10000.0);
    EXPECT_EQ(midGraph->outputs()[2]->def->kind, sandy::ir::mid_ir::OpKind::Permute);

    int ropeCount = 0;
    for (auto* op : midGraph->entry()->ops) {
        if (op->kind == sandy::ir::mid_ir::OpKind::RoPE)
            ropeCount++;
    }
    EXPECT_EQ(ropeCount, 2);
}

TEST(CompilerTest, ProgrammaticKVAttentionMaterializesUnbatchedHKeyValue) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* qWeight = highGraph.addWeight("q.weight");
    auto* kWeight = highGraph.addWeight("k.weight");
    auto* vWeight = highGraph.addWeight("v.weight");
    auto* oWeight = highGraph.addWeight("o.weight");
    auto results = highGraph.addBuiltin(
        "kv_attention",
        {x, qWeight, kWeight, vWeight, oWeight},
        {
            sandy::ir::high_ir::Attr::fromInt("heads", 2),
            sandy::ir::high_ir::Attr::fromInt("kv_heads", 2),
            sandy::ir::high_ir::Attr::fromInt("head_dim", 2),
        },
        3);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    weights.add(sandy::core::TensorDesc("q.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("k.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("v.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("o.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({3, 4}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 3u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({3, 4}));
    EXPECT_EQ(midGraph->outputs()[1]->shape, sandy::core::Shape({2, 3, 2}));
    EXPECT_EQ(midGraph->outputs()[2]->shape, sandy::core::Shape({2, 3, 2}));
}

TEST(CompilerTest, ProgrammaticAttentionMaterializesBatchedCachedKV) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* k = highGraph.addInput("k");
    auto* v = highGraph.addInput("v");
    auto* qWeight = highGraph.addWeight("q.weight");
    auto* oWeight = highGraph.addWeight("o.weight");
    auto results = highGraph.addBuiltin(
        "attention",
        {x, k, v, qWeight, oWeight},
        {
            sandy::ir::high_ir::Attr::fromInt("heads", 2),
            sandy::ir::high_ir::Attr::fromInt("kv_heads", 1),
            sandy::ir::high_ir::Attr::fromInt("head_dim", 2),
            sandy::ir::high_ir::Attr::fromInt("window", 2),
            sandy::ir::high_ir::Attr::fromFloat("rope_theta", 10000.0),
        },
        1);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    weights.add(sandy::core::TensorDesc("q.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("o.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["k"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 1, 3, 2}), sandy::core::DType::F32);
    options.input_tensor_descs["v"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 1, 3, 2}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({2, 3, 4}));
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::MatMul);

    int ropeCount = 0;
    for (auto* op : midGraph->entry()->ops) {
        if (op->kind == sandy::ir::mid_ir::OpKind::RoPE)
            ropeCount++;
    }
    EXPECT_EQ(ropeCount, 1);
}

TEST(CompilerTest, ProgrammaticAttentionMaterializesUnbatchedCachedKV) {
    sandy::ir::high_ir::Graph highGraph;
    auto* x = highGraph.addInput("x");
    auto* k = highGraph.addInput("k");
    auto* v = highGraph.addInput("v");
    auto* qWeight = highGraph.addWeight("q.weight");
    auto* oWeight = highGraph.addWeight("o.weight");
    auto results = highGraph.addBuiltin(
        "attention",
        {x, k, v, qWeight, oWeight},
        {
            sandy::ir::high_ir::Attr::fromInt("heads", 2),
            sandy::ir::high_ir::Attr::fromInt("kv_heads", 2),
            sandy::ir::high_ir::Attr::fromInt("head_dim", 2),
        },
        1);
    highGraph.setOutputs(results);

    sandy::Compiler compiler;
    TestWeights weights;
    weights.add(sandy::core::TensorDesc("q.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));
    weights.add(sandy::core::TensorDesc("o.weight", sandy::core::Shape({4, 4}), sandy::core::DType::F32));

    sandy::ir::mid_ir::MaterializeOptions options;
    options.input_tensor_descs["x"] = sandy::core::TensorDesc(
        sandy::core::Shape({3, 4}), sandy::core::DType::F32);
    options.input_tensor_descs["k"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, 2}), sandy::core::DType::F32);
    options.input_tensor_descs["v"] = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, 2}), sandy::core::DType::F32);

    auto result = compiler.materialize_mid_ir(highGraph, weights, options);
    ASSERT_TRUE(result) << result.error();
    auto midGraph = result.take();

    ASSERT_EQ(midGraph->outputs().size(), 1u);
    EXPECT_EQ(midGraph->outputs()[0]->shape, sandy::core::Shape({3, 4}));
    EXPECT_EQ(midGraph->outputs()[0]->def->kind, sandy::ir::mid_ir::OpKind::MatMul);
}
