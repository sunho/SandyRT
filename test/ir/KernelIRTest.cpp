#include "KernelIR.h"
#include "MidIR.h"
#include "MidIRToKernelIR.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using namespace sandy::ir;

kernel_ir::ValueType tensor_type(
    sandy::core::Shape shape,
    sandy::core::DType dtype = sandy::core::DType::F32)
{
    return kernel_ir::ValueType{
        kernel_ir::ValueKind::Tensor,
        dtype,
        std::move(shape),
    };
}

kernel_ir::ValueType paged_tensor_type(
    sandy::core::Shape shape,
    sandy::core::DType dtype,
    int64_t growDim,
    int64_t pageSize)
{
    return kernel_ir::ValueType{
        kernel_ir::ValueKind::PagedTensor,
        dtype,
        std::move(shape),
        kernel_ir::PagedTensorMeta{growDim, pageSize},
    };
}

const kernel_ir::ElementwiseKernelOp& as_elementwise(
    const std::unique_ptr<kernel_ir::Op>& op)
{
    EXPECT_EQ(op->kind(), kernel_ir::OpKind::ElementwiseKernel);
    return static_cast<const kernel_ir::ElementwiseKernelOp&>(*op);
}

class MidIRToKernelIRTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        mid_ir::register_all_ops();
    }
};

} // namespace

TEST(KernelIRGraphTest, VerifyTracksDefsUsesAndOutputs) {
    kernel_ir::Graph graph;

    auto input = graph.addValue(tensor_type({2, 3}), "x");
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 0, ""},
        input);

    auto output = graph.addValue(tensor_type({2, 3}), "relu");
    std::vector<kernel_ir::ElementwiseInput> inputs = {
        kernel_ir::ElementwiseInput{input, kernel_ir::BroadcastMode::None},
    };
    std::vector<kernel_ir::ScalarNode> scalars = {
        kernel_ir::ScalarNode{
            0, kernel_ir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
        kernel_ir::ScalarNode{
            1, kernel_ir::ScalarOp::ReLU, sandy::core::DType::F32, 0, 0.0, {0}},
    };
    graph.addOp<kernel_ir::ElementwiseKernelOp>(
        std::move(inputs),
        output,
        1,
        std::move(scalars));
    graph.setOutputs({output});

    auto result = graph.verify();
    ASSERT_TRUE(result) << result.error();

    EXPECT_EQ(graph.value(input).uses.size(), 1u);
    EXPECT_EQ(graph.value(input).uses[0].op, 1u);
    EXPECT_EQ(graph.value(output).def.op, 1u);
    EXPECT_EQ(graph.outputs()[0], output);
}

TEST(KernelIRGraphTest, VerifyTracksPagedTensorInput) {
    kernel_ir::Graph graph;

    auto cache = graph.addValue(
        paged_tensor_type({2, -1, 128}, sandy::core::DType::BF16, 1, 16),
        "cache");
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 3, ""},
        cache);
    graph.setOutputs({cache});

    auto result = graph.verify();
    ASSERT_TRUE(result) << result.error();

    EXPECT_EQ(graph.value(cache).def.op, 0u);
    EXPECT_TRUE(graph.value(cache).uses.empty());

    testing::internal::CaptureStdout();
    graph.dump();
    auto dump = testing::internal::GetCapturedStdout();

    EXPECT_NE(dump.find("input()"), std::string::npos);
    EXPECT_NE(dump.find("source=arg(3)"), std::string::npos);
    EXPECT_NE(dump.find("paged_tensor<2x?x128xbf16"), std::string::npos);
    EXPECT_NE(dump.find("grow_dim=1"), std::string::npos);
    EXPECT_NE(dump.find("page_size=16"), std::string::npos);
}

TEST(KernelIRGraphTest, VerifyRejectsLayoutTransformOnPagedInput) {
    kernel_ir::Graph graph;

    auto cache = graph.addValue(
        paged_tensor_type({2, -1, 128}, sandy::core::DType::BF16, 1, 16));
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 0, ""},
        cache);
    auto reshaped = graph.addValue(tensor_type({2, -1, 128}, sandy::core::DType::BF16));
    graph.addOp<kernel_ir::LayoutTransformOp>(
        kernel_ir::LayoutTransformKind::Reshape,
        cache,
        reshaped,
        std::vector<int64_t>{2, -1, 128});
    graph.setOutputs({reshaped});

    auto result = graph.verify();
    EXPECT_FALSE(result);
    EXPECT_NE(
        result.error().find("layout transform cannot consume paged input"),
        std::string::npos);
}

TEST(KernelIRGraphTest, DumpPrintsElementwiseScalarDag) {
    kernel_ir::Graph graph;

    auto lhs = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({2, 3}));

    std::vector<kernel_ir::ElementwiseInput> inputs = {
        kernel_ir::ElementwiseInput{lhs, kernel_ir::BroadcastMode::None},
        kernel_ir::ElementwiseInput{rhs, kernel_ir::BroadcastMode::None},
    };
    std::vector<kernel_ir::ScalarNode> scalars = {
        kernel_ir::ScalarNode{
            0, kernel_ir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
        kernel_ir::ScalarNode{
            1, kernel_ir::ScalarOp::Load, sandy::core::DType::F32, 1, 0.0, {}},
        kernel_ir::ScalarNode{
            2, kernel_ir::ScalarOp::Add, sandy::core::DType::F32, 0, 0.0, {0, 1}},
    };
    graph.addOp<kernel_ir::ElementwiseKernelOp>(
        std::move(inputs),
        output,
        2,
        std::move(scalars));
    graph.setOutputs({output});

    testing::internal::CaptureStdout();
    graph.dump();
    auto dump = testing::internal::GetCapturedStdout();

    EXPECT_NE(dump.find("s0 = load input0"), std::string::npos);
    EXPECT_NE(dump.find("s1 = load input1"), std::string::npos);
    EXPECT_NE(dump.find("s2 = add s0, s1"), std::string::npos);
    EXPECT_NE(dump.find("store %2, s2"), std::string::npos);
}

TEST(KernelIRGraphTest, VerifyRejectsInvalidGraphOutput) {
    kernel_ir::Graph graph;
    graph.setOutputs({42});

    auto result = graph.verify();
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("graph output references invalid value"),
              std::string::npos);
}

TEST(KernelIRGraphTest, VerifyTracksDeviceTransfer) {
    kernel_ir::Graph graph;

    auto input = graph.addValue(tensor_type({2, 3}), "x");
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 0, ""},
        input);

    auto output = graph.addValue(tensor_type({2, 3}), "x_on_device_1", 1);
    graph.addOp<kernel_ir::DeviceTransferOp>(0, 1, input, output);
    graph.setOutputs({output});

    auto result = graph.verify();
    ASSERT_TRUE(result) << result.error();

    testing::internal::CaptureStdout();
    graph.dump();
    auto dump = testing::internal::GetCapturedStdout();

    EXPECT_NE(dump.find("device_transfer(%0)"), std::string::npos);
    EXPECT_NE(dump.find("source_device=0 target_device=1"), std::string::npos);
}

TEST(KernelIRGraphTest, VerifyRejectsSameDeviceTransfer) {
    kernel_ir::Graph graph;

    auto input = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kernel_ir::DeviceTransferOp>(0, 0, input, output);
    graph.setOutputs({output});

    auto result = graph.verify();
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("source and target are equal"), std::string::npos);
}

TEST(KernelIRGraphTest, VerifyRejectsKernelInputOnWrongDevice) {
    kernel_ir::Graph graph;

    auto input = graph.addValue(tensor_type({2, 3}));
    graph.addOp<kernel_ir::InputOp>(
        kernel_ir::InputSource{kernel_ir::InputSourceKind::Argument, 0, ""},
        input);

    auto output = graph.addValue(tensor_type({2, 3}), 1);
    graph.addOp<kernel_ir::ElementwiseKernelOp>(
        std::vector<kernel_ir::ElementwiseInput>{
            kernel_ir::ElementwiseInput{input, kernel_ir::BroadcastMode::None},
        },
        output,
        1,
        std::vector<kernel_ir::ScalarNode>{
            kernel_ir::ScalarNode{
                0, kernel_ir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kernel_ir::ScalarNode{
                1, kernel_ir::ScalarOp::Tanh, sandy::core::DType::F32, 0, 0.0, {0}},
        },
        1);
    graph.setOutputs({output});

    auto result = graph.verify();
    EXPECT_FALSE(result);
    EXPECT_NE(result.error().find("input %0 is on device 0, expected device 1"),
              std::string::npos);
}

TEST_F(MidIRToKernelIRTest, LowersInputAndWeight) {
    mid_ir::Graph midGraph;
    mid_ir::Builder builder(midGraph);

    auto* input = builder.createInput(
        2, sandy::core::Shape({-1, 4}), sandy::core::DType::F32);
    auto* weight = builder.createWeight(
        "layer.weight", sandy::core::Shape({4}), sandy::core::DType::F32);
    mid_ir::Value* outputs[] = {input, weight};
    builder.setOutputs(outputs);

    auto lowered = kernel_ir::lowerMidIRToKernelIR(midGraph);
    ASSERT_TRUE(lowered) << lowered.error();
    auto graph = lowered.take();

    ASSERT_EQ(graph->ops().size(), 2u);
    ASSERT_EQ(graph->outputs().size(), 2u);

    const auto& inputOp =
        static_cast<const kernel_ir::InputOp&>(*graph->ops()[0]);
    EXPECT_EQ(inputOp.source().kind, kernel_ir::InputSourceKind::Argument);
    EXPECT_EQ(inputOp.source().index, 2);

    const auto& weightOp =
        static_cast<const kernel_ir::InputOp&>(*graph->ops()[1]);
    EXPECT_EQ(weightOp.source().kind, kernel_ir::InputSourceKind::Weight);
    EXPECT_EQ(weightOp.source().name, "layer.weight");
}

TEST_F(MidIRToKernelIRTest, LowersPagedTensorInput) {
    mid_ir::Graph midGraph;
    mid_ir::Builder builder(midGraph);

    auto* cache = builder.createPagedTensorInput(
        4,
        sandy::core::Shape({2, -1, 128}),
        sandy::core::DType::BF16,
        1,
        16);
    mid_ir::Value* outputs[] = {cache};
    builder.setOutputs(outputs);

    auto lowered = kernel_ir::lowerMidIRToKernelIR(midGraph);
    ASSERT_TRUE(lowered) << lowered.error();
    auto graph = lowered.take();

    ASSERT_EQ(graph->ops().size(), 1u);
    ASSERT_EQ(graph->outputs().size(), 1u);

    const auto& inputOp =
        static_cast<const kernel_ir::InputOp&>(*graph->ops()[0]);
    EXPECT_EQ(inputOp.kind(), kernel_ir::OpKind::Input);
    EXPECT_EQ(inputOp.source().kind, kernel_ir::InputSourceKind::Argument);
    EXPECT_EQ(inputOp.source().index, 4);
    EXPECT_EQ(graph->value(inputOp.outputs()[0]).type.shape,
              sandy::core::Shape({2, -1, 128}));
    EXPECT_EQ(graph->value(inputOp.outputs()[0]).type.dtype,
              sandy::core::DType::BF16);
    EXPECT_EQ(graph->value(inputOp.outputs()[0]).type.kind,
              kernel_ir::ValueKind::PagedTensor);
    EXPECT_EQ(graph->value(inputOp.outputs()[0]).type.paged.growDim, 1);
    EXPECT_EQ(graph->value(inputOp.outputs()[0]).type.paged.pageSize, 16);
    EXPECT_EQ(graph->outputs()[0], inputOp.outputs()[0]);
}

TEST_F(MidIRToKernelIRTest, LowersUnaryAndBinaryElementwiseOps) {
    mid_ir::Graph midGraph;
    mid_ir::Builder builder(midGraph);

    auto* lhs = builder.createInput(
        0, sandy::core::Shape({2, 3, 4}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(
        1, sandy::core::Shape({3, 1}), sandy::core::DType::F32);
    auto* add = builder.createAdd(lhs, rhs);
    auto* tanh = builder.createTanh(add);
    mid_ir::Value* outputs[] = {tanh};
    builder.setOutputs(outputs);

    auto lowered = kernel_ir::lowerMidIRToKernelIR(midGraph);
    ASSERT_TRUE(lowered) << lowered.error();
    auto graph = lowered.take();

    ASSERT_EQ(graph->ops().size(), 4u);
    const auto& addOp = as_elementwise(graph->ops()[2]);
    ASSERT_EQ(addOp.elementwiseInputs().size(), 2u);
    EXPECT_EQ(addOp.elementwiseInputs()[0].broadcast,
              kernel_ir::BroadcastMode::None);
    EXPECT_EQ(addOp.elementwiseInputs()[1].broadcast,
              kernel_ir::BroadcastMode::RightAligned);
    ASSERT_EQ(addOp.scalars().size(), 3u);
    EXPECT_EQ(addOp.scalars()[2].op, kernel_ir::ScalarOp::Add);

    const auto& tanhOp = as_elementwise(graph->ops()[3]);
    ASSERT_EQ(tanhOp.scalars().size(), 2u);
    EXPECT_EQ(tanhOp.scalars()[1].op, kernel_ir::ScalarOp::Tanh);
    EXPECT_EQ(graph->outputs()[0], tanhOp.outputs()[0]);
}

TEST_F(MidIRToKernelIRTest, LowersConstantAndScalarBroadcastMul) {
    mid_ir::Graph midGraph;
    mid_ir::Builder builder(midGraph);

    auto* input = builder.createInput(
        0, sandy::core::Shape({2, 3}), sandy::core::DType::BF16);
    auto* scale = builder.createConstantF32(0.5f);
    auto* mul = builder.createMul(input, scale);
    mid_ir::Value* outputs[] = {mul};
    builder.setOutputs(outputs);

    auto lowered = kernel_ir::lowerMidIRToKernelIR(midGraph);
    ASSERT_TRUE(lowered) << lowered.error();
    auto graph = lowered.take();

    ASSERT_EQ(graph->ops().size(), 3u);
    const auto& constantOp = as_elementwise(graph->ops()[1]);
    EXPECT_TRUE(constantOp.elementwiseInputs().empty());
    ASSERT_EQ(constantOp.scalars().size(), 1u);
    EXPECT_EQ(constantOp.scalars()[0].op, kernel_ir::ScalarOp::Constant);
    EXPECT_EQ(constantOp.scalars()[0].constant, 0.5);

    const auto& mulOp = as_elementwise(graph->ops()[2]);
    ASSERT_EQ(mulOp.elementwiseInputs().size(), 2u);
    EXPECT_EQ(mulOp.elementwiseInputs()[1].broadcast,
              kernel_ir::BroadcastMode::RightAligned);
    ASSERT_EQ(mulOp.scalars().size(), 3u);
    EXPECT_EQ(mulOp.scalars()[2].op, kernel_ir::ScalarOp::Mul);
    EXPECT_EQ(graph->value(mulOp.outputs()[0]).type.dtype,
              sandy::core::DType::BF16);
}

TEST_F(MidIRToKernelIRTest, AttentionFusorOptionControlsLowering) {
    mid_ir::Graph midGraph;
    mid_ir::Builder builder(midGraph);

    auto* q = builder.createInput(
        0, sandy::core::Shape({1, 8, 16, 256}), sandy::core::DType::BF16);
    auto* k = builder.createInput(
        1, sandy::core::Shape({1, 2, 16, 256}), sandy::core::DType::BF16);
    auto* v = builder.createInput(
        2, sandy::core::Shape({1, 2, 16, 256}), sandy::core::DType::BF16);
    auto* attention = builder.createAttention(q, k, v, 512, 1.0f);
    mid_ir::Value* outputs[] = {attention};
    builder.setOutputs(outputs);

    auto defaultLowered = kernel_ir::lowerMidIRToKernelIR(midGraph);
    ASSERT_TRUE(defaultLowered) << defaultLowered.error();
    auto defaultGraph = defaultLowered.take();
    ASSERT_EQ(defaultGraph->ops().size(), 6u);
    EXPECT_EQ(defaultGraph->ops()[3]->kind(),
              kernel_ir::OpKind::SlidingQueryKeyScoreKernel);
    EXPECT_EQ(defaultGraph->ops()[4]->kind(), kernel_ir::OpKind::SoftmaxKernel);
    EXPECT_EQ(defaultGraph->ops()[5]->kind(), kernel_ir::OpKind::MatMulKernel);

    kernel_ir::LoweringOptions options;
    options.fusor.attention = true;
    auto fusedLowered = kernel_ir::lowerMidIRToKernelIR(midGraph, options);
    ASSERT_TRUE(fusedLowered) << fusedLowered.error();
    auto fusedGraph = fusedLowered.take();
    ASSERT_EQ(fusedGraph->ops().size(), 4u);
    ASSERT_EQ(fusedGraph->ops()[3]->kind(), kernel_ir::OpKind::AttentionKernel);

    const auto& attentionOp =
        static_cast<const kernel_ir::AttentionKernelOp&>(*fusedGraph->ops()[3]);
    EXPECT_EQ(attentionOp.inputs().size(), 3u);
    EXPECT_EQ(attentionOp.window(), 512);
    EXPECT_EQ(attentionOp.scale(), 1.0);
    EXPECT_EQ(fusedGraph->outputs()[0], attentionOp.outputs()[0]);
}
