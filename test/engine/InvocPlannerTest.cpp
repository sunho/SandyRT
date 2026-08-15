#include "InvocPlanner.h"
#include "MidIR.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

class InvocPlannerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sandy::ir::mid_ir::register_all_ops();
    }
};

template<typename T>
const T& payload_as(const sandy::engine::InvocInstruction& instruction) {
    return std::get<T>(instruction.payload);
}

} // namespace

TEST_F(InvocPlannerTest, PlansInputWeightsAndSingleComputeOp) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* w = builder.createWeight("w", sandy::core::Shape({3, 2}), sandy::core::DType::F32);
    auto* b = builder.createWeight("b", sandy::core::Shape({3}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, w, b);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::InvocPlanner planner(7);
    auto draftResult = planner.plan(graph);
    ASSERT_TRUE(draftResult) << draftResult.error();
    auto draft = draftResult.take();

    ASSERT_EQ(draft.programSources.size(), 1u);
    EXPECT_EQ(draft.programSources[0].id, 0u);
    EXPECT_EQ(draft.programSources[0].device, 7u);
    EXPECT_EQ(draft.programSources[0].op, out->def);

    ASSERT_EQ(draft.outputs, std::vector<sandy::engine::InvocValueId>({3}));
    ASSERT_EQ(draft.instructions.size(), 9u);

    ASSERT_EQ(draft.instructions[0].kind, sandy::engine::InvocInstructionKind::LoadInput);
    const auto& loadInput = payload_as<sandy::engine::InvocLoadInput>(draft.instructions[0]);
    EXPECT_EQ(loadInput.device, 7u);
    EXPECT_EQ(loadInput.index, 0);
    EXPECT_EQ(loadInput.value, 0u);

    ASSERT_EQ(draft.instructions[1].kind, sandy::engine::InvocInstructionKind::LoadWeight);
    const auto& loadWeight = payload_as<sandy::engine::InvocLoadWeight>(draft.instructions[1]);
    EXPECT_EQ(loadWeight.device, 7u);
    EXPECT_EQ(loadWeight.name, "w");
    EXPECT_EQ(loadWeight.value, 1u);

    ASSERT_EQ(draft.instructions[2].kind, sandy::engine::InvocInstructionKind::LoadWeight);
    const auto& loadBias = payload_as<sandy::engine::InvocLoadWeight>(draft.instructions[2]);
    EXPECT_EQ(loadBias.device, 7u);
    EXPECT_EQ(loadBias.name, "b");
    EXPECT_EQ(loadBias.value, 2u);

    ASSERT_EQ(draft.instructions[3].kind, sandy::engine::InvocInstructionKind::Alloc);
    const auto& alloc = payload_as<sandy::engine::InvocAlloc>(draft.instructions[3]);
    EXPECT_EQ(alloc.device, 7u);
    EXPECT_EQ(alloc.value, 3u);
    EXPECT_EQ(alloc.desc.shape, sandy::core::Shape({1, 3}));
    EXPECT_EQ(alloc.desc.dtype, sandy::core::DType::F32);

    ASSERT_EQ(draft.instructions[4].kind, sandy::engine::InvocInstructionKind::RunKernel);
    const auto& run = payload_as<sandy::engine::InvocRunKernel>(draft.instructions[4]);
    EXPECT_EQ(run.device, 7u);
    EXPECT_EQ(run.program, 0u);
    EXPECT_EQ(run.inputs, std::vector<sandy::engine::InvocValueId>({0, 1, 2}));
    EXPECT_EQ(run.outputs, std::vector<sandy::engine::InvocValueId>({3}));

    ASSERT_EQ(draft.instructions[5].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[5]).value, 0u);
    ASSERT_EQ(draft.instructions[6].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[6]).value, 1u);
    ASSERT_EQ(draft.instructions[7].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[7]).value, 2u);

    ASSERT_EQ(draft.instructions[8].kind, sandy::engine::InvocInstructionKind::StoreOutputs);
    const auto& store = payload_as<sandy::engine::InvocStoreOutputs>(draft.instructions[8]);
    EXPECT_EQ(store.device, 7u);
    EXPECT_EQ(store.values, std::vector<sandy::engine::InvocValueId>({3}));
}

TEST_F(InvocPlannerTest, ReshapeAllocatesDescriptorChangingOutput) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {3, 2});
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::InvocPlanner planner;
    auto draftResult = planner.plan(graph);
    ASSERT_TRUE(draftResult) << draftResult.error();
    auto draft = draftResult.take();

    ASSERT_EQ(draft.programSources.size(), 1u);
    EXPECT_EQ(draft.programSources[0].op, out->def);
    ASSERT_EQ(draft.outputs, std::vector<sandy::engine::InvocValueId>({1}));
    ASSERT_EQ(draft.instructions.size(), 5u);
    EXPECT_EQ(draft.instructions[0].kind, sandy::engine::InvocInstructionKind::LoadInput);

    const auto& loadInput = payload_as<sandy::engine::InvocLoadInput>(draft.instructions[0]);
    EXPECT_EQ(loadInput.index, 0);
    EXPECT_EQ(loadInput.value, 0u);

    ASSERT_EQ(draft.instructions[1].kind, sandy::engine::InvocInstructionKind::Alloc);
    const auto& alloc = payload_as<sandy::engine::InvocAlloc>(draft.instructions[1]);
    EXPECT_EQ(alloc.value, 1u);
    EXPECT_EQ(alloc.desc.shape, sandy::core::Shape({3, 2}));

    ASSERT_EQ(draft.instructions[2].kind, sandy::engine::InvocInstructionKind::RunKernel);
    const auto& run = payload_as<sandy::engine::InvocRunKernel>(draft.instructions[2]);
    EXPECT_EQ(run.inputs, std::vector<sandy::engine::InvocValueId>({0}));
    EXPECT_EQ(run.outputs, std::vector<sandy::engine::InvocValueId>({1}));

    ASSERT_EQ(draft.instructions[3].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[3]).value, 0u);

    ASSERT_EQ(draft.instructions[4].kind, sandy::engine::InvocInstructionKind::StoreOutputs);
    const auto& store = payload_as<sandy::engine::InvocStoreOutputs>(draft.instructions[4]);
    EXPECT_EQ(store.values, std::vector<sandy::engine::InvocValueId>({1}));
    ASSERT_EQ(store.descs.size(), 1u);
    EXPECT_EQ(store.descs[0].shape, sandy::core::Shape({3, 2}));
}

TEST_F(InvocPlannerTest, ReshapeFeedsComputeWithIndependentProgram) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* reshaped = builder.createReshape(x, {3, 2});
    auto* out = builder.createTanh(reshaped);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::InvocPlanner planner;
    auto draftResult = planner.plan(graph);
    ASSERT_TRUE(draftResult) << draftResult.error();
    auto draft = draftResult.take();

    ASSERT_EQ(draft.programSources.size(), 2u);
    EXPECT_EQ(draft.programSources[0].op, reshaped->def);
    EXPECT_EQ(draft.programSources[1].op, out->def);
    ASSERT_EQ(draft.outputs, std::vector<sandy::engine::InvocValueId>({2}));
    ASSERT_EQ(draft.instructions.size(), 8u);

    EXPECT_EQ(draft.instructions[0].kind, sandy::engine::InvocInstructionKind::LoadInput);
    EXPECT_EQ(draft.instructions[1].kind, sandy::engine::InvocInstructionKind::Alloc);
    EXPECT_EQ(draft.instructions[2].kind, sandy::engine::InvocInstructionKind::RunKernel);
    ASSERT_EQ(draft.instructions[3].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[3]).value, 0u);
    EXPECT_EQ(draft.instructions[4].kind, sandy::engine::InvocInstructionKind::Alloc);
    EXPECT_EQ(draft.instructions[5].kind, sandy::engine::InvocInstructionKind::RunKernel);
    ASSERT_EQ(draft.instructions[6].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[6]).value, 1u);
    EXPECT_EQ(draft.instructions[7].kind, sandy::engine::InvocInstructionKind::StoreOutputs);

    const auto& run = payload_as<sandy::engine::InvocRunKernel>(draft.instructions[2]);
    EXPECT_EQ(run.inputs, std::vector<sandy::engine::InvocValueId>({0}));
    EXPECT_EQ(run.outputs, std::vector<sandy::engine::InvocValueId>({1}));

    const auto& tanhRun = payload_as<sandy::engine::InvocRunKernel>(draft.instructions[5]);
    EXPECT_EQ(tanhRun.inputs, std::vector<sandy::engine::InvocValueId>({1}));
    EXPECT_EQ(tanhRun.outputs, std::vector<sandy::engine::InvocValueId>({2}));

    const auto& store = payload_as<sandy::engine::InvocStoreOutputs>(draft.instructions[7]);
    EXPECT_EQ(store.values, std::vector<sandy::engine::InvocValueId>({2}));
}

TEST_F(InvocPlannerTest, DeallocsValuesImmediatelyAfterLastConsumer) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* tanh = builder.createTanh(x);
    auto* out = builder.createReLU(tanh);
    sandy::ir::mid_ir::Value* outputs[] = {out};
    builder.setOutputs(outputs);

    sandy::engine::InvocPlanner planner;
    auto draftResult = planner.plan(graph);
    ASSERT_TRUE(draftResult) << draftResult.error();
    auto draft = draftResult.take();

    ASSERT_EQ(draft.instructions.size(), 8u);
    EXPECT_EQ(draft.instructions[0].kind, sandy::engine::InvocInstructionKind::LoadInput);
    EXPECT_EQ(draft.instructions[1].kind, sandy::engine::InvocInstructionKind::Alloc);
    EXPECT_EQ(draft.instructions[2].kind, sandy::engine::InvocInstructionKind::RunKernel);
    ASSERT_EQ(draft.instructions[3].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[3]).value, 0u);
    EXPECT_EQ(draft.instructions[4].kind, sandy::engine::InvocInstructionKind::Alloc);
    EXPECT_EQ(draft.instructions[5].kind, sandy::engine::InvocInstructionKind::RunKernel);
    ASSERT_EQ(draft.instructions[6].kind, sandy::engine::InvocInstructionKind::Dealloc);
    EXPECT_EQ(payload_as<sandy::engine::InvocDealloc>(draft.instructions[6]).value, 1u);
    EXPECT_EQ(draft.instructions[7].kind, sandy::engine::InvocInstructionKind::StoreOutputs);
}

TEST_F(InvocPlannerTest, ReshapeOutputKeepsReshapedBufferAlive) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);

    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* reshaped = builder.createReshape(x, {3, 2});
    auto* dead = builder.createTanh(reshaped);
    sandy::ir::mid_ir::Value* outputs[] = {reshaped};
    builder.setOutputs(outputs);
    (void)dead;

    sandy::engine::InvocPlanner planner;
    auto draftResult = planner.plan(graph);
    ASSERT_TRUE(draftResult) << draftResult.error();
    auto draft = draftResult.take();

    ASSERT_EQ(draft.outputs, std::vector<sandy::engine::InvocValueId>({1}));
    ASSERT_FALSE(draft.instructions.empty());
    ASSERT_EQ(draft.instructions.back().kind, sandy::engine::InvocInstructionKind::StoreOutputs);

    for (size_t index = 0; index + 1 < draft.instructions.size(); index++) {
        if (draft.instructions[index].kind != sandy::engine::InvocInstructionKind::Dealloc)
            continue;
        EXPECT_NE(payload_as<sandy::engine::InvocDealloc>(draft.instructions[index]).value, 1u);
    }

    const auto& store = payload_as<sandy::engine::InvocStoreOutputs>(draft.instructions.back());
    EXPECT_EQ(store.values, std::vector<sandy::engine::InvocValueId>({1}));
    ASSERT_EQ(store.descs.size(), 1u);
    EXPECT_EQ(store.descs[0].shape, sandy::core::Shape({3, 2}));
}
