#include "Allocator.h"
#include "CpuDevice.h"
#include "MidIRCpuInterpreter.h"
#include "MidIRToKernelIR.h"
#include "MidIR.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

class TestTensorBuffer final : public sandy::core::TensorBuffer {
public:
    TestTensorBuffer(sandy::core::TensorDesc desc, std::vector<uint8_t> data)
        : TensorBuffer(std::move(desc)), data_(std::move(data)) {}

private:
    Result<void> load() override { return {}; }
    void unload() override {}
    std::span<const uint8_t> data() const override { return data_; }

    std::vector<uint8_t> data_;
};

class CpuDeviceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        sandy::ir::mid_ir::register_all_ops();
    }
};

std::vector<uint8_t> f32_bytes(std::initializer_list<float> values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    size_t index = 0;
    for (float value : values) {
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
        index++;
    }
    return bytes;
}

std::shared_ptr<TestTensorBuffer> make_f32_buffer(
        const std::string& name,
        sandy::core::Shape shape,
        std::initializer_list<float> values) {
    return std::make_shared<TestTensorBuffer>(
        sandy::core::TensorDesc(name, std::move(shape), sandy::core::DType::F32),
        f32_bytes(values));
}

float read_f32(std::span<const uint8_t> bytes, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
    return value;
}

void expect_f32_output(
        sandy::device::CpuDevice& device,
        sandy::device::DeviceBufferId buffer,
        std::initializer_list<float> expected) {
    auto read = device.read(buffer);
    ASSERT_TRUE(read) << read.error();
    auto access = (*read)->access();
    ASSERT_TRUE(access) << access.error();
    auto data = (*access).data();
    ASSERT_EQ(data.size(), expected.size() * sizeof(float));

    size_t index = 0;
    for (float value : expected) {
        EXPECT_NEAR(read_f32(data, index), value, 1.0e-5f);
        index++;
    }
}

sandy::ir::kernel_ir::ValueType tensor_type(
        sandy::core::Shape shape,
        sandy::core::DType dtype = sandy::core::DType::F32) {
    return sandy::ir::kernel_ir::ValueType{
        sandy::ir::kernel_ir::ValueKind::Tensor,
        dtype,
        std::move(shape),
    };
}

sandy::ir::kernel_ir::ValueType paged_tensor_type(
        sandy::core::Shape shape,
        sandy::core::DType dtype,
        int64_t growDim,
        int64_t pageSize) {
    return sandy::ir::kernel_ir::ValueType{
        sandy::ir::kernel_ir::ValueKind::PagedTensor,
        dtype,
        std::move(shape),
        sandy::ir::kernel_ir::PagedTensorMeta{growDim, pageSize},
    };
}

sandy::device::DeviceTensorView tensor_view(
        sandy::device::CpuDevice& device,
        sandy::device::DeviceBufferId buffer,
        sandy::core::TensorDesc desc) {
    auto view = device.defaultView(std::move(desc));
    return sandy::device::DeviceTensorView{
        buffer,
        view.take(),
    };
}

} // namespace

TEST(CoreAllocatorTest, FixedPagePoolAllocatesAndReusesPages) {
    sandy::core::FixedPagePool pool;
    auto init = pool.initialize(16, 1, 2);
    ASSERT_TRUE(init) << init.error();
    EXPECT_EQ(pool.page_count(), 1u);
    EXPECT_EQ(pool.free_page_count(), 1u);

    auto first = pool.allocate();
    ASSERT_TRUE(first) << first.error();
    EXPECT_EQ(*first, 0u);

    auto second = pool.allocate();
    ASSERT_TRUE(second) << second.error();
    EXPECT_EQ(*second, 1u);

    auto third = pool.allocate();
    EXPECT_FALSE(third);
    EXPECT_NE(third.error().find("capacity"), std::string::npos);

    auto freed = pool.deallocate(*first);
    ASSERT_TRUE(freed) << freed.error();
    auto reused = pool.allocate();
    ASSERT_TRUE(reused) << reused.error();
    EXPECT_EQ(*reused, *first);
}

TEST_F(CpuDeviceTest, LoadReadAndDeallocBuffer) {
    sandy::device::CpuDevice device;
    auto host = make_f32_buffer("x", sandy::core::Shape({2}), {1.0f, 2.0f});

    auto loaded = device.load(*host);
    ASSERT_TRUE(loaded) << loaded.error();
    expect_f32_output(device, *loaded, {1.0f, 2.0f});

    auto dealloc = device.dealloc(*loaded);
    EXPECT_TRUE(dealloc) << dealloc.error();
    auto readAfterDealloc = device.read(*loaded);
    EXPECT_FALSE(readAfterDealloc);
}

TEST_F(CpuDeviceTest, PagedPoolAllocReserveAppendAndMeta) {
    sandy::device::CpuDevice device;
    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(
        sandy::core::Shape({2, 3, sandy::core::Shape::kDynamic, 4}),
        sandy::core::DType::F32);
    poolDesc.growDim = 2;
    poolDesc.pageSize = 2;
    poolDesc.initialPages = 1;
    poolDesc.maxPages = 4;

    auto pool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(pool) << pool.error();

    auto tensor = device.allocPaged(
        *pool,
        sandy::core::Shape({2, 3, 0, 4}));
    ASSERT_TRUE(tensor) << tensor.error();

    auto meta = device.pagedMeta(*tensor);
    ASSERT_TRUE(meta) << meta.error();
    EXPECT_EQ(meta->growLength, 0);
    EXPECT_EQ(meta->pageCount, 0);
    EXPECT_EQ(meta->pageElementCount, 2 * 3 * 2 * 4);

    auto chunk = make_f32_buffer(
        "chunk",
        sandy::core::Shape({2, 3, 3, 4}),
        {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f,
            28.0f, 29.0f, 30.0f, 31.0f,
            32.0f, 33.0f, 34.0f, 35.0f,
            36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f,
            44.0f, 45.0f, 46.0f, 47.0f,
            48.0f, 49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f,
            60.0f, 61.0f, 62.0f, 63.0f,
            64.0f, 65.0f, 66.0f, 67.0f,
            68.0f, 69.0f, 70.0f, 71.0f,
        });

    auto append = device.appendPaged(*tensor, *chunk);
    ASSERT_TRUE(append) << append.error();

    meta = device.pagedMeta(*tensor);
    ASSERT_TRUE(meta) << meta.error();
    EXPECT_EQ(meta->growLength, 3);
    EXPECT_EQ(meta->logicalDesc.shape, sandy::core::Shape({2, 3, 3, 4}));
    EXPECT_EQ(meta->growDim, 2);
    EXPECT_EQ(meta->pageSize, 2);
    EXPECT_EQ(meta->pageCount, 2);
    EXPECT_EQ(meta->pageElementCount, 48);

    EXPECT_TRUE(device.deallocPaged(*tensor));
    EXPECT_TRUE(device.destroyPagedPool(*pool));
}

TEST_F(CpuDeviceTest, PagedPoolAppendRejectsShapeMismatch) {
    sandy::device::CpuDevice device;
    sandy::device::DevicePagedPoolDesc poolDesc;
    poolDesc.templateDesc = sandy::core::TensorDesc(
        sandy::core::Shape({2, sandy::core::Shape::kDynamic, 4}),
        sandy::core::DType::F32);
    poolDesc.growDim = 1;
    poolDesc.pageSize = 2;
    poolDesc.maxPages = 4;

    auto pool = device.createPagedPool(poolDesc);
    ASSERT_TRUE(pool) << pool.error();
    auto tensor = device.allocPaged(*pool, sandy::core::Shape({2, 0, 4}));
    ASSERT_TRUE(tensor) << tensor.error();
    auto chunk = make_f32_buffer(
        "bad",
        sandy::core::Shape({3, 1, 4}),
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
         6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    auto append = device.appendPaged(*tensor, *chunk);
    EXPECT_FALSE(append);
    EXPECT_NE(append.error().find("non-grow dimension"), std::string::npos);

    EXPECT_TRUE(device.deallocPaged(*tensor));
    EXPECT_TRUE(device.destroyPagedPool(*pool));
}

TEST_F(CpuDeviceTest, CompileRejectsPagedTensorValue) {
    sandy::ir::kernel_ir::Graph graph;
    auto cache = graph.addValue(
        paged_tensor_type(
            sandy::core::Shape({2, sandy::core::Shape::kDynamic, 4}),
            sandy::core::DType::F32,
            1,
            2));
    graph.addOp<sandy::ir::kernel_ir::InputOp>(
        sandy::ir::kernel_ir::InputSource{
            sandy::ir::kernel_ir::InputSourceKind::Argument,
            0,
            ""},
        cache);
    graph.setOutputs({cache});

    sandy::device::CpuDevice device;
    auto compiled = device.compile(graph);
    EXPECT_FALSE(compiled);
    EXPECT_NE(compiled.error().find("paged tensor values yet"), std::string::npos);
}

TEST_F(CpuDeviceTest, DefaultViewRejectsDynamicShape) {
    sandy::device::CpuDevice device;
    auto view = device.defaultView(
        sandy::core::TensorDesc(sandy::core::Shape({sandy::core::Shape::kDynamic, 2}),
                                sandy::core::DType::F32));
    EXPECT_FALSE(view);
    EXPECT_NE(view.error().find("dynamic shape"), std::string::npos);
}

TEST_F(CpuDeviceTest, RunReshapeF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2, 3}), sandy::core::DType::F32);
    auto* out = builder.createReshape(x, {3, 2});
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    });
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_TRUE(device.dealloc(*xBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, RunAddF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* lhs = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createAdd(lhs, rhs);
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto lhsHost = make_f32_buffer("lhs", sandy::core::Shape({2}), {1.0f, 2.0f});
    auto rhsHost = make_f32_buffer("rhs", sandy::core::Shape({2}), {3.0f, 4.0f});
    auto lhsBuffer = device.load(*lhsHost);
    ASSERT_TRUE(lhsBuffer) << lhsBuffer.error();
    auto rhsBuffer = device.load(*rhsHost);
    ASSERT_TRUE(rhsBuffer) << rhsBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *lhsBuffer, lhsHost->desc()),
        tensor_view(device, *rhsBuffer, rhsHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {4.0f, 6.0f});
    EXPECT_TRUE(device.dealloc(*lhsBuffer));
    EXPECT_TRUE(device.dealloc(*rhsBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, RunLinearF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({1, 2}), sandy::core::DType::F32);
    auto* weight = builder.createWeight("w", sandy::core::Shape({2, 2}), sandy::core::DType::F32);
    auto* bias = builder.createWeight("b", sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createLinear(x, weight, bias);
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({1, 2}), {1.0f, 2.0f});
    auto weightHost = make_f32_buffer("w", sandy::core::Shape({2, 2}), {3.0f, 4.0f, 5.0f, 6.0f});
    auto biasHost = make_f32_buffer("b", sandy::core::Shape({2}), {7.0f, 8.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto weightBuffer = device.load(*weightHost);
    ASSERT_TRUE(weightBuffer) << weightBuffer.error();
    auto biasBuffer = device.load(*biasHost);
    ASSERT_TRUE(biasBuffer) << biasBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
        tensor_view(device, *weightBuffer, weightHost->desc()),
        tensor_view(device, *biasBuffer, biasHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {18.0f, 25.0f});
    EXPECT_TRUE(device.dealloc(*xBuffer));
    EXPECT_TRUE(device.dealloc(*weightBuffer));
    EXPECT_TRUE(device.dealloc(*biasBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, RunTanhF32) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* x = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createTanh(x);
    sandy::ir::mid_ir::Value* graphOutputs[] = {out};
    builder.setOutputs(graphOutputs);

    sandy::device::CpuDevice device;
    auto kernelGraph = sandy::ir::kernel_ir::lowerMidIRToKernelIR(graph);
    ASSERT_TRUE(kernelGraph) << kernelGraph.error();
    auto op = (*kernelGraph)->value((*kernelGraph)->outputs()[0]).def.op;
    auto compiled = device.compile(**kernelGraph);
    ASSERT_TRUE(compiled) << compiled.error();

    auto xHost = make_f32_buffer("x", sandy::core::Shape({2}), {0.0f, 1.0f});
    auto xBuffer = device.load(*xHost);
    ASSERT_TRUE(xBuffer) << xBuffer.error();
    auto outBuffer = device.alloc(sandy::core::TensorDesc(out->shape, out->dtype));
    ASSERT_TRUE(outBuffer) << outBuffer.error();

    std::vector<sandy::device::DeviceTensorView> inputs = {
        tensor_view(device, *xBuffer, xHost->desc()),
    };
    std::vector<sandy::device::DeviceTensorView> outputs = {
        tensor_view(device, *outBuffer, sandy::core::TensorDesc(out->shape, out->dtype)),
    };
    auto run = device.run(*compiled, op, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    expect_f32_output(device, *outBuffer, {0.0f, std::tanh(1.0f)});
    EXPECT_TRUE(device.dealloc(*xBuffer));
    EXPECT_TRUE(device.dealloc(*outBuffer));
}

TEST_F(CpuDeviceTest, CompileRejectsChainedUnaryElementwiseKernel) {
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto input = graph.addValue(tensor_type({2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        input);
    auto output = graph.addValue(tensor_type({2}));
    graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{input, kir::BroadcastMode::None},
        },
        output,
        2,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Sqrt, sandy::core::DType::F32, 0, 0.0, {0}},
            kir::ScalarNode{2, kir::ScalarOp::Tanh, sandy::core::DType::F32, 0, 0.0, {1}},
        });
    graph.setOutputs({output});

    sandy::device::CpuDevice device;
    auto compiled = device.compile(graph);
    EXPECT_FALSE(compiled);
    EXPECT_NE(compiled.error().find("unary kernel must be a single op"), std::string::npos);
}

TEST_F(CpuDeviceTest, CompileRejectsChainedBinaryElementwiseKernel) {
    namespace kir = sandy::ir::kernel_ir;

    kir::Graph graph;
    auto lhs = graph.addValue(tensor_type({2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 0, ""},
        lhs);
    auto rhs = graph.addValue(tensor_type({2}));
    graph.addOp<kir::InputOp>(
        kir::InputSource{kir::InputSourceKind::Argument, 1, ""},
        rhs);
    auto output = graph.addValue(tensor_type({2}));
    graph.addOp<kir::ElementwiseKernelOp>(
        std::vector<kir::ElementwiseInput>{
            kir::ElementwiseInput{lhs, kir::BroadcastMode::None},
            kir::ElementwiseInput{rhs, kir::BroadcastMode::None},
        },
        output,
        3,
        std::vector<kir::ScalarNode>{
            kir::ScalarNode{0, kir::ScalarOp::Load, sandy::core::DType::F32, 0, 0.0, {}},
            kir::ScalarNode{1, kir::ScalarOp::Load, sandy::core::DType::F32, 1, 0.0, {}},
            kir::ScalarNode{2, kir::ScalarOp::Add, sandy::core::DType::F32, 0, 0.0, {0, 1}},
            kir::ScalarNode{3, kir::ScalarOp::Mul, sandy::core::DType::F32, 0, 0.0, {2, 1}},
        });
    graph.setOutputs({output});

    sandy::device::CpuDevice device;
    auto compiled = device.compile(graph);
    EXPECT_FALSE(compiled);
    EXPECT_NE(compiled.error().find("binary kernel must be a single op"), std::string::npos);
}

TEST_F(CpuDeviceTest, DebugMidIRCpuInterpreterRunsOldPerOpDispatch) {
    sandy::ir::mid_ir::Graph graph;
    sandy::ir::mid_ir::Builder builder(graph);
    auto* lhs = builder.createInput(0, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* rhs = builder.createInput(1, sandy::core::Shape({2}), sandy::core::DType::F32);
    auto* out = builder.createAdd(lhs, rhs);

    auto lhsBytes = f32_bytes({1.0f, 2.0f});
    auto rhsBytes = f32_bytes({3.0f, 4.0f});
    std::vector<uint8_t> outBytes(2 * sizeof(float));

    auto lhsRef = sandy::core::make_tensor_ref(
        sandy::core::TensorDesc(lhs->shape, lhs->dtype),
        lhsBytes);
    ASSERT_TRUE(lhsRef) << lhsRef.error();
    auto rhsRef = sandy::core::make_tensor_ref(
        sandy::core::TensorDesc(rhs->shape, rhs->dtype),
        rhsBytes);
    ASSERT_TRUE(rhsRef) << rhsRef.error();
    auto outRef = sandy::core::make_mutable_tensor_ref(
        sandy::core::TensorDesc(out->shape, out->dtype),
        outBytes);
    ASSERT_TRUE(outRef) << outRef.error();

    std::vector<sandy::core::TensorRef> inputs = {*lhsRef, *rhsRef};
    std::vector<sandy::core::MutableTensorRef> outputs = {*outRef};
    auto run = sandy::device::debug::runMidIROpOnCpu(*out->def, inputs, outputs);
    ASSERT_TRUE(run) << run.error();

    EXPECT_FLOAT_EQ(read_f32(outBytes, 0), 4.0f);
    EXPECT_FLOAT_EQ(read_f32(outBytes, 1), 6.0f);
}
