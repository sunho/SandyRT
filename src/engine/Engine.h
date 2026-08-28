#pragma once

#include "Device.h"
#include "DeviceWiseCopier.h"
#include "EngineTypes.h"
#include "KernelIR.h"
#include "MidIR.h"
#include "MidIRToKernelIR.h"
#include "Result.h"
#include "RuntimeScratchPlan.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::engine {

struct EngineProfileEvent {
    size_t opIndex = 0;
    ir::kernel_ir::OpId op = 0;
    uint32_t device = 0;
    DeviceCompiledGraphId deviceGraph = 0;
    ir::kernel_ir::OpKind opKind = ir::kernel_ir::OpKind::Input;
    size_t inputCount = 0;
    size_t outputCount = 0;
    double elapsedMs = 0.0;
};

struct EngineProfileStageEvent {
    std::string stage;
    size_t opIndex = 0;
    ir::kernel_ir::OpId op = 0;
    ir::kernel_ir::OpKind opKind = ir::kernel_ir::OpKind::Input;
    double elapsedMs = 0.0;
};

struct EngineDeviceRunBoundaryEvent {
    enum class Boundary {
        Begin,
        End,
    };

    Boundary boundary = Boundary::Begin;
    size_t opIndex = 0;
    ir::kernel_ir::OpId op = 0;
    uint32_t device = 0;
    DeviceCompiledGraphId deviceGraph = 0;
    ir::kernel_ir::OpKind opKind = ir::kernel_ir::OpKind::Input;
    size_t inputCount = 0;
    size_t outputCount = 0;
    double elapsedMs = 0.0;
};

struct EngineRunOptions {
    std::function<void(const EngineProfileEvent&)> profileKernel;
    std::function<void(const EngineProfileStageEvent&)> profileStage;
    std::function<void(const EngineDeviceRunBoundaryEvent&)> profileDeviceRunBoundary;
};

struct EngineCompileOptions {
    ir::kernel_ir::FusorOptions fusor;
};

class Engine {
public:
    explicit Engine(
        std::vector<std::unique_ptr<device::Device>> devices,
        std::unique_ptr<device::DeviceWiseCopier> copier = nullptr);

    Result<std::unique_ptr<CompiledKernelGraph>> compile(
        const ir::mid_ir::Graph& graph,
        const EngineCompileOptions* options = nullptr);

    Result<std::unique_ptr<DeviceWeightMap>> loadWeights(
        const CompiledKernelGraph& compiled,
        const TensorMap& weights);

    Result<void> deallocWeights(DeviceWeightMap& weights);

    Result<std::vector<TensorBufferPtr>> run(
        const CompiledKernelGraph& compiled,
        std::span<TensorBufferPtr const> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options = nullptr);

    Result<std::vector<TensorBufferPtr>> run(
        const CompiledKernelGraph& compiled,
        std::span<TensorBufferPtr const> inputs,
        const DeviceWeightMap& weights,
        const EngineRunOptions* options = nullptr);

    Result<std::vector<RunOutput>> runValues(
        const CompiledKernelGraph& compiled,
        std::span<const RunInput> inputs,
        const TensorMap& weights,
        const EngineRunOptions* options = nullptr);

    RuntimePlanCacheStats runtimePlanCacheStats() const {
        return runtimePlanCache_.stats();
    }

    Result<std::vector<RunOutput>> runValues(
        const CompiledKernelGraph& compiled,
        std::span<const RunInput> inputs,
        const DeviceWeightMap& weights,
        const EngineRunOptions* options = nullptr);

private:
    class StaticScratchLease;

    Result<std::unique_ptr<StaticScratchLease>> acquireStaticScratch(
        const CompiledKernelGraph& compiled);
    void releaseStaticScratch(
        CompiledProgramId program,
        RuntimeScratchPlan plan);
    void discardStaticScratch(RuntimeScratchPlan plan);

    Result<std::vector<RunOutput>> runValuesImpl(
        const CompiledKernelGraph& compiled,
        std::span<const RunInput> inputs,
        const TensorMap* hostWeights,
        const DeviceWeightMap* deviceWeights,
        const EngineRunOptions* options);

    std::vector<std::unique_ptr<device::Device>> devices_;
    std::unique_ptr<device::DeviceWiseCopier> copier_;
    std::mutex staticScratchMutex_;
    std::unordered_map<CompiledProgramId, RuntimeScratchLayout> staticScratchLayouts_;
    std::unordered_map<CompiledProgramId, std::vector<RuntimeScratchPlan>> staticScratchAvailable_;
    RuntimePlanCache runtimePlanCache_;
};

} // namespace sandy::engine
