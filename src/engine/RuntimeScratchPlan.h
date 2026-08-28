#pragma once

#include "CacheKey.h"
#include "Device.h"
#include "EngineTypes.h"
#include "RuntimeTensorDesc.h"

#include <memory>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sandy::engine {

struct RuntimeScratchPlan {
    std::unordered_map<ir::kernel_ir::ValueId, device::DeviceTensorView> views;
    std::unordered_map<ir::kernel_ir::ValueId, ir::kernel_ir::DeviceId> devices;
    std::unordered_map<ir::kernel_ir::DeviceId, device::DeviceBufferId> buffers;
};

struct RuntimeScratchLayout {
    std::unordered_map<ir::kernel_ir::DeviceId, device::DeviceScratchLayout> devices;
};

struct CachedInvocationPlan {
    RuntimeTensorDescs tensorDescs;
    RuntimeScratchLayout scratchLayout;
};

struct RuntimePlanCacheStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t entries = 0;
};

class RuntimePlanCache {
public:
    using PlanPtr = std::shared_ptr<const CachedInvocationPlan>;
    using Factory = std::function<Result<CachedInvocationPlan>()>;

    Result<PlanPtr> getOrCreate(const core::CacheKey& key, const Factory& factory);
    RuntimePlanCacheStats stats() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<core::CacheKey, PlanPtr, core::CacheKeyHash> entries_;
    size_t hits_ = 0;
    size_t misses_ = 0;
};

// Simulates the graph in execution order. Value use counts are consumed here,
// before execution, and translated into allocator alloc/free events.
Result<RuntimeScratchLayout> planRuntimeScratchLayout(
    const CompiledKernelGraph& compiled,
    const RuntimeTensorDescs& tensorDescs,
    std::vector<std::unique_ptr<device::Device>>& devices);

Result<RuntimeScratchPlan> instantiateRuntimeScratch(
    const RuntimeScratchLayout& layout,
    std::vector<std::unique_ptr<device::Device>>& devices);

} // namespace sandy::engine
