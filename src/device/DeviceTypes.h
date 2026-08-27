#pragma once

#include "TensorBuffer.h"
#include "KernelIR.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sandy::device {

using DeviceBufferId = uint32_t;
using DeviceCompiledGraphId = uint32_t;
using DeviceProgramId = DeviceCompiledGraphId;
using DevicePagedPoolId = uint32_t;
using DevicePagedTensorId = uint32_t;

using TensorBufferPtr = std::shared_ptr<core::TensorBuffer>;
using TensorMap = std::unordered_map<std::string, TensorBufferPtr>;

struct TensorViewDesc {
    core::TensorDesc desc;
    std::vector<int64_t> strides;
    int64_t storageOffset = 0;
};

struct DeviceTensorView {
    DeviceBufferId buffer = 0;
    TensorViewDesc view;
};

struct DevicePagedPoolDesc {
    core::TensorDesc templateDesc;
    int64_t growDim = -1;
    int64_t pageSize = -1;
    int64_t initialPages = 0;
    int64_t maxPages = -1;
};

struct DevicePagedTensorMeta {
    DevicePagedPoolId pool = 0;
    core::TensorDesc logicalDesc;
    int64_t growDim = -1;
    int64_t pageSize = -1;
    int64_t growLength = 0;
    int64_t pageCount = 0;
    int64_t pageElementCount = 0;
};

struct DevicePagedTensorView {
    DevicePagedTensorId tensor = 0;
    DevicePagedTensorMeta meta;
};

using DeviceTensorLikeView = std::variant<DeviceTensorView, DevicePagedTensorView>;

struct DeviceTensorTupleView {
    std::vector<DeviceTensorLikeView> elements;
};

using DeviceRunValue = std::variant<
    DeviceTensorView,
    DevicePagedTensorView,
    DeviceTensorTupleView>;

struct DeviceScratchAllocation {
    DeviceBufferId buffer = 0;
    std::unordered_map<ir::kernel_ir::ValueId, DeviceTensorView> views;
};

struct DeviceScratchPlacement {
    core::TensorDesc desc;
    size_t byteOffset = 0;
};

struct DeviceScratchLayout {
    size_t bytes = 0;
    std::unordered_map<ir::kernel_ir::ValueId, DeviceScratchPlacement> placements;
};

} // namespace sandy::device
