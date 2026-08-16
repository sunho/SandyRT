#pragma once

#include "TensorBuffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::device {

using DeviceBufferId = uint32_t;
using DeviceCompiledGraphId = uint32_t;
using DeviceProgramId = DeviceCompiledGraphId;

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

} // namespace sandy::device
