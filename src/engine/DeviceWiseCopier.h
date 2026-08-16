#pragma once

#include "Device.h"
#include "EngineTypes.h"
#include "Result.h"

namespace sandy::engine {

class DeviceWiseCopier {
public:
    virtual ~DeviceWiseCopier() = default;

    virtual Result<DeviceBufferId> copy(
        Device& source,
        DeviceBufferId sourceBuffer,
        Device& target) = 0;
};

class HostBounceDeviceWiseCopier final : public DeviceWiseCopier {
public:
    Result<DeviceBufferId> copy(
        Device& source,
        DeviceBufferId sourceBuffer,
        Device& target) override;
};

} // namespace sandy::engine
