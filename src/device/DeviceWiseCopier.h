#pragma once

#include "Device.h"
#include "Result.h"

namespace sandy::device {

class DeviceWiseCopier {
public:
    virtual ~DeviceWiseCopier() = default;

    virtual Result<DeviceTensorView> copy(
        Device& sourceDevice,
        DeviceTensorView sourceView,
        Device& target) = 0;
};

class HostBounceDeviceWiseCopier final : public DeviceWiseCopier {
public:
    Result<DeviceTensorView> copy(
        Device& sourceDevice,
        DeviceTensorView sourceView,
        Device& target) override;
};

} // namespace sandy::device
