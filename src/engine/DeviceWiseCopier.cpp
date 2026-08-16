#include "DeviceWiseCopier.h"

namespace sandy::engine {

Result<DeviceTensorView> HostBounceDeviceWiseCopier::copy(
        Device& sourceDevice,
        DeviceTensorView sourceView,
        Device& target) {
    auto host = sourceDevice.read(std::move(sourceView));
    if (!host)
        return make_error(host.error());
    auto loaded = target.load(**host);
    if (!loaded)
        return make_error(loaded.error());
    auto view = target.defaultView((*host)->desc());
    if (!view)
        return make_error(view.error());
    return DeviceTensorView{
        loaded.take(),
        view.take(),
    };
}

} // namespace sandy::engine
