#include "DeviceWiseCopier.h"

namespace sandy::engine {

Result<DeviceBufferId> HostBounceDeviceWiseCopier::copy(
        Device& source,
        DeviceBufferId sourceBuffer,
        Device& target) {
    auto host = source.read(sourceBuffer);
    if (!host)
        return make_error(host.error());
    auto loaded = target.load(**host);
    if (!loaded)
        return make_error(loaded.error());
    return loaded.take();
}

} // namespace sandy::engine
