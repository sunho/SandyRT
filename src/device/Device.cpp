#include "Device.h"

#include <utility>
#include <vector>

namespace sandy::device {

Result<std::vector<int64_t>> Device::defaultStrides(const core::Shape& shape) const {
    if (shape.has_dynamic())
        return make_error("cannot compute default strides for dynamic shape");

    std::vector<int64_t> strides(static_cast<size_t>(shape.rank()), 1);
    int64_t stride = 1;
    for (int i = shape.rank() - 1; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= shape.dim(i);
    }
    return strides;
}

Result<TensorViewDesc> Device::defaultView(core::TensorDesc desc) const {
    auto strides = defaultStrides(desc.shape);
    if (!strides)
        return make_error(strides.error());

    TensorViewDesc view;
    view.desc = std::move(desc);
    view.strides = strides.take();
    view.storageOffset = 0;
    return view;
}

Result<bool> Device::isDefaultView(const TensorViewDesc& view) const {
    auto strides = defaultStrides(view.desc.shape);
    if (!strides)
        return make_error(strides.error());
    return view.storageOffset == 0 && view.strides == *strides;
}

} // namespace sandy::device
