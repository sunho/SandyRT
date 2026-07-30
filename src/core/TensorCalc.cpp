#include "TensorCalc.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace sandy::core {

namespace {

Result<void> require_f32(const TensorDesc& desc, const std::string& name) {
    if (desc.dtype != DType::F32)
        return make_error(name + " must be f32");
    return {};
}

Result<void> require_bytes(std::span<const uint8_t> data,
                           const TensorDesc& desc,
                           const std::string& name) {
    auto numel = desc.shape.numel();
    if (numel < 0)
        return make_error(name + " must have static shape");

    auto expected = static_cast<size_t>(numel) * dtype_size(desc.dtype);
    if (data.size() != expected)
        return make_error(name + " byte size mismatch");

    return {};
}

float read_f32(std::span<const uint8_t> data, size_t index) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + index * sizeof(float), sizeof(float));
    return value;
}

void write_f32(std::vector<uint8_t>& data, size_t index, float value) {
    std::memcpy(data.data() + index * sizeof(float), &value, sizeof(float));
}

} // namespace

Result<OwnedTensor> linear_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc,
        std::span<const uint8_t> weight,
        const TensorDesc& weightDesc,
        std::span<const uint8_t> bias,
        const TensorDesc& biasDesc) {
    auto xDtype = require_f32(xDesc, "linear input");
    if (!xDtype) return make_error(xDtype.error());

    auto weightDtype = require_f32(weightDesc, "linear weight");
    if (!weightDtype) return make_error(weightDtype.error());

    auto biasDtype = require_f32(biasDesc, "linear bias");
    if (!biasDtype) return make_error(biasDtype.error());

    if (xDesc.shape.rank() != 2)
        return make_error("linear input must have rank 2");
    if (weightDesc.shape.rank() != 2)
        return make_error("linear weight must have rank 2");
    if (biasDesc.shape.rank() != 1)
        return make_error("linear bias must have rank 1");

    int64_t batch = xDesc.shape.dim(0);
    int64_t inFeatures = xDesc.shape.dim(1);
    int64_t outFeatures = weightDesc.shape.dim(0);
    if (batch < 0 || inFeatures < 0 || outFeatures < 0)
        return make_error("linear inputs must have static shape");
    if (weightDesc.shape.dim(1) != inFeatures)
        return make_error("linear weight input dimension mismatch");
    if (biasDesc.shape.dim(0) != outFeatures)
        return make_error("linear bias dimension mismatch");

    auto xBytes = require_bytes(x, xDesc, "linear input");
    if (!xBytes) return make_error(xBytes.error());

    auto weightBytes = require_bytes(weight, weightDesc, "linear weight");
    if (!weightBytes) return make_error(weightBytes.error());

    auto biasBytes = require_bytes(bias, biasDesc, "linear bias");
    if (!biasBytes) return make_error(biasBytes.error());

    OwnedTensor out;
    out.desc = TensorDesc(Shape({batch, outFeatures}), DType::F32);
    out.data.resize(static_cast<size_t>(batch * outFeatures) * sizeof(float));

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t o = 0; o < outFeatures; o++) {
            float acc = read_f32(bias, static_cast<size_t>(o));
            for (int64_t i = 0; i < inFeatures; i++) {
                float xv = read_f32(x, static_cast<size_t>(b * inFeatures + i));
                float wv = read_f32(weight, static_cast<size_t>(o * inFeatures + i));
                acc += xv * wv;
            }
            write_f32(out.data, static_cast<size_t>(b * outFeatures + o), acc);
        }
    }

    return out;
}

Result<OwnedTensor> relu_f32(
        std::span<const uint8_t> x,
        const TensorDesc& xDesc) {
    auto xDtype = require_f32(xDesc, "relu input");
    if (!xDtype) return make_error(xDtype.error());

    auto xBytes = require_bytes(x, xDesc, "relu input");
    if (!xBytes) return make_error(xBytes.error());

    OwnedTensor out;
    out.desc = xDesc;
    out.data.resize(x.size());

    size_t count = x.size() / sizeof(float);
    for (size_t i = 0; i < count; i++)
        write_f32(out.data, i, std::max(0.0f, read_f32(x, i)));

    return out;
}

} // namespace sandy::core
