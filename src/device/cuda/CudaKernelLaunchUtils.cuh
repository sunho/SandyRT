#pragma once

#include "CudaKernels.h"

#include <string>

namespace sandy::device {

inline Result<void> validate_context(
        const CudaLaunchContext& context,
        size_t minInputs,
        size_t minOutputs,
        const char* kernelName) {
    if (!context.stream)
        return make_error(std::string("cuda ") + kernelName + " launch has null stream");
    if (context.inputs.size() < minInputs)
        return make_error(std::string("cuda ") + kernelName + " input arity mismatch");
    if (context.outputs.size() < minOutputs)
        return make_error(std::string("cuda ") + kernelName + " output arity mismatch");
    for (const auto& input : context.inputs) {
        if (!input.data && input.bytes != 0)
            return make_error(std::string("cuda ") + kernelName + " input buffer is null");
    }
    for (const auto& output : context.outputs) {
        if (!output.data && output.bytes != 0)
            return make_error(std::string("cuda ") + kernelName + " output buffer is null");
    }
    return {};
}

inline Result<void> unimplemented(const char* kernelName) {
    return make_error(std::string("cuda ") + kernelName + " kernel is not implemented yet");
}

inline bool is_float_compute_dtype(core::DType dtype) {
    return dtype == core::DType::F32 || dtype == core::DType::BF16;
}

} // namespace sandy::device
