#include "CudaKernels.h"

namespace sandy::engine {

namespace {

Result<void> validate_context(
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

Result<void> unimplemented(const char* kernelName) {
    return make_error(std::string("cuda ") + kernelName + " kernel is not implemented yet");
}

} // namespace

Result<void> launch_cuda_elementwise(
        const CudaLaunchContext& context,
        const CudaElementwiseProgram& program) {
    auto valid = validate_context(
        context,
        program.elementwiseInputs.size(),
        1,
        "elementwise");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("elementwise");
}

Result<void> launch_cuda_layout_transform(
        const CudaLaunchContext& context,
        const CudaLayoutTransformProgram&) {
    auto valid = validate_context(context, 1, 1, "layout_transform");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("layout_transform");
}

Result<void> launch_cuda_matmul(
        const CudaLaunchContext& context,
        const CudaMatMulProgram&) {
    auto valid = validate_context(context, 2, 1, "matmul");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("matmul");
}

Result<void> launch_cuda_gather(const CudaLaunchContext& context) {
    auto valid = validate_context(context, 2, 1, "gather");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("gather");
}

Result<void> launch_cuda_softmax(
        const CudaLaunchContext& context,
        const CudaSoftmaxProgram&) {
    auto valid = validate_context(context, 1, 1, "softmax");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("softmax");
}

Result<void> launch_cuda_norm(
        const CudaLaunchContext& context,
        const CudaNormProgram&) {
    auto valid = validate_context(context, 1, 1, "norm");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("norm");
}

Result<void> launch_cuda_rope(
        const CudaLaunchContext& context,
        const CudaRoPEProgram&) {
    auto valid = validate_context(context, 1, 1, "rope");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("rope");
}

Result<void> launch_cuda_sliding_query_key_score(
        const CudaLaunchContext& context,
        const CudaSlidingQueryKeyScoreProgram&) {
    auto valid = validate_context(context, 2, 1, "sliding_query_key_score");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("sliding_query_key_score");
}

Result<void> launch_cuda_reduction(
        const CudaLaunchContext& context,
        const CudaReductionProgram&) {
    auto valid = validate_context(context, 1, 1, "reduction");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("reduction");
}

Result<void> launch_cuda_custom(
        const CudaLaunchContext& context,
        const CudaCustomProgram& program) {
    auto valid = validate_context(context, 0, 1, "custom");
    if (!valid)
        return make_error(valid.error());
    return unimplemented(program.customName.empty() ? "custom" : program.customName.c_str());
}

} // namespace sandy::engine
