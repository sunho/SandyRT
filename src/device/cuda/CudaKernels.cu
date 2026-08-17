#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"

namespace sandy::device {

Result<void> launch_cuda_layout_transform(
        const CudaLaunchContext& context,
        const CudaLayoutTransformProgram&) {
    auto valid = validate_context(context, 1, 1, "layout_transform");
    if (!valid)
        return make_error(valid.error());
    return unimplemented("layout_transform");
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

} // namespace sandy::device
