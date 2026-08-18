#include "CudaKernels.h"
#include "CudaKernelLaunchUtils.cuh"

namespace sandy::device {

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

} // namespace sandy::device
