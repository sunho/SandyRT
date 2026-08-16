#include "CudaRuntime.h"

namespace sandy::device {

Result<void> cuda_check(cudaError_t status, const std::string& context) {
    if (status == cudaSuccess)
        return {};
    return make_error(context + ": " + cudaGetErrorString(status));
}

} // namespace sandy::device
