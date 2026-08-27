#include "CudaRoPEJit.h"
#include "CudaJitEmbeddedSources.h"
#include "CudaJitPolicy.h"
#include <sstream>
namespace sandy::device {
Result<CudaJitCache::KernelPtr> compileCudaRoPEJit(
        int device, CudaJitCache& cache, core::DType dtype,
        bool splitHalf, bool hasPositions, core::DType positionDtype,
        int inputAccess, int positionAccess, int outputAccess) {
    std::ostringstream c;
    c << "#pragma once\ninline constexpr int SandyRoPEDType = "
      << (dtype == core::DType::F32 ? "SANDY_JIT_F32" : "SANDY_JIT_BF16")
      << ";\ninline constexpr bool SandyRoPESplitHalf = " << (splitHalf ? "true" : "false")
      << ";\ninline constexpr bool SandyRoPEHasPositions = " << (hasPositions ? "true" : "false")
      << ";\nusing SandyRoPEPositionType = "
      << (positionDtype == core::DType::I32 ? "int32_t" : "int64_t") << ";\n";
    const int accesses[] = {inputAccess, positionAccess, outputAccess};
    const char* names[] = {"SandyRoPEInputAccess", "SandyRoPEPositionAccess", "SandyRoPEOutputAccess"};
    for (int i = 0; i < 3; ++i) {
        auto a = cudaJitAccessConstant(accesses[i]);
        if (!a) return make_error(a.error());
        c << "inline constexpr int " << names[i] << " = " << *a << ";\n";
    }
    CudaJitRequest r{"CudaJitRoPEKernel.cu", std::string(embeddedRoPEKernelSource()),
        embeddedRoPEHeaders(), "sandy_jit_rope", {"-lineinfo"}};
    r.headers.push_back({"CudaJitRoPEConfig.cuh", c.str()});
    return cache.getOrCompile(device, r);
}
}
