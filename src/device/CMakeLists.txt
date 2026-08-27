add_library(DeviceLib
  Device.cpp
  DeviceWiseCopier.cpp
  cpu/CpuDevice.cpp
  cpu/MidIRCpuInterpreter.cpp
)

if(SANDY_ENABLE_CUDA)
  set(SANDY_JIT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda/jit/generated")
  file(MAKE_DIRECTORY "${SANDY_JIT_GENERATED_DIR}")

  function(sandy_embed_cuda_source source symbol output_name)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    file(READ "${source}" source_contents)
    string(FIND "${source_contents}" ")SANDY_JIT_SOURCE\"" delimiter_collision)
    if(NOT delimiter_collision EQUAL -1)
      message(FATAL_ERROR "JIT source contains reserved raw-string delimiter: ${source}")
    endif()
    file(WRITE "${SANDY_JIT_GENERATED_DIR}/${output_name}"
      "#pragma once\nnamespace sandy::device::cuda_jit_embedded {\n"
      "inline constexpr char ${symbol}[] = R\"SANDY_JIT_SOURCE(${source_contents})SANDY_JIT_SOURCE\";\n"
      "}\n")
  endfunction()

  set(SANDY_JIT_TEMPLATE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda/jit/templates")
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitAbi.cuh"
    kAbi CudaJitAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitTensorAccess.cuh"
    kTensorAccess CudaJitTensorAccessSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitElementwiseTemplate.cuh"
    kElementwiseTemplate CudaJitElementwiseTemplateSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitElementwiseKernel.cu"
    kElementwiseKernel CudaJitElementwiseKernelSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/generated/ElementwiseEvaluator.cuh"
    kEvaluatorStub CudaJitEvaluatorStubSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/generated/ElementwiseConfig.cuh"
    kElementwiseConfigStub CudaJitElementwiseConfigStubSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitReductionAbi.cuh"
    kReductionAbi CudaJitReductionAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitReductionKernel.cu"
    kReductionKernel CudaJitReductionKernelSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitLayoutTransformAbi.cuh"
    kLayoutTransformAbi CudaJitLayoutTransformAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitLayoutTransformKernel.cu"
    kLayoutTransformKernel CudaJitLayoutTransformKernelSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitGatherAbi.cuh"
    kGatherAbi CudaJitGatherAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitGatherKernel.cu"
    kGatherKernel CudaJitGatherKernelSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitSoftmaxAbi.cuh"
    kSoftmaxAbi CudaJitSoftmaxAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitSoftmaxKernel.cu"
    kSoftmaxKernel CudaJitSoftmaxKernelSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitNormAbi.cuh"
    kNormAbi CudaJitNormAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitNormKernel.cu"
    kNormKernel CudaJitNormKernelSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitRoPEAbi.cuh"
    kRoPEAbi CudaJitRoPEAbiSource.generated.h)
  sandy_embed_cuda_source("${SANDY_JIT_TEMPLATE_DIR}/CudaJitRoPEKernel.cu"
    kRoPEKernel CudaJitRoPEKernelSource.generated.h)

  target_sources(DeviceLib
    PRIVATE
    cuda/CudaRuntime.cpp
    cuda/jit/CudaElementwiseJit.cpp
    cuda/jit/CudaJit.cpp
    cuda/jit/CudaJitEmbeddedSources.cpp
    cuda/jit/CudaGatherJit.cpp
    cuda/jit/CudaLayoutTransformJit.cpp
    cuda/jit/CudaNormJit.cpp
    cuda/jit/CudaReductionJit.cpp
    cuda/jit/CudaRoPEJit.cpp
    cuda/jit/CudaSoftmaxJit.cpp
    cuda/CudaScratchAllocator.cpp
    cuda/CudaDevice.cpp
    cuda/CudaPagedTensor.cpp
    cuda/CudaAttentionKernels.cu
    cuda/CudaElementwiseKernels.cu
    cuda/CudaGatherKernels.cu
    cuda/CudaKernels.cu
    cuda/CudaLayoutTransformKernels.cu
    cuda/CudaMatMulKernels.cu
    cuda/CudaMoeScatterSumKernels.cu
    cuda/CudaNormKernels.cu
    cuda/CudaRoPEKernels.cu
    cuda/CudaSoftmaxKernels.cu
    cuda/CudaTopKKernels.cu
  )
endif()

target_include_directories(DeviceLib
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/jit
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/jit/templates
  ${SANDY_JIT_GENERATED_DIR}
)

target_link_libraries(DeviceLib
  PUBLIC
  CoreLib
  IRLib
)

if(SANDY_ENABLE_CUDA)
  target_link_libraries(DeviceLib
    PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cuda_driver
    CUDA::nvrtc
  )
endif()

set_target_properties(DeviceLib PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

if(SANDY_ENABLE_CUDA)
  set_target_properties(DeviceLib PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  set_source_files_properties(
    cuda/CudaElementwiseKernels.cu
    cuda/CudaAttentionKernels.cu
    cuda/CudaGatherKernels.cu
    cuda/CudaKernels.cu
    cuda/CudaLayoutTransformKernels.cu
    cuda/CudaMatMulKernels.cu
    cuda/CudaMoeScatterSumKernels.cu
    cuda/CudaNormKernels.cu
    cuda/CudaRoPEKernels.cu
    cuda/CudaSoftmaxKernels.cu
    cuda/CudaTopKKernels.cu
    PROPERTIES LANGUAGE CUDA
  )
endif()
