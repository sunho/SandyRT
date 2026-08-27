#include "CudaJitAttentionAbi.cuh"
#include "CudaJitTensorAccess.cuh"

// Prefill has a separate highlighted source module. It remains on the compiled
// CUDA fallback during the decode-only attention migration.
