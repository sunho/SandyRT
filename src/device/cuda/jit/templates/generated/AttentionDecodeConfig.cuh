#pragma once

inline constexpr int SandyAttentionDType = SANDY_JIT_BF16;
inline constexpr int SandyAttentionQAccess = SANDY_JIT_CONTIGUOUS;
inline constexpr int SandyAttentionOutputAccess = SANDY_JIT_CONTIGUOUS;
inline constexpr int SandyAttentionPositionDType = SANDY_JIT_I64;
inline constexpr int SandyAttentionPositionAccess = SANDY_JIT_CONTIGUOUS;
inline constexpr int SandyAttentionHeadDim = 256;
inline constexpr int SandyAttentionRank = 4;
inline constexpr int SandyAttentionQueryHeadsPerKv = 2;
inline constexpr bool SandyAttentionHasPositionOffsets = true;
inline constexpr int64_t SandyAttentionWindow = 4096;
inline constexpr float SandyAttentionScale = 0.0625f;
