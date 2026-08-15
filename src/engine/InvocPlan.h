#pragma once

#include "Device.h"
#include "Tensor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sandy::engine {

using InvocDeviceId = uint32_t;
using InvocValueId = uint32_t;
using InvocProgramId = uint32_t;

enum class InvocInstructionKind {
    Alloc,
    Dealloc,
    LoadInput,
    LoadWeight,
    RunKernel,
};

struct InvocAlloc {
    InvocDeviceId device = 0;
    InvocValueId value = 0;
    core::TensorDesc desc;
};

struct InvocDealloc {
    InvocDeviceId device = 0;
    InvocValueId value = 0;
};

struct InvocLoadInput {
    InvocDeviceId device = 0;
    int64_t index = 0;
    InvocValueId value = 0;
};

struct InvocLoadWeight {
    InvocDeviceId device = 0;
    std::string name;
    InvocValueId value = 0;
};

struct InvocRunKernel {
    InvocDeviceId device = 0;
    InvocProgramId program = 0;
    std::vector<InvocValueId> inputs;
    std::vector<InvocValueId> outputs;
};

using InvocInstructionPayload = std::variant<
    InvocAlloc,
    InvocDealloc,
    InvocLoadInput,
    InvocLoadWeight,
    InvocRunKernel>;

struct InvocInstruction {
    InvocInstructionKind kind = InvocInstructionKind::Alloc;
    InvocInstructionPayload payload;

    static InvocInstruction alloc(InvocAlloc value) {
        return {InvocInstructionKind::Alloc, std::move(value)};
    }

    static InvocInstruction dealloc(InvocDealloc value) {
        return {InvocInstructionKind::Dealloc, std::move(value)};
    }

    static InvocInstruction load_input(InvocLoadInput value) {
        return {InvocInstructionKind::LoadInput, std::move(value)};
    }

    static InvocInstruction load_weight(InvocLoadWeight value) {
        return {InvocInstructionKind::LoadWeight, std::move(value)};
    }

    static InvocInstruction run_kernel(InvocRunKernel value) {
        return {InvocInstructionKind::RunKernel, std::move(value)};
    }
};

struct InvocProgram {
    InvocProgramId id = 0;
    InvocDeviceId device = 0;
    DeviceProgramId deviceProgram = 0;
};

struct InvocPlan {
    std::vector<InvocProgram> programs;
    std::vector<InvocInstruction> instructions;
    std::vector<InvocValueId> outputs;
};

} // namespace sandy::engine
