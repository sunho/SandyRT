#pragma once

#include "TensorBuffer.h"
#include "Weight.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::weight {

class EagerSafeTensorsBuffer : public core::TensorBuffer {
public:
    EagerSafeTensorsBuffer(core::TensorDesc desc, const uint8_t* data, size_t size);

private:
    Result<void> load() override;
    void unload() override;
    std::span<const uint8_t> data() const override;

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

class EagerSafeTensorWeights : public Weights {
public:
    static std::unique_ptr<EagerSafeTensorWeights> load(const std::string& path);

    std::vector<core::TensorDesc> descriptors() const override;
    std::shared_ptr<core::TensorBuffer> get_tensor(
        const std::string& name) const override;

private:
    struct TensorInfo {
        core::Shape shape;
        core::DType dtype;
        size_t offset;
        size_t size;
    };

    std::vector<uint8_t> data_;
    size_t dataOffset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::vector<std::string> names_;
};

} // namespace sandy::weight
