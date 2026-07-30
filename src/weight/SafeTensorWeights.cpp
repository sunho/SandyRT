#include "SafeTensorWeights.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <utility>

namespace sandy::weight {

static core::DType parse_dtype(const std::string& s) {
    if (s == "F32") return core::DType::F32;
    if (s == "F16") return core::DType::F16;
    if (s == "BF16") return core::DType::BF16;
    if (s == "I32") return core::DType::I32;
    if (s == "I64") return core::DType::I64;
    if (s == "U8" || s == "BOOL") return core::DType::U8;
    fprintf(stderr, "unsupported safetensors dtype: %s\n", s.c_str());
    abort();
}

EagerSafeTensorsBuffer::EagerSafeTensorsBuffer(core::TensorDesc desc,
                                               const uint8_t* data,
                                               size_t size)
    : TensorBuffer(std::move(desc)), data_(data), size_(size) {}

Result<void> EagerSafeTensorsBuffer::load() {
    return {};
}

void EagerSafeTensorsBuffer::unload() {}

std::span<const uint8_t> EagerSafeTensorsBuffer::data() const {
    if (!is_mounted()) {
        fprintf(stderr, "EagerSafeTensorsBuffer::data() called while unmounted\n");
        abort();
    }
    return {data_, size_};
}

std::unique_ptr<EagerSafeTensorWeights> EagerSafeTensorWeights::load(
        const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        abort();
    }

    size_t fileSize = file.tellg();
    file.seekg(0);

    auto w = std::make_unique<EagerSafeTensorWeights>();
    w->data_.resize(fileSize);
    file.read(reinterpret_cast<char*>(w->data_.data()), fileSize);

    if (fileSize < 8) {
        fprintf(stderr, "safetensors file too small\n");
        abort();
    }

    uint64_t headerSize = 0;
    memcpy(&headerSize, w->data_.data(), 8);
    w->dataOffset_ = 8 + headerSize;

    if (w->dataOffset_ > fileSize) {
        fprintf(stderr, "safetensors header size exceeds file\n");
        abort();
    }

    std::string headerStr(
        reinterpret_cast<char*>(w->data_.data() + 8), headerSize);
    auto header = nlohmann::json::parse(headerStr);

    for (auto& [key, val] : header.items()) {
        if (key == "__metadata__") continue;

        auto dtype = parse_dtype(val["dtype"].get<std::string>());
        auto shapeArr = val["shape"].get<std::vector<int64_t>>();
        auto offsets = val["data_offsets"].get<std::vector<size_t>>();

        TensorInfo info;
        info.shape = core::Shape(std::move(shapeArr));
        info.dtype = dtype;
        info.offset = offsets[0];
        info.size = offsets[1] - offsets[0];

        w->names_.push_back(key);
        w->tensors_[key] = std::move(info);
    }

    return w;
}

std::vector<core::TensorDesc> EagerSafeTensorWeights::descriptors() const {
    std::vector<core::TensorDesc> descs;
    for (auto& name : names_) {
        auto& info = tensors_.at(name);
        descs.push_back({name, info.shape, info.dtype});
    }
    return descs;
}

std::shared_ptr<core::TensorBuffer> EagerSafeTensorWeights::get_tensor(
    const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;

    const auto& info = it->second;
    core::TensorDesc desc(name, info.shape, info.dtype);
    return std::make_shared<EagerSafeTensorsBuffer>(
        std::move(desc), data_.data() + dataOffset_ + info.offset, info.size);
}

} // namespace sandy::weight
