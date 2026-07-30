#include "SafeTensorWeights.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace weight {

static ir::DType parse_dtype(const std::string& s) {
    if (s == "F32") return ir::DType::F32;
    if (s == "F16") return ir::DType::F16;
    if (s == "BF16") return ir::DType::BF16;
    if (s == "I32") return ir::DType::I32;
    if (s == "I64") return ir::DType::I64;
    if (s == "U8" || s == "BOOL") return ir::DType::U8;
    fprintf(stderr, "unsupported safetensors dtype: %s\n", s.c_str());
    abort();
}

EagerSafeTensorWeights EagerSafeTensorWeights::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        abort();
    }

    size_t fileSize = file.tellg();
    file.seekg(0);

    EagerSafeTensorWeights w;
    w.data_.resize(fileSize);
    file.read(reinterpret_cast<char*>(w.data_.data()), fileSize);

    if (fileSize < 8) {
        fprintf(stderr, "safetensors file too small\n");
        abort();
    }

    uint64_t headerSize = 0;
    memcpy(&headerSize, w.data_.data(), 8);
    w.dataOffset_ = 8 + headerSize;

    if (w.dataOffset_ > fileSize) {
        fprintf(stderr, "safetensors header size exceeds file\n");
        abort();
    }

    std::string headerStr(
        reinterpret_cast<char*>(w.data_.data() + 8), headerSize);
    auto header = nlohmann::json::parse(headerStr);

    for (auto& [key, val] : header.items()) {
        if (key == "__metadata__") continue;

        auto dtype = parse_dtype(val["dtype"].get<std::string>());
        auto shapeArr = val["shape"].get<std::vector<int64_t>>();
        auto offsets = val["data_offsets"].get<std::vector<size_t>>();

        TensorInfo info;
        info.shape = ir::Shape(std::move(shapeArr));
        info.dtype = dtype;
        info.offset = offsets[0];
        info.size = offsets[1] - offsets[0];

        w.names_.push_back(key);
        w.tensors_[key] = std::move(info);
    }

    return w;
}

std::vector<ir::TensorDesc> EagerSafeTensorWeights::get_descriptors() const {
    std::vector<ir::TensorDesc> descs;
    for (auto& name : names_) {
        auto& info = tensors_.at(name);
        descs.push_back({name, info.shape, info.dtype});
    }
    return descs;
}

ir::TensorDesc EagerSafeTensorWeights::get_descriptor(
    const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        fprintf(stderr, "weight not found: %s\n", name.c_str());
        abort();
    }
    return {name, it->second.shape, it->second.dtype};
}

bool EagerSafeTensorWeights::has(const std::string& name) const {
    return tensors_.count(name) > 0;
}

std::span<const uint8_t> EagerSafeTensorWeights::get_buffer(
    const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        fprintf(stderr, "weight not found: %s\n", name.c_str());
        abort();
    }
    return {data_.data() + dataOffset_ + it->second.offset, it->second.size};
}

} // namespace weight
