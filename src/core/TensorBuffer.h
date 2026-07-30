#pragma once

#include "Result.h"
#include "Tensor.h"

#include <cstdint>
#include <span>

namespace sandy::core {

class TensorBuffer {
public:
    class Access {
    public:
        Access(const Access&) = delete;
        Access& operator=(const Access&) = delete;
        Access(Access&& other) noexcept;
        Access& operator=(Access&& other) noexcept;
        ~Access();

        const TensorDesc& desc() const;
        std::span<const uint8_t> data() const;

    private:
        friend class TensorBuffer;
        explicit Access(TensorBuffer& buffer);

        TensorBuffer* buffer_;
    };

    virtual ~TensorBuffer() = default;

    const TensorDesc& desc() const;
    Result<void> mount();
    void unmount();
    Result<Access> access();

protected:
    explicit TensorBuffer(TensorDesc desc);

    bool is_mounted() const;

private:
    virtual Result<void> load() = 0;
    virtual void unload() = 0;
    virtual std::span<const uint8_t> data() const = 0;

    TensorDesc desc_;
    int mountDepth_ = 0;
};

} // namespace sandy::core
