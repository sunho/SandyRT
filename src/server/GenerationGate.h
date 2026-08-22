#pragma once

#include "RequestControl.h"
#include "Result.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>

namespace sandy::server {

class GenerationGate {
public:
    class Lease {
    public:
        explicit Lease(GenerationGate& gate) : gate_(&gate) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease();

    private:
        GenerationGate* gate_;
    };

    Result<std::unique_ptr<Lease>> acquire(const RequestControl* control);

private:
    void release();

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<uint64_t> pending_;
    uint64_t nextTicket_ = 0;
    bool active_ = false;

    friend class Lease;
};

} // namespace sandy::server
