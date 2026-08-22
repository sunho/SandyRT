#include "GenerationGate.h"

#include <algorithm>
#include <chrono>

namespace sandy::server {

GenerationGate::Lease::~Lease() {
    if (gate_)
        gate_->release();
}

Result<std::unique_ptr<GenerationGate::Lease>> GenerationGate::acquire(
        const RequestControl* control) {
    std::unique_lock<std::mutex> lock(mutex_);
    uint64_t ticket = nextTicket_++;
    pending_.push_back(ticket);

    for (;;) {
        auto reason = control ? control->stopReason() : RequestStopReason::None;
        if (reason != RequestStopReason::None) {
            auto position = std::find(pending_.begin(), pending_.end(), ticket);
            if (position != pending_.end())
                pending_.erase(position);
            changed_.notify_all();
            return make_error(requestStopMessage(reason));
        }

        if (!active_ && !pending_.empty() && pending_.front() == ticket) {
            pending_.pop_front();
            active_ = true;
            return std::make_unique<Lease>(*this);
        }

        changed_.wait_for(lock, std::chrono::milliseconds(20));
    }
}

void GenerationGate::release() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
    }
    changed_.notify_all();
}

} // namespace sandy::server
