#pragma once

#include <chrono>
#include <functional>

namespace sandy::server {

enum class RequestStopReason {
    None,
    ClientCancelled,
    DeadlineExceeded,
};

struct RequestControl {
    using Clock = std::chrono::steady_clock;

    Clock::time_point submittedAt = Clock::now();
    Clock::time_point deadline = Clock::time_point::max();
    std::function<bool()> clientCancelled;

    RequestStopReason stopReason() const {
        if (clientCancelled && clientCancelled())
            return RequestStopReason::ClientCancelled;
        if (Clock::now() >= deadline)
            return RequestStopReason::DeadlineExceeded;
        return RequestStopReason::None;
    }
};

inline const char* requestStopMessage(RequestStopReason reason) {
    switch (reason) {
        case RequestStopReason::ClientCancelled:
            return "request cancelled";
        case RequestStopReason::DeadlineExceeded:
            return "request deadline exceeded";
        case RequestStopReason::None:
            return "";
    }
    return "request stopped";
}

} // namespace sandy::server
