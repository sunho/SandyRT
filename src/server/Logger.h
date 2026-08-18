#pragma once

#include "Engine.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sandy::server {

struct LoggerConfig {
    bool debug = false;
    bool profile = false;
    std::string requestLogDir = "logs/requests";
};

class RequestLogger {
public:
    static std::unique_ptr<RequestLogger> create(
        const LoggerConfig& config,
        const std::string& requestId);

    RequestLogger(const RequestLogger&) = delete;
    RequestLogger& operator=(const RequestLogger&) = delete;
    ~RequestLogger();

    bool enabled() const { return stream_.is_open(); }
    bool profileEnabled() const { return profile_; }
    const std::string& path() const { return path_; }

    void log(std::string_view message);
    void logf(const char* format, ...);
    void logProfileKernel(
        std::string_view phase,
        const engine::EngineProfileEvent& event);
    void logProfileStage(
        std::string_view phase,
        const engine::EngineProfileStageEvent& event);
    void logDeviceBoundary(
        std::string_view phase,
        const engine::EngineDeviceRunBoundaryEvent& event);
    void logServerStage(
        std::string_view phase,
        std::string_view stage,
        double elapsedMs);

    struct AggregateStats {
        size_t count = 0;
        double totalMs = 0.0;
        double maxMs = 0.0;

        void add(double elapsedMs);
        void add(const AggregateStats& other);
        double avg() const;
    };

    struct KernelAggregate {
        std::string phase;
        std::string kind;
        uint32_t device = 0;
        engine::DeviceCompiledGraphId deviceGraph = 0;
        AggregateStats stats;
    };

    struct StageAggregate {
        std::string phase;
        std::string stage;
        std::string kind;
        AggregateStats stats;
    };

    struct DeviceRunAggregate {
        std::string phase;
        std::string kind;
        uint32_t device = 0;
        engine::DeviceCompiledGraphId deviceGraph = 0;
        size_t beginCount = 0;
        size_t endCount = 0;
        AggregateStats endStats;
    };

private:
    RequestLogger(
        bool debug,
        bool profile,
        std::string requestId,
        std::string path,
        std::ofstream stream);
    void flushProfileSummaries();

    bool debug_ = false;
    bool profile_ = false;
    std::string requestId_;
    std::string path_;
    std::ofstream stream_;
    std::mutex mutex_;
    std::unordered_map<std::string, StageAggregate> serverStageAggregates_;
    std::unordered_map<std::string, KernelAggregate> kernelAggregates_;
    std::unordered_map<std::string, StageAggregate> stageAggregates_;
    std::unordered_map<std::string, DeviceRunAggregate> deviceRunAggregates_;
};

class ServerStageScope {
public:
    ServerStageScope(RequestLogger* logger, std::string phase, std::string stage);
    ServerStageScope(const ServerStageScope&) = delete;
    ServerStageScope& operator=(const ServerStageScope&) = delete;
    ~ServerStageScope();

private:
    RequestLogger* logger_ = nullptr;
    std::string phase_;
    std::string stage_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace sandy::server
