#include "Logger.h"

#include "KernelIR.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sandy::server {

namespace {

std::atomic<uint64_t> g_requestLogCounter{0};

std::string timestamp_utc() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%dT%H%M%SZ", &tm);
    return buffer;
}

std::string sanitize_filename(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        auto c = static_cast<unsigned char>(ch);
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            ch == '_' ||
            ch == '-' ||
            ch == '.') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
        if (out.size() >= 80)
            break;
    }
    if (out.empty())
        out = "request";
    return out;
}

std::string aggregate_key(std::initializer_list<std::string_view> parts) {
    std::string key;
    for (auto part : parts) {
        key.append(part.data(), part.size());
        key.push_back('\x1f');
    }
    return key;
}

std::string phase_bucket(std::string_view phase) {
    if (phase.starts_with("prefill_"))
        return "prefill";
    if (phase.starts_with("decode_step_"))
        return "decode";
    if (phase.starts_with("prompt_token_"))
        return "prompt_eval";
    return std::string(phase);
}

template<typename Aggregate>
std::vector<Aggregate> map_values(const std::unordered_map<std::string, Aggregate>& map) {
    std::vector<Aggregate> values;
    values.reserve(map.size());
    for (const auto& [_, aggregate] : map)
        values.push_back(aggregate);
    return values;
}

template<typename Aggregate>
void sort_by_total_desc(std::vector<Aggregate>& aggregates) {
    std::sort(aggregates.begin(), aggregates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.stats.totalMs != rhs.stats.totalMs)
            return lhs.stats.totalMs > rhs.stats.totalMs;
        if (lhs.phase != rhs.phase)
            return lhs.phase < rhs.phase;
        if constexpr (requires { lhs.stage; rhs.stage; }) {
            if (lhs.stage != rhs.stage)
                return lhs.stage < rhs.stage;
        }
        if (lhs.kind != rhs.kind)
            return lhs.kind < rhs.kind;
        if constexpr (requires { lhs.device; rhs.device; }) {
            return lhs.device < rhs.device;
        }
        return false;
    });
}

void sort_device_runs_by_total_desc(
        std::vector<RequestLogger::DeviceRunAggregate>& aggregates) {
    std::sort(aggregates.begin(), aggregates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.endStats.totalMs != rhs.endStats.totalMs)
            return lhs.endStats.totalMs > rhs.endStats.totalMs;
        if (lhs.phase != rhs.phase)
            return lhs.phase < rhs.phase;
        if (lhs.kind != rhs.kind)
            return lhs.kind < rhs.kind;
        return lhs.device < rhs.device;
    });
}

std::vector<RequestLogger::KernelAggregate> kernel_bucket_values(
        const std::vector<RequestLogger::KernelAggregate>& kernels) {
    std::unordered_map<std::string, RequestLogger::KernelAggregate> buckets;
    for (const auto& kernel : kernels) {
        auto bucketPhase = phase_bucket(kernel.phase);
        auto device = std::to_string(kernel.device);
        auto deviceGraph = std::to_string(kernel.deviceGraph);
        auto key = aggregate_key({bucketPhase, kernel.kind, device, deviceGraph});
        auto& bucket = buckets[key];
        if (bucket.stats.count == 0) {
            bucket.phase = std::move(bucketPhase);
            bucket.kind = kernel.kind;
            bucket.device = kernel.device;
            bucket.deviceGraph = kernel.deviceGraph;
        }
        bucket.stats.add(kernel.stats);
    }
    return map_values(buckets);
}

std::vector<RequestLogger::StageAggregate> stage_bucket_values(
        const std::vector<RequestLogger::StageAggregate>& stages) {
    std::unordered_map<std::string, RequestLogger::StageAggregate> buckets;
    for (const auto& stage : stages) {
        auto bucketPhase = phase_bucket(stage.phase);
        auto key = aggregate_key({bucketPhase, stage.stage, stage.kind});
        auto& bucket = buckets[key];
        if (bucket.stats.count == 0) {
            bucket.phase = std::move(bucketPhase);
            bucket.stage = stage.stage;
            bucket.kind = stage.kind;
        }
        bucket.stats.add(stage.stats);
    }
    return map_values(buckets);
}

std::vector<RequestLogger::DeviceRunAggregate> device_run_bucket_values(
        const std::vector<RequestLogger::DeviceRunAggregate>& runs) {
    std::unordered_map<std::string, RequestLogger::DeviceRunAggregate> buckets;
    for (const auto& run : runs) {
        auto bucketPhase = phase_bucket(run.phase);
        auto device = std::to_string(run.device);
        auto deviceGraph = std::to_string(run.deviceGraph);
        auto key = aggregate_key({bucketPhase, run.kind, device, deviceGraph});
        auto& bucket = buckets[key];
        if (bucket.beginCount == 0 && bucket.endCount == 0) {
            bucket.phase = std::move(bucketPhase);
            bucket.kind = run.kind;
            bucket.device = run.device;
            bucket.deviceGraph = run.deviceGraph;
        }
        bucket.beginCount += run.beginCount;
        bucket.endCount += run.endCount;
        bucket.endStats.add(run.endStats);
    }
    return map_values(buckets);
}

} // namespace

std::unique_ptr<RequestLogger> RequestLogger::create(
        const LoggerConfig& config,
        const std::string& requestId) {
    if (!config.debug && !config.profile)
        return nullptr;

    std::error_code ec;
    std::filesystem::create_directories(config.requestLogDir, ec);
    if (ec) {
        std::fprintf(
            stderr,
            "failed to create request log directory %s: %s\n",
            config.requestLogDir.c_str(),
            ec.message().c_str());
        return nullptr;
    }

    auto counter = g_requestLogCounter.fetch_add(1, std::memory_order_relaxed);
    auto name = timestamp_utc() + "_" + std::to_string(counter) + "_" +
        sanitize_filename(requestId) + ".log";
    auto path = (std::filesystem::path(config.requestLogDir) / name).string();

    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream.is_open()) {
        std::fprintf(stderr, "failed to open request log: %s\n", path.c_str());
        return nullptr;
    }

    auto logger = std::unique_ptr<RequestLogger>(new RequestLogger(
        config.debug,
        config.profile,
        requestId,
        path,
        std::move(stream)));
    logger->logf(
        "request_log.start request_id=%s debug=%d profile=%d path=%s",
        requestId.c_str(),
        config.debug ? 1 : 0,
        config.profile ? 1 : 0,
        path.c_str());
    std::fprintf(
        stderr,
        "request log: request_id=%s path=%s\n",
        requestId.c_str(),
        path.c_str());
    return logger;
}

RequestLogger::RequestLogger(
        bool debug,
        bool profile,
        std::string requestId,
        std::string path,
        std::ofstream stream)
    : debug_(debug),
      profile_(profile),
      requestId_(std::move(requestId)),
      path_(std::move(path)),
      stream_(std::move(stream)) {}

RequestLogger::~RequestLogger() {
    if (!stream_.is_open())
        return;
    flushProfileSummaries();
    log("request_log.end");
    stream_.flush();
}

void RequestLogger::AggregateStats::add(double elapsedMs) {
    count++;
    totalMs += elapsedMs;
    maxMs = std::max(maxMs, elapsedMs);
}

void RequestLogger::AggregateStats::add(const AggregateStats& other) {
    count += other.count;
    totalMs += other.totalMs;
    maxMs = std::max(maxMs, other.maxMs);
}

double RequestLogger::AggregateStats::avg() const {
    return count == 0 ? 0.0 : totalMs / static_cast<double>(count);
}

void RequestLogger::log(std::string_view message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open())
        return;
    stream_ << timestamp_utc() << " " << message << '\n';
}

void RequestLogger::logf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    char stackBuffer[1024];
    va_list argsCopy;
    va_copy(argsCopy, args);
    auto written = std::vsnprintf(stackBuffer, sizeof(stackBuffer), format, args);
    va_end(args);
    if (written < 0) {
        va_end(argsCopy);
        return;
    }
    if (static_cast<size_t>(written) < sizeof(stackBuffer)) {
        va_end(argsCopy);
        log(stackBuffer);
        return;
    }

    std::vector<char> heapBuffer(static_cast<size_t>(written) + 1);
    std::vsnprintf(heapBuffer.data(), heapBuffer.size(), format, argsCopy);
    va_end(argsCopy);
    log(heapBuffer.data());
}

void RequestLogger::logProfileKernel(
        std::string_view phase,
        const engine::EngineProfileEvent& event) {
    if (!profile_)
        return;
    auto kind = std::string(ir::kernel_ir::op_kind_name(event.opKind));
    auto device = std::to_string(event.device);
    auto deviceGraph = std::to_string(event.deviceGraph);
    auto key = aggregate_key({phase, kind, device, deviceGraph});

    std::lock_guard<std::mutex> lock(mutex_);
    auto& aggregate = kernelAggregates_[key];
    if (aggregate.stats.count == 0) {
        aggregate.phase = std::string(phase);
        aggregate.kind = std::move(kind);
        aggregate.device = event.device;
        aggregate.deviceGraph = event.deviceGraph;
    }
    aggregate.stats.add(event.elapsedMs);
}

void RequestLogger::logProfileStage(
        std::string_view phase,
        const engine::EngineProfileStageEvent& event) {
    if (!profile_)
        return;
    auto kind = std::string(ir::kernel_ir::op_kind_name(event.opKind));
    auto key = aggregate_key({phase, event.stage, kind});

    std::lock_guard<std::mutex> lock(mutex_);
    auto& aggregate = stageAggregates_[key];
    if (aggregate.stats.count == 0) {
        aggregate.phase = std::string(phase);
        aggregate.stage = event.stage;
        aggregate.kind = std::move(kind);
    }
    aggregate.stats.add(event.elapsedMs);
}

void RequestLogger::logDeviceBoundary(
        std::string_view phase,
        const engine::EngineDeviceRunBoundaryEvent& event) {
    if (!profile_)
        return;
    auto kind = std::string(ir::kernel_ir::op_kind_name(event.opKind));
    auto device = std::to_string(event.device);
    auto deviceGraph = std::to_string(event.deviceGraph);
    auto key = aggregate_key({phase, kind, device, deviceGraph});

    std::lock_guard<std::mutex> lock(mutex_);
    auto& aggregate = deviceRunAggregates_[key];
    if (aggregate.beginCount == 0 && aggregate.endCount == 0) {
        aggregate.phase = std::string(phase);
        aggregate.kind = std::move(kind);
        aggregate.device = event.device;
        aggregate.deviceGraph = event.deviceGraph;
    }
    if (event.boundary == engine::EngineDeviceRunBoundaryEvent::Boundary::Begin) {
        aggregate.beginCount++;
    } else {
        aggregate.endCount++;
        aggregate.endStats.add(event.elapsedMs);
    }
}

void RequestLogger::logServerStage(
        std::string_view phase,
        std::string_view stage,
        double elapsedMs) {
    if (profile_) {
        auto key = aggregate_key({phase, stage});

        std::lock_guard<std::mutex> lock(mutex_);
        auto& aggregate = serverStageAggregates_[key];
        if (aggregate.stats.count == 0) {
            aggregate.phase = std::string(phase);
            aggregate.stage = std::string(stage);
        }
        aggregate.stats.add(elapsedMs);
        return;
    }

    logf(
        "server.stage phase=%.*s stage=%.*s elapsed_ms=%.6f",
        static_cast<int>(phase.size()),
        phase.data(),
        static_cast<int>(stage.size()),
        stage.data(),
        elapsedMs);
}

void RequestLogger::flushProfileSummaries() {
    if (!profile_)
        return;

    auto kernels = map_values(kernelAggregates_);
    auto stages = map_values(stageAggregates_);
    auto deviceRuns = map_values(deviceRunAggregates_);
    auto serverStages = map_values(serverStageAggregates_);
    auto kernelBuckets = kernel_bucket_values(kernels);
    auto stageBuckets = stage_bucket_values(stages);
    auto deviceRunBuckets = device_run_bucket_values(deviceRuns);
    auto serverStageBuckets = stage_bucket_values(serverStages);

    sort_by_total_desc(kernels);
    sort_by_total_desc(stages);
    sort_device_runs_by_total_desc(deviceRuns);
    sort_by_total_desc(serverStages);
    sort_by_total_desc(kernelBuckets);
    sort_by_total_desc(stageBuckets);
    sort_device_runs_by_total_desc(deviceRunBuckets);
    sort_by_total_desc(serverStageBuckets);

    logf(
        "profile.summary.start kernel_groups=%zu kernel_bucket_groups=%zu "
        "engine_stage_groups=%zu engine_stage_bucket_groups=%zu "
        "device_run_groups=%zu device_run_bucket_groups=%zu "
        "server_stage_groups=%zu server_stage_bucket_groups=%zu",
        kernels.size(),
        kernelBuckets.size(),
        stages.size(),
        stageBuckets.size(),
        deviceRuns.size(),
        deviceRunBuckets.size(),
        serverStages.size(),
        serverStageBuckets.size());

    for (const auto& aggregate : serverStageBuckets) {
        logf(
            "server.stage.bucket phase=%s stage=%s count=%zu total_ms=%.6f "
            "avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.stage.c_str(),
            aggregate.stats.count,
            aggregate.stats.totalMs,
            aggregate.stats.avg(),
            aggregate.stats.maxMs);
    }
    for (const auto& aggregate : serverStages) {
        logf(
            "server.stage.aggregate phase=%s stage=%s count=%zu total_ms=%.6f "
            "avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.stage.c_str(),
            aggregate.stats.count,
            aggregate.stats.totalMs,
            aggregate.stats.avg(),
            aggregate.stats.maxMs);
    }

    for (const auto& aggregate : kernelBuckets) {
        logf(
            "profile.kernel.bucket phase=%s kind=%s device=%u device_graph=%llu "
            "count=%zu total_ms=%.6f avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.kind.c_str(),
            aggregate.device,
            static_cast<unsigned long long>(aggregate.deviceGraph),
            aggregate.stats.count,
            aggregate.stats.totalMs,
            aggregate.stats.avg(),
            aggregate.stats.maxMs);
    }
    for (const auto& aggregate : kernels) {
        logf(
            "profile.kernel.aggregate phase=%s kind=%s device=%u device_graph=%llu "
            "count=%zu total_ms=%.6f avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.kind.c_str(),
            aggregate.device,
            static_cast<unsigned long long>(aggregate.deviceGraph),
            aggregate.stats.count,
            aggregate.stats.totalMs,
            aggregate.stats.avg(),
            aggregate.stats.maxMs);
    }

    for (const auto& aggregate : stageBuckets) {
        logf(
            "profile.engine_stage.bucket phase=%s stage=%s kind=%s count=%zu "
            "total_ms=%.6f avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.stage.c_str(),
            aggregate.kind.c_str(),
            aggregate.stats.count,
            aggregate.stats.totalMs,
            aggregate.stats.avg(),
            aggregate.stats.maxMs);
    }
    for (const auto& aggregate : stages) {
        logf(
            "profile.engine_stage.aggregate phase=%s stage=%s kind=%s count=%zu "
            "total_ms=%.6f avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.stage.c_str(),
            aggregate.kind.c_str(),
            aggregate.stats.count,
            aggregate.stats.totalMs,
            aggregate.stats.avg(),
            aggregate.stats.maxMs);
    }

    for (const auto& aggregate : deviceRunBuckets) {
        logf(
            "profile.device_run.bucket phase=%s kind=%s device=%u device_graph=%llu "
            "begin_count=%zu end_count=%zu total_ms=%.6f avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.kind.c_str(),
            aggregate.device,
            static_cast<unsigned long long>(aggregate.deviceGraph),
            aggregate.beginCount,
            aggregate.endCount,
            aggregate.endStats.totalMs,
            aggregate.endStats.avg(),
            aggregate.endStats.maxMs);
    }
    for (const auto& aggregate : deviceRuns) {
        logf(
            "profile.device_run.aggregate phase=%s kind=%s device=%u device_graph=%llu "
            "begin_count=%zu end_count=%zu total_ms=%.6f avg_ms=%.6f max_ms=%.6f",
            aggregate.phase.c_str(),
            aggregate.kind.c_str(),
            aggregate.device,
            static_cast<unsigned long long>(aggregate.deviceGraph),
            aggregate.beginCount,
            aggregate.endCount,
            aggregate.endStats.totalMs,
            aggregate.endStats.avg(),
            aggregate.endStats.maxMs);
    }

    log("profile.summary.end");
}

ServerStageScope::ServerStageScope(
        RequestLogger* logger,
        std::string phase,
        std::string stage)
    : logger_(logger),
      phase_(std::move(phase)),
      stage_(std::move(stage)),
      start_(std::chrono::steady_clock::now()) {}

ServerStageScope::~ServerStageScope() {
    if (!logger_)
        return;
    auto end = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration<double, std::milli>(end - start_).count();
    logger_->logServerStage(phase_, stage_, elapsedMs);
}

} // namespace sandy::server
