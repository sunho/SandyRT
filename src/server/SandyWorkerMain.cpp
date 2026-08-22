#include "Model.h"
#include "SandyInferenceService.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::string modelPath;
    std::string prefillModelPath;
    std::string weightsPath;
    std::string listen = "127.0.0.1:50051";
    std::string modelId;
    std::string architecture = "gemma4e2b";
    int eosTokenId = -1;
    int maxContextTokens = 0;
    int prefillChunkTokens = sandy::server::kDefaultPrefillChunkTokens;
    int requestTimeoutMs = 300000;
    bool debug = false;
    bool profile = false;
    std::string logDir = "logs/requests";
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --model <eval_token.sandy.go> --weights <weights.safetensors> "
        "[--prefill-model <prefill.sandy.go>] [--prefill-chunk-tokens <n>] "
        "[--listen <addr>] [--model-id <id>] [--architecture <name>] "
        "[--eos-token-id <id>] [--max-context-tokens <n>] "
        "[--request-timeout-ms <n>] "
        "[--debug] [--profile] [--log-dir <dir>]\n",
        argv0);
}

bool parse_int(std::string_view text, int& out) {
    try {
        size_t pos = 0;
        int value = std::stoi(std::string(text), &pos);
        if (pos != text.size())
            return false;
        out = value;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_args(int argc, char* argv[], Args& args) {
    for (int i = 1; i < argc; i++) {
        std::string_view arg(argv[i]);
        auto require_value = [&](std::string& out) -> bool {
            if (i + 1 >= argc)
                return false;
            out = argv[++i];
            return true;
        };

        if (arg == "--model") {
            if (!require_value(args.modelPath)) return false;
        } else if (arg == "--prefill-model") {
            if (!require_value(args.prefillModelPath)) return false;
        } else if (arg == "--prefill-chunk-tokens" ||
                   arg == "--prefill-chunk-size") {
            std::string value;
            if (!require_value(value) || !parse_int(value, args.prefillChunkTokens)) return false;
            if (args.prefillChunkTokens < 0)
                return false;
        } else if (arg == "--weights") {
            if (!require_value(args.weightsPath)) return false;
        } else if (arg == "--listen") {
            if (!require_value(args.listen)) return false;
        } else if (arg == "--model-id") {
            if (!require_value(args.modelId)) return false;
        } else if (arg == "--architecture") {
            if (!require_value(args.architecture)) return false;
        } else if (arg == "--eos-token-id") {
            std::string value;
            if (!require_value(value) || !parse_int(value, args.eosTokenId)) return false;
        } else if (arg == "--max-context-tokens") {
            std::string value;
            if (!require_value(value) || !parse_int(value, args.maxContextTokens)) return false;
        } else if (arg == "--request-timeout-ms") {
            std::string value;
            if (!require_value(value) || !parse_int(value, args.requestTimeoutMs)) return false;
            if (args.requestTimeoutMs < 0)
                return false;
        } else if (arg == "--debug") {
            args.debug = true;
        } else if (arg == "--profile") {
            args.profile = true;
        } else if (arg == "--log-dir" || arg == "--request-log-dir") {
            if (!require_value(args.logDir)) return false;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return !args.modelPath.empty() && !args.weightsPath.empty();
}

} // namespace

int main(int argc, char* argv[]) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 1;
    }

    sandy::server::ModelConfig config;
    config.modelId = args.modelId;
    config.architecture = args.architecture;
    config.modelPath = args.modelPath;
    config.prefillModelPath = args.prefillModelPath;
    config.weightsPath = args.weightsPath;
    config.eosTokenId = args.eosTokenId;
    config.maxContextTokens = args.maxContextTokens;
    config.logging.debug = args.debug || args.profile;
    config.logging.profile = args.profile;
    config.logging.requestLogDir = args.logDir;
    config.session.prefillChunkTokens = args.prefillChunkTokens;

    auto modelResult = sandy::server::Model::load(std::move(config));
    if (!modelResult) {
        std::fprintf(stderr, "model load error: %s\n", modelResult.error().c_str());
        return 1;
    }

    auto model = std::shared_ptr<sandy::server::Model>(modelResult.take().release());
    sandy::server::SandyInferenceService service(
        model,
        std::chrono::milliseconds(args.requestTimeoutMs));

    grpc::ServerBuilder builder;
    int selectedPort = 0;
    builder.AddListeningPort(
        args.listen,
        grpc::InsecureServerCredentials(),
        &selectedPort);
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    if (!server) {
        std::fprintf(stderr, "failed to listen on %s\n", args.listen.c_str());
        return 1;
    }

    std::fprintf(stderr,
        "sandy_grpc_worker listening on %s (selected port %d)\n",
        args.listen.c_str(),
        selectedPort);
    std::fprintf(stderr,
        "sandy_grpc_worker logging debug=%d profile=%d log_dir=%s\n",
        args.debug || args.profile ? 1 : 0,
        args.profile ? 1 : 0,
        args.logDir.c_str());
    std::fprintf(stderr,
        "sandy_grpc_worker request_timeout_ms=%d\n",
        args.requestTimeoutMs);
    server->Wait();
    return 0;
}
