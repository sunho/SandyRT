#include "GemmaModel.h"
#include "SandyInferenceService.h"

#include <grpcpp/grpcpp.h>

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::string modelPath;
    std::string weightsPath;
    std::string listen = "127.0.0.1:50051";
    std::string modelId = "gemma4e2b";
    int eosTokenId = 1;
    int maxContextTokens = 0;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --model <eval_token.sandy.go> --weights <weights.safetensors> "
        "[--listen <addr>] [--model-id <id>] [--eos-token-id <id>] "
        "[--max-context-tokens <n>]\n",
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
        } else if (arg == "--weights") {
            if (!require_value(args.weightsPath)) return false;
        } else if (arg == "--listen") {
            if (!require_value(args.listen)) return false;
        } else if (arg == "--model-id") {
            if (!require_value(args.modelId)) return false;
        } else if (arg == "--eos-token-id") {
            std::string value;
            if (!require_value(value) || !parse_int(value, args.eosTokenId)) return false;
        } else if (arg == "--max-context-tokens") {
            std::string value;
            if (!require_value(value) || !parse_int(value, args.maxContextTokens)) return false;
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

    sandy::server::GemmaModelConfig config;
    config.modelId = args.modelId;
    config.modelPath = args.modelPath;
    config.weightsPath = args.weightsPath;
    config.eosTokenId = args.eosTokenId;
    config.maxContextTokens = args.maxContextTokens;

    auto modelResult = sandy::server::GemmaModel::load(std::move(config));
    if (!modelResult) {
        std::fprintf(stderr, "model load error: %s\n", modelResult.error().c_str());
        return 1;
    }

    auto model = std::shared_ptr<sandy::server::GemmaModel>(modelResult.take().release());
    sandy::server::SandyInferenceService service(model);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(args.listen, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    if (!server) {
        std::fprintf(stderr, "failed to listen on %s\n", args.listen.c_str());
        return 1;
    }

    std::fprintf(stderr, "sandy_grpc_worker listening on %s\n", args.listen.c_str());
    server->Wait();
    return 0;
}
