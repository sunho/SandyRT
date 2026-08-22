#include "SandyInferenceService.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace sandy::server {

SandyInferenceService::SandyInferenceService(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

grpc::Status SandyInferenceService::Health(
        grpc::ServerContext*,
        const HealthRequest*,
        HealthResponse* response) {
    response->set_ok(model_ != nullptr);
    response->set_message(model_ ? "ok" : "model not loaded");
    return grpc::Status::OK;
}

grpc::Status SandyInferenceService::ModelInfo(
        grpc::ServerContext*,
        const ModelInfoRequest*,
        ModelInfoResponse* response) {
    if (!model_)
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "model not loaded");

    const auto& config = model_->config();
    response->set_model_id(config.modelId);
    response->set_backend(model_->backend());
    response->set_max_context_tokens(config.maxContextTokens);
    return grpc::Status::OK;
}

grpc::Status SandyInferenceService::Generate(
        grpc::ServerContext*,
        const GenerateRequest* request,
        GenerateResponse* response) {
    auto logger = model_
        ? RequestLogger::create(model_->config().logging, request->request_id())
        : nullptr;
    ServerStageScope totalTimer(
        logger.get(),
        "request",
        "server.grpc.generate.total");
    if (logger) {
        logger->logf(
            "server.grpc.generate.start request_id=%s input_ids=%d max_tokens=%d "
            "stop_token_ids=%d",
            request->request_id().c_str(),
            request->input_ids_size(),
            request->max_tokens(),
            request->stop_token_ids_size());
    }

    response->set_request_id(request->request_id());
    response->set_prompt_tokens(request->input_ids_size());

    if (!model_) {
        response->set_error("model not loaded");
        if (logger)
            logger->log("server.grpc.generate.error message=model_not_loaded");
        return grpc::Status::OK;
    }

    std::vector<int64_t> inputIds;
    {
        ServerStageScope parseTimer(
            logger.get(),
            "request",
            "server.grpc.parse_input_ids");
        inputIds.reserve(static_cast<size_t>(request->input_ids_size()));
        for (auto token : request->input_ids())
            inputIds.push_back(token);
    }

    std::vector<int64_t> stopTokenIds;
    {
        ServerStageScope parseTimer(
            logger.get(),
            "request",
            "server.grpc.parse_stop_token_ids");
        stopTokenIds.reserve(static_cast<size_t>(request->stop_token_ids_size()));
        for (auto token : request->stop_token_ids())
            stopTokenIds.push_back(token);
    }

    auto generated = model_->generate(
        request->request_id(),
        inputIds,
        request->max_tokens(),
        stopTokenIds,
        logger.get(),
        SamplingOverrides{
            request->has_top_p() ? std::optional<float>(request->top_p()) : std::nullopt,
            request->has_temperature()
                ? std::optional<float>(request->temperature())
                : std::nullopt});
    if (!generated) {
        response->set_error(generated.error());
        if (logger) {
            logger->logf(
                "server.grpc.generate.error message=%s",
                generated.error().c_str());
        }
        return grpc::Status::OK;
    }

    auto result = generated.take();
    {
        ServerStageScope responseTimer(
            logger.get(),
            "request",
            "server.grpc.write_response");
        for (auto token : result.outputIds)
            response->add_output_ids(token);
        response->set_finish_reason(result.finishReason);
        response->set_prompt_tokens(result.promptTokens);
        response->set_completion_tokens(result.completionTokens);
    }
    if (logger) {
        logger->logf(
            "server.grpc.generate.done finish_reason=%s prompt_tokens=%d "
            "completion_tokens=%d",
            result.finishReason.c_str(),
            result.promptTokens,
            result.completionTokens);
    }
    return grpc::Status::OK;
}

} // namespace sandy::server
