#include "SandyInferenceService.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace sandy::server {

SandyInferenceService::SandyInferenceService(std::shared_ptr<GemmaModel> model)
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
    response->set_backend("cpu");
    response->set_max_context_tokens(config.maxContextTokens);
    return grpc::Status::OK;
}

grpc::Status SandyInferenceService::Generate(
        grpc::ServerContext*,
        const GenerateRequest* request,
        GenerateResponse* response) {
    response->set_request_id(request->request_id());
    response->set_prompt_tokens(request->input_ids_size());

    if (!model_) {
        response->set_error("model not loaded");
        return grpc::Status::OK;
    }

    std::vector<int64_t> inputIds;
    inputIds.reserve(static_cast<size_t>(request->input_ids_size()));
    for (auto token : request->input_ids())
        inputIds.push_back(token);

    std::vector<int64_t> stopTokenIds;
    stopTokenIds.reserve(static_cast<size_t>(request->stop_token_ids_size()));
    for (auto token : request->stop_token_ids())
        stopTokenIds.push_back(token);

    auto generated = model_->generate(inputIds, request->max_tokens(), stopTokenIds);
    if (!generated) {
        response->set_error(generated.error());
        return grpc::Status::OK;
    }

    auto result = generated.take();
    for (auto token : result.outputIds)
        response->add_output_ids(token);
    response->set_finish_reason(result.finishReason);
    response->set_prompt_tokens(result.promptTokens);
    response->set_completion_tokens(result.completionTokens);
    return grpc::Status::OK;
}

} // namespace sandy::server
