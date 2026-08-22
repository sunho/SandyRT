#include "SandyInferenceService.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace sandy::server {

namespace {

RequestControl make_request_control(
        grpc::ServerContext* context,
        std::chrono::milliseconds timeout) {
    RequestControl control;
    control.submittedAt = RequestControl::Clock::now();
    if (timeout.count() > 0)
        control.deadline = control.submittedAt + timeout;
    control.clientCancelled = [context]() { return context->IsCancelled(); };
    return control;
}

SamplingOverrides sampling_overrides(const GenerateRequest& request) {
    return SamplingOverrides{
        request.has_top_p() ? std::optional<float>(request.top_p()) : std::nullopt,
        request.has_temperature()
            ? std::optional<float>(request.temperature())
            : std::nullopt};
}

std::vector<int64_t> copy_input_ids(const GenerateRequest& request) {
    return {request.input_ids().begin(), request.input_ids().end()};
}

std::vector<int64_t> copy_stop_token_ids(const GenerateRequest& request) {
    return {request.stop_token_ids().begin(), request.stop_token_ids().end()};
}

grpc::Status stopped_status(RequestStopReason reason) {
    if (reason == RequestStopReason::DeadlineExceeded) {
        return grpc::Status(
            grpc::StatusCode::DEADLINE_EXCEEDED,
            requestStopMessage(reason));
    }
    return grpc::Status(
        grpc::StatusCode::CANCELLED,
        requestStopMessage(RequestStopReason::ClientCancelled));
}

} // namespace

SandyInferenceService::SandyInferenceService(
        std::shared_ptr<Model> model,
        std::chrono::milliseconds requestTimeout)
    : model_(std::move(model)),
      requestTimeout_(requestTimeout) {}

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
        grpc::ServerContext* context,
        const GenerateRequest* request,
        GenerateResponse* response) {
    auto control = make_request_control(context, requestTimeout_);
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
        sampling_overrides(*request),
        &control);
    if (!generated) {
        auto reason = control.stopReason();
        if (reason != RequestStopReason::None)
            return stopped_status(reason);
        response->set_error(generated.error());
        if (logger) {
            logger->logf(
                "server.grpc.generate.error message=%s",
                generated.error().c_str());
        }
        return grpc::Status::OK;
    }

    auto result = generated.take();
    auto reason = control.stopReason();
    if (reason != RequestStopReason::None)
        return stopped_status(reason);
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

grpc::Status SandyInferenceService::GenerateStream(
        grpc::ServerContext* context,
        const GenerateRequest* request,
        grpc::ServerWriter<GenerateStreamResponse>* writer) {
    auto control = make_request_control(context, requestTimeout_);
    auto logger = model_
        ? RequestLogger::create(model_->config().logging, request->request_id())
        : nullptr;
    if (!model_)
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "model not loaded");

    bool writeFailed = false;
    auto onToken = [&](int64_t tokenId) {
        if (control.stopReason() != RequestStopReason::None)
            return false;
        GenerateStreamResponse event;
        event.set_request_id(request->request_id());
        event.mutable_token()->set_token_id(tokenId);
        if (!writer->Write(event)) {
            writeFailed = true;
            return false;
        }
        return true;
    };

    auto generated = model_->generate(
        request->request_id(),
        copy_input_ids(*request),
        request->max_tokens(),
        copy_stop_token_ids(*request),
        logger.get(),
        sampling_overrides(*request),
        &control,
        onToken);
    if (!generated) {
        auto reason = control.stopReason();
        if (reason != RequestStopReason::None)
            return stopped_status(reason);
        if (writeFailed)
            return stopped_status(RequestStopReason::ClientCancelled);

        GenerateStreamResponse event;
        event.set_request_id(request->request_id());
        event.mutable_error()->set_message(generated.error());
        if (!writer->Write(event))
            return stopped_status(RequestStopReason::ClientCancelled);
        return grpc::Status::OK;
    }

    auto reason = control.stopReason();
    if (reason != RequestStopReason::None)
        return stopped_status(reason);

    GenerateStreamResponse event;
    event.set_request_id(request->request_id());
    auto* done = event.mutable_done();
    done->set_finish_reason(generated->finishReason);
    done->set_prompt_tokens(generated->promptTokens);
    done->set_completion_tokens(generated->completionTokens);
    if (!writer->Write(event))
        return stopped_status(RequestStopReason::ClientCancelled);
    return grpc::Status::OK;
}

} // namespace sandy::server
