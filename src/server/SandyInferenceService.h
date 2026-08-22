#pragma once

#include "Model.h"
#include "sandy_inference.grpc.pb.h"

#include <chrono>
#include <memory>

namespace sandy::server {

class SandyInferenceService final : public sandy::server::SandyInference::Service {
public:
    SandyInferenceService(
        std::shared_ptr<Model> model,
        std::chrono::milliseconds requestTimeout);

    grpc::Status Health(
        grpc::ServerContext* context,
        const HealthRequest* request,
        HealthResponse* response) override;

    grpc::Status ModelInfo(
        grpc::ServerContext* context,
        const ModelInfoRequest* request,
        ModelInfoResponse* response) override;

    grpc::Status Generate(
        grpc::ServerContext* context,
        const GenerateRequest* request,
        GenerateResponse* response) override;

    grpc::Status GenerateStream(
        grpc::ServerContext* context,
        const GenerateRequest* request,
        grpc::ServerWriter<GenerateStreamResponse>* writer) override;

private:
    std::shared_ptr<Model> model_;
    std::chrono::milliseconds requestTimeout_;
};

} // namespace sandy::server
