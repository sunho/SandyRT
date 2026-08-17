#pragma once

#include "Model.h"
#include "sandy_inference.grpc.pb.h"

#include <memory>

namespace sandy::server {

class SandyInferenceService final : public sandy::server::SandyInference::Service {
public:
    explicit SandyInferenceService(std::shared_ptr<Model> model);

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

private:
    std::shared_ptr<Model> model_;
};

} // namespace sandy::server
