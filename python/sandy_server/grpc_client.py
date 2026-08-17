from __future__ import annotations

import sys
from pathlib import Path

import grpc

from .config import normalize_grpc_target

_GENERATED_DIR = Path(__file__).resolve().parent / "generated"
if str(_GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATED_DIR))

try:
    import sandy_inference_pb2
    import sandy_inference_pb2_grpc
except ImportError as exc:
    raise RuntimeError(
        "missing Python gRPC stubs; run: "
        "python scripts/generate_python_grpc.py"
    ) from exc


class SandyGrpcClient:
    def __init__(self, target: str):
        self._target = normalize_grpc_target(target)
        self._channel = grpc.insecure_channel(self._target)
        self._stub = sandy_inference_pb2_grpc.SandyInferenceStub(self._channel)

    def health(self) -> tuple[bool, str]:
        response = self._stub.Health(sandy_inference_pb2.HealthRequest())
        return bool(response.ok), response.message

    def model_info(self):
        return self._stub.ModelInfo(sandy_inference_pb2.ModelInfoRequest())

    def generate(
            self,
            request_id: str,
            input_ids: list[int],
            max_tokens: int,
            stop_token_ids: list[int]) -> object:
        request = sandy_inference_pb2.GenerateRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )
        return self._stub.Generate(request)
