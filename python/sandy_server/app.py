from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path
from typing import Any

import uvicorn
import grpc
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import ServerConfig
from .grpc_client import SandyGrpcClient
from .openai_compat import (
    chat_completion_response,
    model_list_response,
    new_chat_completion_id,
)
from .tokenizer import decode_tokens, encode_messages, load_tokenizer


def model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=32, ge=0)
    temperature: float = 0.0
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = 1


def normalize_auth_token(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def make_auth_dependency(auth_token: str | None):
    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if auth_token is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        authorized = (
            scheme.lower() == "bearer"
            and bool(token)
            and secrets.compare_digest(token, auth_token)
        )
        if not authorized:
            raise HTTPException(
                status_code=401,
                detail="invalid authorization token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_auth


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="Sandy Server")
    tokenizer = load_tokenizer(config.tokenizer_path)
    grpc_client = SandyGrpcClient(config.grpc_target)
    api = APIRouter(
        prefix="/v1",
        dependencies=[Depends(make_auth_dependency(config.auth_token))],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            ok, message = grpc_client.health()
        except grpc.RpcError as exc:
            return {"ok": False, "message": exc.details() or str(exc)}
        return {"ok": ok, "message": message}

    @api.get("/models")
    def models() -> dict[str, Any]:
        return model_list_response(config.model_id)

    @api.post("/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported in MVP")
        if request.n != 1:
            raise HTTPException(status_code=400, detail="n > 1 is not supported in MVP")
        if request.temperature != 0:
            raise HTTPException(status_code=400, detail="only temperature=0 is supported in MVP")
        if request.model is not None and request.model != config.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        if request.stop is not None:
            raise HTTPException(status_code=400, detail="text stop sequences are not supported in MVP")

        completion_id = new_chat_completion_id()
        try:
            input_ids = encode_messages(
                tokenizer,
                [model_dump(message) for message in request.messages],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stop_token_ids: list[int] = []
        if config.eos_token_id is not None:
            stop_token_ids.append(config.eos_token_id)

        try:
            response = grpc_client.generate(
                request_id=completion_id,
                input_ids=input_ids,
                max_tokens=request.max_tokens,
                stop_token_ids=stop_token_ids,
            )
        except grpc.RpcError as exc:
            raise HTTPException(
                status_code=503,
                detail=exc.details() or "gRPC worker unavailable",
            ) from exc
        if response.error:
            raise HTTPException(status_code=500, detail=response.error)

        output_ids = [int(token_id) for token_id in response.output_ids]
        content = decode_tokens(tokenizer, output_ids)
        finish_reason = response.finish_reason or "stop"
        return chat_completion_response(
            completion_id=completion_id,
            model_id=config.model_id,
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=int(response.prompt_tokens),
            completion_tokens=int(response.completion_tokens),
        )

    app.include_router(api)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sandy OpenAI-compatible HTTP server.")
    parser.add_argument("--model-id", default="gemma4e2b")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--grpc", default="127.0.0.1:50051")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--eos-token-id", default=1, type=int)
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Bearer token required for /v1 requests. Defaults to --auth-token-env.",
    )
    parser.add_argument(
        "--auth-token-env",
        default="SANDY_API_TOKEN",
        help="Environment variable to read the bearer token from.",
    )
    args = parser.parse_args()
    auth_token = normalize_auth_token(args.auth_token)
    if auth_token is None and args.auth_token_env:
        auth_token = normalize_auth_token(os.environ.get(args.auth_token_env))

    app = create_app(ServerConfig(
        model_id=args.model_id,
        tokenizer_path=args.tokenizer,
        grpc_target=args.grpc,
        eos_token_id=args.eos_token_id,
        auth_token=auth_token,
    ))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
