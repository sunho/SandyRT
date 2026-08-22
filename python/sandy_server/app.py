from __future__ import annotations

import argparse
import asyncio
import os
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
import grpc
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import ServerConfig
from .grpc_client import SandyGrpcClient
from .openai_compat import (
    chat_completion_response,
    chat_completion_chunk,
    model_list_response,
    new_chat_completion_id,
    sse_data,
)
from .tokenizer import (
    IncrementalTokenDecoder,
    decode_tokens,
    encode_messages,
    load_tokenizer,
)


def model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=32, ge=0)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    chat_template_kwargs: dict[str, Any] | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = 1
    stream_options: StreamOptions | None = None


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


def grpc_http_error(exc: grpc.RpcError) -> HTTPException:
    code = exc.code()
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        status = 504
    elif code == grpc.StatusCode.CANCELLED:
        status = 499
    elif code == grpc.StatusCode.UNAVAILABLE:
        status = 503
    else:
        status = 500
    return HTTPException(status_code=status, detail=exc.details() or str(exc))


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
    def chat_completions(request: ChatCompletionRequest, http_request: Request):
        if request.n != 1:
            raise HTTPException(status_code=400, detail="n > 1 is not supported in MVP")
        if request.model is not None and request.model != config.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        if request.stop is not None:
            raise HTTPException(status_code=400, detail="text stop sequences are not supported in MVP")

        completion_id = new_chat_completion_id()
        try:
            input_ids = encode_messages(
                tokenizer,
                [model_dump(message) for message in request.messages],
                request.chat_template_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stop_token_ids = config.stop_token_ids()

        if request.stream:
            call = grpc_client.generate_stream(
                request_id=completion_id,
                input_ids=input_ids,
                max_tokens=request.max_tokens,
                stop_token_ids=stop_token_ids,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            include_usage = bool(
                request.stream_options and request.stream_options.include_usage)
            created = int(time.time())
            pending: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=32)
            stop_reader = threading.Event()

            def enqueue(kind: str, value: Any) -> bool:
                while not stop_reader.is_set():
                    try:
                        pending.put((kind, value), timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def read_worker_stream() -> None:
                try:
                    for event in call:
                        if not enqueue("event", event):
                            return
                except grpc.RpcError as exc:
                    enqueue("rpc_error", exc)
                finally:
                    enqueue("end", None)

            reader = threading.Thread(target=read_worker_stream, daemon=True)
            reader.start()

            async def events():
                decoder = IncrementalTokenDecoder(tokenizer)
                yield sse_data(chat_completion_chunk(
                    completion_id,
                    config.model_id,
                    created,
                    delta={"role": "assistant", "content": ""},
                ))
                try:
                    while True:
                        if await http_request.is_disconnected():
                            return
                        try:
                            kind, value = pending.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.02)
                            continue

                        if kind == "rpc_error":
                            yield sse_data({
                                "error": {
                                    "message": value.details() or str(value),
                                    "type": "server_error",
                                }
                            })
                            yield sse_data("[DONE]")
                            return
                        if kind == "end":
                            yield sse_data({
                                "error": {
                                    "message": "worker stream ended without a final event",
                                    "type": "server_error",
                                }
                            })
                            yield sse_data("[DONE]")
                            return

                        event = value
                        if event.HasField("token"):
                            delta = decoder.push(int(event.token.token_id))
                            if delta:
                                yield sse_data(chat_completion_chunk(
                                    completion_id,
                                    config.model_id,
                                    created,
                                    delta={"content": delta},
                                ))
                        elif event.HasField("error"):
                            yield sse_data({
                                "error": {
                                    "message": event.error.message,
                                    "type": "server_error",
                                }
                            })
                            yield sse_data("[DONE]")
                            return
                        elif event.HasField("done"):
                            tail = decoder.finish()
                            if tail:
                                yield sse_data(chat_completion_chunk(
                                    completion_id,
                                    config.model_id,
                                    created,
                                    delta={"content": tail},
                                ))
                            yield sse_data(chat_completion_chunk(
                                completion_id,
                                config.model_id,
                                created,
                                delta={},
                                finish_reason=event.done.finish_reason or "stop",
                            ))
                            if include_usage:
                                prompt_tokens = int(event.done.prompt_tokens)
                                completion_tokens = int(event.done.completion_tokens)
                                yield sse_data(chat_completion_chunk(
                                    completion_id,
                                    config.model_id,
                                    created,
                                    usage={
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": prompt_tokens + completion_tokens,
                                    },
                                ))
                            yield sse_data("[DONE]")
                            return
                finally:
                    stop_reader.set()
                    call.cancel()
                    reader.join(timeout=0.2)

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            response = grpc_client.generate(
                request_id=completion_id,
                input_ids=input_ids,
                max_tokens=request.max_tokens,
                stop_token_ids=stop_token_ids,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except grpc.RpcError as exc:
            raise grpc_http_error(exc) from exc
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
    parser.add_argument("--end-of-turn-token-id", default=None, type=int)
    parser.add_argument("--tool-response-token-id", default=None, type=int)
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
        end_of_turn_token_id=args.end_of_turn_token_id,
        tool_response_token_id=args.tool_response_token_id,
        auth_token=auth_token,
    ))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
