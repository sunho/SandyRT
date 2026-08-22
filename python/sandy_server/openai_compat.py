from __future__ import annotations

import time
import uuid
import json
from typing import Any


def new_chat_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def model_list_response(model_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "sandy",
            }
        ],
    }


def chat_completion_response(
        completion_id: str,
        model_id: str,
        content: str,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_completion_chunk(
        completion_id: str,
        model_id: str,
        created: int,
        delta: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    if delta is not None or finish_reason is not None:
        choices.append({
            "index": 0,
            "delta": delta or {},
            "finish_reason": finish_reason,
        })
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": choices,
        "usage": usage,
    }


def sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
