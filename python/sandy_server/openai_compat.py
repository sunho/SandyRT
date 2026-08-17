from __future__ import annotations

import time
import uuid
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
