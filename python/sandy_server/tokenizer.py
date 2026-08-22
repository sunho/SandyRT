from __future__ import annotations

from pathlib import Path
from typing import Any


def load_tokenizer(tokenizer_path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for Sandy chat tokenization") from exc
    return AutoTokenizer.from_pretrained(tokenizer_path)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        if not isinstance(content, str):
            raise ValueError("MVP chat server only supports string message content")
        normalized.append({"role": role, "content": content})
    return normalized


def encode_messages(
        tokenizer,
        messages: list[dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None = None) -> list[int]:
    normalized = _normalize_messages(messages)
    if getattr(tokenizer, "chat_template", None):
        template_kwargs = dict(chat_template_kwargs or {})
        reserved = {"tokenize", "add_generation_prompt"} & template_kwargs.keys()
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"chat_template_kwargs cannot override: {names}")
        ids = tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    else:
        prompt = _fallback_chat_prompt(tokenizer, normalized)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _fallback_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    bos = tokenizer.bos_token or ""
    sot = getattr(tokenizer, "sot_token", None) or "<|turn>"
    eot = getattr(tokenizer, "eot_token", None) or "<turn|>"
    parts = [bos]
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        parts.append(f"{sot}{role}\n{message['content']}{eot}")
    parts.append(f"{sot}model\n")
    return "".join(parts)


def decode_tokens(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


class IncrementalTokenDecoder:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._emitted = ""

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        decoded = decode_tokens(self._tokenizer, self._ids)
        if not decoded.startswith(self._emitted):
            return ""
        delta = decoded[len(self._emitted):]
        replacement = delta.find("\ufffd")
        if replacement >= 0:
            delta = delta[:replacement]
        self._emitted += delta
        return delta

    def finish(self) -> str:
        decoded = decode_tokens(self._tokenizer, self._ids)
        if not decoded.startswith(self._emitted):
            return ""
        delta = decoded[len(self._emitted):]
        self._emitted = decoded
        return delta
