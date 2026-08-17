from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    model_id: str
    tokenizer_path: Path
    grpc_target: str
    eos_token_id: int | None = 1


def normalize_grpc_target(value: str) -> str:
    if value.startswith("unix:/") and not value.startswith("unix:///"):
        return "unix://" + value[len("unix:"):]
    return value
