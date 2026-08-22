from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    model_id: str
    tokenizer_path: Path
    grpc_target: str
    eos_token_id: int | None = 1
    end_of_turn_token_id: int | None = None
    tool_response_token_id: int | None = None
    auth_token: str | None = None

    def stop_token_ids(self) -> list[int]:
        return list(dict.fromkeys(
            token_id for token_id in (
                self.eos_token_id,
                self.end_of_turn_token_id,
                self.tool_response_token_id,
            ) if token_id is not None
        ))


def normalize_grpc_target(value: str) -> str:
    if value.startswith("unix:/") and not value.startswith("unix:///"):
        return "unix://" + value[len("unix:"):]
    return value
