#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

try:
    import numpy as np
except ImportError:
    np = None


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ = 32
PAD_ID = 0
EOS_ID = 2
LAYERS = 22

DTYPE_SIZE = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def require_numpy():
    if np is None:
        raise RuntimeError("numpy is required; install dependencies with: pip install -r scripts/requirements.txt")
    return np


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_runner(root: pathlib.Path) -> pathlib.Path:
    for runner in [
        root / "build-cuda/test/cuda_runner",
        root / "build-cuda/cuda_runner",
        root / "build-cublas12/cpu_runner",
        root / "build-cublas/cpu_runner",
        root / "build-fast/cpu_runner",
        root / "build-opt/test/cpu_runner",
        root / "build/test/cpu_runner",
    ]:
        if runner.exists():
            return runner
    return root / "build-fast/cpu_runner"


def default_eval_runner(root: pathlib.Path) -> pathlib.Path:
    for runner in [
        root / "build-cuda/test/cuda_multi_gemma4_runner",
        root / "build-cuda/cuda_multi_gemma4_runner",
        root / "build-cublas12/multi_gemma4_runner",
        root / "build-cublas/multi_gemma4_runner",
        root / "build-fast/multi_gemma4_runner",
        root / "build-opt/test/multi_gemma4_runner",
        root / "build/test/multi_gemma4_runner",
    ]:
        if runner.exists():
            return runner
    return root / "build-cuda/test/cuda_multi_gemma4_runner"


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: list[int]
    path: pathlib.Path
    data_offset: int
    nbytes: int


@dataclass
class OutputTensor:
    name: str
    dtype: str
    shape: list[int]
    nbytes: int
    write: Callable[[object], None]


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_SIZE[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype}") from exc


def read_safetensors_header(path: pathlib.Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def discover_safetensors(root: pathlib.Path) -> dict[str, TensorEntry]:
    if root.is_file():
        files = [root]
    else:
        index = root / "model.safetensors.index.json"
        if index.exists():
            data = json.loads(index.read_text())
            files = sorted({root / name for name in data.get("weight_map", {}).values()})
        else:
            files = sorted(root.glob("*.safetensors"))

    entries: dict[str, TensorEntry] = {}
    for path in files:
        header, data_start = read_safetensors_header(path)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            entries[name] = TensorEntry(
                name=name,
                dtype=meta["dtype"],
                shape=list(meta["shape"]),
                path=path,
                data_offset=data_start + begin,
                nbytes=end - begin,
            )
    return entries


def read_entry(entry: TensorEntry) -> bytes:
    with entry.path.open("rb") as f:
        f.seek(entry.data_offset)
        return f.read(entry.nbytes)


def f32_to_bf16_bytes(arr) -> bytes:
    numpy = require_numpy()
    f32 = numpy.asarray(arr, dtype=numpy.float32)
    bits = f32.view(numpy.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16).astype("<u2").tobytes()


def entry_as_bf16_array(entry: TensorEntry):
    numpy = require_numpy()
    raw = read_entry(entry)
    if entry.dtype == "BF16":
        return numpy.frombuffer(raw, dtype="<u2").copy().reshape(entry.shape)
    if entry.dtype == "F16":
        arr = numpy.frombuffer(raw, dtype="<f2").astype(numpy.float32)
        return numpy.frombuffer(f32_to_bf16_bytes(arr), dtype="<u2").reshape(entry.shape)
    if entry.dtype == "F32":
        arr = numpy.frombuffer(raw, dtype="<f4")
        return numpy.frombuffer(f32_to_bf16_bytes(arr), dtype="<u2").reshape(entry.shape)
    raise ValueError(f"{entry.name}: expected floating dtype, got {entry.dtype}")


def bf16_tensor(out_name: str, src: TensorEntry) -> OutputTensor:
    count = 1
    for dim in src.shape:
        count *= dim
    nbytes = count * 2

    def write(f) -> None:
        f.write(entry_as_bf16_array(src).astype("<u2", copy=False).tobytes())

    return OutputTensor(out_name, "BF16", src.shape, nbytes, write)


def bytes_tensor(out_name: str, dtype: str, shape: list[int], data: bytes) -> OutputTensor:
    expected = dtype_size(dtype)
    for dim in shape:
        expected *= dim
    if expected != len(data):
        raise ValueError(f"{out_name}: expected {expected} bytes, got {len(data)}")
    return OutputTensor(out_name, dtype, shape, len(data), lambda f: f.write(data))


def write_safetensors(path: pathlib.Path, tensors: list[OutputTensor]) -> None:
    offset = 0
    header = {}
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for index, tensor in enumerate(tensors, 1):
            print(f"[weights] {index:4d}/{len(tensors)} {tensor.name} {tensor.dtype}{tensor.shape}")
            tensor.write(f)
    tmp.replace(path)


def resolve(entries: dict[str, TensorEntry], *names: str) -> TensorEntry:
    for name in names:
        if name in entries:
            return entries[name]
    raise KeyError("missing tensor; tried: " + ", ".join(names))


def maybe_resolve(entries: dict[str, TensorEntry], *names: str) -> TensorEntry | None:
    for name in names:
        if name in entries:
            return entries[name]
    return None


def convert_weights(src_dir: pathlib.Path, dst: pathlib.Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[weights] exists: {dst}")
        return

    entries = discover_safetensors(src_dir)
    if not entries:
        raise FileNotFoundError(f"no safetensors found under {src_dir}")

    out: list[OutputTensor] = []
    out.append(bf16_tensor("model.embed_tokens.weight", resolve(entries, "model.embed_tokens.weight")))
    out.append(bf16_tensor("model.norm.weight", resolve(entries, "model.norm.weight")))

    lm_head = maybe_resolve(entries, "lm_head.weight")
    if lm_head is None:
        lm_head = resolve(entries, "model.embed_tokens.weight")
    out.append(bf16_tensor("model.lm_head.weight", lm_head))

    for layer in range(LAYERS):
        src_base = f"model.layers.{layer}"
        out_base = f"model.layers.{layer}"
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            out.append(bf16_tensor(f"{out_base}.{suffix}", resolve(entries, f"{src_base}.{suffix}")))

    write_safetensors(dst, out)
    print(f"[weights] wrote: {dst}")


def download_snapshot(artifacts: pathlib.Path, model_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --download") from exc
    snapshot_download(
        repo_id=model_id,
        local_dir=artifacts,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "*.model",
            "*.json",
        ],
    )


def load_tokenizer(artifacts: pathlib.Path, model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for prompt tokenization; pass --ids or install transformers") from exc
    return AutoTokenizer.from_pretrained(artifacts if artifacts.exists() else model_id)


def encode_prompt(prompt: str, artifacts: pathlib.Path, model_id: str) -> tuple[list[int], str]:
    tokenizer = load_tokenizer(artifacts, model_id)
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids], rendered


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def write_input(path: pathlib.Path, ids: list[int], max_seq: int) -> int:
    real_ids = ids[-max_seq:] or [EOS_ID]
    token_index = len(real_ids) - 1
    padded = real_ids + [PAD_ID] * (max_seq - len(real_ids))
    input_data = struct.pack(f"<{len(padded)}q", *padded)
    write_safetensors(path, [
        bytes_tensor("input_ids", "I64", [1, max_seq], input_data),
    ])
    return token_index


def parse_ids(value: str) -> list[int]:
    ids = [int(part) for part in value.replace(" ", "").split(",") if part]
    if not ids:
        raise argparse.ArgumentTypeError("--ids must contain at least one token id")
    return ids


def parse_runner_topk(stdout: str, token_index: int) -> list[tuple[int, float]]:
    pattern = re.compile(r"^\s+top5\[(\d+)\]:(.*)$", re.MULTILINE)
    for match in pattern.finditer(stdout):
        if int(match.group(1)) != token_index:
            continue
        entries = []
        for token_id, score in re.findall(r"\s(\d+)\(([^)]+)\)", match.group(2)):
            entries.append((int(token_id), float(score)))
        if entries:
            return entries
    raise RuntimeError(f"cpu_runner output did not contain top5[{token_index}]")


def parse_generated(stdout: str) -> list[int]:
    match = re.search(r"^\[generated\](.*)$", stdout, re.MULTILINE)
    if not match:
        return []
    return [int(part) for part in match.group(1).split()]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Prepare and run Sandy TinyLlama 1.1B Chat BF16.")
    parser.add_argument("prompt", nargs="?", default="Hello, my name is")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--artifacts", default=root / "experiments/tinyllama", type=pathlib.Path)
    parser.add_argument("--hf-weights", default=None, type=pathlib.Path,
                        help="Directory or safetensors file with original HF weights.")
    parser.add_argument("--sandy-weights", default=None, type=pathlib.Path)
    parser.add_argument("--model", default=root / "src/models/tinyllama.sandy.go", type=pathlib.Path)
    parser.add_argument("--eval-model", default=root / "src/models/tinyllama/eval_token.sandy.go", type=pathlib.Path)
    parser.add_argument("--runner", default=default_runner(root), type=pathlib.Path)
    parser.add_argument("--eval-runner", default=default_eval_runner(root), type=pathlib.Path)
    parser.add_argument("--max-seq", default=MAX_SEQ, type=int)
    parser.add_argument("--max-answer-tokens", default=1, type=int)
    parser.add_argument("--ids", type=parse_ids, default=None,
                        help="Comma-separated token ids. Skips tokenizer loading.")
    parser.add_argument("--eval-token", action="store_true",
                        help="Run the decoder eval-token model with paged KV caches.")
    parser.add_argument("--download", action="store_true",
                        help="Download the HF snapshot if artifacts are missing.")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-input", action="store_true")
    parser.add_argument("--instrument", action="store_true",
                        help="Print per-kernel CPU engine timing from cpu_runner.")
    args = parser.parse_args()

    artifacts = args.artifacts
    hf_weights = args.hf_weights or artifacts
    sandy_weights = args.sandy_weights or (artifacts / "sandy_model.bf16.safetensors")

    if args.download and not list(artifacts.glob("*.safetensors")):
        print(f"[download] {args.model_id} -> {artifacts}")
        download_snapshot(artifacts, args.model_id)

    if not pathlib.Path(hf_weights).exists():
        print(f"missing original weights: {hf_weights}", file=sys.stderr)
        print(f"download with: {sys.argv[0]} --download --prepare-only", file=sys.stderr)
        return 1

    convert_weights(pathlib.Path(hf_weights), sandy_weights, args.force_convert)

    rendered_prompt = None
    tokenizer = None
    if args.ids is not None:
        ids = args.ids
    else:
        tokenizer = load_tokenizer(artifacts, args.model_id)
        ids, rendered_prompt = encode_prompt(args.prompt, artifacts, args.model_id)

    if rendered_prompt is not None:
        print(f"[chat] rendered prompt:\n{rendered_prompt}")
    print(f"[tokenizer] ids: {ids}")

    if args.eval_token:
        if args.max_answer_tokens < 0:
            print("--max-answer-tokens must be >= 0", file=sys.stderr)
            return 1
        if not args.eval_runner.exists():
            print(f"missing eval-token runner: {args.eval_runner}", file=sys.stderr)
            print("build with: cmake --build build-cuda --target cuda_multi_gemma4_runner", file=sys.stderr)
            return 1
        if not args.eval_model.exists():
            print(f"missing eval-token model: {args.eval_model}", file=sys.stderr)
            return 1

        cmd = [
            str(args.eval_runner),
            "--eval-token",
            "--architecture", "tinyllama",
        ]
        if args.instrument:
            cmd.append("--profile")
        cmd.extend([
            str(args.eval_model),
            str(sandy_weights),
            str(args.max_answer_tokens),
            *[str(token_id) for token_id in ids],
        ])
        print("[run]", " ".join(cmd))
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            return result.returncode
        generated = parse_generated(result.stdout)
        if tokenizer is not None and generated:
            print("[generated decoded]")
            print(tokenizer.decode(generated, skip_special_tokens=False))
        return 0

    if args.max_seq != MAX_SEQ:
        print(f"this SandyGo model is fixed to max_seq={MAX_SEQ}", file=sys.stderr)
        return 1

    if args.keep_input:
        input_path = artifacts / "input_latest.safetensors"
    else:
        fd, tmp_name = tempfile.mkstemp(prefix="sandy_tinyllama_input_", suffix=".safetensors")
        os.close(fd)
        input_path = pathlib.Path(tmp_name)
    token_index = write_input(input_path, ids, args.max_seq)

    if len(ids) > args.max_seq:
        print(f"[input] prompt has {len(ids)} tokens; using last {args.max_seq}", file=sys.stderr)
    print(f"[input] next-token logits position: {token_index}")
    print(f"[input] safetensors: {input_path}")
    print(f"[weights] sandy: {sandy_weights}")

    if args.prepare_only:
        return 0

    if not args.runner.exists():
        print(f"missing cpu_runner: {args.runner}", file=sys.stderr)
        print("build with: cmake --build build-fast --target cpu_runner", file=sys.stderr)
        return 1

    cmd = [str(args.runner)]
    if args.instrument:
        cmd.append("--instrument")
    cmd.extend([str(args.model), str(sandy_weights), str(input_path)])
    print("[run]", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(result.stdout, end="")
        return result.returncode

    print(result.stdout, end="")
    if tokenizer is not None:
        print("[top-k decoded]")
        for rank, (token_id, logit) in enumerate(parse_runner_topk(result.stdout, token_index), 1):
            print(f"{rank:2d}. id={token_id:5d} logit={logit: .6g} text={decode_token(tokenizer, token_id)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
