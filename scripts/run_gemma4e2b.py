#!/usr/bin/env python3

import argparse
import json
import mmap
import os
import pathlib
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


MODEL_ID = "google/gemma-4-E2B-it"
MAX_SEQ = 16
PAD_ID = 0
EOS_ID = 1
HIDDEN = 1536
PLE_DIM = 256
LAYERS = 35
FIRST_SHARED_KV_LAYER = 15


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
    opt_runner = root / "build-opt/test/cpu_runner"
    if opt_runner.exists():
        return opt_runner
    return root / "build/test/cpu_runner"


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


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_SIZE[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype}") from exc


def read_entry(entry: TensorEntry) -> bytes:
    with entry.path.open("rb") as f:
        f.seek(entry.data_offset)
        return f.read(entry.nbytes)


def direct_tensor(out_name: str, src: TensorEntry) -> OutputTensor:
    def write(f) -> None:
        with src.path.open("rb") as in_f:
            in_f.seek(src.data_offset)
            remaining = src.nbytes
            while remaining:
                chunk = in_f.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"unexpected EOF while reading {src.path}")
                f.write(chunk)
                remaining -= len(chunk)

    return OutputTensor(out_name, src.dtype, src.shape, src.nbytes, write)


def bytes_tensor(out_name: str, dtype: str, shape: list[int], data: bytes) -> OutputTensor:
    expected = dtype_size(dtype)
    for dim in shape:
        expected *= dim
    if expected != len(data):
        raise ValueError(f"{out_name}: expected {expected} bytes, got {len(data)}")
    return OutputTensor(out_name, dtype, shape, len(data), lambda f: f.write(data))


def ones_tensor(out_name: str, dtype: str, shape: list[int]) -> OutputTensor:
    count = 1
    for dim in shape:
        count *= dim
    if dtype == "BF16":
        data = b"\x80\x3f" * count
    elif dtype == "F32":
        data = struct.pack("<f", 1.0) * count
    elif dtype == "F16":
        data = require_numpy().ones(count, dtype="<f2").tobytes()
    else:
        raise ValueError(f"cannot create ones tensor for dtype {dtype}")
    return bytes_tensor(out_name, dtype, shape, data)


def row_slice_transposed(out_name: str, src: TensorEntry, row_start: int, rows: int) -> OutputTensor:
    if len(src.shape) != 2:
        raise ValueError(f"{src.name} must be rank-2")
    src_rows, src_cols = src.shape
    if row_start < 0 or row_start + rows > src_rows:
        raise ValueError(f"{src.name}: row slice out of bounds")
    elem = dtype_size(src.dtype)
    row_bytes = src_cols * elem
    with src.path.open("rb") as f:
        f.seek(src.data_offset + row_start * row_bytes)
        raw = f.read(rows * row_bytes)
    numpy = require_numpy()
    arr = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(rows, src_cols, elem)
    transposed = numpy.ascontiguousarray(arr.transpose(1, 0, 2)).tobytes()
    return bytes_tensor(out_name, src.dtype, [src_cols, rows], transposed)


def transposed_2d(out_name: str, src: TensorEntry) -> OutputTensor:
    if len(src.shape) != 2:
        raise ValueError(f"{src.name} must be rank-2")
    src_rows, src_cols = src.shape
    elem = dtype_size(src.dtype)
    raw = read_entry(src)
    numpy = require_numpy()
    arr = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(src_rows, src_cols, elem)
    transposed = numpy.ascontiguousarray(arr.transpose(1, 0, 2)).tobytes()
    return bytes_tensor(out_name, src.dtype, [src_cols, src_rows], transposed)


def column_slice(out_name: str, src: TensorEntry, col_start: int, cols: int) -> OutputTensor:
    if len(src.shape) != 2:
        raise ValueError(f"{src.name} must be rank-2")
    rows, src_cols = src.shape
    if col_start < 0 or col_start + cols > src_cols:
        raise ValueError(f"{src.name}: column slice out of bounds")
    elem = dtype_size(src.dtype)
    nbytes = rows * cols * elem

    def write(f) -> None:
        row_bytes = src_cols * elem
        slice_bytes = cols * elem
        batch_rows = 1024
        with src.path.open("rb") as in_f:
            mm = mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                base = src.data_offset
                for row0 in range(0, rows, batch_rows):
                    row1 = min(rows, row0 + batch_rows)
                    chunk = bytearray((row1 - row0) * slice_bytes)
                    out = 0
                    for row in range(row0, row1):
                        start = base + row * row_bytes + col_start * elem
                        chunk[out:out + slice_bytes] = mm[start:start + slice_bytes]
                        out += slice_bytes
                    f.write(chunk)
            finally:
                mm.close()

    return OutputTensor(out_name, src.dtype, [rows, cols], nbytes, write)


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


def text_name(name: str) -> tuple[str, str, str]:
    return (
        f"model.language_model.{name}",
        f"language_model.model.{name}",
        f"model.{name}",
        name,
    )


def convert_weights(src_dir: pathlib.Path, dst: pathlib.Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[weights] exists: {dst}")
        return

    entries = discover_safetensors(src_dir)
    if not entries:
        raise FileNotFoundError(f"no safetensors found under {src_dir}")

    out: list[OutputTensor] = []

    def add_exact(out_name: str, *src_names: str) -> TensorEntry:
        src = resolve(entries, *src_names)
        out.append(direct_tensor(out_name, src))
        return src

    embed = add_exact("language_model.model.embed_tokens.weight", *text_name("embed_tokens.weight"))
    add_exact("language_model.model.norm.weight", *text_name("norm.weight"))

    ple_embed = resolve(entries, *text_name("embed_tokens_per_layer.weight"))
    ple_proj = resolve(entries, *text_name("per_layer_model_projection.weight"))
    ple_norm = resolve(entries, *text_name("per_layer_projection_norm.weight"))

    for layer in range(LAYERS):
        base = f"language_model.model.layers.{layer}"
        src_base_names = (
            f"model.language_model.layers.{layer}",
            f"language_model.model.layers.{layer}",
            f"model.layers.{layer}",
            f"layers.{layer}",
        )

        def add_layer(suffix: str, required: bool = True) -> TensorEntry | None:
            names = tuple(f"{prefix}.{suffix}" for prefix in src_base_names)
            src = maybe_resolve(entries, *names)
            if src is None:
                if required:
                    raise KeyError("missing tensor; tried: " + ", ".join(names))
                return None
            out.append(direct_tensor(f"{base}.{suffix}", src))
            return src

        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "post_per_layer_input_norm.weight",
        ]:
            add_layer(suffix)

        input_gate = add_layer("per_layer_input_gate.weight", required=False)
        if input_gate is None:
            raise KeyError(f"missing tensor: {src_base_names[0]}.per_layer_input_gate.weight")
        out.pop()
        out.append(transposed_2d(f"{base}.per_layer_input_gate.weight", input_gate))

        projection = add_layer("per_layer_projection.weight", required=False)
        if projection is None:
            raise KeyError(f"missing tensor: {src_base_names[0]}.per_layer_projection.weight")
        out.pop()
        out.append(transposed_2d(f"{base}.per_layer_projection.weight", projection))

        if layer < FIRST_SHARED_KV_LAYER:
            for suffix in [
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.k_norm.weight",
            ]:
                add_layer(suffix)

        skip = add_layer("skip_scale", required=False)
        if skip is None:
            layer_scalar = add_layer("layer_scalar", required=False)
            if layer_scalar is not None:
                out[-1].name = f"{base}.skip_scale"
            else:
                out.append(ones_tensor(f"{base}.skip_scale", embed.dtype, [1]))

        ple_prefix = f"language_model.model.per_layer_inputs.{layer}"
        col0 = layer * PLE_DIM
        out.append(row_slice_transposed(f"{ple_prefix}.model_projection.weight", ple_proj, col0, PLE_DIM))
        out.append(direct_tensor(f"{ple_prefix}.projection_norm.weight", ple_norm))
        out.append(column_slice(f"{ple_prefix}.embedding.weight", ple_embed, col0, PLE_DIM))

    write_safetensors(dst, out)
    print(f"[weights] wrote: {dst}")


def write_input(path: pathlib.Path, ids: list[int], max_seq: int) -> int:
    real_ids = ids[-max_seq:] or [EOS_ID]
    token_index = len(real_ids) - 1
    padded = real_ids + [PAD_ID] * (max_seq - len(real_ids))
    input_data = require_numpy().asarray([padded], dtype="<i8").tobytes()
    write_safetensors(path, [
        bytes_tensor("input_ids", "I64", [1, max_seq], input_data),
    ])
    return token_index


def load_tokenizer(artifacts: pathlib.Path, model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for prompt tokenization; pass --ids or install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(artifacts if artifacts.exists() else model_id)


def encode_prompt(prompt: str, artifacts: pathlib.Path, model_id: str) -> list[int]:
    tokenizer = load_tokenizer(artifacts, model_id)
    return list(tokenizer.encode(prompt, add_special_tokens=False))


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
            "tokenizer*",
            "*.model",
            "*.json",
        ],
    )


def parse_ids(value: str) -> list[int]:
    ids = [int(part) for part in value.replace(" ", "").split(",") if part]
    if not ids:
        raise argparse.ArgumentTypeError("--ids must contain at least one token id")
    return ids


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Prepare and run Sandy Gemma4_E2B text-only BF16.")
    parser.add_argument("prompt", nargs="?", default="Hello, my name is")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--artifacts", default=root / "experiments/gemma4_e2b", type=pathlib.Path)
    parser.add_argument("--hf-weights", default=None, type=pathlib.Path,
                        help="Directory or safetensors file with original HF weights.")
    parser.add_argument("--sandy-weights", default=None, type=pathlib.Path)
    parser.add_argument("--model", default=root / "src/models/gemma4e2b.sandy.go", type=pathlib.Path)
    parser.add_argument("--runner", default=default_runner(root), type=pathlib.Path)
    parser.add_argument("--max-seq", default=MAX_SEQ, type=int)
    parser.add_argument("--ids", type=parse_ids, default=None,
                        help="Comma-separated token ids. Skips tokenizer loading.")
    parser.add_argument("--download", action="store_true",
                        help="Download the gated HF snapshot if artifacts are missing.")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-input", action="store_true")
    parser.add_argument("--instrument", action="store_true",
                        help="Print per-kernel CPU engine timing from cpu_runner.")
    args = parser.parse_args()

    if args.max_seq != MAX_SEQ:
        print(f"this SandyGo model is fixed to max_seq={MAX_SEQ}", file=sys.stderr)
        return 1

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

    ids = args.ids if args.ids is not None else encode_prompt(args.prompt, artifacts, args.model_id)
    if args.keep_input:
        input_path = artifacts / "input_latest.safetensors"
    else:
        fd, tmp_name = tempfile.mkstemp(prefix="sandy_gemma4e2b_input_", suffix=".safetensors")
        os.close(fd)
        input_path = pathlib.Path(tmp_name)
    token_index = write_input(input_path, ids, args.max_seq)

    print(f"[tokenizer] ids: {ids}")
    print(f"[input] next-token logits position: {token_index}")
    print(f"[input] safetensors: {input_path}")
    print(f"[weights] sandy: {sandy_weights}")

    if args.prepare_only:
        return 0

    if not args.runner.exists():
        print(f"missing cpu_runner: {args.runner}", file=sys.stderr)
        print("build with: cmake --build build --target cpu_runner", file=sys.stderr)
        return 1

    cmd = [str(args.runner)]
    if args.instrument:
        cmd.append("--instrument")
    cmd.extend([str(args.model), str(sandy_weights), str(input_path)])
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
