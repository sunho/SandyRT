#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass

import numpy as np


MAX_SEQ = 16
GPT2_EOS = 50256


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_runner(root: pathlib.Path) -> pathlib.Path:
    opt_runner = root / "build-opt/test/cpu_runner"
    if opt_runner.exists():
        return opt_runner
    return root / "build/test/cpu_runner"


@dataclass
class SafeTensor:
    dtype: str
    shape: list[int]
    data: bytes


def read_safetensors(path: pathlib.Path) -> dict[str, SafeTensor]:
    blob = path.read_bytes()
    if len(blob) < 8:
        raise ValueError(f"{path} is not a safetensors file")

    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header_start = 8
    data_start = header_start + header_len
    header = json.loads(blob[header_start:data_start])

    tensors: dict[str, SafeTensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        tensors[name] = SafeTensor(
            dtype=meta["dtype"],
            shape=list(meta["shape"]),
            data=blob[data_start + begin:data_start + end],
        )
    return tensors


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()

    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)
    tmp.replace(path)


def tensor_as_f32(tensors: dict[str, SafeTensor], name: str) -> np.ndarray:
    tensor = tensors[name]
    if tensor.dtype != "F32":
        raise ValueError(f"{name} must be F32, got {tensor.dtype}")
    return np.frombuffer(tensor.data, dtype="<f4").reshape(tensor.shape)


def f32_tensor(array: np.ndarray) -> tuple[str, list[int], bytes]:
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return "F32", list(contiguous.shape), contiguous.tobytes()


def i64_tensor(array: np.ndarray) -> tuple[str, list[int], bytes]:
    contiguous = np.ascontiguousarray(array, dtype="<i8")
    return "I64", list(contiguous.shape), contiguous.tobytes()


def convert_gpt2_weights(src: pathlib.Path, dst: pathlib.Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[weights] exists: {dst}")
        return

    print(f"[weights] converting: {src} -> {dst}")
    hf = read_safetensors(src)
    out: dict[str, tuple[str, list[int], bytes]] = {}

    for name in ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]:
        out[name] = f32_tensor(tensor_as_f32(hf, name))

    hidden = 768
    for layer in range(12):
        prefix = f"h.{layer}"
        for name in ["ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"]:
            full = f"{prefix}.{name}"
            out[full] = f32_tensor(tensor_as_f32(hf, full))

        c_attn_w = tensor_as_f32(hf, f"{prefix}.attn.c_attn.weight")
        c_attn_b = tensor_as_f32(hf, f"{prefix}.attn.c_attn.bias")
        for proj, begin in [("q_proj", 0), ("k_proj", hidden), ("v_proj", hidden * 2)]:
            end = begin + hidden
            out[f"{prefix}.attn.{proj}.weight"] = f32_tensor(c_attn_w[:, begin:end].T)
            out[f"{prefix}.attn.{proj}.bias"] = f32_tensor(c_attn_b[begin:end])

        out[f"{prefix}.attn.c_proj.weight"] = f32_tensor(
            tensor_as_f32(hf, f"{prefix}.attn.c_proj.weight").T)
        out[f"{prefix}.attn.c_proj.bias"] = f32_tensor(
            tensor_as_f32(hf, f"{prefix}.attn.c_proj.bias"))

        out[f"{prefix}.mlp.c_fc.weight"] = f32_tensor(
            tensor_as_f32(hf, f"{prefix}.mlp.c_fc.weight").T)
        out[f"{prefix}.mlp.c_fc.bias"] = f32_tensor(
            tensor_as_f32(hf, f"{prefix}.mlp.c_fc.bias"))
        out[f"{prefix}.mlp.c_proj.weight"] = f32_tensor(
            tensor_as_f32(hf, f"{prefix}.mlp.c_proj.weight").T)
        out[f"{prefix}.mlp.c_proj.bias"] = f32_tensor(
            tensor_as_f32(hf, f"{prefix}.mlp.c_proj.bias"))

    write_safetensors(dst, out)
    print(f"[weights] wrote: {dst}")


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class GPT2Tokenizer:
    def __init__(self, vocab_path: pathlib.Path, merges_path: pathlib.Path):
        self.encoder: dict[str, int] = json.loads(vocab_path.read_text())
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: dict[str, str] = {}

        merges: list[tuple[str, str]] = []
        for line in merges_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split()
            merges.append((left, right))
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in self._pretokens(text):
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for piece in self._bpe(encoded).split(" "):
                ids.append(self.encoder[piece])
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in ids)
        return bytes(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")

    def _pretokens(self, text: str) -> list[str]:
        # Approximation of GPT-2's regex pre-tokenizer using only stdlib re.
        # It preserves leading spaces on word/number/punctuation chunks.
        pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+",
            re.UNICODE,
        )
        return [m.group(0) for m in pattern.finditer(text)]

    def _bpe(self, token: str) -> str:
        cached = self.cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        if len(word) == 1:
            return token

        pairs = self._pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, math.inf))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    @staticmethod
    def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def write_input(path: pathlib.Path, ids: list[int], max_seq: int) -> int:
    real_ids = ids[-max_seq:]
    if not real_ids:
        real_ids = [GPT2_EOS]
    real_len = len(real_ids)
    padded = real_ids + [GPT2_EOS] * (max_seq - real_len)
    positions = list(range(max_seq))
    write_safetensors(path, {
        "input_ids": i64_tensor(np.array([padded], dtype=np.int64)),
        "position_ids": i64_tensor(np.array([positions], dtype=np.int64)),
    })
    return real_len - 1


def parse_output_logits(stdout: str, token_index: int, vocab_size: int) -> np.ndarray:
    marker = "output0:"
    start = stdout.find(marker)
    if start < 0:
        raise RuntimeError("cpu_runner output did not contain output0")
    first_newline = stdout.find("\n", start)
    left = stdout.find("[", first_newline)
    right = stdout.find("]\n", left)
    if left < 0 or right < 0:
        raise RuntimeError("could not parse output0 tensor values")
    values = np.fromstring(stdout[left + 1:right], sep=",", dtype=np.float32)
    expected = MAX_SEQ * vocab_size
    if values.size != expected:
        raise RuntimeError(f"expected {expected} logits, parsed {values.size}")
    return values.reshape(1, MAX_SEQ, vocab_size)[0, token_index]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Hello, my name is")
    parser.add_argument("--runner", default=default_runner(root), type=pathlib.Path)
    parser.add_argument("--model", default=root / "src/models/gpt2_small.sandy.go", type=pathlib.Path)
    parser.add_argument("--artifacts", default=root / "experiments/gpt2_small", type=pathlib.Path)
    parser.add_argument("--max-seq", default=MAX_SEQ, type=int)
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-input", action="store_true")
    args = parser.parse_args()

    if args.max_seq != MAX_SEQ:
        print(f"this SandyGo model is fixed to max_seq={MAX_SEQ}", file=sys.stderr)
        return 1
    if not args.runner.exists():
        print(f"missing cpu_runner: {args.runner}", file=sys.stderr)
        print("build with: cmake --build build-opt --target cpu_runner", file=sys.stderr)
        return 1

    hf_weights = args.artifacts / "model.safetensors"
    sandy_weights = args.artifacts / "sandy_model.safetensors"
    vocab = args.artifacts / "vocab.json"
    merges = args.artifacts / "merges.txt"
    for path in [hf_weights, vocab, merges]:
        if not path.exists():
            print(f"missing {path}; run scripts/download_gpt2_small.sh", file=sys.stderr)
            return 1

    convert_gpt2_weights(hf_weights, sandy_weights, args.force_convert)

    tokenizer = GPT2Tokenizer(vocab, merges)
    ids = tokenizer.encode(args.prompt)
    if args.keep_input:
        input_path = args.artifacts / "input_latest.safetensors"
    else:
        fd, tmp_name = tempfile.mkstemp(prefix="sandy_gpt2_input_", suffix=".safetensors")
        os.close(fd)
        input_path = pathlib.Path(tmp_name)
    token_index = write_input(input_path, ids, args.max_seq)

    print(f"[tokenizer] ids: {ids}")
    print(f"[tokenizer] decoded: {tokenizer.decode(ids)}")
    print(f"[input] next-token logits position: {token_index}")
    print(f"[input] safetensors: {input_path}")

    if args.prepare_only:
        return 0

    cmd = [str(args.runner), str(args.model), str(sandy_weights), str(input_path)]
    print("[run]", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(result.stdout, end="")
        return result.returncode

    logits = parse_output_logits(result.stdout, token_index, len(tokenizer.encoder))
    top = np.argpartition(-logits, args.top_k)[:args.top_k]
    top = top[np.argsort(-logits[top])]

    print("[top-k]")
    for rank, token_id in enumerate(top, 1):
        piece = tokenizer.decode([int(token_id)])
        print(f"{rank:2d}. id={int(token_id):5d} logit={float(logits[token_id]): .6g} text={piece!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
