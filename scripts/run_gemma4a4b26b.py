#!/usr/bin/env python3

import argparse
import json
import mmap
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import run_gemma4e4b as gemma4e4b


MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_MAX_SEQ = 0
EOS_ID = 1
LAYERS = 30
MOE_INTERMEDIATE = 704


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_runner(root: pathlib.Path) -> pathlib.Path:
    for runner in [
        root / "build-cuda/test/cuda_runner",
        root / "build/test/cuda_runner",
        root / "build-fast/cpu_runner",
        root / "build-opt/test/cpu_runner",
    ]:
        if runner.exists():
            return runner
    return root / "build-cuda/test/cuda_runner"


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


def parse_generated(stdout: str) -> list[int]:
    match = re.search(r"^\[generated\](.*)$", stdout, re.MULTILINE)
    if not match:
        return []
    return [int(part) for part in match.group(1).split()]


def load_text_config(root: pathlib.Path) -> dict:
    config_path = root / "config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text())
    return config.get("text_config", config)


def layer_source_names(layer: int, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    prefixes = (
        f"model.language_model.layers.{layer}",
        f"language_model.model.layers.{layer}",
        f"model.layers.{layer}",
        f"layers.{layer}",
    )
    return tuple(f"{prefix}.{suffix}" for suffix in suffixes for prefix in prefixes)


def rank3_dim1_slice(
        out_name: str,
        src: gemma4e4b.TensorEntry,
        start: int,
        rows: int) -> gemma4e4b.OutputTensor:
    if len(src.shape) != 3:
        raise ValueError(f"{src.name} must be rank-3")
    experts, src_rows, hidden = src.shape
    if start < 0 or rows <= 0 or start + rows > src_rows:
        raise ValueError(f"{src.name}: dim-1 slice out of bounds")

    elem = gemma4e4b.dtype_size(src.dtype)
    expert_bytes = src_rows * hidden * elem
    slice_bytes = rows * hidden * elem
    offset_in_expert = start * hidden * elem
    nbytes = experts * slice_bytes

    def write(f) -> None:
        with src.path.open("rb") as in_f:
            mm = mmap.mmap(in_f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                base = src.data_offset
                for expert in range(experts):
                    read_start = base + expert * expert_bytes + offset_in_expert
                    f.write(mm[read_start:read_start + slice_bytes])
            finally:
                mm.close()

    return gemma4e4b.OutputTensor(out_name, src.dtype, [experts, rows, hidden], nbytes, write)


def convert_weights(src_dir: pathlib.Path, dst: pathlib.Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[weights] exists: {dst}")
        return

    entries = gemma4e4b.discover_safetensors(src_dir)
    if not entries:
        raise FileNotFoundError(f"no safetensors found under {src_dir}")

    config = load_text_config(src_dir if src_dir.is_dir() else src_dir.parent)
    attention_k_eq_v = bool(config.get("attention_k_eq_v", True))

    out: list[gemma4e4b.OutputTensor] = []

    def add_exact(out_name: str, *src_names: str) -> gemma4e4b.TensorEntry:
        src = gemma4e4b.resolve(entries, *src_names)
        out.append(gemma4e4b.direct_tensor(out_name, src))
        return src

    embed = add_exact(
        "language_model.model.embed_tokens.weight",
        *gemma4e4b.text_name("embed_tokens.weight"))
    add_exact("language_model.model.norm.weight", *gemma4e4b.text_name("norm.weight"))

    for layer in range(LAYERS):
        base = f"language_model.model.layers.{layer}"

        def resolve_layer(*suffixes: str, required: bool = True) -> gemma4e4b.TensorEntry | None:
            names = layer_source_names(layer, suffixes)
            src = gemma4e4b.maybe_resolve(entries, *names)
            if src is None and required:
                raise KeyError("missing tensor; tried: " + ", ".join(names))
            return src

        def add_layer(out_suffix: str, *src_suffixes: str) -> gemma4e4b.TensorEntry:
            src = resolve_layer(*(src_suffixes or (out_suffix,)))
            assert src is not None
            out.append(gemma4e4b.direct_tensor(f"{base}.{out_suffix}", src))
            return src

        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "router.scale",
            "router.proj.weight",
        ]:
            add_layer(suffix)

        add_layer("router.per_expert_scale.weight", "router.per_expert_scale", "router.per_expert_scale.weight")

        k_proj = add_layer("self_attn.k_proj.weight")
        v_proj = resolve_layer("self_attn.v_proj.weight", required=False)
        if v_proj is not None:
            out.append(gemma4e4b.direct_tensor(f"{base}.self_attn.v_proj.weight", v_proj))
        elif attention_k_eq_v:
            out.append(gemma4e4b.direct_tensor(f"{base}.self_attn.v_proj.weight", k_proj))
        else:
            raise KeyError(f"missing tensor: model.language_model.layers.{layer}.self_attn.v_proj.weight")

        add_layer("skip_scale", "skip_scale", "layer_scalar")

        add_layer("experts.down_proj.weight", "experts.down_proj", "experts.down_proj.weight")
        gate_up = resolve_layer(
            "experts.gate_up_proj",
            "experts.gate_up_proj.weight",
            "experts.up_gate_proj",
            "experts.up_gate_proj.weight")
        assert gate_up is not None
        if gate_up.shape[1] != 2 * MOE_INTERMEDIATE:
            raise ValueError(
                f"{gate_up.name}: expected packed gate/up dim {2 * MOE_INTERMEDIATE}, "
                f"got {gate_up.shape[1]}")
        out.append(rank3_dim1_slice(f"{base}.experts.gate_proj.weight", gate_up, 0, MOE_INTERMEDIATE))
        out.append(rank3_dim1_slice(
            f"{base}.experts.up_proj.weight",
            gate_up,
            MOE_INTERMEDIATE,
            MOE_INTERMEDIATE))

    print(f"[weights] source text tensors: {sum(name.startswith('model.language_model') for name in entries)}")
    print(f"[weights] output tensors: {len(out)}")
    print(f"[weights] embedding dtype: {embed.dtype}")
    gemma4e4b.write_safetensors(dst, out)
    print(f"[weights] wrote: {dst}")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Prepare and run Sandy Gemma 4 26B-A4B MoE BF16 chat.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Hello, my name is",
        help="User-turn text. The runner applies the model chat template.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--artifacts", default=root / "experiments/gemma4_a4b26b", type=pathlib.Path)
    parser.add_argument(
        "--hf-weights",
        default=None,
        type=pathlib.Path,
        help="Directory or safetensors file with original HF weights.")
    parser.add_argument("--sandy-weights", default=None, type=pathlib.Path)
    parser.add_argument("--model", default=root / "src/models/gemma4a4b26b.sandy.go", type=pathlib.Path)
    parser.add_argument("--eval-model", default=root / "src/models/gemma4a4b26b/eval_token.sandy.go", type=pathlib.Path)
    parser.add_argument("--prefill-model", default=root / "src/models/gemma4a4b26b/prefill.sandy.go", type=pathlib.Path)
    parser.add_argument("--runner", default=default_runner(root), type=pathlib.Path)
    parser.add_argument("--eval-runner", default=default_eval_runner(root), type=pathlib.Path)
    parser.add_argument(
        "--max-seq",
        default=DEFAULT_MAX_SEQ,
        type=int,
        help="Optional truncation cap. 0 keeps the full prompt.")
    parser.add_argument("--max-answer-tokens", default=1, type=int)
    parser.add_argument(
        "--ids",
        type=gemma4e4b.parse_ids,
        default=None,
        help="Comma-separated token ids. Skips tokenizer loading.")
    parser.add_argument(
        "--eval-token",
        action="store_true",
        help="Run the decoder eval-token model with paged KV caches.")
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="Run the paged-KV prefill model once for the full prompt.")
    parser.add_argument("--download", action="store_true", help="Download the HF snapshot.")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-input", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Print per-kernel engine timing from the runner.")
    parser.add_argument("--instrument", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_seq < 0:
        print("--max-seq must be >= 0", file=sys.stderr)
        return 1
    if args.max_answer_tokens < 0:
        print("--max-answer-tokens must be >= 0", file=sys.stderr)
        return 1

    artifacts = args.artifacts
    hf_weights = args.hf_weights or artifacts
    sandy_weights = args.sandy_weights or (artifacts / "sandy_model.bf16.safetensors")

    if args.download:
        print(f"[download] {args.model_id} -> {artifacts}")
        gemma4e4b.download_snapshot(artifacts, args.model_id)

    if not pathlib.Path(hf_weights).exists():
        print(f"missing original weights: {hf_weights}", file=sys.stderr)
        print(f"download with: {sys.argv[0]} --download --prepare-only", file=sys.stderr)
        return 1

    convert_weights(pathlib.Path(hf_weights), sandy_weights, args.force_convert)

    rendered_prompt = None
    if args.ids is not None:
        ids = args.ids
    else:
        ids, rendered_prompt = gemma4e4b.encode_prompt(args.prompt, artifacts, args.model_id)
        if args.max_seq > 0 and len(ids) > args.max_seq:
            print(
                f"chat-formatted prompt is {len(ids)} tokens; truncating to the last "
                f"{args.max_seq} tokens",
                file=sys.stderr,
            )

    if args.keep_input:
        input_path = artifacts / "input_latest.safetensors"
    else:
        fd, tmp_name = tempfile.mkstemp(prefix="sandy_gemma4a4b26b_input_", suffix=".safetensors")
        os.close(fd)
        input_path = pathlib.Path(tmp_name)
    token_index = gemma4e4b.write_input(input_path, ids, args.max_seq)

    if rendered_prompt is not None:
        print(f"[chat] rendered prompt:\n{rendered_prompt}")
    print(f"[tokenizer] ids: {ids}")

    if args.eval_token or args.prefill:
        if args.max_seq > 0 and len(ids) > args.max_seq:
            ids = ids[-args.max_seq:]
            print(f"[tokenizer] truncated ids: {ids}")
        print(f"[weights] sandy: {sandy_weights}")
        if args.prepare_only:
            return 0
        if not args.eval_runner.exists():
            print(f"missing eval-token runner: {args.eval_runner}", file=sys.stderr)
            print("build with: cmake --build build-cuda --target cuda_multi_gemma4_runner", file=sys.stderr)
            return 1
        run_model = args.prefill_model if args.prefill else args.eval_model
        if not run_model.exists():
            print(f"missing paged-KV model: {run_model}", file=sys.stderr)
            return 1

        cmd = [
            str(args.eval_runner),
            "--prefill" if args.prefill else "--eval-token",
            "--architecture", "gemma4a4b26b",
        ]
        if args.profile or args.instrument:
            cmd.append("--profile")
        cmd.extend([
            str(run_model),
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
        if generated and args.ids is None:
            tokenizer = gemma4e4b.load_tokenizer(artifacts, args.model_id)
            print("[generated decoded]")
            print(tokenizer.decode(generated, skip_special_tokens=False))
        return 0

    print(f"[input] next-token logits position: {token_index}")
    print(f"[input] safetensors: {input_path}")
    print(f"[weights] sandy: {sandy_weights}")

    if args.prepare_only:
        return 0

    if not args.runner.exists():
        print(f"missing runner: {args.runner}", file=sys.stderr)
        print("build with: cmake --build build-cuda --target cuda_runner", file=sys.stderr)
        return 1

    cmd = [str(args.runner)]
    if args.profile or args.instrument:
        cmd.append("--profile")
    cmd.extend([str(args.model), str(sandy_weights), str(input_path)])
    print("[run]", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        gemma4e4b.print_decoded_runner_tokens(result.stdout, artifacts, args.model_id)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
