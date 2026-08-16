#!/usr/bin/env python3

import argparse
import pathlib
import re
import subprocess
import sys

import run_gemma4e4b as gemma4e4b


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_runner(root: pathlib.Path) -> pathlib.Path:
    for runner in [
        root / "build-fast/multi_gemma4_runner",
        root / "build-opt/test/multi_gemma4_runner",
        root / "build/test/multi_gemma4_runner",
    ]:
        if runner.exists():
            return runner
    return root / "build-fast/multi_gemma4_runner"


def parse_ids(value: str) -> list[int]:
    return gemma4e4b.parse_ids(value)


def decode_tokens(tokenizer, ids: list[int]) -> list[str]:
    out = []
    for token_id in ids:
        try:
            out.append(tokenizer.decode([token_id]))
        except Exception:
            out.append("<decode-error>")
    return out


def parse_generated(stdout: str) -> list[int]:
    match = re.search(r"^\[generated\](.*)$", stdout, re.MULTILINE)
    if not match:
        return []
    return [int(part) for part in match.group(1).split()]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Run Gemma4 E4B for multiple argmax-generated tokens through Sandy."
    )
    parser.add_argument("prompt", nargs="?", default="Hello",
                        help="User-turn text. Ignored when --ids is passed.")
    parser.add_argument("--emit-count", type=int, default=8)
    parser.add_argument("--ids", type=parse_ids, default=None,
                        help="Comma-separated raw token ids. Skips chat tokenization.")
    parser.add_argument("--model-id", default=gemma4e4b.MODEL_ID)
    parser.add_argument("--artifacts", default=root / "experiments/gemma4_e4b", type=pathlib.Path)
    parser.add_argument("--hf-weights", default=None, type=pathlib.Path)
    parser.add_argument("--sandy-weights", default=None, type=pathlib.Path)
    parser.add_argument("--model", default=root / "src/models/gemma4e4b.sandy.go", type=pathlib.Path)
    parser.add_argument("--runner", default=default_runner(root), type=pathlib.Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    args = parser.parse_args()

    if args.emit_count < 0:
        print("--emit-count must be >= 0", file=sys.stderr)
        return 1

    artifacts = args.artifacts
    hf_weights = args.hf_weights or artifacts
    sandy_weights = args.sandy_weights or (artifacts / "sandy_model.bf16.safetensors")

    if args.download and not list(artifacts.glob("*.safetensors")):
        print(f"[download] {args.model_id} -> {artifacts}")
        gemma4e4b.download_snapshot(artifacts, args.model_id)

    if not pathlib.Path(hf_weights).exists():
        print(f"missing original weights: {hf_weights}", file=sys.stderr)
        print(f"download with: {sys.argv[0]} --download --emit-count 0", file=sys.stderr)
        return 1

    gemma4e4b.convert_weights(pathlib.Path(hf_weights), sandy_weights, args.force_convert)

    rendered_prompt = None
    if args.ids is not None:
        ids = args.ids
    else:
        ids, rendered_prompt = gemma4e4b.encode_prompt(args.prompt, artifacts, args.model_id)

    if not args.runner.exists():
        print(f"missing multi_gemma4_runner: {args.runner}", file=sys.stderr)
        print("build with: cmake --build build-fast --target multi_gemma4_runner", file=sys.stderr)
        return 1

    if rendered_prompt is not None:
        print(f"[chat] rendered prompt:\n{rendered_prompt}")
    print(f"[tokenizer] ids: {ids}")
    print(f"[weights] sandy: {sandy_weights}")

    cmd = [
        str(args.runner),
        str(args.model),
        str(sandy_weights),
        str(args.emit_count),
        *[str(token_id) for token_id in ids],
    ]
    print("[run]", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    generated = parse_generated(result.stdout)
    if not generated:
        return 0
    try:
        tokenizer = gemma4e4b.load_tokenizer(artifacts, args.model_id)
    except Exception as exc:
        print(f"[decode] skipped: {exc}", file=sys.stderr)
        return 0
    pieces = decode_tokens(tokenizer, generated)
    print(f"[generated_decoded] {pieces!r}")
    print(f"[generated_text] {tokenizer.decode(generated)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
