#!/usr/bin/env python3

import argparse
import pathlib
import re
import subprocess
import sys

import run_gemma4e2b as gemma4e2b


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_runner(root: pathlib.Path) -> pathlib.Path:
    for runner in [
        root / "build-cublas12/multi_gemma4_runner",
        root / "build-fast/multi_gemma4_runner",
        root / "build-opt/test/multi_gemma4_runner",
        root / "build/test/multi_gemma4_runner",
    ]:
        if runner.exists():
            return runner
    return root / "build-cublas12/multi_gemma4_runner"


def parse_generated(stdout: str) -> list[int]:
    match = re.search(r"^\[generated\](.*)$", stdout, re.MULTILINE)
    if not match:
        return []
    return [int(part) for part in match.group(1).split()]


def load_tokenizer(artifacts: pathlib.Path, model_id: str):
    tokenizer = gemma4e2b.load_tokenizer(artifacts, model_id)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = gemma4e2b.load_chat_template(artifacts)
    return tokenizer


def encode_messages(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], str]:
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
    return list(ids), rendered


def truncate_at_eos(ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return ids
    try:
        index = ids.index(eos_token_id)
    except ValueError:
        return ids
    return ids[:index]


def run_eval(
        runner: pathlib.Path,
        model: pathlib.Path,
        weights: pathlib.Path,
        max_answer_tokens: int,
        ids: list[int],
        profile: bool) -> tuple[list[int], str, str, int]:
    cmd = [str(runner), "--eval-token"]
    if profile:
        cmd.append("--profile")
    cmd.extend([
        str(model),
        str(weights),
        str(max_answer_tokens),
        *[str(token_id) for token_id in ids],
    ])
    result = subprocess.run(cmd, text=True, capture_output=True)
    return parse_generated(result.stdout), result.stdout, result.stderr, result.returncode


def answer_once(
        tokenizer,
        messages: list[dict[str, str]],
        runner: pathlib.Path,
        model: pathlib.Path,
        weights: pathlib.Path,
        max_answer_tokens: int,
        profile: bool,
        verbose: bool) -> tuple[str, list[int], int]:
    ids, rendered = encode_messages(tokenizer, messages)
    if verbose:
        print(f"[chat] rendered prompt:\n{rendered}")
        print(f"[tokenizer] ids: {ids}")
        print(
            "[run]",
            " ".join([
                str(runner),
                "--eval-token",
                str(model),
                str(weights),
                str(max_answer_tokens),
                "<ids>",
            ]),
        )

    generated, stdout, stderr, code = run_eval(
        runner,
        model,
        weights,
        max_answer_tokens,
        ids,
        profile)
    if stdout and (verbose or profile):
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    if code != 0:
        return "", generated, code
    if not generated:
        print("runner did not print a [generated] line", file=sys.stderr)
        return "", generated, 1

    trimmed = truncate_at_eos(generated, getattr(tokenizer, "eos_token_id", None))
    text = tokenizer.decode(trimmed, skip_special_tokens=True)
    return text, generated, 0


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Chat with Sandy Gemma4 E2B eval-token runner."
    )
    parser.add_argument("question", nargs="?",
                        help="Question to answer once. Omit for an interactive chat loop.")
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--model-id", default=gemma4e2b.MODEL_ID)
    parser.add_argument("--artifacts", default=root / "experiments/gemma4_e2b", type=pathlib.Path)
    parser.add_argument("--hf-weights", default=None, type=pathlib.Path)
    parser.add_argument("--sandy-weights", default=None, type=pathlib.Path)
    parser.add_argument("--model", default=root / "src/models/gemma4e2b/eval_token.sandy.go", type=pathlib.Path)
    parser.add_argument("--runner", default=default_runner(root), type=pathlib.Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_answer_tokens < 0:
        print("--max-answer-tokens must be >= 0", file=sys.stderr)
        return 1

    artifacts = args.artifacts
    hf_weights = args.hf_weights or artifacts
    sandy_weights = args.sandy_weights or (artifacts / "sandy_model.bf16.safetensors")

    if args.download and not list(artifacts.glob("*.safetensors")):
        print(f"[download] {args.model_id} -> {artifacts}")
        gemma4e2b.download_snapshot(artifacts, args.model_id)

    if not pathlib.Path(hf_weights).exists():
        print(f"missing original weights: {hf_weights}", file=sys.stderr)
        print(f"download with: {sys.argv[0]} --download", file=sys.stderr)
        return 1

    if args.force_convert or not sandy_weights.exists():
        gemma4e2b.convert_weights(pathlib.Path(hf_weights), sandy_weights, args.force_convert)

    if not args.runner.exists():
        print(f"missing multi_gemma4_runner: {args.runner}", file=sys.stderr)
        print("build with: cmake --build build-cublas12 --target multi_gemma4_runner", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"missing eval-token model: {args.model}", file=sys.stderr)
        return 1

    tokenizer = load_tokenizer(artifacts, args.model_id)
    messages: list[dict[str, str]] = []

    def ask(question: str) -> int:
        messages.append({"role": "user", "content": question})
        text, generated, code = answer_once(
            tokenizer,
            messages,
            args.runner,
            args.model,
            sandy_weights,
            args.max_answer_tokens,
            args.profile,
            args.verbose)
        if code != 0:
            return code
        messages.append({"role": "assistant", "content": text})
        if args.verbose:
            print(f"[generated_ids] {generated}")
        print(text)
        return 0

    if args.question is not None:
        return ask(args.question)

    print("Enter an empty line or Ctrl-D to exit.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not question:
            return 0
        code = ask(question)
        if code != 0:
            return code


if __name__ == "__main__":
    raise SystemExit(main())
