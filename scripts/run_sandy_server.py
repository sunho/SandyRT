#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import os
import pathlib
import signal
import subprocess
import sys
import time


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_worker(root: pathlib.Path) -> pathlib.Path:
    for worker in [
        root / "build-server-cuda/src/server/sandy_grpc_worker",
        root / "build-server-cublas/src/server/sandy_grpc_worker",
        root / "build-server/src/server/sandy_grpc_worker",
        root / "build/src/server/sandy_grpc_worker",
        root / "build-fast/src/server/sandy_grpc_worker",
        root / "build-opt/src/server/sandy_grpc_worker",
    ]:
        if worker.exists():
            return worker
    return root / "build-server/src/server/sandy_grpc_worker"


def model_defaults(root: pathlib.Path, architecture: str) -> dict[str, object]:
    if architecture == "tinyllama":
        return {
            "model": root / "src/models/tinyllama/eval_token.sandy.go",
            "weights": root / "experiments/tinyllama/sandy_model.bf16.safetensors",
            "tokenizer": root / "experiments/tinyllama",
            "model_id": "tinyllama",
            "eos_token_id": 2,
            "max_context_tokens": 2048,
        }
    return {
        "model": root / "src/models/gemma4e2b/eval_token.sandy.go",
        "weights": root / "experiments/gemma4_e2b/sandy_model.bf16.safetensors",
        "tokenizer": root / "experiments/gemma4_e2b",
        "model_id": "gemma4e2b",
        "eos_token_id": 1,
        "max_context_tokens": 0,
    }


def with_pythonpath(root: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    package_root = str(root / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
    return env


def with_cuda_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    wsl_lib = "/usr/lib/wsl/lib"
    current = env.get("LD_LIBRARY_PATH")
    parts = current.split(os.pathsep) if current else []
    if wsl_lib not in parts:
        env["LD_LIBRARY_PATH"] = wsl_lib if not current else wsl_lib + os.pathsep + current
    return env


def popen(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    kwargs: dict[str, object] = {"env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def terminate(process: subprocess.Popen, name: str) -> None:
    if process.poll() is not None:
        return
    print(f"[stop] {name}")
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def wait_or_kill(process: subprocess.Popen, name: str) -> int:
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[kill] {name}", file=sys.stderr)
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        return process.wait()


def stop_processes(processes: list[tuple[subprocess.Popen, str]]) -> int:
    exit_code = 0
    for process, name in processes:
        terminate(process, name)
    for process, name in processes:
        code = wait_or_kill(process, name)
        if exit_code == 0 and code != 0:
            exit_code = code
    return exit_code


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Run Sandy C++ gRPC worker and OpenAI-compatible HTTP server."
    )
    parser.add_argument(
        "--architecture",
        default="tinyllama",
        choices=["gemma4e2b", "gemma4e4b", "gemma", "tinyllama"],
    )
    parser.add_argument("--worker", default=default_worker(root), type=pathlib.Path)
    parser.add_argument(
        "--model",
        default=None,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        type=pathlib.Path,
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--grpc-listen", default="127.0.0.1:50051")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", default=8000, type=int)
    parser.add_argument("--eos-token-id", default=None, type=int)
    parser.add_argument("--max-context-tokens", default=None, type=int)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    defaults = model_defaults(root, args.architecture)
    if args.model is None:
        args.model = defaults["model"]
    if args.weights is None:
        args.weights = defaults["weights"]
    if args.tokenizer is None:
        args.tokenizer = defaults["tokenizer"]
    if args.model_id is None:
        args.model_id = defaults["model_id"]
    if args.eos_token_id is None:
        args.eos_token_id = defaults["eos_token_id"]
    if args.max_context_tokens is None:
        args.max_context_tokens = defaults["max_context_tokens"]

    if not args.worker.exists():
        print(f"missing sandy_grpc_worker: {args.worker}", file=sys.stderr)
        print(
            "build with: python3 scripts/build_sandy_server_cuda.py",
            file=sys.stderr,
        )
        return 1
    if not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 1
    if not args.weights.exists():
        print(f"missing weights: {args.weights}", file=sys.stderr)
        return 1
    if not args.tokenizer.exists():
        print(f"missing tokenizer path: {args.tokenizer}", file=sys.stderr)
        return 1

    worker_cmd = [
        str(args.worker),
        "--model", str(args.model),
        "--weights", str(args.weights),
        "--listen", args.grpc_listen,
        "--model-id", args.model_id,
        "--architecture", args.architecture,
        "--eos-token-id", str(args.eos_token_id),
    ]
    if args.max_context_tokens > 0:
        worker_cmd.extend(["--max-context-tokens", str(args.max_context_tokens)])

    http_cmd = [
        args.python,
        "-m", "sandy_server.app",
        "--tokenizer", str(args.tokenizer),
        "--grpc", args.grpc_listen,
        "--host", args.http_host,
        "--port", str(args.http_port),
        "--model-id", args.model_id,
        "--eos-token-id", str(args.eos_token_id),
    ]

    print("[start]", " ".join(worker_cmd))
    worker = popen(worker_cmd, env=with_cuda_worker_env())
    time.sleep(0.5)
    if worker.poll() is not None:
        return worker.returncode or 1

    print("[start]", " ".join(http_cmd))
    http = popen(http_cmd, env=with_pythonpath(root))

    processes = [(http, "http server"), (worker, "grpc worker")]
    stopping = False
    cleaned_up = False

    def cleanup() -> int:
        nonlocal cleaned_up
        if cleaned_up:
            return 0
        cleaned_up = True
        return stop_processes(processes)

    atexit.register(cleanup)

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(f"[signal] {signum}")
        cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    print(f"[ready] http://{args.http_host}:{args.http_port}")
    exit_code = 0
    try:
        while True:
            for process, name in processes:
                code = process.poll()
                if code is not None:
                    print(f"[exit] {name}: {code}")
                    exit_code = code
                    cleanup_code = cleanup()
                    if exit_code == 0:
                        exit_code = cleanup_code
                    return exit_code
            time.sleep(0.5)
    except KeyboardInterrupt:
        cleanup_code = cleanup()
        if exit_code == 0:
            exit_code = cleanup_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
