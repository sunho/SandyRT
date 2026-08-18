#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import os
import pathlib
import signal
import socket
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


DEFAULT_PREFILL_CHUNK_TOKENS = 2048


def model_defaults(root: pathlib.Path, architecture: str) -> dict[str, object]:
    if architecture == "tinyllama":
        return {
            "model": root / "src/models/tinyllama/eval_token.sandy.go",
            "prefill_model": root / "src/models/tinyllama/prefill.sandy.go",
            "weights": root / "experiments/tinyllama/sandy_model.bf16.safetensors",
            "tokenizer": root / "experiments/tinyllama",
            "model_id": "tinyllama",
            "eos_token_id": 2,
            "max_context_tokens": 2048,
            "prefill_chunk_tokens": DEFAULT_PREFILL_CHUNK_TOKENS,
        }
    if architecture == "gemma4e4b":
        return {
            "model": root / "src/models/gemma4e4b/eval_token.sandy.go",
            "prefill_model": root / "src/models/gemma4e4b/prefill.sandy.go",
            "weights": root / "experiments/gemma4_e4b/sandy_model.bf16.safetensors",
            "tokenizer": root / "experiments/gemma4_e4b",
            "model_id": "gemma4e4b",
            "eos_token_id": 1,
            "max_context_tokens": 0,
            "prefill_chunk_tokens": DEFAULT_PREFILL_CHUNK_TOKENS,
        }
    if architecture in {"gemma4a4b26b", "gemma4a4b", "gemma4moe"}:
        return {
            "model": root / "src/models/gemma4a4b26b/eval_token.sandy.go",
            "prefill_model": root / "src/models/gemma4a4b26b/prefill.sandy.go",
            "weights": root / "experiments/gemma4_a4b26b/sandy_model.bf16.safetensors",
            "tokenizer": root / "experiments/gemma4_a4b26b",
            "model_id": "gemma4a4b26b",
            "eos_token_id": 1,
            "max_context_tokens": 0,
            "prefill_chunk_tokens": DEFAULT_PREFILL_CHUNK_TOKENS,
        }
    return {
        "model": root / "src/models/gemma4e2b/eval_token.sandy.go",
        "prefill_model": root / "src/models/gemma4e2b/prefill.sandy.go",
        "weights": root / "experiments/gemma4_e2b/sandy_model.bf16.safetensors",
        "tokenizer": root / "experiments/gemma4_e2b",
        "model_id": "gemma4e2b",
        "eos_token_id": 1,
        "max_context_tokens": 0,
        "prefill_chunk_tokens": DEFAULT_PREFILL_CHUNK_TOKENS,
    }


def with_pythonpath(root: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    package_root = str(root / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
    return env


def with_cuda_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_lib = "/usr/local/cuda/lib64"
    wsl_lib = "/usr/lib/wsl/lib"
    current = env.get("LD_LIBRARY_PATH")
    parts = current.split(os.pathsep) if current else []
    additions = [path for path in [cuda_lib, wsl_lib] if path not in parts]
    if additions:
        prefix = os.pathsep.join(additions)
        env["LD_LIBRARY_PATH"] = prefix if not current else prefix + os.pathsep + current
    return env


def popen(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    kwargs: dict[str, object] = {"env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def grpc_tcp_endpoint(target: str) -> tuple[str, int] | None:
    if target.startswith("unix:"):
        return None
    host, sep, port_text = target.rpartition(":")
    if not sep:
        return None
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host:
        host = "127.0.0.1"
    try:
        port = int(port_text)
    except ValueError:
        return None
    return host, port


def wait_for_grpc_port(
        process: subprocess.Popen,
        target: str,
        timeout_seconds: float) -> bool:
    endpoint = grpc_tcp_endpoint(target)
    if endpoint is None:
        time.sleep(0.5)
        return process.poll() is None
    host, port = endpoint
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


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
        choices=[
            "gemma4e2b",
            "gemma4e4b",
            "gemma4a4b26b",
            "gemma4a4b",
            "gemma4moe",
            "gemma",
            "tinyllama",
        ],
    )
    parser.add_argument("--worker", default=default_worker(root), type=pathlib.Path)
    parser.add_argument(
        "--model",
        default=None,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--prefill-model",
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
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Bearer token required for /v1 HTTP requests. Prefer setting "
            "--auth-token-env instead of passing secrets on the command line."
        ),
    )
    parser.add_argument(
        "--auth-token-env",
        default="SANDY_API_TOKEN",
        help="Environment variable used to pass the bearer token to the HTTP server.",
    )
    parser.add_argument("--eos-token-id", default=None, type=int)
    parser.add_argument("--max-context-tokens", default=None, type=int)
    parser.add_argument("--prefill-chunk-tokens", default=None, type=int)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write one server-side log file per request.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write per-request logs and include engine kernel/stage profiling.",
    )
    parser.add_argument(
        "--log-dir",
        default=root / "logs/requests",
        type=pathlib.Path,
        help="Directory for request logs when --debug or --profile is enabled.",
    )
    parser.add_argument("--worker-start-timeout", default=300.0, type=float)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    defaults = model_defaults(root, args.architecture)
    if args.model is None:
        args.model = defaults["model"]
    if args.prefill_model is None:
        args.prefill_model = defaults["prefill_model"]
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
    if args.prefill_chunk_tokens is None:
        args.prefill_chunk_tokens = defaults["prefill_chunk_tokens"]

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
    if args.prefill_chunk_tokens < 0:
        print("--prefill-chunk-tokens must be >= 0", file=sys.stderr)
        return 1
    if args.prefill_chunk_tokens > 0 and not args.prefill_model.exists():
        print(f"missing prefill model: {args.prefill_model}", file=sys.stderr)
        return 1
    if not args.weights.exists():
        print(f"missing weights: {args.weights}", file=sys.stderr)
        return 1
    if not args.tokenizer.exists():
        print(f"missing tokenizer path: {args.tokenizer}", file=sys.stderr)
        return 1
    if args.auth_token is not None and not args.auth_token_env:
        print("--auth-token requires a non-empty --auth-token-env", file=sys.stderr)
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
    if args.prefill_chunk_tokens > 0:
        worker_cmd.extend([
            "--prefill-model", str(args.prefill_model),
            "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
        ])
    if args.debug:
        worker_cmd.append("--debug")
    if args.profile:
        worker_cmd.append("--profile")
    if args.debug or args.profile:
        worker_cmd.extend(["--log-dir", str(args.log_dir)])

    http_cmd = [
        args.python,
        "-m", "sandy_server.app",
        "--tokenizer", str(args.tokenizer),
        "--grpc", args.grpc_listen,
        "--host", args.http_host,
        "--port", str(args.http_port),
        "--model-id", args.model_id,
        "--eos-token-id", str(args.eos_token_id),
        "--auth-token-env", args.auth_token_env,
    ]
    http_env = with_pythonpath(root)
    if args.auth_token is not None:
        http_env[args.auth_token_env] = args.auth_token

    print("[start]", " ".join(worker_cmd))
    worker = popen(worker_cmd, env=with_cuda_worker_env())
    if not wait_for_grpc_port(worker, args.grpc_listen, args.worker_start_timeout):
        if worker.poll() is not None:
            return worker.returncode or 1
        terminate(worker, "grpc worker")
        wait_or_kill(worker, "grpc worker")
        print("[error] grpc worker did not become ready", file=sys.stderr)
        return 1
    if worker.poll() is not None:
        return worker.returncode or 1

    print("[start]", " ".join(http_cmd))
    http = popen(http_cmd, env=http_env)

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
