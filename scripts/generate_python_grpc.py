#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import argparse
import importlib.util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Sandy Python gRPC stubs.")
    parser.add_argument("--protoc", help="Path to a protoc binary.")
    parser.add_argument(
        "--grpc-python-plugin",
        help="Path to grpc_python_plugin, required when --protoc is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    proto = root / "proto" / "sandy_inference.proto"
    out = root / "python" / "sandy_server" / "generated"
    out.mkdir(parents=True, exist_ok=True)

    if args.protoc:
        if not args.grpc_python_plugin:
            print("--grpc-python-plugin is required with --protoc", file=sys.stderr)
            return 2
        cmd = [
            args.protoc,
            f"-I{proto.parent}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            f"--plugin=protoc-gen-grpc_python={args.grpc_python_plugin}",
            str(proto),
        ]
    else:
        if importlib.util.find_spec("grpc_tools.protoc") is None:
            print(
                "grpc_tools.protoc is not installed. Install grpcio-tools or pass "
                "--protoc and --grpc-python-plugin from a CMake gRPC build.",
                file=sys.stderr,
            )
            return 1
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto.parent}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            str(proto),
        ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
