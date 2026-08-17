#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Build Sandy gRPC worker with CUDA execution."
    )
    parser.add_argument("--build-dir", default=root / "build-server-cuda", type=pathlib.Path)
    parser.add_argument(
        "--grpc-prefix",
        type=pathlib.Path,
        help="Install prefix containing gRPCConfig.cmake, if gRPC is not on CMake's default path.",
    )
    parser.add_argument("--jobs", "-j", type=int)
    args = parser.parse_args()

    configure_cmd = [
        "cmake",
        "-S", str(root),
        "-B", str(args.build_dir),
        "-DSANDY_ENABLE_SERVER=ON",
        "-DSANDY_BUILD_TESTS=OFF",
        "-DSANDY_ENABLE_CUDA=ON",
        "-DSANDY_USE_BUNDLED_GRPC=OFF",
    ]
    if args.grpc_prefix is not None:
        configure_cmd.append(f"-DCMAKE_PREFIX_PATH={args.grpc_prefix}")

    build_cmd = [
        "cmake",
        "--build", str(args.build_dir),
        "--target", "sandy_grpc_worker",
    ]
    if args.jobs is not None:
        build_cmd.extend(["-j", str(args.jobs)])

    print("[configure]", " ".join(configure_cmd))
    code = subprocess.call(configure_cmd)
    if code != 0:
        return code

    print("[build]", " ".join(build_cmd))
    return subprocess.call(build_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
