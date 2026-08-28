#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${SANDY_VENV_DIR:-${repo_root}/.venv}"
build_dir="${SANDY_BUILD_DIR:-${repo_root}/build-cuda}"
jobs="${SANDY_BUILD_JOBS:-$(nproc)}"
cuda_arch="${SANDY_CUDA_ARCHITECTURES:-}"

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc is required for the SandyRT CUDA runner" >&2; exit 1; }

if [[ -z "${cuda_arch}" ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required to detect the native CUDA architecture" >&2
    exit 1
  }
  cuda_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | head -n1 | tr -d '.[:space:]')"
  [[ -n "${cuda_arch}" ]] || { echo "could not detect CUDA compute capability" >&2; exit 1; }
fi

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install -r "${repo_root}/scripts/requirements-gemma4.txt"

"${venv_dir}/bin/cmake" \
  -S "${repo_root}" \
  -B "${build_dir}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O2 -DNDEBUG" \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
  -DSANDY_ENABLE_CUDA=ON \
  -DSANDY_BUILD_TESTS=ON
"${venv_dir}/bin/cmake" --build "${build_dir}" \
  --target cuda_runner cuda_multi_gemma4_runner cuda_device_tests \
  --parallel "${jobs}"

echo
echo "Gemma 4 MoE environment is ready."
echo "Activate it with: source ${venv_dir}/bin/activate"
echo "Download/prepare with: python scripts/run_gemma4a4b26b.py --download --prepare-only"
echo "Run with: python scripts/run_gemma4a4b26b.py --eval-token --max-answer-tokens 16 'Hello'"
