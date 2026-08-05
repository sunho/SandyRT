#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${repo_root}/experiments/gpt2_small"
base_url="https://huggingface.co/openai-community/gpt2/resolve/main"

mkdir -p "${out_dir}"

download() {
    local name="$1"
    local url="${base_url}/${name}"
    local out="${out_dir}/${name}"
    local tmp="${out}.tmp"

    if [[ -s "${out}" ]]; then
        echo "exists: ${out}"
        return
    fi

    echo "download: ${url}"
    curl -L --fail --retry 3 --retry-delay 2 -o "${tmp}" "${url}"
    mv "${tmp}" "${out}"
}

download "config.json"
download "vocab.json"
download "merges.txt"
download "model.safetensors"

echo "done: ${out_dir}"
