# Gemma 4 MoE setup

This repository runs the text path of `google/gemma-4-26B-A4B-it` through the
SandyRT CUDA backend. The model uses BF16 weights and requires roughly 52 GB for
the original checkpoint, plus space for the converted SandyRT checkpoint.

## Bootstrap

```bash
./scripts/setup_gemma4_moe.sh
source .venv/bin/activate
```

The setup creates an isolated Python environment and builds the CUDA single-pass,
paged-KV generation, and device-test runners in `build-cuda/`.

## Model access and download

The Hugging Face repository may require accepting Google's model terms and
authenticating first:

```bash
hf auth login
python scripts/run_gemma4a4b26b.py --download --prepare-only
```

The second command downloads the original checkpoint into
`experiments/gemma4_a4b26b/` and converts it to
`sandy_model.bf16.safetensors`. Keep at least 110 GB free while both copies are
present. To use an existing snapshot elsewhere, pass `--hf-weights PATH` and
optionally `--sandy-weights PATH`.

## Run

Generate with the paged KV cache:

```bash
python scripts/run_gemma4a4b26b.py \
  --eval-token \
  --max-answer-tokens 32 \
  "Explain mixture-of-experts models briefly."
```

For a one-pass next-token smoke test, omit `--eval-token`. Add `--profile` to
either command for per-kernel timings.

