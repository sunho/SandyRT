"""Export Gemma 4 E2B model to StableHLO MLIR.

Usage:
    # Graph only (dummy weights, fast):
    python scripts/export_gemma_stablehlo.py -o model.mlir

    # With real weights from GCS (needs gcloud auth):
    python scripts/export_gemma_stablehlo.py \
        --checkpoint gs://gemma-data/checkpoints/gemma4-e2b-pt \
        -o model.mlir

Requirements:
    pip install gemma jax jaxlib
"""

import argparse
import numpy as np
# https://github.com/google-research/kauldron/issues/1416
if not hasattr(np, "float128"):
    np.float128 = np.longdouble

import jax
import jax.numpy as jnp
from jax import export as jax_export
import gemma.gm as gm


def main():
    parser = argparse.ArgumentParser(description="Export Gemma4-E2B to StableHLO")
    parser.add_argument("-o", "--output", required=True, help="Output .mlir file path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Orbax checkpoint path (GCS or local). If omitted, uses random params.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--text-only", action="store_true", default=True,
                        help="Export text-only forward pass (no vision/audio)")
    args = parser.parse_args()

    print("Creating Gemma4_E2B model...")
    model = gm.nn.Gemma4_E2B()

    dummy_tokens = jnp.ones((args.batch_size, args.seq_len), dtype=jnp.int32)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        params = gm.ckpts.load_params(args.checkpoint, text_only=args.text_only)
    else:
        print("Initializing with random parameters (graph export only)...")
        params = model.init(jax.random.PRNGKey(0), dummy_tokens)

    def forward(params, tokens):
        output = model.apply(params, tokens)
        return output.logits

    print(f"Exporting with shape: batch={args.batch_size}, seq_len={args.seq_len}...")
    jitted = jax.jit(forward)
    exported = jax_export.export(jitted)(params, dummy_tokens)

    mlir_module = exported.mlir_module()
    if isinstance(mlir_module, str):
        mlir_text = mlir_module
    else:
        mlir_text = mlir_module.operation.get_asm(
            large_elements_limit=0,
            enable_debug_info=False,
        )

    with open(args.output, "w") as f:
        f.write(mlir_text)

    print(f"StableHLO written to {args.output}")
    print(f"  Size: {len(mlir_text) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
