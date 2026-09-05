"""Merge an adapter, quantise to MLX, and freeze the prompt beside the weights.

Packaging used to be three commands typed by hand, which is how the prompt came
adrift from the weights: `lfx/prompts.py` was edited for schema v2.0, and the
already-shipped 3B -- trained on the older wording -- fell from 0.559 to 0.265
form accuracy and 1.000 to 0.853 schema validity on test_real without a single
weight changing. Nothing warned, because nothing tied the two together.

So the prompt is written INTO the model directory here, and `lfx.prompts.for_model`
reads it back at inference. A model now carries the contract it was trained under.

    python pipeline/train/package.py --adapter models/lora_3b \\
        --base unsloth/Qwen2.5-3B-Instruct --name 3b
"""

import argparse
import pathlib
import subprocess
import sys

from lfx.prompts import SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct")
ap.add_argument("--name", required=True, help="short tag, e.g. 3b -> models/mlx_3b_8bit")
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--skip-merge", action="store_true", help="reuse an existing merge")
args = ap.parse_args()

merged = pathlib.Path(f"models/merged_{args.name}")
mlx = pathlib.Path(f"models/mlx_{args.name}_{args.bits}bit")


def run(cmd):
    print("==>", " ".join(cmd))
    if subprocess.run(cmd).returncode:
        sys.exit(f"failed: {' '.join(cmd)}")


if not args.skip_merge:
    run([sys.executable, "pipeline/train/merge_adapter.py",
         "--base", args.base, "--adapter", args.adapter, "--out", str(merged)])
run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(merged),
     "--mlx-path", str(mlx), "-q", "--q-bits", str(args.bits)])

# The step that used to be missing.
(mlx / "prompt.txt").write_text(SYSTEM)
print(f"\nwrote {mlx}/prompt.txt ({len(SYSTEM)} chars) — the prompt this model "
      f"was trained with, and the one inference will use for it")
print(f"packaged {mlx}")
