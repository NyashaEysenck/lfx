"""Fold the trained LoRA adapter into the base weights, producing a plain model.

LoRA keeps the base frozen and learns a low-rank correction: the effective weight
is W + (alpha/r) * B @ A. Merging evaluates that once and writes the result as
ordinary weights, so the output needs no `peft` at inference and can be handed to
MLX or llama.cpp, neither of which understands adapters.

Runs on CPU in float32 — the arithmetic is a one-off, and doing it in fp16 on MPS
risks rounding the correction away on small-magnitude weights.

    python merge_adapter.py --adapter lora_e6 --out merged_model
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", default="lora_e6")
ap.add_argument("--out", default="merged_model")
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
print(f"adapter    r={cfg['r']}  alpha={cfg['lora_alpha']}  "
      f"scaling={cfg['lora_alpha'] / cfg['r']:.2f}")
print(f"targets    {', '.join(sorted(cfg['target_modules']))}")

print(f"\nloading base {args.base} (fp32, cpu)")
model = AutoModelForCausalLM.from_pretrained(
    args.base, torch_dtype=torch.float32, device_map="cpu")
tok = AutoTokenizer.from_pretrained(args.base)

# Snapshot one targeted weight so we can prove the merge actually changed something.
probe = "model.layers.0.self_attn.q_proj.weight"
before = dict(model.named_parameters())[probe].detach().clone()

print(f"applying adapter {args.adapter}")
model = PeftModel.from_pretrained(model, args.adapter)

print("merging")
model = model.merge_and_unload()          # W <- W + (alpha/r) B@A, adapter discarded

after = dict(model.named_parameters())[probe].detach()
delta = (after - before).abs()
print(f"\nsanity check on {probe}")
print(f"  max |delta| {delta.max().item():.6f}   mean |delta| {delta.mean().item():.8f}")
if delta.max().item() == 0:
    raise SystemExit("ERROR: weights unchanged — the adapter did not apply")

print(f"\nsaving to {args.out}/ (fp16)")
model = model.half()
model.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)

size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
print(f"done — {size / 1e9:.2f} GB")
