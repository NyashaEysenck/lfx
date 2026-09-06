"""Materialise the tied output head so GGUF converters can read the model.

Qwen2.5 sets tie_word_embeddings=True: one weight matrix serves as both the input
embedding and the output projection, and `lm_head.weight` is simply absent from
the checkpoint. transformers and MLX know to tie them. Ollama's safetensors
importer does not -- it builds a model with no valid output head, which generates
pure noise.

That failure is nasty because everything else looks right. `ollama show --system`
returns the correct prompt, the architecture and parameter count are correct, the
quantisation reports success, and the merged weights score perfectly under
transformers. Only the output is wrong, and it is wrong in a way that reads like
a broken quantisation rather than a missing tensor: this cost a full q4 build, an
F16 build, and a template fix before the cause was found.

So the head is written out explicitly, and tie_word_embeddings is turned off to
match. The result is still a valid HF checkpoint -- untied is the ordinary case --
and costs vocab_size * hidden_size * dtype bytes, about 600 MB for the 3B.

    python pipeline/deploy/untie_embeddings.py --model models/merged_v5
"""

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="float16")
args = ap.parse_args()

d = pathlib.Path(args.model)
cfg = json.loads((d / "config.json").read_text())

if not cfg.get("tie_word_embeddings"):
    print(f"{d}: tie_word_embeddings already False — nothing to do")
    raise SystemExit(0)

st = d / "model.safetensors"
tensors = load_file(str(st))
emb = next(k for k in tensors if k.endswith("embed_tokens.weight"))
if "lm_head.weight" in tensors:
    print("lm_head.weight already present — nothing to do")
    raise SystemExit(0)

dtype = getattr(torch, args.dtype)
print(f"tying {emb} -> lm_head.weight  {tuple(tensors[emb].shape)}  as {args.dtype}")
tensors = {k: v.to(dtype) for k, v in tensors.items()}
tensors["lm_head.weight"] = tensors[emb].clone()

shutil.copy(st, st.with_suffix(".safetensors.bak"))
save_file(tensors, str(st), metadata={"format": "pt"})

cfg["tie_word_embeddings"] = False
cfg["torch_dtype"] = args.dtype
(d / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"wrote {st} with {len(tensors)} tensors; tie_word_embeddings=False, "
      f"torch_dtype={args.dtype}\nbackup at {st.with_suffix('.safetensors.bak')}")
