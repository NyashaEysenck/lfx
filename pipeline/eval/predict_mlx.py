"""Run the local MLX model over a split, in the same output format as predict.py,
so evaluate.py can compare the packaged model against the Colab-side numbers.

Same prompts, same greedy decoding, same JSON extraction — the only difference
from predict.py is the runtime. Any score gap is therefore attributable to
merging and quantization, which is the thing being measured.
"""
import argparse, json
from lfx.jsonio import extract_json
from lfx.prompts import SYSTEM
from mlx_lm import generate, load

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--model", default="models/mlx_v2_8bit")
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = [json.loads(l) for l in open(f"data/splits/{args.split}.jsonl")]
model, tok = load(args.model)
sys_msg = SYSTEM

with open(args.out, "w") as fh:
    for i, r in enumerate(rows, 1):
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": r["text"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        raw = generate(model, tok, prompt=prompt, max_tokens=768, verbose=False)
        fh.write(json.dumps({"id": r["id"], "raw": raw, "parsed": extract_json(raw)},
                            ensure_ascii=False) + "\n")
        fh.flush()
        if i % 10 == 0: print(f"  {i}/{len(rows)}", flush=True)
print(f"wrote {args.out}")
