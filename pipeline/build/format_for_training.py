"""Convert a labeled split into the chat format Unsloth/TRL expects.

Emits {"conversations": [{"role": "...", "content": "..."}]} per line — the
schema instruction as the system turn, the raw passage as the user turn, and the
JSON label as the assistant turn.

The system prompt comes from lfx.prompts so training, evaluation and the shipped
CLI cannot drift apart; if they did, the before/after comparison would measure
prompt wording rather than the effect of fine-tuning.
"""

import argparse
import json

from lfx.prompts import SYSTEM


ap = argparse.ArgumentParser()
ap.add_argument("splits", nargs="+", help="e.g. train val test_gen test_real")
args = ap.parse_args()

for name in args.splits:
    rows = [json.loads(l) for l in open(f"data/splits/{name}.jsonl")]
    out = f"data/splits/{name}.chat.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            label = {k: r["label"][k] for k in
                     ("premises", "conclusion", "argument_type", "form", "suppressed_premise")}
            fh.write(json.dumps({"conversations": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["text"]},
                {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
            ]}, ensure_ascii=False) + "\n")
    print(f"{out}: {len(rows)} examples")
