"""How often is the correct label the model's SECOND choice?

If top-1 is mediocre but top-2 is high, the model has narrowed correctly and the
task is contested between two plausible labels — a different problem from the model
misreading the passage, and one that calls for a different fix.

Candidates are scored against the model's OWN generated prefix, truncated at the
form field. Scoring after a synthetic prefix measures nothing: a first attempt used
an empty premises array and reported top-1 of 0.147 where real generation scores
0.500, because the model never emits that context.
"""

import argparse
import collections
import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from lfx.prompts import SYSTEM
from lfx.schema import FORMS

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test_real")
ap.add_argument("--preds", default="results/preds_v4_real.jsonl")
ap.add_argument("--model", default="models/mlx_v4_8bit")
ap.add_argument("--k", type=int, default=3)
args = ap.parse_args()

rows = [json.loads(l) for l in open(f"data/splits/{args.split}.jsonl")]
preds = {json.loads(l)["id"]: json.loads(l) for l in open(args.preds)}
model, tok = load(args.model)

MARK = '"form": "'
hits, ranks, skipped, agree = collections.Counter(), [], 0, []

for i, r in enumerate(rows, 1):
    raw = (preds.get(r["id"]) or {}).get("raw", "") or ""
    if MARK not in raw:
        skipped += 1
        continue
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    forced = prompt + raw[: raw.index(MARK) + len(MARK)]
    n_pref = len(tok.encode(forced)) - 1

    scored = []
    for form in FORMS:
        ids = mx.array([tok.encode(forced + form + '"')])
        lp = nn.log_softmax(model(ids[:, :-1]), axis=-1)
        tgt = ids[0, 1:]
        tok_lp = mx.take_along_axis(lp[0], tgt[:, None], axis=-1).squeeze(-1)
        vals = tok_lp[n_pref:]
        # MEAN, not sum: summing log-probs penalises multi-token labels
        # ("inference to the best explanation") against short ones ("causal"),
        # which is a length artefact, not the model's preference.
        scored.append((float(vals.mean()), form))
    scored.sort(reverse=True)

    order = [f for _, f in scored]
    # sanity: does this ranking reproduce what the model actually generated?
    gen = ((preds.get(r["id"]) or {}).get("parsed") or {}).get("form")
    if gen is not None:
        agree.append(order[0] == gen)
    gold = r["label"]["form"]
    rank = order.index(gold) + 1 if gold in order else 99
    ranks.append(rank)
    for k in range(1, args.k + 1):
        if rank <= k:
            hits[k] += 1
    if i % 10 == 0:
        print(f"  {i}/{len(rows)}", flush=True)

n = len(ranks)
print(f"\n{args.split}: {n} scored, {skipped} skipped (no form field in output)\n")
for k in range(1, args.k + 1):
    print(f"  top-{k}   {hits[k]}/{n} = {hits[k] / n:.3f}")
print(f"\n  median rank of the correct label: {sorted(ranks)[n // 2]}")
if agree:
    a = sum(agree) / len(agree)
    print(f"\n  VALIDITY CHECK — this ranking's top-1 matches what the model "
          f"actually generated {sum(agree)}/{len(agree)} = {a:.0%} of the time.")
    if a < 0.8:
        print("  Below 80%: the scoring does not reproduce the model's own choice, "
              "so the top-k figures above are NOT trustworthy.")
