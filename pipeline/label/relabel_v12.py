"""Re-label the records most likely to move under schema v1.2.

The question this answers: how much of `other` is actually inference to the best
explanation? `other` carries 7 of the 16 form errors on test_real and is the
largest class there, so if IBE absorbs a real share of it that is the single
biggest accuracy win available. If it absorbs little, IBE is a minor addition and
the effort belongs in Phase 3 instead.

Only candidate classes are re-labelled — `sign` (expected to become IBE wholesale),
`other`, `analogy` and `causal` (both plausibly IBE), and `categorical syllogism`
(some may be application of generalization). Everything else is left alone: a full
re-label would churn labels that are already settled and cost far more calls.

    python pipeline/label/relabel_v12.py --corpus data/interim/corpus_all.jsonl
"""

import argparse
import collections
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from lfx.vertex import build_examples, client, label_one

CANDIDATES = {"sign", "other", "analogy", "causal", "categorical syllogism",
              "generalization"}

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/interim/corpus_all.jsonl")
ap.add_argument("--fewshot", default="data/raw/fewshot.jsonl")
ap.add_argument("--out", default="data/interim/relabel_v12.jsonl")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.corpus)]
todo = [r for r in rows if r["label"]["form"] in CANDIDATES]
if args.limit:
    todo = todo[: args.limit]
print(f"{len(rows)} records, {len(todo)} in candidate classes "
      f"({', '.join(sorted(CANDIDATES))})\n")

cl = client()
examples = build_examples(args.fewshot)
out = open(args.out, "w")
lock, n = threading.Lock(), [0]


def one(r):
    try:
        new = label_one(cl, examples, r["text"])
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {r['id']}: {type(exc).__name__}"[:110], file=sys.stderr)
        return
    with lock:
        out.write(json.dumps({
            "id": r["id"], "origin": r.get("origin"), "text": r["text"],
            "old_form": r["label"]["form"], "new_form": new["form"],
            "new_label": new,
        }, ensure_ascii=False) + "\n")
        out.flush()
        n[0] += 1
        if n[0] % 25 == 0:
            print(f"  {n[0]}/{len(todo)}", flush=True)


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(pool.map(one, todo))
out.close()

res = [json.loads(l) for l in open(args.out)]
print(f"\nre-labelled {len(res)}\n")
moves = collections.Counter((r["old_form"], r["new_form"]) for r in res)
stayed = sum(v for (a, b), v in moves.items() if a == b)
print(f"unchanged {stayed}/{len(res)}  ·  moved {len(res) - stayed}\n")
print("movements (old -> new):")
for (a, b), v in moves.most_common():
    if a != b:
        print(f"  {v:>3}  {a}  ->  {b}")
