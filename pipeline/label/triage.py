"""Compare each generated passage's INTENDED form against the form the labeler
independently assigned, and rank what a human should look at.

The labeler never sees the intended form, so agreement is real evidence the
passage instantiates what it was meant to. Disagreement means one of two things,
both worth a human's time: the generator wrote the wrong structure, or the
labeler misread it.
"""

import collections
import json
import sys

lab = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1
                                                        else "data/interim/generated_labeled.jsonl")}
gen = {json.loads(l)["id"]: json.loads(l) for l in open("data/interim/generated_corpus.jsonl")}

rows = []
for rid, g in gen.items():
    if rid not in lab:
        continue
    got = lab[rid]["label"]["form"]
    rows.append((rid, g["intended_form"], got, g["domain"], got == g["intended_form"]))

agree = sum(r[4] for r in rows)
print(f"{len(rows)} labeled  ·  {agree} agree ({agree/max(len(rows),1):.0%})  ·  "
      f"{len(rows)-agree} for review\n")

# Which intended forms does the generator fail to produce reliably?
per = collections.defaultdict(lambda: [0, 0])
for _, want, _, _, ok in rows:
    per[want][1] += 1
    per[want][0] += ok
print("agreement by intended form (low = generator or labeler is unreliable here):")
for form, (ok, n) in sorted(per.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
    bar = "#" * round(ok / max(n, 1) * 20)
    print(f"  {form:<26} {ok:>3}/{n:<3} {bar}")

# The specific confusions, most common first — these say what to fix.
conf = collections.Counter((w, g) for _, w, g, _, ok in rows if not ok)
print("\nmost common confusions (intended -> labeled):")
for (w, g), n in conf.most_common(15):
    print(f"  {n:>3}  {w}  ->  {g}")

with open("data/interim/triage_queue.jsonl", "w") as fh:
    for rid, want, got, dom, ok in rows:
        if not ok:
            fh.write(json.dumps({
                "id": rid, "intended_form": want, "labeled_form": got,
                "domain": dom, "text": gen[rid]["text"], "label": lab[rid]["label"],
            }, ensure_ascii=False) + "\n")
print(f"\n{len(rows)-agree} disagreements -> data/interim/triage_queue.jsonl")
