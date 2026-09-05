"""Blind-relabel named classes and report where the human label is not supported.

Phase 5 left `begging the question` leaking to `equivocation` on 12 of 80 held-out
records. Reading the equivocation TRAINING records explained why: only about half
of the 44 are equivocation at all. The rest are straw man, appeal to authority, a
sorites, and two passages that are not arguments. A class that small and that
noisy has no usable boundary, so it acts as an attractor for anything involving
restatement or word-play -- which is exactly what circular arguments look like.

This is the technique that caught the `faulty generalization` mapping error: the
labeler never sees the human label, so agreement is independent evidence and
disagreement is a queue worth a human's time. It does NOT decide anything on its
own; it ranks what to look at.

Deliberately operates on the corpus, not on a split, so a correction flows through
merge and split like any other label. Held-out records are never edited here.

    python pipeline/label/audit_classes.py --classes "equivocation,begging the question"
"""

import argparse
import collections
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from lfx.vertex import build_examples, client, label_one

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/interim/corpus_v12.jsonl")
ap.add_argument("--fewshot", default="data/raw/fewshot.jsonl")
ap.add_argument("--classes", help="comma-separated form names; default all")
ap.add_argument("--sample", type=int,
                help="audit a random sample of this many, for measuring a set's "
                     "label noise rather than correcting it")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/interim/audit_classes.jsonl")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.corpus)]
if args.classes:
    targets = {c.strip() for c in args.classes.split(",") if c.strip()}
    todo = [r for r in rows if r["label"]["form"] in targets]
    scope = ", ".join(sorted(targets))
else:
    todo, scope = list(rows), "all classes"
if args.sample and args.sample < len(todo):
    # A seeded sample so the estimate is reproducible and cannot be quietly
    # re-drawn until it says something convenient.
    todo = random.Random(args.seed).sample(todo, args.sample)
    scope += f" (random sample of {args.sample}, seed {args.seed})"
if args.limit:
    todo = todo[: args.limit]

print(f"{len(rows)} records in {args.corpus}; auditing {len(todo)}: {scope}\n")

cl = client()
examples = build_examples(args.fewshot)
out = open(args.out, "w")
lock, done = threading.Lock(), [0]


def one(r):
    try:
        got = label_one(cl, examples, r["text"])
    except Exception as exc:  # noqa: BLE001 - one bad record must not stop the run
        print(f"  ! {r['id']}: {type(exc).__name__}"[:110], file=sys.stderr)
        return
    rec = dict(r)
    rec["gemini_form"] = got.get("form")
    rec["gemini_label"] = got
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{len(todo)}")


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(pool.map(one, todo))
out.close()

labeled = [json.loads(l) for l in open(args.out)]
print(f"\nlabeled {len(labeled)}/{len(todo)}\n")

per = collections.defaultdict(lambda: [0, 0])
confusion = collections.defaultdict(collections.Counter)
for r in labeled:
    human, gem = r["label"]["form"], r["gemini_form"]
    per[human][0] += 1
    per[human][1] += human == gem
    if human != gem:
        confusion[human][gem] += 1

print(f"{'class':26}{'n':>5}{'agree':>8}")
for form, (n, ok) in sorted(per.items()):
    print(f"{form:26}{n:>5}{ok / n:>8.0%}")

for form in sorted(confusion):
    print(f"\n{form} -> what the blind labeler called it instead:")
    for other, n in confusion[form].most_common():
        print(f"   {n:>3}  {other}")

print(f"\nreview queue written to {args.out} "
      f"({sum(len(c) for c in confusion.values())} distinct disagreements)")
