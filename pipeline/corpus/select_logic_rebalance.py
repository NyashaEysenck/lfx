"""Select unused LOGIC rows for the informal fallacies v6 lost ground on.

v6 added 988 constructed items for formal and inductive classes and nothing for the
informal fallacies, which shrank them from ~7% of training each to ~2-3%. It then
fell 0.765 -> 0.625 on external-human prose, almost entirely by sending 93 of 200
`ad hominem` records to `straw man` -- predicting straw man 152 times against 54
true. These are the only classes that cannot be constructed honestly, so the
rebalance has to come from human-labelled prose.

Only classes whose EXTENSION was checked against our definitions, via the KEEP map
in import_logic.py. `faulty generalization` stays out: LOGIC uses it for any bad
inductive leap, ours is the narrow one, and Gemini disagreed with the human label
on 21 of 33 sampled. A name match is not a class match.

Excluded: everything already in the corpus, the benchmark, or any held-out split,
checked text-by-text. The benchmark's external-human tier IS LOGIC prose, so this
is not a formality -- a slip here would train on the test set.

    python pipeline/corpus/select_logic_rebalance.py --apply
"""

import argparse
import collections
import csv
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from import_logic import KEEP, is_argument  # noqa: E402

# the informal fallacies only: the formal classes are now well covered by
# construction and do not need more from a noisier source
TARGET = {k: v for k, v in KEEP.items()
          if v in {"ad hominem", "straw man", "ad populum", "false dilemma",
                   "begging the question", "equivocation"}}


def key(t):
    return " ".join(t.lower().split())[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interim/logic_rebalance.jsonl")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    excluded = set()
    for p in ("data/interim/corpus_v20.jsonl", "data/benchmark/lfx_bench_v1.jsonl",
              "data/splits/train.jsonl", "data/splits/val.jsonl",
              "data/splits/test_gen.jsonl", "data/splits/test_real.jsonl",
              "data/splits/extra_test_human.jsonl"):
        for l in open(p):
            excluded.add(key(json.loads(l)["text"]))

    picked, why = [], collections.Counter()
    for path in sorted(glob.glob("data/logic/edu_*.csv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                lab = (row.get("updated_label") or row.get("label") or "").strip().lower()
                txt = (row.get("source_article") or "").strip()
                if lab not in TARGET or not txt:
                    continue
                k = key(txt)
                if k in excluded:
                    why["already used or held out"] += 1
                    continue
                if not is_argument(txt):
                    why["not an argument (definition/quiz text)"] += 1
                    continue
                if len(txt) < 40:
                    why["too short"] += 1
                    continue
                excluded.add(k)
                picked.append({"id": f"logic_rb_{len(picked):04d}", "text": txt,
                               "origin": "logic-human", "logic_class": lab,
                               "gold_form": TARGET[lab]})

    c = collections.Counter(r["gold_form"] for r in picked)
    print(f"{len(picked)} rows selected")
    for f, n in c.most_common():
        print(f"  {n:4}  {f}")
    print("\nskipped:", dict(why))
    if args.apply:
        with open(args.out, "w") as fh:
            for r in picked:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\nDRY RUN — re-run with --apply to write")


if __name__ == "__main__":
    sys.exit(main())
