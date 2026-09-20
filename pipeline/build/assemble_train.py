"""Merge the constructed training items into the training split.

Kept strictly apart from the benchmark. The generators run at temperature 1.0, so
a collision is unlikely but not impossible, and "unlikely" is how this project
found three separate train/test leaks before. Every candidate is checked against
the benchmark text-by-text and dropped on a match.

Different seeds too -- the benchmark was built at seeds 7 and 11, this at 101 and
103 -- so the sampled domain, register and skeleton sequences differ as well.

    python pipeline/build/assemble_train.py --apply
"""

import argparse
import collections
import json
import sys

from lfx.schema import forms_of, schema_ok, inconsistency

NEW = ["data/interim/train_formal.jsonl", "data/interim/train_specified.jsonl"]


def key(t):
    return " ".join(t.lower().split())[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/splits/train.jsonl")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bench = {key(json.loads(l)["text"])
             for l in open("data/benchmark/lfx_bench_v1.jsonl")}
    held = set(bench)
    for p in ("data/splits/test_real.jsonl", "data/splits/extra_test_human.jsonl",
              "data/splits/test_gen.jsonl", "data/splits/val.jsonl"):
        for l in open(p):
            held.add(key(json.loads(l)["text"]))

    existing = [json.loads(l) for l in open("data/splits/train.jsonl")]
    seen = {key(r["text"]) for r in existing}
    out = list(existing)
    added, leaked, dup, invalid = 0, 0, 0, 0

    for path in NEW:
        for l in open(path):
            r = json.loads(l)
            k = key(r["text"])
            if k in held:
                leaked += 1
                continue
            if k in seen:
                dup += 1
                continue
            if not schema_ok(r["label"]) or inconsistency(r["label"]):
                invalid += 1
                continue
            seen.add(k)
            out.append({"id": f"tr_{r['id']}", "text": r["text"],
                        "origin": "constructed", "label": r["label"]})
            added += 1

    print(f"train: {len(existing)} -> {len(out)}  (+{added})")
    print(f"  dropped: {leaked} benchmark/held-out collisions, {dup} duplicates, "
          f"{invalid} schema-invalid")

    c = collections.Counter(" + ".join(forms_of(r["label"])) for r in out)
    o = collections.Counter(r.get("origin") for r in out)
    print(f"\norigin: {dict(o)}")
    print(f"\n{'class':34} {'n':>5}")
    for f, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {f:34} {n:5}")

    if args.apply:
        with open(args.out, "w") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\nDRY RUN — re-run with --apply to write")


if __name__ == "__main__":
    sys.exit(main())
