"""Build v7 training data: test the dilution hypothesis.

Keeps all v6 data (preserves the reductio fix), adds the 61 labeled LOGIC
rebalance rows (excluding any leaks into benchmark / held-out sets), and
upsamples the five never-constructed informal classes x2:
  - ad hominem
  - straw man
  - ad populum
  - false dilemma
  - begging the question

This restores their share of the training distribution toward ~25%.
"""

import argparse
import collections
import json
import random
import sys

from lfx.schema import forms_of, inconsistency, schema_ok

INFORMAL_5 = {
    "ad hominem",
    "straw man",
    "ad populum",
    "false dilemma",
    "begging the question",
}


def key(t):
    return " ".join(t.lower().split())[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6-train", default="data/splits/train_v6.jsonl")
    ap.add_argument("--new-logic", default="data/interim/logic_rebalance_labeled.jsonl")
    ap.add_argument("--out", default="data/splits/train.jsonl")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # 1. Collect all benchmark and held-out texts to guard against leaks
    held = set()
    for p in ("data/benchmark/lfx_bench_v1.jsonl", "data/splits/test_real.jsonl",
              "data/splits/extra_test_human.jsonl", "data/splits/test_gen.jsonl",
              "data/splits/val.jsonl"):
        for l in open(p):
            held.add(key(json.loads(l)["text"]))

    # 2. Load existing v6 train split
    v6 = [json.loads(l) for l in open(args.v6_train)]
    seen = {key(r["text"]) for r in v6}
    print(f"Loaded v6 train: {len(v6)} records")

    # 3. Add new labeled LOGIC rows
    added, leaked, dup, invalid = 0, 0, 0, 0
    base_pool = list(v6)

    for l in open(args.new_logic):
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
        rec = {
            "id": r["id"],
            "text": r["text"],
            "origin": "logic-human",
            "label": r["label"],
        }
        base_pool.append(rec)
        added += 1

    print(f"Added new LOGIC rows: +{added} (total base: {len(base_pool)}, dropped: {leaked} leak, {dup} dup, {invalid} invalid)")

    # 4. Upsample the 5 informal fallacies x2
    # Each record that contains at least one of INFORMAL_5 gets 1 duplicate copy added.
    upsampled_copies = []
    for r in base_pool:
        forms = set(forms_of(r["label"]))
        if forms & INFORMAL_5:
            dup_rec = dict(r)
            dup_rec["id"] = f"{r['id']}_dup"
            dup_rec["is_duplicate"] = True
            upsampled_copies.append(dup_rec)

    print(f"Upsampled duplicates created: +{len(upsampled_copies)}")

    v7_all = base_pool + upsampled_copies

    # Shuffle with fixed seed for uniform batch distribution during training
    rng = random.Random(args.seed)
    rng.shuffle(v7_all)

    # 5. Compute and print statistics
    c = collections.Counter()
    c_informal5 = 0
    for r in v7_all:
        forms = forms_of(r["label"])
        joined = " + ".join(forms)
        c[joined] += 1
        if set(forms) & INFORMAL_5:
            c_informal5 += 1

    o = collections.Counter(r.get("origin", "unknown") for r in v7_all)

    print(f"\nFinal v7 total records: {len(v7_all)}")
    print(f"Origins: {dict(o)}")
    share_pct = 100.0 * c_informal5 / len(v7_all)
    print(f"Informal-5 share: {c_informal5}/{len(v7_all)} ({share_pct:.1f}%) [target ~25%]")
    print(f"\n{'class':34} {'n':>5}")
    for f, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {f:34} {n:5}")

    if args.apply:
        with open(args.out, "w") as fh:
            for r in v7_all:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\nDRY RUN — re-run with --apply to write")


if __name__ == "__main__":
    sys.exit(main())
