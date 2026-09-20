"""Assemble LFX-Bench: one evaluation set, every record carrying its provenance.

The problem this fixes is not that the old evaluation was small. It is that it was
LOPSIDED and did not say so. "0.765 on 485 held-out human arguments" covered five
informal fallacy classes; seventeen classes had three real-prose records or fewer,
three had none. One averaged number over that composition is not a benchmark, it
is a summary of whichever classes happened to be populated.

So provenance is a FIELD, not a footnote. Four kinds, and they are not equivalent:

  constructed-symbolic   label specified by construction, verified by lfx.formal
                         deriving the same form from a blind re-formalisation.
                         The strongest guarantee here: no human or model judged it.
  constructed-consensus  label specified by construction, verified by two models
                         from different families agreeing blind. Removes
                         idiosyncratic error, NOT systematic error.
  external-human         labelled by the LOGIC annotators, inherited with their
                         reported noise (~8-12% by this project's own re-labelling).
  project-authored       condensed from public-domain sources by this project and
                         reviewed here. The weakest provenance: our own judgement.

Anyone using this can filter to the tiers they trust. That is more useful than a
flat "gold" claim, and it is the honest description of what the labels are.

Report per class, never one average across tiers. Constructed prose is easier than
found prose -- 0.812 against 0.765 measured on this project's own splits -- so a
blended headline would flatter whichever model suits the constructed style.

    python pipeline/build/assemble_benchmark.py --apply
"""

import argparse
import collections
import json
import sys

from lfx.schema import FORMS, FAMILY, forms_of

SOURCES = [
    ("constructed-symbolic", "data/splits/tier1_formal.jsonl", "target_form"),
    ("constructed-consensus", "data/interim/inductive_tier1b_20260920_181518.jsonl",
     "target_form"),
    ("external-human", "data/splits/extra_test_human.jsonl", None),
    ("project-authored", "data/splits/test_real.jsonl", None),
]


def key(text):
    return " ".join(text.lower().split())[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark/lfx_bench_v1.jsonl")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # Anything the model was trained on cannot measure it. Checked, not assumed:
    # this project has already found three separate train/test leaks.
    trained = set()
    for p in ("data/splits/train.jsonl", "data/splits/val.jsonl",
              "data/splits/test_gen.jsonl"):
        for line in open(p):
            trained.add(key(json.loads(line)["text"]))

    out, seen, leaks, dups = [], set(), 0, 0
    for tier, path, form_field in SOURCES:
        for line in open(path):
            r = json.loads(line)
            k = key(r["text"])
            if k in trained:
                leaks += 1
                continue
            if k in seen:
                dups += 1
                continue
            seen.add(k)
            form = r[form_field] if form_field else None
            forms = [form] if form else forms_of(r["label"])
            rec = {"id": f"{tier.split('-')[0][:4]}_{r['id']}", "text": r["text"],
                   "label": r["label"], "form": forms, "provenance": tier}
            # difficulty, where construction measured it
            if "unaided" in r:
                rec["frontier_unaided_correct"] = r["unaided"]["correct"]
            if "verifiers" in r:
                rec["verifier_votes"] = r["verifiers"]
            if "skeleton" in r:
                rec["skeleton"] = r["skeleton"]
            out.append(rec)

    print(f"{len(out)} records  ({leaks} dropped as train/val/test_gen leaks, "
          f"{dups} as duplicates)\n")

    by_tier = collections.Counter(r["provenance"] for r in out)
    print("by provenance:")
    for t, n in by_tier.most_common():
        print(f"  {n:5}  {t}")

    print(f"\n{'class':34} {'family':18} {'n':>5}   provenance")
    missing = []
    for f in FORMS:
        sub = [r for r in out if r["form"] == [f]]
        if not sub:
            missing.append(f)
            print(f"  {f:34} {FAMILY[f][:17]:18} {0:5}   -- NO COVERAGE")
            continue
        mix = collections.Counter(r["provenance"] for r in sub)
        desc = ", ".join(f"{k.split('-')[0]}:{v}" for k, v in mix.most_common())
        print(f"  {f:34} {FAMILY[f][:17]:18} {len(sub):5}   {desc}")

    covered = len(FORMS) - len(missing)
    print(f"\n{covered}/{len(FORMS)} classes covered; missing: "
          f"{', '.join(missing) if missing else 'none'}")

    if args.apply:
        with open(args.out, "w") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\nDRY RUN — re-run with --apply to write")


if __name__ == "__main__":
    sys.exit(main())
