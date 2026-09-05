"""Migrate labels from schema v1.2 (form: string) to v2.0 (form: list).

Every existing record becomes a one-element list, which is exactly what it always
meant: one form applies. Nothing is reinterpreted here -- the records that need a
real second form are the `other` chains, and those are re-labelled separately by
the labeller, which can now return more than one.

Set equality against a one-element list is the same comparison v1.2's string
equality made, so every number measured before this migration stays comparable
with every number measured after it.

    python pipeline/build/migrate_v2.py --apply
"""

import argparse
import glob
import json
import pathlib

from lfx.schema import forms_of, inconsistency, schema_ok

ap = argparse.ArgumentParser()
ap.add_argument("--paths", nargs="*", default=[
    "data/splits/*.jsonl", "data/interim/corpus_v13.jsonl", "data/raw/fewshot.jsonl",
])
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

files = [pathlib.Path(p) for pat in args.paths for p in sorted(glob.glob(pat))]
total = 0
for path in files:
    if path.name.endswith(".chat.jsonl"):
        continue  # rebuilt from the plain split by format_for_training
    rows = [json.loads(l) for l in open(path)]
    changed = 0
    for r in rows:
        lab = r.get("label")
        if not isinstance(lab, dict) or isinstance(lab.get("form"), list):
            continue
        lab["form"] = forms_of(lab)
        changed += 1

    bad = [r.get("id") for r in rows
           if isinstance(r.get("label"), dict)
           and (not schema_ok(r["label"]) or inconsistency(r["label"]))]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} records invalid after migration, "
                         f"first few {bad[:5]}")

    total += changed
    print(f"{str(path):44} {len(rows):>5} records, {changed:>5} migrated")
    if args.apply and changed:
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n{'applied' if args.apply else 'DRY RUN'}: {total} labels migrated")
if not args.apply:
    print("re-run with --apply to write")
