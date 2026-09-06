"""End-to-end consistency check. Run it before trusting any number.

Most of this project's worst bugs were consistency failures rather than logic
errors: a prompt that drifted from the schema, predictions scored against a
reshuffled split, one id naming two passages, a fix applied to a generated file
instead of its source. None raised an exception; each quietly changed a number.

Every check below corresponds to something that actually went wrong.

    python pipeline/verify.py
"""

import collections
import glob
import json
import pathlib
import re
import sys

from lfx.prompts import SYSTEM, for_model
from lfx.schema import FORMS, MAX_FORMS, forms_of, inconsistency, schema_ok

FAILS, WARNS = [], []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def warn(name, detail):
    print(f"  [WARN] {name} — {detail}")
    WARNS.append(name)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def key(text):
    return " ".join(text.lower().split())[:150]


SPLITS = ["train", "val", "test_gen", "test_real", "extra_test_human"]

print("\n1. SCHEMA — every label valid under v2.0")
data = {}
for name in SPLITS:
    rows = load(f"data/splits/{name}.jsonl")
    data[name] = rows
    bad = [r["id"] for r in rows if not schema_ok(r["label"])]
    inc = [r["id"] for r in rows if inconsistency(r["label"])]
    check(f"{name}: {len(rows)} records, labels valid", not bad and not inc,
          f"invalid={len(bad)} inconsistent={len(inc)}" if bad or inc else "")
corpus = load("data/interim/corpus_v20.jsonl")
check(f"corpus_v20: {len(corpus)} records, labels valid",
      not [r for r in corpus if not schema_ok(r["label"]) or inconsistency(r["label"])])
check("every form used is in the enum",
      all(f in FORMS for r in corpus + sum(data.values(), []) for f in forms_of(r["label"])))

print("\n2. IDS — one id names one passage (a duplicate silently drops a record)")
for name, rows in data.items():
    dup = [k for k, v in collections.Counter(r["id"] for r in rows).items() if v > 1]
    check(f"{name}: ids unique", not dup, f"{len(dup)} duplicated" if dup else "")
seen = {}
collide = 0
for name, rows in data.items():
    for r in rows:
        if r["id"] in seen and seen[r["id"]] != key(r["text"]):
            collide += 1
        seen[r["id"]] = key(r["text"])
check("no id names different passages across splits", collide == 0,
      f"{collide} collisions" if collide else "")

print("\n3. LEAKAGE — no passage on both sides of the train/test line")
train_txt = {key(r["text"]) for r in data["train"]} | {key(r["text"]) for r in data["val"]}
for name in ("test_gen", "test_real", "extra_test_human"):
    over = train_txt & {key(r["text"]) for r in data[name]}
    check(f"train/val vs {name}: no shared text", not over,
          f"{len(over)} shared" if over else "")
over = {key(r["text"]) for r in data["train"]} & {key(r["text"]) for r in data["val"]}
check("train vs val: no shared text", not over, f"{len(over)} shared" if over else "")

print("\n4. PROMPT — one contract, and every model carries its own")
listed = re.search(r"each one of: (.*?)\), and suppressed", SYSTEM, re.S)
in_prompt = [x.strip() for x in listed.group(1).split(",")] if listed else []
check("SYSTEM lists exactly the schema's forms", sorted(in_prompt) == sorted(FORMS),
      f"prompt={len(in_prompt)} schema={len(FORMS)}")
check(f"SYSTEM states the list bound ({MAX_FORMS})", f"1-{MAX_FORMS}" in SYSTEM)
for d in sorted(glob.glob("models/mlx_*")):
    p = pathlib.Path(d) / "prompt.txt"
    if not p.is_file():
        warn(f"{d}: no prompt.txt", "inference falls back to the current SYSTEM")
    else:
        same = p.read_text() == SYSTEM
        print(f"  [INFO] {d}: prompt.txt "
              f"{'matches current SYSTEM' if same else 'is an OLDER contract (expected for pre-v2.0 models)'}")

print("\n5. TRAINING FILES — chat targets agree with the labels they came from")
for name in ("train", "val"):
    chat = load(f"data/splits/{name}.chat.jsonl")
    check(f"{name}.chat.jsonl row count matches {name}.jsonl",
          len(chat) == len(data[name]), f"{len(chat)} vs {len(data[name])}")
    mism = 0
    for row, src in zip(chat, data[name]):
        try:
            got = json.loads(row["conversations"][-1]["content"])
        except Exception:
            mism += 1
            continue
        if forms_of(got) != forms_of(src["label"]):
            mism += 1
    check(f"{name}.chat.jsonl targets match labels", mism == 0, f"{mism} mismatched")
    sysmsg = chat[0]["conversations"][0]["content"]
    check(f"{name}.chat.jsonl carries the current SYSTEM", sysmsg == SYSTEM)

print("\n6. PREDICTIONS — each file scores against the split that produced it")
for path in sorted(glob.glob("results/preds_v5_*.jsonl")):
    split = pathlib.Path(path).stem.replace("preds_v5_", "")
    if split not in data:
        continue
    pids = {json.loads(l)["id"] for l in open(path)}
    gids = {r["id"] for r in data[split]}
    check(f"{pathlib.Path(path).name} ids align with {split}",
          pids == gids, f"{len(pids - gids)} orphan preds, {len(gids - pids)} missing")

print("\n7. CHAINS — the multi-form field is unexercised, by decision")
# This used to warn on every run. Nothing was ever going to action it, and a
# warning that always fires trains you to skim the ones that matter. The state
# is now asserted instead: if a chain label ever appears, that is a real change
# and the check fails so someone looks at it. See lfx/schema.py above MAX_FORMS
# for why the field stayed empty -- the corpus holds overlay, not chains.
tot = sum(1 for rows in data.values() for r in rows if len(forms_of(r["label"])) > 1)
check("no chain labels present, as expected", tot == 0,
      f"{tot} multi-form labels appeared — the schema note in lfx/schema.py is "
      f"now out of date; decide whether this is the overlay phase starting"
      if tot else "field reserved; see lfx/schema.py")

print("\n" + "=" * 62)
print(f"{len(FAILS)} failed, {len(WARNS)} warnings")
for f in FAILS:
    print(f"  FAIL: {f}")
for w in WARNS:
    print(f"  WARN: {w}")
sys.exit(1 if FAILS else 0)
