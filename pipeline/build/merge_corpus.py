"""Combine every source into one corpus, capping per class and per origin.

Sources, and what each is good for:
  corpus_all.jsonl        Gemini prose + Gemini labels, broad coverage of all forms,
                          weak on fallacies, ~12% label noise floor.
  logic_labeled_v12.jsonl real passages, HUMAN form labels (LOGIC, Jin et al. 2022).
  formal_generated.jsonl  template-built, labels correct by construction, but
  disjunctive_generated   syntactically narrow.

TWO CAPS, for two different problems.

1. PER-CLASS cap. Without it the corpus tilts hard toward whatever was easiest to
   source: the LOGIC import alone offers 272 ad hominem against 21 analogy, and a
   model trained on that learns the prior, not the task. The v2 corpus was 41%
   fallacies; adding four more fallacy classes uncapped would push it past half.

2. PER-ORIGIN cap on TEMPLATE records. This is the composition fix. Five classes
   were 73-80% template, and the v2 run measured templates scoring 18/18 on
   templated text against 2/6 on natural prose — the model learned the template,
   not the form. Templates are therefore limited to a fraction of each class.
   Where that leaves a class short, the shortfall is REPORTED rather than filled
   with more templates: it is a work order for prose generation.

Surplus human-labelled records go to extra_test_human.jsonl. Real passages with
human labels are worth more as evaluation than as the Nth example of a class.
"""

import argparse
import collections
import json
import random

ap = argparse.ArgumentParser()
# Caps are ORIGIN-AWARE, and that is the point. A uniform cap of 45 was tried and
# regressed the two best-measured classes on real prose: ad hominem 0.76 -> 0.63 and
# false dilemma 0.82 -> 0.54, because it cut their training data from ~80 to 45.
# Those are the classes with the MOST abundant human-labelled real prose in the
# corpus, and balance is not worth paying for by discarding them. Balance earns its
# keep by stopping the model learning a prior instead of a task — it is not a
# virtue in itself.
ap.add_argument("--cap", type=int, default=45,
                help="max GENERATED records per form")
# 60 reproduces the v2 recipe that scored 0.76 / 0.82 on those classes: roughly
# 60 human plus ~20 generated. 90 was tried and pushed the fallacy share to 55%,
# since every human-labelled class here happens to be a fallacy.
ap.add_argument("--human-cap", type=int, default=60,
                help="max human-labelled records per form")
ap.add_argument("--template-cap", type=int, default=18,
                help="max template records per form — these do not transfer to prose")
ap.add_argument("--out", default="data/interim/corpus_v12.jsonl")
ap.add_argument("--surplus", default="data/splits/extra_test_human.jsonl")
ap.add_argument("--seed", type=int, default=20260904)
args = ap.parse_args()

rng = random.Random(args.seed)


def load(path, origin=None):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if origin:
            r["origin"] = origin
        rows.append(r)
    return rows


pool = (load("data/interim/corpus_all.jsonl")
        + load("data/interim/fill_labeled.jsonl", "gemini")
        + load("data/interim/logic_labeled_v12.jsonl", "logic-human")
        + load("data/interim/formal_generated.jsonl", "template")
        + load("data/interim/disjunctive_generated.jsonl", "template"))

# drop the pre-v1.2 LOGIC import, superseded by logic_labeled_v12
seen_text, dedup = set(), []
for r in pool:
    key = " ".join(r["text"].lower().split())[:180]
    if key in seen_text:
        continue
    seen_text.add(key)
    dedup.append(r)
print(f"{len(pool)} records in, {len(pool) - len(dedup)} duplicates dropped\n")

by_form = collections.defaultdict(list)
for r in dedup:
    by_form[r["label"]["form"]].append(r)

kept, surplus, short = [], [], []

for form, rows in by_form.items():
    human = [r for r in rows if r.get("origin") == "logic-human"]
    tmpl = [r for r in rows if r.get("origin") == "template"]
    gen = [r for r in rows if r.get("origin") not in ("logic-human", "template")]
    for lst in (human, tmpl, gen):
        rng.shuffle(lst)

    take_human = human[: args.human_cap]
    take_gen = gen[: args.cap]
    # templates only fill what real prose could not
    room = max(0, args.cap - len(take_gen))
    take_tmpl = tmpl[: min(room, args.template_cap)]

    chosen = take_human + take_gen + take_tmpl
    kept += chosen
    surplus += human[len(take_human):]          # spare human labels -> external test
    target = args.cap + (args.human_cap if human else 0)
    if len(chosen) < args.cap:
        short.append((form, len(chosen), args.cap - len(chosen),
                      len(take_tmpl) / max(len(chosen), 1)))

rng.shuffle(kept)
with open(args.out, "w") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
# Deduplicate: the surplus previously shipped 630 lines for 571 unique ids, and
# evaluate.py keys on id, so 59 records were silently dropped from every score.
seen_ids, uniq = set(), []
for r in surplus:
    if r["id"] in seen_ids:
        continue
    seen_ids.add(r["id"])
    uniq.append(r)
surplus = uniq
with open(args.surplus, "w") as fh:
    for r in surplus:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

counts = collections.Counter(r["label"]["form"] for r in kept)
origins = collections.Counter(r.get("origin", "?") for r in kept)
print(f"{len(kept)} records -> {args.out}")
print(f"  origins: {dict(origins)}")
print(f"  surplus (human-labelled, held out) -> {args.surplus}: {len(surplus)}\n")

w = max(len(f) for f in counts) + 2
print(f"{'form':<{w}}{'n':>5}{'gemini':>8}{'human':>7}{'tmpl':>6}{'tmpl%':>7}")
for form, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    c = lambda o: sum(1 for r in kept
                      if r["label"]["form"] == form and r.get("origin") == o)
    t = c("template")
    print(f"{form:<{w}}{n:>5}{c('gemini'):>8}{c('logic-human'):>7}{t:>6}{t / n:>6.0%}")

fall = {"ad hominem", "hasty generalization", "false dilemma", "ad populum",
        "begging the question", "straw man", "equivocation",
        "affirming the consequent", "denying the antecedent"}
nf = sum(v for k, v in counts.items() if k in fall)
print(f"\nfallacies: {nf}/{len(kept)} = {nf / len(kept):.0%}")

if short:
    print(f"\nWORK ORDER — classes below the cap of {args.cap}, i.e. what prose "
          f"generation still owes:")
    for form, n, need, share in sorted(short, key=lambda x: -x[2]):
        flag = "  TEMPLATE-HEAVY" if share > 0.5 else ""
        print(f"  {form:<34} have {n:>3}, need {need:>3} more{flag}")
