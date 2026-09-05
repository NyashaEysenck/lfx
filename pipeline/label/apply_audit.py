"""Apply the adjudicated results of the equivocation / begging-the-question audit.

Every decision here was made by a human reading the passage, using the blind
labeler's disagreement only to decide WHAT to read. Gemini's answer is adopted
where a human agreed with it, never automatically -- it was wrong on the two
cleanest equivocations in the corpus ("only man is rational, no women is man" and
the rare-steak pun), which is why the queue needed a person at all.

Three groups:
  DROP      passages that are not arguments -- two are glossary fragments cut off
            before the phrase that would have caught them in import_logic.py
  RELABEL   the human label was wrong; adopt the form a human confirmed
  KEEP      everything else, including the 5 where the blind labeler was wrong

Writes a new corpus rather than editing in place, so the change is reviewable and
the previous file stays as it was when v4 and the 3B run were measured.
"""

import argparse
import collections
import json

from lfx.schema import INDUCTIVE_FORMS, inconsistency, schema_ok

# Not arguments. Nothing to extract, so no label could be right.
DROP = {
    "logic_0807": "truncated glossary fragment: '...is an example of'",
    "logic_0576": "a report of what someone said; no argument",
    "logic_0138": "'If you are open to it, love will find you' -- no argument",
    "logic_0007": "'Science shows us that...' -- an assertion, no argument",
    "logic_0465": "amphiboly in a single sentence; no conclusion drawn",
    "logic_0827": "truncated glossary fragment: '...I am using'",
}

# The human label was wrong. Value is the form a human confirmed after reading it.
RELABEL = {
    # blind labeler right, human label wrong
    "logic_0242": "hasty generalization",   # 'paranormal is real because I saw a ghost'
    "logic_0372": "causal",                 # got sick, so it was the students
    "logic_0524": "causal",                 # pirates vs global temperature
    "logic_0109": "ad hominem",             # 'you say that because you are a Republican'
    "logic_0420": "straw man",              # 'you are declaring war on Mother Nature'
    "logic_0537": "straw man",              # BLM -> looting
    "logic_0848": "authority",              # 'Historians agree...'
    "logic_0128": "authority",              # 'a program I saw on TV once'
    "logic_0599": "begging the question",   # first life came from a living being: God
    "logic_0549": "other",                  # lost book proving leprechauns
    "logic_0349": "other",                  # therapist vs placebo; a continuum fallacy
    # adjudicated one by one from the contested queue
    "logic_0695": "authority",              # 'because it says so in the Bible'
    "logic_0325": "authority",              # Freud on repression
    "logic_0874": "categorical syllogism",  # the abortion syllogism; see below
    "logic_0096": "false dilemma",          # the dirty-bomb passage; see below
}

# logic_0874: structurally a categorical syllogism. The equivocation charge rests
# on 'human being' meaning a biological organism in one premise and a moral person
# in the other -- a contested reading of the content, not a property of the form,
# and `form` describes structure.
#
# logic_0096: the rhetorical question frames the choice as tolerate rights
# violations OR accept a dirty bomb, treating those as exhaustive when
# proportional restrictions, warrants and targeted measures sit in the gap. The
# original label had to invent a premise absent from the text and convert the
# question into a conclusion; read as a false dilemma it needs neither.
REWRITE = {
    "logic_0096": {
        "premises": [
            "Either the rights of some psycho-violent street thug are possibly "
            "violated, or a dirty bomb goes off on Wall Street and much of this "
            "part of Brooklyn is destroyed.",
            "The destruction of much of this part of Brooklyn is not acceptable.",
        ],
        "conclusion": "Possible violations of the rights of some psycho-violent "
                      "street thug are worth it.",
        "argument_type": "deductive",
        "form": "false dilemma",
        "suppressed_premise": "There is no third option between violating those "
                              "rights and accepting the risk of a dirty bomb.",
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/interim/corpus_v12.jsonl")
ap.add_argument("--audit", default="data/interim/audit_eq_btq.jsonl")
ap.add_argument("--out", default="data/interim/corpus_v13.jsonl")
args = ap.parse_args()

audit = {json.loads(l)["id"]: json.loads(l) for l in open(args.audit)}
rows = [json.loads(l) for l in open(args.corpus)]

# 15 ids in the corpus name TWO different passages -- an id collision from the
# import, not a repeated record. It matters here because every decision below is
# keyed by id, and it matters generally because evaluate.py keys on id too and so
# silently drops one of each pair. Targets are therefore matched on id AND text,
# and a decision that cannot be pinned to exactly one record is refused rather
# than applied to both.
by_id = collections.defaultdict(list)
for r in rows:
    by_id[r["id"]].append(r)


def decided(r):
    """True if r is the passage a decision was written about, not its id-twin."""
    twins = by_id[r["id"]]
    if len(twins) == 1:
        return True
    return r["text"] == audit[r["id"]]["text"] if r["id"] in audit else False


ambiguous = [rid for rid in set(DROP) | set(RELABEL) | set(REWRITE)
             if len(by_id.get(rid, [])) > 1 and rid not in audit]
if ambiguous:
    raise SystemExit(f"cannot disambiguate {ambiguous} -- refusing")

kept, dropped, relabeled = [], [], []
for r in rows:
    rid = r["id"]
    if rid in DROP and decided(r):
        dropped.append((rid, DROP[rid]))
        continue
    if rid in REWRITE and decided(r):
        r = dict(r)
        r["label"] = REWRITE[rid]
        relabeled.append((rid, "-> " + r["label"]["form"]))
    elif rid in RELABEL and decided(r):
        want = RELABEL[rid]
        gem = audit.get(rid, {}).get("gemini_label")
        old = r["label"]["form"]
        r = dict(r)
        # Adopt the blind labeler's full record only when it reached the same form
        # a human confirmed; otherwise keep the existing fields and move the form.
        if gem and gem.get("form") == want:
            r["label"] = gem
        else:
            r["label"] = dict(r["label"], form=want)
        relabeled.append((rid, f"{old} -> {want}"))
    kept.append(r)

# Five records in the corpus contradict themselves: an inductive form carrying
# argument_type "deductive". They pre-date this audit -- four are in train.jsonl,
# so every model from v2 through the 3B trained on them. The schema is explicit
# that form implies argument_type, so the form is authoritative and the type is
# the field to correct. Surfaced by the guard below, not sought out.
repaired = []
for r in kept:
    why = inconsistency(r["label"])
    if not why:
        continue
    want = "inductive" if r["label"]["form"] in INDUCTIVE_FORMS else "deductive"
    r["label"] = dict(r["label"], argument_type=want)
    repaired.append((r["id"], r["label"]["form"], want))

bad = [(r["id"], inconsistency(r["label"])) for r in kept
       if not schema_ok(r["label"]) or inconsistency(r["label"])]
if bad:
    for rid, why in bad:
        print(f"  INVALID {rid}: {why or 'fails schema_ok'}")
    raise SystemExit(f"{len(bad)} records would be written invalid -- refusing")

# Give the colliding ids distinct names on the way out, so downstream code that
# keys on id (evaluate.py, the splitter) stops losing one record of each pair.
seen, renamed = set(), []
for r in kept:
    if r["id"] in seen:
        base = r["id"]
        n = 2
        while f"{base}_{n}" in seen:
            n += 1
        r["id"] = f"{base}_{n}"
        renamed.append((base, r["id"]))
    seen.add(r["id"])

with open(args.out, "w") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"in  {len(rows)}\nout {len(kept)}   dropped {len(dropped)}   "
      f"relabeled {len(relabeled)}\n")
print("dropped as not-an-argument:")
for rid, why in dropped:
    print(f"  {rid}  {why}")
print("\nrelabeled:")
for rid, what in relabeled:
    print(f"  {rid}  {what}")
if repaired:
    print("\nargument_type repaired to match form (pre-existing contradictions):")
    for rid, form, want in repaired:
        print(f"  {rid}  {form} -> {want}")
if renamed:
    print(f"\nid collisions given distinct names ({len(renamed)}):")
    for base, now in renamed:
        print(f"  {base} -> {now}")
print(f"\nwrote {args.out}")
