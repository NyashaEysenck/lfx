"""Sanity-check the candidate labels before the human review pass."""
import collections
import json
import sys

DEDUCTIVE_FORMS = {"modus ponens", "modus tollens", "hypothetical syllogism",
                   "disjunctive syllogism", "categorical syllogism",
                   "reductio ad absurdum", "affirming the consequent",
                   "denying the antecedent"}
INDUCTIVE_FORMS = {"generalization", "analogy", "causal", "sign", "authority",
                   "hasty generalization"}

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1
                                   else "labeled_candidates.jsonl")]

forms = collections.Counter(r["label"]["form"] for r in rows)
types_ = collections.Counter(r["label"]["argument_type"] for r in rows)

print(f"{len(rows)} labeled\n")
print("argument_type:", dict(types_))
print("\nform distribution:")
for form, n in forms.most_common():
    print(f"  {n:3d}  {form}")

print("\nflags for human review:")
flagged = 0
for r in rows:
    lb, issues = r["label"], []
    if lb["argument_type"] == "deductive" and lb["form"] in INDUCTIVE_FORMS:
        issues.append(f"deductive but inductive form '{lb['form']}'")
    if lb["argument_type"] == "inductive" and lb["form"] in DEDUCTIVE_FORMS:
        issues.append(f"inductive but deductive form '{lb['form']}'")
    if len(lb["premises"]) < 2:
        issues.append(f"only {len(lb['premises'])} premise(s)")
    if not lb["conclusion"].strip():
        issues.append("empty conclusion")
    if lb["form"] == "other":
        issues.append("form='other' (schema gap?)")
    if issues:
        flagged += 1
        print(f"  {r['id']:<20} {'; '.join(issues)}")
print(f"\n{flagged}/{len(rows)} rows flagged")
