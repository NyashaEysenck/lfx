"""Score predictions against gold labels, field by field.

Deliberately not a single accuracy number. The task has four distinguishable
failure modes and they call for different fixes:
  - the model emits something that is not valid JSON / not the schema  -> format
  - it splits premises differently than the gold label                  -> extraction
  - it picks the wrong `form`                                           -> classification
  - it invents or omits a suppressed premise                            -> hallucination

Runs locally on CPU. Generation happens elsewhere (predict.py, on a GPU).

Usage:
    python evaluate.py test_real.jsonl preds_base_real.jsonl [preds_tuned_real.jsonl]
"""

import collections
import json
import sys
from difflib import SequenceMatcher

from lfx.schema import forms_of, is_chain, schema_ok

MATCH = 0.6          # SequenceMatcher ratio above which two premises are "the same"


def norm(s):
    return " ".join(str(s).lower().split())


def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def premise_prf(gold, pred):
    """Greedy best-match F1 over premise lists. Order-independent: the model is
    not required to reproduce the gold ordering, only the same set of claims."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    unused, hits = list(pred), 0
    for g in gold:
        best, bi = 0.0, None
        for i, p in enumerate(unused):
            r = sim(g, p)
            if r > best:
                best, bi = r, i
        if best >= MATCH:
            hits += 1
            unused.pop(bi)
    prec = hits / len(pred)
    rec = hits / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def score(gold_path, pred_path):
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(gold_path)}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(pred_path)}

    m = collections.Counter()
    sums = collections.defaultdict(float)
    per_form = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    n = 0

    for rid, g in gold.items():
        n += 1
        gl = g["label"]
        pr = preds.get(rid)
        p = pr.get("parsed") if pr else None

        if p is None:
            m["json_invalid"] += 1
            per_form[" + ".join(forms_of(gl))][1] += 1
            m[("chain" if is_chain(gl) else "single") + "_n"] += 1
            continue
        m["json_valid"] += 1
        # Predictions made under v1.2 hold `form` as a bare string. Coerce them to
        # the v2.0 shape for scoring so historical runs stay comparable; a model
        # trained under v1.2 is judged by whether it produced a valid v1.2 label,
        # which is what it was asked for. Nothing else about the record changes.
        if isinstance(p.get("form"), str):
            p = dict(p, form=[p["form"]])
            m["legacy_coerced"] += 1
        if not schema_ok(p):
            m["schema_invalid"] += 1
            per_form[" + ".join(forms_of(gl))][1] += 1
            m[("chain" if is_chain(gl) else "single") + "_n"] += 1
            continue
        m["schema_valid"] += 1

        m["type_correct"] += p["argument_type"] == gl["argument_type"]
        # v2.0 scores the form SET. Nearly every record holds one form, where set
        # equality is the same comparison v1.2 made -- so these numbers stay
        # comparable across the schema change. Only a chain can now be fully
        # right, where before whichever form the annotator wrote down was the one
        # correct answer and the model was marked wrong for naming the other.
        gforms, pforms = forms_of(gl), forms_of(p)
        okf = set(gforms) == set(pforms)
        m["form_correct"] += okf
        # Chains are reported separately: they are the records a single-value
        # field could not express, so they are where this change should show.
        bucket = "chain" if is_chain(gl) else "single"
        m[f"{bucket}_n"] += 1
        m[f"{bucket}_correct"] += okf
        # Partial credit, reported but never used as the headline: on a chain,
        # naming one of two forms is more useful than naming neither.
        if gforms:
            sums["form_jaccard"] += (len(set(gforms) & set(pforms))
                                     / len(set(gforms) | set(pforms)))
        key = " + ".join(gforms)
        per_form[key][0] += okf
        per_form[key][1] += 1
        if not okf:
            confusion[(key, " + ".join(pforms))] += 1

        _, _, f1 = premise_prf(gl["premises"], p["premises"])
        sums["premise_f1"] += f1
        sums["conclusion_sim"] += sim(gl["conclusion"], p["conclusion"])

        g_has, p_has = gl["suppressed_premise"] is not None, p["suppressed_premise"] is not None
        m["supp_presence_correct"] += g_has == p_has
        if g_has and p_has:
            sums["supp_sim"] += sim(gl["suppressed_premise"], p["suppressed_premise"])
            m["supp_both"] += 1

    ok = max(m["schema_valid"], 1)
    if m["legacy_coerced"]:
        print(f"  note: {pred_path} holds {m['legacy_coerced']} v1.2-shaped labels "
              f"(form as a string); coerced to a one-element list for scoring")
    return {
        "n": n,
        "json_valid": m["json_valid"] / n,
        "schema_valid": m["schema_valid"] / n,
        # End-to-end: counts a schema-invalid output as wrong, because downstream
        # it IS wrong. The conditional numbers below are diagnostic only — they are
        # computed over schema-valid rows, so a model that fails schema on its hard
        # cases scores better on them by shrinking its own denominator.
        "form_acc_e2e": m["form_correct"] / n,
        "form_acc": m["form_correct"] / ok,
        "form_jaccard": sums["form_jaccard"] / ok,
        "single_acc": m["single_correct"] / max(m["single_n"], 1),
        "chain_acc": m["chain_correct"] / max(m["chain_n"], 1),
        "chain_n": m["chain_n"],
        "type_acc": m["type_correct"] / ok,
        "premise_f1": sums["premise_f1"] / ok,
        "conclusion_sim": sums["conclusion_sim"] / ok,
        "supp_presence_acc": m["supp_presence_correct"] / ok,
        "supp_sim": sums["supp_sim"] / max(m["supp_both"], 1),
    }, per_form, confusion


def main():
    gold_path, pred_paths = sys.argv[1], sys.argv[2:]
    results = [(p, *score(gold_path, p)) for p in pred_paths]

    keys = ["json_valid", "schema_valid", "form_acc_e2e", "form_acc",
            "single_acc", "chain_acc", "form_jaccard", "type_acc",
            "premise_f1", "conclusion_sim", "supp_presence_acc", "supp_sim"]
    labels = {"json_valid": "JSON parses", "schema_valid": "schema valid",
              "form_acc_e2e": "FORM ACC (end-to-end)",
              "form_acc": "form acc | schema ok",
              "single_acc": "  on single-form records",
              "chain_acc": "  on multi-form chains",
              "form_jaccard": "form overlap (partial credit)",
              "type_acc": "argument_type accuracy",
              "premise_f1": "premise F1", "conclusion_sim": "conclusion similarity",
              "supp_presence_acc": "suppressed present/absent", "supp_sim": "suppressed similarity"}

    names = [p.replace(".jsonl", "").replace("preds_", "") for p, _, _, _ in results]
    print(f"gold: {gold_path}  ({results[0][1]['n']} examples)\n")
    print(f"{'metric':<28}" + "".join(f"{nm:>16}" for nm in names)
          + ("      delta" if len(results) == 2 else ""))
    for k in keys:
        row = f"{labels[k]:<28}" + "".join(f"{r[1][k]:>16.3f}" for r in results)
        if len(results) == 2:
            d = results[1][1][k] - results[0][1][k]
            row += f"  {d:>+9.3f}"
        print(row)

    _, _, per_form, confusion = results[-1]
    print(f"\nper-form accuracy ({names[-1]}):")
    for form, (ok_, tot) in sorted(per_form.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        if tot:
            print(f"  {form:<26} {ok_:>3}/{tot:<3} {'#' * round(ok_ / tot * 20)}")
    if confusion:
        print(f"\ntop confusions ({names[-1]}, gold -> predicted):")
        for (g, p), c in confusion.most_common(10):
            print(f"  {c:>3}  {g}  ->  {p}")


if __name__ == "__main__":
    main()
