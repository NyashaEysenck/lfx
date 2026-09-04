"""The v1.2 label schema: the form enum, its families, and validity checks.

v1.2 aligns the enum with Duke's "Think Again" specialization, whose syllabi were
checked on 2026-09-04. Two corrections came out of that:

  - `sign` is gone, merged into `inference to the best explanation`. Every record
    it held was observed-indicator -> underlying-condition (spectral lines ->
    atmospheric composition, charcoal -> past wildfire, destroyed evidence ->
    guilt), which is IBE in Duke's framing. Sign is a finer slice than Duke cuts.
  - `application of generalization` and `inference to the best explanation` are
    added. Duke's Think Again III teaches generalization, application, IBE,
    analogy and causal — NOT the generalization/analogy/causal/sign/authority list
    the project brief originally attributed to it, which is a different textbook
    tradition.

`authority` stays. Duke files Appeals to Authority under Think Again IV's fallacies
of relevance, but the lesson teaches telling legitimate appeals from illegitimate
ones rather than rejecting the form, and this schema extracts rather than evaluates.

The four informal fallacies added are Duke Think Again IV topics with real
human-labelled data available in data/logic/.
"""

VALID_DEDUCTIVE = ["modus ponens", "modus tollens", "hypothetical syllogism",
                   "disjunctive syllogism", "categorical syllogism",
                   "reductio ad absurdum"]
INDUCTIVE = ["generalization", "application of generalization",
             "inference to the best explanation", "analogy", "causal", "authority"]
FORMAL_FALLACY = ["affirming the consequent", "denying the antecedent"]
INFORMAL_FALLACY = ["ad hominem", "hasty generalization", "false dilemma",
                    "ad populum", "begging the question", "straw man", "equivocation"]

FORMS = VALID_DEDUCTIVE + INDUCTIVE + FORMAL_FALLACY + INFORMAL_FALLACY + ["other"]

FAMILY = {
    **{f: "valid deductive" for f in VALID_DEDUCTIVE},
    **{f: "inductive" for f in INDUCTIVE},
    **{f: "FORMAL FALLACY" for f in FORMAL_FALLACY},
    **{f: "informal fallacy" for f in INFORMAL_FALLACY},
    "other": "multi-step / unclassified",
}

# `form` implies `argument_type`. A formal fallacy is still presented as
# necessary, so it counts as deductive; hasty generalization is a failed
# induction, so it counts as inductive.
DEDUCTIVE_FORMS = set(VALID_DEDUCTIVE) | set(FORMAL_FALLACY)
INDUCTIVE_FORMS = set(INDUCTIVE) | {"hasty generalization"}

REQUIRED_KEYS = {"premises", "conclusion", "argument_type", "form", "suppressed_premise"}


def inconsistency(d):
    """Describe an internal contradiction between form and argument_type, or None.

    The model violates this in roughly 0-2% of outputs. It is not cosmetic: a
    contradictory pair reliably means the model is unsure of the form.
    """
    t, f = d.get("argument_type"), d.get("form")
    if t == "deductive" and f in INDUCTIVE_FORMS:
        return f"'{f}' is an inductive form but argument_type says deductive"
    if t == "inductive" and f in DEDUCTIVE_FORMS:
        return f"'{f}' is a deductive form but argument_type says inductive"
    return None


def schema_ok(p):
    """True if p is a well-formed label. Does not check factual correctness."""
    if not isinstance(p, dict) or set(p) != REQUIRED_KEYS:
        return False
    return (isinstance(p.get("premises"), list)
            and all(isinstance(x, str) for x in p["premises"])
            and isinstance(p.get("conclusion"), str)
            and p.get("argument_type") in {"deductive", "inductive"}
            and p.get("form") in FORMS
            and (p.get("suppressed_premise") is None
                 or isinstance(p["suppressed_premise"], str)))
