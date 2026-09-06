"""The v2.0 label schema: the form enum, its families, and validity checks.

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

# v2.0: `form` is a LIST, ordered most-central move first.
#
# A single value could not describe a multi-step argument. "Forty of the Swedes I
# met were Lutheran, so most Swedes are; therefore Sven probably is" performs a
# generalization AND an application of it, and whichever the annotator happened to
# write became the one correct answer while the other scored as an error -- so a
# model reading the passage correctly was marked wrong. Those records were pushed
# into `other`, which then meant two unrelated things: "several forms apply" and
# "no form applies". `equivocation` showed what a double-meaning label does to a
# class, so the two were separated:
#
#   form = ["generalization", "application of generalization"]   several apply
#   form = ["other"]                                             none applies
#
# Nearly every record has exactly one entry, so set equality against a one-element
# list is the same comparison v1.2 made -- the metric stays comparable across the
# change, and only chains can now be fully right.
MAX_FORMS = 3

# STATUS, decided 2026-09-06: the list is UNEXERCISED, deliberately, and that is
# not an oversight to be fixed later without thinking about it.
#
# No split holds a multi-form label, so the model gets no signal to produce one
# and no gold record can test one. verify.py used to warn about this on every
# run. An un-actioned warning that fires every time teaches you to skim warnings,
# which costs more than the thing it warns about, so it now states this decision
# instead.
#
# The reason it stayed empty is worth recording. The list was designed for
# SEQUENTIAL chains -- generalization, then application of it -- and the corpus
# turns out to hold almost none. What it holds instead, roughly 116 times, is
# OVERLAY: one argument that two competent labellers each described correctly
# with a different single form.
#
#   "Either determinism is true, or we possess genuine free will. Our experience
#    of agency suggests we are not cogs in a deterministic machine, so
#    determinism must be false. Therefore we have free will."
#
# Structurally a `disjunctive syllogism`; the disjunction is false because it
# excludes compatibilism, so also a `false dilemma`. Neither label is wrong and
# neither is a step the other follows. The same shape recurs as `begging the
# question` vs `categorical syllogism`, and `false dilemma` vs `modus tollens`.
#
# Those records are currently recorded as labeller DISAGREEMENTS and pushed into
# train as noise -- see split_corpus.py, which routes contested records there.
#
# So the honest next phase is not to manufacture chains. It is to redefine this
# field as "every form that describes this argument" and mine the disagreements,
# which are already human-adjudicated. That is deferred, not rejected: it resets
# every metric in the project and requires re-reviewing gold that currently holds
# one form per record. Do it as its own phase, against a frozen baseline, or not
# at all.


def forms_of(d):
    """The form list, tolerating a v1.2 record that still holds a bare string."""
    f = d.get("form")
    if f is None:
        return []
    return [f] if isinstance(f, str) else list(f)


def is_chain(d):
    """True if more than one form applies -- a multi-step argument."""
    return len(forms_of(d)) > 1


def inconsistency(d):
    """Describe an internal contradiction between form and argument_type, or None.

    The model violates this in roughly 0-2% of outputs. It is not cosmetic: a
    contradictory pair reliably means the model is unsure of the form.

    Checked on SINGLE-form labels only. The rule was first written to apply to
    every entry of a chain, which promptly rejected a real argument: night shifts
    -> poor sleep -> clinical error is causal and inductive, while "a hospital must
    not adopt such a policy, so withdraw the rota" is a deductive modus tollens.
    Real chains mix families, and forbidding that would only move the single-value
    bottleneck from `form` to `argument_type`. On a chain, argument_type describes
    how the argument as a whole presents its conclusion -- as necessary, or as
    probable -- not how each step works.
    """
    t, forms = d.get("argument_type"), forms_of(d)
    if len(forms) != 1:
        return None
    f = forms[0]
    if t == "deductive" and f in INDUCTIVE_FORMS:
        return f"'{f}' is an inductive form but argument_type says deductive"
    if t == "inductive" and f in DEDUCTIVE_FORMS:
        return f"'{f}' is a deductive form but argument_type says inductive"
    return None


def schema_ok(p):
    """True if p is a well-formed label. Does not check factual correctness."""
    if not isinstance(p, dict) or set(p) != REQUIRED_KEYS:
        return False
    forms = p.get("form")
    forms_valid = (isinstance(forms, list) and 1 <= len(forms) <= MAX_FORMS
                   and all(f in FORMS for f in forms)
                   and len(set(forms)) == len(forms)
                   # `other` means no form applies, so it cannot sit beside one
                   and not ("other" in forms and len(forms) > 1))
    return (isinstance(p.get("premises"), list)
            and all(isinstance(x, str) for x in p["premises"])
            and isinstance(p.get("conclusion"), str)
            and p.get("argument_type") in {"deductive", "inductive"}
            and forms_valid
            and (p.get("suppressed_premise") is None
                 or isinstance(p["suppressed_premise"], str)))
