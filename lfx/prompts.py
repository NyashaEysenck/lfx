"""The task prompt — single source of truth.

This exact string is used at training time, at baseline-eval time, and at
inference time. If any of those three drift apart the before/after comparison
measures prompt wording rather than fine-tuning, and the packaged CLI silently
underperforms the model it shipped. Change it in one place or not at all.
"""

SYSTEM = (
    "Extract the logical structure of the argument into JSON with keys: "
    "premises (array of strings), conclusion (string), "
    "argument_type (\"deductive\" or \"inductive\"), "
    "form (one of: modus ponens, modus tollens, hypothetical syllogism, "
    "disjunctive syllogism, categorical syllogism, reductio ad absurdum, "
    "generalization, analogy, causal, sign, authority, "
    "affirming the consequent, denying the antecedent, ad hominem, "
    "hasty generalization, false dilemma, other), "
    "and suppressed_premise (string or null). "
    "Return only the JSON object."
)
