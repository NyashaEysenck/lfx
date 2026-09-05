"""The task prompt — single source of truth.

This exact string is used at training time, at baseline-eval time, and at
inference time. If any of those three drift apart the before/after comparison
measures prompt wording rather than fine-tuning, and the packaged CLI silently
underperforms the model it shipped. Change it in one place or not at all.

That is exactly what went wrong before v2.0: the enum was written out by hand
here and then schema v1.2 changed it. Every model from v1.2 onward was trained,
evaluated and shipped with a prompt that still offered the deleted `sign` and
never mentioned `straw man`, `ad populum`, `begging the question`, `equivocation`,
`application of generalization` or `inference to the best explanation` -- four of
which are classes in the main test set. The models learned them from the training
targets while being told a label space that did not contain them.

The list is therefore BUILT FROM `FORMS` now rather than restated. A form added to
the schema appears here automatically, and the two cannot disagree again.
"""

from lfx.schema import FORMS, MAX_FORMS

SYSTEM = (
    "Extract the logical structure of the argument into JSON with keys: "
    "premises (array of strings), conclusion (string), "
    'argument_type ("deductive" or "inductive"), '
    "form (array of 1"
    f"-{MAX_FORMS} strings, each one of: {', '.join(FORMS)}), "
    "and suppressed_premise (string or null). "
    "List one form for a single-step argument. List every form a multi-step "
    "argument performs, most central first. Use [\"other\"] alone, and only when "
    "no other form applies. "
    "Return only the JSON object."
)
