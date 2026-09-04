"""Gemini labeling via Vertex AI.

NOTE the credentials story, because it is not obvious: the GEMINI_API_KEY in .env
is a Cloud-console key restricted to generativelanguage.googleapis.com, which
bills against a prepaid balance and returns 429 "prepayment credits depleted".
The aiplatform (Vertex) surface reaches the same models, bills the ordinary Cloud
way, and authenticates off Application Default Credentials. So there is no API key
here at all — run `gcloud auth application-default login` once.
"""

import json
import os
import sys
import time

from google.genai import types

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "agentic-school-506719")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = "gemini-2.5-flash"

from lfx.schema import FORMS  # noqa: E402

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "premises": {"type": "ARRAY", "items": {"type": "STRING"}},
        "conclusion": {"type": "STRING"},
        "argument_type": {"type": "STRING", "enum": ["deductive", "inductive"]},
        "form": {"type": "STRING", "enum": FORMS},
        "suppressed_premise": {"type": "STRING", "nullable": True},
    },
    "required": ["premises", "conclusion", "argument_type", "form", "suppressed_premise"],
    "property_ordering": ["premises", "conclusion", "argument_type", "form",
                          "suppressed_premise"],
}

LABELER_INSTRUCTION = """\
You extract the logical structure of an argument into a fixed JSON schema.

This is a PARSING task, not an evaluation task. Do not judge whether the argument
is good, valid, sound, or persuasive. Report only its structure as given.

Field rules:
- premises: each distinct stated premise, one per array element. Restate in clean,
  self-contained declarative form; do not quote rhetorical padding, and do not
  invent premises the passage does not state.
- conclusion: the single main claim the premises are offered in support of. If the
  passage states it more than once, use the clearest statement.
- argument_type: "deductive" if the conclusion is presented as following with
  necessity; "inductive" if presented as probable or well-supported. A formally
  invalid argument that is *presented* as necessary is still "deductive".
- form: the named pattern.
    * Valid deductive: modus ponens, modus tollens, hypothetical syllogism,
      disjunctive syllogism, categorical syllogism (quantified premises about
      classes, e.g. "All A are B; s is A; so s is B"), reductio ad absurdum
      (assume a claim, derive a contradiction or absurdity, reject the claim).
    * Inductive: generalization, analogy, causal, sign, authority.
    * Formal fallacies: affirming the consequent, denying the antecedent.
    * Informal fallacies: ad hominem, hasty generalization, false dilemma.
    * "other": use only when no listed form fits — typically a multi-step passage
      chaining several distinct inferences with no single dominant pattern. Prefer
      a specific form when one clearly dominates the argument's main inference.
  form must agree with argument_type: never pair a deductive form with
  argument_type "inductive", or an inductive form with "deductive".

  PRECEDENCE RULES for form — apply these in order, they override the lists above:
  1. FALLACY BEATS THE VALID FORM IT RESEMBLES. Do not read charitably. If the
     argument commits one of the named fallacies, name the fallacy, even when the
     inference pattern looks respectable.
     - A disjunction presented as exhaustive when other options plainly exist is
       "false dilemma", NOT "disjunctive syllogism" — the inference there is valid
       and the fallacy lives entirely in the first premise. Ask whether the two
       options are genuinely the only ones; if not, it is a false dilemma.
     - Check the DIRECTION of every conditional before assigning a form.
       "If P then Q; Q; therefore P" is "affirming the consequent", never
       "modus ponens", "causal", or "sign".
       "If P then Q; not P; therefore not Q" is "denying the antecedent", never
       "modus tollens".
     - Generalizing from one or two cases is "hasty generalization", not
       "generalization".
  2. "causal" IS NARROW. Use it only when the CONCLUSION ITSELF asserts that one
     thing causes another. It is not a catch-all for arguments that explain,
     motivate, or offer evidence. An argument that infers a cause but reaches it by
     affirming a consequent is "affirming the consequent", not "causal".
  3. "other" WINS OVER A PARTIAL MATCH. If the passage chains two or more DISTINCT
     inferences — eliminating alternatives and then inferring a best explanation, or
     a causal chain that terminates in a no-infinite-regress step, or a principle
     plus a subsumption plus a normative conclusion — label it "other" even when one
     of its steps, taken alone, resembles a named form. Only assign a named form when
     that single form accounts for the WHOLE argument.
- suppressed_premise: an unstated assumption the argument needs in order to work.
  Give exactly one, the load-bearing one. Use null if the argument is complete as
  stated. Do not pad this field.
"""


def client():
    from google import genai
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


def build_examples(path):
    """Hand-labeled few-shot turns, as alternating user/model content."""
    turns = []
    with open(path) as fh:
        for line in fh:
            ex = json.loads(line)
            turns.append(types.Content(role="user", parts=[types.Part(text=ex["text"])]))
            turns.append(types.Content(
                role="model",
                parts=[types.Part(text=json.dumps(ex["label"], ensure_ascii=False))]))
    return turns


def label_one(cl, examples, text, retries=3):
    """Label one passage. Retries with backoff; raises if all attempts fail."""
    contents = examples + [types.Content(role="user", parts=[types.Part(text=text)])]
    config = types.GenerateContentConfig(
        system_instruction=LABELER_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.0,
    )
    for attempt in range(retries):
        try:
            resp = cl.models.generate_content(model=MODEL, contents=contents, config=config)
            return json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001 - surface anything, retry, then give up
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries - 1} after "
                  f"{type(exc).__name__}: {exc}"[:200], file=sys.stderr)
            time.sleep(2 ** attempt)
