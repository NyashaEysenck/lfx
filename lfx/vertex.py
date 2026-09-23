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

from lfx.schema import FORMS, MAX_FORMS  # noqa: E402

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "premises": {"type": "ARRAY", "items": {"type": "STRING"}},
        "conclusion": {"type": "STRING"},
        "argument_type": {"type": "STRING", "enum": ["deductive", "inductive"]},
        # v2.0: a list, so a multi-step argument can name every move it makes
        # instead of being forced into one value or dumped into `other`.
        "form": {"type": "ARRAY", "items": {"type": "STRING", "enum": FORMS},
                 "minItems": 1, "maxItems": MAX_FORMS},
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
- form: an ARRAY of the named patterns the argument performs, most central
  first. One entry for a single-step argument; see rules 5-7 for multi-step ones.
    * Valid deductive: modus ponens, modus tollens, hypothetical syllogism,
      disjunctive syllogism, categorical syllogism (quantified premises about
      classes, e.g. "All A are B; s is A; so s is B"), reductio ad absurdum
      (assume a claim, derive a contradiction or absurdity, reject the claim).
    * Inductive:
      - generalization: from an observed SAMPLE to a broader population.
        "Every raven we sampled was black, so ravens are black."
      - application of generalization: the INVERSE — from a population-level
        generalization down to a particular case. "Most Swedes are Lutheran;
        Ingrid is Swedish; so Ingrid is probably Lutheran." Note the quantifier:
        "most"/"usually" makes this inductive and probable. If the premise says
        "all" and the conclusion is presented as certain, it is a CATEGORICAL
        SYLLOGISM instead, and deductive.
      - inference to the best explanation: an observation needs explaining, a
        hypothesis would explain it, no rival explains it as well, so the
        hypothesis is probably true. This covers reasoning from an INDICATOR to
        the underlying condition it evidences — symptoms to a diagnosis, charcoal
        in sediment to a past wildfire, destroyed evidence to guilt. The indicator
        does not CAUSE the condition; it is evidence of it.
      - analogy: two things share relevant features, one has a further property,
        so the other probably does too.
      - causal: the CONCLUSION ITSELF asserts that one thing causes another.
      - authority: a qualified, relevant authority asserts a claim, so it is
        probably true.
    * Formal fallacies: affirming the consequent, denying the antecedent.
    * Informal fallacies:
      - ad hominem: rejects a claim by attacking the person advancing it.
      - hasty generalization: generalizes from a sample far too small or biased.
      - false dilemma: presents two options as exhaustive when others plainly exist.
      - ad populum: infers that a claim is true because many people believe it, or
        because it is popular or traditional.
      - begging the question: the conclusion is assumed by one of the premises;
        the argument moves in a circle. "He is a wonderful writer because he
        writes so well."
      - straw man: misrepresents or exaggerates an opponent's position, then
        refutes the distorted version rather than what was actually claimed.
      - equivocation: a key word or phrase shifts MEANING between premises, so the
        argument only appears to work. The structure is often impeccable; the flaw
        is semantic. "We have a right to free speech, therefore it is right to
        speak falsely."
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
  3. QUANTIFIER DECIDES between categorical syllogism and application of
     generalization. "All A are B" with a conclusion presented as certain is a
     categorical syllogism (deductive). "Most A are B" or "A are usually B" with a
     hedged conclusion ("probably", "likely") is application of generalization
     (inductive). Read the quantifier and the hedge before choosing.
  4. EVIDENCE-TO-CONDITION IS IBE, NOT CAUSAL. If the conclusion says the evidence
     indicates some condition obtains, that is inference to the best explanation.
     Reserve "causal" for a conclusion that asserts one thing CAUSES another.
  5. A CHAIN OF THE SAME KIND IS ONE FORM. A passage running several steps of ONE
     sort of inference takes that form's name, once. A four-step causal chain
     (cycling -> exercise -> energy -> concentration -> better work) is
     ["causal"] — length alone does not add entries.
  6. A CHAIN THAT CROSSES KINDS LISTS EACH FORM, most central first. "Forty of
     the Swedes I met were Lutheran, so most Swedes are; so Sven probably is"
     performs two different moves and is
     ["generalization", "application of generalization"]. A causal chain ending
     in a normative "therefore we should..." lists both moves. Give at most three,
     and only forms the passage actually performs — do not pad the list.
  7. "other" MEANS NO FORM IN THE LIST APPLIES. It is the escape hatch for a real
     argument this vocabulary cannot name — an appeal to ignorance, a continuum
     fallacy. It is NOT for multi-step arguments: those name their moves under
     rule 6. Return ["other"] alone, never beside another form.
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


def normalize_forms(label):
    """Repair form-list constraints the API response schema cannot express.

    `minItems`/`maxItems`/`enum` are enforced by the schema; "no duplicates" and
    "`other` never sits beside a real form" are not, and the labeler does violate
    the latter -- it returned ["inference to the best explanation", "other"],
    which reads as "this is IBE, and also unnameable".

    Applied to LABELER output only, never to model predictions: training data must
    be clean, but a model's output has to be measured as it actually came out or
    schema validity stops meaning anything.
    """
    forms = label.get("form")
    if isinstance(forms, str):
        forms = [forms]
    if not isinstance(forms, list):
        return label
    seen = [f for i, f in enumerate(forms) if f not in forms[:i]]
    if len(seen) > 1:
        seen = [f for f in seen if f != "other"] or ["other"]
    return dict(label, form=seen[:MAX_FORMS])


def label_one(cl, examples, text, retries=3):
    """Label one passage. Retries with backoff; raises if all attempts fail."""
    contents = examples + [types.Content(role="user", parts=[types.Part(text=text)])]
    config = types.GenerateContentConfig(
        system_instruction=LABELER_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.0,
        http_options=types.HttpOptions(timeout=45000),
    )
    for attempt in range(retries):
        try:
            resp = cl.models.generate_content(model=MODEL, contents=contents, config=config)
            return normalize_forms(json.loads(resp.text))
        except Exception as exc:  # noqa: BLE001 - surface anything, retry, then give up
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries - 1} after "
                  f"{type(exc).__name__}: {exc}"[:200], file=sys.stderr)
            time.sleep(2 ** attempt)
