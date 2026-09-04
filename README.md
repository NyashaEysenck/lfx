# Logical Form Extractor

A fine-tuned 1.5B model that reads an argument and returns its structure as JSON.
Runs locally on Apple silicon at ~0.6s per passage, no network.

```console
$ ./logicalform "John says we should be vegetarians. John is a plant farmer,
                 so he is biased, and we should not take his view seriously."

  P1  John says we should be vegetarians.
  P2  John is a plant farmer.
  ∴   We should not take John's opinion on diet seriously.

  deductive · ad hominem  (informal fallacy)
  unstated: If someone has a vested interest in a topic, their opinion on it is
            biased and should not be taken seriously.
```

Built as an end-to-end fine-tuning exercise: data → labels → train → eval → package.
See [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) for the full log, including the
mistakes and what they cost.

## What it outputs

```json
{
  "premises": ["...", "..."],
  "conclusion": "...",
  "argument_type": "deductive" | "inductive",
  "form": "<one of 17>",
  "suppressed_premise": "..." | null
}
```

`form` covers six valid deductive patterns, five inductive ones, five fallacies,
and `other` for multi-step arguments that chain several inferences. Full enum in
[`lfx/schema.py`](lfx/schema.py).

## How well it works

Measured on 293 real passages with human labels, never trained on:

| metric | score |
|---|---|
| schema valid | 0.952 |
| **form accuracy** (end-to-end) | **0.775** |
| argument_type accuracy | 0.864 |

And on 34 hand-reviewed passages condensed from Aquinas, Hume, Madison, MLK:

| metric | base model | fine-tuned |
|---|---|---|
| premise F1 | 0.778 | **0.881** |
| conclusion similarity | 0.868 | **0.951** |
| form accuracy | 0.029 | 0.529 |

**Read that split honestly.** Decomposition is the strong half — pulling a paragraph
apart into premises and a conclusion is reliable. Form classification is the weak
half: right about half the time on hard philosophical prose, better (76-82%) on the
fallacy classes backed by human-labeled training data.

### Known limits

- **It repairs some fallacies into valid forms.** "If it rained the sidewalk would
  be wet; the sidewalk is wet; so it rained" still comes back as `modus ponens`
  rather than `affirming the consequent`. The failure is asymmetric — it calls bad
  arguments good, never the reverse.
- **It knows five fallacies.** Equivocation, ad populum, straw man, circular
  reasoning, slippery slope and the rest have no representation, and it does not
  fall back to `other` — it names the valid form the argument resembles.
- **It always answers.** There is no "this isn't an argument" output. The CLI
  rejects obviously degenerate input, but a real paragraph that makes no argument
  will still get a fabricated extraction.
- **When the form is wrong, `suppressed_premise` usually reads as nonsense.** A
  useful self-check in practice.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[local]"          # add [merge] or [colab] for the pipeline
```

Local inference needs a model in `models/` — not in the repo (9.6GB). Rebuild with
`scripts/run_v2.sh`, then `pipeline/train/merge_adapter.py`, then `mlx_lm convert`.

Corpus building needs Vertex AI via ADC, not an API key:

```bash
gcloud auth application-default login
```

> A Gemini API key from the Cloud console targets `generativelanguage.googleapis.com`,
> which bills a prepaid balance separate from your Cloud credits and fails with
> `429 prepayment credits depleted`. Vertex (`aiplatform`) reaches the same models
> and bills normally. See [`lfx/vertex.py`](lfx/vertex.py).

## Layout

```
lfx/              importable library — prompt, schema, JSON parsing, CLI
  prompts.py      the task prompt, used by training AND inference (single source)
  schema.py       form enum, families, form/argument_type consistency check
pipeline/
  corpus/         generate and import passages
  label/          label with Gemini, triage, repair
  build/          merge sources, split, format for training
  train/          LoRA fine-tune, predict, merge adapter   (train/ runs on Colab)
  eval/           score predictions field by field
scripts/          drive a Colab T4 from the terminal
data/raw/         seed corpus, few-shot examples, imported datasets
data/interim/     generated and labeled intermediates
data/splits/      train / val / test_gen / test_real / extra_test_human
results/          predictions from every run, for comparison
```

## Rebuilding from scratch

```bash
python pipeline/corpus/generate_corpus.py          # 400 passages, stratified
python pipeline/label/label_corpus.py --corpus data/interim/generated_corpus.jsonl \
    --out data/interim/generated_labeled.jsonl     # blind labeling
python pipeline/label/triage.py                    # intent vs label disagreements
python pipeline/corpus/import_logic.py             # human-labeled fallacies
python pipeline/label/label_logic.py               # extraction fields only
python pipeline/corpus/gen_formal.py               # template-built formal forms
python pipeline/corpus/gen_disjunctive.py
python pipeline/build/merge_corpus.py --cap 60
python pipeline/build/split_corpus.py
python pipeline/build/format_for_training.py train val test_gen test_real
./scripts/run_v2.sh                                # train + eval on a Colab T4
```

## Two findings worth carrying elsewhere

**Real data generalizes; templates teach the template.** Forms trained on
constructed examples scored 18/18 on constructed test text and 2/6 on natural
prose. The same run's `false dilemma`, trained on real passages, scored 10/12 on
real passages.

**Match what a class *contains*, not what it's called.** Importing LOGIC's
`faulty generalization` as `hasty generalization` looked like a clean 1:1 mapping
by name. It isn't — LOGIC uses it as an umbrella for any bad inductive leap,
including slippery slopes and false cause. Reading twenty passages would have
caught what the class names and counts could not.

## Data

`data/logic/` holds the LOGIC / LogicClimate datasets (Jin et al., *Logical Fallacy
Detection*, Findings of EMNLP 2022). **The source repository declares no license**,
so treat this as private study use — see [`data/logic/SOURCE.md`](data/logic/SOURCE.md)
before publishing anything derived from it.
