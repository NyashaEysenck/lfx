# Logical Form Extractor (LFX)

[![Model](https://img.shields.io/badge/Base_Model-Qwen2.5--3B--Instruct-blue.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Ollama](https://img.shields.io/badge/Ollama-nyashastino%2Flfx-black.svg)](https://ollama.com/nyashastino/lfx)
[![Accuracy](https://img.shields.io/badge/Form_Accuracy-89.7%25_(Bench)-brightgreen.svg)](#how-well-it-works)
[![License](https://img.shields.io/badge/License-Non--Commercial_Research-lightgrey.svg)](#license--attribution)

**Logical Form Extractor** is a fine-tuned 3B parameter language model that parses natural-language arguments into structured JSON: isolating individual premises, identifying the conclusion, classifying the argument as deductive or inductive, naming the specific logical form or fallacy, and reconstructing any suppressed (enthymematic) premises.

Available out-of-the-box on **Ollama** (`nyashastino/lfx`) and locally on Apple Silicon via **MLX** (`logicalform`).

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

---

## Quickstart with Ollama

The fastest way to use Logical Form Extractor is via [Ollama](https://ollama.com).

### 1. Run from the Terminal

```bash
ollama run nyashastino/lfx "All men are mortal. Socrates is a man. Therefore Socrates is mortal."
```

Output:
```json
{
  "premises": [
    "All men are mortal.",
    "Socrates is a man."
  ],
  "conclusion": "Socrates is mortal.",
  "argument_type": "deductive",
  "form": [
    "categorical syllogism"
  ],
  "suppressed_premise": null
}
```

### Available Tags

| Tag | Quantization | Size | End-to-End Form Accuracy | Notes |
|---|---|---|---|---|
| `nyashastino/lfx:latest`<br>`nyashastino/lfx:3b-q8` | `Q8_0` | 3.3 GB | **89.7%** (Full Benchmark)<br>**78.1%** (Held-Out Human Prose) | **Primary Release.** Best overall accuracy, highest premise extraction fidelity, 0.861 Macro-F1 across 22 classes. |

To pull the model:
```bash
ollama pull nyashastino/lfx
ollama pull nyashastino/lfx:3b-q8
```

### 2. Python Integration

```python
import json
import ollama

passage = (
    "If the city raises parking meter rates, downtown retail foot traffic will plunge. "
    "Foot traffic has dropped by 30% this quarter. Therefore, the city must have raised meter rates."
)

response = ollama.chat(
    model="nyashastino/lfx",
    messages=[{"role": "user", "content": passage}],
)

data = json.loads(response["message"]["content"])
print(f"Type: {data['argument_type']}")
print(f"Form: {', '.join(data['form'])}")
print(f"Conclusion: {data['conclusion']}")
```

### 3. REST API (`curl`)

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "nyashastino/lfx",
  "prompt": "If it rained, the sidewalk would be wet. It did not rain. Thus, the sidewalk is not wet.",
  "stream": false
}'
```

---

## Output Schema (v2.0)

The model guarantees a valid JSON object matching this schema:

```json
{
  "premises": ["<premise 1>", "<premise 2>", "..."],
  "conclusion": "<the asserted conclusion>",
  "argument_type": "deductive" | "inductive",
  "form": ["<form 1>", "..."],
  "suppressed_premise": "<enthymeme / unstated assumption>" | null
}
```

### Schema Notes
- **`premises`**: An array of explicit claims functioning as evidence or support.
- **`conclusion`**: The primary claim the premises aim to establish.
- **`argument_type`**: `"deductive"` if the conclusion is presented as necessarily following; `"inductive"` if presented as probable or supported by evidence.
- **`form`**: An array of 1 to 3 logical forms (ordered primary move first). Lists a single form for standard arguments, or multiple forms when distinct inferential steps are combined.
- **`suppressed_premise`**: A reconstructed unstated premise necessary to link premises to conclusion, or `null` if the argument is fully explicit.

---

## Supported Logical Forms (22 Categories)

The taxonomy aligns with formal logic and critical thinking curricula (modeled on Duke University's *Think Again* framework):

| Family | Forms |
|---|---|
| **Valid Deductive (6)** | `modus ponens`, `modus tollens`, `hypothetical syllogism`, `disjunctive syllogism`, `categorical syllogism`, `reductio ad absurdum` |
| **Inductive (6)** | `generalization`, `application of generalization`, `inference to the best explanation`, `analogy`, `causal`, `authority` |
| **Formal Fallacies (2)** | `affirming the consequent`, `denying the antecedent` |
| **Informal Fallacies (7)** | `ad hominem`, `hasty generalization`, `false dilemma`, `ad populum`, `begging the question`, `straw man`, `equivocation` |
| **Other (1)** | `other` *(reserved for multi-step arguments or patterns outside this taxonomy)* |

Detailed definitions and validation logic are implemented in [`lfx/schema.py`](lfx/schema.py).

---

## How Well It Works

Evaluation on the unified **LFX-Bench v1** (1,521 arguments covering all 22 classes with verified provenance) demonstrates strong premise extraction, near-perfect schema compliance, and high form accuracy across both symbolic logic and authentic human prose.

### 1. Full Benchmark Evaluation (`LFX-Bench v1`, n=1521, 22 Classes)

Scored using greedy decoding (temperature 0) against ground-truth labels:

| Metric | Base Model (Qwen2.5-3B) | Tuned LFX (3B-Q8) |
|---|---|---|
| **End-to-End Form Accuracy** | 0.029 | **0.897** (89.7%) |
| **Macro-F1 (across 22 classes)** | ~0.03 | **0.861** |
| **Schema Validity** | 0.735 | **0.998** (99.8%) |
| **JSON Parses** | 0.812 | **0.999** (99.9%) |
| **Argument Type Accuracy** | 0.560 | **0.944** (94.4%) |
| **Premise F1** | 0.778 | **0.890** |
| **Conclusion Similarity** | 0.868 | **0.911** |
| **Suppressed Premise Match** | 0.480 | **0.900** |

### 2. Breakdown by Provenance Tier

The benchmark evaluates four distinct tiers to prevent ease-of-generation biases from masking real-world performance:

| Tier | n | Description / Guarantee | Form Accuracy |
|---|---|---|---|
| **constructed-symbolic** | 693 | Label fixed by construction; verified by blind formalization round-trip through `lfx.formal` | **98.8%** (685 / 693) |
| **constructed-consensus** | 309 | Inductive forms; verified by two distinct model families agreeing blind | **92.9%** (287 / 309) |
| **external-human** | 485 | Authentic human prose from the LOGIC dataset (never seen in training) | **78.1%** (379 / 485) |
| **project-authored** | 34 | Condensed historical & philosophical texts (Aquinas, Hume, Douglass, etc.)* | **41.2%** (14 / 34) |

*\*On `project-authored`, 12 of the 34 passages are multi-step chained arguments classified under `other`. On the 22 single-form passages, form accuracy reaches **63.6%** (14 / 22).*

### 3. Precision, Recall, and F1 on Key Forms

| Form | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| `hypothetical syllogism` | 1.000 | 1.000 | **1.000** | Perfect rule extraction |
| `affirming the consequent` | 0.989 | 0.989 | **0.989** | Overcomes charity repair |
| `reductio ad absurdum` | 0.963 | 0.988 | **0.975** | Raised from 2.5% in early checkpoints |
| `categorical syllogism` | 0.864 | 0.989 | **0.922** | Robust quantified premise recognition |
| `ad hominem` | 0.962 | 0.750 | **0.843** | Rebalanced; clean separation from straw man |
| `ad populum` | 0.835 | 0.827 | **0.831** | Stable on human political prose |
| `false dilemma` | 0.772 | 0.830 | **0.800** | Distinguishes disjunctions from false dilemmas |
| `begging the question` | 0.885 | 0.711 | **0.788** | Catches circular premises |
| `straw man` | 0.568 | 0.852 | **0.681** | Rescued from over-prediction |
| `equivocation` | 0.643 | 0.643 | **0.643** | Detects semantic term shifts |

---

## Capabilities and Known Limitations

- **Decomposition vs. Evaluation:** LFX is an extraction parser, not an epistemic fact-checker. It extracts the argument structure as presented by the author; it does not judge whether the premises are true.
- **Asymmetric Fallacy Identification:** The model rarely misidentifies a valid argument as a fallacy. However, subtle formal fallacies can occasionally be "repaired" into their valid structural counterparts (e.g., an ambiguous *affirming the consequent* may be labeled *modus ponens*).
- **Closed Fallacy Taxonomy:** Unsupported fallacies (e.g., *slippery slope*, *red herring*, *circular reasoning*) are mapped to their nearest structural form or `other`.
- **Argument Assumption:** The fine-tuning corpus consists exclusively of arguments. When fed arbitrary non-argument text (e.g., pure descriptive prose), the model may attempt to extract an argument. The local CLI includes guard heuristics to reject degenerate inputs (<6 words or repetitive tokens).
- **Suppressed Premise Diagnostic:** If the predicted `form` is incorrect, the `suppressed_premise` often reveals incongruity or nonsensical assumptions—providing a useful self-check during review.

---

## Local Inference on Apple Silicon (MLX)

For local development on macOS with Apple Silicon:

```bash
# 1. Clone & create virtual environment
git clone https://github.com/NyashaEysenck/lfx.git
cd lfx
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install package in editable mode
pip install -e ".[local]"
```

Run arguments directly or pipe input:
```bash
# Direct argument
./logicalform "Either we cut spending or face bankruptcy. We cannot cut spending, so we will go bankrupt."

# Reading from clipboard / stdin
pbpaste | ./logicalform

# Interactive REPL session
./logicalform --repl

# Raw JSON output for downstream tooling
./logicalform --json < passage.txt
```

---

## Repository Structure

```
lfx/                     Core library package
  prompts.py             Canonical system prompt (pinned across training and inference)
  schema.py              22-form enum, taxonomic families, and consistency validation
  cli.py                 CLI runner and formatted terminal rendering
  jsonio.py              Robust JSON extraction and repair utilities

pipeline/                Data and training pipeline
  corpus/                Corpus generation and external dataset importers
  label/                 Automated labeling (Vertex AI), triage, and human review tools
  build/                 Deduplication, split stratification, and ChatML formatting
  train/                 LoRA fine-tuning, adapter merging, and quantization scripts
  eval/                  Field-by-field scoring harness (F1, sequence similarity, form accuracy)
  deploy/                llama.cpp conversion and Ollama Modelfile packaging

data/                    Dataset splits and raw sources
  benchmark/             lfx_bench_v1 (1521 records across 22 classes with provenance)
  splits/                train (2069), val (160), test_gen (160), test_real (34), extra_test_human (485)
  logic/                 LOGIC / LogicClimate dataset (Jin et al., EMNLP Findings 2022)
  mafalda/               MAFALDA gold standard validation set (Helwe et al., 2024)

docs/                    In-depth documentation
  PROJECT_BRIEF.md       Comprehensive technical report and engineering log
  OLLAMA_MODEL_CARD.md   Model card and reference guide for Ollama
```

---

## Key Engineering Takeaways

1. **Real Data Generalizes; Templates Teach the Template:** Early experiments with template-generated formal fallacies scored 18/18 on synthetic phrasing but collapsed to 2/6 on natural human prose. Training on real-world human passages (from LOGIC) immediately generalized (scoring >76% on unseen prose).
2. **The Capacity Lever:** Scaling from 1.5B to 3B parameters yielded a negligible change on synthetic text (+3.1%), but delivered a massive **+11.5% jump in form accuracy on natural human prose** (McNemar test $z=5.13$, $p < 10^{-6}$). Parameter capacity specifically enables generalization across complex rhetorical variations.
3. **Data Leaks and Upstream Corrections:** In multi-source datasets, deduplication must be enforced at the text level *before* splitting. In-place fixes on split files are brittle and easily overwritten by upstream pipeline re-runs.
4. **Quantization Profile:** Converting the 3B model from 8-bit to 4-bit (`Q4_K_M`) costs only 0.010 in form accuracy while slashing model size from 3.3 GB to 1.9 GB. However, premise span precision and suppressed premise nuance suffer more noticeable degradations under 4-bit quantization, making 8-bit (`Q8_0`) the preferred release for desktop environments.

For the complete chronology, ablations, and pipeline post-mortem, see [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md).

---

## License & Attribution

- **Base Model:** Fine-tuned from `Qwen2.5-3B-Instruct`. Qwen is licensed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) (Copyright © Alibaba Cloud). **Non-commercial research use only.**
- **Attribution Notice:** Built with Qwen.
- **Training Data:** Incorporates the LOGIC dataset under the MIT License (Copyright © 2022 Zhijing Jin et al.).
- **Notices:** Detailed notices and licenses are documented in [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Citation
```bibtex
@inproceedings{jin-etal-2022-logical,
  title = "Logical Fallacy Detection",
  author = "Jin, Zhijing and Lalwani, Abhinav and Vaidhya, Tejas and Shen, Xiaoyu and Ding, Yiwen and Lyu, Zhiheng and Sachan, Mrinmaya and Mihalcea, Rada and Schoelkopf, Bernhard",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
  year = "2022",
  pages = "7180--7198",
  url = "https://aclanthology.org/2022.findings-emnlp.532"
}
```
