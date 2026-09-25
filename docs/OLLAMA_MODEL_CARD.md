# nyashastino/lfx — Logical Form Extractor

**Extract the logical skeleton of natural-language arguments into structured JSON.**

`nyashastino/lfx` is a fine-tuned 3B parameter model (based on `Qwen2.5-3B-Instruct`) engineered specifically to deconstruct natural-language arguments. Given an argumentative passage, it extracts individual premises, isolates the asserted conclusion, determines whether the argument is deductive or inductive, identifies the specific logical form or fallacy (from a 22-class taxonomy), and reconstructs unstated enthymematic premises.

---

## Quick Start

Run directly in your terminal:

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

---

## Model Tags & Quantizations

| Tag | Quantization | Size | RAM Required | Best For |
|---|---|---|---|---|
| `nyashastino/lfx:latest`<br>`nyashastino/lfx:3b-q8` | `Q8_0` (8-bit) | 3.3 GB | ~4.5 GB | **Primary Release.** Best overall accuracy (89.7% benchmark accuracy, 78.1% on held-out human prose) and highest premise extraction fidelity. |

To pull the model:
```bash
ollama pull nyashastino/lfx
ollama pull nyashastino/lfx:3b-q8
```

---

## Output Schema

The model responds strictly with a single, valid JSON object matching the following structure:

```json
{
  "premises": [
    "First premise statement",
    "Second premise statement"
  ],
  "conclusion": "The claim being argued for",
  "argument_type": "deductive" | "inductive",
  "form": [
    "primary logical form or fallacy"
  ],
  "suppressed_premise": "Unstated assumption required for validity, or null"
}
```

### Field Definitions

- **`premises`** *(array of strings)*: The explicit statements provided by the speaker as evidence, reasons, or justifications.
- **`conclusion`** *(string)*: The central asserted claim that the premises are intended to support.
- **`argument_type`** *(string)*:
  - `"deductive"`: The author presents the conclusion as following necessarily from the premises.
  - `"inductive"`: The author presents the premises as making the conclusion probable or empirically likely.
- **`form`** *(array of 1–3 strings)*: The inferential pattern or fallacy exhibited. Most single-step arguments have exactly 1 form. Multi-step arguments list their inferential moves ordered by the most central step first.
- **`suppressed_premise`** *(string or null)*: An unstated premise (enthymeme) necessary to bridge the premises to the conclusion. If the argument is fully explicit, this field is `null`.

---

## Supported Logical Forms (22 Categories)

Modeled on classical formal logic and critical thinking frameworks (aligned with Duke University's *Think Again* curriculum):

### 1. Valid Deductive Forms
- `modus ponens` (*If P then Q; P; therefore Q*)
- `modus tollens` (*If P then Q; not Q; therefore not P*)
- `hypothetical syllogism` (*If P then Q; if Q then R; therefore if P then R*)
- `disjunctive syllogism` (*P or Q; not P; therefore Q*)
- `categorical syllogism` (*All A are B; C is A; therefore C is B*)
- `reductio ad absurdum` (*Assume P; derive contradiction; therefore not P*)

### 2. Inductive Forms
- `generalization` (*Sampling instances to infer a general pattern*)
- `application of generalization` (*Applying an established empirical generalization to a specific instance*)
- `inference to the best explanation` (*Inferring a hypothesis because it best accounts for observed facts*)
- `analogy` (*Inferring similarity in an unobserved aspect based on known shared properties*)
- `causal` (*Inferring a cause-and-effect relationship from correlation or sequence*)
- `authority` (*Supporting a claim by citing relevant, credible expertise*)

### 3. Formal Fallacies
- `affirming the consequent` (*If P then Q; Q; therefore P*)
- `denying the antecedent` (*If P then Q; not P; therefore not Q*)

### 4. Informal Fallacies
- `ad hominem` (*Attacking the person's character, motive, or background rather than their argument*)
- `hasty generalization` (*Drawing a sweeping inductive conclusion from an unrepresentative or tiny sample*)
- `false dilemma` (*Falsely presenting complex alternatives as an exhaustive either-or choice*)
- `ad populum` (*Arguing something must be true or good simply because many people believe it*)
- `begging the question` (*Assuming the conclusion in the premises; circular reasoning*)
- `straw man` (*Misrepresenting or exaggerating an opponent's position to make it easier to attack*)
- `equivocation` (*Exploiting multiple meanings of a word or phrase within the same reasoning*)

### 5. Other
- `other` (*Reserved for complex multi-step chaining or arguments falling outside the above 21 categories*)

---

## Examples

### Example 1: Enthymeme with Suppressed Premise
**Input:**
> "John says we should all become vegetarians, but he owns a massive commercial organic farm, so we shouldn't take his dietary lectures seriously."

**Output:**
```json
{
  "premises": [
    "John says we should all become vegetarians.",
    "John owns a massive commercial organic farm."
  ],
  "conclusion": "We should not take John's dietary lectures seriously.",
  "argument_type": "deductive",
  "form": [
    "ad hominem"
  ],
  "suppressed_premise": "If someone stands to benefit financially from a cause, their arguments for that cause are invalid."
}
```

### Example 2: Formal Fallacy
**Input:**
> "If the fire alarm triggers, the entire building will be evacuated immediately. The building is being evacuated right now, so the fire alarm must have gone off."

**Output:**
```json
{
  "premises": [
    "If the fire alarm triggers, the entire building will be evacuated immediately.",
    "The building is being evacuated right now."
  ],
  "conclusion": "The fire alarm must have gone off.",
  "argument_type": "deductive",
  "form": [
    "affirming the consequent"
  ],
  "suppressed_premise": null
}
```

### Example 3: Inference to the Best Explanation
**Input:**
> "The kitchen window was forced open, muddy footprints lead directly to the safe, and the jewelry is gone. A burglar broke in."

**Output:**
```json
{
  "premises": [
    "The kitchen window was forced open.",
    "Muddy footprints lead directly to the safe.",
    "The jewelry is missing."
  ],
  "conclusion": "A burglar broke into the house.",
  "argument_type": "inductive",
  "form": [
    "inference to the best explanation"
  ],
  "suppressed_premise": null
}
```

---

## Usage Integrations

### Python (`ollama-python`)

```python
import json
import ollama

client = ollama.Client()

prompt = (
    "Either humanity dramatically reduces carbon emissions within a decade, "
    "or we face ecological collapse. We will not cut emissions, therefore "
    "ecological collapse is inevitable."
)

res = client.chat(
    model="nyashastino/lfx",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.0},
)

result = json.loads(res["message"]["content"])
print(json.dumps(result, indent=2))
```

### REST API (`curl`)

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "nyashastino/lfx",
  "messages": [
    {"role": "user", "content": "Nobody has proven that ghosts don'\''t exist, so they must be real."}
  ],
  "stream": false
}'
```

### JavaScript / TypeScript

```javascript
import ollama from 'ollama';

const response = await ollama.chat({
  model: 'nyashastino/lfx',
  messages: [{
    role: 'user',
    content: 'Smartphones cause attention deficits because constantly switching apps fragments concentration.'
  }],
});

const analysis = JSON.parse(response.message.content);
console.log(analysis);
```

---

## Benchmark Performance

Evaluated on the unified **LFX-Bench v1** (1,521 unseen argumentative passages across all 22 classes with verified provenance):

| Benchmark Metric | Full Benchmark (n=1521) | Held-Out Human Prose (n=485) |
|---|---|---|
| **End-to-End Form Accuracy** | **89.7%** (1365 / 1521) | **78.1%** (379 / 485) |
| **Macro-F1 (22 Classes)** | **0.861** | **0.840** |
| **Schema Validity** | **99.8%** | **99.8%** |
| **JSON Parses** | **99.9%** | **100%** |
| **Argument Type Accuracy** | **94.4%** | **84.5%** |
| **Premise F1** | **0.890** | **0.748** |
| **Conclusion Similarity** | **0.911** | **0.925** |
| **Suppressed Premise Match** | **0.900** | **0.812** |

### Breakdown by Provenance Tier:
- **`constructed-symbolic` (n=693):** **98.8%** (label specified by construction, verified by blind formalization through symbolic rules)
- **`constructed-consensus` (n=309):** **92.9%** (inductive forms verified by blind multi-model agreement)
- **`external-human` (n=485):** **78.1%** (authentic human op-eds and speech passages from LOGIC)
- **`project-authored` (n=34):** **41.2%** (condensed classical philosophical texts; 12 of 34 are multi-step `other`)

### Per-Class Performance on Key Forms:
- **`hypothetical syllogism`:** 1.000 F1 (100% recall, 100% precision)
- **`affirming the consequent`:** 0.989 F1 (98.9% recall, 98.9% precision)
- **`reductio ad absurdum`:** 0.975 F1 (98.8% recall, 96.3% precision)
- **`categorical syllogism`:** 0.922 F1 (98.9% recall, 86.4% precision)
- **`ad hominem`:** 0.843 F1 (75.0% recall, 96.2% precision)
- **`ad populum`:** 0.831 F1 (82.7% recall, 83.5% precision)
- **`false dilemma`:** 0.800 F1 (83.0% recall, 77.2% precision)
- **`begging the question`:** 0.788 F1 (71.1% recall, 88.5% precision)
- **`straw man`:** 0.681 F1 (85.2% recall, 56.8% precision)
- **`equivocation`:** 0.643 F1 (64.3% recall, 64.3% precision)

---

## Modelfile & Generation Settings

The model uses the ChatML template and zero-temperature decoding for deterministic extraction:

```dockerfile
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

---

## Limitations & Best Practices

1. **Extraction, Not Truth-Checking:** The model extracts the internal structural relationship between premises and conclusions. It does not perform real-world fact-checking to determine if premises are factually accurate.
2. **Asymmetric Fallacy Identification:** The model rarely flags a sound argument as a fallacy. However, weak or subtly flawed arguments may occasionally be categorized under their valid structural twin (e.g., treating *affirming the consequent* as *modus ponens*).
3. **Closed Taxonomy:** Arguments based on informal fallacies outside the 22 categories (e.g., *slippery slope*, *no true Scotsman*) will be mapped to the closest applicable category or labeled `other`.
4. **Non-Argument Text:** The model was trained specifically on arguments. If supplied with purely descriptive or non-argumentative prose (e.g., a weather report or novel excerpt), it may attempt to manufacture an argument structure. Ensure input passages contain an intended point or argument.

---

## License & Attribution

- **Base Weights:** Fine-tuned from `Qwen2.5-3B-Instruct`. Qwen is licensed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) (Copyright © Alibaba Cloud). **Non-commercial research use only.**
- **Attribution Notice:** Built with Qwen.
- **Dataset Attribution:** Fine-tuned on the LOGIC dataset (Jin et al., Findings of EMNLP 2022), licensed under MIT (Copyright © 2022 Zhijing Jin et al.).

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
