# LOGIC / LogicClimate

Downloaded 2026-09-02 from https://github.com/causalNLP/logical-fallacy (`data/`).

**Paper:** Jin, Lalwani, Vaidhya, Shen, Ding, Lyu, Sachan, Mihalcea, Schoelkopf.
"Logical Fallacy Detection." Findings of EMNLP 2022, pp. 7180-7198.
https://aclanthology.org/2022.findings-emnlp.532

**Files**
- `edu_{train,dev,test}.csv` — LOGIC, 2,449 examples, 13 fallacy classes.
  Columns that matter: `source_article` (the passage), `updated_label` (the class).
  Several leading `Unnamed:` index columns are junk from the authors' pandas export.
- `climate_all.csv` — LogicClimate, real climate-journalism prose.
  Columns differ: `source_article`, `logical_fallacies`, `original_url`.

**LICENSING — RESOLVED (non-commercial).** The dataset is MIT-licensed. The grant
is not in the repo -- there is no LICENSE file, and the HuggingFace mirror says
"unknown" -- it is in the PAPER, Appendix A, the datasheet under "Dataset Overview
for Responsible NLP":

> "The dataset is open-sourced with the MIT license, and the intended use is for
> academic research but not commercial purposes."

Checked 2026-09-06. An earlier pass here concluded "no grant of reuse" by reasoning
from the repo's silence and never opened the appendix. That was wrong. The missing
LICENSE file is a packaging omission, not an absence of a licence.

MIT permits use, modification and redistribution of derived work -- including model
weights -- so long as the copyright notice and licence text travel with it. Ship
both with anything derived from this data, alongside the citation below.

The trailing clause is the one ambiguity: MIT itself permits commercial use, so
"academic research but not commercial" states an intent the licence text does not
encode. For non-commercial work the two agree and there is nothing to resolve. Ask
the authors before any commercial use, and only then.

Separately, the arXiv version of the PAPER is CC BY-NC-SA 4.0
(https://arxiv.org/abs/2202.13758, "Rights to this article"). That governs the
article, not the dataset; quote the paper under those terms, use the data under MIT.

**Known label noise.** Human-annotated is not error-free. Example labeled
`faulty generalization`: "If we ban Hummers because they are bad for the
environment, eventually the government will ban all cars..." — that is a slippery
slope. Apply the same intent-vs-label triage used on the generated corpus.
