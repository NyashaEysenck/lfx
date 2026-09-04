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

**LICENSING — UNRESOLVED.** The repository declares NO license, which under default
copyright means no grant of reuse. Fine for private study; NOT clearly usable in
anything published or redistributed. Resolve with the authors before any public
release of a model or dataset derived from this.

**Known label noise.** Human-annotated is not error-free. Example labeled
`faulty generalization`: "If we ban Hummers because they are bad for the
environment, eventually the government will ban all cars..." — that is a slippery
slope. Apply the same intent-vs-label triage used on the generated corpus.
