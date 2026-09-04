# MAFALDA

Downloaded 2026-09-04 from https://github.com/ChadiHelwe/MAFALDA (`datasets/`).

**Paper:** Helwe, Calamai, Paris, Clavel, Suchanek. "MAFALDA: A Benchmark and
Comprehensive Study of Fallacy Detection and Classification." NAACL 2024.
https://aclanthology.org/2024.naacl-long.270/

**Licence: CC-BY-SA** — a real licence, unlike LOGIC's none. Attribution and
share-alike required if anything derived from it is published.

`gold_standard_dataset.jsonl` — 200 texts, 272 annotated spans, 25 fallacy classes.
Span-level with a disjunctive scheme: several annotations may be correct for one
span, which is an honest treatment of how contested fallacy labelling is.

**Used as a TEST set, never for training.** Per-class counts are small (7-28), so
it is thin training material but the best external validation available: expert
annotated, real prose (Reddit and similar), and independent of this project's
generator, labeller and reviewer.

Classes overlapping our schema: hasty generalization 28, false dilemma 18,
ad hominem 16, ad populum 14, straw man 13, circular reasoning 11, equivocation 7.
Also carries slippery slope 11 and false analogy 12, which our schema lacks.
