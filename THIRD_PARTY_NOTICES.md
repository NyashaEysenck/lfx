# Third-party notices

This repository redistributes data from two external datasets, and its training
scripts build on a third-party base model. Each is listed below with the notice
its licence requires.

The project's own code, generated corpus and documentation are MIT-licensed; see
`LICENSE`.

---

## LOGIC / LogicClimate

Redistributed verbatim as `data/logic/edu_{train,dev,test}.csv` and
`data/logic/climate_all.csv`. Passages derived from it also appear throughout
`data/interim/` and `data/splits/` under `logic_*` and `esc_*` ids.

**Licence: MIT.** The upstream repository ships no `LICENSE` file; the grant is
stated in the paper, Appendix A, in the datasheet under "Dataset Overview for
Responsible NLP":

> "The dataset is open-sourced with the MIT license, and the intended use is for
> academic research but not commercial purposes."

The authors published no copyright line, so the notice below names them as the
copyright holders. MIT permits commercial use; the sentence above states an
intent it does not encode. This project is non-commercial, so the two agree here.
Anyone intending commercial use should take that up with the authors.

```
MIT License

Copyright (c) 2022 Zhijing Jin, Abhinav Lalwani, Tejas Vaidhya, Xiaoyu Shen,
Yiwen Ding, Zhiheng Lyu, Mrinmaya Sachan, Rada Mihalcea, Bernhard Schoelkopf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Source: https://github.com/causalNLP/logical-fallacy

```bibtex
@inproceedings{jin-etal-2022-logical,
    title = "Logical Fallacy Detection",
    author = "Jin, Zhijing and Lalwani, Abhinav and Vaidhya, Tejas and
      Shen, Xiaoyu and Ding, Yiwen and Lyu, Zhiheng and Sachan, Mrinmaya and
      Mihalcea, Rada and Schoelkopf, Bernhard",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    pages = "7180--7198",
    url = "https://aclanthology.org/2022.findings-emnlp.532",
    doi = "10.18653/v1/2022.findings-emnlp.532"
}
```

The paper itself (arXiv version) is separately licensed CC BY-NC-SA 4.0. That
governs the article, not the dataset.

---

## MAFALDA

Redistributed verbatim as `data/mafalda/gold_standard_dataset.jsonl`.

**Licence: CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0/

Used as an external validation set only. Nothing was taken FROM MAFALDA into this
project's corpus or splits, so no ShareAlike obligation attaches to anything here.

27 passages do appear on both sides, and they are not counter-examples: every one
carries `origin: logic-human`, meaning it entered through the LOGIC import in
Phase 2. MAFALDA consolidates several earlier fallacy datasets, LOGIC among them,
so these are passages the two corpora share upstream. They reached this project
under LOGIC's MIT grant, not under CC BY-SA. Verified by origin, not assumed from
the text match.

Source: https://github.com/ChadiHelwe/MAFALDA

```bibtex
@inproceedings{helwe-etal-2024-mafalda,
    title = "{MAFALDA}: A Benchmark and Comprehensive Study of Fallacy Detection
      and Classification",
    author = "Helwe, Chadi and Calamai, Tom and Paris, Pierre-Henri and
      Clavel, Chlo{\'e} and Suchanek, Fabian",
    booktitle = "Proceedings of NAACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.naacl-long.270/"
}
```

---

## Qwen2.5 (base model)

No model weights are distributed from this repository — `models/` is gitignored,
and every artifact under it is regenerable from the scripts in `pipeline/train/`.
This notice covers the scripts that fine-tune against the base model, and applies
in full to any model artifact published elsewhere.

**Licence: Qwen RESEARCH LICENSE AGREEMENT** — non-commercial use only, defined
as "research or evaluation purposes only".

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT,
> Copyright (c) Alibaba Cloud. All Rights Reserved.

Built with Qwen.

Base: `Qwen/Qwen2.5-3B-Instruct` (via `unsloth/Qwen2.5-3B-Instruct`).
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

Any published derivative must ship a copy of that agreement, mark its
modifications, and display "Built with Qwen" in its documentation.

---

## Generated corpus

Records with `origin: gemini` (612 in `corpus_v20.jsonl`) and `origin: template`
(87) were produced by this project — the former with the Gemini API, the latter
from templates in `pipeline/corpus/`. They are covered by this repository's own
MIT licence. They are noted here for provenance, not because a third party holds
rights in them.

## Test passages

`data/splits/test_real.jsonl` contains condensed restatements of arguments from
public-domain sources (Aquinas, Anselm, Pascal, Descartes, Hume, Plato, Epicurus,
the Federalist Papers, Lincoln, Douglass, 1 Kings) alongside original examples
written for this project. They are paraphrases of argument structure, not quoted
text. Nothing in-copyright is reproduced.
