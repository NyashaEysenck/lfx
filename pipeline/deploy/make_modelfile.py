"""Generate an Ollama Modelfile for a packaged model.

The SYSTEM block is read from the model's own prompt.txt, never retyped. Pasting
it by hand is how it drifts, and drift here is silent: the weights keep working,
the wording changes underneath them, and the score falls with nothing to point
at. That already cost this project a shipped 3B falling 0.559 -> 0.265 form
accuracy on test_real with no weight changed.

The TEMPLATE block is not optional. Recent transformers writes the chat template
to a separate chat_template.jinja, while Ollama looks for it inside
tokenizer_config.json -- so a merged Qwen2.5 imports with the template
`{{ .Prompt }}`, a bare passthrough. The model then never sees the ChatML it was
fine-tuned on and emits noise: the first build of this scored 0/34 parseable on
test_real, with weights and quantisation both fine. Nothing warned; `ollama show
--system` even returned the correct prompt. So the chat format is pinned here
alongside the prompt, for the same reason.

The licence blocks are obligations, not decoration. Qwen's research licence
requires that a redistributed derivative carry a copy of the agreement, mark its
modifications, and display "Built with Qwen". LOGIC is MIT and requires its
notice to travel with anything substantial derived from it. Ollama's LICENSE
directive is where both belong, so they ship with the model rather than living
in a README nobody pulls.

    python pipeline/deploy/make_modelfile.py --model models/merged_v5 \\
        --name lfx-3b --out Modelfile.lfx-3b
"""

import argparse
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF-format dir Ollama imports from")
ap.add_argument("--prompt", help="prompt.txt to pin (default: <model>/prompt.txt, "
                                 "else the sibling mlx dir's copy)")
ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--adapter", default="models/lora_3b")
ap.add_argument("--out", required=True)
args = ap.parse_args()

model = pathlib.Path(args.model)
prompt_path = pathlib.Path(args.prompt) if args.prompt else model / "prompt.txt"
if not prompt_path.is_file():
    raise SystemExit(
        f"no prompt.txt at {prompt_path}. Refusing to guess: a model shipped with "
        f"the wrong prompt looks fine and scores badly. Pass --prompt explicitly.")
SYSTEM = prompt_path.read_text().strip()

NOTICE = f"""Logical Form Extractor — argument structure extraction.

Built with Qwen.

Base model: {args.base}
Modification: LoRA fine-tune (r=16, alpha=32, 6 epochs), merged into the base
weights. Adapter: {args.adapter}. Weights are modified from the original.

Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT,
Copyright (c) Alibaba Cloud. All Rights Reserved.

NON-COMMERCIAL USE ONLY. The Qwen Research License permits use, reproduction,
distribution and derivative works FOR NON-COMMERCIAL PURPOSES ONLY, defined as
research or evaluation. Full terms:
https://huggingface.co/{args.base}/blob/main/LICENSE

Training data: the LOGIC dataset (Jin et al., Findings of EMNLP 2022), MIT
licensed, plus prose generated for this project.

MIT License, Copyright (c) 2022 Zhijing Jin, Abhinav Lalwani, Tejas Vaidhya,
Xiaoyu Shen, Yiwen Ding, Zhiheng Lyu, Mrinmaya Sachan, Rada Mihalcea,
Bernhard Schoelkopf. Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions: The
above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS
IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

Citation: Jin et al. "Logical Fallacy Detection." Findings of EMNLP 2022,
7180-7198. https://aclanthology.org/2022.findings-emnlp.532
"""

# temperature 0: the evaluation harness decodes greedily, so anything else makes
# the served model a different model from the measured one.
# ChatML, matching Qwen2.5-Instruct's own chat_template.jinja. Ollama cannot read
# that file, so it is restated here rather than left to a default.
TEMPLATE = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

modelfile = f'''FROM {model}

TEMPLATE """{TEMPLATE}"""

SYSTEM """{SYSTEM}"""

PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

LICENSE """{NOTICE}"""
'''

pathlib.Path(args.out).write_text(modelfile)
print(f"wrote {args.out}")
print(f"  FROM      {model}")
print(f"  SYSTEM    {len(SYSTEM)} chars, read verbatim from {prompt_path}")
print(f"  TEMPLATE  ChatML, pinned (Ollama cannot read chat_template.jinja)")
print(f"  LICENSE   {len(NOTICE)} chars (Qwen research + LOGIC MIT)")
