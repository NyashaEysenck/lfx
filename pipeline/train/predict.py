"""Run a model over a split and write predictions. Runs on the Colab GPU.

Same script for the baseline and the tuned model — pass --adapter for the latter.
Using one script for both is deliberate: it guarantees the prompt, sampling
params, and parsing are identical on each side of the comparison, which is the
whole point of step 8.

    python predict.py --split test_real --out preds_base_real.jsonl
    python predict.py --split test_real --adapter ./lora_out --out preds_tuned_real.jsonl
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/Qwen2.5-1.5B-Instruct"


def extract_json(text):
    """Models wrap JSON in prose or fences. Take the first balanced object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="e.g. test_real (reads <split>.chat.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", help="path to the trained LoRA adapter; omit for baseline")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{args.split}.chat.jsonl")]
    ids = [json.loads(l)["id"] for l in open(f"{args.split}.jsonl")]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    with open(args.out, "w") as fh:
        for i, (rid, row) in enumerate(zip(ids, rows), 1):
            # drop the assistant turn — that is the answer we are asking for
            msgs = [m for m in row["conversations"] if m["role"] != "assistant"]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            fh.write(json.dumps({"id": rid, "raw": text, "parsed": extract_json(text)},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
