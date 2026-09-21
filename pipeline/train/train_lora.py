"""LoRA fine-tune Qwen2.5 on the extraction task. Runs on Colab (T4).

Loss is computed on the assistant turn only — the model is graded on the JSON it
produces, not on its ability to echo back the schema instruction and the passage.

    python train_lora.py --epochs 3 --out lora_out
    python train_lora.py --epochs 1 --init-adapter lora_v6 --out lora_v6

`--init-adapter` continues from an adapter trained in an EARLIER SESSION, which is
what makes training survive a free Colab VM. Three consecutive runs died before
finishing: one when a laptop lid closed and stopped the CLI keep-alive, one when
Colab reclaimed the VM at ~60 minutes of sustained GPU load, and one to a
keep-alive failure at 8 minutes that has no explanation. Each lost everything,
because checkpoints were written with save_strategy="epoch" onto a VM that then
disappeared, and nothing ever moved them off it.

Training in one-epoch chunks, carrying the adapter out after each, turns any of
those failures into the loss of a single epoch. The adapter is ~300 MB for a 3B,
which moves in a couple of minutes; a full HF checkpoint carrying optimizer state
would be several times that.

What that costs: the optimizer state does NOT carry across chunks, so each chunk
restarts Adam's moments and the LR schedule. For LoRA at this scale that is a mild
cost and a known one -- and it is strictly better than losing the run.
"""

# Unsloth MUST be imported before trl / transformers / peft, or its patches
# never get applied and training silently runs slower and hotter on memory.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse
import inspect

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen2.5-1.5B-Instruct")
ap.add_argument("--train", default="train.chat.jsonl")
ap.add_argument("--val", default="val.chat.jsonl")
ap.add_argument("--out", default="lora_out")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--max-seq", type=int, default=1024)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--init-adapter", help="continue from this adapter directory")
args = ap.parse_args()

import os

if args.init_adapter and os.path.isdir(args.init_adapter):
    # Loading the adapter directly restores the trained weights; get_peft_model
    # would discard them and start from a fresh zero-initialised LoRA.
    print(f"resuming from adapter {args.init_adapter}", flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.init_adapter, max_seq_length=args.max_seq,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)
else:
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq,
        dtype=None,        # let Unsloth pick: fp16 on T4, bf16 on newer cards
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=20260902,
    )

def to_text(batch):
    return {"text": [tok.apply_chat_template(c, tokenize=False)
                     for c in batch["conversations"]]}

ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
ds = ds.map(to_text, batched=True)

# TRL renames these across releases (max_seq_length -> max_length,
# tokenizer -> processing_class). Bind to whatever this install actually accepts
# so the script survives the next rename instead of dying at epoch 0.
cfg_params = inspect.signature(SFTConfig.__init__).parameters
cfg = dict(
    output_dir=args.out, dataset_text_field="text",
    per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
    num_train_epochs=args.epochs, learning_rate=args.lr,
    warmup_ratio=0.05, lr_scheduler_type="linear", optim="adamw_8bit",
    weight_decay=0.01, logging_steps=5,
    eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, seed=20260902, report_to="none",
)
cfg["max_length" if "max_length" in cfg_params else "max_seq_length"] = args.max_seq
cfg = {k: v for k, v in cfg.items() if k in cfg_params}

tr_params = inspect.signature(SFTTrainer.__init__).parameters
tok_kw = "processing_class" if "processing_class" in tr_params else "tokenizer"
trainer = SFTTrainer(
    model=model, **{tok_kw: tok},
    train_dataset=ds["train"], eval_dataset=ds["val"],
    args=SFTConfig(**cfg),
)
# grade the model on its answer, not on the prompt it was handed
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.train()
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"adapter saved -> {args.out}")
