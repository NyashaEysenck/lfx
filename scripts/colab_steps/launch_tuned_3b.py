"""Detached prediction pass for the 3B adapter. Same rationale as launch_3b.py."""
JOB = "tuned_3b"
CMD = ("python predict.py --split test_real --base unsloth/Qwen2.5-3B-Instruct "
       "--adapter lora_3b --out preds_3b_real.jsonl && "
       "python predict.py --split test_gen --base unsloth/Qwen2.5-3B-Instruct "
       "--adapter lora_3b --out preds_3b_gen.jsonl")
exec(open("/content/lfjob.py").read())
