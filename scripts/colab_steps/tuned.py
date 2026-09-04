import subprocess
for split, out in (("test_real","preds_tuned_real.jsonl"), ("test_gen","preds_tuned_gen.jsonl")):
    subprocess.run(["python","predict.py","--split",split,"--adapter","lora_out","--out",out],
                   cwd="/content", check=True)
print("TUNED DONE")
