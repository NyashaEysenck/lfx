import subprocess
for split,out in (("test_real","preds_v4_real.jsonl"),("test_gen","preds_v4_gen.jsonl")):
    subprocess.run(["python","predict.py","--split",split,"--adapter","lora_v4","--out",out],
                   cwd="/content", check=True)
print("TUNED DONE")
