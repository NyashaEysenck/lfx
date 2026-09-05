import subprocess
for split,out in (("test_real","preds_v3_real.jsonl"),("test_gen","preds_v3_gen.jsonl")):
    subprocess.run(["python","predict.py","--split",split,"--adapter","lora_v3","--out",out],
                   cwd="/content", check=True)
print("TUNED DONE")
