import subprocess
for split,out in (("test_real","preds_v2_real.jsonl"),("test_gen","preds_v2_gen.jsonl")):
    subprocess.run(["python","predict.py","--split",split,"--adapter","lora_v2","--out",out],
                   cwd="/content", check=True)
print("TUNED DONE")
