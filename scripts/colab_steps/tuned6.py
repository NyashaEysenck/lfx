import subprocess
for split,out in (("test_real","preds_e6_real.jsonl"),("test_gen","preds_e6_gen.jsonl")):
    subprocess.run(["python","predict.py","--split",split,"--adapter","lora_e6","--out",out],
                   cwd="/content", check=True)
print("TUNED DONE")
