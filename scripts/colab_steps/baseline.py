import subprocess
for split, out in (("test_real","preds_base_real.jsonl"), ("test_gen","preds_base_gen.jsonl")):
    subprocess.run(["python","predict.py","--split",split,"--out",out], cwd="/content", check=True)
print("BASELINE DONE")
