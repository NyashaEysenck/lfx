import subprocess
r = subprocess.run(["python","train_lora.py","--epochs","0.05","--out","smoke_out"],
                   cwd="/content", capture_output=True, text=True)
print("EXIT", r.returncode)
if r.returncode: print(r.stderr[-3000:])
else: print("SMOKE OK"); print(r.stdout[-800:])
