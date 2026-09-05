import subprocess
r = subprocess.run(["python","train_lora.py","--epochs","6","--out","lora_v4"],
                   cwd="/content", capture_output=True, text=True)
print("EXIT", r.returncode)
print(r.stdout[-1200:] if r.returncode==0 else r.stderr[-3000:])
if r.returncode==0: print("TRAIN DONE")
