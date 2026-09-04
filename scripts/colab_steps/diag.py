import subprocess
r = subprocess.run(["python","train_lora.py","--epochs","3","--out","lora_out"],
                   cwd="/content", capture_output=True, text=True)
print("EXIT", r.returncode)
print("---- STDOUT tail ----"); print(r.stdout[-3000:])
print("---- STDERR tail ----"); print(r.stderr[-5000:])
