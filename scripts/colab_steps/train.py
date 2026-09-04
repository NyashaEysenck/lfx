import subprocess
subprocess.run(["python","train_lora.py","--epochs","3","--out","lora_out"], cwd="/content", check=True)
print("TRAIN DONE")
