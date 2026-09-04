import json, subprocess
r = subprocess.run(["python","train_lora.py","--epochs","6","--out","lora_relog"],
                   cwd="/content", capture_output=True, text=True)
open("/content/trainlog.txt","w").write(r.stdout + "\n===STDERR===\n" + r.stderr)
print("EXIT", r.returncode, "LOGGED")
