"""Start the 3B fine-tune as a DETACHED process on the VM and return at once.

Three previous long runs were lost because `colab exec` blocks for the whole job:
when the local CLI process died (sleep, interrupt, closed terminal) the remote
kernel kept working and nobody was left to collect the output. Detaching inverts
that -- the job is owned by the VM, the local side only polls. Losing the laptop
now costs nothing.

Idempotent: re-running while a job is alive reports RUNNING instead of starting
a second trainer on the same GPU.
"""
JOB = "train_3b"
CMD = ("python train_lora.py --base unsloth/Qwen2.5-3B-Instruct "
       "--epochs 6 --out lora_3b")
exec(open("/content/lfjob.py").read())
