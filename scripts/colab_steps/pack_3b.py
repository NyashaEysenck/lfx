"""Tar the trained adapter so it can be pulled off the VM in one download.

A finished 3B adapter was lost once because it sat on the VM while the much
longer prediction step ran and the runtime went away underneath it. Training is
the expensive part; get it onto local disk before anything else starts.
"""
import os
import subprocess

if not os.path.isdir("/content/lora_3b"):
    print("PACK MISSING")
else:
    subprocess.run(["tar", "czf", "/content/lora_3b.tar.gz", "-C", "/content", "lora_3b"],
                   check=True)
    print("PACK DONE", os.path.getsize("/content/lora_3b.tar.gz"), "bytes")
