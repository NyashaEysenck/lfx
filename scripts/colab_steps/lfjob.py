"""Detached-job control, shared by the launch and poll steps.

Deliberately NOT named joblib.py: /content is on sys.path, so that name shadowed
the real joblib package and broke sklearn, which transformers imports at startup.

Callers set JOB (marker prefix) and CMD (shell command, or None to just poll).
State lives in three files on the VM so it survives the kernel being restarted
or the local CLI dying:  <job>.log, <job>.pid, <job>.done (holding the exit code).
"""
import os
import subprocess

_log, _pid, _done = f"/content/{JOB}.log", f"/content/{JOB}.pid", f"/content/{JOB}.done"


def _alive():
    if not os.path.exists(_pid):
        return False
    try:
        os.kill(int(open(_pid).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _tail(n=25):
    if not os.path.exists(_log):
        return ""
    return "".join(open(_log, errors="replace").readlines()[-n:])


if CMD and not _alive() and not os.path.exists(_done):
    for f in (_log, _done):
        if os.path.exists(f):
            os.remove(f)
    # `sh -c` records the exit code on the VM itself rather than in this kernel,
    # so the marker outlives a kernel restart.
    _p = subprocess.Popen(["sh", "-c", f"{CMD} >>{JOB}.log 2>&1; echo $? >{JOB}.done"],
                          cwd="/content", stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True)
    open(_pid, "w").write(str(_p.pid))
    open("/content/current_job", "w").write(JOB)
    print(f"{JOB} LAUNCHED pid={_p.pid}")
elif os.path.exists(_done):
    _code = open(_done).read().strip()
    print(_tail())
    print(f"{JOB} FINISHED rc={_code}")
    print("JOB OK" if _code == "0" else "JOB FAILED")
elif _alive():
    print(_tail())
    print(f"{JOB} RUNNING")
else:
    print(_tail())
    # No marker and no process: killed outright (OOM, VM reclaim).
    print(f"{JOB} DEAD")
