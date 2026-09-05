"""Report the current detached job's state. Safe to call any number of times."""
JOB = open("/content/current_job").read().strip()
CMD = None
exec(open("/content/lfjob.py").read())
