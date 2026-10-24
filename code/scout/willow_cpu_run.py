"""Run one command (no shell) and record its wall time and the CPU time of its process tree (getrusage of the
waited-for children), appending a line to a TSV log; exit with the command's code.
Usage: python3 -I cpu_run.py LOG LABEL -- CMD ARG... Scout, 2026-10-24."""
import datetime
import resource
import subprocess
import sys
import time

log, label = sys.argv[1], sys.argv[2]
assert sys.argv[3] == "--"
cmd = sys.argv[4:]
r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
t0, w0 = datetime.datetime.now(datetime.timezone.utc), time.monotonic()
code = subprocess.call(cmd)
w1 = time.monotonic()
r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
user, sys_ = r1.ru_utime - r0.ru_utime, r1.ru_stime - r0.ru_stime
line = "%s\t%s\twall %.1f\tuser %.2f\tsys %.2f\tcpu %.2f\texit %d\n" % (
    t0.strftime("%Y-%m-%dT%H:%M:%SZ"), label, w1 - w0, user, sys_, user + sys_, code)
with open(log, "a", encoding="utf-8") as fh:
    fh.write(line)
sys.stdout.flush()
print(line, end="", file=sys.stderr)
sys.exit(code)
