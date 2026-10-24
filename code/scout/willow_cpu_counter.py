"""Read the kernel's counters of the Claude Code process (PID given): its own CPU and its reaped children's CPU,
in seconds, with the UTC time; append a line to a TSV log. Usage: python3 -I counter.py PID LOG LABEL. Scout, 2026-10-24."""
import datetime
import os
import sys

pid, log, label = int(sys.argv[1]), sys.argv[2], sys.argv[3]
tck = os.sysconf("SC_CLK_TCK")
s = open("/proc/%d/stat" % pid).read()
rest = s[s.rindex(")") + 2:].split()
utime, stime, cutime, cstime = (int(x) / tck for x in rest[11:15])
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
line = "%s\t%s\tpid %d\tself %.2f\tchildren %.2f\n" % (now, label, pid, utime + stime, cutime + cstime)
with open(log, "a", encoding="utf-8") as fh:
    fh.write(line)
print(line, end="")
