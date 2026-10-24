#!/usr/bin/env python3
"""willow_scout_cpu_since.py -- scout's compute in a later window, added to an earlier record of
willow_scout_cpu.py (S11's line for the text's builds and checks and the reproduction entry point's runs).
Scout, 2026-10-24 (draft 9.3).

Scout's commands are read from scout's own transcript (one JSON object per line), as in willow_scout_cpu.py: each
Bash command has a call time and a result time, and a background command runs from its launch to its output file's
last change. A command in the window [--start, --until) is classed as in willow_scout_cpu.py. Its CPU time is:

  * measured, if it ran under the wrapper `cpu_run.py` (code/scout/willow_cpu_run.py) and the wrapper logged it
    (the logs given with --cpu-log): the CPU of its process tree;
  * otherwise, for a run of the entry point, a build of the text or of the letter, or a check of the text, the
    measured CPU per command of its class (the mean of the wrapped steps of that class in the logs given), or, for a
    wrapped command that was stopped before the wrapper could log it, that mean scaled by its wall time over the
    class's mean wall time, at most the mean;
  * for "other" and "transcript", whose commands are too many to time, their wall time times a rate (--rate, CPU-s
    per second of wall time: the rate of the earlier record, from the kernel's counter, an upper bound).

The kernel's counter (the Claude Code process's reaped children, read by code/scout/willow_cpu_counter.py; given
with --counter as the boot time and readings ISO=SECONDS) holds every agent's commands since the container's last
restart, so it bounds the window's part after that restart from above, where the window has a reading at its end.
Steps still to come when the record is made (--remaining LABEL=CPU_S, --remaining-other-wall S) are added as
estimates. The earlier record (--base, its JSON) gives the compute before the window. Output: a JSON record and a
short report.

Usage: python3 -I willow_scout_cpu_since.py --transcript JSONL --tasks DIR --start ISO --until ISO
    --cpu-log TSV [TSV ...] --rate R --base JSON [--restart ISO --counter ISO=SECONDS ...]
    [--remaining LABEL=CPU_S ...] [--remaining-other-wall S] --json OUT --md OUT [--list TSV]
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

UTC = datetime.timezone.utc
WRAPPED = re.compile(r"python3 [^|;&]*cpu_run\.py")  # a step run under the CPU wrapper: measured
LABEL = re.compile(r"cpu_run\.py\s+\S+\s+(\"[^\"]*\"|'[^']*'|\S+)")  # the wrapper's LABEL argument
TRANSCRIPT = [""]
CLASSES = ("entry point, full", "entry point, quick", "build", "letter build", "checks")


def ts(s):
    return datetime.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def kind(cmd):
    """As willow_scout_cpu.py's."""
    if TRANSCRIPT[0] and TRANSCRIPT[0] in cmd:
        return "transcript"
    m = re.search(r"python3 [^|;&]*willow_reproduce\w*\.py[^|;&]*--out[^|;&]*", cmd)
    if m:
        return "entry point, quick" if "--skip" in m.group(0) else "entry point, full"
    if re.search(r"python3 [^|;&]*src/build\.py", cmd):
        return "build"
    if re.search(r"python3 [^|;&]*build_letter\.py", cmd):
        return "letter build"
    if re.search(r"python3 [^|;&]*(check_\w+|run_checks)\.py", cmd):
        return "checks"
    return "other"


def log_kind(label):
    """As willow_scout_cpu.py's classing of a wrapper log's label."""
    return ("entry point, quick" if "quick" in label else "entry point, full" if "run" in label else
            "letter build" if "letter" in label else "build" if "build" in label else
            "checks" if "check" in label else "transcript" if "record" in label else "other")


def read_commands(path, tasks):
    """As willow_scout_cpu.py's."""
    calls, results, bg = {}, {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            t = d.get("timestamp")
            c = (d.get("message") or {}).get("content")
            if not t or not isinstance(c, list):
                continue
            for x in c:
                if not isinstance(x, dict):
                    continue
                if x.get("type") == "tool_use" and x.get("name") == "Bash" and x.get("id") not in calls:
                    inp = x.get("input") or {}
                    calls[x["id"]] = (ts(t), inp.get("command", ""), bool(inp.get("run_in_background")))
                elif x.get("type") == "tool_result" and x.get("tool_use_id") not in results:
                    results[x.get("tool_use_id")] = ts(t)
                    s = x.get("content")
                    s = s if isinstance(s, str) else json.dumps(s)
                    m = re.match(r"\s*Command running in background with ID: (\w+)", s)
                    if m:
                        bg[x.get("tool_use_id")] = m.group(1)
    out = []
    for cid, (t0, cmd, isbg) in calls.items():
        if isbg:
            f = os.path.join(tasks, (bg.get(cid) or "?") + ".output")
            t1 = datetime.datetime.fromtimestamp(os.path.getmtime(f), UTC) if os.path.exists(f) else None
        else:
            t1 = results.get(cid)
        out.append({"t0": t0, "t1": t1, "kind": kind(cmd), "bg": isbg, "cmd": cmd})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--cpu-log", nargs="+", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--restart")
    ap.add_argument("--counter", nargs="*", default=[])
    ap.add_argument("--remaining", nargs="*", default=[])
    ap.add_argument("--remaining-other-wall", type=float, default=0.0)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--list")
    a = ap.parse_args()
    TRANSCRIPT[0] = os.path.basename(a.transcript)
    TS, TU = ts(a.start), ts(a.until)
    base = json.load(open(a.base, encoding="utf-8"))
    base_md5 = hashlib.md5(open(a.base, "rb").read()).hexdigest()

    cmds = [c for c in read_commands(a.transcript, a.tasks) if TS <= c["t0"] < TU and c["t1"] is not None]
    for c in cmds:
        c["wall"] = max(0.0, (c["t1"] - c["t0"]).total_seconds())
        c["wrapped"] = bool(WRAPPED.search(c["cmd"]))
        m = LABEL.search(c["cmd"]) if c["wrapped"] else None
        c["label"] = m.group(1).strip("\"'") if m else None

    # the wrapper's logs: the measured steps in the window
    logged = []
    for f in a.cpu_log:
        for line in open(f, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            t = ts(p[0])
            if not TS <= t < TU:
                continue
            logged.append({"label": p[1], "kind": log_kind(p[1]), "t": iso(t), "wall_s": float(p[2].split()[1]),
                           "cpu_s": float(p[5].split()[1]), "how": "cpu_run.py", "log": os.path.basename(
                               os.path.dirname(os.path.abspath(f))) + "/" + os.path.basename(f)})
    per_cmd, per_wall = {}, {}
    for k in CLASSES:
        xs = [m for m in logged if m["kind"] == k]
        if xs:
            per_cmd[k] = sum(m["cpu_s"] for m in xs) / len(xs)
            per_wall[k] = sum(m["wall_s"] for m in xs) / len(xs)
    base_per_cmd = base.get("cpu_per_command_s", {})
    for k in CLASSES:  # a class with no wrapped step in the window takes the earlier record's
        if k not in per_cmd and k in base_per_cmd:
            per_cmd[k] = base_per_cmd[k]

    # each command's CPU: measured (its log line), or estimated
    used_logs = set()
    rows = []
    for c in sorted(cmds, key=lambda c: c["t0"]):
        cpu, how = None, None
        if c["wrapped"] and c["label"]:
            for i, m in enumerate(logged):
                if i not in used_logs and m["label"] == c["label"] and abs((ts(m["t"]) - c["t0"]).total_seconds()) < 120:
                    used_logs.add(i)
                    cpu, how = m["cpu_s"], "measured"
                    break
        k = c["kind"]
        if cpu is None and c["wrapped"]:
            k = log_kind(c["label"] or "") if k == "other" else k
            if k in per_cmd:
                frac = min(1.0, c["wall"] / per_wall[k]) if per_wall.get(k) else 1.0
                cpu, how = per_cmd[k] * frac, "estimate (stopped before the wrapper logged it)"
        if cpu is None and k in per_cmd:
            cpu, how = per_cmd[k], "estimate (the class's measured CPU per command)"
        if cpu is None:
            cpu, how = c["wall"] * a.rate, "estimate (wall time at the rate)"
        rows.append({"t0": iso(c["t0"]), "wall_s": round(c["wall"], 1), "kind": k, "cpu_s": cpu, "how": how,
                     "cmd": " ".join(c["cmd"].split())[:300]})
    unmatched = [m for i, m in enumerate(logged) if i not in used_logs]
    if a.list:
        with open(a.list, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write("%s\t%.1f\t%s\t%.2f\t%s\t%s\n" % (r["t0"], r["wall_s"], r["kind"], r["cpu_s"], r["how"], r["cmd"]))

    by = {}
    for r in rows:
        key = "measured" if r["how"] == "measured" else r["how"]
        b = by.setdefault(r["kind"], {}).setdefault(key, {"n": 0, "wall_s": 0.0, "cpu_s": 0.0})
        b["n"] += 1
        b["wall_s"] += r["wall_s"]
        b["cpu_s"] += r["cpu_s"]
    measured_s = sum(r["cpu_s"] for r in rows if r["how"] == "measured")
    estimated_s = sum(r["cpu_s"] for r in rows if r["how"] != "measured")
    remaining = []
    for r in a.remaining:
        label, v = r.rsplit("=", 1)
        remaining.append({"label": label, "cpu_s": float(v)})
    if a.remaining_other_wall:
        remaining.append({"label": "other commands to the hand-back, about %.0f s of wall time at the rate"
                                   % a.remaining_other_wall, "cpu_s": round(a.remaining_other_wall * a.rate, 1)})
    rem = sum(r["cpu_s"] for r in remaining)
    window_s = measured_s + estimated_s + rem

    # the counter's bound on the part of the window after the restart
    bound = None
    if a.restart and a.counter:
        TR = ts(a.restart)
        reads = [(ts(x.split("=")[0]), float(x.split("=")[1])) for x in a.counter]
        tl, vl = max(reads)
        after = sum(r["cpu_s"] for r in rows if ts(r["t0"]) >= TR and ts(r["t0"]) < tl)
        bound = {"restart": a.restart, "reading": iso(tl), "children_s": vl,
                 "scout_after_restart_to_reading_s": round(after, 1), "at_most_counter": after <= vl}

    base_s = base["since_cutoff"]["cpu_s"]
    total_s = base_s + window_s
    rec = {
        "made": iso(datetime.datetime.now(UTC)),
        "script": "code/scout/willow_scout_cpu_since.py",
        "script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        "what": "scout's CPU time in a later window: builds and checks of the text, the letter, the reproduction entry "
                "point's runs and other commands (estimate; parts measured), added to the earlier record",
        "base": {"file": os.path.basename(a.base), "md5": base_md5, "cutoff": base["cutoff"],
                 "until": base["until"], "cpu_s": base_s, "core_hours": base["since_cutoff"]["core_hours"]},
        "start": a.start, "until": a.until,
        "commands_n": len(rows),
        "wall_s": round(sum(r["wall_s"] for r in rows), 1),
        "by_kind": {k: {h: {"n": v["n"], "wall_s": round(v["wall_s"], 1), "cpu_s": round(v["cpu_s"], 2)}
                        for h, v in d.items()} for k, d in by.items()},
        "measured_steps": logged,
        "logged_not_in_window_commands": unmatched,
        "cpu_per_command_s": {k: round(v, 2) for k, v in per_cmd.items()},
        "rate_other_cpu_per_wall_s": a.rate,
        "remaining": remaining,
        "window": {"measured_s": round(measured_s, 2), "estimated_s": round(estimated_s, 1),
                   "remaining_s": round(rem, 1), "cpu_s": round(window_s, 1),
                   "core_hours": round(window_s / 3600, 3), "core_hours_unrounded": window_s / 3600},
        "counter_bound": bound,
        "since_cutoff": {"cpu_s": round(total_s, 1), "core_hours": round(total_s / 3600, 2),
                         "core_hours_unrounded": total_s / 3600},
        "first_command": rows[0]["t0"] if rows else None,
        "last_command": rows[-1]["t0"] if rows else None,
    }
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    L = ["# Scout's compute from %s to %s" % (a.start, a.until), "",
         "Made %s by `code/scout/willow_scout_cpu_since.py` (md5 %s). An estimate, with parts measured: the CPU time "
         "of scout's commands from %s to %s, for S11's line on the text's builds and checks and the reproduction "
         "entry point's runs, added to the earlier record `%s` (md5 %s; %s to %s, %.1f CPU-s, %.2f core-hours)."
         % (rec["made"], rec["script_md5"], a.start, a.until, os.path.basename(a.base), base_md5, base["cutoff"],
            base["until"], base_s, base["since_cutoff"]["core_hours"]), "",
         "| class | how | commands | wall (s) | CPU (s) |", "|---|---|---|---|---|"]
    for k, d in by.items():
        for h, v in d.items():
            L.append("| %s | %s | %d | %.0f | %.2f |" % (k, h, v["n"], v["wall_s"], v["cpu_s"]))
    L += ["", "In the window: %.2f CPU-s measured, %.1f estimated, %.1f still to come (estimates): %.1f CPU-s, "
              "%.3f core-hours. With the earlier record: %.1f CPU-s, %.2f core-hours since %s."
          % (measured_s, estimated_s, rem, window_s, window_s / 3600, total_s, total_s / 3600, base["cutoff"]), "",
          "Measured CPU per command (the wrapped steps in the window, or the earlier record's): "
          + "; ".join("%s %.2f s" % kv for kv in rec["cpu_per_command_s"].items()) + ".", "",
          "Rate for other commands: %.4f CPU-s per second of wall time (the earlier record's, from the kernel's "
          "counter; an upper bound, since the counter holds every agent's commands)." % a.rate]
    if bound:
        L += ["", "The kernel's counter at %s, %.2f s for every agent's commands since the restart at %s, bounds "
                  "scout's estimate for that part of the window, %.1f s, from above: %s."
              % (bound["reading"], bound["children_s"], bound["restart"], bound["scout_after_restart_to_reading_s"],
                 bound["at_most_counter"])]
    L += ["", "| measured step | class | wall (s) | CPU (s) | log |", "|---|---|---|---|---|"]
    for m in logged:
        L.append("| %s | %s | %.1f | %.2f | %s |" % (m["label"], m["kind"], m["wall_s"], m["cpu_s"], m["log"]))
    if remaining:
        L += ["", "Still to come when this was made (estimates): " +
              "; ".join("%s %.1f s" % (r["label"], r["cpu_s"]) for r in remaining) + "."]
    with open(a.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(json.dumps({"window": rec["window"], "since_cutoff": rec["since_cutoff"], "per_cmd": rec["cpu_per_command_s"],
                      "by_kind": rec["by_kind"], "counter_bound": bound, "unmatched_logs": len(unmatched)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
