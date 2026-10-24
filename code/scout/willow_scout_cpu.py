#!/usr/bin/env python3
"""willow_scout_cpu.py -- scout's compute for the revision since a cut-off (S11's line for the text's builds and
checks and the reproduction entry point's runs). Scout, 2026-10-24.

Scout's commands are read from scout's own transcript (one JSON object per line), where each Bash command has a call
time and a result time; a background command runs from its launch to its output file's last change. Each command is
classed as an entry-point run (full, or quick with layers skipped), a build of the text, a build of the letter, a
check of the text, a read of the transcript itself, or other. The CPU time of a class is:

  * measured, where it was: the shell's `time` on two full runs of draft 9 (files given with --time), and the CPU of
    the process tree of each step that was run under the wrapper `cpu_run.py` (code/scout/willow_cpu_run.py; its log
    given with --cpu-log);
  * otherwise estimated, as the class's count times its measured CPU per command (the mean of the measured ones);
  * for "other" and "transcript", whose commands are too many to time, as their wall time times a rate: the CPU of the
    container's reaped children between two readings of the kernel's counter (--counter), less the wrapped steps in
    between, per second of scout's other commands' wall time in between. The counter holds every agent's commands, so
    the rate is an upper bound for scout's.

The kernel's counter itself (the Claude Code process's reaped children since the container's restart, read by
code/scout/willow_cpu_counter.py) bounds the total after the restart from above. Steps still to come when the record
is made (--remaining LABEL=CPU_S) are added as estimates. Output: a JSON record and a short report.

Usage: python3 -I willow_scout_cpu.py --transcript JSONL --tasks DIR --cutoff ISO --restart ISO --until ISO
    --time FILE... --cpu-log TSV --counter ISO=SECONDS ISO=SECONDS [--remaining LABEL=CPU_S ...] --json OUT --md OUT
    [--list TSV]
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
TRANSCRIPT = [""]  # the transcript's file name, set in main()


def ts(s):
    return datetime.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def kind(cmd):
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


def read_commands(path, tasks):
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
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--time", nargs="+", default=[])
    ap.add_argument("--cpu-log", required=True)
    ap.add_argument("--counter", nargs=2, required=True)
    ap.add_argument("--remaining", nargs="*", default=[])
    ap.add_argument("--remaining-other-wall", type=float, default=0.0,
                    help="seconds of wall time of other commands still to come, counted at the rate")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--list", help="also write every command in the windows, with its class, to this TSV")
    a = ap.parse_args()
    TRANSCRIPT[0] = os.path.basename(a.transcript)
    T0, TR, TU = ts(a.cutoff), ts(a.restart), ts(a.until)
    (c0t, c0v), (c1t, c1v) = (x.split("=") for x in a.counter)
    C0, C1, c0v, c1v = ts(c0t), ts(c1t), float(c0v), float(c1v)

    cmds = [c for c in read_commands(a.transcript, a.tasks) if T0 <= c["t0"] < TU and c["t1"] is not None]
    if a.list:
        with open(a.list, "w", encoding="utf-8") as fh:
            for c in sorted(cmds, key=lambda c: c["t0"]):
                fh.write("%s\t%.1f\t%s\t%s\t%s\n" % (iso(c["t0"]), (c["t1"] - c["t0"]).total_seconds(), c["kind"],
                                                      "bg" if c["bg"] else "fg", " ".join(c["cmd"].split())[:300]))
    windows = {"before_restart": (T0, TR), "after_restart": (TR, C0), "draft_9_1b": (C0, TU)}
    tab = {}
    for w, (lo, hi) in windows.items():
        tab[w] = {}
        for c in cmds:
            if lo <= c["t0"] < hi:
                k = tab[w].setdefault(c["kind"], {"n": 0, "wall_s": 0.0})
                k["n"] += 1
                k["wall_s"] += max(0.0, (c["t1"] - c["t0"]).total_seconds())

    # measured CPU: the shell's time on draft 9's two full runs, and the wrapped steps of draft 9.1b
    measured = []
    for f in a.time:
        txt = open(f, encoding="utf-8").read()
        u = re.search(r"user\s+(\d+)m([\d.]+)s", txt)
        s = re.search(r"sys\s+(\d+)m([\d.]+)s", txt)
        r = re.search(r"real\s+(\d+)m([\d.]+)s", txt)
        cpu = 60 * int(u.group(1)) + float(u.group(2)) + 60 * int(s.group(1)) + float(s.group(2))
        measured.append({"label": "draft 9, full run (%s)" % os.path.basename(f), "kind": "entry point, full",
                         "wall_s": round(60 * int(r.group(1)) + float(r.group(2)), 1), "cpu_s": round(cpu, 2),
                         "how": "the shell's time"})
    wrapped = []
    for line in open(a.cpu_log, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        t, label = ts(p[0]), p[1]
        wall, cpu = float(p[2].split()[1]), float(p[5].split()[1])
        k = ("entry point, quick" if "quick" in label else "entry point, full" if "run" in label else
             "letter build" if "letter" in label else "build" if "build" in label else
             "checks" if "check" in label else "transcript" if "record" in label else "other")
        wrapped.append({"label": label, "kind": k, "t": iso(t), "wall_s": wall, "cpu_s": cpu, "how": "cpu_run.py"})
    measured += wrapped
    per_cmd = {}
    for k in ("entry point, full", "entry point, quick", "build", "letter build", "checks"):
        xs = [m["cpu_s"] for m in measured if m["kind"] == k]
        if xs:
            per_cmd[k] = sum(xs) / len(xs)

    # the rate for "other" and "transcript" commands, from the counter between C0 and C1
    wrapped_in = sum(m["cpu_s"] for m in wrapped if C0 <= ts(m["t"]) < C1)
    # (a command run under cpu_run.py is measured, so its wall time is left out of the rate's sums)
    wall_rest = sum(max(0.0, (c["t1"] - c["t0"]).total_seconds()) for c in cmds
                    if C0 <= c["t0"] < C1 and c["kind"] in ("other", "transcript") and not WRAPPED.search(c["cmd"]))
    rate = max(0.0, (c1v - c0v) - wrapped_in) / wall_rest if wall_rest > 0 else 0.0

    est = {}
    for w in ("before_restart", "after_restart"):
        cpu = 0.0
        detail = {}
        for k, v in tab[w].items():
            if k in per_cmd:
                x = v["n"] * per_cmd[k]
                if w == "after_restart" and k == "entry point, full":
                    # draft 9's two full runs were measured; the others take the mean
                    nm = sum(1 for m in measured if m["how"] == "the shell's time")
                    x = sum(m["cpu_s"] for m in measured if m["how"] == "the shell's time") + (v["n"] - nm) * per_cmd[k]
            else:
                x = v["wall_s"] * rate
            detail[k] = round(x, 1)
            cpu += x
        est[w] = {"cpu_s": round(cpu, 1), "by_kind_cpu_s": detail}
    # draft 9.1b: the wrapped steps, the counter's increase for the rest up to C1, the commands after C1 at the rate,
    # and the steps still to come
    after_c1 = sum(max(0.0, (c["t1"] - c["t0"]).total_seconds()) for c in cmds
                   if c["t0"] >= C1 and c["kind"] in ("other", "transcript") and not WRAPPED.search(c["cmd"]))
    wrapped_after = sum(m["cpu_s"] for m in wrapped if ts(m["t"]) >= C1)
    remaining = []
    for r in a.remaining:
        label, v = r.rsplit("=", 1)
        remaining.append({"label": label, "cpu_s": float(v)})
    if a.remaining_other_wall:
        remaining.append({"label": "other commands to the hand-back, about %.0f s of wall time at the rate"
                                   % a.remaining_other_wall, "cpu_s": round(a.remaining_other_wall * rate, 1)})
    rem = sum(r["cpu_s"] for r in remaining)
    d91b = (c1v - c0v) + wrapped_after + after_c1 * rate + rem
    est["draft_9_1b"] = {"cpu_s": round(d91b, 1), "counter_increase_s": round(c1v - c0v, 2),
                         "wrapped_after_C1_s": round(wrapped_after, 2), "other_after_C1_s": round(after_c1 * rate, 1),
                         "remaining_s": round(rem, 1)}
    total_s = sum(e["cpu_s"] for e in est.values())
    rec = {
        "made": iso(datetime.datetime.now(UTC)),
        "script": "code/scout/willow_scout_cpu.py",
        "script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        "what": "scout's CPU time since the cut-off: builds and checks of the text, the letter, the reproduction "
                "entry point's runs and other commands (estimate; parts measured)",
        "cutoff": a.cutoff, "container_restart": a.restart, "until": a.until,
        "commands": {w: {k: {"n": v["n"], "wall_s": round(v["wall_s"], 1)} for k, v in tab[w].items()} for w in tab},
        "commands_n": {w: sum(v["n"] for v in tab[w].values()) for w in tab},
        "wall_s": {w: round(sum(v["wall_s"] for v in tab[w].values()), 1) for w in tab},
        "measured": measured,
        "cpu_per_command_s": {k: round(v, 2) for k, v in per_cmd.items()},
        "counter": {"what": "the Claude Code process's reaped children (utime + stime), every agent's commands in the "
                            "container since its restart; an upper bound for scout's",
                    "readings": [{"t": c0t, "children_s": c0v}, {"t": c1t, "children_s": c1v}]},
        "rate_other_cpu_per_wall_s": round(rate, 4),
        "remaining": remaining,
        "estimate_cpu_s": est,
        "since_cutoff": {"cpu_s": round(total_s, 1), "core_hours": round(total_s / 3600, 2),
                         "core_hours_unrounded": total_s / 3600},
        "check_after_restart_at_most_counter": est["after_restart"]["cpu_s"] <= c0v,
        "first_command_after_cutoff": iso(min(c["t0"] for c in cmds)) if cmds else None,
    }
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    L = ["# Scout's compute since %s" % a.cutoff, "",
         "Made %s by `code/scout/willow_scout_cpu.py` (md5 %s). An estimate, with parts measured: the CPU time of "
         "scout's commands from %s to %s, for S11's line on the text's builds and checks and the reproduction entry "
         "point's runs. Scout's first command after the cut-off was at %s." % (
             rec["made"], rec["script_md5"], a.cutoff, a.until, rec["first_command_after_cutoff"]), "",
         "| window | commands | wall (s) | CPU (s), estimate |", "|---|---|---|---|"]
    names = {"before_restart": "to the container's restart, %s" % a.restart,
             "after_restart": "from the restart to the first counter reading, %s" % c0t,
             "draft_9_1b": "from %s, with the steps still to come" % c0t}
    for w in ("before_restart", "after_restart", "draft_9_1b"):
        L.append("| %s | %d | %.0f | %.1f |" % (names[w], rec["commands_n"][w], rec["wall_s"][w], est[w]["cpu_s"]))
    L += ["", "Since the cut-off: %.1f CPU-s, %.2f core-hours." % (total_s, total_s / 3600), "",
          "Measured CPU per command: " + "; ".join("%s %.2f s" % kv for kv in rec["cpu_per_command_s"].items()) + ".",
          "",
          "Rate for other commands: %.4f CPU-s per second of wall time (the counter's increase from %s to %s, %.2f s, "
          "less %.2f s of measured steps, over %.0f s of scout's other commands; an upper bound, since the counter "
          "holds every agent's commands)." % (rate, c0t, c1t, c1v - c0v, wrapped_in, wall_rest), "",
          "The kernel's counter at %s, %.2f s for every agent's commands since the restart, bounds the after-restart "
          "estimate, %.1f s, from above: %s." % (c0t, c0v, est["after_restart"]["cpu_s"],
                                                 rec["check_after_restart_at_most_counter"]), "",
          "| measured step | class | wall (s) | CPU (s) | how |", "|---|---|---|---|---|"]
    for m in measured:
        L.append("| %s | %s | %.1f | %.2f | %s |" % (m["label"], m["kind"], m["wall_s"], m["cpu_s"], m["how"]))
    if remaining:
        L += ["", "Still to come when this was made (estimates): " +
              "; ".join("%s %.1f s" % (r["label"], r["cpu_s"]) for r in remaining) + "."]
    with open(a.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(json.dumps({"since_cutoff": rec["since_cutoff"], "estimate_cpu_s": est, "rate": rate,
                      "per_cmd": rec["cpu_per_command_s"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
