#!/usr/bin/env python3
"""K2's extra rung, leaf by leaf (notes/decoder/willow_prereg.md section 6; section 9 entries 10 and 11).

The orchestrator's decision of 2026-10-10 (board/main.md, [decision] post headed 10:23Z): run the extra rung
(k2_extra_rung.py, unchanged) on every failing d = 5 ladder leaf as soon as its ladder finishes, one named leaf at a
time, cheapest first; stop and report if the extra rungs' CPU would pass 16 core-hours in total (raised to 28 by the
orchestrator at about 12:16Z, after decoder's projection of about 22, and to 34 at 15:10Z); apply the kill rule only
once all 32 d = 5 leaves have their verdicts. Since the orchestrator's notice of 19:07Z (the queue finished; decoder may
use 4 processes), up to --parallel leaves run at a time, still started cheapest first. This script only schedules; every
verdict comes from k2_extra_rung.py.

Loop:
  1. List the finished d = 5 ladder leaves and their state, with k2_extra_rung.py's rules (re-implemented here so that
     this process stays small): "pass (ladder)"; the verdict in the leaf's extra_rung.json if it was made for the current
     ladder file (same md5); otherwise pending.
  2. When all --expect (32) leaves have a verdict, write <out>/k2_tally.json and exit 0. The tally counts the verdicts
     and applies section 6's kill rule: kill d = 5 ML if more than a quarter of the leaves still fail.
  3. CPU so far: per leaf, the CPU of the runs this script made (its ledger, kept in <out>/runner_ledger.json and in a
     local copy, --ledger), or, for a leaf run before this script, tau3 + check CPU from its extra_rung.json plus 30 s
     (set-up, assumed).
  4. Every k2_extra_rung.py process already running holds a slot, whoever started it (one left by an earlier runner
     is waited for, and its leaf is not started again). With a free slot (fewer than --parallel running), take the
     cheapest pending leaf that is not running: fewest rounds, then fewest band shots. Projected cost: band shots x
     rounds x the largest tau3 cost per round per band shot measured so far (at least 0.14 s; 0.12-0.22 s on the first
     9 leaves) + 60 s (the tau2 check and set-up). Committed: each running leaf at its projected cost, or its CPU so far
     + 60 s if that is larger. If CPU so far + committed + projected > --budget-hours (34), log "K2 budget reached",
     start nothing more, wait for the running leaves, write the tally so far and exit 3.
     Before that, the projection for all the extra rungs, by the method of decoder's board post headed 11:24:53Z: CPU
     so far; each pending leaf at the largest tau3 cost measured on its own patch (--assume-rate, 0.2 s, on a patch
     with none measured yet); each expected leaf whose ladder is not finished (the d = 5 patches in --root, X and Z,
     at the rounds of the finished ladder leaves) at the mean band shots of the finished leaves with the same rounds,
     assumed to need the extra rung; 60 s per leaf. It is logged, and if it is past --budget-hours the script logs
     "K2 projection past the budget", starts nothing more, waits for the running leaves, writes the tally so far and
     exits 6 (the orchestrator's rule: if the projection goes past the stop, stop before the next leaf and tell main).
     A running leaf counts at its estimate, or at its CPU so far + 60 s if that is larger.
  5. Start k2_extra_rung.py --leaf <leaf> as a child process (its output appended to <logs>/k2_<leaf tag>.out). When a
     child ends, record its CPU (rusage of the child) in the ledger. If it exited non-zero, start nothing more, wait
     for the others and exit 5.
  6. If nothing can start, look again when a running leaf ends (checked every 30 s) or, with nothing running and
     nothing pending, after --poll seconds (default 300).
A leaf whose rebuild check fails gets "not run: ..." from k2_extra_rung.py; it is listed in the tally as without a
verdict, and the script exits 4 once nothing else is pending.
A leaf cut off by a container restart has no extra_rung.json, so it is pending again and is rerun from its start; the
CPU of the cut-off run is not in the ledger.

Run (up to --parallel children at a time; see code/decoder/round2/RELAUNCH.md):
  NUMBA_CACHE_DIR=<local> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -I -B k2_runner.py --root <local extract> \
      --res results/decoder/willow --out results/decoder/willow/supplementary/k2_extra_rung \
      --logs <local log dir> --ledger <local ledger json> --budget-hours 34 --parallel 2
"""
import argparse, hashlib, json, os, subprocess, sys, time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEC_PER_ROUND_BAND_SHOT = 0.14
SETUP_S = 60.0
SETUP_S_OLD = 30.0


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def ladder_leaves(res):
    """As k2_extra_rung.ladder_leaves: every finished d = 5 ladder leaf (leaf rel, tag, decoder name, summary)."""
    out = []
    for dp, dn, fn in os.walk(os.path.join(res, "decode")):
        dn[:] = sorted(d for d in dn if not d.startswith(("d3_", "d7_")))
        if "summary.json" not in fn:
            continue
        try:
            s = json.load(open(os.path.join(dp, "summary.json")))
        except Exception:
            continue
        if s.get("status") != "complete" or int(s.get("meta", {}).get("distance", 0) or 0) != 5:
            continue
        for name, ds in (s.get("decoders") or {}).items():
            if name.startswith("ml_tau2") and isinstance(ds, dict) and "ladder" in ds:
                out.append((s["experiment"], s["dem_tag"], name, s))
    return sorted(out, key=lambda x: (x[0], x[1]))


def ladder_passes(lad, band):
    """As k2_extra_rung.ladder_rules."""
    e1, ef = lad["errors_tau1"], lad["errors_final"]
    return abs(ef - e1) <= max(3, 0.01 * ef) and lad["max_abs_dP_redecoded"] < band


def band_of(name):
    parts = name.split(":")
    return float(parts[2]) if len(parts) > 2 else 0.1


def states(a):
    rows, rates, by_patch = [], [SEC_PER_ROUND_BAND_SHOT], {}
    for rel, tag, name, summ in ladder_leaves(a.res):
        lad = summ["decoders"][name]["ladder"]
        row = {"leaf": rel, "tag": tag, "rounds": int(summ.get("rounds") or 0), "patch": rel.split("/")[-3]}
        zpath = os.path.join(a.res, "decode", rel, tag, name.replace(":", "_").replace("/", "-") + ".npz")
        row["n_band"] = int(np.load(zpath)["redecoded"].sum())
        if ladder_passes(lad, band_of(name)):
            row["verdict"] = "pass (ladder)"; rows.append(row); continue
        pj = os.path.join(a.out, rel, tag, "extra_rung.json")
        old = None
        if os.path.exists(pj):
            try:
                old = json.load(open(pj))
            except Exception:
                old = None
        if old and old.get("ladder_npz_md5") == md5_file(zpath):
            row["verdict"] = old["verdict"]
            row["cpu_old"] = float((old.get("tau3") or {}).get("cpu_seconds_tau3", 0.0)) + \
                float((old.get("check") or {}).get("cpu_seconds", 0.0)) + SETUP_S_OLD
            ms = (old.get("tau3") or {}).get("ms_per_round_per_band_shot_tau3")
            if ms:
                rates.append(float(ms) / 1e3)
                by_patch.setdefault(row["patch"], []).append(float(ms) / 1e3)
        else:
            row["verdict"] = "pending"
        rows.append(row)
    rate = max(rates)
    for row in rows:
        if row["verdict"] == "pending":
            row["projected_s"] = row["n_band"] * row["rounds"] * rate + SETUP_S
            row["rate_s"] = rate
    return rows, by_patch


def full_projection(a, rows, used, by_patch, floor=None):
    """Projection for all the extra rungs (see the module docstring, step 4); core-seconds and its parts. floor: leaf ->
    CPU so far + 60 s of a running leaf, which counts if it is larger than the leaf's estimate."""
    floor = floor or {}

    def prate(patch):
        return max(by_patch[patch]) if by_patch.get(patch) else a.assume_rate
    pend = [x for x in rows if x["verdict"] == "pending"]
    known = sum(max(x["n_band"] * x["rounds"] * prate(x["patch"]) + SETUP_S, floor.get(x["leaf"], 0.0)) for x in pend)
    done = {x["leaf"] for x in rows}
    rdirs = {x["leaf"].split("/")[-1]: x["rounds"] for x in rows}
    bands = {}
    for x in rows:
        bands.setdefault(x["rounds"], []).append(x["n_band"])
    unknown = []
    for exp in sorted({x["leaf"].split("/")[0] for x in rows}):
        base = os.path.join(a.root, exp)
        for patch in sorted(d for d in os.listdir(base) if d.startswith("d5_")):
            for basis in ("X", "Z"):
                for rdir, r in sorted(rdirs.items()):
                    leaf = f"{exp}/{patch}/{basis}/{rdir}"
                    if leaf not in done and os.path.isdir(os.path.join(base, patch, basis, rdir)):
                        unknown.append((leaf, patch, r))
    est = sum(float(np.mean(bands[r])) * r * prate(patch) + SETUP_S for leaf, patch, r in unknown)
    parts = {"used": used, "pending": known, "n_pending": len(pend), "not_laddered": est,
             "n_not_laddered": len(unknown), "n_expected": len(rows) + len(unknown),
             "rates": {k: max(v) for k, v in sorted(by_patch.items())}}
    return used + known + est, parts


def running_k2():
    """pids of k2_extra_rung.py processes now running (any parent)."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except Exception:
            continue
        if any(c.endswith(b"k2_extra_rung.py") for c in cmd):
            out.append(int(d))
    return out


def leaf_of(pid):
    """The --leaf argument of a running k2_extra_rung.py process (None if it is gone or has none)."""
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except Exception:
        return None
    for i in range(len(cmd) - 1):
        if cmd[i] == b"--leaf":
            return cmd[i + 1].decode()
    return None


def cpu_of(pid):
    """CPU seconds (user + system) of a running process so far (0 if it is gone)."""
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return 0.0


def load_ledger(a):
    runs = {}
    for p in (os.path.join(a.out, "runner_ledger.json"), a.ledger):
        try:
            for r in json.load(open(p))["runs"]:
                runs[(r["leaf"], r["started"])] = r
        except Exception:
            pass
    return sorted(runs.values(), key=lambda r: r["started"])


def save_ledger(a, runs):
    obj = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": runs}
    for p in (a.ledger, os.path.join(a.out, "runner_ledger.json")):
        write_json(p, obj)


def cpu_so_far(rows, runs):
    by_leaf = {}
    for r in runs:
        by_leaf[r["leaf"]] = by_leaf.get(r["leaf"], 0.0) + float(r["cpu_seconds"])
    tot = sum(by_leaf.values())
    tot += sum(x.get("cpu_old", 0.0) for x in rows if x["leaf"] not in by_leaf)
    return tot


def tally(a, rows, used, final):
    still = [x["leaf"] for x in rows if x["verdict"].startswith("still fails")]
    none = [x["leaf"] for x in rows if x["verdict"] == "pending" or x["verdict"].startswith("not run")]
    counts = {}
    for x in rows:
        counts[x["verdict"]] = counts.get(x["verdict"], 0) + 1
    t = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "prereg": "notes/decoder/willow_prereg.md section 6; section 9 entries 10 and 11",
         "expected_leaves": a.expect, "finished_ladder_leaves": len(rows), "verdicts": counts,
         "still_failing": still, "without_verdict": none, "cpu_core_hours_extra_rungs": used / 3600,
         "kill_rule": "kill d = 5 ML if more than a quarter of the d = 5 leaves still fail after the extra rung",
         "complete": final}
    if final:
        t["kill"] = len(still) > a.expect / 4
    write_json(os.path.join(a.out, "k2_tally.json"), t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local extract of the archive")
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", required=True, help="local folder for the children's output")
    ap.add_argument("--ledger", required=True, help="local copy of the ledger (json)")
    ap.add_argument("--expect", type=int, default=32)
    ap.add_argument("--budget-hours", type=float, default=34.0)
    ap.add_argument("--poll", type=float, default=300.0)
    ap.add_argument("--assume-rate", type=float, default=0.2,
                    help="tau3 s per round per band shot for a patch with none measured (projection only)")
    ap.add_argument("--parallel", type=int, default=1, help="leaves run at a time (the orchestrator, 19:07Z: up to 4 "
                    "processes for decoder in all)")
    a = ap.parse_args()
    os.makedirs(a.logs, exist_ok=True)
    logf = open(os.path.join(a.out, "k2_runner.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    budget = 3600.0 * a.budget_hours
    script = os.path.join(HERE, "k2_extra_rung.py")
    log(f"k2_runner (md5 {md5_file(os.path.abspath(__file__))}) with k2_extra_rung.py (md5 {md5_file(script)}); "
        f"budget {a.budget_hours} core-hours; expecting {a.expect} leaves; up to {a.parallel} at a time")
    kids = {}               # pid -> (Popen, row, start time) of the children this runner started
    stop = None             # (exit status, reason) once no new leaf may start
    idle, waiting, idle_busy = False, False, None
    while True:
        ended = False
        for pid in list(kids):                   # record the children that have ended
            try:
                rp, status, ru = os.wait4(pid, os.WNOHANG)
            except ChildProcessError:
                rp, status, ru = pid, None, None
            if rp == 0:
                continue
            p, x, started = kids.pop(pid)
            ended = True
            code = os.waitstatus_to_exitcode(status) if status is not None else -1
            p.returncode = code
            cpu = (ru.ru_utime + ru.ru_stime) if ru is not None else float("nan")
            runs = load_ledger(a)
            runs.append({"leaf": x["leaf"], "started": started, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "cpu_seconds": cpu, "exit": code, "projected_s": x["projected_s"], "n_band": x["n_band"],
                         "rounds": x["rounds"]})
            save_ledger(a, runs)
            log(f"finished {x['leaf']}: exit {code}, {cpu / 60:.1f} CPU min")
            if code != 0 and stop is None:
                stop = (5, f"k2_extra_rung.py exited {code} on {x['leaf']}")
                log(f"{stop[1]}. Stopping: no new leaf starts" + (f"; waiting for the {len(kids)} running" if kids else "")
                    + "; tell decoder.")
        busy = {pid: leaf_of(pid) for pid in running_k2()}
        for pid, kid in kids.items():            # this runner's own children, even before they show in /proc
            busy[pid] = kid[1]["leaf"]
        if stop is not None:
            if busy:
                time.sleep(30)
                continue
            rows, by_patch = states(a)
            tally(a, rows, cpu_so_far(rows, load_ledger(a)), final=False)
            log(f"no k2_extra_rung.py is running now; exit {stop[0]} ({stop[1]})")
            sys.exit(stop[0])
        if len(busy) >= a.parallel:
            if not waiting:
                log(f"{len(busy)} k2_extra_rung.py running (pid {sorted(busy)}: "
                    f"{', '.join('/'.join(str(v).split('/')[-3:]) for v in busy.values())}); waiting for a free slot "
                    f"(up to {a.parallel} at a time)")
            waiting = True
            time.sleep(30)
            continue
        waiting = False
        if idle and busy and set(busy) == idle_busy and not ended:
            time.sleep(30)                       # nothing to start until a running leaf ends
            continue
        rows, by_patch = states(a)
        runs = load_ledger(a)
        used = cpu_so_far(rows, runs)
        running = {lf: pid for pid, lf in busy.items() if lf}
        pend_all = [x for x in rows if x["verdict"] == "pending"]
        pend = sorted((x for x in pend_all if x["leaf"] not in running), key=lambda x: (x["rounds"], x["n_band"], x["leaf"]))
        problems = [x["leaf"] for x in rows if x["verdict"].startswith("not run")]
        if len(rows) >= a.expect and not pend_all and not busy:
            t = tally(a, rows, used, final=not problems)
            if problems:
                log(f"all {len(rows)} ladder leaves done, but without a verdict: {problems}. Tell decoder.")
                sys.exit(4)
            log(f"all {len(rows)} leaves have verdicts: {t['verdicts']}; still failing {len(t['still_failing'])}; "
                f"kill = {t['kill']}; extra rungs used {used / 3600:.2f} core-hours. Done.")
            return
        if not pend:
            if not idle:
                log(f"{len(rows)} of {a.expect} ladder leaves finished, none to start ({len(busy)} running); CPU so far "
                    f"{used / 3600:.2f} core-hours; looking again every {30 if busy else a.poll:.0f} s")
                tally(a, rows, used, final=False)
            idle, idle_busy = True, set(busy)
            time.sleep(30 if busy else a.poll)
            continue
        idle = False
        x = pend[0]
        run_rows = [r for r in pend_all if r["leaf"] in running]
        floor = {r["leaf"]: cpu_of(running[r["leaf"]]) + SETUP_S for r in run_rows}
        committed = sum(max(r["projected_s"], floor[r["leaf"]]) for r in run_rows)
        proj_all = sum(p["projected_s"] for p in pend)
        log(f"{len(rows)} of {a.expect} ladder leaves finished, {len(pend)} pending and {len(running)} running (pending "
            f"projected {proj_all / 3600:.2f} core-hours at {1e3 * x['rate_s']:.0f} ms per round per band shot; running "
            f"committed {committed / 3600:.2f}); CPU so far {used / 3600:.2f} core-hours")
        tot, pp = full_projection(a, rows, used, by_patch, floor)
        rtxt = ", ".join(f"{k[6:]} {1e3 * v:.0f}" for k, v in pp["rates"].items())
        log(f"projection for all the extra rungs: {tot / 3600:.1f} core-hours (budget {a.budget_hours}): used "
            f"{used / 3600:.2f}; {pp['n_pending']} pending or running {pp['pending'] / 3600:.1f}; {pp['n_not_laddered']} "
            f"ladders not finished {pp['not_laddered'] / 3600:.1f} (estimate); largest tau3 ms per round per band shot: "
            f"{rtxt}; assumed {1e3 * a.assume_rate:.0f} elsewhere"
            + ("" if pp["n_expected"] == a.expect else f"; WARNING: {pp['n_expected']} leaves expected, not {a.expect}"))
        if tot > budget:
            tally(a, rows, used, final=False)
            log(f"K2 projection past the budget: {tot / 3600:.1f} core-hours projected against {a.budget_hours}. "
                f"Stopping before {x['leaf']}; tell decoder, who tells @main."
                + (f" No new leaf starts; waiting for the {len(busy)} running." if busy else ""))
            stop = (6, "projection past the budget")
            continue
        if used + committed + x["projected_s"] > budget:
            tally(a, rows, used, final=False)
            log(f"K2 budget reached: {used / 3600:.2f} core-hours used, {committed / 3600:.2f} committed to the running "
                f"leaves, and {x['leaf']} would add about {x['projected_s'] / 3600:.2f} (budget {a.budget_hours}). "
                f"Stopping; tell decoder, who tells @main."
                + (f" No new leaf starts; waiting for the {len(busy)} running." if busy else ""))
            stop = (3, "budget reached")
            continue
        tagname = "_".join(x["leaf"].split("/")[-3:])
        log(f"running {x['leaf']}: r = {x['rounds']}, {x['n_band']} band shots, projected {x['projected_s'] / 60:.0f} "
            f"CPU min (slot {len(busy) + 1} of {a.parallel})")
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(a.logs, f"k2_{tagname}.out"), "a") as fo:
            p = subprocess.Popen([sys.executable, "-I", "-B", script, "--root", a.root, "--res", a.res,
                                  "--out", a.out, "--leaf", x["leaf"]], stdout=fo, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL)
        kids[p.pid] = (p, x, started)
        time.sleep(5)                            # until the child shows in /proc


if __name__ == "__main__":
    main()
