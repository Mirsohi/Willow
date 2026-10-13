"""Synthetic check of pr3_r10_tau4.py's scheduler, rules and tally (amendment 10a). The r = 10 driver runs first with
fake units (fake_full_decode_willow.py), so the tally and the ledger are the driver's own; then the tau4 scheduler
runs with fake tau4 parts (fake_tau4_worker.py) in place of run_rung().
The machine may be running the real r = 10 driver: both schedulers' heavy-process scans are limited to this test's
processes, and driver_alive() is stubbed except in scenario 8.
  1. X still fails after tau3, tau4 = tau3: "pass (tau4 rung)"; the rule values equal a direct computation; k3 equals
     the r = 10 tally's; the projection is the driver's CPU + 5.5 x the extra parts' CPU; exit 0, final. A rerun
     starts nothing. A part lost in a restart is rerun alone, its seen CPU booked lost.
  2. tau4 moves every band shot: "still fails after tau4".
  3. Both leaves still fail and the cap is below the projection: nothing starts, exit 6, both "tau4 incomplete".
  4. The parts burn CPU past the cap: all are killed, exit 6, "tau4 incomplete", final.
  5. A part passes the RSS cap (one slot): exit 7, the leaf's other parts skipped, final.
  6. A part crashes (one slot): exit 5, no new start, not final; a rerun retries it and passes.
  7. No leaf still fails: "not needed", exit 0.
  8. The driver still runs: exit 2, nothing written. 9. An earlier tau4 part is alive: exit 2.
  10. A tau3 check fails (one slot): exit 4, the leaf's other parts skipped, final.
  11. Another heavy process holds the only slot: the parts wait for it.
usage: PR3_CODE=<code/decoder> python3 -I -B test_r10_tau4.py <code/decoder/pr3> <tau4 script folder> <scratch>"""
import json, os, shutil, subprocess, sys, tempfile, time, types
CODE_PR3, T4CODE, SCR = sys.argv[1:4]
sys.path.insert(0, CODE_PR3)
import numpy as np
import pr3_r10_posthoc as D
C, PRR = D.C, D.PRR
sys.path.insert(0, T4CODE)
import pr3_r10_tau4 as T4
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_full_decode_willow.py")
FAKE4 = os.path.join(HERE, "fake_tau4_worker.py")
LX, LZ = C.leaf("X", 10), C.leaf("Z", 10)
REAL_DRIVER_ALIVE = T4.driver_alive
T4.driver_alive = lambda: False
# The frozen runner drops its Popen objects and reaps its units with os.wait4. subprocess's own _cleanup(), run at
# every new Popen(), can reap a dropped unit first when the unit has already ended; the runner's wait4 then raises
# ChildProcessError. Fake units end within milliseconds, so a second driver run in one process hits this; real units
# run for tens of minutes. The test turns _cleanup() off (run_capped() keeps its Popen object and needs none).
subprocess._cleanup = lambda: None
ok = []


def check(cond, what):
    ok.append(bool(cond)); print(("ok   " if cond else "FAIL ") + what, flush=True)


def only(T, scan):
    """The scan, limited to processes whose arguments name the test folder T."""
    key = T.encode()
    return lambda me: [p for p in scan(me) if any(key in c for c in p["cmd"])]


def run_driver(modes):
    T = tempfile.mkdtemp(dir=SCR)
    pilot = {"measured_rates": {"c1_10": 43.0072098317, "s_10": 0.005, "c2_10": 383.1532430062512, "c1_50": 51.2932625268,
                                "cb_50": 1.047663489999997, "s_50": 0.02, "c2_50": 325.92149488, "c3_50": 1035.233558995,
                                "ovh_extra_s": 4.195043083999984, "ovh_ladder_s": 0.0},
             "stage1_ladder": {"peak_rss": 536133632}, "stage2_extra": {"peak_rss": 349929472}}
    json.dump(pilot, open(os.path.join(T, "pilot.json"), "w"))
    a = types.SimpleNamespace(root="/nonexistent", res=os.path.join(T, "res"), project="/nonexistent",
                              work=os.path.join(T, "w"), state=os.path.join(T, "s"), logs=os.path.join(T, "l"),
                              pilot=os.path.join(T, "pilot.json"), design="25k", guard_core_hours=13.0, max_heavy=4,
                              min_free_gib=3, cap_gib=4, poll=0.3, dry_run=False)
    R = D.R10(a)
    R.procs.scan = only(T, R.procs.scan)

    def cmd(u):
        lf, j = u["leaf"], u["part"]
        d = R.ladder_dir(lf, j) if u["kind"] == "ladder" else os.path.dirname(R.extra_json(lf, j))
        arg = {"kind": u["kind"], "dir": d, "leaf": lf, "part": j, "slice": C.part_slice("25k", j),
               "ladder_npz": os.path.join(R.ladder_dir(lf, j), PRR.LADDER_DN + ".npz"), "modes": modes}
        return [sys.executable, "-I", "-B", FAKE, json.dumps(arg)]
    R.cmd = cmd
    code = R.run()
    assert code == 0, code
    return T, a


def copy_of(base, a):
    T = tempfile.mkdtemp(dir=SCR); shutil.rmtree(T); shutil.copytree(base, T)
    return T, types.SimpleNamespace(**dict(vars(a), res=os.path.join(T, "res"), state=os.path.join(T, "s")))


def tau4(T, a, mode, guard=13.0, cap_gib=4.0, seconds=20, max_heavy=4):
    b = types.SimpleNamespace(root="/nonexistent", res=a.res, ledger=os.path.join(a.state, "ledger.json"),
                              work=os.path.join(T, "w4"), state=os.path.join(T, "s4"), logs=os.path.join(T, "l4"),
                              design="25k", guard_core_hours=guard, max_heavy=max_heavy, min_free_gib=3.0,
                              cap_gib=cap_gib, poll=0.2, dry_run=False, worker=None)
    S = T4.Tau4(b)
    S.procs.scan = only(T, S.procs.scan)
    starts, logs = [], []

    def log(m):
        logs.append(m); print("    " + m, flush=True)
    S.log = log

    def cmd(lf, j):
        starts.append((lf, j))
        return [sys.executable, "-I", "-B", FAKE4, json.dumps({"odir": S.odir(lf, j), "edir": S.edir(lf, j), "mode": mode,
                                                              "part": j, "seconds": seconds})]
    S.cmd = cmd
    code = S.run()
    tl = os.path.join(a.res, "r10_tau4_tally.json")
    return code, (json.load(open(tl)) if os.path.exists(tl) else None), starts, logs


def direct(res, lf):
    idx, obs, df = [], [], []
    b3, l3, l4 = [], [], []
    for j in range(5):
        dd = os.path.join(res, "parts", f"p{j}", "decode", lf, "rl_25k")
        z = np.load(os.path.join(dd, PRR.LADDER_DN + ".npz")); sh = np.load(os.path.join(dd, "shots.npz"))
        idx.append(z["idx"]); obs.append(sh["obs"]); df.append(z["pred"][:, 0])
        e = np.load(os.path.join(res, "parts", f"p{j}", "extra_rung", lf, "rl_25k", "extra_rung.npz"))
        f = np.load(os.path.join(res, "parts", f"p{j}", "tau4_rung", lf, "rl_25k", "tau4_rung.npz"))
        b3.append(e["idx"]); l3.append(e["llr_t3"]); l4.append(f["llr_t4"])
    idx, obs, df = np.concatenate(idx), np.concatenate(obs), np.concatenate(df)
    b3, l3, l4 = np.concatenate(b3), np.concatenate(l3), np.concatenate(l4)
    final3, final4 = df.copy(), df.copy()
    pos = {int(x): i for i, x in enumerate(idx)}
    bi = np.array([pos[int(x)] for x in b3])
    final3[bi] = (l3 > 0); final4[bi] = (l4 > 0)
    m = idx % 25 != 0
    k3, k4 = int(((final3 != obs) & m).sum()), int(((final4 != obs) & m).sum())
    ch = int(((l3 > 0) != (l4 > 0))[m[bi]].sum())
    return k3, k4, ch


def ext_h(a, lf):
    L = json.load(open(os.path.join(a.state, "ledger.json")))["units"]
    return sum((L.get(f"extra|{lf}|{j}") or {}).get("cpu_s", 0) for j in range(5)) / 3600


def drv_h(a):
    L = json.load(open(os.path.join(a.state, "ledger.json")))["units"]
    return sum((x.get("cpu_s") or 0) + (x.get("lost_cpu_s") or 0) for x in L.values()) / 3600


# ---- 1: X still fails after tau3; tau4 = tau3
T, a = run_driver({"X": "fail", "Z": "pass"})
Tl = json.load(open(os.path.join(a.res, "r10_tally.json")))
check(Tl["complete"] and Tl["still_failing"] == [LX], f"driver: X still fails after tau3, Z passes ({Tl['still_failing']})")
base = os.path.join(SCR, "base_xfail"); shutil.rmtree(base, ignore_errors=True); shutil.copytree(T, base)
code, t, starts, _ = tau4(T, a, "pass")
v = t["verdicts"][LX]
k3, k4, ch = direct(a.res, LX)
check(code == 0 and t["final"] and v["verdict"] == "pass (tau4 rung)" and len(starts) == 5 and set(t["verdicts"]) == {LX},
      f"1: exit {code}, {v['verdict']}, {len(starts)} parts started, final {t['final']}")
check((v["errors_tau3"], v["errors_tau4"], v["decisions_changed"]) == (k3, k4, ch) and v["errors_tau3_equal_r10_tally"]
      and abs(v["limit"] - max(3, 0.01 * k4)) < 1e-12 and v["tau3_check"]["identical_in_every_part"],
      f"1: k3 {v['errors_tau3']} = {k3}, k4 {v['errors_tau4']} = {k4}, changed {v['decisions_changed']} = {ch}; k3 equals the "
      f"tally's; tau3 check {v['tau3_check']}")
P = t["projection"]
check(abs(P["per_leaf_h"][LX] - 5.5 * ext_h(a, LX)) < 1e-12 and abs(P["driver_h"] - drv_h(a)) < 1e-12
      and abs(t["total_cpu_core_h"] - (drv_h(a) + t["tau4_cpu_core_h"])) < 1e-12,
      f"1: projection: driver {P['driver_h'] * 3600:.1f} s, X {P['per_leaf_h'][LX] * 3600:.1f} s = 5.5 x its extra parts")
code1b, t1b, starts1b, _ = tau4(T, a, "pass")
check(code1b == 0 and not starts1b and t1b["verdicts"][LX]["verdict"] == "pass (tau4 rung)", "1: a rerun starts nothing")
L4 = json.load(open(os.path.join(T, "s4", "tau4_ledger.json")))
k = f"tau4|{LX}|2"
L4["units"][k]["state"] = "running"; L4["units"][k]["seen_cpu_s"] = 5.0
json.dump(L4, open(os.path.join(T, "s4", "tau4_ledger.json"), "w"))
os.remove(os.path.join(a.res, "parts", "p2", "tau4_rung", LX, "rl_25k", "tau4_rung.json"))
code1c, t1c, starts1c, _ = tau4(T, a, "pass")
L4b = json.load(open(os.path.join(T, "s4", "tau4_ledger.json")))
check(code1c == 0 and starts1c == [(LX, 2)] and abs(L4b["units"][k]["lost_cpu_s"] - 5.0) < 1e-9
      and t1c["verdicts"][LX]["verdict"] == "pass (tau4 rung)",
      f"1: a part lost in a restart is rerun alone ({starts1c}), its seen CPU booked lost ({L4b['units'][k]['lost_cpu_s']} s)")

# ---- 2: tau4 moves the band
T2, a2 = copy_of(base, a)
code, t, starts, _ = tau4(T2, a2, "fail")
v = t["verdicts"][LX]
k3, k4, ch = direct(a2.res, LX)
check(code == 0 and v["verdict"].startswith("still fails after tau4") and (v["errors_tau4"], v["decisions_changed"]) == (k4, ch)
      and ch > v["limit"], f"2: exit {code}, {v['verdict']}, changed {ch} > limit {v['limit']:.2f}")

# ---- 3: both still fail, cap below the projection
T3, a3 = run_driver({"X": "fail", "Z": "fail"})
code, t, starts, _ = tau4(T3, a3, "pass", guard=1e-6)
check(code == 6 and not starts and t["status"].startswith("not started") and t["final"]
      and all(t["verdicts"][lf]["verdict"] == "tau4 incomplete: leaf rows only" for lf in (LX, LZ))
      and not os.path.exists(os.path.join(a3.res, "parts", "p0", "tau4_rung")),
      f"3: exit {code}, {t['status']!r}, nothing started, both incomplete")

# ---- 4: CPU past the cap while running
T4d, a4 = copy_of(base, a)
guard = drv_h(a4) + 5.5 * ext_h(a4, LX) + 12 / 3600
t0 = time.time()
code, t, starts, _ = tau4(T4d, a4, "burn", guard=guard, seconds=30)
check(code == 6 and t["status"].startswith("stopped: the cap") and t["verdicts"][LX]["verdict"] == "tau4 incomplete: leaf rows only"
      and t["final"] and time.time() - t0 < 25 and t["total_cpu_core_h"] < guard + 8 / 3600,
      f"4: exit {code}, {t['status']!r}, stopped after {time.time() - t0:.1f} s, total {t['total_cpu_core_h'] * 3600:.1f} s "
      f"against a cap of {guard * 3600:.1f} s")
code4b, t4b, starts4b, _ = tau4(T4d, a4, "pass", guard=guard)
check(code4b == 6 and not starts4b and t4b["status"].startswith("not started") and t4b["final"],
      f"4: a rerun under the same cap starts nothing: exit {code4b}, {t4b['status']!r}")

# ---- 5: memory cap, one slot
T5, a5 = copy_of(base, a)
code, t, starts, _ = tau4(T5, a5, "mem", cap_gib=0.3, max_heavy=1)
ps = t["verdicts"][LX].get("parts", {})
check(code == 7 and len(starts) == 1 and t["final"] and t["verdicts"][LX]["verdict"] == "tau4 incomplete: leaf rows only"
      and list(ps.values()).count("failed: memory cap") == 1 and sum(s.startswith("skipped") for s in ps.values()) == 4,
      f"5: mem: exit {code}, {len(starts)} part started, parts {ps}")

# ---- 6: a crash, one slot; a rerun retries
T6, a6 = copy_of(base, a)
code, t, starts, _ = tau4(T6, a6, "crash", max_heavy=1)
check(code == 5 and len(starts) == 1 and not t["final"] and t["verdicts"][LX]["verdict"] == "tau4 incomplete: leaf rows only",
      f"6: crash: exit {code}, {len(starts)} part started, final {t['final']}")
code6b, t6b, starts6b, _ = tau4(T6, a6, "pass")
check(code6b == 0 and len(starts6b) == 5 and t6b["final"] and t6b["verdicts"][LX]["verdict"] == "pass (tau4 rung)",
      f"6: the rerun retries: exit {code6b}, {len(starts6b)} parts, {t6b['verdicts'][LX]['verdict']}")

# ---- 7: nothing to do
T7, a7 = run_driver({"X": "extra", "Z": "pass"})
code, t, starts, _ = tau4(T7, a7, "pass")
check(code == 0 and not starts and t["status"].startswith("not needed") and t["verdicts"] == {} and t["final"],
      f"7: exit {code}, {t['status']!r}")

# ---- 8: the driver still runs (the real check)
T8, a8 = copy_of(base, a)
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "pr3_r10_posthoc.py"])
time.sleep(0.5)
T4.driver_alive = REAL_DRIVER_ALIVE
code, t, starts, _ = tau4(T8, a8, "pass")
T4.driver_alive = lambda: False
alive_then = p.pid in T4.python_procs(b"pr3_r10_posthoc.py")
p.kill(); p.wait()
check(code == 2 and not starts and t is None and alive_then and p.pid not in T4.python_procs(b"pr3_r10_posthoc.py"),
      f"8: the driver still runs: exit {code}, nothing started or written; the stand-in was seen, and is gone after the kill")

# ---- 9: an earlier tau4 part is alive
T9, a9 = copy_of(base, a)
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "/x/pr3_r10_tau4.py", "--worker", LX, "0"])
time.sleep(0.5)
code, t, starts, logs = tau4(T9, a9, "pass")
p.kill(); p.wait()
check(code == 2 and not starts and t is None and any(str(p.pid) in m for m in logs),
      f"9: an earlier tau4 part (pid {p.pid}) is alive: exit {code}, nothing started")

# ---- 10: a tau3 check fails, one slot
T10, a10 = copy_of(base, a)
code, t, starts, _ = tau4(T10, a10, "checkfail", max_heavy=1)
ps = t["verdicts"][LX].get("parts", {})
check(code == 4 and len(starts) == 1 and t["final"] and list(ps.values()).count("failed: tau3 check") == 1
      and sum(s.startswith("skipped") for s in ps.values()) == 4,
      f"10: tau3 check fails: exit {code}, {len(starts)} part started, parts {ps}")

# ---- 11: another heavy process holds the only slot
T11, a11 = copy_of(base, a)
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(6)", os.path.join(T11, "dummy_decode_willow.py")])
t0 = time.time()
code, t, starts, logs = tau4(T11, a11, "pass", max_heavy=1)
el = time.time() - t0
p.wait()
check(code == 0 and t["verdicts"][LX]["verdict"] == "pass (tau4 rung)" and el > 5
      and any(m.startswith("waiting: 0 tau4 parts and 1 other heavy") for m in logs),
      f"11: the parts waited for another heavy process: exit {code}, {el:.1f} s")
print(f"{sum(ok)} of {len(ok)} checks pass")
sys.exit(0 if all(ok) else 1)
