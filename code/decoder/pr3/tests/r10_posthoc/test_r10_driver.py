"""Synthetic end-to-end check of pr3_r10_posthoc.py: the frozen runner's loop, with fake units in place of
decode_willow.py and pr3_extra_rung.py. X/r10 is built to fail the ladder rules and pass the extra rung, Z/r10 to pass
the ladder. Checks: only the two r = 10 leaves are scheduled; no slot is held; at most 4 units run at once; the rule
values equal a direct computation on the PR3 sample (probe shots left out); extra parts run for X/r10 only;
r10_tally.json is complete, post hoc and has no kill field; exit 0. Then a guard below the projection starts nothing
and exits 6.
usage: python3 -I -B test_r10_driver.py <code/decoder/pr3 folder> <scratch folder>"""
import json, os, sys, tempfile, types
CODE_PR3, SCR = sys.argv[1], sys.argv[2]
sys.path.insert(0, CODE_PR3)
import numpy as np
import pr3_r10_posthoc as D
C, PRR = D.C, D.PRR
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_decode_willow.py")
ok = []


def check(cond, what):
    ok.append(bool(cond)); print(("ok   " if cond else "FAIL ") + what)


def make(guard):
    T = tempfile.mkdtemp(dir=SCR)
    pilot = {"measured_rates": {"c1_10": 43.0072098317, "s_10": 0.005, "c2_10": 383.1532430062512, "c1_50": 51.2932625268,
                                "cb_50": 1.047663489999997, "s_50": 0.02, "c2_50": 325.92149488, "c3_50": 1035.233558995,
                                "ovh_extra_s": 4.195043083999984, "ovh_ladder_s": 0.0},
             "stage1_ladder": {"peak_rss": 536133632}, "stage2_extra": {"peak_rss": 349929472}}
    json.dump(pilot, open(os.path.join(T, "pilot.json"), "w"))
    a = types.SimpleNamespace(root="/nonexistent", res=os.path.join(T, "res"), project="/nonexistent",
                              work=os.path.join(T, "w"), state=os.path.join(T, "s"), logs=os.path.join(T, "l"),
                              pilot=os.path.join(T, "pilot.json"), design="25k", guard_core_hours=guard, max_heavy=4,
                              min_free_gib=3, cap_gib=4, poll=0.3, dry_run=False)
    R = D.R10(a)
    seen = {"max_kids": 0}

    def cmd(u):
        lf, j = u["leaf"], u["part"]
        d = R.ladder_dir(lf, j) if u["kind"] == "ladder" else os.path.dirname(R.extra_json(lf, j))
        arg = {"kind": u["kind"], "dir": d, "leaf": lf, "part": j, "slice": C.part_slice("25k", j),
               "ladder_npz": os.path.join(R.ladder_dir(lf, j), PRR.LADDER_DN + ".npz")}
        seen["max_kids"] = max(seen["max_kids"], len(R.kids) + 1)
        return [sys.executable, "-I", "-B", FAKE, json.dumps(arg)]
    R.cmd = cmd
    return R, a, seen


# ---- run 1: everything
R, a, seen = make(13.0)
check(sorted({u["leaf"] for u in R.units.values()}) == [C.leaf("X", 10), C.leaf("Z", 10)] and len(R.units) == 20,
      f"units: the two r = 10 leaves only ({len(R.units)} units)")
check(R.holds([]) == ({}, False) and R.killed() is False, "no hold, no kill")
code = R.run()
check(code == 0, f"exit {code}")
check(seen["max_kids"] <= 4, f"at most 4 units at once (max {seen['max_kids']})")
T = json.load(open(os.path.join(a.res, "r10_tally.json")))
check(T["complete"] and T["still_failing"] == [] and "kill" not in T and T["what"].startswith("post hoc"),
      f"tally: complete {T['complete']}, post hoc, no kill field")
vx, vz = T["verdicts"][C.leaf("X", 10)], T["verdicts"][C.leaf("Z", 10)]
check(vx["verdict"] == "pass (extra rung)" and vz["verdict"] == "pass (ladder)", f"verdicts {vx['verdict']} / {vz['verdict']}")
L = json.load(open(os.path.join(a.state, "ledger.json")))["units"]
ext = sorted(k for k, x in L.items() if k.startswith("extra|") and x.get("state") == "done")
check(len(ext) == 5 and all("/X/r10|" in k for k in ext), f"extra parts run: {len(ext)}, all X/r10")
# direct computation on the PR3 sample
for b, v in (("X", vx), ("Z", vz)):
    lf = C.leaf(b, 10); k1 = kf = ch = n = band = 0
    for j in range(C.PARTS):
        d = R.ladder_dir(lf, j)
        z = np.load(os.path.join(d, PRR.LADDER_DN + ".npz")); s = np.load(os.path.join(d, "shots.npz"))
        m = z["idx"] % 25 != 0
        d1 = (z["llr_t1"] > 0).astype(np.uint8); df = z["pred"][:, 0]
        k1 += int(((d1 != s["obs"]) & m).sum()); kf += int(((df != s["obs"]) & m).sum()); ch += int(((d1 != df) & m).sum())
        n += int(m.sum()); band += int((z["redecoded"] & m).sum())
    lad = v["ladder"]
    check((lad["errors_tau1"], lad["errors_tau2"], lad["decisions_changed"], v["n_sample"], v["band_shots_sample"])
          == (k1, kf, ch, n, band) and n == 24000,
          f"{b}/r10 ladder values equal a direct computation (k {k1} -> {kf}, changed {ch}, sample {n}, band {band})")
check(vx["extra"]["decisions_changed"] == 0 and vx["extra"]["errors_tau3"] == vx["ladder"]["errors_tau2"],
      "X/r10 extra rung: 0 changes, errors unchanged")

# ---- run 2: a guard below the projection starts nothing
R2, a2, seen2 = make(5.0)
code2 = R2.run()
L2 = json.load(open(os.path.join(a2.state, "ledger.json")))["units"]
T2 = json.load(open(os.path.join(a2.res, "r10_tally.json")))
check(code2 == 6 and not any(x.get("started") for x in L2.values()) and T2["status"].startswith("stopped: projection"),
      f"guard 5 < projection: exit {code2}, nothing started, status {T2['status']!r}")
print(f"{sum(ok)} of {len(ok)} checks pass")
sys.exit(0 if all(ok) else 1)
