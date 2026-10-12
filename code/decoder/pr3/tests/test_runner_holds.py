"""Synthetic check of pr3_runner.Runner.holds() (version 4): in-sample part i is held from the end of mech part i until it
has been seen; reservoir's hold starts when no mech process is alive and follows its r2m_ scripts only; the ledger
migration from versions 2 and 3."""
import json, os, sys, types, tempfile
sys.path.insert(0, "/home/claude/work/decoder/pr3")
import pr3_runner as PRR

T = tempfile.mkdtemp(dir="/home/claude/work/decoder/pr3/tests")
pilot = {"measured_rates": {"c1_10": 43.0, "s_10": 0.005, "c2_10": 383.0, "c1_50": 51.3, "s_50": 0.02, "c2_50": 326.0,
                            "c3_50": 1035.0, "cb_50": 1.0, "ovh_extra_s": 4.0, "ovh_ladder_s": 0.0},
         "stage1_ladder": {"peak_rss": 5e8}, "stage2_extra": {"peak_rss": 3.5e8}}
json.dump(pilot, open(os.path.join(T, "pilot.json"), "w"))


def runner(ledger):
    st = tempfile.mkdtemp(dir=T)
    json.dump(ledger, open(os.path.join(st, "ledger.json"), "w"))
    a = types.SimpleNamespace(root="/x", res=os.path.join(T, "res"), project=os.path.join(T, "proj"),
                              work=os.path.join(T, "w"), state=st, logs=os.path.join(T, "l"),
                              pilot=os.path.join(T, "pilot.json"), design="25k", guard_core_hours=116.8, max_heavy=4,
                              min_free_gib=3, cap_gib=4, poll=1, dry_run=True)
    R = PRR.Runner(a); R.procs_all = []
    return R


base = lambda: {"made": "x", "design": "25k", "units": {}, "holds": {"insample": {"state": "held", "last_seen": None},
                "reservoir": {"state": "released", "last_seen": 1791665860.0}, "final_round1": {"state": "held", "last_seen": None}}}
R2 = runner(base())                                          # a version-2 ledger: both reset
assert R2.ledger["holds"]["reservoir"]["state"] == "held" and R2.ledger["holds"]["insample"]["parts_seen"] == []
L3 = base(); L3["holds_version"] = 3; L3["holds"]["reservoir"] = {"state": "seen", "last_seen": 5.0}
R3 = runner(L3)                                              # a version-3 ledger: in-sample reset, reservoir kept
assert R3.ledger["holds"]["reservoir"]["state"] == "seen" and R3.ledger["holds_version"] == 4
print("migrations ok:", R3.ledger["notes"])

R = runner(base())
H = R.ledger["holds"]
clock = [1_000_000.0]
PRR.time.time = lambda: clock[0]
P = lambda *args: {"pid": 1, "pgid": 1, "cpu": 0, "rss": 0, "cmd": [b"python3", b"-I", b"-B"] + [x.encode() for x in args]}
M0, M1 = P("/x/run_mech.py", "--extract", "e", "--part", "0/2"), P("/x/run_mech.py", "--extract", "e", "--part", "1/2")
I0, I1 = P("/x/run_insample.py", "--out", "o", "--part", "0/2"), P("/x/run_insample.py", "--out", "o", "--part", "1/2")
res_other = P("/mnt/x/code/reservoir/willow_gapmech/mpe.py")


def step(heavy, dt, label, want):
    clock[0] += dt
    out, pause = R.holds(heavy)
    got = (out["insample"], out["reservoir"])
    print(f"{label:52s} -> insample {got[0]}, reservoir {got[1]} | seen {H['insample']['parts_seen']} "
          f"{H['insample']['state']}/{H['reservoir']['state']}")
    assert got == want, (label, got, want)


step([M0, M1, res_other], 0, "both mech parts, a reservoir MPE run", (0, 0))
step([M0, M1, I1], 10, "in-sample 1/2 during mech part 1: not counted", (0, 0))
assert H["insample"]["parts_seen"] == []
step([M0], 30, "mech part 1 ended", (1, 0))
step([M0, I1], 30, "in-sample 1/2 running", (0, 0))
step([M0], 2000, "in-sample 1/2 done, mech part 0 running", (0, 0))
step([M0], 300, "still mech part 0 (no release yet)", (0, 0))
assert H["insample"]["state"] == "held"
step([], 300, "mech part 0 ended", (1, 1))
step([res_other], 30, "a non-check reservoir script", (1, 1))
step([I0], 30, "in-sample 0/2 running", (0, 1))
step([], 60, "in-sample 0/2 gone 60 s", (0, 1))
step([], 61, "in-sample 0/2 gone 121 s: released", (0, 1))
assert H["insample"]["state"] == "released"
step([P("/usr/bin/python3", "/mnt/x/code/reservoir/willow_r2check/r2m_check.py", "a")], 30, "r2m_check running", (0, 0))
step([], 700, "r2m_check gone 700 s (localmax not seen)", (0, 1))
step([P("/mnt/x/code/reservoir/willow_r2check/r2m_localmax.py")], 30, "r2m_localmax running", (0, 0))
step([], 590, "localmax gone 590 s", (0, 1))
step([], 20, "localmax gone 610 s: released", (0, 0))
assert H["reservoir"]["state"] == "released"

# --part 0/1 counts for both parts; the 60-minute fallback for reservoir
R = runner(base()); H = R.ledger["holds"]
step([], 0, "fresh: no mech", (2, 1))
step([P("/x/run_insample.py", "--part", "0/1")], 30, "in-sample --part 0/1", (0, 1))
step([P("/x/run_insample.py", "--series", "a/b")], 30, "a --series run does not count", (0, 1))
step([P("/x/r2m_redecode.py")], 30, "r2m_redecode", (0, 0))
step([], 3590, "fallback: gone 3590 s", (0, 1))
step([], 20, "fallback: gone 3610 s", (0, 0))
# control.json releases
R = runner(base()); H = R.ledger["holds"]
json.dump({"insample": "released", "reservoir": "released"}, open(os.path.join(R.a.state, "control.json"), "w"))
step([], 0, "control.json releases both", (0, 0))
print("named heavy:", [n for n in PRR.HEAVY_NAMES if n.startswith(b"r2m_")])
print("ALL OK")
