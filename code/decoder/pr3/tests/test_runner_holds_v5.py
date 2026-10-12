"""Synthetic check of the non-band hold added to pr3_runner.Runner.holds(): no slot before the shot lists exist or while
a mech process is alive; 1 slot until nonband_check.py has been seen and then gone for 600 s; an immediate release when
its summary is final (not a test run); a release by control.json."""
import json, os, sys, types, tempfile
sys.path.insert(0, "/home/claude/work/decoder/pr3")
import pr3_runner as PRR

T = tempfile.mkdtemp(dir="/home/claude/work/decoder/pr3/tests")
pilot = {"measured_rates": {"c1_10": 43.0, "s_10": 0.005, "c2_10": 383.0, "c1_50": 51.3, "s_50": 0.02, "c2_50": 326.0,
                            "c3_50": 1035.0, "cb_50": 1.0, "ovh_extra_s": 4.0, "ovh_ladder_s": 0.0},
         "stage1_ladder": {"peak_rss": 5e8}, "stage2_extra": {"peak_rss": 3.5e8}}
json.dump(pilot, open(os.path.join(T, "pilot.json"), "w"))


def runner():
    st = tempfile.mkdtemp(dir=T); proj = tempfile.mkdtemp(dir=T)
    L = {"made": "x", "design": "25k", "units": {}, "holds_version": 4,
         "holds": {"insample": {"state": "released", "last_seen": None, "parts_seen": ["0/2", "1/2"]},
                   "reservoir": {"state": "released", "last_seen": None}, "final_round1": {"state": "held", "last_seen": None}}}
    json.dump(L, open(os.path.join(st, "ledger.json"), "w"))
    a = types.SimpleNamespace(root="/x", res=os.path.join(T, "res"), project=proj, work=os.path.join(T, "w"), state=st,
                              logs=os.path.join(T, "l"), pilot=os.path.join(T, "pilot.json"), design="25k",
                              guard_core_hours=116.8, max_heavy=4, min_free_gib=3, cap_gib=4, poll=1, dry_run=True)
    R = PRR.Runner(a); R.procs_all = []
    return R


clock = [2_000_000.0]
PRR.time.time = lambda: clock[0]
P = lambda *args: {"pid": 1, "pgid": 1, "cpu": 0, "rss": 0, "cmd": [b"python3", b"-I", b"-B"] + [x.encode() for x in args]}
M0 = P("/x/run_mech.py", "--part", "0/2")
NB = P("/x/code/decoder/supplementary/nonband_check.py", "--lists", "l", "--out", "o")
CH = P("/x/code/decoder/supplementary/post_mech_chain.py", "--state", "s")      # the chain itself is not heavy-named


def step(R, heavy, dt, label, want):
    clock[0] += dt
    out, pause = R.holds(heavy)
    print(f"{label:58s} -> nonband {out['nonband']} | {R.ledger['holds']['nonband']['state']}")
    assert out["nonband"] == want, (label, out, want)
    return out


def arm(R):
    d = os.path.join(R.a.project, PRR.NONBAND); os.makedirs(d, exist_ok=True)
    json.dump({"seed": 1}, open(os.path.join(d, "shot_lists.json"), "w"))


def summary(R, **kw):
    json.dump(kw, open(os.path.join(R.a.project, PRR.NONBAND, "summary.json"), "w"))


R = runner()
step(R, [], 0, "no shot lists yet", 0)
arm(R)
step(R, [M0], 10, "lists exist, a mech part alive", 0)
o = step(R, [CH], 10, "mech gone, the chain waiting", 1)
assert sum(o.values()) == 1, o
step(R, [NB], 30, "nonband_check running", 0)
assert R.ledger["holds"]["nonband"]["state"] == "seen"
summary(R, status="running", test_only=False)
step(R, [], 300, "gone 300 s (summary still running)", 1)
step(R, [], 290, "gone 590 s", 1)
step(R, [], 20, "gone 610 s: released", 0)
assert R.ledger["holds"]["nonband"]["state"] == "released"
step(R, [NB], 30, "a later run gets no hold", 0)

R = runner(); arm(R)
summary(R, status="complete", test_only=True)
step(R, [], 0, "a test-only summary does not release", 1)
summary(R, status="stopped: cpu budget 4.0 core-hours reached", test_only=False)
step(R, [], 10, "a guard stop releases at once", 0)

R = runner(); arm(R)
summary(R, status="complete", test_only=False)
step(R, [M0], 0, "a final summary while mech alive", 0)
assert R.ledger["holds"]["nonband"]["state"] == "released"

R = runner(); arm(R)
json.dump({"nonband": "released"}, open(os.path.join(R.a.state, "control.json"), "w"))
step(R, [], 0, "control.json release", 0)
assert R.ledger["holds"]["nonband"]["state"] == "released"
assert b"nonband_check.py" in PRR.HEAVY_NAMES and b"analyze_explore.py" in PRR.HEAVY_NAMES
assert b"post_mech_chain.py" not in PRR.HEAVY_NAMES
print("ALL OK")
