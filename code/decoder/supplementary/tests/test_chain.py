"""Synthetic check of post_mech_chain.py with fake steps (4 series; scripts with the real names under a temporary code
tree, so the chain ignores the real processes). Scenarios: the normal order (each in-sample part starts when its own mech
part ends; explore after both, with the earlier file kept; the check after explore; the analysis after round 1's final
analysis); a restart that adopts a running part; a blocked part (mech incomplete) that lets the check go ahead, then
starts when its mech part completes; a failed part; a second chain with the same --out exits at once."""
import json, os, subprocess, sys, tempfile, time

CHAIN = "/home/claude/work/decoder/nonband/post_mech_chain.py"
T = tempfile.mkdtemp(dir="/home/claude/work/decoder/nonband")
ARCH = "google_105Q_surface_code_d3_d5_d7"
CODE, RES = os.path.join(T, "code", "decoder"), os.path.join(T, "res")
EV = os.path.join(T, "events.txt")
SER = [f"{ARCH}/d3_at_q{p}/{b}" for p in ("A", "B") for b in ("X", "Z")]
W5 = (10, 13, 30, 50)

COMMON = f'''import os, sys, time, json
import numpy as np
EV = {EV!r}
def ev(m):
    open(EV, "a").write(f"{{time.time():.3f}} {{m}}\\n")
def arg(f, d=None):
    return sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d
SER = {sorted(SER)!r}
'''
FAKES = {
    "round2/run_mech.py": COMMON + '''
part = arg("--part"); i = int(part[0]); out = arg("--out")
ev(f"mech{i} start"); time.sleep(float(arg("--sleep", 3)))
if arg("--incomplete"):
    ev(f"mech{i} died"); sys.exit(1)
for s in SER[i::2]:
    os.makedirs(os.path.join(out, s), exist_ok=True)
    for fn in ("mech_fit_T.json", "mech_fit_D.json"):
        json.dump({}, open(os.path.join(out, s, fn), "w"))
    for r in (10, 13, 30, 50):
        os.makedirs(os.path.join(out, f"{s}/r{r}"), exist_ok=True); json.dump({}, open(os.path.join(out, f"{s}/r{r}", "eval_mech.json"), "w"))
ev(f"mech{i} end")
''',
    "round2/run_insample.py": COMMON + '''
part = arg("--part"); i = int(part[0]); out = arg("--out")
ev(f"ins{i} start"); time.sleep(float(os.environ.get("FAKE_INS_SLEEP", "2")))
if os.environ.get("FAKE_INS_FAIL") == str(i):
    ev(f"ins{i} fail"); sys.exit(2)
for s in SER[i::2]:
    for r in (10, 13, 30, 50):
        d = os.path.join(out, f"{s}/r{r}"); os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "insample_data.npz"), **{f"{t}_pred": np.zeros(3) for t in ("RL", "T", "D", "Tm", "Dm")})
ev(f"ins{i} end")
''',
    "round2/analyze_explore.py": COMMON + '''
ev("explore start"); time.sleep(1)
json.dump({"new": True}, open(os.path.join(arg("--r2"), "explore_analysis.json"), "w")); ev("explore end")
''',
    "supplementary/nonband_check.py": COMMON + '''
ev("nonband start"); time.sleep(2)
st = os.environ.get("FAKE_NB_STATUS", "complete")
json.dump({"status": st, "test_only": False, "cpu_core_hours": 0.1}, open(os.path.join(arg("--out"), "summary.json"), "w"))
ev("nonband end"); sys.exit(3 if st.startswith("stopped") else 0)
''',
    "supplementary/nonband_analyze.py": COMMON + '''
ev("analysis start"); os.makedirs(arg("--out"), exist_ok=True)
json.dump({"ok": 1}, open(os.path.join(arg("--out"), "nonband_results.json"), "w")); ev("analysis end")
''',
}
for rel, src in FAKES.items():
    os.makedirs(os.path.dirname(os.path.join(CODE, rel)), exist_ok=True)
    open(os.path.join(CODE, rel), "w").write(src)
for s in SER:
    for r in W5:
        d = os.path.join(RES, "fit", s, f"r{r}", "rl_tsplit"); os.makedirs(d, exist_ok=True); open(os.path.join(d, "fit_data.npz"), "w").close()
    os.makedirs(os.path.join(RES, "round2", "f1", s), exist_ok=True); json.dump({}, open(os.path.join(RES, "round2", "f1", s, "fit_series.json"), "w"))
NB = os.path.join(RES, "supplementary", "nonband_check"); os.makedirs(NB)
json.dump({"old": True}, open(os.path.join(RES, "round2", "explore_analysis.json"), "w"))
F1 = os.path.join(RES, "final_round1", "final_round1.json")


def mech(i, sleep, incomplete=False):
    cmd = [sys.executable, "-I", "-B", os.path.join(CODE, "round2", "run_mech.py"), "--extract", "e", "--res", RES,
           "--out", os.path.join(RES, "round2", "mech"), "--part", f"{i}/2", "--sleep", str(sleep)] + (["--incomplete", "1"] if incomplete else [])
    return subprocess.Popen(cmd, start_new_session=True)


def chain(out, *extra, env=None):
    cmd = [sys.executable, "-I", "-B", CHAIN, "--project", T, "--code", CODE, "--res", RES, "--extract", os.path.join(T, "extract"),
           "--out", out, "--work", os.path.join(T, "work"), "--poll", "0.5", "--grace", "4", "--min-free-gib", "0", "--test"] + list(extra)
    return subprocess.Popen(cmd, env=dict(os.environ, **(env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def events():
    out = {}
    for line in open(EV):
        t, m = line.split(" ", 1); out.setdefault(m.strip(), []).append(float(t))
    return out


def states(out):
    return {k: v["state"] for k, v in json.load(open(os.path.join(out, "state.json")))["steps"].items()}


def wait(p, t=90):
    o, _ = p.communicate(timeout=t)
    return p.returncode, o


# ---- 1. the normal order
OUT1 = os.path.join(T, "chain1")
m0, m1 = mech(0, 2), mech(1, 8)
c = chain(OUT1)
time.sleep(1.5)
dup = chain(OUT1); rc_dup, o_dup = wait(dup, 30)
assert rc_dup == 0 and "another chain" in o_dup, o_dup
time.sleep(2)
os.makedirs(os.path.dirname(F1), exist_ok=True); json.dump({"x": 1}, open(F1, "w"))
rc, o = wait(c)
print(o[-2500:])
E = events()
print({k: [round(x - E["mech0 start"][0], 1) for x in v] for k, v in E.items()})
assert rc == 0, rc
assert E["ins0 start"][0] >= E["mech0 end"][0] and E["ins0 start"][0] < E["mech1 end"][0], "part 0 must start before mech part 1 ends"
assert E["ins1 start"][0] >= E["mech1 end"][0]
assert E["explore start"][0] >= max(E["ins0 end"][0], E["ins1 end"][0])
assert E["nonband start"][0] >= E["explore end"][0] and E["analysis start"][0] >= E["nonband end"][0]
assert json.load(open(os.path.join(OUT1, "explore_analysis.before_chain.json"))) == {"old": True}
assert json.load(open(os.path.join(RES, "round2", "explore_analysis.json"))) == {"new": True}
assert all(v == "done" for v in states(OUT1).values()), states(OUT1)
c = chain(OUT1); rc, o = wait(c, 30)                 # a repeat run finds everything done
print(o[-600:]); assert rc == 0 and "chain ends (exit 0)" in o
print("scenario 1 ok")

# ---- 2. restart, adoption, blocked part, guard stop, then the blocked part completes
for sub in ("mech", "insample"):
    subprocess.run(["rm", "-rf", "--", os.path.join(RES, "round2", sub)], check=True)
os.remove(F1); os.remove(os.path.join(NB, "summary.json")); os.remove(EV)
OUT2 = os.path.join(T, "chain2")
m0, m1 = mech(0, 1), mech(1, 2, incomplete=True)
env = {"FAKE_INS_SLEEP": "6", "FAKE_NB_STATUS": "stopped: cpu budget 4.0 core-hours reached"}
c = chain(OUT2, env=env)
for _ in range(60):
    time.sleep(0.5)
    if os.path.exists(EV) and "ins0 start" in events():
        break
time.sleep(1)
c.kill(); c.wait()                                       # the chain dies while part 0 runs
c = chain(OUT2, "--no-wait-final", env=env)
rc, o = wait(c)
print(o[-2500:])
assert "insample_0: adopted the running pid" in o, "the running part must be adopted"
E = events(); assert len(E["ins0 start"]) == 1, E
s2 = states(OUT2); print(s2)
assert s2 == {"insample_0": "done", "insample_1": "blocked", "explore": "waiting", "nonband": "stopped",
              "nonband_analysis": "waiting"} and rc == 1
m1 = mech(1, 1); m1.wait()                              # main relaunches mech part 1 and it completes
json.dump({"x": 2}, open(F1, "w"))
c = chain(OUT2, env=env); rc, o = wait(c)
print(o[-1500:])
s2 = states(OUT2); print(s2)
assert s2 == {"insample_0": "done", "insample_1": "done", "explore": "done", "nonband": "stopped",
              "nonband_analysis": "done"} and rc == 0      # a guard stop is a designed outcome
E = events(); assert len(E["nonband start"]) == 1 and len(E["ins1 start"]) == 1
print("scenario 2 ok")

# ---- 3. a failed part: no explore; the check and its analysis still run
for sub in ("mech", "insample"):
    subprocess.run(["rm", "-rf", "--", os.path.join(RES, "round2", sub)], check=True)
os.remove(os.path.join(NB, "summary.json")); os.remove(EV)
subprocess.run(["rm", "-rf", "--", os.path.join(NB, "analysis")], check=True)
OUT3 = os.path.join(T, "chain3")
m0, m1 = mech(0, 1), mech(1, 1)
c = chain(OUT3, env={"FAKE_INS_FAIL": "1"}); rc, o = wait(c)
print(o[-1500:])
s3 = states(OUT3); print(s3)
assert s3 == {"insample_0": "done", "insample_1": "failed", "explore": "waiting", "nonband": "done",
              "nonband_analysis": "done"} and rc == 1
st = json.load(open(os.path.join(OUT3, "state.json")))
assert st["steps"]["insample_1"]["rc"] == 2 and "0 of 2 series" in st["steps"]["insample_1"]["note"]
print("scenario 3 ok")
print("ALL OK", T)
