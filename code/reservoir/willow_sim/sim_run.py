"""Orchestrator of the simulation with the RL DEM as the true noise (notes/reservoir/sim_design.md, approved by main
10 Oct 2026: stages A + B, guard 6.0 core-hours, one process at a time, normal priority, one thread).
EXPLORATORY: post hoc question, pre-specified design.

reservoir, 2026-10-10.
  plan OUT [tiny]   writes the run plan (leaves in the design's order with their sampler seeds, shots, decoders, flags,
                    chunk sizes, reference subsets, certificate settings, guard, stage-B deadline, paths, projected CPU)
  run PLAN          runs the plan's steps one child process at a time (python3 -I -B, OMP/OPENBLAS/MKL/NUMBA threads 1,
                    NUMBA_CACHE_DIR in scratch, the parent's priority):
     stage A (r = 10): sample the d = 3 leaves; per d = 3 leaf decode_willow.py (ml_exact, mwpm, corr_mwpm; dem tag
       rl_sim) then sim_mpe.py d3; sample the d = 5 leaves; per d = 5 leaf decode_willow.py (ml_tau2:1e-6/1e-8:0.1,
       mwpm, corr_mwpm) then sim_mpe.py d5; per d = 5 leaf the tau2 and tau3 references (decode_willow.py,
       ml_sparse:1e-8:2000000 on the first shots, dem tag rl_sim_t2; ml_sparse:1e-9:2000000, rl_sim_t3); analysis;
       copy of the outputs to results/reservoir/willow_sim/ (through qrc.outpaths).
     stage B (r = 30): the same without the MPE; analysis of A + B; copy.
  CPU guard: the children's CPU (RUSAGE_CHILDREN) is added up in state.json. Before each step the run stops if the
  CPU used plus the step's projected CPU would pass the guard; while a step runs its CPU is read from /proc every 15 s
  and the step is stopped if the total passes the guard. Stage B starts only before the plan's deadline (main: no new
  stage at or after 05:00Z) and only if the CPU used plus stage B's projection (scaled by stage A's actual/projected
  ratio when above 1) stays within the guard. A step that fails is retried once (decode_willow.py and sim_mpe.py
  resume from their checkpoints); a step that fails twice, or a failed copy, is recorded in state.json and the run goes
  on (steps that need its output then fail and are recorded too), so that one failure does not cost the night's
  decoding; only the guard, the stage-B rule and SIGTERM stop the run. SIGTERM stops the current child and the run.
  State: <scratch>/state.json (steps done, CPU, stop reason); restarting `run` skips the steps already done.
Usage: python3 -I sim_run.py plan OUT [tiny] | run PLAN
"""
from __future__ import annotations

import datetime as dt
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.outpaths import out_path  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARCH = "google_105Q_surface_code_d3_d5_d7"
DEM_REL = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
P3 = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
BASES = ["X", "Z"]
SCRATCH = "/home/claude/work/reservoir/willow_sim"
CLK = os.sysconf("SC_CLK_TCK")
STOP = {"flag": False}


def rel_of(d, patch, basis, r):
    return f"d{d}_at_{patch}/{basis}/r{r:02d}"


def cmd_plan(out, tiny=False):
    scratch = os.path.join(SCRATCH, "tiny") if tiny else SCRATCH
    if tiny:
        spec = [("A", 10, {3: 600, 5: 120}, {3: P3[6:7], 5: P5[1:2]}, ["Z"]),
                ("B", 30, {3: 300, 5: 20}, {3: P3[6:7], 5: P5[1:2]}, ["Z"])]
    else:
        spec = [("A", 10, {3: 50000, 5: 20000}, {3: P3, 5: P5}, BASES),
                ("B", 30, {3: 10000, 5: 2500}, {3: P3, 5: P5}, BASES)]
    leaves = []
    for stage, r, shots, pats, bases in spec:
        for d in (3, 5):
            for p in pats[d]:
                for b in bases:
                    leaves.append({"stage": stage, "d": d, "patch": p, "basis": b, "r": r, "leaf": rel_of(d, p, b, r),
                                   "shots": shots[d]})
    ss = np.random.SeedSequence(20261011).spawn(len(leaves))
    k5 = 0
    for i, L in enumerate(leaves):
        L["seed_index"] = i
        L["seed"] = int(ss[i].generate_state(1, dtype=np.uint64)[0])
        if L["stage"] == "A" and L["d"] == 5:
            L["d5_index"] = k5
            k5 += 1
    X = ROOT
    sparse = lambda tau: {"decoder": f"ml_sparse:{tau}:2000000", "file": f"ml_sparse_{tau}_2000000.npz"}  # noqa: E731
    plan = {
        "note": ("Run plan of the simulation with the RL DEM as the true noise (reservoir, 2026-10-10), approved by main "
                 "as stages A + B (notes/reservoir/sim_design.md, md5 c6e36d4a6c44ae7ca335200aa846a198, and its "
                 "pre-run addendum). Leaves in the design's order (stage A then B; d = 3 then d = 5; patches as listed; "
                 "X then Z); sampler seed of leaf i = SeedSequence(20261011).spawn(n_leaves)[i].generate_state(1, "
                 "uint64)[0]." + (" TINY TEST PLAN." if tiny else "")),
        "tiny": tiny,
        "paths": {"real_arch": "/home/claude/work/decoder/willow_run/extract/" + ARCH,
                  "sim_root": os.path.join(scratch, "archive"), "decode_out": os.path.join(scratch, "decode"),
                  "decode_work": os.path.join(scratch, "work"), "mpe_dir": os.path.join(scratch, "mpe"),
                  "analysis_dir": os.path.join(scratch, "analysis"), "numba_cache": os.path.join(scratch, "numba_cache"),
                  "scratch": scratch, "results": os.path.join(X, "results", "reservoir", "willow_sim"),
                  "decode_willow": os.path.join(X, "code", "decoder", "decode_willow.py"),
                  "real_decode": os.path.join(X, "results", "decoder", "willow", "decode"),
                  "k2_dir": os.path.join(X, "results", "decoder", "willow", "supplementary", "k2_extra_rung"),
                  "stage1_dir": os.path.join(X, "results", "reservoir", "willow_gapmech")},
        "seeds": {"sampling": 20261011, "subsets": 20261012, "bootstrap": 20261012},
        "leaves": leaves,
        "decode": {"A3": {"decoders": "ml_exact,mwpm,corr_mwpm", "chunk": 5000},
                   "A5": {"decoders": "ml_tau2:1e-6/1e-8:0.1,mwpm,corr_mwpm", "chunk": 2000},
                   "B3": {"decoders": "ml_exact,mwpm,corr_mwpm", "chunk": 5000},
                   "B5": {"decoders": "ml_tau2:1e-6/1e-8:0.1,mwpm,corr_mwpm", "chunk": 500},
                   "flags": {"--dem-file": DEM_REL, "--dem-tag": "rl_sim", "--synd-ref": "0",
                             "--recompute-check": "none", "--shots": "0:<n>:1", "--max-shots": "<n>"}},
        "refs": {"r10": {"t2": dict(sparse("1e-8"), shots=2 if tiny else 125, chunk=25, dem_tag="rl_sim_t2"),
                         "t3": dict(sparse("1e-9"), shots=1 if tiny else 13, chunk=13, dem_tag="rl_sim_t3")},
                 "r30": {"t2": dict(sparse("1e-8"), shots=2 if tiny else 50, chunk=10, dem_tag="rl_sim_t2"),
                         "t3": dict(sparse("1e-9"), shots=1 if tiny else 10, chunk=5, dem_tag="rl_sim_t3")}},
        "mpe_d5": {"time_limit_s": 60, "concordant_per_leaf": 10 if tiny else 250, "concordant_seed": 20261012,
                   "leaves_d5_r10": k5},
        "bootstrap_B": 10000,
        "guard_cpu_s": 1800 if tiny else 21600,
        "stage_b_deadline_utc": "2026-10-11T05:00:00Z",
        "projected_cpu_s": {"sample": {"A3": 60, "A5": 20, "B3": 20, "B5": 10},
                            "decode": {"A3": 40, "A5": 800, "B3": 25, "B5": 580},
                            "mpe": {"A3": 50, "A5": 480}, "t2": {"A": 70, "B": 100}, "t3": {"A": 35, "B": 60},
                            "analysis": {"A": 180, "B": 240}, "copy": 5},
        "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"},
    }
    if tiny:
        plan["projected_cpu_s"]["decode"] = {"A3": 30, "A5": 30, "B3": 30, "B5": 30}
        plan["projected_cpu_s"]["mpe"] = {"A3": 30, "A5": 60}
    with open(out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"plan: {len(leaves)} leaves, {sum(L['shots'] for L in leaves)} shots -> {out}")


# ------------------------------------------------------------------ steps
def leaves_of(plan, stage, d):
    return [L for L in plan["leaves"] if L["stage"] == stage and L["d"] == d]


def make_steps(plan):
    S = []
    for stage in ("A", "B"):
        if not any(L["stage"] == stage for L in plan["leaves"]):
            continue
        for d in (3, 5):
            if not leaves_of(plan, stage, d):
                continue
            S.append({"id": f"{stage}/sample/d{d}", "kind": "sample", "stage": stage, "d": d, "key": f"{stage}{d}"})
            for L in leaves_of(plan, stage, d):
                S.append({"id": f"{stage}/decode/{L['leaf']}", "kind": "decode", "stage": stage, "d": d,
                          "leaf": L, "key": f"{stage}{d}"})
                if stage == "A":
                    S.append({"id": f"{stage}/mpe/{L['leaf']}", "kind": "mpe", "stage": stage, "d": d, "leaf": L,
                              "key": f"{stage}{d}"})
        for L in leaves_of(plan, stage, 5):
            for t in ("t2", "t3"):
                S.append({"id": f"{stage}/{t}/{L['leaf']}", "kind": t, "stage": stage, "d": 5, "leaf": L, "key": stage})
        S.append({"id": f"{stage}/analysis", "kind": "analysis", "stage": stage, "key": stage})
        S.append({"id": f"{stage}/copy", "kind": "copy", "stage": stage})
    return S


def projected(plan, st):
    p = plan["projected_cpu_s"]
    if st["kind"] == "copy":
        return p["copy"]
    return p[st["kind"]][st["key"]]


def command(plan, st):
    P = plan["paths"]
    py = [sys.executable, "-I", "-B"]
    if st["kind"] == "sample":
        return py + [os.path.join(HERE, "sim_sample.py"), PLAN_PATH[0], st["stage"], str(st["d"])]
    if st["kind"] == "mpe":
        return py + [os.path.join(HERE, "sim_mpe.py"), f"d{st['d']}", PLAN_PATH[0], st["leaf"]["leaf"]]
    if st["kind"] == "analysis":
        stages = "A" if st["stage"] == "A" else "AB"
        return py + [os.path.join(HERE, "sim_analysis.py"), PLAN_PATH[0], stages,
                     os.path.join(P["analysis_dir"], f"sim_analysis_{stages}.json")]
    L = st["leaf"]
    if st["kind"] == "decode":
        D = plan["decode"][st["key"]]
        n, tag, decs, chunk = L["shots"], "rl_sim", D["decoders"], D["chunk"]
    else:
        R = plan["refs"][f"r{L['r']}"][st["kind"]]
        n, tag, decs, chunk = R["shots"], R["dem_tag"], R["decoder"], R["chunk"]
    return py + [P["decode_willow"], "--root", P["sim_root"], "--out", P["decode_out"], "--work", P["decode_work"],
                 "--leaf", f"{ARCH}/{L['leaf']}", "--dem-file", DEM_REL, "--dem-tag", tag, "--decoders", decs,
                 "--max-shots", str(n), "--shots", f"0:{n}:1", "--chunk", str(chunk), "--synd-ref", "0",
                 "--recompute-check", "none"]


def check_decode(plan, st):
    L = st["leaf"]
    if st["kind"] == "decode":
        tag, n, decs = "rl_sim", L["shots"], plan["decode"][st["key"]]["decoders"].split(",")
    else:
        R = plan["refs"][f"r{L['r']}"][st["kind"]]
        tag, n, decs = R["dem_tag"], R["shots"], [R["decoder"]]
    p = os.path.join(plan["paths"]["decode_out"], ARCH, L["leaf"], tag, "summary.json")
    if not os.path.exists(p):
        return f"no summary.json at {p}"
    s = json.load(open(p))
    if s.get("status") != "complete":
        return f"status {s.get('status')}"
    for dname in decs:
        ds = s.get("decoders", {}).get(dname, {})
        if ds.get("n") != n:
            return f"{dname}: n {ds.get('n')} (expected {n})"
    return None


def check_step(plan, st):
    P = plan["paths"]
    if st["kind"] in ("decode", "t2", "t3"):
        return check_decode(plan, st)
    if st["kind"] == "mpe":
        f = os.path.join(P["mpe_dir"], st["leaf"]["leaf"].replace("/", "_"))
        return None if os.path.exists(f + ".npz") and os.path.exists(f + ".json") else "MPE outputs missing"
    if st["kind"] == "sample":
        for L in leaves_of(plan, st["stage"], st["d"]):
            f = os.path.join(P["sim_root"], ARCH, L["leaf"], "sample.json")
            if not os.path.exists(f) or json.load(open(f)).get("seed") != L["seed"]:
                return f"sample missing for {L['leaf']}"
        return None
    if st["kind"] == "analysis":
        stages = "A" if st["stage"] == "A" else "AB"
        f = os.path.join(P["analysis_dir"], f"sim_analysis_{stages}.json")
        return None if os.path.exists(f) else "analysis output missing"
    return None


def child_cpu(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            x = f.read()
        v = x[x.rindex(")") + 2:].split()
        return (int(v[11]) + int(v[12])) / CLK
    except (OSError, ValueError, IndexError):
        return 0.0


def run_child(cmd, env, used, guard, log):
    """Runs one child; returns (exit code or None if stopped, cpu seconds, wall seconds, reason)."""
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    w0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, cwd=SCRATCH_CWD[0])
    reason = None
    while True:
        try:
            rc = p.wait(timeout=15)
            break
        except subprocess.TimeoutExpired:
            pass
        if STOP["flag"]:
            reason = "sigterm"
        elif used + child_cpu(p.pid) > guard:
            reason = "guard_hard"
        if reason:
            log(f"stopping child {p.pid}: {reason}")
            p.terminate()
            try:
                rc = p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
            break
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return (None if reason else rc), cpu, time.time() - w0, reason


def copy_outputs(plan, stage, state, log):
    P = plan["paths"]
    res = P["results"]
    n = 0

    where = []

    def cp(src, rel_dst):
        nonlocal n
        dst = out_path(os.path.join(res, rel_dst))
        if not where:
            where.append(dst[:len(dst) - len(rel_dst)])
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        n += 1

    samples = []
    for L in [x for x in plan["leaves"] if x["stage"] == stage]:
        samples.append(json.load(open(os.path.join(P["sim_root"], ARCH, L["leaf"], "sample.json"))))
        tags = ["rl_sim"] + ([plan["refs"][f"r{L['r']}"][t]["dem_tag"] for t in ("t2", "t3")] if L["d"] == 5 else [])
        for tag in tags:
            src = os.path.join(P["decode_out"], ARCH, L["leaf"], tag)
            for fn in sorted(os.listdir(src)):
                if fn == "summary.json" or (fn.endswith(".npz") and ".tmp" not in fn):
                    cp(os.path.join(src, fn), os.path.join("decode", ARCH, L["leaf"], tag, fn))
        if stage == "A":
            b = L["leaf"].replace("/", "_")
            for ext in (".npz", ".json"):
                cp(os.path.join(P["mpe_dir"], b + ext), os.path.join("mpe", b + ext))
    sp = out_path(os.path.join(res, f"samples_{stage}.json"))
    with open(sp + ".tmp", "w") as f:
        json.dump(samples, f, indent=1)
    os.replace(sp + ".tmp", sp)
    stages = "A" if stage == "A" else "AB"
    cp(os.path.join(P["analysis_dir"], f"sim_analysis_{stages}.json"), f"sim_analysis_{stages}.json")
    cp(PLAN_PATH[0], "sim_plan.json")
    log(f"copied {n + 1} files for stage {stage} to {where[0] if where else res}")


def save_state(state):
    p = os.path.join(SCRATCH_CWD[0], "state.json")
    with open(p + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(p + ".tmp", p)


PLAN_PATH = [""]
SCRATCH_CWD = [""]


def cmd_run(plan_path):
    PLAN_PATH[0] = os.path.abspath(plan_path)
    plan = json.load(open(plan_path))
    P = plan["paths"]
    SCRATCH_CWD[0] = P["scratch"]
    for k in ("sim_root", "decode_out", "decode_work", "mpe_dir", "analysis_dir", "numba_cache"):
        os.makedirs(P[k], exist_ok=True)
    env = dict(os.environ)
    env.update(plan["env"])
    env["NUMBA_CACHE_DIR"] = P["numba_cache"]
    guard = plan["guard_cpu_s"]
    deadline = dt.datetime.strptime(plan["stage_b_deadline_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    sp = os.path.join(P["scratch"], "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"steps": {}, "cpu_used_s": 0.0, "stopped": None}
    state["stopped"] = None
    state.setdefault("runs", []).append({"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
                                         "nice": os.nice(0)})
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__("flag", True))

    def log(msg):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [sim_run] {msg}", flush=True)

    log(f"start: plan {plan_path}; guard {guard} CPU-s; used so far {state['cpu_used_s']:.0f} s; nice {os.nice(0)}; "
        f"stage-B deadline {plan['stage_b_deadline_utc']}")
    steps = make_steps(plan)
    for st in steps:
        sid = st["id"]
        if state["steps"].get(sid, {}).get("status") == "done":
            continue
        if STOP["flag"]:
            state["stopped"] = "sigterm"
            break
        if st["stage"] == "B" and not any(state["steps"].get(x["id"], {}).get("status") == "done"
                                          for x in steps if x["stage"] == "B"):
            if dt.datetime.now(dt.timezone.utc) >= deadline:
                state["stopped"] = "deadline_stage_B_not_started"
                log("stage B not started: the deadline has passed")
                break
            done_a = [x for x in steps if x["stage"] == "A" and x["kind"] not in ("analysis", "copy")]
            act = sum(state["steps"].get(x["id"], {}).get("cpu_s", 0.0) for x in done_a)
            pro = sum(projected(plan, x) for x in done_a)
            ratio = act / pro if pro else 1.0
            proj_b = sum(projected(plan, x) for x in steps if x["stage"] == "B") * max(1.0, ratio)
            state["stage_b_gate"] = {"cpu_used_s": state["cpu_used_s"], "stage_a_actual_over_projected": ratio,
                                     "stage_b_projected_s": proj_b, "guard_s": guard}
            if state["cpu_used_s"] + proj_b > guard:
                state["stopped"] = "guard_projection_stage_B"
                log(f"stage B not started: used {state['cpu_used_s']:.0f} s + projected {proj_b:.0f} s > guard")
                break
            log(f"stage B gate passed: used {state['cpu_used_s']:.0f} s, stage A actual/projected {ratio:.2f}, "
                f"stage B projected {proj_b:.0f} s")
        if state["cpu_used_s"] + projected(plan, st) > guard:
            state["stopped"] = "guard_projection"
            log(f"stop before {sid}: used {state['cpu_used_s']:.0f} s + projected {projected(plan, st)} s > guard")
            break
        if st["kind"] == "copy":
            try:
                copy_outputs(plan, st["stage"], state, log)
                state["steps"][sid] = {"status": "done", "cpu_s": 0.0, "end": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                         time.gmtime())}
            except Exception as ex:  # noqa: BLE001
                log(f"copy failed (recorded; the run goes on): {ex!r}")
                state["steps"][sid] = {"status": "failed", "error": repr(ex), "cpu_s": 0.0}
                state.setdefault("failed_steps", []).append(sid)
            save_state(state)
            continue
        ok = False
        for attempt in (1, 2):
            log(f"step {sid} (attempt {attempt}; used {state['cpu_used_s']:.0f} s)")
            rc, cpu, wall, reason = run_child(command(plan, st), env, state["cpu_used_s"], guard, log)
            state["cpu_used_s"] += cpu
            rec = state["steps"].setdefault(sid, {"cpu_s": 0.0, "wall_s": 0.0, "attempts": 0})
            rec["cpu_s"] += cpu
            rec["wall_s"] += wall
            rec["attempts"] += 1
            rec["exit"] = rc
            if reason:
                rec["status"] = "stopped"
                state["stopped"] = reason
                break
            err = check_step(plan, st) if rc == 0 else f"exit {rc}"
            if err is None:
                rec["status"] = "done"
                rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                log(f"done {sid}: cpu {cpu:.1f} s, wall {wall:.1f} s (projected {projected(plan, st)} s)")
                ok = True
                break
            rec["status"] = "failed"
            rec["error"] = err
            log(f"step {sid} failed: {err}")
            save_state(state)
        save_state(state)
        if not ok and state["stopped"]:
            break
        if not ok:
            state.setdefault("failed_steps", []).append(sid)
            log(f"step {sid} failed twice (recorded; the run goes on)")
            save_state(state)
    own = resource.getrusage(resource.RUSAGE_SELF)
    state["orchestrator_cpu_s_this_run"] = own.ru_utime + own.ru_stime
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["remaining_steps"] = [x["id"] for x in steps if state["steps"].get(x["id"], {}).get("status") != "done"]
    save_state(state)
    log(f"end: stopped={state['stopped']}; children CPU {state['cpu_used_s']:.0f} s "
        f"({state['cpu_used_s'] / 3600:.2f} core-hours); steps left {len(state['remaining_steps'])}")
    try:
        res = P["results"]
        for fn in ("state.json", "run.log"):
            src = os.path.join(P["scratch"], fn)
            if os.path.exists(src):
                dst = out_path(os.path.join(res, fn))
                shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
        log("state.json and run.log copied")
    except Exception as ex:  # noqa: BLE001
        log(f"final copy failed: {ex!r}")
    log("END")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "plan" and len(a) in (2, 3):
        cmd_plan(a[1], tiny=(len(a) == 3 and a[2] == "tiny"))
    elif a and a[0] == "run" and len(a) == 2:
        cmd_run(a[1])
    else:
        raise SystemExit(__doc__)
