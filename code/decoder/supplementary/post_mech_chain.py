#!/usr/bin/env python3
"""Decoder's chain after round 2's mech fits (code/decoder/round2/RELAUNCH.md section 2.6). Light: it polls once a
minute and starts each step as a child in its own session, so a step outlives the chain. The slot order after the mech
fits, main's decisions relayed at 20:00Z and 21:40Z on 2026-10-10: round 2's in-sample decodes, reservoir's mech check
(main starts it), the post hoc non-band check, then PR3; pr3_runner.py holds a slot for each of the first three.

Steps:
  insample_i (i = 0, 1)  once no run_mech.py of part i/2 is alive and that part is complete (each of its series has
      mech_fit_T.json, mech_fit_D.json and eval_mech.json for its 4 W5 leaves: run_mech.py's own test), start
      run_insample.py --part i/2 (PR2 section 8.4) as the RELAUNCH block does. Done when every series of the part has
      the decoders RL, T, D, Tm and Dm in its 4 leaves' insample_data.npz (run_insample.py's own test). A part whose mech
      part is incomplete with no mech process alive for --grace seconds is "blocked"; it starts if that mech part later
      completes.
  explore  once both in-sample parts are done: analyze_explore.py (writes round2/explore_analysis.json, which
      reservoir's mech check reads; the earlier file is first copied here as explore_analysis.before_chain.json).
  nonband  once no mech or in-sample process is alive, neither in-sample part is waiting or running, and explore has
      ended (or cannot run, a part being blocked or failed): nonband_check.py with its 4.0 core-hour guard
      (willow_prereg.md section 9 entry 13).
  nonband_analysis  once the check has a final summary (complete, or stopped by its guard) and round 1's final
      analysis exists (final_round1.json, written by final_round1_trigger.py after K2's complete tally):
      nonband_analyze.py. Then the chain exits 0.
A step that fails is not retried by this process; running the chain again (the RELAUNCH block) retries it.
Restartable: the state is <out>/state.json; a step whose process is alive is adopted, not started again; a finished step
is recognised from its outputs. A second chain with the same --out exits at once.
Run: python3 -I -B post_mech_chain.py --extract /home/claude/work/decoder/willow_run/extract \
         --out <X>/results/decoder/willow/supplementary/post_mech_chain   (see RELAUNCH.md)
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, zipfile

sys.dont_write_bytecode = True
X = "/mnt/project-files/tensor-networks/deep-dives/experimental"
ARCH = "google_105Q_surface_code_d3_d5_d7"          # run_f1.ARCH
W5 = (10, 13, 30, 50)                                # cond_fit.W5
STEPS = ("insample_0", "insample_1", "explore", "nonband", "nonband_analysis")
FINISHED = ("done", "failed", "stopped")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def all_series(res):
    """run_f1.all_series, copied (no import of the fitting code)."""
    out = set()
    for dp, dn, fn in os.walk(os.path.join(res, "fit", ARCH)):
        if os.path.basename(dp) == "rl_tsplit" and "fit_data.npz" in fn:
            rel = os.path.relpath(os.path.dirname(dp), os.path.join(res, "fit"))
            out.add(os.path.dirname(rel))
    return sorted(out)


def npz_keys(p):
    try:
        with zipfile.ZipFile(p) as z:
            return {n[:-4] for n in z.namelist() if n.endswith(".npy")}
    except (OSError, zipfile.BadZipFile):
        return set()


def procs():
    """(pid, args, cwd) of every process but this one."""
    out, me = [], os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cl = [c.decode(errors="replace") for c in f.read().split(b"\0") if c]
            cwd = os.readlink(f"/proc/{d}/cwd")
        except OSError:
            continue
        if cl:
            out.append((int(d), cl, cwd))
    return out


def runs_script(cl, cwd, script):
    """True if one of the arguments is this script (absolute, or relative to the process's cwd)."""
    base, real = os.path.basename(script), os.path.realpath(script)
    for c in cl[:6]:
        if c.endswith(base) and os.path.realpath(c if os.path.isabs(c) else os.path.join(cwd, c)) == real:
            return True
    return False


def arg_of(cl, flag):
    for i, c in enumerate(cl[:-1]):
        if c == flag:
            return cl[i + 1]
    return None


class Chain:
    def __init__(self, a):
        self.a = a
        self.code, self.res = a.code, a.res
        self.r2 = os.path.join(a.res, "round2")
        self.S = {"mech": os.path.join(a.code, "round2", "run_mech.py"),
                  "insample": os.path.join(a.code, "round2", "run_insample.py"),
                  "explore": os.path.join(a.code, "round2", "analyze_explore.py"),
                  "nonband": os.path.join(a.code, "supplementary", "nonband_check.py"),
                  "nonband_analysis": os.path.join(a.code, "supplementary", "nonband_analyze.py"),
                  "chain": os.path.abspath(__file__)}
        os.makedirs(a.out, exist_ok=True)
        self.state_p = os.path.join(a.out, "state.json")
        self.logf = open(os.path.join(a.out, "chain.log"), "a")
        self.kids = {}                                  # step -> Popen (this process's children)
        self.series = all_series(a.res)
        self.part = {i: self.series[i::2] for i in (0, 1)}
        self.gone_since = {0: None, 1: None}

    def log(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True); self.logf.write(line + "\n"); self.logf.flush()

    # ------------------------------------------------------------------ tests on outputs
    def mech_done(self, s):
        m = os.path.join(self.r2, "mech")
        return (os.path.exists(os.path.join(m, s, "mech_fit_T.json")) and os.path.exists(os.path.join(m, s, "mech_fit_D.json"))
                and all(os.path.exists(os.path.join(m, f"{s}/r{r}", "eval_mech.json")) for r in W5))

    def insample_done(self, s):
        want = {"RL", "T"}
        if os.path.exists(os.path.join(self.r2, "f1", s, "fit_series.json")):
            want.add("D")
        for tag, fn in (("Tm", "mech_fit_T.json"), ("Dm", "mech_fit_D.json")):
            if os.path.exists(os.path.join(self.r2, "mech", s, fn)):
                want.add(tag)
        if not {"Tm", "Dm"} <= want:
            return False
        return all({f"{t}_pred" for t in want} <= npz_keys(os.path.join(self.r2, "insample", f"{s}/r{r}", "insample_data.npz"))
                   for r in W5)

    def nonband_summary(self):
        try:
            s = json.load(open(os.path.join(self.a.nonband, "summary.json")))
        except Exception:
            return None
        if s.get("test_only") or not str(s.get("status", "")).startswith(("complete", "stopped")):
            return None
        return s

    # ------------------------------------------------------------------ processes
    def alive(self, P, key, part=None):
        out = []
        for pid, cl, cwd in P:
            if not runs_script(cl, cwd, self.S[key]):
                continue
            if part is not None:
                p = arg_of(cl, "--part") or "0/1"
                if p not in (f"{part}/2", "0/1"):
                    continue
            out.append(pid)
        return out

    def start(self, step, cmd, env_extra, cwd):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        env.update(env_extra)
        os.makedirs(cwd, exist_ok=True)
        if "NUMBA_CACHE_DIR" in env_extra:
            os.makedirs(env_extra["NUMBA_CACHE_DIR"], exist_ok=True)
        f = open(os.path.join(self.a.out, f"{step}.out"), "ab")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd, env=env,
                             start_new_session=True)
        f.close()
        self.kids[step] = p
        st = self.st["steps"][step]
        st.update({"state": "running", "pid": p.pid, "started": now(), "ended": None, "rc": None, "cmd": cmd})
        self.log(f"{step}: started pid {p.pid}: {' '.join(cmd)}")
        self.save()

    def ended(self, step, P, key, part=None):
        """None while the step's process runs; else its exit status (None if it was adopted)."""
        p = self.kids.get(step)
        if p is not None:
            rc = p.poll()
            if rc is None:
                return "running"
            del self.kids[step]
            return rc
        return "running" if self.alive(P, key, part) else None

    def mem_ok(self):
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 >= self.a.min_free_gib * 2 ** 30
        except OSError:
            pass
        return True

    def save(self):
        self.st["updated"] = now()
        write_json(self.state_p, self.st)

    def finish(self, step, state, rc=None, note=None, **kw):
        st = self.st["steps"][step]
        st.update({"state": state, "ended": now(), "rc": rc, "note": note, **kw})
        self.log(f"{step}: {state}" + (f" (exit {rc})" if rc is not None else "") + (f": {note}" if note else ""))
        self.save()

    # ------------------------------------------------------------------ the loop
    def run(self):
        a = self.a
        P = procs()
        other = [pid for pid, cl, cwd in P if runs_script(cl, cwd, self.S["chain"])
                 and os.path.realpath(arg_of(cl, "--out") or "") == os.path.realpath(a.out)]
        if other:
            print(f"{now()} another chain with this --out is running (pid {other}); exiting", flush=True)
            return 0
        self.st = json.load(open(self.state_p)) if os.path.exists(self.state_p) else {"made": now(), "steps": {}, "runs": []}
        me = md5_file(self.S["chain"])
        self.st["runs"].append({"started": now(), "pid": os.getpid(), "md5": me,
                                "scripts_md5": {k: (md5_file(v) if os.path.exists(v) else None) for k, v in self.S.items()}})
        assert len(self.series) == 18 or a.test, f"{len(self.series)} series, not 18"
        for s in STEPS:
            st = self.st["steps"].setdefault(s, {"state": "waiting"})
            if st["state"] in ("running", "failed", "blocked"):
                st["state"] = "waiting"                  # re-examined below; a failed step is retried by a new chain
        self.log(f"chain (md5 {me}) started; {len(self.series)} series; state {json.dumps({s: self.st['steps'][s]['state'] for s in STEPS})}")
        self.save()
        while True:
            P = procs()
            mech_alive = {i: self.alive(P, "mech", i) for i in (0, 1)}
            ins_alive = {i: self.alive(P, "insample", i) for i in (0, 1)}
            # ---- in-sample decodes, one per mech part
            for i in (0, 1):
                step = f"insample_{i}"; st = self.st["steps"][step]
                part_done = all(self.insample_done(s) for s in self.part[i])
                if st["state"] in ("waiting", "blocked"):
                    if part_done and not ins_alive[i]:
                        self.finish(step, "done", note="every series of the part already has Tm and Dm")
                        continue
                    if ins_alive[i]:
                        st.update({"state": "running", "pid": ins_alive[i][0], "started": st.get("started") or now()})
                        self.log(f"{step}: adopted the running pid {ins_alive[i][0]}"); self.save()
                        continue
                    if mech_alive[i]:
                        self.gone_since[i] = None
                        if st["state"] == "blocked":
                            st["state"] = "waiting"; self.log(f"{step}: mech part {i}/2 is running again; waiting"); self.save()
                        continue
                    if all(self.mech_done(s) for s in self.part[i]):
                        if not self.mem_ok():
                            continue
                        self.start(step, [sys.executable, "-I", "-B", self.S["insample"], "--extract", a.extract,
                                          "--res", a.res, "--f1", os.path.join(self.r2, "f1"), "--mech",
                                          os.path.join(self.r2, "mech"), "--out", os.path.join(self.r2, "insample"),
                                          "--part", f"{i}/2"],
                                   {"NUMBA_CACHE_DIR": os.path.join(a.work, "round2", "numba_cache")},
                                   os.path.join(a.work, "round2"))
                        continue
                    self.gone_since[i] = self.gone_since[i] or time.time()
                    if st["state"] == "waiting" and time.time() - self.gone_since[i] > a.grace:
                        k = sum(self.mech_done(s) for s in self.part[i])
                        st["state"] = "blocked"
                        self.log(f"{step}: blocked: mech part {i}/2 is not running and has {k} of {len(self.part[i])} "
                                 f"series complete (decoder or main decides)"); self.save()
                elif st["state"] == "running":
                    rc = self.ended(step, P, "insample", i)
                    if rc == "running":
                        continue
                    if all(self.insample_done(s) for s in self.part[i]):
                        self.finish(step, "done", rc)
                    else:
                        k = sum(self.insample_done(s) for s in self.part[i])
                        self.finish(step, "failed", rc, f"{k} of {len(self.part[i])} series complete")
            S = self.st["steps"]
            # ---- analyze_explore.py once both parts are done
            st = S["explore"]
            if st["state"] == "waiting" and S["insample_0"]["state"] == "done" and S["insample_1"]["state"] == "done" \
                    and not ins_alive[0] and not ins_alive[1]:
                if self.alive(P, "explore"):
                    st.update({"state": "running", "started": now()}); self.log("explore: adopted a running analyze_explore.py")
                else:
                    ea = os.path.join(self.r2, "explore_analysis.json")
                    keep = os.path.join(a.out, "explore_analysis.before_chain.json")
                    if os.path.exists(ea) and not os.path.exists(keep):
                        shutil.copy2(ea, keep)
                        self.log(f"explore: the earlier explore_analysis.json (md5 {md5_file(keep)}) copied to {keep}")
                    st["t0"] = time.time()
                    self.start("explore", [sys.executable, "-I", "-B", self.S["explore"], "--res", a.res, "--r2", self.r2,
                                           "--m3", a.m3],
                               {"NUMBA_CACHE_DIR": os.path.join(a.work, "round2", "numba_cache")}, os.path.join(a.work, "round2"))
            elif st["state"] == "running":
                rc = self.ended("explore", P, "explore")
                if rc != "running":
                    ea = os.path.join(self.r2, "explore_analysis.json")
                    fresh = os.path.exists(ea) and os.path.getmtime(ea) >= st.get("t0", 0) - 5
                    if rc in (0, None) and fresh:
                        self.finish("explore", "done", rc, md5=md5_file(ea))
                    else:
                        self.finish("explore", "failed", rc, "explore_analysis.json not rewritten")
            # ---- the non-band check
            st = S["nonband"]
            ins_over = all(S[f"insample_{i}"]["state"] in ("done", "failed", "blocked") for i in (0, 1))
            explore_over = S["explore"]["state"] in FINISHED or (ins_over and not (S["insample_0"]["state"] == "done"
                                                                                  and S["insample_1"]["state"] == "done"))
            if st["state"] == "waiting":
                summ = self.nonband_summary()
                if summ:
                    self.finish("nonband", "done" if summ["status"] == "complete" else "stopped", note=summ["status"])
                elif self.alive(P, "nonband"):
                    st.update({"state": "running", "started": now()}); self.log("nonband: adopted a running nonband_check.py")
                    self.save()
                elif ins_over and explore_over and not any(mech_alive.values()) and not any(ins_alive.values()) \
                        and not self.alive(P, "explore") and self.mem_ok():
                    self.start("nonband", [sys.executable, "-I", "-B", self.S["nonband"], "--root", a.extract, "--res", a.res,
                                           "--lists", a.nonband, "--out", a.nonband, "--cpu-budget-hours", str(a.nonband_budget)],
                               {"NUMBA_CACHE_DIR": os.path.join(a.work, "nonband", "numba_cache")}, os.path.join(a.work, "nonband"))
            elif st["state"] == "running":
                rc = self.ended("nonband", P, "nonband")
                if rc != "running":
                    summ = self.nonband_summary()
                    if summ is None:
                        self.finish("nonband", "failed", rc, "no final summary.json")
                    else:
                        note = summ["status"] + (f"; rebuild failed: {summ.get('rebuild_failed')}" if summ.get("rebuild_failed") else "")
                        self.finish("nonband", "done" if summ["status"] == "complete" else "stopped", rc, note,
                                    cpu_core_hours=summ.get("cpu_core_hours"))
            # ---- its analysis, once round 1's final analysis exists
            st = S["nonband_analysis"]
            if st["state"] == "waiting" and S["nonband"]["state"] in ("done", "stopped") and os.path.exists(a.final_r1):
                outp = os.path.join(a.nonband, "analysis", "nonband_results.json")
                f1md5 = md5_file(a.final_r1)
                if os.path.exists(outp) and st.get("final_r1_md5") == f1md5 and st.get("check_summary_md5") == \
                        md5_file(os.path.join(a.nonband, "summary.json")):
                    self.finish("nonband_analysis", "done", note="already analysed")
                elif self.alive(P, "nonband_analysis"):
                    st.update({"state": "running", "started": now()}); self.log("nonband_analysis: adopted a running process")
                else:
                    st["final_r1_md5"] = f1md5; st["check_summary_md5"] = md5_file(os.path.join(a.nonband, "summary.json"))
                    self.start("nonband_analysis", [sys.executable, "-I", "-B", self.S["nonband_analysis"], "--res", a.res,
                                                    "--lists", a.nonband, "--check", a.nonband, "--final-r1", a.final_r1,
                                                    "--out", os.path.join(a.nonband, "analysis")],
                               {}, os.path.join(a.work, "nonband"))
            elif st["state"] == "running":
                rc = self.ended("nonband_analysis", P, "nonband_analysis")
                if rc != "running":
                    outp = os.path.join(a.nonband, "analysis", "nonband_results.json")
                    if rc in (0, None) and os.path.exists(outp):
                        self.finish("nonband_analysis", "done", rc, md5=md5_file(outp))
                    else:
                        self.finish("nonband_analysis", "failed", rc)
            self.save()
            states = {s: S[s]["state"] for s in STEPS}
            if "running" not in states.values() and (states["nonband_analysis"] in FINISHED or states["nonband"] == "failed" or
                                                     (states["nonband"] in ("done", "stopped") and a.no_wait_final)):
                code = 0 if states["nonband_analysis"] == "done" and all(states[s] == "done" for s in STEPS[:3]) else 1
                self.log(f"chain ends (exit {code}): {json.dumps(states)}")
                return code
            time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=X)
    ap.add_argument("--extract", required=True, help="the local Willow extract (run_mech.py's --extract)")
    ap.add_argument("--code", default=None, help="default <project>/code/decoder")
    ap.add_argument("--res", default=None, help="default <project>/results/decoder/willow")
    ap.add_argument("--out", required=True, help="the chain's folder: state.json, chain.log, the steps' output")
    ap.add_argument("--nonband", default=None, help="default <res>/supplementary/nonband_check")
    ap.add_argument("--final-r1", default=None, help="default <res>/final_round1/final_round1.json")
    ap.add_argument("--m3", default=None, help="default <res>/analysis_m3/prereg_results.json")
    ap.add_argument("--work", default="/home/claude/work/decoder")
    ap.add_argument("--nonband-budget", type=float, default=4.0)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--grace", type=float, default=900.0)
    ap.add_argument("--min-free-gib", type=float, default=3.0)
    ap.add_argument("--no-wait-final", action="store_true", help="tests only: end without waiting for final_round1.json")
    ap.add_argument("--test", action="store_true", help="tests only: any number of series")
    a = ap.parse_args()
    a.code = a.code or os.path.join(a.project, "code", "decoder")
    a.res = a.res or os.path.join(a.project, "results", "decoder", "willow")
    a.nonband = a.nonband or os.path.join(a.res, "supplementary", "nonband_check")
    a.final_r1 = a.final_r1 or os.path.join(a.res, "final_round1", "final_round1.json")
    a.m3 = a.m3 or os.path.join(a.res, "analysis_m3", "prereg_results.json")
    sys.exit(Chain(a).run())


if __name__ == "__main__":
    main()
