#!/usr/bin/env python3
"""PR3's post hoc r = 10 follow-up, amendment 10a (notes/decoder/willow_prereg_r3.md): the tau4 rung for an r = 10
leaf that still fails after tau3. Run it once the r = 10 driver (pr3_r10_posthoc.py) has exited.

Rule (10a). A leaf whose r10_tally.json verdict is "still fails" gets exactly one more rung, tau4 = tau3/10 = 1e-9, on
its band shots (the shots the ladder re-decoded at tau2), part by part:
  - the decoder is rebuilt as k2_extra_rung.py rebuilds it (the same DEM file, md5 checked against summary.json; the
    tau1 decoder's compiled program shared by the deeper rungs);
  - tau3 rebuild check: the first 10 band shots of the part are decoded again at tau3 with tau3's term cap (2,000,000)
    and must give the stored tau3 llr (extra_rung.npz) bitwise; otherwise tau4 does not run (exit 4);
  - tau4 = 1e-9 with a term cap of 8,000,000 on all the part's band shots, in chunks of 5 shots that are kept, so a
    restarted part resumes.
Then, per leaf on the PR3 sample (the r = 10 probe shots left out): rule 1, |k4 - k3| <= max(3, 1% of k4), and rule 2',
the band shots whose decision changes between tau3 and tau4 number at most the same limit, with no NaN at tau4. k3 and
k4 are the leaf's error counts with the band shots' decisions taken from tau3 and from tau4 (the ladder's elsewhere).
Verdict: "pass (tau4 rung)" or "still fails after tau4: excluded and flagged". A leaf whose tau4 parts do not all
finish gets "tau4 incomplete: leaf rows only".

Budget (10a). The hard cap is --guard-core-hours (13) in total, with the driver's CPU (its final ledger) included.
  - Projection: the driver's CPU + 5.5 x the CPU of each failing leaf's 5 extra parts (minus the CPU of tau4 parts
    already done, plus their own CPU). If it is over the cap, no tau4 part starts, the tally says so, exit 6.
  - While tau4 runs: if the driver's CPU plus every tau4 part's CPU (live) reaches the cap, every tau4 part is killed,
    no new one starts, and the unfinished leaves get "tau4 incomplete". Exit 6.
  - At most --max-heavy (4) heavy processes on the machine, whoever started them (pr3_runner.Procs, as PR3 counts);
    a part starts only if MemAvailable stays 3 GiB above its expected 2.5 GiB; each part runs under a 4 GiB RSS cap
    (pr3_common.run_capped).
  - A part that hits the memory cap or fails the tau3 check ends its leaf's tau4 (the leaf's other pending parts are
    skipped; leaf rows only). A part that crashes stops new starts; a rerun retries it.
Outputs: <res>/parts/p<j>/tau4_rung/<leaf>/<tag>/tau4_rung.json and .npz (idx, llr_t4, logp_t4, maxk_t4);
<res>/r10_tau4_tally.json (status, projection, CPU, verdicts with the rules' values, "final"); <state>/tau4_ledger.json.
Exits: 0 done (or no leaf needs tau4), 2 the driver or an earlier tau4 part is still running, 4 a tau3 check failed,
5 a part crashed (the tally is not final), 6 the cap, 7 the memory cap.

  NUMBA_CACHE_DIR=<scratch> python3 -I -B pr3_r10_tau4.py --root <extract> --res results/decoder/willow/pr3/r10_posthoc \\
      --ledger <driver's final ledger.json> --work <scratch>/tau4_work --state <scratch>/state --logs <scratch>/logs/tau4 \\
      --guard-core-hours 13 [--dry-run]
One part (what the scheduler starts): ... --worker <leaf> <part>
"""
import argparse, hashlib, json, os, signal, sys, threading, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("PR3_CODE") or os.path.dirname(HERE)            # code/decoder
for _p in (os.path.join(CODE, "supplementary"), CODE, os.path.join(CODE, "pr3"), HERE):
    sys.path.insert(0, _p)
import numpy as np
import pr3_common as C                                                # read-only
import pr3_runner as PRR                                              # read-only: its machine-wide heavy count (Procs)

R = 10
KMAX_T4 = 8_000_000
FACTOR = 5.5
NEED_GIB = 2.5
CHUNK = 5
LADDER_DN = C.LADDER.replace(":", "_").replace("/", "-")
WHAT = ("post hoc r = 10 follow-up, amendment 10a (willow_prereg_r3.md): tau4 = 1e-9 for an r = 10 leaf that still "
        "fails after tau3; PR3's kill stands and H7 stays not evaluated")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sigmoid(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-x.astype(np.float64)))


def decide(llr):
    return (np.nan_to_num(sigmoid(llr), nan=0.0) > 0.5).astype(np.uint8)


def nanmax_or(x, empty):
    return float(np.nanmax(x)) if x.size and not np.all(np.isnan(x)) else empty


def python_procs(script, *flags):
    """pids of live Python processes running script (an argument ending with it), with every flag among the arguments."""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == os.getpid():
            continue
        try:
            cl = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if (cl and b"python" in os.path.basename(cl[0]) and any(c.endswith(script) for c in cl[1:])
                and all(f in cl for f in flags)):
            out.append(int(p))
    return out


def driver_alive():
    """True if the r = 10 driver (pr3_r10_posthoc.py) is alive."""
    return bool(python_procs(b"pr3_r10_posthoc.py"))


def workers_alive():
    """pids of live tau4 parts (pr3_r10_tau4.py --worker), for example from an earlier scheduler that was stopped."""
    return python_procs(b"pr3_r10_tau4.py", b"--worker")


def save_npz(path, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- one part: the tau3 check and tau4
def run_rung(root, ldir, edir, odir, wdir, rel, ladder_name, log, check_shots=C.CHECK_SHOTS_PER_PART, kmax4=KMAX_T4,
             first=None):
    """The tau3 rebuild check and tau4 on one ladder output's band shots. first: decode only the first n band shots
    (tests only). Returns an exit code (0 done, 4 check failed)."""
    import stim
    from dem_utils import parse_dem
    from ml_sparse import SparseFrontierDecoder
    from willow_data import Experiment
    from decode_willow import KMAX_LADDER, _ml_out           # frozen; read-only
    import k2_extra_rung as K2X                               # unchanged; read-only (rungs())
    t1, t2, band = K2X.rungs(ladder_name)
    t3, t4 = t2 / 10, t2 / 100
    dn = ladder_name.replace(":", "_").replace("/", "-")
    summ = json.load(open(os.path.join(ldir, "summary.json")))
    assert summ.get("status") == "complete", f"{ldir}: the ladder is not complete"
    zpath = os.path.join(ldir, dn + ".npz")
    z = np.load(zpath); sh = np.load(os.path.join(ldir, "shots.npz"))
    idx = z["idx"].astype(np.int64); red = z["redecoded"].astype(bool)
    assert np.array_equal(idx, sh["idx"].astype(np.int64)), f"{ldir}: ladder and shots.npz select different shots"
    obs = sh["obs"].astype(np.uint8)
    ej = json.load(open(os.path.join(edir, "extra_rung.json")))
    zmd5 = md5(zpath)
    assert "tau3" in ej and ej.get("ladder_npz_md5") == zmd5, f"{edir}: the extra rung does not match the ladder output"
    epath = os.path.join(edir, "extra_rung.npz")
    emd5 = md5(epath)
    e = np.load(epath)
    b = np.where(red)[0]
    assert np.array_equal(e["idx"].astype(np.int64), idx[b]), f"{edir}: tau3's band shots differ from the ladder's band"
    l3 = e["llr_t3"]
    if first is not None:
        b, l3 = b[:first], l3[:first]
    folder = os.path.join(root, rel)
    dem_path = os.path.join(folder, summ["job"]["dem_file"])
    assert md5(dem_path) == summ["dem_md5"], f"{rel}: DEM md5 differs from summary.json"
    exp = Experiment(folder)
    det_all, obs_all = exp.detection_events(), exp.obs_flips_actual()
    assert np.array_equal(obs_all[idx, 0].astype(np.uint8), obs), f"{rel}: observed flips differ from shots.npz"
    detb = det_all[idx[b]]
    dm = parse_dem(stim.DetectorErrorModel.from_file(dem_path))
    D1 = SparseFrontierDecoder(dm, tau=t1, kmax=KMAX_LADDER)

    def deeper(tau, kmax):
        D = SparseFrontierDecoder.__new__(SparseFrontierDecoder)
        D.dm, D.tau, D.kmax, D.prog = dm, tau, kmax, D1.prog
        return D
    r = int(summ.get("rounds") or 0)
    nc = min(len(b), check_shots)
    log(f"{rel}: {len(idx)} shots, {len(b)} band shots, r = {r}; tau3 check at {t3:g} (kmax {KMAX_LADDER}) on {nc}")
    c0 = time.process_time()
    o3 = _ml_out(*deeper(t3, KMAX_LADDER).class_probs(detb[:nc])) if nc else {"llr": np.zeros(0, np.float32)}
    secc = time.process_time() - c0
    same = bool(np.array_equal(o3["llr"], l3[:nc], equal_nan=True))
    check = {"tau3": t3, "kmax": KMAX_LADDER, "n_checked": int(nc), "n_band": int(len(b)), "identical": same,
             "max_abs_llr_diff": float(np.nanmax(np.abs(o3["llr"].astype(np.float64) - l3[:nc]))) if nc else 0.0,
             "cpu_seconds": secc}
    rec = {"leaf": rel, "ladder_npz_md5": zmd5, "extra_rung_npz_md5": emd5, "check": check, "kmax_t4": kmax4,
           "first": first, "script_md5": {"pr3_r10_tau4.py": md5(os.path.abspath(__file__)),
                                          "k2_extra_rung.py": md5(K2X.__file__)}}
    if not same:
        log(f"  tau3 check FAILED (max |llr diff| {check['max_abs_llr_diff']:.3g}); tau4 not run")
        rec.update(verdict="not run: the rebuilt tau3 rung does not reproduce the stored tau3", made=now())
        C.write_json(os.path.join(odir, "tau4_rung.json"), rec)
        return 4
    log(f"  tau3 check passed ({secc:.0f} CPU s); tau4 = {t4:g} (kmax {kmax4}) on {len(b)} band shots, chunks of {CHUNK}")
    D4 = deeper(t4, kmax4)
    os.makedirs(wdir, exist_ok=True)
    llr4, logp4, mk4, sec4 = [], [], [], 0.0
    for k0 in range(0, len(b), CHUNK):
        cp = os.path.join(wdir, f"chunk_{k0:05d}.npz")
        n = min(CHUNK, len(b) - k0)
        if os.path.exists(cp):
            c = np.load(cp)
            if str(c["extra_md5"]) == emd5 and int(c["n"]) == n and int(c["kmax"]) == kmax4:
                llr4.append(c["llr"]); logp4.append(c["logp"]); mk4.append(c["maxk"]); sec4 += float(c["sec"])
                continue
        c0 = time.process_time()
        post, lp = D4.class_probs(detb[k0:k0 + n])
        o = _ml_out(post, lp)
        s = time.process_time() - c0
        mk = np.asarray(D4.last_maxk).copy()
        save_npz(cp, llr=o["llr"], logp=o["logp"], maxk=mk, sec=s, n=n, kmax=kmax4, extra_md5=emd5)
        llr4.append(o["llr"]); logp4.append(o["logp"]); mk4.append(mk); sec4 += s
        log(f"  chunk {k0}: {n} shots, {s:.0f} CPU s, max K {int(mk.max()) if n else 0}")
    cat = lambda v, dt: np.concatenate(v) if v else np.zeros(0, dt)
    llr4, logp4, mk4 = cat(llr4, np.float32), cat(logp4, np.float64), cat(mk4, np.int64)
    os.makedirs(odir, exist_ok=True)
    save_npz(os.path.join(odir, "tau4_rung.npz"), idx=idx[b].astype(np.int32), llr_t4=llr4, logp_t4=logp4, maxk_t4=mk4)
    rec["tau4"] = {"tau4": t4, "n_band": int(len(b)), "nan_tau4": int(np.isnan(llr4).sum()),
                   "max_K_tau4": int(mk4.max()) if len(b) else 0, "n_at_kmax_tau4": int(np.sum(mk4 >= kmax4)),
                   "cpu_seconds_tau4": sec4, "ms_per_round_per_band_shot_tau4": 1e3 * sec4 / max(len(b), 1) / max(r, 1)}
    rec.update(verdict="done", made=now())
    C.write_json(os.path.join(odir, "tau4_rung.json"), rec)
    log(f"  tau4 done: {sec4:.0f} CPU s, max K {rec['tau4']['max_K_tau4']}, {rec['tau4']['n_at_kmax_tau4']} at kmax")
    return 0


# ---------------------------------------------------------------- the scheduler
class Tau4:
    def __init__(self, a):
        self.a = a
        self.tag = f"rl_{a.design}"
        os.makedirs(a.state, exist_ok=True); os.makedirs(a.logs, exist_ok=True)
        self.lp = os.path.join(a.state, "tau4_ledger.json")
        try:
            self.ledger = json.load(open(self.lp))
        except Exception:
            self.ledger = {"units": {}}
        self.lock = threading.Lock()
        self.live = {}                   # key -> CPU seconds of the running part
        self.pg = {}                     # key -> process group
        self.stop = None                 # the cap: every running part is killed, no new part starts
        self.halt = None                 # a part crashed: no new part starts; a rerun retries
        self.procs = PRR.Procs()

    def others(self):
        """Heavy processes on the machine other than this scheduler's parts. PR3's rule: at most --max-heavy heavy
        processes run on the machine, whoever started them."""
        with self.lock:
            own = set(self.pg.values())
        return [p for p in self.procs.scan({os.getpid()}) if p["pgid"] not in own and p["pid"] not in own]

    def log(self, msg):
        print(f"{now()} {msg}", flush=True)

    def save(self):
        with self.lock:
            C.write_json(self.lp, self.ledger)

    def ldir(self, lf, j):
        return os.path.join(self.a.res, "parts", f"p{j}", "decode", lf, self.tag)

    def edir(self, lf, j):
        return os.path.join(self.a.res, "parts", f"p{j}", "extra_rung", lf, self.tag)

    def odir(self, lf, j):
        return os.path.join(self.a.res, "parts", f"p{j}", "tau4_rung", lf, self.tag)

    def cmd(self, lf, j):
        a = self.a
        return [sys.executable, "-I", "-B", os.path.abspath(__file__), "--root", a.root, "--res", a.res, "--ledger",
                a.ledger, "--work", a.work, "--state", a.state, "--logs", a.logs, "--design", a.design,
                "--guard-core-hours", str(a.guard_core_hours), "--worker", lf, str(j)]

    def done(self, lf, j):
        try:
            t = json.load(open(os.path.join(self.odir(lf, j), "tau4_rung.json")))
        except Exception:
            return False
        return ("tau4" in t and t.get("first") is None
                and t.get("extra_rung_npz_md5") == md5(os.path.join(self.edir(lf, j), "extra_rung.npz")))

    def run(self):
        a = self.a
        T = json.load(open(os.path.join(a.res, "r10_tally.json")))
        if not T.get("complete"):
            self.log("r10_tally.json is not complete: the driver has not finished"); return 2
        if driver_alive():
            self.log("the r = 10 driver is still running"); return 2
        w = workers_alive()
        if w:
            self.log(f"tau4 parts of an earlier run are still alive (pids {w}): wait for them or stop them first"); return 2
        DL = json.load(open(a.ledger))["units"]
        drv_h = sum((x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0) for x in DL.values()) / 3600
        failing = sorted(lf for lf, v in T["verdicts"].items() if str(v.get("verdict", "")).startswith("still fails"))
        U = self.ledger["units"]
        for lf in failing:
            for j in range(C.PARTS):
                k = f"tau4|{lf}|{j}"
                x = DL.get(f"extra|{lf}|{j}") or {}
                u = U.setdefault(k, {"leaf": lf, "part": j, "state": "pending", "cpu_s": 0.0, "lost_cpu_s": 0.0})
                u["extra_cpu_s"] = (x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0)
                if self.done(lf, j):
                    u["state"] = "done"
                elif u["state"] in ("running", "done"):     # a run lost in a restart: book what was seen, restart
                    u["lost_cpu_s"] += u.pop("seen_cpu_s", 0.0); u["state"] = "pending"
                elif u["state"].startswith(("failed: exit", "stopped")):   # a crash, or the cap: the projection decides
                    u["state"] = "pending"
        self.ledger.update(made=now(), what=WHAT, failing=failing, driver_core_h=drv_h, guard_core_h=a.guard_core_hours,
                           kmax_t4=KMAX_T4, factor=FACTOR)
        self.save()

        def will_run(u):                    # pending, and no part of its leaf has failed for good
            return u["state"] in ("pending", "running") and not any(
                x["leaf"] == u["leaf"] and x["state"] in ("failed: memory cap", "failed: tau3 check") for x in U.values())

        def projection():
            t4_used = sum(u["cpu_s"] + u["lost_cpu_s"] for u in U.values()) / 3600
            rest = sum(FACTOR * u["extra_cpu_s"] for u in U.values() if will_run(u)) / 3600
            return {"driver_h": drv_h, "tau4_used_h": t4_used, "tau4_remaining_h": rest,
                    "per_leaf_h": {lf: FACTOR * sum(U[f"tau4|{lf}|{j}"]["extra_cpu_s"] for j in range(C.PARTS)) / 3600
                                   for lf in failing},
                    "total_h": drv_h + t4_used + rest}
        P = projection()
        self.log(f"pr3_r10_tau4 (md5 {md5(os.path.abspath(__file__))}): {WHAT}; leaves still failing: "
                 + (", ".join(lf.split('/', 2)[-1] for lf in failing) or "none") + f"; projection {P}")
        if not failing:
            self.tally("not needed: no leaf still fails after tau3", P); return 0
        if a.dry_run:
            return 0
        if P["total_h"] > a.guard_core_hours:
            self.log(f"projection {P['total_h']:.2f} core-hours is over the cap {a.guard_core_hours}: no tau4 part starts")
            self.tally(f"not started: projection {P['total_h']:.2f} core-hours is over the cap {a.guard_core_hours}", P)
            return 6
        todo = [k for k, u in U.items() if u["state"] == "pending"]
        todo.sort(key=lambda k: -U[k]["extra_cpu_s"])
        threads = []

        def total_now():
            with self.lock:
                return drv_h + (sum(u["cpu_s"] + u["lost_cpu_s"] for u in U.values())
                                + sum(self.live.values())) / 3600

        def go(k):
            u = U[k]
            lf, j = u["leaf"], u["part"]
            env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
            logp = os.path.join(a.logs, f"tau4_{lf.split('/')[-2]}_{lf.split('/')[-1]}_p{j}.out")

            def on_poll(pid, rss, peak):
                cpu = C.group_usage(pid)[1]
                with self.lock:
                    self.pg[k] = pid; self.live[k] = cpu; u["seen_cpu_s"] = cpu
                if self.stop is None and total_now() >= a.guard_core_hours:
                    self.stop = f"the cap of {a.guard_core_hours} core-hours was reached"
                if self.stop:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            res = C.run_capped(self.cmd(lf, j), env, logp, int(a.cap_gib * C.GiB), poll=a.poll, on_poll=on_poll)
            with self.lock:
                self.live.pop(k, None); self.pg.pop(k, None)
                u.pop("seen_cpu_s", None)
                u["cpu_s"] += res["cpu_s"]; u["peak_rss"] = max(u.get("peak_rss") or 0, res["peak_rss"])
                u["ended"] = now(); u["exit"] = res["exit"]
                if res["exit"] == 0 and self.done(lf, j):
                    u["state"] = "done"
                elif res["capped"]:
                    u["state"] = "failed: memory cap"
                elif self.stop:
                    u["state"] = "stopped: " + self.stop
                elif res["exit"] == 4:
                    u["state"] = "failed: tau3 check"
                else:
                    u["state"] = f"failed: exit {res['exit']}"
                    self.halt = self.halt or f"tau4 {lf.split('/', 2)[-1]} p{j} failed (exit {res['exit']})"
            self.log(f"{u['state']}: tau4 {lf.split('/', 2)[-1]} p{j}, {res['cpu_s'] / 3600:.2f} core-hours, peak "
                     f"{res['peak_rss'] / C.GiB:.2f} GiB")
            self.save()

        last_wait = None
        while todo and not self.stop and not self.halt:
            running = [t for t in threads if t.is_alive()]
            oth = self.others()
            if len(running) + len(oth) < a.max_heavy and C.mem_available() - NEED_GIB * C.GiB >= a.min_free_gib * C.GiB:
                k = todo.pop(0)
                with self.lock:
                    dead = [x["state"] for x in U.values() if x["leaf"] == U[k]["leaf"]
                            and x["state"] in ("failed: memory cap", "failed: tau3 check")]
                if dead:                                   # the leaf's tau4 cannot finish: save the budget
                    with self.lock:
                        U[k]["state"] = f"skipped: another part of the leaf {dead[0]}"
                    self.log(f"skip tau4 {U[k]['leaf'].split('/', 2)[-1]} p{U[k]['part']}: {U[k]['state']}")
                    self.save(); continue
                if total_now() >= a.guard_core_hours:
                    self.stop = f"the cap of {a.guard_core_hours} core-hours was reached"; todo.insert(0, k); break
                with self.lock:
                    U[k]["state"] = "running"; U[k]["started"] = now()
                self.save()
                self.log(f"start tau4 {U[k]['leaf'].split('/', 2)[-1]} p{U[k]['part']} (projected "
                         f"{FACTOR * U[k]['extra_cpu_s'] / 3600:.2f} core-hours)")
                t = threading.Thread(target=go, args=(k,), daemon=True); t.start(); threads.append(t)
            else:
                if oth and (len(running), len(oth)) != last_wait:
                    self.log(f"waiting: {len(running)} tau4 parts and {len(oth)} other heavy processes are running")
                last_wait = (len(running), len(oth))
                time.sleep(a.poll)
        for t in threads:
            t.join()
        P = projection()
        st = [u["state"] for u in U.values()]
        if self.stop:
            status, code = f"stopped: {self.stop}", 6
        elif "failed: memory cap" in st:
            status, code = "a tau4 part hit the memory cap", 7
        elif "failed: tau3 check" in st:
            status, code = "a tau3 rebuild check failed", 4
        elif any(s.startswith("failed") for s in st):
            status, code = "a tau4 part failed: a rerun retries it", 5
        else:
            status, code = "all tau4 parts done", 0
        self.tally(status, P)
        self.log(f"exit {code}: {status}")
        return code

    # ------------------------------------------------------------ verdicts on the PR3 sample
    def rules(self, lf, r=R):
        idx, obs, df, red, part_of = [], [], [], [], []
        for j in range(C.PARTS):
            z = np.load(os.path.join(self.ldir(lf, j), LADDER_DN + ".npz"))
            sh = np.load(os.path.join(self.ldir(lf, j), "shots.npz"))
            assert np.array_equal(z["idx"], sh["idx"])
            idx.append(z["idx"].astype(np.int64)); obs.append(sh["obs"].astype(np.uint8)); df.append(z["pred"][:, 0])
            red.append(z["redecoded"].astype(bool)); part_of.append(np.full(len(z["idx"]), j))
        cat = np.concatenate
        idx, obs, df, red, part_of = cat(idx), cat(obs), cat(df).astype(np.uint8), cat(red), cat(part_of)
        m = C.in_sample(idx, r)
        pos = {int(x): i for i, x in enumerate(idx)}
        bi, l3, l4, mk = [], [], [], []
        for j in range(C.PARTS):
            e = np.load(os.path.join(self.edir(lf, j), "extra_rung.npz"))
            f = np.load(os.path.join(self.odir(lf, j), "tau4_rung.npz"))
            assert np.array_equal(e["idx"], f["idx"]), f"{lf} p{j}: tau4's band shots differ from tau3's"
            bj = np.array([pos[int(x)] for x in e["idx"]], np.int64)
            assert np.array_equal(np.sort(bj), np.where(red & (part_of == j))[0]), f"{lf} p{j}: band shots differ"
            bi.append(bj); l3.append(e["llr_t3"]); l4.append(f["llr_t4"]); mk.append(f["maxk_t4"])
        bi, l3, l4, mk = cat(bi), cat(l3), cat(l4), cat(mk)
        keep = m[bi]; bk = bi[keep]
        d3, d4 = decide(l3), decide(l4)
        kf = int(np.sum((df != obs) & m))
        base = kf - int(np.sum(df[bk] != obs[bk]))
        k3 = base + int(np.sum(d3[keep] != obs[bk]))
        k4 = base + int(np.sum(d4[keep] != obs[bk]))
        ch = int(np.sum(d4[keep] != d3[keep])); nan4 = int(np.isnan(l4[keep]).sum())
        lim = max(3, 0.01 * k4)
        dP = np.abs(sigmoid(l4) - sigmoid(l3))[keep]
        v = {"errors_tau2": kf, "errors_tau3": k3, "errors_tau4": k4, "abs_change_errors": abs(k4 - k3),
             "rule1": abs(k4 - k3) <= lim, "decisions_changed": ch, "rule2prime": ch <= lim and nan4 == 0, "limit": lim,
             "nan_tau4": nan4, "max_abs_dP_descriptive": nanmax_or(dP, None) if nan4 < dP.size else None,
             "band_shots_sample": int(keep.sum()), "kmax_t4": KMAX_T4, "band_shots_at_kmax_tau4": int(np.sum(mk[keep] >= KMAX_T4))}
        v["passes"] = v["rule1"] and v["rule2prime"]
        return v

    def tally(self, status, P):
        a = self.a
        T = json.load(open(os.path.join(a.res, "r10_tally.json")))
        U = self.ledger["units"]
        verdicts = {}
        for lf in self.ledger.get("failing", []):
            parts = [U[f"tau4|{lf}|{j}"] for j in range(C.PARTS)]
            if all(u["state"] == "done" for u in parts):
                v = self.rules(lf)
                v["errors_tau3_equal_r10_tally"] = v["errors_tau3"] == ((T["verdicts"][lf].get("extra") or {}).get("errors_tau3"))
                v["verdict"] = "pass (tau4 rung)" if v["passes"] else "still fails after tau4: excluded and flagged"
                J = [json.load(open(os.path.join(self.odir(lf, j), "tau4_rung.json"))) for j in range(C.PARTS)]
                v["tau3_check"] = {"identical_in_every_part": all(x["check"].get("identical") is True for x in J),
                                   "n_checked": sum(int(x["check"].get("n_checked") or 0) for x in J)}
                v["tau4_parts"] = {"n_band": sum(int(x["tau4"].get("n_band") or 0) for x in J),
                                   "max_K": max(int(x["tau4"].get("max_K_tau4") or 0) for x in J),
                                   "n_at_kmax": sum(int(x["tau4"].get("n_at_kmax_tau4") or 0) for x in J)}
            else:
                v = {"verdict": "tau4 incomplete: leaf rows only",
                     "parts": {f"p{u['part']}": u["state"] for u in parts}}
            v["cpu_core_h"] = sum(u["cpu_s"] + u["lost_cpu_s"] for u in parts) / 3600
            v["peak_rss_gib"] = max((u.get("peak_rss") or 0) for u in parts) / C.GiB
            verdicts[lf] = v
        t = {"made": now(), "what": WHAT, "status": status, "design": a.design, "guard_core_h": a.guard_core_hours,
             "kmax_t4": KMAX_T4, "factor": FACTOR, "projection": P, "failing_after_tau3": self.ledger.get("failing", []),
             "verdicts": verdicts,
             "tau4_cpu_core_h": sum(u["cpu_s"] + u["lost_cpu_s"] for u in U.values()) / 3600,
             "total_cpu_core_h": P["driver_h"] + sum(u["cpu_s"] + u["lost_cpu_s"] for u in U.values()) / 3600,
             "final": status.startswith(("not started", "not needed", "stopped")) or all(
                 u["state"] == "done" or u["state"].startswith(("failed: memory cap", "failed: tau3 check", "skipped"))
                 for u in U.values()),
             "code_md5": {"pr3_r10_tau4.py": md5(os.path.abspath(__file__)), "pr3_common.py": md5(C.__file__),
                          "pr3_runner.py": md5(PRR.__file__)}}
        C.write_json(os.path.join(a.res, "r10_tau4_tally.json"), t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--ledger", required=True, help="the r = 10 driver's final ledger.json")
    ap.add_argument("--work", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--design", default="25k", choices=sorted(C.DESIGNS))
    ap.add_argument("--guard-core-hours", type=float, required=True)
    ap.add_argument("--max-heavy", type=int, default=4)
    ap.add_argument("--min-free-gib", type=float, default=3.0)
    ap.add_argument("--cap-gib", type=float, default=4.0)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker", nargs=2, metavar=("LEAF", "PART"))
    a = ap.parse_args()
    if os.path.normpath(a.res).endswith(os.path.join("pr3", "main")):
        sys.exit("--res must be the r = 10 follow-up's folder (r10_posthoc), not PR3's main folder")
    if a.worker:
        lf, j = a.worker[0], int(a.worker[1])
        S = Tau4.__new__(Tau4); S.a = a; S.tag = f"rl_{a.design}"
        log = lambda msg: print(f"{now()} {msg}", flush=True)
        sys.exit(run_rung(a.root, S.ldir(lf, j), S.edir(lf, j), S.odir(lf, j), os.path.join(a.work, f"p{j}", lf), lf,
                          C.LADDER, log))
    sys.exit(Tau4(a).run())


if __name__ == "__main__":
    main()
