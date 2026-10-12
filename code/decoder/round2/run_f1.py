#!/usr/bin/env python3
"""Round 2, family F1 on real data (notes/decoder/willow_prereg_r2.md): per series (patch x basis), fit D-hat on the
5,000 round-1 fit shots of each of its 4 W5 leaves (cond_fit.fit_series, pooled), then decode the leaves' held-out
halves under D-hat. Reads the local archive extract (python3 -I) and round 1's fit_data.npz / fit.json (read-only).

Checks per leaf before fitting: the DEM file's md5 equals fit.json's; the rebuilt classes and their fault counts equal
fit.json's; the held-out logical outcomes equal fit_data.npz's obs_test.
Outputs (restartable; a series whose outputs exist is skipped; the fit checkpoints to <out>/<series>/fit_ckpt.ckpt.npz
after every accepted step and resumes from it; a leaf whose eval.json matches the fit is not decoded again):
  <out>/<series>/fit_series.json                 scales, SEs, convergence, history, timings
  <out>/<leaf>/d_data.npz                        test_idx, obs_test, D_pred, D_llr (float32), D_logp
  <out>/<leaf>/eval.json                         held-out counts of ML(D-hat), ML(RL), ML(T-hat), paired cells, timings
CPU budget (--cpu-budget-hours H; the coordinator's 12-core-hour guard): F1's CPU so far is the sum, over every series
under --out, of fit_series.json's cpu_seconds_total, or for an unfinished series its checkpoint's fit CPU plus its
eval.json decode CPU. It is checked before each series and before every new evaluation; when it reaches H the process
logs "CPU budget reached", keeps its checkpoint and exits with status 3. CPU spent after a process's last checkpoint
is not counted (at most one evaluation per process).
Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B run_f1.py --extract <root> --res results/decoder/willow \
         --out results/decoder/willow/round2/f1 --part 0/2 --cpu-budget-hours 12   (see RELAUNCH.md)
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import Leaf, load_dm, t_classes, fit_series, write_json, remove_ckpt, read_ckpt_state, RL_DEM, W5
from willow_data import Experiment
import analyze_willow as AW            # frozen: count2

ARCH = "google_105Q_surface_code_d3_d5_d7"
MAX_ITER = 25          # PR2 section 7, 2026-10-10: cap raised from 15 before any real-data fit


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def all_series(res):
    out = set()
    for dp, dn, fn in os.walk(os.path.join(res, "fit", ARCH)):
        if os.path.basename(dp) == "rl_tsplit" and "fit_data.npz" in fn:
            rel = os.path.relpath(os.path.dirname(dp), os.path.join(res, "fit"))     # ARCH/d3_at_q../B/rNN
            out.add(os.path.dirname(rel))
    return sorted(out)


def load_leaf(a, series, r, log):
    rel = f"{series}/r{r}"
    fp = os.path.join(a.res, "fit", rel, "rl_tsplit")
    fj = json.load(open(os.path.join(fp, "fit.json")))
    z = np.load(os.path.join(fp, "fit_data.npz"))
    folder = os.path.join(a.extract, rel)
    dem_path = os.path.join(folder, RL_DEM)
    assert md5_file(dem_path) == fj["dem_md5"], f"{rel}: DEM md5 differs from fit.json"
    dm = load_dm(dem_path)
    labels, _ = t_classes(dm, len(z["fit_idx"]))
    names, counts = np.unique(labels, return_counts=True)
    assert sorted(names.tolist()) == sorted(fj["classes"]), f"{rel}: classes differ from fit.json"
    assert all(int(c) == int(fj["n_faults_per_class"][nm]) for nm, c in zip(names, counts)), f"{rel}: class sizes differ"
    exp = Experiment(folder)
    det_all, obs_all = exp.detection_events(), exp.obs_flips_actual()
    fi, ti = z["fit_idx"].astype(np.int64), z["test_idx"].astype(np.int64)
    assert np.array_equal(obs_all[ti, 0].astype(np.uint8), z["obs_test"]), f"{rel}: held-out outcomes differ"
    extra = {"rel": rel, "test_idx": ti, "obs_test": z["obs_test"], "shipped_pred": z["shipped_pred"],
             "fitted_pred": z["fitted_pred"], "dem_md5": fj["dem_md5"]}
    lf = Leaf(rel, r, dm, labels, det_all[fi], obs_all[fi, 0], det_all[ti], obs_all[ti, 0], extra=extra)
    log(f"  {rel}: {dm.num_faults} faults, {len(lf.names)} classes (= fit.json), fit {len(fi)} / held-out {len(ti)} shots")
    return lf


def f1_cpu_seconds(out, cache):
    """F1's CPU so far (see the module docstring); cache keeps each series' last readable value."""
    for dp, dn, fn in os.walk(out):
        if "fit_series.json" in fn:
            try:
                cache[dp] = float(json.load(open(os.path.join(dp, "fit_series.json")))["cpu_seconds_total"])
            except Exception:
                pass
        elif "fit_ckpt.ckpt.npz" in fn:
            st = read_ckpt_state(os.path.join(dp, "fit_ckpt"))
            if st is None:
                continue
            dec = 0.0
            for r in W5:
                try:
                    dec += float(json.load(open(os.path.join(dp, f"r{r}", "eval.json")))["cpu_seconds_decode"])
                except Exception:
                    pass
            cache[dp] = float(st["cpu_seconds"]) + dec
    return sum(cache.values())


def done(a, series):
    if not os.path.exists(os.path.join(a.out, series, "fit_series.json")):
        return False
    return all(os.path.exists(os.path.join(a.out, f"{series}/r{r}", "eval.json")) for r in W5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1", help="i/n: this process takes series i, i+n, i+2n, ...")
    ap.add_argument("--series", default=None, help="comma-separated series (overrides --part)")
    ap.add_argument("--cpu-budget-hours", type=float, default=None, help="stop when F1's CPU so far reaches this")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    i, n = (int(x) for x in a.part.split("/"))
    ser = a.series.split(",") if a.series else all_series(a.res)[i::n]
    logf = open(os.path.join(a.out, f"run_f1_part{i}of{n}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log(f"part {a.part}: {len(ser)} series: {', '.join(s.split('/', 1)[1] for s in ser)}")
    cache = {}
    budget = None if a.cpu_budget_hours is None else a.cpu_budget_hours * 3600.0
    over = lambda: budget is not None and f1_cpu_seconds(a.out, cache) >= budget

    def stop_budget(where):
        log(f"CPU budget reached: F1 has used {f1_cpu_seconds(a.out, cache) / 3600:.2f} of {a.cpu_budget_hours} "
            f"core-hours; stopping {where} (checkpoints kept). Tell decoder; do not raise the budget without @main.")
        sys.exit(3)
    if budget is not None:
        log(f"CPU budget {a.cpu_budget_hours} core-hours; F1 so far {f1_cpu_seconds(a.out, cache) / 3600:.2f}")
    for series in ser:
        if done(a, series):
            log(f"{series}: done already"); continue
        if over():
            stop_budget(f"before {series}")
        t0 = time.time(); c0 = time.process_time()
        log(f"{series}: loading")
        leaves = [load_leaf(a, series, r, log) for r in W5]
        ck = os.path.join(a.out, series, "fit_ckpt")
        fr = fit_series(leaves, log=log, ckpt=ck, max_iter=MAX_ITER, stop=over)
        if fr.get("stopped"):
            stop_budget(f"during {series}")
        fr.update(series=series, leaves=[lf.name for lf in leaves], cpu_seconds_fit=fr["cpu_seconds_fit_total"],
                  wall_seconds_fit_this_process=time.time() - t0, prereg="notes/decoder/willow_prereg_r2.md")
        phi = np.array(fr["phi_hat"])
        cpu_dec = 0.0
        for lf in leaves:
            d = os.path.join(a.out, lf.name)
            pe = os.path.join(d, "eval.json")
            if os.path.exists(pe) and os.path.exists(os.path.join(d, "d_data.npz")):
                old = json.load(open(pe))
                if old.get("fingerprint") == fr["fingerprint"] and np.allclose(
                        [old["theta"][nm] for nm in lf.names], lf.theta(phi), rtol=1e-12, atol=0):
                    cpu_dec += old["cpu_seconds_decode"]
                    log(f"  {lf.name}: decoded already"); continue
            c1 = time.process_time()
            pred, llr, logp = lf.decode_test(phi)
            o = lf.extra["obs_test"].astype(np.int8)
            S = lf.extra["shipped_pred"].astype(np.int8); T = lf.extra["fitted_pred"].astype(np.int8)
            ev = {"leaf": lf.name, "rounds": lf.rounds, "n_test": int(len(o)), "series": series,
                  "theta": dict(zip(lf.names, lf.theta(phi).tolist())),
                  "errors": {"D": int(np.sum(pred != o)), "RL": int(np.sum(S != o)), "T": int(np.sum(T != o))},
                  "cells_RL_vs_D": AW.count2(S, pred.astype(np.int8), o), "cells_T_vs_D": AW.count2(T, pred.astype(np.int8), o),
                  "nan_posteriors": int(np.isnan(llr).sum()), "cpu_seconds_decode": time.process_time() - c1,
                  "converged": fr["converged"], "fingerprint": fr["fingerprint"]}
            cpu_dec += ev["cpu_seconds_decode"]
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "d_data.tmp.npz"), test_idx=lf.extra["test_idx"].astype(np.int32),
                                obs_test=lf.extra["obs_test"], D_pred=pred, D_llr=llr, D_logp=logp)
            os.replace(os.path.join(d, "d_data.tmp.npz"), os.path.join(d, "d_data.npz"))
            write_json(os.path.join(d, "eval.json"), ev)
            log(f"  {lf.name}: held-out errors D {ev['errors']['D']}, RL {ev['errors']['RL']}, T {ev['errors']['T']}")
        fr["cpu_seconds_decode"] = cpu_dec
        fr["cpu_seconds_total"] = fr["cpu_seconds_fit"] + cpu_dec
        fr["wall_seconds_this_process"] = time.time() - t0
        write_json(os.path.join(a.out, series, "fit_series.json"), fr)
        remove_ckpt(ck)
        log(f"{series}: done, {fr['cpu_seconds_total'] / 60:.1f} CPU min (converged={fr['converged']}, "
            f"{fr['iterations']} steps, {fr['n_eval']} evaluations); F1 so far "
            f"{f1_cpu_seconds(a.out, cache) / 3600:.2f} core-hours")


if __name__ == "__main__":
    main()
