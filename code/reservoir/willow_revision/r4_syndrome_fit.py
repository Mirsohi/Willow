"""R4 of the revision (review 1.6): a matched comparison of the refit objectives at the class level. POST HOC revision
analysis, not a pre-registered test; declared in notes/reservoir/willow_revision_declarations.md before it ran.

reservoir, 2026-10-13. S-hat is D-hat (round 2's family F1, code/decoder/round2/run_f1.py) with one change, the objective:
the syndrome log-likelihood sum_s ln P_theta(s) of the fit shots instead of the conditional sum_s ln P_theta(L_obs | s).
Everything else is decoder's code, imported read-only and called as run_f1 calls it:
  parameterization  T-hat's classes per leaf (fit_willow.class_labels, first/last-layer split, 30 expected events in
                    5,000 fit shots); one scale per class name shared by the 4 W5 leaves of a series (pooled per
                    series); p_j = p0_j * theta_c on the shipped RL DEM;
  fit shots         T-hat's 5,000 per leaf (round 1's fit_data.npz fit_idx), 20,000 per series;
  regularization    none (as D-hat);
  optimiser         cond_fit.fit_series with run_f1's settings: Fisher scoring in ln theta from theta = 1 with the OPG
                    information, Powell-damped BFGS updates, step halving, steps capped at 1 in ln theta, stop when every
                    component of the next step is below 0.02 SE, cap 25 iterations (run_f1.MAX_ITER), a class with
                    singular information fixed at 1;
  evaluation        the same held-out halves (round 1's test_idx, 25,000 per leaf), decoded by exact ML under S-hat
                    (cond_fit.Leaf.decode_test).
The per-shot syndrome term and its score come from the same exact forward-backward pass as D-hat's
(fb_cond.ForwardBackwardCond.run: lp_marg = ln P(s) and g_marg, the score of ln P(s) through the same Jacobian), so the
objective is the only change. cond_fit's log lines say "ln P(L|s)"; for S-hat that number is the mean ln P(s). Its
"se_exp_ln" (the conditional information) has no meaning for S-hat; the OPG SEs are the ones used.

Outputs, in run_f1's layout (restartable in the same way: a finished series is skipped, a fit resumes from its checkpoint
OUT/<series>/fit_ckpt.ckpt.npz):
  OUT/<series>/fit_series.json     scales, SEs, convergence, history, CPU (objective "syndrome ln P(s)")
  OUT/<leaf>/s_data.npz            test_idx, obs_test, S_pred, S_llr (float32), S_logp (ln P(s) under S-hat)
  OUT/<leaf>/eval.json             held-out errors of ML(S-hat), ML(RL), ML(T-hat), ML(D-hat) and paired cells
Usage: NUMBA_CACHE_DIR=<scratch> python3 -I -B r4_syndrome_fit.py fit EXTRACT RES OUT PART GUARD_S
         EXTRACT: the archive extract root; RES: results/decoder/willow; PART i/n: series i, i+n, ... of the 18;
         GUARD_S: R4's CPU so far (every series under OUT, counted as run_f1 counts F1's) is checked before each series
         and before every evaluation; when it reaches GUARD_S the process keeps its checkpoint and exits with status 3.
       python3 -I -B r4_syndrome_fit.py test EXTRACT RES OUT_JSON
         before the declaration, on 300 fit shots of one leaf: S-hat's objective at theta = 1 against decoder's stored
         ln P(s) of the same shots under the RL DEM, and its score against central finite differences.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.realpath(os.path.join(HERE, "..", "..", "decoder", "round2")))
import numpy as np  # noqa: E402
import run_f1 as RF  # noqa: E402  (decoder's, read-only; sets the path to cond_fit, fb_cond and the frozen modules)
from cond_fit import Leaf, fit_series, remove_ckpt, write_json  # noqa: E402
from fb_cond import ForwardBackwardCond  # noqa: E402
import analyze_willow as AW  # noqa: E402  (frozen: count2)

ARCH = RF.ARCH
OBJECTIVE = "syndrome log-likelihood sum_s ln P_theta(s) (S-hat); D-hat's F1 otherwise unchanged"


class SyndromeLeaf(Leaf):
    """cond_fit.Leaf with the syndrome log-likelihood as the per-shot objective (the only change)."""

    def evaluate(self, phi_series, K):
        th = self.theta(phi_series)
        dm = self.model.dm_at(th)
        fb = ForwardBackwardCond(dm, order=self.model.order)
        r = fb.run(self.det_fit, self.L_fit, J=self.model.jacobian(th))
        G = np.zeros((len(self.L_fit), K))
        G[:, self.cols] = r["g_marg"]
        return r["lp_marg"], G


def load(a, series, r, log):
    lf = RF.load_leaf(a, series, r, log)
    lf.__class__ = SyndromeLeaf
    return lf


def cmd_fit(extract, res, out, part, guard):
    a = argparse.Namespace(extract=extract, res=res, out=out)
    guard = float(guard)
    os.makedirs(out, exist_ok=True)
    i, n = (int(x) for x in part.split("/"))
    ser = RF.all_series(res)[i::n]
    f1 = os.path.join(res, "round2", "f1")
    logf = open(os.path.join(out, f"r4_part{i}of{n}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"R4 part {part}: {len(ser)} series; objective: {OBJECTIVE}; guard {guard:.0f} CPU-s (all of R4 under {out}); "
        f"R4 so far {RF.f1_cpu_seconds(out, {}):.0f} CPU-s; nice {os.nice(0)}")
    cache = {}
    over = lambda: RF.f1_cpu_seconds(out, cache) >= guard  # noqa: E731
    for series in ser:
        if RF.done(a, series):
            log(f"{series}: done already")
            continue
        if over():
            log(f"guard reached before {series}: R4 has used {RF.f1_cpu_seconds(out, cache):.0f} CPU-s; stopping")
            sys.exit(3)
        t0 = time.time()
        log(f"{series}: loading")
        leaves = [load(a, series, r, log) for r in RF.W5]
        ck = os.path.join(out, series, "fit_ckpt")
        fr = fit_series(leaves, log=log, ckpt=ck, max_iter=RF.MAX_ITER, stop=over)
        if fr.get("stopped"):
            log(f"guard reached during {series}: R4 has used {RF.f1_cpu_seconds(out, cache):.0f} CPU-s; stopping "
                f"(checkpoint kept)")
            sys.exit(3)
        fr.update(series=series, leaves=[lf.name for lf in leaves], cpu_seconds_fit=fr["cpu_seconds_fit_total"],
                  wall_seconds_fit_this_process=time.time() - t0, objective=OBJECTIVE,
                  note="se_exp_ln has no meaning for S-hat (conditional information); use se_opg_ln")
        phi = np.array(fr["phi_hat"])
        cpu_dec = 0.0
        for lf in leaves:
            d = os.path.join(out, lf.name)
            pe = os.path.join(d, "eval.json")
            if os.path.exists(pe) and os.path.exists(os.path.join(d, "s_data.npz")):
                old = json.load(open(pe))
                if old.get("fingerprint") == fr["fingerprint"] and np.allclose(
                        [old["theta"][nm] for nm in lf.names], lf.theta(phi), rtol=1e-12, atol=0):
                    cpu_dec += old["cpu_seconds_decode"]
                    log(f"  {lf.name}: decoded already")
                    continue
            c1 = time.process_time()
            pred, llr, logp = lf.decode_test(phi)
            o = lf.extra["obs_test"].astype(np.int8)
            S = lf.extra["shipped_pred"].astype(np.int8)
            T = lf.extra["fitted_pred"].astype(np.int8)
            zd = np.load(os.path.join(f1, lf.name, "d_data.npz"))
            if not np.array_equal(zd["test_idx"], lf.extra["test_idx"].astype(np.int32)):
                raise SystemExit(f"{lf.name}: D-hat's held-out shots differ")
            D = zd["D_pred"].astype(np.int8)
            p8 = pred.astype(np.int8)
            ev = {"leaf": lf.name, "rounds": lf.rounds, "n_test": int(len(o)), "series": series,
                  "objective": OBJECTIVE, "theta": dict(zip(lf.names, lf.theta(phi).tolist())),
                  "errors": {"S": int(np.sum(p8 != o)), "RL": int(np.sum(S != o)), "T": int(np.sum(T != o)),
                             "D": int(np.sum(D != o))},
                  "cells_RL_vs_S": AW.count2(S, p8, o), "cells_D_vs_S": AW.count2(D, p8, o),
                  "cells_T_vs_S": AW.count2(T, p8, o), "nan_posteriors": int(np.isnan(llr).sum()),
                  "cpu_seconds_decode": time.process_time() - c1, "converged": fr["converged"],
                  "fingerprint": fr["fingerprint"]}
            cpu_dec += ev["cpu_seconds_decode"]
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "s_data.tmp.npz"), test_idx=lf.extra["test_idx"].astype(np.int32),
                                obs_test=lf.extra["obs_test"], S_pred=pred, S_llr=llr, S_logp=logp)
            os.replace(os.path.join(d, "s_data.tmp.npz"), os.path.join(d, "s_data.npz"))
            write_json(pe, ev)
            log(f"  {lf.name}: held-out errors S {ev['errors']['S']}, RL {ev['errors']['RL']}, D {ev['errors']['D']}, "
                f"T {ev['errors']['T']}")
        fr["cpu_seconds_decode"] = cpu_dec
        fr["cpu_seconds_total"] = fr["cpu_seconds_fit"] + cpu_dec
        fr["wall_seconds_this_process"] = time.time() - t0
        write_json(os.path.join(out, series, "fit_series.json"), fr)
        remove_ckpt(ck)
        log(f"{series}: done, {fr['cpu_seconds_total'] / 60:.1f} CPU min (converged={fr['converged']}, "
            f"{fr['iterations']} steps, {fr['n_eval']} evaluations); R4 so far "
            f"{RF.f1_cpu_seconds(out, cache) / 3600:.2f} core-hours")


def cmd_test(extract, res, out_json):
    a = argparse.Namespace(extract=extract, res=res, out=None)
    series = RF.all_series(res)
    s = [x for x in series if x.endswith("d3_at_q4_5/X")][0]
    lf = load(a, s, 10, print)
    n = 300
    lf.det_fit, lf.L_fit = lf.det_fit[:n], lf.L_fit[:n]
    names = sorted(lf.names)
    lf.bind(names)
    K = len(names)
    t0 = time.process_time()
    lp, G = lf.evaluate(np.zeros(K), K)
    c_eval = time.process_time() - t0
    fi = np.load(os.path.join(res, "fit", lf.name, "rl_tsplit", "fit_data.npz"))["fit_idx"][:n].astype(np.int64)
    ml = np.load(os.path.join(res, "decode", lf.name, "rl", "ml_exact.npz"))
    pos = {int(i): k for k, i in enumerate(ml["idx"])}
    ref = ml["logp"][[pos[int(i)] for i in fi]]
    out = {"label": "R4 pre-declaration test (reservoir, 2026-10-13); 300 fit shots of " + lf.name,
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "classes": K,
           "lnPs_vs_decoder_RL_logp_max_abs": float(np.max(np.abs(lp - ref))), "eval_cpu_s": c_eval, "fd": []}
    h = 1e-5
    for c in sorted(set(np.argsort(-np.abs(G.sum(0)))[:3].tolist() + [0])):
        e = np.zeros(K)
        e[c] = h
        lp_p, _ = lf.evaluate(e, K)
        lp_m, _ = lf.evaluate(-e, K)
        fd = (lp_p.sum() - lp_m.sum()) / (2 * h)
        out["fd"].append({"class": names[c], "score_sum": float(G[:, c].sum()), "finite_difference": float(fd),
                          "rel_diff": float(abs(fd - G[:, c].sum()) / max(abs(fd), 1e-12))})
    print(json.dumps(out, indent=1), flush=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    x = sys.argv[1:]
    if x and x[0] == "fit" and len(x) == 6:
        cmd_fit(*x[1:])
    elif x and x[0] == "test" and len(x) == 4:
        cmd_test(*x[1:])
    else:
        raise SystemExit(__doc__)
