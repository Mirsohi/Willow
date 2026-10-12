#!/usr/bin/env python3
"""Kernel tests for fb_cond (round 2, notes/decoder/willow_prereg_r2.md section 5), on a shipped Willow RL DEM with
shots sampled from the DEM itself (no real shots are read):

1. the joint part (ln P(s, L_obs) and its score) equals the frozen ml_fb_fast run with L, within 1e-10 relative;
2. the marginal part (ln P(s) and its score) equals the frozen run without L, within 1e-10 relative;
3. ln P(L_obs | s) equals the exact decoder's posterior (ml_exact_fast), within 1e-10;
4. the conditional score equals central finite differences of sum ln P(L_obs | s), within 1e-5 relative;
5. timing: the new kernel against one frozen run.
The class scales are set to random values in [0.6, 1.6] so that the test is not at theta = 1.

Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B test_fb_cond.py --dem <leaf>/decoding_results/.../error_model.dem
"""
import argparse, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import load_dm, t_classes, tolist
from fit_noise import DemClassScale
from ml_fb_fast import ForwardBackwardFast
from ml_exact_fast import FastExactDecoder
from mismatch import dm_to_stim
from fb_cond import ForwardBackwardCond


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True, action="append")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-time", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res, ok_all = [], True
    for path in a.dem:
        dm0 = load_dm(path)
        labels, _ = t_classes(dm0)
        model = DemClassScale(dm0, classes=labels)
        rng = np.random.default_rng(a.seed)
        th = rng.uniform(0.6, 1.6, len(model.names))
        dm = model.dm_at(th)
        det, obs, _ = dm_to_stim(dm).compile_sampler(seed=a.seed).sample(max(a.n, a.n_time))
        det = det.astype(np.uint8); L = obs[:, 0].astype(np.int64)
        d, Lt = det[:a.n], L[:a.n]
        J = model.jacobian(th)
        new = ForwardBackwardCond(dm, order=model.order).run(d, Lt, J)
        old = ForwardBackwardFast(dm, order=model.order)
        rj = old.run(d, L=Lt, J=J); rm = old.run(d, L=None, J=J)
        post, _ = FastExactDecoder(dm, order=model.order).class_probs(d)
        cll_dec = np.log(post[np.arange(len(Lt)), Lt])
        # finite differences of sum ln P(L | s) in phi = ln theta
        h = 1e-4
        fd = np.zeros(len(th))
        for c in range(len(th)):
            tp = th.copy(); tm = th.copy(); tp[c] *= np.exp(h); tm[c] *= np.exp(-h)
            fp = ForwardBackwardCond(model.dm_at(tp), order=model.order).run(d, Lt, model.jacobian(tp))["cll"].sum()
            fm = ForwardBackwardCond(model.dm_at(tm), order=model.order).run(d, Lt, model.jacobian(tm))["cll"].sum()
            fd[c] = (fp - fm) / (2 * h)
        an = new["g_cond"].sum(axis=0)
        r = {"dem": path, "faults": dm.num_faults, "classes": len(th), "shots": a.n,
             "joint_logp": rel_err(new["lp_joint"], rj["logp"]), "joint_score": rel_err(new["g_joint"], rj["grads"]),
             "marg_logp": rel_err(new["lp_marg"], rm["logp"]), "marg_score": rel_err(new["g_marg"], rm["grads"]),
             "cll_vs_decoder_abs": float(np.max(np.abs(new["cll"] - cll_dec))),
             "fd_score_rel": float(np.max(np.abs(an - fd)) / max(np.max(np.abs(fd)), 1e-300)),
             "mean_cll": float(new["cll"].mean())}
        # timing on n_time shots (after compilation)
        dT, LT = det[:a.n_time], L[:a.n_time]
        t0 = time.process_time(); ForwardBackwardCond(dm, order=model.order).run(dT, LT, J); t_new = time.process_time() - t0
        t0 = time.process_time(); old.run(dT, L=LT, J=J); t_old = time.process_time() - t0
        rounds = max(c[2] for c in dm0.coords.values() if len(c) >= 3) - min(c[2] for c in dm0.coords.values() if len(c) >= 3)
        r.update(cpu_s_new=t_new, cpu_s_frozen_one_run=t_old, ratio=t_new / t_old, timing_shots=a.n_time,
                 ms_per_shot_new=1e3 * t_new / a.n_time, detector_layers_span=rounds)
        r["pass"] = bool(r["joint_logp"] < 1e-10 and r["joint_score"] < 1e-10 and r["marg_logp"] < 1e-10 and
                         r["marg_score"] < 1e-10 and r["cll_vs_decoder_abs"] < 1e-10 and r["fd_score_rel"] < 1e-5)
        ok_all &= r["pass"]
        print(json.dumps(r, indent=1), flush=True)
        res.append(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(tolist({"tests": res, "pass": ok_all}), f, indent=1)
    print("ALL PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
