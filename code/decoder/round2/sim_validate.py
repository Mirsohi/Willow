#!/usr/bin/env python3
"""Round-2 validation on simulated data (notes/decoder/willow_prereg_r2.md, section 5). No real shots are read.

Model M: the 4 W5 shipped RL DEMs of one series (default d3_at_q10_7/X), with T-hat's classes per leaf.
Truth: V0, M itself; V1, M with every fault that flips L scaled x2 and every other fault x0.87.
Per leaf: 50,000 shots sampled from the truth with stim; split as the real data without sweep bits (even positions
= fit half, odd = held-out half); 5,000 fit shots drawn from the fit half with seed 7 (as fit_willow); 25,000
held-out shots. D-hat: cond_fit.fit_series on the 4 leaves (pooled, as on real data). Held-out: exact ML under M,
under D-hat and under the truth (oracle); m_D = (k_M - k_D)/k_D with the frozen count2 / ratio_from_counts; the
oracle gain (k_M - k_T)/k_T; the recovered share (k_M - k_D)/(k_M - k_T); mean held-out -ln P(L | s) per decoder.
Pass (pre-registered): V0, m_D's 95% CI not above 0; V1, m_D's 95% CI above 0 and recovered share >= 0.5.

Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B sim_validate.py --extract <extract root> --scenario V1 --out <dir>
"""
import argparse, json, math, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import Leaf, load_dm, t_classes, fit_series, write_json, remove_ckpt, RL_DEM, W5
from dem_utils import DemMatrices
from ml_exact_fast import FastExactDecoder
from mismatch import dm_to_stim
import analyze_willow as AW            # frozen: count2, ratio_from_counts


def truth_dm(dm, scenario):
    if scenario == "V0":
        return dm
    s = np.where(np.asarray(dm.obs) != 0, 2.0, 0.87)
    p = np.clip(dm.p * s, 1e-15, 0.5 - 1e-9)
    return DemMatrices(dm.num_detectors, dm.num_observables, dm.dets, dm.obs, p, dm.coords, [])


def nll(post, L):
    P = np.clip(np.nan_to_num(post[np.arange(len(L)), L], nan=0.5), 1e-300, 1.0)
    return -np.log(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--series", default="google_105Q_surface_code_d3_d5_d7/d3_at_q10_7/X")
    ap.add_argument("--scenario", choices=("V0", "V1"), required=True)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--nfit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    logf = open(os.path.join(a.out, f"sim_{a.scenario}.log"), "a") if os.path.isdir(a.out) else None

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()
    leaves, info = [], {}
    for r in W5:
        path = os.path.join(a.extract, a.series, f"r{r}", RL_DEM)
        dm = load_dm(path)
        labels, _ = t_classes(dm, a.nfit)
        dT = truth_dm(dm, a.scenario)
        det, obs, _ = dm_to_stim(dT).compile_sampler(seed=a.seed + r).sample(a.shots)
        det = det.astype(np.uint8); L = obs[:, 0].astype(np.int64)
        fit_all, te_pos = np.arange(0, a.shots, 2), np.arange(1, a.shots, 2)
        rng = np.random.default_rng(7)
        fit_pos = np.sort(rng.choice(fit_all, size=min(a.nfit, len(fit_all)), replace=False))
        lf = Leaf(f"r{r}", r, dm, labels, det[fit_pos], L[fit_pos], det[te_pos], L[te_pos], extra={"truth": dT})
        leaves.append(lf)
        info[f"r{r}"] = {"faults": dm.num_faults, "classes": len(lf.names), "LER_truth_sample": float(L.mean())}
        log(f"r{r}: {dm.num_faults} faults, {len(lf.names)} classes, sampled {a.shots} shots")
    ck = os.path.join(a.out, f"ckpt_sim_{a.scenario}")
    fr = fit_series(leaves, log=log, ckpt=ck); cpu_fit = fr["cpu_seconds_fit_total"]
    log(f"fit: converged={fr['converged']} after {fr['iterations']} steps, {fr['n_eval']} evaluations, {cpu_fit:.0f} CPU s")
    phi = np.array(fr["phi_hat"])
    cells, cells_oracle, per_leaf, tot = [], [], {}, {"M": 0, "D": 0, "T": 0}
    ll = {"M": 0.0, "D": 0.0, "T": 0.0}; nsh = 0
    td = time.process_time()
    for lf in leaves:
        posts = {}
        for tag, dm in (("M", lf.model.dm_at(np.ones(len(lf.names)))), ("D", lf.model.dm_at(lf.theta(phi))),
                        ("T", lf.extra["truth"])):
            post, _ = FastExactDecoder(dm, order=lf.model.order).class_probs(lf.det_test)
            posts[tag] = post
        pred = {k: (np.nan_to_num(v[:, 1], nan=0.0) > 0.5).astype(np.int8) for k, v in posts.items()}
        o = lf.L_test.astype(np.int8)
        cells.append(AW.count2(pred["M"], pred["D"], o)); cells_oracle.append(AW.count2(pred["M"], pred["T"], o))
        k = {t: int(np.sum(pred[t] != o)) for t in pred}
        for t in k:
            tot[t] += k[t]; ll[t] += float(nll(posts[t], lf.L_test).sum())
        nsh += len(o)
        per_leaf[lf.name] = {"errors": k, "n": int(len(o))}
    cpu_dec = time.process_time() - td
    mD = AW.ratio_from_counts(cells); mT = AW.ratio_from_counts(cells_oracle)
    gain = tot["M"] - tot["T"]
    share = (tot["M"] - tot["D"]) / gain if gain else float("nan")
    if a.scenario == "V0":
        passed = not (mD["ci95"][0] > 0)
    else:
        passed = bool(mD["ci95"][0] > 0 and share >= 0.5)
    out = {"scenario": a.scenario, "series": a.series, "truth": "M" if a.scenario == "V0" else
           "M with faults flipping L x2, others x0.87", "leaves": info, "fit": fr, "m_D": mD, "oracle_gain": mT,
           "errors": tot, "per_leaf": per_leaf, "recovered_share_of_oracle_gain": share,
           "heldout_mean_nll": {t: ll[t] / nsh for t in ll}, "pass": passed,
           "cpu_seconds": {"fit": cpu_fit, "decode_3_dems": cpu_dec}, "wall_seconds": time.time() - t0}
    write_json(os.path.join(a.out, f"sim_{a.scenario}.json"), out)
    remove_ckpt(ck)
    log(f"{a.scenario}: m_D = {100 * mD['rel']:+.3f}% [{100 * mD['ci95'][0]:+.3f}, {100 * mD['ci95'][1]:+.3f}]%, "
        f"oracle {100 * mT['rel']:+.3f}%, recovered share {share:.3f}, nll M/D/T "
        f"{out['heldout_mean_nll']['M']:.5f}/{out['heldout_mean_nll']['D']:.5f}/{out['heldout_mean_nll']['T']:.5f} -> "
        f"{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
