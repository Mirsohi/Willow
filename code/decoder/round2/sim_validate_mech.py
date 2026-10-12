#!/usr/bin/env python3
"""Round 2, Amendment A1 (notes/decoder/willow_prereg_r2.md section 8.2): validation of T-hat_mech and D-hat_mech on
simulated data. No real shots are read.

Model M: the 4 W5 shipped RL DEMs of one series (default d3_at_q10_7/X), with T-hat's classes per leaf (as F1's
validation, sim_validate.py).
Truth: V0, M itself; V2, M with one log-odds shift per template drawn from N(0, 0.3^2) (numpy default_rng(2026), one
draw per template in sorted template-name order, shared by the 4 leaves) plus ln 2 on every template that flips L.
Shots, split and fit sample as sim_validate.py: 50,000 shots per leaf from the truth (stim, seed 2026 + r); even
positions = fit half, odd = held-out half (25,000); 5,000 fit shots drawn from the fit half with seed 7.
Fits (mech_fit, pooled over the 4 leaves): T-hat_mech maximises sum ln P(s), no penalty, start 0; D-hat_mech maximises
sum ln P(L|s) - (4/2) sum delta^2, starting from the F1 D-hat of the same simulated shots mapped to templates. For V0 the
F1 D-hat is read from sim_V0.json when its sampled data are the same (checked: each leaf's sample LER); otherwise it is
fitted here with cond_fit.fit_series.
Held-out: exact ML under M, T-hat_mech, D-hat_mech and the truth; m_X = (k_M - k_X)/k_X with the frozen count2 /
ratio_from_counts; the oracle gain; the recovered share (k_M - k_X)/(k_M - k_truth); mean held-out -ln P(L|s).
Pass (pre-registered): V0, neither m_X has a 95% CI above 0; V2, both m_X have 95% CIs above 0 and T-hat_mech
recovers at least half of the oracle gain.

Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B sim_validate_mech.py --extract <root> --scenario V2 --out <dir>
"""
import argparse, json, math, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import Leaf, load_dm, t_classes, fit_series, write_json, remove_ckpt, RL_DEM, W5
from mech_fit import MechSeries, fit_mech, template_keys, key_name, LAMBDA_D
from dem_utils import DemMatrices
from ml_exact_fast import FastExactDecoder
from mismatch import dm_to_stim
import analyze_willow as AW            # frozen: count2, ratio_from_counts


def truth_v2(dms, seed=2026, sd=0.3):
    keys = [template_keys(dm) for dm in dms]
    names = sorted({key_name(k) for ks in keys for k in ks})
    z = np.random.default_rng(seed).normal(0.0, sd, size=len(names))
    shift = dict(zip(names, z.tolist()))
    out = []
    for dm, ks in zip(dms, keys):
        d = np.array([shift[key_name(k)] + (math.log(2.0) if k[1] else 0.0) for k in ks])
        lo = np.log(dm.p) - np.log1p(-dm.p) + d
        p = np.clip(1.0 / (1.0 + np.exp(-lo)), 1e-15, 0.5 - 1e-9)
        out.append(DemMatrices(dm.num_detectors, dm.num_observables, dm.dets, dm.obs, p, dm.coords, []))
    return out, shift


def nll(post, L):
    P = np.clip(np.nan_to_num(post[np.arange(len(L)), L], nan=0.5), 1e-300, 1.0)
    return -np.log(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--series", default="google_105Q_surface_code_d3_d5_d7/d3_at_q10_7/X")
    ap.add_argument("--scenario", choices=("V0", "V2"), required=True)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--nfit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    logf = open(os.path.join(a.out, f"sim_mech_{a.scenario}.log"), "a") if os.path.isdir(a.out) else None

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()
    dms = [load_dm(os.path.join(a.extract, a.series, f"r{r}", RL_DEM)) for r in W5]
    if a.scenario == "V0":
        truths, shift = dms, None
    else:
        truths, shift = truth_v2(dms, a.seed)
    leaves, info = [], {}
    for r, dm, dT in zip(W5, dms, truths):
        labels, _ = t_classes(dm, a.nfit)
        det, obs, _ = dm_to_stim(dT).compile_sampler(seed=a.seed + r).sample(a.shots)
        det = det.astype(np.uint8); L = obs[:, 0].astype(np.int64)
        fit_all, te_pos = np.arange(0, a.shots, 2), np.arange(1, a.shots, 2)
        rng = np.random.default_rng(7)
        fit_pos = np.sort(rng.choice(fit_all, size=min(a.nfit, len(fit_all)), replace=False))
        lf = Leaf(f"r{r}", r, dm, labels, det[fit_pos], L[fit_pos], det[te_pos], L[te_pos], extra={"truth": dT})
        leaves.append(lf)
        info[f"r{r}"] = {"faults": dm.num_faults, "classes": len(lf.names), "LER_truth_sample": float(L.mean())}
        log(f"r{r}: {dm.num_faults} faults, {len(lf.names)} classes, sampled {a.shots} shots")
    ms = MechSeries(leaves)
    log(f"family: {json.dumps(ms.info)}")
    cpu = {}
    # T-hat_mech: syndrome likelihood
    cks = {k: os.path.join(a.out, f"ckpt_sim_mech_{a.scenario}_{k}") for k in ("T", "F1", "D")}
    fT = fit_mech(ms, "marg", lam=0.0, start=None, log=log, ckpt=cks["T"])
    cpu["fit_T_mech"] = fT["cpu_seconds_fit_total"]
    log(f"T_mech: converged={fT['converged']} after {fT['iterations']} steps, {fT['n_eval']} evaluations, "
        f"{cpu['fit_T_mech']:.0f} CPU s")
    # F1 D-hat for the start of D-hat_mech
    f1 = None
    p0 = os.path.join(a.out, "sim_V0.json")
    if a.scenario == "V0" and os.path.exists(p0):
        prev = json.load(open(p0))
        same = all(prev["leaves"][k]["LER_truth_sample"] == info[k]["LER_truth_sample"] for k in info)
        if same and prev["fit"]["converged"]:
            f1 = {"names": prev["fit"]["names"], "phi_hat": prev["fit"]["phi_hat"], "source": "sim_V0.json"}
    if f1 is None:
        fr = fit_series(leaves, log=log, ckpt=cks["F1"])
        cpu["fit_F1"] = fr["cpu_seconds_fit_total"]
        f1 = {"names": fr["names"], "phi_hat": fr["phi_hat"], "source": "fitted here", "converged": fr["converged"],
              "iterations": fr["iterations"]}
    start = ms.start_from_classes(f1["names"], np.array(f1["phi_hat"]))
    fD = fit_mech(ms, "cond", lam=LAMBDA_D, start=start, log=log, ckpt=cks["D"])
    cpu["fit_D_mech"] = fD["cpu_seconds_fit_total"]
    log(f"D_mech: converged={fD['converged']} after {fD['iterations']} steps, {fD['n_eval']} evaluations, "
        f"{cpu['fit_D_mech']:.0f} CPU s")
    dT_, dD_ = np.array(fT["delta_hat"]), np.array(fD["delta_hat"])
    tags = ("M", "Tm", "Dm", "truth")
    cells = {t: [] for t in tags[1:]}
    tot = {t: 0 for t in tags}; ll = {t: 0.0 for t in tags}; per_leaf = {}; nsh = 0
    c0 = time.process_time()
    for l, lf in enumerate(leaves):
        posts = {}
        for tag, dm in (("M", lf.model.dm_at(np.ones(len(lf.names)))), ("Tm", ms.dm(l, dT_)), ("Dm", ms.dm(l, dD_)),
                        ("truth", lf.extra["truth"])):
            posts[tag], _ = FastExactDecoder(dm, order=lf.model.order).class_probs(lf.det_test)
        pred = {k: (np.nan_to_num(v[:, 1], nan=0.0) > 0.5).astype(np.int8) for k, v in posts.items()}
        o = lf.L_test.astype(np.int8)
        for t in cells:
            cells[t].append(AW.count2(pred["M"], pred[t], o))
        k = {t: int(np.sum(pred[t] != o)) for t in tags}
        for t in tags:
            tot[t] += k[t]; ll[t] += float(nll(posts[t], lf.L_test).sum())
        nsh += len(o)
        per_leaf[lf.name] = {"errors": k, "n": int(len(o))}
    cpu["decode_4_dems"] = time.process_time() - c0
    m = {t: AW.ratio_from_counts(cells[t]) for t in cells}
    gain = tot["M"] - tot["truth"]
    share = {t: ((tot["M"] - tot[t]) / gain if gain else float("nan")) for t in ("Tm", "Dm")}
    if a.scenario == "V0":
        passed = bool(not (m["Tm"]["ci95"][0] > 0) and not (m["Dm"]["ci95"][0] > 0))
    else:
        passed = bool(m["Tm"]["ci95"][0] > 0 and m["Dm"]["ci95"][0] > 0 and share["Tm"] >= 0.5)
    out = {"scenario": a.scenario, "series": a.series, "prereg": "notes/decoder/willow_prereg_r2.md section 8.2",
           "truth": "M" if a.scenario == "V0" else "M with per-template log-odds shifts N(0, 0.3^2) (seed 2026) + ln 2 on "
           "templates flipping L", "truth_shifts": shift, "leaves": info, "family": ms.info,
           "fit_T_mech": fT, "fit_D_mech": fD, "f1_start": f1, "m_T_mech": m["Tm"], "m_D_mech": m["Dm"],
           "oracle_gain": m["truth"], "errors": tot, "per_leaf": per_leaf, "recovered_share_of_oracle_gain": share,
           "heldout_mean_nll": {t: ll[t] / nsh for t in ll}, "pass": passed, "cpu_seconds": cpu,
           "wall_seconds": time.time() - t0}
    write_json(os.path.join(a.out, f"sim_mech_{a.scenario}.json"), out)
    for ck in cks.values():
        remove_ckpt(ck)
    f = lambda d: f"{100 * d['rel']:+.3f}% [{100 * d['ci95'][0]:+.3f}, {100 * d['ci95'][1]:+.3f}]%"
    log(f"{a.scenario}: m_T_mech = {f(m['Tm'])}, m_D_mech = {f(m['Dm'])}, oracle {100 * m['truth']['rel']:+.3f}%, "
        f"shares T {share['Tm']:.3f} D {share['Dm']:.3f}, nll M/Tm/Dm/truth "
        + "/".join(f"{ll[t] / nsh:.5f}" for t in tags) + f" -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
