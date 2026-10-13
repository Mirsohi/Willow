"""R4's analysis (review 1.6): the matched class-level comparison of the label fit D-hat and the syndrome fit S-hat on
the held-out halves. POST HOC revision analysis, not a pre-registered test; declared in
notes/reservoir/willow_revision_declarations.md (R4 at 16:01Z; this script's md5 in a later section) before it was run.

reservoir, 2026-10-13. The definitions are decoder's round-2 analysis (code/decoder/round2/analyze_r2.py), with its
helpers imported read-only: analyze_willow.count2 / ratio_from_counts / calib_sums / paired_series, and
p3_heldout_fitted.per_shot / ratio_block / delta / CONF. For each set of series (S-hat converged; all S-hat fitted;
and, as a reproduction check of decoder's f1_analysis.json, D-hat on all 18 series), on the W5 held-out halves:
  errors       m_X = (k_RL - k_X) / k_X for X = D-hat, S-hat (P4's SE, pooled over leaves) and the direct
               (k_S - k_D) / k_D; also against T-hat;
  log-loss     -ln P(L_obs | s) per shot (exact from the stored llr; a NaN posterior counts as P = 1/2): means and the
               paired differences RL - D, RL - S, S - D, T - S (shot as the unit, stratified by leaf);
  calibration  R and R_conf of ML(X) and the share of ML(RL)'s excess of each removed (delta method);
  syndromes    the held-out ln P_X(s) - ln P_RL(s) per shot for X = T-hat, D-hat, S-hat, and S - D (stratified SE);
  NN gap       the approximate share of the ML(RL)-NN gap closed (as P5 / analyze_r2: numerator on the W5 held-out
               halves, denominator on the G1 window);
  scales       S-hat's and D-hat's fitted scales per class: medians over series and the Spearman correlation of
               ln theta over (series, class); convergence, iterations, evaluations and fixed classes per series.
Usage: python3 -I -B r4_analysis.py RES F1_DIR S_DIR M3_JSON OUT_DIR   (RES = results/decoder/willow)
       -> OUT_DIR/r4_analysis.json (through qrc.outpaths)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEC = os.path.realpath(os.path.join(HERE, "..", "..", "decoder"))
for p in (DEC, os.path.join(DEC, "supplementary"), os.path.join(DEC, "round2"), os.path.join(HERE, "..")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
import analyze_willow as AW  # noqa: E402  (decoder's, frozen, read-only)
from p3_heldout_fitted import CONF, delta, per_shot, ratio_block  # noqa: E402  (decoder's, read-only)
from qrc.outpaths import out_path  # noqa: E402

W5 = (10, 13, 30, 50)


def nll_exact(llr, obs):
    """analyze_r2.nll_exact (copied: analyze_r2 has no import guard for its argument parsing)."""
    L = np.nan_to_num(llr.astype(np.float64), nan=0.0)
    s = np.where(obs.astype(bool), 1.0, -1.0)
    return np.logaddexp(0.0, -s * L)


def strat_mean(lst):
    """mean over all shots and its SE with the shot as the unit, stratified by leaf (analyze_r2's)."""
    n = sum(len(x) for x in lst)
    mean = float(np.concatenate(lst).mean())
    var = sum(float(((x - x.mean()) ** 2).sum()) for x in lst)
    se = math.sqrt(var) / n
    return {"mean": mean, "se": se, "z": mean / se if se > 0 else None, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def block(res, f1, sdir, series, models, p5):
    """The comparison on the held-out halves of the listed series, for the listed fitted models (subset of D, S)."""
    out = {"series": len(series), "problems": []}
    cells = {k: [] for k in ("RL_vs_D", "RL_vs_S", "S_vs_D", "T_vs_D", "T_vs_S")}
    pts = {m: {} for m in models}
    nll = {k: [] for k in ("RL", "T") + tuple(models)}
    lps = {k: [] for k in ("T-RL",) + tuple(f"{m}-RL" for m in models) + (("S-D",) if len(models) == 2 else ())}
    cols = ["e_RL", "m_RL", "ec_RL", "mc_RL"] + [f"{c}_{m}" for m in models for c in ("e", "m", "ec", "mc")]
    ix = {c: i for i, c in enumerate(cols)}
    tot = np.zeros(len(cols))
    cov = np.zeros((len(cols), len(cols)))
    acc = {m: {"n": 0, "k": 0, "sum_m": 0.0, "sum_mvar": 0.0, "var_k": 0.0, "conf_k": 0, "conf_m": 0.0, "conf_n": 0}
           for m in models}
    nsh = 0
    for s in series:
        lvs = [f"{s}/r{r}" for r in W5]
        for lf in lvs:
            r = int(lf.rsplit("/r", 1)[1])
            z1 = np.load(os.path.join(res, "fit", lf, "rl_tsplit", "fit_data.npz"))
            o = z1["obs_test"].astype(np.int8)
            RLp, Tp = z1["shipped_pred"].astype(np.int8), z1["fitted_pred"].astype(np.int8)
            pred, llr, logp = {}, {}, {}
            if "D" in models:
                z2 = np.load(os.path.join(f1, lf, "d_data.npz"))
                if not (np.array_equal(z1["test_idx"], z2["test_idx"]) and np.array_equal(z1["obs_test"], z2["obs_test"])):
                    out["problems"].append(f"{lf}: D-hat's held-out shots differ from round 1")
                pred["D"], llr["D"], logp["D"] = z2["D_pred"].astype(np.int8), z2["D_llr"], z2["D_logp"]
            if "S" in models:
                z3 = np.load(os.path.join(sdir, lf, "s_data.npz"))
                if not (np.array_equal(z1["test_idx"], z3["test_idx"]) and np.array_equal(z1["obs_test"], z3["obs_test"])):
                    out["problems"].append(f"{lf}: S-hat's held-out shots differ from round 1")
                pred["S"], llr["S"], logp["S"] = z3["S_pred"].astype(np.int8), z3["S_llr"], z3["S_logp"]
            for m in models:
                c = AW.count2(RLp, pred[m], o)
                cells[f"RL_vs_{m}"].append(c)
                cells[f"T_vs_{m}"].append(AW.count2(Tp, pred[m], o))
                pts[m].setdefault(s, {})[r] = c
            if len(models) == 2:
                cells["S_vs_D"].append(AW.count2(pred["S"], pred["D"], o))
            n_l = {"RL": nll_exact(z1["shipped_llr"], o), "T": nll_exact(z1["fitted_llr"], o)}
            for m in models:
                n_l[m] = nll_exact(llr[m], o)
            for k in nll:
                nll[k].append(n_l[k])
            lps["T-RL"].append(z1["fitted_logp"] - z1["shipped_logp"])
            for m in models:
                lps[f"{m}-RL"].append(logp[m] - z1["shipped_logp"])
            if len(models) == 2:
                lps["S-D"].append(logp["S"] - logp["D"])
            X = np.zeros((len(o), len(cols)))
            have = np.ones(len(o), bool)
            for tag, L in [("RL", z1["shipped_llr"])] + [(m, llr[m]) for m in models]:
                e, mm, _ = per_shot(L, o)
                cf = mm < CONF
                X[:, ix[f"e_{tag}"]], X[:, ix[f"m_{tag}"]] = e, mm
                X[:, ix[f"ec_{tag}"]], X[:, ix[f"mc_{tag}"]] = e * cf, mm * cf
                if tag in acc:
                    cs = AW.calib_sums(L, o, have)
                    a = acc[tag]
                    for k2 in ("n", "k", "sum_m", "sum_mvar"):
                        a[k2] += cs[k2]
                    a["var_k"] += cs["k"] * (1 - cs["k"] / max(cs["n"], 1))
                    a["conf_n"] += int(cf.sum())
                    a["conf_k"] += int(e[cf].sum())
                    a["conf_m"] += float(mm[cf].sum())
            tot += X.sum(0)
            Dv = X - X.mean(0)
            cov += Dv.T @ Dv
            nsh += len(o)
    out["shots"] = nsh
    out["errors"] = {}
    for k, v in cells.items():
        if v:
            out["errors"][k] = AW.ratio_from_counts(v)
    out["logloss"] = {"mean_nll": {k: float(np.concatenate(v).mean()) for k, v in nll.items()}}
    for x, y in (("RL", "D"), ("RL", "S"), ("S", "D"), ("T", "S"), ("T", "D")):
        if x in nll and y in nll:
            out["logloss"][f"diff_{x}-{y}"] = strat_mean([a - b for a, b in zip(nll[x], nll[y])])
    out["syndrome_lnP_per_shot"] = {k: strat_mean(v) for k, v in lps.items() if v}

    def Rt(T, t, c=""):
        return T[ix[f"e{c}_{t}"]] / T[ix[f"m{c}_{t}"]]
    out["calibration"] = {"R_RL_same_shots": float(Rt(tot, "RL")), "R_conf_RL_same_shots": float(Rt(tot, "RL", "c"))}
    for m in models:
        a = acc[m]
        b = ratio_block(a["k"], a["sum_m"], a["sum_mvar"], a["var_k"])
        b["conf"] = {"shots": a["conf_n"], "errors": a["conf_k"], "predicted": a["conf_m"],
                     "ratio": a["conf_k"] / a["conf_m"] if a["conf_m"] > 0 else None}
        sh = {}
        for name, fun in (("R", lambda T, m=m: 1 - (Rt(T, m) - 1) / (Rt(T, "RL") - 1)),
                          ("R_conf", lambda T, m=m: 1 - (Rt(T, m, "c") - 1) / (Rt(T, "RL", "c") - 1))):
            v, se = delta(fun, tot, cov)
            sh[name] = {"share": v, "se": se, "ci95": [v - 1.96 * se, v + 1.96 * se]}
        out["calibration"][m] = {"ML": b, "share_of_RL_excess_removed": sh}
    if p5:
        gap = p5["eps_ML_rl"] - p5["eps_NN"]
        se_gap = math.sqrt(p5["se"] ** 2 + p5["se_NN"] ** 2)
        out["nn_gap_share"] = {}
        for m in models:
            rows = [AW.paired_series(p, W5) for p in pts[m].values()]
            rows = [x for x in rows if x]
            dbar = float(np.mean([x["delta"] for x in rows]))
            se_d = math.sqrt(sum(x["se_delta"] ** 2 for x in rows)) / len(rows)
            shv = dbar / gap
            out["nn_gap_share"][m] = {"series": len(rows), "mean_eps_RL_minus_eps_X": dbar, "se": se_d, "share": shv,
                                      "se_share": abs(shv) * math.sqrt((se_d / dbar) ** 2 + (se_gap / gap) ** 2)
                                      if dbar else None,
                                      "note": "approximate, as P5: numerator on the W5 held-out halves, denominator on "
                                              "the G1 window"}
    return out


def main(res, f1, sdir, m3, out_dir):
    t0 = time.time()
    all18 = sorted(os.path.relpath(dp, f1) for dp, dn, fn in os.walk(f1) if "fit_series.json" in fn)
    sfit = sorted(os.path.relpath(dp, sdir) for dp, dn, fn in os.walk(sdir) if "fit_series.json" in fn)
    fD = {s: json.load(open(os.path.join(f1, s, "fit_series.json"))) for s in all18}
    fS = {s: json.load(open(os.path.join(sdir, s, "fit_series.json"))) for s in sfit}
    conv = [s for s in sfit if fS[s]["converged"]]
    p5 = json.load(open(m3))["P5_ML_rl_vs_NN"]["pooled"]["3"] if m3 and os.path.exists(m3) else None
    R = {"label": "reservoir, revision R4 (review 1.6): matched class-level comparison of the label fit D-hat and the "
                  "syndrome fit S-hat. POST HOC; declared before it was run",
         "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "series": {"D_hat": len(all18), "S_hat_fitted": len(sfit), "S_hat_converged": len(conv),
                    "S_hat_not_converged": [s for s in sfit if not fS[s]["converged"]]}}
    R["reproduction_of_f1_analysis"] = block(res, f1, sdir, all18, ("D",), p5)
    ref = json.load(open(os.path.join(os.path.dirname(f1), "f1_analysis.json")))
    mine = R["reproduction_of_f1_analysis"]
    R["reproduction_check"] = {
        "m_D_rel_diff": mine["errors"]["RL_vs_D"]["rel"] - ref["primary_m_D"]["pooled"]["rel"],
        "m_D_se_diff": mine["errors"]["RL_vs_D"]["se_rel"] - ref["primary_m_D"]["pooled"]["se_rel"],
        "logloss_RL-D_diff": mine["logloss"]["diff_RL-D"]["mean"] - ref["logloss"]["diff_RL-D"]["mean"],
        "R_D_diff": mine["calibration"]["D"]["ML"]["ratio"] - ref["calibration_D"]["ML(D-hat)"]["ratio"],
        "share_R_diff": mine["calibration"]["D"]["share_of_RL_excess_removed"]["R"]["share"]
        - ref["calibration_D"]["share_of_RL_excess_removed"]["R"]["share"],
        "share_R_conf_diff": mine["calibration"]["D"]["share_of_RL_excess_removed"]["R_conf"]["share"]
        - ref["calibration_D"]["share_of_RL_excess_removed"]["R_conf"]["share"]}
    if conv:
        R["matched_S_converged"] = block(res, f1, sdir, conv, ("D", "S"), p5)
    if sfit and len(sfit) > len(conv):
        R["matched_S_all_fitted"] = block(res, f1, sdir, sfit, ("D", "S"), p5)
    sc = {"per_series": {}}
    pairs = []
    for s in sfit:
        d = dict(zip(fD[s]["names"], fD[s]["theta_hat"]))
        e = dict(zip(fS[s]["names"], fS[s]["theta_hat"]))
        sc["per_series"][s] = {"converged": fS[s]["converged"], "iterations": fS[s]["iterations"],
                               "n_eval": fS[s]["n_eval"], "fixed": fS[s]["fixed"],
                               "cpu_seconds_total": fS[s].get("cpu_seconds_total"),
                               "theta_S": e, "theta_D": d, "se_ln_opg_S": dict(zip(fS[s]["names"], fS[s]["se_opg_ln"]))}
        pairs += [(math.log(d[k]), math.log(e[k])) for k in d if k in e and d[k] > 0 and e[k] > 0]
    names = sorted({nm for s in sfit for nm in fS[s]["names"]})
    sc["median_over_series"] = {nm: {"S": float(np.median([fS[s]["theta_hat"][fS[s]["names"].index(nm)]
                                                          for s in sfit if nm in fS[s]["names"]])),
                                     "D": float(np.median([fD[s]["theta_hat"][fD[s]["names"].index(nm)]
                                                          for s in sfit if nm in fD[s]["names"]]))} for nm in names}
    if len(pairs) > 2:
        rho = stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        sc["spearman_ln_theta_D_vs_S"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "pairs": len(pairs)}
    sc["cpu_hours_S"] = sum(fS[s].get("cpu_seconds_total", 0.0) for s in sfit) / 3600
    R["scales"] = sc
    R["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r4_analysis.json")), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(json.dumps(R["reproduction_check"]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
