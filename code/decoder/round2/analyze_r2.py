#!/usr/bin/env python3
"""Round 2 analysis of family F1 (notes/decoder/willow_prereg_r2.md, sections 3 and 4).

Primary: m_D = (k_ML(RL) - k_ML(D-hat)) / k_ML(D-hat) on the held-out halves of the W5 leaves of the converged series,
with the frozen count2 / ratio_from_counts (P4's SE); evaluated if at least 17 of 18 series fits converged.
Secondary: (1) held-out conditional log-loss -ln P(L_obs | s) per shot for ML(RL), ML(T-hat), ML(D-hat), exact from the
stored llr (softplus, a NaN posterior counts as P = 1/2), with paired differences and SEs (shot as the unit,
stratified by leaf); (2) R and R_conf of ML(D-hat) and the share of ML(RL)'s excess removed (definitions of
supplementary/p3_heldout_fitted.py, imported); (3) the share of the ML(RL)-NN gap closed, approximate: the mean over
series of eps_RL - eps_D-hat from the frozen paired_series on the W5 held-out halves, over P5's gap from
analysis_m3/prereg_results.json; (4) m_D without r = 13; (5) the count ratio of ML(T-hat) against ML(D-hat) and the
fitted scales.

Run: python3 -I -B analyze_r2.py --res results/decoder/willow --f1 results/decoder/willow/round2/f1 \
         --m3 results/decoder/willow/analysis_m3/prereg_results.json --out results/decoder/willow/round2
"""
import argparse, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "supplementary"))
sys.path.insert(0, HERE)
import numpy as np
import analyze_willow as AW                                  # frozen
from p3_heldout_fitted import per_shot, ratio_block, delta, CONF
from cond_fit import write_json, W5

MARGIN = 0.03


def nll_exact(llr, obs):
    L = np.nan_to_num(llr.astype(np.float64), nan=0.0)
    s = np.where(obs.astype(bool), 1.0, -1.0)
    return np.logaddexp(0.0, -s * L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--f1", required=True)
    ap.add_argument("--m3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    series = sorted(os.path.relpath(dp, a.f1) for dp, dn, fn in os.walk(a.f1) if "fit_series.json" in fn)
    fits = {s: json.load(open(os.path.join(a.f1, s, "fit_series.json"))) for s in series}
    conv = [s for s in series if fits[s]["converged"]]
    R = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prereg": "notes/decoder/willow_prereg_r2.md",
         "series_fitted": len(series), "series_converged": len(conv),
         "not_converged": [s for s in series if not fits[s]["converged"]], "checks": {"problems": []}}
    cells = {"RL_vs_D": [], "T_vs_D": [], "RL_vs_D_no13": []}
    pts = {}
    nll = {"RL": [], "T": [], "D": []}; dif = {"RL-D": [], "T-D": [], "RL-T": []}
    cols = ["e_D", "m_D", "ec_D", "mc_D", "e_RL", "m_RL", "ec_RL", "mc_RL"]
    ix = {c: i for i, c in enumerate(cols)}
    tot = np.zeros(len(cols)); cov = np.zeros((len(cols), len(cols)))
    acc = {"n": 0, "k": 0, "sum_m": 0.0, "sum_mvar": 0.0, "var_k": 0.0, "conf_k": 0, "conf_m": 0.0, "conf_n": 0, "conf_mvar": 0.0}
    nsh = 0
    for s in conv:
        for lf in fits[s]["leaves"]:
            r = int(lf.rsplit("/r", 1)[1])
            z1 = np.load(os.path.join(a.res, "fit", lf, "rl_tsplit", "fit_data.npz"))
            z2 = np.load(os.path.join(a.f1, lf, "d_data.npz"))
            if not (np.array_equal(z1["test_idx"], z2["test_idx"]) and np.array_equal(z1["obs_test"], z2["obs_test"])):
                R["checks"]["problems"].append(f"{lf}: held-out shots differ from round 1")
            o = z1["obs_test"].astype(np.int8)
            S, T, D = (x.astype(np.int8) for x in (z1["shipped_pred"], z1["fitted_pred"], z2["D_pred"]))
            c = AW.count2(S, D, o)
            cells["RL_vs_D"].append(c); cells["T_vs_D"].append(AW.count2(T, D, o))
            if r != 13:
                cells["RL_vs_D_no13"].append(c)
            pts.setdefault(s, {})[r] = c
            n_l = {"RL": nll_exact(z1["shipped_llr"], o), "T": nll_exact(z1["fitted_llr"], o), "D": nll_exact(z2["D_llr"], o)}
            for k in nll:
                nll[k].append(n_l[k])
            for k, (x, y) in {"RL-D": ("RL", "D"), "T-D": ("T", "D"), "RL-T": ("RL", "T")}.items():
                dif[k].append(n_l[x] - n_l[y])
            have = np.ones(len(o), bool)
            cs = AW.calib_sums(z2["D_llr"], o, have)
            for k2 in ("n", "k", "sum_m", "sum_mvar"):
                acc[k2] += cs[k2]
            acc["var_k"] += cs["k"] * (1 - cs["k"] / max(cs["n"], 1))
            X = np.zeros((len(o), len(cols)))
            for tag, llr in (("D", z2["D_llr"]), ("RL", z1["shipped_llr"])):
                e, m, _ = per_shot(llr, o)
                cf = m < CONF
                X[:, ix[f"e_{tag}"]], X[:, ix[f"m_{tag}"]] = e, m
                X[:, ix[f"ec_{tag}"]], X[:, ix[f"mc_{tag}"]] = e * cf, m * cf
                if tag == "D":
                    acc["conf_n"] += int(cf.sum()); acc["conf_k"] += int(e[cf].sum())
                    acc["conf_m"] += float(m[cf].sum()); acc["conf_mvar"] += float(np.sum(m[cf] * (1 - m[cf])))
            tot += X.sum(0); Dv = X - X.mean(0); cov += Dv.T @ Dv
            nsh += len(o)
    # primary
    mD = AW.ratio_from_counts(cells["RL_vs_D"])
    evaluable = len(conv) >= 17
    if mD and evaluable:
        sig = mD["p"] < 0.05
        if sig and mD["rel"] >= MARGIN:
            dec = "significant and >= +3%: a material gain of ML(D-hat) over ML(RL)"
        elif sig and mD["rel"] > 0:
            dec = "significant and below +3%: a real but small gain of ML(D-hat) over ML(RL)"
        elif sig:
            dec = "significant and negative: ML(D-hat) is worse than ML(RL)"
        else:
            inside = -MARGIN < mD["ci90"][0] and mD["ci90"][1] < MARGIN
            dec = "not significant; 90% CI " + ("inside" if inside else "not inside") + " +-3%"
    else:
        dec = "not evaluable (fewer than 17 of 18 series converged)" if not evaluable else "no data"
    R["primary_m_D"] = {"pooled": mD, "evaluable": evaluable, "decision": dec}
    # secondary 1: log-loss
    NLL = {k: np.concatenate(v) for k, v in nll.items()}
    R["logloss"] = {"mean_nll": {k: float(v.mean()) for k, v in NLL.items()}, "n": nsh}
    for k, lst in dif.items():
        mean = float(np.concatenate(lst).mean())
        var = sum(float(((x - x.mean()) ** 2).sum()) for x in lst)
        se = math.sqrt(var) / nsh
        R["logloss"][f"diff_{k}"] = {"mean": mean, "se": se, "z": mean / se if se > 0 else float("nan"),
                                     "ci95": [mean - 1.96 * se, mean + 1.96 * se]}
    # secondary 2: calibration of ML(D-hat)
    b = ratio_block(acc["k"], acc["sum_m"], acc["sum_mvar"], acc["var_k"])
    Rc = acc["conf_k"] / acc["conf_m"] if acc["conf_m"] > 0 else float("nan")
    b["conf"] = {"shots": acc["conf_n"], "errors": acc["conf_k"], "predicted": acc["conf_m"], "ratio": Rc,
                 "se_sqrtk": math.sqrt(acc["conf_k"]) / acc["conf_m"] if acc["conf_m"] > 0 else float("nan")}

    def Rt(T, t, c=""):
        return T[ix[f"e{c}_{t}"]] / T[ix[f"m{c}_{t}"]]
    shares = {}
    for name, fun in (("R", lambda T: 1 - (Rt(T, "D") - 1) / (Rt(T, "RL") - 1)),
                      ("R_conf", lambda T: 1 - (Rt(T, "D", "c") - 1) / (Rt(T, "RL", "c") - 1))):
        v, se = delta(fun, tot, cov)
        shares[name] = {"share": v, "se": se, "ci95": [v - 1.96 * se, v + 1.96 * se]}
    shares["R_RL_same_shots"] = float(Rt(tot, "RL")); shares["R_conf_RL_same_shots"] = float(Rt(tot, "RL", "c"))
    R["calibration_D"] = {"ML(D-hat)": b, "share_of_RL_excess_removed": shares}
    # secondary 3: share of the ML(RL)-NN gap closed (approximate)
    p5 = json.load(open(a.m3))["P5_ML_rl_vs_NN"]["pooled"]["3"]
    gap = p5["eps_ML_rl"] - p5["eps_NN"]; se_gap = math.sqrt(p5["se"] ** 2 + p5["se_NN"] ** 2)
    rows = [AW.paired_series(p, W5) for p in pts.values()]
    rows = [x for x in rows if x]
    dbar = float(np.mean([x["delta"] for x in rows])); se_d = math.sqrt(sum(x["se_delta"] ** 2 for x in rows)) / len(rows)
    sh = dbar / gap
    R["nn_gap_share"] = {"series": len(rows), "mean_eps_RL_minus_eps_D": dbar, "se": se_d, "gap_RL_minus_NN": gap,
                         "se_gap": se_gap, "share": sh, "se_share": abs(sh) * math.sqrt((se_d / dbar) ** 2 + (se_gap / gap) ** 2)
                         if dbar else float("nan"),
                         "note": "approximate, as P5: numerator on the W5 held-out halves, denominator on the G1 window"}
    # secondary 4, 5
    R["m_D_without_r13"] = AW.ratio_from_counts(cells["RL_vs_D_no13"])
    R["ratio_T_vs_D"] = AW.ratio_from_counts(cells["T_vs_D"])
    names = sorted({nm for s in series for nm in fits[s]["names"]})
    R["scales"] = {"per_series": {s: {"converged": fits[s]["converged"], "iterations": fits[s]["iterations"],
                                      "fixed": fits[s]["fixed"], "theta": dict(zip(fits[s]["names"], fits[s]["theta_hat"])),
                                      "se_ln_opg": dict(zip(fits[s]["names"], fits[s]["se_opg_ln"])),
                                      "se_ln_exp": dict(zip(fits[s]["names"], fits[s].get("se_exp_ln", [None] * len(fits[s]["names"])))),
                                      "n_eval": fits[s]["n_eval"]} for s in series},
                   "median_over_series": {nm: float(np.median([fits[s]["theta_hat"][fits[s]["names"].index(nm)]
                                                                for s in series if nm in fits[s]["names"]]))
                                          for nm in names}}
    R["cpu_hours"] = sum(fits[s].get("cpu_seconds_total", 0.0) for s in series) / 3600
    os.makedirs(a.out, exist_ok=True)
    write_json(os.path.join(a.out, "f1_analysis.json"), R)
    show = {k: R[k] for k in ("series_fitted", "series_converged", "not_converged", "checks", "primary_m_D", "logloss",
                              "calibration_D", "nn_gap_share", "m_D_without_r13", "ratio_T_vs_D", "cpu_hours")}
    show["scales_median"] = R["scales"]["median_over_series"]
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main()
