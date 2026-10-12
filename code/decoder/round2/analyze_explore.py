#!/usr/bin/env python3
"""Round 2, exploratory statistics (notes/decoder/willow_prereg_r2.md sections 8.2 and 8.4; 95% CIs, no test).

Held out (section 8.2): X = T-hat_mech (Tm) and D-hat_mech (Dm), on the held-out halves of the W5 leaves of the series
whose X fit converged (the others are listed).
In sample (section 8.4, descriptive only): X = T-hat (T), D-hat (D), Tm and Dm, on the 5,000 fit shots per leaf that all
four fits used, for the series whose X fit converged (T-hat: every leaf's round-1 fit).
Per X, the same block of statistics for both:
  m_X = (k_ML(RL) - k_ML(X)) / k_ML(X) with the frozen count2 / ratio_from_counts (shot as the unit, stratified by leaf),
  also without r = 13; the count ratio of ML(T-hat) against ML(X) (X != T);
  conditional log-loss -ln P(L_obs | s) of every decoder available on those shots, exact from the stored llr (a NaN
  posterior counts as P = 1/2), with paired differences against X;
  R and R_conf of ML(X) and the share of ML(RL)'s excess removed (definitions of supplementary/p3_heldout_fitted.py);
  the share of the ML(RL)-NN gap closed, approximate as P5 (frozen paired_series on W5; P5's gap from
  analysis_m3/prereg_results.json).
Run: python3 -I -B analyze_explore.py --res results/decoder/willow --r2 results/decoder/willow/round2 \
         --m3 results/decoder/willow/analysis_m3/prereg_results.json
(reads <r2>/f1, <r2>/mech and <r2>/insample; writes <r2>/explore_analysis.json)
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
from analyze_r2 import nll_exact

ARCH = "google_105Q_surface_code_d3_d5_d7"
TAGS = ("RL", "T", "D", "Tm", "Dm")


def load_heldout(a, leaf):
    z1 = np.load(os.path.join(a.res, "fit", leaf, "rl_tsplit", "fit_data.npz"))
    out = {"o": z1["obs_test"].astype(np.int8), "RL": (z1["shipped_pred"], z1["shipped_llr"]),
           "T": (z1["fitted_pred"], z1["fitted_llr"]), "problems": []}
    for path, keys in ((os.path.join(a.r2, "f1", leaf, "d_data.npz"), ("D",)),
                       (os.path.join(a.r2, "mech", leaf, "mech_data.npz"), ("Tm", "Dm"))):
        if os.path.exists(path):
            z = np.load(path)
            if not (np.array_equal(z["test_idx"], z1["test_idx"]) and np.array_equal(z["obs_test"], z1["obs_test"])):
                out["problems"].append(f"{path}: held-out shots differ from round 1")
            for k in keys:
                out[k] = (z[f"{k}_pred"], z[f"{k}_llr"])
    return out


def load_insample(a, leaf):
    z = np.load(os.path.join(a.r2, "insample", leaf, "insample_data.npz"))
    fi = np.load(os.path.join(a.res, "fit", leaf, "rl_tsplit", "fit_data.npz"))["fit_idx"]
    out = {"o": z["obs_fit"].astype(np.int8), "problems": [] if np.array_equal(z["fit_idx"], fi) else
           [f"{leaf}: in-sample fit_idx differs from round 1"]}
    for k in TAGS:
        if f"{k}_pred" in z.files:
            out[k] = (z[f"{k}_pred"], z[f"{k}_llr"])
    return out


def block(a, series, X, loader, p5):
    cells = {"RL_vs_X": [], "T_vs_X": [], "RL_vs_X_no13": []}
    pts, nll, dif = {}, {}, {}
    cols = ["e_X", "m_X", "ec_X", "mc_X", "e_RL", "m_RL", "ec_RL", "mc_RL"]
    ix = {c: i for i, c in enumerate(cols)}
    tot = np.zeros(len(cols)); cov = np.zeros((len(cols), len(cols)))
    acc = {"n": 0, "k": 0, "sum_m": 0.0, "sum_mvar": 0.0, "var_k": 0.0, "conf_k": 0, "conf_m": 0.0, "conf_n": 0}
    problems, nsh, present = [], 0, None
    for s in series:
        for r in W5:
            lf = f"{s}/r{r}"
            d = loader(a, lf)
            problems += d["problems"]
            if X not in d:
                problems.append(f"{lf}: no {X} predictions"); continue
            tags = [t for t in TAGS if t in d]
            present = tags if present is None else [t for t in present if t in tags]
            o = d["o"]
            S, P = d["RL"][0].astype(np.int8), d[X][0].astype(np.int8)
            c = AW.count2(S, P, o)
            cells["RL_vs_X"].append(c)
            if X != "T":
                cells["T_vs_X"].append(AW.count2(d["T"][0].astype(np.int8), P, o))
            if r != 13:
                cells["RL_vs_X_no13"].append(c)
            pts.setdefault(s, {})[r] = c
            n_l = {t: nll_exact(d[t][1], o) for t in tags}
            for t, v in n_l.items():
                nll.setdefault(t, []).append(v)
                if t != X:
                    dif.setdefault(f"{t}-{X}", []).append(v - n_l[X])
            cs = AW.calib_sums(d[X][1], o, np.ones(len(o), bool))
            for k2 in ("n", "k", "sum_m", "sum_mvar"):
                acc[k2] += cs[k2]
            acc["var_k"] += cs["k"] * (1 - cs["k"] / max(cs["n"], 1))
            M = np.zeros((len(o), len(cols)))
            for tag, llr in (("X", d[X][1]), ("RL", d["RL"][1])):
                e, m, _ = per_shot(llr, o)
                cf = m < CONF
                M[:, ix[f"e_{tag}"]], M[:, ix[f"m_{tag}"]] = e, m
                M[:, ix[f"ec_{tag}"]], M[:, ix[f"mc_{tag}"]] = e * cf, m * cf
                if tag == "X":
                    acc["conf_n"] += int(cf.sum()); acc["conf_k"] += int(e[cf].sum()); acc["conf_m"] += float(m[cf].sum())
            tot += M.sum(0); Dv = M - M.mean(0); cov += Dv.T @ Dv
            nsh += len(o)
    out = {"series": len(series), "shots": nsh, "problems": problems, "m_X": AW.ratio_from_counts(cells["RL_vs_X"]),
           "m_X_without_r13": AW.ratio_from_counts(cells["RL_vs_X_no13"]),
           "ratio_T_vs_X": AW.ratio_from_counts(cells["T_vs_X"]) if cells["T_vs_X"] else None}
    if not nsh:
        return out
    keep = set(present or [])
    L = {"mean_nll": {k: float(np.concatenate(v).mean()) for k, v in nll.items() if k in keep}, "n": nsh}
    for k, lst in dif.items():
        if k.split("-")[0] not in keep:
            continue
        mean = float(np.concatenate(lst).mean())
        se = math.sqrt(sum(float(((x - x.mean()) ** 2).sum()) for x in lst)) / nsh
        L[f"diff_{k}"] = {"mean": mean, "se": se, "z": mean / se if se > 0 else float("nan"),
                          "ci95": [mean - 1.96 * se, mean + 1.96 * se]}
    out["logloss"] = L
    b = ratio_block(acc["k"], acc["sum_m"], acc["sum_mvar"], acc["var_k"])
    b["conf"] = {"shots": acc["conf_n"], "errors": acc["conf_k"], "predicted": acc["conf_m"],
                 "ratio": acc["conf_k"] / acc["conf_m"] if acc["conf_m"] > 0 else float("nan"),
                 "se_sqrtk": math.sqrt(acc["conf_k"]) / acc["conf_m"] if acc["conf_m"] > 0 else float("nan")}

    def Rt(T, t, c=""):
        return T[ix[f"e{c}_{t}"]] / T[ix[f"m{c}_{t}"]]
    shares = {"R_RL_same_shots": float(Rt(tot, "RL")), "R_conf_RL_same_shots": float(Rt(tot, "RL", "c"))}
    for name, fun in (("R", lambda T: 1 - (Rt(T, "X") - 1) / (Rt(T, "RL") - 1)),
                      ("R_conf", lambda T: 1 - (Rt(T, "X", "c") - 1) / (Rt(T, "RL", "c") - 1))):
        v, se = delta(fun, tot, cov)
        shares[name] = {"share": v, "se": se, "ci95": [v - 1.96 * se, v + 1.96 * se]}
    out["calibration"] = {"ML(X)": b, "share_of_RL_excess_removed": shares}
    gap = p5["eps_ML_rl"] - p5["eps_NN"]; se_gap = math.sqrt(p5["se"] ** 2 + p5["se_NN"] ** 2)
    rows = [x for x in (AW.paired_series(p, W5) for p in pts.values()) if x]
    if rows:
        dbar = float(np.mean([x["delta"] for x in rows])); se_d = math.sqrt(sum(x["se_delta"] ** 2 for x in rows)) / len(rows)
        sh = dbar / gap
        out["nn_gap_share"] = {"series": len(rows), "mean_eps_RL_minus_eps_X": dbar, "se": se_d, "gap_RL_minus_NN": gap,
                               "se_gap": se_gap, "share": sh,
                               "se_share": abs(sh) * math.sqrt((se_d / dbar) ** 2 + (se_gap / gap) ** 2) if dbar else float("nan"),
                               "note": "approximate, as P5"}
    return out


def converged_series(a):
    allser = sorted({os.path.dirname(os.path.relpath(dp, os.path.join(a.res, "fit"))).rsplit("/", 1)[0]
                     for dp, dn, fn in os.walk(os.path.join(a.res, "fit", ARCH))
                     if os.path.basename(dp) == "rl_tsplit" and "fit_data.npz" in fn})
    conv, fits = {"T": list(allser)}, {}
    for X, sub, fn, key in (("D", "f1", "fit_series.json", "converged"), ("Tm", "mech", "mech_fit_T.json", "converged"),
                            ("Dm", "mech", "mech_fit_D.json", "converged")):
        fits[X] = {s: json.load(open(os.path.join(a.r2, sub, s, fn))) for s in allser
                   if os.path.exists(os.path.join(a.r2, sub, s, fn))}
        conv[X] = [s for s in allser if s in fits[X] and fits[X][s][key]]
    return allser, conv, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--r2", required=True)
    ap.add_argument("--m3", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p5 = json.load(open(a.m3))["P5_ML_rl_vs_NN"]["pooled"]["3"]
    allser, conv, fits = converged_series(a)
    R = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "prereg": "notes/decoder/willow_prereg_r2.md sections 8.2 and 8.4", "status": "EXPLORATORY (95% CIs, no test)",
         "series": len(allser), "fits": {X: {"fitted": len(f), "converged": len(conv[X]),
                                            "not_converged": [s for s in f if s not in conv[X]]} for X, f in fits.items()},
         "heldout": {}, "insample": {}}

    def ready(s, sub, fn):
        return all(os.path.exists(os.path.join(a.r2, sub, f"{s}/r{r}", fn)) for r in W5)
    for X in ("Tm", "Dm"):
        ser = [s for s in conv[X] if ready(s, "mech", "mech_data.npz")]
        if ser:
            R["heldout"][X] = block(a, ser, X, load_heldout, p5)
    for X in ("T", "D", "Tm", "Dm"):
        ser = [s for s in conv[X] if ready(s, "insample", "insample_data.npz")]
        ser = [s for s in ser if all(f"{X}_pred" in np.load(os.path.join(a.r2, "insample", f"{s}/r{r}",
                                                                         "insample_data.npz")).files for r in W5)]
        if ser:
            R["insample"][X] = block(a, ser, X, load_insample, p5)
    for X, f in fits.items():
        if X in ("Tm", "Dm"):
            R["fits"][X]["per_series"] = {s: {"converged": v["converged"], "iterations": v["iterations"], "n_eval": v["n_eval"],
                                              "fixed": v["fixed"], "parameters": len(v["names"]),
                                              "cpu_seconds_fit": v.get("cpu_seconds_fit"),
                                              "delta_quantiles": np.quantile(np.array(v["delta_hat"]), [0, .1, .5, .9, 1]).tolist()}
                                          for s, v in f.items()}
            R["fits"][X]["cpu_hours"] = sum(v.get("cpu_seconds_fit", 0.0) for v in f.values()) / 3600
    out = a.out or os.path.join(a.r2, "explore_analysis.json")
    write_json(out, R)
    for split in ("heldout", "insample"):
        for X, b in R[split].items():
            m = b.get("m_X")
            if m:
                print(f"{split:8s} {X:3s} series {b['series']:2d}: m_X = {100 * m['rel']:+.3f}% "
                      f"[{100 * m['ci95'][0]:+.3f}, {100 * m['ci95'][1]:+.3f}]%")
    print("written", out)


if __name__ == "__main__":
    main()
