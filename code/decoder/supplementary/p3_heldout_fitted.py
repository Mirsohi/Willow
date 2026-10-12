#!/usr/bin/env python3
"""EXPLORATORY -- not pre-registered. Requested by the coordinator at milestone 3 (2026-10-09); logged in section 9 of
notes/decoder/willow_prereg.md (entry 6).

How much of the miscalibration of ML(RL DEM) does the likelihood refit T-hat repair? Computed on the held-out halves
of the 72 fitted d = 3 leaves (r in W5 = {10, 13, 30, 50}; 25,000 shots each, 1.8M in total), for ML(T-hat) and for
ML(RL DEM) on the same shots. The frozen analyze_willow.py pools P3 for ML(RL) on all shots only.

1. P3 statistics. R = errors / sum min(P, 1 - P); SE sqrt(sum_leaf k (1 - k/n)) / sum m, as in the frozen pooled P3;
   90% CI; error-count z = (k - sum m) / sqrt(sum m (1 - m)); excess cross-entropy z. The per-leaf sums come from
   calib_sums() of the frozen analyze_willow.py, imported unchanged (a NaN posterior counts as P = 1/2).
2. R_conf: R on the shots with min(P, 1 - P) < 3e-3 under the decoder's own posterior; SE sqrt(k) / sum m, the
   convention of reservoir's mechanism study (code/reservoir/willow_mech/mech_summary.py).
3. Share of the excess removed by the refit: 1 - (R_T - 1) / (R_RL - 1), likewise for R_conf, and 1 - X_T / X_RL for
   the excess cross-entropy X. Delta-method SE with the shot as the unit, stratified by leaf (both decoders on the same
   shots, so the covariance is included; the expected counts are treated as random here).
4. Reliability bins (frozen EDGES), pooled, for both decoders.
5. Stouffer z over the 72 leaves of the held-out syndrome log-likelihood tests stored in fit.json.
6. Checks: stored decisions against the sign of the stored llr; held-out error counts against fit.json; ML(RL) R and
   R_conf on all shots of the 72 leaves from the decode folders (frozen P3: k = 509,651; reservoir: 1,314 confident
   failures against 407.0 predicted).

Inputs are only read. Run:  nice python3 -I -B code/decoder/supplementary/p3_heldout_fitted.py \
    --res results/decoder/willow --out results/decoder/willow/supplementary
"""
import argparse, csv, json, math, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/decoder
import numpy as np
import analyze_willow as AW            # frozen (md5 52fb23ff8e20c0b29b7e19191f38f0d4); used read-only

CONF = 3e-3
TAGS = ("fitted", "shipped")           # ML(T-hat), ML(RL DEM)
NICE = {"fitted": "ML(T-hat)", "shipped": "ML(RL DEM)"}


def per_shot(llr, obs):
    """Per-shot error indicator, m = min(P, 1 - P) and nll - H, exactly as calib_sums() builds its sums."""
    L = llr.astype(np.float64); o = obs.astype(bool)
    nan = np.isnan(L)
    L = np.where(nan, 0.0, L)
    P1 = 1 / (1 + np.exp(-L)); m = 1 / (1 + np.exp(np.abs(L)))
    err = np.where(nan, False, P1 > 0.5) != o
    P1c = np.clip(P1, 1e-300, 1 - 1e-16); P0c = np.clip(1 - P1, 1e-300, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nll = -np.where(o, np.log(P1c), np.log(P0c))
        H = -(P0c * np.log(P0c) + P1c * np.log(P1c))
    x = np.nan_to_num(nll - H)
    return err.astype(np.float64), m, x


def ratio_block(k, sm, smv, var_k):
    R = k / sm
    se = math.sqrt(var_k) / sm
    return {"errors": int(k), "predicted": sm, "ratio": R, "se": se, "ci90": [R - 1.645 * se, R + 1.645 * se],
            "ci95": [R - 1.96 * se, R + 1.96 * se], "z_errors": (k - sm) / math.sqrt(smv) if smv > 0 else float("nan")}


def delta(fun, tot, cov, h=1e-6):
    """Value and delta-method SE of fun(totals) with Var(totals) = cov (central differences)."""
    v = fun(tot)
    g = np.zeros(len(tot))
    for i in range(len(tot)):
        e = np.zeros(len(tot)); e[i] = h * max(abs(tot[i]), 1.0)
        g[i] = (fun(tot + e) - fun(tot - e)) / (2 * e[i])
    return float(v), float(math.sqrt(max(g @ cov @ g, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="results/decoder/willow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-all-shots", action="store_true", help="skip the ML(RL) all-shot check from the decode folders")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    fit_root = os.path.join(a.res, "fit")
    rels = sorted(os.path.relpath(os.path.dirname(dp), fit_root) for dp, dn, fn in os.walk(fit_root)
                  if os.path.basename(dp) == "rl_tsplit" and "fit_data.npz" in fn and "fit.json" in fn)
    K = len(AW.EDGES) - 1
    # columns of the per-shot matrix: e, m, x, e_conf, m_conf for each decoder
    COLS = [f"{c}_{t}" for t in TAGS for c in ("e", "m", "x", "ec", "mc")]
    ix = {c: i for i, c in enumerate(COLS)}
    tot = np.zeros(len(COLS)); cov = np.zeros((len(COLS), len(COLS)))
    acc = {t: {"n": 0, "k": 0, "sum_m": 0.0, "sum_mvar": 0.0, "var_k": 0.0, "sum_nll": 0.0, "sum_H": 0.0,
               "sum_xvar": 0.0, "nan": 0, "leaves": 0, "conf_n": 0, "conf_k": 0, "conf_m": 0.0, "conf_mvar": 0.0,
               "bins": {"n": [0] * K, "pred": [0.0] * K, "var": [0.0] * K, "obs": [0.0] * K}} for t in TAGS}
    checks = {"leaves": len(rels), "fit_status": {}, "pred_vs_llr_disagreements": {t: 0 for t in TAGS},
              "errors_vs_fit_json_differences": {t: 0 for t in TAGS}, "pred_based_errors": {t: 0 for t in TAGS}}
    sll = {t: {"z_mean": [], "z_disp": [], "var_ratio": []} for t in TAGS}
    rows = []
    for rel in rels:
        fp = os.path.join(fit_root, rel, "rl_tsplit")
        fj = json.load(open(os.path.join(fp, "fit.json")))
        checks["fit_status"][fj.get("status")] = checks["fit_status"].get(fj.get("status"), 0) + 1
        z = np.load(os.path.join(fp, "fit_data.npz"))
        obs = z["obs_test"].astype(np.int8)
        have = np.ones(len(obs), bool)
        X = np.zeros((len(obs), len(COLS)))
        row = {"leaf": rel.split("/", 1)[1], "rounds": fj.get("rounds"), "n": int(len(obs)), "classes": len(fj["classes"])}
        for t in TAGS:
            llr, pred = z[f"{t}_llr"], z[f"{t}_pred"]
            cs = AW.calib_sums(llr, obs, have)
            e, m, x = per_shot(llr, obs)
            assert int(e.sum()) == cs["k"] and abs(float(m.sum()) - cs["sum_m"]) <= 1e-9 * max(1.0, cs["sum_m"])
            conf = m < CONF
            X[:, ix[f"e_{t}"]], X[:, ix[f"m_{t}"]], X[:, ix[f"x_{t}"]] = e, m, x
            X[:, ix[f"ec_{t}"]], X[:, ix[f"mc_{t}"]] = e * conf, m * conf
            A = acc[t]
            for k2 in ("n", "k", "sum_m", "sum_mvar", "sum_nll", "sum_H", "sum_xvar", "nan"):
                A[k2] += cs[k2]
            A["var_k"] += cs["k"] * (1 - cs["k"] / max(cs["n"], 1)); A["leaves"] += 1
            A["conf_n"] += int(conf.sum()); A["conf_k"] += int(e[conf].sum())
            A["conf_m"] += float(m[conf].sum()); A["conf_mvar"] += float(np.sum(m[conf] * (1 - m[conf])))
            for k2 in ("n", "pred", "var", "obs"):
                A["bins"][k2] = [u + v for u, v in zip(A["bins"][k2], cs["bins"][k2])]
            p_llr = np.where(np.isnan(llr), False, 1 / (1 + np.exp(-np.nan_to_num(llr.astype(np.float64)))) > 0.5)
            checks["pred_vs_llr_disagreements"][t] += int(np.sum(p_llr != pred.astype(bool)))
            kp = int(np.sum(pred.astype(np.int8) != obs))
            checks["pred_based_errors"][t] += kp
            checks["errors_vs_fit_json_differences"][t] += int(kp != fj["test"][t]["errors"])
            for k2 in ("z_mean", "z_disp", "var_ratio"):
                sll[t][k2].append(float(fj["test"][t]["syndrome_loglik"][k2]))
            row.update({f"errors_{t}": cs["k"], f"predicted_{t}": cs["sum_m"], f"R_{t}": cs["k"] / cs["sum_m"],
                        f"conf_shots_{t}": int(conf.sum()), f"conf_errors_{t}": int(e[conf].sum()),
                        f"conf_predicted_{t}": float(m[conf].sum()),
                        f"synd_z_mean_{t}": fj["test"][t]["syndrome_loglik"]["z_mean"],
                        f"synd_z_disp_{t}": fj["test"][t]["syndrome_loglik"]["z_disp"]})
        tot += X.sum(0)
        D = X - X.mean(0)
        cov += D.T @ D                      # Var(leaf total) = n Var(per-shot), shots i.i.d. within a leaf
        rows.append(row)

    out = {"label": "EXPLORATORY (not pre-registered): calibration of ML(T-hat) and ML(RL DEM) on the held-out halves",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": os.path.abspath(fit_root),
           "frozen_analysis_md5_used": "52fb23ff8e20c0b29b7e19191f38f0d4", "conf_threshold": CONF, "checks": checks,
           "by_decoder": {}, "refit_share_of_excess_removed": {}, "reliability_bins": {}, "syndrome_loglik_stouffer": {}}
    for t in TAGS:
        A = acc[t]
        b = ratio_block(A["k"], A["sum_m"], A["sum_mvar"], A["var_k"])
        b.update(n=A["n"], leaves=A["leaves"], nan_posteriors=A["nan"],
                 excess_xent_per_shot=(A["sum_nll"] - A["sum_H"]) / A["n"],
                 z_xent=(A["sum_nll"] - A["sum_H"]) / math.sqrt(A["sum_xvar"]))
        Rc = A["conf_k"] / A["conf_m"]
        b["conf"] = {"shots": A["conf_n"], "share_of_shots": A["conf_n"] / A["n"], "errors": A["conf_k"],
                     "predicted": A["conf_m"], "ratio": Rc, "se_sqrtk": math.sqrt(A["conf_k"]) / A["conf_m"],
                     "ci90": [Rc - 1.645 * math.sqrt(A["conf_k"]) / A["conf_m"], Rc + 1.645 * math.sqrt(A["conf_k"]) / A["conf_m"]],
                     "z_errors": (A["conf_k"] - A["conf_m"]) / math.sqrt(A["conf_mvar"])}
        # the rest of the shots (min(P, 1 - P) >= 3e-3)
        kr, mr = A["k"] - A["conf_k"], A["sum_m"] - A["conf_m"]
        b["not_conf"] = {"shots": A["n"] - A["conf_n"], "errors": kr, "predicted": mr, "ratio": kr / mr}
        # design-based SE of R (expected count random as well), for comparison with the frozen convention
        b["se_design"] = delta(lambda T, t=t: T[ix[f"e_{t}"]] / T[ix[f"m_{t}"]], tot, cov)[1]
        out["by_decoder"][NICE[t]] = b
        bins = A["bins"]
        out["reliability_bins"][NICE[t]] = [
            {"lo": float(AW.EDGES[i]), "hi": float(min(AW.EDGES[i + 1], 0.5)), "n": bins["n"][i], "observed": bins["obs"][i],
             "predicted": bins["pred"][i], "ratio": bins["obs"][i] / bins["pred"][i] if bins["pred"][i] > 0 else None}
            for i in range(K) if bins["n"][i]]
        out["syndrome_loglik_stouffer"][NICE[t]] = {
            "leaves": len(sll[t]["z_mean"]), "stouffer_z_mean": float(np.sum(sll[t]["z_mean"]) / math.sqrt(len(sll[t]["z_mean"]))),
            "stouffer_z_disp": float(np.sum(sll[t]["z_disp"]) / math.sqrt(len(sll[t]["z_disp"]))),
            "median_var_ratio": float(np.median(sll[t]["var_ratio"]))}

    def R(T, t, c=""):
        return T[ix[f"e{c}_{t}"]] / T[ix[f"m{c}_{t}"]]
    for name, fun in (
            ("R", lambda T: 1 - (R(T, "fitted") - 1) / (R(T, "shipped") - 1)),
            ("R_conf", lambda T: 1 - (R(T, "fitted", "c") - 1) / (R(T, "shipped", "c") - 1)),
            ("excess_xent", lambda T: 1 - T[ix["x_fitted"]] / T[ix["x_shipped"]])):
        v, se = delta(fun, tot, cov)
        out["refit_share_of_excess_removed"][name] = {"share": v, "se": se, "ci90": [v - 1.645 * se, v + 1.645 * se],
                                                      "ci95": [v - 1.96 * se, v + 1.96 * se]}
    for name, c in (("R", ""), ("R_conf", "c")):
        v, se = delta(lambda T, c=c: R(T, "fitted", c) - R(T, "shipped", c), tot, cov)
        out["refit_share_of_excess_removed"][f"difference_{name}_fitted_minus_shipped"] = {
            "value": v, "se": se, "z": v / se, "ci90": [v - 1.645 * se, v + 1.645 * se]}
    out["refit_share_of_excess_removed"]["definition"] = (
        "share = 1 - (R_T - 1)/(R_RL - 1) on the same held-out shots (1 = fully repaired, 0 = unchanged, < 0 = worse); "
        "for the cross-entropy 1 - X_T/X_RL with X the summed excess nll - H. Delta-method SE, shot as the unit, "
        "stratified by leaf; R_conf uses each decoder's own confident shots.")

    if not a.no_all_shots:            # ML(RL) on all shots of the same 72 leaves, from the decode folders
        al = {"n": 0, "k": 0, "sum_m": 0.0, "conf_n": 0, "conf_k": 0, "conf_m": 0.0}
        for rel in rels:
            obs, ents, ml, info, _ = AW.load_leaf(a.res, rel, ["rl"])
            mm = ml["ML.rl"]
            sel = mm["have"] & (obs >= 0)
            cs = AW.calib_sums(mm["llr"], obs, mm["have"])
            e, m, _x = per_shot(mm["llr"][sel], obs[sel])
            conf = m < CONF
            al["n"] += cs["n"]; al["k"] += cs["k"]; al["sum_m"] += cs["sum_m"]
            al["conf_n"] += int(conf.sum()); al["conf_k"] += int(e[conf].sum()); al["conf_m"] += float(m[conf].sum())
        al.update(R=al["k"] / al["sum_m"], R_conf=al["conf_k"] / al["conf_m"], R_conf_se=math.sqrt(al["conf_k"]) / al["conf_m"],
                  frozen_P3_k=509651, frozen_P3_sum_m=418761.7493722092, reservoir_R_conf=[3.23, 0.09],
                  reservoir_counts="1,314 confident failures against 407.0 predicted (board/reservoir.md 21:33Z)")
        out["ML_RL_all_shots_check"] = al
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(a.out, "p3_heldout_fitted.json"), "w"), indent=1)
    with open(os.path.join(a.out, "p3_heldout_fitted_leaves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({k: out[k] for k in ("checks", "by_decoder", "refit_share_of_excess_removed",
                                          "syndrome_loglik_stouffer")}, indent=1))
    if "ML_RL_all_shots_check" in out:
        print(json.dumps(out["ML_RL_all_shots_check"], indent=1))
    print(f"done in {out['seconds']:.0f} s")


if __name__ == "__main__":
    main()
