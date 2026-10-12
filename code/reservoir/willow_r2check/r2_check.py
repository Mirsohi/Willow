"""Independent check of round 2's F1 result: the primary m_D, its SE and z, the cheap secondaries, the held-out and fit
shots, and the fits' records (checks 1, 4 and 5 of main's brief, 2026-10-10).

reservoir, 2026-10-10. Written from notes/decoder/willow_prereg_r2.md sections 2-4 and 7. The numbers come from the
per-shot predictions: D-hat's (round2/f1/<leaf>/d_data.npz), ML(RL)'s (round 1's full-leaf decodes, decode/<leaf>/rl/
ml_exact.npz, checked against fit_data.npz's shipped_pred), ML(T-hat)'s (fit_data.npz) and the archive's observable
flips; eval.json and f1_analysis.json are only compared with at the end.

Inputs (read only; run with python3 -I because the archive is downloaded data):
  ARCHIVE  the archive extract (detection_events.b8, obs_flips_actual.b8, sweep_bits.b8, the RL DEM)
  DECODE   results/decoder/willow/decode/<arch>: round 1's ML(RL) decodes of all 50,000 shots per leaf
  FIT      results/decoder/willow/fit/<arch>: round 1's fit.json / fit_data.npz (split, fit_idx, ML(T-hat))
  F1       results/decoder/willow/round2/f1/<arch>: fit_series.json per series; eval.json, d_data.npz per leaf
  ANALYSIS results/decoder/willow/round2/f1_analysis.json (compared with, not used)

What is computed (the 72 W5 leaves, 18 series):
  (1) m_D = (K_RL - K_D) / K_D with P4's SE (delta method per shot, stratified by leaf), z, p, 90% and 95% CIs; by r,
      without r = 13, per series (descriptive, with a heterogeneity chi2), ML(T-hat) against ML(D-hat), and P4 again;
  (2) secondaries: held-out conditional log-loss and paired differences (SE stratified by leaf); R and R_conf of ML(D-hat)
      and ML(RL) on the same shots; the share of RL's excess removed (delta method on the four totals, stratified);
  (3) descriptive: ln P(s) under D-hat and T-hat against RL on the held-out shots, and the mean number of detection
      events per shot that each DEM predicts against the data; the series' in-sample gain against its held-out gain;
  (4) checks: the held-out shots are P4's (odd rank within each initialisation bitstring) and equal round 1's test_idx;
      the fit shots (round 1's fit_idx) are in the even ranks and disjoint from them; each series' fingerprint
      recomputed from the archive's fit shots and the rebuilt class names equals fit_series.json's; the DEM file's md5
      equals fit.json's; the rebuilt classes equal fit.json's and eval.json's; eval.json's scales equal exp(phi_hat);
      D_pred is the argmax of D_llr; the fit records (convergence, last step, score, non-finite shots).
Usage: nice -n 10 python3 -I r2_check.py ARCHIVE DECODE FIT F1 ANALYSIS OUT_DIR
  -> OUT_DIR/r2_leaves.csv, OUT_DIR/r2_check.json (through qrc.outpaths)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (ARCH, BASES, CONF, PATCHES, SHOTS, W5, det_marginals, detection_events, dem_path,  # noqa: E402
                       leaf_model, md5_file, min_p_from_llr, nll_from_llr, ratio_se, read_b8_bit0, rel_of, scaled,
                       series_fit, series_of, split_rank, theta_by_name)
from qrc.outpaths import out_path  # noqa: E402

Z90, Z95 = 1.6448536269514722, 1.959963984540054


def ratio_se_strata(num, den, leaf_of):
    """As p4_check.ratio_se, but a stratum with fewer than 2 shots adds no variance (for the confident subsets)."""
    A, B = float(num.sum()), float(den.sum())
    Q = A / B
    v = 0.0
    for l in np.unique(leaf_of):
        s = leaf_of == l
        if s.sum() > 1:
            d = num[s] - Q * den[s]
            v += s.sum() * d.var(ddof=1)
    return Q, math.sqrt(v) / B


def ratio_block(num, den, leaf_of, minus_one=True):
    Q, se = (ratio_se if minus_one else ratio_se_strata)(num.astype(float), den.astype(float), leaf_of)
    m = Q - 1 if minus_one else Q
    z = m / se if minus_one else (Q - 1) / se
    return {"num": int(num.sum()) if num.dtype.kind in "iub" else float(num.sum()),
            "den": int(den.sum()) if den.dtype.kind in "iub" else float(den.sum()),
            "shots": int(num.size), "value": m, "se": se, "z": z, "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "ci90": [m - Z90 * se, m + Z90 * se], "ci95": [m - Z95 * se, m + Z95 * se]}


def mean_se(x, leaf_of):
    """Mean of x with the SE stratified by leaf (shot as the unit)."""
    v = 0.0
    for l in np.unique(leaf_of):
        s = leaf_of == l
        v += s.sum() * x[s].var(ddof=1)
    n = x.size
    mu = float(x.mean())
    se = math.sqrt(v) / n
    return {"mean": mu, "se": se, "z": mu / se if se > 0 else None, "ci95": [mu - Z95 * se, mu + Z95 * se]}


def share_removed(eD, mD, eR, mR, leaf_of, conf=None):
    """1 - (R_D - 1) / (R_RL - 1) with R = sum(e) / sum(m), delta method on the four totals (stratified by leaf)."""
    sel = np.ones(eD.size, bool) if conf is None else conf
    V = np.stack([eD[sel], mD[sel], eR[sel], mR[sel]], 1).astype(float)
    lf = leaf_of[sel]
    T = V.sum(0)
    C = np.zeros((4, 4))
    for l in np.unique(lf):
        s = lf == l
        if s.sum() > 1:
            C += s.sum() * np.cov(V[s].T, ddof=1)
    RD, RR = T[0] / T[1], T[2] / T[3]
    S = 1 - (RD - 1) / (RR - 1)
    g = np.array([-1 / T[1] / (RR - 1), T[0] / T[1] ** 2 / (RR - 1),
                  (RD - 1) / (RR - 1) ** 2 / T[3], -(RD - 1) / (RR - 1) ** 2 * T[2] / T[3] ** 2])
    se = math.sqrt(float(g @ C @ g))
    return {"share": S, "se": se, "ci95": [S - Z95 * se, S + Z95 * se], "R_D": RD, "R_RL": RR,
            "errors_D": T[0], "predicted_D": T[1], "errors_RL": T[2], "predicted_RL": T[3], "shots": int(sel.sum())}


def leaf(arch, dec, fit, f1, patch, basis, r, fs, th_D):
    rel = rel_of(patch, basis, r)
    lname = f"{series_of(patch, basis)}/r{r}"
    f_dir = os.path.join(fit, rel, "rl_tsplit")
    fj = json.load(open(os.path.join(f_dir, "fit.json")))
    fd = np.load(os.path.join(f_dir, "fit_data.npz"), allow_pickle=False)
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    dd = np.load(os.path.join(f1, rel, "d_data.npz"), allow_pickle=False)
    ev = json.load(open(os.path.join(f1, rel, "eval.json")))
    obs = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)
    rank = split_rank(arch, rel)
    chk = {}
    # --- the shots
    ti = dd["test_idx"].astype(np.int64)
    fi = fd["fit_idx"].astype(np.int64)
    chk["d_test_idx_equals_round1_test_idx"] = bool(np.array_equal(ti, fd["test_idx"].astype(np.int64)))
    chk["d_test_idx_is_odd_rank_rule"] = bool(np.array_equal(np.sort(ti), np.flatnonzero(rank % 2 == 1)))
    chk["d_test_idx_sorted_unique"] = bool(np.all(np.diff(ti) > 0))
    chk["fit_idx_in_even_ranks"] = bool(np.all(rank[fi] % 2 == 0))
    chk["fit_idx_unique"] = bool(np.unique(fi).size == fi.size)
    chk["fit_test_disjoint"] = bool(np.intersect1d(fi, ti).size == 0)
    chk["n_fit_n_test"] = [int(fi.size), int(ti.size)]
    o = obs[ti].astype(np.int8)
    chk["d_obs_test_equals_archive"] = bool(np.array_equal(dd["obs_test"].astype(np.int8), o))
    # --- the predictions
    if not np.array_equal(ml["idx"], np.arange(SHOTS)):
        raise SystemExit(f"ML(RL) decode not complete and in order at {rel}")
    rl_pred = ml["pred"].reshape(SHOTS, -1)[:, 0][ti].astype(np.int8)
    rl_llr = ml["llr"][ti]
    chk["ml_rl_pred_equals_shipped_pred"] = bool(np.array_equal(rl_pred, fd["shipped_pred"].astype(np.int8)))
    chk["ml_rl_llr_equals_shipped_llr"] = bool(np.array_equal(rl_llr, fd["shipped_llr"]))
    Dp, Dl, Dlogp = dd["D_pred"].astype(np.int8), dd["D_llr"], dd["D_logp"]
    chk["D_llr_nan"] = int(np.sum(~np.isfinite(Dl)))
    chk["D_llr_exact_zero"] = int(np.sum(Dl == 0))
    chk["D_pred_is_argmax_of_D_llr"] = bool(np.array_equal(Dp, (np.nan_to_num(Dl, nan=0.0) > 0).astype(np.int8)))
    chk["D_logp_nonfinite"] = int(np.sum(~np.isfinite(Dlogp)))
    # --- the model: DEM file, classes, scales
    chk["dem_md5_equals_fit_json"] = md5_file(dem_path(arch, rel)) == fj["dem_md5"]
    thr = float(fj["fit_attempts"][-1]["min_class_events"])
    M = leaf_model(arch, rel, r, fj["n_fit"], thr)
    chk["dem_duplicates_zero_empty"] = [M["n_dup"], M["n_zero"], M["n_empty"]]
    chk["classes_rebuilt_equal_fit_json"] = M["names"] == sorted(fj["classes"])
    chk["classes_rebuilt_equal_eval_json"] = M["names"] == sorted(ev["theta"])
    chk["eval_theta_equals_exp_phi_hat"] = (max(abs(ev["theta"][nm] / th_D[nm] - 1) for nm in M["names"])
                                            if chk["classes_rebuilt_equal_eval_json"] else None)
    chk["eval_fingerprint_equals_fit_series"] = ev.get("fingerprint") == fs["fingerprint"]
    chk["eval_converged"] = bool(ev.get("converged"))
    chk["eval_leaf_name"] = ev.get("leaf") == lname
    # --- per shot
    a = (rl_pred != o).astype(np.int8)
    d = (Dp != o).astype(np.int8)
    t = (fd["fitted_pred"].astype(np.int8) != o).astype(np.int8)
    arr = {"a": a, "d": d, "t": t,
           "nll_RL": nll_from_llr(rl_llr, o), "nll_T": nll_from_llr(fd["fitted_llr"], o), "nll_D": nll_from_llr(Dl, o),
           "m_RL": min_p_from_llr(rl_llr), "m_T": min_p_from_llr(fd["fitted_llr"]), "m_D": min_p_from_llr(Dl),
           "lp_RL": ml["logp"][ti].astype(np.float64), "lp_T": fd["fitted_logp"].astype(np.float64),
           "lp_D": Dlogp.astype(np.float64)}
    chk["logp_RL_ml_vs_shipped_max_abs"] = float(np.max(np.abs(arr["lp_RL"] - fd["shipped_logp"])))
    # --- events per shot: data (held-out shots) against each DEM's prediction
    det_all = detection_events(arch, rel, M["nd"])
    ev_data = float(det_all[ti].sum(1).mean())
    mD, nclip = scaled(M["mech"], M["cls"], th_D)
    thT = dict(zip(fj["classes"], fj["scale_hat"]))
    mT, _ = scaled(M["mech"], M["cls"], thT)
    chk["D_hat_faults_clipped"] = nclip
    events = {"data": ev_data, "RL": float(det_marginals(M["mech"], M["nd"]).sum()),
              "T_hat": float(det_marginals(mT, M["nd"]).sum()), "D_hat": float(det_marginals(mD, M["nd"]).sum())}
    # --- the fit shots, for the fingerprint (bytes exactly as F1's Leaf stores them)
    det_fit = np.ascontiguousarray(det_all[fi], dtype=np.uint8)
    L_fit = np.asarray(obs[fi], dtype=np.int64).reshape(-1)
    # --- comparison with eval.json (after the counts above)
    cells = {"n": int(o.size), "n10": int(np.sum((a == 1) & (d == 0))), "n01": int(np.sum((a == 0) & (d == 1))),
             "n11": int(np.sum((a == 1) & (d == 1)))}
    cellsT = {"n": int(o.size), "n10": int(np.sum((t == 1) & (d == 0))), "n01": int(np.sum((t == 0) & (d == 1))),
              "n11": int(np.sum((t == 1) & (d == 1)))}
    chk["eval_errors_equal"] = ev["errors"] == {"D": int(d.sum()), "RL": int(a.sum()), "T": int(t.sum())}
    chk["eval_cells_RL_vs_D_equal"] = {k: ev["cells_RL_vs_D"][k] for k in cells} == cells
    chk["eval_cells_T_vs_D_equal"] = {k: ev["cells_T_vs_D"][k] for k in cellsT} == cellsT
    row = {"patch": patch, "basis": basis, "r": r, "n_test": int(o.size), "n_fit": int(fi.size),
           "n_classes": len(M["names"]), "K_RL": int(a.sum()), "K_D": int(d.sum()), "K_T": int(t.sum()),
           "RL_wrong_D_right": cells["n10"], "RL_right_D_wrong": cells["n01"], "both_wrong": cells["n11"],
           "nll_RL": float(arr["nll_RL"].mean()), "nll_D": float(arr["nll_D"].mean()),
           "nll_T": float(arr["nll_T"].mean()),
           "R_RL": float(a.sum() / arr["m_RL"].sum()), "R_D": float(d.sum() / arr["m_D"].sum()),
           "lnPs_D_minus_RL": float(np.mean(arr["lp_D"] - arr["lp_RL"])),
           "lnPs_T_minus_RL": float(np.mean(arr["lp_T"] - arr["lp_RL"])),
           "events_data": events["data"], "events_RL": events["RL"], "events_T": events["T_hat"],
           "events_D": events["D_hat"]}
    row.update({f"chk_{k}": v for k, v in chk.items()})
    return row, arr, chk, (lname, det_fit, L_fit), M["names"]


def main(arch, dec, fit, f1, analysis, out_dir):
    t0 = time.time()
    rows, chks, arrs, leaf_ids, ser_of = [], [], [], [], []
    series_rep = {}
    li = 0
    for p in PATCHES:
        for b in BASES:
            s = series_of(p, b)
            fs = series_fit(f1, p, b)
            th = theta_by_name(fs)
            h = hashlib.md5()
            names_all = set()
            leaf_rows = []
            for r in W5:
                row, arr, chk, (lname, det_fit, L_fit), names = leaf(arch, dec, fit, f1, p, b, r, fs, th)
                h.update(lname.encode()); h.update(det_fit.tobytes()); h.update(L_fit.tobytes())
                names_all |= set(names)
                row["_i"] = li
                rows.append(row); chks.append(chk); arrs.append(arr)
                leaf_ids.append(np.full(row["n_test"], li, np.int32)); ser_of.append(s)
                leaf_rows.append(row)
                li += 1
                print(f"{p} {b} r{r}: K_RL {row['K_RL']} K_D {row['K_D']} K_T {row['K_T']} ({time.time() - t0:.0f} s)",
                      flush=True)
            h.update(json.dumps(sorted(names_all)).encode())
            fp = h.hexdigest() + f"|info={fs['information']}|secant={fs['secant']}"
            lc = fs["last_convergence_check"]
            series_rep[s] = {
                "fingerprint_recomputed_equals": fp == fs["fingerprint"],
                "names_equal": sorted(names_all) == fs["names"],
                "converged": fs["converged"], "stopped": fs["stopped"], "iterations": fs["iterations"],
                "n_eval": fs["n_eval"], "fixed": fs["fixed"], "n_shots": fs["n_shots"], "nonfinite": fs["nonfinite"],
                "last_step_max_in_se": lc["max_step_in_se"], "last_step_argmax": lc["argmax"],
                "last_check_iteration_equals_iterations": lc["iteration"] == fs["iterations"],
                "mean_score_max_abs": float(np.max(np.abs(fs["mean_score_at_hat"]))),
                "phi_equals_ln_theta": float(np.max(np.abs(np.array(fs["phi_hat"]) - np.log(fs["theta_hat"])))),
                "theta_equals_last_history": bool(np.allclose(fs["history"][-1]["theta"], fs["theta_hat"], rtol=1e-12,
                                                              atol=0)),
                "ll_equals_last_history": fs["history"][-1]["ll"] == fs["ll_per_shot"],
                "in_sample_ll_RL": fs["history"][0]["ll"], "in_sample_ll_D": fs["ll_per_shot"],
                "in_sample_gain_per_shot": fs["ll_per_shot"] - fs["history"][0]["ll"],
                "history_theta0_is_1": bool(np.all(np.array(fs["history"][0]["theta"]) == 1.0)),
                "n_classes": len(fs["names"]), "cpu_seconds_total": fs.get("cpu_seconds_total")}
    leaf_of = np.concatenate(leaf_ids)
    A = {k: np.concatenate([x[k] for x in arrs]) for k in arrs[0]}
    r_of_leaf = np.array([row["r"] for row in rows])
    r_of = r_of_leaf[leaf_of]
    s_names = sorted(series_rep)
    s_idx_of_leaf = np.array([s_names.index(x) for x in ser_of])
    s_of = s_idx_of_leaf[leaf_of]
    res = {"label": "reservoir, independent check of round 2's F1 (decoder's D-hat), 2026-10-10",
           "inputs": {"archive": arch, "decode": dec, "fit": fit, "f1": f1, "analysis": analysis}}
    # (1) primary and its relatives
    allm = np.ones(leaf_of.size, bool)

    def mD(sel):
        x = ratio_block(A["a"][sel], A["d"][sel], leaf_of[sel])
        x.update({"K_RL": x.pop("num"), "K_D": x.pop("den"), "leaves": int(np.unique(leaf_of[sel]).size),
                  "RL_wrong_D_right": int(np.sum((A["a"][sel] == 1) & (A["d"][sel] == 0))),
                  "RL_right_D_wrong": int(np.sum((A["a"][sel] == 0) & (A["d"][sel] == 1)))})
        return x
    res["primary_m_D"] = mD(allm)
    pm = res["primary_m_D"]
    sig = pm["p_two_sided"] < 0.05
    res["primary_m_D"]["reading_section3"] = (
        "significant and >= +3%" if sig and pm["value"] >= 0.03 else
        "significant and below +3%" if sig and pm["value"] > 0 else
        "significant and negative" if sig else "not significant")
    res["primary_m_D"]["series_converged"] = sum(v["converged"] for v in series_rep.values())
    res["m_D_by_r"] = {str(r): mD(r_of == r) for r in W5}
    res["m_D_without_r13"] = mD(r_of != 13)
    per_s = {s: mD(s_of == i) for i, s in enumerate(s_names)}
    w = np.array([1 / per_s[s]["se"] ** 2 for s in s_names])
    v = np.array([per_s[s]["value"] for s in s_names])
    mbar = float((w * v).sum() / w.sum())
    Qh = float((w * (v - mbar) ** 2).sum())
    res["m_D_per_series"] = {s.split("/", 1)[1]: {k: per_s[s][k] for k in ("value", "se", "K_RL", "K_D")}
                             for s in s_names}
    res["m_D_series_heterogeneity"] = {"inverse_variance_mean": mbar, "Q": Qh, "df": len(s_names) - 1,
                                       "p": float(stats.chi2.sf(Qh, len(s_names) - 1)),
                                       "series_with_m_D_above_0": int(np.sum(v > 0)),
                                       "note": "descriptive; per-series values from the same stratified estimator"}
    x = ratio_block(A["t"], A["d"], leaf_of)
    x.update({"K_T": x.pop("num"), "K_D": x.pop("den")})
    res["ratio_T_vs_D"] = x
    x = ratio_block(A["a"], A["t"], leaf_of)
    x.update({"K_RL": x.pop("num"), "K_T": x.pop("den")})
    res["P4_again_RL_vs_T"] = x
    # (2) secondaries
    ll = {"mean_nll": {k: float(A[f"nll_{k}"].mean()) for k in ("RL", "T", "D")}, "shots": int(leaf_of.size)}
    for k, (u, v_) in {"RL-D": ("RL", "D"), "T-D": ("T", "D"), "RL-T": ("RL", "T")}.items():
        ll[f"diff_{k}"] = mean_se(A[f"nll_{u}"] - A[f"nll_{v_}"], leaf_of)
    ll["diff_RL-D_without_r13"] = mean_se(A["nll_RL"][r_of != 13] - A["nll_D"][r_of != 13], leaf_of[r_of != 13])
    res["logloss"] = ll
    cal = {}
    for tag, e, m in (("D", A["d"], A["m_D"]), ("RL", A["a"], A["m_RL"]), ("T", A["t"], A["m_T"])):
        Rx = ratio_block(e, m, leaf_of, minus_one=False)
        cf = m < CONF
        Rc = ratio_block(e[cf], m[cf], leaf_of[cf], minus_one=False)
        cal[tag] = {"errors": int(e.sum()), "predicted": float(m.sum()), "R": Rx["value"], "R_se": Rx["se"],
                    "R_ci95": Rx["ci95"], "conf_shots": int(cf.sum()), "conf_errors": int(e[cf].sum()),
                    "conf_predicted": float(m[cf].sum()), "R_conf": Rc["value"], "R_conf_se": Rc["se"],
                    "R_conf_se_sqrt_k": math.sqrt(e[cf].sum()) / float(m[cf].sum())}
    res["calibration"] = cal
    cfD, cfR = (A["m_D"] < CONF).astype(float), (A["m_RL"] < CONF).astype(float)
    res["share_of_RL_excess_removed"] = {
        "R": share_removed(A["d"], A["m_D"], A["a"], A["m_RL"], leaf_of),
        "R_conf": share_removed(A["d"] * cfD, A["m_D"] * cfD, A["a"] * cfR, A["m_RL"] * cfR, leaf_of),
        "R_without_r13": share_removed(A["d"], A["m_D"], A["a"], A["m_RL"], leaf_of, conf=(r_of != 13))}
    res["share_of_RL_excess_removed"]["R_conf"]["note"] = ("each R_conf on its own confident shots (min(P, 1 - P) < "
                                                           "3e-3 under D-hat, resp. RL); delta method on the four totals")
    # (3) descriptive
    lp = {}
    for r in W5 + ["all"]:
        s = allm if r == "all" else (r_of == r)
        lp[str(r)] = {"D_minus_RL": float(np.mean(A["lp_D"][s] - A["lp_RL"][s])),
                      "T_minus_RL": float(np.mean(A["lp_T"][s] - A["lp_RL"][s])),
                      "D_minus_RL_per_round": (float(np.mean((A["lp_D"][s] - A["lp_RL"][s]) / r_of[s]))),
                      "RL_mean": float(np.mean(A["lp_RL"][s]))}
    res["lnPs_heldout"] = lp
    evr = {}
    for r in W5:
        rr = [row for row in rows if row["r"] == r]
        evr[str(r)] = {k: float(np.mean([row[f"events_{k}"] for row in rr])) for k in ("data", "RL", "T", "D")}
        evr[str(r)].update({f"{k}_over_data": evr[str(r)][k] / evr[str(r)]["data"] for k in ("RL", "T", "D")})
    res["events_per_shot_by_r"] = evr
    ins = []
    for i, s in enumerate(s_names):
        sel = s_of == i
        ho = mean_se(A["nll_RL"][sel] - A["nll_D"][sel], leaf_of[sel])
        series_rep[s]["heldout_gain_per_shot"] = ho["mean"]
        series_rep[s]["heldout_gain_se"] = ho["se"]
        series_rep[s]["in_sample_minus_heldout"] = series_rep[s]["in_sample_gain_per_shot"] - ho["mean"]
        ins.append(series_rep[s]["in_sample_minus_heldout"])
    ins = np.array(ins)
    res["in_sample_vs_heldout_gain"] = {
        "mean_in_sample_gain": float(np.mean([series_rep[s]["in_sample_gain_per_shot"] for s in s_names])),
        "mean_heldout_gain": float(np.mean([series_rep[s]["heldout_gain_per_shot"] for s in s_names])),
        "mean_difference": float(ins.mean()), "sd_difference": float(ins.std(ddof=1)),
        "expected_optimism_k_over_n": float(np.mean([series_rep[s]["n_classes"] / series_rep[s]["n_shots"]
                                                     for s in s_names])),
        "note": "in-sample: fit_series.json's ll at theta-hat minus at theta = 1 (20,000 fit shots); held out: mean "
                "nll_RL - nll_D over the series' 100,000 held-out shots; AIC-type optimism of the in-sample gain "
                "over the held-out one is about k/n (assumed, a correct model)"}
    res["series"] = {s.split("/", 1)[1]: series_rep[s] for s in s_names}
    # (4) checks, summarised
    agg = {}
    for c in chks:
        for k, v_ in c.items():
            agg.setdefault(k, []).append(v_)
    summ = {}
    for k, vals in agg.items():
        if all(isinstance(x, bool) for x in vals):
            summ[k] = f"{sum(vals)} of {len(vals)}"
        elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals if x is not None):
            vv = [x for x in vals if x is not None]
            summ[k] = {"max": max(vv), "sum": sum(vv)} if vv else None
        else:
            summ[k] = sorted({json.dumps(x) for x in vals})
    for k in ("fingerprint_recomputed_equals", "names_equal", "converged", "history_theta0_is_1",
              "theta_equals_last_history", "ll_equals_last_history", "last_check_iteration_equals_iterations"):
        summ[f"series_{k}"] = f"{sum(bool(series_rep[s][k]) for s in s_names)} of {len(s_names)}"
    summ["series_last_step_max_in_se"] = max(series_rep[s]["last_step_max_in_se"] for s in s_names)
    summ["series_iterations_max"] = max(series_rep[s]["iterations"] for s in s_names)
    summ["series_fixed_classes"] = sum(len(series_rep[s]["fixed"]) for s in s_names)
    summ["series_nonfinite"] = sum(series_rep[s]["nonfinite"] for s in s_names)
    summ["series_mean_score_max_abs"] = max(series_rep[s]["mean_score_max_abs"] for s in s_names)
    res["checks"] = summ
    # comparison with decoder's analysis (read only now)
    fa = json.load(open(analysis))
    cmp_ = {}
    pmD = fa["primary_m_D"]["pooled"]
    cmp_["m_D"] = {"decoder": pmD["rel"], "here": pm["value"], "abs_diff": abs(pmD["rel"] - pm["value"])}
    cmp_["m_D_se"] = {"decoder": pmD["se_rel"], "here": pm["se"], "ratio": pm["se"] / pmD["se_rel"]}
    cmp_["errors"] = {"decoder": [pmD["errors_A"], pmD["errors_B"]], "here": [pm["K_RL"], pm["K_D"]]}
    cmp_["decision"] = {"decoder": fa["primary_m_D"]["decision"], "here": pm["reading_section3"]}
    x13 = fa["m_D_without_r13"]
    cmp_["m_D_without_r13"] = {"decoder": [x13["rel"], x13["se_rel"]],
                               "here": [res["m_D_without_r13"]["value"], res["m_D_without_r13"]["se"]]}
    xt = fa["ratio_T_vs_D"]
    cmp_["ratio_T_vs_D"] = {"decoder": [xt["rel"], xt["se_rel"]],
                            "here": [res["ratio_T_vs_D"]["value"], res["ratio_T_vs_D"]["se"]]}
    fl = fa["logloss"]
    cmp_["logloss_mean_nll"] = {"decoder": fl["mean_nll"], "here": ll["mean_nll"]}
    for k in ("RL-D", "T-D", "RL-T"):
        cmp_[f"logloss_diff_{k}"] = {"decoder": [fl[f"diff_{k}"]["mean"], fl[f"diff_{k}"]["se"]],
                                     "here": [ll[f"diff_{k}"]["mean"], ll[f"diff_{k}"]["se"]]}
    fc = fa["calibration_D"]
    cmp_["R_D"] = {"decoder": [fc["ML(D-hat)"]["ratio"], fc["ML(D-hat)"]["se"]], "here": [cal["D"]["R"], cal["D"]["R_se"]]}
    cmp_["R_conf_D"] = {"decoder": [fc["ML(D-hat)"]["conf"]["errors"], fc["ML(D-hat)"]["conf"]["predicted"]],
                        "here": [cal["D"]["conf_errors"], cal["D"]["conf_predicted"]]}
    sh = fc["share_of_RL_excess_removed"]
    cmp_["share_R"] = {"decoder": [sh["R"]["share"], sh["R"]["se"]],
                       "here": [res["share_of_RL_excess_removed"]["R"]["share"],
                                res["share_of_RL_excess_removed"]["R"]["se"]]}
    cmp_["share_R_conf"] = {"decoder": [sh["R_conf"]["share"], sh["R_conf"]["se"]],
                            "here": [res["share_of_RL_excess_removed"]["R_conf"]["share"],
                                     res["share_of_RL_excess_removed"]["R_conf"]["se"]]}
    cmp_["R_RL_same_shots"] = {"decoder": sh["R_RL_same_shots"], "here": cal["RL"]["R"]}
    cmp_["decoder_problems"] = fa["checks"]["problems"]
    cmp_["decoder_series_converged"] = fa["series_converged"]
    res["comparison_with_f1_analysis"] = cmp_
    res["run_seconds"] = time.time() - t0
    rows_out = [{k: v_ for k, v_ in row.items() if k != "_i"} for row in rows]
    with open(out_path(os.path.join(out_dir, "r2_leaves.csv")), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        wr.writeheader()
        for row in rows_out:
            wr.writerow({k: (json.dumps(v_) if isinstance(v_, (list, dict)) else v_) for k, v_ in row.items()})
    with open(out_path(os.path.join(out_dir, "r2_check.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    show = {k: res[k] for k in ("primary_m_D", "m_D_without_r13", "ratio_T_vs_D", "P4_again_RL_vs_T", "logloss",
                                "calibration", "share_of_RL_excess_removed", "checks", "comparison_with_f1_analysis",
                                "in_sample_vs_heldout_gain", "lnPs_heldout", "events_per_shot_by_r",
                                "m_D_series_heterogeneity")}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
