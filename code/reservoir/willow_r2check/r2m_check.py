"""Independent check of round 2's mech fits, T-hat_mech (Tm) and D-hat_mech (Dm) (notes/decoder/willow_prereg_r2.md
sections 8.2 and 8.4): the held-out and in-sample statistics recomputed from the per-shot predictions, the shots, and
the fits' records and fingerprints (main's brief of 2026-10-10: "the same checks as for F1"). Written before the real
mech fits existed; to be run when main asks.

reservoir, 2026-10-10. Inputs (read only; python3 -I, as the archive is downloaded data):
  ARCHIVE   the archive extract (detection events, observables, sweep bits, the RL DEM)
  DECODE    results/decoder/willow/decode/<arch>: round 1's ML(RL) decodes of all 50,000 shots per leaf
  FIT       results/decoder/willow/fit/<arch>: round 1's split and fit (fit.json, fit_data.npz with ML(T-hat))
  F1        results/decoder/willow/round2/f1/<arch>: fit_series.json per series, d_data.npz (ML(D-hat)) per leaf
  MECH      results/decoder/willow/round2/mech/<arch>: mech_fit_T.json, mech_fit_D.json per series; mech_data.npz,
            eval_mech.json per leaf
  INSAMPLE  results/decoder/willow/round2/insample/<arch>: insample_data.npz per leaf ("-" to skip)
  ANALYSIS  results/decoder/willow/round2/explore_analysis.json, compared with at the end only ("-" to skip)
A series enters X's statistics (X = Tm, Dm) when its X fit converged and its 4 leaves' mech outputs exist (PR2 8.2:
series of a fit that did not converge are left out and listed). What is computed:
  (1) held out (PR2 8.2, exploratory 95% CIs): m_X = (K_RL - K_X) / K_X with P4's SE (delta method per shot,
      stratified by leaf), z, CIs; by r, without r = 13, per series (heterogeneity chi2); ML(T-hat) and ML(D-hat)
      against ML(X); the conditional log-loss of RL, T-hat, D-hat, Tm and Dm with paired differences; R and R_conf of
      ML(X) and ML(RL), and the share of RL's excess removed (delta method on the four totals);
  (2) in sample (PR2 8.4, descriptive only): the same on the fit shots from insample_data.npz, for T, D, Tm, Dm; the
      fits' own in-sample gain against the held-out gain: Dm, mean ln P(L|s) at delta-hat (mech_fit_D.json mean_ll)
      minus RL's on the same fit shots (F1's fit_series.json history[0]); Tm, mean ln P(s) at delta-hat minus RL's
      (round 1's ML(RL) ln P(s) of the fit shots, which must also equal the fit's own start, history[0]);
  (3) checks: held-out shots = round 1's test_idx = P4's odd ranks, sorted and unique; fit_idx in the even ranks,
      unique, disjoint from them; observables equal the archive; the RL DEM's md5 equals fit.json's; ML(RL) equals
      round 1's shipped predictions; X_pred is the argmax of X_llr (a NaN posterior counts as 1/2 and predicts 0) and
      NaN posteriors are counted; eval_mech.json's errors and 2x2 cells equal the ones here. Per series: the family
      rebuilt here (r2m_common) equals the fits' parameter names and family record; each fit's fingerprint recomputed
      from the archive's fit shots equals the stored one; the fit records (converged, stopped, iterations <= 25 and the
      recorded max_iter = 25 (PR2 amendment of 2026-10-10 15:14Z; section 8.2 had 15; changed here 2026-10-11 before
      the run), converged exactly when the last convergence check is below 0.02 SE, a fit not converged having used
      all its steps, the last convergence check at the last iteration, the objective and mean ll equal to the last
      history entry, fixed parameters, events, finite estimates and SEs, 20,000 shots). In-sample files: fit_idx and
      observables equal round 1's and the archive; their RL predictions equal round 1's ML(RL) of those shots.
Usage: nice -n 10 python3 -I r2m_check.py ARCHIVE DECODE FIT F1 MECH INSAMPLE ANALYSIS OUT_DIR [--series p/B,...]
  -> OUT_DIR/r2m_leaves.csv, OUT_DIR/r2m_check.json (through qrc.outpaths)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (BASES, CONF, PATCHES, SHOTS, W5, dem_path, md5_file, min_p_from_llr,  # noqa: E402
                       nll_from_llr, ratio_se, read_b8_bit0, rel_of, series_fit, split_rank)
from r2_check import Z90, Z95, mean_se, ratio_se_strata  # noqa: E402
from r2m_common import fingerprint, fit_shots, fit_suffix, leaf_name, mech_family, series_models  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

TAGS_HO = ("RL", "T", "D", "Tm", "Dm")
MAX_ITER = 25  # PR2 amendment 2026-10-10 15:14Z (section 8.2's 15 before it); changed 2026-10-11, before the run
STEP_TOL = 0.02


def ratio_block(num, den, leaf_of, minus_one=True):
    """r2_check.ratio_block (the same estimators), with z and p left empty when the SE is 0 (identical decisions)."""
    Q, se = (ratio_se if minus_one else ratio_se_strata)(num.astype(float), den.astype(float), leaf_of)
    m = Q - 1 if minus_one else Q
    z = (Q - 1) / se if se > 0 else None
    return {"num": int(num.sum()) if num.dtype.kind in "iub" else float(num.sum()),
            "den": int(den.sum()) if den.dtype.kind in "iub" else float(den.sum()),
            "shots": int(num.size), "value": m, "se": se, "z": z,
            "p_two_sided": float(2 * stats.norm.sf(abs(z))) if z is not None else None,
            "ci90": [m - Z90 * se, m + Z90 * se], "ci95": [m - Z95 * se, m + Z95 * se]}


def share_removed(eD, mD, eR, mR, leaf_of, conf=None):
    """r2_check.share_removed (1 - (R_X - 1) / (R_RL - 1), delta method on the four totals, stratified by leaf), with
    a rounding-negative variance (X identical to RL) read as 0."""
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
    se = math.sqrt(max(float(g @ C @ g), 0.0))
    return {"share": S, "se": se, "ci95": [S - Z95 * se, S + Z95 * se], "R_X": RD, "R_RL": RR,
            "errors_X": T[0], "predicted_X": T[1], "errors_RL": T[2], "predicted_RL": T[3], "shots": int(sel.sum())}


def argmax_pred(llr):
    return (np.nan_to_num(np.asarray(llr, np.float64), nan=0.0) > 0).astype(np.int8)


def cells(x, y, o):
    ex, ey = x != o, y != o
    return {"n": int(o.size), "n10": int(np.sum(ex & ~ey)), "n01": int(np.sum(~ex & ey)), "n11": int(np.sum(ex & ey))}


def leaf(arch, dec, fit, f1, mech, ins, patch, basis, r, M):
    rel = rel_of(patch, basis, r)
    lname = leaf_name(patch, basis, r)
    fd = np.load(os.path.join(fit, rel, "rl_tsplit", "fit_data.npz"), allow_pickle=False)
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    dd = np.load(os.path.join(f1, rel, "d_data.npz"), allow_pickle=False)
    md = np.load(os.path.join(mech, rel, "mech_data.npz"), allow_pickle=False)
    em = json.load(open(os.path.join(mech, rel, "eval_mech.json")))
    obs = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)
    rank = split_rank(arch, rel)
    chk = {}
    ti, fi = md["test_idx"].astype(np.int64), fd["fit_idx"].astype(np.int64)
    chk["test_idx_equals_round1_test_idx"] = bool(np.array_equal(ti, fd["test_idx"].astype(np.int64)))
    chk["test_idx_equals_f1_test_idx"] = bool(np.array_equal(ti, dd["test_idx"].astype(np.int64)))
    chk["test_idx_is_odd_rank_rule"] = bool(np.array_equal(np.sort(ti), np.flatnonzero(rank % 2 == 1)))
    chk["test_idx_sorted_unique"] = bool(np.all(np.diff(ti) > 0))
    chk["fit_idx_in_even_ranks"] = bool(np.all(rank[fi] % 2 == 0))
    chk["fit_idx_unique"] = bool(np.unique(fi).size == fi.size)
    chk["fit_test_disjoint"] = bool(np.intersect1d(fi, ti).size == 0)
    chk["n_fit_n_test"] = [int(fi.size), int(ti.size)]
    o = obs[ti].astype(np.int8)
    chk["obs_test_equals_archive"] = bool(np.array_equal(md["obs_test"].astype(np.int8), o))
    chk["dem_md5_equals_fit_json"] = md5_file(dem_path(arch, rel)) == M["fit_json"]["dem_md5"]
    if not np.array_equal(ml["idx"], np.arange(SHOTS)):
        raise SystemExit(f"ML(RL) decode not complete and in order at {rel}")
    rl_pred_all = ml["pred"].reshape(SHOTS, -1)[:, 0].astype(np.int8)
    rl_pred, rl_llr = rl_pred_all[ti], ml["llr"][ti]
    chk["ml_rl_pred_equals_shipped_pred"] = bool(np.array_equal(rl_pred, fd["shipped_pred"].astype(np.int8)))
    pred = {"RL": rl_pred, "T": fd["fitted_pred"].astype(np.int8), "D": dd["D_pred"].astype(np.int8)}
    llr = {"RL": rl_llr, "T": fd["fitted_llr"], "D": dd["D_llr"]}
    for X in ("Tm", "Dm"):
        pred[X], llr[X] = md[f"{X}_pred"].astype(np.int8), md[f"{X}_llr"]
        chk[f"{X}_pred_is_argmax_of_llr"] = bool(np.array_equal(pred[X], argmax_pred(llr[X])))
        chk[f"{X}_llr_nan"] = int(np.sum(np.isnan(llr[X])))
        chk[f"{X}_llr_inf"] = int(np.sum(np.isinf(llr[X])))
        chk[f"{X}_llr_exact_zero"] = int(np.sum(llr[X] == 0))
        chk[f"eval_mech_nan_{X}_equal"] = em.get("nan_posteriors", {}).get(X) == chk[f"{X}_llr_nan"]
    e = {k: (v != o).astype(np.int8) for k, v in pred.items()}
    mine = {"Tm": int(e["Tm"].sum()), "Dm": int(e["Dm"].sum()), "RL": int(e["RL"].sum()), "T": int(e["T"].sum())}
    chk["eval_mech_errors_equal"] = em.get("errors") == mine
    for a_, b_ in (("RL", "Tm"), ("RL", "Dm"), ("T", "Tm"), ("T", "Dm")):
        c = cells(pred[a_], pred[b_], o)
        st = em.get(f"cells_{a_}_vs_{b_}", {})
        chk[f"eval_mech_cells_{a_}_vs_{b_}_equal"] = {k: st.get(k) for k in c} == c
    chk["eval_mech_leaf_name"] = em.get("leaf") == lname
    arr = {f"e_{k}": e[k] for k in TAGS_HO}
    arr.update({f"nll_{k}": nll_from_llr(llr[k], o) for k in TAGS_HO})
    arr.update({f"m_{k}": min_p_from_llr(llr[k]) for k in TAGS_HO})
    # --- the fit shots: fingerprint bytes, ML(RL)'s ln P(s), in-sample predictions
    fi2, det_fit, L_fit = fit_shots(arch, fit, M)
    if not np.array_equal(fi2, fi):
        raise SystemExit(f"{rel}: fit_idx read twice differs")
    lp_rl_fit = ml["logp"][fi].astype(np.float64)
    ins_arr, ins_chk = None, {}
    p_ins = os.path.join(ins, rel, "insample_data.npz") if ins != "-" else None
    if p_ins and os.path.exists(p_ins):
        z = np.load(p_ins, allow_pickle=False)
        of = obs[fi].astype(np.int8)
        ins_chk["insample_fit_idx_equals_round1"] = bool(np.array_equal(z["fit_idx"].astype(np.int64), fi))
        ins_chk["insample_obs_equals_archive"] = bool(np.array_equal(z["obs_fit"].astype(np.int8), of))
        tags = [k for k in TAGS_HO if f"{k}_pred" in z.files]
        ins_chk["insample_tags"] = tags
        if "RL" in tags:
            ins_chk["insample_RL_pred_equals_round1_ml"] = bool(np.array_equal(z["RL_pred"].astype(np.int8),
                                                                               rl_pred_all[fi]))
            ins_chk["insample_RL_llr_max_abs_diff"] = float(np.nanmax(np.abs(z["RL_llr"].astype(np.float64) -
                                                                             ml["llr"][fi].astype(np.float64))))
        for k in tags:
            ins_chk[f"insample_{k}_pred_is_argmax"] = bool(np.array_equal(z[f"{k}_pred"].astype(np.int8),
                                                                          argmax_pred(z[f"{k}_llr"])))
        ins_arr = {"tags": tags}
        for k in tags:
            p_, l_ = z[f"{k}_pred"].astype(np.int8), z[f"{k}_llr"]
            ins_arr[f"e_{k}"] = (p_ != of).astype(np.int8)
            ins_arr[f"nll_{k}"] = nll_from_llr(l_, of)
            ins_arr[f"m_{k}"] = min_p_from_llr(l_)
    row = {"patch": patch, "basis": basis, "r": r, "n_test": int(o.size), "n_fit": int(fi.size),
           **{f"K_{k}": int(e[k].sum()) for k in TAGS_HO},
           **{f"cells_RL_vs_{X}": json.dumps(cells(pred["RL"], pred[X], o)) for X in ("Tm", "Dm")},
           **{f"nll_{k}": float(arr[f"nll_{k}"].mean()) for k in TAGS_HO},
           **{f"R_{k}": float(e[k].sum() / arr[f"m_{k}"].sum()) for k in ("RL", "Tm", "Dm")},
           "lnPs_RL_fit_mean": float(lp_rl_fit.mean())}
    row.update({f"chk_{k}": v for k, v in chk.items()})
    row.update({f"chk_{k}": v for k, v in ins_chk.items()})
    return row, arr, ins_arr, {**chk, **ins_chk}, (lname, det_fit, L_fit), lp_rl_fit


def fit_record(fj, fam, fp_base, n_expected):
    lc = fj.get("last_convergence_check") or {}
    names = fj["names"]
    fixed = set(fj.get("fixed", []))
    free = np.array([nm not in fixed for nm in names])
    se = np.array([np.nan if v is None else v for v in fj["se"]], float)
    dl = np.array(fj["delta_hat"], float)
    hist = fj.get("history", [])
    return {"names_equal": names == fam["names"], "family_equal": fj.get("family") == fam["info"],
            "fingerprint_recomputed_equals": fp_base + fit_suffix(fj) == fj.get("fingerprint"),
            "objective_type": fj.get("objective_type"), "lambda": fj.get("lambda"), "information": fj.get("information"),
            "secant": fj.get("secant"), "start": fj.get("start"),
            "converged": bool(fj.get("converged")), "stopped": bool(fj.get("stopped")), "iterations": fj.get("iterations"),
            "iterations_within_cap": (fj.get("iterations") or 0) <= MAX_ITER, "n_eval": fj.get("n_eval"),
            "max_iter_recorded": fj.get("max_iter"), "max_iter_equals_cap": fj.get("max_iter") == MAX_ITER,
            "converged_matches_stop_rule": bool(fj.get("converged")) == (lc.get("max_step_in_se") is not None and
                                                                         lc["max_step_in_se"] < STEP_TOL),
            "not_converged_used_all_steps": (bool(fj.get("converged")) or bool(fj.get("stopped")) or
                                             fj.get("iterations") == fj.get("max_iter")),
            "fixed": sorted(fixed), "events": fj.get("events", []), "parameters": len(names),
            "last_step_max_in_se": lc.get("max_step_in_se"), "last_step_argmax": lc.get("argmax"),
            "last_step_below_tol": (lc.get("max_step_in_se") is not None and lc["max_step_in_se"] < STEP_TOL),
            "last_check_iteration_equals_iterations": lc.get("iteration") == fj.get("iterations"),
            "objective_equals_last_history": bool(hist) and hist[-1].get("objective") == fj.get("objective"),
            "mean_ll_equals_last_history": bool(hist) and hist[-1].get("mean_ll") == fj.get("mean_ll"),
            "delta_finite": bool(np.all(np.isfinite(dl))), "se_finite_where_free": bool(np.all(np.isfinite(se[free]))),
            "n_shots_ok": fj.get("n_shots") == n_expected, "max_abs_delta": float(np.max(np.abs(dl))) if dl.size else None,
            "delta_quantiles": np.quantile(dl, [0, .1, .5, .9, 1]).tolist() if dl.size else None,
            "mean_ll": fj.get("mean_ll"), "mean_ll_start": hist[0].get("mean_ll") if hist else None,
            "cpu_seconds_fit": fj.get("cpu_seconds_fit")}


def stat_block(A, leaf_of, r_of, s_of, s_names, X, tags):
    """m_X and its relatives, log-loss, calibration (A: per-shot arrays of the selected leaves)."""
    def mX(sel, ref="RL"):
        x = ratio_block(A[f"e_{ref}"][sel], A[f"e_{X}"][sel], leaf_of[sel])
        x.update({f"K_{ref}": x.pop("num"), f"K_{X}": x.pop("den"), "leaves": int(np.unique(leaf_of[sel]).size),
                  f"{ref}_wrong_{X}_right": int(np.sum((A[f"e_{ref}"][sel] == 1) & (A[f"e_{X}"][sel] == 0))),
                  f"{ref}_right_{X}_wrong": int(np.sum((A[f"e_{ref}"][sel] == 0) & (A[f"e_{X}"][sel] == 1)))})
        return x
    allm = np.ones(leaf_of.size, bool)
    out = {"shots": int(leaf_of.size), "series": len(s_names), "m_X": mX(allm),
           "m_X_by_r": {str(r): mX(r_of == r) for r in W5 if np.any(r_of == r)},
           "m_X_without_r13": mX(r_of != 13)}
    for ref in ("T", "D"):
        if ref in tags and ref != X:
            out[f"ratio_{ref}_vs_X"] = mX(allm, ref)
    per = {s: mX(s_of == i) for i, s in enumerate(s_names) if np.any(s_of == i)}
    if len(per) > 1:
        w = np.array([1 / v["se"] ** 2 for v in per.values()])
        v_ = np.array([v["value"] for v in per.values()])
        mbar = float((w * v_).sum() / w.sum())
        Q = float((w * (v_ - mbar) ** 2).sum())
        out["m_X_series_heterogeneity"] = {"inverse_variance_mean": mbar, "Q": Q, "df": len(per) - 1,
                                           "p": float(stats.chi2.sf(Q, len(per) - 1)),
                                           "series_above_0": int(np.sum(v_ > 0))}
    out["m_X_per_series"] = {s: {k: v[k] for k in ("value", "se", "K_RL", f"K_{X}")} for s, v in per.items()}
    ll = {"mean_nll": {k: float(A[f"nll_{k}"].mean()) for k in tags}}
    for k in tags:
        if k != X:
            ll[f"diff_{k}-{X}"] = mean_se(A[f"nll_{k}"] - A[f"nll_{X}"], leaf_of)
    out["logloss"] = ll
    cal = {}
    for k in ("RL", X):
        e, m = A[f"e_{k}"], A[f"m_{k}"]
        Rx = ratio_block(e, m, leaf_of, minus_one=False)
        cf = m < CONF
        cal[k] = {"errors": int(e.sum()), "predicted": float(m.sum()), "R": Rx["value"], "R_se": Rx["se"],
                  "conf_shots": int(cf.sum()), "conf_errors": int(e[cf].sum()), "conf_predicted": float(m[cf].sum()),
                  "R_conf": float(e[cf].sum() / m[cf].sum()) if m[cf].sum() > 0 else None,
                  "R_conf_se_sqrt_k": math.sqrt(e[cf].sum()) / float(m[cf].sum()) if m[cf].sum() > 0 else None}
    out["calibration"] = cal
    cfX, cfR = (A[f"m_{X}"] < CONF).astype(float), (A["m_RL"] < CONF).astype(float)
    out["share_of_RL_excess_removed"] = {
        "R": share_removed(A[f"e_{X}"], A[f"m_{X}"], A["e_RL"], A["m_RL"], leaf_of),
        "R_conf": share_removed(A[f"e_{X}"] * cfX, A[f"m_{X}"] * cfX, A["e_RL"] * cfR, A["m_RL"] * cfR, leaf_of),
        "R_without_r13": share_removed(A[f"e_{X}"], A[f"m_{X}"], A["e_RL"], A["m_RL"], leaf_of, conf=(r_of != 13))}
    return out


def compare(dec_block, mine, X):
    """Our block against decoder's analyze_explore block (keys as in explore_analysis.json)."""
    if not dec_block:
        return None
    c = {}
    dm, mm = dec_block.get("m_X") or {}, mine["m_X"]
    c["m_X"] = {"decoder": [dm.get("rel"), dm.get("se_rel"), dm.get("errors_A"), dm.get("errors_B")],
                "here": [mm["value"], mm["se"], mm["K_RL"], mm[f"K_{X}"]]}
    d13 = dec_block.get("m_X_without_r13") or {}
    c["m_X_without_r13"] = {"decoder": [d13.get("rel"), d13.get("se_rel")],
                            "here": [mine["m_X_without_r13"]["value"], mine["m_X_without_r13"]["se"]]}
    if dec_block.get("ratio_T_vs_X") and "ratio_T_vs_X" in mine:
        c["ratio_T_vs_X"] = {"decoder": [dec_block["ratio_T_vs_X"].get("rel"), dec_block["ratio_T_vs_X"].get("se_rel")],
                             "here": [mine["ratio_T_vs_X"]["value"], mine["ratio_T_vs_X"]["se"]]}
    dl = dec_block.get("logloss") or {}
    c["logloss_mean_nll"] = {"decoder": dl.get("mean_nll"), "here": mine["logloss"]["mean_nll"]}
    for k, v in mine["logloss"].items():
        if k.startswith("diff_") and k in dl:
            c[f"logloss_{k}"] = {"decoder": [dl[k]["mean"], dl[k]["se"]], "here": [v["mean"], v["se"]]}
    dc = (dec_block.get("calibration") or {}).get("ML(X)") or {}
    c["R_X"] = {"decoder": [dc.get("ratio"), dc.get("se")], "here": [mine["calibration"][X]["R"],
                                                                    mine["calibration"][X]["R_se"]]}
    dcc = dc.get("conf") or {}
    c["R_conf_X"] = {"decoder": [dcc.get("errors"), dcc.get("predicted")],
                     "here": [mine["calibration"][X]["conf_errors"], mine["calibration"][X]["conf_predicted"]]}
    ds = (dec_block.get("calibration") or {}).get("share_of_RL_excess_removed") or {}
    for k in ("R", "R_conf"):
        if k in ds:
            c[f"share_{k}"] = {"decoder": [ds[k].get("share"), ds[k].get("se")],
                               "here": [mine["share_of_RL_excess_removed"][k]["share"],
                                        mine["share_of_RL_excess_removed"][k]["se"]]}
    if "R_RL_same_shots" in ds:
        c["R_RL_same_shots"] = {"decoder": ds["R_RL_same_shots"], "here": mine["calibration"]["RL"]["R"]}
    c["decoder_problems"] = dec_block.get("problems")
    c["decoder_series"] = dec_block.get("series")
    c["m_X_abs_diff"] = (abs(dm["rel"] - mm["value"]) if dm.get("rel") is not None else None)
    return c


def summarise_checks(chks):
    agg = {}
    for c in chks:
        for k, v in c.items():
            agg.setdefault(k, []).append(v)
    out = {}
    for k, vals in agg.items():
        if all(isinstance(x, bool) for x in vals):
            out[k] = f"{sum(vals)} of {len(vals)}"
        elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals if x is not None):
            vv = [x for x in vals if x is not None]
            out[k] = {"max": max(vv), "sum": sum(vv)} if vv else None
        else:
            out[k] = sorted({json.dumps(x) for x in vals})
    return out


def main(a):
    t0 = time.time()
    want = ([tuple(s.split("/")) for s in a.series.split(",")] if a.series
            else [(p, b) for p in PATCHES for b in BASES])
    res = {"label": "reservoir, independent check of round 2's mech fits (T-hat_mech, D-hat_mech), PR2 sections 8.2 "
                    "and 8.4; held-out statistics exploratory (95% CIs), in-sample statistics descriptive",
           "inputs": {k: getattr(a, k) for k in ("archive", "decode", "fit", "f1", "mech", "insample", "analysis")},
           "series_requested": [f"d3_at_{p}/{b}" for p, b in want], "missing": {}, "series": {}}
    rows, chks, arrs, iarrs, meta = [], [], [], [], []
    for p, b in want:
        key = f"d3_at_{p}/{b}"
        sd = os.path.join(a.mech, f"d3_at_{p}", b)
        fits = {X: json.load(open(os.path.join(sd, fn))) for X, fn in (("Tm", "mech_fit_T.json"), ("Dm", "mech_fit_D.json"))
                if os.path.exists(os.path.join(sd, fn))}
        miss = [r for r in W5 if not (os.path.exists(os.path.join(a.mech, rel_of(p, b, r), "mech_data.npz")) and
                                      os.path.exists(os.path.join(a.mech, rel_of(p, b, r), "eval_mech.json")))]
        if len(fits) < 2 or miss:
            res["missing"][key] = {"fits": sorted(fits), "leaves_without_outputs": miss}
            continue
        models = series_models(a.archive, a.fit, p, b)
        fam = mech_family(models)
        fs1 = series_fit(a.f1, p, b)
        lnames, dets, Ls, lp_fit = [], [], [], []
        for r, M in zip(W5, models):
            row, arr, iarr, chk, (ln, det, L), lpf = leaf(a.archive, a.decode, a.fit, a.f1, a.mech, a.insample,
                                                           p, b, r, M)
            rows.append(row); chks.append(chk); arrs.append(arr); iarrs.append(iarr)
            meta.append({"series": key, "r": r, "n": row["n_test"], "n_fit": row["n_fit"]})
            lnames.append(ln); dets.append(det); Ls.append(L); lp_fit.append(lpf)
            print(f"{key} r{r}: K_RL {row['K_RL']} K_Tm {row['K_Tm']} K_Dm {row['K_Dm']} ({time.time() - t0:.0f} s)",
                  flush=True)
        fp_base = fingerprint(lnames, dets, Ls, fam["names"])
        n_fit_series = int(sum(L.size for L in Ls))
        rep = {"family": fam["info"], "Tm": fit_record(fits["Tm"], fam, fp_base, n_fit_series),
               "Dm": fit_record(fits["Dm"], fam, fp_base, n_fit_series)}
        lp_rl = float(np.concatenate(lp_fit).mean())
        rep["Tm"]["in_sample_lnPs_RL"] = lp_rl
        rep["Tm"]["start_equals_RL_lnPs_abs_diff"] = (abs(rep["Tm"]["mean_ll_start"] - lp_rl)
                                                     if rep["Tm"]["mean_ll_start"] is not None else None)
        rep["Tm"]["in_sample_gain_lnPs_per_shot"] = fits["Tm"]["mean_ll"] - lp_rl
        rep["Dm"]["in_sample_ll_RL_f1"] = fs1["history"][0]["ll"]
        rep["Dm"]["in_sample_gain_per_shot"] = fits["Dm"]["mean_ll"] - fs1["history"][0]["ll"]
        rep["Dm"]["in_sample_gain_over_f1_D"] = fits["Dm"]["mean_ll"] - fs1["ll_per_shot"]
        res["series"][key] = rep
    if not rows:
        res["note"] = "no series with both mech fits and all held-out outputs; nothing computed"
    else:
        leaf_of = np.concatenate([np.full(m_["n"], i, np.int32) for i, m_ in enumerate(meta)])
        r_of = np.concatenate([np.full(m_["n"], m_["r"], np.int16) for m_ in meta])
        s_names = sorted({m_["series"] for m_ in meta})
        s_of = np.concatenate([np.full(m_["n"], s_names.index(m_["series"]), np.int16) for m_ in meta])
        A = {k: np.concatenate([x[k] for x in arrs]) for k in arrs[0]}
        res["heldout"], res["insample"], res["excluded"] = {}, {}, {}
        for X in ("Tm", "Dm"):
            ok = [s for s in s_names if res["series"][s][X]["converged"]]
            res["excluded"][X] = [s for s in s_names if s not in ok]
            sel = np.isin(s_of, [s_names.index(s) for s in ok])
            if sel.any():
                sub = {k: v[sel] for k, v in A.items()}
                # series indices into the list of series kept (fixed 2026-10-11 20:16Z: s_of indexes all series, so
                # with a series left out the per-series block and its heterogeneity paired data and names wrongly)
                keep = np.full(len(s_names), -1, np.int16)
                keep[[s_names.index(s) for s in ok]] = np.arange(len(ok))
                res["heldout"][X] = stat_block(sub, leaf_of[sel], r_of[sel], keep[s_of[sel]], ok, X, TAGS_HO)
                # the fits' in-sample gain against their held-out gain, per series (Dm: ln P(L|s); Tm: ln P(s))
                for s in ok:
                    ss = s_of == s_names.index(s)
                    if X == "Dm":
                        ho = mean_se(A["nll_RL"][ss] - A["nll_Dm"][ss], leaf_of[ss])
                        rep = res["series"][s]["Dm"]
                        rep["heldout_gain_per_shot"], rep["heldout_gain_se"] = ho["mean"], ho["se"]
                        rep["in_sample_minus_heldout"] = rep["in_sample_gain_per_shot"] - ho["mean"]
        g = [res["series"][s]["Dm"] for s in s_names if "heldout_gain_per_shot" in res["series"][s]["Dm"]]
        if g:
            res["Dm_in_sample_vs_heldout_gain"] = {
                "series": len(g), "mean_in_sample_gain": float(np.mean([x["in_sample_gain_per_shot"] for x in g])),
                "mean_heldout_gain": float(np.mean([x["heldout_gain_per_shot"] for x in g])),
                "mean_difference": float(np.mean([x["in_sample_minus_heldout"] for x in g])),
                "sd_difference": float(np.std([x["in_sample_minus_heldout"] for x in g], ddof=1)) if len(g) > 1 else None,
                "free_parameters_over_n_mean": float(np.mean([(x["parameters"] - len(x["fixed"])) / 20000 for x in g])),
                "note": "in sample: mech_fit_D.json mean_ll minus F1's RL value on the same 20,000 fit shots; held out: "
                        "mean nll_RL - nll_Dm on the series' 100,000 held-out shots. With the ridge the expected "
                        "optimism is below k/n (k free parameters; assumed reading)"}
        # in sample (PR2 8.4), from decoder's insample_data.npz where present
        if any(x is not None for x in iarrs):
            for X in ("T", "D", "Tm", "Dm"):
                ok_s = [s for s in s_names if X in ("T", "D") or res["series"][s][X]["converged"]]
                idx = [i for i, m_ in enumerate(meta) if m_["series"] in ok_s and iarrs[i] is not None and
                       X in iarrs[i]["tags"] and "RL" in iarrs[i]["tags"]]
                if not idx:
                    continue
                tags = [t for t in TAGS_HO if all(t in iarrs[i]["tags"] for i in idx)]
                IA = {f"{q}_{t}": np.concatenate([iarrs[i][f"{q}_{t}"] for i in idx]) for t in tags
                      for q in ("e", "nll", "m")}
                lo = np.concatenate([np.full(meta[i]["n_fit"], i, np.int32) for i in idx])
                ro = np.concatenate([np.full(meta[i]["n_fit"], meta[i]["r"], np.int16) for i in idx])
                sn = sorted({meta[i]["series"] for i in idx})
                so = np.concatenate([np.full(meta[i]["n_fit"], sn.index(meta[i]["series"]), np.int16) for i in idx])
                res["insample"][X] = stat_block(IA, lo, ro, so, sn, X, tags)
                res["insample"][X]["leaves"] = len(idx)
        res["checks"] = summarise_checks(chks)
        for X in ("Tm", "Dm"):
            reps = [res["series"][s][X] for s in s_names]
            for k in ("names_equal", "family_equal", "fingerprint_recomputed_equals", "converged", "stopped",
                      "iterations_within_cap", "max_iter_equals_cap", "converged_matches_stop_rule",
                      "not_converged_used_all_steps", "last_step_below_tol", "last_check_iteration_equals_iterations",
                      "objective_equals_last_history", "mean_ll_equals_last_history", "delta_finite",
                      "se_finite_where_free", "n_shots_ok"):
                res["checks"][f"{X}_series_{k}"] = f"{sum(bool(x[k]) for x in reps)} of {len(reps)}"
            res["checks"][f"{X}_fixed_parameters"] = sum(len(x["fixed"]) for x in reps)
            res["checks"][f"{X}_max_iterations"] = max(x["iterations"] or 0 for x in reps)
        res["checks"]["Tm_start_equals_RL_lnPs_max_abs_diff"] = max(
            (res["series"][s]["Tm"]["start_equals_RL_lnPs_abs_diff"] or 0.0) for s in s_names)
        # comparison with decoder's analysis, read only now
        if a.analysis != "-" and os.path.exists(a.analysis):
            fa = json.load(open(a.analysis))
            res["comparison_with_explore_analysis"] = {
                "made": fa.get("made"),
                "heldout": {X: compare((fa.get("heldout") or {}).get(X), res["heldout"][X], X) for X in res["heldout"]},
                "insample": {X: compare((fa.get("insample") or {}).get(X), res["insample"][X], X)
                             for X in res["insample"]},
                "fits": {X: {s: {k: v.get(k) for k in ("converged", "iterations", "n_eval", "fixed", "parameters")}
                             for s, v in ((fa.get("fits") or {}).get(X) or {}).get("per_series", {}).items()}
                         for X in ("Tm", "Dm")}}
    res["run_seconds"] = time.time() - t0
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))  # in-sample columns may exist for some leaves only
        with open(out_path(os.path.join(a.out_dir, "r2m_leaves.csv")), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval="")
            wr.writeheader()
            for row in rows:
                wr.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    with open(out_path(os.path.join(a.out_dir, "r2m_check.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    show = {k: res[k] for k in ("missing", "excluded", "checks") if k in res}
    for X, blk in (res.get("heldout") or {}).items():
        show[f"heldout_{X}"] = {k: blk[k] for k in ("m_X", "m_X_without_r13", "logloss")}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("archive", "decode", "fit", "f1", "mech", "insample", "analysis", "out_dir"):
        ap.add_argument(k)
    ap.add_argument("--series", default=None, help="patch/basis[,patch/basis...]; default all 18")
    main(ap.parse_args())
