"""Derived numbers for the Willow manuscript v2.0, draft 3 (scout, 2026-10-13). POST HOC; read only. Nothing is decoded,
and nothing is fitted beyond the final analysis's own estimators applied to subsets of its per-shot outputs.

Why: the fresh-eyes reading of draft 2 (notes/orchestrator/fresh_eyes_v2.0_draft2.md, md5 41857d65...) asked for
  B1  Fig. 2 on matched inputs: ML(RL) and Libra on the same series and round counts;
  B2  Table 3 with the paired absolute excess per cycle at both distances;
  B8  Table 4 with the growth as Delta = ln(1 + g5) - ln(1 + g3) in every row, rule B's growth, and the interim
      leave-one-out rows recomputed on K2's final tally;
  B9  Tables 2 and 5 with every interval labelled, and intervals for R and the log-loss gains;
  C12 (optional) every decoder on Libra's round counts {10, 30, 50}.
The coordinator accepted these on 2026-10-13 (after 18:10Z), with every number to be checked against its source.

Sections of the output JSON:
  checks              the leaf tables and fits reproduce final_round1.json (g3; g5 under rule A, rule B's series and no
                      exclusion), p1_se_covariance.json (P1's g, its SE and the SE of the mean paired difference),
                      analysis.json (every decoder's mean paired difference) and r3_patch.json (the matched Lambdas).
  leave_one_out_final Libra's g5 and Delta with one d = 5 leaf left out, or one round count left out at both
                      distances; rule A and no exclusion; K2's final tally.
  growth_rows         Delta +- SE for the headline, the frozen SE formula, rule B and no exclusion.
  table3_abs_excess   every decoder's mean paired difference eps_A - eps_ML over series, with the corrected SE (entry 7's
                      delta method, slope covariance included, Birge factors), on its W5 windows.
  matched_fig2        ML(RL)'s and Libra's own (unpaired) fits on the same series and round counts {10, 30, 50}, d = 5
                      under rule A and with no exclusion: mean eps over series, SE = sqrt(sum se^2)/n (the convention of
                      the final analysis's Lambda), and Lambda_3/5 = eps3/eps5 with its delta-method SE.
  census_growth       Delta for the census's rows (d2_worstcase.json) and per-round boxes (r5_worstcase.json), g3 fixed.
  b9                  R, R_conf and the log-loss gains with their intervals, read from f1_analysis.json,
                      explore_analysis.json, r2_temper.json and p3_heldout_fitted.json; R_conf's SE is R_conf/sqrt(k)
                      for its k errors, the sources' se_sqrtk, computed here where a source does not store it.
  c12                 every decoder against ML(RL) on Libra's round counts {10, 30, 50} at both distances (Tesseract
                      ran at r <= 30 at d = 5, so it is left out).

The estimator functions (wls, series_fit, pooled, growth and leaf_any) are copied from code/scout/willow_growth_re.py v2
(3dde48e31dd131712ec3272827ac6f61) unchanged, except that series_fit gains the option to omit the slope covariance (the
frozen formula), pooled also returns the SE of the mean paired difference, and own_fit (an unpaired fit of one
decoder, as reservoir's d5_check.eps_fit) is new.
Usage: OMP_NUM_THREADS=1 nice -n 10 python3 -I willow_v2d3_derived.py <experimental-root>/ <out.json>
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
W = root + "results/decoder/willow/"
INPUTS = {"FR": W + "final_round1/final_round1.json", "FA": W + "final_round1/analysis/analysis.json",
          "K2T": W + "supplementary/k2_extra_rung/k2_tally.json", "SEC": W + "supplementary/p1_se_covariance.json",
          "R3": root + "results/reservoir/willow_revision/r3_patch.json",
          "D2": W + "revision/d2_worstcase/d2_worstcase.json",
          "R5": root + "results/reservoir/willow_revision/r5_worstcase.json",
          "F1": W + "round2/f1_analysis.json", "EX": W + "round2/explore_analysis.json",
          "RT": root + "results/reservoir/willow_r2check/r2_temper.json",
          "P3F": W + "supplementary/p3_heldout_fitted.json",
          "GRF": root + "results/scout/willow_growth_re_final.json"}
J = {k: json.load(open(v)) for k, v in INPUTS.items()}
MD5 = {k: hashlib.md5(open(v, "rb").read()).hexdigest() for k, v in INPUTS.items()}
DEC = W + "decode/google_105Q_surface_code_d3_d5_d7/"
P3 = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
LADDER = "ml_tau2_1e-6-1e-8_0.1.npz"
Z95 = stats.norm.ppf(0.975)
R_LIBRA = [10, 30, 50]
R_P1 = [10] + list(range(30, 251, 20))  # Libra's 13 round counts r >= 10
WINDOWS = {"libra": ([10, 30, 50], [10, 30, 50]), "tesseract": ([10, 13, 30, 50], [10, 13, 30]),
           "harmony": ([10, 13, 30, 50], [10, 13, 30, 50]), "corrmatch": ([10, 13, 30, 50], [10, 13, 30, 50]),
           "corr_mwpm": ([10, 13, 30, 50], [10, 13, 30, 50]), "mwpm": ([10, 13, 30, 50], [10, 13, 30, 50]),
           "beliefmatch": ([10, 13, 30, 50], None)}
SHIP = {"libra": "libra_decoder_with_rl_optimized_prior", "harmony": "harmony_decoder_with_rl_optimized_prior",
        "corrmatch": "correlated_matching_decoder_with_rl_optimized_prior"}
OURS = {"tesseract": ("rl_s5", "rl_s5t", "tesseract.npz"), "corr_mwpm": ("rl", "rl_s5", "corr_mwpm.npz"),
        "mwpm": ("rl", "rl_s5", "mwpm.npz"), "beliefmatch": ("rl_s5", None, "beliefmatch.npz")}
MISSING = 255
failing = {s.split("google_105Q_surface_code_d3_d5_d7/")[1] for s in J["K2T"]["still_failing"]}
assert J["K2T"].get("complete") is True and len(failing) == 7, failing


# ---- estimators (from willow_growth_re.py v2) ----
def wls(x, y, var):
    x, y, w = np.asarray(x, float), np.asarray(y, float), 1.0 / np.asarray(var, float)
    X = np.column_stack([np.ones_like(x), x])
    M = np.linalg.inv(X.T @ (w[:, None] * X))
    coef = M @ (X.T @ (w * y))
    a = (M @ (X * w[:, None]).T)[1]  # slope = a . y
    dof = len(x) - 2
    chi2 = float(np.sum(w * (y - X @ coef) ** 2))
    birge = math.sqrt(max(1.0, chi2 / dof)) if dof > 0 else 1.0
    return coef[1], a, math.sqrt(M[1, 1]), birge, (chi2 / dof if dof > 0 else None)


def series_fit(L, use_cov=True):
    """Paired fit of decoder A against ML(RL) over the leaves L of one series (PP section 4; entry 7's SE)."""
    r = np.array([x["r"] for x in L], float)
    n = np.array([x["n"] for x in L], float)
    EA, EM = np.array([x["k_a"] for x in L]) / n, np.array([x["k_ml"] for x in L]) / n
    p11 = np.array([x["n11"] for x in L]) / n
    yA, yM = np.log(1 - 2 * EA), np.log(1 - 2 * EM)
    gA, gM = -2 / (1 - 2 * EA), -2 / (1 - 2 * EM)
    VA, VM, C = EA * (1 - EA) / n, EM * (1 - EM) / n, (p11 - EA * EM) / n
    var_yM = gM ** 2 * VM
    var_z = gA ** 2 * VA + gM ** 2 * VM - 2 * gA * gM * C
    cov_z_yM = gA * gM * C - gM ** 2 * VM
    bM, c, se_bM, BM, chiM = wls(r, yM, var_yM)
    beta, a, se_beta, Bz, chiz = wls(r, yA - yM, var_z)
    V_bM, V_beta = (se_bM * BM) ** 2, (se_beta * Bz) ** 2
    C_bb = float(np.sum(a * c * cov_z_yM)) * BM * Bz if use_cov else 0.0
    eM, eA = (1 - math.exp(bM)) / 2, (1 - math.exp(bM + beta)) / 2
    dlt = eA - eM
    Jd = np.array([(math.exp(bM) - math.exp(bM + beta)) / 2, -math.exp(bM + beta) / 2])  # d delta / d(bM, beta)
    Je = np.array([-math.exp(bM) / 2, 0.0])
    S = np.array([[V_bM, C_bb], [C_bb, V_beta]])
    return {"eps_ml": eM, "eps_a": eA, "delta": dlt, "var_delta": float(Jd @ S @ Jd),
            "var_eps_ml": float(Je @ S @ Je), "cov_delta_eps": float(Jd @ S @ Je),
            "rel": dlt / eM, "rounds": [int(v) for v in r]}


def own_fit(L, key):
    """Unpaired fit of one decoder's error per cycle over the leaves L of one series (reservoir's d5_check.eps_fit)."""
    r = np.array([x["r"] for x in L], float)
    n = np.array([x["n"] for x in L], float)
    E = np.array([x[key] for x in L]) / n
    y, v = np.log(1 - 2 * E), (2 / (1 - 2 * E)) ** 2 * E * (1 - E) / n
    b, _, se_b, B, _ = wls(r, y, v)
    return {"eps": (1 - math.exp(b)) / 2, "se": math.exp(b) / 2 * se_b * B}


def pooled(fits):
    """g = mean(delta)/mean(eps_ML) with entry 7's delta-method SE over independent series."""
    S = len(fits)
    D = np.mean([f["delta"] for f in fits])
    E = np.mean([f["eps_ml"] for f in fits])
    g = D / E
    vD = sum(f["var_delta"] for f in fits) / S ** 2
    vE = sum(f["var_eps_ml"] for f in fits) / S ** 2
    cDE = sum(f["cov_delta_eps"] for f in fits) / S ** 2
    se = math.sqrt(vD - 2 * g * cDE + g * g * vE) / E
    return {"g": float(g), "se": se, "series": S, "mean_eps_ml": float(E), "mean_delta": float(D),
            "se_mean_delta": math.sqrt(vD), "ci95": [float(g - Z95 * se), float(g + Z95 * se)]}


def growth(g3, s3, g5, s5):
    a, b = s3 / (1 + g3), s5 / (1 + g5)
    D = math.log(1 + g5) - math.log(1 + g3)
    se = math.sqrt(a * a + b * b)
    return {"Delta": D, "se": se, "z": D / se, "p_two_sided_normal": float(2 * stats.norm.sf(abs(D / se))),
            "lambda_ratio": math.exp(-D), "lambda_ratio_ci95": [math.exp(-D - Z95 * se), math.exp(-D + Z95 * se)]}


def leaf_any(d, patch, basis, r, a):
    """2x2 table of decoder a against ML(RL) on ML's shots of one leaf where a has a prediction."""
    sub = "rl" if d == 3 else "rl_s5"
    base = "%sd%d_at_%s/%s/r%02d/" % (DEC, d, patch, basis, r)
    sh = np.load(base + sub + "/shots.npz", allow_pickle=False)
    ml = np.load(base + sub + "/" + ("ml_exact.npz" if d == 3 else LADDER), allow_pickle=False)
    idx = ml["idx"].astype(np.int64)
    assert np.array_equal(sh["idx"].astype(np.int64), idx)
    obs = sh["obs"].astype(np.uint8)
    pm = ml["pred"][:, 0].astype(np.uint8)
    assert np.array_equal(pm, (ml["llr"] > 0).astype(np.uint8))
    if a in SHIP:
        names = list(sh["shipped_names"])
        li = [i for i, s in enumerate(names) if "/" + SHIP[a] + "/" in s]
        assert len(li) == 1, (a, names)
        pa = sh["shipped"][li[0]].astype(np.uint8)
    else:
        s3, s5, fn = OURS[a]
        z = np.load(base + (s3 if d == 3 else s5) + "/" + fn, allow_pickle=False)
        ii = z["idx"].astype(np.int64)
        full = np.full(max(int(idx.max()), int(ii.max())) + 1, MISSING, np.uint8)
        full[ii] = z["pred"].reshape(ii.size, -1)[:, 0].astype(np.uint8)
        pa = full[idx]
    m = pa != MISSING
    aw, mw = (pa != obs) & m, (pm != obs) & m
    return {"d": d, "patch": patch, "basis": basis, "r": r, "n": int(m.sum()), "k_ml": int(mw.sum()),
            "k_a": int(aw.sum()), "n11": int(np.sum(aw & mw))}


def name(d, p, b, r):
    return "d%d_at_%s/%s/r%02d" % (d, p, b, r)


def series_set(T, d, rounds, excluded, use_cov=True, series_only=None):
    fits, lost = [], []
    for p in (P3 if d == 3 else P5):
        for b in "XZ":
            if series_only is not None and p + "|" + b not in series_only:
                continue
            L = [T[(d, p, b, r)] for r in rounds if name(d, p, b, r) not in excluded]
            L = [x for x in L if x["n"] > 0]
            if len(L) < 2:
                lost.append("%s|%s (%d round%s)" % (p, b, len(L), "" if len(L) == 1 else "s"))
                continue
            f = series_fit(L, use_cov)
            f.update(patch=p, basis=b, leaves=L)
            fits.append(f)
    return fits, lost


def table(a, d, rounds):
    return {(d, p, b, r): leaf_any(d, p, b, r, a) for p in (P3 if d == 3 else P5) for b in "XZ" for r in rounds}


def rel(x, y):
    return abs(x / y - 1) if y else abs(x)


res = {"label": "scout, POST HOC, read only: derived numbers for manuscript v2.0 draft 3 (fresh-eyes points B1, B2, B8, "
                "B9, C12); estimators of the final round-1 analysis on subsets of its per-shot outputs; K2's final "
                "tally (rule A: 7 leaves still fail)",
       "inputs": {k: {"path": v.replace(root, ""), "md5": MD5[k]} for k, v in INPUTS.items()},
       "k2_still_failing": sorted(failing)}
FR, FA = J["FR"], J["FA"]
E7 = FR["entry7_se"]
checks = {}

# ---- Libra's leaf tables on {10, 30, 50} and on P1's 13 round counts ----
TL = {}
for d in (3, 5):
    TL.update(table("libra", d, R_LIBRA))
TP1 = table("libra", 3, R_P1)

# P1: g, its corrected SE and the SE of the mean paired difference, against p1_se_covariance.json
fP1, _ = series_set(TP1, 3, R_P1, set())
pP1 = pooled(fP1)
sec = J["SEC"]["full_delta_method_inflated"]
checks["P1_vs_p1_se_covariance"] = {"g": rel(pP1["g"], sec["g"]), "se": rel(pP1["se"], sec["se"]),
                                    "se_mean_delta": rel(pP1["se_mean_delta"], sec["se_mean_delta"]),
                                    "mean_delta": rel(pP1["mean_delta"], J["SEC"]["frozen_pool_gap"]["mean_delta"])}
res["P1_abs_excess"] = {"mean_delta": pP1["mean_delta"], "se_mean_delta": pP1["se_mean_delta"],
                        "mean_eps_ml": pP1["mean_eps_ml"], "series": pP1["series"], "rounds": R_P1}

# Libra on {10, 30, 50}: g3; g5 under rule A and with no exclusion; rule B's series
f3, _ = series_set(TL, 3, R_LIBRA, set())
p3 = pooled(f3)
checks["g3_vs_final_round1"] = {"g": rel(p3["g"], E7["pooled_gaps_W5:d3|libra.rl vs ML.rl"]["g"]),
                                "se": rel(p3["se"], E7["pooled_gaps_W5:d3|libra.rl vs ML.rl"]["se"])}
DR = FR["decisions"]["d5_rules"]
fits5, p5 = {}, {}
for rule, excl, only in (("A", failing, None), ("none", set(), None),
                         ("B", set(), {s for s in DR["B"]["g5"]["series"]})):
    fits5[rule], lost = series_set(TL, 5, R_LIBRA, excl, series_only=only)
    p5[rule] = pooled(fits5[rule])
    checks["g5_%s_vs_final_round1" % rule] = {"g": rel(p5[rule]["g"], DR[rule]["g5"]["g"]),
                                              "se": rel(p5[rule]["se"], DR[rule]["g5"]["se"]),
                                              "series": [p5[rule]["series"], DR[rule]["g5"]["n_series"]],
                                              "lost": lost}

# ---- growth rows (Table 4) ----
g3, s3, s3f = p3["g"], E7["pooled_gaps_W5:d3|libra.rl vs ML.rl"]["se"], \
    E7["pooled_gaps_W5:d3|libra.rl vs ML.rl"]["frozen_formula_se"]
rows = {}
for rule in ("A", "B", "none"):
    G = DR[rule]["g5"]
    gr = growth(g3, s3, G["g"], G["se"])
    pr = DR[rule]["paired_ratio_libra_over_ML_rl"]
    gr["check_vs_final_round1_ln_ratio"] = rel(-gr["Delta"], pr["ln_ratio"])
    gr["check_vs_final_round1_se_ln"] = rel(gr["se"], pr["se_ln"])
    gr["g5"], gr["g5_se"], gr["g5_ci95"], gr["series"] = G["g"], G["se"], G["ci95"], G["n_series"]
    rows[{"A": "headline_rule_A", "B": "rule_B", "none": "no_exclusion"}[rule]] = gr
G = DR["A"]["g5"]
rows["frozen_se_formula_rule_A"] = growth(g3, s3f, G["g"], G["se_frozen_formula"])
rows["frozen_se_formula_rule_A"]["g5_ci95"] = [G["g"] - Z95 * G["se_frozen_formula"],
                                               G["g"] + Z95 * G["se_frozen_formula"]]
res["growth_rows"] = rows
res["g3_W5"] = {"g": g3, "se": s3, "se_frozen_formula": s3f, "ci95": p3["ci95"], "series": p3["series"]}

# ---- leave one out on the final tally ----
loo = {}
for rule, excl in (("A", failing), ("none", set())):
    base = fits5[rule]
    one_leaf = []
    for f in base:
        for x in f["leaves"]:
            nm = name(5, x["patch"], x["basis"], x["r"])
            fl, lost = series_set(TL, 5, R_LIBRA, set(excl) | {nm})
            q = pooled(fl)
            gr = growth(g3, s3, q["g"], q["se"])
            one_leaf.append({"left_out": nm, "series": q["series"], "lost": lost, "g5": q["g"], "g5_se": q["se"],
                             "Delta": gr["Delta"], "Delta_se": gr["se"], "lambda_ratio": gr["lambda_ratio"]})
    one_round = []
    for r in R_LIBRA:
        rr = [x for x in R_LIBRA if x != r]
        fl3, _ = series_set(TL, 3, rr, set())
        fl5, lost = series_set(TL, 5, rr, excl)
        q3, q5 = pooled(fl3), pooled(fl5)
        gr = growth(q3["g"], q3["se"], q5["g"], q5["se"])
        one_round.append({"left_out_r": r, "series3": q3["series"], "series5": q5["series"], "lost5": lost,
                          "g3": q3["g"], "g5": q5["g"], "g5_se": q5["se"], "Delta": gr["Delta"], "Delta_se": gr["se"],
                          "lambda_ratio": gr["lambda_ratio"]})
    rng = lambda L, k: [min(x[k] for x in L), max(x[k] for x in L)]  # noqa: E731
    loo[rule] = {"one_d5_leaf": {"leaves": len(one_leaf), "g5_range": rng(one_leaf, "g5"),
                                 "Delta_range": rng(one_leaf, "Delta"), "rows": one_leaf},
                 "one_round_count_both_distances": {"g5_range": rng(one_round, "g5"),
                                                    "g3_range": rng(one_round, "g3"),
                                                    "Delta_range": rng(one_round, "Delta"), "rows": one_round}}
res["leave_one_out_final"] = loo

# ---- Table 3: every decoder's mean paired difference on its W5 windows (corrected SE) ----
FAW = FA["pooled_gaps_W5"]
t3, tables = {}, {}
for a, (w3, w5) in WINDOWS.items():
    T = TL if a == "libra" else {}
    if a != "libra":
        T.update(table(a, 3, w3))
        if w5:
            T.update(table(a, 5, w5))
    tables[a] = T
    o = {"windows": {"3": w3, "5": w5}}
    for d, w in ((3, w3), (5, w5)):
        if not w:
            continue
        fl, lost = series_set(T, d, w, failing if d == 5 else set())
        q = pooled(fl)
        qf = pooled(series_set(T, d, w, failing if d == 5 else set(), use_cov=False)[0])
        key = "d%d|%s.rl vs ML.rl" % (d, a)
        o[str(d)] = {"series": q["series"], "lost": lost, "g": q["g"], "g_se": q["se"], "mean_eps_ml": q["mean_eps_ml"],
                     "mean_delta": q["mean_delta"], "se_mean_delta": q["se_mean_delta"],
                     "se_mean_delta_frozen_formula": qf["se_mean_delta"],
                     "analysis_json": {k: FAW[key][k] for k in ("n_series", "mean_eps_ML", "mean_delta", "se_delta")},
                     "check_mean_delta_vs_analysis_json": rel(q["mean_delta"], FAW[key]["mean_delta"]),
                     "check_se_frozen_vs_analysis_json": rel(qf["se_mean_delta"], FAW[key]["se_delta"]),
                     "check_g_vs_final_round1": rel(q["g"], E7["pooled_gaps_W5:" + key]["g"]),
                     "check_g_se_vs_final_round1": rel(q["se"], E7["pooled_gaps_W5:" + key]["se"])}
    t3[a] = o
res["table3_abs_excess"] = t3

# ---- Fig. 2: matched own fits of ML(RL) and Libra ----
fig = {}
R3L = J["R3"]["decoders"]["libra"]
for rule, R3k in (("A", "ruleA"), ("none", "no_exclusion")):
    o = {}
    for d, fl in ((3, f3), (5, fits5[rule])):
        own_m = [own_fit(f["leaves"], "k_ml") for f in fl]
        own_a = [own_fit(f["leaves"], "k_a") for f in fl]
        n = len(fl)
        o[str(d)] = {"series": n, "series_names": [f["patch"] + "|" + f["basis"] for f in fl],
                     "eps_ml": float(np.mean([x["eps"] for x in own_m])),
                     "se_ml": math.sqrt(sum(x["se"] ** 2 for x in own_m)) / n,
                     "eps_libra": float(np.mean([x["eps"] for x in own_a])),
                     "se_libra": math.sqrt(sum(x["se"] ** 2 for x in own_a)) / n,
                     "eps_libra_paired": float(np.mean([f["eps_a"] for f in fl])),
                     "own_ml_minus_paired_ml_max": max(abs(x["eps"] - f["eps_ml"]) for x, f in zip(own_m, fl))}
    for who, k, ks in (("ml", "eps_ml", "se_ml"), ("libra", "eps_libra", "se_libra")):
        lam = o["3"][k] / o["5"][k]
        o["Lambda_" + who] = lam
        o["Lambda_" + who + "_se"] = lam * math.sqrt((o["3"][ks] / o["3"][k]) ** 2 + (o["5"][ks] / o["5"][k]) ** 2)
    o["own_ratio"] = o["Lambda_libra"] / o["Lambda_ml"]
    t7 = R3L[R3k]["table7"]
    o["check_vs_r3"] = {"Lambda_libra": rel(o["Lambda_libra"], t7["lambda_a_own_matched"]),
                        "Lambda_ml": rel(o["Lambda_ml"], t7["lambda_ml_own_matched"]),
                        "own_ratio": rel(o["own_ratio"], t7["own_ratio_matched"])}
    o["paired_ratio"] = DR[{"A": "A", "none": "none"}[rule]]["paired_ratio_libra_over_ML_rl"]
    fig[rule] = o
res["matched_fig2"] = fig

# ---- census rows: Delta with g3 fixed ----
D2A = J["D2"]["rules"]["A"]
D2N = J["D2"]["rules"]["none"]
dl = lambda g5: math.log1p(g5) - math.log1p(g3)  # noqa: E731
cen = {"g3": g3}
for lab, R in (("A", D2A), ("none", D2N)):
    cen[lab] = {k: {"g5": v["g5"]["g"], "Delta": dl(v["g5"]["g"]), "count_gap": v["count_gap"]["rel"]}
                for k, v in R["rows"].items()}
    cen[lab]["leaf_weighted"] = {"g5": R["estimate_leaf_weighted"]["g5"]["g"],
                                 "Delta": dl(R["estimate_leaf_weighted"]["g5"]["g"]),
                                 "count_gap": R["estimate_leaf_weighted"]["count_gap"]["rel"]}
    cen[lab]["by_round_estimate"] = {"g5": R["estimate_by_round"]["g5"]["g"],
                                     "Delta": dl(R["estimate_by_round"]["g5"]["g"]),
                                     "count_gap": R["estimate_by_round"]["count_gap"]["rel"]}
PB = J["R5"]["per_round_boxes"]
for key in ("C_only|each95|A", "C_only|each95|none", "C_only|alpha_over_3|A", "C_only|alpha_over_4|A",
            "C_only|joint95_bonferroni|A"):
    b = PB[key]
    cen[key] = {"g5_min": b["g5_min"], "g5_max": b["g5_max"], "Delta_min": dl(b["g5_min"]),
                "Delta_max": dl(b["g5_max"]), "count_gap_min": b["count_gap_min"], "count_gap_max": b["count_gap_max"]}
cen["check_d2_box_by_round_vs_r5_each95_A"] = {
    "g5_min": rel(D2A["box_by_round"]["without_tau3_only"]["g5_min"]["value"], PB["C_only|each95|A"]["g5_min"]),
    "g5_max": rel(D2A["box_by_round"]["without_tau3_only"]["g5_max"]["value"], PB["C_only|each95|A"]["g5_max"])}
res["census_growth"] = cen

# ---- B9: Table 5's intervals ----
F1, EX, RT, P3F = J["F1"], J["EX"], J["RT"], J["P3F"]
cal = RT["heldout"]["calibration"]
b9 = {"RL": {"R": P3F["by_decoder"]["ML(RL DEM)"]["ratio"], "R_ci95": P3F["by_decoder"]["ML(RL DEM)"]["ci95"],
             "R_conf": P3F["by_decoder"]["ML(RL DEM)"]["conf"]["ratio"],
             "R_conf_se": P3F["by_decoder"]["ML(RL DEM)"]["conf"]["se_sqrtk"]},
      "temperature_series": {"R": cal["temper_series"]["R"], "R_se": cal["temper_series"]["R_se"],
                             "R_ci95": [cal["temper_series"]["R"] - Z95 * cal["temper_series"]["R_se"],
                                        cal["temper_series"]["R"] + Z95 * cal["temper_series"]["R_se"]],
                             "R_conf": cal["temper_series"]["R_conf"],
                             "R_conf_se_sqrtk_computed": cal["temper_series"]["R_conf"]
                             / math.sqrt(cal["temper_series"]["conf_errors"]),
                             "conf_errors": cal["temper_series"]["conf_errors"],
                             "share": RT["heldout"]["share_of_RL_excess_removed"]["temper_series_R"]["share"],
                             "share_ci95": RT["heldout"]["share_of_RL_excess_removed"]["temper_series_R"]["ci95"],
                             "logloss_gain": RT["heldout"]["diff_RL_minus_temper_series"]["mean"],
                             "logloss_gain_ci95": RT["heldout"]["diff_RL_minus_temper_series"]["ci95"]},
      "T_hat": {"R": P3F["by_decoder"]["ML(T-hat)"]["ratio"], "R_ci95": P3F["by_decoder"]["ML(T-hat)"]["ci95"],
                "R_conf": P3F["by_decoder"]["ML(T-hat)"]["conf"]["ratio"],
                "R_conf_se": P3F["by_decoder"]["ML(T-hat)"]["conf"]["se_sqrtk"],
                "share": P3F["refit_share_of_excess_removed"]["R"]["share"],
                "share_ci95": P3F["refit_share_of_excess_removed"]["R"]["ci95"],
                "logloss_gain": F1["logloss"]["diff_RL-T"]["mean"], "logloss_gain_ci95": F1["logloss"]["diff_RL-T"]["ci95"]},
      "D_hat": {"R": F1["calibration_D"]["ML(D-hat)"]["ratio"], "R_ci95": F1["calibration_D"]["ML(D-hat)"]["ci95"],
                "R_conf": F1["calibration_D"]["ML(D-hat)"]["conf"]["ratio"],
                "R_conf_se": F1["calibration_D"]["ML(D-hat)"]["conf"]["se_sqrtk"],
                "share": F1["calibration_D"]["share_of_RL_excess_removed"]["R"]["share"],
                "share_ci95": F1["calibration_D"]["share_of_RL_excess_removed"]["R"]["ci95"],
                "logloss_gain": F1["logloss"]["diff_RL-D"]["mean"], "logloss_gain_ci95": F1["logloss"]["diff_RL-D"]["ci95"]}}
for k, X in (("T_hat_mech", "Tm"), ("D_hat_mech", "Dm")):
    h = EX["heldout"][X]
    c = h["calibration"]["ML(X)"]
    s = h["calibration"]["share_of_RL_excess_removed"]
    b9[k] = {"R": c["ratio"], "R_ci95": c["ci95"], "R_conf": c["conf"]["ratio"], "R_conf_se": c["conf"]["se_sqrtk"],
             "R_RL_same_shots": s["R_RL_same_shots"], "R_conf_RL_same_shots": s["R_conf_RL_same_shots"],
             "share": s["R"]["share"], "share_ci95": s["R"]["ci95"],
             "logloss_gain": h["logloss"]["diff_RL-" + X]["mean"], "logloss_gain_ci95": h["logloss"]["diff_RL-" + X]["ci95"]}
res["b9"] = b9

# ---- C12: every decoder on Libra's round counts {10, 30, 50} ----
c12 = {}
for a in ("harmony", "corrmatch", "corr_mwpm", "mwpm"):
    T = tables[a]  # W5 = {10, 13, 30, 50} at both distances holds {10, 30, 50}
    assert all((d, p, b, r) in T for d, PP in ((3, P3), (5, P5)) for p in PP for b in "XZ" for r in R_LIBRA)
    o = {}
    fl3, _ = series_set(T, 3, R_LIBRA, set())
    q3 = pooled(fl3)
    for rule, excl in (("A", failing), ("none", set())):
        fl5, lost = series_set(T, 5, R_LIBRA, excl)
        q5 = pooled(fl5)
        gr = growth(q3["g"], q3["se"], q5["g"], q5["se"])
        o[rule] = {"g3": q3["g"], "g3_ci95": q3["ci95"], "series3": q3["series"], "g5": q5["g"], "g5_ci95": q5["ci95"],
                   "series5": q5["series"], "lost5": lost, "lambda_ratio": gr["lambda_ratio"],
                   "lambda_ratio_ci95": gr["lambda_ratio_ci95"], "Delta": gr["Delta"], "Delta_se": gr["se"]}
    c12[a] = o
res["c12_libra_windows"] = c12
res["checks"] = checks

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "leaves"}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(clean(res), open(out, "w"), indent=1)
print(json.dumps(clean({"checks": checks,
                        "growth_rows": {k: {kk: v[kk] for kk in ("Delta", "se", "lambda_ratio")} for k, v in rows.items()},
                        "loo": {r: {k: {kk: v[kk] for kk in v if kk.endswith("range")} for k, v in L.items()}
                                for r, L in loo.items()}}), indent=1))
for a, o in t3.items():
    print(a, {d: ("%.4f e-4 (+- %.4f; frozen %.4f; FA %.4f) chk %.1e %.1e" % (
        1e4 * o[d]["mean_delta"], 1e4 * o[d]["se_mean_delta"], 1e4 * o[d]["se_mean_delta_frozen_formula"],
        1e4 * o[d]["analysis_json"]["se_delta"], o[d]["check_mean_delta_vs_analysis_json"],
        o[d]["check_g_vs_final_round1"])) for d in ("3", "5") if d in o})
for rule, o in fig.items():
    print("fig2", rule, {d: {k: o[d][k] for k in ("series", "eps_ml", "se_ml", "eps_libra", "se_libra",
                                                  "eps_libra_paired")} for d in ("3", "5")},
          "Lambda ML %.4f +- %.4f, Libra %.4f +- %.4f, ratio %.5f" % (o["Lambda_ml"], o["Lambda_ml_se"],
                                                                     o["Lambda_libra"], o["Lambda_libra_se"],
                                                                     o["own_ratio"]), o["check_vs_r3"])
for a, o in c12.items():
    print("c12", a, {r: "g3 %.2f g5 %.2f (%d) ratio %.3f [%.3f, %.3f]" % (
        100 * v["g3"], 100 * v["g5"], v["series5"], v["lambda_ratio"], *v["lambda_ratio_ci95"]) for r, v in o.items()})
