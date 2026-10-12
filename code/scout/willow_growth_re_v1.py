"""Post hoc sensitivity for the growth of Libra's gap to ML(RL) from d = 3 to d = 5 (scout, 2026-10-10).

Main's answer (d) to scout's internal review, relayed by the coordinator after 21:27Z: a random-effects SE and a rank test for g5's growth, labelled post hoc
sensitivity, no decoding. Descriptive; not a pre-registered statistic.

What it does, on Libra's W5 rounds {10, 30, 50} at both distances (18 series at d = 3, 8 at d = 5):
  1. Builds every leaf's 2x2 table (Libra wrong, ML(RL) wrong) from decoder's stored per-shot outputs, and checks it
     against reservoir's d5_leaves.csv.
  2. Refits every series with the paired estimator of PP section 4 and the SE of PP section 9 entry 7 (delta method
     with Cov(beta, b_M), Birge factors on both fits), and checks the pooled g3 and g5 against the posted values.
  3. Random effects: DerSimonian-Laird on the per-series relative gaps, with the Hartung-Knapp-Sidik-Jonkman
     t-interval; and the pre-registered pooled ratio g = mean(d_s)/mean(eps_ML,s) with a series-level (cluster) SE.
  4. The growth ln(1 + g5) - ln(1 + g3) under each SE, with Welch-Satterthwaite degrees of freedom.
  5. A rank test: exact one-sided Mann-Whitney of the 8 d = 5 per-series gaps against the 18 at d = 3.
  6. Holm over the six decoders from reservoir's matched table (gap_table.json, rule A), and the same growth with a
     series-level SE for every decoder from reservoir's per-series file (gap_series.csv), with Holm again on its
     two-sided t p-values.
Rule A (PP section 9 entry 12): a leaf that still fails K2 leaves the ML(RL) fits; its series keeps its other rounds.

Inputs (read only): results/decoder/willow/decode/ (per-shot outputs), results/decoder/willow/supplementary/
k2_extra_rung/summary.json, results/reservoir/willow_d5check/d5_leaves.csv, results/reservoir/willow_gapmech/
gap_table.json and gap_series.csv.
Usage: OMP_NUM_THREADS=1 nice -n 10 python3 -I willow_growth_re.py <experimental-root>/ <out.json>
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
DEC = root + "results/decoder/willow/decode/google_105Q_surface_code_d3_d5_d7/"
P3 = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
ROUNDS = [10, 30, 50]
LADDER = "ml_tau2_1e-6-1e-8_0.1.npz"
Z95 = stats.norm.ppf(0.975)


def leaf_table(d, patch, basis, r):
    sub = "rl" if d == 3 else "rl_s5"
    base = "%sd%d_at_%s/%s/r%02d/%s/" % (DEC, d, patch, basis, r, sub)
    sh = np.load(base + "shots.npz", allow_pickle=False)
    ml = np.load(base + ("ml_exact.npz" if d == 3 else LADDER), allow_pickle=False)
    assert np.array_equal(sh["idx"], ml["idx"])
    names = list(sh["shipped_names"])
    li = [i for i, s in enumerate(names) if "libra_decoder_with_rl_optimized_prior" in s]
    assert len(li) == 1, names
    obs = sh["obs"].astype(np.uint8)
    pm = ml["pred"][:, 0].astype(np.uint8)
    assert np.array_equal(pm, (ml["llr"] > 0).astype(np.uint8))
    pl = sh["shipped"][li[0]].astype(np.uint8)
    mw, lw = pm != obs, pl != obs
    return {"d": d, "patch": patch, "basis": basis, "r": r, "n": int(obs.size), "k_ml": int(mw.sum()),
            "k_libra": int(lw.sum()), "n11": int(np.sum(mw & lw))}


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
    """Paired fit of Libra against ML(RL) over the leaves L of one series (PP section 4; entry 7's SE)."""
    r = np.array([x["r"] for x in L], float)
    n = np.array([x["n"] for x in L], float)
    EA, EM = np.array([x["k_libra"] for x in L]) / n, np.array([x["k_ml"] for x in L]) / n
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
    return {"eps_ml": eM, "eps_libra": eA, "delta": dlt, "var_delta": float(Jd @ S @ Jd),
            "var_eps_ml": float(Je @ S @ Je), "cov_delta_eps": float(Jd @ S @ Je),
            "rel": dlt / eM, "rounds": [int(v) for v in r], "birge_ml": BM, "birge_z": Bz,
            "chi2_red_ml": chiM, "chi2_red_z": chiz}


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
    # series-level (cluster) SE of the same ratio estimator: residuals u_s = delta_s - g eps_s
    u = np.array([f["delta"] - g * f["eps_ml"] for f in fits])
    se_cl = math.sqrt(S / (S - 1) * np.sum(u ** 2)) / (S * E)
    return {"g": g, "se_entry7": se, "se_cluster": se_cl, "series": S}


def dersimonian_laird(fits):
    rel = np.array([f["rel"] for f in fits])
    v = np.array([(f["var_delta"] - 2 * f["rel"] * f["cov_delta_eps"] + f["rel"] ** 2 * f["var_eps_ml"])
                  / f["eps_ml"] ** 2 for f in fits])
    w = 1 / v
    S = len(rel)
    mu_fe = np.sum(w * rel) / np.sum(w)
    Q = float(np.sum(w * (rel - mu_fe) ** 2))
    Cc = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (S - 1)) / Cc)
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * rel) / np.sum(ws))
    se = math.sqrt(1 / np.sum(ws))
    se_hk = math.sqrt(float(np.sum(ws * (rel - mu) ** 2)) / ((S - 1) * np.sum(ws)))
    t = stats.t.ppf(0.975, S - 1)
    return {"series": S, "fixed_effect_mean": float(mu_fe), "Q": Q, "df": S - 1,
            "p_Q": float(stats.chi2.sf(Q, S - 1)), "I2": max(0.0, (Q - (S - 1)) / Q) if Q > 0 else 0.0,
            "tau2": tau2, "tau": math.sqrt(tau2), "mu_re": mu, "se_re": se, "se_hksj": se_hk,
            "ci95_hksj_t": [mu - t * se_hk, mu + t * se_hk], "per_series_rel": rel.tolist(),
            "per_series_se": np.sqrt(v).tolist()}


def growth(g3, s3, df3, g5, s5, df5):
    a, b = s3 / (1 + g3), s5 / (1 + g5)
    D = math.log(1 + g5) - math.log(1 + g3)
    se = math.sqrt(a * a + b * b)
    out = {"ln_growth": D, "se": se, "z": D / se, "p_one_sided_normal": float(stats.norm.sf(D / se)),
           "p_two_sided_normal": float(2 * stats.norm.sf(abs(D / se))),
           "lambda_ratio": math.exp(-D),
           "lambda_ratio_ci95_normal": [math.exp(-D - Z95 * se), math.exp(-D + Z95 * se)]}
    if df3 and df5:
        df = (a * a + b * b) ** 2 / (a ** 4 / df3 + b ** 4 / df5)
        t = stats.t.ppf(0.975, df)
        out.update(df_welch=df, p_one_sided_t=float(stats.t.sf(D / se, df)),
                   p_two_sided_t=float(2 * stats.t.sf(abs(D / se), df)),
                   lambda_ratio_ci95_t=[math.exp(-D - t * se), math.exp(-D + t * se)])
    return out


# ---- 1. leaf tables from decoder's per-shot outputs, checked against reservoir's ----
leaves = [leaf_table(3, p, b, r) for p in P3 for b in "XZ" for r in ROUNDS]
leaves += [leaf_table(5, p, b, r) for p in P5 for b in "XZ" for r in ROUNDS]
ref = {(int(x["d"]), x["patch"], x["basis"], int(x["r"])): x
       for x in csv.DictReader(open(root + "results/reservoir/willow_d5check/d5_leaves.csv")) if x["has_libra"] == "True"}
mism = [(x["d"], x["patch"], x["basis"], x["r"]) for x in leaves
        if any(int(ref[(x["d"], x["patch"], x["basis"], x["r"])][k]) != x[k] for k in ("n", "k_ml", "k_libra", "n11"))]

k2 = json.load(open(root + "results/decoder/willow/supplementary/k2_extra_rung/summary.json"))
failing = {s.split("google_105Q_surface_code_d3_d5_d7/")[1] for s in k2["still_failing"]}


def fits_for(d, rule):
    out = []
    for p in (P3 if d == 3 else P5):
        for b in "XZ":
            L = [x for x in leaves if x["d"] == d and x["patch"] == p and x["basis"] == b]
            if rule == "A" and d == 5:
                L = [x for x in L if "d5_at_%s/%s/r%02d" % (p, b, x["r"]) not in failing]
            f = series_fit(L)
            f.update(patch=p, basis=b)
            out.append(f)
    return out


res = {"label": "scout, POST HOC SENSITIVITY (main's answer (d), relayed after 21:27Z); descriptive; no decoding; "
                "Libra against ML(RL) on W5 rounds {10, 30, 50}; d = 5 interim (K2 as in summary.json)",
       "k2_summary_made": k2["made"], "k2_still_failing": sorted(failing),
       "leaf_tables": {"leaves": len(leaves), "mismatches_vs_reservoir_d5_leaves_csv": mism}}
f3 = fits_for(3, None)
for rule in ("no_exclusion", "A"):
    f5 = fits_for(5, "A" if rule == "A" else None)
    p3, p5 = pooled(f3), pooled(f5)
    dl3, dl5 = dersimonian_laird(f3), dersimonian_laird(f5)
    x3, x5 = np.array(dl3["per_series_rel"]), np.array(dl5["per_series_rel"])
    mw = stats.mannwhitneyu(x5, x3, alternative="greater", method="exact")
    res[rule] = {
        "pooled_g3_entry7": p3, "pooled_g5_entry7": p5,
        "growth_entry7_se": growth(p3["g"], p3["se_entry7"], None, p5["g"], p5["se_entry7"], None),
        "growth_cluster_se": growth(p3["g"], p3["se_cluster"], 17, p5["g"], p5["se_cluster"], 7),
        "random_effects_d3": dl3, "random_effects_d5": dl5,
        "growth_random_effects_hksj": growth(dl3["mu_re"], dl3["se_hksj"], 17, dl5["mu_re"], dl5["se_hksj"], 7),
        "growth_random_effects_dl": growth(dl3["mu_re"], dl3["se_re"], None, dl5["mu_re"], dl5["se_re"], None),
        "rank_test": {"mann_whitney_U_d5_over_d3": float(mw.statistic), "pairs": int(x5.size * x3.size),
                      "p_one_sided_exact": float(mw.pvalue),
                      "p_two_sided_exact": float(stats.mannwhitneyu(x5, x3, alternative="two-sided",
                                                                    method="exact").pvalue),
                      "d5_series_above_d3_max": int(np.sum(x5 > x3.max())),
                      "d5_series_above_d3_median": int(np.sum(x5 > np.median(x3))),
                      "d3_range": [float(x3.min()), float(x3.max())], "d5_range": [float(x5.min()), float(x5.max())],
                      "d5_min_series": [f["patch"] + "/" + f["basis"] for f in f5 if f["rel"] == x5.min()]},
        "series_d5": [{k: f[k] for k in ("patch", "basis", "rounds", "rel", "eps_ml", "delta", "birge_z",
                                         "chi2_red_z")} for f in f5]}
res["series_d3"] = [{k: f[k] for k in ("patch", "basis", "rounds", "rel", "eps_ml", "delta", "birge_z",
                                       "chi2_red_z")} for f in f3]

# ---- 6. every decoder: Holm on reservoir's z, and a series-level SE from reservoir's per-series file ----
GT = json.load(open(root + "results/reservoir/willow_gapmech/gap_table.json"))
decs = ["libra", "tesseract", "harmony", "corrmatch", "corr_mwpm", "mwpm"]
zs = {k: GT["table"][k]["matched|ruleA"]["growth_z"] for k in decs}
ps = {k: 2 * stats.norm.sf(abs(z)) for k, z in zs.items()}
order = sorted(decs, key=lambda k: ps[k])
holm, run = {}, 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(decs) - i) * ps[k]))
    holm[k] = run
rows = list(csv.DictReader(open(root + "results/reservoir/willow_gapmech/gap_series.csv")))
every = {}
for k in decs:
    o = {"reservoir_growth_z": zs[k], "p_two_sided": ps[k], "holm_adjusted_p": holm[k]}
    for d in ("3", "5"):
        R = [x for x in rows if x["decoder"] == k and x["rounds_set"] == "matched" and x["k2"] == "ruleA"
             and x["d"] == d]
        eM = np.array([float(x["eps_ml"]) for x in R])
        dl = np.array([float(x["eps_a"]) for x in R]) - eM
        g = dl.sum() / eM.sum()
        S = len(R)
        o["g" + d] = g
        o["se_cluster_" + d] = math.sqrt(S / (S - 1) * np.sum((dl - g * eM) ** 2)) / eM.sum()
        o["series_" + d] = S
    o["growth_cluster_se"] = growth(o["g3"], o["se_cluster_3"], o["series_3"] - 1,
                                    o["g5"], o["se_cluster_5"], o["series_5"] - 1)
    every[k] = o
# Holm again, on the two-sided t p-values of the growth with series-level SEs (Welch df)
pt = {k: 2 * float(stats.t.sf(abs(every[k]["growth_cluster_se"]["z"]), every[k]["growth_cluster_se"]["df_welch"]))
      for k in decs}
run = 0.0
for i, k in enumerate(sorted(decs, key=lambda k: pt[k])):
    run = max(run, min(1.0, (len(decs) - i) * pt[k]))
    every[k]["growth_cluster_se"]["p_two_sided_t"] = pt[k]
    every[k]["growth_cluster_se"]["holm_adjusted_p_t"] = run
res["every_decoder_rule_A_matched_rounds"] = every
res["input_md5_note"] = "see code/scout/README.md"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"mismatches": mism, "A": {k: res["A"][k] for k in ("pooled_g3_entry7", "pooled_g5_entry7")}},
                 indent=1))
