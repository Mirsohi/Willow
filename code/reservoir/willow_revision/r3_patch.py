"""R3 of the revision (review point 1.5): target population and patch-level sensitivity of the final analysis, and
the paired Lambda-ratio estimator against the ratio of separately fitted Lambdas (Table 7).

reservoir, 2026-10-13. POST HOC: a revision analysis prompted by the review, not pre-registered. Read-only; nothing is
decoded. The estimators are the final analysis's: entry 7's paired gap per series (d5_check.paired), pooled as
g = mean_s(eps_A - eps_ML) / mean_s(eps_ML) (d5_check.g_stat); rule A on K2's final tally is primary and no exclusion
the sensitivity row; each decoder on its matched rounds (Libra 10, 30, 50; Tesseract 10, 13, 30; the others 10, 13,
30, 50), as in reservoir's matched table (willow_gapmech/gapmech.py), whose d = 5 gaps equal decoder's final_round1.

What it adds:
 1. Three kinds of interval for g_3, g_5 and the growth G = ln(1 + g_5) - ln(1 + g_3) (G = -ln of the paired Lambda
    ratio):
      conditional  entry 7's SE: shot noise, conditional on the fixed collection of experiments (the paper's);
      series       a cluster (CR1) SE with the series as the unit, t with S - 1 df at each distance and Welch df for
                   G (scout's willow_growth_re.py convention, reproduced here as a check);
      patch        the same with the patch (its X and Z series together) as the unit: 9 clusters at d = 3, 4 at
                   d = 5; and the delete-one-patch jackknife.
 2. Per-patch d = 5 gaps (conditional SEs) and each patch's own growth against the pooled d = 3 gap.
 3. Leave one patch out at d = 5 (d = 3 kept whole), and leave one region out: a d = 5 patch together with the d = 3
    patches that lie inside it (all 17 of their qubits shared; from the decode summaries' qubit coordinates).
 4. X and Z within a patch: across patches, the correlation of the X and Z series gaps, and a one-way ICC.
 5. Table 7, per decoder: (a) the paired ratio (1 + g_3)/(1 + g_5); (b) the ratio of separately fitted Lambdas on the
    same rounds, series and shots, Lambda_X = mean_s eps_X(d = 3) / mean_s eps_X(d = 5), with each decoder's own
    unpaired fit (d5_check.eps_fit); conditional CIs for (a), (b) and (b) - (a) from one bootstrap of the leaves' 2x2
    tables (multinomial within leaf, which is the same as resampling shots within leaf for these statistics;
    B = 10,000, numpy default_rng(20261013)); patch-jackknife intervals with Welch df; and (c) v1.1's Table 7 ratio
    from its Lambda column (decoder's final_round1.json decisions.lambda_3_5: each decoder's own window and shots,
    rule A on ML(RL) only), read and cited, not recomputed.
 6. Qubit overlap of the patches.

Usage: nice -n 10 python3 -I r3_patch.py ARCHIVE DECODE K2_DIR PROJECT_ROOT
       -> results/reservoir/willow_revision/r3_patch.json and r3_patch_rows.csv (through qrc.outpaths)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("willow_gapmech", "willow_d5check", "willow_check", ""):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
from d5_check import ARCH, BASES, P3, P5, W5, Z95, eps_fit, g_stat, k2_verdicts, md5, paired, rel_of  # noqa: E402
from gapmech import DECODERS, leaf_counts, read_leaf  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

ROUNDS = {"libra": [10, 30, 50], "tesseract": [10, 13, 30]}
B, SEED = 10000, 20261013
RULES = ("ruleA", "no_exclusion")


# ---------------------------------------------------------------------------------------------------------------
def build_series(leaves, a, d, rounds, excluded):
    ser, lost = [], []
    for p in (P3 if d == 3 else P5):
        for b in BASES:
            L = []
            for r in rounds:
                if rel_of(d, p, b, r) in excluded:
                    continue
                rec = leaves[(d, p, b, r)]
                if a not in rec["preds"]:
                    continue
                c = leaf_counts(rec, a)
                if c["n"] > 0:
                    L.append((r, c))
            if len(L) < 2:
                lost.append(f"{p}/{b}")
                continue
            rr = [x[0] for x in L]
            kA = [x[1]["kA"] for x in L]
            kM = [x[1]["kM"] for x in L]
            n11 = [x[1]["n11"] for x in L]
            n = [x[1]["n"] for x in L]
            pr = paired(rr, kA, kM, n11, n)
            own_a, own_m = eps_fit(rr, kA, n), eps_fit(rr, kM, n)
            ser.append({"patch": p, "basis": b, "rounds": rr, "kA": kA, "kM": kM, "n11": n11, "n": n, "pr": pr,
                        "own_a": own_a, "own_m": own_m})
    return ser, lost


def pooled_g(ser):
    d = np.array([s["pr"]["d"] for s in ser])
    e = np.array([s["pr"]["eps_ml"] for s in ser])
    return float(d.sum() / e.sum())


def cluster_se(ser, unit):
    """CR1 cluster SE of g = sum(d)/sum(e): u_c = sum_{s in c} (d_s - g e_s); se = sqrt(C/(C-1) sum u_c^2)/sum(e)."""
    g = pooled_g(ser)
    e = sum(s["pr"]["eps_ml"] for s in ser)
    u = {}
    for s in ser:
        k = s["patch"] if unit == "patch" else (s["patch"], s["basis"])
        u[k] = u.get(k, 0.0) + s["pr"]["d"] - g * s["pr"]["eps_ml"]
    C = len(u)
    se = math.sqrt(C / (C - 1) * sum(v * v for v in u.values())) / e if C > 1 else float("nan")
    return {"g": g, "se": se, "clusters": C, "df": C - 1}


def jackknife(ser, stat):
    pats = sorted({s["patch"] for s in ser})
    vals = [stat([s for s in ser if s["patch"] != p]) for p in pats]
    C = len(pats)
    m = float(np.mean(vals))
    return {"left_out": pats, "values": [float(v) for v in vals], "se": math.sqrt((C - 1) / C *
            float(np.sum((np.array(vals) - m) ** 2))), "clusters": C, "df": C - 1}


def growth_t(l3, s3, df3, l5, s5, df5):
    """G = l5 - l3 (log scale), se from the two parts, Welch-Satterthwaite df, t interval; Lambda ratio = exp(-G)."""
    G = l5 - l3
    se = math.hypot(s3, s5)
    out = {"G": G, "se": se, "lambda_ratio": math.exp(-G),
           "lambda_ratio_ci95_normal": [math.exp(-G - Z95 * se), math.exp(-G + Z95 * se)],
           "p_two_sided_normal": float(2 * stats.norm.sf(abs(G / se)))}
    if df3 and df5:
        df = (s3 ** 2 + s5 ** 2) ** 2 / (s3 ** 4 / df3 + s5 ** 4 / df5)
        t = float(stats.t.ppf(0.975, df))
        out.update(df_welch=df, t975=t, p_two_sided_t=float(2 * stats.t.sf(abs(G / se), df)),
                   lambda_ratio_ci95_t=[math.exp(-G - t * se), math.exp(-G + t * se)])
    return out


def g_ci_t(g, se, df):
    t = float(stats.t.ppf(0.975, df)) if df else Z95
    return {"g": g, "se": se, "df": df, "ci95": [g - t * se, g + t * se]}


def conditional(ser3, ser5):
    o3, o5 = g_stat([s["pr"] for s in ser3]), g_stat([s["pr"] for s in ser5])
    l3, l5 = math.log1p(o3["g"]), math.log1p(o5["g"])
    s3, s5 = o3["se_inflated"] / (1 + o3["g"]), o5["se_inflated"] / (1 + o5["g"])
    gr = growth_t(l3, s3, None, l5, s5, None)
    return {"g3": o3["g"], "g3_se": o3["se_inflated"], "g3_ci95": o3["ci95"], "g5": o5["g"], "g5_se": o5["se_inflated"],
            "g5_ci95": o5["ci95"], "series3": o3["series"], "series5": o5["series"], **gr}


def clustered(ser3, ser5, unit):
    c3, c5 = cluster_se(ser3, unit), cluster_se(ser5, unit)
    l3, l5 = math.log1p(c3["g"]), math.log1p(c5["g"])
    gr = growth_t(l3, c3["se"] / (1 + c3["g"]), c3["df"], l5, c5["se"] / (1 + c5["g"]), c5["df"])
    return {"g3": g_ci_t(c3["g"], c3["se"], c3["df"]), "g5": g_ci_t(c5["g"], c5["se"], c5["df"]),
            "clusters3": c3["clusters"], "clusters5": c5["clusters"], **gr}


def jack_growth(ser3, ser5):
    j3 = jackknife(ser3, lambda s: math.log1p(pooled_g(s)))
    j5 = jackknife(ser5, lambda s: math.log1p(pooled_g(s)))
    l3, l5 = math.log1p(pooled_g(ser3)), math.log1p(pooled_g(ser5))
    gr = growth_t(l3, j3["se"], j3["df"], l5, j5["se"], j5["df"])
    return {"ln1p_g3_jack": j3, "ln1p_g5_jack": j5, **gr}


# ---------------------------------------------------------------------------------------------------------------
def own_lambda_ratio(ser3, ser5, which="own"):
    """Ratio of separately fitted Lambdas on the given series: (mean eps_A3/mean eps_A5)/(mean eps_M3/mean eps_M5)."""
    a3 = np.mean([s["own_a"]["eps"] for s in ser3]); a5 = np.mean([s["own_a"]["eps"] for s in ser5])
    m3 = np.mean([s["own_m"]["eps"] for s in ser3]); m5 = np.mean([s["own_m"]["eps"] for s in ser5])
    return float((a3 / a5) / (m3 / m5)), float(a3 / a5), float(m3 / m5)


def slope_vec(x, y, v):
    w = 1.0 / v
    Sw, Sx, Sy = w.sum(1), (w * x).sum(1), (w * y).sum(1)
    Sxx, Sxy = (w * x * x).sum(1), (w * x * y).sum(1)
    return (Sw * Sxy - Sx * Sy) / (Sw * Sxx - Sx * Sx)


def boot_series(rng, s):
    """B bootstrap replicates of a series' paired eps_ML, eps_A (paired) and eps_A (own fit)."""
    R = len(s["rounds"])
    x = np.array(s["rounds"], float)[None, :]
    kA = np.empty((B, R)); kM = np.empty((B, R)); n11 = np.empty((B, R)); nn = np.empty((1, R))
    for i in range(R):
        n, a, m, c = s["n"][i], s["kA"][i], s["kM"][i], s["n11"][i]
        p = np.array([c, a - c, m - c, n - a - m + c], float) / n
        draw = rng.multinomial(n, p, size=B)
        n11[:, i] = draw[:, 0]; kA[:, i] = draw[:, 0] + draw[:, 1]; kM[:, i] = draw[:, 0] + draw[:, 2]
        nn[0, i] = n
    EA, EM = kA / nn, kM / nn
    yA, yM = np.log(1 - 2 * EA), np.log(1 - 2 * EM)
    vA = 4 * EA * (1 - EA) / (nn * (1 - 2 * EA) ** 2)
    vM = 4 * EM * (1 - EM) / (nn * (1 - 2 * EM) ** 2)
    cAM = 4 * (n11 / nn - EA * EM) / (nn * (1 - 2 * EA) * (1 - 2 * EM))
    vz = vA + vM - 2 * cAM
    bM = slope_vec(x, yM, vM)
    bz = slope_vec(x, yA - yM, vz)
    bA = slope_vec(x, yA, vA)
    return (1 - np.exp(bM)) / 2, (1 - np.exp(bM + bz)) / 2, (1 - np.exp(bA)) / 2


def table7_boot(rng, ser3, ser5):
    reps = {}
    for d, ser in ((3, ser3), (5, ser5)):
        M, Ap, Ao = [], [], []
        for s in ser:
            m, ap, ao = boot_series(rng, s)
            M.append(m); Ap.append(ap); Ao.append(ao)
        M, Ap, Ao = np.array(M), np.array(Ap), np.array(Ao)
        reps[d] = {"g": (Ap - M).sum(0) / M.sum(0), "lam_a": Ao.mean(0), "lam_m": M.mean(0)}
    paired_ratio = (1 + reps[3]["g"]) / (1 + reps[5]["g"])
    own_ratio = (reps[3]["lam_a"] / reps[5]["lam_a"]) / (reps[3]["lam_m"] / reps[5]["lam_m"])
    ok = np.isfinite(paired_ratio) & np.isfinite(own_ratio)

    def pc(v):
        v = v[ok]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"B": B, "finite": int(ok.sum()), "paired_ratio_ci95": pc(paired_ratio), "own_ratio_ci95": pc(own_ratio),
            "own_minus_paired_ci95": pc(own_ratio - paired_ratio), "g3_ci95": pc(reps[3]["g"]),
            "g5_ci95": pc(reps[5]["g"]), "paired_ratio_sd": float(np.std(paired_ratio[ok])),
            "own_ratio_sd": float(np.std(own_ratio[ok]))}


def table7_jack(ser3, ser5):
    """Patch jackknife of ln(paired ratio) and ln(own ratio), each split into its d = 3 and d = 5 parts."""
    def own_part(s):
        return math.log(np.mean([x["own_a"]["eps"] for x in s]) / np.mean([x["own_m"]["eps"] for x in s]))
    out = {}
    for name, f in (("paired", lambda s: math.log1p(pooled_g(s))), ("own", own_part)):
        j3, j5 = jackknife(ser3, f), jackknife(ser5, f)
        l3, l5 = f(ser3), f(ser5)
        gr = growth_t(l3, j3["se"], j3["df"], l5, j5["se"], j5["df"])
        out[name] = {"ratio": gr["lambda_ratio"], "se_ln": gr["se"], "df_welch": gr.get("df_welch"),
                     "ci95_t": gr.get("lambda_ratio_ci95_t"), "d3_jack": j3, "d5_jack": j5}
    return out


# ---------------------------------------------------------------------------------------------------------------
def qubits(dec, d, p):
    sub = "rl" if d == 3 else "rl_s5"
    m = json.load(open(os.path.join(dec, ARCH, rel_of(d, p, "X", 10), sub, "summary.json")))["meta"]["props"]
    return {tuple(x) for x in m["data_qubit_coords"]} | {tuple(x) for x in m["meas_qubit_coords"]}


def xz_corr(ser):
    by = {}
    for s in ser:
        by.setdefault(s["patch"], {})[s["basis"]] = s["pr"]["d"] / s["pr"]["eps_ml"]
    pairs = [(v["X"], v["Z"]) for v in by.values() if "X" in v and "Z" in v]
    o = {"patches_with_both": len(pairs)}
    if len(pairs) >= 3:
        x, z = np.array(pairs).T
        o["pearson_r"] = float(np.corrcoef(x, z)[0, 1])
        o["spearman_rho"] = float(stats.spearmanr(x, z).statistic)
        allv = np.concatenate([x, z]); k = len(x)
        msb = 2 * np.sum(((x + z) / 2 - allv.mean()) ** 2) / (k - 1)
        msw = np.sum((x - (x + z) / 2) ** 2 + (z - (x + z) / 2) ** 2) / k
        o["icc1"] = float((msb - msw) / (msb + msw))
        o["pairs"] = [[float(a), float(b)] for a, b in pairs]
    return o


def main(arch, dec, k2_dir, proj):
    t0 = time.time()
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = k2_verdicts(k2_dir)
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    leaves = {}
    for d, pats in ((3, P3), (5, P5)):
        for p in pats:
            for b in BASES:
                for r in W5:
                    leaves[(d, p, b, r)] = read_leaf(arch, dec, d, p, b, r, chk)
        print(f"d = {d}: leaves read ({time.time() - t0:.0f} s)", flush=True)
    Q3 = {p: qubits(dec, 3, p) for p in P3}
    Q5 = {p: qubits(dec, 5, p) for p in P5}
    inside = {p5: [p3 for p3 in P3 if Q3[p3] <= Q5[p5]] for p5 in P5}
    res = {"label": "reservoir, revision R3 (review 1.5): patch-level sensitivity and Table 7's two Lambda-ratio "
                    "estimators. POST HOC; read-only; nothing decoded.",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "k2": {"verdicts": len(k2), "still_fail": sorted(failing)}, "checks": chk,
           "code_md5": {f: md5(os.path.join(HERE, "..", f)) for f in
                        ("willow_revision/r3_patch.py", "willow_gapmech/gapmech.py", "willow_d5check/d5_check.py",
                         "willow_check/willow_check.py")},
           "bootstrap": {"B": B, "seed": SEED, "draw_order": "decoders in " + ",".join(DECODERS) + "; rules ruleA, "
                         "no_exclusion; d = 3 then 5; series in table order; leaves by round; 4 cells per leaf"},
           "patch_overlap": {"qubits_per_patch": {"d3": len(Q3[P3[0]]), "d5": len(Q5[P5[0]])},
                             "d5_pairwise_shared": {a: {b: len(Q5[a] & Q5[b]) for b in P5} for a in P5},
                             "d5_x_d3_shared": {a: {b: len(Q5[a] & Q3[b]) for b in P3} for a in P5},
                             "d3_inside_d5": inside,
                             "union_d5": len(set().union(*Q5.values())), "union_d3": len(set().union(*Q3.values()))},
           "decoders": {}}
    ref = json.load(open(os.path.join(proj, "results", "reservoir", "willow_gapmech", "final_k2", "gap_table.json")))
    grf = json.load(open(os.path.join(proj, "results", "scout", "willow_growth_re_final.json")))
    fr1 = json.load(open(os.path.join(proj, "results", "decoder", "willow", "final_round1", "final_round1.json")))
    lam = fr1["decisions"]["lambda_3_5"]
    rng = np.random.default_rng(SEED)
    rows = []
    repro = {"gap_table_max_abs_diff": 0.0, "grf_series_cluster_max_abs_diff": 0.0, "eps_ml_paired_vs_own_max": 0.0}
    for a in DECODERS:
        rounds = ROUNDS.get(a, W5)
        res["decoders"][a] = {"rounds": rounds}
        for rule in RULES:
            excl = failing if rule == "ruleA" else set()
            ser3, lost3 = build_series(leaves, a, 3, rounds, set())
            ser5, lost5 = build_series(leaves, a, 5, rounds, excl)
            for s in ser3 + ser5:
                repro["eps_ml_paired_vs_own_max"] = max(repro["eps_ml_paired_vs_own_max"],
                                                        abs(s["pr"]["eps_ml"] - s["own_m"]["eps"]))
            o = {"lost_series_d3": lost3, "lost_series_d5": lost5,
                 "series_d5": [f"{s['patch']}/{s['basis']}" for s in ser5]}
            o["conditional"] = conditional(ser3, ser5)
            rt = ref["table"][a][f"matched|{rule}"]
            repro["gap_table_max_abs_diff"] = max(repro["gap_table_max_abs_diff"],
                                                  abs(rt["g3"]["g"] - o["conditional"]["g3"]),
                                                  abs(rt["g5"]["g"] - o["conditional"]["g5"]),
                                                  abs(rt["g5"]["se_inflated"] - o["conditional"]["g5_se"]))
            o["series_cluster"] = clustered(ser3, ser5, "series")
            if rule == "ruleA":
                e = grf["every_decoder_rule_A_matched_rounds"][a]
                repro["grf_series_cluster_max_abs_diff"] = max(
                    repro["grf_series_cluster_max_abs_diff"], abs(e["se_cluster_3"] - o["series_cluster"]["g3"]["se"]),
                    abs(e["se_cluster_5"] - o["series_cluster"]["g5"]["se"]),
                    abs(e["growth_cluster_se"]["df_welch"] - o["series_cluster"]["df_welch"]))
            o["patch_cluster"] = clustered(ser3, ser5, "patch")
            o["patch_jackknife"] = jack_growth(ser3, ser5)
            g3 = o["conditional"]["g3"]
            l3, s3 = math.log1p(g3), o["conditional"]["g3_se"] / (1 + g3)
            o["per_patch_d5"], o["leave_one_patch_out_d5"], o["leave_one_region_out"] = [], [], []
            for p in P5:
                sp = [s for s in ser5 if s["patch"] == p]
                if sp:
                    q = g_stat([s["pr"] for s in sp])
                    gr = growth_t(l3, s3, None, math.log1p(q["g"]), q["se_inflated"] / (1 + q["g"]), None)
                    o["per_patch_d5"].append({"patch": p, "series": len(sp), "g5": q["g"], "g5_se": q["se_inflated"],
                                              "g5_ci95": q["ci95"], "lambda_ratio": gr["lambda_ratio"],
                                              "lambda_ratio_ci95": gr["lambda_ratio_ci95_normal"]})
                rest = [s for s in ser5 if s["patch"] != p]
                q = g_stat([s["pr"] for s in rest])
                gr = growth_t(l3, s3, None, math.log1p(q["g"]), q["se_inflated"] / (1 + q["g"]), None)
                o["leave_one_patch_out_d5"].append({"left_out": p, "series5": len(rest), "g5": q["g"],
                                                    "g5_se": q["se_inflated"], "g5_ci95": q["ci95"],
                                                    "lambda_ratio": gr["lambda_ratio"],
                                                    "lambda_ratio_ci95": gr["lambda_ratio_ci95_normal"],
                                                    "G": gr["G"], "G_se": gr["se"]})
                r3 = [s for s in ser3 if s["patch"] not in inside[p]]
                q3 = g_stat([s["pr"] for s in r3])
                gr = growth_t(math.log1p(q3["g"]), q3["se_inflated"] / (1 + q3["g"]), None, math.log1p(q["g"]),
                              q["se_inflated"] / (1 + q["g"]), None)
                o["leave_one_region_out"].append({"left_out_d5": p, "left_out_d3": inside[p], "series3": len(r3),
                                                  "series5": len(rest), "g3": q3["g"], "g5": q["g"],
                                                  "lambda_ratio": gr["lambda_ratio"],
                                                  "lambda_ratio_ci95": gr["lambda_ratio_ci95_normal"]})
            o["xz_within_patch"] = {"d3": xz_corr(ser3), "d5": xz_corr(ser5)}
            own, lam_a_own, lam_m_own = own_lambda_ratio(ser3, ser5)
            o["table7"] = {"paired_ratio": o["conditional"]["lambda_ratio"],
                           "paired_ratio_ci95_analytic": o["conditional"]["lambda_ratio_ci95_normal"],
                           "own_ratio_matched": own, "lambda_a_own_matched": lam_a_own,
                           "lambda_ml_own_matched": lam_m_own,
                           "bootstrap": table7_boot(rng, ser3, ser5), "patch_jackknife": table7_jack(ser3, ser5)}
            o["per_series"] = [{"d": d, "patch": s["patch"], "basis": s["basis"], "rounds": s["rounds"],
                                "g_series": s["pr"]["d"] / s["pr"]["eps_ml"], "eps_ml": s["pr"]["eps_ml"],
                                "eps_a_paired": s["pr"]["eps_a_paired"], "eps_a_own": s["own_a"]["eps"]}
                               for d, ser in ((3, ser3), (5, ser5)) for s in ser]
            res["decoders"][a][rule] = o
            c, sc, pc_, pj = o["conditional"], o["series_cluster"], o["patch_cluster"], o["patch_jackknife"]
            rows.append({"decoder": a, "rule": rule, "rounds": " ".join(map(str, rounds)),
                         "series3": c["series3"], "series5": c["series5"], "g3": c["g3"], "g5": c["g5"],
                         "lambda_ratio": c["lambda_ratio"],
                         "ci_conditional": c["lambda_ratio_ci95_normal"],
                         "ci_series_cluster_t": sc.get("lambda_ratio_ci95_t"), "df_series": sc.get("df_welch"),
                         "p_series_t": sc.get("p_two_sided_t"),
                         "ci_patch_cluster_t": pc_.get("lambda_ratio_ci95_t"), "df_patch": pc_.get("df_welch"),
                         "p_patch_t": pc_.get("p_two_sided_t"),
                         "ci_patch_jack_t": pj.get("lambda_ratio_ci95_t"), "df_jack": pj.get("df_welch"),
                         "p_jack_t": pj.get("p_two_sided_t"),
                         "lopo_lambda_range": [min(x["lambda_ratio"] for x in o["leave_one_patch_out_d5"]),
                                               max(x["lambda_ratio"] for x in o["leave_one_patch_out_d5"])],
                         "own_ratio_matched": own})
            print(f"{a} {rule} done ({time.time() - t0:.0f} s)", flush=True)
        res["decoders"][a]["v11_table7"] = {"lambda_own_window": lam.get(f"{a}.rl", {}).get("Lambda"),
                                            "lambda_ml_rule_A": lam["ML.rl"]["Lambda"],
                                            "ratio": (lam[f"{a}.rl"]["Lambda"] / lam["ML.rl"]["Lambda"])
                                            if f"{a}.rl" in lam else None,
                                            "source": "final_round1.json decisions.lambda_3_5 (cited)"}
    res["reproduction"] = repro
    res["run_seconds"] = time.time() - t0
    p = out_path(os.path.join(proj, "results", "reservoir", "willow_revision", "r3_patch.json"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    p2 = out_path(os.path.join(proj, "results", "reservoir", "willow_revision", "r3_patch_rows.csv"))
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("reproduction", json.dumps(repro), flush=True)
    print(f"done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
