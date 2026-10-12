"""Comparison of reservoir's independent Willow P1 check with decoder's milestone-2 analysis.

Order of work (UTC, 2026-10-09, from the session transcript). The brief (19:59Z) quoted decoder's posted numbers.
willow_check.py wrote leaves.csv, series.csv and willow_check.json at 20:07Z, and their md5s were recorded at 20:08:32Z
(55abda3a..., 1d6afcfe..., 4fd385a2...; unchanged since). bootstrap_g.json followed at 20:09Z. decoder's analysis code
and results/decoder/willow/analysis_m2/ were first read at 20:10:26Z, board/decoder.md after that. variants.py,
ties.py and this script came later; none of them changes a number of willow_check.py.

Inputs (read only):
  OUT_DIR  results/reservoir/willow_check/: leaves.csv, series.csv, willow_check.json, variants.json, bootstrap_g.json
  DEC_DIR  results/decoder/willow/analysis_m2/: leaf_table.csv, series_eps.csv, paired_gaps.csv, prereg_results.json,
           analysis.json
Output: OUT_DIR/comparison.json through qrc.outpaths (a re-run writes to results/reservoir/rebuild/willow_check/
        unless QRC_OVERWRITE=1).
Usage: nice -n 19 python3 -I compare_decoder.py OUT_DIR DEC_DIR
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.dont_write_bytecode = True  # no __pycache__ in the shared code folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from willow_check import REPORTED, in_path, out_path  # noqa: E402


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rdiff(a, b):
    return abs(a - b) / abs(b)


def main(out_dir, dec_dir):
    L = rows(in_path(os.path.join(out_dir, "leaves.csv")))
    S = rows(in_path(os.path.join(out_dir, "series.csv")))
    W = json.load(open(in_path(os.path.join(out_dir, "willow_check.json"))))
    V = json.load(open(in_path(os.path.join(out_dir, "variants.json"))))
    B = json.load(open(in_path(os.path.join(out_dir, "bootstrap_g.json"))))
    lt = [r for r in rows(os.path.join(dec_dir, "leaf_table.csv")) if r["d"] == "3"]
    se_rows = rows(os.path.join(dec_dir, "series_eps.csv"))
    pg = rows(os.path.join(dec_dir, "paired_gaps.csv"))
    pr = json.load(open(os.path.join(dec_dir, "prereg_results.json")))
    an = json.load(open(os.path.join(dec_dir, "analysis.json")))
    out = {}

    # (a) per-leaf error counts and shots
    dec = {(r["patch"], r["basis"], int(r["rounds"]), r["entity"]): (int(r["errors"]), int(r["n"])) for r in lt}
    diff, missing = Counter(), Counter()
    for x in L:
        for ent, col in (("ML.rl", "k_ml"), ("libra.rl", "k_libra")):
            v = dec.get((x["patch"], x["basis"], int(x["r"]), ent))
            if v is None:
                missing[ent] += 1
            elif v != (int(x[col]), int(x["n"])):
                diff[ent] += 1
    out["a_leaf_counts"] = {"leaves": len(L), "differences": {e: diff[e] for e in ("ML.rl", "libra.rl")},
                            "missing_in_decoder_table": {e: missing[e] for e in ("ML.rl", "libra.rl")}}

    # (b) per-series Libra eps (its 13 round counts) against series_eps.csv
    ser = {tuple(r["series_entity"].split("|")): r for r in se_rows if r["window"] == "primary"}
    mx, npts = Counter(), set()
    for s in S:
        r = ser[("d3", s["patch"], s["basis"], "libra.rl")]
        npts.add(int(r["npts"]))
        mx["eps"] = max(mx["eps"], rdiff(float(s["eps_libra"]), float(r["eps"])))
        mx["se"] = max(mx["se"], rdiff(float(s["se_libra"]), float(r["se_eps"])))
        mx["chi2_red"] = max(mx["chi2_red"], rdiff(float(s["chi2_red_libra"]), float(r["chi2_red"])))
    out["b_series_libra_eps"] = {"series": len(S), "decoder_npts": sorted(npts),
                                 "max_relative_difference": {k: mx[k] for k in ("eps", "se", "chi2_red")}}

    # (c) per-series paired gaps against paired_gaps.csv (primary window: Libra's 13 round counts), and decoder's SE
    # convention rebuilt from reservoir's fits: se_delta = e^(bM + bz) se_z / 2 (z-slope term only),
    # se_eps_ML = e^bM se_M / 2, rel se = sqrt((se_delta / eM)^2 + (delta se_eps_ML / eM^2)^2), both fits inflated;
    # pooled the same way with no covariance between the mean gap and the mean eps_ML.
    pgd = {tuple(r["series_pair"].split(" vs ")[0].split("|")): r for r in pg
           if r["window"] == "primary" and r["series_pair"].endswith(" vs ML.rl")}
    mx = Counter()
    se_d, se_m, d, em, rel = [], [], [], [], []
    for s in S:
        r = pgd[("d3", s["patch"], s["basis"], "libra.rl")]
        bM, bz = float(s["paired_slope_ml"]), float(s["paired_slope_z"])
        sd = math.exp(bM + bz) * math.sqrt(float(s["paired_var_slope_z"]) * max(1.0, float(s["paired_chi2_red_z"]))) / 2
        sm = math.sqrt(float(s["paired_var_eps_ml_inflated"]))
        dd, ee = float(s["paired_d"]), float(s["paired_eps_ml"])
        se_d.append(sd); se_m.append(sm); d.append(dd); em.append(ee); rel.append(dd / ee)
        mx["rel_abs"] = max(mx["rel_abs"], abs(dd / ee - float(r["rel"])))
        mx["eps_ML"] = max(mx["eps_ML"], rdiff(ee, float(r["eps_ML"])))
        mx["eps_A"] = max(mx["eps_A"], rdiff(float(s["paired_eps_a_paired"]), float(r["eps_A"])))
        mx["chi2_red_z"] = max(mx["chi2_red_z"], rdiff(float(s["paired_chi2_red_z"]), float(r["chi2_red"])))
        mx["se_rel_decoder_convention"] = max(mx["se_rel_decoder_convention"],
                                              rdiff(math.sqrt((sd / ee) ** 2 + (dd * sm / ee ** 2) ** 2), float(r["se_rel"])))
    n = len(S)
    D, M = float(np.mean(d)), float(np.mean(em))
    sD, sM = math.sqrt(sum(x * x for x in se_d)) / n, math.sqrt(sum(x * x for x in se_m)) / n
    se_dec_conv = math.sqrt((sD / M) ** 2 + (D * sM / M ** 2) ** 2)
    out["c_series_paired_gaps"] = {
        "series": n, "series_with_positive_gap": int(sum(x > 0 for x in rel)), "rel_range": [min(rel), max(rel)],
        "max_difference": {"rel_abs": mx["rel_abs"], **{k: mx[k] for k in ("eps_ML", "eps_A", "chi2_red_z",
                                                                               "se_rel_decoder_convention")}},
        "note": "max_difference of rel is absolute; the others are relative"}

    # headline numbers: reservoir (M), decoder's analysis_m2 files (M by decoder), decoder's board post (R)
    g, p1 = W["g"], pr["P1_libra_vs_ML_rl_d3"]["pooled"]
    g1 = pr["G1_reproduce_libra"]["by_distance"]["3"]["r>=10"]
    p5 = pr["P5_ML_rl_vs_NN"]["pooled"]["3"]
    me, pc = an["mean_eps"], an["pooled_counts"]["d3|libra.rl vs ML.rl"]
    lib = W["eps_mean"]["libra"]
    v13, v14 = V["eps_ml_rl_13_rounds_libra_window"], V["eps_ml_rl_14_rounds_r_ge_10"]
    pool = W["pooled_ratio"]
    out["headline"] = {
        "libra_mean_eps_13_rounds": {"reservoir": [lib["mean"], lib["se"]], "decoder_files": [g1["ours"], g1["se"]],
                                     "decoder_posted": REPORTED["decoder_libra_eps"],
                                     "table_S1": REPORTED["table_S1_libra_eps"],
                                     "z_vs_table_S1": {"reservoir": lib["z_vs_table_S1_libra_eps"], "decoder_files": g1["z"]}},
        "ml_rl_mean_eps_13_rounds_P1_denominator": {"reservoir": [v13["mean"], v13["se"]],
                                                    "decoder_files_P1_and_P5": [p1["mean_eps_ML"], p5["se"]]},
        "ml_rl_mean_eps_14_rounds_own_window": {"reservoir": [v14["mean"], v14["se"]],
                                                "decoder_files": [me["d3|ML.rl"]["eps"], me["d3|ML.rl"]["se"]],
                                                "decoder_posted": REPORTED["decoder_ml_rl_eps"]},
        "mean_gap_delta": {"reservoir": g["mean_d"], "decoder_files": p1["mean_delta"]},
        "g": {"reservoir": g["g"], "decoder_files": p1["rel"], "decoder_posted": REPORTED["decoder_g"][0]},
        "se_g": {"decoder_files": p1["se_rel"], "decoder_posted": REPORTED["decoder_g"][1],
                 "decoder_convention_rebuilt_from_reservoir_fits": se_dec_conv,
                 "reservoir_full_delta_method_inflated": g["se_inflated"],
                 "reservoir_full_delta_method_no_inflation": g["se_no_inflation"],
                 "reservoir_shot_bootstrap_sd": B["bootstrap_sd"]["g"]},
        "z_g": {"decoder_files": p1["z"], "reservoir_full_delta_method_inflated": g["g"] / g["se_inflated"]},
        "ci95_g": {"decoder_files": p1["ci95"], "reservoir_bootstrap_percentile": B["g_percentile_95"],
                   "reservoir_full_delta_method_inflated": [g["g"] - 1.96 * g["se_inflated"], g["g"] + 1.96 * g["se_inflated"]]},
        "pooled_count_ratio": {"reservoir": [pool["ratio"], pool["se"]], "decoder_files": [pc["rel"], pc["se_rel"]],
                               "sum_k_libra": [pool["sum_k_libra"], pc["errors_A"]], "sum_k_ml": [pool["sum_k_ml"], pc["errors_B"]],
                               "reservoir_mcnemar_z": pool["mcnemar_z"]},
    }

    # leaves with E >= 1/2 at d = 3 (no rule in prereg section 4; decoder's analysis drops them from a fit)
    half = Counter(r["entity"] for r in lt if 2 * int(r["errors"]) >= int(r["n"]))
    out["leaves_with_E_ge_half_d3"] = {"by_entity": dict(sorted(half.items())),
                                       "ML.rl": half.get("ML.rl", 0), "libra.rl": half.get("libra.rl", 0)}

    h = out["headline"]
    out["summary"] = {
        "counts_identical": out["a_leaf_counts"]["differences"] == {"ML.rl": 0, "libra.rl": 0}
                            and out["a_leaf_counts"]["missing_in_decoder_table"] == {"ML.rl": 0, "libra.rl": 0},
        "libra_eps_relative_difference": rdiff(lib["mean"], g1["ours"]),
        "g_relative_difference": rdiff(g["g"], p1["rel"]),
        "ml_eps_14_rounds_relative_difference": rdiff(v14["mean"], me["d3|ML.rl"]["eps"]),
        "ml_eps_13_rounds_relative_difference": rdiff(v13["mean"], p1["mean_eps_ML"]),
        "se_g_decoder_convention_relative_difference": rdiff(se_dec_conv, p1["se_rel"]),
        "pooled_ratio_relative_difference": rdiff(pool["ratio"], pc["rel"]),
        "se_g_full_over_decoder": h["se_g"]["reservoir_full_delta_method_inflated"] / p1["se_rel"],
    }
    with open(out_path(os.path.join(out_dir, "comparison.json")), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
