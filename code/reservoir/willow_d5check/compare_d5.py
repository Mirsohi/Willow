"""Compare reservoir's d = 5 / Lambda check (d5_check.py's stored outputs) with decoder's final-analysis outputs.

reservoir, 2026-10-10. Run only after d5_check.py's outputs were stored, so that the check stayed independent of
decoder's numbers. decoder's files are read only: results/decoder/willow/analysis/{prereg_results.json, analysis.json,
leaf_table.csv, series_eps.csv, paired_gaps.csv} (made by analyze_willow.py, written 19:30Z).

Compared:
  leaves   error counts and shot numbers, ML(RL) and Libra, every W5 leaf at d = 3 and d = 5
  series   eps, SE and chi2_red per series on W5 (ML(RL) and Libra, d = 3 and 5); the paired per-series gaps
           (paired_gaps.csv, window W5: d = 5 = g_5's series, d = 3 = g_3 on {10, 30, 50})
  pooled   g_5 and its CI (prereg_results.json P1 d5_replication, analysis.json pooled_gaps_W5): which SE convention
           reproduces decoder's CI; pooled error counts
  Lambda   per-distance mean eps; Lambda_ML(RL) (decoder fits d = 3, 5, 7; the pre-registered Lambda is d = 3/5,
           prereg section 5); Lambda_Libra; the paired ratio and its SE under the frozen and the corrected SE of g
  decoder's own d = 5 gap check (optional 4th argument, results/decoder/willow/pr3_planning/d5_gap_check.json, posted
           19:52Z and read only after this check was stored): g_5 and its SEs, rule A (= without the K2-failing
           leaves), per-series gaps, discordant pairs, ladder and extra-rung sums
Usage: python3 -I compare_d5.py DEC_ANALYSIS_DIR MINE_DIR OUT_DIR [DEC_D5_GAP_CHECK_JSON]
       -> OUT_DIR/compare_d5.json (through qrc.outpaths)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qrc.outpaths import in_path, out_path  # noqa: E402

Z90, Z95 = 1.6448536269514722, 1.959963984540054


def rd_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def rdiff(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def maxdiff(pairs):
    """Largest relative difference over (mine, decoder) pairs, and the number of pairs."""
    d = [rdiff(a, b) for a, b in pairs]
    return {"pairs": len(d), "max_rel_diff": max(d) if d else None, "identical": sum(x == 0.0 for x in d)}


def main():
    dec, mine, out_dir = sys.argv[1:4]
    M = json.load(open(in_path(os.path.join(mine, "d5_check.json"))))
    ML = rd_csv(in_path(os.path.join(mine, "d5_leaves.csv")))
    MS = rd_csv(in_path(os.path.join(mine, "d5_series.csv")))
    P = json.load(open(os.path.join(dec, "prereg_results.json")))
    A = json.load(open(os.path.join(dec, "analysis.json")))
    LT = rd_csv(os.path.join(dec, "leaf_table.csv"))
    SE = rd_csv(os.path.join(dec, "series_eps.csv"))
    PG = rd_csv(os.path.join(dec, "paired_gaps.csv"))
    res = {"label": "reservoir, 2026-10-10: comparison of willow_d5check/ (independent) with decoder's analysis/ "
                    "(written 19:30Z; analysis.json 'made' " + str(A.get("made")) + "). Measured.",
           "decoder_files": {k: os.path.join(dec, k) for k in ("prereg_results.json", "analysis.json",
                                                              "leaf_table.csv", "series_eps.csv",
                                                              "paired_gaps.csv")}}

    # leaves
    lt = {}
    for r in LT:
        if r["entity"] in ("ML.rl", "libra.rl") and r["d"] in ("3", "5"):
            lt[(r["d"], r["patch"], r["basis"], int(r["rounds"]), r["entity"])] = (int(r["n"]), int(r["errors"]))
    leaves = {"ML.rl": {"compared": 0, "equal": 0, "mismatch": []}, "libra.rl": {"compared": 0, "equal": 0,
                                                                                "mismatch": []}}
    mine_keys = set()
    for r in ML:
        d, p, b, rr = r["d"], r["patch"], r["basis"], int(r["r"])
        for ent, kcol in (("ML.rl", "k_ml"), ("libra.rl", "k_libra")):
            if ent == "libra.rl" and r["has_libra"] != "True":
                continue
            # d = 3 Libra counts in d5_leaves.csv are on all 50,000 shots, d = 5 on the ML subset (n)
            mine_n = int(r["n"])
            key = (d, p, b, rr, ent)
            mine_keys.add(key)
            got = lt.get(key)
            leaves[ent]["compared"] += 1
            if got is not None and got == (mine_n, int(r[kcol])):
                leaves[ent]["equal"] += 1
            else:
                leaves[ent]["mismatch"].append({"leaf": f"d{d}/{p}/{b}/r{rr}", "mine": [mine_n, int(r[kcol])],
                                                "decoder": got})
    in_window = {k for k in lt if (k[0] == "5") or (k[0] == "3" and k[3] in (10, 13, 30, 50))}
    leaves["decoder_W5_rows_not_in_mine"] = sorted("/".join(map(str, k)) for k in in_window - mine_keys)
    res["leaves"] = leaves

    # series eps on W5
    se = {}
    for r in SE:
        if r["window"] != "W5":
            continue
        d, p, b, ent = r["series_entity"].split("|")
        se[(d[1:], p, b, ent)] = r
    ser = {}
    for tag, d, ent in (("eps_d3_ml_rl", "3", "ML.rl"), ("eps_d3_libra", "3", "libra.rl"),
                        ("eps_d5_ml_rl", "5", "ML.rl"), ("eps_d5_libra", "5", "libra.rl")):
        rows = [r for r in MS if r["set"] == tag]
        pe, ps, pc, miss, rounds_eq = [], [], [], [], 0
        for r in rows:
            o = se.get((d, r["patch"], r["basis"], ent))
            if o is None:
                miss.append(f"{r['patch']}/{r['basis']}")
                continue
            pe.append((r["eps"], o["eps"]))
            ps.append((r["se"], o["se_eps"]))
            if r["chi2_red"] not in ("", "None") and o["chi2_red"] not in ("", "None"):
                pc.append((r["chi2_red"], o["chi2_red"]))
            rounds_eq += json.loads(r["rounds"]) == [int(x) for x in o["rounds"].split()]
        ser[tag] = {"series": len(rows), "missing_in_decoder": miss, "rounds_equal": rounds_eq,
                    "eps": maxdiff(pe), "se": maxdiff(ps), "chi2_red": maxdiff(pc)}
    res["series_eps_W5"] = ser

    # paired per-series gaps, window W5
    pg = {}
    for r in PG:
        if r["window"] != "W5":
            continue
        d, p, b, pair = r["series_pair"].split("|")
        if pair == "libra.rl vs ML.rl":
            pg[(d[1:], p, b)] = r
    pairs = {}
    for tag, d in (("g5_all", "5"), ("g3_on_10_30_50", "3")):
        rows = [r for r in MS if r["set"] == tag]
        pr, pa, pm, pc, miss = [], [], [], [], []
        for r in rows:
            o = pg.get((d, r["patch"], r["basis"]))
            if o is None:
                miss.append(f"{r['patch']}/{r['basis']}")
                continue
            pr.append((r["g_series"], o["rel"]))
            pa.append((r["eps_libra_paired"], o["eps_A"]))
            pm.append((r["eps_ml"], o["eps_ML"]))
            if r["chi2_red_z"] not in ("", "None"):
                pc.append((r["chi2_red_z"], o["chi2_red"]))
        pairs[tag] = {"series": len(rows), "missing_in_decoder": miss, "g_series": maxdiff(pr),
                      "eps_libra_paired": maxdiff(pa), "eps_ml": maxdiff(pm), "chi2_red_z": maxdiff(pc)}
    res["paired_series_W5"] = pairs

    # pooled g_5 and its CI
    g5 = M["g5"]
    rep = P["P1_libra_vs_ML_rl_d3"]["d5_replication"]
    pw = A["pooled_gaps_W5"]["d5|libra.rl vs ML.rl"]
    half90 = (rep["ci90"][1] - rep["ci90"][0]) / 2
    res["g5"] = {
        "mine": g5["g"], "decoder_prereg": rep["rel"], "decoder_pooled_gaps_W5": pw["rel"],
        "rel_diff": rdiff(g5["g"], rep["rel"]),
        "decoder_se_rel": pw["se_rel"],
        "decoder_ci90_half_width_over_1.645": half90 / 1.645,
        "mine_se_frozen_formula_inflated": g5["se_frozen_formula_inflated"],
        "mine_se_covariance_corrected_inflated": g5["se_inflated"],
        "decoder_se_equals_frozen_formula": rdiff(pw["se_rel"], g5["se_frozen_formula_inflated"]) < 1e-9,
        "decoder_ci_quantiles": "1.645 and 1.96 (half-widths / se_rel)",
        "decoder_ci90": rep["ci90"], "decoder_ci95": pw["ci95"], "decoder_z": pw["z"], "decoder_p": pw["p"],
        "corrected_ci90": g5["ci90"], "corrected_ci95": g5["ci95"], "corrected_z": g5["z"],
        "corrected_p": g5["p_two_sided"],
        "replicated_decoder": rep["replicated"], "replicated_corrected": M["replication"]["replicated"],
        "note": "prereg section 9 entry 7 (Scope): the covariance correction of the SE applies to every g of section "
                "4, the d = 5 replication included, at the final analysis. decoder's CI uses the frozen formula.",
    }
    g3w = A["pooled_gaps_W5"]["d3|libra.rl vs ML.rl"]
    res["g3_W5"] = {"mine_on_10_30_50": M["g3_on_10_30_50"]["g"], "decoder_W5": g3w["rel"],
                    "rel_diff": rdiff(M["g3_on_10_30_50"]["g"], g3w["rel"]),
                    "decoder_se_rel": g3w["se_rel"],
                    "mine_se_frozen_formula_inflated": M["g3_on_10_30_50"]["se_frozen_formula_inflated"],
                    "mine_se_covariance_corrected_inflated": M["g3_on_10_30_50"]["se_inflated"]}
    pc = A["pooled_counts"]["d5|libra.rl vs ML.rl"]
    pm = M["pooled_d5"]
    res["pooled_counts_d5"] = {
        "decoder": {k: pc[k] for k in ("leaves", "n", "errors_A", "errors_B", "rel", "se_rel", "ci95")},
        "mine": {k: pm[k] for k in ("leaves", "shots", "k_libra", "k_ml", "ratio", "se", "ci95")},
        "counts_equal": (pc["leaves"], pc["n"], pc["errors_A"], pc["errors_B"]) ==
                        (pm["leaves"], pm["shots"], pm["k_libra"], pm["k_ml"]),
        "rel_diff_ratio": rdiff(pc["rel"], pm["ratio"]), "rel_diff_se": rdiff(pc["se_rel"], pm["se"])}

    # Lambda
    la = A["lambda"]
    ml, lb = M["lambda"]["ml_rl"], M["lambda"]["libra"]
    e = la["ML.rl"]["eps"]
    # decoder's Lambda_ML(RL) over d = 3, 5, 7: WLS of ln eps_d on d, weights (eps/se)^2, Lambda = exp(-2 slope)
    pts = [(int(d), math.log(v[0]), (v[1] / v[0]) ** -2) for d, v in e.items()]
    sw = sum(w for _, _, w in pts)
    xb = sum(w * x for x, _, w in pts) / sw
    yb = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - xb) ** 2 for x, _, w in pts)
    slope = sum(w * (x - xb) * (y - yb) for x, y, w in pts) / sxx
    lam357 = math.exp(-2 * slope)
    se357 = 2 * lam357 / math.sqrt(sxx)
    lam35_dec = e["3"][0] / e["5"][0]
    res["lambda"] = {
        "ML_rl": {"decoder": la["ML.rl"]["Lambda"], "decoder_se": la["ML.rl"]["se"],
                  "decoder_distances": la["ML.rl"]["distances"],
                  "refit_d357_from_decoder_eps": lam357, "refit_d357_se_no_inflation": se357,
                  "refit_rel_diff": rdiff(lam357, la["ML.rl"]["Lambda"]),
                  "Lambda_3_5_from_decoder_eps": lam35_dec, "mine_Lambda_3_5": ml["Lambda"], "mine_se": ml["se"],
                  "rel_diff_3_5": rdiff(lam35_dec, ml["Lambda"]),
                  "eps_d3": maxdiff([(ml["eps_mean_d3"], e["3"][0]), (ml["se_d3"], e["3"][1])]),
                  "eps_d5": maxdiff([(ml["eps_mean_d5"], e["5"][0]), (ml["se_d5"], e["5"][1])]),
                  "d7_eps_decoder": e.get("7"),
                  "note": "prereg section 5: Lambda_3/5 on W5; d = 7 is exploratory. decoder's table value includes "
                          "d = 7 (2 series, eps SE 70% of eps), which moves Lambda_ML by +0.003."},
        "libra": {"decoder": la["libra.rl"]["Lambda"], "decoder_se": la["libra.rl"]["se"],
                  "mine": lb["Lambda"], "mine_se": lb["se"], "rel_diff": rdiff(la["libra.rl"]["Lambda"], lb["Lambda"]),
                  "rel_diff_se": rdiff(la["libra.rl"]["se"], lb["se"])},
    }
    lp = A["lambda_paired_libra_over_ML_rl"]
    pr = M["lambda"]["paired_ratio_libra_over_ml"]
    g3, g5v = M["g3_on_10_30_50"], M["g5"]
    se_ln_frozen = math.hypot(g3["se_frozen_formula_inflated"] / (1 + g3["g"]),
                              g5v["se_frozen_formula_inflated"] / (1 + g5v["g"]))
    se_ln_corr = math.hypot(g3["se_inflated"] / (1 + g3["g"]), g5v["se_inflated"] / (1 + g5v["g"]))
    ln = math.log(pr["ratio"])
    res["lambda"]["paired_ratio"] = {
        "decoder": lp["ratio"], "mine": pr["ratio"], "rel_diff": rdiff(lp["ratio"], pr["ratio"]),
        "decoder_se_ln": lp["se_ln"], "decoder_ci95": lp["ci95"],
        "se_ln_from_frozen_formula_ses": se_ln_frozen,
        "decoder_se_ln_equals_frozen": rdiff(lp["se_ln"], se_ln_frozen) < 1e-9,
        "se_ln_from_corrected_ses": se_ln_corr,
        "ci95_log_scale_corrected": [math.exp(ln - Z95 * se_ln_corr), math.exp(ln + Z95 * se_ln_corr)],
        "mine_delta_se_corrected": pr["se"], "mine_ci95_delta_corrected": pr["ci95"]}
    # optional: decoder's own d = 5 gap check (pr3_planning/d5_gap_check.json, posted 19:52Z; read after the store)
    if len(sys.argv) > 4:
        G = json.load(open(sys.argv[4]))
        sa = M["sensitivity"]["without_k2_failing"]["g5"]
        mser = {f"{r['patch']}/{r['basis']}": r for r in MS if r["set"] == "g5_all"}
        tr = M["flags"]["truncation"]
        res["decoder_d5_gap_check"] = {
            "file": sys.argv[4], "made": G.get("made"),
            "base_g": maxdiff([(g5["g"], G["base"]["g"])]),
            "base_se_cov_vs_mine_corrected": maxdiff([(g5["se_inflated"], G["base"]["se_cov"])]),
            "base_se_frozen": maxdiff([(g5["se_frozen_formula_inflated"], G["base"]["se_frozen"])]),
            "ruleA_g_vs_mine_without_k2_failing": maxdiff([(sa["g"], G["excl_now_ruleA"]["g"])]),
            "ruleA_se_cov": maxdiff([(sa["se_inflated"], G["excl_now_ruleA"]["se_cov"])]),
            "ruleB_g_decoder": G["excl_now_ruleB"]["g"], "ruleB_se_cov_decoder": G["excl_now_ruleB"]["se_cov"],
            "per_series_rel": maxdiff([(mser[k]["g_series"], v["rel"]) for k, v in G["per_series"].items()]),
            "discordant_counts_equal": (G["discordance"]["pooled"]["n10"], G["discordance"]["pooled"]["n01"]) ==
                                       (M["pooled_d5"]["libra_wrong_ml_right"], M["pooled_d5"]["libra_right_ml_wrong"]),
            "ladder_sums_equal": (G["ladder_tau1_to_tau2"]["sum_errors_tau1"], G["ladder_tau1_to_tau2"]["sum_errors_final"],
                                  G["ladder_tau1_to_tau2"]["sum_decisions_changed"]) ==
                                 (tr["ladder_tau1_to_final_errors"]["sum_k_tau1"],
                                  tr["ladder_tau1_to_final_errors"]["sum_k_final"],
                                  tr["ladder_tau1_to_final_errors"]["decisions_changed"]),
            "extra_rung_sums_equal": (G["extra_rung_tau2_to_tau3"]["leaves"], G["extra_rung_tau2_to_tau3"]["sum_k2"],
                                      G["extra_rung_tau2_to_tau3"]["sum_k3"]) ==
                                     (tr["extra_rung_tau2_to_tau3"]["leaves"],
                                      tr["extra_rung_tau2_to_tau3"]["sum_k_ladder_final"],
                                      tr["extra_rung_tau2_to_tau3"]["sum_k_tau3"]),
            "leave_one_leaf_out_range_decoder": [G["leave_one_out"]["min"], G["leave_one_out"]["max"]],
            "ruleA_more_failures_range_decoder": {k: [v["g_min"], v["g_max"]]
                                                  for k, v in G["scenarios_more_failures_ruleA"].items()}}
    p = out_path(os.path.join(out_dir, "compare_d5.json"))
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    brief = {k: res[k] for k in ("g5", "pooled_counts_d5", "lambda", "decoder_d5_gap_check") if k in res}
    brief["leaves"] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "mismatch"}
                           | {"n_mismatch": len(v["mismatch"])}) for k, v in res["leaves"].items()}
    brief["series_eps_W5"] = res["series_eps_W5"]
    brief["paired_series_W5"] = res["paired_series_W5"]
    print(json.dumps(brief, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    main()
