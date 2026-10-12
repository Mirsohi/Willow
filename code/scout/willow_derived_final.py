"""Derived numbers for the Willow manuscript from round 1's final analysis (scout, 2026-10-11).

The final-analysis counterpart of willow_derived.py (which worked on the frozen analysis of 2026-10-10 19:30Z, interim
at d = 5, and whose output results/scout/willow_derived_d3d5.json is kept as that record). Descriptive arithmetic on
posted outputs only; no decoding, no new statistic of the pre-registration. Main's decisions 4 and 8 (paper.md): the
other decoders' paired Lambda ratios from final_round1.json, labelled descriptive, compared with reservoir's
independent matched table.

Inputs (read only; folder results/decoder/willow/ unless stated):
  final_round1/final_round1.json                 decoder's final round-1 analysis, rule A (made 2026-10-11 17:30:14Z)
  final_round1/analysis/analysis.json            the frozen code's output on the rule-A view (17:13:19Z)
  final_round1/analysis/calibration_leaves.csv   R per leaf on the rule-A view
  supplementary/p1_se_covariance.json            entry 7's SE of P1 (absolute delta)
  results/reservoir/willow_gapmech/final_k2/gap_table.json   reservoir's matched table under K2's final tally
Reported values (Willow paper, Table S1 [V]) are typed in below and labelled.
Usage: python3 -I willow_derived_final.py <experimental-root>/ <out.json>
"""
import csv
import json
import math
import sys

root, out = sys.argv[1], sys.argv[2]
W = root + "results/decoder/willow/"
FR = json.load(open(W + "final_round1/final_round1.json"))
FA = json.load(open(W + "final_round1/analysis/analysis.json"))
SEC = json.load(open(W + "supplementary/p1_se_covariance.json"))
GT = json.load(open(root + "results/reservoir/willow_gapmech/final_k2/gap_table.json"))
Z95 = 1.959963984540054
E7 = FR["entry7_se"]
DA = FR["decisions"]["d5_rules"]
res = {"label": "scout, descriptive arithmetic on decoder's final round-1 analysis (rule A; entry 7's SE) and "
                "reservoir's matched table under K2's final tally", "made_from": {"final_round1.json": FR["made"],
                                                                                "k2": FR["k2"]["verdicts"]}}


def ratio_ci(g3, s3, g5, s5):
    r = (1 + g3) / (1 + g5)
    se = math.sqrt((s3 / (1 + g3)) ** 2 + (s5 / (1 + g5)) ** 2)
    return r, se, [r * math.exp(-Z95 * se), r * math.exp(Z95 * se)]


# ---- 1. every decoder's paired gap to ML(RL) on W5 at d = 3 and 5, entry 7's SE, and the paired Lambda ratio ----
gaps = {}
for a in ["libra", "tesseract", "harmony", "corrmatch", "corr_mwpm", "mwpm", "beliefmatch"]:
    row = {}
    for d in ("3", "5"):
        k = "pooled_gaps_W5:d%s|%s.rl vs ML.rl" % (d, a)
        if k in E7:
            e = E7[k]
            row.update({"g" + d: e["g"], "se" + d: e["se"], "ci95_" + d: e["ci95"], "n_series_" + d: e["n_series"]})
    if "g5" in row:
        r, se, ci = ratio_ci(row["g3"], row["se3"], row["g5"], row["se5"])
        row.update(growth_g5_over_g3=row["g5"] / row["g3"], paired_lambda_ratio=r, paired_lambda_ratio_se_ln=se,
                   paired_lambda_ratio_ci95_log=ci, z_ln_ratio=math.log(r) / se, g5_minus_g3=row["g5"] - row["g3"],
                   g5_minus_g3_se=math.hypot(row["se3"], row["se5"]))
        G = GT["table"][a]["matched|ruleA"]
        row["reservoir_matched"] = {"rounds": G["rounds"], "g3": G["g3"]["g"], "g5": G["g5"]["g"],
                                    "lambda_ratio": G["lambda_ratio"], "lambda_ratio_ci95": G["lambda_ratio_ci95"]}
        row["difference_vs_reservoir"] = {"g5": row["g5"] - G["g5"]["g"], "g3": row["g3"] - G["g3"]["g"],
                                          "lambda_ratio": r - G["lambda_ratio"]}
    gaps[a] = row
gaps["note"] = ("W5 at both distances (Libra r in {10, 30, 50}); Tesseract's d = 3 W5 includes r = 50 and its d = 5 "
                "rounds are {10, 13, 30}, so its ratio here mixes windows; reservoir's matched table uses {10, 13, 30} "
                "at both distances for it")
res["gaps_W5_entry7"] = gaps

# ---- 2. Lambda_3/5 per decoder (decoder's final table: two points, unpaired) ----
lam = {k: {"Lambda_3_5": v["Lambda"], "se": v["se"], "eps3": v["eps"]["3"], "eps5": v["eps"]["5"],
           "n_series": v["n_series"]} for k, v in FR["decisions"]["lambda_3_5"].items()}
t3, t5 = FA["mean_eps"]["d3|tesseract.rl"], FA["mean_eps"]["d5|tesseract.rl"]
lam["tesseract_windows_differ"] = {"eps3": t3["eps"], "se3": t3["se"], "eps5": t5["eps"], "se5": t5["se"],
                                   "Lambda_3_5": t3["eps"] / t5["eps"],
                                   "note": "d = 3 on r in {10, 13, 30, 50}, d = 5 on r in {10, 13, 30}; not comparable"}
lam["frozen_fit_d3_5_7_exploratory"] = FR["decisions"]["lambda_frozen_fit_exploratory"]
res["lambda_3_5"] = lam

# ---- 3. Libra (rule A, B and no exclusion) ----
lib = {}
for rule in ("A", "B", "none"):
    pr = DA[rule]["paired_ratio_libra_over_ML_rl"]
    g5 = DA[rule]["g5"]
    g3, s3 = E7["pooled_gaps_W5:d3|libra.rl vs ML.rl"]["g"], E7["pooled_gaps_W5:d3|libra.rl vs ML.rl"]["se"]
    lib[rule] = {"g3_W5": g3, "g3_se": s3, "g5": g5["g"], "g5_se": g5["se"], "g5_ci95": g5["ci95"], "g5_ci90": g5["ci90"],
                 "n_series_5": g5["n_series"], "ratio": pr["ratio"], "ratio_ci95_log": pr["ci95"],
                 "z_ln_ratio": pr["ln_ratio"] / pr["se_ln"], "one_minus_ratio": 1 - pr["ratio"],
                 "growth_g5_over_g3": g5["g"] / g3, "g5_minus_g3": g5["g"] - g3,
                 "g5_minus_g3_se": math.hypot(s3, g5["se"]),
                 "Lambda_ML_rl": DA[rule]["Lambda_ML_rl_3_5"]["Lambda"], "Lambda_ML_rl_se": DA[rule]["Lambda_ML_rl_3_5"]["se"],
                 "Lambda_ML_rl_ci95": [DA[rule]["Lambda_ML_rl_3_5"]["Lambda"] - Z95 * DA[rule]["Lambda_ML_rl_3_5"]["se"],
                                       DA[rule]["Lambda_ML_rl_3_5"]["Lambda"] + Z95 * DA[rule]["Lambda_ML_rl_3_5"]["se"]]}
pg5 = FA["pooled_gaps_W5"]["d5|libra.rl vs ML.rl"]
lib["A"]["abs_d5"] = {"mean_delta": pg5["mean_delta"], "mean_eps_ML": pg5["mean_eps_ML"],
                      "se_entry7": DA["A"]["g5"]["se"] * pg5["mean_eps_ML"],
                      "note": "absolute per-cycle gap; SE = entry 7's relative SE times ML's paired eps (scout)"}
lib["A"]["unpaired_ratio_Lambda_libra_over_ML"] = (FR["decisions"]["lambda_3_5"]["libra.rl"]["Lambda"]
                                                   / FR["decisions"]["lambda_3_5"]["ML.rl"]["Lambda"])
lib["A"]["H7_threshold_c"] = math.log(1.11 / (1 + DA["A"]["g5"]["g"]))
lib["A"]["libra_over_corrmatch"] = {"paired_through_ML": gaps["libra"]["paired_lambda_ratio"]
                                    / gaps["corrmatch"]["paired_lambda_ratio"],
                                    "Lambda_column": FR["decisions"]["lambda_3_5"]["libra.rl"]["Lambda"]
                                    / FR["decisions"]["lambda_3_5"]["corrmatch.rl"]["Lambda"]}
res["libra"] = lib

# ---- 4. ML(RL) against the reported NN, d = 3 and d = 5, and the ratio of Lambdas (approximate, unpaired) ----
P5_3 = FA["prereg"]["P5_ML_rl_vs_NN"]["pooled"]["3"]
h3, sh3 = P5_3["rel"], P5_3["se_rel"]
nn = {"h3": h3, "h3_se": sh3}
for rule in ("A", "B", "none"):
    p = DA[rule]["P5_d5"]
    r, se, ci = ratio_ci(h3, sh3, p["rel"], p["se_rel"])
    nn[rule] = {"h5": p["rel"], "h5_se": p["se_rel"], "h5_ci95": p["ci95"], "p": p["p"], "n_series": p["n_series"],
                "eps_ML_rl": p["eps_ML_rl"], "eps_ML_rl_se": p["se"], "rel_se_eps_ML": p["se"] / p["eps_ML_rl"],
                "se_ratio_to_d3": p["se_rel"] / sh3,
                "Lambda_ML_over_Lambda_NN": r, "ci95_log": ci}
nn["rel_se_eps_ML_d3"] = P5_3["se"] / P5_3["eps_ML_rl"]
res["ml_vs_nn"] = nn

# ---- 5. inference share of Libra's reported gap to the NN (approximate, unpaired) ----
reported = {"eps_libra": {"3": [7.12e-3, 0.06e-3], "5": [3.49e-3, 0.04e-3]},
            "eps_nn": {"3": [6.50e-3, 0.06e-3], "5": [3.03e-3, 0.03e-3]},
            "source": "Willow paper (arXiv:2408.13687v1) Table S1, reported [V]"}
P1 = FA["prereg"]["P1_libra_vs_ML_rl_d3"]["pooled"]
share = {}
for d, num, nse in (("3", P1["mean_delta"], SEC["full_delta_method_inflated"]["se_mean_delta"]),
                    ("5", lib["A"]["abs_d5"]["mean_delta"], lib["A"]["abs_d5"]["se_entry7"])):
    den = reported["eps_libra"][d][0] - reported["eps_nn"][d][0]
    dse = math.hypot(reported["eps_libra"][d][1], reported["eps_nn"][d][1])
    s = num / den
    share[d] = {"numerator_libra_minus_ML_paired": num, "numerator_se": nse, "denominator_libra_minus_nn_reported": den,
                "denominator_se": dse, "share": s, "share_se": s * math.hypot(nse / num, dse / den)}
share["note"] = ("approximate: the numerator is our paired Libra - ML(RL) difference (d = 3: Libra's 13 round counts, "
                 "r >= 10; d = 5: r in {10, 30, 50}, rule A); the denominator is Table S1's Libra - NN (reported; "
                 "window not stated); unpaired")
share["reported"] = reported
res["inference_share_of_libra_nn_gap"] = share

# ---- 6. calibration R by distance on the rule-A view (W5; by r; on r in {10, 13}) ----
rows = [r for r in csv.DictReader(open(W + "final_round1/analysis/calibration_leaves.csv")) if r["entity"] == "ML.rl"]
cal = {}


def pool(sel):
    E = sum(float(r["errors"]) for r in sel)
    P = sum(float(r["predicted"]) for r in sel)
    return {"leaves": len(sel), "errors": E, "predicted": P, "R": E / P if P else None}


for d in ("3", "5", "7"):
    Ld = [r for r in rows if "/d%s_at_" % d in r["leaf"]]
    cal["d%s_W5" % d] = pool([r for r in Ld if int(r["rounds"]) in (10, 13, 30, 50)])
    cal["d%s_r10_13" % d] = pool([r for r in Ld if int(r["rounds"]) in (10, 13)])
    for rr in (10, 13, 30, 50):
        s = [r for r in Ld if int(r["rounds"]) == rr]
        if s:
            cal["d%s_r%d" % (d, rr)] = pool(s)
cal["d5_final_round1"] = {k: FR["decisions"]["calibration_d5_ML_rl"][k] for k in ("leaves", "ratio", "ci90", "ci95")}
cal["note"] = ("pooled sum(errors)/sum(predicted) over ML(RL) leaves, as the frozen P3 statistic, on the rule-A view "
               "(the 7 leaves that still fail K2 are out at d = 5); d = 7 is the 4 probe leaves")
res["calibration_R_by_distance"] = cal

# ---- 7. Tesseract: its own eps at d = 5 and ML's paired eps on its rounds (rule A) ----
pt = FA["pooled_gaps_W5"]["d5|tesseract.rl vs ML.rl"]
res["tesseract_d5"] = {"eps_tesseract": t5["eps"], "se": t5["se"], "mean_eps_ML_paired": pt["mean_eps_ML"],
                       "rounds": "10, 13, 30"}

json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("libra", "ml_vs_nn", "inference_share_of_libra_nn_gap", "calibration_R_by_distance",
                                      "tesseract_d5")}, indent=1))
for a, row in gaps.items():
    if isinstance(row, dict) and "g5" in row:
        print(a, "g3 %.2f [%.2f, %.2f]" % (100 * row["g3"], 100 * row["ci95_3"][0], 100 * row["ci95_3"][1]),
              "g5 %.2f [%.2f, %.2f] (%d)" % (100 * row["g5"], 100 * row["ci95_5"][0], 100 * row["ci95_5"][1],
                                             row["n_series_5"]),
              "x%.1f" % row["growth_g5_over_g3"], "ratio %.4f [%.4f, %.4f] z %.1f" % (
                  row["paired_lambda_ratio"], *row["paired_lambda_ratio_ci95_log"], row["z_ln_ratio"]),
              "| res %.4f [%.4f, %.4f]" % (row["reservoir_matched"]["lambda_ratio"],
                                          *row["reservoir_matched"]["lambda_ratio_ci95"]),
              "diff %.1e" % row["difference_vs_reservoir"]["lambda_ratio"], "+%.1f pts" % (100 * row["g5_minus_g3"]))
    elif isinstance(row, dict):
        print(a, "d = 3 only:", "g3 %.2f [%.2f, %.2f]" % (100 * row["g3"], 100 * row["ci95_3"][0], 100 * row["ci95_3"][1]))
for k, v in lam.items():
    if "Lambda_3_5" in v:
        print(k, "%.3f" % v["Lambda_3_5"], "± %.3f" % v["se"] if "se" in v else "")
