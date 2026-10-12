"""Derived numbers for the restructured Willow manuscript (scout, 2026-10-10 evening).

Descriptive arithmetic on frozen or posted outputs only; no decoding, no new statistic of the pre-registration.
Inputs (read only):
  results/decoder/willow/analysis/analysis.json        frozen analyze_willow.py output, written 19:30Z (interim at d = 5)
  results/decoder/willow/analysis/calibration_leaves.csv
  results/reservoir/willow_d5check/d5_check.json       reservoir's independent check (entry 7 SEs)
  results/reservoir/willow_d5check/compare_d5.json
  results/decoder/willow/supplementary/p1_se_covariance.json   entry 7's SE of P1 (absolute delta)
Reported values (Willow paper, Table S1 and main text [V]) are typed in below and labelled.
Usage: python3 -I willow_derived.py <experimental-root>/ <out.json>
"""
import csv
import json
import math
import sys

root, out = sys.argv[1], sys.argv[2]
A = json.load(open(root + "results/decoder/willow/analysis/analysis.json"))
R5 = json.load(open(root + "results/reservoir/willow_d5check/d5_check.json"))
C5 = json.load(open(root + "results/reservoir/willow_d5check/compare_d5.json"))

Z95 = 1.959963984540054
res = {"label": "scout, descriptive arithmetic on the frozen analysis (19:30Z, interim at d = 5) and reservoir's "
                "check (19:58Z); frozen SEs unless 'corrected'", "inputs": {}}

# ---- 1. Every decoder's paired gap to ML(RL) on W5 at d = 3 and d = 5 (frozen SEs) ----
pg = A["pooled_gaps_W5"]
decs = ["libra", "tesseract", "harmony", "corrmatch", "corr_mwpm", "mwpm", "beliefmatch"]
gaps = {}
for dname in decs:
    row = {}
    for d in ("3", "5"):
        k = "d%s|%s.rl vs ML.rl" % (d, dname)
        if k in pg:
            e = pg[k]
            row["g" + d] = e["rel"]
            row["se" + d] = e["se_rel"]
            row["ci95_" + d] = e["ci95"]
            row["n_series_" + d] = e["n_series"]
            row["mean_eps_ML_paired_" + d] = e["mean_eps_ML"]
    if "g3" in row and "g5" in row:
        g3, g5, s3, s5 = row["g3"], row["g5"], row["se3"], row["se5"]
        ratio = (1 + g3) / (1 + g5)
        se_ln = math.sqrt((s3 / (1 + g3)) ** 2 + (s5 / (1 + g5)) ** 2)
        row["growth_g5_over_g3"] = g5 / g3
        row["paired_lambda_ratio"] = ratio
        row["paired_lambda_ratio_se_ln"] = se_ln
        row["paired_lambda_ratio_ci95_log"] = [ratio * math.exp(-Z95 * se_ln), ratio * math.exp(Z95 * se_ln)]
        row["z_ln_ratio"] = math.log(ratio) / se_ln
        row["g5_minus_g3"] = g5 - g3
        row["g5_minus_g3_se"] = math.sqrt(s3 ** 2 + s5 ** 2)
    gaps[dname] = row
res["gaps_W5_frozen_se"] = gaps

# ---- 2. Lambda_3/5 per decoder from the W5 mean eps (unpaired, independent SEs) ----
lam = {}
for dname in ["ML", "libra", "harmony", "corrmatch", "corr_mwpm", "mwpm"]:
    key = dname + ".rl"
    e = A["lambda"][key]["eps"]
    e3, s3 = e["3"]
    e5, s5 = e["5"]
    L = e3 / e5
    lam[dname] = {"eps3_W5": e3, "se3": s3, "eps5_W5": e5, "se5": s5, "Lambda_3_5": L,
                  "se": L * math.sqrt((s3 / e3) ** 2 + (s5 / e5) ** 2),
                  "Lambda_fit_frozen_table": A["lambda"][key]["Lambda"],
                  "distances_frozen_table": A["lambda"][key]["distances"]}
# Tesseract: d = 3 on r <= 50 (its whole subset), d = 5 on r <= 30; windows differ (indicative only)
t3 = A["mean_eps"]["d3|tesseract.rl"]
t5 = A["mean_eps"]["d5|tesseract.rl"]
lam["tesseract_windows_differ"] = {"eps3": t3["eps"], "se3": t3["se"], "eps5": t5["eps"], "se5": t5["se"],
                                   "Lambda_3_5": t3["eps"] / t5["eps"],
                                   "note": "d = 3 on r in {10, 13, 30, 50}, d = 5 on r in {10, 13, 30}; not comparable"}
res["lambda_3_5_unpaired"] = lam

# ---- 3. Libra with entry 7's SEs (reservoir), and the frozen paired ratio ----
pr = R5["lambda"]["paired_ratio_libra_over_ml"]
res["libra_corrected"] = {
    "g3_10_30_50": pr["g3_on_10_30_50"], "g3_se_corrected": pr["g3_se"],
    "g5": pr["g5"], "g5_se_corrected": pr["g5_se"],
    "g5_ci95_corrected": R5["g5"]["ci95"], "g5_ci90_corrected": R5["g5"]["ci90"],
    "ratio": pr["ratio"], "ratio_ci95_delta_corrected": pr["ci95"],
    "ratio_ci95_log_corrected": C5["lambda"]["paired_ratio"]["ci95_log_scale_corrected"],
    "ratio_ci95_frozen": A["lambda_paired_libra_over_ML_rl"]["ci95"],
}
g3c, g5c = pr["g3_on_10_30_50"], pr["g5"]
s3c, s5c = pr["g3_se"], pr["g5_se"]
se_ln_c = math.sqrt((s3c / (1 + g3c)) ** 2 + (s5c / (1 + g5c)) ** 2)
res["libra_corrected"]["z_ln_ratio_corrected"] = math.log((1 + g3c) / (1 + g5c)) / se_ln_c
res["libra_corrected"]["g5_minus_g3"] = g5c - g3c
res["libra_corrected"]["g5_minus_g3_se_corrected"] = math.sqrt(s3c ** 2 + s5c ** 2)

# ---- 4. ML(RL) against the reported NN, and the inference-versus-model split (approximate, unpaired) ----
P5 = A["prereg"]["P5_ML_rl_vs_NN"]["pooled"]
h3, sh3 = P5["3"]["rel"], P5["3"]["se_rel"]
h5, sh5 = P5["5"]["rel"], P5["5"]["se_rel"]
r_ml_nn = (1 + h3) / (1 + h5)
se_ln_mlnn = math.sqrt((sh3 / (1 + h3)) ** 2 + (sh5 / (1 + h5)) ** 2)
reported = {"eps_libra": {"3": [7.12e-3, 0.06e-3], "5": [3.49e-3, 0.04e-3], "7": [1.71e-3, 0.03e-3]},
            "eps_nn": {"3": [6.50e-3, 0.06e-3], "5": [3.03e-3, 0.03e-3], "7": [1.43e-3, 0.03e-3]},
            "Lambda_libra": [2.04, 0.02], "Lambda_nn": [2.14, 0.02],
            "source": "Willow paper (arXiv:2408.13687v1) Table S1 and main text, reported [V]"}
lib_nn = {d: reported["eps_libra"][d][0] / reported["eps_nn"][d][0] - 1 for d in ("3", "5", "7")}
res["ml_vs_nn"] = {
    "h3": h3, "h3_se": sh3, "h5": h5, "h5_se": sh5,
    "Lambda_ML_over_Lambda_NN_3_5": r_ml_nn,
    "ci95_log": [r_ml_nn * math.exp(-Z95 * se_ln_mlnn), r_ml_nn * math.exp(Z95 * se_ln_mlnn)],
    "reported_libra_over_nn_eps_minus_1": lib_nn,
    "reported_Lambda_libra_over_Lambda_nn": reported["Lambda_libra"][0] / reported["Lambda_nn"][0],
    "reported_Lambda35_libra_over_nn_from_tableS1":
        (reported["eps_libra"]["3"][0] / reported["eps_libra"]["5"][0]) /
        (reported["eps_nn"]["3"][0] / reported["eps_nn"]["5"][0]),
    "reported": reported,
}
# inference share of Libra's reported gap to the NN: (Libra - ML, paired, ours) / (Libra - NN, Table S1)
P1 = A["prereg"]["P1_libra_vs_ML_rl_d3"]["pooled"]
SEC = json.load(open(root + "results/decoder/willow/supplementary/p1_se_covariance.json"))
num3, num3_se = P1["mean_delta"], SEC["full_delta_method_inflated"]["se_mean_delta"]  # entry 7's SE
e5 = A["pooled_gaps_W5"]["d5|libra.rl vs ML.rl"]
num5 = e5["mean_delta"]
num5_se = s5c * e5["mean_eps_ML"]  # corrected relative SE times ML's paired eps
share = {}
for d, num, nse in (("3", num3, num3_se), ("5", num5, num5_se)):
    den = reported["eps_libra"][d][0] - reported["eps_nn"][d][0]
    dse = math.hypot(reported["eps_libra"][d][1], reported["eps_nn"][d][1])
    s = num / den
    share[d] = {"numerator_libra_minus_ML_paired": num, "numerator_se": nse,
                "denominator_libra_minus_nn_reported": den, "denominator_se": dse,
                "share": s, "share_se": s * math.hypot(nse / num, dse / den)}
share["note"] = ("approximate: the numerator is our paired Libra - ML(RL) difference (d = 3: Libra's 13 round counts, "
                 "r >= 10; d = 5: r in {10, 30, 50}); the denominator is Table S1's Libra - NN (reported; window not "
                 "stated); unpaired")
res["inference_share_of_libra_nn_gap"] = share

# ---- 5. Calibration R by distance on a common window (r in {10, 13}) and on W5 ----
rows = list(csv.DictReader(open(root + "results/decoder/willow/analysis/calibration_leaves.csv")))
cal = {}
for win_name, win in (("r10_13", {10, 13}), ("W5", {10, 13, 30, 50})):
    for d in ("3", "5", "7"):
        E = P = 0.0
        n = 0
        for r in rows:
            if r["entity"] != "ML.rl" or ("/d%s_at_" % d) not in r["leaf"]:
                continue
            if int(r["rounds"]) not in win:
                continue
            E += float(r["errors"])
            P += float(r["predicted"])
            n += 1
        if n:
            cal["d%s_%s" % (d, win_name)] = {"leaves": n, "errors": E, "predicted": P, "R": E / P}
for d in ("3", "5", "7"):
    for rr in (10, 13, 30, 50):
        E = P = 0.0
        n = 0
        for r in rows:
            if r["entity"] == "ML.rl" and ("/d%s_at_" % d) in r["leaf"] and int(r["rounds"]) == rr:
                E += float(r["errors"])
                P += float(r["predicted"])
                n += 1
        if n:
            cal["d%s_r%d" % (d, rr)] = {"leaves": n, "errors": E, "predicted": P, "R": E / P}
cal["note"] = ("pooled sum(errors)/sum(predicted) over ML(RL) leaves, as the frozen P3 statistic; d = 5 includes the "
               "leaves that still fail K2 (interim); d = 7 is the 4 probe leaves (2,000 shots each, r = 10 and 13)")
res["calibration_R_by_distance"] = cal

json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
