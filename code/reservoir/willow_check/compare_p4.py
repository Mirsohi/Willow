"""Compare the independent P4 check (p4_check.json, p4_leaves.csv; stored 2026-10-09 22:31Z, md5s in the log) with
decoder's milestone-3 analysis, which was read only after that (analysis_m3/prereg_results.json and analysis.json:
P4, the decomposition, the per-leaf 2x2 cells, the sensitivity row). Also recomputes, from the fit outputs, decoder's
exploratory held-out calibration of ML(RL) and ML(T-hat) (supplementary/p3_heldout_fitted.json): R and R_conf on the
held-out halves, and summarises the BHHH histories (step halving, steps, the last step taken). Those parts were computed
after reading decoder's numbers and code/decoder/fit_noise.py's fit_bhhh.
Usage: nice -n 19 python3 -I compare_p4.py OUT_DIR ANALYSIS_M3 SUPPLEMENTARY FIT
  -> OUT_DIR/p4_comparison.json (through qrc.outpaths)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import in_path, out_path  # noqa: E402

W5 = [10, 13, 30, 50]
CONF = 3e-3


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def calib(llr, obs):
    L = llr.astype(np.float64)
    m = 1 / (1 + np.exp(np.abs(L)))
    err = (L > 0).astype(np.uint8) != obs
    conf = m < CONF
    return {"errors": int(err.sum()), "predicted": float(m.sum()), "conf_errors": int(err[conf].sum()),
            "conf_predicted": float(m[conf].sum())}


def main(out_dir, m3, supp, fit):
    mine = json.load(open(in_path(os.path.join(out_dir, "p4_check.json"))))
    rows = list(csv.DictReader(open(in_path(os.path.join(out_dir, "p4_leaves.csv")))))
    pr = json.load(open(os.path.join(m3, "prereg_results.json")))
    an = json.load(open(os.path.join(m3, "analysis.json")))
    sp = json.load(open(os.path.join(supp, "p3_heldout_fitted.json")))
    res = {"label": "reservoir, P4 check against decoder's analysis_m3 (read after p4_check.json was stored)"}
    d4 = pr["P4_mismatch_cost_d3"]["pooled"]
    m4 = mine["P4_all_72"]
    res["P4"] = {"m": [m4["m"], d4["rel"], m4["m"] == d4["rel"]],
                 "K_RL": [m4["K_RL"], d4["errors_A"]], "K_T": [m4["K_T"], d4["errors_B"]],
                 "se": [m4["se"], d4["se_rel"], m4["se"] / d4["se_rel"]],
                 "se_note": "ratio = sqrt(25000/24999): reservoir uses the within-leaf variance with n - 1, decoder with n",
                 "z": [m4["z"], d4["z"]], "p": [m4["p_two_sided"], d4["p"]],
                 "p_holm": [mine["holm"]["p_adjusted"]["P4"], pr["P4_mismatch_cost_d3"]["p_holm"]],
                 "ci90": [m4["ci90"], d4["ci90"]], "ci95": [m4["ci95"], d4["ci95"]]}
    dp = an["decomposition"]["pooled"]
    dm = mine["decomposition_54"]
    res["decomposition"] = {
        "counts_mine": dm["counts"], "K_mine": [dm["K_Libra"], dm["K_RL"], dm["K_T"]],
        "K_decoder": [dp["total"]["errors_A"], dp["mismatch_libra_leaves"]["errors_A"], dp["total"]["errors_B"]],
        "total": [dm["total"]["value"], dp["total"]["rel"], dm["total"]["se"], dp["total"]["se_rel"]],
        "suboptimality": [dm["suboptimality"]["value"], dp["suboptimality"]["rel"], dm["suboptimality"]["se"],
                          dp["suboptimality"]["se_rel"]],
        "suboptimality_se_note": ("decoder: SE of (K_L - K_RL)/K_RL rescaled by K_RL/K_T (K_T treated as fixed); "
                                  "reservoir: delta method for (K_L - K_RL)/K_T directly. Values agree to 0.4% of SE."),
        "mismatch_54": [dm["mismatch_m54"]["value"], dp["mismatch_libra_leaves"]["rel"], dm["mismatch_m54"]["se"],
                        dp["mismatch_libra_leaves"]["se_rel"]]}
    sens = an["sensitivity_excl_r13"]["P4_mismatch_d3"]
    ms = mine["sensitivity_without_r13"]
    res["sensitivity_without_r13"] = {"m": [ms["m"], sens["rel"]], "se": [ms["se"], sens["se_rel"]],
                                      "p": [ms["p_two_sided"], sens["p"]]}
    # per-leaf 2x2 cells
    cells = {}
    for x in an["decomposition"]["leaves"]:
        if x["d"] == 3 and x["rounds"] in W5:
            cells[(x["patch"], x["basis"], int(x["rounds"]))] = x["cells"]
    diff = []
    for r_ in rows:
        k = (r_["patch"], r_["basis"], int(r_["r"]))
        c = cells.get(k)
        if c is None:
            diff.append(f"{k}: missing in decoder")
            continue
        mm = c["mismatch"]
        mine_mm = (int(r_["RL_wrong_T_right"]), int(r_["RL_right_T_wrong"]), int(r_["both_wrong"]), int(r_["n_test"]))
        if mine_mm != (mm["n10"], mm["n01"], mm["n11"], mm["n"]):
            diff.append(f"{k}: mismatch cells {mine_mm} vs {mm}")
        if r_["K_Libra"] != "":
            t, s = c.get("total"), c.get("suboptimality")
            if (int(r_["Libra_wrong_T_right"]), int(r_["Libra_right_T_wrong"])) != (t["n10"], t["n01"]):
                diff.append(f"{k}: total cells")
            if (int(r_["Libra_wrong_RL_right"]), int(r_["Libra_right_RL_wrong"])) != (s["n10"], s["n01"]):
                diff.append(f"{k}: suboptimality cells")
            if int(r_["K_Libra"]) != t["n10"] + t["n11"]:
                diff.append(f"{k}: Libra errors")
    res["per_leaf_cells"] = {"leaves_compared": len(rows), "differences": diff}
    # held-out calibration of both decoders (decoder's exploratory supplementary), recomputed from fit_data
    acc = {"RL": [0, 0.0, 0, 0.0], "T_hat": [0, 0.0, 0, 0.0]}
    for r_ in rows:
        rel_ = os.path.join(f"d3_at_{r_['patch']}", r_["basis"], f"r{int(r_['r']):02d}", "rl_tsplit", "fit_data.npz")
        fd = np.load(os.path.join(fit, rel_), allow_pickle=False)
        for name, key in (("RL", "shipped"), ("T_hat", "fitted")):
            c = calib(fd[f"{key}_llr"], fd["obs_test"])
            a = acc[name]
            a[0] += c["errors"]; a[1] += c["predicted"]; a[2] += c["conf_errors"]; a[3] += c["conf_predicted"]
    mine_cal = {k: {"errors": v[0], "predicted": v[1], "R": v[0] / v[1], "conf_errors": v[2],
                    "conf_predicted": v[3], "R_conf": v[2] / v[3], "R_conf_se": v[2] / v[3] / math.sqrt(v[2])}
                for k, v in acc.items()}
    res["heldout_calibration_recomputed"] = {"mine": mine_cal, "decoder_supplementary": {
        k: sp[k] for k in ("by_decoder", "refit_share_of_excess_removed") if k in sp}}
    # BHHH history (read from fit.json after reading code/decoder/fit_noise.py's fit_bhhh): the stop test is applied
    # to the next proposed step, which is not taken and not stored, so the stored last step need not be < 0.02 SE
    lam_ok, steps, last = 0, [], []
    for r_ in rows:
        rel_ = os.path.join(f"d3_at_{r_['patch']}", r_["basis"], f"r{int(r_['r']):02d}", "rl_tsplit", "fit.json")
        h = json.load(open(os.path.join(fit, rel_)))["history"]
        lam_ok += all(x.get("lambda", 1.0) == 1.0 for x in h[1:])
        steps.append(len(h) - 1)
        last.append(max(abs(v) for v in h[-1]["step_in_se"]))
    res["bhhh"] = {"fits_without_step_halving": lam_ok, "fits": len(rows), "steps_min_max": [min(steps), max(steps)],
                   "last_taken_step_max_abs_in_se_min_median_max": [min(last), float(np.median(last)), max(last)],
                   "note": "fit_bhhh stops when every component of the next proposed step is below 0.02 SE; that step "
                           "is not taken or stored, so the stop test itself cannot be checked from fit.json"}
    with open(out_path(os.path.join(out_dir, "p4_comparison.json")), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("P4", "decomposition", "sensitivity_without_r13", "per_leaf_cells")},
                     indent=1, default=str))
    print(json.dumps(mine_cal, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
