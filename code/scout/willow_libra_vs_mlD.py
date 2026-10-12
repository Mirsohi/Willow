"""Libra against ML(D-hat) at d = 3 on the held-out halves (scout, 2026-10-10). Descriptive, post hoc; no decoding.

Main's answer (c) to scout's internal review, relayed by the coordinator after 21:27Z: "Libra against ML(D-hat) at
d = 3 on the held-out halves: yours, descriptive, no decoding."

Reference decoders, both exact maximum likelihood (ML) at d = 3:
  ML(RL)    the optimal decoder under the shipped RL prior (decode/.../rl/ml_exact.npz, every shot);
  ML(D-hat) the optimal decoder under decoder's F1 prior D-hat, fitted on the even shots of each series and evaluated
            on the odd (held-out) shots (round2/f1/.../d_data.npz, 25,000 shots per leaf).
Libra is Google's shipped "libra_decoder_with_rl_optimized_prior" output (decode/.../rl/shots.npz).

On the 54 held-out leaves of Libra's W5 rounds {10, 30, 50} (9 patches x 2 bases x 3 rounds; 18 series):
  1. Checks every leaf: obs[test_idx] equals obs_test; D_pred equals (D_llr > 0); the ML(RL) and ML(D-hat) error
     counts and the RL-against-D 2x2 cells equal decoder's eval.json.
  2. Count ratios, pooled over leaves, for Libra/ML(D-hat), Libra/ML(RL) and ML(RL)/ML(D-hat), with a stratified
     per-shot delta-method SE (u = x - R y, centred within each leaf), overall and per round count.
     Libra/ML(RL) is also given on all 50,000 shots of the same leaves, to show the held-out half is typical.
  3. Per-cycle paired gaps for the same three pairs: the paired estimator of PP section 4 with the SE of PP section 9
     entry 7 (delta method with Cov(beta, b_ref), Birge factors on both fits), pooled as g = mean(delta)/mean(eps_ref),
     with entry 7's SE and a series-level (cluster) SE.
Inputs (read only): results/decoder/willow/decode/ and results/decoder/willow/round2/f1/.
Usage: OMP_NUM_THREADS=1 nice -n 10 python3 -I willow_libra_vs_mlD.py <experimental-root>/ <out.json>
"""
import json
import math
import sys

import numpy as np
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
DEC = root + "results/decoder/willow/decode/google_105Q_surface_code_d3_d5_d7/"
F1 = root + "results/decoder/willow/round2/f1/google_105Q_surface_code_d3_d5_d7/"
P3 = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
ROUNDS = [10, 30, 50]
Z95 = stats.norm.ppf(0.975)
PAIRS = [("libra", "mlD"), ("libra", "mlRL"), ("mlRL", "mlD")]


def load_leaf(patch, basis, r):
    base = "%sd3_at_%s/%s/r%02d/" % (DEC, patch, basis, r)
    fb = "%sd3_at_%s/%s/r%02d/" % (F1, patch, basis, r)
    sh = np.load(base + "rl/shots.npz", allow_pickle=False)
    ml = np.load(base + "rl/ml_exact.npz", allow_pickle=False)
    dd = np.load(fb + "d_data.npz", allow_pickle=False)
    ev = json.load(open(fb + "eval.json"))
    assert np.array_equal(sh["idx"], ml["idx"]) and np.array_equal(ml["idx"], np.arange(ml["idx"].size))
    names = list(sh["shipped_names"])
    li = [i for i, s in enumerate(names) if "libra_decoder_with_rl_optimized_prior" in s]
    assert len(li) == 1, names
    obs = sh["obs"].astype(np.uint8)
    pm = ml["pred"][:, 0].astype(np.uint8)
    assert np.array_equal(pm, (ml["llr"] > 0).astype(np.uint8))
    pl = sh["shipped"][li[0]].astype(np.uint8)
    t = dd["test_idx"].astype(np.int64)
    ot = dd["obs_test"].astype(np.uint8)
    pd = dd["D_pred"].astype(np.uint8)
    checks = {
        "obs_test_matches": bool(np.array_equal(obs[t], ot)),
        "D_pred_is_sign_of_llr": bool(np.array_equal(pd, (dd["D_llr"] > 0).astype(np.uint8))),
        "n_test": int(t.size) == int(ev["n_test"]),
    }
    w = {"libra": pl[t] != ot, "mlRL": pm[t] != ot, "mlD": pd != ot}
    a, b = w["mlRL"], w["mlD"]
    checks["errors_RL_match"] = int(a.sum()) == int(ev["errors"]["RL"])
    checks["errors_D_match"] = int(b.sum()) == int(ev["errors"]["D"])
    c = ev["cells_RL_vs_D"]
    checks["cells_RL_vs_D_match"] = (int((a & ~b).sum()) == c["n10"] and int((~a & b).sum()) == c["n01"]
                                     and int((a & b).sum()) == c["n11"] and int(t.size) == c["n"])
    full = {"libra": pl != obs, "mlRL": pm != obs}
    return {"patch": patch, "basis": basis, "r": r, "n": int(t.size), "w": w, "full": full,
            "n_full": int(obs.size), "checks": checks}


def count_ratio(leaves, A, B, key="w"):
    """R = sum x_A / sum x_B over leaves; SE from u = x_A - R x_B centred within each leaf (stratified)."""
    sx = sum(int(L[key][A].sum()) for L in leaves)
    sy = sum(int(L[key][B].sum()) for L in leaves)
    R = sx / sy
    ss = 0.0
    for L in leaves:
        u = L[key][A].astype(float) - R * L[key][B].astype(float)
        ss += float(np.sum((u - u.mean()) ** 2))
    se = math.sqrt(ss) / sy
    n10 = sum(int((L[key][A] & ~L[key][B]).sum()) for L in leaves)
    n01 = sum(int((~L[key][A] & L[key][B]).sum()) for L in leaves)
    return {"errors_A": sx, "errors_B": sy, "n10_A_wrong_only": n10, "n01_B_wrong_only": n01,
            "rel": R - 1, "se": se, "ci95": [R - 1 - Z95 * se, R - 1 + Z95 * se], "z": (R - 1) / se,
            "leaves": len(leaves), "shots": sum(L["n" if key == "w" else "n_full"] for L in leaves)}


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


def series_fit(L, A, B):
    """Paired per-cycle fit of A against reference B over the leaves L of one series (PP section 4; entry 7's SE)."""
    r = np.array([x["r"] for x in L], float)
    n = np.array([x["n"] for x in L], float)
    EA = np.array([x["w"][A].sum() for x in L]) / n
    EB = np.array([x["w"][B].sum() for x in L]) / n
    p11 = np.array([(x["w"][A] & x["w"][B]).sum() for x in L]) / n
    yA, yB = np.log(1 - 2 * EA), np.log(1 - 2 * EB)
    gA, gB = -2 / (1 - 2 * EA), -2 / (1 - 2 * EB)
    VA, VB, C = EA * (1 - EA) / n, EB * (1 - EB) / n, (p11 - EA * EB) / n
    var_yB = gB ** 2 * VB
    var_z = gA ** 2 * VA + gB ** 2 * VB - 2 * gA * gB * C
    cov_z_yB = gA * gB * C - gB ** 2 * VB
    bB, c, se_bB, BB, chiB = wls(r, yB, var_yB)
    beta, a, se_beta, Bz, chiz = wls(r, yA - yB, var_z)
    V_bB, V_beta = (se_bB * BB) ** 2, (se_beta * Bz) ** 2
    C_bb = float(np.sum(a * c * cov_z_yB)) * BB * Bz
    eB, eA = (1 - math.exp(bB)) / 2, (1 - math.exp(bB + beta)) / 2
    Jd = np.array([(math.exp(bB) - math.exp(bB + beta)) / 2, -math.exp(bB + beta) / 2])
    Je = np.array([-math.exp(bB) / 2, 0.0])
    S = np.array([[V_bB, C_bb], [C_bb, V_beta]])
    return {"patch": L[0]["patch"], "basis": L[0]["basis"], "eps_ref": eB, "eps_A": eA, "delta": eA - eB,
            "var_delta": float(Jd @ S @ Jd), "var_eps_ref": float(Je @ S @ Je), "cov_delta_eps": float(Jd @ S @ Je),
            "rel": (eA - eB) / eB, "birge_ref": BB, "birge_z": Bz, "chi2_red_z": chiz}


def pooled(fits):
    S = len(fits)
    D = np.mean([f["delta"] for f in fits])
    E = np.mean([f["eps_ref"] for f in fits])
    g = D / E
    vD = sum(f["var_delta"] for f in fits) / S ** 2
    vE = sum(f["var_eps_ref"] for f in fits) / S ** 2
    cDE = sum(f["cov_delta_eps"] for f in fits) / S ** 2
    se = math.sqrt(vD - 2 * g * cDE + g * g * vE) / E
    u = np.array([f["delta"] - g * f["eps_ref"] for f in fits])
    se_cl = math.sqrt(S / (S - 1) * np.sum(u ** 2)) / (S * E)
    t = stats.t.ppf(0.975, S - 1)
    rel = np.array([f["rel"] for f in fits])
    return {"g": g, "se_entry7": se, "ci95_entry7": [g - Z95 * se, g + Z95 * se], "se_cluster": se_cl,
            "ci95_cluster_t": [g - t * se_cl, g + t * se_cl], "series": S,
            "series_above_zero": int(np.sum(rel > 0)), "series_rel_median": float(np.median(rel)),
            "series_rel_range": [float(rel.min()), float(rel.max())]}


def main():
    leaves = [load_leaf(p, b, r) for p in P3 for b in ("X", "Z") for r in ROUNDS]
    bad = [(L["patch"], L["basis"], L["r"], k) for L in leaves for k, v in L["checks"].items() if not v]
    res = {"made_by": "scout, code/scout/willow_libra_vs_mlD.py",
           "label": "descriptive, post hoc (main's answer (c) to scout's internal review); no decoding",
           "leaves": len(leaves), "series": 18, "rounds": ROUNDS, "held_out_shots_per_leaf": 25000,
           "check_failures": bad, "count_ratio": {}, "count_ratio_by_r": {}, "per_cycle": {}, "per_series": {}}
    for A, B in PAIRS:
        k = "%s_vs_%s" % (A, B)
        res["count_ratio"][k] = count_ratio(leaves, A, B)
        res["count_ratio_by_r"][k] = {str(r): count_ratio([L for L in leaves if L["r"] == r], A, B) for r in ROUNDS}
        fits = []
        for p in P3:
            for b in ("X", "Z"):
                fits.append(series_fit([L for L in leaves if L["patch"] == p and L["basis"] == b], A, B))
        res["per_cycle"][k] = pooled(fits)
        res["per_series"][k] = [{kk: (round(v, 8) if isinstance(v, float) else v) for kk, v in f.items()}
                                for f in fits]
    res["count_ratio"]["libra_vs_mlRL_all_shots"] = count_ratio(leaves, "libra", "mlRL", key="full")
    # multiplicative decomposition of Libra's excess over ML(D-hat): (1 + g_LD) = (1 + g_LR)(1 + g_RD) for counts
    cr = res["count_ratio"]
    lLD = math.log1p(cr["libra_vs_mlD"]["rel"])
    res["decomposition_count_log_scale"] = {
        "ln_libra_vs_mlD": lLD,
        "share_inference_loss_at_RL": math.log1p(cr["libra_vs_mlRL"]["rel"]) / lLD,
        "share_RL_vs_Dhat": math.log1p(cr["mlRL_vs_mlD"]["rel"]) / lLD,
        "note": "exact identity for pooled counts on the same shots; shares are descriptive"}
    pc = res["per_cycle"]
    lLDc = math.log1p(pc["libra_vs_mlD"]["g"])
    res["decomposition_per_cycle_log_scale"] = {
        "ln_libra_vs_mlD": lLDc,
        "share_inference_loss_at_RL": math.log1p(pc["libra_vs_mlRL"]["g"]) / lLDc,
        "share_RL_vs_Dhat": math.log1p(pc["mlRL_vs_mlD"]["g"]) / lLDc,
        "note": "approximate for per-cycle gaps (separate fits); descriptive"}
    json.dump(res, open(out, "w"), indent=1)
    print("check failures:", bad)
    for k, v in res["count_ratio"].items():
        print("count", k, "%+.3f%% [%.3f, %.3f] z %.2f  A %d B %d n10 %d n01 %d" % (
            100 * v["rel"], 100 * v["ci95"][0], 100 * v["ci95"][1], v["z"], v["errors_A"], v["errors_B"],
            v["n10_A_wrong_only"], v["n01_B_wrong_only"]))
    for k, d in res["count_ratio_by_r"].items():
        print("by r", k, {r: "%+.2f%% +- %.2f" % (100 * v["rel"], 100 * v["se"]) for r, v in d.items()})
    for k, v in res["per_cycle"].items():
        print("per-cycle", k, "%+.3f%% +- %.3f (entry 7), +- %.3f (cluster); series > 0: %d of 18; median %+.2f%%" % (
            100 * v["g"], 100 * v["se_entry7"], 100 * v["se_cluster"], v["series_above_zero"],
            100 * v["series_rel_median"]))
    print("decomposition count", res["decomposition_count_log_scale"])
    print("decomposition per-cycle", res["decomposition_per_cycle_log_scale"])


main()
