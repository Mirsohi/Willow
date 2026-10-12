"""Independent check of the first confirmatory Willow result (P1: Libra against exact ML with the RL DEM, d = 3).

reservoir, 2026-10-09. Written from notes/decoder/willow_prereg.md section 4 alone. decoder's analysis code
(code/decoder/analyze_willow.py), its analysis outputs (results/decoder/willow/analysis_m2/) and board/decoder.md were
not read before these numbers were computed.

Inputs (read only; run with python3 -I because the archive is downloaded data):
  archive  ARCHIVE/d3_at_<patch>/<basis>/r<NN>/obs_flips_actual.b8 and
           .../decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8 (stim b8, 1 observable)
  decoder  DECODE/d3_at_<patch>/<basis>/r<NN>/rl/ml_exact.npz (pred, llr, logp, idx) and shots.npz (idx, obs, shipped);
           rl/summary.json only for the md5 of the DEM file that ML used
Scope: d = 3, 9 patches x {X, Z} = 18 series; r in {10, 30, 50, ..., 250}, the 13 round counts r >= 10 with Libra
predictions (Libra has none at r = 1 and r = 13).

Estimators (prereg section 4):
  eps of a series   WLS of y_r = ln(1 - 2 E_r) on r with a free intercept, weights 1 / Var(y_r), delta method
                    Var(y_r) = 4 E (1 - E) / (n (1 - 2 E)^2); eps = (1 - e^slope) / 2; se x sqrt(max(1, chi2_red)).
  mean over series  arithmetic mean; se = sqrt(sum se_s^2) / S.
  paired gap g      per series, WLS of z_r = y_r(Libra) - y_r(ML) on r with Var(z_r) = Var(y_A) + Var(y_ML) - 2 Cov from
                    the leaf's 2x2 table of (Libra wrong, ML wrong); eps_A = (1 - e^(slope_ML + slope_z)) / 2;
                    g = mean_s(eps_A - eps_ML) / mean_s(eps_ML). Delta-method se, including Cov(slope_ML, slope_z)
                    (both fits use the same shots). Reported without and with the chi2_red inflation of each fit.
  pooled ratio      (sum k_Libra - sum k_ML) / sum k_ML; per-shot delta method, stratified by leaf.
Usage: nice -n 19 python3 -I willow_check.py ARCHIVE_ROOT DECODE_ROOT OUT_DIR
  -> OUT_DIR/leaves.csv (per-leaf counts, 2x2 tables, checks), OUT_DIR/series.csv, OUT_DIR/willow_check.json
  The stored results are in results/reservoir/willow_check/. With OUT_DIR there, a re-run writes to
  results/reservoir/rebuild/willow_check/ unless QRC_OVERWRITE=1 (code/reservoir/qrc/outpaths.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import stim

sys.dont_write_bytecode = True  # no __pycache__ in the shared code folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import in_path, out_path  # noqa: E402,F401

PATCHES = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
BASES = ["X", "Z"]
ROUNDS = [10] + list(range(30, 251, 20))  # 13 round counts
LIBRA = "libra_decoder_with_rl_optimized_prior"
CM_RL = "correlated_matching_decoder_with_rl_optimized_prior"
REPORTED = {"table_S1_libra_eps": [7.12e-3, 0.06e-3],
            "decoder_libra_eps": [7.116e-3, 0.063e-3], "decoder_ml_rl_eps": [7.008e-3, 0.059e-3],
            "decoder_g": [0.01644, 0.00095]}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def read_obs_b8(path):
    """One observable per shot: stim's reader, cross-checked against the raw bytes (bit 0, padding bits zero)."""
    a = stim.read_shot_data_file(path=path, format="b8", num_observables=1)
    raw = np.fromfile(path, dtype=np.uint8)
    if a.shape != (raw.size, 1) or not np.array_equal(a[:, 0], (raw & 1).astype(bool)):
        raise SystemExit(f"b8 parse mismatch in {path}")
    return a[:, 0].astype(np.uint8), bool(np.all((raw >> 1) == 0))


def leaf(arch, dec, patch, basis, r):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir, d_dir = os.path.join(arch, rel), os.path.join(dec, rel, "rl")
    act, pad_a = read_obs_b8(os.path.join(a_dir, "obs_flips_actual.b8"))
    lib, pad_l = read_obs_b8(os.path.join(a_dir, "decoding_results", LIBRA, "obs_flips_predicted.b8"))
    n_all = act.size
    ml = np.load(os.path.join(d_dir, "ml_exact.npz"), allow_pickle=False)
    sh = np.load(os.path.join(d_dir, "shots.npz"), allow_pickle=False)
    idx, pred, llr = ml["idx"].astype(np.int64), ml["pred"].reshape(len(ml["idx"]), -1)[:, 0], ml["llr"]
    rec = {"patch": patch, "basis": basis, "r": r, "n_archive": int(n_all), "n_decoded": int(len(idx)),
           "b8_padding_zero": pad_a and pad_l}
    # (e) sanity checks
    rec["idx_is_all_shots_in_order"] = bool(np.array_equal(idx, np.arange(n_all)))
    rec["shots_idx_equal"] = bool(np.array_equal(sh["idx"].astype(np.int64), idx))
    rec["obs_mismatch"] = int(np.sum(sh["obs"].astype(np.uint8) != act[idx]))
    tie = llr == 0
    rec["llr_nan"] = int(np.isnan(llr).sum())
    rec["llr_zero"] = int(tie.sum())
    rec["llr_abs_below_1e-6"] = int((np.abs(llr) < 1e-6).sum())
    rec["pred_vs_argmax_mismatch_if_llr_ln_p1_over_p0"] = int(np.sum(pred[~tie] != (llr[~tie] > 0)))
    rec["pred_vs_argmax_mismatch_if_llr_ln_p0_over_p1"] = int(np.sum(pred[~tie] != (llr[~tie] < 0)))
    names = [str(x) for x in sh["shipped_names"]]
    li = [i for i, s in enumerate(names) if LIBRA in s]
    rec["shipped_libra_in_shots_npz_mismatch"] = (int(np.sum(sh["shipped"][li[0]] != lib[idx])) if li else None)
    rec["dem_md5_libra"] = md5(os.path.join(a_dir, "decoding_results", LIBRA, "error_model.dem"))
    rec["dem_md5_corr_matching_rl"] = md5(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    rec["dem_md5_used_by_ml"] = json.load(open(os.path.join(d_dir, "summary.json"))).get("dem_md5")
    # (a) counts and the 2x2 table
    ml_wrong = pred != act[idx]
    lib_wrong = lib[idx] != act[idx]
    rec.update(n=int(len(idx)), k_ml=int(ml_wrong.sum()), k_libra=int(lib_wrong.sum()),
               n11=int(np.sum(lib_wrong & ml_wrong)), n10=int(np.sum(lib_wrong & ~ml_wrong)),
               n01=int(np.sum(~lib_wrong & ml_wrong)), n00=int(np.sum(~lib_wrong & ~ml_wrong)))
    return rec


def y_var(k, n):
    E = np.asarray(k, float) / np.asarray(n, float)
    if np.any(E >= 0.5):
        raise SystemExit("E_r >= 1/2: ln(1 - 2E) undefined (no rule in the prereg)")
    return E, np.log(1 - 2 * E), 4 * E * (1 - E) / (np.asarray(n, float) * (1 - 2 * E) ** 2)


def wls(x, y, var):
    w = 1.0 / var
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    coef = cov @ (X.T * w)  # beta = coef @ y
    beta = coef @ y
    chi2 = float(np.sum(w * (y - X @ beta) ** 2))
    return {"a": float(beta[0]), "b": float(beta[1]), "var_b": float(cov[1, 1]), "c_b": coef[1],
            "chi2_red": chi2 / (len(x) - 2)}


def eps_fit(r, k, n):
    E, y, v = y_var(k, n)
    f = wls(r, y, v)
    infl = max(1.0, f["chi2_red"])
    eb = math.exp(f["b"])
    return {"eps": (1 - eb) / 2, "se": eb / 2 * math.sqrt(f["var_b"] * infl), "se_no_inflation": eb / 2 * math.sqrt(f["var_b"]),
            "chi2_red": f["chi2_red"], "slope": f["b"]}


def paired(r, kA, kM, n11, n):
    EA, yA, vA = y_var(kA, n)
    EM, yM, vM = y_var(kM, n)
    p11 = np.asarray(n11, float) / np.asarray(n, float)
    cAM = 4 * (p11 - EA * EM) / (np.asarray(n, float) * (1 - 2 * EA) * (1 - 2 * EM))  # Cov(y_A, y_ML)
    vz = vA + vM - 2 * cAM
    fM, fz = wls(r, yM, vM), wls(r, yA - yM, vz)
    cov_b = float(np.sum(fM["c_b"] * fz["c_b"] * (cAM - vM)))  # Cov(slope_ML, slope_z); Cov(y_ML, z) = cAM - vM
    bM, bz = fM["b"], fz["b"]
    epsM, epsA = (1 - math.exp(bM)) / 2, (1 - math.exp(bM + bz)) / 2
    gM = np.array([-math.exp(bM) / 2, 0.0])
    gd = np.array([(math.exp(bM) - math.exp(bM + bz)) / 2, -math.exp(bM + bz) / 2])
    out = {"eps_ml": epsM, "eps_a_paired": epsA, "d": epsA - epsM, "slope_ml": bM, "slope_z": bz,
           "chi2_red_ml": fM["chi2_red"], "chi2_red_z": fz["chi2_red"], "var_slope_z": fz["var_b"],
           "corr_slopes": cov_b / math.sqrt(fM["var_b"] * fz["var_b"])}
    for tag, iM, iz in (("", 1.0, 1.0), ("_inflated", max(1.0, fM["chi2_red"]), max(1.0, fz["chi2_red"]))):
        V = np.array([[fM["var_b"] * iM, cov_b * math.sqrt(iM * iz)], [cov_b * math.sqrt(iM * iz), fz["var_b"] * iz]])
        out["var_d" + tag], out["var_eps_ml" + tag], out["cov_d_eps_ml" + tag] = gd @ V @ gd, gM @ V @ gM, gd @ V @ gM
    return out


def main(arch, dec, out_dir):
    t0 = time.time()
    leaves = []
    for p in PATCHES:
        for b in BASES:
            for r in ROUNDS:
                leaves.append(leaf(arch, dec, p, b, r))
        print(f"{p}: {len(leaves)} leaves read, {time.time() - t0:.0f} s", flush=True)
    keys = list(leaves[0].keys())
    with open(out_path(os.path.join(out_dir, "leaves.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(leaves)

    # (b), (c) per series
    series, S = [], len(PATCHES) * len(BASES)
    for p in PATCHES:
        for b in BASES:
            L = [x for x in leaves if x["patch"] == p and x["basis"] == b]
            r = np.array([x["r"] for x in L], float)
            n = np.array([x["n"] for x in L], float)
            kM, kA, n11 = (np.array([x[k] for x in L], float) for k in ("k_ml", "k_libra", "n11"))
            sM, sA, pr = eps_fit(r, kM, n), eps_fit(r, kA, n), paired(r, kA, kM, n11, n)
            series.append({"patch": p, "basis": b, "eps_ml": sM["eps"], "se_ml": sM["se"], "chi2_red_ml": sM["chi2_red"],
                           "eps_libra": sA["eps"], "se_libra": sA["se"], "chi2_red_libra": sA["chi2_red"],
                           "se_ml_no_inflation": sM["se_no_inflation"], "se_libra_no_inflation": sA["se_no_inflation"],
                           **{f"paired_{k}": v for k, v in pr.items()}})
    with open(out_path(os.path.join(out_dir, "series.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(series[0].keys()))
        w.writeheader()
        w.writerows(series)

    def mean_se(key, sekey):
        return float(np.mean([s[key] for s in series])), float(math.sqrt(sum(s[sekey] ** 2 for s in series)) / S)

    res = {"scope": {"series": S, "rounds": ROUNDS, "leaves": len(leaves), "shots_per_leaf": sorted({x["n"] for x in leaves})},
           "eps_mean": {}, "g": {}, "pooled_ratio": {}, "checks": {}, "reported": REPORTED}
    for dname, key in (("libra", "eps_libra"), ("ml_rl", "eps_ml")):
        m, se = mean_se(key, key.replace("eps", "se"))
        m2, se2 = mean_se(key, key.replace("eps", "se") + "_no_inflation")
        res["eps_mean"][dname] = {"mean": m, "se": se, "se_no_inflation": se2,
                                  "n_series_chi2_red_above_1": int(sum(s["chi2_red_" + dname.split("_")[0]] > 1 for s in series))}
    lib = res["eps_mean"]["libra"]
    for ref in ("table_S1_libra_eps", "decoder_libra_eps"):
        v, s = REPORTED[ref]
        lib["z_vs_" + ref] = (lib["mean"] - v) / math.sqrt(lib["se"] ** 2 + s ** 2)
    v, s = REPORTED["decoder_ml_rl_eps"]
    res["eps_mean"]["ml_rl"]["diff_vs_decoder"] = res["eps_mean"]["ml_rl"]["mean"] - v

    D = float(np.mean([s["paired_d"] for s in series]))
    M = float(np.mean([s["paired_eps_ml"] for s in series]))
    g = D / M
    for tag in ("", "_inflated"):
        vD = sum(s["paired_var_d" + tag] for s in series) / S ** 2
        vM = sum(s["paired_var_eps_ml" + tag] for s in series) / S ** 2
        cDM = sum(s["paired_cov_d_eps_ml" + tag] for s in series) / S ** 2
        res["g"]["se" + (tag or "_no_inflation")] = math.sqrt(max(vD - 2 * g * cDM + g * g * vM, 0)) / M
    res["g"].update(g=g, mean_d=D, mean_eps_ml=M, mean_eps_libra_paired=float(np.mean([s["paired_eps_a_paired"] for s in series])),
                    z=g / res["g"]["se_inflated"], g_unpaired_from_separate_fits=(lib["mean"] - res["eps_mean"]["ml_rl"]["mean"])
                    / res["eps_mean"]["ml_rl"]["mean"], series_with_d_positive=int(sum(s["paired_d"] > 0 for s in series)))

    # (d) pooled count ratio, per-shot delta method stratified by leaf
    n = np.array([x["n"] for x in leaves], float)
    kA, kM, n11 = (np.array([x[k] for x in leaves], float) for k in ("k_libra", "k_ml", "n11"))
    pA, pM, p11 = kA / n, kM / n, n11 / n
    KA, KM = kA.sum(), kM.sum()
    vA, vM, cAM = np.sum(n * pA * (1 - pA)), np.sum(n * pM * (1 - pM)), np.sum(n * (p11 - pA * pM))
    q = KA / KM
    n10, n01 = sum(x["n10"] for x in leaves), sum(x["n01"] for x in leaves)
    res["pooled_ratio"] = {"sum_k_libra": int(KA), "sum_k_ml": int(KM), "ratio": q - 1,
                           "se": math.sqrt(vA - 2 * q * cAM + q * q * vM) / KM,
                           "discordant_libra_wrong_ml_right": int(n10), "discordant_libra_right_ml_wrong": int(n01),
                           "mcnemar_z": (n10 - n01) / math.sqrt(n10 + n01), "both_wrong": int(n11.sum()),
                           "shots": int(n.sum())}

    # (e) checks summed over leaves
    c = res["checks"]
    c["all_shots_decoded_in_order"] = all(x["idx_is_all_shots_in_order"] and x["shots_idx_equal"] for x in leaves)
    c["obs_mismatch_shots"] = sum(x["obs_mismatch"] for x in leaves)
    c["llr_nan"] = sum(x["llr_nan"] for x in leaves)
    c["llr_exactly_zero"] = sum(x["llr_zero"] for x in leaves)
    c["llr_abs_below_1e-6"] = sum(x["llr_abs_below_1e-6"] for x in leaves)
    c["pred_vs_argmax_mismatch_if_llr_ln_p1_over_p0"] = sum(x["pred_vs_argmax_mismatch_if_llr_ln_p1_over_p0"] for x in leaves)
    c["pred_vs_argmax_mismatch_if_llr_ln_p0_over_p1"] = sum(x["pred_vs_argmax_mismatch_if_llr_ln_p0_over_p1"] for x in leaves)
    c["shipped_libra_copy_in_shots_npz_mismatch"] = sum((x["shipped_libra_in_shots_npz_mismatch"] or 0) for x in leaves)
    c["b8_padding_zero"] = all(x["b8_padding_zero"] for x in leaves)
    c["libra_dem_equals_corr_matching_rl_dem_md5"] = all(x["dem_md5_libra"] == x["dem_md5_corr_matching_rl"] for x in leaves)
    c["ml_used_that_dem_md5"] = all(x["dem_md5_used_by_ml"] == x["dem_md5_corr_matching_rl"] for x in leaves)
    res["seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "willow_check.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "reported"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
