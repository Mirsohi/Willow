"""Independent check of P1's d = 5 replication and of Lambda (notes/decoder/willow_prereg.md sections 2, 4, 5, 6 and
section 9 entries 7, 8, 10, 11). Brief from main, 10 Oct 2026, after the queue's final analysis step (19:30Z).

reservoir, 2026-10-10. Written from the pre-registration; the estimators are those of reservoir's P1 check
(willow_check/willow_check.py, from which y_var and read_obs_b8 are imported; wls, eps_fit and paired are copied with
one change, a guard for fits with 2 rounds, which have no chi2 and so no inflation). decoder's analysis code is not
read; its analysis outputs (results/decoder/willow/analysis/) are read only by compare_d5.py, after these outputs are
stored.

Inputs (read only; python3 -I because the archive is downloaded data):
  archive  ARCHIVE/d<d>_at_<patch>/<basis>/r<NN>/obs_flips_actual.b8 and
           decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8 (all 50,000 shots)
  decoder  DECODE/d5_at_<patch>/<basis>/r<NN>/rl_s5/ml_tau2_1e-6-1e-8_0.1.npz (idx, pred, llr, llr_t1, redecoded),
           shots.npz and summary.json (DEM md5); DECODE/d3_at_<patch>/<basis>/r<NN>/rl/ml_exact.npz and summary.json
  K2       K2_DIR/<arch>/d5_at_<patch>/<basis>/r<NN>/rl_s5/extra_rung.json (per-leaf verdicts as found when this runs;
           interim) and K2_DIR/summary.json (the list made at 18:37:33Z)
Scope and estimators (prereg sections 2, 4, 5):
  d = 5   4 patches x {X, Z} = 8 series; ML(RL) = the tau ladder on W5 = r in {10, 13, 30, 50}, every 5th shot;
          Libra on r in {10, 30, 50} (none at 13), on the same shots (pairing by shot).
  g_5     per series: WLS of z_r = y_r(Libra) - y_r(ML) on r, Var(z_r) from the leaf's 2x2 table; eps_A from
          slope_ML + slope_z; g = mean_s(eps_A - eps_ML) / mean_s(eps_ML); delta-method SE with Cov(slope_ML,
          slope_z) and the chi2_red inflation of each fit (section 9 entry 7); also the SE without the covariance terms
          (the frozen formula's convention, entry 7) and without inflation. Replicated if g_5 has g_3's sign and its
          90% CI excludes 0.
  Lambda  Lambda_3/5 = mean_s eps (d = 3) / mean_s eps (d = 5) on W5 (ML(RL): 10, 13, 30, 50; Libra: 10, 30, 50);
          paired ratio Lambda_Libra / Lambda_ML = (1 + g_3) / (1 + g_5), both on {10, 30, 50}.
  counts  (sum k_Libra - sum k_ML) / sum k_ML over leaves, per-shot delta method stratified by leaf.
Sensitivity (requested by main): without the leaves that K2 still fails; without those and the leaves with no K2
verdict yet; leave one patch out; leave one r out; counts by r, patch and basis. Flags: window, missing Libra rounds,
truncation (tau1 -> final and the extra rung's tau2 -> tau3 error counts), the every-5th-shot subset (Libra's error
rate on it against all shots), NaN and tied posteriors, DEM files.
Usage: nice -n 10 python3 -I d5_check.py ARCHIVE DECODE K2_DIR OUT_DIR -> OUT_DIR/d5_leaves.csv, d5_series.csv,
       d5_check.json (through qrc.outpaths)
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

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_check"))
sys.path.insert(0, os.path.join(HERE, ".."))
from willow_check import read_obs_b8, y_var  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

ARCH = "google_105Q_surface_code_d3_d5_d7"
P3 = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
BASES = ["X", "Z"]
W5 = [10, 13, 30, 50]
WL = [10, 30, 50]  # W5 rounds with Libra predictions
LIBRA = "libra_decoder_with_rl_optimized_prior"
CM_RL = "correlated_matching_decoder_with_rl_optimized_prior"
ML5 = "ml_tau2_1e-6-1e-8_0.1.npz"
Z90, Z95 = 1.6448536269514722, 1.959963984540054


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def rel_of(d, patch, basis, r):
    return f"d{d}_at_{patch}/{basis}/r{r:02d}"


# ---- estimators: willow_check.py's, with a guard for 2-round fits (no chi2_red, no inflation) ----
def wls(x, y, var):
    x, y, var = (np.asarray(v, float) for v in (x, y, var))
    w = 1.0 / var
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    coef = cov @ (X.T * w)
    beta = coef @ y
    chi2 = float(np.sum(w * (y - X @ beta) ** 2))
    return {"a": float(beta[0]), "b": float(beta[1]), "var_b": float(cov[1, 1]), "c_b": coef[1],
            "chi2_red": chi2 / (len(x) - 2) if len(x) > 2 else None}


def infl(f):
    return max(1.0, f["chi2_red"]) if f["chi2_red"] is not None else 1.0


def eps_fit(r, k, n):
    E, y, v = y_var(k, n)
    f = wls(r, y, v)
    eb = math.exp(f["b"])
    return {"eps": (1 - eb) / 2, "se": eb / 2 * math.sqrt(f["var_b"] * infl(f)),
            "se_no_inflation": eb / 2 * math.sqrt(f["var_b"]), "chi2_red": f["chi2_red"], "slope": f["b"],
            "rounds": [int(v) for v in r]}


def paired(r, kA, kM, n11, n):
    EA, yA, vA = y_var(kA, n)
    EM, yM, vM = y_var(kM, n)
    n = np.asarray(n, float)
    p11 = np.asarray(n11, float) / n
    cAM = 4 * (p11 - EA * EM) / (n * (1 - 2 * EA) * (1 - 2 * EM))  # Cov(y_A, y_ML)
    vz = vA + vM - 2 * cAM
    fM, fz = wls(r, yM, vM), wls(r, yA - yM, vz)
    cov_b = float(np.sum(fM["c_b"] * fz["c_b"] * (cAM - vM)))  # Cov(slope_ML, slope_z)
    bM, bz = fM["b"], fz["b"]
    epsM, epsA = (1 - math.exp(bM)) / 2, (1 - math.exp(bM + bz)) / 2
    gM = np.array([-math.exp(bM) / 2, 0.0])
    gd = np.array([(math.exp(bM) - math.exp(bM + bz)) / 2, -math.exp(bM + bz) / 2])
    out = {"eps_ml": epsM, "eps_a_paired": epsA, "d": epsA - epsM, "slope_ml": bM, "slope_z": bz,
           "chi2_red_ml": fM["chi2_red"], "chi2_red_z": fz["chi2_red"],
           "corr_slopes": cov_b / math.sqrt(fM["var_b"] * fz["var_b"]), "rounds": [int(v) for v in r]}
    for tag, iM, iz in (("", 1.0, 1.0), ("_inflated", infl(fM), infl(fz))):
        V = np.array([[fM["var_b"] * iM, cov_b * math.sqrt(iM * iz)], [cov_b * math.sqrt(iM * iz), fz["var_b"] * iz]])
        out["var_d" + tag], out["var_eps_ml" + tag], out["cov_d_eps_ml" + tag] = gd @ V @ gd, gM @ V @ gM, gd @ V @ gM
        # the frozen formula's convention (entry 7): only d(delta)/d(beta) in var_d, and no covariance
        out["var_d_frozen" + tag] = gd[1] ** 2 * fz["var_b"] * iz
    return out


def g_stat(prs):
    """g and its SEs from a list of paired() results (one per series)."""
    S = len(prs)
    D = float(np.mean([p["d"] for p in prs]))
    M = float(np.mean([p["eps_ml"] for p in prs]))
    g = D / M
    o = {"g": g, "series": S, "mean_d": D, "mean_eps_ml": M,
         "mean_eps_libra_paired": float(np.mean([p["eps_a_paired"] for p in prs])),
         "series_with_d_positive": int(sum(p["d"] > 0 for p in prs))}
    for tag in ("", "_inflated"):
        vD = sum(p["var_d" + tag] for p in prs) / S ** 2
        vM = sum(p["var_eps_ml" + tag] for p in prs) / S ** 2
        cDM = sum(p["cov_d_eps_ml" + tag] for p in prs) / S ** 2
        o["se" + (tag or "_no_inflation")] = math.sqrt(max(vD - 2 * g * cDM + g * g * vM, 0.0)) / M
        vDf = sum(p["var_d_frozen" + tag] for p in prs) / S ** 2
        o["se_frozen_formula" + (tag or "_no_inflation")] = math.sqrt(vDf + g * g * vM) / M
    se = o["se_inflated"]
    o.update(z=g / se, p_two_sided=math.erfc(abs(g / se) / math.sqrt(2)),
             ci90=[g - Z90 * se, g + Z90 * se], ci95=[g - Z95 * se, g + Z95 * se])
    return o


def pooled(L):
    """(sum k_Libra - sum k_ML) / sum k_ML over leaves L, per-shot delta method stratified by leaf."""
    n = np.array([x["n"] for x in L], float)
    kA, kM, n11 = (np.array([x[k] for x in L], float) for k in ("k_libra", "k_ml", "n11"))
    pA, pM, p11 = kA / n, kM / n, n11 / n
    KA, KM = kA.sum(), kM.sum()
    vA, vM, cAM = np.sum(n * pA * (1 - pA)), np.sum(n * pM * (1 - pM)), np.sum(n * (p11 - pA * pM))
    q = KA / KM
    se = math.sqrt(max(vA - 2 * q * cAM + q * q * vM, 0.0)) / KM
    n10, n01 = int(sum(x["n10"] for x in L)), int(sum(x["n01"] for x in L))
    return {"leaves": len(L), "shots": int(n.sum()), "k_libra": int(KA), "k_ml": int(KM), "ratio": q - 1, "se": se,
            "ci95": [q - 1 - Z95 * se, q - 1 + Z95 * se], "libra_wrong_ml_right": n10, "libra_right_ml_wrong": n01}


def mean_eps(fits):
    m = float(np.mean([f["eps"] for f in fits]))
    se = math.sqrt(sum(f["se"] ** 2 for f in fits)) / len(fits)
    return m, se


def ratio_se(a, sa, b, sb):
    q = a / b
    return q, q * math.sqrt((sa / a) ** 2 + (sb / b) ** 2)


# ---- leaves ----
def k2_verdicts(k2_dir):
    out = {}
    root = os.path.join(k2_dir, ARCH)
    for p in P5:
        for b in BASES:
            for r in W5:
                f = os.path.join(root, rel_of(5, p, b, r), "rl_s5", "extra_rung.json")
                if os.path.exists(f):
                    j = json.load(open(f))
                    if "verdict" in j:
                        e = j.get("tau3") if isinstance(j.get("tau3"), dict) else j.get("extra", {})
                        out[rel_of(5, p, b, r)] = {"verdict": j["verdict"], "errors_ladder_final": e.get("errors_ladder_final"),
                                                   "errors_with_tau3": e.get("errors_with_tau3"), "n_band": e.get("n_band"),
                                                   "decisions_changed_tau2_tau3": e.get("decisions_changed_tau2_tau3"),
                                                   "max_abs_dP_tau2_tau3": e.get("max_abs_dP_tau2_tau3"),
                                                   "ladder_errors_tau1": j.get("ladder", {}).get("errors_tau1"),
                                                   "ladder_errors_final": j.get("ladder", {}).get("errors_final")}
    return out


def leaf5(arch, dec, patch, basis, r, k2):
    rel = rel_of(5, patch, basis, r)
    a_dir, d_dir = os.path.join(arch, rel), os.path.join(dec, ARCH, rel, "rl_s5")
    act, pad_a = read_obs_b8(os.path.join(a_dir, "obs_flips_actual.b8"))
    ml = np.load(os.path.join(d_dir, ML5), allow_pickle=False)
    sh = np.load(os.path.join(d_dir, "shots.npz"), allow_pickle=False)
    sm = json.load(open(os.path.join(d_dir, "summary.json")))
    idx = ml["idx"].astype(np.int64)
    pred = ml["pred"].reshape(len(idx), -1)[:, 0].astype(np.uint8)
    llr, llr1 = ml["llr"].astype(np.float64), ml["llr_t1"].astype(np.float64)
    o = act[idx]
    rec = {"d": 5, "patch": patch, "basis": basis, "r": r, "n_archive": int(act.size), "n": int(idx.size),
           "idx_is_every_5th_shot": bool(np.array_equal(idx, np.arange(0, act.size, 5))),
           "shots_idx_equal": bool(np.array_equal(sh["idx"].astype(np.int64), idx)),
           "obs_mismatch": int(np.sum(sh["obs"].astype(np.uint8) != o)), "b8_padding_zero": pad_a,
           "llr_nan": int(np.isnan(llr).sum()), "llr_zero": int(np.sum(llr == 0)),
           "llr_t1_nan": int(np.isnan(llr1).sum())}
    fin = np.isfinite(llr) & (llr != 0)
    rec["pred_vs_llr_gt_0_mismatch"] = int(np.sum(pred[fin] != (llr[fin] > 0)))
    rec["dem_md5_used"] = sm.get("dem_md5")
    rec["dem_md5_ok"] = sm.get("dem_md5") == md5(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    ml_wrong = pred != o
    rec["k_ml"] = int(ml_wrong.sum())
    band = ml["redecoded"].astype(bool)
    pred1 = (np.nan_to_num(llr1, nan=0.0) > 0).astype(np.uint8)
    rec["band_shots"] = int(band.sum())
    rec["k_ml_tau1"] = int(np.sum(pred1 != o))
    rec["decisions_changed_tau1_final"] = int(np.sum(pred1 != pred))
    rec["decisions_changed_outside_band"] = int(np.sum((pred1 != pred) & ~band))
    lp = os.path.join(a_dir, "decoding_results", LIBRA, "obs_flips_predicted.b8")
    rec["has_libra"] = os.path.exists(lp)
    if rec["has_libra"]:
        lib, pad_l = read_obs_b8(lp)
        rec["b8_padding_zero"] = pad_a and pad_l
        rec["libra_dem_equals_cm_rl_dem"] = (md5(os.path.join(a_dir, "decoding_results", LIBRA, "error_model.dem"))
                                             == md5(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem")))
        names = [str(x) for x in sh["shipped_names"]]
        li = [i for i, s in enumerate(names) if LIBRA in s]
        rec["shipped_libra_copy_mismatch"] = int(np.sum(sh["shipped"][li[0]] != lib[idx])) if li else None
        lw = lib[idx] != o
        rec.update(k_libra=int(lw.sum()), n11=int(np.sum(lw & ml_wrong)), n10=int(np.sum(lw & ~ml_wrong)),
                   n01=int(np.sum(~lw & ml_wrong)), n00=int(np.sum(~lw & ~ml_wrong)),
                   k_libra_all_shots=int(np.sum(lib != act)), k_libra_other_shots=int(np.sum(lib != act)) - int(lw.sum()))
    v = k2.get(rel)
    rec["k2_verdict"] = v["verdict"] if v else "none_yet"
    rec["k2_status"] = "none_yet" if not v else ("pass" if v["verdict"] == "pass" else "fail")
    if v:
        rec.update({f"k2_{k}": v[k] for k in ("errors_ladder_final", "errors_with_tau3", "n_band",
                                              "decisions_changed_tau2_tau3", "max_abs_dP_tau2_tau3")})
    return rec


def leaf3(arch, dec, patch, basis, r):
    rel = rel_of(3, patch, basis, r)
    a_dir, d_dir = os.path.join(arch, rel), os.path.join(dec, ARCH, rel, "rl")
    act, _ = read_obs_b8(os.path.join(a_dir, "obs_flips_actual.b8"))
    ml = np.load(os.path.join(d_dir, "ml_exact.npz"), allow_pickle=False)
    idx = ml["idx"].astype(np.int64)
    pred = ml["pred"].reshape(len(idx), -1)[:, 0].astype(np.uint8)
    o = act[idx]
    rec = {"d": 3, "patch": patch, "basis": basis, "r": r, "n_archive": int(act.size), "n": int(idx.size),
           "idx_is_all_shots": bool(np.array_equal(idx, np.arange(act.size))),
           "dem_md5_ok": json.load(open(os.path.join(d_dir, "summary.json"))).get("dem_md5")
           == md5(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))}
    ml_wrong = pred != o
    rec["k_ml"] = int(ml_wrong.sum())
    lp = os.path.join(a_dir, "decoding_results", LIBRA, "obs_flips_predicted.b8")
    rec["has_libra"] = os.path.exists(lp)
    if rec["has_libra"]:
        lib, _ = read_obs_b8(lp)
        lw = lib[idx] != o
        rec.update(k_libra=int(lw.sum()), n11=int(np.sum(lw & ml_wrong)), n10=int(np.sum(lw & ~ml_wrong)),
                   n01=int(np.sum(~lw & ml_wrong)), n00=int(np.sum(~lw & ~ml_wrong)))
    return rec


def series_g(leaves, rounds_allowed, patches):
    """Per-series paired fits over the allowed rounds (series with fewer than 2 rounds are skipped) and g."""
    prs, per, skipped = [], [], []
    for p in patches:
        for b in BASES:
            L = sorted([x for x in leaves if x["patch"] == p and x["basis"] == b and x["r"] in rounds_allowed
                        and x.get("has_libra")], key=lambda x: x["r"])
            if len(L) < 2:
                skipped.append(f"{p}/{b} ({len(L)} round)")
                continue
            r = [x["r"] for x in L]
            n, kA, kM, n11 = ([x[k] for x in L] for k in ("n", "k_libra", "k_ml", "n11"))
            pr = paired(r, kA, kM, n11, n)
            prs.append(pr)
            per.append({"patch": p, "basis": b, "rounds": r, "g_series": pr["d"] / pr["eps_ml"], "eps_ml": pr["eps_ml"],
                        "eps_libra_paired": pr["eps_a_paired"], "chi2_red_z": pr["chi2_red_z"]})
    if not prs:
        return {"skipped": skipped}
    o = g_stat(prs)
    o["skipped_series"] = skipped
    o["per_series"] = per
    return o


def main(arch, dec, k2_dir, out_dir):
    t0 = time.time()
    k2 = k2_verdicts(k2_dir)
    k2s = json.load(open(os.path.join(k2_dir, "summary.json")))
    L5 = [leaf5(arch, dec, p, b, r, k2) for p in P5 for b in BASES for r in W5]
    print(f"d = 5: {len(L5)} leaves read ({time.time() - t0:.0f} s)", flush=True)
    L3 = [leaf3(arch, dec, p, b, r) for p in P3 for b in BASES for r in W5]
    print(f"d = 3: {len(L3)} leaves read ({time.time() - t0:.0f} s)", flush=True)
    keys = []
    for x in L5 + L3:
        for k in x:
            if k not in keys:
                keys.append(k)
    with open(out_path(os.path.join(out_dir, "d5_leaves.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(L5 + L3)

    res = {"label": "reservoir, independent check of P1's d = 5 replication and Lambda, 2026-10-10 (interim: K2 verdicts "
                    "as found at run time)", "k2": {}, "checks": {}}
    # K2 status as read now, and the summary file's lists
    v5 = {rel_of(5, x["patch"], x["basis"], x["r"]): x["k2_status"] for x in L5}
    res["k2"] = {"read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "pass": sorted(k for k, v in v5.items() if v == "pass"),
                 "still_fail": sorted(k for k, v in v5.items() if v == "fail"),
                 "no_verdict_yet": sorted(k for k, v in v5.items() if v == "none_yet"),
                 "summary_json_made": k2s.get("made"),
                 "summary_json_still_failing": [s.split(ARCH + "/")[-1] for s in k2s.get("still_failing", [])],
                 "summary_json_pending": [s.split(ARCH + "/")[-1] for s in k2s.get("pending_or_not_run", [])]}

    # checks
    c = res["checks"]
    c["d5_every_5th_shot"] = all(x["idx_is_every_5th_shot"] and x["shots_idx_equal"] for x in L5)
    c["d3_all_shots"] = all(x["idx_is_all_shots"] for x in L3)
    c["d5_obs_mismatch"] = sum(x["obs_mismatch"] for x in L5)
    c["d5_llr_nan"] = sum(x["llr_nan"] for x in L5)
    c["d5_llr_t1_nan"] = sum(x["llr_t1_nan"] for x in L5)
    c["d5_llr_zero"] = sum(x["llr_zero"] for x in L5)
    c["d5_pred_vs_llr_gt_0_mismatch"] = sum(x["pred_vs_llr_gt_0_mismatch"] for x in L5)
    c["dem_md5_ok_all"] = all(x["dem_md5_ok"] for x in L5 + L3)
    c["d5_libra_dem_equals_cm_rl_dem"] = all(x.get("libra_dem_equals_cm_rl_dem", True) for x in L5)
    c["d5_shipped_libra_copy_mismatch"] = sum(x.get("shipped_libra_copy_mismatch") or 0 for x in L5)
    c["b8_padding_zero"] = all(x["b8_padding_zero"] for x in L5)
    c["d5_libra_rounds"] = sorted({x["r"] for x in L5 if x["has_libra"]})
    c["d3_libra_rounds_in_W5"] = sorted({x["r"] for x in L3 if x["has_libra"]})
    c["k2_ladder_final_equals_k_ml"] = all(x.get("k2_errors_ladder_final") in (None, x["k_ml"]) for x in L5)

    # 1. g_5 on the pre-registered sample, and the pooled counts
    lib5 = [x for x in L5 if x["has_libra"]]
    res["g5"] = series_g(L5, WL, P5)
    res["pooled_d5"] = pooled(lib5)
    g5 = res["g5"]
    # 2. sensitivity
    fail = set(res["k2"]["still_fail"])
    nov = set(res["k2"]["no_verdict_yet"])
    keep1 = [x for x in L5 if rel_of(5, x["patch"], x["basis"], x["r"]) not in fail]
    keep2 = [x for x in keep1 if rel_of(5, x["patch"], x["basis"], x["r"]) not in nov]
    sens = {}
    for tag, L in (("without_k2_failing", keep1), ("k2_pass_only", keep2)):
        sens[tag] = {"g5": series_g(L, WL, P5), "pooled": pooled([x for x in L if x["has_libra"]]),
                     "leaves_with_libra": sorted(rel_of(5, x["patch"], x["basis"], x["r"]) for x in L if x["has_libra"])}
    # the same leaves for both: pooled counts on the 24 leaves split by K2 status
    sens["pooled_by_k2_status"] = {s: pooled([x for x in lib5 if x["k2_status"] == s]) if any(
        x["k2_status"] == s for x in lib5) else None for s in ("pass", "fail", "none_yet")}
    fit_ser = {f"{x['patch']}/{x['basis']}" for x in sens["k2_pass_only"]["g5"].get("per_series", [])}
    sens["k2_pass_only_series_all_their_leaves"] = {
        "series": sorted(fit_ser),
        "g5": {k: v for k, v in series_g([x for x in L5 if f"{x['patch']}/{x['basis']}" in fit_ser], WL, P5).items()
               if k != "per_series"},
        "note": "the series that keep 2 rounds after dropping failing and undecided leaves, fitted on all their W5 "
                "Libra rounds: separates the choice of series from the K2 verdicts"}
    sens["leave_one_patch_out"] = {p: {k: series_g(L5, WL, [q for q in P5 if q != p])[k] for k in
                                       ("g", "se_inflated", "ci95")} for p in P5}
    sens["leave_one_r_out"] = {f"r{r}": {k: series_g(L5, [q for q in WL if q != r], P5)[k] for k in
                                         ("g", "se_inflated", "ci95")} for r in WL}
    sens["pooled_by_r"] = {f"r{r}": pooled([x for x in lib5 if x["r"] == r]) for r in WL}
    sens["pooled_by_patch"] = {p: pooled([x for x in lib5 if x["patch"] == p]) for p in P5}
    sens["pooled_by_basis"] = {b: pooled([x for x in lib5 if x["basis"] == b]) for b in BASES}
    sens["pooled_by_series"] = {f"{p}/{b}": pooled([x for x in lib5 if x["patch"] == p and x["basis"] == b])
                                for p in P5 for b in BASES}
    res["sensitivity"] = sens

    # 3. Lambda
    fit = {}
    for d, L, P in ((3, L3, P3), (5, L5, P5)):
        for dec_name, key, rounds in (("ml_rl", "k_ml", W5), ("libra", "k_libra", WL)):
            fs = []
            for p in P:
                for b in BASES:
                    S = sorted([x for x in L if x["patch"] == p and x["basis"] == b and x["r"] in rounds], key=lambda x: x["r"])
                    fs.append(eps_fit([x["r"] for x in S], [x[key] for x in S], [x["n"] for x in S]))
            fit[(d, dec_name)] = fs
    lam = {}
    for dec_name in ("ml_rl", "libra"):
        m3, s3 = mean_eps(fit[(3, dec_name)])
        m5, s5 = mean_eps(fit[(5, dec_name)])
        q, sq = ratio_se(m3, s3, m5, s5)
        lam[dec_name] = {"eps_mean_d3": m3, "se_d3": s3, "eps_mean_d5": m5, "se_d5": s5, "Lambda": q, "se": sq,
                         "ci95": [q - Z95 * sq, q + Z95 * sq],
                         "rounds": W5 if dec_name == "ml_rl" else WL}
    # ML on Libra's rounds, to see the window's effect on Lambda_ML
    fs3 = [eps_fit(*zip(*[(x["r"], x["k_ml"], x["n"]) for x in sorted(L3, key=lambda x: x["r"]) if x["patch"] == p
                          and x["basis"] == b and x["r"] in WL])) for p in P3 for b in BASES]
    fs5 = [eps_fit(*zip(*[(x["r"], x["k_ml"], x["n"]) for x in sorted(L5, key=lambda x: x["r"]) if x["patch"] == p
                          and x["basis"] == b and x["r"] in WL])) for p in P5 for b in BASES]
    m3, s3 = mean_eps(fs3)
    m5, s5 = mean_eps(fs5)
    q, sq = ratio_se(m3, s3, m5, s5)
    lam["ml_rl_on_libra_rounds"] = {"eps_mean_d3": m3, "se_d3": s3, "eps_mean_d5": m5, "se_d5": s5, "Lambda": q, "se": sq,
                                    "rounds": WL}
    g3 = series_g(L3, WL, P3)
    r_ = (1 + g3["g"]) / (1 + g5["g"])
    sr = r_ * math.sqrt((g3["se_inflated"] / (1 + g3["g"])) ** 2 + (g5["se_inflated"] / (1 + g5["g"])) ** 2)
    lam["paired_ratio_libra_over_ml"] = {"g3_on_10_30_50": g3["g"], "g3_se": g3["se_inflated"], "g5": g5["g"],
                                         "g5_se": g5["se_inflated"], "ratio": r_, "se": sr,
                                         "ci95": [r_ - Z95 * sr, r_ + Z95 * sr]}
    lam["unpaired_ratio_libra_over_ml"] = lam["libra"]["Lambda"] / lam["ml_rl"]["Lambda"]
    res["lambda"] = lam
    res["g3_on_10_30_50"] = g3
    res["pooled_d3_on_10_30_50"] = pooled([x for x in L3 if x["has_libra"] and x["r"] in WL])
    res["pooled_d3_by_r"] = {f"r{r}": pooled([x for x in L3 if x["has_libra"] and x["r"] == r]) for r in WL}
    res["replication"] = {"g3_reported_P1": 0.01644, "same_sign_as_g3": g5["g"] > 0, "ci90_excludes_0": g5["ci90"][0] > 0,
                          "replicated": bool(g5["g"] > 0 and g5["ci90"][0] > 0)}

    # 4. flags
    fl = {}
    w = [x for x in lib5]
    fl["subset_every_5th"] = {
        "libra_error_rate_subset": sum(x["k_libra"] for x in w) / sum(x["n"] for x in w),
        "libra_error_rate_other_shots": sum(x["k_libra_other_shots"] for x in w) / sum(x["n_archive"] - x["n"] for x in w),
        "per_leaf_subset_minus_other": {rel_of(5, x["patch"], x["basis"], x["r"]): x["k_libra"] / x["n"]
                                        - x["k_libra_other_shots"] / (x["n_archive"] - x["n"]) for x in w}}
    tr = [x for x in L5 if x.get("k2_errors_with_tau3") is not None]
    fl["truncation"] = {
        "ladder_tau1_to_final_errors": {"sum_k_tau1": sum(x["k_ml_tau1"] for x in L5), "sum_k_final": sum(x["k_ml"] for x in L5),
                                        "decisions_changed": sum(x["decisions_changed_tau1_final"] for x in L5),
                                        "changed_outside_band": sum(x["decisions_changed_outside_band"] for x in L5),
                                        "band_shots": sum(x["band_shots"] for x in L5)},
        "extra_rung_tau2_to_tau3": {"leaves": len(tr), "sum_k_ladder_final": sum(x["k2_errors_ladder_final"] for x in tr),
                                    "sum_k_tau3": sum(x["k2_errors_with_tau3"] for x in tr),
                                    "decisions_changed": sum(x["k2_decisions_changed_tau2_tau3"] or 0 for x in tr),
                                    "in_libra_window": {
                                        "leaves": sum(1 for x in tr if x["has_libra"]),
                                        "sum_k_ladder_final": sum(x["k2_errors_ladder_final"] for x in tr if x["has_libra"]),
                                        "sum_k_tau3": sum(x["k2_errors_with_tau3"] for x in tr if x["has_libra"])}},
        "per_leaf": {rel_of(5, x["patch"], x["basis"], x["r"]): {
            "k_tau1": x["k_ml_tau1"], "k_final": x["k_ml"], "band": x["band_shots"],
            "k_tau3": x.get("k2_errors_with_tau3"), "k2": x["k2_verdict"]} for x in L5}}
    KA, KM = sum(x["k_libra"] for x in lib5), sum(x["k_ml"] for x in lib5)
    fl["if_every_ml_count_were_1pct_higher"] = {
        "pooled_ratio": KA / (1.01 * KM) - 1,
        "note": "K2 rule 1 lets a leaf's ML count move by max(3, 1%) between rungs; this is the pooled gap if every ML "
                "count were 1% too low"}
    res["flags"] = fl
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "d5_check.json")), "w") as f:
        json.dump(res, f, indent=1)
    ser = []
    for tag, o in (("g5_all", res["g5"]), ("g5_without_k2_failing", sens["without_k2_failing"]["g5"]),
                   ("g5_k2_pass_only", sens["k2_pass_only"]["g5"]), ("g3_on_10_30_50", g3)):
        for s in o.get("per_series", []):
            ser.append({"set": tag, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in s.items()}})
    for (d, dn), fs in fit.items():
        for (p, b), f in zip([(p, b) for p in (P3 if d == 3 else P5) for b in BASES], fs):
            ser.append({"set": f"eps_d{d}_{dn}", "patch": p, "basis": b, "rounds": json.dumps(f["rounds"]),
                        "eps": f["eps"], "se": f["se"], "chi2_red": f["chi2_red"]})
    keys = []
    for s in ser:
        for k in s:
            if k not in keys:
                keys.append(k)
    with open(out_path(os.path.join(out_dir, "d5_series.csv")), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=keys, restval="")
        w_.writeheader()
        w_.writerows(ser)
    brief = {k: res[k] for k in ("checks", "k2", "pooled_d5", "replication", "lambda", "pooled_d3_on_10_30_50")}
    brief["g5"] = {k: v for k, v in g5.items() if k != "per_series"}
    brief["g3_on_10_30_50"] = {k: v for k, v in g3.items() if k != "per_series"}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
