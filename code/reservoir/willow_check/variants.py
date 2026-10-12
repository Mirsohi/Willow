"""Variants of willow_check.py's estimators, to locate small differences from decoder's posted numbers.

  (1) eps of ML(RL) on the prereg's own-window for ML at d = 3: every r >= 10 (14 round counts, r = 13 included),
      against the 13 Libra round counts used for P1. Needs the 18 r = 13 leaves (archive actual flips, decoder's ML).
  (2) The standard error of g under other readings of "delta-method SE" in prereg section 4: with or without the
      chi2_red inflation of each fit, with or without Cov(slope_ML, slope_z), and with or without the uncertainty of
      the denominator.
Usage: nice -n 19 python3 -I variants.py ARCHIVE_ROOT DECODE_ROOT OUT_DIR   (reads OUT_DIR/leaves.csv and series.csv,
       writes OUT_DIR/variants.json)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True  # no __pycache__ in the shared code folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from willow_check import BASES, PATCHES, eps_fit, in_path, out_path, read_obs_b8  # noqa: E402


def ml_count(arch, dec, patch, basis, r):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    act, _ = read_obs_b8(os.path.join(arch, rel, "obs_flips_actual.b8"))
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    idx = ml["idx"].astype(np.int64)
    pred = ml["pred"].reshape(len(idx), -1)[:, 0]
    if not np.array_equal(pred, (ml["llr"] > 0).astype(pred.dtype)):
        raise SystemExit(f"argmax check fails at {rel}")
    return int(np.sum(pred != act[idx])), int(len(idx))


def main(arch, dec, out_dir):
    L = list(csv.DictReader(open(in_path(os.path.join(out_dir, "leaves.csv")))))
    S = list(csv.DictReader(open(in_path(os.path.join(out_dir, "series.csv")))))
    out = {}
    # (1) ML eps on 14 round counts
    e13, e14, r13 = [], [], {}
    for p in PATCHES:
        for b in BASES:
            rows = sorted([x for x in L if x["patch"] == p and x["basis"] == b], key=lambda x: int(x["r"]))
            k13, n13 = ml_count(arch, dec, p, b, 13)
            r13[f"{p}/{b}"] = k13
            r = np.array([int(x["r"]) for x in rows] + [13], float)
            k = np.array([int(x["k_ml"]) for x in rows] + [k13], float)
            n = np.array([int(x["n"]) for x in rows] + [n13], float)
            o = np.argsort(r)
            e14.append(eps_fit(r[o], k[o], n[o]))
            e13.append(eps_fit(r[o][r[o] != 13], k[o][r[o] != 13], n[o][r[o] != 13]))
    for tag, e in (("13_rounds_libra_window", e13), ("14_rounds_r_ge_10", e14)):
        out[f"eps_ml_rl_{tag}"] = {"mean": float(np.mean([x["eps"] for x in e])),
                                  "se": math.sqrt(sum(x["se"] ** 2 for x in e)) / len(e),
                                  "se_no_inflation": math.sqrt(sum(x["se_no_inflation"] ** 2 for x in e)) / len(e)}
    out["ml_errors_r13"] = r13
    # (2) se of g
    f = {k: np.array([float(s[k]) for s in S]) for k in S[0] if k.startswith("paired_")}
    n = len(S)
    bM, bz = f["paired_slope_ml"], f["paired_slope_z"]
    D, M = f["paired_d"].mean(), f["paired_eps_ml"].mean()
    g = D / M
    # rebuild var(slope_ML) from var(eps_ML): eps_ML = (1 - e^bM)/2 -> var_eps = (e^bM/2)^2 var_bM
    var_bM = f["paired_var_eps_ml"] / (np.exp(bM) / 2) ** 2
    var_bz = f["paired_var_slope_z"]
    rho = f["paired_corr_slopes"]
    iM, iz = np.maximum(1, f["paired_chi2_red_ml"]), np.maximum(1, f["paired_chi2_red_z"])
    gd_M, gd_z = (np.exp(bM) - np.exp(bM + bz)) / 2, -np.exp(bM + bz) / 2
    gM = -np.exp(bM) / 2

    def se(inflate_M, inflate_z, with_cov, with_denominator):
        vM_ = var_bM * (iM if inflate_M else 1)
        vz_ = var_bz * (iz if inflate_z else 1)
        c_ = rho * np.sqrt(var_bM * var_bz) * np.sqrt((iM if inflate_M else 1) * (iz if inflate_z else 1)) * (1 if with_cov else 0)
        var_d = gd_M ** 2 * vM_ + gd_z ** 2 * vz_ + 2 * gd_M * gd_z * c_
        var_e = gM ** 2 * vM_
        cov_de = gd_M * gM * vM_ + gd_z * gM * c_
        vD, vE, cDE = var_d.sum() / n ** 2, var_e.sum() / n ** 2, cov_de.sum() / n ** 2
        v = vD - 2 * g * cDE + g * g * vE if with_denominator else vD
        return math.sqrt(v) / M

    out["g"] = g
    out["se_g"] = {f"inflate_ml={a}, inflate_z={b}, cov={c}, denominator={d}": se(a, b, c, d)
                   for a in (False, True) for b in (False, True) for c in (True, False) for d in (True, False)}
    with open(out_path(os.path.join(out_dir, "variants.json")), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
