"""Shot-level bootstrap of the paired gap g and the mean eps (check of the delta-method standard errors).

Within a leaf the shots are exchangeable and the 2x2 table (Libra wrong, ML wrong) is a sufficient statistic for every
estimator in willow_check.py. A bootstrap replicate therefore redraws each leaf's table from a multinomial with the
observed cell frequencies (n shots), then recomputes eps per series, the means and g exactly as willow_check.py does.
This captures binomial shot noise only; the chi2_red inflation of the prereg (excess scatter across round counts) is a
model-level correction that a shot bootstrap cannot reproduce, so compare with the uninflated delta-method se.
Leaves near E = 1/2: at r = 230-250 some series have 1 - 2E within 0.5-2.6 sd of zero, so a replicate can give E >= 1/2,
where ln(1 - 2E) is undefined (the prereg has no rule; the observed data never reach it). In the bootstrap only,
1 - 2E is then clipped at 0.5 / n, which gives that point a weight of about 1/n, i.e. drops it from the fit. The number
of clipped leaf-decoder points is reported.
Usage: nice -n 19 python3 -I bootstrap_g.py OUT_DIR [B] [SEED]   (reads OUT_DIR/leaves.csv, writes OUT_DIR/bootstrap_g.json)
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
import willow_check as wc  # noqa: E402
from willow_check import eps_fit, in_path, out_path, paired  # noqa: E402

CLIPPED = [0]


def y_var_clipped(k, n):
    """willow_check.y_var with 1 - 2E clipped at 0.5 / n (bootstrap replicates only, see the docstring)."""
    n = np.asarray(n, float)
    one = 1 - 2 * np.asarray(k, float) / n
    CLIPPED[0] += int(np.sum(one < 0.5 / n))
    one = np.maximum(one, 0.5 / n)
    E = (1 - one) / 2
    return E, np.log(one), 4 * E * (1 - E) / (n * one ** 2)


def stats(L):
    """eps means and g from per-leaf tables, grouped by series (same arithmetic as willow_check.main)."""
    ser = {}
    for x in L:
        ser.setdefault((x["patch"], x["basis"]), []).append(x)
    eM, eA, d, epsM_p = [], [], [], []
    for rows in ser.values():
        rows = sorted(rows, key=lambda x: x["r"])
        r = np.array([x["r"] for x in rows], float)
        n = np.array([x["n"] for x in rows], float)
        kM = np.array([x["n11"] + x["n01"] for x in rows], float)
        kA = np.array([x["n11"] + x["n10"] for x in rows], float)
        n11 = np.array([x["n11"] for x in rows], float)
        eM.append(eps_fit(r, kM, n)["eps"])
        eA.append(eps_fit(r, kA, n)["eps"])
        p = paired(r, kA, kM, n11, n)
        d.append(p["d"])
        epsM_p.append(p["eps_ml"])
    return float(np.mean(eA)), float(np.mean(eM)), float(np.mean(d) / np.mean(epsM_p))


def main(out_dir, B=2000, seed=12345):
    L = []
    with open(in_path(os.path.join(out_dir, "leaves.csv"))) as f:
        for row in csv.DictReader(f):
            L.append({"patch": row["patch"], "basis": row["basis"], "r": int(row["r"]), "n": int(row["n"]),
                      "n11": int(row["n11"]), "n10": int(row["n10"]), "n01": int(row["n01"]), "n00": int(row["n00"])})
    base = stats(L)  # observed data: the unclipped estimator of willow_check.py
    wc.y_var = y_var_clipped  # eps_fit and paired look y_var up in willow_check's namespace at call time
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        Lb = []
        for x in L:
            c = rng.multinomial(x["n"], np.array([x["n11"], x["n10"], x["n01"], x["n00"]], float) / x["n"])
            Lb.append(dict(x, n11=int(c[0]), n10=int(c[1]), n01=int(c[2]), n00=int(c[3])))
        reps.append(stats(Lb))
    reps = np.array(reps)
    out = {"B": B, "seed": seed, "estimate": {"eps_libra": base[0], "eps_ml": base[1], "g": base[2]},
           "bootstrap_sd": {"eps_libra": float(reps[:, 0].std(ddof=1)), "eps_ml": float(reps[:, 1].std(ddof=1)),
                            "g": float(reps[:, 2].std(ddof=1))},
           "bootstrap_mean_minus_estimate": {"eps_libra": float(reps[:, 0].mean() - base[0]),
                                             "eps_ml": float(reps[:, 1].mean() - base[1]), "g": float(reps[:, 2].mean() - base[2])},
           "g_percentile_95": [float(np.percentile(reps[:, 2], 2.5)), float(np.percentile(reps[:, 2], 97.5))],
           "clipped_leaf_points_per_replicate": CLIPPED[0] / 2 / B,  # y_var runs twice per point (eps fit, paired fit)
           "replicates": int(B)}
    with open(out_path(os.path.join(out_dir, "bootstrap_g.json")), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000, int(sys.argv[3]) if len(sys.argv) > 3 else 12345)
