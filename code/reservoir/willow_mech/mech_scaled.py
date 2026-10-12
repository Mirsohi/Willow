"""EXPLORATORY (not pre-registered), extra to task 1: is the data only a cleaner copy of the RL DEM, or is its shape
different too? Compares the data with the leaf's RL DEM after every mechanism probability is multiplied by one factor c,
chosen so that the scaled DEM's expected number of detection events equals the data's mean (exact expectation:
P(detector fires) = (1 - prod_e (1 - 2 c p_e)) / 2 over the mechanisms e that touch it). Under "the data are the DEM
with uniformly lower rates", the scaled DEM matches the data's variance, tails and bursts as well. No decoding is done.

For each W5 leaf (d = 3, r in {10, 13, 30, 50}): 10^6 samples of the scaled DEM (seeded by the leaf name); event count
N, the largest number of events in one layer B, and the two windows of mech_leaves.py; compared with the data's values
stored by mech_leaves.py in LEAVES.
Usage: nice -n 19 python3 -I mech_scaled.py ARCHIVE LEAVES OUT_DIR [N_DEM]   (writes OUT_DIR/scaled_leaves.csv and
       OUT_DIR/scaled.json through qrc.outpaths)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import zlib

import numpy as np
import stim

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from mech_leaves import BASES, PATCHES, RL, W5, features, layer_counts, windows  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

LABEL = "EXPLORATORY, not pre-registered (reservoir, 2026-10-09); no multiplicity correction"


def mechanisms(dem):
    ps, dets = [], []
    for ins in dem.flattened():
        if ins.type == "error":
            ps.append(ins.args_copy()[0])
            dets.append(sorted({x.val for x in ins.targets_copy() if x.is_relative_detector_id()}))
    return np.array(ps), dets


def expected_events(c, ps, inc, nd):
    lg = np.zeros(nd)
    np.add.at(lg, inc[1], np.log1p(-2 * c * ps[inc[0]]))
    return float(np.sum((1 - np.exp(lg)) / 2))


def scaled_dem(dem, c):
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type == "error":
            out.append("error", [c * ins.args_copy()[0]], ins.targets_copy())
        else:
            out.append(ins)
    return out


def stats_of(x):
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std()
    z = (x - mu) / sd
    return mu, sd ** 2, float(np.mean(z ** 3)), float(np.mean(z >= 3)), float(np.mean(z >= 4))


def leaf(arch, leaves, patch, basis, r, n_dem):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    dem = stim.DetectorErrorModel.from_file(os.path.join(arch, rel, "decoding_results", RL, "error_model.dem"))
    nd = dem.num_detectors
    d = np.load(os.path.join(leaves, f"{patch}_{basis}_r{r:03d}.npz"))
    Nd = d["N"].astype(float)
    ps, dets = mechanisms(dem)
    inc = (np.array([i for i, ds in enumerate(dets) for _ in ds]), np.array([j for ds in dets for j in ds]))
    target = Nd.mean()
    lo, hi = 0.2, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected_events(mid, ps, inc, nd) < target else (lo, mid)
    c = (lo + hi) / 2
    sdem = scaled_dem(dem, c)
    sampler = sdem.compile_sampler(seed=zlib.crc32(("scaled/" + rel).encode()))
    win = windows(r)
    Ns, Bs, hB = [], [], np.zeros(9, np.int64)
    mom = np.zeros(6)
    left = n_dem
    while left:
        k = min(50000, left)
        det, _, _ = sampler.sample(shots=k, bit_packed=True)
        f = features(layer_counts(det, r), r, win)
        Ns.append(f["N"].astype(np.int16))
        hB += np.bincount(f["B"], minlength=9)[:9]
        A, Bw = f["NA"].astype(float), f["NB"].astype(float)
        mom += [A.sum(), Bw.sum(), (A * A).sum(), (Bw * Bw).sum(), (A * Bw).sum(), k]
        left -= k
    Ns = np.concatenate(Ns).astype(float)
    md, vd, skd, t3d, t4d = stats_of(Nd)
    ms, vs, sks, t3s, t4s = stats_of(Ns)
    q99, q999 = np.quantile(Ns, 0.99, method="inverted_cdf"), np.quantile(Ns, 0.999, method="inverted_cdf")
    hBd = np.bincount(d["B"].astype(np.int64), minlength=9)[:9]
    n_m = mom[5]
    ea, eb = mom[0] / n_m, mom[1] / n_m
    row = {"patch": patch, "basis": basis, "r": r, "c": c, "expected_events_scaled": expected_events(c, ps, inc, nd),
           "mean_data": md, "mean_scaled": ms, "var_data": vd, "var_scaled": vs, "var_ratio": vd / vs,
           "skew_data": skd, "skew_scaled": sks, "z_ge3_data": t3d, "z_ge3_scaled": t3s, "z_ge4_data": t4d, "z_ge4_scaled": t4s,
           "q99_scaled": float(q99), "q999_scaled": float(q999),
           "above_q99_data": float(np.mean(Nd > q99)), "above_q99_scaled": float(np.mean(Ns > q99)),
           "above_q999_data": float(np.mean(Nd > q999)), "above_q999_scaled": float(np.mean(Ns > q999)),
           "n_data": len(Nd), "n_scaled": len(Ns), "win_cov_scaled": mom[4] / n_m - ea * eb}
    for k in (3, 4, 5, 6):
        row[f"B_ge{k}_data"] = float(hBd[k:].sum() / hBd.sum())
        row[f"B_ge{k}_scaled"] = float(hB[k:].sum() / hB.sum())
    return row


def main(arch, leaves, out_dir, n_dem=1_000_000):
    rows = []
    for r in W5:
        for p in PATCHES:
            for b in BASES:
                rows.append(leaf(arch, leaves, p, b, r, n_dem))
                x = rows[-1]
                print(p, b, r, f"c = {x['c']:.3f}, var ratio {x['var_ratio']:.3f}, >q999 {x['above_q999_data']:.4f} vs "
                      f"{x['above_q999_scaled']:.4f}, B>=6 {x['B_ge6_data']:.4f} vs {x['B_ge6_scaled']:.4f}", flush=True)
    with open(out_path(os.path.join(out_dir, "scaled_leaves.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"label": LABEL, "n_dem_per_leaf": n_dem, "by_r": {}}
    for key in [str(r) for r in W5] + ["all"]:
        R = rows if key == "all" else [x for x in rows if str(x["r"]) == key]
        o = {"leaves": len(R), "c_mean": float(np.mean([x["c"] for x in R])),
             "c_min": float(min(x["c"] for x in R)), "c_max": float(max(x["c"] for x in R)),
             "var_ratio_median": float(np.median([x["var_ratio"] for x in R])),
             "var_ratio_geomean": float(math.exp(np.mean([math.log(x["var_ratio"]) for x in R]))),
             "leaves_var_ratio_above_1": int(sum(x["var_ratio"] > 1 for x in R)),
             "skew_data_mean": float(np.mean([x["skew_data"] for x in R])),
             "skew_scaled_mean": float(np.mean([x["skew_scaled"] for x in R]))}
        for t in ("above_q99", "above_q999", "z_ge3", "z_ge4", "B_ge4", "B_ge5", "B_ge6"):
            O = sum(x[f"{t}_data"] * x["n_data"] for x in R)
            E = sum(x[f"{t}_scaled"] * x["n_data"] for x in R)
            o[t] = {"data_share": O / sum(x["n_data"] for x in R), "scaled_share": E / sum(x["n_data"] for x in R),
                    "ratio_pooled": O / E if E else float("nan"), "ratio_se": (O / E) / math.sqrt(max(O, 1)) if E else float("nan"),
                    "ratio_median": float(np.median([x[f"{t}_data"] / x[f"{t}_scaled"] for x in R if x[f"{t}_scaled"] > 0]))}
        res["by_r"][key] = o
    with open(out_path(os.path.join(out_dir, "scaled.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["by_r"]["all"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000)
