"""EXPLORATORY (not pre-registered), extra to tasks 1-3: which detector correlations does the RL DEM over- or
underpredict? A moment check of the DEM's relative rates by fault class, without decoding and without decoder's fits.

For each W5 leaf (d = 3, r in {10, 13, 30, 50}) and each detector pair (i, j) that shares at least one DEM mechanism,
the pair probability
    p_ij = 1/2 - 1/2 sqrt(1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>))
is computed on the data and on fresh samples of the leaf's RL DEM (the same estimator on both, so hyperedges and
overlapping mechanisms enter both sides alike). Pairs are classed by the detectors' own coordinates (stabilizer x, y and
round t, the first three coordinates): timelike (same stabilizer, adjacent rounds: measurement faults), spacelike (same
round, different stabilizers: data faults between rounds), diagonal (adjacent rounds, different stabilizers: faults
inside the stabilizer circuit); pairs touching the first (t = 0) or the last (t = r) layer are classed separately.
Per class: sum of data p_ij / sum of DEM p_ij. Single-detector rates are compared by layer type as well.
Pairs that share no mechanism have zero covariance under the DEM; their mean covariance in the data is reported in
groups: the same stabilizer 2 rounds apart, the same stabilizer 3-5 rounds apart (a leakage-like signature: a leaked
qubit makes its stabilizers fire repeatedly), different stabilizers at most 1 round apart, different stabilizers 2-5
rounds apart, and all pairs at least 6 rounds apart (the reference for shot-level heterogeneity, which adds a small
positive covariance to every pair).
Streaks: per shot, the largest number of events of one stabilizer within 4 consecutive bulk rounds (S). Under the DEM
S >= 3 needs two separate faults on one stabilizer within 4 rounds. Data and DEM histograms go to pairs_leaves.csv;
the data's per-shot S goes to STREAKS/<patch>_<basis>_r<NNN>.npz for mech_summary.py (task 2 by streak).
Unconnected coincidences: per shot, Ku = the number of firing pairs of different stabilizers at most 1 round apart that
share no DEM mechanism (the DEM can explain such a pair only by two separate faults), and Ks2 = the same for one
stabilizer 2 rounds apart. Data per shot go to the same STREAKS file; data and DEM shares of Ku >= 1 to the CSV.
Usage: nice -n 19 python3 -I mech_pairs.py ARCHIVE OUT_DIR STREAKS [N_DEM]   (writes OUT_DIR/pairs.json and
       pairs_leaves.csv through qrc.outpaths; N_DEM defaults to 200,000)
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
from scipy import sparse

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import out_path  # noqa: E402

PATCHES = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
BASES = ["X", "Z"]
W5 = [10, 13, 30, 50]
RL = "correlated_matching_decoder_with_rl_optimized_prior"
CLASSES = ["timelike", "spacelike", "diagonal", "boundary_first", "boundary_last", "other"]
LABEL = "EXPLORATORY, not pre-registered (reservoir, 2026-10-09); no multiplicity correction"


def moments(X, chunk=50000):
    """First and second moments of a 0/1 matrix (rows = shots), accumulated in float64."""
    n, k = X.shape
    s1, s2 = np.zeros(k), np.zeros((k, k))
    for a in range(0, n, chunk):
        x = X[a:a + chunk].astype(np.float32)
        s1 += x.sum(0)
        s2 += (x.T @ x).astype(np.float64)
    return s1 / n, s2 / n


def pij(m1, m2, I, J):
    xi, xj, xij = m1[I], m1[J], m2[I, J]
    num = 4 * (xij - xi * xj)
    den = 1 - 2 * xi - 2 * xj + 4 * xij
    inner = np.clip(1 - num / den, 0, None)
    return 0.5 - 0.5 * np.sqrt(inner)


def streaks(E, slot, r):
    """Largest number of events of one stabilizer within 4 consecutive bulk rounds, per shot (slot[l, k] = detector
    index of stabilizer k in bulk layer l)."""
    X = E[:, slot].astype(np.int8)  # shots x (r - 1) layers x 8 stabilizers
    if X.shape[1] < 4:
        return X.sum(1).max(1)
    c = np.cumsum(X, axis=1, dtype=np.int16)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
    w = c[:, 4:] - c[:, :-4]
    return w.max(axis=(1, 2)).astype(np.int8)


def leaf(arch, patch, basis, r, n_dem, streak_dir=None):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir = os.path.join(arch, rel)
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", RL, "error_model.dem"))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    xy = [tuple(co[i][:2]) for i in range(nd)]
    t = np.array([int(co[i][2]) for i in range(nd)])
    pairs = set()
    for ins in dem.flattened():
        if ins.type == "error":
            ds = sorted({x.val for x in ins.targets_copy() if x.is_relative_detector_id()})
            for a in range(len(ds)):
                for b in range(a + 1, len(ds)):
                    pairs.add((ds[a], ds[b]))
    P = np.array(sorted(pairs))
    I, J = P[:, 0], P[:, 1]
    cls = []
    for i, j in P:
        if t[i] == 0 or t[j] == 0:
            cls.append("boundary_first")
        elif t[i] == r or t[j] == r:
            cls.append("boundary_last")
        elif xy[i] == xy[j] and abs(t[i] - t[j]) == 1:
            cls.append("timelike")
        elif t[i] == t[j]:
            cls.append("spacelike")
        elif abs(t[i] - t[j]) == 1:
            cls.append("diagonal")
        else:
            cls.append("other")
    cls = np.array(cls)
    # bulk-layer slots: the 8 stabilizers in the order of layer 1, located in every bulk layer by their (x, y)
    stabs = [xy[i] for i in range(nd) if t[i] == 1]
    where = {(t[i], xy[i]): i for i in range(nd)}
    slot = np.array([[where[(lay, s_)] for s_ in stabs] for lay in range(1, r)]) if r > 1 else None
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(-1, nd // 8)
    D = np.unpackbits(raw, axis=1, bitorder="little")[:, :nd]
    m1d, m2d = moments(D)
    sampler = dem.compile_sampler(seed=zlib.crc32(("pairs/" + rel).encode()))
    det, _, _ = sampler.sample(shots=n_dem, bit_packed=True)
    S = np.unpackbits(det, axis=1, bitorder="little")[:, :nd]
    m1m, m2m = moments(S)
    pd, pm = pij(m1d, m2d, I, J), pij(m1m, m2m, I, J)
    row = {"patch": patch, "basis": basis, "r": r, "pairs": len(P)}
    for c in CLASSES:
        sel = cls == c
        row[f"{c}_pairs"] = int(sel.sum())
        row[f"{c}_sum_p_data"] = float(pd[sel].sum())
        row[f"{c}_sum_p_dem"] = float(pm[sel].sum())
    lay = {"first": t == 0, "bulk": (t > 0) & (t < r), "last": t == r}
    for k, sel in lay.items():
        row[f"rate_{k}_data"] = float(m1d[sel].mean())
        row[f"rate_{k}_dem"] = float(m1m[sel].mean())
    # pairs that share no mechanism (DEM: zero covariance), in groups; mean covariance per pair
    covd = m2d - np.outer(m1d, m1d)
    covm = m2m - np.outer(m1m, m1m)
    iu = np.triu_indices(nd, 1)
    conn = np.zeros((nd, nd), bool)
    conn[I, J] = True
    dt = np.abs(t[:, None] - t[None, :])[iu]
    same = np.array([[xy[i] == xy[j] for j in range(nd)] for i in range(nd)])[iu]
    bulk = ((t[:, None] > 0) & (t[:, None] < r) & (t[None, :] > 0) & (t[None, :] < r))[iu]
    un = ~conn[iu] & bulk
    groups = {"same_stab_dt2": un & same & (dt == 2), "same_stab_dt3to5": un & same & (dt >= 3) & (dt <= 5),
              "other_dt0to1": un & ~same & (dt <= 1), "other_dt2to5": un & ~same & (dt >= 2) & (dt <= 5),
              "far_dt6plus": un & (dt >= 6)}
    for g, sel in groups.items():
        row[f"un_{g}_pairs"] = int(sel.sum())
        row[f"un_{g}_cov_data"] = float(covd[iu][sel].mean()) if sel.any() else float("nan")
        row[f"un_{g}_cov_dem"] = float(covm[iu][sel].mean()) if sel.any() else float("nan")
        row[f"un_{g}_cov_dem_se"] = float(covm[iu][sel].std() / math.sqrt(sel.sum())) if sel.sum() > 1 else float("nan")
    # unconnected coincidences per shot (sparse pair matrices)
    def coincidences(E, sel):
        U = sparse.csr_matrix((np.ones(int(sel.sum()), np.float32), (iu[0][sel], iu[1][sel])), shape=(nd, nd))
        out = np.empty(len(E), np.float32)
        for a0 in range(0, len(E), 50000):
            Ef = E[a0:a0 + 50000].astype(np.float32)
            Y = Ef @ U
            Y = Y.toarray() if sparse.issparse(Y) else np.asarray(Y)
            out[a0:a0 + 50000] = (Y * Ef).sum(1)
        return np.rint(out).astype(np.int32)

    ku_sel, ks2_sel = groups["other_dt0to1"], groups["same_stab_dt2"]
    Kud, Ks2d = coincidences(D, ku_sel), coincidences(D, ks2_sel)
    Kum, Ks2m = coincidences(S, ku_sel), coincidences(S, ks2_sel)
    row["Ku_ge1_data"], row["Ku_ge1_dem"] = float(np.mean(Kud >= 1)), float(np.mean(Kum >= 1))
    row["Ks2_ge1_data"], row["Ks2_ge1_dem"] = float(np.mean(Ks2d >= 1)), float(np.mean(Ks2m >= 1))
    # streaks (bulk rounds), data per shot and DEM histogram
    if slot is not None:
        sd, sm = streaks(D, slot, r), streaks(S, slot, r)
        hd, hm = np.bincount(sd, minlength=5)[:5], np.bincount(sm, minlength=5)[:5]
        for k in range(1, 5):
            row[f"streak_ge{k}_data"] = float(np.sum(sd >= k) / len(sd))
            row[f"streak_ge{k}_dem"] = float(np.sum(sm >= k) / len(sm))
        row["streak_ge4_data_count"] = int(np.sum(sd >= 4))
        if streak_dir:
            np.savez_compressed(os.path.join(streak_dir, f"{patch}_{basis}_r{r:03d}.npz"), S=sd,
                                Ku=Kud.astype(np.int16), Ks2=Ks2d.astype(np.int16))
    return row


def main(arch, out_dir, streak_dir, n_dem=200_000):
    os.makedirs(streak_dir, exist_ok=True)
    rows = []
    for r in W5:
        for p in PATCHES:
            for b in BASES:
                rows.append(leaf(arch, p, b, r, n_dem, streak_dir))
                print(p, b, r, {c: round(rows[-1][f"{c}_sum_p_data"] / rows[-1][f"{c}_sum_p_dem"], 3)
                                for c in CLASSES if rows[-1][f"{c}_pairs"]}, flush=True)
    keys = list(rows[0].keys())
    with open(out_path(os.path.join(out_dir, "pairs_leaves.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    res = {"label": LABEL, "n_dem_per_leaf": n_dem, "leaves": len(rows), "by_r": {}, "pooled": {}}

    def summ(R):
        o = {}
        for c in CLASSES:
            if not sum(x[f"{c}_pairs"] for x in R):
                continue
            ratios = np.array([x[f"{c}_sum_p_data"] / x[f"{c}_sum_p_dem"] for x in R if x[f"{c}_pairs"]])
            o[c] = {"pairs_per_leaf": float(np.mean([x[f"{c}_pairs"] for x in R])),
                    "data_over_dem_pooled": sum(x[f"{c}_sum_p_data"] for x in R) / sum(x[f"{c}_sum_p_dem"] for x in R),
                    "per_leaf_median": float(np.median(ratios)), "per_leaf_min": float(ratios.min()),
                    "per_leaf_max": float(ratios.max()), "leaves_below_1": int((ratios < 1).sum()), "leaves": len(ratios)}
        for k in ("first", "bulk", "last"):
            o[f"rate_{k}"] = {"data": float(np.mean([x[f"rate_{k}_data"] for x in R])),
                              "dem": float(np.mean([x[f"rate_{k}_dem"] for x in R])),
                              "data_over_dem": float(np.mean([x[f"rate_{k}_data"] / x[f"rate_{k}_dem"] for x in R]))}
        for g in ("same_stab_dt2", "same_stab_dt3to5", "other_dt0to1", "other_dt2to5", "far_dt6plus"):
            v = [x for x in R if x[f"un_{g}_pairs"]]
            if v:
                o[f"unconnected_{g}"] = {"pairs_per_leaf": float(np.mean([x[f"un_{g}_pairs"] for x in v])),
                                         "mean_cov_data": float(np.mean([x[f"un_{g}_cov_data"] for x in v])),
                                         "mean_cov_dem": float(np.mean([x[f"un_{g}_cov_dem"] for x in v])),
                                         "se_mean_cov_dem": float(math.sqrt(sum(x[f"un_{g}_cov_dem_se"] ** 2 for x in v)) / len(v))}
        for k in range(1, 5):
            o[f"streak_ge{k}"] = {"data": float(np.mean([x[f"streak_ge{k}_data"] for x in R])),
                                  "dem": float(np.mean([x[f"streak_ge{k}_dem"] for x in R]))}
        for k in ("Ku", "Ks2"):
            o[f"{k}_ge1"] = {"data": float(np.mean([x[f"{k}_ge1_data"] for x in R])),
                             "dem": float(np.mean([x[f"{k}_ge1_dem"] for x in R]))}
        return o

    for r in W5:
        res["by_r"][str(r)] = summ([x for x in rows if x["r"] == r])
    res["pooled"] = summ(rows)
    with open(out_path(os.path.join(out_dir, "pairs.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["pooled"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200_000)
