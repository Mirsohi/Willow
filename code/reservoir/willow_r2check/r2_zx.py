"""EXPLORATORY and descriptive (not pre-registered; main's follow-up brief after the F1 check, 2026-10-10): why is
F1's gain in errors (ML(D-hat) against ML(RL) on the held-out halves) in the Z basis only?

reservoir, 2026-10-10. Existing outputs only, no decoding: D_pred and D_llr (round2/f1/<leaf>/d_data.npz), round 1's
ML(RL) predictions and llr (decode/<leaf>/rl/ml_exact.npz), the fitted scales and their OPG SEs (fit_series.json),
plus, for simple syndrome features, the archive's observable flips and detection events, and the RL DEM's detector
coordinates and T-hat classes (r2_common; for the (7, 7) detector of the twin study and each class's share of the
DEM's probability mass). No multiplicity correction; the patterns below were looked for after seeing the X/Z split.
Notation: b = held-out shots where ML(RL) is wrong and ML(D-hat) right, c = the reverse. With a binary observable the
changed decisions (D_pred != RL's) are exactly b + c. Hit rate h = b / (b + c); contribution to m_D = (b - c) / K_D of
the basis (contributions add up to the basis' m_D over any partition of its shots). min-P = min(P, 1 - P).
Parts:
 1 scales: per class, ln theta-hat paired by patch (Z - X: mean, SD over patches, t, inverse-variance weighted mean
   with the OPG SEs and the chi2 of the differences), and the class's share of the RL DEM's probability mass by basis;
   the logical tilt ln theta(c+L) - ln theta(c) for the four bulk base classes; the bulk scale (mean ln theta-hat over
   the five non-logical bulk classes).
 2 decisions: b, c, h and the contribution, by basis and r, basis and patch, direction of the change (D-hat calls 1 or
   0), min-P under RL and under D-hat (bins); with the hit rate each model gives D-hat's changed calls (RL: mean
   min-P_RL; D-hat: mean 1 - min-P_D).
 3 involvement: (a) the (7, 7) j-shots (an event at the last-layer detector of measure qubit (7, 7)) of q6_7 X and
   q8_9 X against their other held-out shots; (b) syndrome features: detection events in the last layer (0, 1, >= 2),
   in the first layer (0, >= 1), timelike pairs (events at one stabilizer in two consecutive layers below the last;
   0, >= 1).
 4 across the 18 series: m_D, the share of changed decisions and h against the bulk scale and the logical tilt
   (Spearman, Pearson), and a weighted least-squares fit m_D ~ 1 + bulk scale + [basis Z] (weights 1 / SE^2, SEs
   scaled by the residual chi2 per degree of freedom when above 1); the temperature each series' ML(RL) posteriors
   want (r2_temper.json, EXPLORATORY) against the bulk scale.
 5 the Z - X contrast of m_D leaving out one patch at a time (r2_basis.group on r2_leaves.csv).
Usage: nice -n 10 python3 -I r2_zx.py ARCHIVE DECODE FIT F1 OUT_DIR
  (also reads OUT_DIR/r2_check.json, r2_leaves.csv and r2_temper.json) -> OUT_DIR/r2_zx.json, r2_zx_cells.csv
  (through qrc.outpaths)
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import sys
import time

import numpy as np
import stim
from scipy import stats

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (BASES, PATCHES, SHOTS, W5, classes, detection_events, dem_path, min_p_from_llr,  # noqa: E402
                       read_b8_bit0, rel_of, series_fit)
from r2_basis import group  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

LABEL = ("EXPLORATORY, descriptive, not pre-registered (reservoir, 2026-10-10): why F1's gain in errors is in the Z "
         "basis only; existing outputs only, no decoding; no multiplicity correction")
BULK = ["boundary", "spacelike", "timelike", "diagonal", "hyperedge"]
PAIRS = ["boundary", "spacelike", "diagonal", "hyperedge"]
TARGET = (7.0, 7.0)
TWIN_SERIES = [("q6_7", "X"), ("q8_9", "X")]
MP_EDGES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
FIELDS = ["shots", "changed", "b", "c", "K_RL", "K_D", "sum_minP_RL_changed", "sum_1_minus_minP_D_changed"]


def mp_bin(m):
    k = np.clip(np.searchsorted(MP_EDGES, m, side="right") - 1, 0, len(MP_EDGES) - 2)
    return np.array([f"[{MP_EDGES[i]:g},{MP_EDGES[i + 1]:g})" for i in range(len(MP_EDGES) - 1)])[k]


def cell(sel, e_rl, e_d, mp_rl, mp_d):
    ch = sel & (e_rl != e_d)
    return {"shots": int(sel.sum()), "changed": int(ch.sum()), "b": int((sel & (e_rl == 1) & (e_d == 0)).sum()),
            "c": int((sel & (e_rl == 0) & (e_d == 1)).sum()), "K_RL": int(e_rl[sel].sum()), "K_D": int(e_d[sel].sum()),
            "sum_minP_RL_changed": float(mp_rl[ch].sum()), "sum_1_minus_minP_D_changed": float((1 - mp_d[ch]).sum())}


def leaf(arch, dec, fit, f1, patch, basis, r):
    rel = rel_of(patch, basis, r)
    dd = np.load(os.path.join(f1, rel, "d_data.npz"), allow_pickle=False)
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    ti = dd["test_idx"].astype(np.int64)
    if not np.array_equal(ml["idx"], np.arange(SHOTS)):
        raise SystemExit(f"{rel}: ML(RL) decode order")
    obs = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)[ti].astype(np.int8)
    if not np.array_equal(obs, dd["obs_test"].astype(np.int8)):
        raise SystemExit(f"{rel}: observables differ")
    rl_pred = ml["pred"].reshape(SHOTS, -1)[ti, 0].astype(np.int8)
    d_pred = dd["D_pred"].astype(np.int8)
    mp_rl = min_p_from_llr(ml["llr"][ti])
    mp_d = min_p_from_llr(dd["D_llr"])
    e_rl, e_d = (rl_pred != obs).astype(np.int8), (d_pred != obs).astype(np.int8)
    dem = stim.DetectorErrorModel.from_file(dem_path(arch, rel))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    E = detection_events(arch, rel, nd, ti)
    n_last, n_first = E[:, t_of == r].sum(1), E[:, t_of == 0].sum(1)
    key = {(co[i][0], co[i][1], int(t_of[i])): i for i in range(nd) if t_of[i] < r}
    pa = [(i, key[(x, y, t + 1)]) for (x, y, t), i in key.items() if (x, y, t + 1) in key]
    tl = (E[:, [a for a, _ in pa]] & E[:, [b for _, b in pa]]).sum(1)
    jflag = None
    if (patch, basis) in TWIN_SERIES:
        js = [i for i in range(nd) if t_of[i] == r and tuple(co[i][-3:-1]) == TARGET]
        if len(js) != 1:
            raise SystemExit(f"{rel}: (7, 7) last-layer detector not found once")
        jflag = E[:, js[0]] == 1
    # each class's share of the RL DEM's probability mass (T-hat's classes, as rebuilt for the F1 check)
    fj = json.load(open(os.path.join(fit, rel, "rl_tsplit", "fit.json")))
    mech, _, cls, _ = classes(dem, r, fj["n_fit"], float(fj["fit_attempts"][-1]["min_class_events"]))
    mass = collections.defaultdict(float)
    for (p, _, _), c in zip(mech, cls):
        mass[c] += p
    tot = sum(mass.values())
    allsel = np.ones(ti.size, bool)
    factors = {"all": {"all": allsel},
               "minP_RL": {k: mp_bin(mp_rl) == k for k in np.unique(mp_bin(mp_rl))},
               "minP_D": {k: mp_bin(mp_d) == k for k in np.unique(mp_bin(mp_d))},
               "direction": {"D_calls_1": d_pred == 1, "D_calls_0": d_pred == 0},
               "n_last": {"0": n_last == 0, "1": n_last == 1, ">=2": n_last >= 2},
               "n_first": {"0": n_first == 0, ">=1": n_first >= 1},
               "timelike_pairs": {"0": tl == 0, ">=1": tl >= 1}}
    if jflag is not None:
        factors["j77"] = {"j": jflag, "not_j": ~jflag}
    rows = []
    for f, levels in factors.items():
        for lv, sel in levels.items():
            rows.append({"patch": patch, "basis": basis, "r": r, "factor": f, "level": lv,
                         **cell(sel, e_rl, e_d, mp_rl, mp_d)})
    return rows, {c: v / tot for c, v in mass.items()}


def summarise(cells, K_D_basis):
    s = {k: sum(c[k] for c in cells) for k in FIELDS}
    n = s["changed"]
    h = s["b"] / n if n else None  # None (JSON null) where no decision changed
    out = {"shots": s["shots"], "changed": n, "changed_share": n / s["shots"] if s["shots"] else None,
           "b": s["b"], "c": s["c"], "hit_rate": h, "hit_rate_se": math.sqrt(h * (1 - h) / n) if n else None,
           "hit_rate_by_RL": s["sum_minP_RL_changed"] / n if n else None,
           "hit_rate_by_D": s["sum_1_minus_minP_D_changed"] / n if n else None,
           "contribution_to_m_D": (s["b"] - s["c"]) / K_D_basis,
           "contribution_se": math.sqrt(max(n - (s["b"] - s["c"]) ** 2 / n, 0.0)) / K_D_basis if n else 0.0}
    if s["K_D"]:
        out["m_D_within"] = s["K_RL"] / s["K_D"] - 1.0
    return out


def main(arch, dec, fit, f1, out_dir):
    t0 = time.time()
    res = {"label": LABEL}
    # ---- part 2-3 per leaf
    cells, mass = [], {}
    for p in PATCHES:
        for b in BASES:
            for r in W5:
                rows, ms = leaf(arch, dec, fit, f1, p, b, r)
                cells += rows
                mass[(p, b, r)] = ms
    KD = {b: sum(c["K_D"] for c in cells if c["basis"] == b and c["factor"] == "all") for b in BASES}
    chk = json.load(open(os.path.join(out_dir, "r2_check.json")))
    if KD["X"] + KD["Z"] != chk["primary_m_D"]["K_D"]:
        raise SystemExit("K_D differs from r2_check.json")
    dec_ = {}
    for f in sorted({c["factor"] for c in cells}):
        dec_[f] = {}
        for b in BASES:
            lv = sorted({c["level"] for c in cells if c["factor"] == f and c["basis"] == b})
            if lv:
                dec_[f][b] = {v: summarise([c for c in cells if c["factor"] == f and c["basis"] == b and c["level"] == v],
                                           KD[b]) for v in lv}
    res["decisions"] = dec_
    allc = [c for c in cells if c["factor"] == "all"]
    res["decisions_by_basis_and_r"] = {f"{b}/r{r}": summarise([c for c in allc if c["basis"] == b and c["r"] == r], KD[b])
                                       for b in BASES for r in W5}
    res["decisions_by_series"] = {f"d3_at_{p}/{b}": summarise([c for c in allc if c["basis"] == b and c["patch"] == p],
                                                               KD[b]) for p in PATCHES for b in BASES}
    # ---- part 1: scales
    fs = {(p, b): series_fit(f1, p, b) for p in PATCHES for b in BASES}
    ph = {k: dict(zip(v["names"], v["phi_hat"])) for k, v in fs.items()}
    se = {k: dict(zip(v["names"], v["se_opg_ln"])) for k, v in fs.items()}
    names = sorted({n for v in ph.values() for n in v})
    sc = {}
    for n in names:
        pp = [p for p in PATCHES if n in ph[(p, "X")] and n in ph[(p, "Z")]]
        e = {"in_series": {b: sum(1 for p in PATCHES if n in ph[(p, b)]) for b in BASES},
             "mass_share_mean": {b: float(np.mean([mass[(p, b, r)].get(n, 0.0) for p in PATCHES for r in W5]))
                                 for b in BASES}}
        for b in BASES:
            v = [ph[(p, b)][n] for p in PATCHES if n in ph[(p, b)]]
            e[f"theta_median_{b}"] = float(np.exp(np.median(v))) if v else None
        if len(pp) >= 2:
            d = np.array([ph[(p, "Z")][n] - ph[(p, "X")][n] for p in pp])
            s2 = np.array([se[(p, "Z")][n] ** 2 + se[(p, "X")][n] ** 2 for p in pp])
            w = 1.0 / s2
            ivm = float(np.sum(w * d) / np.sum(w))
            e.update({"paired_patches": len(pp), "dlnZ_minus_X_mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                      "t": float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size))), "patches_Z_above_X": int((d > 0).sum()),
                      "iv_mean": ivm, "iv_se": float(1 / math.sqrt(np.sum(w))), "iv_z": ivm * math.sqrt(np.sum(w)),
                      "chi2_about_iv_mean": float(np.sum(w * (d - ivm) ** 2)), "chi2_df": len(pp) - 1})
        sc[n] = e
    res["scales_by_class"] = sc
    per = {}
    for (p, b), v in fs.items():
        tilt = {c: ph[(p, b)][c + "+L"] - ph[(p, b)][c] for c in PAIRS}
        tse = {c: math.hypot(se[(p, b)][c + "+L"], se[(p, b)][c]) for c in PAIRS}
        key = f"d3_at_{p}/{b}"
        ds = res["decisions_by_series"][key]
        per[key] = {"basis": b, "bulk_scale": float(np.mean([ph[(p, b)][c] for c in BULK])),
                    "bulk_spread": float(np.std([ph[(p, b)][c] for c in BULK])),
                    "logical_tilt": tilt, "logical_tilt_se": tse,
                    "logical_tilt_mean": float(np.mean(list(tilt.values()))),
                    "logical_tilt_norm": float(math.sqrt(sum((tilt[c] / tse[c]) ** 2 for c in PAIRS))),
                    "m_D": chk["m_D_per_series"][key]["value"], "m_D_se": chk["m_D_per_series"][key]["se"],
                    "changed_share": ds["changed_share"], "hit_rate": ds["hit_rate"]}
    res["series"] = per
    res["tilt_by_basis"] = {b: {c: {"median": float(np.median([per[k]["logical_tilt"][c] for k in per if per[k]["basis"] == b])),
                                    "mean": float(np.mean([per[k]["logical_tilt"][c] for k in per if per[k]["basis"] == b]))}
                                for c in PAIRS} for b in BASES}
    res["bulk_scale_by_basis"] = {b: {"mean": float(np.mean([per[k]["bulk_scale"] for k in per if per[k]["basis"] == b])),
                                      "values": [per[k]["bulk_scale"] for k in per if per[k]["basis"] == b]} for b in BASES}
    # ---- part 4: across series
    keys = sorted(per)
    y = {"m_D": np.array([per[k]["m_D"] for k in keys]), "changed_share": np.array([per[k]["changed_share"] for k in keys]),
         "hit_rate": np.array([per[k]["hit_rate"] for k in keys])}
    xs = {"bulk_scale": np.array([per[k]["bulk_scale"] for k in keys]),
          "logical_tilt_mean": np.array([per[k]["logical_tilt_mean"] for k in keys]),
          "logical_tilt_norm": np.array([per[k]["logical_tilt_norm"] for k in keys]),
          "bulk_spread": np.array([per[k]["bulk_spread"] for k in keys])}
    cor = {}
    for yn, yv in y.items():
        for xn, xv in xs.items():
            sp, pr = stats.spearmanr(xv, yv), stats.pearsonr(xv, yv)
            cor[f"{yn}~{xn}"] = {"spearman": float(sp[0]), "spearman_p": float(sp[1]), "pearson": float(pr[0]),
                                 "pearson_p": float(pr[1])}
    # the temperature each series' ML(RL) posteriors want (r2_temper.py, EXPLORATORY) against D-hat's bulk scale
    tb = json.load(open(os.path.join(out_dir, "r2_temper.json")))["series"]
    beta = np.array([tb[k]["beta"] for k in keys])
    for k, bv in zip(keys, beta):
        per[k]["temper_beta"] = float(bv)
    for xn in ("bulk_scale", "logical_tilt_norm"):
        sp, pr = stats.spearmanr(xs[xn], beta), stats.pearsonr(xs[xn], beta)
        cor[f"temper_beta~{xn}"] = {"spearman": float(sp[0]), "spearman_p": float(sp[1]), "pearson": float(pr[0]),
                                    "pearson_p": float(pr[1])}
    res["temper_beta_by_basis"] = {b: {"median": float(np.median([tb[k]["beta"] for k in keys if per[k]["basis"] == b])),
                                       "values": [tb[k]["beta"] for k in keys if per[k]["basis"] == b]} for b in BASES}
    res["across_series_correlations"] = cor
    zb = np.array([1.0 if per[k]["basis"] == "Z" else 0.0 for k in keys])
    w = 1.0 / np.array([per[k]["m_D_se"] for k in keys]) ** 2
    wls = {}
    for nm, cols in (("basis_only", [np.ones(len(keys)), zb]), ("bulk_only", [np.ones(len(keys)), xs["bulk_scale"]]),
                     ("bulk_and_basis", [np.ones(len(keys)), xs["bulk_scale"], zb])):
        A = np.column_stack(cols)
        W = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A * W, y["m_D"] * np.sqrt(w), rcond=None)
        resid = y["m_D"] - A @ coef
        chi2 = float(np.sum(w * resid ** 2))
        cov = np.linalg.inv((A * w[:, None]).T @ A)
        dof = len(keys) - A.shape[1]
        scale = max(chi2 / dof, 1.0)
        wls[nm] = {"coef": coef.tolist(), "se_scaled": np.sqrt(np.diag(cov) * scale).tolist(), "chi2": chi2, "dof": dof,
                   "terms": ["1"] + (["bulk_scale"] if nm != "basis_only" else []) + (["Z"] if nm != "bulk_only" else [])}
    res["across_series_wls_m_D"] = wls
    # ---- part 5: leave one patch out
    lrows = list(csv.DictReader(open(os.path.join(out_dir, "r2_leaves.csv"))))
    lopo = {}
    for p in ["none"] + list(PATCHES):
        keep = [r_ for r_ in lrows if r_["patch"] != p]
        gx, gz = group([r_ for r_ in keep if r_["basis"] == "X"]), group([r_ for r_ in keep if r_["basis"] == "Z"])
        d = gz["m_D"] - gx["m_D"]
        s_ = math.hypot(gx["se"], gz["se"])
        lopo[f"without_{p}"] = {"m_D_X": gx["m_D"], "se_X": gx["se"], "m_D_Z": gz["m_D"], "se_Z": gz["se"],
                                "Z_minus_X": d, "se": s_, "z": d / s_}
    res["leave_one_patch_out"] = lopo
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r2_zx_cells.csv")), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["patch", "basis", "r", "factor", "level"] + FIELDS)
        wr.writeheader()
        for c in cells:
            wr.writerow(c)
    with open(out_path(os.path.join(out_dir, "r2_zx.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("decisions_by_basis_and_r", "leave_one_patch_out", "across_series_wls_m_D")},
                     indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
