"""Do the shipped RL DEMs differ between the X and Z experiments of a patch? Asked by scout (board, 11:53Z) and main
(10 Oct) after F1's gain turned out to be in Z only. reservoir, 2026-10-10. Descriptive and post hoc; DEM parsing and
counting only, no decoding.

For each patch and each r in W5, the RL DEMs (decoding_results/<CM_RL>/error_model.dem) of the X and Z experiments:
  1. total probability per symptom class: T-hat's classes (willow_check/p4_check.classes) before the rare-class merge
     ("raw": the same names in both bases) and after it (F1's classes), the latter also under D-hat (F1's scales from
     fit_series.json, clipped as F1's model does);
  2. expected detection events per round: the sum over detectors of (1 - prod over its faults of (1 - 2p)) / 2,
     divided by r; also per detector in the bulk layers (1 to r - 1, both stabiliser types) and in layers 0 and r;
     under RL and D-hat, with the observed rates (the archive's detection events, all 50,000 shots);
  3. the expected logical-flip probability, (1 - prod over the L-flipping faults of (1 - 2p)) / 2, and the sum of p
     over those faults, under RL and D-hat, with the observed raw flip rate (obs_flips_actual, all shots);
  4. fault by fault: symptoms (the sorted (x, y, t) of a fault's detectors) present in both DEMs, their probabilities
     XOR-combined over the observable flag, and ln(p_Z / p_X), by layer tag (first: touches layer 0; last: touches
     layer r; else bulk); symptoms present in one DEM only are counted, with their total probability. Control: the
     same comparison for the shipped SI1000-prior DEMs (correlated_matching_decoder_with_si1000_prior), one physical
     noise model applied to each basis's circuit; and the number of lines that differ between the X and Z
     circuit_ideal.stim files;
  5. within each RL DEM, whether a bulk template (r2m_common.template_key: geometry relative to its first layer and
     the L flag) has one rate in every round, and whether that rate is the same at the four r of a series.
Then per patch: D-hat's bulk scale (the mean ln theta-hat over the five non-logical bulk classes, as in r2_zx.py),
Z - X, against the RL prior's Z - X differences: the same five classes' total probability (RL and D-hat), the expected
events per round, and the ratio of expected to observed events (Spearman, Pearson, and an OLS line over the 9 patches).
Usage: nice -n 10 python3 -I r2_xzprior.py ARCHIVE FIT F1 OUT_DIR -> OUT_DIR/r2_xzprior.json, r2_xzprior_classes.csv
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
from r2_common import (BASES, PATCHES, SHOTS, W5, classes, dem_path, det_marginals, detection_events,  # noqa: E402
                       read_b8_bit0, rel_of, scaled, series_fit, theta_by_name)
from p4_check import mechanisms  # noqa: E402  (willow_check, on the path through r2_common)
from r2m_common import template_key  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

BULK = ["boundary", "spacelike", "timelike", "diagonal", "hyperedge"]
TAGS = ("bulk", "first", "last")
CM_SI = "correlated_matching_decoder_with_si1000_prior"


def symptoms(mm, co, tl, r):
    """symptom (sorted (x, y, t) of the detectors) -> (p XOR-combined over its faults, the same over the L-flipping
    ones, layer tag)."""
    sym = {}
    for q, ds, l in mm:
        k = tuple(sorted((round(float(co[d][0]), 6), round(float(co[d][1]), 6), round(float(co[d][2]), 6))
                         for d in ds))
        ts = {int(tl[d]) for d in ds}
        tag = "first" if 0 in ts else ("last" if r in ts else "bulk")
        qa, qL, tg = sym.get(k, (0.0, 0.0, tag))
        qa = qa + q - 2 * qa * q
        if l:
            qL = qL + q - 2 * qL * q
        sym[k] = (qa, qL, tg)
    return sym


def si1000_symptoms(arch, rel, r):
    dem = stim.DetectorErrorModel.from_file(os.path.join(arch, rel, "decoding_results", CM_SI, "error_model.dem"))
    co = dem.get_detector_coordinates()
    tl = np.array([int(round(co[i][2])) for i in range(dem.num_detectors)])
    return symptoms([m for m in mechanisms(dem) if m[0] > 0 and m[1]], co, tl, r)


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X_ = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X_, y, rcond=None)
    e = y - X_ @ b
    s2 = float(e @ e) / (x.size - 2)
    cov = s2 * np.linalg.inv(X_.T @ X_)
    return {"intercept": float(b[0]), "intercept_se": float(math.sqrt(cov[0, 0])), "slope": float(b[1]),
            "slope_se": float(math.sqrt(cov[1, 1])), "n": int(x.size)}


def p_odd(ps):
    """P(an odd number of independent events with probabilities ps)."""
    ps = np.asarray(ps, float)
    return float(0.5 * (1.0 - np.exp(np.sum(np.log1p(-2.0 * ps))))) if ps.size else 0.0


def leaf(arch, fit, patch, basis, r, theta):
    rel = rel_of(patch, basis, r)
    fj = json.load(open(os.path.join(fit, rel, "rl_tsplit", "fit.json")))
    dem = stim.DetectorErrorModel.from_file(dem_path(arch, rel))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    tl = np.array([int(round(co[i][2])) for i in range(nd)])
    if tl.min() != 0 or tl.max() != r:
        raise SystemExit(f"{rel}: detector layers {tl.min()}..{tl.max()}")
    mech, raw, cur, _ = classes(dem, r, fj["n_fit"], float(fj["fit_attempts"][-1]["min_class_events"]))
    pos, mm, rc, fc = {}, [], [], []
    n_zero = n_empty = n_dup = 0
    for (p, ds, l), a, b in zip(mech, raw, cur):
        if p == 0:
            n_zero += 1
            continue
        if not ds:
            n_empty += 1
            continue
        k = (ds, l)
        if k in pos:  # merged as r2_common.leaf_model does (none expected)
            n_dup += 1
            i = pos[k]
            q = mm[i][0]
            mm[i] = (q + p - 2 * q * p, ds, l)
            continue
        pos[k] = len(mm)
        mm.append((p, ds, l))
        rc.append(a)
        fc.append(b)
    mD, nclip = scaled(mm, fc, theta)
    p = np.array([m[0] for m in mm])
    pD = np.array([m[0] for m in mD])
    L = np.array([bool(m[2]) for m in mm])
    out = {"rel": rel, "detectors": nd, "faults": len(mm), "zero_p": n_zero, "empty": n_empty,
           "duplicates_merged": n_dup, "clipped_under_D": nclip}
    raw_t = collections.defaultdict(lambda: [0, 0.0])
    for c, q in zip(rc, p):
        raw_t[c][0] += 1
        raw_t[c][1] += q
    f1_t = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for c, q, qD in zip(fc, p, pD):
        f1_t[c][0] += 1
        f1_t[c][1] += q
        f1_t[c][2] += qD
    out["raw_classes"] = {c: {"n": v[0], "sum_p": v[1]} for c, v in sorted(raw_t.items())}
    out["f1_classes"] = {c: {"n": v[0], "sum_p": v[1], "theta": theta[c], "sum_p_D": v[2]}
                         for c, v in sorted(f1_t.items())}
    out["sum_p"] = {"all": float(p.sum()), "flips_L": float(p[L].sum()), "no_L": float(p[~L].sum()),
                    "bulk5": float(sum(raw_t[c][1] for c in BULK if c in raw_t))}
    out["sum_p_D"] = {"all": float(pD.sum()), "bulk5": float(sum(f1_t[c][2] for c in BULK if c in f1_t))}
    ev = detection_events(arch, rel, nd)
    obs = ev.sum(axis=0, dtype=np.int64) / float(ev.shape[0])
    del ev
    lay = {"bulk": (tl > 0) & (tl < r), "first": tl == 0, "last": tl == r}
    out["detectors_by_layer"] = {g: int(s.sum()) for g, s in lay.items()}
    out["events"] = {}
    for nm, v in (("RL", det_marginals(mm, nd)), ("D", det_marginals(mD, nd)), ("observed", obs)):
        e = {"per_round": float(v.sum() / r)}
        e.update({f"per_detector_{g}": float(v[s].mean()) for g, s in lay.items()})
        out["events"][nm] = e
    of = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)
    out["logical"] = {"L_faults": int(L.sum()), "P_L_RL": p_odd(p[L]), "P_L_D": p_odd(pD[L]),
                      "sum_p_L_RL": float(p[L].sum()), "sum_p_L_D": float(pD[L].sum()),
                      "observed_flip_rate": float(np.mean(np.asarray(of, float)))}
    g = collections.defaultdict(list)
    t_first, t_last = min(co[i][2] for i in range(nd)), max(co[i][2] for i in range(nd))
    for q, ds, l in mm:
        k = template_key(co, ds, l, t_first, t_last)
        if k[0] == "bulk":
            g[k].append(q)
    out["bulk_templates"] = len(g)
    out["bulk_templates_with_more_than_one_rate"] = int(sum(1 for v in g.values() if max(v) / min(v) - 1 > 1e-9))
    return out, symptoms(mm, co, tl, r), {k: float(np.median(v)) for k, v in g.items()}


def match(sx, sz):
    res, ds = {}, {}
    for tag in TAGS:
        kx = {k for k, v in sx.items() if v[2] == tag}
        kz = {k for k, v in sz.items() if v[2] == tag}
        both = sorted(kx & kz)
        d = np.array([math.log(sz[k][0] / sx[k][0]) for k in both])
        w = np.array([sx[k][0] for k in both])
        lx = np.array([sx[k][1] > 0 for k in both], bool)
        lz = np.array([sz[k][1] > 0 for k in both], bool)
        res[tag] = {"symptoms_X": len(kx), "symptoms_Z": len(kz), "in_both": len(both),
                    "identical_to_1e-9": int(np.sum(np.abs(d) < 1e-9)),
                    "abs_ln_ratio_median": float(np.median(np.abs(d))) if d.size else None,
                    "abs_ln_ratio_max": float(np.max(np.abs(d))) if d.size else None,
                    "ln_ratio_mean": float(d.mean()) if d.size else None,
                    "ln_ratio_p_weighted_mean": float(np.sum(w * d) / np.sum(w)) if d.size else None,
                    "sum_p_X_in_both": float(w.sum()), "sum_p_Z_in_both": float(sum(sz[k][0] for k in both)),
                    "sum_p_X_only": float(sum(sx[k][0] for k in kx - kz)),
                    "sum_p_Z_only": float(sum(sz[k][0] for k in kz - kx)),
                    "flips_L_in_X": int(lx.sum()), "flips_L_in_Z": int(lz.sum()),
                    "L_flag_differs": int(np.sum(lx != lz))}
        ds[tag] = d
    return res, ds


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    s = stats.spearmanr(x, y)
    return {"n": int(x.size), "spearman": float(s.correlation), "spearman_p": float(s.pvalue),
            "pearson": float(np.corrcoef(x, y)[0, 1])}


def main(arch, fit, f1, out_dir):
    t0 = time.time()
    res = {"label": "reservoir, 2026-10-10: the shipped RL DEMs of X and Z compared, per patch and r (descriptive, "
                    "post hoc; no decoding)", "leaves": {}, "pairs": {}}
    rows, dpool, dpool_si = [], {t: [] for t in TAGS}, {t: [] for t in TAGS}
    tmpl = collections.defaultdict(dict)
    bscale = {}
    for patch in PATCHES:
        for r in W5:
            st, sy = {}, {}
            for b in BASES:
                fs = series_fit(f1, patch, b)
                th = theta_by_name(fs)
                ph = dict(zip(fs["names"], fs["phi_hat"]))
                bscale[(patch, b)] = float(np.mean([ph[c] for c in BULK]))
                o, s, tr = leaf(arch, fit, patch, b, r, th)
                st[b], sy[b] = o, s
                tmpl[(patch, b)][r] = tr
                res["leaves"][o["rel"]] = o
                for c, v in o["raw_classes"].items():
                    rows.append({"patch": patch, "basis": b, "r": r, "labels": "raw", "class": c, "n_faults": v["n"],
                                 "sum_p": v["sum_p"], "sum_p_per_round": v["sum_p"] / r, "theta": "", "sum_p_D": ""})
                for c, v in o["f1_classes"].items():
                    rows.append({"patch": patch, "basis": b, "r": r, "labels": "F1", "class": c, "n_faults": v["n"],
                                 "sum_p": v["sum_p"], "sum_p_per_round": v["sum_p"] / r, "theta": v["theta"],
                                 "sum_p_D": v["sum_p_D"]})
            m, ds = match(sy["X"], sy["Z"])
            msi, dsi = match(si1000_symptoms(arch, rel_of(patch, "X", r), r),
                             si1000_symptoms(arch, rel_of(patch, "Z", r), r))
            for t in TAGS:
                dpool[t].append(ds[t])
                dpool_si[t].append(dsi[t])
            cx = open(os.path.join(arch, rel_of(patch, "X", r), "circuit_ideal.stim")).read().splitlines()
            cz = open(os.path.join(arch, rel_of(patch, "Z", r), "circuit_ideal.stim")).read().splitlines()
            dl = [i for i, (a, z) in enumerate(zip(cx, cz)) if a != z]
            circ = {"lines_X": len(cx), "lines_Z": len(cz), "lines_differing": len(dl) + abs(len(cx) - len(cz)),
                    "differing_line_numbers": [i + 1 for i in dl],
                    "differing_line_kinds": dict(collections.Counter(cx[i].split("(")[0].split(" ")[0] for i in dl))}
            X_, Z_ = st["X"], st["Z"]
            names = sorted(set(X_["raw_classes"]) | set(Z_["raw_classes"]))
            lr = {}
            for c in names:
                a, z = X_["raw_classes"].get(c), Z_["raw_classes"].get(c)
                lr[c] = (math.log(z["sum_p"] / a["sum_p"]) if a and z and a["sum_p"] > 0 and z["sum_p"] > 0 else None)
            ev = {nm: math.log(Z_["events"][nm]["per_round"] / X_["events"][nm]["per_round"])
                  for nm in ("RL", "D", "observed")}
            evb = {nm: math.log(Z_["events"][nm]["per_detector_bulk"] / X_["events"][nm]["per_detector_bulk"])
                   for nm in ("RL", "D", "observed")}
            res["pairs"][f"d3_at_{patch}/r{r:02d}"] = {
                "symptoms": m, "symptoms_SI1000": msi, "ideal_circuit": circ, "ln_Z_over_X_raw_class_totals": lr,
                "ln_Z_over_X_bulk5_RL": math.log(Z_["sum_p"]["bulk5"] / X_["sum_p"]["bulk5"]),
                "ln_Z_over_X_bulk5_D": math.log(Z_["sum_p_D"]["bulk5"] / X_["sum_p_D"]["bulk5"]),
                "ln_Z_over_X_all_RL": math.log(Z_["sum_p"]["all"] / X_["sum_p"]["all"]),
                "ln_Z_over_X_events_per_round": ev, "ln_Z_over_X_events_per_bulk_detector": evb,
                "ln_Z_over_X_expected_over_observed_RL": ev["RL"] - ev["observed"],
                "ln_Z_over_X_expected_over_observed_bulk_RL": evb["RL"] - evb["observed"],
                "P_L": {b: {k: st[b]["logical"][k] for k in ("P_L_RL", "P_L_D", "observed_flip_rate")} for b in BASES}}
            print(f"d3_at_{patch}/r{r:02d}: bulk symptoms in both {m['bulk']['in_both']}, identical RL "
                  f"{m['bulk']['identical_to_1e-9']} (SI1000 {msi['bulk']['identical_to_1e-9']} of "
                  f"{msi['bulk']['in_both']}), max |ln ratio| {m['bulk']['abs_ln_ratio_max']}; circuit lines differing "
                  f"{circ['lines_differing']} of {circ['lines_X']}; "
                  f"ln Z/X bulk5 RL {res['pairs'][f'd3_at_{patch}/r{r:02d}']['ln_Z_over_X_bulk5_RL']:+.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    # pooled symptom comparison
    def pooled(pool, key):
        sm = {}
        for t in TAGS:
            d = np.concatenate(pool[t]) if pool[t] else np.zeros(0)
            a = np.abs(d)
            P = [res["pairs"][k][key][t] for k in res["pairs"]]
            sm[t] = {"in_both": int(d.size), "identical_to_1e-9": int(np.sum(a < 1e-9)),
                     "abs_ln_ratio_quantiles_50_90_99_max": ([float(np.quantile(a, q)) for q in (0.5, 0.9, 0.99)]
                                                             + [float(a.max())]) if d.size else None,
                     "ln_ratio_mean": float(d.mean()) if d.size else None,
                     "symptoms_X_only": int(sum(v["symptoms_X"] - v["in_both"] for v in P)),
                     "symptoms_Z_only": int(sum(v["symptoms_Z"] - v["in_both"] for v in P)),
                     "L_flag_differs": int(sum(v["L_flag_differs"] for v in P))}
        return sm

    cr = {"series": len(tmpl), "templates_at_all_r": 0, "rate_differs_between_r": 0}
    for d in tmpl.values():
        for k in set.intersection(*(set(v) for v in d.values())):
            vals = [d[r][k] for r in W5]
            cr["templates_at_all_r"] += 1
            cr["rate_differs_between_r"] += int(max(vals) / min(vals) - 1 > 1e-9)
    res["rl_rates_across_rounds_and_r"] = {
        "bulk_templates_leaf_level": int(sum(o["bulk_templates"] for o in res["leaves"].values())),
        "with_more_than_one_rate_across_rounds": int(sum(o["bulk_templates_with_more_than_one_rate"]
                                                         for o in res["leaves"].values())), **cr}
    res["symptom_summary"] = pooled(dpool, "symptoms")
    res["symptom_summary_SI1000"] = pooled(dpool_si, "symptoms_SI1000")
    res["ideal_circuit_summary"] = {
        "pairs": len(res["pairs"]),
        "lines_differing_range": [min(v["ideal_circuit"]["lines_differing"] for v in res["pairs"].values()),
                                  max(v["ideal_circuit"]["lines_differing"] for v in res["pairs"].values())],
        "differing_line_kinds": dict(sum((collections.Counter(v["ideal_circuit"]["differing_line_kinds"])
                                          for v in res["pairs"].values()), collections.Counter()))}
    # class totals: ln(Z / X) over the 36 (patch, r) pairs
    cs = collections.defaultdict(list)
    for k, v in res["pairs"].items():
        for c, x in v["ln_Z_over_X_raw_class_totals"].items():
            if x is not None:
                cs[c].append(x)
    mass = collections.defaultdict(float)
    tot = 0.0
    for o in res["leaves"].values():
        for c, v in o["raw_classes"].items():
            mass[c] += v["sum_p"]
        tot += o["sum_p"]["all"]
    res["class_summary"] = {c: {"pairs": len(v), "ratio_Z_over_X_geomean": float(math.exp(np.mean(v))),
                                "ratio_min": float(math.exp(min(v))), "ratio_max": float(math.exp(max(v))),
                                "share_of_mass": mass[c] / tot} for c, v in sorted(cs.items())}

    def by_basis(f):
        return {b: float(np.mean([f(o) for o in res["leaves"].values() if f"/{b}/" in o["rel"]])) for b in BASES}

    res["by_basis_mean_over_patches_and_r"] = {
        "bulk5_sum_p_per_round_RL": by_basis(lambda o: o["sum_p"]["bulk5"] / int(o["rel"][-2:])),
        "bulk5_sum_p_per_round_D": by_basis(lambda o: o["sum_p_D"]["bulk5"] / int(o["rel"][-2:])),
        "all_sum_p_per_round_RL": by_basis(lambda o: o["sum_p"]["all"] / int(o["rel"][-2:])),
        "events_per_round_RL": by_basis(lambda o: o["events"]["RL"]["per_round"]),
        "events_per_round_D": by_basis(lambda o: o["events"]["D"]["per_round"]),
        "events_per_round_observed": by_basis(lambda o: o["events"]["observed"]["per_round"]),
        "events_RL_over_observed": by_basis(lambda o: o["events"]["RL"]["per_round"] / o["events"]["observed"]["per_round"]),
        "events_D_over_observed": by_basis(lambda o: o["events"]["D"]["per_round"] / o["events"]["observed"]["per_round"]),
        "bulk_detector_RL_over_observed": by_basis(lambda o: o["events"]["RL"]["per_detector_bulk"]
                                                   / o["events"]["observed"]["per_detector_bulk"]),
        "P_L_RL": by_basis(lambda o: o["logical"]["P_L_RL"]),
        "P_L_D": by_basis(lambda o: o["logical"]["P_L_D"]),
        "P_L_observed": by_basis(lambda o: o["logical"]["observed_flip_rate"]),
        "sum_p_L_per_round_RL": by_basis(lambda o: o["logical"]["sum_p_L_RL"] / int(o["rel"][-2:]))}
    res["by_basis_and_r"] = {}
    for r in W5:
        sel = {b: [o for o in res["leaves"].values() if f"/{b}/" in o["rel"] and o["rel"].endswith(f"r{r:02d}")]
               for b in BASES}
        res["by_basis_and_r"][f"r{r:02d}"] = {b: {
            "events_per_round_RL": float(np.mean([o["events"]["RL"]["per_round"] for o in sel[b]])),
            "events_per_round_observed": float(np.mean([o["events"]["observed"]["per_round"] for o in sel[b]])),
            "P_L_RL": float(np.mean([o["logical"]["P_L_RL"] for o in sel[b]])),
            "P_L_D": float(np.mean([o["logical"]["P_L_D"] for o in sel[b]])),
            "P_L_observed": float(np.mean([o["logical"]["observed_flip_rate"] for o in sel[b]]))} for b in BASES}
    # per patch: D-hat's bulk scale Z - X against the prior's Z - X differences (means over r)
    per = {}
    for patch in PATCHES:
        pr = [res["pairs"][f"d3_at_{patch}/r{r:02d}"] for r in W5]
        per[patch] = {"bulk_scale_Z_minus_X": bscale[(patch, "Z")] - bscale[(patch, "X")],
                      "bulk_scale_X": bscale[(patch, "X")], "bulk_scale_Z": bscale[(patch, "Z")],
                      "ln_Z_over_X_bulk5_RL": float(np.mean([v["ln_Z_over_X_bulk5_RL"] for v in pr])),
                      "ln_Z_over_X_bulk5_D": float(np.mean([v["ln_Z_over_X_bulk5_D"] for v in pr])),
                      "ln_Z_over_X_events_RL": float(np.mean([v["ln_Z_over_X_events_per_round"]["RL"] for v in pr])),
                      "ln_Z_over_X_events_observed": float(np.mean([v["ln_Z_over_X_events_per_round"]["observed"]
                                                                    for v in pr])),
                      "ln_Z_over_X_expected_over_observed_RL": float(np.mean(
                          [v["ln_Z_over_X_expected_over_observed_RL"] for v in pr]))}
    ks = list(per)
    y = [per[k]["bulk_scale_Z_minus_X"] for k in ks]
    res["mirror"] = {
        "per_patch": per,
        "mean_over_patches": {f: float(np.mean([per[k][f] for k in ks])) for f in per[ks[0]]},
        "corr_with_bulk_scale_Z_minus_X": {f: corr([per[k][f] for k in ks], y) for f in (
            "ln_Z_over_X_bulk5_RL", "ln_Z_over_X_events_RL", "ln_Z_over_X_events_observed",
            "ln_Z_over_X_expected_over_observed_RL")},
        "ols_bulk_scale_Z_minus_X_on": {f: ols([per[k][f] for k in ks], y) for f in (
            "ln_Z_over_X_bulk5_RL", "ln_Z_over_X_events_RL", "ln_Z_over_X_expected_over_observed_RL")},
        "note": "if D-hat only undid a basis difference in the prior, bulk_scale_Z_minus_X would be about minus "
                "ln_Z_over_X_bulk5_RL (or minus the expected-over-observed difference) and ln_Z_over_X_bulk5_D about 0"}
    res["checks"] = {"duplicates_merged": int(sum(o["duplicates_merged"] for o in res["leaves"].values())),
                     "zero_p_faults": int(sum(o["zero_p"] for o in res["leaves"].values())),
                     "empty_faults": int(sum(o["empty"] for o in res["leaves"].values())),
                     "clipped_under_D": int(sum(o["clipped_under_D"] for o in res["leaves"].values())),
                     "leaves": len(res["leaves"])}
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r2_xzprior.json")), "w") as f:
        json.dump(res, f, indent=1)
    with open(out_path(os.path.join(out_dir, "r2_xzprior_classes.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: res[k] for k in ("rl_rates_across_rounds_and_r", "symptom_summary", "symptom_summary_SI1000",
                                          "ideal_circuit_summary",
                                          "class_summary", "by_basis_mean_over_patches_and_r", "by_basis_and_r",
                                          "checks")}, indent=1))
    print(json.dumps({k: v for k, v in res["mirror"].items() if k != "per_patch"}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
