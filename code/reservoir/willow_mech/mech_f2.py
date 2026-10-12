"""EXPLORATORY: candidate extra mechanisms for decoder's round-2 family F2 (notes/decoder/willow_prereg_r2.md section 6).
Nothing is fitted here. Every number is computed on the fit halves only: within each initialisation bitstring the
shots of even rank (the complement of P4's held-out test_idx, which is checked against each fit_data.npz). The held-out
halves are not read. The choice of what to look at (short-range pairs, the last layer, the X basis) comes from
reservoir's mechanism study (board/reservoir.md 21:33Z), which saw all shots, so the design leakage that the round-2
addendum notes remains.

A. Pairs of detectors that share no mechanism of the leaf's RL DEM (the DEM gives them zero covariance), short range:
   different stabilizers at most one round apart, or the same stabilizer two rounds apart. Per pair instance:
     q = (Cov(x_i, x_j) - s2 <x_i><x_j>) / ((1 - 2<x_i>)(1 - 2<x_j>)),
   the probability of one extra independent fault that flips exactly i and j (first order). s2 is the leaf's shot-level
   rate variance, the mean of Cov / (<x_i><x_j>) over unconnected pairs at least 6 rounds apart, so slow common-rate
   fluctuations are not counted as a pair mechanism. Instances are pooled per series (patch x basis, its 4 W5 leaves)
   into keys: the two stabilizers, the round offset, and the region (bulk; touching layer 0; touching layer r, with
   the offsets from that layer). A last-layer detector's stabilizer is the measure qubit of its last coordinate
   triple (its first coordinates are a data qubit). Selected: keys with q > 0 and z = q / se_used >= 3, where se_used
   is the larger of the analytic SE (independent shots) and the scatter SE of the instances' q (conservative: the
   scatter is often several times the analytic SE, because q varies between instances).
   Observable flag: under the RL DEM the pair {i, j} is explained at lowest order by two faults e1, e2 with
   D(e1) XOR D(e2) = {i, j}. Their summed probability by observable parity gives the flag a fault there would most
   plausibly carry (the dominant parity) and how clear that is (its share; flag_clear when >= 0.9). suggested_class
   names a pooled class in T-hat's style, e.g. pair_diff_stab_dt1+L@last; decoder decides the classes.
B. Last-layer (and, for contrast, first-layer) single events. Shots with exactly one detection event, at detector j:
   ML(RL) failures against the sum of min(P, 1 - P) (R_j), per series and detector and per basis, plus the DEM's
   lowest-order explanations of {j} by observable parity. R_j > 1 says the minority-parity explanations are under-
   weighted. The candidate is a "twin" of the single-detector fault at j with the opposite observable flag; its
   implied probability, q_twin = (R_j - 1) f_j w_j, has f_j the mean predicted failure and w_j the DEM's total weight
   of explanations of {j} (first order, an initial value for a fit and not an estimate of the truth).
   final_layer_classes pools the single events by the class of the fault at j and splits the predicted failures over
   the classes of the minority explanations, per basis. relation_to_pairs_json cross-checks the bulk pair groups
   against pairs.json.
Usage: nice -n 19 python3 -I mech_f2.py ARCHIVE DECODE FIT OUT_DIR   -> OUT_DIR/f2_candidates.json (qrc.outpaths)
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

import numpy as np
import stim

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_check"))
sys.path.insert(0, os.path.join(HERE, ".."))  # code/reservoir (qrc.outpaths)
from p4_check import CM_RL, SHOTS, block_codes, classes, read_b8_bit0  # noqa: E402
from qrc.outpaths import in_path, out_path  # noqa: E402

PATCHES = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
BASES = ["X", "Z"]
W5 = [10, 13, 30, 50]
LABEL = ("EXPLORATORY, not pre-registered (reservoir, 2026-10-09): candidates for F2, nothing fitted; "
         "computed on the fit halves only")
Z_SEL = 3.0


def poisson_upper(k, lam):
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    if k > lam:  # sum the upper tail directly, so that small p-values do not round to 0
        term = math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
        s, i = 0.0, k
        while term > 0 and term > 1e-17 * s:
            s += term
            i += 1
            term *= lam / i
        return min(1.0, s)
    below = sum(math.exp(i * math.log(lam) - lam - math.lgamma(i + 1)) for i in range(k))
    return max(0.0, 1.0 - below)


def explain(target, mech, fp, by_det, top=4):
    """Lowest-order (one or two fault) explanations of a detector set under the DEM: summed probability by observable
    parity, and the leading explanations (all of them if top is None)."""
    w = [0.0, 0.0]
    lead = []
    t = frozenset(target)
    seen = set()
    for d in target:
        for k in by_det[d]:
            p, ds, l = mech[k]
            s = frozenset(ds)
            if s == t:
                if ("1", k) not in seen:
                    seen.add(("1", k))
                    w[l] += p
                    lead.append((p, l, (k,)))
                continue
            rest = s ^ t
            k2 = fp.get(rest)
            if k2 is not None and k2 != k:
                key = ("2",) + tuple(sorted((k, k2)))
                if key in seen:
                    continue
                seen.add(key)
                p2, _, l2 = mech[k2]
                w[l ^ l2] += p * p2
                lead.append((p * p2, l ^ l2, (k, k2)))
    lead.sort(key=lambda x: -x[0])
    return w, lead if top is None else lead[:top]


def leaf(arch, dec, fit, patch, basis, r):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir = os.path.join(arch, rel)
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    stab = [tuple(co[i][-3:-1]) if t_of[i] == r else tuple(co[i][:2]) for i in range(nd)]
    mech, raw, _, _ = classes(dem, r, 5000, 30.0)  # raw class names (before merging) for labelling only
    fp = {frozenset(ds): k for k, (p, ds, l) in enumerate(mech)}
    by_det = collections.defaultdict(list)
    conn = np.zeros((nd, nd), bool)
    for k, (p, ds, l) in enumerate(mech):
        for d in ds:
            by_det[d].append(k)
        for a in ds:
            for b in ds:
                conn[a, b] = True
    # fit halves only: even rank within each bitstring; never a held-out shot
    code = block_codes(os.path.join(a_dir, "sweep_bits.b8"), SHOTS)
    rank = np.zeros(SHOTS, np.int64)
    for c in np.unique(code):
        pos = np.flatnonzero(code == c)
        rank[pos] = np.arange(pos.size)
    fit_half = np.flatnonzero(rank % 2 == 0)
    test_idx = np.load(os.path.join(fit, rel, "rl_tsplit", "fit_data.npz"))["test_idx"]
    if np.intersect1d(fit_half, test_idx).size or fit_half.size + test_idx.size != SHOTS:
        raise SystemExit(f"fit half overlaps the held-out half at {rel}")
    raw_b = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(SHOTS, -1)
    E = np.unpackbits(raw_b[fit_half], axis=1, bitorder="little")[:, :nd]
    n = E.shape[0]
    Ef = E.astype(np.float32)
    m1 = Ef.mean(0).astype(np.float64)
    m2 = (Ef.T @ Ef).astype(np.float64) / n
    cov = m2 - np.outer(m1, m1)
    # A. unconnected short-range pairs
    iu, ju = np.triu_indices(nd, 1)
    unc = ~conn[iu, ju]
    dt = np.abs(t_of[iu] - t_of[ju])
    same = np.array([stab[a] == stab[b] for a, b in zip(iu, ju)])
    far = unc & (dt >= 6) & (t_of[iu] > 0) & (t_of[ju] > 0) & (t_of[iu] < r) & (t_of[ju] < r)
    s2 = float(np.mean(cov[iu[far], ju[far]] / (m1[iu[far]] * m1[ju[far]]))) if far.any() else 0.0
    cand = unc & ((~same & (dt <= 1)) | (same & (dt == 2)))
    inst = []
    for a, b in zip(iu[cand], ju[cand]):
        if t_of[a] > t_of[b] or (t_of[a] == t_of[b] and stab[a] > stab[b]):
            a, b = b, a
        den = (1 - 2 * m1[a]) * (1 - 2 * m1[b])
        q = (cov[a, b] - s2 * m1[a] * m1[b]) / den
        se = math.sqrt(m1[a] * (1 - m1[a]) * m1[b] * (1 - m1[b]) / n) / den
        if t_of[a] == 0 or t_of[b] == 0:
            region, toff = "first", (int(t_of[a]), int(t_of[b]))
        elif t_of[a] == r or t_of[b] == r:
            region, toff = "last", (int(r - t_of[a]), int(r - t_of[b]))
        else:
            region, toff = "bulk", (0, int(t_of[b] - t_of[a]))
        w, lead = explain((int(a), int(b)), mech, fp, by_det)
        key = (stab[a], stab[b], int(t_of[b] - t_of[a]), region, toff)
        inst.append({"key": key, "i": int(a), "j": int(b), "q": q, "se": se, "cov": float(cov[a, b]),
                     "w0": w[0], "w1": w[1],
                     "lead": [(pp, ll, "+".join(raw[k] for k in ks)) for pp, ll, ks in lead]})
    # B. single events (N = 1) in the first and last layers
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    llr = ml["llr"].astype(np.float64)[fit_half]
    pred = ml["pred"].reshape(SHOTS, -1)[:, 0][fit_half]
    obs = read_b8_bit0(os.path.join(a_dir, "obs_flips_actual.b8"), SHOTS)[fit_half]
    fail = (pred != obs)
    m = 1 / (1 + np.exp(np.abs(llr)))
    N = E.sum(1)
    one = np.flatnonzero(N == 1)
    jdet = E[one].argmax(1)
    single = {}
    for j in sorted(set(np.flatnonzero((t_of == 0) | (t_of == r)).tolist())):
        sel = one[jdet == j]
        w, lead = explain((j,), mech, fp, by_det, top=None)
        domp = int(w[1] > w[0])
        ones = [raw[ks[0]] for pp, ll, ks in lead if len(ks) == 1]
        minority = collections.defaultdict(float)  # class combination of each minority-parity explanation -> weight
        for pp, ll, ks in lead:
            if ll != domp:
                minority["+".join(raw[k] for k in ks)] += pp
        single[j] = {"layer": "last" if t_of[j] == r else "first", "stab": stab[j], "shots": int(sel.size),
                     "fail": int(fail[sel].sum()), "pred": float(m[sel].sum()), "w0": w[0], "w1": w[1],
                     "one_fault_class": ones[0] if ones else "none", "minority": dict(minority),
                     "lead": [(pp, ll, "+".join(raw[k] for k in ks)) for pp, ll, ks in lead[:4]]}
    conf = m < 3e-3
    calib = {"shots": int(n), "fail": int(fail.sum()), "pred": float(m.sum()),
             "conf_fail": int(fail[conf].sum()), "conf_pred": float(m[conf].sum())}
    return {"s2": s2, "n": n, "inst": inst, "single": single, "calib": calib, "nd": nd}


def main(arch, dec, fit, out_dir):
    res = {"label": LABEL,
           "selection_rule": (f"per series and key: q > 0 and z = q / se_used >= {Z_SEL}, se_used = max(analytic SE, "
                              "scatter SE of the instances' q)"),
           "definitions": __doc__.split("Usage:")[0].strip(), "series": {}, "instances_by_leaf": {},
           "single_events": {}, "by_basis": {}}
    basis_acc = collections.defaultdict(lambda: [0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0])
    fl_dom = collections.defaultdict(lambda: [0, 0, 0.0])  # (basis, layer, one-fault class) -> shots, fail, pred
    fl_min = collections.defaultdict(float)  # (basis, layer, minority class combination) -> predicted failures
    xcheck = collections.defaultdict(list)  # pairs.json cross-check: bulk group -> per-leaf (pairs, mean cov)
    for p in PATCHES:
        for b in BASES:
            keys = collections.defaultdict(list)
            per_leaf = {}
            sing = collections.defaultdict(lambda: {"shots": 0, "fail": 0, "pred": 0.0, "w": [0.0, 0.0], "lead": None,
                                                    "n_leaves": 0, "detector_by_r": {}, "one_fault_class": set()})
            for r in W5:
                L = leaf(arch, dec, fit, p, b, r)
                per_leaf[r] = L
                for x in L["inst"]:
                    keys[x["key"]].append(x)
                for g in ("same_stab_dt2", "other_dt0to1"):
                    cv = [x["cov"] for x in L["inst"] if x["key"][3] == "bulk"
                          and (x["key"][0] == x["key"][1]) == (g == "same_stab_dt2")]
                    xcheck[g].append((len(cv), float(np.mean(cv))))
                for j, s in L["single"].items():
                    k = (s["layer"], s["stab"])
                    a = sing[k]
                    a["shots"] += s["shots"]; a["fail"] += s["fail"]; a["pred"] += s["pred"]
                    a["w"][0] += s["w0"] / len(W5); a["w"][1] += s["w1"] / len(W5); a["n_leaves"] += 1
                    a["detector_by_r"][str(r)] = int(j)
                    a["one_fault_class"].add(s["one_fault_class"])
                    if a["lead"] is None or r == 10:
                        a["lead"] = s["lead"]
                    d = fl_dom[(b, s["layer"], s["one_fault_class"])]
                    d[0] += s["shots"]; d[1] += s["fail"]; d[2] += s["pred"]
                    wm = sum(s["minority"].values())
                    for combo, wc in s["minority"].items():
                        fl_min[(b, s["layer"], combo)] += s["pred"] * wc / wm if wm > 0 else 0.0
                c = L["calib"]
                for lay in ("last", "first"):
                    for j, s in L["single"].items():
                        if s["layer"] == lay:
                            acc = basis_acc[(b, lay)]
                            acc[0] += s["shots"]; acc[1] += s["pred"]; acc[2] += s["fail"]
                acc = basis_acc[(b, "all")]
                acc[0] += c["shots"]; acc[1] += c["pred"]; acc[2] += c["fail"]; acc[4] += c["conf_fail"]; acc[5] += c["conf_pred"]
                print(f"{p} {b} r={r}: s2 {L['s2']:.4f}, {len(L['inst'])} pair instances", flush=True)
            rows = []
            for k, xs in keys.items():
                q = np.array([x["q"] for x in xs])
                se = np.array([x["se"] for x in xs])
                qm = float(q.mean())
                sem = float(math.sqrt(np.sum(se ** 2)) / len(xs))
                sd_emp = float(q.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
                se_used = max(sem, sd_emp) if len(xs) > 1 else sem
                w0 = float(np.mean([x["w0"] for x in xs]))
                w1 = float(np.mean([x["w1"] for x in xs]))
                dom = [int(x["w1"] > x["w0"]) for x in xs if x["w0"] + x["w1"] > 0]
                dpar = int(w1 > w0) if w0 + w1 > 0 else None
                share = (max(w0, w1) / (w0 + w1)) if w0 + w1 > 0 else None
                group = "same_stab_dt2" if k[0] == k[1] else f"diff_stab_dt{k[2]}"
                rows.append({"stab_a": list(k[0]), "stab_b": list(k[1]), "dt": k[2], "region": k[3],
                             "layer_offsets": list(k[4]), "instances": len(xs), "q": qm, "se": sem,
                             "se_scatter": sd_emp, "se_used": se_used, "z": qm / se_used, "z_analytic": qm / sem,
                             "observable_flag": {"dominant_parity": dpar, "dominant_share": share,
                                                 "instances_with_parity_1": sum(dom), "instances_explained": len(dom),
                                                 "two_fault_weight": w0 + w1},
                             "suggested_class": "pair_" + group + ("+L" if dpar else "")
                                                + ("" if k[3] == "bulk" else "@" + k[3]),
                             "flag_clear": bool(share is not None and share >= 0.9),
                             "leading_two_fault_explanations": xs[len(xs) // 2]["lead"]})
            rows.sort(key=lambda x: -x["z"])
            sel = [x for x in rows if x["z"] >= Z_SEL and x["q"] > 0]
            ser = f"{p} {b}"
            res["series"][ser] = {
                "keys_tested": len(rows), "selected": sel,
                "summary_by_group": {
                    name: {"keys": len(v), "selected": sum(1 for x in v if x["z"] >= Z_SEL and x["q"] > 0),
                           "mean_q": float(np.mean([x["q"] for x in v])) if v else None}
                    for name, v in (("diff_stab_dt0", [x for x in rows if x["dt"] == 0 and x["stab_a"] != x["stab_b"]]),
                                    ("diff_stab_dt1", [x for x in rows if x["dt"] == 1 and x["stab_a"] != x["stab_b"]]),
                                    ("same_stab_dt2", [x for x in rows if x["stab_a"] == x["stab_b"]]))},
                "s2_by_r": {str(r): per_leaf[r]["s2"] for r in W5}}
            selkeys = {(tuple(x["stab_a"]), tuple(x["stab_b"]), x["dt"], x["region"], tuple(x["layer_offsets"])): x["q"]
                       for x in sel}
            for r in W5:
                res["instances_by_leaf"][f"{ser} r{r}"] = [
                    [x["i"], x["j"], int(x["w1"] > x["w0"]) if x["w0"] + x["w1"] > 0 else None, selkeys[x["key"]]]
                    for x in per_leaf[r]["inst"] if x["key"] in selkeys]
            srows = []
            for (lay, st), a in sorted(sing.items()):
                R = a["fail"] / a["pred"] if a["pred"] > 0 else float("nan")
                f = a["pred"] / a["shots"] if a["shots"] else float("nan")
                wt = a["w"][0] + a["w"][1]
                srows.append({"layer": lay, "stab": list(st), "detector_by_r": a["detector_by_r"],
                              "one_fault_class": sorted(a["one_fault_class"]), "shots": a["shots"],
                              "failures": a["fail"], "predicted": a["pred"], "excess": a["fail"] - a["pred"],
                              "R": R, "R_se": R / math.sqrt(max(a["fail"], 1)),
                              "p_poisson_upper": poisson_upper(a["fail"], a["pred"]),
                              "dominant_parity": int(a["w"][1] > a["w"][0]),
                              "minority_share": min(a["w"]) / wt if wt else None,
                              "q_twin_first_order": (R - 1) * f * wt if a["shots"] else None,
                              "leading_explanations_r10": a["lead"]})
            res["single_events"][ser] = srows
    for (b, lay), a in sorted(basis_acc.items()):
        o = {"shots": a[0], "failures": a[2], "predicted": a[1], "R": a[2] / a[1], "R_se": a[2] / a[1] / math.sqrt(a[2])}
        if lay == "all":
            o.update({"conf_failures": a[4], "conf_predicted": a[5], "R_conf": a[4] / a[5],
                      "R_conf_se": a[4] / a[5] / math.sqrt(a[4])})
        res["by_basis"][f"{b} {lay if lay == 'all' else 'N=1 ' + lay + ' layer'}"] = o
    # final-layer classes behind the largest R (single events), per basis
    fl = {}
    for b in BASES:
        o = {}
        for lay in ("last", "first"):
            dom_rows = []
            for (bb, ll, cl), (sh, fa, pr) in sorted(fl_dom.items()):
                if bb == b and ll == lay:
                    dom_rows.append({"one_fault_class": cl, "shots": sh, "failures": fa, "predicted": pr,
                                     "R": fa / pr if pr > 0 else None, "p_poisson_upper": poisson_upper(fa, pr)})
            tot = sum(v for (bb, ll, _), v in fl_min.items() if bb == b and ll == lay)
            mins = sorted(((cl, v) for (bb, ll, cl), v in fl_min.items() if bb == b and ll == lay),
                          key=lambda x: -x[1])
            o[f"N=1 {lay} layer"] = {
                "by_one_fault_class": dom_rows,
                "minority_explanations_share_of_predicted_failures": [
                    {"classes": cl, "predicted_failures": v, "share": v / tot if tot else None} for cl, v in mins[:12]],
                "stabilizers_ranked_by_excess": sorted(
                    ({"series": ser, **{k: x[k] for k in ("stab", "detector_by_r", "one_fault_class", "shots",
                                                          "failures", "predicted", "excess", "R", "p_poisson_upper",
                                                          "minority_share", "q_twin_first_order")}}
                     for ser, xs in res["single_events"].items() if ser.endswith(" " + b)
                     for x in xs if x["layer"] == lay), key=lambda x: -x["excess"])[:12]}
        fl[b] = o
    res["final_layer_classes"] = {
        "note": ("Single events (N = 1) at a first- or last-layer detector j, pooled by the class of the one-fault "
                 "explanation of {j} (the single-detector fault at j) and, for the predicted failures, split over "
                 "the class combinations of the minority-parity explanations in proportion to their DEM weight. "
                 "R > 1: those minority explanations (or a missing twin of the fault at j with the opposite "
                 "observable flag) are under-weighted. Observed failures cannot be split by explanation."),
        "by_basis": fl}
    pj = json.load(open(in_path(os.path.join(out_dir, "pairs.json"))))["pooled"]
    res["relation_to_pairs_json"] = {
        "note": ("pairs.json (reservoir 21:24Z, all shots) has the two short-range classes as means over the 72 "
                 "leaves of bulk pairs: unconnected_same_stab_dt2 and unconnected_other_dt0to1. This file splits "
                 "them, and their first- and last-layer analogues, into per-stabilizer keys on the fit halves only. "
                 "Cross-check on the bulk pairs: pairs per leaf (must be equal) and the mean raw covariance "
                 "(fit halves here, all shots there)."),
        "same_stab_dt2": {"pairs_per_leaf_here": float(np.mean([c[0] for c in xcheck["same_stab_dt2"]])),
                          "pairs_per_leaf_pairs_json": pj["unconnected_same_stab_dt2"]["pairs_per_leaf"],
                          "mean_cov_here": float(np.mean([c[1] for c in xcheck["same_stab_dt2"]])),
                          "mean_cov_pairs_json": pj["unconnected_same_stab_dt2"]["mean_cov_data"]},
        "other_dt0to1": {"pairs_per_leaf_here": float(np.mean([c[0] for c in xcheck["other_dt0to1"]])),
                         "pairs_per_leaf_pairs_json": pj["unconnected_other_dt0to1"]["pairs_per_leaf"],
                         "mean_cov_here": float(np.mean([c[1] for c in xcheck["other_dt0to1"]])),
                         "mean_cov_pairs_json": pj["unconnected_other_dt0to1"]["mean_cov_data"]},
        "far_dt6plus_mean_cov_pairs_json": pj["unconnected_far_dt6plus"]["mean_cov_data"]}
    allsel = [x for s in res["series"].values() for x in s["selected"]]
    res["summary"] = {
        "series": len(res["series"]), "selected_pair_keys": len(allsel),
        "selected_by_group": collections.Counter(
            ("same_stab_dt2" if x["stab_a"] == x["stab_b"] else f"diff_stab_dt{x['dt']}") + "|" + x["region"]
            for x in allsel),
        "selected_dominant_parity_1": sum(1 for x in allsel if x["observable_flag"]["dominant_parity"] == 1),
        "selected_ambiguous_share_below_0.9": sum(1 for x in allsel if (x["observable_flag"]["dominant_share"] or 1) < 0.9),
        "selected_q_median": float(np.median([x["q"] for x in allsel])) if allsel else None,
        "selected_flag_clear": sum(1 for x in allsel if x["flag_clear"]),
        "selected_by_suggested_class": collections.Counter(x["suggested_class"] for x in allsel),
        "keys_tested": sum(s["keys_tested"] for s in res["series"].values()),
        "pair_instances_listed": sum(len(v) for v in res["instances_by_leaf"].values()),
        "expected_false_selections_at_z3": "about 0.00135 x keys tested (one-sided normal tail), see keys_tested"}
    with open(out_path(os.path.join(out_dir, "f2_candidates.json")), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res["summary"], indent=1, default=str))
    print(json.dumps(res["by_basis"], indent=1))
    print(json.dumps(res["relation_to_pairs_json"], indent=1))
    fx = res["final_layer_classes"]["by_basis"]
    print(json.dumps({b: {k: v for k, v in fx[b]["N=1 last layer"].items() if k != "stabilizers_ranked_by_excess"}
                      for b in BASES}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
