"""Independent check of P4 (mismatch cost: exact ML with the RL DEM against exact ML with the fitted DEM T-hat, d = 3)
and of the decomposition, from the stored fits and decodes.

reservoir, 2026-10-09. Written from notes/decoder/willow_prereg.md sections 2, 4 and 5 and the fit outputs alone.
The P4 parts of code/decoder/analyze_willow.py, results/decoder/willow/analysis_m3/, results/decoder/willow/
supplementary/ and board/decoder.md 22:13:57Z were not read before these numbers were stored.

Inputs (read only; run with python3 -I because the archive is downloaded data):
  archive  ARCHIVE/d3_at_<patch>/<basis>/r<NN>/{obs_flips_actual.b8, sweep_bits.b8,
           decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem (the RL DEM),
           decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8 (r = 10, 30, 50 only)}
  decoder  DECODE/<leaf>/rl/ml_exact.npz (ML(RL) on all 50,000 shots: pred, llr, logp, idx)
           FIT/<leaf>/rl_tsplit/{fit.json, fit_data.npz} (the split, the class scales, ML(T-hat) on the held-out shots)
Scope: the 72 fitted leaves (9 patches x {X, Z} x r in W5 = {10, 13, 30, 50}).

What is computed:
  (a) per leaf, on the held-out shots: errors of ML(RL) and ML(T-hat) (each against the archive's observable flips),
      and the 2 x 2 table (RL wrong, T-hat wrong);
  (b) m = (K_RL - K_T) / K_T over the 72 leaves, with the section 4 SE: delta method per shot, stratified by leaf,
      Var(m) = sum_leaf n_l Var_l(a - Q b) / K_T^2 with a, b the per-shot error indicators and Q = K_RL / K_T;
      also by r, without r = 13 (the section 5 sensitivity row), and the Holm step with P1;
  (c) the decomposition on the 54 leaves with Libra predictions (r = 10, 30, 50):
      (K_L - K_T) / K_T = (K_L - K_RL) / K_T + (K_RL - K_T) / K_T, each with the same SE, the identity in counts;
  (d) checks: the held-out shots follow the pre-registered split (alternating within each initialisation bitstring)
      and are disjoint from the fit sample; the archive's flips equal the stored ones; the stored ML(RL) decodes on the
      held-out shots equal the full-leaf ML(RL) decodes (predictions and float32 LLRs exactly, ln P(s) to rounding: the
      q10_7 X leaves were partly decoded with the frozen code, prereg section 9 entries 3-4); ML(T-hat)'s predictions
      equal the argmax of its posteriors;
      the class scales equal the last BHHH iterate of the fit's history; and the class structure (faults and expected
      fit-sample events per class, merges) rebuilt here from the RL DEM and section 2's rule equals the fit's record.
Usage: nice -n 19 python3 -I p4_check.py ARCHIVE DECODE FIT OUT_DIR
  -> OUT_DIR/p4_leaves.csv, OUT_DIR/p4_check.json (through qrc.outpaths: a re-run writes to results/reservoir/rebuild/
     unless QRC_OVERWRITE=1)
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

sys.dont_write_bytecode = True  # no __pycache__ in the shared code folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import out_path  # noqa: E402

PATCHES = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
BASES = ["X", "Z"]
W5 = [10, 13, 30, 50]
LIBRA_R = [10, 30, 50]
CM_RL = "correlated_matching_decoder_with_rl_optimized_prior"
LIBRA = "libra_decoder_with_rl_optimized_prior"
SHOTS = 50000
REPORTED = {"decoder_m": -0.00413, "decoder_se_m": 0.00091, "decoder_K_RL": 254802, "decoder_K_T": 255859,
            "decoder_decomposition": {"total": 0.00765, "suboptimality": 0.01131, "mismatch": -0.00367,
                                      "counts": [1682, 2489, -807]}}


def read_b8_bit0(path, n):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != n or np.any(raw >> 1):
        raise SystemExit(f"unexpected b8 layout in {path}")
    return raw & 1


def block_codes(path, n):
    sw = np.fromfile(path, dtype=np.uint8).reshape(n, -1)
    return np.unpackbits(sw, axis=1, bitorder="little")[:, :9].astype(np.int64) @ (1 << np.arange(9))


def mechanisms(dem):
    """(p, detectors, flips L0) per error instruction; the symptom is the XOR of all targets (decomposition marks and
    repeated targets handled as stim does)."""
    out = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        cnt, obs = collections.Counter(), 0
        for x in ins.targets_copy():
            if x.is_relative_detector_id():
                cnt[x.val] += 1
            elif x.is_logical_observable_id():
                obs ^= 1
        out.append((ins.args_copy()[0], tuple(sorted(k for k, v in cnt.items() if v % 2)), obs))
    return out


def classes(dem, r, n_fit, thr):
    """Section 2's classes rebuilt from the DEM: base symptom class (boundary = 1 detector; spacelike = 2 in one layer;
    timelike = 2 in adjacent layers at the same (x, y); diagonal = 2 in adjacent layers elsewhere; hyperedge = 3 or
    more), '+L' if the fault flips the observable, '@first' / '@last' if it touches detector layer 0 / r (the layer is
    the third coordinate, prereg section 9 entry 4). A class with fewer than thr expected events in the fit sample
    (n_fit x sum of p) is merged by dropping its layer tag, then its logical tag; anything still too small is pooled."""
    co = dem.get_detector_coordinates()
    t = {i: int(round(co[i][2])) for i in co}
    xy = {i: (co[i][0], co[i][1]) for i in co}
    mech = mechanisms(dem)
    names = []
    for p, ds, l in mech:
        if len(ds) == 1:
            base = "boundary"
        elif len(ds) == 2:
            i, j = ds
            base = ("spacelike" if t[i] == t[j] else "timelike" if abs(t[i] - t[j]) == 1 and xy[i] == xy[j]
                    else "diagonal" if abs(t[i] - t[j]) == 1 else "other")
        else:
            base = "hyperedge" if ds else "logical_only"
        name = base + ("+L" if l else "")
        ts = {t[i] for i in ds}
        if 0 in ts:
            name += "@first"
        elif r in ts:
            name += "@last"
        names.append(name)
    p = np.array([m[0] for m in mech])
    cur = list(names)

    def ev(lst):
        acc = collections.defaultdict(float)
        for k, pp in zip(lst, p):
            acc[k] += n_fit * pp
        return acc

    merged = {}
    for step in ("layer", "logical"):
        e = ev(cur)
        small = {k for k, v in e.items() if v < thr}
        for k in sorted(small):
            if step == "layer" and "@" in k:
                tgt = k.split("@")[0]
            elif step == "logical" and "+L" in k and "@" not in k:
                tgt = k.replace("+L", "")
            else:
                continue
            merged[k] = {"n_faults": sum(1 for x in cur if x == k), "expected_events": e[k], "into": tgt}
            cur = [tgt if x == k else x for x in cur]
    e = ev(cur)
    small = sorted(k for k, v in e.items() if v < thr)
    if small:
        for k in small:
            merged[k] = {"n_faults": sum(1 for x in cur if x == k), "expected_events": e[k], "into": "pooled"}
        cur = ["pooled" if x in small else x for x in cur]
    return mech, names, cur, merged


def ratio_se(num_ind, den_ind, leaf_of):
    """Q = sum(num) / sum(den) with the per-shot delta method stratified by leaf."""
    A, B = float(num_ind.sum()), float(den_ind.sum())
    Q = A / B
    v = 0.0
    for l in np.unique(leaf_of):
        s = leaf_of == l
        d = num_ind[s] - Q * den_ind[s]
        v += s.sum() * d.var(ddof=1)
    return Q, math.sqrt(v) / B


def leaf(arch, dec, fit, patch, basis, r):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir = os.path.join(arch, rel)
    f_dir = os.path.join(fit, rel, "rl_tsplit")
    fj = json.load(open(os.path.join(f_dir, "fit.json")))
    fd = np.load(os.path.join(f_dir, "fit_data.npz"), allow_pickle=False)
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    obs = read_b8_bit0(os.path.join(a_dir, "obs_flips_actual.b8"), SHOTS)
    code = block_codes(os.path.join(a_dir, "sweep_bits.b8"), SHOTS)
    chk = {}
    chk["fit_json_experiment"] = fj["experiment"].endswith(rel.replace(os.sep, "/"))
    chk["fit_status_complete"] = fj["status"] == "complete" and bool(fj["fit_attempts"][-1]["ok"])
    # the split: within each initialisation bitstring, successive shots alternate fit / held-out
    rank = np.zeros(SHOTS, np.int64)
    for c in np.unique(code):
        pos = np.flatnonzero(code == c)
        rank[pos] = np.arange(pos.size)
    held_rule = np.flatnonzero(rank % 2 == 1)
    fit_half = np.flatnonzero(rank % 2 == 0)
    test_idx, fit_idx = fd["test_idx"].astype(np.int64), fd["fit_idx"].astype(np.int64)
    chk["test_idx_equals_rule_odd_rank_within_bitstring"] = bool(np.array_equal(np.sort(test_idx), held_rule))
    chk["test_idx_sorted_unique"] = bool(np.all(np.diff(test_idx) > 0))
    chk["fit_idx_unique"] = bool(np.unique(fit_idx).size == fit_idx.size)
    chk["fit_idx_in_fit_half"] = bool(np.isin(fit_idx, fit_half).all())
    chk["fit_test_disjoint"] = bool(np.intersect1d(fit_idx, test_idx).size == 0)
    chk["n_fit_n_test"] = [int(fit_idx.size), int(test_idx.size)] == [fj["n_fit"], fj["n_test"]]
    o = obs[test_idx]
    chk["obs_test_equals_archive"] = bool(np.array_equal(fd["obs_test"], o))
    if not np.array_equal(ml["idx"], np.arange(SHOTS)):
        raise SystemExit(f"ML(RL) decode not complete and in order at {rel}")
    rl_pred = ml["pred"].reshape(SHOTS, -1)[:, 0][test_idx]
    chk["shipped_pred_equals_ml_rl"] = bool(np.array_equal(fd["shipped_pred"], rl_pred))
    chk["shipped_llr_equals_ml_rl"] = bool(np.array_equal(fd["shipped_llr"], ml["llr"][test_idx]))
    chk["shipped_logp_max_abs_diff_ml_rl"] = float(np.max(np.abs(fd["shipped_logp"] - ml["logp"][test_idx])))
    fl = fd["fitted_llr"].astype(np.float64)
    chk["fitted_pred_is_argmax"] = bool(np.array_equal(fd["fitted_pred"], (fl > 0).astype(fd["fitted_pred"].dtype)))
    chk["shipped_pred_is_argmax"] = bool(np.array_equal(fd["shipped_pred"], (fd["shipped_llr"] > 0).astype(np.uint8)))
    chk["fitted_llr_exact_zero"] = int(np.sum(fl == 0))
    chk["fitted_llr_nan"] = int(np.sum(~np.isfinite(fl)))
    # the class scales against the fit's own log
    hist = fj["history"]
    chk["scale_hat_equals_last_iterate"] = bool(hist[-1]["theta"] == fj["scale_hat"])
    chk["n_eval_equals_history"] = fj["n_eval"] == len(hist)
    chk["n_classes"] = len(fj["classes"]) == len(fj["scale_hat"]) == fj["fit_attempts"][-1]["classes"]
    chk["ll_hat_equals_last_iterate"] = hist[-1]["ll_per_shot"] == fj["ll_per_shot_fit"]
    last_steps = hist[-1].get("step_in_se", [])
    chk["last_step_max_abs_in_se"] = float(max(abs(x) for x in last_steps)) if last_steps else None
    chk["bhhh_steps"] = len(hist) - 1
    # the class structure rebuilt from the RL DEM
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    thr = float(fj["fit_attempts"][-1]["min_class_events"])
    mech, raw_names, cur, merged = classes(dem, r, fj["n_fit"], thr)
    nf = collections.Counter(cur)
    evc = collections.defaultdict(float)
    for k, m in zip(cur, mech):
        evc[k] += fj["n_fit"] * m[0]
    chk["classes_rebuilt_equal"] = sorted(nf) == sorted(fj["classes"])
    chk["n_faults_per_class_equal"] = dict(nf) == fj["n_faults_per_class"]
    chk["expected_events_per_class_max_rel_diff"] = (
        max(abs(evc[k] / fj["expected_events_per_class"][k] - 1) for k in fj["expected_events_per_class"])
        if chk["classes_rebuilt_equal"] else None)
    rec = fj.get("merged_classes", {})
    chk["merges_equal"] = ({k: v["into"] for k, v in merged.items()} == {k: v["into"] for k, v in rec.items()} and
                           all(merged[k]["n_faults"] == rec[k]["n_faults"] for k in rec))
    # errors on the held-out shots
    a = (fd["shipped_pred"].astype(np.uint8) != o).astype(np.int8)  # ML(RL) wrong
    b = (fd["fitted_pred"].astype(np.uint8) != o).astype(np.int8)  # ML(T-hat) wrong
    lp = os.path.join(a_dir, "decoding_results", LIBRA, "obs_flips_predicted.b8")
    c = (read_b8_bit0(lp, SHOTS)[test_idx] != o).astype(np.int8) if os.path.exists(lp) else None
    t = fj.get("test", {})
    chk["errors_equal_fit_json"] = (int(a.sum()) == t.get("shipped", {}).get("errors") and
                                    int(b.sum()) == t.get("fitted", {}).get("errors"))
    pr = t.get("paired_fitted_vs_shipped", {})
    chk["paired_equal_fit_json"] = (int(np.sum((b == 1) & (a == 0))) == pr.get("a_wrong_b_right") and
                                    int(np.sum((b == 0) & (a == 1))) == pr.get("a_right_b_wrong"))
    row = {"patch": patch, "basis": basis, "r": r, "n_test": int(test_idx.size), "n_classes": len(fj["classes"]),
           "K_RL": int(a.sum()), "K_T": int(b.sum()),
           "both_wrong": int(np.sum((a == 1) & (b == 1))), "RL_wrong_T_right": int(np.sum((a == 1) & (b == 0))),
           "RL_right_T_wrong": int(np.sum((a == 0) & (b == 1))), "both_right": int(np.sum((a == 0) & (b == 0))),
           "K_Libra": int(c.sum()) if c is not None else "",
           "Libra_wrong_T_right": int(np.sum((c == 1) & (b == 0))) if c is not None else "",
           "Libra_right_T_wrong": int(np.sum((c == 0) & (b == 1))) if c is not None else "",
           "Libra_wrong_RL_right": int(np.sum((c == 1) & (a == 0))) if c is not None else "",
           "Libra_right_RL_wrong": int(np.sum((c == 0) & (a == 1))) if c is not None else ""}
    row.update({f"chk_{k}": v for k, v in chk.items()})
    return row, a, b, c, chk


def pooled(rows, A, B, Cs, leaf_of, sel_leaves):
    s = np.isin(leaf_of, sel_leaves)
    KR, KT = int(A[s].sum()), int(B[s].sum())
    Q, se = ratio_se(A[s].astype(float), B[s].astype(float), leaf_of[s])
    m = Q - 1
    z = m / se
    return {"leaves": len(sel_leaves), "shots": int(s.sum()), "K_RL": KR, "K_T": KT, "m": m, "se": se, "z": z,
            "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "ci90": [m - 1.6448536269514722 * se, m + 1.6448536269514722 * se],
            "ci95": [m - 1.959963984540054 * se, m + 1.959963984540054 * se],
            "discordant": [int(sum(r_["RL_wrong_T_right"] for r_ in rows if r_["_i"] in sel_leaves)),
                           int(sum(r_["RL_right_T_wrong"] for r_ in rows if r_["_i"] in sel_leaves))]}


def main(arch, dec, fit, out_dir):
    t0 = time.time()
    rows, As, Bs, Cs, L = [], [], [], [], []
    for r in W5:
        for p in PATCHES:
            for b in BASES:
                row, a, bb, c, chk = leaf(arch, dec, fit, p, b, r)
                row["_i"] = len(rows)
                rows.append(row)
                As.append(a)
                Bs.append(bb)
                Cs.append(c)
                L.append(np.full(a.size, row["_i"]))
                bad = [k for k, v in chk.items() if v is False]
                print(f"{p} {b} r={r}: K_RL {row['K_RL']} K_T {row['K_T']} disc {row['RL_wrong_T_right']}/"
                      f"{row['RL_right_T_wrong']} checks failing: {bad}", flush=True)
    A, B, leaf_of = np.concatenate(As), np.concatenate(Bs), np.concatenate(L)
    allidx = [x["_i"] for x in rows]
    res = {"label": "reservoir, independent check of P4 (prereg section 5), 2026-10-09",
           "P4_all_72": pooled(rows, A, B, Cs, leaf_of, allidx),
           "by_r": {str(r): pooled(rows, A, B, Cs, leaf_of, [x["_i"] for x in rows if x["r"] == r]) for r in W5},
           "sensitivity_without_r13": pooled(rows, A, B, Cs, leaf_of, [x["_i"] for x in rows if x["r"] != 13])}
    # Holm with P1 (my P1 check: z = 16.79 with the covariance; p is far below P4's)
    p1 = float(2 * stats.norm.sf(16.79))
    p4 = res["P4_all_72"]["p_two_sided"]
    ps = sorted([("P1", p1), ("P4", p4)], key=lambda x: x[1])
    adj, run = {}, 0.0
    for k, (name, pv) in enumerate(ps):
        run = max(run, min(1.0, (len(ps) - k) * pv))
        adj[name] = run
    res["holm"] = {"p_raw": {"P1": p1, "P4": p4}, "p_adjusted": adj,
                   "note": "P1's p from the corrected delta-method SE (z = 16.79, results/reservoir/willow_check)"}
    # decomposition on the 54 leaves with Libra predictions
    lib = [x["_i"] for x in rows if x["r"] in LIBRA_R]
    s = np.isin(leaf_of, lib)
    C = np.concatenate([Cs[i] for i in lib])
    Ls = leaf_of[s]
    As_, Bs_ = A[s].astype(float), B[s].astype(float)
    KL, KR, KT = int(C.sum()), int(As_.sum()), int(Bs_.sum())
    tot, se_tot = ratio_se(C.astype(float) - Bs_, Bs_, Ls)
    sub, se_sub = ratio_se(C.astype(float) - As_, Bs_, Ls)
    mm, se_mm = ratio_se(As_ - Bs_, Bs_, Ls)
    res["decomposition_54"] = {
        "leaves": len(lib), "shots": int(s.sum()), "K_Libra": KL, "K_RL": KR, "K_T": KT,
        "counts": {"Libra_minus_T": KL - KT, "Libra_minus_RL": KL - KR, "RL_minus_T": KR - KT,
                   "identity_holds": (KL - KT) == (KL - KR) + (KR - KT)},
        "total": {"value": tot, "se": se_tot}, "suboptimality": {"value": sub, "se": se_sub},
        "mismatch_m54": {"value": mm, "se": se_mm},
        "identity_residual": tot - (sub + mm)}
    keys = [k for k in rows[0] if not k.startswith("_")]
    res["checks_all_leaves"] = {k[4:]: (all(x[k] for x in rows) if isinstance(rows[0][k], bool) else
                                       [min(x[k] for x in rows if x[k] is not None),
                                        max(x[k] for x in rows if x[k] is not None)])
                                for k in keys if k.startswith("chk_")}
    res["leaves_failing_any_check"] = [f"{x['patch']} {x['basis']} r{x['r']}" for x in rows
                                       if any(x[k] is False for k in keys if k.startswith("chk_"))]
    res["reported_by_decoder_for_comparison"] = REPORTED
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "p4_leaves.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: x[k] for k in keys} for x in rows])
    with open(out_path(os.path.join(out_dir, "p4_check.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("P4_all_72", "decomposition_54", "sensitivity_without_r13", "holm",
                                          "checks_all_leaves", "leaves_failing_any_check")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
