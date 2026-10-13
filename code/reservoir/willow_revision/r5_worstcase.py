"""R5 of the revision (review 1.3): an independent reproduction of the "95% worst cases" of decoder's post hoc check of
the decisions outside the band (deviation 13 of notes/decoder/willow_prereg.md), and of D2's derivation of them
(results/decoder/willow/revision/d2_worstcase/). POST HOC; read-only; nothing is decoded.

reservoir, 2026-10-13. Written from deviation 13's analysis text and the check's raw outputs, without reading decoder's
nonband_analyze.py or D2's script (rev_d2_worstcase.py). The estimators are the final analysis's (the paired gap of
prereg section 4 with entry 7's SE, imported from willow_d5check/d5_check.py, which reproduces final_round1.json); the
exhaustive per-round enumeration uses a vectorised copy of its point estimate, checked against it at random vertices.

Inputs: the 24 d = 5 leaves of P1's window (4 patches x X, Z x r = 10, 30, 50; ML's every 5th shot, 10,000 per leaf):
round 1's ladder (decode/<leaf>/rl_s5), Libra's shipped predictions and the outcomes (archive), K2's verdicts (rule A),
and the check's shot_lists.npz / .json and leaves/<k>.npz.

Per shot of the check: d1 = the round-1 decision (tau1's; outside the band the ladder's final one); final = tau3's where
tau3 ran, else tau2's; a flip is final != d1. A flip moves a D10 shot (Libra wrong, ML right) to both wrong, a D01 shot
to both right, a C00 shot (both right) to Libra right / ML wrong, and a C11 shot to Libra wrong / ML right. Per leaf the
2x2 table of (Libra, ML) changes by the expected flips (fractions allowed), and g_5 (rule A: rounds 10, 30, 50 less
K2's failing leaves; or no exclusion) and the pooled count gap (sum k_Libra - sum k_ML) / sum k_ML are recomputed.

Rows (deviation 13):
  base       no flips (must reproduce final_round1.json's g_5);
  estimate   D's flips as observed; each C type's flips at its pooled sample rate times each leaf's count of that type;
             the tau3-only term (the random subsample's change rate, pooled within D and within C) on the shots that
             tau2 did not move and tau3 did not visit ("unrun");
  worst      D as observed; C00 and C11 at one-sided 95% Clopper-Pearson bounds: the type whose flips push g the named
             way at its upper bound, the other at its lower bound ("lower g": C00 up, C11 down; "higher g" the mirror).
             Variants for the other type (zero; its observed rate) are sensitivity rows;
  looser     the worst case plus the tau3-only term at its upper bound on the unrun shots of the adverse types (D10 and
             C00 for lower g; D01 and C11 for higher g). D's unrun shots are counted (census). For C the unrun count is
             the leaf's population times an unrun share of the sample, taken five ways: pooled over all leaves (the
             reading that reproduces the published rows), pooled within the round count, the leaf's own, the
             population less the visited sample shots, the whole population.
Checks of D2's claims: the vertices of the pooled boxes plus random interior points; the Bonferroni level of each row;
per-round-count rates and bounds (C only: 64 vertices, at 95% each, at 1 - 0.05/k for k = 2, 3, 4 (an extreme that uses
k non-trivial bounds then holds jointly at 95%), and at a joint 95% for the whole box by Bonferroni; with the tau3-only
term per round count: all 2^18 vertices, exhaustively); other ways of combining the C strata for the estimate; D's flips
alone; the change of g_5 per added ML error by round count; the C weights.
Usage: python3 -I -B r5_worstcase.py ARCHIVE DECODE K2_DIR CHECK_DIR OUT_DIR -> OUT_DIR/r5_worstcase.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_gapmech"))
from gapmech import read_leaf  # noqa: E402  (also puts willow_d5check and code/reservoir on the path)
from d5_check import ARCH, g_stat, k2_verdicts, md5, paired, rel_of  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

TYPES = ("D10", "D01", "C00", "C11")
KEYS = ("C00", "C11", "t3D10", "t3D01", "t3C00", "t3C11")
SRC = {"C00": "C00", "C11": "C11", "t3D10": "t3D", "t3D01": "t3D", "t3C00": "t3C", "t3C11": "t3C"}
UNRUN = ("pooled_share", "round_share", "leaf_share", "pop_minus_visited", "whole_population")
ROUNDS = (10, 30, 50)
ML5 = "ml_tau2_1e-6-1e-8_0.1.npz"
SEED = 20261013  # random interior points and check vertices only
ALPHA = 0.05


def cp_up(k, n, a=ALPHA):
    return float(stats.beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0


def cp_lo(k, n, a=ALPHA):
    return float(stats.beta.ppf(a, k, n - k + 1)) if k > 0 else 0.0


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def load(arch, dec, k2_dir, chk_dir):
    S = np.load(os.path.join(chk_dir, "shot_lists.npz"), allow_pickle=False)
    J = json.load(open(os.path.join(chk_dir, "shot_lists.json")))
    k2 = k2_verdicts(k2_dir)
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    mine = {k: 0 for k in ("d1_vs_llr_t1_sign", "llr_t1_leaf_vs_list", "outside_band_final_llr_not_tau1",
                           "population_mismatch", "band_mismatch", "type_mismatch", "tau2_missing", "tau2_llr_nan",
                           "tau3_llr_nan", "tau3_pred_invalid", "targeted_without_tau3", "moved_rule_disagrees",
                           "random_rule_disagrees", "unvisited_rule_disagrees", "pred_t2_vs_llr_t2_sign",
                           "pred_t3_vs_llr_t3_sign")}
    problems, leaves = [], []
    if [str(x) for x in S["leaves"]] != [L["leaf"] for L in J["leaves"]]:
        problems.append("leaf order differs between shot_lists.npz and shot_lists.json")
    for k, info in enumerate(J["leaves"]):
        p, b, r = info["patch"], info["basis"], int(info["rounds"])
        rel = rel_of(5, p, b, r)
        if info["leaf"] != f"{ARCH}/{rel}":
            problems.append(f"{info['leaf']}: name differs from {rel}")
        rec = read_leaf(arch, dec, 5, p, b, r, chk)
        ml = np.load(os.path.join(dec, ARCH, rel, "rl_s5", ML5), allow_pickle=False)
        band = ml["redecoded"].astype(bool)
        out = ~band
        mine["outside_band_final_llr_not_tau1"] += int(np.sum(ml["llr"][out] != ml["llr_t1"][out]) +
                                                       np.isnan(ml["llr"][out]).sum())
        A = rec["preds"]["libra"] != rec["obs"]
        M = rec["ml"] != rec["obs"]
        base = {"n": int(rec["n"]), "n10": int(np.sum(A & ~M)), "n01": int(np.sum(~A & M)),
                "n11": int(np.sum(A & M)), "k_libra": int(A.sum()), "k_ml": int(M.sum())}
        pop = {"D10": int(np.sum(A & ~M & out)), "D01": int(np.sum(~A & M & out)),
               "C00": int(np.sum(~A & ~M & out)), "C11": int(np.sum(A & M & out))}
        mine["population_mismatch"] += int(pop != info["population"])
        mine["band_mismatch"] += int(int(band.sum()) != info["band"])
        if base["k_ml"] != info["errors_ML"] or base["k_libra"] != info["errors_libra"]:
            problems.append(f"{rel}: base error counts differ from shot_lists.json")
        rows = np.flatnonzero(S["leaf_id"] == k)
        R = np.load(os.path.join(chk_dir, "leaves", f"{k:02d}.npz"), allow_pickle=False)
        if not np.array_equal(R["row"], rows):
            problems.append(f"{rel}: leaf rows differ from the shot list")
            continue
        pos = S["pos"][rows]
        if not (np.array_equal(S["idx"][rows], np.asarray(ml["idx"], np.int64)[pos]) and np.all(out[pos])
                and np.array_equal(S["ml"][rows], rec["ml"][pos])
                and np.array_equal(S["libra"][rows], rec["preds"]["libra"][pos])
                and np.array_equal(S["obs"][rows], rec["obs"][pos])):
            problems.append(f"{rel}: shot-list shots, decisions or outcomes differ from the stored ones")
        typ = S["type"][rows].astype(str)
        aw, mw = S["libra"][rows] != S["obs"][rows], S["ml"][rows] != S["obs"][rows]
        exp_type = np.where(aw & ~mw, "D10", np.where(~aw & mw, "D01", np.where(~aw & ~mw, "C00", "C11")))
        mine["type_mismatch"] += int(np.sum(typ != exp_type))
        d1 = S["ml"][rows].astype(np.uint8)
        l1 = S["llr_t1"][rows].astype(np.float64)
        mine["d1_vs_llr_t1_sign"] += int(np.sum(d1 != (l1 > 0)))
        mine["llr_t1_leaf_vs_list"] += int(np.sum(R["llr_t1"] != S["llr_t1"][rows]))
        done2, done3 = R["done_t2"].astype(bool), R["done_t3"].astype(bool)
        mine["tau2_missing"] += int(np.sum(~done2))
        l2, l3 = R["llr_t2"].astype(np.float64), R["llr_t3"].astype(np.float64)
        mine["tau2_llr_nan"] += int(np.sum(done2 & np.isnan(l2)))
        mine["tau3_llr_nan"] += int(np.sum(done3 & np.isnan(l3)))
        mine["tau3_pred_invalid"] += int(np.sum(done3 & (R["pred_t3"] > 1)))
        ok2 = done2 & np.isfinite(l2) & (l2 != 0)
        mine["pred_t2_vs_llr_t2_sign"] += int(np.sum(R["pred_t2"][ok2] != (l2[ok2] > 0)))
        ok3 = done3 & np.isfinite(l3) & (l3 != 0)
        mine["pred_t3_vs_llr_t3_sign"] += int(np.sum(R["pred_t3"][ok3] != (l3[ok3] > 0)))
        reason = R["t3_reason"].astype(int)
        # tau2's "moved" rule of deviation 13, recomputed: decision changed, |P2 - 1/2| < 0.1, |P2 - P1| >= 0.05, NaN
        P1, P2 = sig(l1), sig(l2)
        moved = (R["pred_t2"] != d1) | (np.abs(P2 - 0.5) < 0.1) | (np.abs(P2 - P1) >= 0.05) | np.isnan(l2)
        u = S["u"][rows]
        mine["moved_rule_disagrees"] += int(np.sum(moved != (reason == 1)))
        mine["random_rule_disagrees"] += int(np.sum((~moved & (u < 0.1)) != (reason == 2)))
        mine["unvisited_rule_disagrees"] += int(np.sum((~moved & (u >= 0.1)) != (reason == 0)))
        mine["targeted_without_tau3"] += int(np.sum((reason == 1) & ~done3))
        final = np.where(done3, R["pred_t3"], R["pred_t2"])
        flip = final != d1
        flip2 = R["pred_t2"] != d1
        L = {"leaf": rel, "patch": p, "basis": b, "r": r, "base": base, "pop": pop,
             "excluded_A": rel in failing, "rows": {}}
        for t in TYPES:
            m = typ == t
            L["rows"][t] = {"n": int(m.sum()), "flips": int(np.sum(flip & m)), "flips_tau2": int(np.sum(flip2 & m)),
                            "targeted": int(np.sum((reason == 1) & m)),
                            "targeted_flips": int(np.sum((reason == 1) & flip & m)),
                            "random": int(np.sum((reason == 2) & m)),
                            "random_flips": int(np.sum((reason == 2) & flip & m)),
                            "unvisited": int(np.sum((reason == 0) & m)),
                            "unvisited_flips": int(np.sum((reason == 0) & flip & m))}
        leaves.append(L)
    return leaves, failing, problems, chk, mine


def stats_of(leaves, add, rule):
    """g_5 (entry 7's SE) and the count gap after the per-leaf changes add[i] = (d_n10, d_n01, d_n11)."""
    by = {}
    tot_a = tot_m = 0.0
    for L, (x10, x01, x11) in zip(leaves, add):
        if rule == "A" and L["excluded_A"]:
            continue
        n10, n01, n11 = L["base"]["n10"] + x10, L["base"]["n01"] + x01, L["base"]["n11"] + x11
        kA, kM = n10 + n11, n01 + n11
        tot_a += kA
        tot_m += kM
        by.setdefault((L["patch"], L["basis"]), []).append((L["r"], kA, kM, n11, L["base"]["n"]))
    prs = []
    for _, v in sorted(by.items()):
        if len(v) < 2:
            continue
        v.sort()
        prs.append(paired([x[0] for x in v], [x[1] for x in v], [x[2] for x in v], [x[3] for x in v],
                          [x[4] for x in v]))
    g = g_stat(prs)
    return {"g5": g["g"], "se": g["se_inflated"], "ci95": g["ci95"], "series": len(prs),
            "count_gap": (tot_a - tot_m) / tot_m, "count_gap_errors": tot_a - tot_m, "k_libra": tot_a, "k_ml": tot_m}


def unrun_pop(leaves):
    """For the tau3-only term on C: per leaf and type, the number of C shots that tau2 did not move and tau3 did not
    visit, five ways (UNRUN)."""
    def share(sel, t):
        n = sum(L["rows"][t]["n"] for L in sel)
        return sum(L["rows"][t]["unvisited"] for L in sel) / n
    pooled = {t: share(leaves, t) for t in ("C00", "C11")}
    by_r = {r: {t: share([L for L in leaves if L["r"] == r], t) for t in ("C00", "C11")} for r in ROUNDS}
    U = {m: [] for m in UNRUN}
    for L in leaves:
        for m in U:
            U[m].append({})
        for t in ("C00", "C11"):
            R, pop = L["rows"][t], L["pop"][t]
            U["pooled_share"][-1][t] = pop * pooled[t]
            U["round_share"][-1][t] = pop * by_r[L["r"]][t]
            U["leaf_share"][-1][t] = pop * R["unvisited"] / R["n"] if R["n"] else pop * pooled[t]
            U["pop_minus_visited"][-1][t] = float(pop - (R["n"] - R["unvisited"]))
            U["whole_population"][-1][t] = float(pop)
    return U, pooled, by_r


def parts(L, q, Ui):
    """Expected flips of each type in leaf L at rates q: D's observed flips plus the tau3-only term on D's unrun shots
    (rates t3D10, t3D01), C's flips at rates C00, C11 on the population plus the tau3-only term on Ui (t3C00, t3C11)."""
    R = L["rows"]
    return (R["D10"]["flips"] + q.get("t3D10", 0.0) * R["D10"]["unvisited"],
            R["D01"]["flips"] + q.get("t3D01", 0.0) * R["D01"]["unvisited"],
            q.get("C00", 0.0) * L["pop"]["C00"] + q.get("t3C00", 0.0) * Ui["C00"],
            q.get("C11", 0.0) * L["pop"]["C11"] + q.get("t3C11", 0.0) * Ui["C11"])


def changes(leaves, rr, U, per_round=False):
    """Per-leaf (d_n10, d_n01, d_n11). D10 flip: n10 - 1, n11 + 1; D01 flip: n01 - 1; C00 flip: n01 + 1; C11 flip:
    n11 - 1, n10 + 1."""
    out = []
    for i, L in enumerate(leaves):
        f10, f01, c00, c11 = parts(L, rr[L["r"]] if per_round else rr, U[i])
        out.append((-f10 + c11, -f01 + c00, f10 - c11))
    return out


def flip_totals(leaves, rr, U, rule, per_round=False):
    tot = np.zeros(4)
    for i, L in enumerate(leaves):
        if rule == "A" and L["excluded_A"]:
            continue
        tot += np.array(parts(L, rr[L["r"]] if per_round else rr, U[i]))
    return dict(zip(TYPES, (float(x) for x in tot)))


def rate_box(k, n, a=ALPHA):
    return {"flips": k, "n": n, "rate": k / n if n else 0.0, "lo": cp_lo(k, n, a), "up": cp_up(k, n, a)}


def materialize(spec, B):
    return {key: {"up": B[SRC[key]]["up"], "lo": B[SRC[key]]["lo"], "est": B[SRC[key]]["rate"], "zero": 0.0}[w]
            for key, w in spec.items()}


def nontrivial(spec, B):
    st = {(SRC[key], w) for key, w in spec.items()
          if (w == "up" and B[SRC[key]]["up"] < 1) or (w == "lo" and B[SRC[key]]["lo"] > 0)}
    return sorted(st)


def row_specs(other):
    o = other
    return {"estimate": {k: "est" for k in KEYS},
            "worst_lower_g": {"C00": "up", "C11": o},
            "worst_higher_g": {"C00": o, "C11": "up"},
            "worst_lower_g_with_tau3_only": {"C00": "up", "C11": o, "t3D10": "up", "t3C00": "up", "t3D01": o,
                                             "t3C11": o},
            "worst_higher_g_with_tau3_only": {"C00": o, "C11": "up", "t3D01": "up", "t3C11": "up", "t3D10": o,
                                              "t3C00": o}}


def extremes(vals):
    i_min = int(np.argmin([v[0] for v in vals]))
    i_max = int(np.argmax([v[0] for v in vals]))
    return {"points": len(vals), "g5_min": vals[i_min][0], "g5_max": vals[i_max][0],
            "count_gap_min": min(v[1] for v in vals), "count_gap_max": max(v[1] for v in vals),
            "count_gap_errors_min": min(v[2] for v in vals), "count_gap_errors_max": max(v[2] for v in vals),
            "argmin_g5": vals[i_min][3], "argmax_g5": vals[i_max][3]}


# ---- exhaustive enumeration of per-round boxes (vectorised point estimate of d5_check.paired / g_stat) ----
def wls_slope(x, y, var):
    w = 1.0 / var
    Sw, Sx, Sy = w.sum(1), (w * x).sum(1), (w * y).sum(1)
    Sxx, Sxy = (w * x * x).sum(1), (w * x * y).sum(1)
    return (Sw * Sxy - Sx * Sy) / (Sw * Sxx - Sx * Sx)


def paired_point(r, kA, kM, n11, n):
    """d5_check.paired's point estimates (d, eps_ml), vectorised over the rows of kA, kM, n11 (shape V x rounds)."""
    EA, EM = kA / n, kM / n
    vA = 4 * EA * (1 - EA) / (n * (1 - 2 * EA) ** 2)
    vM = 4 * EM * (1 - EM) / (n * (1 - 2 * EM) ** 2)
    cAM = 4 * (n11 / n - EA * EM) / (n * (1 - 2 * EA) * (1 - 2 * EM))
    yA, yM = np.log(1 - 2 * EA), np.log(1 - 2 * EM)
    bM = wls_slope(r, yM, vM)
    bz = wls_slope(r, yA - yM, vA + vM - 2 * cAM)
    eM = (1 - np.exp(bM)) / 2
    return (1 - np.exp(bM + bz)) / 2 - eM, eM


def enumerate_rounds(leaves, rule, corner_changes, bits):
    """g_5 (point) and the count gap at every vertex of a per-round box. Vertex v: bits j*bits .. (j+1)*bits - 1 are
    round count ROUNDS[j]'s corner code; corner_changes[i][c] is leaf i's (d10, d01, d11) at corner code c."""
    V = 1 << (3 * bits)
    v = np.arange(V)
    code = {r: (v >> (bits * j)) & ((1 << bits) - 1) for j, r in enumerate(ROUNDS)}
    totA, totM, sumD, sumE = np.zeros(V), np.zeros(V), np.zeros(V), np.zeros(V)
    series = {}
    for i, L in enumerate(leaves):
        if not (rule == "A" and L["excluded_A"]):
            series.setdefault((L["patch"], L["basis"]), []).append(i)
    for _, idx in sorted(series.items()):
        cols = []
        for i in sorted(idx, key=lambda j: leaves[j]["r"]):
            L = leaves[i]
            ch = corner_changes[i][code[L["r"]]]
            n10, n01, n11 = L["base"]["n10"] + ch[:, 0], L["base"]["n01"] + ch[:, 1], L["base"]["n11"] + ch[:, 2]
            kA, kM = n10 + n11, n01 + n11
            totA += kA
            totM += kM
            cols.append((L["r"], kA, kM, n11, L["base"]["n"]))
        if len(cols) < 2:
            continue
        d, e = paired_point(np.array([c[0] for c in cols], float), np.stack([c[1] for c in cols], 1),
                            np.stack([c[2] for c in cols], 1), np.stack([c[3] for c in cols], 1),
                            np.array([c[4] for c in cols], float))
        sumD += d
        sumE += e
    return sumD / sumE, (totA - totM) / totM, totA - totM


def round_boxes(leaves, a):
    """Per round count: C00 and C11 flip rates of the sample, and the tau3-only rates of D's and C's random subsample,
    with one-sided bounds at level 1 - a."""
    out = {}
    for r in ROUNDS:
        sel = [L for L in leaves if L["r"] == r]
        s = lambda t, k: sum(L["rows"][t][k] for L in sel)  # noqa: E731
        out[r] = {"C00": rate_box(s("C00", "flips"), s("C00", "n"), a),
                  "C11": rate_box(s("C11", "flips"), s("C11", "n"), a),
                  "t3D": rate_box(s("D10", "random_flips") + s("D01", "random_flips"),
                                  s("D10", "random") + s("D01", "random"), a),
                  "t3C": rate_box(s("C00", "random_flips") + s("C11", "random_flips"),
                                  s("C00", "random") + s("C11", "random"), a)}
    return out


def corner_rates(box, keys, c):
    return {k: box[SRC[k]]["up" if (c >> j) & 1 else "lo"] for j, k in enumerate(keys)}


def per_round_exhaustive(leaves, rule, boxes, keys, Umode, check_n, rng, interior_n=0):
    bits = len(keys)
    cc = []
    for i, L in enumerate(leaves):
        cc.append(np.array([changes([L], corner_rates(boxes[L["r"]], keys, c), [Umode[i]])[0]
                            for c in range(1 << bits)]))
    g, cg, cge = enumerate_rounds(leaves, rule, cc, bits)
    beyond = 0
    for _ in range(interior_n):  # random interior points of the box, with the final analysis's estimator
        rr = {r: {k: boxes[r][SRC[k]]["lo"] + rng.random() * (boxes[r][SRC[k]]["up"] - boxes[r][SRC[k]]["lo"])
                  for k in keys} for r in ROUNDS}
        s = stats_of(leaves, changes(leaves, rr, Umode, per_round=True), rule)
        beyond += int(not (g.min() - 1e-12 <= s["g5"] <= g.max() + 1e-12) or
                      not (cg.min() - 1e-12 <= s["count_gap"] <= cg.max() + 1e-12))

    def decode(v):
        return {f"r{r}|{k}": ("up" if (v >> (bits * j + jj)) & 1 else "lo")
                for j, r in enumerate(ROUNDS) for jj, k in enumerate(keys)}

    def rates_of(v):
        return {r: corner_rates(boxes[r], keys, (v >> (bits * j)) & ((1 << bits) - 1)) for j, r in enumerate(ROUNDS)}

    def n_bounds(v):
        """Non-trivial one-sided bounds that vertex v uses (an upper bound below 1, a lower bound above 0)."""
        n = 0
        for j, r in enumerate(ROUNDS):
            for jj, k in enumerate(keys):
                b = boxes[r][SRC[k]]
                n += int(b["up"] < 1) if (v >> (bits * j + jj)) & 1 else int(b["lo"] > 0)
        return n
    i_min, i_max = int(np.argmin(g)), int(np.argmax(g))
    j_min, j_max = int(np.argmin(cg)), int(np.argmax(cg))
    vs = [i_min, i_max, j_min, j_max] + [int(x) for x in rng.integers(0, g.size, check_n)]
    dev = 0.0
    for v in vs:
        s = stats_of(leaves, changes(leaves, rates_of(v), Umode, per_round=True), rule)
        dev = max(dev, abs(s["g5"] - g[v]), abs(s["count_gap"] - cg[v]))
    return {"vertices": int(g.size), "finite": int(np.isfinite(g).sum()), "g5_min": float(g[i_min]),
            "g5_max": float(g[i_max]), "count_gap_min": float(cg[j_min]), "count_gap_max": float(cg[j_max]),
            "count_gap_errors_min": float(cge[j_min]), "count_gap_errors_max": float(cge[j_max]),
            "argmin_g5": decode(i_min), "argmax_g5": decode(i_max), "argmin_count_gap": decode(j_min),
            "argmax_count_gap": decode(j_max),
            "nontrivial_bounds_used": {"g5_min": n_bounds(i_min), "g5_max": n_bounds(i_max),
                                       "count_gap_min": n_bounds(j_min), "count_gap_max": n_bounds(j_max)},
            "checked_vertices": len(vs), "max_abs_diff_vs_d5_check_estimator": dev, "interior_points": interior_n,
            "interior_outside_vertex_range": beyond}


def main(arch, dec, k2_dir, chk_dir, out_dir):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    leaves, failing, problems, chk, mine = load(arch, dec, k2_dir, chk_dir)
    U, share_pooled, share_round = unrun_pop(leaves)
    keys = ("n", "flips", "flips_tau2", "targeted", "targeted_flips", "random", "random_flips", "unvisited",
            "unvisited_flips")
    agg = {t: {k: sum(L["rows"][t][k] for L in leaves) for k in keys} for t in TYPES}
    for t in TYPES:
        agg[t]["population"] = sum(L["pop"][t] for L in leaves)
    B = {"C00": rate_box(agg["C00"]["flips"], agg["C00"]["n"]), "C11": rate_box(agg["C11"]["flips"], agg["C11"]["n"]),
         "t3D": rate_box(agg["D10"]["random_flips"] + agg["D01"]["random_flips"],
                         agg["D10"]["random"] + agg["D01"]["random"]),
         "t3C": rate_box(agg["C00"]["random_flips"] + agg["C11"]["random_flips"],
                         agg["C00"]["random"] + agg["C11"]["random"])}
    wts = [(L["pop"]["C00"] + L["pop"]["C11"]) / (L["rows"]["C00"]["n"] + L["rows"]["C11"]["n"]) for L in leaves]
    res = {"label": "reservoir, revision R5 (review 1.3): independent reproduction of the non-band check's worst cases "
                    "and of D2's derivation of them. POST HOC; read-only",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "inputs_md5": {f: md5(os.path.join(chk_dir, f)) for f in ("shot_lists.npz", "shot_lists.json")},
           "problems": problems, "read_leaf_checks": chk, "my_checks": mine,
           "k2_failing_in_window": sorted(f for f in failing if f.endswith(("r10", "r30", "r50"))),
           "strata": agg, "pooled_rates_and_cp95": B, "C_weight_range": [min(wts), max(wts)],
           "C_unrun_share": {"pooled": share_pooled, "by_round": {str(r): v for r, v in share_round.items()}},
           "rows": {}}
    res["inputs_md5"].update({f"leaves/{k:02d}.npz": md5(os.path.join(chk_dir, "leaves", f"{k:02d}.npz"))
                              for k in range(len(leaves))})
    zero = [(0.0, 0.0, 0.0)] * len(leaves)
    for rule in ("A", "none"):
        res["rows"][f"{rule}|base"] = stats_of(leaves, zero, rule)
    # the published rows (non-adverse C type at its lower bound; tau3-only C term on the pooled unrun share) and the
    # sensitivity variants
    for other in ("lo", "zero", "est"):
        for name, spec in row_specs(other).items():
            if other != "lo" and name == "estimate":
                continue
            for mode in (UNRUN if "tau3" in name else ("pooled_share",)):
                rr = materialize(spec, B)
                for rule in ("A", "none"):
                    s = stats_of(leaves, changes(leaves, rr, U[mode]), rule)
                    s["flips_total"] = flip_totals(leaves, rr, U[mode], rule)
                    s["bounds_used"] = nontrivial(spec, B)
                    s["bonferroni_joint_level"] = 1 - ALPHA * len(s["bounds_used"])
                    tag = name if other == "lo" else f"{name}|other_type_{other}"
                    tag = tag if mode == "pooled_share" else f"{tag}|unrun_C_{mode}"
                    res["rows"][f"{rule}|{tag}"] = s
    # vertices of the pooled boxes, and random interior points
    vert = {}
    for tag, ks, modes in (("C_only", ("C00", "C11"), ("pooled_share",)),
                           ("with_tau3_only", KEYS, ("pooled_share", "leaf_share"))):
        for mode in modes:
            for rule in ("A", "none"):
                vals = []
                for corner in itertools.product(("lo", "up"), repeat=len(ks)):
                    spec = dict(zip(ks, corner))
                    s = stats_of(leaves, changes(leaves, materialize(spec, B), U[mode]), rule)
                    vals.append((s["g5"], s["count_gap"], s["count_gap_errors"], spec))
                e = extremes(vals)
                inner = []
                for _ in range(500):
                    rr = {k: B[SRC[k]]["lo"] + rng.random() * (B[SRC[k]]["up"] - B[SRC[k]]["lo"]) for k in ks}
                    s = stats_of(leaves, changes(leaves, rr, U[mode]), rule)
                    inner.append((s["g5"], s["count_gap"]))
                e["interior_points"] = len(inner)
                e["interior_outside_vertex_range"] = int(sum(
                    not (e["g5_min"] - 1e-12 <= v[0] <= e["g5_max"] + 1e-12) or
                    not (e["count_gap_min"] - 1e-12 <= v[1] <= e["count_gap_max"] + 1e-12) for v in inner))
                vert[f"{tag}|unrun_C_{mode}|{rule}" if tag != "C_only" else f"{tag}|{rule}"] = e
    res["vertex_extremes_pooled"] = vert
    # per-round-count rates and bounds
    box95 = round_boxes(leaves, ALPHA)
    nnt_C = sum(int(box95[r][t]["up"] < 1) + int(box95[r][t]["lo"] > 0) for r in ROUNDS for t in ("C00", "C11"))
    nnt_all = nnt_C + sum(int(box95[r][t]["up"] < 1) + int(box95[r][t]["lo"] > 0) for r in ROUNDS
                          for t in ("t3D", "t3C"))
    res["per_round_rates_cp95"] = {str(r): box95[r] for r in ROUNDS}
    res["per_round_nontrivial_bounds"] = {"C_only": nnt_C, "with_tau3_only": nnt_all}
    per = {}
    for level, aC, aT in (("each95", ALPHA, ALPHA), ("joint95_bonferroni", ALPHA / nnt_C, ALPHA / nnt_all)):
        boxC = round_boxes(leaves, aC)
        for rule in ("A", "none"):
            est = {r: {t: box95[r][t]["rate"] for t in ("C00", "C11")} for r in ROUNDS}
            e = per_round_exhaustive(leaves, rule, boxC, ("C00", "C11"), U["pooled_share"], 100, rng,
                                     500 if level == "each95" else 0)
            e["estimate"] = stats_of(leaves, changes(leaves, est, U["pooled_share"], per_round=True), rule)
            per[f"C_only|{level}|{rule}"] = e
        boxT = round_boxes(leaves, aT)
        for mode in ("round_share", "pooled_share"):
            for rule in ("A", "none"):
                per[f"with_tau3_only|unrun_C_{mode}|{level}|{rule}"] = per_round_exhaustive(
                    leaves, rule, boxT, KEYS, U[mode], 100, rng, 500 if level == "each95" else 0)
    # C only, with the bounds at 1 - 0.05/k: each extreme of the 95% box uses k non-trivial bounds (see
    # nontrivial_bounds_used), so the extreme of this box that uses k bounds holds jointly at 95% by Bonferroni
    for k in (2, 3, 4):
        boxC = round_boxes(leaves, ALPHA / k)
        for rule in ("A", "none"):
            per[f"C_only|alpha_over_{k}|{rule}"] = per_round_exhaustive(leaves, rule, boxC, ("C00", "C11"),
                                                                         U["pooled_share"], 20, rng, 0)
    res["per_round_boxes"] = per
    # other ways of combining the C strata for the estimate, and D's flips alone
    alt = {}
    for rule in ("A", "none"):
        o = {"pooled_per_type": res["rows"][f"{rule}|estimate"]}
        est = {r: {t: box95[r][t]["rate"] for t in ("C00", "C11")} for r in ROUNDS}
        o["per_type_per_round"] = stats_of(leaves, changes(leaves, est, U["pooled_share"], per_round=True), rule)
        add = []
        for i, L in enumerate(leaves):
            R = L["rows"]
            q = {t: (R[t]["flips"] / R[t]["n"] if R[t]["n"] else 0.0) for t in ("C00", "C11")}
            add.append(changes([L], q, [U["pooled_share"][i]])[0])
        o["per_leaf_per_type"] = stats_of(leaves, add, rule)
        add = []
        for L in leaves:  # Horvitz-Thompson within the leaf: each sampled C shot stands for pop_C / n_C shots
            R = L["rows"]
            w = (L["pop"]["C00"] + L["pop"]["C11"]) / (R["C00"]["n"] + R["C11"]["n"])
            f10, f01 = R["D10"]["flips"], R["D01"]["flips"]
            c00, c11 = w * R["C00"]["flips"], w * R["C11"]["flips"]
            add.append((-f10 + c11, -f01 + c00, f10 - c11))
        o["per_leaf_horvitz_thompson"] = stats_of(leaves, add, rule)
        gs = [v["g5"] for v in o.values()]
        o["range_of_combinations_g5"] = [min(gs), max(gs)]
        o["d_flips_only"] = stats_of(leaves, changes(leaves, {}, U["pooled_share"]), rule)
        o["d_flips_net_ml_errors_by_round"] = {
            str(r): sum(L["rows"]["D10"]["flips"] - L["rows"]["D01"]["flips"] for L in leaves
                        if L["r"] == r and not (rule == "A" and L["excluded_A"])) for r in ROUNDS}
        alt[rule] = o
    res["estimate_by_combination"] = alt
    # the change of g_5 per added ML error (an n01 shot), by round count: forward differences of h errors on one leaf,
    # divided by h (h = 1; and h = 10, the step that D2's table turns out to use)
    fd = {}
    for h in (1, 10):
        for rule in ("A", "none"):
            g0 = res["rows"][f"{rule}|base"]["g5"]
            by = {r: [] for r in ROUNDS}
            for i, L in enumerate(leaves):
                if rule == "A" and L["excluded_A"]:
                    continue
                add = [(0.0, float(h), 0.0) if j == i else (0.0, 0.0, 0.0) for j in range(len(leaves))]
                by[L["r"]].append((stats_of(leaves, add, rule)["g5"] - g0) / h)
            fd[f"h{h}|{rule}"] = {str(r): {"leaves": len(v), "mean_points": 100 * float(np.mean(v)),
                                           "min_points": 100 * min(v), "max_points": 100 * max(v)}
                                  for r, v in by.items()}
    res["dg5_per_added_ml_error"] = fd
    res["per_leaf"] = leaves
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r5_worstcase.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    for k, v in res["rows"].items():
        print(f"{k:80s} g5 {100 * v['g5']:+.2f}%  count gap {100 * v['count_gap']:+.2f}% "
              f"({v['count_gap_errors']:+.0f})", flush=True)
    for k, v in per.items():
        print(f"per round {k:55s} g5 {100 * v['g5_min']:+.2f}% .. {100 * v['g5_max']:+.2f}%  count gap "
              f"{100 * v['count_gap_min']:+.2f}% .. {100 * v['count_gap_max']:+.2f}%  check {v['max_abs_diff_vs_d5_check_estimator']:.1e}",
              flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
