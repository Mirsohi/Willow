"""R5 of the revision (review 1.3): an independent reproduction of the "95% worst cases" of decoder's post hoc check of
the decisions outside the band (deviation 13 of notes/decoder/willow_prereg.md; D2). POST HOC; read-only; nothing is
decoded.

reservoir, 2026-10-13. Written from deviation 13's analysis text and the check's raw outputs, without decoder's
nonband_analyze.py or rev_d2_worstcase.py. The estimators are the final analysis's (paired gap of prereg section 4 with
entry 7's SE, imported from willow_d5check/d5_check.py, which reproduces final_round1.json).

Inputs: the 24 d = 5 leaves of P1's window (4 patches x X, Z x r = 10, 30, 50; ML's every 5th shot, 10,000 per leaf);
round 1's ladder (decode/<leaf>/rl_s5), Libra's shipped predictions and the outcomes (archive), K2's verdicts (rule A),
and the check's shot_lists.npz / .json and leaves/<k>.npz.
Per shot of the check: d1 = the ladder's (tau1's) decision; final = tau3's where tau3 ran, else tau2's; a flip is
final != d1. A flip moves a D10 shot (Libra wrong, ML right) to both wrong, a D01 shot to both right, a C00 shot (both
right) to Libra right / ML wrong, and a C11 shot to Libra wrong / ML right. Per leaf the 2x2 table of (Libra, ML) is
changed by the flips, and g_5 (rule A: rounds 10, 30, 50 less K2's failing leaves; or no exclusion) and the pooled
count gap (sum k_Libra - sum k_ML) / sum k_ML are recomputed.
Rows (my reading of deviation 13, checked against the published table):
  base       no flips;
  estimate   D's flips as observed; each C type's flips at its pooled sample rate times each leaf's population of that
             type; the tau3-only term at the random subsample's rate (pooled within D and within C) times the shots not
             moved by tau2 and not visited by tau3;
  worst      D as observed; C00 and C11 at one-sided 95% Clopper-Pearson bounds of their pooled rates, upper for the
             type whose flips push g in the named direction and lower for the other ("lower g" = the direction that
             adds ML errors: C00 up, C11 down; "higher g" the mirror); with "_with_tau3_only", also the tau3-only rates
             at their upper bounds on the unvisited shots of the adverse types (D10 and C00, or D01 and C11).
Extensions (D2's claims, checked independently): every vertex of the box of allowed pooled rates (g_5's extremes);
per-round-count rates and bounds; alternative ways of combining the C strata for the estimate; the Bonferroni level
of each row.
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
from gapmech import read_leaf  # noqa: E402
from d5_check import g_stat, k2_verdicts, paired  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

TYPES = ("D10", "D01", "C00", "C11")


def cp_up(k, n):
    return float(stats.beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0


def cp_lo(k, n):
    return float(stats.beta.ppf(0.05, k, n - k + 1)) if k > 0 else 0.0


def load(arch, dec, k2_dir, chk_dir):
    S = np.load(os.path.join(chk_dir, "shot_lists.npz"), allow_pickle=False)
    J = json.load(open(os.path.join(chk_dir, "shot_lists.json")))
    k2 = k2_verdicts(k2_dir)
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    leaves, problems = [], []
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    for k, info in enumerate(J["leaves"]):
        name = info["leaf"]
        rel = name.split("/", 1)[1]
        p, b, r = rel.split("/")[0][len("d5_at_"):], rel.split("/")[1], int(rel.split("/")[2][1:])
        rec = read_leaf(arch, dec, 5, p, b, r, chk)
        A = rec["preds"]["libra"] != rec["obs"]
        M = rec["ml"] != rec["obs"]
        base = {"n": int(rec["n"]), "n10": int(np.sum(A & ~M)), "n01": int(np.sum(~A & M)), "n11": int(np.sum(A & M))}
        if int(M.sum()) != info["errors_ML"] or int(A.sum()) != info["errors_libra"]:
            problems.append(f"{rel}: base error counts differ from shot_lists.json")
        rows = np.flatnonzero(S["leaf_id"] == k)
        R = np.load(os.path.join(chk_dir, "leaves", f"{k:02d}.npz"), allow_pickle=False)
        if not np.array_equal(R["row"], rows):
            problems.append(f"{rel}: leaf rows differ from the shot list")
        pos = S["pos"][rows]
        if not (np.array_equal(S["ml"][rows], rec["ml"][pos]) and np.array_equal(S["libra"][rows],
                                                                                   rec["preds"]["libra"][pos])
                and np.array_equal(S["obs"][rows], rec["obs"][pos])):
            problems.append(f"{rel}: shot-list decisions or outcomes differ from the stored ones")
        typ = S["type"][rows]
        a_w, m_w = S["libra"][rows] != S["obs"][rows], S["ml"][rows] != S["obs"][rows]
        exp_type = np.where(a_w & ~m_w, "D10", np.where(~a_w & m_w, "D01", np.where(~a_w & ~m_w, "C00", "C11")))
        if not np.array_equal(typ, exp_type):
            problems.append(f"{rel}: stored types differ from (Libra, ML) correctness")
        d1 = S["ml"][rows]
        done3 = R["done_t3"].astype(bool)
        final = np.where(done3, R["pred_t3"], R["pred_t2"])
        reason = R["t3_reason"].astype(int)
        if not R["done_t2"].all():
            problems.append(f"{rel}: rows without tau2")
        targeted_no_t3 = int(np.sum((reason == 1) & ~done3))
        flip = final != d1
        leaves.append({"rel": rel, "patch": p, "basis": b, "r": r, "base": base, "pop": info["population"],
                       "excluded_A": f"d5_at_{p}/{b}/r{r:02d}" in failing or rel in failing,
                       "rows": {t: {"n": int(np.sum(typ == t)), "flips": int(np.sum(flip & (typ == t))),
                                    "random": int(np.sum((reason == 2) & (typ == t))),
                                    "random_flips": int(np.sum((reason == 2) & flip & (typ == t))),
                                    "unvisited": int(np.sum((reason == 0) & (typ == t))),
                                    "targeted": int(np.sum((reason == 1) & (typ == t)))} for t in TYPES},
                       "targeted_without_t3": targeted_no_t3})
    return leaves, failing, problems, chk


def stats_of(leaves, add, rule):
    """g_5 (entry 7's SE) and the count gap after adding the per-leaf changes add[i] = (d_n10, d_n01, d_n11)."""
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
    for key, v in sorted(by.items()):
        if len(v) < 2:
            continue
        v.sort()
        prs.append(paired([x[0] for x in v], [x[1] for x in v], [x[2] for x in v], [x[3] for x in v],
                          [x[4] for x in v]))
    g = g_stat(prs)
    return {"g5": g["g"], "se": g["se_inflated"], "ci95": g["ci95"], "series": len(prs),
            "count_gap": (tot_a - tot_m) / tot_m, "count_gap_errors": tot_a - tot_m}


def changes(leaves, rates, scope="pooled"):
    """Per-leaf (d_n10, d_n01, d_n11) for D's observed flips plus C flips at the given rates (per type; a dict per
    round count when scope = 'per_round'), plus tau3-only flips at rates t3D10, t3D01 (D unvisited rows of that type)
    and t3C00, t3C11 (C population scaled by the share of the leaf's C rows that are unvisited)."""
    out = []
    for L in leaves:
        rr = rates[L["r"]] if scope == "per_round" else rates
        R = L["rows"]
        f10, f01 = R["D10"]["flips"], R["D01"]["flips"]
        c00 = rr["C00"] * L["pop"]["C00"]
        c11 = rr["C11"] * L["pop"]["C11"]
        t10 = rr.get("t3D10", 0.0) * R["D10"]["unvisited"]
        t01 = rr.get("t3D01", 0.0) * R["D01"]["unvisited"]
        nC = R["C00"]["n"] + R["C11"]["n"]
        u00 = L["pop"]["C00"] * (R["C00"]["unvisited"] / R["C00"]["n"]) if R["C00"]["n"] else 0.0
        u11 = L["pop"]["C11"] * (R["C11"]["unvisited"] / R["C11"]["n"]) if R["C11"]["n"] else 0.0
        t00 = rr.get("t3C00", 0.0) * u00
        t11 = rr.get("t3C11", 0.0) * u11
        del nC
        # D10 flip: n10 -1, n11 +1; D01 flip: n01 -1; C00 flip: n01 +1; C11 flip: n11 -1, n10 +1
        d10 = -(f10 + t10) + (c11 + t11)
        d01 = -(f01 + t01) + (c00 + t00)
        d11 = (f10 + t10) - (c11 + t11)
        out.append((d10, d01, d11))
    return out


def main(arch, dec, k2_dir, chk_dir, out_dir):
    t0 = time.time()
    leaves, failing, problems, chk = load(arch, dec, k2_dir, chk_dir)
    agg = {t: {k: sum(L["rows"][t][k] for L in leaves) for k in ("n", "flips", "random", "random_flips", "unvisited",
                                                                 "targeted")} for t in TYPES}
    nC00, kC00 = agg["C00"]["n"], agg["C00"]["flips"]
    nC11, kC11 = agg["C11"]["n"], agg["C11"]["flips"]
    nDr = agg["D10"]["random"] + agg["D01"]["random"]
    kDr = agg["D10"]["random_flips"] + agg["D01"]["random_flips"]
    nCr = agg["C00"]["random"] + agg["C11"]["random"]
    kCr = agg["C00"]["random_flips"] + agg["C11"]["random_flips"]
    rate = {"C00": kC00 / nC00, "C11": kC11 / nC11, "t3D": kDr / nDr, "t3C": kCr / nCr}
    bnd = {"C00": (cp_lo(kC00, nC00), cp_up(kC00, nC00)), "C11": (cp_lo(kC11, nC11), cp_up(kC11, nC11)),
           "t3D": (cp_lo(kDr, nDr), cp_up(kDr, nDr)), "t3C": (cp_lo(kCr, nCr), cp_up(kCr, nCr))}
    est = {"C00": rate["C00"], "C11": rate["C11"], "t3D10": rate["t3D"], "t3D01": rate["t3D"], "t3C00": rate["t3C"],
           "t3C11": rate["t3C"]}
    rows_def = {
        "base": None,
        "estimate": est,
        "worst_lower_g": {"C00": bnd["C00"][1], "C11": bnd["C11"][0]},
        "worst_higher_g": {"C00": bnd["C00"][0], "C11": bnd["C11"][1]},
        "worst_lower_g_with_tau3_only": {"C00": bnd["C00"][1], "C11": bnd["C11"][0], "t3D10": bnd["t3D"][1],
                                         "t3C00": bnd["t3C"][1]},
        "worst_higher_g_with_tau3_only": {"C00": bnd["C00"][0], "C11": bnd["C11"][1], "t3D01": bnd["t3D"][1],
                                          "t3C11": bnd["t3C"][1]},
    }
    bonf = {"base": None, "estimate": None, "worst_lower_g": 1, "worst_higher_g": 2, "worst_lower_g_with_tau3_only": 3,
            "worst_higher_g_with_tau3_only": 4}
    res = {"label": "reservoir, revision R5 (review 1.3): independent reproduction of D2's derivation of the non-band "
                    "check's worst cases. POST HOC; read-only",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "problems": problems, "checks": chk,
           "k2_failing_in_window": sorted(f for f in failing if f.endswith(("r10", "r30", "r50"))),
           "strata": agg, "pooled_rates": rate, "cp_bounds_one_sided_95": bnd, "rows": {}, "bounded_rates_per_row": {},
           "targeted_without_t3": sum(L["targeted_without_t3"] for L in leaves)}
    zero = [(0.0, 0.0, 0.0)] * len(leaves)
    for rule in ("A", "none"):
        for name, rr in rows_def.items():
            add = zero if rr is None else changes(leaves, rr)
            res["rows"][f"{rule}|{name}"] = stats_of(leaves, add, rule)
    for name, k in bonf.items():
        res["bounded_rates_per_row"][name] = {"rates_at_a_bound": k,
                                              "bonferroni_joint_level": None if k is None else 1 - 0.05 * k}
    # every vertex of the box of allowed pooled rates
    vert = {}
    for tag, keys in (("C_only", ("C00", "C11")), ("with_tau3_only", ("C00", "C11", "t3D10", "t3D01", "t3C00", "t3C11"))):
        for rule in ("A", "none"):
            vals = []
            for corner in itertools.product((0, 1), repeat=len(keys)):
                rr = {}
                for key, c in zip(keys, corner):
                    src = key if key in bnd else ("t3D" if key.startswith("t3D") else "t3C")
                    rr[key] = bnd[src][c]
                s = stats_of(leaves, changes(leaves, rr), rule)
                vals.append((s["g5"], s["count_gap"], corner))
            vert[f"{tag}|{rule}"] = {"vertices": len(vals), "g5_min": min(v[0] for v in vals),
                                     "g5_max": max(v[0] for v in vals),
                                     "count_gap_min": min(v[1] for v in vals), "count_gap_max": max(v[1] for v in vals)}
    res["vertex_extremes"] = vert
    # per-round-count rates (C00, C11 per r) and their bounds: estimate and every vertex
    per = {}
    for r in (10, 30, 50):
        n00 = sum(L["rows"]["C00"]["n"] for L in leaves if L["r"] == r)
        k00 = sum(L["rows"]["C00"]["flips"] for L in leaves if L["r"] == r)
        n11 = sum(L["rows"]["C11"]["n"] for L in leaves if L["r"] == r)
        k11 = sum(L["rows"]["C11"]["flips"] for L in leaves if L["r"] == r)
        per[r] = {"C00": (k00, n00, cp_lo(k00, n00), cp_up(k00, n00)), "C11": (k11, n11, cp_lo(k11, n11), cp_up(k11, n11))}
    res["per_round_rates"] = {str(r): {t: dict(zip(("flips", "n", "lo", "up"), v)) for t, v in d.items()}
                              for r, d in per.items()}
    for rule in ("A", "none"):
        rr = {r: {"C00": per[r]["C00"][0] / per[r]["C00"][1], "C11": per[r]["C11"][0] / per[r]["C11"][1]}
              for r in per}
        e = stats_of(leaves, changes(leaves, rr, "per_round"), rule)
        vals = []
        for corner in itertools.product((0, 1), repeat=6):
            rr = {}
            for i, r in enumerate((10, 30, 50)):
                rr[r] = {"C00": per[r]["C00"][2 + corner[2 * i]], "C11": per[r]["C11"][2 + corner[2 * i + 1]]}
            s = stats_of(leaves, changes(leaves, rr, "per_round"), rule)
            vals.append((s["g5"], s["count_gap"]))
        res[f"per_round|{rule}"] = {"estimate": e, "g5_min": min(v[0] for v in vals), "g5_max": max(v[0] for v in vals),
                                    "count_gap_min": min(v[1] for v in vals), "count_gap_max": max(v[1] for v in vals)}
    # alternative combinations of the C strata for the estimate
    alt = {}
    for rule in ("A", "none"):
        o = {}
        # (i) pooled per type (the estimate row); (ii) per type per round count; (iii) per leaf and type (the leaf's
        # own sample rate); (iv) C pooled over both types, flips scaled by the C population and split by type share
        o["pooled_per_type"] = res["rows"][f"{rule}|estimate"]["g5"]
        rr = {r: {"C00": per[r]["C00"][0] / per[r]["C00"][1], "C11": per[r]["C11"][0] / per[r]["C11"][1]} for r in per}
        o["per_type_per_round"] = stats_of(leaves, changes(leaves, rr, "per_round"), rule)["g5"]
        add = []
        for L in leaves:
            R = L["rows"]
            l00 = R["C00"]["flips"] / R["C00"]["n"] if R["C00"]["n"] else 0.0
            l11 = R["C11"]["flips"] / R["C11"]["n"] if R["C11"]["n"] else 0.0
            add.append(changes([L], {"C00": l00, "C11": l11})[0])
        o["per_leaf_per_type"] = stats_of(leaves, add, rule)["g5"]
        add = []
        for L in leaves:
            R = L["rows"]
            nC = R["C00"]["n"] + R["C11"]["n"]
            w = (L["pop"]["C00"] + L["pop"]["C11"]) / nC
            f10, f01 = R["D10"]["flips"], R["D01"]["flips"]
            c00, c11 = w * R["C00"]["flips"], w * R["C11"]["flips"]
            add.append((-f10 + c11, -f01 + c00, f10 - c11))
        o["per_leaf_expansion_weight"] = stats_of(leaves, add, rule)["g5"]
        o["d_only"] = stats_of(leaves, changes(leaves, {"C00": 0.0, "C11": 0.0}), rule)["g5"]
        o["range"] = [min(v for k, v in o.items() if k != "range"), max(v for k, v in o.items() if k != "range")]
        alt[rule] = o
    res["estimate_by_combination"] = alt
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r5_worstcase.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    for k, v in res["rows"].items():
        print(f"{k:38s} g5 {100 * v['g5']:+.2f}%  count gap {100 * v['count_gap']:+.2f}% ({v['count_gap_errors']:+.0f})",
              flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
