"""R1, POST HOC additions, not declared: the declared worst-case range in full, at a joint 95% level, and a phase 2 plan.

reservoir, 2026-10-13. Declared (notes/reservoir/willow_revision_declarations.md, R1, "Reported"): the worst-case
range of g5(MPE|ML) and Lambda_MPE/Lambda_ML "with each leaf's MPE error count moved by up to u x shots in either
direction". r12_analysis.r1_certificate, as declared and run, evaluates two points of that box only: every leaf moved
by ceil(u_r n_leaf) in the same direction (mpe_better, mpe_worse). g5 rests on each series' slope over r, so moves in
opposite directions at different r change it more, and those two points do not bound the declared range.
This script computes, for {10, 13, 30} and {10, 30}, rule A and no exclusion:
 1. the declared box in full. g5 = mean_s d_s / mean_s eps_ML_s (d5_check.g_stat) and eps_ML does not move, so the
    extremes of g5 are the sums of each series' extremes of d_s. Each series' d_s is evaluated at every corner of its
    box (each leaf's count at kA - or + ceil(u_r n), n11 capped at the new count, as in r1_certificate), and a grid of
    9 levels per leaf checks that no interior point lies outside the corners' extremes;
 2. the same with each round count's bound at 1 - 0.05/k (Bonferroni: the k bounds of a window then hold together at
    95% or more), where the declared bounds hold together at 1 - 0.05 k only (85% on {10, 13, 30}, 90% on {10, 30});
 3. a planning projection for declared phase 2 (138 more shots per r = 30 leaf, so n_30 = 2,208), assuming that it
    finds no miss. Not a result.
Checks: the declared certificate is rerun (r12_analysis.py imported unchanged) and must equal the stored
r1_certificate.json in OUT_DIR, and this script's same-direction corners must reproduce its mpe_better and mpe_worse
g5 exactly. Reads only; writes OUT_DIR/r1_posthoc.json.
Usage: python3 -I -B r1_posthoc.py ARCHIVE DECODE K2_DIR STAGE_DIR R1_RECORDS OUT_DIR
       (the write goes through qrc.outpaths: QRC_OVERWRITE=1 for the stored location)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r12_analysis as R12  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402  (on sys.path through r12_analysis's imports)

N30_PHASE2 = 2208  # r = 30 shots with stage 2, phase 1 and declared phase 2 (8 leaves x 276)
GRID = 9           # interior check: levels per leaf between -ceil(u n) and +ceil(u n)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def cp_upper_at(alpha):
    """r12_analysis.cp_upper with the one-sided level 1 - alpha in place of 0.95; same edge cases."""
    def f(k, n):
        return float(stats.beta.ppf(1 - alpha, k + 1, n - k)) if n and k < n else (1.0 if n else None)
    return f


def series_data(leaves, excl, rounds):
    """Per d = 5 series (patch, basis): [(r, kA, kM, n11, n)], built exactly as r1_certificate's worst case builds it."""
    out = []
    for p in R12.P5:
        for b in R12.BASES:
            L = []
            for r in rounds:
                rec = leaves[(5, p, b, r)]
                if R12.rel_of(5, p, b, r) in excl or "tesseract" not in rec["preds"]:
                    continue
                m = R12.mask_of(rec, ["tesseract"])
                aw = (rec["preds"]["tesseract"] != rec["obs"]) & m
                mw = (rec["ml"] != rec["obs"]) & m
                L.append((r, int(aw.sum()), int(mw.sum()), int(np.sum(aw & mw)), int(m.sum())))
            if len(L) >= 2:
                out.append((f"{p}/{b}", L))
    return out


def moved(L, u_r, steps):
    """paired() on the series with each leaf's MPE count moved by step * ceil(u_r n); step in [-1, 1]."""
    rs, kA2, kM, n11, n = [], [], [], [], []
    for (r, kA, km, x11, nn), s in zip(L, steps):
        k2 = max(kA + int(round(s * math.ceil(u_r[r] * nn))), 0)
        rs.append(r)
        kA2.append(k2)
        kM.append(km)
        n11.append(min(x11, k2))
        n.append(nn)
    return R12.paired(rs, kA2, kM, n11, n)


def box(series, u_r, g3):
    lo, hi, same_lo, same_hi, nom, per, interior_ok, worst_excess = [], [], [], [], [], [], True, 0.0
    for name, L in series:
        k = len(L)
        cs = sorted(((moved(L, u_r, sg), sg) for sg in itertools.product((-1, 1), repeat=k)), key=lambda t: t[0]["d"])
        lo.append(cs[0][0])
        hi.append(cs[-1][0])
        same_lo.append(moved(L, u_r, (-1,) * k))
        same_hi.append(moved(L, u_r, (1,) * k))
        nom.append(moved(L, u_r, (0,) * k))
        dmin, dmax = cs[0][0]["d"], cs[-1][0]["d"]
        for st in itertools.product(np.linspace(-1, 1, GRID), repeat=k):
            dd = moved(L, u_r, st)["d"]
            ex = max(dmin - dd, dd - dmax)
            if ex > 1e-12:
                interior_ok = False
            worst_excess = max(worst_excess, ex)
        per.append({"series": name, "rounds": [x[0] for x in L],
                    "moves": {f"r{x[0]}": math.ceil(u_r[x[0]] * x[4]) for x in L},
                    "d_min_corner": [int(v) for v in cs[0][1]], "d_max_corner": [int(v) for v in cs[-1][1]],
                    "d_nominal": nom[-1]["d"], "d_min": dmin, "d_max": dmax})
    G = {lab: R12.gsum(v) for lab, v in (("nominal", nom), ("box_min", lo), ("box_max", hi),
                                          ("same_direction_down", same_lo), ("same_direction_up", same_hi))}
    o = {"u_per_r": u_r, "g3_exact": g3["g"], "series_d5": len(series),
         "g5": {lab: G[lab]["g"] for lab in G},
         "lambda_mpe_over_ml": {lab: (1 + g3["g"]) / (1 + G[lab]["g"]) for lab in G},
         "box_growth_kept": bool(G["box_min"]["g"] > g3["g"]), "box_g5_positive": bool(G["box_min"]["g"] > 0),
         "same_direction_growth_kept": bool(min(G["same_direction_down"]["g"], G["same_direction_up"]["g"]) > g3["g"]),
         "interior_check": {"grid_levels_per_leaf": GRID, "corners_hold_extremes": interior_ok,
                            "largest_excess": worst_excess},
         "series": per}
    return o


def main(arch, dec, k2_dir, stage_dir, r1_records, out_dir):
    t0 = time.time()
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = R12.k2_verdicts(k2_dir)
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    leaves, _ = R12.load_leaves(arch, dec, stage_dir, None, chk)
    print(f"leaves read ({time.time() - t0:.0f} s)", flush=True)
    declared = R12.r1_certificate(leaves, failing, stage_dir, r1_records)
    stored_p = os.path.join(out_dir, "r1_certificate.json")
    stored = json.load(open(stored_p)) if os.path.isfile(stored_p) else None
    ser = (lambda x: json.dumps(x, sort_keys=True))  # the stored file went through JSON; NaN-safe comparison
    equal = None if stored is None else all(ser(json.loads(ser(declared[k]))) == ser(stored[k])
                                            for k in ("summary", "leverage", "worst_case"))
    U = declared["summary"]["union"]
    kn = {r: (U[f"r{r}"]["misses"], U[f"r{r}"]["proven"]) for r in (10, 13, 30)}
    if any(cp_upper_at(0.05)(*kn[r]) != R12.cp_upper(*kn[r]) for r in kn):
        raise SystemExit("cp_upper_at(0.05) differs from r12_analysis.cp_upper")
    out = {"label": "reservoir, revision R1, POST HOC additions, not declared: the declared worst-case box in full "
                    "(every corner), the same at a joint 95% level (Bonferroni), and a phase 2 projection (planning "
                    "only: assumes phase 2 finds no miss)",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "r12_analysis_md5": md5(R12.__file__), "r1_records": r1_records, "r1_records_md5": md5(r1_records),
           "stored_certificate": stored_p if stored is not None else None,
           "stored_certificate_md5": md5(stored_p) if stored is not None else None,
           "declared_rerun_equals_stored": equal, "checks": chk, "k2_still_fail": sorted(failing),
           "union_misses_proven": {f"r{r}": list(v) for r, v in kn.items()}, "results": {}}
    same_ok = True
    for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
        for w in ("10_13_30", "10_30"):
            rounds = R12.WINDOWS[w]
            k = len(rounds)
            f3 = R12.series_fits(leaves, "mpe_exact", "ml", ["mpe_exact"], 3, rounds, set())
            g3 = R12.gsum(list(f3.values()))
            series = series_data(leaves, excl, rounds)
            key = f"{w}|{kname}"
            res = {"round_counts": k, "joint_level_of_each95": 1 - 0.05 * k, "each_level_joint95": 1 - 0.05 / k}
            for lab, alpha, proj in (("each95", 0.05, False), ("joint95", 0.05 / k, False),
                                     ("phase2_projection_each95", 0.05, True),
                                     ("phase2_projection_joint95", 0.05 / k, True)):
                cp = cp_upper_at(alpha)
                u_r = {r: (cp(0, N30_PHASE2) if (proj and r == 30) else cp(*kn[r])) for r in rounds}
                res[lab] = box(series, u_r, g3)
            dw = declared["worst_case"][key]["union"]
            sd = res["each95"]["g5"]
            res["same_direction_reproduces_declared"] = bool(sd["same_direction_down"] == dw["mpe_better"]["g5"] and
                                                             sd["same_direction_up"] == dw["mpe_worse"]["g5"] and
                                                             sd["nominal"] == dw["nominal"]["g5"])
            same_ok &= res["same_direction_reproduces_declared"]
            out["results"][key] = res
    out["same_direction_reproduces_declared_all"] = bool(same_ok)
    out["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r1_posthoc.json")), "w") as f:
        json.dump(out, f, indent=1)
    print(f"declared rerun equals stored: {equal}; same-direction corners reproduce declared: {same_ok}", flush=True)
    for key, res in out["results"].items():
        for lab in ("each95", "joint95", "phase2_projection_each95", "phase2_projection_joint95"):
            v = res[lab]
            g = v["g5"]
            u = ", ".join(f"r{r} {100 * x:.3f}%" for r, x in v["u_per_r"].items())
            print(f"{key:22s} {lab:26s} u {u} | g3 {100 * v['g3_exact']:.2f}% g5 {100 * g['nominal']:.2f}% "
                  f"same-dir [{100 * g['same_direction_down']:.2f}, {100 * g['same_direction_up']:.2f}] "
                  f"box [{100 * g['box_min']:.2f}, {100 * g['box_max']:.2f}] Lambda box "
                  f"[{v['lambda_mpe_over_ml']['box_max']:.4f}, {v['lambda_mpe_over_ml']['box_min']:.4f}] "
                  f"growth kept {v['box_growth_kept']} g5>0 {v['box_g5_positive']} "
                  f"interior ok {v['interior_check']['corners_hold_extremes']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
