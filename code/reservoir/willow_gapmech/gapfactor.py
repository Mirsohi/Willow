"""EXPLORATORY, post hoc: each decoder's Lambda deficit to ML(RL), split through Tesseract (a most-probable-error search)
on the same shots. Part of item 2 of main's brief (10 Oct 2026); from existing per-shot outputs only, nothing decoded.

reservoir, 2026-10-10. Not pre-registered. With eps_X the paired per-round error rate of decoder X (prereg section 4,
paired() and g_stat() imported from willow_d5check/d5_check.py with section 9 entry 7's SE), the gap of X to a
reference Y is g(X|Y) = mean_s(eps_X - eps_Y) / mean_s(eps_Y), and the paired Lambda ratio Lambda_X / Lambda_Y =
(1 + g_3(X|Y)) / (1 + g_5(X|Y)). Since eps_A / eps_ML = (eps_A / eps_T)(eps_T / eps_ML) per series, the deficit
factorises (exactly per series, approximately after pooling):
    Lambda_A / Lambda_ML  ~  (Lambda_A / Lambda_T) x (Lambda_T / Lambda_ML).
If Tesseract's decisions are the exact most-probable-error (MPE) class (checked at d = 3 on a timing sample, mpe.py),
Lambda_T / Lambda_ML is the degeneracy factor (what deciding by the single most probable error costs against summing over
errors) and Lambda_A / Lambda_T the rest (A's approximation, relative to MPE). That reading is inferred, not measured.

Shots: those with A's, Tesseract's and ML's predictions (d = 3: Tesseract's every 5th; d = 5: ML's every 5th). Rounds:
the rounds where A and Tesseract both have predictions at both distances (Libra 10, 30; the others 10, 13, 30; Tesseract
has none at d = 5, r = 50). K2 at d = 5: rule A (main 19:58Z) and no exclusion, verdicts as found at run time.
Inputs as for gapmech.py. Usage: nice -n 10 python3 -I gapfactor.py ARCHIVE DECODE K2_DIR OUT_DIR
  -> OUT_DIR/gap_factor.json (through qrc.outpaths)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gapmech import MISSING, growth, gsum, read_leaf  # noqa: E402
from d5_check import BASES, P3, P5, k2_verdicts, paired, rel_of  # noqa: E402  (path set by gapmech)
from qrc.outpaths import out_path  # noqa: E402

ROUNDS = [10, 13, 30]
OTHERS = ["libra", "harmony", "corrmatch", "corr_mwpm", "mwpm"]


def wrong(rec, x):
    return rec["ml"] != rec["obs"] if x == "ml" else rec["preds"][x] != rec["obs"]


def series_pair(leaves, x, y, a, d, rounds, excluded):
    """paired() per series for X against reference Y, on the shots where A, Tesseract and ML all have predictions."""
    prs, lost = [], []
    for p in (P3 if d == 3 else P5):
        for b in BASES:
            L = []
            for r in rounds:
                rec = leaves.get((d, p, b, r))
                if rec is None or rel_of(d, p, b, r) in excluded:
                    continue
                m = (rec["preds"][a] != MISSING) & (rec["preds"]["tesseract"] != MISSING)
                xw, yw = wrong(rec, x) & m, wrong(rec, y) & m
                if m.sum():
                    L.append((r, int(m.sum()), int(xw.sum()), int(yw.sum()), int(np.sum(xw & yw))))
            if len(L) < 2:
                lost.append(f"{p}/{b}")
                continue
            pr = paired([q[0] for q in L], [q[2] for q in L], [q[3] for q in L], [q[4] for q in L],
                        [q[1] for q in L])
            pr["basis"] = b
            prs.append(pr)
    return prs, lost


def main(arch, dec, k2_dir, out_dir):
    t0 = time.time()
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = k2_verdicts(k2_dir)
    k2_read = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    leaves = {}
    for d, pats in ((3, P3), (5, P5)):
        for p in pats:
            for b in BASES:
                for r in ROUNDS:
                    leaves[(d, p, b, r)] = read_leaf(arch, dec, d, p, b, r, chk)
        print(f"d = {d}: leaves read ({time.time() - t0:.0f} s)", flush=True)
    res = {"label": "EXPLORATORY, post hoc (reservoir, 2026-10-10): each decoder's paired Lambda deficit to ML(RL) "
                    "split through Tesseract on the same shots (item 2 of main's brief). Gaps and ratios measured; "
                    "reading Tesseract as the exact MPE decoder is an inference. Interim (K2 as found at run time).",
           "k2": {"read_at": k2_read, "verdicts_found": len(k2), "still_fail": sorted(failing)}, "checks": chk,
           "decoders": {}}
    for a in OTHERS:
        rounds = [r for r in ROUNDS if all(a in leaves[(d, p, b, r)]["preds"] and
                                           "tesseract" in leaves[(d, p, b, r)]["preds"]
                                           for d, pats in ((3, P3), (5, P5)) for p in pats for b in BASES)]
        res["decoders"][a] = {"rounds": rounds}
        for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
            o = {}
            for tag, x, y in (("A|ML", a, "ml"), ("T|ML", "tesseract", "ml"), ("A|T", a, "tesseract")):
                prs3, lost3 = series_pair(leaves, x, y, a, 3, rounds, set())
                prs5, lost5 = series_pair(leaves, x, y, a, 5, rounds, excl)
                g3, g5 = gsum(prs3), gsum(prs5)
                q = {"g3": g3, "g5": g5, "lost_series_d3": lost3, "lost_series_d5": lost5}
                if g3 and g5:
                    q.update(growth(g3, g5))
                o[tag] = q
            if all("lambda_ratio" in o[t] for t in o):
                la, lt, lat = (math.log(o[t]["lambda_ratio"]) for t in ("A|ML", "T|ML", "A|T"))
                o["share_of_ln_deficit_in_T|ML"] = lt / la if la else None
                o["factorisation_check_ln"] = {"A|ML": la, "T|ML + A|T": lt + lat}
            res["decoders"][a][kname] = o
        print(f"{a} done ({time.time() - t0:.0f} s)", flush=True)
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "gap_factor.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(f"done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 4:
        raise SystemExit(__doc__)
    main(*a)
