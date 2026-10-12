"""EXPLORATORY, post hoc: how many d = 5 ML(RL) decisions outside K2's band could truncation have flipped?

reservoir, 2026-10-10. Part of the d = 5 / Lambda check for main (flag 4: "anything that could make d = 5 look better
for ML than it is, e.g. truncation"). Not pre-registered.

The tau ladder (prereg section 6) re-decodes at tau2 only the band shots, |P_tau1 - 1/2| < 0.1; every other shot keeps
its tau1 decision, which K2 never checks. On the band shots the tau1 -> tau2 change of the log-likelihood ratio,
dL = llr - llr_t1, is known. Assumption (untested, and probably pessimistic, because band syndromes are the ambiguous,
hard ones): an out-of-band shot's tau1 LLR is off by a dL drawn from its own leaf's band shots. Then the expected number
of out-of-band decision flips is sum_i P(sign(L1_i + dL) != sign(L1_i)). A flip changes the leaf's ML error count by
+1 or -1; the worst case for the comparison with Libra is that every flip would add an ML error under exact decoding.

Input (read only): DECODE/d5_at_<patch>/<basis>/r<NN>/rl_s5/ml_tau2_1e-6-1e-8_0.1.npz (llr, llr_t1, redecoded, pred),
DECODE/.../shots.npz (obs). DECODE is results/decoder/willow/decode/google_105Q_surface_code_d3_d5_d7.
Usage: nice -n 10 python3 -I band_extrap.py DECODE OUT_DIR -> OUT_DIR/band_extrap.json (through qrc.outpaths)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qrc.outpaths import out_path  # noqa: E402

P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
W5 = [10, 13, 30, 50]
WL = [10, 30, 50]
K_LIBRA, K_ML = 21853, 20836  # pooled counts over the 24 Libra leaves (d5_check.json pooled_d5; decoder's equal)


def main():
    t0 = time.time()
    dec, out_dir = sys.argv[1:3]
    rows = []
    for p in P5:
        for b in ("X", "Z"):
            for r in W5:
                leaf = os.path.join(dec, f"d5_at_{p}", b, f"r{r:02d}", "rl_s5")
                z = np.load(os.path.join(leaf, "ml_tau2_1e-6-1e-8_0.1.npz"), allow_pickle=False)
                L1 = z["llr_t1"].astype(np.float64).ravel()
                L = z["llr"].astype(np.float64).ravel()
                rd = z["redecoded"].astype(bool).ravel()
                pred = z["pred"].astype(np.int64).ravel()
                with np.errstate(invalid="ignore"):
                    dL = L[rd] - L1[rd]
                if not np.all(np.isfinite(dL)):
                    raise SystemExit(f"non-finite dL on band shots in {leaf}")
                nb = ~rd
                a = L1[nb]
                s = np.sort(dL)
                n = len(s)
                q = np.where(a > 0, np.searchsorted(s, -a, side="left") / n,
                             np.where(a < 0, (n - np.searchsorted(s, -a, side="right")) / n, 0.5))
                # band decisions changed tau1 -> final, recomputed from the LLRs (decisions: llr > 0)
                ch_band = int(np.sum((L[rd] > 0) != (L1[rd] > 0)))
                p_eq = bool(np.array_equal(pred, (L > 0).astype(np.int64)))
                rows.append({
                    "leaf": f"d5_at_{p}/{b}/r{r:02d}", "patch": p, "basis": b, "r": r, "libra": r in WL,
                    "band": int(rd.sum()), "out_of_band": int(nb.sum()),
                    "band_decisions_changed": ch_band, "pred_equals_llr_gt_0": p_eq,
                    "band_abs_dL_median": float(np.median(np.abs(dL))),
                    "band_abs_dL_q90": float(np.quantile(np.abs(dL), 0.9)),
                    "band_abs_dL_max": float(np.abs(dL).max()),
                    "band_share_abs_dL_gt_0.405": float(np.mean(np.abs(dL) > 0.405)),
                    "out_of_band_inf_llr_t1": int(np.sum(~np.isfinite(a))),
                    "min_abs_llr_t1_out_of_band": float(np.min(np.abs(a))),
                    "expected_flips_out_of_band": float(q.sum()),
                })
    tot = {}
    for tag, sel in (("all_32_leaves", lambda x: True), ("libra_24_leaves", lambda x: x["libra"])):
        rr = [x for x in rows if sel(x)]
        F = sum(x["expected_flips_out_of_band"] for x in rr)
        tot[tag] = {"leaves": len(rr), "band": sum(x["band"] for x in rr),
                    "out_of_band": sum(x["out_of_band"] for x in rr),
                    "band_decisions_changed": sum(x["band_decisions_changed"] for x in rr),
                    "expected_flips_out_of_band": F,
                    "max_leaf_expected_flips": max(x["expected_flips_out_of_band"] for x in rr)}
    F = tot["libra_24_leaves"]["expected_flips_out_of_band"]
    res = {
        "label": "EXPLORATORY, post hoc (reservoir, 2026-10-10): extrapolation of the band shots' tau1 -> tau2 LLR "
                 "changes to the out-of-band shots that K2 never re-decodes. Measured inputs; the extrapolation is an "
                 "assumption.",
        "totals": tot,
        "pooled_gap_d5": {
            "as_measured": K_LIBRA / K_ML - 1,
            "if_every_expected_flip_added_an_ML_error": (K_LIBRA - (K_ML + F)) / (K_ML + F),
            "errors_needed_to_erase_the_gap": K_LIBRA - K_ML},
        "per_leaf": rows,
        "run_seconds": time.time() - t0,
    }
    p = out_path(os.path.join(out_dir, "band_extrap.json"))
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("totals", "pooled_gap_d5", "run_seconds")}, indent=1))
    print("pred == (llr > 0) in every leaf:", all(x["pred_equals_llr_gt_0"] for x in rows))
    print("wrote", p)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main()
