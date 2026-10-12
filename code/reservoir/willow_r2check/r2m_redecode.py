"""Re-decode random held-out shots of the W5 leaves under T-hat_mech and D-hat_mech with the exact decoder written here,
and compare with decoder's stored predictions and llr (mech_data.npz). Also the held-out ln P(s) under each fit against
RL's on the same shots (mech_data.npz stores no ln P(s)), for T-hat_mech's in-sample against held-out gain. Part of
the mech check of main's brief of 2026-10-10 ("the same checks as for F1"); written before the real mech fits.

reservoir, 2026-10-10. Model: r2m_common (the family rebuilt from PR2 section 8.2; each fit's parameter names must
equal it, else the series is skipped and listed). Decoder: r2_common.layered_ml (exact, layer by layer).
Shots: N_PER_LEAF per leaf, drawn without replacement from the leaf's held-out shots with numpy's default_rng(SEED),
leaves in the order of the series requested and W5. Control: the first CONTROL sampled shots of each series' r = 10
leaf (the cheapest; a decode's cost is mostly per call, not per shot) are also decoded with every shift at 0 and
compared with round 1's ML(RL) llr and ln P(s) (decode/<leaf>/rl/ml_exact.npz); this checks the mech DEM path (logit,
sigmoid, clip) at delta = 0.
Usage: nice -n 10 python3 -I r2m_redecode.py ARCHIVE DECODE FIT MECH OUT_DIR [--n-per-leaf 28] [--seed 20261010]
       [--control 4] [--series p/B,...]  -> OUT_DIR/r2m_redecode.json (through qrc.outpaths)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import BASES, PATCHES, SHOTS, W5, detection_events, layered_ml, rel_of  # noqa: E402
from r2m_common import mech_family, mech_faults, series_models  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402


def compare(llr, logp, ref_llr, ref_pred, ref_logp=None):
    ref = np.asarray(ref_llr, np.float64)
    fin = np.isfinite(ref)
    dl = np.abs(llr[fin] - ref[fin])
    o = {"shots": int(llr.size), "stored_nonfinite": int((~fin).sum()),
         "max_abs_dllr": float(dl.max()) if dl.size else None,
         "max_rel_dllr": float(np.max(dl / np.maximum(1.0, np.abs(ref[fin])))) if dl.size else None,
         "pred_equal": int(np.sum((llr > 0).astype(np.int8) == np.asarray(ref_pred).astype(np.int8))),
         "min_abs_llr": float(np.min(np.abs(llr)))}
    if ref_logp is not None:
        o["max_abs_dlogp"] = float(np.max(np.abs(logp - np.asarray(ref_logp, np.float64))))
    return o


def main(a):
    t0 = time.time()
    want = ([tuple(s.split("/")) for s in a.series.split(",")] if a.series
            else [(p, b) for p in PATCHES for b in BASES])
    rng = np.random.default_rng(a.seed)
    res = {"label": "reservoir, mech check: exact re-decode of random held-out shots under T-hat_mech and D-hat_mech, "
                    "2026-10-10", "n_per_leaf": a.n_per_leaf, "seed": a.seed, "control_shots_per_series_r10": a.control,
           "skipped": {}, "leaves": {}}
    tot = {X: {"shots": 0, "pred_equal": 0, "max_abs_dllr": 0.0, "max_rel_dllr": 0.0, "stored_nonfinite": 0,
               "min_abs_llr": math.inf} for X in ("Tm", "Dm")}
    tot["control_RL"] = {"shots": 0, "pred_equal": 0, "max_abs_dllr": 0.0, "max_rel_dllr": 0.0, "max_abs_dlogp": 0.0,
                         "stored_nonfinite": 0, "min_abs_llr": math.inf}
    dlp = {X: [] for X in ("Tm", "Dm")}  # per leaf: ln P(s) under X minus RL's, per sampled shot
    for p, b in want:
        key = f"d3_at_{p}/{b}"
        sd = os.path.join(a.mech, f"d3_at_{p}", b)
        paths = {X: os.path.join(sd, fn) for X, fn in (("Tm", "mech_fit_T.json"), ("Dm", "mech_fit_D.json"))}
        if not all(os.path.exists(v) for v in paths.values()):
            res["skipped"][key] = "fits missing"
            continue
        fits = {X: json.load(open(v)) for X, v in paths.items()}
        models = series_models(a.archive, a.fit, p, b)
        fam = mech_family(models)
        if any(fits[X]["names"] != fam["names"] for X in fits):
            res["skipped"][key] = "parameter names differ from the family rebuilt here"
            continue
        delta = {X: np.array(fits[X]["delta_hat"], float) for X in fits}
        for r, M, ix in zip(W5, models, fam["idx"]):
            rel = rel_of(p, b, r)
            mp = os.path.join(a.mech, rel, "mech_data.npz")
            if not os.path.exists(mp):
                res["skipped"][rel] = "mech_data.npz missing"
                continue
            md = np.load(mp, allow_pickle=False)
            ml = np.load(os.path.join(a.decode, rel, "rl", "ml_exact.npz"), allow_pickle=False)
            ti = md["test_idx"].astype(np.int64)
            pos = np.sort(rng.choice(ti.size, size=min(a.n_per_leaf, ti.size), replace=False))
            shots = ti[pos]
            syn = detection_events(a.archive, rel, M["nd"], shots)
            out = {"shots": shots.tolist()}
            lp_rl = ml["logp"][shots].astype(np.float64)
            for X in ("Tm", "Dm"):
                t1 = time.time()
                faults, _ = mech_faults(M, ix, delta[X])
                llr, logp = layered_ml(faults, M["t_of"], r, syn)
                o = compare(llr, logp, md[f"{X}_llr"][pos], md[f"{X}_pred"][pos])
                o["mean_lnPs_minus_RL"] = float(np.mean(logp - lp_rl))
                o["seconds"] = time.time() - t1
                out[X] = o
                dlp[X].append(logp - lp_rl)
                T = tot[X]
                T["shots"] += o["shots"]; T["pred_equal"] += o["pred_equal"]
                T["stored_nonfinite"] += o["stored_nonfinite"]
                T["min_abs_llr"] = min(T["min_abs_llr"], o["min_abs_llr"])
                for k in ("max_abs_dllr", "max_rel_dllr"):
                    if o[k] is not None:
                        T[k] = max(T[k], o[k])
            if a.control > 0 and r == W5[0]:
                c = slice(0, min(a.control, shots.size))
                faults0, _ = mech_faults(M, ix, np.zeros(len(fam["names"])))
                llr0, logp0 = layered_ml(faults0, M["t_of"], r, syn[c])
                o = compare(llr0, logp0, ml["llr"][shots[c]], ml["pred"].reshape(SHOTS, -1)[shots[c], 0],
                            ml["logp"][shots[c]])
                out["control_RL"] = o
                T = tot["control_RL"]
                T["shots"] += o["shots"]; T["pred_equal"] += o["pred_equal"]
                T["stored_nonfinite"] += o["stored_nonfinite"]
                T["min_abs_llr"] = min(T["min_abs_llr"], o["min_abs_llr"])
                for k in ("max_abs_dllr", "max_rel_dllr", "max_abs_dlogp"):
                    if o.get(k) is not None:
                        T[k] = max(T[k], o[k])
            res["leaves"][rel] = out
            print(f"{rel}: Tm pred {out['Tm']['pred_equal']}/{out['Tm']['shots']} max|dllr| {out['Tm']['max_abs_dllr']}; "
                  f"Dm pred {out['Dm']['pred_equal']}/{out['Dm']['shots']} max|dllr| {out['Dm']['max_abs_dllr']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    for T in tot.values():  # nothing decoded: no minimum (keeps the output strict JSON)
        if not math.isfinite(T["min_abs_llr"]):
            T["min_abs_llr"] = None
    res["total"] = tot
    lp = {}
    for X in ("Tm", "Dm"):
        if dlp[X]:
            x = np.concatenate(dlp[X])
            v = sum(len(y) * np.var(y, ddof=1) for y in dlp[X] if len(y) > 1)
            lp[X] = {"shots": int(x.size), "mean_lnPs_minus_RL_per_shot": float(x.mean()),
                     "se": float(math.sqrt(v) / x.size) if v > 0 else None,
                     "note": "held out, on the sampled shots only (SE stratified by leaf); compare with the fit's "
                             "in-sample ln P(s) gain (r2m_check: Tm in_sample_gain_lnPs_per_shot)"}
    res["heldout_lnPs_sample"] = lp
    res["note"] = ("stored llr are float32, so |dllr| up to about 1e-7 x |llr| is rounding; round 1's ML(RL) ln P(s) is "
                   "float64; a stored non-finite llr is not compared (counted)")
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(a.out_dir, "r2m_redecode.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("total", "heldout_lnPs_sample", "skipped")}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("archive", "decode", "fit", "mech", "out_dir"):
        ap.add_argument(k)
    ap.add_argument("--n-per-leaf", type=int, default=28)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--control", type=int, default=4)
    ap.add_argument("--series", default=None)
    main(ap.parse_args())
