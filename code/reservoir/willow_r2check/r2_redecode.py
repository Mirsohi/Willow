"""Re-decode random held-out shots of all 72 W5 leaves under the fitted D-hat with the exact decoder written here, and
compare with decoder's stored D_pred, D_llr and D_logp (check 2 of main's brief, 2026-10-10). The same shots are also
decoded under the RL DEM and compared with round 1's ML(RL) decodes, as a control of the decoder.

reservoir, 2026-10-10. D-hat for a leaf: the shipped RL DEM with each fault's probability times exp(phi_hat) of its class
name (fit_series.json of the leaf's series), clipped as F1's model; classes rebuilt by p4_check.classes (r2_common).
Shots: N_PER_LEAF per leaf, drawn without replacement from the leaf's held-out shots with numpy's default_rng(SEED).
Usage: nice -n 10 python3 -I r2_redecode.py ARCHIVE DECODE FIT F1 OUT_DIR [N_PER_LEAF SEED]   (defaults 28, 20261010)
  -> OUT_DIR/r2_redecode.json (through qrc.outpaths)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (BASES, PATCHES, SHOTS, W5, detection_events, layered_ml, leaf_model, rel_of,  # noqa: E402
                       scaled, series_fit, theta_by_name)
from qrc.outpaths import out_path  # noqa: E402


def main(arch, dec, fit, f1, out_dir, n_per_leaf=28, seed=20261010):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    res = {"label": "reservoir, round-2 F1 check 2: exact re-decode of random held-out shots under D-hat, 2026-10-10",
           "shots_per_leaf": n_per_leaf, "seed": seed, "leaves": {}}
    tot = {"D": {"max_abs_dllr": 0.0, "max_rel_dllr": 0.0, "max_abs_dlogp": 0.0, "pred_equal": 0, "min_abs_llr": np.inf},
           "RL": {"max_abs_dllr": 0.0, "max_rel_dllr": 0.0, "max_abs_dlogp": 0.0, "pred_equal": 0, "min_abs_llr": np.inf},
           "shots": 0, "clipped_faults": 0, "duplicates": 0}
    for p in PATCHES:
        for b in BASES:
            th = theta_by_name(series_fit(f1, p, b))
            for r in W5:
                rel = rel_of(p, b, r)
                fj = json.load(open(os.path.join(fit, rel, "rl_tsplit", "fit.json")))
                dd = np.load(os.path.join(f1, rel, "d_data.npz"), allow_pickle=False)
                ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
                M = leaf_model(arch, rel, r, fj["n_fit"], float(fj["fit_attempts"][-1]["min_class_events"]))
                mD, nclip = scaled(M["mech"], M["cls"], th)
                pos = np.sort(rng.choice(dd["test_idx"].size, size=n_per_leaf, replace=False))
                shots = dd["test_idx"][pos].astype(np.int64)
                syn = detection_events(arch, rel, M["nd"], shots)
                out = {"shots": shots.tolist(), "clipped_faults": nclip, "duplicates": M["n_dup"]}
                for tag, mech, ref_llr, ref_logp, ref_pred in (
                        ("D", mD, dd["D_llr"][pos], dd["D_logp"][pos], dd["D_pred"][pos]),
                        ("RL", M["mech"], ml["llr"][shots], ml["logp"][shots], ml["pred"].reshape(SHOTS, -1)[shots, 0])):
                    t1 = time.time()
                    llr, logp = layered_ml(mech, M["t_of"], r, syn)
                    ref = ref_llr.astype(np.float64)
                    dl = np.abs(llr - ref)
                    o = {"max_abs_dllr": float(dl.max()), "max_rel_dllr": float(np.max(dl / np.maximum(1.0, np.abs(ref)))),
                         "max_abs_dlogp": float(np.max(np.abs(logp - ref_logp))),
                         "pred_equal": int(np.sum((llr > 0).astype(np.uint8) == ref_pred)),
                         "min_abs_llr": float(np.min(np.abs(llr))), "seconds": time.time() - t1}
                    out[tag] = o
                    for k in ("max_abs_dllr", "max_rel_dllr", "max_abs_dlogp"):
                        tot[tag][k] = max(tot[tag][k], o[k])
                    tot[tag]["pred_equal"] += o["pred_equal"]
                    tot[tag]["min_abs_llr"] = min(tot[tag]["min_abs_llr"], o["min_abs_llr"])
                tot["shots"] += n_per_leaf
                tot["clipped_faults"] += nclip
                tot["duplicates"] += M["n_dup"]
                res["leaves"][rel] = out
                print(f"{rel}: D max|dllr| {out['D']['max_abs_dllr']:.2e} pred {out['D']['pred_equal']}/{n_per_leaf}; "
                      f"RL max|dllr| {out['RL']['max_abs_dllr']:.2e} ({time.time() - t0:.0f} s)", flush=True)
    res["total"] = tot
    res["note"] = ("stored D_llr and ML(RL) llr are float32, so |dllr| up to about 1e-7 x |llr| is rounding; D_logp and "
                   "the ML(RL) logp are float64")
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r2_redecode.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (5, 7):
        raise SystemExit(__doc__)
    main(*a[:5], *(int(x) for x in a[5:]))
