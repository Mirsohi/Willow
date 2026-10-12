"""Validation and timing of ml_exact_fast.FastExactDecoder (compact slots + shot batching):
(1) brute-force enumeration on tiny random and circuit sub-DEMs; (2) identical posteriors and log P(s) to the
reference ExactFrontierDecoder on d=3 SI1000 circuits (r = 10, 30); (3) cost per round per shot for both.
Run: python3 tests/test_exact_fast.py"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from dem_utils import parse_dem
from ml_exact import ExactFrontierDecoder, brute_force_table
from ml_exact_fast import FastExactDecoder
from mismatch import si_family
from test_exact import random_dem, sub_dem

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    worst = 0.0
    c3 = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    dm3 = parse_dem(si_family(c3, 3e-3, 1.5e-2, 3e-4).detector_error_model(decompose_errors=False))
    dms = [random_dem(rng, 8, 14, nobs=1) for _ in range(4)]
    dms += [sub_dem(dm3, np.sort(rng.choice(dm3.num_faults, size=16, replace=False)), rng, pscale=30) for _ in range(4)]
    for t, dm in enumerate(dms):
        tab = brute_force_table(dm)
        keys = list(tab.keys())[:64]
        S = np.array([[(k >> i) & 1 for i in range(dm.num_detectors)] for k in keys], np.uint8)
        ref = np.array([tab[k] for k in keys])
        for B in (1, 3, 16):
            post, logp = FastExactDecoder(dm, batch=B).class_probs(S)
            e1 = np.max(np.abs(post - ref / ref.sum(1, keepdims=True)))
            e2 = np.max(np.abs(logp - np.log(ref.sum(1))))
            worst = max(worst, e1, e2)
        print(f"tiny DEM #{t}: M={dm.num_faults} max|P - brute|={e1:.1e} max|logP - brute|={e2:.1e}")
    for r in (10, 30):
        ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=r)
        c = si_family(ideal, 3e-3, 1.5e-2, 3e-4)
        dm = parse_dem(c.detector_error_model(decompose_errors=False))
        det = c.compile_detector_sampler(seed=r).sample(3000).astype(np.uint8)
        E = ExactFrontierDecoder(dm); Fd = FastExactDecoder(dm)
        E.class_probs(det[:8]); Fd.class_probs(det[:8])
        t0 = time.time(); pe, le = E.class_probs(det); te = time.time() - t0
        t0 = time.time(); pf, lf = Fd.class_probs(det); tf = time.time() - t0
        d1 = np.nanmax(np.abs(pf - pe)); d2 = np.nanmax(np.abs(lf - le))
        worst = max(worst, d1, d2)
        print(f"d3 r{r} SI1000(3e-3), {len(det)} shots: max|dP|={d1:.1e} max|dlogP|={d2:.1e}; reference "
              f"{1e3 * te / len(det) / r:.3f} ms/round/shot, fast {1e3 * tf / len(det) / r:.3f} ms/round/shot "
              f"({te / tf:.1f}x; fraction of dense work {Fd.work_ratio:.2f})")
    print("PASS" if worst < 1e-10 else "FAIL")
