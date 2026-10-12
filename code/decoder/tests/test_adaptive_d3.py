"""Adaptive tau ladder (ml_sparse.AdaptiveSparseDecoder: re-decode at the next tau while |2P - 1| <= kappa * D) against
EXACT ML at d = 3 on random shots (SI1000 3.5e-3; r = 10: 4000 shots, r = 50: 1000 shots; fixed seeds).
Usage: python3 tests/test_adaptive_d3.py   (output kept in results/decoder/speed/adaptive_d3_vs_exact.log)
"""
import sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, stim
from dem_utils import parse_dem
from mismatch import si_family
from ml_sparse import AdaptiveSparseDecoder
from ml_exact_fast import FastExactDecoder
p = 3.5e-3
for r, n in ((10, 4000), (50, 1000)):
    c = si_family(stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=r), p, 5 * p, p / 10)
    dm = parse_dem(c.detector_error_model(decompose_errors=False))
    det = c.compile_detector_sampler(seed=5 + r).sample(n).astype(np.uint8)
    pe, le = FastExactDecoder(dm).class_probs(det)
    for taus, kappa in (((1e-4, 1e-6, 1e-8), 2.0), ((1e-4, 1e-6, 1e-8), 4.0), ((1e-5, 1e-7), 2.0)):
        A = AdaptiveSparseDecoder(dm, taus=taus, kappa=kappa)
        t0 = time.time(); pa, la = A.class_probs(det); t = time.time() - t0
        print(f"r={r} n={n} taus={taus} kappa={kappa}: levels {np.bincount(A.last_level, minlength=len(taus)).tolist()} "
              f"decisions differ {int(np.sum((pa[:, 1] > .5) != (pe[:, 1] > .5)))} max|dP| {float(np.max(np.abs(pa[:, 1] - pe[:, 1]))):.2e} "
              f"{1e3 * t / n / r:.3f} ms/round/shot", flush=True)
