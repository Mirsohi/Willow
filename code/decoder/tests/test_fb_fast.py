"""ml_fb_fast.ForwardBackwardFast against the validated ml_fb_exact.ForwardBackwardExact (itself checked against
brute force and finite differences in test_fb_exact.py): log P, all fault posteriors q_j, per-shot scores, joint
and marginal, d=3 SI1000-family circuits at r = 4, 10; plus timing. Run: python3 tests/test_fb_fast.py"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import stim
from dem_utils import parse_dem
from mismatch import si_family
from ml_fb_exact import ForwardBackwardExact
from ml_fb_fast import ForwardBackwardFast

if __name__ == "__main__":
    worst = 0.0
    rng = np.random.default_rng(4)
    for r, n in ((4, 300), (10, 1000)):
        ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=r)
        c = si_family(ideal, 2.1e-3, 4.5e-2, 3e-4)
        dm = parse_dem(c.detector_error_model(decompose_errors=False))
        det, obs = c.compile_detector_sampler(seed=r).sample(n, separate_observables=True)
        det = det.astype(np.uint8); L = obs[:, 0].astype(np.int64)
        J = rng.normal(size=(dm.num_faults, 3)) * dm.p[:, None]
        A = ForwardBackwardExact(dm); Bf = ForwardBackwardFast(dm)
        A.run(det[:4], L=L[:4], J=J); Bf.run(det[:4], L=L[:4], J=J)
        for joint in (True, False):
            t0 = time.time(); ra = A.run(det, L=L if joint else None, J=J, store_q=True); ta = time.time() - t0
            t0 = time.time(); rb = Bf.run(det, L=L if joint else None, J=J, store_q=True); tb = time.time() - t0
            e = [np.max(np.abs(ra["logp"] - rb["logp"])), np.max(np.abs(ra["q"] - rb["q"])),
                 np.max(np.abs(ra["qsum"] - rb["qsum"]) / n), np.max(np.abs(ra["grads"] - rb["grads"]) / (1 + np.abs(ra["grads"])))]
            worst = max(worst, *e)
            print(f"d3 r{r} {'joint' if joint else 'marginal'} n={n}: max|dlogP|={e[0]:.1e} max|dq|={e[1]:.1e} "
                  f"max|dqsum|/n={e[2]:.1e} max rel dscore={e[3]:.1e}; reference {1e3 * ta / n / r:.3f} ms/round/shot, "
                  f"fast {1e3 * tb / n / r:.3f} ms/round/shot ({ta / tb:.1f}x)", flush=True)
    print("PASS" if worst < 1e-9 else "FAIL")
