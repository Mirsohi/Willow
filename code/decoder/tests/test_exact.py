"""Validation of ExactFrontierDecoder against brute-force enumeration on tiny DEMs (random sub-DEMs of
real circuit-level DEMs and random synthetic DEMs). Run: python3 test_exact.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from dem_utils import DemMatrices, parse_dem
from ml_exact import ExactFrontierDecoder, brute_force_table
from willow_sim import memory_circuit


def sub_dem(dm, faults, rng, pscale=None):
    dets = [dm.dets[j] for j in faults]
    used = sorted(set(d for ds in dets for d in ds))
    remap = {d: i for i, d in enumerate(used)}
    nd = [tuple(sorted(remap[d] for d in ds)) for ds in dets]
    p = dm.p[faults].copy()
    if pscale is not None:
        p = np.clip(p * pscale, 1e-4, 0.45)
    return DemMatrices(len(used), dm.num_observables, nd, dm.obs[faults].copy(), p,
                       {remap[d]: dm.coords[d] for d in used}, [])


def random_dem(rng, nd, M, nobs=1, pmax=0.3):
    dets = []
    for _ in range(M):
        k = rng.integers(1, 5)
        dets.append(tuple(sorted(rng.choice(nd, size=min(k, nd), replace=False).tolist())))
    obs = rng.integers(0, 1 << nobs, size=M)
    p = rng.uniform(0.001, pmax, size=M)
    # merge duplicates is not required for brute force; the decoder treats them as separate faults
    return DemMatrices(nd, nobs, dets, obs.astype(np.int64), p, {}, [])


def check(dm, label):
    table = brute_force_table(dm)
    dec = ExactFrontierDecoder(dm)
    synds = list(table.keys())
    S = np.array([[(s >> i) & 1 for i in range(dm.num_detectors)] for s in synds], dtype=np.uint8)
    post, logp = dec.class_probs(S)
    maxerr_post, maxerr_logp = 0.0, 0.0
    for k, s in enumerate(synds):
        v = table[s]
        tot = v.sum()
        maxerr_post = max(maxerr_post, np.max(np.abs(post[k] - v / tot)))
        maxerr_logp = max(maxerr_logp, abs(logp[k] - np.log(tot)))
    # probabilities over all syndromes must sum to 1
    total = sum(v.sum() for v in table.values())
    print(f"{label}: M={dm.num_faults} N={dm.num_detectors} W={dec.W} syndromes={len(synds)} "
          f"max|post err|={maxerr_post:.2e} max|logP err|={maxerr_logp:.2e} sum P(s)={total:.12f}")
    return maxerr_post, maxerr_logp


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    worst = 0
    c = memory_circuit(3, 3, "Z", 3e-3)
    dm = parse_dem(c.detector_error_model(decompose_errors=False))
    for t in range(6):
        faults = np.sort(rng.choice(dm.num_faults, size=18, replace=False))
        e1, e2 = check(sub_dem(dm, faults, rng, pscale=30), f"sub-DEM d3r3 #{t}")
        worst = max(worst, e1, e2)
    for t in range(6):
        dm2 = random_dem(rng, nd=int(rng.integers(4, 12)), M=18, nobs=int(rng.integers(1, 3)))
        e1, e2 = check(dm2, f"random DEM #{t}")
        worst = max(worst, e1, e2)
    print("WORST", worst)
    assert worst < 1e-9
    print("PASS")
