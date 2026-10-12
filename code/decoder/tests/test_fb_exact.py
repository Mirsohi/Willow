"""Validation of ml_fb_exact.ForwardBackwardExact:
(1) fault posterior marginals q_j = P(e_j=1 | s, L) and P(e_j=1 | s) against brute-force enumeration on tiny DEMs;
(2) log-likelihoods against ml_exact; (3) d ln P / d p_j = (q_j - p_j)/(p_j(1-p_j)) against finite differences
on a real d=3 circuit DEM. Run: python3 tests/test_fb_exact.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from dem_utils import DemMatrices, parse_dem
from ml_exact import ExactFrontierDecoder
from ml_fb_exact import ForwardBackwardExact
from willow_sim import memory_circuit
from test_exact import sub_dem, random_dem


def brute(dm):
    M = dm.num_faults
    cfg = np.arange(1 << M, dtype=np.int64)
    syn = np.zeros(1 << M, np.int64); cls = np.zeros(1 << M, np.int64); logw = np.zeros(1 << M)
    on_all = []
    for j in range(M):
        on = (cfg >> j) & 1
        m = 0
        for d in dm.dets[j]:
            m ^= (1 << d)
        syn ^= np.where(on == 1, m, 0); cls ^= np.where(on == 1, dm.obs[j], 0)
        logw += np.where(on == 1, np.log(dm.p[j]), np.log1p(-dm.p[j]))
        on_all.append(on)
    return syn, cls, np.exp(logw), np.array(on_all)


def check_tiny(dm, label, rng):
    syn, cls, w, on = brute(dm)
    synds = np.unique(syn)
    synds = synds[rng.permutation(len(synds))[:40]]
    S = np.array([[(s >> i) & 1 for i in range(dm.num_detectors)] for s in synds], np.uint8)
    fb = ForwardBackwardExact(dm)
    worst_q, worst_l = 0.0, 0.0
    for joint in [False, True]:
        for k, s in enumerate(synds):
            sel = syn == s
            if joint:
                Lval = int(cls[sel][0] & 1)
                sel = sel & ((cls & 1) == Lval)
                r = fb.run(S[k:k + 1], L=np.array([Lval]), store_q=True)
            else:
                r = fb.run(S[k:k + 1], store_q=True)
            Pst = w[sel].sum()
            qb = (on[:, sel] * w[sel]).sum(axis=1) / Pst
            worst_q = max(worst_q, np.max(np.abs(r["q"][0] - qb)))
            worst_l = max(worst_l, abs(r["logp"][0] - np.log(Pst)))
    print(f"{label}: M={dm.num_faults} W={fb.prog['W']} max|q - brute|={worst_q:.2e} max|logP - brute|={worst_l:.2e}")
    return max(worst_q, worst_l)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    worst = 0.0
    c = memory_circuit(3, 3, "Z", 3e-3)
    dm = parse_dem(c.detector_error_model(decompose_errors=False))
    for t in range(4):
        faults = np.sort(rng.choice(dm.num_faults, size=16, replace=False))
        worst = max(worst, check_tiny(sub_dem(dm, faults, rng, pscale=30), f"sub-DEM d3r3 #{t}", rng))
    for t in range(4):
        worst = max(worst, check_tiny(random_dem(rng, 8, 15, nobs=1), f"random DEM #{t}", rng))
    # real DEM: log-likelihoods vs ml_exact, gradient vs finite differences
    c = memory_circuit(3, 4, "Z", 3e-3)
    dm = parse_dem(c.detector_error_model(decompose_errors=False))
    det, obs = c.compile_detector_sampler(seed=5).sample(200, separate_observables=True)
    det = det.astype(np.uint8); L = obs[:, 0].astype(np.int64)
    fb = ForwardBackwardExact(dm)
    post, lp = ExactFrontierDecoder(dm).class_probs(det)
    r = fb.run(det, L=L, store_q=True)
    lj = lp + np.log(post[np.arange(len(L)), L])
    print(f"d3 r4 joint logP vs ml_exact: max diff {np.max(np.abs(r['logp'] - lj)):.2e}")
    r0 = fb.run(det, store_q=True)
    print(f"d3 r4 marginal logP vs ml_exact: max diff {np.max(np.abs(r0['logp'] - lp)):.2e}")
    worst = max(worst, np.max(np.abs(r['logp'] - lj)), np.max(np.abs(r0['logp'] - lp)))
    gerr = 0.0
    for j in rng.choice(dm.num_faults, size=8, replace=False):
        h = 1e-6 * dm.p[j]
        vals = []
        for sgn in (+1, -1):
            p2 = dm.p.copy(); p2[j] += sgn * h
            d2 = DemMatrices(dm.num_detectors, dm.num_observables, dm.dets, dm.obs, p2, dm.coords, dm.undetectable)
            q2, l2 = ExactFrontierDecoder(d2).class_probs(det)
            vals.append(np.sum(l2 + np.log(q2[np.arange(len(L)), L])))
        fd = (vals[0] - vals[1]) / (2 * h)
        an = np.sum((r["q"][:, j] - dm.p[j]) / (dm.p[j] * (1 - dm.p[j])))
        gerr = max(gerr, abs(fd - an) / max(1.0, abs(an)))
        print(f"  fault {j}: p={dm.p[j]:.2e} dlnL/dp finite-diff={fd:.6e} analytic={an:.6e}")
    print(f"max relative gradient error {gerr:.2e}")
    print("PASS" if worst < 1e-10 and gerr < 1e-5 else "FAIL")
