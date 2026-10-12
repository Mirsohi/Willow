#!/usr/bin/env python3
"""Optimiser comparison for D-hat (PR2 section 7, 2026-10-10 01:19Z). Simulated data only.

Data: the RL DEMs of one series at the given rounds (default d3_at_q10_7/X, r = 10 and 13), n shots each sampled from
the V1 truth of sim_validate.py (faults flipping L x2, others x0.87; stim seed 77 + r); T-hat's classes per leaf.
Fits cond_fit.fit_series with each variant: info = opg | expected; secant = False | one | bfgs | hybrid. Prints convergence,
iterations, evaluations, the final mean ln P(L|s), the last convergence check and the steps taken.
Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B test_optimizer.py <series dir in the extract> 4000 10,13 \
         opg:False expected:False expected:one expected:bfgs opg:bfgs
"""
import sys, os, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import Leaf, load_dm, t_classes, RL_DEM, fit_series
from sim_validate import truth_dm
from mismatch import dm_to_stim

E, n, rs = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
variants = [v.split(":") for v in (sys.argv[4:] or ["opg:False", "expected:False", "expected:one", "expected:bfgs", "opg:bfgs"])]
leaves = []
for r in rs:
    dm = load_dm(os.path.join(E, f"r{r}", RL_DEM)); labels, _ = t_classes(dm, 5000)
    det, obs, _ = dm_to_stim(truth_dm(dm, "V1")).compile_sampler(seed=77 + r).sample(n)
    leaves.append(Leaf(f"r{r}", r, dm, labels, det.astype(np.uint8), obs[:, 0].astype(np.int64)))
res = {}
for info, sec in variants:
    sec = {"False": False, "one": "one", "bfgs": "bfgs", "hybrid": "hybrid"}[sec]
    t = time.process_time()
    f = fit_series(leaves, log=lambda m: None, info=info, secant=sec)
    tag = f"{info}+{sec}"
    res[tag] = f
    steps = [round(max(abs(x) for x in h["step_in_se"]), 3) for h in f["history"][1:]]
    print(f"{tag:15s}: converged {f['converged']}, iterations {f['iterations']}, n_eval {f['n_eval']}, "
          f"ll {f['ll_per_shot']:.9f}, last {f['last_convergence_check']['max_step_in_se']:.3g}, "
          f"cpu {time.process_time() - t:.0f} s\n   steps taken (max |step|/se): {steps}", flush=True)
tags = list(res)
se = np.array(res[tags[0]]["se_opg_ln"], float)
for a in tags[1:]:
    d = np.abs(np.array(res[a]["phi_hat"]) - np.array(res[tags[0]]["phi_hat"])) / se
    print(f"max |phi({a}) - phi({tags[0]})| / se_opg = {np.nanmax(d):.3g}")
