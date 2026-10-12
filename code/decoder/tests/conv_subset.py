"""Convergence of the sparse ML decoder at small tau on a subset of a bench_sim run.
Loads <bench dir>/ml_sparse_<tau0>.npz (posteriors over all shots), re-samples the same shots, picks the
n_hard shots with posterior closest to 1/2 plus n_rand random shots, runs smaller tau values and compares.
Usage: python3 conv_subset.py --d 5 --rounds 5 --p 3e-3 --basis Z --shots 2000 --seed 12345 --bench <dir>
           --tau0 1e-08 --taus 1e-10,1e-12 --n-hard 60 --n-rand 60 --kmax 1000000 --budget 900 --out <json>"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_sparse import SparseFrontierDecoder

ap = argparse.ArgumentParser()
for k, t, dflt in [("--d", int, 5), ("--rounds", int, 5), ("--p", float, 3e-3), ("--basis", str, "Z"), ("--shots", int, 2000),
                   ("--seed", int, 12345), ("--bench", str, None), ("--tau0", str, "1e-08"), ("--taus", str, "1e-10"),
                   ("--n-hard", int, 60), ("--n-rand", int, 60), ("--kmax", int, 1000000), ("--budget", float, 900), ("--out", str, None)]:
    ap.add_argument(k, type=t, default=dflt)
a = ap.parse_args()
circ = memory_circuit(a.d, a.rounds, a.basis, a.p)
dm = parse_dem(circ.detector_error_model(decompose_errors=False))
det, obs = circ.compile_detector_sampler(seed=a.seed + 1000 * a.rounds + a.d).sample(a.shots, separate_observables=True)
det = det.astype(np.uint8)
z = np.load(os.path.join(a.bench, f"ml_sparse_{a.tau0}.npz"))
p0 = z["post1"].astype(float); l0 = z["logp"]
rng = np.random.default_rng(0)
hard = np.argsort(np.abs(p0 - 0.5))[:a.n_hard]
rest = np.setdiff1d(np.arange(len(p0)), hard)
sel = np.concatenate([hard, rng.choice(rest, a.n_rand, replace=False)])
res = {"args": vars(a), "sel": sel.tolist(), "tau0_post1": p0[sel].tolist(), "runs": {}}
ref_p, ref_l, ref_name = p0[sel], l0[sel], a.tau0
for tau in [float(t) for t in a.taus.split(",")]:
    s = SparseFrontierDecoder(dm, tau=tau, kmax=a.kmax); s.class_probs(det[sel[:1]])
    P, L, T, K = [], [], [], []
    t_start = time.time()
    for i in sel:
        t0 = time.time(); post, lp = s.class_probs(det[i:i + 1]); T.append(time.time() - t0)
        P.append(post[0, 1]); L.append(lp[0]); K.append(int(s.last_maxk[0]))
        if time.time() - t_start > a.budget:
            break
    n = len(P); P = np.array(P); L = np.array(L)
    dP = np.abs(P - ref_p[:n]); dL = np.abs(L - ref_l[:n])
    rec = {"n": n, "ms_per_shot": 1e3 * float(np.mean(T)), "meanK": float(np.mean(K)), "maxK": int(np.max(K)),
           "kmax_hit": int(np.sum(np.array(K) >= a.kmax)), "post1": P.tolist(), "logp": L.tolist(),
           f"max_abs_dP_vs_{ref_name}": float(dP.max()), f"max_abs_dlogP_vs_{ref_name}": float(dL.max()),
           f"decisions_differ_vs_{ref_name}": int(np.sum((P > 0.5) != (ref_p[:n] > 0.5))),
           "hard_shots_included": int(min(n, a.n_hard))}
    res["runs"][f"{tau:.0e}"] = rec
    print(f"tau={tau:.0e}: n={n} {rec['ms_per_shot']:.0f} ms/shot meanK={rec['meanK']:.0f} maxK={rec['maxK']} kmax_hit={rec['kmax_hit']} "
          f"| vs {ref_name}: max|dP|={dP.max():.2e} max|dlogP|={dL.max():.2e} decisions differ={rec[f'decisions_differ_vs_{ref_name}']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    ref_p, ref_l, ref_name = P, L, f"{tau:.0e}"
    if n < len(sel):
        break
