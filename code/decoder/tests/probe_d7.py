"""d=7 feasibility probe for the sparse-frontier ML decoder (joint X+Z DEM, SI1000 p=3e-3, Z memory).
Measures frontier width W, number of kept frontier states K, time per shot and self-convergence across tau,
shot by shot, with a wall-clock budget per tau.  Usage: python3 probe_d7.py <rounds> <shots> <budget_s> <out.json>"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_sparse import SparseFrontierDecoder
import baselines

r = int(sys.argv[1]); N = int(sys.argv[2]); budget = float(sys.argv[3]); out = sys.argv[4]
d, p = 7, 3e-3
c = memory_circuit(d, r, "Z", p)
demu = c.detector_error_model(decompose_errors=False)
demd = c.detector_error_model(decompose_errors=True)
dm = parse_dem(demu)
det, obs = c.compile_detector_sampler(seed=31).sample(N, separate_observables=True)
det = det.astype(np.uint8)
res = {"d": d, "r": r, "p": p, "N": N, "num_detectors": int(c.num_detectors), "faults": int(dm.num_faults), "tau": {}}
for name in ["mwpm", "corr_mwpm", "tesseract"]:
    D = baselines.make(name, demu, demd); t0 = time.time(); pr = D.decode(det); dt = time.time() - t0
    res[name] = {"pred": pr[:, 0].tolist(), "ms_per_shot": 1e3 * dt / N}
res["obs"] = obs[:, 0].astype(int).tolist()
posts = {}
for tau, kmax in [(1e-3, 300000), (1e-4, 300000), (1e-5, 300000), (1e-6, 300000)]:
    s = SparseFrontierDecoder(dm, tau=tau, kmax=kmax)
    if "W" not in res:
        res["W"] = int(s.prog["W"]); print(f"d=7 r={r}: detectors={c.num_detectors} faults={dm.num_faults} W={res['W']}", flush=True)
    s.class_probs(det[:1])  # JIT warm-up
    P, T, K = [], [], []
    t_start = time.time()
    for i in range(N):
        t0 = time.time(); post, lp = s.class_probs(det[i:i + 1]); T.append(time.time() - t0)
        P.append(float(post[0, 1])); K.append(int(s.last_maxk[0]))
        if time.time() - t_start > budget:
            break
    n = len(P); posts[tau] = np.array(P)
    rec = {"n": n, "ms_per_shot_mean": 1e3 * float(np.mean(T)), "ms_per_shot_max": 1e3 * float(np.max(T)),
           "meanK": float(np.mean(K)), "maxK": int(np.max(K)), "kmax": kmax, "kmax_hit": int(np.sum(np.array(K) >= kmax)),
           "post1": P}
    res["tau"][f"{tau:.0e}"] = rec
    msg = f"tau={tau:.0e}: n={n} {rec['ms_per_shot_mean']:.0f} ms/shot (max {rec['ms_per_shot_max']:.0f}) meanK={rec['meanK']:.0f} maxK={rec['maxK']} kmax_hit={rec['kmax_hit']}"
    prev = [t for t in posts if t > tau]
    if prev:
        q = posts[min(prev)]; m = min(len(q), n)
        msg += f" | vs tau={min(prev):.0e} on {m} shots: max|dP|={np.max(np.abs(q[:m]-posts[tau][:m])):.2e} decisions differ={int(np.sum((q[:m]>0.5)!=(posts[tau][:m]>0.5)))}"
    print(msg, flush=True)
    json.dump(res, open(out, "w"), indent=1)
    if n < 3:
        break
print("done", flush=True)
