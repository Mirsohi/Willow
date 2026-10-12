"""d=7 r=10 feasibility probe on HARD shots: sparse ML at tau=1e-5 on N shots, then tau = 1e-6, 1e-7, 1e-8 on the
n_hard most uncertain shots (+ n_rand random), with self-convergence, plus baselines on all N shots and a p-sensitivity
run (cost at a higher physical error rate). Usage: python3 probe_d7b.py <rounds> <N> <n_hard> <n_rand> <budget_s> <out.json>"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_sparse import SparseFrontierDecoder
import baselines

r, N, nh, nr, budget, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
d, p = 7, 3e-3
res = {"d": d, "r": r, "p": p, "N": N}
def setup(p):
    c = memory_circuit(d, r, "Z", p)
    demu = c.detector_error_model(decompose_errors=False)
    return c, demu, c.detector_error_model(decompose_errors=True), parse_dem(demu)
c, demu, demd, dm = setup(p)
det, obs = c.compile_detector_sampler(seed=41).sample(N, separate_observables=True)
det = det.astype(np.uint8); obs = obs[:, 0].astype(np.uint8)
res.update(num_detectors=int(c.num_detectors), faults=int(dm.num_faults), p_det=float(det.mean()))
def run(dm, tau, idx, budget, kmax=1000000):
    s = SparseFrontierDecoder(dm, tau=tau, kmax=kmax); s.class_probs(det[idx[:1]])
    P, T, K = [], [], []; t_start = time.time()
    for i in idx:
        t0 = time.time(); post, lp = s.class_probs(det[i:i + 1]); T.append(time.time() - t0)
        P.append(float(post[0, 1])); K.append(int(s.last_maxk[0]))
        if time.time() - t_start > budget:
            break
    return np.array(P), np.array(T), np.array(K), int(s.prog["W"])
P5, T5, K5, W = run(dm, 1e-5, np.arange(N), budget)
n5 = len(P5)
res["W"] = W
res["tau_1e-05_all"] = {"n": n5, "ms_per_shot": 1e3 * float(T5.mean()), "meanK": float(K5.mean()), "maxK": int(K5.max()),
                        "errors": int(np.sum((P5 > 0.5) != obs[:n5])), "post1": P5.tolist()}
print(f"d=7 r={r} p={p}: dets={c.num_detectors} faults={dm.num_faults} W={W} p_det={det.mean():.4f}", flush=True)
print(f"tau=1e-05 on {n5} shots: {1e3*T5.mean():.0f} ms/shot meanK={K5.mean():.0f} maxK={K5.max()} ML errors={res['tau_1e-05_all']['errors']}", flush=True)
for name in ["mwpm", "corr_mwpm", "tesseract"]:
    D = baselines.make(name, demu, demd); t0 = time.time(); pr = D.decode(det[:n5]); dt = time.time() - t0
    e = pr[:, 0] != obs[:n5]
    res[name] = {"ms_per_shot": 1e3 * dt / n5, "errors": int(e.sum()),
                 "ML_wrong_other_right": int(np.sum(((P5 > 0.5) != obs[:n5]) & ~e)), "ML_right_other_wrong": int(np.sum(((P5 > 0.5) == obs[:n5]) & e))}
    print(f"{name}: errors={e.sum()}/{n5} {1e3*dt/n5:.1f} ms/shot paired vs ML(1e-5): {res[name]['ML_wrong_other_right']} / {res[name]['ML_right_other_wrong']}", flush=True)
json.dump(res, open(out, "w"), indent=1)
rng = np.random.default_rng(1)
hard = np.argsort(np.abs(P5 - 0.5))[:nh]
sel = np.concatenate([hard, rng.choice(np.setdiff1d(np.arange(n5), hard), nr, replace=False)])
res["sel"] = sel.tolist(); ref = P5[sel]; refname = "1e-05"
for tau in [1e-6, 1e-7, 1e-8]:
    P, T, K, _ = run(dm, tau, sel, budget)
    n = len(P)
    rec = {"n": n, "ms_per_shot": 1e3 * float(T.mean()), "ms_per_shot_max": 1e3 * float(T.max()), "meanK": float(K.mean()), "maxK": int(K.max()),
           "max_abs_dP_vs_prev": float(np.max(np.abs(P - ref[:n]))), "decisions_differ_vs_prev": int(np.sum((P > 0.5) != (ref[:n] > 0.5))),
           "prev": refname, "post1": P.tolist()}
    res[f"tau_{tau:.0e}_subset"] = rec
    print(f"tau={tau:.0e} on {n} hard/random shots: {rec['ms_per_shot']:.0f} ms/shot (max {rec['ms_per_shot_max']:.0f}) meanK={rec['meanK']:.0f} maxK={rec['maxK']} "
          f"| vs {refname}: max|dP|={rec['max_abs_dP_vs_prev']:.2e} decisions differ={rec['decisions_differ_vs_prev']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    ref, refname = P, f"{tau:.0e}"
    if n < len(sel):
        break
# p sensitivity: cost at a Willow-like detection probability
for p2 in [3.5e-3]:
    c2, demu2, demd2, dm2 = setup(p2)
    det = c2.compile_detector_sampler(seed=43).sample(40, separate_observables=True)[0].astype(np.uint8)
    for tau in [1e-5, 1e-6]:
        P, T, K, _ = run(dm2, tau, np.arange(40), budget / 2)
        res[f"p{p2:.1e}_tau{tau:.0e}"] = {"p_det": float(det.mean()), "n": len(P), "ms_per_shot": 1e3 * float(T.mean()), "meanK": float(K.mean()), "maxK": int(K.max())}
        print(f"p={p2:.1e} (p_det={det.mean():.4f}) tau={tau:.0e}: {1e3*T.mean():.0f} ms/shot meanK={K.mean():.0f} maxK={K.max()} n={len(P)}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
print("done", flush=True)
