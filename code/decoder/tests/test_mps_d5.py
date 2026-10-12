"""Independent chi-controlled cross-check at d=5 (r=5): boundary-MPS ML (ml_mps_fast, reweighted) on the hardest shots
of the conv_subset run, compared with the sparse-frontier posteriors at tau=1e-10.
Usage: python3 test_mps_d5.py <conv_d5_r5.json> <n_shots> <budget_s per chi> <out.json>"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_mps_fast import FastMPSDecoder

conv = json.load(open(sys.argv[1])); n = int(sys.argv[2]); budget = float(sys.argv[3]); out = sys.argv[4]
a = conv["args"]
c = memory_circuit(a["d"], a["rounds"], a["basis"], a["p"])
dm = parse_dem(c.detector_error_model(decompose_errors=False))
det, obs = c.compile_detector_sampler(seed=a["seed"] + 1000 * a["rounds"] + a["d"]).sample(a["shots"], separate_observables=True)
det = det.astype(np.uint8)
sel = np.array(conv["sel"])
ref = np.array(conv["runs"]["1e-10"]["post1"]); m = min(n, len(ref)); sel = sel[:m]; ref = ref[:m]
res = {"n": int(m), "ref": "sparse tau=1e-10", "chi": {}}
for chi in [8, 16, 32, 64]:
    D = FastMPSDecoder(dm, chi=chi, cutoff=1e-14, reweight=True); D.class_probs(det[sel[:1]])
    P, T = [], []; t_start = time.time()
    for i in sel:
        t0 = time.time(); post, lp = D.class_probs(det[i:i + 1]); T.append(time.time() - t0); P.append(float(post[0, 1]))
        if time.time() - t_start > budget:
            break
    k = len(P); P = np.array(P)
    rec = {"n": k, "ms_per_shot": 1e3 * float(np.mean(T)), "max_abs_dP_vs_ref": float(np.nanmax(np.abs(P - ref[:k]))),
           "decisions_differ": int(np.sum((np.nan_to_num(P, nan=0.5) > 0.5) != (ref[:k] > 0.5))), "nan": int(np.isnan(P).sum()), "post1": P.tolist()}
    res["chi"][str(chi)] = rec
    print(f"MPS chi={chi}: n={k} {rec['ms_per_shot']:.0f} ms/shot | vs sparse 1e-10: max|dP|={rec['max_abs_dP_vs_ref']:.2e} decisions differ={rec['decisions_differ']} nan={rec['nan']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    if k < m:
        break
