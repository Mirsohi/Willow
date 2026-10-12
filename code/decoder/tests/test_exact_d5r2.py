# Exact joint ML reference at d=5, r=2 (dense frontier W=23) vs sparse frontier and MPS.
import sys, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_exact import ExactFrontierDecoder
from ml_sparse import SparseFrontierDecoder
d, r, p = 5, 2, 3e-3
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
c = memory_circuit(d, r, "Z", p)
dm = parse_dem(c.detector_error_model(decompose_errors=False))
det, obs = c.compile_detector_sampler(seed=21).sample(20000, separate_observables=True)
det = det.astype(np.uint8)
# enrich with "hard" shots: take shots with the most detection events
order = np.argsort(-det.sum(1))
sel = np.concatenate([order[:N // 2], np.arange(N // 2)])
det, obs = det[sel], obs[sel]
out = (sys.argv[2] if len(sys.argv) > 2 else "exact_d5r2.npz")
ex = ExactFrontierDecoder(dm)
print("W =", ex.W, flush=True)
if os.path.exists(out):
    z = np.load(out); pe, le = z["pe"], z["le"]
else:
    t0 = time.time(); pe, le = ex.class_probs(det); t1 = time.time()
    np.savez(out, pe=pe, le=le, sel=sel)
    print(f"exact: {(t1-t0)/len(det):.2f} s/shot, LER={np.mean(pe.argmax(1)!=obs[:,0]):.4f}", flush=True)
for tau in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    s = SparseFrontierDecoder(dm, tau=tau, kmax=1000000); s.class_probs(det[:2])
    t0 = time.time(); ps_, ls_ = s.class_probs(det); t1 = time.time()
    print(f"sparse tau={tau:.0e}: max|post diff|={np.nanmax(np.abs(ps_[:,1]-pe[:,1])):.2e} mean={np.nanmean(np.abs(ps_[:,1]-pe[:,1])):.2e} max|logP diff|={np.nanmax(np.abs(ls_-le)):.2e} disagree={np.sum(ps_.argmax(1)!=pe.argmax(1))}/{len(det)} {1e3*(t1-t0)/len(det):.2f} ms/shot meanK={s.last_maxk.mean():.0f} maxK={s.last_maxk.max()}", flush=True)
