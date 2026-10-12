import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_exact import ExactFrontierDecoder
from ml_mps_fast import FastMPSDecoder
from ml_sparse import SparseFrontierDecoder
d, r, p = 3, 5, 3e-3
c = memory_circuit(d, r, "Z", p)
dm = parse_dem(c.detector_error_model(decompose_errors=False))
det, obs = c.compile_detector_sampler(seed=5).sample(300, separate_observables=True)
ex = ExactFrontierDecoder(dm)
pe, le = ex.class_probs(det)
def rep(label, pm, lm, dt, extra):
    print(f"{label}: max|post diff|={np.nanmax(np.abs(pm[:,1]-pe[:,1])):.2e} mean={np.nanmean(np.abs(pm[:,1]-pe[:,1])):.2e} max|logP diff|={np.nanmax(np.abs(lm-le)):.2e} "
          f"disagree={np.sum(np.nan_to_num(pm,nan=.5).argmax(1)!=pe.argmax(1))}/{len(det)} t/shot={1e3*dt/len(det):.2f} ms {extra}", flush=True)
for tau in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    s = SparseFrontierDecoder(dm, tau=tau, kmax=100000)
    s.class_probs(det[:2])
    t0 = time.time(); pm, lm = s.class_probs(det); t1 = time.time()
    rep(f"sparse tau={tau:.0e}", pm, lm, t1 - t0, f"maxK={s.last_maxk.max()} meanK={s.last_maxk.mean():.0f}")
for rw in [True, False]:
    for chi in [2, 4, 8, 16, 32]:
        m = FastMPSDecoder(dm, chi=chi, cutoff=1e-14, reweight=rw)
        m.class_probs(det[:2])
        t0 = time.time(); pm, lm = m.class_probs(det); t1 = time.time()
        rep(f"mps rw={rw} chi={chi}", pm, lm, t1 - t0, f"maxbond={m.last_maxbond.max()}")
