import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem, restrict_dem
from ml_exact import ExactFrontierDecoder
from ml_sparse import SparseFrontierDecoder
import baselines
d, r, p = 5, 5, 3e-3
c = memory_circuit(d, r, "Z", p)
demu = c.detector_error_model(decompose_errors=False); demd = c.detector_error_model(decompose_errors=True)
dm = parse_dem(demu)
N = 2000
det, obs = c.compile_detector_sampler(seed=9).sample(N, separate_observables=True)
det = det.astype(np.uint8)
# Z-only exact reference vs sparse on the Z-only DEM
pos0 = set((dm.coords[i][0], dm.coords[i][1]) for i in range(dm.num_detectors) if dm.coords[i][-1] == 0)
keep = [i for i in range(dm.num_detectors) if (dm.coords[i][0], dm.coords[i][1]) in pos0]
dz = restrict_dem(dm, keep)
ez = ExactFrontierDecoder(dz); ez.class_probs(det[:2, keep])
t0 = time.time(); pz, lz = ez.class_probs(det[:, keep]); t1 = time.time()
print(f"Z-only exact: W={ez.W} LER={np.mean(pz.argmax(1)!=obs[:,0]):.4f} {1e3*(t1-t0)/N:.2f} ms/shot", flush=True)
for tau in [1e-6, 1e-8, 1e-10]:
    s = SparseFrontierDecoder(dz, tau=tau, kmax=200000); s.class_probs(det[:2, keep])
    t0 = time.time(); ps_, ls_ = s.class_probs(det[:, keep]); t1 = time.time()
    print(f"Z-only sparse tau={tau:.0e}: max|post diff|={np.nanmax(np.abs(ps_[:,1]-pz[:,1])):.2e} disagree={np.sum(ps_.argmax(1)!=pz.argmax(1))} {1e3*(t1-t0)/N:.2f} ms/shot meanK={s.last_maxk.mean():.0f} maxK={s.last_maxk.max()}", flush=True)
# joint DEM: sparse convergence
prev = None
res = {}
for tau in [1e-4, 1e-6, 1e-8, 1e-10]:
    s = SparseFrontierDecoder(dm, tau=tau, kmax=400000); s.class_probs(det[:2])
    t0 = time.time(); ps_, ls_ = s.class_probs(det); t1 = time.time()
    res[tau] = ps_
    msg = f"joint sparse tau={tau:.0e}: LER={np.mean(ps_.argmax(1)!=obs[:,0]):.4f} {1e3*(t1-t0)/N:.2f} ms/shot meanK={s.last_maxk.mean():.0f} maxK={s.last_maxk.max()}"
    if prev is not None:
        msg += f" | vs prev tau: max|post diff|={np.nanmax(np.abs(ps_[:,1]-prev[:,1])):.2e} disagree={np.sum(ps_.argmax(1)!=prev.argmax(1))}"
    print(msg, flush=True)
    prev = ps_
for name in ["mwpm", "corr_mwpm", "beliefmatch", "tesseract"]:
    D = baselines.make(name, demu, demd); t0 = time.time(); pr = D.decode(det); t1 = time.time()
    print(f"{name}: LER={np.mean(pr[:,0]!=obs[:,0]):.4f} {1e3*(t1-t0)/N:.2f} ms/shot", flush=True)
print(f"Z-only exact ML LER={np.mean(pz.argmax(1)!=obs[:,0]):.4f}; joint sparse(1e-10) LER={np.mean(prev.argmax(1)!=obs[:,0]):.4f}")
