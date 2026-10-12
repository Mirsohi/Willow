import sys, time
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
import numpy as np
from willow_sim import memory_circuit
from dem_utils import parse_dem
from ml_exact import ExactFrontierDecoder
from ml_cotengra import CotengraExact
for (d, r) in [(3, 3), (3, 6)]:
    c = memory_circuit(d, r, "Z", 3e-3)
    dm = parse_dem(c.detector_error_model(decompose_errors=False))
    det, obs = c.compile_detector_sampler(seed=3).sample(100, separate_observables=True)
    det = det.astype(np.uint8)
    t0 = time.time(); ce = CotengraExact(dm, max_time=20); t1 = time.time()
    pc, lc = ce.class_probs(det); t2 = time.time()
    pe, le = ExactFrontierDecoder(dm).class_probs(det)
    print(f"d={d} r={r}: cotengra width={ce.width:.1f} log2cost={np.log2(ce.cost):.1f} path {t1-t0:.1f}s contract {1e3*(t2-t1)/len(det):.1f} ms/shot | max|post diff|={np.max(np.abs(pc[:,1]-pe[:,1])):.2e} max|logP diff|={np.max(np.abs(lc-le)):.2e}", flush=True)
