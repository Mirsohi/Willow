"""scout: independent cross-check of decoder's exact TN-ML posteriors with scout's bucket-elimination ML (ml_dp.py).
Reads a bundle directory (dem_undecomposed.dem, detection_events.b8, obs_flips_actual.01, decoder_exact_post1.npy,
decoder_exact_logp.npy) as data only. Writes results/scout/xcheck_<bundle name>.json."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np
import stim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_dp import DPDecoder

b = sys.argv[1].rstrip("/")
out = sys.argv[2]
dem = stim.DetectorErrorModel.from_file(os.path.join(b, "dem_undecomposed.dem"))
nd = dem.num_detectors
det = stim.read_shot_data_file(path=os.path.join(b, "detection_events.b8"), format="b8", num_detectors=nd)
obs = stim.read_shot_data_file(path=os.path.join(b, "obs_flips_actual.01"), format="01", num_observables=1)[:, 0]
their_post = np.load(os.path.join(b, "decoder_exact_post1.npy"), allow_pickle=False)
their_logp = np.load(os.path.join(b, "decoder_exact_logp.npy"), allow_pickle=False)
if len(sys.argv) > 3:  # optional subset: k first and k last shots (bundles may store hard shots first)
    k = int(sys.argv[3])
    idx = np.unique(np.r_[np.arange(min(k, len(det))), np.arange(max(0, len(det) - k), len(det))])
    det, obs, their_post, their_logp = det[idx], obs[idx], their_post[idx], their_logp[idx]
dec = DPDecoder(dem)
t = time.time()
post, logp = dec.posterior_chunked(det, return_logp=True)
dt = time.time() - t
ok = np.isfinite(post) & np.isfinite(their_post)
res = dict(bundle=os.path.basename(b), shots=int(len(det)), num_detectors=int(nd), n_err=len(dec.errs),
           max_frontier=dec.max_frontier, sec_per_shot=dt / len(det),
           nan_mine=int((~np.isfinite(post)).sum()), nan_theirs=int((~np.isfinite(their_post)).sum()),
           max_abs_dpost=float(np.max(np.abs(post[ok] - their_post[ok]))),
           max_abs_dlogp=float(np.max(np.abs(logp[ok] - their_logp[ok]))),
           decisions_differ=int(((post[ok] > 0.5) != (their_post[ok] > 0.5)).sum()),
           near_ties_mine=int((np.abs(post[ok] - 0.5) < 1e-9).sum()),
           ml_fail_mine=int(((post > 0.5) != obs.astype(bool)).sum()),
           ml_fail_theirs=int(((their_post > 0.5) != obs.astype(bool)).sum()))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
