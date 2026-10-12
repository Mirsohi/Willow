"""Re-decode held-out shots of one fitted leaf with an exact ML decoder written here, under the RL DEM and under the
fitted DEM T-hat, and compare with decoder's stored posteriors (check (d) of the P4 check, optional item).

reservoir, 2026-10-09. The decoder is independent of code/decoder/: an exact sum over all fault configurations, done
layer by layer. Every fault of these DEMs touches one detector layer t or two adjacent layers t, t + 1 (checked).
Before layer t is closed, a vector W(a, l) holds, for each parity pattern a of layer t's detectors and each observable
parity l, the probability that the faults of the earlier layers reproduce the observed detectors below t. The faults
whose lowest layer is t have a joint parity distribution F_t(a, b, l) over (layer t, layer t + 1, observable), built
exactly by convolving the faults one by one. Closing layer t on the observed pattern s_t gives
    W'(b, l) = sum_{a, l'} W(a XOR s_t, l') F_t(a, b, l XOR l'),
two matrix products per layer for a batch of shots. After the last layer W(l) = P(syndrome, observable flip = l), so
llr = ln P(s, 1) - ln P(s, 0) and ln P(s) = ln(P(s, 0) + P(s, 1)). Each shot's vector is rescaled after every layer.
The fitted DEM is the RL DEM with every fault probability multiplied by its class scale (fit.json scale_hat; classes
rebuilt by p4_check.classes, which reproduces the fit's class record on all 72 leaves).
Usage: nice -n 19 python3 -I p4_redecode.py ARCHIVE FIT OUT_DIR [PATCH BASIS R N_SHOTS]
       defaults q8_5 X 10 2000: every 12th held-out shot, the first N_SHOTS of them
  -> OUT_DIR/p4_redecode.json (through qrc.outpaths)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import stim

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from p4_check import CM_RL, SHOTS, classes  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402


def layered_ml(mech, t_of, r, syn):
    """mech: list of (p, detectors, flips_L); t_of: layer of each detector; syn: (S, n_det) 0/1 detection events.
    Returns llr (ln P(s,1) - ln P(s,0)) and ln P(s) per shot."""
    nd = len(t_of)
    layers = [np.flatnonzero(t_of == t) for t in range(r + 1)]
    pos = np.zeros(nd, np.int64)
    for t in range(r + 1):
        pos[layers[t]] = np.arange(layers[t].size)
    by_lo = [[] for _ in range(r + 1)]
    for p, ds, l in mech:
        ts = sorted({int(t_of[d]) for d in ds})
        if not ts or ts[-1] - ts[0] > 1:
            raise SystemExit(f"fault spans more than two layers or has no detector: {ds}")
        by_lo[ts[0]].append((p, ds, l))
    S = syn.shape[0]
    W = np.zeros((S, 1 << layers[0].size, 2))
    W[:, 0, 0] = 1.0
    logscale = np.zeros(S)
    for t in range(r + 1):
        na = layers[t].size
        nb = layers[t + 1].size if t < r else 0
        F = np.zeros((1 << na, 1 << nb, 2))
        F[0, 0, 0] = 1.0
        ia, ib = np.arange(1 << na), np.arange(1 << nb)
        for p, ds, l in by_lo[t]:
            ma = sum(1 << int(pos[d]) for d in ds if t_of[d] == t)
            mb = sum(1 << int(pos[d]) for d in ds if t_of[d] == t + 1)
            G = F[ia ^ ma][:, ib ^ mb]
            if l:
                G = G[:, :, ::-1]
            F = (1 - p) * F + p * G
        # close layer t on the observed pattern
        s_t = (syn[:, layers[t]].astype(np.int64) << np.arange(na)).sum(1)
        Wt = W[np.arange(S)[:, None], ia[None, :] ^ s_t[:, None], :]  # W(a XOR s_t, l')
        Wn = np.empty((S, 1 << nb, 2))
        Wn[:, :, 0] = Wt[:, :, 0] @ F[:, :, 0] + Wt[:, :, 1] @ F[:, :, 1]
        Wn[:, :, 1] = Wt[:, :, 0] @ F[:, :, 1] + Wt[:, :, 1] @ F[:, :, 0]
        mx = Wn.reshape(S, -1).max(1)
        if np.any(mx <= 0):
            raise SystemExit("a shot has zero probability under the DEM")
        W = Wn / mx[:, None, None]
        logscale += np.log(mx)
    P0, P1 = W[:, 0, 0], W[:, 0, 1]
    return np.log(P1) - np.log(P0), np.log(P0 + P1) + logscale


def main(arch, fit, out_dir, patch="q8_5", basis="X", r=10, n_shots=2000):
    t0 = time.time()
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir = os.path.join(arch, rel)
    f_dir = os.path.join(fit, rel, "rl_tsplit")
    fj = json.load(open(os.path.join(f_dir, "fit.json")))
    fd = np.load(os.path.join(f_dir, "fit_data.npz"), allow_pickle=False)
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    mech, _, cls, _ = classes(dem, r, fj["n_fit"], float(fj["fit_attempts"][-1]["min_class_events"]))
    theta = dict(zip(fj["classes"], fj["scale_hat"]))
    mech_fit = [(theta[c] * p, ds, l) for (p, ds, l), c in zip(mech, cls)]
    sel = np.arange(0, fd["test_idx"].size, 12)[:n_shots]
    shots = fd["test_idx"][sel].astype(np.int64)
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(SHOTS, -1)
    syn = np.unpackbits(raw[shots], axis=1, bitorder="little")[:, :nd]
    res = {"label": "reservoir, P4 check (d): exact ML re-decode with an independent layered decoder, 2026-10-09",
           "leaf": f"{patch} {basis} r{r}", "shots": int(shots.size), "shot_rule": "every 12th held-out shot",
           "max_fault_probability_fitted": float(max(m[0] for m in mech_fit)), "results": {}}
    for name, mm, key in (("RL", mech, "shipped"), ("T_hat", mech_fit, "fitted")):
        t1 = time.time()
        llr, logp = layered_ml(mm, t_of, r, syn)
        ref_llr = fd[f"{key}_llr"][sel].astype(np.float64)
        ref_logp = fd[f"{key}_logp"][sel]
        pred = (llr > 0).astype(np.uint8)
        res["results"][name] = {
            "max_abs_diff_llr": float(np.max(np.abs(llr - ref_llr))),
            "max_rel_diff_llr": float(np.max(np.abs(llr - ref_llr) / np.maximum(1.0, np.abs(ref_llr)))),
            "max_abs_diff_logp": float(np.max(np.abs(logp - ref_logp))),
            "predictions_equal": int(np.sum(pred == fd[f"{key}_pred"][sel])),
            "min_abs_llr": float(np.min(np.abs(llr))), "seconds": time.time() - t1}
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "p4_redecode.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 3:
        main(a[0], a[1], a[2], a[3], a[4], int(a[5]), int(a[6]))
    else:
        main(a[0], a[1], a[2])
