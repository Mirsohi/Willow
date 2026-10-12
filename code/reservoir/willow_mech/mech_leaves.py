"""EXPLORATORY (not pre-registered): why is the RL DEM miscalibrated at d = 3 (P3)? Step 1, per-leaf extraction.

For each d = 3 leaf this writes one compressed npz to a scratch folder (OUT, outside the project tree):
  per shot (data, 50,000 shots, in file order):
    N     detection events in the shot
    B     largest number of events in one detector layer (all layers; layers 0 and r have 4 detectors, the rest 8)
    Bb    the same over the bulk layers 1..r-1 only
    Bt    the layer of the first maximum
    F, L  events in the first (t = 0) and the last (t = r) detector layer
    NA, NB  events in two windows of bulk layers, the first and the last third (1..w and r-w..r-1, w = floor((r-1)/3));
          the gap between them is wider than any fault of the DEM reaches, so under the DEM Cov(NA, NB) = 0 exactly
    m     min(P, 1 - P) from decoder's exact ML posterior (P = sigmoid(llr)); fail_ml: ML(RL) prediction != actual flip
    logp  ln P(s) under the RL DEM (decoder's exact ML); fail_libra: Libra wrong (-1 where Libra has no predictions)
    block shot // 5000 (the 10 blocks of one initialisation bitstring each, checked against sweep_bits.b8)
  per block: the 9-bit initialisation code and its Hamming weight
  DEM samples (stim, seeded by the leaf name): histograms of N, B, Bb, F, L; moments of NA and NB; mean events per layer
  data: mean events per layer
Detector layers come from the DEM's coordinates (third coordinate = round); the script checks that the layers are
contiguous in detector order with sizes 4, 8, ..., 8, 4, which lets it count events per layer from the packed bytes.

Inputs (read only; run with python3 -I because the archive is downloaded data):
  ARCHIVE/d3_at_<patch>/<basis>/r<NN>/{detection_events.b8, obs_flips_actual.b8, sweep_bits.b8,
      decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem (the RL DEM),
      decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8}
  DECODE/d3_at_<patch>/<basis>/r<NN>/rl/{ml_exact.npz, summary.json}   (decoder's exact ML with the RL DEM)
Usage: nice -n 19 python3 -I mech_leaves.py ARCHIVE DECODE OUT [w5|rest|all] [N_DEM_W5] [N_DEM_REST]
       defaults: all, 10^6 DEM samples per W5 leaf (r in {10, 13, 30, 50}), 10^5 for the other leaves.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib

import numpy as np
import stim

sys.dont_write_bytecode = True

PATCHES = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
BASES = ["X", "Z"]
ROUNDS = [1, 10, 13] + list(range(30, 251, 20))
W5 = [10, 13, 30, 50]
RL = "correlated_matching_decoder_with_rl_optimized_prior"
LIBRA = "libra_decoder_with_rl_optimized_prior"
SHOTS, BLOCK = 50000, 5000
LO = np.array([bin(i & 15).count("1") for i in range(256)], np.uint8)
HI = np.array([bin(i >> 4).count("1") for i in range(256)], np.uint8)
CHUNK = 50000


def layer_counts(packed, r):
    """Events per detector layer from b8-packed rows: layer 0 = bits 0-3 of byte 0, layer t (1 <= t < r) = bits 4-7 of
    byte t-1 and bits 0-3 of byte t, layer r = bits 4-7 of byte r-1."""
    lo, hi = LO[packed], HI[packed]
    C = np.empty((packed.shape[0], r + 1), np.int16)
    C[:, 0] = lo[:, 0]
    if r > 1:
        C[:, 1:r] = hi[:, : r - 1].astype(np.int16) + lo[:, 1:r]
    C[:, r] = hi[:, r - 1]
    return C


def windows(r):
    w = (r - 1) // 3
    return ((1, w), (r - w, r - 1)) if w >= 1 else None


def features(C, r, win):
    f = {"N": C.sum(1, dtype=np.int32), "B": C.max(1).astype(np.int8), "Bt": C.argmax(1).astype(np.int16),
         "F": C[:, 0].astype(np.int8), "L": C[:, r].astype(np.int8)}
    f["Bb"] = C[:, 1:r].max(1).astype(np.int8) if r > 1 else np.zeros(len(C), np.int8)
    if win:
        (a0, a1), (b0, b1) = win
        f["NA"] = C[:, a0:a1 + 1].sum(1, dtype=np.int32)
        f["NB"] = C[:, b0:b1 + 1].sum(1, dtype=np.int32)
    else:
        f["NA"] = f["NB"] = np.zeros(len(C), np.int32)
    return f


def check_layers(dem, r):
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t = np.array([co[i][2] for i in range(nd)], float)
    if nd != 8 * r or not np.all(np.diff(t) >= 0) or not np.all(t == np.round(t)):
        raise SystemExit(f"unexpected detector layout: {nd} detectors at r = {r}")
    sizes = np.bincount(t.astype(int), minlength=r + 1)
    expect = np.array([4] + [8] * (r - 1) + [4]) if r > 1 else np.array([4, 4])
    if len(sizes) != r + 1 or not np.array_equal(sizes, expect):
        raise SystemExit(f"unexpected layer sizes at r = {r}")
    span = 0
    for ins in dem.flattened():
        if ins.type == "error":
            ts = [t[x.val] for x in ins.targets_copy() if x.is_relative_detector_id()]
            if ts:
                span = max(span, int(max(ts) - min(ts)))
    return t.astype(int), span


def leaf(arch, dec, out, patch, basis, r, n_dem):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir, d_dir = os.path.join(arch, rel), os.path.join(dec, rel, "rl")
    dem_path = os.path.join(a_dir, "decoding_results", RL, "error_model.dem")
    dem = stim.DetectorErrorModel.from_file(dem_path)
    t, span = check_layers(dem, r)
    win = windows(r)
    if win and win[1][0] - win[0][1] <= span:
        raise SystemExit(f"window gap too small at r = {r} (DEM span {span})")
    # data
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8)
    if raw.size != SHOTS * r:
        raise SystemExit(f"detection_events.b8 size {raw.size} at {rel}")
    C = layer_counts(raw.reshape(SHOTS, r), r)
    fd = features(C, r, win)
    layer_mean_data = C.mean(0)
    obs_raw = np.fromfile(os.path.join(a_dir, "obs_flips_actual.b8"), dtype=np.uint8)
    if obs_raw.size != SHOTS or np.any(obs_raw >> 1):
        raise SystemExit(f"obs_flips_actual.b8 unexpected at {rel}")
    obs = obs_raw & 1
    ml = np.load(os.path.join(d_dir, "ml_exact.npz"), allow_pickle=False)
    if not np.array_equal(ml["idx"], np.arange(SHOTS)):
        raise SystemExit(f"ML shots not all in order at {rel}")
    llr = ml["llr"].astype(np.float64)
    pred = ml["pred"].reshape(SHOTS, -1)[:, 0]
    if not np.array_equal(pred, (llr > 0).astype(pred.dtype)):
        raise SystemExit(f"pred != argmax at {rel}")
    e = np.exp(-np.abs(llr))
    m = e / (1 + e)
    fail_ml = pred != obs
    lp = os.path.join(a_dir, "decoding_results", LIBRA, "obs_flips_predicted.b8")
    fail_libra = ((np.fromfile(lp, dtype=np.uint8) & 1) != obs).astype(np.int8) if os.path.exists(lp) \
        else np.full(SHOTS, -1, np.int8)
    sw = np.fromfile(os.path.join(a_dir, "sweep_bits.b8"), dtype=np.uint8).reshape(SHOTS, -1)
    code = np.unpackbits(sw, axis=1, bitorder="little")[:, :9].astype(np.int64) @ (1 << np.arange(9))
    blocks = code.reshape(SHOTS // BLOCK, BLOCK)
    if not np.all(blocks == blocks[:, :1]):
        raise SystemExit(f"initialisation code not constant within blocks at {rel}")
    bcode = blocks[:, 0]
    if not np.all((bcode[0::2] ^ bcode[1::2]) == 511):
        raise SystemExit(f"blocks 2k, 2k+1 not complementary at {rel}")
    summ = json.load(open(os.path.join(d_dir, "summary.json")))
    # DEM samples
    seed = zlib.crc32(rel.encode())
    sampler = dem.compile_sampler(seed=seed)
    hist = {k: np.zeros(n, np.int64) for k, n in (("N", 8 * r + 1), ("B", 9), ("Bb", 9), ("F", 5), ("L", 5))}
    mom = np.zeros(6)  # sum NA, sum NB, sum NA^2, sum NB^2, sum NA NB, n
    layer_sum = np.zeros(r + 1)
    left = n_dem
    while left:
        k = min(CHUNK, left)
        det, _, _ = sampler.sample(shots=k, bit_packed=True)
        Cs = layer_counts(det, r)
        fs = features(Cs, r, win)
        for key in hist:
            hist[key] += np.bincount(fs[key], minlength=len(hist[key]))[: len(hist[key])]
        A, Bw = fs["NA"].astype(np.float64), fs["NB"].astype(np.float64)
        mom += [A.sum(), Bw.sum(), (A * A).sum(), (Bw * Bw).sum(), (A * Bw).sum(), k]
        layer_sum += Cs.sum(0)
        left -= k
    meta = {"patch": patch, "basis": basis, "r": r, "shots": SHOTS, "n_dem": n_dem, "seed": seed,
            "dem_num_errors": dem.num_errors, "dem_max_layer_span": span, "windows": win,
            "decoder_p_det": summ.get("p_det"), "decoder_dem_md5": summ.get("dem_md5"),
            "libra": bool(os.path.exists(lp)), "stim": stim.__version__, "numpy": np.__version__}
    np.savez_compressed(os.path.join(out, f"{patch}_{basis}_r{r:03d}.npz"),
                        meta=json.dumps(meta), N=fd["N"].astype(np.int16), B=fd["B"], Bb=fd["Bb"], Bt=fd["Bt"],
                        F=fd["F"], L=fd["L"], NA=fd["NA"].astype(np.int16), NB=fd["NB"].astype(np.int16),
                        m=m, fail_ml=fail_ml, logp=ml["logp"].astype(np.float64), fail_libra=fail_libra,
                        block=(np.arange(SHOTS) // BLOCK).astype(np.int8), block_code=bcode,
                        block_weight=np.array([bin(int(c)).count("1") for c in bcode]),
                        layer_mean_data=layer_mean_data, layer_mean_dem=layer_sum / n_dem,
                        dem_hist_N=hist["N"], dem_hist_B=hist["B"], dem_hist_Bb=hist["Bb"], dem_hist_F=hist["F"],
                        dem_hist_L=hist["L"], dem_mom_window=mom)
    return float(fd["N"].mean()), float((layer_sum / n_dem).sum()), summ.get("p_det")


def main(arch, dec, out, which="all", n_w5=1_000_000, n_rest=100_000):
    os.makedirs(out, exist_ok=True)
    todo = []
    for r in W5 + [x for x in ROUNDS if x not in W5]:
        if (which == "w5" and r not in W5) or (which == "rest" and r in W5):
            continue
        for p in PATCHES:
            for b in BASES:
                todo.append((p, b, r))
    t0 = time.time()
    for i, (p, b, r) in enumerate(todo):
        f = os.path.join(out, f"{p}_{b}_r{r:03d}.npz")
        if os.path.exists(f):
            continue
        md, mm, pdet = leaf(arch, dec, out, p, b, r, n_w5 if r in W5 else n_rest)
        print(f"{i + 1}/{len(todo)} {p} {b} r={r}: mean events data {md:.3f} (decoder p_det x 8r "
              f"{(pdet or 0) * 8 * r:.3f}), DEM {mm:.3f}; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else "all", int(a[4]) if len(a) > 4 else 1_000_000,
         int(a[5]) if len(a) > 5 else 100_000)
