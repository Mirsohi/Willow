"""Shared helpers for reservoir's independent check of round 2's F1 result (decoder's D-hat).

reservoir, 2026-10-10. Written from notes/decoder/willow_prereg_r2.md sections 2-4 and the stored outputs. Nothing in
code/decoder/ is imported; the model is rebuilt here:
  - classes: T-hat's per-fault class labels, rebuilt by willow_check/p4_check.classes (equal to every fit.json's class
    record in the P4 check);
  - D-hat's DEM for a leaf: the shipped RL DEM with each fault's probability multiplied by the series' scale of its
    class name, clipped to [1e-15, 0.5 - 1e-9] as F1's model does (code/decoder/fit_noise.py DemClassScale.probs, read
    to learn the convention); faults with identical (detectors, observable) would be merged before scaling, as
    decoder's parse_dem does (counted; none exist in the 72 RL DEMs if the counts below are 0);
  - exact ML: willow_check/p4_redecode.layered_ml (an exact layer-by-layer sum over fault configurations).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import stim

sys.dont_write_bytecode = True  # no __pycache__ in the shared code folder
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_check"))
sys.path.insert(0, os.path.join(HERE, ".."))  # code/reservoir (qrc.outpaths)
from p4_check import BASES, CM_RL, PATCHES, SHOTS, W5, block_codes, classes, ratio_se, read_b8_bit0  # noqa: E402,F401
from p4_redecode import layered_ml  # noqa: E402,F401

ARCH = "google_105Q_surface_code_d3_d5_d7"
CLIP_LO, CLIP_HI = 1e-15, 0.5 - 1e-9
CONF = 3e-3  # R_conf threshold on min(P, 1 - P) (round 1's P3 definition)


def rel_of(patch, basis, r):
    return f"d3_at_{patch}/{basis}/r{r:02d}"


def series_of(patch, basis):
    return f"{ARCH}/d3_at_{patch}/{basis}"


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def dem_path(arch, rel):
    return os.path.join(arch, rel, "decoding_results", CM_RL, "error_model.dem")


def leaf_model(arch, rel, r, n_fit, thr):
    """The RL DEM of a leaf as (p, detectors, flips L) faults with T-hat's class label per fault, the detector layers,
    and counts of merged duplicates and zero-probability faults."""
    dem = stim.DetectorErrorModel.from_file(dem_path(arch, rel))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    mech, _, cls, _ = classes(dem, r, n_fit, thr)
    pos, mm, cc = {}, [], []
    n_dup = n_zero = n_empty = 0
    for (p, ds, l), c in zip(mech, cls):
        if p == 0:
            n_zero += 1
            continue
        if not ds:
            n_empty += 1  # an undetectable fault: decoder's F1 asserts there are none
            continue
        k = (ds, l)
        if k in pos:
            n_dup += 1
            i = pos[k]
            q = mm[i][0]
            if cc[i] != c:
                raise SystemExit(f"{rel}: duplicate symptom {k} in two classes")
            mm[i] = (q + p - 2 * q * p, ds, l)
        else:
            pos[k] = len(mm)
            mm.append((p, ds, l))
            cc.append(c)
    return {"nd": nd, "t_of": t_of, "mech": mm, "cls": cc, "n_dup": n_dup, "n_zero": n_zero, "n_empty": n_empty,
            "names": sorted(set(cc))}


def scaled(mech, cls, theta_by_name):
    """D-hat's faults: p0 * theta(class), clipped as F1's model; also the number of clipped faults."""
    out, nclip = [], 0
    for (p, ds, l), c in zip(mech, cls):
        q = p * theta_by_name[c]
        qc = min(max(q, CLIP_LO), CLIP_HI)
        nclip += int(qc != q)
        out.append((qc, ds, l))
    return out, nclip


def det_marginals(mech, nd):
    """P(detector fires) per detector under independent faults: (1 - prod(1 - 2p)) / 2."""
    acc = np.zeros(nd)  # sum of ln|1 - 2p|
    for p, ds, _ in mech:
        v = np.log1p(-2.0 * p)
        for d in ds:
            acc[d] += v
    return (1.0 - np.exp(acc)) / 2.0


def detection_events(arch, rel, nd, idx=None):
    raw = np.fromfile(os.path.join(arch, rel, "detection_events.b8"), dtype=np.uint8).reshape(SHOTS, -1)
    if idx is not None:
        raw = raw[idx]
    return np.ascontiguousarray(np.unpackbits(raw, axis=1, bitorder="little")[:, :nd])


def split_rank(arch, rel):
    """Rank of each shot within its initialisation bitstring (the 9 sweep bits); odd rank = held out (P4's split)."""
    code = block_codes(os.path.join(arch, rel, "sweep_bits.b8"), SHOTS)
    rank = np.zeros(SHOTS, np.int64)
    for c in np.unique(code):
        pos = np.flatnonzero(code == c)
        rank[pos] = np.arange(pos.size)
    return rank


def nll_from_llr(llr, obs):
    """-ln P(L_obs | s) from llr = ln P(1|s) - ln P(0|s); a NaN llr counts as P = 1/2."""
    x = np.nan_to_num(np.asarray(llr, np.float64), nan=0.0)
    s = np.where(np.asarray(obs).astype(bool), 1.0, -1.0)
    return np.logaddexp(0.0, -s * x)


def min_p_from_llr(llr):
    """min(P, 1 - P) from the llr; NaN counts as 1/2."""
    x = np.abs(np.nan_to_num(np.asarray(llr, np.float64), nan=0.0))
    return np.exp(-np.logaddexp(0.0, x))


def series_fit(f1, patch, basis):
    return json.load(open(os.path.join(f1, f"d3_at_{patch}", basis, "fit_series.json")))


def theta_by_name(fs):
    return {nm: float(np.exp(ph)) for nm, ph in zip(fs["names"], fs["phi_hat"])}
