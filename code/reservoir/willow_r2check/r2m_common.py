"""Shared helpers for reservoir's independent check of round 2's mech fits, T-hat_mech and D-hat_mech
(notes/decoder/willow_prereg_r2.md section 8: amendment A1, with 8.4). reservoir, 2026-10-10; written before any real
mech fit existed, to be run when main asks.

The family is rebuilt here from PR2 section 8.2; nothing in code/decoder/ is imported. On the shipped RL DEM of each
leaf (the faults of r2_common.leaf_model: duplicates merged, T-hat's class per fault rebuilt by p4_check.classes):
  logit p_j = logit p0_j + delta_t(j), clipped to [1e-15, 0.5 - 1e-9] as F1's model (faults left at the RL value
  pass through the same sigmoid and clip);
  template t(j) = (layer tag, observable flag, geometry): tag "first" if fault j touches detector layer 0, else
  "last" if it touches the final layer, else "bulk" (layers from the third detector coordinate; first / last = the
  smallest / largest over the leaf's detectors); geometry = the sorted (x, y, t - t_min) of its detectors from their
  first three coordinates, rounded to 1e-6;
  one delta per template, shared by every round and by the 4 W5 leaves of a series;
  rare templates (fewer than 30 expected fault events in the series' fit sample: sum of p0 x 5,000 over their faults in
  the 4 leaves) share one delta per F1 (T-hat) class name, "R:<class>"; a group still under 30 joins "R:rare"; if that
  is under 30 too, its faults stay at the RL value (no parameter).
Parameter names and the fingerprint follow the conventions of decoder's mech_fit.py and cond_fit.py (read to learn the
format only): "<tag>|L<flag>|x,y,t;x,y,t" with %g numbers, "R:<class>" or "R:rare", sorted as strings; fingerprint =
md5 over (leaf name, fit-shot detection events as uint8, fit-shot observables as int64) for the 4 leaves in W5 order,
then json.dumps of the parameter names, followed by "|<objective>|<lambda>|info=<opg|expected>|secant=<secant>".
D-hat_mech's start (PR2 section 8.2): the mean of ln theta-hat (F1, the series' fit_series.json) over each parameter's
faults in the 4 leaves.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

import numpy as np
import stim

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (CLIP_HI, CLIP_LO, SHOTS, W5, dem_path, detection_events, leaf_model,  # noqa: E402,F401
                       read_b8_bit0, rel_of, series_of)

MIN_EVENTS = 30.0
LAMBDA_D = 4.0
N_FIT = 5000


def template_key(co, ds, flag, t_first, t_last):
    xyz = [tuple(float(v) for v in co[d][:3]) for d in ds]
    if not xyz or any(len(c) != 3 for c in xyz):
        raise SystemExit(f"fault without three detector coordinates: {ds}")
    ts = [c[2] for c in xyz]
    tag = "first" if abs(min(ts) - t_first) < 1e-9 else ("last" if abs(max(ts) - t_last) < 1e-9 else "bulk")
    tm = min(ts)
    geo = tuple(sorted((round(c[0], 6), round(c[1], 6), round(c[2] - tm, 6)) for c in xyz))
    return (tag, int(flag != 0), geo)


def key_name(k):
    return f"{k[0]}|L{k[1]}|" + ";".join(f"{x:g},{y:g},{t:g}" for x, y, t in k[2])


def series_models(arch, fit, patch, basis):
    """The 4 leaves of a series: leaf_model (faults, classes, layers) plus the detector coordinates and fit.json."""
    out = []
    for r in W5:
        rel = rel_of(patch, basis, r)
        fj = json.load(open(os.path.join(fit, rel, "rl_tsplit", "fit.json")))
        M = leaf_model(arch, rel, r, fj["n_fit"], float(fj["fit_attempts"][-1]["min_class_events"]))
        dem = stim.DetectorErrorModel.from_file(dem_path(arch, rel))
        M.update({"rel": rel, "r": r, "co": dem.get_detector_coordinates(), "fit_json": fj,
                  "p0": np.array([p for p, _, _ in M["mech"]], float)})
        out.append(M)
    return out


def mech_family(models, n_fit=N_FIT):
    keys, ev = [], collections.defaultdict(float)
    for M in models:
        ts = [c[2] for c in M["co"].values() if len(c) >= 3]
        t_first, t_last = min(ts), max(ts)
        ks = [template_key(M["co"], ds, l, t_first, t_last) for _, ds, l in M["mech"]]
        keys.append(ks)
        for k, p in zip(ks, M["p0"]):
            ev[k] += p * n_fit
    big = {k for k, v in ev.items() if v >= MIN_EVENTS}
    pname = [[key_name(k) if k in big else "R:" + str(c) for k, c in zip(ks, M["cls"])] for ks, M in zip(keys, models)]
    gev = collections.defaultdict(float)
    for names, M in zip(pname, models):
        for nm, p in zip(names, M["p0"]):
            if nm.startswith("R:"):
                gev[nm] += p * n_fit
    small = {nm for nm, v in gev.items() if v < MIN_EVENTS}
    pname = [["R:rare" if nm in small else nm for nm in names] for names in pname]
    rare_ev = sum(v for nm, v in gev.items() if nm in small)
    fixed_rare = bool(small) and rare_ev < MIN_EVENTS
    if fixed_rare:
        pname = [[None if nm == "R:rare" else nm for nm in names] for names in pname]
    names = sorted({nm for nms in pname for nm in nms if nm is not None})
    index = {nm: i for i, nm in enumerate(names)}
    idx = [np.array([index[nm] if nm is not None else -1 for nm in nms], np.int64) for nms in pname]
    info = {"templates": len(ev), "templates_ge_30_events": len(big), "rare_groups": sorted(gev),
            "rare_groups_merged": sorted(small), "rare_fixed_at_0": bool(fixed_rare), "parameters": len(names),
            "template_flags_L": int(sum(1 for k in ev if k[1]))}
    return {"names": names, "idx": idx, "info": info, "keys": keys, "template_events": ev}


def mech_faults(M, idx, delta):
    """The leaf's faults under the shifts delta: [(p, detectors, flips L)] for layered_ml, and the probabilities."""
    p0 = M["p0"]
    d = np.where(idx >= 0, np.asarray(delta, float)[np.maximum(idx, 0)], 0.0)
    lo = np.log(p0) - np.log1p(-p0) + d
    p = np.clip(1.0 / (1.0 + np.exp(-lo)), CLIP_LO, CLIP_HI)
    return [(float(q), ds, l) for q, (_, ds, l) in zip(p, M["mech"])], p


def start_from_f1(models, fam, f1_names, phi_hat):
    m = dict(zip(f1_names, phi_hat))
    s = np.zeros(len(fam["names"]))
    c = np.zeros(len(fam["names"]))
    for M, ix in zip(models, fam["idx"]):
        for i, cl in zip(ix, M["cls"]):
            if i >= 0 and cl in m:
                s[i] += m[cl]
                c[i] += 1
    return np.where(c > 0, s / np.maximum(c, 1), 0.0)


def fit_shots(arch, fit, M):
    """The leaf's fit shots (round 1's fit_idx) as F1's and the mech fits' Leaf stores them: events uint8, L int64."""
    fd = np.load(os.path.join(fit, M["rel"], "rl_tsplit", "fit_data.npz"), allow_pickle=False)
    fi = fd["fit_idx"].astype(np.int64)
    det = np.ascontiguousarray(detection_events(arch, M["rel"], M["nd"], fi), dtype=np.uint8)
    obs = read_b8_bit0(os.path.join(arch, M["rel"], "obs_flips_actual.b8"), SHOTS)
    return fi, det, np.asarray(obs[fi], dtype=np.int64).reshape(-1)


def fit_suffix(fj):
    info = "expected" if str(fj.get("information", "")).startswith("conditional") else "opg"
    return f"|{fj['objective_type']}|{fj['lambda']}|info={info}|secant={fj['secant']}"


def fingerprint(lnames, dets, Ls, names):
    h = hashlib.md5()
    for ln, det, L in zip(lnames, dets, Ls):
        h.update(ln.encode())
        h.update(det.tobytes())
        h.update(L.tobytes())
    h.update(json.dumps(list(names)).encode())
    return h.hexdigest()


def leaf_name(patch, basis, r):
    return f"{series_of(patch, basis)}/r{r}"
