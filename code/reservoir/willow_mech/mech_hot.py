"""EXPLORATORY (not pre-registered), extra to task 3: what changes in the hot blocks of q8_5 Z r = 30?

In that leaf, blocks 8 and 9 have about 30% more detection events than blocks 0-7, yet the ML(RL) calibration ratio
does not rise (mech_summary.py, blocks.csv). This script asks where the extra events sit. For the two block groups it
gives: the event rate of each stabilizer in the bulk layers (1..r-1) and in the first and last layers; and the pair
probability p_ij (the estimator of mech_pairs.py) summed by class over the detector pairs that share a DEM mechanism,
in total and per stabilizer for the timelike pairs (same stabilizer, adjacent rounds: measurement-like faults).
Block order is file order; whether it is acquisition order is not known [U].
Usage: nice -n 19 python3 -I mech_hot.py ARCHIVE OUT_DIR [PATCH BASIS R HOT_BLOCKS]
       defaults q8_5 Z 30 8,9; writes OUT_DIR/hot_<patch>_<basis>_r<NNN>.json through qrc.outpaths
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import stim

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from mech_pairs import RL, moments, pij  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

LABEL = "EXPLORATORY, not pre-registered (reservoir, 2026-10-09); no multiplicity correction"
BLOCK = 5000


def main(arch, out_dir, patch="q8_5", basis="Z", r=30, hot=(8, 9)):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir = os.path.join(arch, rel)
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", RL, "error_model.dem"))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    xy = [tuple(float(v) for v in co[i][:2]) for i in range(nd)]
    t = np.array([int(co[i][2]) for i in range(nd)])
    pairs = set()
    for ins in dem.flattened():
        if ins.type == "error":
            ds = sorted({x.val for x in ins.targets_copy() if x.is_relative_detector_id()})
            pairs.update((ds[a], ds[b]) for a in range(len(ds)) for b in range(a + 1, len(ds)))
    P = np.array(sorted(pairs))
    I, J = P[:, 0], P[:, 1]
    cls = np.array(["boundary_first" if t[i] == 0 or t[j] == 0 else "boundary_last" if t[i] == r or t[j] == r
                    else "timelike" if xy[i] == xy[j] and abs(t[i] - t[j]) == 1 else "spacelike" if t[i] == t[j]
                    else "diagonal" if abs(t[i] - t[j]) == 1 else "other" for i, j in P])
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(-1, nd // 8)
    D = np.unpackbits(raw, axis=1, bitorder="little")[:, :nd]
    blk = np.arange(D.shape[0]) // BLOCK
    groups = {"other_blocks": ~np.isin(blk, hot), "hot_blocks": np.isin(blk, hot)}
    stabs = sorted({xy[i] for i in range(nd) if 0 < t[i] < r})
    res = {"label": LABEL, "leaf": f"{patch} {basis} r{r}", "hot_blocks": list(hot),
           "note": "block order is file order; whether it is acquisition order is not known [U]", "groups": {}}
    for g, sel in groups.items():
        m1, m2 = moments(D[sel])
        p = pij(m1, m2, I, J)
        o = {"shots": int(sel.sum()), "mean_events": float(D[sel].sum(1).mean()),
             "rate_first": float(m1[t == 0].mean()), "rate_last": float(m1[t == r].mean()),
             "rate_bulk": float(m1[(t > 0) & (t < r)].mean()),
             "sum_p_by_class": {c: float(p[cls == c].sum()) for c in sorted(set(cls))},
             "bulk_rate_by_stabilizer": {}, "timelike_p_by_stabilizer": {}}
        for s in stabs:
            k = f"{s[0]:g},{s[1]:g}"
            o["bulk_rate_by_stabilizer"][k] = float(m1[[i for i in range(nd) if xy[i] == s and 0 < t[i] < r]].mean())
            tl = (cls == "timelike") & np.array([xy[i] == s for i in I])
            o["timelike_p_by_stabilizer"][k] = float(p[tl].mean())
        res["groups"][g] = o
    a, b = res["groups"]["other_blocks"], res["groups"]["hot_blocks"]
    res["hot_over_other"] = {
        "mean_events": b["mean_events"] / a["mean_events"],
        "rate_first": b["rate_first"] / a["rate_first"], "rate_last": b["rate_last"] / a["rate_last"],
        "rate_bulk": b["rate_bulk"] / a["rate_bulk"],
        "sum_p_by_class": {c: b["sum_p_by_class"][c] / a["sum_p_by_class"][c] for c in a["sum_p_by_class"]},
        "bulk_rate_by_stabilizer": {k: b["bulk_rate_by_stabilizer"][k] / a["bulk_rate_by_stabilizer"][k]
                                    for k in a["bulk_rate_by_stabilizer"]},
        "timelike_p_by_stabilizer": {k: b["timelike_p_by_stabilizer"][k] / a["timelike_p_by_stabilizer"][k]
                                     for k in a["timelike_p_by_stabilizer"]}}
    with open(out_path(os.path.join(out_dir, f"hot_{patch}_{basis}_r{r:03d}.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["hot_over_other"], indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 2:
        main(a[0], a[1], a[2], a[3], int(a[4]), tuple(int(x) for x in a[5].split(",")))
    else:
        main(a[0], a[1])
