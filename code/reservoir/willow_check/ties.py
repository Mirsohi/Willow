"""Near-tie shots of the exact ML decoder: |llr| below a threshold, where P(flip | syndrome) is 1/2 to within rounding and
either prediction is maximum-likelihood. Counts how often ML and Libra are wrong on them, and their share of the
discordant pairs behind g (P1 scope: d = 3, 18 series, the 13 Libra round counts).
Usage: nice -n 19 python3 -I ties.py ARCHIVE_ROOT DECODE_ROOT OUT_DIR   -> OUT_DIR/ties.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True  # no __pycache__ in the shared code folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from willow_check import BASES, LIBRA, PATCHES, ROUNDS, out_path, read_obs_b8  # noqa: E402

THRESHOLDS = [1e-9, 1e-6, 1e-3, 1e-2]


def main(arch, dec, out_dir):
    acc = {t: {"shots": 0, "ml_wrong": 0, "libra_wrong": 0, "n10": 0, "n01": 0} for t in THRESHOLDS}
    smallest = []
    for p in PATCHES:
        for b in BASES:
            for r in ROUNDS:
                rel = os.path.join(f"d3_at_{p}", b, f"r{r:02d}")
                act, _ = read_obs_b8(os.path.join(arch, rel, "obs_flips_actual.b8"))
                lib, _ = read_obs_b8(os.path.join(arch, rel, "decoding_results", LIBRA, "obs_flips_predicted.b8"))
                ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
                idx = ml["idx"].astype(np.int64)
                pred, llr = ml["pred"].reshape(len(idx), -1)[:, 0], np.abs(ml["llr"].astype(np.float64))
                mw, lw = pred != act[idx], lib[idx] != act[idx]
                smallest.extend(np.sort(llr)[:3].tolist())
                for t in THRESHOLDS:
                    m = llr < t
                    a = acc[t]
                    a["shots"] += int(m.sum())
                    a["ml_wrong"] += int(np.sum(mw & m))
                    a["libra_wrong"] += int(np.sum(lw & m))
                    a["n10"] += int(np.sum(lw & ~mw & m))
                    a["n01"] += int(np.sum(~lw & mw & m))
    out = {"by_threshold": {f"|llr|<{t:g}": v for t, v in acc.items()},
           "smallest_abs_llr": sorted(smallest)[:10],
           "note": "n10 = Libra wrong, ML right; n01 = Libra right, ML wrong (on the near-tie shots only)"}
    with open(out_path(os.path.join(out_dir, "ties.json")), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
