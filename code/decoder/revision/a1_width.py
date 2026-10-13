#!/usr/bin/env python3
"""A1 / B10: the frontier width W of our exact elimination order at d = 5 (Willow RL DEMs, r = 10 and 13), and what a
dense exact sweep would need: memory 8 x 2^(W + n_obs) bytes, and work sum_k 2^(open_k + n_obs) cell updates per shot
(open_k = open detectors when fault k is applied). Calibrated against d = 3 (ml_exact_fast's measured speed).
No decoding; reads DEM files only (decoder, 2026-10-13).
usage: python3 -I -B a1_width.py --code <code/decoder> --root <extract> --out OUT.json DEM_REL [DEM_REL ...]
"""
import argparse, hashlib, json, os, sys, time


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("dems", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.code))
    import numpy as np
    import stim
    from dem_utils import parse_dem, best_sweep_order, frontier_profile
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "dem_utils_md5": md5(os.path.join(a.code, "dem_utils.py")), "files": []}
    for rel in a.dems:
        p = os.path.join(a.root, rel)
        t0 = time.process_time()
        dm = parse_dem(stim.DetectorErrorModel.from_file(p))
        order, mx, mean, name = best_sweep_order(dm)
        _, _, cnt = frontier_profile(dm, order)
        nobs = int(dm.num_observables)
        work = float(np.sum(np.power(2.0, cnt.astype(np.float64) + nobs)))
        top = np.sort(cnt)[::-1]
        out["files"].append({"dem": rel, "md5": md5(p), "detectors": int(dm.num_detectors), "faults": int(len(order)),
                             "n_obs": nobs, "W": int(mx), "mean_open": mean, "sweep": name,
                             "steps_at_W": int((cnt == mx).sum()), "steps_at_W_minus_1": int((cnt == mx - 1).sum()),
                             "dense_table_bytes": 8.0 * 2 ** (mx + nobs), "work_cell_updates_per_shot": work,
                             "cpu_s": time.process_time() - t0})
        f = out["files"][-1]
        print(f"{rel}: det {f['detectors']} faults {f['faults']} W {f['W']} mean {f['mean_open']:.1f} "
              f"table {f['dense_table_bytes']/1e9:.3g} GB work {f['work_cell_updates_per_shot']:.3g} ({f['cpu_s']:.1f} s)", flush=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
