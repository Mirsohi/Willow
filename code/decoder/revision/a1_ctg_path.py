#!/usr/bin/env python3
"""A1 / B10: cotengra contraction-tree search (no contraction, no decoding) for Willow d = 5 RL DEMs, to estimate the
cost of an exact tree contraction per shot: width (log2 of the largest intermediate) and cost (cotengra's
contraction_cost, scalar operations). Uses ml_cotengra.CotengraExact's network and search settings unchanged
(parallel=False). decoder, 2026-10-13.
usage: python3 -I -B a1_ctg_path.py --code <code/decoder> --root <extract> --max-time S --out OUT.json DEM_REL [...]
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
    ap.add_argument("--max-time", type=float, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("dems", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.code))
    import math
    import stim
    from dem_utils import parse_dem
    from ml_cotengra import CotengraExact
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "ml_cotengra_md5": md5(os.path.join(a.code, "ml_cotengra.py")), "max_time_s": a.max_time, "files": []}
    for rel in a.dems:
        p = os.path.join(a.root, rel)
        t0 = time.process_time(); w0 = time.time()
        dm = parse_dem(stim.DetectorErrorModel.from_file(p))
        C = CotengraExact(dm, max_time=a.max_time)
        tr = C.tree
        rec = {"dem": rel, "md5": md5(p), "tensors": len(C.inputs), "width": float(C.width),
               "largest_intermediate_bytes": 8.0 * 2 ** float(C.width), "cost": float(C.cost),
               "log10_cost": math.log10(float(C.cost)), "cpu_s": time.process_time() - t0, "wall_s": time.time() - w0}
        try:
            rec["peak_size_bytes"] = 8.0 * float(tr.peak_size())
        except Exception as e:                                   # older/newer cotengra: optional
            rec["peak_size_error"] = f"{type(e).__name__}"
        out["files"].append(rec)
        print(json.dumps(rec), flush=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
