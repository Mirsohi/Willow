"""Reference values for the per-class scale fits (run_fit.py "<S>+dem", "<S>+demt" with time-split classes): for each symptom class c of the decoding
model's DEM M, the ratio of class totals sum_{j in c} p_j(T) / sum_{j in c} p_j(M), where p_j(T) is the truth's
probability for the same symptom (0 if T has no such fault), and the spread of the per-fault ratios inside the class.
If the per-fault ratio were constant inside every class, the class-scale family would contain T and the fit should
return these ratios; the spread shows how far T is from the family.
Usage: python3 dem_class_truth.py [--res ../../results/decoder]  -> <res>/fit/dem_class_truth.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from dem_utils import parse_dem
from mismatch import si_family, graphlike_dm
from fit_noise import symptom_class


def class_ratios(dmM, dmT, time_split=False):
    pT = {(d, int(o)): p for d, o, p in zip(dmT.dets, dmT.obs, dmT.p)}
    out = {}
    cls = np.array([symptom_class(dmM, j, time_split=time_split) for j in range(dmM.num_faults)])
    matched = 0.0
    for c in sorted(set(cls)):
        idx = np.where(cls == c)[0]
        pm = dmM.p[idx]
        pt = np.array([pT.get((dmM.dets[j], int(dmM.obs[j])), 0.0) for j in idx])
        matched += pt.sum()
        r = pt / pm
        out[c] = {"n_faults": int(len(idx)), "ratio_of_totals": float(pt.sum() / pm.sum()),
                  "per_fault_ratio_min": float(r.min()), "per_fault_ratio_max": float(r.max()),
                  "per_fault_ratio_weighted_sd": float(np.sqrt(np.average((r - r.mean()) ** 2, weights=pm)))}
    out["_T_mass_not_representable"] = float(dmT.p.sum() - matched)
    out["_T_mass_total"] = float(dmT.p.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "decoder"))
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args()
    p = a.p
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=a.rounds)
    dm_M = parse_dem(si_family(ideal, p, 5 * p, p / 10).detector_error_model(decompose_errors=False))
    res = {}
    # S2: T = si_family(0.7 p, 15 p, p / 10), M = SI1000(p)
    dm_T2 = parse_dem(si_family(ideal, 0.7 * p, 15 * p, p / 10).detector_error_model(decompose_errors=False))
    res["S2+dem"] = class_ratios(dm_M, dm_T2)
    res["S2+demt"] = class_ratios(dm_M, dm_T2, time_split=True)
    # S3a: T = SI1000(p) (full DEM, with hyperedges), M = its graphlike approximation
    base = si_family(ideal, p, 5 * p, p / 10)
    dm_G = graphlike_dm(base.detector_error_model(decompose_errors=True))
    res["S3a+dem"] = class_ratios(dm_G, dm_M)
    res["S3a+demt"] = class_ratios(dm_G, dm_M, time_split=True)
    json.dump(res, open(os.path.join(a.res, "fit", "dem_class_truth.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, f"(T mass not representable in M's symptom set: {v['_T_mass_not_representable']:.4f} of {v['_T_mass_total']:.4f})")
        for c, x in v.items():
            if not c.startswith("_"):
                print(f"   {c:12s} n={x['n_faults']:4d} ratio of totals {x['ratio_of_totals']:.3f}  per-fault [{x['per_fault_ratio_min']:.2f}, {x['per_fault_ratio_max']:.2f}]")


if __name__ == "__main__":
    main()
