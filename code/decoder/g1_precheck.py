"""Informal G1 pre-check from the shipped predictions on the local extraction, using the frozen estimator
functions of analyze_willow.py (series_eps, mean_eps). The official G1 is the analysis run on the decode outputs."""
import sys, os, glob, json, math, collections
sys.path.insert(0, "/mnt/project-files/tensor-networks/deep-dives/experimental/code/decoder")
import numpy as np
import analyze_willow as A

base = sys.argv[1]
pws = ["libra_decoder_with_rl_optimized_prior", "correlated_matching_decoder_with_rl_optimized_prior",
       "harmony_decoder_with_rl_optimized_prior", "correlated_matching_decoder_with_si1000_prior",
       "harmony_decoder_with_si1000_prior"]
pts = collections.defaultdict(dict)   # (d, series, pw) -> {r: (n, k)}
for L in sorted(glob.glob(base + "/d*_at_*/[XZ]/r*")):
    act = os.path.join(L, "obs_flips_actual.b8")
    if not os.path.exists(act):
        continue
    rel = os.path.relpath(L, base); patch, basis, rr = rel.split("/"); d = int(patch.split("_")[0][1:]); r = int(rr[1:])
    a = np.fromfile(act, dtype=np.uint8) & 1
    for pw in pws:
        f = os.path.join(L, "decoding_results", pw, "obs_flips_predicted.b8")
        if not os.path.exists(f):
            continue
        p = np.fromfile(f, dtype=np.uint8) & 1
        assert p.size == a.size == 50000, (rel, pw, p.size, a.size)
        pts[(d, patch + "/" + basis, pw)][r] = (int(a.size), int(np.sum(p != a)))
out = {}
for d, window, rep in ((3, set(range(10, 251)), (7.12e-3, 0.06e-3)), (5, {10, 13, 30, 50}, (3.49e-3, 0.04e-3))):
    for pw in pws:
        fits = []
        for (dd, s, p), P in sorted(pts.items()):
            if dd != d or p != pw:
                continue
            f = A.series_eps(P, window & set(P))
            if f: fits.append(f)
        if not fits:
            continue
        m, se = A.mean_eps(fits)
        row = {"series": len(fits), "rounds_per_series": sorted({len(f["rounds"]) for f in fits}), "eps": m, "se": se,
               "chi2_red_median": float(np.median([f["chi2_red"] for f in fits]))}
        if pw.startswith("libra"):
            row["reported"] = rep[0]; row["z"] = (m - rep[0]) / math.sqrt(se ** 2 + rep[1] ** 2)
        out[f"d{d} {pw}"] = row
        print(f"d{d} {pw:52s} S={len(fits):2d} rounds={row['rounds_per_series']} eps={m:.5e} se={se:.2e}"
              + (f"  reported {rep[0]:.2e}  z={row['z']:+.2f}" if 'z' in row else ""))
json.dump(out, open(sys.argv[2], "w"), indent=1)
