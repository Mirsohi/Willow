"""PR3 planning numbers (no decoding): shipped decoders' d = 7 error counts on all 50,000 shots, Libra's eps7 against
Table S1 (G1 at d = 7), and the expected precision of g_7 for candidate designs, with the frozen estimators."""
import json, math, os, sys, time
sys.dont_write_bytecode = True
X = "/mnt/project-files/tensor-networks/deep-dives/experimental"
sys.path.insert(0, X + "/code/decoder"); sys.path.insert(0, X + "/code/decoder/supplementary")
import numpy as np
import analyze_willow as AW
import p1_se_covariance as PC
E = "/home/claude/work/decoder/willow_run/extract/google_105Q_surface_code_d3_d5_d7/d7_at_q6_7"
rep = json.load(open(X + "/code/decoder/willow_reported.json"))
DEC = ("libra_decoder_with_rl_optimized_prior", "harmony_decoder_with_rl_optimized_prior",
       "correlated_matching_decoder_with_rl_optimized_prior", "harmony_decoder_with_si1000_prior",
       "correlated_matching_decoder_with_si1000_prior")
import zipfile
ZP = "/home/claude/work/decoder/willow_run/zip/google_105Q_surface_code_d3_d5_d7.zip"
ZF = zipfile.ZipFile(ZP)
NAMES = set(ZF.namelist())
PFX = "google_105Q_surface_code_d3_d5_d7/d7_at_q6_7"
def read_bits(rel):
    """1 observable per shot, one byte per shot (b8): the local extract if present, else the archive (read only)."""
    lp = f"{E}/{rel}"
    if os.path.exists(lp):
        return np.fromfile(lp, np.uint8) & 1
    nm = f"{PFX}/{rel}"
    if nm not in NAMES:
        return None
    return np.frombuffer(ZF.read(nm), np.uint8) & 1
rdirs = {int(d[1:]): d for d in os.listdir(E + "/X") if d.startswith("r")}
rounds = sorted(r for r in rdirs if read_bits(f"X/{rdirs[r]}/obs_flips_actual.b8") is not None)
cnt = {}
for b in ("X", "Z"):
    for r in rounds:
        obs = read_bits(f"{b}/{rdirs[r]}/obs_flips_actual.b8")
        for dn in DEC:
            pr = read_bits(f"{b}/{rdirs[r]}/decoding_results/{dn}/obs_flips_predicted.b8")
            if pr is not None:
                assert len(pr) == len(obs)
                cnt.setdefault(dn, {}).setdefault(b, {})[r] = (int(len(obs)), int(np.sum(pr != obs)))
out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rounds": rounds,
       "libra_rounds": {b: sorted(cnt[DEC[0]][b]) for b in ("X", "Z")}}
fits = {}
for dn in DEC:
    for b in ("X", "Z"):
        pts = cnt[dn][b]
        for wn, w in (("r>=10", tuple(x for x in sorted(pts) if x >= 10)), ("10,30,50", (10, 30, 50)),
                      ("10..90", (10, 30, 50, 70, 90))):
            f = AW.series_eps(pts, w)
            if f:
                fits.setdefault(dn, {}).setdefault(wn, {})[b] = {"eps": f["eps"], "se": f["se_eps"], "chi2_red": f["chi2_red"],
                                                                  "npts": f["npts"]}
out["fits"] = fits
g1 = {}
for wn in ("r>=10", "10,30,50", "10..90"):
    fx, fz = fits[DEC[0]][wn]["X"], fits[DEC[0]][wn]["Z"]
    m = (fx["eps"] + fz["eps"]) / 2; se = math.sqrt(fx["se"] ** 2 + fz["se"] ** 2) / 2
    rv, rs = rep["table_S1_fig1d"]["libra"]["eps7"]
    g1[wn] = {"ours": m, "se": se, "reported": rv, "z": (m - rv) / math.sqrt(se ** 2 + rs ** 2)}
out["G1_d7_libra"] = g1
out["libra_errors"] = {b: {str(r): cnt[DEC[0]][b][r] for r in sorted(cnt[DEC[0]][b])} for b in ("X", "Z")}
print(json.dumps({"G1_d7": g1, "libra_X": out["libra_errors"]["X"], "libra_Z": out["libra_errors"]["Z"]}, indent=0, default=float))
for dn in DEC:
    print(dn[:28], {wn: {b: round(1e3 * v["eps"], 4) for b, v in d.items()} for wn, d in fits[dn].items()})

# ---- expected precision of g_7 (frozen estimators on expected 2x2 tables)
F01 = 0.107            # n01 / k_ML, pooled over the 24 d = 5 Libra leaves (measured, 19:40Z)
def expected_tables(basis, window, N, g):
    """Libra: its measured E_r on all shots (line through ln(1 - 2E) on the window). ML: the same intercept, per-round
    eps_ML = eps_Libra / (1 + g). n01 = F01 k_ML."""
    pts = cnt[DEC[0]][basis]
    f = AW.series_eps(pts, window)
    a, b = f["intercept"], f["slope"]
    epsA = (1 - math.exp(b)) / 2; epsM = epsA / (1 + g); bM = math.log(1 - 2 * epsM)
    tab = {}
    for r in window:
        EA = (1 - math.exp(a + b * r)) / 2; EM = (1 - math.exp(a + bM * r)) / 2
        kA, kM = N * EA, N * EM
        n01 = F01 * kM; n11 = kM - n01; n10 = kA - n11
        tab[r] = {"n": N, "n10": n10, "n01": n01, "n11": n11}
    return tab
def precision(window, N, g):
    S, rows, cells = [], [], []
    for basis in ("X", "Z"):
        t = expected_tables(basis, window, N, g)
        S.append(PC.series_params(t, window)); rows.append(AW.paired_series(t, window)); cells += list(t.values())
    se0 = PC.g_and_se(S, "uninflated", full=True, with_cov=True)["se"]
    cr = AW.ratio_from_counts(cells)
    kM = sum(c["n01"] + c["n11"] for c in cells)
    return {"g": AW.pool_gap(rows)["rel"], "se_slope_uninflated": se0, "count_ratio": cr["rel"], "se_count": cr["se_rel"],
            "ML_errors": kM}
des = {}
for wn, w in (("10,30,50", (10, 30, 50)), ("10..90", (10, 30, 50, 70, 90))):
    for N in (10000, 25000, 50000):
        for g in (0.06, 0.11):
            des[f"{wn}|N={N}|g={g}"] = precision(w, N, g)
out["designs"] = des
for k, v in des.items():
    print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})
json.dump(out, open("/home/claude/work/decoder/pr3/d7_numbers.json", "w"), indent=1, default=float)
