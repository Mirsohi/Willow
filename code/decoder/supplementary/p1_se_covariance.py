#!/usr/bin/env python3
"""Analysis correction for the SE of g in P1 (section 9 of notes/decoder/willow_prereg.md, entry 7; 2026-10-09).

Section 4 asks for a delta-method SE of g = mean_s(eps_A - eps_ML) / mean_s(eps_ML). In each series s both terms come
from fits to the same ML counts: eps_ML from the slope b_M of y_M,r = ln(1 - 2 E_ML,r) on r, and eps_A from b_M + beta,
with beta the slope of z_r = y_A,r - y_M,r. The frozen pool_gap() of analyze_willow.py (md5 52fb23ff...) treats the
series-mean gap and the series-mean eps_ML as independent and keeps only d(delta)/d(beta). This script

1. rebuilds the 2x2 counts (Libra wrong, ML(RL) wrong) of the 18 d = 3 series from the decode folders with the frozen
   load_leaf() and count2(), and the window from plan.json with the frozen planned_rounds();
2. reproduces the frozen paired_series() and pool_gap() results (g and its frozen SE) and checks them against
   analysis_m3/analysis.json;
3. computes the full first-order delta method: per series the 2x2 covariance of (beta, b_M), with
   Cov(beta, b_M) = sum_r a_r c_r Cov(z_r, y_M,r), Cov(z_r, y_M,r) = gA gM Cv - gM^2 VM (the variances and covariance
   of the leaf's 2x2 table, as in paired_series), a_r and c_r the WLS slope coefficients of the two fits; the Birge
   factors scale Var(beta) by f_z^2, Var(b_M) by f_M^2 and the covariance by f_z f_M (correlation unchanged);
   then Var(g) = sum_s grad_s' Sigma_s grad_s over the independent series;
4. checks it against a shot bootstrap (multinomial resampling of each leaf's 2x2 table; captures the covariance but
   not the Birge inflation, so it is compared with the uninflated delta method).

Inputs are only read; nothing in code/decoder is changed. Run:
    nice python3 -I -B code/decoder/supplementary/p1_se_covariance.py --res results/decoder/willow \
        --check results/decoder/willow/analysis_m3/analysis.json --out results/decoder/willow/supplementary
"""
import argparse, json, math, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/decoder
import numpy as np
import analyze_willow as AW            # frozen; used read-only

A_ENT, M_ENT = "libra.rl", "ML.rl"


def points_arrays(pts, window):
    """Per-point quantities of paired_series() (same filters) for the rounds it keeps."""
    rows = []
    for r in sorted(pts):
        if r not in window:
            continue
        c = pts[r]; n = c["n"]
        if not n:
            continue
        EA = (c["n10"] + c["n11"]) / n; EM = (c["n01"] + c["n11"]) / n; p11 = c["n11"] / n
        if not (0 < EA < 0.5 and 0 < EM < 0.5):
            continue
        gA, gM = -2 / (1 - 2 * EA), -2 / (1 - 2 * EM)
        VA, VM, Cv = EA * (1 - EA) / n, EM * (1 - EM) / n, (p11 - EA * EM) / n
        rows.append({"r": r, "z": math.log(1 - 2 * EA) - math.log(1 - 2 * EM),
                     "vz": max(gA ** 2 * VA + gM ** 2 * VM - 2 * gA * gM * Cv, 1e-18),
                     "yM": math.log(1 - 2 * EM), "vM": gM ** 2 * VM, "cov_zy": gA * gM * Cv - gM ** 2 * VM})
    return rows


def slope_coef(r, v):
    w = 1 / np.asarray(v, float); r = np.asarray(r, float)
    rb = (w * r).sum() / w.sum(); Srr = (w * (r - rb) ** 2).sum()
    return w * (r - rb) / Srr


def series_params(pts, window):
    """(beta, b_M, Var beta, Var b_M, Cov) with and without the Birge factors, plus the frozen per-series result."""
    fr = AW.paired_series(pts, window)
    P = points_arrays(pts, window)
    r = [p["r"] for p in P]
    fz = AW.wls(r, [p["z"] for p in P], [p["vz"] for p in P])
    fm = AW.wls(r, [p["yM"] for p in P], [p["vM"] for p in P])
    a = slope_coef(r, [p["vz"] for p in P]); c = slope_coef(r, [p["vM"] for p in P])
    cov0 = float(np.sum(a * c * np.array([p["cov_zy"] for p in P])))
    var_b0, var_m0 = float(np.sum(a ** 2 * [p["vz"] for p in P])), float(np.sum(c ** 2 * [p["vM"] for p in P]))
    fz_f = math.sqrt(max(1.0, fz["chi2_red"])); fm_f = math.sqrt(max(1.0, fm["chi2_red"]))
    assert abs(math.sqrt(var_b0) * fz_f - fz["se"]) <= 1e-12 * fz["se"] and abs(math.sqrt(var_m0) * fm_f - fm["se"]) <= 1e-12 * fm["se"]
    return {"frozen": fr, "beta": fz["slope"], "bM": fm["slope"], "npts": len(P),
            "uninflated": {"var_beta": var_b0, "var_bM": var_m0, "cov": cov0},
            "inflated": {"var_beta": var_b0 * fz_f ** 2, "var_bM": var_m0 * fm_f ** 2, "cov": cov0 * fz_f * fm_f},
            "birge": [fz_f, fm_f], "corr_beta_bM": cov0 / math.sqrt(var_b0 * var_m0)}


def g_and_se(S, mode, full=True, with_cov=True):
    """g and its delta-method SE from the per-series parameters. full=False, with_cov=False is the frozen formula."""
    n = len(S)
    eM = np.array([(1 - math.exp(s["bM"])) / 2 for s in S])
    dl = np.array([(math.exp(s["bM"]) - math.exp(s["bM"] + s["beta"])) / 2 for s in S])
    D, E = dl.mean(), eM.mean(); g = D / E
    var = 0.0; vD = vE = cDE = 0.0
    for s, d_s in zip(S, dl):
        V = s[mode]
        dd_db = -math.exp(s["bM"] + s["beta"]) / 2            # d delta / d beta
        dd_dm = d_s if full else 0.0                             # d delta / d b_M
        de_dm = -math.exp(s["bM"]) / 2                           # d eps_ML / d b_M
        cv = V["cov"] if with_cov else 0.0
        vd = dd_db ** 2 * V["var_beta"] + dd_dm ** 2 * V["var_bM"] + 2 * dd_db * dd_dm * cv
        ve = de_dm ** 2 * V["var_bM"]
        cde = dd_db * de_dm * cv + dd_dm * de_dm * V["var_bM"]
        vD += vd / n ** 2; vE += ve / n ** 2; cDE += cde / n ** 2
    var = vD / E ** 2 + D ** 2 * vE / E ** 4 - 2 * D * cDE / E ** 3
    se = math.sqrt(var)
    return {"g": g, "se": se, "z": g / se, "p": AW.pval(g / se), "ci90": [g - 1.645 * se, g + 1.645 * se],
            "ci95": [g - 1.96 * se, g + 1.96 * se], "se_mean_delta": math.sqrt(vD), "se_mean_eps_ML": math.sqrt(vE),
            "corr_mean_delta_mean_eps_ML": cDE / math.sqrt(vD * vE) if with_cov or full else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="analysis.json of the frozen run to compare with")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    t0 = time.time()
    plan = json.load(open(os.path.join(a.res, "plan.json")))
    PR = AW.planned_rounds(plan)
    dec_root = os.path.join(a.res, "decode")
    pair_pts = {}
    for dp, dn, fn in os.walk(dec_root):
        if "summary.json" not in fn or os.path.basename(dp) != "rl":
            continue
        rel = os.path.relpath(os.path.dirname(dp), dec_root)
        obs, ents, ml, info, shipped = AW.load_leaf(a.res, rel, ["rl"])
        meta = info.get("meta") or {}
        if meta.get("distance") != 3 or A_ENT not in ents or M_ENT not in ents:
            continue
        c = AW.count2(ents[A_ENT], ents[M_ENT], obs)
        if c["n"]:
            pair_pts.setdefault((3, meta["patch"], meta["basis"]), {})[info["rounds"]] = c
    S, keys = [], []
    for key in sorted(pair_pts):
        both = (PR.get(key + (A_ENT,)) or set()) & (PR.get(key + (M_ENT,)) or set())
        window = tuple(sorted(x for x in both if x >= 10))
        sp = series_params(pair_pts[key], window)
        if sp["frozen"] and sp["frozen"]["complete"]:
            S.append(sp); keys.append(f"d3|{key[1]}|{key[2]}|{A_ENT} vs {M_ENT}")
    frozen_pool = AW.pool_gap([s["frozen"] for s in S])
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "series": len(S),
           "points_per_series": sorted({s["npts"] for s in S}), "leaves": sum(len(v) for v in pair_pts.values()),
           "frozen_pool_gap": frozen_pool,
           "frozen_formula_rebuilt": g_and_se(S, "inflated", full=False, with_cov=False),
           "covariance_only_in_cross_term": g_and_se(S, "inflated", full=False, with_cov=True),
           "full_delta_method_inflated": g_and_se(S, "inflated", full=True, with_cov=True),
           "full_delta_method_uninflated": g_and_se(S, "uninflated", full=True, with_cov=True),
           "frozen_formula_uninflated": g_and_se(S, "uninflated", full=False, with_cov=False),
           "per_series": [{"series": k, "beta": s["beta"], "bM": s["bM"], "birge_z_M": s["birge"],
                           "corr_beta_bM": s["corr_beta_bM"], "rel": s["frozen"]["rel"]} for k, s in zip(keys, S)]}
    cb = [s["corr_beta_bM"] for s in S]
    out["corr_beta_bM_range"] = [min(cb), max(cb)]
    # check against the frozen run's analysis.json
    if a.check:
        A = json.load(open(a.check))
        dif = []
        for k, s in zip(keys, S):
            f0 = A["paired_series"][k]
            for q in ("eps_ML", "eps_A", "delta", "se_delta", "se_eps_ML", "rel", "chi2_red"):
                dif.append(abs(s["frozen"][q] - f0[q]) / max(abs(f0[q]), 1e-300))
            assert list(s["frozen"]["rounds"]) == list(f0["rounds"]), k
        pg = A["pooled_gaps"]["d3|ML.rl"] if "d3|ML.rl" in A["pooled_gaps"] else None
        pg = A["pooled_gaps"].get(f"d3|{A_ENT} vs {M_ENT}", pg)
        out["check_vs_frozen_run"] = {"max_relative_difference_per_series": max(dif),
                                      "frozen_run_g": pg["rel"], "frozen_run_se": pg["se_rel"],
                                      "rebuilt_g": frozen_pool["rel"], "rebuilt_se": frozen_pool["se_rel"]}
    # shot bootstrap: resample each leaf's 2x2 table (multinomial), recompute with the frozen functions
    rng = np.random.default_rng(a.seed)
    sk = sorted(pair_pts)
    windows = {k: tuple(sorted(x for x in ((PR.get(k + (A_ENT,)) or set()) & (PR.get(k + (M_ENT,)) or set())) if x >= 10))
               for k in sk}
    gs, Ds, Es = [], [], []
    for b in range(a.boot):
        rows = []
        for k in sk:
            pts = {}
            for r, c in pair_pts[k].items():
                if r not in windows[k]:
                    continue
                n00 = c["n"] - c["n10"] - c["n01"] - c["n11"]
                x = rng.multinomial(c["n"], np.array([c["n10"], c["n01"], c["n11"], n00], float) / c["n"])
                pts[r] = {"n": c["n"], "n10": int(x[0]), "n01": int(x[1]), "n11": int(x[2])}
            f = AW.paired_series(pts, windows[k])
            if f and f["complete"]:
                rows.append(f)
        if len(rows) == len(S):
            p = AW.pool_gap(rows)
            gs.append(p["rel"]); Ds.append(p["mean_delta"]); Es.append(p["mean_eps_ML"])
    gs = np.array(gs)
    out["bootstrap"] = {"replicates": int(len(gs)), "seed": a.seed, "sd_g": float(gs.std(ddof=1)),
                        "percentile_ci95": [float(np.quantile(gs, 0.025)), float(np.quantile(gs, 0.975))],
                        "corr_mean_delta_mean_eps_ML": float(np.corrcoef(Ds, Es)[0, 1])}
    out["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "p1_se_covariance.json"), "w"), indent=1, default=float)
    show = {k: out[k] for k in out if k not in ("per_series", "frozen_pool_gap")}
    show["frozen_pool_gap"] = {k: frozen_pool[k] for k in ("rel", "se_rel", "z", "ci95", "ci90")}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main()
