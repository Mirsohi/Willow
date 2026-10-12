"""Pre-specified analysis of the exact test (notes/reservoir/gapmech_design.md, fixed before stage 1 ran). EXPLORATORY:
the stages and this analysis were fixed in advance, but the question came after the Tesseract split was seen.

reservoir, 2026-10-10. MPE decisions: at d = 3 the exact max-product class from stage 1 (exact_run.py; ties, if any,
stay at class 0 and are counted); at d = 5 decoder's stored Tesseract (r <= 30), certified on stage 2's sample. For each
decoder A (Libra on rounds {10, 30}; Harmony, correlated matching, our correlated MWPM and MWPM on {10, 13, 30}), on the
shots where A, the MPE and ML(RL) all have decisions (every 5th shot at both distances), the paired gaps of prereg
section 4 with entry 7's SE (gapmech/d5_check):
    g(A|ML), g(MPE|ML), g(A|MPE) at d = 3 and d = 5, and the Lambda ratios (1 + g_3)/(1 + g_5) with log-scale 95% CIs
    (independent distances): Lambda_MPE/Lambda_ML (the degeneracy factor), Lambda_A/Lambda_MPE, Lambda_A/Lambda_ML;
    the share of ln(Lambda_A/Lambda_ML) carried by ln(Lambda_MPE/Lambda_ML), point estimate from the pooled gaps, 95% CI
    by a percentile bootstrap over series (B = 10,000, seed 20261010; d = 3 and d = 5 series resampled independently,
    the same draw for the three pairings).
Reading rule (main, 10 Oct, fixed before the run):
    "degeneracy carries most of A's growth" only if the share's lower 95% bound is above 0.5;
    "all of it, within error" only if the whole 95% CI of Lambda_A/Lambda_MPE lies inside [0.98, 1.02] (the analytic
    log-scale CI, as in the matched table); since "all" implies "most", it is used only when both conditions hold;
    otherwise the estimate and its CI, with neither phrase.
Rule A (main 19:58Z) is primary and no exclusion the sensitivity row; K2's verdicts as found at run time. Every d = 5
number that relies on the stored Tesseract carries stage 2's certificate (misses among proven shots, one-sided 95%
Clopper-Pearson upper bound). Also: the degeneracy factor on {10, 13, 30} and {10, 30} for the MPE alone, g_MPE at
d = 3 on W5 (descriptive), stage 1's count of shots where the stored Tesseract is not the exact MPE, and shot categories
(A against ML, MPE against ML) with net losses; with each d = 5 number, the leverage of the certificate (leverage()).
Usage: python3 -I exact_analysis.py ARCHIVE DECODE K2_DIR OUT_DIR [STAGE_DIR]   (STAGE_DIR defaults to OUT_DIR)
  -> OUT_DIR/exact_analysis.json (through qrc.outpaths)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gapmech import MISSING, growth, gsum, read_leaf  # noqa: E402
from d5_check import BASES, P3, P5, k2_verdicts, paired, rel_of  # noqa: E402  (path set by gapmech)
from qrc.outpaths import out_path  # noqa: E402

OTHERS = ["libra", "harmony", "corrmatch", "corr_mwpm", "mwpm"]
ROUNDS = [10, 13, 30]
B_BOOT = 10000
SEED = 20261010
EQUIV = (0.98, 1.02)


def wrong(rec, x):
    return rec["ml"] != rec["obs"] if x == "ml" else rec["preds"][x] != rec["obs"]


def mask_of(rec, need):
    m = np.ones(rec["obs"].size, bool)
    for x in need:
        if x != "ml":
            m &= rec["preds"][x] != MISSING
    return m


def series_fits(leaves, x, y, need, d, rounds, excluded):
    """paired() of X against reference Y per series, on the shots where every decoder in `need` has a decision."""
    out = {}
    for p in (P3 if d == 3 else P5):
        for b in BASES:
            L = []
            for r in rounds:
                rec = leaves.get((d, p, b, r))
                if rec is None or rel_of(d, p, b, r) in excluded:
                    continue
                if any(z != "ml" and z not in rec["preds"] for z in need):
                    continue
                m = mask_of(rec, need)
                xw, yw = wrong(rec, x) & m, wrong(rec, y) & m
                if m.sum():
                    L.append((r, int(m.sum()), int(xw.sum()), int(yw.sum()), int(np.sum(xw & yw))))
            if len(L) < 2:
                continue
            pr = paired([q[0] for q in L], [q[2] for q in L], [q[3] for q in L], [q[4] for q in L],
                        [q[1] for q in L])
            pr["basis"] = b
            pr["rounds"] = [q[0] for q in L]
            pr["shots"] = [q[1] for q in L]
            out[(p, b)] = pr
    return out


def ratio(f3, f5):
    """g_3, g_5 and the Lambda ratio with its analytic CI from two dicts of series fits."""
    g3, g5 = gsum(list(f3.values())), gsum(list(f5.values()))
    o = {"g3": g3, "g5": g5, "series_d3": len(f3), "series_d5": len(f5)}
    if g3 and g5:
        o.update(growth(g3, g5))
    return o


def boot_arrays(fits3, fits5, keys3, keys5):
    """Per pairing: arrays of d and eps_ref over the common series, for the bootstrap."""
    return {t: (np.array([fits3[t][k]["d"] for k in keys3]), np.array([fits3[t][k]["eps_ml"] for k in keys3]),
                np.array([fits5[t][k]["d"] for k in keys5]), np.array([fits5[t][k]["eps_ml"] for k in keys5]))
            for t in fits3}


def bootstrap(arr, rng, B):
    n3 = arr["A|ML"][0].size
    n5 = arr["A|ML"][2].size
    i3 = rng.integers(0, n3, (B, n3))
    i5 = rng.integers(0, n5, (B, n5))
    ln = {}
    for t, (d3, e3, d5, e5) in arr.items():
        g3 = d3[i3].sum(1) / e3[i3].sum(1)
        g5 = d5[i5].sum(1) / e5[i5].sum(1)
        ln[t] = np.log1p(g3) - np.log1p(g5)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = ln["MPE|ML"] / ln["A|ML"]
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"B": B, "share_ci95": pc(share), "share_nonfinite": int(np.sum(~np.isfinite(share))),
            "lambda_ci95": {t: pc(np.exp(v)) for t, v in ln.items()}}


def certificate(stage_dir):
    p = os.path.join(stage_dir, "exact_stage2.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    a = s["all"]
    c = a["conservative_unproven_as_misses"]
    txt = (f"d = 5 MPE = stored Tesseract; certified: {a['misses']} misses in {a['proven']} proven shots, one-sided 95% "
           f"upper bound {100 * a['miss_rate_upper95_one_sided_clopper_pearson']:.2f}% ({a['unproven']} unproven; "
           f"counting them as misses: {100 * c['upper95_one_sided_clopper_pearson']:.2f}%)"
           if a["proven"] else "d = 5 MPE = stored Tesseract; no certified shots")
    return {"text": txt, "summary": s}


def leverage(leaves, need, rounds, excluded, cert):
    """Descriptive: per-shot error rates of the MPE (the stored Tesseract) and ML on the analysed d = 5 shots, D = the
    per-shot net difference (k_MPE - k_ML) / n, and the certificate's upper bound u on the per-shot miss rate. A miss
    can move the MPE's error count by at most one, so u / D says how much of the d = 5 MPE - ML difference the worst case
    allowed by the certificate could move. 'all' uses the per-round bounds weighted by the analysed shots per round."""
    o = {}
    for tag, rr in [(f"r{r}", [r]) for r in rounds] + [("all", rounds)]:
        n = km = kl = 0
        for (d, p, b, r), rec in leaves.items():
            if d != 5 or r not in rr or rel_of(d, p, b, r) in excluded or any(z not in rec["preds"] for z in need):
                continue
            m = mask_of(rec, need)
            n += int(m.sum())
            km += int(np.sum((rec["preds"]["mpe"] != rec["obs"]) & m))
            kl += int(np.sum((rec["ml"] != rec["obs"]) & m))
        D = (km - kl) / n if n else None
        if tag == "all":
            us = [(o[f"r{r}"]["u_upper95"], o[f"r{r}"]["shots"]) for r in rounds]
            u = sum(a * b for a, b in us) / sum(b for _, b in us) if all(a is not None for a, _ in us) and n else None
        else:
            sm = cert["summary"].get(tag) if cert else None
            u = sm["miss_rate_upper95_one_sided_clopper_pearson"] if sm else None
        o[tag] = {"shots": n, "k_mpe": km, "k_ml": kl, "D_per_shot": D, "u_upper95": u,
                  "u_over_D": u / D if (u is not None and D) else None}
    return o


def categories(leaves, a, d, rounds, excluded):
    cat = {}
    for (dd, p, b, r), rec in leaves.items():
        if dd != d or r not in rounds or rel_of(d, p, b, r) in excluded or a not in rec["preds"] \
                or "mpe" not in rec["preds"]:
            continue
        m = mask_of(rec, [a, "mpe"])
        A, M, ML, o = rec["preds"][a][m], rec["preds"]["mpe"][m], rec["ml"][m], rec["obs"][m]
        for name, sel in (("A=ML,MPE=ML", (A == ML) & (M == ML)), ("A!=ML,MPE=ML", (A != ML) & (M == ML)),
                          ("A=ML,MPE!=ML", (A == ML) & (M != ML)), ("A!=ML,MPE!=ML", (A != ML) & (M != ML))):
            c = cat.setdefault(name, {"shots": 0, "A_net": 0, "MPE_net": 0, "k_ml": 0})
            c["shots"] += int(sel.sum())
            c["k_ml"] += int(np.sum((ML != o) & sel))
            c["A_net"] += int(np.sum((A != o) & sel)) - int(np.sum((ML != o) & sel))
            c["MPE_net"] += int(np.sum((M != o) & sel)) - int(np.sum((ML != o) & sel))
    return cat


def reading(share_lo, lam_ci):
    """main's rule; both phrases are 'only if', and 'all of it' also needs the condition for 'most'."""
    most = share_lo is not None and share_lo > 0.5
    inside = lam_ci is not None and EQUIV[0] <= lam_ci[0] and lam_ci[1] <= EQUIV[1]
    if most and inside:
        txt = "degeneracy carries all of A's growth, within error"
    elif most:
        txt = "degeneracy carries most of A's growth"
    else:
        txt = "estimate and CI only (neither phrase)"
    return {"share_lower95_above_0.5": most, "lambda_A_over_MPE_ci95_inside_0.98_1.02": inside, "reading": txt}


def main(arch, dec, k2_dir, out_dir, stage_dir=None):
    stage_dir = stage_dir or out_dir
    t0 = time.time()
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = k2_verdicts(k2_dir)
    k2_read = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    leaves = {}
    missing_stage1 = []
    for d, pats, rounds in ((3, P3, [10, 13, 30, 50]), (5, P5, ROUNDS)):
        for p in pats:
            for b in BASES:
                for r in rounds:
                    rec = read_leaf(arch, dec, d, p, b, r, chk)
                    rel = rel_of(d, p, b, r)
                    if d == 3:
                        f = os.path.join(stage_dir, "exact", rel.replace("/", "_") + ".npz")
                        if os.path.exists(f):  # at d = 3 ML has every shot (checked: ml_idx_rule_ok)
                            z = np.load(f, allow_pickle=False)
                            full = np.full(rec["n"], MISSING, np.uint8)
                            full[z["idx"].astype(np.int64)] = z["mpe"]
                            rec["preds"]["mpe"] = full
                        else:
                            missing_stage1.append(rel)
                    elif "tesseract" in rec["preds"]:
                        rec["preds"]["mpe"] = rec["preds"]["tesseract"]
                    leaves[(d, p, b, r)] = rec
        print(f"d = {d}: leaves read ({time.time() - t0:.0f} s)", flush=True)
    cert = certificate(stage_dir)
    s1p = os.path.join(stage_dir, "exact_stage1.json")
    s1 = json.load(open(s1p))["summary"] if os.path.exists(s1p) else None
    res = {"label": "EXPLORATORY (reservoir, 2026-10-10): pre-specified analysis of the exact test "
                    "(notes/reservoir/gapmech_design.md). Interim: K2 verdicts as found at run time.",
           "k2": {"read_at": k2_read, "verdicts_found": len(k2), "still_fail": sorted(failing)}, "checks": chk,
           "stage1_missing_leaves": missing_stage1, "stage1_summary": s1, "d5_certificate": cert,
           "reading_rule": {"most": "share's lower 95% bound > 0.5", "all": "analytic 95% CI of Lambda_A/Lambda_MPE "
                            "inside [0.98, 1.02], and the condition for 'most'", "share_ci": "percentile bootstrap over series, B = 10,000, seed 20261010, "
                            "d = 3 and d = 5 series resampled independently"},
           "degeneracy_factor": {}, "g_mpe_d3_W5": {}, "decoders": {}}
    cert_txt = cert["text"] if cert else "d = 5 MPE = stored Tesseract; stage 2 not available"
    for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
        for rname, rounds in (("10_13_30", ROUNDS), ("10_30", [10, 30])):
            f3 = series_fits(leaves, "mpe", "ml", ["mpe"], 3, rounds, set())
            f5 = series_fits(leaves, "mpe", "ml", ["mpe"], 5, rounds, excl)
            o = ratio(f3, f5)
            o["d5_certificate"] = cert_txt
            o["d5_leverage"] = leverage(leaves, ["mpe"], rounds, excl, cert)
            res["degeneracy_factor"][f"{rname}|{kname}"] = o
        if kname == "ruleA":
            f3 = series_fits(leaves, "mpe", "ml", ["mpe"], 3, [10, 13, 30, 50], set())
            res["g_mpe_d3_W5"] = gsum(list(f3.values()))
    rng = np.random.default_rng(SEED)
    for a in OTHERS:
        rounds = [r for r in ROUNDS if all(a in leaves[(d, p, b, r)]["preds"] and "mpe" in leaves[(d, p, b, r)]["preds"]
                                           for d, pats in ((3, P3), (5, P5)) for p in pats for b in BASES)]
        res["decoders"][a] = {"rounds": rounds}
        for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
            fits3, fits5 = {}, {}
            for t, x, y in (("A|ML", a, "ml"), ("MPE|ML", "mpe", "ml"), ("A|MPE", a, "mpe")):
                fits3[t] = series_fits(leaves, x, y, [a, "mpe"], 3, rounds, set())
                fits5[t] = series_fits(leaves, x, y, [a, "mpe"], 5, rounds, excl)
            o = {t: ratio(fits3[t], fits5[t]) for t in fits3}
            keys3 = sorted(set.intersection(*(set(fits3[t]) for t in fits3)))
            keys5 = sorted(set.intersection(*(set(fits5[t]) for t in fits5)))
            share = None
            if all("lambda_ratio" in o[t] for t in o):
                la, lm = math.log(o["A|ML"]["lambda_ratio"]), math.log(o["MPE|ML"]["lambda_ratio"])
                share = lm / la if la else None
                bt = bootstrap(boot_arrays(fits3, fits5, keys3, keys5), rng, B_BOOT)
                o["share"] = {"estimate": share, "ci95_bootstrap": bt["share_ci95"],
                              "bootstrap_nonfinite": bt["share_nonfinite"]}
                o["bootstrap_lambda_ci95"] = bt["lambda_ci95"]
                o["reading"] = reading(bt["share_ci95"][0], o["A|MPE"].get("lambda_ratio_ci95"))
                o["factorisation_check_ln"] = {"A|ML": la, "MPE|ML + A|MPE": lm + math.log(o["A|MPE"]["lambda_ratio"])}
            o["series_common"] = {"d3": len(keys3), "d5": len(keys5)}
            o["categories"] = {"d3": categories(leaves, a, 3, rounds, set()),
                               "d5": categories(leaves, a, 5, rounds, excl)}
            o["d5_certificate"] = cert_txt
            o["d5_leverage"] = leverage(leaves, [a, "mpe"], rounds, excl, cert)
            res["decoders"][a][kname] = o
        print(f"{a} done ({time.time() - t0:.0f} s)", flush=True)
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "exact_analysis.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(f"done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (4, 5):
        raise SystemExit(__doc__)
    main(*a)
