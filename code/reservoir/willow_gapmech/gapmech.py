"""EXPLORATORY, post hoc: why does every decoder's gap to ML(RL) grow from d = 3 to d = 5? Items 1 and 2 of main's brief
(10 Oct 2026, after the d = 5 check), from existing per-shot outputs only; nothing is decoded here.

reservoir, 2026-10-10. Not pre-registered. The estimators are the pre-registered ones of notes/decoder/willow_prereg.md
section 4 (imported from willow_d5check/d5_check.py: paired(), g_stat(); g's SE with section 9 entry 7's covariance
term and the chi2_red inflation), applied to other decoders and other round sets than the prereg's.

Item 1, matched gap table. For every decoder A with RL-prior predictions at both d = 3 and d = 5 (Libra, Harmony and
correlated matching as shipped; Tesseract, corr. MWPM and MWPM from decoder's runs), the paired gap g_d = mean_s(eps_A -
eps_ML) / mean_s(eps_ML) at d = 3 (18 series) and d = 5 (8 series) on the same rounds at both distances:
  matched  every W5 round on which A has predictions at both distances (Libra 10, 30, 50; Tesseract 10, 13, 30; the
           others 10, 13, 30, 50);
  common   rounds 10 and 30 for every decoder.
Shots: at d = 3 every shot that has both predictions (all 50,000, or Tesseract's every 5th); at d = 5 ML's every 5th.
K2 at d = 5: rule A (main 19:58Z: a still-failing leaf's round leaves the ML fits; its series stays, if it keeps two
rounds) and no exclusion; verdicts as found at run time. Reported: g_3, g_5, the paired Lambda ratio
Lambda_A / Lambda_ML = (1 + g_3) / (1 + g_5) with a log-scale CI, the growth ln(1 + g_5) - ln(1 + g_3), per-series gaps
(Mann-Whitney between the d = 5 and d = 3 series), and g by basis.

Item 2, discordant shots. Per decoder and distance (matched rounds, rule A): b = A wrong and ML right, c = A right and ML
wrong; their counts by ML's posterior margin |llr|, by the shot's syndrome weight (number of detection events, as a
quartile within its leaf), by r and by basis; ML's predicted share of b among b + c (mean of 1/(1 + e^-|llr|) over the
discordant shots) against the observed share. Overlap with Tesseract (a most-probable-error search decoder) on the
shots both have: the share of each decoder's net loss b - c on shots where Tesseract also disagrees with ML.

Inputs (read only; python3 -I, the archive is downloaded data):
  ARCHIVE/d<d>_at_<patch>/<basis>/r<NN>/{obs_flips_actual.b8, detection_events.b8, decoding_results/<pathway>/...}
  DECODE/<arch>/d<d>_at_<patch>/<basis>/r<NN>/{rl, rl_s5, rl_s5t}/*.npz and summary.json (decoder's outputs)
  K2_DIR/<arch>/d5_at_<patch>/<basis>/r<NN>/rl_s5/extra_rung.json
Usage: nice -n 10 python3 -I gapmech.py ARCHIVE DECODE K2_DIR OUT_DIR
       -> OUT_DIR/gap_table.json, gap_series.csv, discord.json, discord_bins.csv (through qrc.outpaths)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_d5check"))
sys.path.insert(0, os.path.join(HERE, ".."))
from d5_check import (ARCH, BASES, CM_RL, P3, P5, W5, Z90, Z95, g_stat, k2_verdicts, md5, paired,  # noqa: E402
                      rel_of)
from willow_check import read_obs_b8  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

DECODERS = ["libra", "tesseract", "harmony", "corrmatch", "corr_mwpm", "mwpm"]
SHIPPED = {"libra": "libra_decoder_with_rl_optimized_prior", "harmony": "harmony_decoder_with_rl_optimized_prior",
           "corrmatch": "correlated_matching_decoder_with_rl_optimized_prior"}
OURS = {"tesseract": ("rl_s5", "rl_s5t", "tesseract.npz"), "corr_mwpm": ("rl", "rl_s5", "corr_mwpm.npz"),
        "mwpm": ("rl", "rl_s5", "mwpm.npz")}
ML_FILE = {3: ("rl", "ml_exact.npz"), 5: ("rl_s5", "ml_tau2_1e-6-1e-8_0.1.npz")}
COMMON = [10, 30]
MISSING = 255
POPC = np.array([bin(i).count("1") for i in range(256)], np.uint8)
MARGIN_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.6, np.inf]


def read_leaf(arch, dec, d, patch, basis, r, chk):
    rel = rel_of(d, patch, basis, r)
    a_dir = os.path.join(arch, rel)
    sub, fn = ML_FILE[d]
    ml = np.load(os.path.join(dec, ARCH, rel, sub, fn), allow_pickle=False)
    idx = ml["idx"].astype(np.int64)
    act, pad = read_obs_b8(os.path.join(a_dir, "obs_flips_actual.b8"))
    N = act.size
    rec = {"d": d, "patch": patch, "basis": basis, "r": r, "n": int(idx.size), "obs": act[idx]}
    rec["ml"] = ml["pred"].reshape(idx.size, -1)[:, 0].astype(np.uint8)
    llr = ml["llr"].astype(np.float64)
    rec["llr"] = llr
    fin = np.isfinite(llr) & (llr != 0)
    chk["ml_pred_vs_llr_mismatch"] += int(np.sum(rec["ml"][fin] != (llr[fin] > 0)))
    chk["ml_llr_nan"] += int(np.isnan(llr).sum())
    chk["ml_idx_rule_ok"] &= bool(np.array_equal(idx, np.arange(0, N, 1 if d == 3 else 5)))
    rl_md5 = md5(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8)
    nb = raw.size // N
    n_det = (d * d - 1) * r
    if raw.size != N * nb or nb != (n_det + 7) // 8:
        raise SystemExit(f"detection_events.b8 size unexpected in {rel}")
    raw = raw.reshape(N, nb)[idx]
    if n_det % 8:
        chk["det_padding_nonzero"] += int(np.sum(raw[:, -1] >> (n_det % 8)))
    rec["w"] = POPC[raw].sum(1, dtype=np.int32)
    rec["n_det"] = n_det
    preds = {}
    for name, sh in SHIPPED.items():
        p = os.path.join(a_dir, "decoding_results", sh, "obs_flips_predicted.b8")
        if os.path.exists(p):
            v, _ = read_obs_b8(p)
            preds[name] = v[idx]
            dm = md5(os.path.join(a_dir, "decoding_results", sh, "error_model.dem"))
            chk["shipped_dem_not_rl"] += int(dm != rl_md5)
    sums = {}
    for name, (s3, s5, f) in OURS.items():
        s = s3 if d == 3 else s5
        p = os.path.join(dec, ARCH, rel, s, f)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        ii = z["idx"].astype(np.int64)
        full = np.full(N, MISSING, np.uint8)
        full[ii] = z["pred"].reshape(ii.size, -1)[:, 0].astype(np.uint8)
        preds[name] = full[idx]
        if s not in sums:
            sums[s] = json.load(open(os.path.join(dec, ARCH, rel, s, "summary.json"))).get("dem_md5")
        chk["own_run_dem_not_rl"] += int(sums[s] != rl_md5)
        chk["own_run_subset_not_every_5th_or_all"] += int(not (np.array_equal(ii, np.arange(0, N, 5))
                                                              or np.array_equal(ii, np.arange(N))))
    ms = sums.get(sub) or json.load(open(os.path.join(dec, ARCH, rel, sub, "summary.json"))).get("dem_md5")
    chk["ml_dem_not_rl"] += int(ms != rl_md5)
    rec["preds"] = preds
    # within-leaf syndrome-weight percentile (average ranks), on ML's shots
    rec["wq"] = stats.rankdata(rec["w"], method="average") / rec["w"].size
    return rec


def leaf_counts(rec, a, mask_extra=None):
    pa = rec["preds"][a]
    m = pa != MISSING
    if mask_extra is not None:
        m &= mask_extra
    aw = (pa != rec["obs"]) & m
    mw = (rec["ml"] != rec["obs"]) & m
    return {"n": int(m.sum()), "kA": int(aw.sum()), "kM": int(mw.sum()), "n11": int(np.sum(aw & mw)),
            "b": int(np.sum(aw & ~mw)), "c": int(np.sum(~aw & mw))}


def series_set(leaves, a, d, rounds, excluded):
    """paired() per series over the allowed rounds; series with fewer than 2 rounds are listed as lost."""
    prs, per, lost = [], [], []
    for p in (P3 if d == 3 else P5):
        for b in BASES:
            L = []
            for r in rounds:
                rec = leaves.get((d, p, b, r))
                if rec is None or a not in rec["preds"] or rel_of(d, p, b, r) in excluded:
                    continue
                cnt = leaf_counts(rec, a)
                if cnt["n"] > 0:
                    L.append((r, cnt))
            if len(L) < 2:
                lost.append(f"{p}/{b} ({len(L)} round{'s' if len(L) != 1 else ''})")
                continue
            rr = [x[0] for x in L]
            pr = paired(rr, [x[1]["kA"] for x in L], [x[1]["kM"] for x in L], [x[1]["n11"] for x in L],
                        [x[1]["n"] for x in L])
            pr["basis"] = b
            prs.append(pr)
            per.append({"patch": p, "basis": b, "rounds": rr, "shots_per_leaf": [x[1]["n"] for x in L],
                        "g_series": pr["d"] / pr["eps_ml"], "eps_ml": pr["eps_ml"], "eps_a": pr["eps_a_paired"],
                        "chi2_red_z": pr["chi2_red_z"]})
    return prs, per, lost


def gsum(prs):
    if not prs:
        return None
    o = g_stat(prs)
    keep = ("g", "series", "mean_eps_ml", "mean_eps_libra_paired", "series_with_d_positive", "se_inflated",
            "se_frozen_formula_inflated", "z", "p_two_sided", "ci90", "ci95")
    o = {k: o[k] for k in keep}
    o["mean_eps_a_paired"] = o.pop("mean_eps_libra_paired")
    return o


def growth(g3, g5):
    l3, l5 = math.log1p(g3["g"]), math.log1p(g5["g"])
    s3, s5 = g3["se_inflated"] / (1 + g3["g"]), g5["se_inflated"] / (1 + g5["g"])
    se = math.hypot(s3, s5)
    lr = l3 - l5  # ln of the Lambda ratio
    return {"lambda_ratio": math.exp(lr), "lambda_ratio_ci95": [math.exp(lr - Z95 * se), math.exp(lr + Z95 * se)],
            "growth_ln": l5 - l3, "growth_se": se, "growth_z": (l5 - l3) / se,
            "growth_p_one_sided": 0.5 * math.erfc(((l5 - l3) / se) / math.sqrt(2))}


def per_series_stats(per3, per5):
    x3 = np.array([s["g_series"] for s in per3])
    x5 = np.array([s["g_series"] for s in per5])
    mw = stats.mannwhitneyu(x5, x3, alternative="greater", method="exact")
    return {"d3_series": int(x3.size), "d5_series": int(x5.size), "d3_median": float(np.median(x3)),
            "d5_median": float(np.median(x5)), "d3_range": [float(x3.min()), float(x3.max())],
            "d5_range": [float(x5.min()), float(x5.max())],
            "d5_series_above_d3_median": int(np.sum(x5 > np.median(x3))),
            "d5_series_above_d3_max": int(np.sum(x5 > x3.max())),
            "mann_whitney_U": float(mw.statistic), "mann_whitney_p_one_sided": float(mw.pvalue),
            "auc_d5_above_d3": float(mw.statistic / (x3.size * x5.size))}


def discord_tables(leaves, a, d, rounds, excluded):
    """Shot-level arrays over the decoder's matched rounds (rule A at d = 5) and the binned counts."""
    cols = {k: [] for k in ("aw", "mw", "absllr", "wq", "w_per_det", "r", "basis_z")}
    for p in (P3 if d == 3 else P5):
        for b in BASES:
            for r in rounds:
                rec = leaves.get((d, p, b, r))
                if rec is None or a not in rec["preds"] or rel_of(d, p, b, r) in excluded:
                    continue
                pa = rec["preds"][a]
                m = pa != MISSING
                cols["aw"].append((pa != rec["obs"])[m])
                cols["mw"].append((rec["ml"] != rec["obs"])[m])
                cols["absllr"].append(np.abs(rec["llr"][m]))
                cols["wq"].append(rec["wq"][m])
                cols["w_per_det"].append(rec["w"][m] / rec["n_det"])
                cols["r"].append(np.full(int(m.sum()), r))
                cols["basis_z"].append(np.full(int(m.sum()), b == "Z"))
    if not cols["aw"]:
        return None
    A = {k: np.concatenate(v) for k, v in cols.items()}
    return A


def counts_in(A, sel):
    aw, mw = A["aw"][sel], A["mw"][sel]
    b, c = int(np.sum(aw & ~mw)), int(np.sum(~aw & mw))
    pr = 1.0 / (1.0 + np.exp(-A["absllr"][sel][aw != mw]))
    return {"shots": int(sel.sum()), "k_ml": int(mw.sum()), "k_a": int(aw.sum()), "b": b, "c": c, "net": b - c,
            "disagree_rate": (b + c) / max(int(sel.sum()), 1),
            "share_b_observed": b / (b + c) if b + c else None,
            "share_b_predicted_by_ml": float(pr.mean()) if pr.size else None}


def discord_summary(A):
    tot = counts_in(A, np.ones(A["aw"].size, bool))
    out = {"all": tot, "by_margin": [], "by_weight_quartile": [], "by_r": [], "by_basis": []}
    for lo, hi in zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:]):
        sel = (A["absllr"] >= lo) & (A["absllr"] < hi)
        o = counts_in(A, sel)
        o.update(bin=[lo, None if hi == np.inf else hi],
                 share_of_net=o["net"] / tot["net"] if tot["net"] else None,
                 share_of_disagreements=(o["b"] + o["c"]) / max(tot["b"] + tot["c"], 1))
        out["by_margin"].append(o)
    for q, (lo, hi) in enumerate(((0, .25), (.25, .5), (.5, .75), (.75, 1.0001))):
        sel = (A["wq"] > lo) & (A["wq"] <= hi)
        o = counts_in(A, sel)
        o.update(quartile=q + 1, share_of_net=o["net"] / tot["net"] if tot["net"] else None,
                 mean_w_per_det=float(A["w_per_det"][sel].mean()))
        out["by_weight_quartile"].append(o)
    for r in sorted(set(A["r"].tolist())):
        o = counts_in(A, A["r"] == r)
        o.update(r=int(r), net_over_k_ml=o["net"] / o["k_ml"] if o["k_ml"] else None)
        out["by_r"].append(o)
    for bz, name in ((False, "X"), (True, "Z")):
        o = counts_in(A, A["basis_z"] == bz)
        o.update(basis=name, net_over_k_ml=o["net"] / o["k_ml"] if o["k_ml"] else None)
        out["by_basis"].append(o)
    dis = A["aw"] != A["mw"]
    out["mean_w_per_det"] = {"all": float(A["w_per_det"].mean()), "disagreements": float(A["w_per_det"][dis].mean()),
                             "b": float(A["w_per_det"][A["aw"] & ~A["mw"]].mean()) if tot["b"] else None,
                             "c": float(A["w_per_det"][~A["aw"] & A["mw"]].mean()) if tot["c"] else None}
    out["median_abs_llr"] = {"all": float(np.median(A["absllr"])),
                             "disagreements": float(np.median(A["absllr"][dis])) if dis.any() else None}
    return out


def overlap_with_tesseract(leaves, a, d, rounds, excluded):
    """On shots where both A and Tesseract have predictions: A's and Tesseract's net loss split by agreement."""
    cat = {}
    for p in (P3 if d == 3 else P5):
        for b in BASES:
            for r in rounds:
                rec = leaves.get((d, p, b, r))
                if rec is None or rel_of(d, p, b, r) in excluded or a not in rec["preds"] \
                        or "tesseract" not in rec["preds"]:
                    continue
                pa, pt = rec["preds"][a], rec["preds"]["tesseract"]
                m = (pa != MISSING) & (pt != MISSING)
                o, ml = rec["obs"][m], rec["ml"][m]
                a_dis, t_dis = pa[m] != ml, pt[m] != ml
                mw = ml != o
                for key, sel in (("A=ML,T=ML", ~a_dis & ~t_dis), ("A!=ML,T=ML", a_dis & ~t_dis),
                                 ("A=ML,T!=ML", ~a_dis & t_dis), ("A!=ML,T!=ML", a_dis & t_dis)):
                    x = cat.setdefault(key, {"shots": 0, "k_ml": 0, "A_b": 0, "A_c": 0, "T_b": 0, "T_c": 0})
                    x["shots"] += int(sel.sum())
                    x["k_ml"] += int(np.sum(mw & sel))
                    aw, tw = (pa[m] != o) & sel, (pt[m] != o) & sel
                    x["A_b"] += int(np.sum(aw & ~mw))
                    x["A_c"] += int(np.sum(~aw & mw & sel))
                    x["T_b"] += int(np.sum(tw & ~mw))
                    x["T_c"] += int(np.sum(~tw & mw & sel))
    if not cat:
        return None
    for x in cat.values():
        x["A_net"], x["T_net"] = x["A_b"] - x["A_c"], x["T_b"] - x["T_c"]
    An = sum(x["A_net"] for x in cat.values())
    Tn = sum(x["T_net"] for x in cat.values())
    kml = sum(x["k_ml"] for x in cat.values())
    a_dis = cat["A!=ML,T=ML"]["shots"] + cat["A!=ML,T!=ML"]["shots"]
    t_dis = cat["A=ML,T!=ML"]["shots"] + cat["A!=ML,T!=ML"]["shots"]
    return {"categories": cat, "shots": sum(x["shots"] for x in cat.values()), "k_ml": kml,
            "A_net": An, "A_net_over_k_ml": An / kml if kml else None,
            "T_net": Tn, "T_net_over_k_ml": Tn / kml if kml else None,
            "A_net_share_where_T_also_disagrees": cat["A!=ML,T!=ML"]["A_net"] / An if An else None,
            "T_net_share_where_A_also_disagrees": cat["A!=ML,T!=ML"]["T_net"] / Tn if Tn else None,
            "P_T_disagrees_given_A_disagrees": cat["A!=ML,T!=ML"]["shots"] / a_dis if a_dis else None,
            "P_A_disagrees_given_T_disagrees": cat["A!=ML,T!=ML"]["shots"] / t_dis if t_dis else None}


def main(arch, dec, k2_dir, out_dir):
    t0 = time.time()
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = k2_verdicts(k2_dir)
    k2_read = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    leaves = {}
    for d, pats in ((3, P3), (5, P5)):
        for p in pats:
            for b in BASES:
                for r in W5:
                    leaves[(d, p, b, r)] = read_leaf(arch, dec, d, p, b, r, chk)
        print(f"d = {d}: leaves read ({time.time() - t0:.0f} s)", flush=True)
    avail = {a: {d: sorted({r for (dd, p, b, r), rec in leaves.items() if dd == d and a in rec["preds"]})
                 for d in (3, 5)} for a in DECODERS}
    shots = {a: {d: sorted({int(np.sum(rec["preds"][a] != MISSING)) for (dd, p, b, r), rec in leaves.items()
                            if dd == d and a in rec["preds"]}) for d in (3, 5)} for a in DECODERS}
    k2_state = {"read_at": k2_read, "verdicts_found": len(k2),
                "still_fail": sorted(failing), "pass": sorted(k for k, v in k2.items() if v["verdict"] == "pass")}
    res = {"label": "EXPLORATORY, post hoc (reservoir, 2026-10-10): matched gap table, item 1 of main's brief. "
                    "Measured; interim (K2 verdicts as found at run time).",
           "k2": k2_state, "checks": chk, "coverage": {}, "table": {}}
    for a in DECODERS:
        matched = [r for r in W5 if r in avail[a][3] and r in avail[a][5]]
        res["coverage"][a] = {"rounds_d3_in_W5": avail[a][3], "rounds_d5_in_W5": avail[a][5],
                              "matched_rounds": matched, "shots_per_leaf_d3": shots[a][3],
                              "shots_per_leaf_d5": shots[a][5],
                              "source": "shipped (archive)" if a in SHIPPED else "decoder's runs (decode/)"}
    ser_rows = []
    for a in DECODERS:
        matched = res["coverage"][a]["matched_rounds"]
        res["table"][a] = {}
        for vname, rounds in (("matched", matched), ("common_10_30", COMMON)):
            for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
                prs3, per3, lost3 = series_set(leaves, a, 3, rounds, set())
                prs5, per5, lost5 = series_set(leaves, a, 5, rounds, excl)
                g3, g5 = gsum(prs3), gsum(prs5)
                o = {"rounds": rounds, "g3": g3, "g5": g5, "lost_series_d3": lost3, "lost_series_d5": lost5}
                if g3 and g5:
                    o.update(growth(g3, g5))
                    o["per_series"] = per_series_stats(per3, per5)
                    o["by_basis"] = {}
                    for bb in BASES:
                        q3 = gsum([x for x in prs3 if x["basis"] == bb])
                        q5 = gsum([x for x in prs5 if x["basis"] == bb])
                        if q3 and q5:
                            o["by_basis"][bb] = {"g3": q3["g"], "g3_se": q3["se_inflated"], "g5": q5["g"],
                                                 "g5_se": q5["se_inflated"], "series_d3": q3["series"],
                                                 "series_d5": q5["series"]}
                res["table"][a][f"{vname}|{kname}"] = o
                for d, per in ((3, per3), (5, per5)):
                    for s in per:
                        ser_rows.append({"decoder": a, "rounds_set": vname, "k2": kname, "d": d, **s,
                                         "rounds": " ".join(map(str, s["rounds"])),
                                         "shots_per_leaf": " ".join(map(str, s["shots_per_leaf"]))})
        print(f"table {a} done ({time.time() - t0:.0f} s)", flush=True)
    res["run_seconds_table"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "gap_table.json")), "w") as f:
        json.dump(res, f, indent=1)
    keys = []
    for s in ser_rows:
        for k in s:
            if k not in keys:
                keys.append(k)
    with open(out_path(os.path.join(out_dir, "gap_series.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(ser_rows)

    # item 2: discordant shots
    dres = {"label": "EXPLORATORY, post hoc (reservoir, 2026-10-10): discordant shots, item 2 of main's brief. Counts "
                     "measured; any mechanism is inferred. b = decoder wrong and ML right; c = decoder right and ML "
                     "wrong; net = b - c. Matched rounds, rule A at d = 5.",
            "k2": k2_state, "margin_edges": [x if x != np.inf else None for x in MARGIN_EDGES], "decoders": {},
            "overlap_with_tesseract": {}}
    bin_rows = []
    for a in DECODERS:
        rounds = res["coverage"][a]["matched_rounds"]
        dres["decoders"][a] = {"rounds": rounds}
        for d in (3, 5):
            A = discord_tables(leaves, a, d, rounds, failing if d == 5 else set())
            if A is None:
                continue
            sm = discord_summary(A)
            dres["decoders"][a][f"d{d}"] = sm
            for part in ("by_margin", "by_weight_quartile", "by_r", "by_basis"):
                for o in sm[part]:
                    lab = {"by_margin": lambda o: f"|llr| {o['bin'][0]}-{o['bin'][1]}",
                           "by_weight_quartile": lambda o: f"weight Q{o['quartile']}",
                           "by_r": lambda o: f"r{o['r']}", "by_basis": lambda o: o["basis"]}[part](o)
                    bin_rows.append({"decoder": a, "d": d, "split": part, "bin": lab,
                                     **{k: o[k] for k in ("shots", "k_ml", "k_a", "b", "c", "net", "disagree_rate",
                                                          "share_b_observed", "share_b_predicted_by_ml")},
                                     "share_of_net": o.get("share_of_net")})
        if a != "tesseract":
            dres["overlap_with_tesseract"][a] = {}
            for d in (3, 5):
                rr = [r for r in rounds if r in res["coverage"]["tesseract"]["matched_rounds"]]
                for tag, rset in (("shared_matched_rounds", rr), ("common_10_30", COMMON)):
                    ov = overlap_with_tesseract(leaves, a, d, rset, failing if d == 5 else set())
                    if ov:
                        ov["rounds"] = rset
                        dres["overlap_with_tesseract"][a][f"d{d}|{tag}"] = ov
    dres["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "discord.json")), "w") as f:
        json.dump(dres, f, indent=1)
    keys = []
    for s in bin_rows:
        for k in s:
            if k not in keys:
                keys.append(k)
    with open(out_path(os.path.join(out_dir, "discord_bins.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(bin_rows)
    print(f"done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:5])
