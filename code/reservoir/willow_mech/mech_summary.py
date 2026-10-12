"""EXPLORATORY (not pre-registered): why is the RL DEM miscalibrated at d = 3 (P3)? Step 2, statistics.

Reads the per-leaf files of mech_leaves.py (LEAVES, a scratch folder) and writes to OUT_DIR
(results/reservoir/willow_mech/, through qrc.outpaths: a re-run writes to results/reservoir/rebuild/ unless
QRC_OVERWRITE=1):
  events_leaves.csv   task 1 per leaf: data against the leaf's RL DEM samples (event count N, burst B, layers, windows)
  events_by_r.csv     task 1 pooled over the 18 leaves of each round count
  blocks.csv          task 3 per leaf and block: events, ML(RL) errors, sum min(P, 1 - P), Libra errors, ln P(s), ...
  mech.json           tasks 1-3 pooled: tails, windows, overconfident failures (task 2) with the calibration of simple
                      syndromes (exact N; N <= 2 by layer; by basis), block heterogeneity, trend and
                      member effects, the same-bitstring check, within-block shot order (autocorrelation of N,
                      runs of extreme shots and of failures, the neighbours of confident failures, adjacent blocks
                      within against between complement pairs), and the P3 reproduction (R) as a sanity check
Every number here is exploratory: it was not pre-registered, and the tests are not corrected for multiplicity.
With STREAKS (per-shot streaks written by mech_pairs.py), task 2 is also broken down by the streak S (the largest
number of events of one stabilizer within 4 consecutive bulk rounds, a leakage-like signature).
Usage: nice -n 19 python3 -I mech_summary.py LEAVES OUT_DIR [STREAKS]
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import out_path  # noqa: E402

W5 = [10, 13, 30, 50]
CONF = 3e-3  # "nearly certain": predicted error min(P, 1 - P) below this
LABEL = "EXPLORATORY, not pre-registered (reservoir, 2026-10-09); no multiplicity correction"


# ----------------------------------------------------------------------------------------------- helpers
def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def hstats(h):
    """n, mean, variance, kurtosis of a histogram over 0, 1, 2, ..."""
    h = np.asarray(h, float)
    x = np.arange(len(h))
    n = h.sum()
    mu = (x * h).sum() / n
    v = ((x - mu) ** 2 * h).sum() / n
    m4 = ((x - mu) ** 4 * h).sum() / n
    return n, mu, v, (m4 / v ** 2 if v > 0 else float("nan"))


def upper(hd, hm, level):
    """Threshold q = the DEM's `level` quantile; shares above q (strictly) in data and DEM."""
    cm = np.cumsum(hm) / hm.sum()
    q = int(np.argmax(cm >= level))
    sm = 1 - cm[q]
    sd = hd[q + 1:].sum() / hd.sum()
    return q, sd, sm


def lower(hd, hm, level):
    """Threshold q = the largest value with P_DEM(N <= q) <= level (None if P_DEM(N = 0) > level)."""
    cm = np.cumsum(hm) / hm.sum()
    ok = np.flatnonzero(cm <= level)
    if not len(ok):
        return None, float("nan"), float("nan")
    q = int(ok[-1])
    return q, hd[: q + 1].sum() / hd.sum(), cm[q]


def pad(h, n):
    return np.pad(np.asarray(h, float), (0, max(0, n - len(h))))[:n]


def oe(obs, exp, var_exp=0.0):
    """Observed / expected with an SE from Poisson(obs) and the DEM sampling variance of exp."""
    if exp <= 0:
        return {"O": obs, "E": exp, "ratio": float("nan"), "se": float("nan")}
    rt = obs / exp
    return {"O": float(obs), "E": float(exp), "ratio": rt, "se": rt * math.sqrt(1 / max(obs, 1) + var_exp / exp ** 2)}


def wls(X, y, s):
    """Weighted least squares with known SEs s; returns coefficients, their SEs and the residual chi2."""
    w = 1 / s ** 2
    A = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(A)
    beta = cov @ (X.T @ (w * y))
    res = y - X @ beta
    return beta, np.sqrt(np.diag(cov)), float(np.sum(w * res ** 2))


def fixed_effect(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    m = float(np.sum(w * est) / np.sum(w))
    s = float(1 / math.sqrt(np.sum(w)))
    q = float(np.sum(w * (est - m) ** 2))
    return {"estimate": m, "se": s, "z": m / s, "Q_heterogeneity": q, "dof": len(est) - 1,
            "p_heterogeneity": float(stats.chi2.sf(q, len(est) - 1)) if len(est) > 1 else float("nan")}


def chi2_pool(chis, dofs):
    c, d = float(np.sum(chis)), float(np.sum(dofs))
    return {"chi2": c, "dof": d, "ratio": c / d, "z": (c - d) / math.sqrt(2 * d), "p": float(stats.chi2.sf(c, d))}


# ----------------------------------------------------------------------------------------------- task 1
def task1(L):
    rows = []
    for d in L:
        mt, r = d["meta"], d["meta"]["r"]
        hd = np.bincount(d["N"].astype(np.int64), minlength=8 * r + 1)[: 8 * r + 1].astype(float)
        hm = d["dem_hist_N"].astype(float)
        nd, md, vd, kd = hstats(hd)
        nm, mm, vm, km = hstats(hm)
        row = {"patch": mt["patch"], "basis": mt["basis"], "r": r, "w5": r in W5, "n_data": int(nd), "n_dem": int(nm),
               "mean_data": md, "mean_dem": mm, "rel_mean": md / mm - 1,
               "se_rel_mean": math.sqrt(vd / nd / mm ** 2 + md ** 2 * vm / nm / mm ** 4),
               "var_data": vd, "var_dem": vm, "var_ratio": vd / vm,
               "se_ln_var_ratio": math.sqrt((kd - 1) / nd + (km - 1) / nm),
               "fano_data": vd / md, "fano_dem": vm / mm,
               "p0_data": hd[0] / nd, "p0_dem": hm[0] / nm}
        for lev, tag in ((0.99, "99"), (0.999, "999")):
            q, sd, sm = upper(hd, hm, lev)
            row.update({f"N_q{tag}_dem": q, f"N_above_q{tag}_data": sd, f"N_above_q{tag}_dem": sm})
        q, sd, sm = lower(hd, hm, 0.01)
        row.update({"N_qlo01_dem": q, "N_at_or_below_qlo01_data": sd, "N_at_or_below_qlo01_dem": sm})
        for key in ("B", "Bb"):
            hdB = np.bincount(d[key].astype(np.int64), minlength=9)[:9].astype(float)
            hmB = d[f"dem_hist_{key}"].astype(float)
            for lev, tag in ((0.99, "99"), (0.999, "999")):
                q, sd, sm = upper(hdB, hmB, lev)
                row.update({f"{key}_q{tag}_dem": q, f"{key}_above_q{tag}_data": sd, f"{key}_above_q{tag}_dem": sm})
            for k in (3, 4, 5, 6):
                row[f"{key}_ge{k}_data"] = hdB[k:].sum() / hdB.sum()
                row[f"{key}_ge{k}_dem"] = hmB[k:].sum() / hmB.sum()
        ld, lm = d["layer_mean_data"], d["layer_mean_dem"]
        row.update({"first_layer_rel": ld[0] / lm[0] - 1, "last_layer_rel": ld[r] / lm[r] - 1,
                    "bulk_rel": (ld[1:r].sum() / lm[1:r].sum() - 1) if r > 1 else float("nan")})
        # windows: shot-level heterogeneity that persists across the gap (DEM: Cov = 0 exactly)
        if mt["windows"]:
            A, B = d["NA"].astype(float), d["NB"].astype(float)
            Ac, Bc = A - A.mean(), B - B.mean()
            prod = Ac * Bc
            blk = d["block"].astype(int)
            Aw = A - np.bincount(blk, A)[blk] / np.bincount(blk)[blk]
            Bw = B - np.bincount(blk, B)[blk] / np.bincount(blk)[blk]
            pw = Aw * Bw
            mo = d["dem_mom_window"]
            n_m = mo[5]
            ea, eb = mo[0] / n_m, mo[1] / n_m
            cov_m = mo[4] / n_m - ea * eb
            va_m, vb_m = mo[2] / n_m - ea ** 2, mo[3] / n_m - eb ** 2
            norm = A.mean() * B.mean()
            row.update({"win_mean_A": A.mean(), "win_mean_B": B.mean(),
                        "win_cov_data": prod.mean(), "win_se_cov_data": prod.std(ddof=1) / math.sqrt(len(prod)),
                        "win_cov_within_block": pw.mean(), "win_se_cov_within_block": pw.std(ddof=1) / math.sqrt(len(pw)),
                        "win_cov_dem": cov_m, "win_se_cov_dem": math.sqrt(va_m * vb_m / n_m),
                        "win_sigma2_rel": prod.mean() / norm, "win_se_sigma2_rel": prod.std(ddof=1) / math.sqrt(len(prod)) / norm,
                        "win_sigma2_rel_within_block": pw.mean() / norm,
                        "win_se_sigma2_rel_within_block": pw.std(ddof=1) / math.sqrt(len(pw)) / norm,
                        "win_corr_data": float(np.corrcoef(A, B)[0, 1])})
        rows.append(row)

    by_r = []
    for r in sorted({x["r"] for x in rows}):
        R = [x for x in rows if x["r"] == r]
        S = len(R)
        out = {"r": r, "leaves": S,
               "rel_mean": float(np.mean([x["rel_mean"] for x in R])),
               "se_rel_mean": math.sqrt(sum(x["se_rel_mean"] ** 2 for x in R)) / S,
               "rel_mean_min": min(x["rel_mean"] for x in R), "rel_mean_max": max(x["rel_mean"] for x in R),
               "leaves_mean_below_dem_2se": sum(x["rel_mean"] < -2 * x["se_rel_mean"] for x in R),
               "leaves_mean_above_dem_2se": sum(x["rel_mean"] > 2 * x["se_rel_mean"] for x in R),
               "var_ratio_geomean": math.exp(np.mean([math.log(x["var_ratio"]) for x in R])),
               "se_ln_var_ratio": math.sqrt(sum(x["se_ln_var_ratio"] ** 2 for x in R)) / S,
               "var_ratio_min": min(x["var_ratio"] for x in R), "var_ratio_max": max(x["var_ratio"] for x in R),
               "leaves_var_above_dem_2se": sum(math.log(x["var_ratio"]) > 2 * x["se_ln_var_ratio"] for x in R),
               "fano_data": float(np.mean([x["fano_data"] for x in R])), "fano_dem": float(np.mean([x["fano_dem"] for x in R])),
               "p0_data": float(np.mean([x["p0_data"] for x in R])), "p0_dem": float(np.mean([x["p0_dem"] for x in R])),
               "first_layer_rel": float(np.mean([x["first_layer_rel"] for x in R])),
               "last_layer_rel": float(np.mean([x["last_layer_rel"] for x in R]))}
        out["bulk_rel"] = float(np.mean([x["bulk_rel"] for x in R])) if r > 1 else float("nan")
        out["var_ratio_median"] = float(np.median([x["var_ratio"] for x in R]))
        for tag in ("99", "999"):
            out[f"N_above_q{tag}_ratio_median"] = float(np.median([x[f"N_above_q{tag}_data"] / x[f"N_above_q{tag}_dem"] for x in R]))
        for key in ("N", "B", "Bb"):
            for tag in ("99", "999"):
                O = sum(x[f"{key}_above_q{tag}_data"] * x["n_data"] for x in R)
                E = sum(x[f"{key}_above_q{tag}_dem"] * x["n_data"] for x in R)
                vE = sum(x["n_data"] ** 2 * x[f"{key}_above_q{tag}_dem"] * (1 - x[f"{key}_above_q{tag}_dem"]) / x["n_dem"] for x in R)
                o = oe(O, E, vE)
                out.update({f"{key}_above_q{tag}_share_data": O / sum(x["n_data"] for x in R),
                            f"{key}_above_q{tag}_share_dem": E / sum(x["n_data"] for x in R),
                            f"{key}_above_q{tag}_ratio": o["ratio"], f"{key}_above_q{tag}_ratio_se": o["se"]})
        Rl = [x for x in R if x["N_qlo01_dem"] is not None]
        if Rl:
            O = sum(x["N_at_or_below_qlo01_data"] * x["n_data"] for x in Rl)
            E = sum(x["N_at_or_below_qlo01_dem"] * x["n_data"] for x in Rl)
            o = oe(O, E)
            out.update({"N_lower01_leaves": len(Rl), "N_lower01_ratio": o["ratio"], "N_lower01_ratio_se": o["se"]})
        for k in (3, 4, 5, 6):
            O = sum(x[f"B_ge{k}_data"] * x["n_data"] for x in R)
            E = sum(x[f"B_ge{k}_dem"] * x["n_data"] for x in R)
            o = oe(O, E)
            out.update({f"B_ge{k}_share_data": O / sum(x["n_data"] for x in R), f"B_ge{k}_ratio": o["ratio"],
                        f"B_ge{k}_ratio_se": o["se"]})
        if r > 1 and all("win_cov_data" in x for x in R):
            for tag in ("", "_within_block"):
                fe = fixed_effect([x["win_sigma2_rel" + tag] for x in R], [x["win_se_sigma2_rel" + tag] for x in R])
                out.update({"win_sigma2_rel" + tag: fe["estimate"], "win_se_sigma2_rel" + tag: fe["se"],
                            "win_z" + tag: fe["z"]})
            zc = [x["win_cov_dem"] / x["win_se_cov_dem"] for x in R]
            out["win_dem_control_stouffer_z"] = float(np.sum(zc) / math.sqrt(S))
            out["win_corr_data_mean"] = float(np.mean([x["win_corr_data"] for x in R]))
        by_r.append(out)
    return rows, by_r


# ----------------------------------------------------------------------------------------------- task 2
def task2(L, streaks=None):
    """Overconfident ML(RL) failures (min(P, 1 - P) < 3e-3) in the W5 leaves, against all shots and against the
    confident successes of the same leaves; and observed / DEM-predicted failures among confident shots by feature."""
    feats = ["N", "B", "Bb", "F", "L"]
    res = {"definition": f"confident: min(P, 1 - P) < {CONF}; OF = confident and ML(RL) wrong; CS = confident and right",
           "by_r": {}, "pooled": {}}
    of_dev = {f: [] for f in feats + ["logp", "pct_N", "pct_N_cs", "zN", "zB", "zlogp"]}
    cs_dev = {f: np.zeros(3) for f in feats}  # sum, sum of squares, count of (x - leaf mean) over CS shots
    per_r, conf_rows = {}, []
    tot = {"shots": 0, "conf": 0, "OF": 0, "pred_conf": 0.0, "errors": 0, "pred_all": 0.0}
    for d in L:
        r = d["meta"]["r"]
        if r not in W5:
            continue
        m, fail = d["m"], d["fail_ml"]
        conf = m < CONF
        of, cs = conf & fail, conf & ~fail
        pr = per_r.setdefault(r, {"shots": 0, "conf": 0, "OF": 0, "pred_conf": 0.0,
                                  "sum": {g: {f: 0.0 for f in feats} for g in ("OF", "CS", "ALL")},
                                  "cnt": {g: 0 for g in ("OF", "CS", "ALL")},
                                  "share": {g: {k: 0 for k in ("F1", "L1", "B3", "B4")} for g in ("OF", "CS", "ALL")}})
        for k, v in (("shots", len(m)), ("conf", int(conf.sum())), ("OF", int(of.sum())), ("pred_conf", float(m[conf].sum()))):
            pr[k] += v
            tot[k] += v
        tot["errors"] += int(fail.sum())
        tot["pred_all"] += float(m.sum())
        for g, sel in (("OF", of), ("CS", cs), ("ALL", np.ones(len(m), bool))):
            pr["cnt"][g] += int(sel.sum())
            for f in feats:
                pr["sum"][g][f] += float(d[f][sel].astype(float).sum())
            pr["share"][g]["F1"] += int((d["F"][sel] >= 1).sum())
            pr["share"][g]["L1"] += int((d["L"][sel] >= 1).sum())
            pr["share"][g]["B3"] += int((d["B"][sel] >= 3).sum())
            pr["share"][g]["B4"] += int((d["B"][sel] >= 4).sum())
        for f in feats:
            x = d[f].astype(float)
            mu = x.mean()
            of_dev[f].extend((x[of] - mu).tolist())
            dc = x[cs] - mu
            cs_dev[f] += [dc.sum(), (dc * dc).sum(), len(dc)]
            if f in ("N", "B"):
                of_dev["z" + f].extend(((x[of] - x[cs].mean()) / x[cs].std()).tolist())
        lp = d["logp"]
        of_dev["logp"].extend((lp[of] - lp[cs].mean()).tolist())
        of_dev["zlogp"].extend(((lp[of] - lp[cs].mean()) / lp[cs].std()).tolist())
        N = d["N"].astype(float)
        for key, ref in (("pct_N", np.sort(N)), ("pct_N_cs", np.sort(N[cs]))):
            lo, hi = np.searchsorted(ref, N[of], "left"), np.searchsorted(ref, N[of], "right")
            of_dev[key].extend(((lo + hi) / 2 / len(ref)).tolist())
        rank = np.argsort(np.argsort(N + np.random.default_rng(r).uniform(0, 1e-3, len(N))))
        quint = (rank * 5 // len(N)).astype(int)
        key = (d["meta"]["patch"], d["meta"]["basis"], r)
        have = bool(streaks) and key in streaks
        Sx = streaks[key]["S"] if have else np.full(len(m), -1, np.int8)
        Ku = streaks[key]["Ku"] if have else np.full(len(m), -1, np.int16)
        Ks2 = streaks[key]["Ks2"] if have else np.full(len(m), -1, np.int16)
        if streaks:
            sx = Sx.astype(float)
            of_dev.setdefault("S", []).extend((sx[of] - sx.mean()).tolist())
            of_dev.setdefault("zS", []).extend(((sx[of] - sx[cs].mean()) / sx[cs].std()).tolist())
            for g, sel in (("OF", of), ("CS", cs), ("ALL", np.ones(len(m), bool))):
                pr.setdefault("S", {}).setdefault(g, [0.0, 0, 0])
                pr["S"][g][0] += float(sx[sel].sum())
                pr["S"][g][1] += int((sx[sel] >= 3).sum())
                pr["S"][g][2] += int(sel.sum())
        conf_rows.append({"r": np.full(int(conf.sum()), r), "fail": fail[conf], "m": m[conf], "S": Sx[conf].astype(int),
                          "Ku": Ku[conf].astype(int), "Ks2": Ks2[conf].astype(int),
                          "B": d["B"][conf].astype(int),
                          "Bb": d["Bb"][conf].astype(int), "F": d["F"][conf].astype(int), "L": d["L"][conf].astype(int),
                          "quint": quint[conf], "block": d["block"][conf].astype(int),
                          "Bt_edge": np.isin(d["Bt"][conf], [0, 1, r - 1, r])})
    res["totals_w5"] = tot | {"R_all": tot["errors"] / tot["pred_all"], "R_conf": tot["OF"] / tot["pred_conf"],
                              "R_conf_se": math.sqrt(tot["OF"]) / tot["pred_conf"]}
    for r, pr in sorted(per_r.items()):
        row = {"shots": pr["shots"], "confident_shots": pr["conf"], "OF": pr["OF"], "pred_failures_confident": pr["pred_conf"],
               "R_conf": pr["OF"] / pr["pred_conf"], "R_conf_se": math.sqrt(max(pr["OF"], 1)) / pr["pred_conf"]}
        for g in ("OF", "CS", "ALL"):
            c = max(pr["cnt"][g], 1)
            for f in feats:
                row[f"mean_{f}_{g}"] = pr["sum"][g][f] / c
            for k, lab in (("F1", "F_ge1"), ("L1", "L_ge1"), ("B3", "B_ge3"), ("B4", "B_ge4")):
                row[f"share_{lab}_{g}"] = pr["share"][g][k] / c
            if "S" in pr:
                row[f"mean_S_{g}"] = pr["S"][g][0] / max(pr["S"][g][2], 1)
                row[f"share_S_ge3_{g}"] = pr["S"][g][1] / max(pr["S"][g][2], 1)
        res["by_r"][str(r)] = row
    nOF = len(of_dev["N"])
    pooled = {"OF_shots": nOF}
    for f in feats:
        a = np.asarray(of_dev[f])
        s1, s2, n = cs_dev[f]
        pooled[f"OF_minus_leaf_mean_{f}"] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(nOF))}
        pooled[f"CS_minus_leaf_mean_{f}"] = {"mean": s1 / n, "se": math.sqrt(max(s2 / n - (s1 / n) ** 2, 0) / n)}
    if "S" in of_dev:
        a = np.asarray(of_dev["S"])
        pooled["OF_minus_leaf_mean_S"] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(nOF))}
    for f in ("zN", "zB", "zlogp") + (("zS",) if "zS" in of_dev else ()):
        a = np.asarray(of_dev[f])
        pooled[f"OF_vs_CS_standardised_{f[1:]}"] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(nOF))}
    a = np.asarray(of_dev["logp"])
    pooled["OF_minus_CS_lnP_nats"] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(nOF))}
    for f, ref in (("pct_N", "all_shots"), ("pct_N_cs", "confident_successes")):
        a = np.asarray(of_dev[f])
        pooled[f"OF_mean_percentile_of_N_among_{ref}"] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(nOF)),
                                                          "null": 0.5}
    res["pooled"] = pooled
    cr = {k: np.concatenate([x[k] for x in conf_rows]) for k in conf_rows[0]}

    def bins(labels, sels):
        out = []
        for lab, sel in zip(labels, sels):
            O, E = int(cr["fail"][sel].sum()), float(cr["m"][sel].sum())
            out.append({"bin": lab, "confident_shots": int(sel.sum()), "observed": O, "predicted": E,
                        "ratio": O / E if E > 0 else float("nan"), "se": math.sqrt(max(O, 1)) / E if E > 0 else float("nan")})
        return out

    B, Bb, F, Lx, q = cr["B"], cr["Bb"], cr["F"], cr["L"], cr["quint"]
    res["confident_by_feature"] = {
        "B_max_events_in_one_layer": bins(["0-1", "2", "3", ">=4"], [B <= 1, B == 2, B == 3, B >= 4]),
        "Bb_bulk_layers_only": bins(["0-1", "2", "3", ">=4"], [Bb <= 1, Bb == 2, Bb == 3, Bb >= 4]),
        "F_first_layer_events": bins(["0", "1", ">=2"], [F == 0, F == 1, F >= 2]),
        "L_last_layer_events": bins(["0", "1", ">=2"], [Lx == 0, Lx == 1, Lx >= 2]),
        "N_quintile_within_leaf": bins(["1 (fewest)", "2", "3", "4", "5 (most)"], [q == k for k in range(5)]),
        "largest_layer_at_edge": bins(["layer 0, 1, r-1 or r", "elsewhere"], [cr["Bt_edge"], ~cr["Bt_edge"]]),
        "m": bins(["<1e-4", "1e-4 to 1e-3", "1e-3 to 3e-3"], [cr["m"] < 1e-4, (cr["m"] >= 1e-4) & (cr["m"] < 1e-3), cr["m"] >= 1e-3]),
        "r": bins([str(r) for r in W5], [cr["r"] == r for r in W5])}
    if streaks:
        Sx = cr["S"]
        res["confident_by_feature"]["S_streak_one_stabilizer_4_rounds"] = bins(["0-1", "2", "3", ">=4"],
                                                                            [Sx <= 1, Sx == 2, Sx == 3, Sx >= 4])
        Ku, Ks2 = cr["Ku"], cr["Ks2"]
        res["confident_by_feature"]["Ku_unconnected_coincidences_dt0to1"] = bins(["0", "1", ">=2"], [Ku == 0, Ku == 1, Ku >= 2])
        res["confident_by_feature"]["Ks2_same_stabilizer_dt2_coincidences"] = bins(["0", ">=1"], [Ks2 == 0, Ks2 >= 1])
    res["confident_by_block_position"] = bins([str(b) for b in range(10)], [cr["block"] == b for b in range(10)])
    return res


def calib_simple(L, rounds):
    """Observed / DEM-predicted ML(RL) failures (all shots and confident shots) by exact event count N, by where the
    events of N <= 2 shots lie (first layer, last layer, bulk), and by basis. A failure in a shot with N = 0 needs an
    undetectable logical fault; at N = 1 or 2 the posterior rests on a handful of mechanisms."""
    nb = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 20), (21, 10 ** 6)]
    cats = ["N0", "N1_first", "N1_last", "N1_bulk", "N2_first_only", "N2_last_only", "N2_bulk_only", "N2_mixed"]
    acc = {}

    def add(key, s, m, fl, conf):
        a = acc.setdefault(key, [0, 0, 0.0, 0, 0, 0.0])
        sc = s & conf
        a[0] += int(s.sum()); a[1] += int(fl[s].sum()); a[2] += float(m[s].sum())
        a[3] += int(sc.sum()); a[4] += int(fl[sc].sum()); a[5] += float(m[sc].sum())

    for d in L:
        if d["meta"]["r"] not in rounds:
            continue
        N, F, La = d["N"].astype(int), d["F"].astype(int), d["L"].astype(int)
        m, fl = d["m"], d["fail_ml"].astype(bool)
        conf = m < CONF
        Bk = N - F - La
        for lo, hi in nb:
            add(("N", f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}")), (N >= lo) & (N <= hi), m, fl, conf)
        sel = {"N0": N == 0, "N1_first": (N == 1) & (F == 1), "N1_last": (N == 1) & (La == 1), "N1_bulk": (N == 1) & (Bk == 1),
               "N2_first_only": (N == 2) & (F == 2), "N2_last_only": (N == 2) & (La == 2), "N2_bulk_only": (N == 2) & (Bk == 2),
               "N2_mixed": (N == 2) & (F < 2) & (La < 2) & (Bk < 2)}
        for c in cats:
            add(("layers", c), sel[c], m, fl, conf)
        b = d["meta"]["basis"]
        add(("basis", b), np.ones(len(N), bool), m, fl, conf)
        add(("basis", b + " N<=2"), N <= 2, m, fl, conf)
    out = {}
    for (g, k), a in acc.items():
        out.setdefault(g, []).append({"bin": k, "shots": a[0], "failures": a[1], "predicted": a[2],
                                      "ratio": a[1] / a[2] if a[2] else None,
                                      "ratio_se": (a[1] / a[2]) / math.sqrt(max(a[1], 1)) if a[2] else None,
                                      "confident_shots": a[3], "confident_failures": a[4], "confident_predicted": a[5],
                                      "confident_ratio": a[4] / a[5] if a[5] else None,
                                      "confident_ratio_se": (a[4] / a[5]) / math.sqrt(max(a[4], 1)) if a[5] else None})
    return out


# ----------------------------------------------------------------------------------------------- task 3
def task3(L):
    rows, leafstats = [], []
    for d in L:
        mt, r = d["meta"], d["meta"]["r"]
        blk = d["block"].astype(int)
        nb = np.bincount(blk)
        K = len(nb)
        N, m, fail, lp = d["N"].astype(float), d["m"], d["fail_ml"].astype(float), d["logp"]
        F, Lx = d["F"].astype(float), d["L"].astype(float)
        lib = d["fail_libra"].astype(float)
        has_lib = bool(np.all(lib >= 0))

        def bmean(x):
            return np.bincount(blk, x) / nb

        def bwvar(x):
            mu = bmean(x)[blk]
            return float(np.sum((x - mu) ** 2) / (len(x) - K))

        yN, ym, ylp, yF, yL = bmean(N), bmean(m), bmean(lp), bmean(F), bmean(Lx)
        e = np.bincount(blk, fail)
        p = np.bincount(blk, m)
        q2 = np.bincount(blk, m * m)
        l = np.bincount(blk, lib) if has_lib else np.full(K, np.nan)
        conf = m < CONF
        e_conf = np.bincount(blk, fail * conf, minlength=K)
        p_conf = np.bincount(blk, m * conf, minlength=K)
        code, wt = d["block_code"], d["block_weight"]
        for b in range(K):
            rows.append({"patch": mt["patch"], "basis": mt["basis"], "r": r, "w5": r in W5, "block": b,
                         "code": int(code[b]), "weight": int(wt[b]), "member": "even" if b % 2 == 0 else "odd",
                         "heavy": int(wt[b] >= 5), "shots": int(nb[b]), "mean_events": yN[b], "mean_F": yF[b], "mean_L": yL[b],
                         "ml_errors": int(e[b]), "sum_min_P": p[b], "R": e[b] / p[b],
                         "libra_errors": int(l[b]) if has_lib else "", "mean_lnP": ylp[b],
                         "confident_failures": int(e_conf[b]), "confident_predicted": p_conf[b]})
        s2N, s2m, s2lp, s2F, s2L = bwvar(N), bwvar(m), bwvar(lp), bwvar(F), bwvar(Lx)
        c_hat = e.sum() / p.sum()
        Vc = c_hat * p - c_hat ** 2 * q2
        eb = e.sum() / nb.sum()
        st = {"patch": mt["patch"], "basis": mt["basis"], "r": r, "blocks": K, "c_hat": c_hat,
              "chi2_events": float(np.sum(nb * (yN - N.mean()) ** 2) / s2N),
              "chi2_predicted": float(np.sum(nb * (ym - m.mean()) ** 2) / s2m),
              "chi2_lnP": float(np.sum(nb * (ylp - lp.mean()) ** 2) / s2lp),
              "chi2_first_layer": float(np.sum(nb * (yF - F.mean()) ** 2) / s2F),
              "chi2_last_layer": float(np.sum(nb * (yL - Lx.mean()) ** 2) / s2L),
              "chi2_ml_errors": float(np.sum((e - nb * eb) ** 2 / (nb * eb * (1 - eb)))),
              "chi2_calibration": float(np.sum((e - c_hat * p) ** 2 / Vc)),
              "tau2_rel_events": float((np.sum(nb * (yN - N.mean()) ** 2) / (K - 1) - s2N) / (nb.mean() * N.mean() ** 2))}
        if has_lib:
            lb = l.sum() / nb.sum()
            st["chi2_libra_errors"] = float(np.sum((l - nb * lb) ** 2 / (nb * lb * (1 - lb))))
        # trend (u = (b - 4.5)/9: change from block 0 to block 9) and member effects, relative scales
        u = (np.arange(K) - (K - 1) / 2) / (K - 1)
        par = np.where(np.arange(K) % 2 == 0, 0.5, -0.5)
        hv = np.where(wt >= 5, 0.5, -0.5)
        series = {"events": (yN / N.mean(), np.sqrt(s2N / nb) / N.mean()),
                  "ml_error_rate": (e / nb / eb, np.sqrt(eb * (1 - eb) / nb) / eb),
                  "predicted": (ym / m.mean(), np.sqrt(s2m / nb) / m.mean()),
                  "calibration": (e / p / c_hat, np.sqrt(Vc) / p / c_hat),
                  "lnP": (ylp - lp.mean(), np.sqrt(s2lp / nb)),
                  "first_layer": (yF / F.mean(), np.sqrt(s2F / nb) / F.mean()),
                  "last_layer": (yL / Lx.mean(), np.sqrt(s2L / nb) / Lx.mean())}
        if has_lib:
            series["libra_error_rate"] = (l / nb / lb, np.sqrt(lb * (1 - lb) / nb) / lb)
        for name, (y, s) in series.items():
            for tag, lab in (("parity", par), ("heavy", hv)):
                X = np.column_stack([np.ones(K), u, lab])
                beta, se, chi_res = wls(X, y, s)
                st[f"{name}_{tag}_trend"], st[f"{name}_{tag}_trend_se"] = float(beta[1]), float(se[1])
                st[f"{name}_{tag}_member"], st[f"{name}_{tag}_member_se"] = float(beta[2]), float(se[2])
                st[f"{name}_{tag}_chi2_resid"] = chi_res
            # does the block's calibration track its event rate? (slope of calibration on the event deviation)
        yc, sc = series["calibration"]
        x = yN / N.mean() - 1
        X = np.column_stack([np.ones(K), x])
        beta, se, _ = wls(X, yc, sc)
        st["calibration_on_events_slope"], st["calibration_on_events_slope_se"] = float(beta[1]), float(se[1])
        # same-bitstring blocks (leaves where one pair repeats): standardised differences of the event rate
        y, s = series["events"]
        same, other = [], []
        for i in range(K):
            for j in range(i + 1, K):
                zz = (y[i] - y[j]) ** 2 / (s[i] ** 2 + s[j] ** 2)
                (same if code[i] == code[j] else other).append(zz)
        st["same_code_pairs_z2"] = same
        st["other_pairs_mean_z2"] = float(np.mean(other))
        leafstats.append(st)
    return rows, leafstats


def shot_order(L, rounds):
    """Within-block structure in file order. Whether file order (and block order) is acquisition order is not known [U].
    (a) lag-k autocorrelation of the event count N inside blocks (standardised per block; the -1/n bias of a sample
        autocorrelation is added back), as a Stouffer z over leaves and as a mean coefficient;
    (b) consecutive shots that are both extreme (N above the leaf's 99th percentile), observed against expected from
        each block's own share (so drift between blocks does not count);
    (c) consecutive shots that are both ML(RL) failures, observed against expected from each block's own failure rate;
    (d) mean event count of the shots 1, 2 and 5 places away from a confident ML failure (m < 3e-3), in units of the
        block's SD, against the same for confident successes;
    (e) squared z of the difference between adjacent block means, within a complement pair (2k, 2k+1) against between
        pairs (2k+1, 2k+2): state dependence predicts within > between, smooth drift predicts within ~ between."""
    lags = [1, 2, 5, 10, 100, 1000]
    acc = {k: [] for k in lags}
    slow = {k: [] for k in lags}  # within-block autocovariance of N at lag k over mean(N)^2: the relative variance of a
    # rate multiplier that is shared by shots k apart (if the excess correlation is a common rate)
    O = E = Of = Ef = 0.0
    near = {k: [0.0, 0.0, 0.0, 0.0] for k in (1, 2, 5)}  # sum z near OF, count; sum z near CS, count
    within, between = [], []
    for d in L:
        if d["meta"]["r"] not in rounds:
            continue
        N = d["N"].astype(float)
        K = int(d["block"].max()) + 1
        Nb = N.reshape(K, -1)
        n_b = Nb.shape[1]
        Z = (Nb - Nb.mean(1, keepdims=True)) / Nb.std(1, keepdims=True)
        D = Nb - Nb.mean(1, keepdims=True)
        mu = float(N.mean())
        for k in lags:
            c = float(np.mean(Z[:, k:] * Z[:, :-k])) + 1.0 / n_b
            acc[k].append(c * math.sqrt(K * (n_b - k)))
            slow[k].append((float(np.mean(D[:, k:] * D[:, :-k])) + float(np.mean(D * D)) / n_b) / mu ** 2)
        q = np.quantile(N, 0.99)
        X = (Nb > q).astype(float)
        pb = X.mean(1)
        O += float(np.sum(X[:, 1:] * X[:, :-1]))
        E += float(np.sum((n_b - 1) * pb * pb))
        F = d["fail_ml"].astype(float).reshape(K, -1)
        fb = F.mean(1)
        Of += float(np.sum(F[:, 1:] * F[:, :-1]))
        Ef += float(np.sum((n_b - 1) * fb * fb))
        conf = (d["m"] < 3e-3).reshape(K, -1)
        of = conf & (F > 0)
        cs = conf & (F == 0)
        for k in near:
            for sel, j in ((of, 0), (cs, 2)):
                for sh in (k, -k):
                    if sh > 0:
                        zz = Z[:, sh:][sel[:, :-sh]]
                    else:
                        zz = Z[:, :sh][sel[:, -sh:]]
                    near[k][j] += float(zz.sum())
                    near[k][j + 1] += zz.size
        bm = Nb.mean(1)
        se = Nb.std(1) / math.sqrt(n_b)
        for b in range(K - 1):
            zz = (bm[b] - bm[b + 1]) ** 2 / (se[b] ** 2 + se[b + 1] ** 2)
            (within if b % 2 == 0 else between).append(zz)
    nl = len(acc[1])
    out = {"note": "file order assumed to be acquisition order within a block, and block index to be time order; neither is known [U]",
           "leaves": nl,
           "autocorr_N_stouffer_z": {str(k): float(np.sum(v) / math.sqrt(len(v))) for k, v in acc.items()},
           "autocorr_N_mean_coefficient": {str(k): float(np.mean(v) / math.sqrt(10 * (5000 - k))) for k, v in acc.items()},
           "slow_rate_var_rel": {str(k): {"mean": float(np.mean(v)), "se": float(np.std(v) / math.sqrt(len(v))),
                                          "median": float(np.median(v))} for k, v in slow.items()},
           "extreme_N_consecutive": {"observed_pairs": O, "expected_pairs": E, "ratio": O / E,
                                     "se": (O / E) / math.sqrt(max(O, 1))},
           "ml_failure_consecutive": {"observed_pairs": Of, "expected_pairs": Ef, "ratio": Of / Ef,
                                      "se": (Of / Ef) / math.sqrt(max(Of, 1))},
           "N_z_near_confident_failure": {str(k): {"near_OF_mean_z": v[0] / max(v[1], 1), "n_OF": v[1],
                                                   "near_OF_se": 1 / math.sqrt(max(v[1], 1)),
                                                   "near_CS_mean_z": v[2] / max(v[3], 1), "n_CS": v[3]}
                                          for k, v in near.items()},
           "adjacent_blocks_mean_z2": {"within_pair_2k_2k1": float(np.mean(within)),
                                       "between_pairs_2k1_2k2": float(np.mean(between)),
                                       "n_within": len(within), "n_between": len(between),
                                       "within_over_between": float(np.mean(within) / np.mean(between))}}
    return out


def pool_task3(LS, rounds):
    S = [x for x in LS if x["r"] in rounds]
    out = {"leaves": len(S), "note": "blocks are the 10 contiguous runs of 5,000 shots in file order; the trend uses the "
           "block index as time, but whether block order is acquisition order is not known [U]"}
    het = {}
    for key in ("events", "predicted", "lnP", "first_layer", "last_layer", "ml_errors", "calibration", "libra_errors"):
        ck = f"chi2_{key}"
        v = [x[ck] for x in S if ck in x]
        if v:
            het[key] = chi2_pool(v, [x["blocks"] - 1 for x in S if ck in x])
    out["heterogeneity_across_blocks"] = het
    out["tau_rel_events"] = {"mean_tau2": float(np.mean([x["tau2_rel_events"] for x in S])),
                             "sd_block_rate_rel": math.sqrt(max(0.0, float(np.mean([x["tau2_rel_events"] for x in S]))))}
    eff = {}
    for name in ("events", "ml_error_rate", "predicted", "calibration", "lnP", "first_layer", "last_layer", "libra_error_rate"):
        for tag in ("parity", "heavy"):
            k = f"{name}_{tag}"
            T = [x for x in S if f"{k}_trend" in x]
            if not T:
                continue
            tr = fixed_effect([x[f"{k}_trend"] for x in T], [x[f"{k}_trend_se"] for x in T])
            mb = fixed_effect([x[f"{k}_member"] for x in T], [x[f"{k}_member_se"] for x in T])
            ztr = np.array([x[f"{k}_trend"] / x[f"{k}_trend_se"] for x in T])
            zmb = np.array([x[f"{k}_member"] / x[f"{k}_member_se"] for x in T])
            eff[k] = {"trend_pooled": tr, "member_pooled": mb,
                      "trend_chi2_any_direction": chi2_pool(ztr ** 2, [1] * len(T)),
                      "member_chi2_any_direction": chi2_pool(zmb ** 2, [1] * len(T)),
                      "residual_chi2": chi2_pool([x[f"{k}_chi2_resid"] for x in T], [x["blocks"] - 3 for x in T])}
    out["trend_and_member"] = eff
    out["calibration_on_events_slope"] = fixed_effect([x["calibration_on_events_slope"] for x in S],
                                                      [x["calibration_on_events_slope_se"] for x in S])
    same = [z for x in S for z in x["same_code_pairs_z2"]]
    out["same_bitstring_check"] = {"leaves_with_repeat": sum(1 for x in S if x["same_code_pairs_z2"]),
                                   "same_code_pairs": len(same), "same_code_mean_z2": float(np.mean(same)) if same else None,
                                   "other_pairs_mean_z2_in_those_leaves": float(np.mean([x["other_pairs_mean_z2"] for x in S
                                                                                         if x["same_code_pairs_z2"]]))
                                   if same else None,
                                   "note": "z2 = squared standardised difference of block event rates; 1 under no heterogeneity"}
    return out


# ----------------------------------------------------------------------------------------------- main
def cross_leaf(rows1, rows3):
    """Spearman correlation, within each W5 round count, between a leaf's calibration ratio R and its event statistics;
    combined over the four round counts by Fisher z (18 leaves each)."""
    agg = {}
    for x in rows3:
        k = (x["patch"], x["basis"], x["r"])
        a = agg.setdefault(k, [0, 0.0])
        a[0] += x["ml_errors"]
        a[1] += x["sum_min_P"]
    E = {(x["patch"], x["basis"], x["r"]): x for x in rows1}
    feats = {"rel_mean": lambda x: x["rel_mean"], "var_ratio": lambda x: x["var_ratio"], "fano_data": lambda x: x["fano_data"],
             "N_above_q999_ratio": lambda x: x["N_above_q999_data"] / x["N_above_q999_dem"],
             "B_ge4_ratio": lambda x: x["B_ge4_data"] / x["B_ge4_dem"],
             "win_sigma2_rel": lambda x: x.get("win_sigma2_rel", float("nan")),
             "last_layer_rel": lambda x: x["last_layer_rel"], "first_layer_rel": lambda x: x["first_layer_rel"]}
    out = {}
    for f, fn in feats.items():
        zs, per = [], {}
        for r in W5:
            keys = [k for k in agg if k[2] == r]
            Rv = np.array([agg[k][0] / agg[k][1] for k in keys])
            v = np.array([fn(E[k]) for k in keys])
            rho = float(stats.spearmanr(Rv, v).statistic)
            per[str(r)] = rho
            zs.append(math.atanh(rho) * math.sqrt(len(keys) - 3))
        out[f] = {"rho_by_r": per, "combined_z": float(np.sum(zs) / math.sqrt(len(zs)))}
    return out


def write_csv(path, rows):
    keys = []
    for x in rows:
        for k in x:
            if k not in keys:
                keys.append(k)
    with open(out_path(path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(leaves_dir, out_dir, streak_dir=None):
    files = sorted(glob.glob(os.path.join(leaves_dir, "*.npz")))
    streaks = None
    if streak_dir:
        streaks = {}
        for f in sorted(glob.glob(os.path.join(streak_dir, "*.npz"))):
            p, b, rr = os.path.basename(f)[:-4].rsplit("_", 2)
            z = np.load(f)
            streaks[(p, b, int(rr[1:]))] = {k: z[k] for k in z.files}
    L = [load(f) for f in files]
    L.sort(key=lambda d: (d["meta"]["r"], d["meta"]["patch"], d["meta"]["basis"]))
    rows1, by_r = task1(L)
    write_csv(os.path.join(out_dir, "events_leaves.csv"), rows1)
    write_csv(os.path.join(out_dir, "events_by_r.csv"), by_r)
    t2 = task2(L, streaks)
    rows3, ls3 = task3(L)
    write_csv(os.path.join(out_dir, "blocks.csv"), rows3)
    W5rows = [x for x in rows1 if x["w5"]]
    res = {"label": LABEL, "leaves": len(L), "w5_leaves": len(W5rows),
           "inputs": {"leaves_dir": leaves_dir, "dem_samples_per_leaf": sorted({d["meta"]["n_dem"] for d in L}),
                      "stim": sorted({d["meta"]["stim"] for d in L})},
           "sanity": {"p3_R_reproduced": t2["totals_w5"]["R_all"], "p3_errors": t2["totals_w5"]["errors"],
                      "p3_predicted": t2["totals_w5"]["pred_all"],
                      "data_mean_events_vs_decoder_p_det_max_abs_diff": max(
                          abs(float(d["N"].mean()) - d["meta"]["decoder_p_det"] * 8 * d["meta"]["r"]) for d in L)},
           "task1_w5_pooled": {k: (float(np.mean([x[k] for x in W5rows])) if isinstance(W5rows[0][k], float) else None)
                               for k in ("rel_mean", "var_ratio", "fano_data", "fano_dem")},
           "task1_by_r": by_r, "task2": t2, "cross_leaf_R_vs_events_w5": cross_leaf(rows1, rows3),
           "task3_w5": pool_task3(ls3, W5), "task3_all_r_ge10": pool_task3(ls3, [r for r in sorted({x["r"] for x in ls3}) if r >= 10]),
           "task3_by_r": {str(r): pool_task3(ls3, [r]) for r in sorted({x["r"] for x in ls3})},
           "task2_calibration_simple_w5": calib_simple(L, W5),
           "task2_calibration_simple_r1": calib_simple(L, [1]),
           "task3_shot_order_w5": shot_order(L, W5),
           "task3_shot_order_by_r": {str(r): shot_order(L, [r]) for r in sorted({d["meta"]["r"] for d in L})}}
    with open(out_path(os.path.join(out_dir, "mech.json")), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"sanity": res["sanity"], "task1_w5_pooled": res["task1_w5_pooled"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
