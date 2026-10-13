#!/usr/bin/env python3
"""PR3's post hoc r = 10 follow-up: the analysis declared in notes/decoder/willow_prereg_r3.md section 10, before any
r = 10 part started, with amendment 10a (the tau4 rung). Descriptive only: no test and no p-value. PR3's kill stands,
and H7 stays not evaluated.

Inputs, all read-only: the follow-up's parts, r10_tally.json and, if it exists, r10_tau4_tally.json (--res); the
driver's final ledger (--ledger) and the tau4 ledger (--tau4-ledger, if tau4 ran); round 1's decode folders (--round1:
results/decoder/willow); final_round1.json (--final-r1: rule A's excluded leaves); round 1's final analysis.json
(--final-analysis: for cross-checks); PR3's leaf table (--pr3-leaf-table). Refused: a tally that is not complete (a
leaf without a final verdict); a leaf that still fails after tau3 with no r10_tau4_tally.json; a tau4 tally that is
not final or names other leaves.

1. Leaf rows of X/r10 and Z/r10, built as pr3_leaf_table.py builds PR3's (same keys, same formulas): the verdict, the
   parts that ran, the rules' values between tau1 and tau2 and between tau2 and tau3 (rule 1: |k_new - k_old|;
   rule 2': decisions changed; both against max(3, 1% of k_new)), the largest |dP| (descriptive), NaN posteriors, band
   shots at kmax, the tau2 rebuild checks, CPU from the ledger (cpu_s + lost_cpu_s). The values must equal the tally's.
   A leaf that still failed after tau3 also gets its tau4 parts and the rules' values between tau3 and tau4 (10a),
   recomputed here, which must equal r10_tau4_tally.json's, with the tau3 rebuild checks and the tau4 CPU.
2. A leaf that passes (verdict "pass (ladder)", "pass (extra rung)" or, under 10a, "pass (tau4 rung)") gives d = 7
   numbers. A leaf that still fails after its last rung, or whose tau4 did not finish or did not start, gives none
   (PR3 section 5): its leaf row is reported instead. If only one basis passes, every number below uses that basis
   alone, at d = 3, 5 and 7, and is labelled so. If neither passes, only the leaf rows are reported.
3. Paired 2x2 tables (A wrong, ML(RL) wrong) at d = 7, r = 10, on the PR3 sample (the 25,000 decoded shots minus the
   1,000 probe shots: 24,000 per leaf). ML(RL)'s decisions are the ladder's final (tau2) decisions, as in round 1, K2
   and PR3; the extra rung only checks convergence. Under 10a, a leaf that passes at the tau4 rung takes tau3's
   decisions on its band shots and the ladder's elsewhere (tau4 checks tau3). A in: Libra (RL), Harmony (RL, SI1000), the shipped correlated
   matching (RL, SI1000), our corr. MWPM and MWPM (RL DEM). Tesseract has no d = 7 predictions and is left out.
   At d = 3 and 5, r = 10, the same tables from round 1's RL decode folders ("rl": all 50,000 shots, ML(RL) exact;
   "rl_s5": every 5th shot, ML(RL) the tau2 ladder), rebuilt with the frozen load_leaf() and count2(); the leaves rule
   A excludes (final_round1.json) give no ML(RL) number.
4. The paired gap at one round count. Round 1's estimator (willow_prereg.md section 4) fits y_r = ln(1 - 2E_r) on r per
   series and needs 2 rounds (section 9 entry 8). With one round it is taken through the origin: slope y/r, so
   eps = (1 - (1 - 2E)^(1/r))/2, round 1's eps_1pt. Per series (one leaf): b_M = y_ML/r and beta = (y_A - y_ML)/r,
   with variances and covariance from the leaf's 2x2 table (p1_se_covariance.points_arrays, divided by r^2). Then
   g = mean_s(eps_A - eps_ML) / mean_s(eps_ML) with entry 7's full delta-method SE, covariance included
   (p1_se_covariance.g_and_se, unchanged), and no Birge factor (one round leaves no residual degree of freedom).
   95% CI: g +- 1.96 SE. A series with E outside (0, 1/2) is dropped (entry 8) and listed.
5. Growth of the gap with distance: g_5/g_3 and g_7/g_5 per decoder, with Fieller 95% CIs (independent estimates:
   different patches). If the denominator's 95% CI contains 0, the interval is unbounded and reported so.
6. eps at d = 7, r = 10 (one round count) for ML(RL) and Libra: eps_1pt per basis with its delta-method SE, and the
   mean over the bases used, SE sqrt(sum se^2)/S. Lambda_5/7 at r = 10 for both: the frozen lam_fit() on these means
   (d = 5: ML(RL) on its leaves under rule A, Libra on all its leaves, each on the leaf's decoded shots), with a 95% CI
   on the log scale. One-round values include SPAM spread over 10 rounds: comparable across d at r = 10, not with
   round 1's fitted per-round values.
7. Checks: the parts cover the design's shots exactly and the sample has 24,000 shots per leaf; the recomputed rule
   values equal the tallies'; the sample's ML(RL) errors equal the tally's (tau2's, or tau3's for a leaf that passes at
   the tau4 rung); the tau3 rebuild checks are identical in every tau4 part; every decoder has a prediction on every
   sample shot; round 1's rebuilt counts and eps_1pt equal the final analysis's for every leaf and decoder used; the
   pooled g and its frozen-formula SE equal the frozen pool_gap()'s on the same series; a shot bootstrap of each d = 7
   row (2,000 multinomial resamples of each leaf's 2x2 table, seed 20261013) gives an SD of g close to the SE.
Outputs: <out>/analysis/r10_results.json and .md; <out>/leaf_table/pr3_leaf_table_with_r10.json and .md (PR3's leaf
table rows, unchanged, with the two r = 10 rows replaced by this run's). Exit 1 if a check fails.

  python3 -I -B pr3_r10_analyze.py --res results/decoder/willow/pr3/r10_posthoc \\
      --ledger results/decoder/willow/pr3/r10_posthoc/logs/ledger.json \\
      [--tau4-ledger results/decoder/willow/pr3/r10_posthoc/logs/tau4_ledger.json] --round1 results/decoder/willow \\
      --final-r1 results/decoder/willow/final_round1/final_round1.json \\
      --final-analysis results/decoder/willow/final_round1/analysis/analysis.json \\
      --pr3-leaf-table results/decoder/willow/pr3/leaf_table/pr3_leaf_table.json --out results/decoder/willow/pr3/r10_posthoc
"""
import argparse, hashlib, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("PR3_CODE") or os.path.dirname(HERE)            # code/decoder
for _p in (os.path.join(CODE, "supplementary"), CODE, os.path.join(CODE, "pr3"), HERE):
    sys.path.insert(0, _p)
import numpy as np
import analyze_willow as AW               # frozen; read-only
import p1_se_covariance as PC             # section 9 entry 7; read-only
import pr3_common as C                    # read-only

R = 10
ML, LIBRA = "ML.rl", "libra.rl"
ENTS = (LIBRA, "harmony.rl", "harmony.si1000", "corrmatch.rl", "corrmatch.si1000", "corr_mwpm.rl", "mwpm.rl")
LADDER_DN = C.LADDER.replace(":", "_").replace("/", "-")
Z95 = 1.96                                 # as the frozen ci_block() and g_and_se()
BOOT, SEED = 2000, 20261013
TAG1 = {3: "rl", 5: "rl_s5"}               # round 1's RL decode folder at each distance


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sigmoid(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-x.astype(np.float64)))


def decide(llr):
    return (np.nan_to_num(sigmoid(llr), nan=0.0) > 0.5).astype(np.uint8)


def nanmax_or(x, empty):
    return float(np.nanmax(x)) if x.size and not np.all(np.isnan(x)) else empty


def rel_diff(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------- the estimator at one round count
def eps_1pt(n, k, r=R):
    """eps = (1 - (1 - 2E)^(1/r)) / 2 and its delta-method SE, E = k / n."""
    E = k / n
    return AW.eps_from(E, r), (1 - 2 * E) ** (1 / r - 1) / r * math.sqrt(E * (1 - E) / n)


def series_1pt(c, r=R):
    """One series' paired parameters from its 2x2 table at round count r; None if E is outside (0, 1/2)."""
    P = PC.points_arrays({r: c}, (r,))
    if not P:
        return None
    p = P[0]
    V = {"var_beta": p["vz"] / r ** 2, "var_bM": p["vM"] / r ** 2, "cov": p["cov_zy"] / r ** 2}
    bM, beta = p["yM"] / r, p["z"] / r
    eM, eA = (1 - math.exp(bM)) / 2, (1 - math.exp(bM + beta)) / 2
    return {"bM": bM, "beta": beta, "uninflated": V, "inflated": V, "npts": 1, "table": dict(c),
            "eps_A": eA, "eps_ML": eM, "rel": eA / eM - 1, "delta": eA - eM,
            "se_delta": math.exp(bM + beta) * math.sqrt(V["var_beta"]) / 2,
            "se_eps_ML": math.exp(bM) * math.sqrt(V["var_bM"]) / 2}


def pooled(series):
    """g over the series with entry 7's full delta-method SE (no Birge factor), and the check against pool_gap()."""
    keys = sorted(series)
    S = [series[k] for k in keys]
    if not S:
        return None
    cs = PC.g_and_se(S, "uninflated", full=True, with_cov=True)
    ff = PC.g_and_se(S, "uninflated", full=False, with_cov=False)
    pg = AW.pool_gap(S)                                   # the frozen estimator on the same per-series values
    return {"g": cs["g"], "se": cs["se"], "ci95": [cs["g"] - Z95 * cs["se"], cs["g"] + Z95 * cs["se"]],
            "n_series": len(S), "series": keys, "per_series_rel": {k: s["rel"] for k, s in zip(keys, S)},
            "mean_eps_ML": float(np.mean([s["eps_ML"] for s in S])), "mean_eps_A": float(np.mean([s["eps_A"] for s in S])),
            "errors_A": sum(s["table"]["n10"] + s["table"]["n11"] for s in S),
            "errors_ML": sum(s["table"]["n01"] + s["table"]["n11"] for s in S),
            "discordant_A_only_ML_only": [sum(s["table"]["n10"] for s in S), sum(s["table"]["n01"] for s in S)],
            "shots": sum(s["table"]["n"] for s in S), "se_frozen_formula": ff["se"],
            "check_vs_frozen_pool_gap": {"g": rel_diff(cs["g"], pg["rel"]), "se_frozen_formula": rel_diff(ff["se"], pg["se_rel"])}}


def boot_sd(series, n_boot=BOOT, seed=SEED):
    """SD of g over multinomial resamples of each series' 2x2 table (a shot bootstrap)."""
    rng = np.random.default_rng(seed)
    keys = sorted(series)
    gs = []
    for _ in range(n_boot):
        S = []
        for k in keys:
            c = series[k]["table"]
            n00 = c["n"] - c["n10"] - c["n01"] - c["n11"]
            x = rng.multinomial(c["n"], np.array([c["n10"], c["n01"], c["n11"], n00], float) / c["n"])
            s = series_1pt({"n": c["n"], "n10": int(x[0]), "n01": int(x[1]), "n11": int(x[2])})
            if s is None:
                break
            S.append(s)
        else:
            gs.append(PC.g_and_se(S, "uninflated", full=True, with_cov=True)["g"])
    return {"replicates": len(gs), "seed": seed, "sd_g": float(np.std(gs, ddof=1)) if len(gs) > 1 else None}


def fieller(a, sa, b, sb, z=Z95):
    """a / b with its Fieller interval for independent normal estimates; unbounded if b's CI contains 0."""
    A = b * b - z * z * sb * sb
    out = {"ratio": a / b if b else None}
    if A <= 0:
        out.update(ci95=None, note="unbounded: the denominator's 95% CI contains 0")
        return out
    s = math.sqrt(a * a * sb * sb + sa * sa * A)
    out["ci95"] = sorted([(a * b - z * s) / A, (a * b + z * s) / A])
    return out


def eps_mean(rows):
    """{series: (eps, se)} -> mean and sqrt(sum se^2) / S."""
    v = list(rows.values())
    return float(np.mean([x[0] for x in v])), math.sqrt(sum(x[1] ** 2 for x in v)) / len(v)


def lam(e5, e7):
    f = AW.lam_fit({5: e5, 7: e7})
    s = f["se"] / f["Lambda"]
    f["ci95_log_scale"] = [f["Lambda"] * math.exp(-Z95 * s), f["Lambda"] * math.exp(Z95 * s)]
    return f


# ---------------------------------------------------------------- the leaf rows (as pr3_leaf_table.py)
def leaf_row(res, tag, design, lf, r, tv, units, t4=None, units4=None):
    """One leaf's row with pr3_leaf_table.py's keys and formulas, and its checks against the tally's verdict. t4: the
    leaf's r10_tau4_tally.json verdict, for a leaf that still failed after tau3 (amendment 10a); units4: the tau4
    ledger's units."""
    checks = []
    nm = lf.split("/", 2)[-1]
    ldir = lambda j: os.path.join(res, "parts", f"p{j}", "decode", lf, tag)
    edir = lambda j: os.path.join(res, "parts", f"p{j}", "extra_rung", lf, tag)

    def ladder_done(j):                         # as pr3_runner.Runner.ladder_done
        try:
            s = json.load(open(os.path.join(ldir(j), "summary.json")))
        except Exception:
            return False
        return (s.get("status") == "complete" and C.LADDER in (s.get("decoders") or {})
                and s.get("selection") == C.part_slice(design, j))

    def extra_done(j):                          # as pr3_runner.Runner.extra_done
        try:
            e = json.load(open(os.path.join(edir(j), "extra_rung.json")))
        except Exception:
            return False
        return "tau3" in e and e.get("ladder_npz_md5") == md5(os.path.join(ldir(j), LADDER_DN + ".npz"))

    row = {"verdict": tv.get("verdict"), "parts": {}}
    cpu = {"ladder": 0.0, "extra": 0.0, "lost_in_restart": 0.0}
    for j in range(C.PARTS):
        pj = {}
        for kind in ("ladder", "extra"):
            x = units.get(f"{kind}|{lf}|{j}")
            done = ladder_done(j) if kind == "ladder" else extra_done(j)
            ent = {"done": done, "ledger_state": (x or {}).get("state", "never started")}
            if x:
                c = (x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0)
                ent.update(cpu_core_h=c / 3600, lost_cpu_core_h=(x.get("lost_cpu_s") or 0.0) / 3600,
                           started=x.get("started"), ended=x.get("ended"), peak_rss_gib=(x.get("peak_rss") or 0) / 2 ** 30)
                cpu[kind] += c; cpu["lost_in_restart"] += x.get("lost_cpu_s") or 0.0
            if kind == "extra" and done:
                e = json.load(open(os.path.join(edir(j), "extra_rung.json")))
                t3, ck = e.get("tau3", {}), e.get("check", {})
                ent.update(n_band=t3.get("n_band"), decisions_changed_tau2_tau3=t3.get("decisions_changed_tau2_tau3"),
                           max_K_tau3=t3.get("max_K_tau3"), kmax=e.get("kmax"),
                           ms_per_round_per_band_shot_tau3=t3.get("ms_per_round_per_band_shot_tau3"),
                           rebuild_check={"n_checked": ck.get("n_checked"), "identical": ck.get("identical"),
                                          "max_abs_llr_diff": ck.get("max_abs_llr_diff")})
            pj[kind] = ent
        row["parts"][f"p{j}"] = pj
    row["ladder_parts_done"] = [j for j in range(C.PARTS) if row["parts"][f"p{j}"]["ladder"]["done"]]
    row["extra_parts_done"] = [j for j in range(C.PARTS) if row["parts"][f"p{j}"]["extra"]["done"]]
    row["cpu_core_h"] = {k: v / 3600 for k, v in cpu.items()}
    row["cpu_core_h"]["total"] = (cpu["ladder"] + cpu["extra"]) / 3600
    Ld = row["ladder_parts_done"]
    checks.append({"what": f"{nm}: all 5 ladder parts done", "ok": len(Ld) == C.PARTS})
    if len(Ld) != C.PARTS:
        return row, checks
    idx, obs, df, llr, llr1, red, part_of = [], [], [], [], [], [], []
    for j in Ld:
        z = np.load(os.path.join(ldir(j), LADDER_DN + ".npz")); sh = np.load(os.path.join(ldir(j), "shots.npz"))
        assert np.array_equal(z["idx"], sh["idx"]), f"{lf} p{j}: ladder and shots.npz select different shots"
        idx.append(z["idx"].astype(np.int64)); obs.append(sh["obs"].astype(np.uint8))
        df.append(z["pred"][:, 0]); llr.append(z["llr"]); llr1.append(z["llr_t1"])
        red.append(z["redecoded"].astype(bool)); part_of.append(np.full(len(z["idx"]), j))
    cat = np.concatenate
    idx, obs, df, llr, llr1, red, part_of = (cat(idx), cat(obs), cat(df).astype(np.uint8), cat(llr), cat(llr1), cat(red),
                                             cat(part_of))
    m = C.in_sample(idx, r)
    d1 = decide(llr1)
    k1, kf = int(np.sum((d1 != obs) & m)), int(np.sum((df != obs) & m))
    ch = int(np.sum((d1 != df) & m))
    dP = np.abs(sigmoid(llr) - sigmoid(llr1))[red & m]
    lim = max(3, 0.01 * kf)
    lad = {"parts": Ld, "complete": True, "n_decoded": int(len(idx)), "n_sample": int(m.sum()),
           "band_shots_sample": int((red & m).sum()), "errors_tau1": k1, "errors_tau2": kf,
           "abs_change_errors": abs(kf - k1), "rule1": abs(kf - k1) <= lim, "decisions_changed": ch,
           "rule2prime": ch <= lim, "limit": lim, "max_abs_dP_descriptive": nanmax_or(dP, 0.0),
           "nan_tau2": int(np.isnan(llr[m]).sum())}
    lad["passes"] = lad["rule1"] and lad["rule2prime"]
    row["tau1_tau2"] = lad
    tl = tv.get("ladder") or {}
    same = all(tl.get(k) == lad[k] for k in ("errors_tau1", "errors_tau2", "rule1", "decisions_changed", "rule2prime",
                                             "nan_tau2", "passes")) \
        and abs(tl.get("limit", -1) - lad["limit"]) < 1e-9 \
        and abs(tl.get("max_abs_dP_descriptive", -1) - lad["max_abs_dP_descriptive"]) < 1e-12 \
        and all(tv.get(k) == lad[k] for k in ("n_sample", "n_decoded", "band_shots_sample"))
    checks.append({"what": f"{nm}: tau1 -> tau2 values equal the tally's", "ok": bool(same)})
    Ed = row["extra_parts_done"]
    if Ed:
        pos = {int(x): i for i, x in enumerate(idx)}
        bi, l3, mk, kmax = [], [], [], None
        for j in Ed:
            e = np.load(os.path.join(edir(j), "extra_rung.npz"))
            bj = np.array([pos[int(x)] for x in e["idx"]], np.int64)
            want = np.where(red & (part_of == j))[0]
            assert np.array_equal(np.sort(bj), want), f"{lf} p{j}: extra-rung shots differ from the part's band"
            bi.append(bj); l3.append(e["llr_t3"]); mk.append(e["maxk_t3"])
            kmax = json.load(open(os.path.join(edir(j), "extra_rung.json"))).get("kmax")
        bi, l3, mk = cat(bi), cat(l3), cat(mk)
        keep = m[bi]
        d3 = decide(l3)
        bk = bi[keep]
        k3 = kf - int(np.sum(df[bk] != obs[bk])) + int(np.sum(d3[keep] != obs[bk]))
        ch3 = int(np.sum(d3[keep] != df[bk])); nan3 = int(np.isnan(l3[keep]).sum())
        lim3 = max(3, 0.01 * k3)
        dP3 = np.abs(sigmoid(l3) - sigmoid(llr[bi]))[keep]
        full = len(Ed) == C.PARTS
        checks.append({"what": f"{nm}: extra parts all 5 or none", "ok": full})
        ex = {"parts": Ed, "complete": full, "band_shots_sample_covered": int(keep.sum()),
              "band_shots_sample_leaf": lad["band_shots_sample"],
              "errors_tau2": kf, "errors_tau3" if full else "errors_tau3_on_these_parts_tau2_elsewhere": k3,
              "abs_change_errors": abs(k3 - kf), "decisions_changed": ch3, "limit": lim3, "nan_tau3": nan3,
              "max_abs_dP_descriptive": nanmax_or(dP3, None) if nan3 < dP3.size else None,
              "kmax": kmax, "band_shots_at_kmax_tau3": int(np.sum(mk[keep] >= kmax)) if kmax else None}
        if full:
            ex.update(rule1=abs(k3 - kf) <= lim3, rule2prime=ch3 <= lim3 and nan3 == 0)
            ex["passes"] = ex["rule1"] and ex["rule2prime"]
            te = tv.get("extra") or {}
            same = all(te.get(k) == v for k, v in (("errors_tau2", kf), ("errors_tau3", k3), ("decisions_changed", ch3),
                                                    ("nan_tau3", nan3), ("rule1", ex["rule1"]),
                                                    ("rule2prime", ex["rule2prime"]), ("passes", ex["passes"]))) \
                and abs(te.get("limit", -1) - lim3) < 1e-9 \
                and abs((te.get("max_abs_dP_descriptive") or 0) - (ex["max_abs_dP_descriptive"] or 0)) < 1e-12
            checks.append({"what": f"{nm}: tau2 -> tau3 values equal the tally's", "ok": bool(same)})
            rc = [row["parts"][f"p{j}"]["extra"]["rebuild_check"] for j in Ed]
            checks.append({"what": f"{nm}: tau2 rebuild checks identical in all {len(rc)} extra parts",
                           "ok": all(c.get("identical") for c in rc)})
        row["tau2_tau3"] = ex
    want = "pass (ladder)" if lad["passes"] else (("pass (extra rung)" if row.get("tau2_tau3", {}).get("passes")
                                                   else "still fails: excluded and flagged") if Ed else None)
    checks.append({"what": f"{nm}: the verdict follows from the recomputed rules", "ok": want == tv.get("verdict")})
    if t4 is not None:
        tau4_rows(row, checks, res, tag, lf, nm, t4, units4 or {}, idx, obs, df, red, part_of, m, kf, edir)
    return row, checks


def tau4_rows(row, checks, res, tag, lf, nm, t4, units4, idx, obs, df, red, part_of, m, kf, edir):
    """Amendment 10a: the leaf's tau4 parts and the rules between tau3 and tau4, recomputed (as
    pr3_r10_tau4.Tau4.rules) and checked against r10_tau4_tally.json."""
    tdir = lambda j: os.path.join(res, "parts", f"p{j}", "tau4_rung", lf, tag)
    row["verdict_after_tau3"] = row["verdict"]
    row["verdict"] = t4.get("verdict")

    def t4_done(j):                             # as pr3_r10_tau4.Tau4.done
        try:
            x = json.load(open(os.path.join(tdir(j), "tau4_rung.json")))
        except Exception:
            return False
        return ("tau4" in x and x.get("first") is None
                and x.get("extra_rung_npz_md5") == md5(os.path.join(edir(j), "extra_rung.npz")))
    cpu4 = 0.0
    for j in range(C.PARTS):
        x = units4.get(f"tau4|{lf}|{j}")
        ent = {"done": t4_done(j), "ledger_state": (x or {}).get("state", "never started")}
        if x:
            c = (x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0)
            ent.update(cpu_core_h=c / 3600, lost_cpu_core_h=(x.get("lost_cpu_s") or 0.0) / 3600, started=x.get("started"),
                       ended=x.get("ended"), peak_rss_gib=(x.get("peak_rss") or 0) / 2 ** 30)
            cpu4 += c
        jp = os.path.join(tdir(j), "tau4_rung.json")
        if os.path.exists(jp):
            x4 = json.load(open(jp))
            ck, s4 = x4.get("check", {}), x4.get("tau4", {})
            ent.update(part_verdict=x4.get("verdict"), n_band=s4.get("n_band"), max_K_tau4=s4.get("max_K_tau4"),
                       n_at_kmax_tau4=s4.get("n_at_kmax_tau4"), kmax=x4.get("kmax_t4"), nan_tau4=s4.get("nan_tau4"),
                       ms_per_round_per_band_shot_tau4=s4.get("ms_per_round_per_band_shot_tau4"),
                       tau3_rebuild_check={"n_checked": ck.get("n_checked"), "identical": ck.get("identical"),
                                           "max_abs_llr_diff": ck.get("max_abs_llr_diff")})
        row["parts"][f"p{j}"]["tau4"] = ent
    D4 = [j for j in range(C.PARTS) if row["parts"][f"p{j}"]["tau4"]["done"]]
    row["tau4_parts_done"] = D4
    row["cpu_core_h"]["tau4"] = cpu4 / 3600
    row["cpu_core_h"]["total"] += cpu4 / 3600
    full = len(D4) == C.PARTS and len(row["extra_parts_done"]) == C.PARTS
    if full:
        pos = {int(x): i for i, x in enumerate(idx)}
        bi, l3, l4, mk, kmax4 = [], [], [], [], None
        for j in D4:
            e = np.load(os.path.join(edir(j), "extra_rung.npz")); f = np.load(os.path.join(tdir(j), "tau4_rung.npz"))
            assert np.array_equal(e["idx"], f["idx"]), f"{lf} p{j}: tau4's band shots differ from tau3's"
            bj = np.array([pos[int(x)] for x in e["idx"]], np.int64)
            assert np.array_equal(np.sort(bj), np.where(red & (part_of == j))[0]), f"{lf} p{j}: band shots differ"
            bi.append(bj); l3.append(e["llr_t3"]); l4.append(f["llr_t4"]); mk.append(f["maxk_t4"])
            kmax4 = json.load(open(os.path.join(tdir(j), "tau4_rung.json"))).get("kmax_t4")
        cat = np.concatenate
        bi, l3, l4, mk = cat(bi), cat(l3), cat(l4), cat(mk)
        keep = m[bi]; bk = bi[keep]
        d3, d4 = decide(l3), decide(l4)
        base = kf - int(np.sum(df[bk] != obs[bk]))
        k3, k4 = base + int(np.sum(d3[keep] != obs[bk])), base + int(np.sum(d4[keep] != obs[bk]))
        ch4 = int(np.sum(d4[keep] != d3[keep])); nan4 = int(np.isnan(l4[keep]).sum())
        lim4 = max(3, 0.01 * k4)
        dP4 = np.abs(sigmoid(l4) - sigmoid(l3))[keep]
        q = {"parts": D4, "complete": True, "band_shots_sample_covered": int(keep.sum()), "errors_tau3": k3,
             "errors_tau4": k4, "abs_change_errors": abs(k4 - k3), "rule1": abs(k4 - k3) <= lim4, "decisions_changed": ch4,
             "rule2prime": ch4 <= lim4 and nan4 == 0, "limit": lim4, "nan_tau4": nan4,
             "max_abs_dP_descriptive": nanmax_or(dP4, None) if nan4 < dP4.size else None, "kmax": kmax4,
             "band_shots_at_kmax_tau4": int(np.sum(mk[keep] >= kmax4)) if kmax4 else None}
        q["passes"] = q["rule1"] and q["rule2prime"]
        row["tau3_tau4"] = q
        same = all(t4.get(k) == q[k] for k in ("errors_tau3", "errors_tau4", "decisions_changed", "nan_tau4", "rule1",
                                                "rule2prime", "passes")) \
            and abs(t4.get("limit", -1) - lim4) < 1e-9 \
            and abs((t4.get("max_abs_dP_descriptive") or 0) - (q["max_abs_dP_descriptive"] or 0)) < 1e-12 \
            and t4.get("band_shots_at_kmax_tau4") == q["band_shots_at_kmax_tau4"]
        checks.append({"what": f"{nm}: tau3 -> tau4 values equal the tau4 tally's", "ok": bool(same)})
        ex = row.get("tau2_tau3") or {}
        checks.append({"what": f"{nm}: tau3's errors at the tau4 step equal the tau2 -> tau3 step's ({k3})",
                       "ok": k3 == ex.get("errors_tau3")})
        rc = [row["parts"][f"p{j}"]["tau4"].get("tau3_rebuild_check") or {} for j in D4]
        checks.append({"what": f"{nm}: tau3 rebuild checks identical in all {len(rc)} tau4 parts",
                       "ok": all(c.get("identical") is True for c in rc)})
    want4 = (("pass (tau4 rung)" if row["tau3_tau4"]["passes"] else "still fails after tau4: excluded and flagged")
             if full else "tau4 incomplete: leaf rows only")
    checks.append({"what": f"{nm}: the tau4 verdict follows from the recomputed rules", "ok": want4 == t4.get("verdict")})


# ---------------------------------------------------------------- d = 7, r = 10: this run's parts
def load_d7(res, tag, lf, design, r=R, ml_tau3=False):
    """The leaf's 5 parts on the PR3 sample: obs, ML(RL) (the ladder's final decisions; with ml_tau3, for a leaf that
    passes at the tau4 rung under 10a, tau3's decisions on the band shots), the shipped decoders and our corr. MWPM and
    MWPM (int8; -1 = no prediction)."""
    cols = {k: [] for k in ("idx", "obs", ML) + ENTS}
    want = []
    for j in range(C.PARTS):
        st, sp, stp = (int(x) for x in C.part_slice(design, j).split(":"))
        want.append(np.arange(st, sp, stp))
        d = os.path.join(res, "parts", f"p{j}", "decode", lf, tag)
        s = json.load(open(os.path.join(d, "summary.json")))
        assert s["status"] == "complete" and s.get("selection") == C.part_slice(design, j), f"{d}: not complete"
        z = np.load(os.path.join(d, LADDER_DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        assert np.array_equal(z["idx"], sh["idx"]), f"{d}: the ladder and shots.npz select different shots"
        n = len(sh["idx"])
        cols["idx"].append(sh["idx"].astype(np.int64)); cols["obs"].append(sh["obs"].astype(np.int8))
        ml = z["pred"][:, 0].astype(np.int8)
        if ml_tau3:
            e = np.load(os.path.join(res, "parts", f"p{j}", "extra_rung", lf, tag, "extra_rung.npz"))
            pos = {int(x): i for i, x in enumerate(z["idx"])}
            bj = np.array([pos[int(x)] for x in e["idx"]], np.int64)
            assert np.array_equal(np.sort(bj), np.where(z["redecoded"].astype(bool))[0]), f"{d}: tau3's shots differ from the band"
            ml[bj] = decide(e["llr_t3"]).astype(np.int8)
        cols[ML].append(ml)
        got = {}
        for nm, row in zip(sh["shipped_names"], sh["shipped"]):
            e = AW.shipped_entity(str(nm))
            assert e not in got, f"{d}: two shipped rows for {e}"
            got[e] = row.astype(np.int8)
        for b in ("corr_mwpm", "mwpm"):
            p = os.path.join(d, b + ".npz")
            if os.path.exists(p):
                zb = np.load(p)
                assert np.array_equal(zb["idx"], sh["idx"]), f"{d}: {b} selects different shots"
                got[b + ".rl"] = zb["pred"][:, 0].astype(np.int8)
        for e in ENTS:
            cols[e].append(got.get(e, np.full(n, -1, np.int8)))
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    cover = bool(np.array_equal(cat["idx"], np.concatenate(want)))
    m = C.in_sample(cat["idx"], r)
    return {k: v[m] for k, v in cat.items()}, int(len(m)), int(m.sum()), cover


# ---------------------------------------------------------------- d = 3 and 5, r = 10: round 1's leaves
def round1(res1, final_r1, final_an, r=R):
    """Round 1's 2x2 tables (A, ML(RL)) and the ML(RL) and Libra marginals at round count r, rebuilt with the frozen
    load_leaf() and count2(); rule A's excluded leaves give no ML(RL) number. Checked against the final analysis."""
    excl = set(json.load(open(final_r1))["excluded_leaves"])
    L = json.load(open(final_an))["leaves"]
    tables = {3: {}, 5: {}}
    marg = {3: {ML: {}, LIBRA: {}}, 5: {ML: {}, LIBRA: {}}}
    checks, excluded, missing, problems = [], [], [], []
    for rel in sorted(L):
        v = L[rel]
        if v.get("rounds") != r or v.get("d") not in (3, 5):
            continue
        d = v["d"]; ser = f"{v['patch']}|{v['basis']}"; tagd = TAG1[d]
        assert tagd in v["folders"], f"{rel}: no {tagd} folder"
        obs, ents, ml, info, shipped = AW.load_leaf(res1, rel, [tagd])
        problems += [f"{rel}: {p}" for p in info["problems"]]
        ex = rel in excl
        if ex:
            excluded.append(rel)
        fe_all = v["entities"]
        for e in (ML, LIBRA):
            if e == ML and ex:
                continue
            if e not in ents:
                missing.append(f"{rel}: {e}"); continue
            a = ents[e]; mm = (a >= 0) & (obs >= 0)
            n, k = int(mm.sum()), int(np.sum(a[mm] != obs[mm]))
            fe = fe_all.get(e, {})
            checks.append({"leaf": rel, "entity": e, "what": "marginal",
                           "ok": n == fe.get("n") and k == fe.get("errors")
                           and rel_diff(AW.eps_from(k / n, r), fe.get("eps_1pt", -1)) < 1e-12})
            marg[d][e][ser] = (n, k)
        if ex:
            continue
        for e in ENTS:
            if e not in ents:
                missing.append(f"{rel}: {e}"); continue
            c = AW.count2(ents[e], ents[ML], obs)
            fe, fm = fe_all.get(e, {}), fe_all.get(ML, {})
            s = series_1pt(c, r)
            ok = (c["n"] == fe.get("n") == fm.get("n") and c["n10"] + c["n11"] == fe.get("errors")
                  and c["n01"] + c["n11"] == fm.get("errors") and s is not None
                  and rel_diff(s["eps_ML"], fm.get("eps_1pt", -1)) < 1e-12 and rel_diff(s["eps_A"], fe.get("eps_1pt", -1)) < 1e-12)
            checks.append({"leaf": rel, "entity": e, "what": "paired table", "ok": bool(ok)})
            tables[d].setdefault(e, {})[ser] = c
    return tables, marg, checks, excluded, missing, problems


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--tau4-ledger", default=None, help="pr3_r10_tau4.py's tau4_ledger.json, if tau4 ran (10a)")
    ap.add_argument("--round1", required=True, help="results/decoder/willow (round 1's decode folders)")
    ap.add_argument("--final-r1", required=True)
    ap.add_argument("--final-analysis", required=True)
    ap.add_argument("--pr3-leaf-table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tally_p = os.path.join(a.res, "r10_tally.json")
    T = json.load(open(tally_p))
    if not T.get("complete"):
        sys.exit("the r = 10 tally is not complete: every leaf needs a final verdict")
    still = sorted(lf for lf, v in (T.get("verdicts") or {}).items() if str(v.get("verdict", "")).startswith("still fails"))
    t4_p = os.path.join(a.res, "r10_tau4_tally.json")
    T4 = json.load(open(t4_p)) if os.path.exists(t4_p) else None
    if still and T4 is None:
        sys.exit("a leaf still fails after tau3 and r10_tau4_tally.json does not exist: run pr3_r10_tau4.py first (10a)")
    if T4 is not None and not T4.get("final"):
        sys.exit(f"r10_tau4_tally.json is not final ({T4.get('status')})")
    if T4 is not None and sorted(T4.get("failing_after_tau3") or []) != still:
        sys.exit("r10_tau4_tally.json names other leaves than the ones that still fail in r10_tally.json")
    LG = json.load(open(a.ledger))
    LG4 = json.load(open(a.tau4_ledger)) if a.tau4_ledger else {"units": {}}
    design = T["design"]; tag = f"rl_{design}"
    units, units4 = LG["units"], LG4["units"]
    o = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "what": ("PR3's post hoc r = 10 follow-up (willow_prereg_r3.md section 10, with amendment 10a): descriptive "
                  "only, no test and no p-value; PR3's kill stands and H7 stays not evaluated"),
         "design": design, "tag": tag, "round": R,
         "tally": {k: T.get(k) for k in ("made", "status", "complete", "still_failing")},
         "tau4": (None if T4 is None else {k: T4.get(k) for k in ("made", "status", "final", "failing_after_tau3",
                                                                    "projection", "guard_core_h", "kmax_t4", "factor",
                                                                    "tau4_cpu_core_h", "total_cpu_core_h")}),
         "inputs_md5": {"r10_tally.json": md5(tally_p), "ledger.json": md5(a.ledger), "final_round1.json": md5(a.final_r1),
                        "final_round1/analysis/analysis.json": md5(a.final_analysis),
                        "pr3_leaf_table.json": md5(a.pr3_leaf_table)},
         "code_md5": {"pr3_r10_analyze.py": md5(os.path.abspath(__file__)), "analyze_willow.py": md5(AW.__file__),
                      "p1_se_covariance.py": md5(PC.__file__), "pr3_common.py": md5(C.__file__)},
         "checks": [], "leaves": {}}
    if T4 is not None:
        o["inputs_md5"]["r10_tau4_tally.json"] = md5(t4_p)
    if a.tau4_ledger:
        o["inputs_md5"]["tau4_ledger.json"] = md5(a.tau4_ledger)

    def chk(what, ok):
        o["checks"].append({"what": what, "ok": bool(ok)})

    # 1. leaf rows and verdicts
    passing = []
    for b in C.BASES:
        lf = C.leaf(b, R)
        tv = (T.get("verdicts") or {}).get(lf, {})
        t4 = (T4.get("verdicts") or {}).get(lf) if (T4 is not None and lf in still) else None
        row, cks = leaf_row(a.res, tag, design, lf, R, tv, units, t4, units4)
        o["checks"] += cks
        o["leaves"][lf] = row
        if str(row["verdict"]).startswith("pass"):
            passing.append(b)
    if T4 is not None and still:
        tv4 = sum(o["leaves"][lf]["cpu_core_h"].get("tau4", 0.0) for lf in still)
        chk(f"tau4 CPU from the tau4 ledger ({tv4:.4f} core-hours) equals the tau4 tally's",
            abs(tv4 - (T4.get("tau4_cpu_core_h") or 0.0)) < 1e-9)
    o["passing_bases"] = passing
    tot = sum((x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0) for x in units.values())
    tot4 = sum((x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0) for x in units4.values())
    o["cpu"] = {"core_h": (tot + tot4) / 3600, "driver_core_h": tot / 3600, "tau4_core_h": tot4 / 3600,
                "of_which_lost_core_h": (sum(x.get("lost_cpu_s") or 0.0 for x in units.values())
                                         + sum(x.get("lost_cpu_s") or 0.0 for x in units4.values())) / 3600,
                "cap_core_h": T.get("guard_core_h"), "projection_at_exit": LG.get("projection"),
                "tau4_projection": (T4 or {}).get("projection")}
    nopass = [f"{b}/r10 ({o['leaves'][C.leaf(b, R)]['verdict']})" for b in C.BASES if b not in passing]
    o["basis_rule"] = ("X and Z" if len(passing) == 2 else (f"{passing[0]} only: " + "; ".join(nopass)
                       + " gives no number" if passing else "no leaf passes: leaf rows only"))

    if passing:
        bases = tuple(passing)
        # 2. d = 7 tables
        d7 = {}
        for b in bases:
            lf = C.leaf(b, R)
            at4 = o["leaves"][lf]["verdict"] == "pass (tau4 rung)"
            S, n_dec, n_s, cover = load_d7(a.res, tag, lf, design, ml_tau3=at4)
            chk(f"{b}/r10: the parts cover the design's shots exactly ({n_dec}); sample {n_s}",
                cover and n_dec == 25000 and n_s == 24000)
            k_ml = int(np.sum(S[ML] != S["obs"]))
            if at4:
                chk(f"{b}/r10: the sample's ML(RL) errors, tau3's on the band (10a), equal the tallies' ({k_ml})",
                    k_ml == T["verdicts"][lf]["extra"]["errors_tau3"] == T4["verdicts"][lf]["errors_tau3"])
            else:
                chk(f"{b}/r10: the sample's ML(RL) errors equal the tally's ({k_ml})", k_ml == T["verdicts"][lf]["ladder"]["errors_tau2"])
            d7[b] = {"n_sample": n_s, "errors": {ML: k_ml}, "tables": {},
                     "ml_rl_decisions": ("tau3 on the band shots, the ladder elsewhere (amendment 10a)" if at4
                                         else "the ladder's final (tau2)")}
            for e in ENTS:
                c = AW.count2(S[e], S[ML], S["obs"])
                chk(f"{b}/r10: {e} has a prediction on every sample shot", c["n"] == n_s)
                if c["n"]:
                    d7[b]["tables"][e] = c; d7[b]["errors"][e] = c["n10"] + c["n11"]
        o["d7_counts"] = d7
        # 3. round 1 at r = 10
        t1, m1, ck1, excl1, miss1, prob1 = round1(a.round1, a.final_r1, a.final_analysis)
        chk(f"round 1, r = 10: rebuilt tables and marginals equal the final analysis's ({len(ck1)} leaf x decoder)",
            ck1 and all(x["ok"] for x in ck1))
        chk("round 1, r = 10: load_leaf() reports no problem", not prob1)
        o["round1_r10"] = {"excluded_under_rule_A": excl1, "missing": miss1, "problems": prob1,
                           "failed_checks": [x for x in ck1 if not x["ok"]]}

        def series_set(d, e):
            src = ({b: d7[b]["tables"][e] for b in bases if e in d7[b]["tables"]} if d == 7
                   else {s: c for s, c in t1[d].get(e, {}).items() if s.split("|")[1] in bases})
            out, dropped = {}, []
            for s, c in src.items():
                p = series_1pt(c)
                if p is None:
                    dropped.append(s)
                else:
                    out[s] = p
            return out, dropped

        # 4. gaps, growth, bootstrap
        o["gaps"], o["growth"], o["bootstrap_d7"], o["dropped_series"] = {}, {}, {}, {}
        for e in ENTS:
            G = {}
            for d in (3, 5, 7):
                S, dropped = series_set(d, e)
                if dropped:
                    o["dropped_series"][f"{e}|d{d}"] = dropped
                p = pooled(S)
                if p:
                    G[str(d)] = p
                    chk(f"d = {d}, {e}: g and its frozen-formula SE equal pool_gap()'s",
                        p["check_vs_frozen_pool_gap"]["g"] < 1e-9 and p["check_vs_frozen_pool_gap"]["se_frozen_formula"] < 1e-9)
            o["gaps"][e] = G
            gr = {}
            for lo, hi in ((3, 5), (5, 7)):
                if str(lo) in G and str(hi) in G:
                    gr[f"g{hi}/g{lo}"] = fieller(G[str(hi)]["g"], G[str(hi)]["se"], G[str(lo)]["g"], G[str(lo)]["se"])
            o["growth"][e] = gr
            if "7" in G:
                bs = boot_sd(series_set(7, e)[0])
                bs["se_delta_method"] = G["7"]["se"]
                bs["ratio_sd_to_se"] = bs["sd_g"] / G["7"]["se"] if bs["sd_g"] else None
                o["bootstrap_d7"][e] = bs
        # 5. eps at d = 7 and Lambda_5/7
        o["eps_d7_r10"], o["lambda_5_7_r10"] = {}, {}
        for e in (ML, LIBRA):
            rows7 = {b: eps_1pt(d7[b]["n_sample"], d7[b]["errors"][e]) for b in bases}
            m7, se7 = eps_mean(rows7)
            o["eps_d7_r10"][e] = {"per_basis": {b: {"eps": v[0], "se": v[1], "errors": d7[b]["errors"][e],
                                                    "n": d7[b]["n_sample"]} for b, v in rows7.items()},
                                  "mean": m7, "se": se7, "bases": list(bases)}
            rows5 = {s: eps_1pt(n, k) for s, (n, k) in m1[5][e].items() if s.split("|")[1] in bases}
            m5, se5 = eps_mean(rows5)
            o["lambda_5_7_r10"][e] = {"Lambda_5_7": lam((m5, se5), (m7, se7)), "eps_d5_mean": m5, "se_d5": se5,
                                      "d5_series": sorted(rows5), "d7_bases": list(bases)}
    o["seconds"] = time.time() - t0
    o["all_checks_ok"] = all(c["ok"] for c in o["checks"])

    # outputs
    ad = os.path.join(a.out, "analysis"); os.makedirs(ad, exist_ok=True)
    for name, text in (("r10_results.json", json.dumps(o, indent=1, default=float)), ("r10_results.md", write_md(o))):
        p = os.path.join(ad, name)
        with open(p + ".tmp", "w") as f:
            f.write(text)
        os.replace(p + ".tmp", p)
    LT = json.load(open(a.pr3_leaf_table))
    comb = {"made": o["made"],
            "what": ("PR3's leaf table with its two r = 10 rows from the post hoc follow-up (willow_prereg_r3.md section "
                     "10); the other 4 rows are pr3_leaf_table.json's, unchanged"),
            "pr3_leaf_table_md5": md5(a.pr3_leaf_table), "r10_results": "analysis/r10_results.json", "leaves": {}}
    for lf, row in LT["leaves"].items():
        comb["leaves"][lf] = (dict(o["leaves"][lf], source="post hoc r = 10 follow-up") if lf in o["leaves"]
                              else dict(row, source="PR3 main run (pr3_leaf_table.json)"))
    comb["pr3_totals"] = LT.get("totals")
    comb["r10_cpu"] = o["cpu"]
    ld = os.path.join(a.out, "leaf_table"); os.makedirs(ld, exist_ok=True)
    for name, text in (("pr3_leaf_table_with_r10.json", json.dumps(comb, indent=1, default=float)),
                       ("pr3_leaf_table_with_r10.md", write_leaf_md(comb))):
        p = os.path.join(ld, name)
        with open(p + ".tmp", "w") as f:
            f.write(text)
        os.replace(p + ".tmp", p)
    print(json.dumps({"all_checks_ok": o["all_checks_ok"], "n_checks": len(o["checks"]), "passing": passing,
                      "seconds": o["seconds"]}, indent=1))
    if not o["all_checks_ok"]:
        sys.exit("a check failed: see r10_results.json")


# ---------------------------------------------------------------- markdown
NICE = {LIBRA: "Libra (RL)", "harmony.rl": "Harmony (RL)", "harmony.si1000": "Harmony (SI1000)",
        "corrmatch.rl": "corr. matching, shipped (RL)", "corrmatch.si1000": "corr. matching, shipped (SI1000)",
        "corr_mwpm.rl": "corr. MWPM, ours (RL)", "mwpm.rl": "MWPM, ours (RL)", ML: "ML(RL)"}


def pct(x):
    return "–" if x is None else f"{100 * x:+.2f}%"


def short(lf):
    return lf.split("/", 2)[-1]


def rung_rows(n, lad, ex, q=None):
    """The 'rules between rungs' rows of pr3_leaf_table.md for one leaf, with the tau3 -> tau4 row (10a) if q."""
    if not lad:
        return f"| {n} | – | no part ran | | | | | | |\n"
    s = rung_rows_pr3(n, lad, ex)
    if q:
        dp = "–" if q["max_abs_dP_descriptive"] is None else format(q["max_abs_dP_descriptive"], ".3f")
        s += (f"| {n} | τ3 → τ4 (10a) | {q['errors_tau3']} → {q['errors_tau4']} | {'pass' if q['rule1'] else 'fail'} | "
              f"{q['decisions_changed']} / {q['limit']:.2f} | {'pass' if q['rule2prime'] else 'fail'} | {dp} | "
              f"{q['nan_tau4']} | {q['band_shots_at_kmax_tau4']} of {q['band_shots_sample_covered']} |\n")
    return s


def cpu_cell(c):
    return (f"{c['total']:.2f} ({c['ladder']:.2f} + {c['extra']:.2f}" + (f" + {c['tau4']:.2f}" if "tau4" in c else "")
            + ")")


def rung_rows_pr3(n, lad, ex):
    s = (f"| {n} | τ1 → τ2 | {lad['errors_tau1']} → {lad['errors_tau2']} | {'pass' if lad['rule1'] else 'fail'} | "
         f"{lad['decisions_changed']} / {lad['limit']:.2f} | {'pass' if lad['rule2prime'] else 'fail'} | "
         f"{lad['max_abs_dP_descriptive']:.3f} | {lad['nan_tau2']} | |\n")
    if ex:
        k3 = ex.get("errors_tau3", ex.get("errors_tau3_on_these_parts_tau2_elsewhere"))
        dp = "–" if ex["max_abs_dP_descriptive"] is None else format(ex["max_abs_dP_descriptive"], ".3f")
        if ex["complete"]:
            s += (f"| {n} | τ2 → τ3 | {ex['errors_tau2']} → {k3} | {'pass' if ex['rule1'] else 'fail'} | "
                  f"{ex['decisions_changed']} / {ex['limit']:.2f} | {'pass' if ex['rule2prime'] else 'fail'} | {dp} | "
                  f"{ex['nan_tau3']} | {ex['band_shots_at_kmax_tau3']} of {ex['band_shots_sample_covered']} |\n")
        else:
            s += (f"| {n} | τ2 → τ3, partial (parts {ex['parts']}; {ex['band_shots_sample_covered']} of "
                  f"{ex['band_shots_sample_leaf']} band shots) | {ex['errors_tau2']} → {k3} (τ3 on these parts) | no verdict | "
                  f"{ex['decisions_changed']} / ({ex['limit']:.2f}, for reference) | no verdict | {dp} | "
                  f"{ex['nan_tau3']} | {ex['band_shots_at_kmax_tau3']} of {ex['band_shots_sample_covered']} |\n")
    return s


RUNG_HEAD = ("| leaf | rung | k (old → new) | rule 1 | decisions changed / limit | rule 2′ | max abs ΔP | NaN | "
             "band shots at kmax (new rung) |\n|---|---|---|---|---|---|---|---|---|\n")


def tau4_md(o):
    """The tau4 paragraph (amendment 10a)."""
    t4 = o.get("tau4")
    if not t4:
        return ""
    s = f"\n**τ4 rung (amendment 10a):** {t4['status']}."
    P = t4.get("projection") or {}
    if str(t4["status"]).startswith(("not started", "stopped")):
        s += (f" Projection: {P.get('total_h', 0):.2f} core-hours in total (the driver's {P.get('driver_h', 0):.2f}, τ4 "
              "used " + f"{P.get('tau4_used_h', 0):.2f}, τ4 to come " + f"{P.get('tau4_remaining_h', 0):.2f}; per leaf "
              + ", ".join(f"{short(lf)} {h:.2f}" for lf, h in (P.get("per_leaf_h") or {}).items())
              + f") against the cap of {t4['guard_core_h']}.")
    s += "\n\n"
    for lf, x in o["leaves"].items():
        if "tau4_parts_done" in x:
            rc = [x["parts"][f"p{j}"]["tau4"].get("tau3_rebuild_check") or {} for j in x["tau4_parts_done"]]
            mk = [x["parts"][f"p{j}"]["tau4"].get("max_K_tau4") for j in x["tau4_parts_done"]]
            s += (f"- {short(lf)}: after τ3, {x['verdict_after_tau3']}; after τ4, {x['verdict']}. τ4 parts done: "
                  f"{len(x['tau4_parts_done'])} of 5. τ3 rebuild checks identical in "
                  f"{sum(1 for c in rc if c.get('identical') is True)} of {len(rc)} parts"
                  + (f"; largest τ4 term count {max(m for m in mk if m is not None)}" if any(m is not None for m in mk) else "")
                  + f". τ4 CPU {x['cpu_core_h'].get('tau4', 0):.2f} core-hours.\n")
    return s


def write_md(o):
    s = ("# PR3 post hoc r = 10 follow-up: results (descriptive)\n\n"
         f"Made {o['made']} by `pr3_r10_analyze.py` (md5 {o['code_md5']['pr3_r10_analyze.py']}). The analysis was "
         "declared in `willow_prereg_r3.md` §10 before any r = 10 part started. **Post hoc and descriptive: no test and "
         "no p-value. PR3's kill stands, and H7 stays not evaluated.**\n\n"
         "## Leaf rows (PR3's rules, on the PR3 sample of 24,000 shots per leaf)\n\n"
         "| leaf | verdict | band shots | CPU, core-hours (ladder + extra [+ τ4]) |\n|---|---|---|---|\n")
    for lf, x in o["leaves"].items():
        lad = x.get("tau1_tau2") or {}
        s += f"| {short(lf)} | {x['verdict']} | {lad.get('band_shots_sample', '–')} | {cpu_cell(x['cpu_core_h'])} |\n"
    s += "\n" + RUNG_HEAD
    for lf, x in o["leaves"].items():
        s += rung_rows(short(lf), x.get("tau1_tau2"), x.get("tau2_tau3"), x.get("tau3_tau4"))
    s += tau4_md(o)
    c = o["cpu"]
    s += (f"\nCPU of the follow-up: {c['core_h']:.2f} core-hours (the driver's {c['driver_core_h']:.2f} + τ4 "
          f"{c['tau4_core_h']:.2f}) against the cap of {c['cap_core_h']}.\n\n")
    if not o["passing_bases"]:
        s += "**No r = 10 leaf passes, so no gap number is given (PR3 §5's rule).** The leaf rows above are the result.\n"
        return s + checks_md(o)
    bases = o["passing_bases"]
    if len(bases) == 1:
        s += (f"**Only {bases[0]}/r10 passes. Every number below uses the {bases[0]} basis alone, at d = 3, 5 and 7.** "
              "The other leaf gives no number; its row is above.\n\n")
    s += "## Error counts at d = 7, r = 10 (PR3 sample)\n\n"
    for b in bases:
        if o["d7_counts"][b].get("ml_rl_decisions", "").startswith("tau3"):
            s += (f"ML(RL) at {b}/r10 takes τ3's decisions on the band shots and the ladder's elsewhere: the leaf passes "
                  "at the τ4 rung (amendment 10a).\n\n")
    s += ("| decoder | " + " | ".join(f"{b}/r10" for b in bases) + " |\n|---|" + "---|" * len(bases) + "\n")
    for e in (ML,) + ENTS:
        s += f"| {NICE[e]} | " + " | ".join(f"{o['d7_counts'][b]['errors'].get(e, '–')} of {o['d7_counts'][b]['n_sample']}"
                                            for b in bases) + " |\n"
    nser = {d: next((G[str(d)]["n_series"] for G in o["gaps"].values() if str(d) in G), 0) for d in (3, 5, 7)}
    s += ("\n## Paired gap g to ML(RL) at r = 10, one round count (95% CIs)\n\n"
          "Per leaf eps = (1 − (1 − 2E)^(1/10))/2, paired through the leaf's 2×2 table; g pooled over the leaves with "
          "entry 7's full delta method, without a Birge factor. SPAM is not separated, so these values compare across d "
          "at r = 10; they are not round 1's fitted g_3 = +1.64% or g_5 = +6.66%. Leaves (Libra row): "
          + ", ".join(f"d = {d}: {nser[d]}" for d in (3, 5, 7)) + f" ({'X and Z' if len(bases) == 2 else bases[0] + ' only'}).\n\n"
          "| decoder | g_3 | g_5 | g_7 | g_5 / g_3 | g_7 / g_5 |\n|---|---|---|---|---|---|\n")

    def cell(G, d):
        if d not in G:
            return "–"
        x = G[d]
        return f"{pct(x['g'])} [{pct(x['ci95'][0])}, {pct(x['ci95'][1])}]"

    def fr(gr, k):
        if k not in gr:
            return "–"
        x = gr[k]
        return f"{x['ratio']:.2f} " + (f"[{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]" if x["ci95"] else "(CI unbounded)")
    for e in ENTS:
        G, gr = o["gaps"].get(e, {}), o["growth"].get(e, {})
        s += f"| {NICE[e]} | {cell(G, '3')} | {cell(G, '5')} | {cell(G, '7')} | {fr(gr, 'g5/g3')} | {fr(gr, 'g7/g5')} |\n"
    if o.get("dropped_series"):
        s += f"\nSeries dropped (E outside (0, 1/2)): {o['dropped_series']}.\n"
    s += ("\n## eps at d = 7, r = 10 and Λ_5/7 at r = 10 (one round count; SPAM included)\n\n| decoder | "
          + " | ".join(f"eps, {b}/r10" for b in bases) + " | eps at d = 7, mean | eps at d = 5, mean (leaves) | Λ_5/7 [95% CI] |\n|---|"
          + "---|" * (len(bases) + 3) + "\n")
    for e in (ML, LIBRA):
        x, l_ = o["eps_d7_r10"][e], o["lambda_5_7_r10"][e]
        L = l_["Lambda_5_7"]
        s += (f"| {NICE[e]} | " + " | ".join(f"{x['per_basis'][b]['eps']:.3e} ± {x['per_basis'][b]['se']:.1e}" for b in bases)
              + f" | {x['mean']:.3e} ± {x['se']:.1e} | {l_['eps_d5_mean']:.3e} ± {l_['se_d5']:.1e} ({len(l_['d5_series'])}) | "
              f"{L['Lambda']:.2f} [{L['ci95_log_scale'][0]:.2f}, {L['ci95_log_scale'][1]:.2f}] |\n")
    bs = o.get("bootstrap_d7") or {}
    if bs:
        s += ("\nShot bootstrap of the d = 7 rows (SD of g against the delta-method SE, percentage points): "
              + "; ".join(f"{NICE[e]} {v['sd_g'] * 100:.2f} vs {v['se_delta_method'] * 100:.2f}" for e, v in bs.items()
                          if v.get("sd_g") is not None) + ".\n")
    return s + checks_md(o)


def checks_md(o):
    bad = [c for c in o["checks"] if not c["ok"]]
    s = f"\n## Checks\n\n{len(o['checks']) - len(bad)} of {len(o['checks'])} checks pass.\n"
    s += "".join(f"\n- {c['what']}: {'ok' if c['ok'] else '**FAILED**'}" for c in o["checks"])
    return s + "\n"


def write_leaf_md(c):
    s = ("# PR3 leaf table with the r = 10 rows (post hoc)\n\n"
         f"Made {c['made']}. The X/r30, X/r50, Z/r30 and Z/r50 rows are PR3's (`pr3_leaf_table.json`, md5 "
         f"{c['pr3_leaf_table_md5']}), unchanged. The X/r10 and Z/r10 rows come from the post hoc r = 10 follow-up "
         "(`willow_prereg_r3.md` §10), run after PR3's kill with PR3's design, decoders and rules; the kill stands, and "
         "H7 stays not evaluated. Rules per leaf on the PR3 sample: rule 1, |k_new − k_old| ≤ max(3, 1% of k_new); "
         "rule 2′, decisions changed ≤ the same limit. |ΔP| is descriptive. Under amendment 10a, an r = 10 leaf that "
         "still failed after τ3 got one more rung, τ4 = 1e-9, with the same rules between τ3 and τ4.\n\n"
         "## Verdicts, parts and CPU\n\n| leaf | source | verdict | ladder parts done | extra parts done | τ4 parts done | "
         "band shots | CPU, core-hours (ladder + extra [+ τ4]) |\n|---|---|---|---|---|---|---|---|\n")
    for lf in sorted(c["leaves"]):
        x = c["leaves"][lf]
        lad = x.get("tau1_tau2") or {}
        t4d = (f"{len(x['tau4_parts_done'])} of 5 {x['tau4_parts_done'] or ''}" if "tau4_parts_done" in x else "–")
        s += (f"| {short(lf)} | {'post hoc' if x['source'].startswith('post hoc') else 'PR3'} | {x['verdict']} | "
              f"{len(x['ladder_parts_done'])} of 5 {x['ladder_parts_done'] or ''} | {len(x['extra_parts_done'])} of 5 "
              f"{x['extra_parts_done'] or ''} | {t4d} | {lad.get('band_shots_sample', '–')} | {cpu_cell(x['cpu_core_h'])} |\n")
    t, r = c["pr3_totals"], c["r10_cpu"]
    s += (f"\nPR3's main run: {t['cpu_core_h']:.2f} core-hours against its {t['guard_core_h']} guard. The r = 10 "
          f"follow-up: {r['core_h']:.2f} core-hours (the driver's {r['driver_core_h']:.2f} + τ4 {r['tau4_core_h']:.2f}) "
          f"against its cap of {r['cap_core_h']}.\n\n## Rules between rungs\n\n" + RUNG_HEAD)
    for lf in sorted(c["leaves"]):
        x = c["leaves"][lf]
        s += rung_rows(short(lf), x.get("tau1_tau2"), x.get("tau2_tau3"), x.get("tau3_tau4"))
    for lf in sorted(c["leaves"]):
        ex = c["leaves"][lf].get("tau2_tau3") or {}
        if ex and not ex["complete"]:
            s += (f"\n{short(lf)}, partial τ2 → τ3 (PR3): {ex['decisions_changed']} decisions changed on "
                  f"{ex['band_shots_sample_covered']} of {ex['band_shots_sample_leaf']} band shots; the largest limit the "
                  f"leaf could reach is {ex['limit_upper_bound']:.2f}, so rule 2′ "
                  + ("fails whatever the other parts give." if ex["rule2prime_fails_whatever_the_other_parts"]
                     else "is not decided by these parts.") + "\n")
    return s


if __name__ == "__main__":
    main()
