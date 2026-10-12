"""Pre-specified analysis of the simulation with the RL DEM as the true noise (notes/reservoir/sim_design.md sections
4-5 and the addendum logged before the run). EXPLORATORY: post hoc question, pre-specified design.

reservoir, 2026-10-10. Inputs: the plan; the simulated leaves' decode_willow.py outputs (dem tags rl_sim, rl_sim_t2,
rl_sim_t3) and sim_mpe.py outputs in scratch; for the comparison, the stored real outputs (decoder's decode folder, the
exact test's stage 1, K2's verdicts as found at run time), read only, through gapmech.read_leaf.

Q1, Q2 (r = 10): for A in {MPE, MWPM, corr. MWPM}, rho_A(d) = k_A / k_ML pooled over the leaves of distance d, with the
  per-shot delta-method SE stratified by leaf (d5_check.pooled's variance; CI on ln rho); Lambda_A/Lambda_ML =
  rho_A(3) / rho_A(5), log-scale CI with the distances independent. MPE: exact (mpe.py) at d = 3; at d = 5 Tesseract
  corrected by the certificate (sim_mpe.py). The same estimator on the real outputs at r = 10 (MPE: stage 1 at d = 3,
  the stored Tesseract at d = 5; rule A primary, no exclusion as sensitivity); s_A = ln Lambda_sim / ln Lambda_real
  and ln Lambda_real - ln Lambda_sim with delta-method CIs (simulation and real data independent).
Q3 (r = 10, and r = 30 when stage B ran): R_sim(d, r) = k_ML / sum min(P, 1 - P) (analyze_willow.py's calib_sums
  definitions: a NaN posterior counts as P = 1/2; SE = sqrt(sum_leaves k (1 - k/n)) / sum m); the gate on R_sim(3, r);
  beta(r) = sum m_ladder / sum m_tau2 - 1 on the tau2 reference shots, beta'(r) = sum m_tau2 / sum m_tau3 - 1 on the
  tau3 shots, percentile bootstrap over shots stratified by leaf (B = 10,000, default_rng(20261012), drawn in the order
  beta(10), beta'(10), beta(30), beta'(30)).
Readings: the phrases of the design's section 5, each with its CI condition (sim_design.md addendum for precedence).
Usage: python3 -I sim_analysis.py PLAN STAGES OUT_JSON   (STAGES: A or AB)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_gapmech"))
from gapmech import MISSING, read_leaf  # noqa: E402
from d5_check import BASES, P3, P5, k2_verdicts, rel_of  # noqa: E402  (path set by gapmech)

ARCH = "google_105Q_surface_code_d3_d5_d7"
ML_FILE = {3: "ml_exact.npz", 5: "ml_tau2_1e-6-1e-8_0.1.npz"}
Z95 = 1.959963984540054
DECS = ["mpe", "mwpm", "corr_mwpm"]
EQUIV = (0.98, 1.02)
GATE = (0.97, 1.03)
BETA_MARGIN = 0.03


def m_of(llr):
    L = np.asarray(llr, np.float64)
    L = np.where(np.isnan(L), 0.0, L)
    return 1 / (1 + np.exp(np.abs(L)))


def err_of(llr, obs):
    L = np.asarray(llr, np.float64)
    nan = np.isnan(L)
    return np.where(nan, False, np.where(nan, 0.0, L) > 0) != obs.astype(bool)


# ------------------------------------------------------------------ simulated leaves
def sim_leaf(plan, L):
    d, r, rel = L["d"], L["r"], L["leaf"]
    base = os.path.join(plan["paths"]["decode_out"], ARCH, rel, "rl_sim")
    sh = np.load(os.path.join(base, "shots.npz"), allow_pickle=False)
    n = L["shots"]
    idx = sh["idx"].astype(np.int64)
    if not np.array_equal(idx, np.arange(n)):
        raise SystemExit(f"{rel}: shots idx is not 0..n-1")
    rec = {"d": d, "r": r, "leaf": rel, "n": n, "obs": sh["obs"].astype(np.uint8), "preds": {}}
    z = np.load(os.path.join(base, ML_FILE[d]), allow_pickle=False)
    if not np.array_equal(z["idx"].astype(np.int64), idx):
        raise SystemExit(f"{rel}: ML idx mismatch")
    rec["ml"] = z["pred"].reshape(n, -1)[:, 0].astype(np.uint8)
    rec["llr"] = z["llr"].astype(np.float64)
    for a in ("mwpm", "corr_mwpm"):
        q = np.load(os.path.join(base, a + ".npz"), allow_pickle=False)
        if not np.array_equal(q["idx"].astype(np.int64), idx):
            raise SystemExit(f"{rel}: {a} idx mismatch")
        rec["preds"][a] = q["pred"].reshape(n, -1)[:, 0].astype(np.uint8)
    mp = os.path.join(plan["paths"]["mpe_dir"], rel.replace("/", "_") + ".npz")
    if r == 10 and os.path.exists(mp):
        q = np.load(mp, allow_pickle=False)
        if not np.array_equal(q["idx"].astype(np.int64), idx):
            raise SystemExit(f"{rel}: MPE idx mismatch")
        rec["preds"]["mpe"] = q["mpe"].astype(np.uint8)
        if d == 5:
            rec["preds"]["tesseract_uncorrected"] = q["tess"].astype(np.uint8)
            if not np.array_equal(q["ml"], rec["ml"]):
                raise SystemExit(f"{rel}: ML used by the certificate differs")
    rec["summary"] = json.load(open(os.path.join(base, "summary.json")))
    return rec


# ------------------------------------------------------------------ estimators
def leaf_counts(rec, a):
    pa = rec["preds"][a]
    m = pa != MISSING
    aw = (pa != rec["obs"]) & m
    mw = (rec["ml"] != rec["obs"]) & m
    return {"n": int(m.sum()), "kA": int(aw.sum()), "kM": int(mw.sum()), "n11": int(np.sum(aw & mw)),
            "b": int(np.sum(aw & ~mw)), "c": int(np.sum(~aw & mw))}


def pooled_rho(recs, a):
    """rho = sum k_A / sum k_ML over leaves; d5_check.pooled's stratified per-shot variance; CI on ln rho."""
    rows = [leaf_counts(x, a) for x in recs if a in x["preds"]]
    if not rows:
        return None
    n = np.array([x["n"] for x in rows], float)
    kA, kM, n11 = (np.array([x[k] for x in rows], float) for k in ("kA", "kM", "n11"))
    pA, pM, p11 = kA / n, kM / n, n11 / n
    KA, KM = kA.sum(), kM.sum()
    vA, vM, cAM = np.sum(n * pA * (1 - pA)), np.sum(n * pM * (1 - pM)), np.sum(n * (p11 - pA * pM))
    if KA == 0 or KM == 0:
        return {"leaves": len(rows), "shots": int(n.sum()), "k_A": int(KA), "k_ML": int(KM), "rho": None,
                "D_per_shot": (KA - KM) / n.sum() if n.sum() else None}
    q = KA / KM
    se = math.sqrt(max(vA - 2 * q * cAM + q * q * vM, 0.0)) / KM
    sl = se / q
    return {"leaves": len(rows), "shots": int(n.sum()), "k_A": int(KA), "k_ML": int(KM), "rho": q, "rho_minus_1": q - 1,
            "se_ln": sl, "ci95": [q * math.exp(-Z95 * sl), q * math.exp(Z95 * sl)],
            "A_wrong_ML_right": int(sum(x["b"] for x in rows)), "A_right_ML_wrong": int(sum(x["c"] for x in rows)),
            "D_per_shot": (KA - KM) / n.sum(), "LER_A": KA / n.sum(), "LER_ML": KM / n.sum()}


def lam(r3, r5):
    if not r3 or not r5 or r3["rho"] is None or r5["rho"] is None:
        return None
    ln = math.log(r3["rho"]) - math.log(r5["rho"])
    se = math.hypot(r3["se_ln"], r5["se_ln"])
    return {"lambda_ratio": math.exp(ln), "ln": ln, "se_ln": se,
            "ci95": [math.exp(ln - Z95 * se), math.exp(ln + Z95 * se)]}


def reading_sim(L):
    lo, hi = L["ci95"]
    grows, shrinks = hi < 1, lo > 1
    inside = EQUIV[0] <= lo and hi <= EQUIV[1]
    if grows:
        txt = "A's loss to ML grows with d under the exact prior"
        if inside:
            txt += " (by less than 2%: the CI also lies inside [0.98, 1.02])"
    elif shrinks:
        txt = "it shrinks"
        if inside:
            txt += " (by less than 2%: the CI also lies inside [0.98, 1.02])"
    elif inside:
        txt = "no growth, within +-2%"
    else:
        txt = "estimate and CI only (no phrase)"
    return {"ci_below_1": grows, "ci_above_1": shrinks, "ci_inside_0.98_1.02": inside, "reading": txt}


def against_real(Ls, Lr):
    if not Ls or not Lr:
        return None
    a, sa, b, sb = Ls["ln"], Ls["se_ln"], Lr["ln"], Lr["se_ln"]
    s = a / b
    ss = math.sqrt((sa / b) ** 2 + (a * sb / b ** 2) ** 2)
    dlt, sd = b - a, math.hypot(sa, sb)
    most = s - Z95 * ss > 0.5
    exceeds = dlt + Z95 * sd < 0
    txt = []
    if most:
        txt.append("most of the real growth appears under the exact prior")
    if exceeds:
        txt.append("the real growth exceeds the exact prior's")
    return {"share_s": s, "share_se": ss, "share_ci95": [s - Z95 * ss, s + Z95 * ss],
            "ln_real_minus_ln_sim": dlt, "se": sd, "ci95": [dlt - Z95 * sd, dlt + Z95 * sd],
            "share_lower95_above_0.5": most, "difference_ci_below_0": exceeds,
            "reading": "; ".join(txt) if txt else "estimate and CI only (no phrase)"}


def calib(recs):
    k = sm = vk = 0.0
    n = nan = 0
    for x in recs:
        e = err_of(x["llr"], x["obs"])
        kk = int(e.sum())
        k += kk
        sm += float(m_of(x["llr"]).sum())
        vk += kk * (1 - kk / max(x["n"], 1))
        n += x["n"]
        nan += int(np.isnan(x["llr"]).sum())
    if not sm:
        return None
    R = k / sm
    se = math.sqrt(vk) / sm
    return {"leaves": len(recs), "shots": n, "k": int(k), "sum_m": sm, "R": R, "se": se,
            "ci95": [R - Z95 * se, R + Z95 * se], "nan_posteriors": nan}


def boot_ratio(groups, rng, B):
    """sum a / sum b - 1 with a percentile bootstrap over shots stratified by group (list of (a, b) arrays)."""
    A = sum(float(a.sum()) for a, _ in groups)
    Bs = sum(float(b.sum()) for _, b in groups)
    sa = np.zeros(B)
    sb = np.zeros(B)
    for a, b in groups:
        ii = rng.integers(0, a.size, (B, a.size))
        sa += a[ii].sum(1)
        sb += b[ii].sum(1)
    v = sa / sb - 1
    return {"estimate": A / Bs - 1, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "shots": int(sum(a.size for a, _ in groups)), "B": B}


def ref_llr(plan, rel, tag, fname):
    p = os.path.join(plan["paths"]["decode_out"], ARCH, rel, tag, fname)
    z = np.load(p, allow_pickle=False)
    return z["idx"].astype(np.int64), z["llr"].astype(np.float64)


def beta_block(plan, recs5, r, rng, B):
    rf = plan["refs"][f"r{r}"]
    g2, g3, info = [], [], {"t2_nan": 0, "t3_nan": 0}
    for x in recs5:
        i2, l2 = ref_llr(plan, x["leaf"], rf["t2"]["dem_tag"], rf["t2"]["file"])
        i3, l3 = ref_llr(plan, x["leaf"], rf["t3"]["dem_tag"], rf["t3"]["file"])
        if not (np.array_equal(i2, np.arange(rf["t2"]["shots"])) and np.array_equal(i3, np.arange(rf["t3"]["shots"]))):
            raise SystemExit(f"{x['leaf']}: reference idx unexpected")
        g2.append((m_of(x["llr"][i2]), m_of(l2)))
        g3.append((m_of(l2[i3]), m_of(l3)))
        info["t2_nan"] += int(np.isnan(l2).sum())
        info["t3_nan"] += int(np.isnan(l3).sum())
        info.setdefault("ladder_decision_ne_t2", 0)
        info["ladder_decision_ne_t2"] += int(np.sum((np.nan_to_num(x["llr"][i2]) > 0) != (np.nan_to_num(l2) > 0)))
        info.setdefault("t2_decision_ne_t3", 0)
        info["t2_decision_ne_t3"] += int(np.sum((np.nan_to_num(l2[i3]) > 0) != (np.nan_to_num(l3) > 0)))
    b = boot_ratio(g2, rng, B)
    bp = boot_ratio(g3, rng, B)
    inside = lambda c: -BETA_MARGIN <= c[0] and c[1] <= BETA_MARGIN  # noqa: E731
    both = inside(b["ci95"]) and inside(bp["ci95"])
    return {"beta": b, "beta_prime": bp, "info": info, "beta_ci_inside_0.03": inside(b["ci95"]),
            "beta_prime_ci_inside_0.03": inside(bp["ci95"]),
            "reading": ("the ladder's truncation moves the predicted errors by less than 3%" if both
                        else "estimate and CI only (no phrase)")}


def ladder_totals(recs5):
    keys = ["n", "n_redecoded", "decisions_changed", "errors_tau1", "errors_final", "n_dP_gt_0.05", "nan_final"]
    t = {k: 0 for k in keys}
    mx = {"max_K_tau1": 0, "max_K_tau2": 0, "max_abs_dP_redecoded": 0.0}
    for x in recs5:
        ld = x["summary"]["decoders"][ML_NAME5]["ladder"]
        for k in keys:
            t[k] += ld[k]
        for k in mx:
            mx[k] = max(mx[k], ld[k])
    t.update(mx)
    t["share_redecoded"] = t["n_redecoded"] / t["n"] if t["n"] else None
    return t


ML_NAME5 = "ml_tau2:1e-6/1e-8:0.1"


# ------------------------------------------------------------------ real data
def real_block(plan, rounds, need_mpe):
    P = plan["paths"]
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = k2_verdicts(P["k2_dir"])
    failing = sorted(k for k, v in k2.items() if v["verdict"] != "pass")
    out = {"k2": {"read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "verdicts_found": len(k2),
                  "still_fail": failing}, "checks": chk, "recs": {}}
    for r in rounds:
        for d, pats in ((3, P3), (5, P5)):
            for p in pats:
                for b in BASES:
                    rec = read_leaf(P["real_arch"], P["real_decode"], d, p, b, r, chk)
                    rel = rel_of(d, p, b, r)
                    rec["leaf"] = rel
                    rec["d"], rec["r"] = d, r
                    if need_mpe and r == 10:
                        if d == 3:
                            z = np.load(os.path.join(P["stage1_dir"], "exact", rel.replace("/", "_") + ".npz"),
                                        allow_pickle=False)
                            full = np.full(rec["n"], MISSING, np.uint8)
                            full[z["idx"].astype(np.int64)] = z["mpe"]
                            rec["preds"]["mpe"] = full
                        else:
                            rec["preds"]["mpe"] = rec["preds"]["tesseract"]
                    for k in ("w", "wq"):
                        rec.pop(k, None)
                    out["recs"][(d, r, rel)] = rec
    return out


def sel(recs, d, r, excluded=()):
    return [x for (dd, rr, rel), x in recs.items() if dd == d and rr == r and rel not in excluded]


# ------------------------------------------------------------------ main
def main(plan_path, stages, out_json):
    t0 = time.time()
    plan = json.load(open(plan_path))
    rounds = [10] + ([30] if "B" in stages else [])
    leaves = [L for L in plan["leaves"] if L["stage"] in stages]
    sim = {}
    for L in leaves:
        sim[(L["d"], L["r"], L["leaf"])] = sim_leaf(plan, L)
    print(f"simulated leaves read: {len(sim)} ({time.time() - t0:.0f} s)", flush=True)
    real = real_block(plan, rounds, True)
    failing = set(real["k2"]["still_fail"])
    print(f"real leaves read: {len(real['recs'])} ({time.time() - t0:.0f} s)", flush=True)
    res = {"label": "EXPLORATORY (reservoir): pre-specified analysis of the simulation with the RL DEM as the true "
                    "noise (notes/reservoir/sim_design.md and its pre-run addendum). Post hoc question.",
           "stages": stages, "plan": os.path.abspath(plan_path), "real_k2": real["k2"], "real_checks": real["checks"],
           "q1_q2": {}, "q3": {}, "certificate": {}, "descriptive": {}}
    # certificate (d = 5, r = 10)
    cj = [json.load(open(os.path.join(plan["paths"]["mpe_dir"], L["leaf"].replace("/", "_") + ".json")))
          for L in leaves if L["d"] == 5 and L["r"] == 10]
    cert = {}
    for kind in ("discordant", "low_confidence", "concordant_sample"):
        c = {"shots": sum(x["certificate"][kind]["shots"] for x in cj),
             "proven": sum(x["certificate"][kind]["proven"] for x in cj),
             "misses": sum(x["certificate"][kind]["misses"] for x in cj),
             "verdicts": {v: sum(x["certificate"][kind]["verdicts"][v] for x in cj)
                          for v in ("same_class", "tie", "miss", "miss_undetermined", "unproven")},
             "milp_cpu_s": sum(x["certificate"][kind]["milp_cpu_s"] for x in cj)}
        c["miss_rate_upper95_one_sided_clopper_pearson"] = (
            float(stats.beta.ppf(0.95, c["misses"] + 1, c["proven"] - c["misses"]))
            if c["proven"] and c["misses"] < c["proven"] else (1.0 if c["proven"] else None))
        cert[kind] = c
    cert["tesseract"] = {"shots": sum(x["shots"] for x in cj),
                         "low_confidence_inf_beam": sum(x["tesseract"]["low_confidence_inf_beam"] for x in cj),
                         "low_confidence_after_fallback": sum(x["tesseract"]["low_confidence_after_fallback"] for x in cj),
                         "tesseract_ne_ml": sum(x["tesseract_ne_ml"] for x in cj),
                         "mpe_corrected_shots": sum(x["mpe_corrected_shots"] for x in cj),
                         "cpu_s": sum(x["tesseract"]["cpu_s"] for x in cj)}
    u = cert["concordant_sample"]["miss_rate_upper95_one_sided_clopper_pearson"]
    cert["u_upper95"] = u
    res["certificate"] = cert
    # Q1, Q2 at r = 10
    s3, s5 = sel(sim, 3, 10), sel(sim, 5, 10)
    r3 = sel(real["recs"], 3, 10)
    for a in DECS + ["tesseract_uncorrected"]:
        o = {}
        a_real = "mpe" if a == "tesseract_uncorrected" else a
        o["sim"] = {"rho_d3": pooled_rho(s3, "mpe" if a == "tesseract_uncorrected" else a), "rho_d5": pooled_rho(s5, a)}
        o["sim"]["lambda"] = lam(o["sim"]["rho_d3"], o["sim"]["rho_d5"])
        if o["sim"]["lambda"]:
            o["sim"]["reading"] = reading_sim(o["sim"]["lambda"])
        if a in ("mpe", "tesseract_uncorrected") and o["sim"]["rho_d5"]:
            D = o["sim"]["rho_d5"]["D_per_shot"]
            o["sim"]["d5_certificate"] = {"u_upper95": u, "D_per_shot": D, "u_over_D": (u / D) if (u is not None and D)
                                          else None}
        for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
            rr = {"rho_d3": pooled_rho(r3, a_real), "rho_d5": pooled_rho(sel(real["recs"], 5, 10, excl), a_real)}
            rr["lambda"] = lam(rr["rho_d3"], rr["rho_d5"])
            rr["against_sim"] = against_real(o["sim"]["lambda"], rr["lambda"])
            o["real_" + kname] = rr
        res["q1_q2"][a] = o
    # Q3
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    for r in rounds:
        q = {}
        S3, S5 = sel(sim, 3, r), sel(sim, 5, r)
        q["R_sim_d3"], q["R_sim_d5"] = calib(S3), calib(S5)
        g = q["R_sim_d3"]["ci95"]
        q["gate"] = {"R_sim_d3_ci95_inside_0.97_1.03": GATE[0] <= g[0] and g[1] <= GATE[1]}
        q["beta"] = beta_block(plan, S5, r, rng, plan["bootstrap_B"])
        if q["gate"]["R_sim_d3_ci95_inside_0.97_1.03"]:
            c5 = q["R_sim_d5"]["ci95"]
            q["ladder_inflates_R"] = c5[0] > 1
            q["reading_R"] = ("the ladder inflates R at d = 5" if c5[0] > 1 else "estimate and CI only (no phrase)")
            q["reading_beta"] = q["beta"]["reading"]
        else:
            q["reading_R"] = "withheld: the gate failed (R_sim(3, r) CI not inside [0.97, 1.03]); pipeline to be checked"
            q["reading_beta"] = "withheld with the d = 5 readings at this r"
        q["ladder"] = ladder_totals(S5)
        for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
            R3r, R5r = calib(sel(real["recs"], 3, r)), calib(sel(real["recs"], 5, r, excl))
            q["R_real_" + kname] = {"d3": R3r, "d5": R5r, "ratio_d5_over_d3": R5r["R"] / R3r["R"]}
            q["descriptive_" + kname] = {
                "share_of_real_d5_excess": (q["R_sim_d5"]["R"] - 1) / (R5r["R"] - 1) if R5r["R"] != 1 else None,
                "R_sim_d5_over_d3": q["R_sim_d5"]["R"] / q["R_sim_d3"]["R"], "R_real_d5_over_d3": R5r["R"] / R3r["R"]}
        res["q3"][f"r{r}"] = q
    # descriptive: LERs, simulated against real
    led = {}
    for r in rounds:
        for d in (3, 5):
            S = sel(sim, d, r)
            Rr = sel(real["recs"], d, r, failing if d == 5 else ())
            row = {}
            for a in ["ml"] + DECS:
                def ler(recs, a=a):
                    k = n = 0
                    for x in recs:
                        if a == "ml":
                            k += int(np.sum(x["ml"] != x["obs"]))
                            n += x["n"]
                        elif a in x["preds"]:
                            m = x["preds"][a] != MISSING
                            k += int(np.sum((x["preds"][a] != x["obs"]) & m))
                            n += int(m.sum())
                    return {"k": k, "n": n, "LER": k / n if n else None}
                row[a] = {"sim": ler(S), "real_ruleA": ler(Rr)}
            led[f"d{d}_r{r}"] = row
    res["descriptive"]["LER"] = led
    res["descriptive"]["sim_mpe_d3_ties"] = sum(
        json.load(open(os.path.join(plan["paths"]["mpe_dir"], L["leaf"].replace("/", "_") + ".json")))["ties"]
        for L in leaves if L["d"] == 3 and L["r"] == 10)
    res["descriptive"]["sim_mpe_ne_ml"] = {
        f"d{d}": int(sum(np.sum(x["preds"]["mpe"] != x["ml"]) for x in sel(sim, d, 10) if "mpe" in x["preds"]))
        for d in (3, 5)}
    res["run_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json + ".tmp", "w") as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(out_json + ".tmp", out_json)
    print(f"analysis {stages} done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 3 or a[1] not in ("A", "AB"):
        raise SystemExit(__doc__)
    main(*a)
