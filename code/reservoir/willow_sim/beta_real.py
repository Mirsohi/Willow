"""beta and beta' on the real d = 5 shots: the ladder's truncation measured on real syndromes (notes/reservoir/
sim_design.md, block "beta on real shots (main's item 3a): pre-specified at 23:19Z", 2026-10-10, before the non-band
check started). EXPLORATORY, outcome-free, read-only: no outcome (observable) is read, and nothing is decoded.

reservoir, 2026-10-11. Inputs (read only; python3 -I):
  DECODE  results/decoder/willow/decode: <arch>/<leaf>/rl_s5/ml_tau2_1e-6-1e-8_0.1.npz, round 1's ladder (tau1 = 1e-6;
          the band |P_tau1 - 1/2| < 0.1 re-decoded at tau2 = 1e-8), 10,000 shots per d = 5 leaf (every 5th shot)
  CHECK   results/decoder/willow/supplementary/nonband_check: shot_lists.npz and .json (strata, populations, selections,
          u), leaves/<k>.npz (llr at tau2 and tau3 = 1e-9, done flags, tau3 reason), summary.json ("complete" required)
  SIM     results/reservoir/willow_sim/sim_analysis_AB.json: the simulation's beta and beta' (description only)
Per r in (10, 30), and r = 50 as a description, pooled over the 8 leaves at r and all their ladder shots, with
m = min(P, 1 - P) = 1 / (1 + exp|llr|) and a NaN posterior counted as P = 1/2 (m = 1/2):
  beta_real = sum m_ladder / sum m_tau2 - 1.
    Numerator, exact: the ladder's final llr (tau2 on the band shots B, tau1 on the other shots N).
    Denominator: sum_B m_ladder + sum of m_tau2 over the D rows (a census of N and D) + sum_{N and C} m_tau1 x
    [sum_C w m_tau2 / sum_C w m_tau1] over the C sample of the leaves at r (a combined ratio), with w = the leaf's C
    population / its C selected. sum_{N and C} m_tau1 = sum_N m_ladder - sum of m_tau1 over the D rows.
  beta'_real = sum_N m_tau2 / sum_N m_tau3 - 1, on the non-band shots only (the band shots' tau3 is K2's). Both sums are
    Horvitz-Thompson over the rows where tau3 ran: weight w for targeted rows (t3_reason 1, tau2 moved the shot) and
    w / 0.1 for random rows (t3_reason 2: not moved and u < 0.1); w = 1 for D rows and the leaf's C weight for C rows.
CIs: percentile bootstrap, B = 10,000, numpy default_rng(20261013), drawn in the order beta(10), beta'(10), beta(30),
beta'(30), beta(50), beta'(50); within one statistic, the leaves in shot-list order; within a leaf (beta'), the D rows
not moved first, then the C rows; an empty group draws nothing. Rows are resampled with replacement within leaf and
stratum, and the census parts are held fixed (the CI is design-based, over the check's sampling of the real shots):
  beta:  the C rows are resampled; the band shots, the D rows and the N-and-C tau1 total are fixed;
  beta': the C rows are resampled, each with its own tau3 status, and so are the D rows that tau2 did not move, each
         with its random-tau3 flag (this mimics the Bernoulli(0.1) selection); the D rows that tau2 moved (all of them
         have tau3) are fixed. The design names only beta's census parts; this is the reading for beta', fixed here
         before any estimate was computed.
Reading per r in (10, 30), as in the simulation: "the ladder's truncation moves the predicted errors by less than 3%"
(on the real shots) only if the 95% CIs of beta_real and beta'_real both lie inside [-0.03, 0.03]; otherwise the
estimates and CIs. Also given: beta_real against the simulation's beta at the same r, as a description only.
Checks: the check's summary is "complete" with script md5 b847ea8d...; shot_lists.npz's md5 equals the one recorded;
each ladder file's md5 equals the one shot_lists.json recorded; per leaf, outside the band the ladder's llr equals its
tau1 llr bitwise and is not NaN, the band count equals the recorded one; the shot-list rows of the leaf lie outside the
band with idx and tau1 llr equal to the ladder's; D rows = D10 + D01, C rows = the C selected, D + C populations = the
non-band count; leaves/<k>.npz rows are the leaf's shot-list rows with the same tau1 llr; every row has tau2; the tau3
reasons equal the check's rule recomputed here (targeted: decision changed, |P_tau2 - 1/2| < 0.1,
|P_tau2 - P_tau1| >= 0.05 or NaN; random: not targeted and u < 0.1) and every row with a reason has tau3.
Usage: nice -n 10 python3 -I beta_real.py DECODE CHECK SIM OUT_DIR  -> OUT_DIR/beta_real.json (through qrc.outpaths)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import out_path  # noqa: E402

LADDER = "rl_s5/ml_tau2_1e-6-1e-8_0.1.npz"
CHECK_SCRIPT_MD5 = "b847ea8db40163b317fbdb09fdfe5dce"
BAND = 0.1
RANDOM_P = 0.1
B_BOOT = 10_000
SEED = 20261013
RS = (10, 30, 50)
READ_RS = (10, 30)
PHRASE = "the ladder's truncation moves the predicted errors by less than 3% (on the real shots)"
NO_PHRASE = "estimate and CI only (no phrase)"


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def m_of(llr):
    """min(P, 1 - P) = 1 / (1 + exp|llr|); a NaN posterior counts as P = 1/2."""
    x = np.abs(np.asarray(llr, np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        m = 1.0 / (1.0 + np.exp(x))
    return np.where(np.isnan(x), 0.5, m)


def sig(llr):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-np.asarray(llr, np.float64)))


def load_leaf(dec, chk, k, name, info, S, lists_md5, checks):
    """One leaf: the ladder's sums and the check's rows, with every consistency check recorded."""
    lp = os.path.join(dec, name, LADDER)
    key = f"decode/{name}/{LADDER}"
    c = {"ladder_md5_equals_recorded": md5_file(lp) == lists_md5.get(key)}
    z = np.load(lp, allow_pickle=False)
    llr, llr1, red = z["llr"], z["llr_t1"], z["redecoded"].astype(bool)
    n = llr.size
    c["n_equals_recorded"] = n == info["n"]
    c["band_equals_recorded"] = int(red.sum()) == info["band"]
    c["nonband_llr_equals_t1_bitwise"] = bool(np.array_equal(llr[~red].view(np.uint32), llr1[~red].view(np.uint32)))
    c["nonband_llr_nan"] = int(np.isnan(llr[~red]).sum())
    with np.errstate(invalid="ignore"):
        band_rule = ~(np.abs(sig(llr1) - 0.5) >= BAND)
    c["band_rule_recomputed_differs"] = int(np.sum(band_rule != red))
    rows = np.flatnonzero(S["leaf_id"] == k)
    pos = S["pos"][rows]
    c["rows_outside_band"] = bool(not red[pos].any())
    c["rows_idx_equal_ladder"] = bool(np.array_equal(S["idx"][rows], z["idx"][pos].astype(np.int64)))
    c["rows_t1_equal_ladder_bitwise"] = bool(np.array_equal(S["llr_t1"][rows].view(np.uint32),
                                                            llr[pos].view(np.uint32)))
    pred_lad_rows = z["pred"].reshape(n, -1)[pos, 0].astype(np.uint8)
    c["rows_ml_equals_ladder_decision"] = bool(np.array_equal(S["ml"][rows].astype(np.uint8), pred_lad_rows))
    st = S["stratum"][rows]
    isD, isC = st == 0, st == 1
    popD = info["population"]["D10"] + info["population"]["D01"]
    popC = info["population"]["C00"] + info["population"]["C11"]
    c["D_rows_equal_population"] = int(isD.sum()) == popD
    c["C_rows_equal_selected"] = int(isC.sum()) == info["selected"]["C"]
    c["D_plus_C_population_equals_nonband"] = popD + popC == n - int(red.sum())
    R = np.load(os.path.join(chk, "leaves", f"{k:02d}.npz"), allow_pickle=False)
    c["leaf_rows_equal_shot_list_rows"] = bool(np.array_equal(R["row"], rows))
    c["leaf_t1_equal_bitwise"] = bool(np.array_equal(R["llr_t1"].view(np.uint32), S["llr_t1"][rows].view(np.uint32)))
    c["all_rows_have_t2"] = bool(R["done_t2"].all())
    P2, P1 = sig(R["llr_t2"]), sig(R["llr_t1"])
    ml = S["ml"][rows]
    with np.errstate(invalid="ignore"):
        moved = R["done_t2"] & ((R["pred_t2"] != ml) | ~(np.abs(P2 - 0.5) >= BAND) | ~(np.abs(P2 - P1) < 0.05))
    u = S["u"][rows]
    reason = R["t3_reason"].astype(int)
    c["t3_reason1_equals_moved"] = bool(np.array_equal(reason == 1, moved))
    c["t3_reason2_equals_unmoved_u_below_0.1"] = bool(np.array_equal(reason == 2, ~moved & (u < RANDOM_P)))
    c["t3_done_where_reason"] = bool(np.array_equal(R["done_t3"].astype(bool), reason > 0))
    for kk, v in c.items():
        checks.setdefault(kk, []).append(v)
    mt2, mt3, mt1 = m_of(R["llr_t2"]), m_of(np.where(reason > 0, R["llr_t3"], 0.0)), m_of(R["llr_t1"])
    wC = popC / info["selected"]["C"]
    m_lad = m_of(llr)
    t3 = reason > 0
    return {
        "name": name, "r": int(info["rounds"]), "n": int(n), "band": int(red.sum()),
        "sum_m_ladder_all": float(m_lad.sum()), "sum_m_ladder_band": float(m_lad[red].sum()),
        "sum_m_ladder_nonband": float(m_lad[~red].sum()),
        "D_t1": mt1[isD], "D_t2": mt2[isD], "D_t3": mt3[isD], "D_moved": moved[isD], "D_reason": reason[isD],
        "C_t1": mt1[isC], "C_t2": mt2[isC], "C_t3": mt3[isC], "C_reason": reason[isC], "wC": wC,
        "popD": popD, "popC": popC, "selC": int(isC.sum()),
        "nan": {"t1_rows": int(np.isnan(R["llr_t1"]).sum()), "t2_rows": int(np.isnan(R["llr_t2"]).sum()),
                "t3_rows": int(np.isnan(R["llr_t3"][t3]).sum())},
        "decisions": {"ladder_ne_t2_rows": int(np.sum(R["pred_t2"] != pred_lad_rows)),
                      "t2_ne_t3_rows": int(np.sum(R["pred_t3"][t3] != R["pred_t2"][t3])), "t3_rows": int(t3.sum())},
    }


def beta_parts(L):
    """Fixed parts and the C sums for beta at one r."""
    num = sum(x["sum_m_ladder_all"] for x in L)
    fixed = sum(x["sum_m_ladder_band"] + x["D_t2"].sum() for x in L)
    nc_t1 = sum(x["sum_m_ladder_nonband"] - x["D_t1"].sum() for x in L)
    return num, fixed, nc_t1


def beta_point(L):
    num, fixed, nc_t1 = beta_parts(L)
    a = sum(x["wC"] * x["C_t2"].sum() for x in L)
    b = sum(x["wC"] * x["C_t1"].sum() for x in L)
    ratio = a / b
    den = fixed + nc_t1 * ratio
    comp = {"sum_m_ladder": num, "sum_m_ladder_band": sum(x["sum_m_ladder_band"] for x in L),
            "sum_m_ladder_nonband": sum(x["sum_m_ladder_nonband"] for x in L),
            "D_sum_m_t1": float(sum(x["D_t1"].sum() for x in L)), "D_sum_m_t2": float(sum(x["D_t2"].sum() for x in L)),
            "NC_sum_m_t1": nc_t1, "C_ratio_t2_over_t1": ratio, "NC_sum_m_t2_estimated": nc_t1 * ratio,
            "sum_m_t2_estimated": den}
    return num / den - 1, comp


def beta_boot(L, rng):
    num, fixed, nc_t1 = beta_parts(L)
    a = np.zeros(B_BOOT)
    b = np.zeros(B_BOOT)
    for x in L:
        nC = x["C_t2"].size
        if nC == 0:
            continue
        ix = rng.integers(0, nC, size=(B_BOOT, nC))
        a += x["wC"] * x["C_t2"][ix].sum(1)
        b += x["wC"] * x["C_t1"][ix].sum(1)
    return num / (fixed + nc_t1 * a / b) - 1


def betap_arrays(x):
    """Per-row HT contributions for beta' (the sums are over the rows where tau3 ran)."""
    Dw = np.where(x["D_reason"] == 1, 1.0, np.where(x["D_reason"] == 2, 1.0 / RANDOM_P, 0.0))
    Cw = x["wC"] * np.where(x["C_reason"] == 1, 1.0, np.where(x["C_reason"] == 2, 1.0 / RANDOM_P, 0.0))
    mv = x["D_moved"]
    return {"Dfix2": float((Dw * x["D_t2"])[mv].sum()), "Dfix3": float((Dw * x["D_t3"])[mv].sum()),
            "Du2": (Dw * x["D_t2"])[~mv], "Du3": (Dw * x["D_t3"])[~mv], "C2": Cw * x["C_t2"], "C3": Cw * x["C_t3"]}


def betap_point(L):
    A = [betap_arrays(x) for x in L]
    s2 = sum(a["Dfix2"] + a["Du2"].sum() + a["C2"].sum() for a in A)
    s3 = sum(a["Dfix3"] + a["Du3"].sum() + a["C3"].sum() for a in A)
    return s2 / s3 - 1, {"HT_sum_m_t2": s2, "HT_sum_m_t3": s3}


def betap_boot(L, rng):
    s2 = np.zeros(B_BOOT)
    s3 = np.zeros(B_BOOT)
    for x in L:
        a = betap_arrays(x)
        s2 += a["Dfix2"]
        s3 += a["Dfix3"]
        for k2, k3 in (("Du2", "Du3"), ("C2", "C3")):
            n = a[k2].size
            if n == 0:
                continue
            ix = rng.integers(0, n, size=(B_BOOT, n))
            s2 += a[k2][ix].sum(1)
            s3 += a[k3][ix].sum(1)
    return s2 / s3 - 1


def ci(v):
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def main(a):
    t0, c0 = time.time(), time.process_time()
    summ = json.load(open(os.path.join(a.check, "summary.json")))
    lists = json.load(open(os.path.join(a.check, "shot_lists.json")))
    S = np.load(os.path.join(a.check, "shot_lists.npz"), allow_pickle=False)
    top = {"check_status_complete": summ.get("status") == "complete",
           "check_script_md5": summ.get("script_md5") == CHECK_SCRIPT_MD5,
           "shot_lists_md5_equals_recorded": md5_file(os.path.join(a.check, "shot_lists.npz")) ==
           lists.get("shot_lists_npz_md5") == summ.get("shot_lists_npz_md5")}
    if not top["check_status_complete"]:
        raise SystemExit("the non-band check's summary.json does not say complete; nothing computed")
    names = [str(x) for x in S["leaves"]]
    infos = {L["leaf"]: L for L in lists["leaves"]}
    checks = {}
    leaves = [load_leaf(a.decode, a.check, k, nm, infos[nm], S, lists["inputs_md5"], checks)
              for k, nm in enumerate(names)]
    sim = json.load(open(a.sim)) if a.sim != "-" and os.path.exists(a.sim) else None
    rng = np.random.default_rng(SEED)
    res = {"label": "reservoir, beta and beta' on the real d = 5 shots (sim_design.md, 23:19Z block); EXPLORATORY, "
                    "outcome-free, read-only",
           "inputs": {"decode": a.decode, "check": a.check, "sim": a.sim,
                      "check_summary_md5": md5_file(os.path.join(a.check, "summary.json")),
                      "shot_lists_npz_md5": md5_file(os.path.join(a.check, "shot_lists.npz")),
                      "shot_lists_json_md5": md5_file(os.path.join(a.check, "shot_lists.json")),
                      "sim_analysis_md5": md5_file(a.sim) if sim is not None else None,
                      "check_cpu_core_hours": summ.get("cpu_core_hours"), "check_made": summ.get("made")},
           "design": {"B": B_BOOT, "seed": SEED, "band": BAND, "random_tau3_p": RANDOM_P,
                      "draw_order": "beta(10), beta'(10), beta(30), beta'(30), beta(50), beta'(50); leaves in "
                                    "shot-list order; within a leaf (beta') the D rows not moved, then the C rows"},
           "checks": {**top, **{k: (f"{sum(bool(x) for x in v)} of {len(v)}" if all(isinstance(x, bool) for x in v)
                                    else {"max": max(v), "sum": sum(v)}) for k, v in checks.items()}},
           "per_r": {}}
    for r in RS:
        L = [x for x in leaves if x["r"] == r]
        b_est, comp = beta_point(L)
        b_bs = beta_boot(L, rng)
        bp_est, comp2 = betap_point(L)
        bp_bs = betap_boot(L, rng)
        b_ci, bp_ci = ci(b_bs), ci(bp_bs)
        inside = {"beta": -0.03 <= b_ci[0] and b_ci[1] <= 0.03, "beta_prime": -0.03 <= bp_ci[0] and bp_ci[1] <= 0.03}
        cnt = {"leaves": len(L), "shots": sum(x["n"] for x in L), "band": sum(x["band"] for x in L),
               "D_rows": sum(x["D_t2"].size for x in L), "D_moved": int(sum(x["D_moved"].sum() for x in L)),
               "D_random_t3": int(sum((x["D_reason"] == 2).sum() for x in L)),
               "D_not_moved": int(sum((~x["D_moved"]).sum() for x in L)),
               "C_rows": sum(x["selC"] for x in L), "C_population": sum(x["popC"] for x in L),
               "C_targeted_t3": int(sum((x["C_reason"] == 1).sum() for x in L)),
               "C_random_t3": int(sum((x["C_reason"] == 2).sum() for x in L)),
               "nan": {k: sum(x["nan"][k] for x in L) for k in ("t1_rows", "t2_rows", "t3_rows")},
               "decisions": {k: sum(x["decisions"][k] for x in L) for k in ("ladder_ne_t2_rows", "t2_ne_t3_rows",
                                                                            "t3_rows")}}
        out = {"counts": cnt, "beta": {"estimate": b_est, "ci95": b_ci, "B": B_BOOT, "components": comp},
               "beta_prime": {"estimate": bp_est, "ci95": bp_ci, "B": B_BOOT, "components": comp2},
               "ci_inside_0.03": inside}
        if r in READ_RS:
            out["reading"] = PHRASE if (inside["beta"] and inside["beta_prime"]) else NO_PHRASE
        else:
            out["reading"] = "descriptive only (r = 50): no reading"
        if sim is not None and f"r{r}" in (sim.get("q3") or {}):
            sb = sim["q3"][f"r{r}"]["beta"]
            out["simulation_description"] = {"beta": sb["beta"], "beta_prime": sb["beta_prime"],
                                             "note": "simulated shots (RL DEM as the truth); description only"}
        res["per_r"][str(r)] = out
        print(f"r = {r}: beta {100 * b_est:+.2f}% [{100 * b_ci[0]:+.2f}, {100 * b_ci[1]:+.2f}]%; beta' {100 * bp_est:+.3f}% "
              f"[{100 * bp_ci[0]:+.3f}, {100 * bp_ci[1]:+.3f}]%; {out['reading']}", flush=True)
    res["run_seconds"] = time.time() - t0
    res["cpu_seconds"] = time.process_time() - c0
    with open(out_path(os.path.join(a.out_dir, "beta_real.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["checks"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("decode", "check", "sim", "out_dir"):
        ap.add_argument(k)
    main(ap.parse_args())
