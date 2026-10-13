#!/usr/bin/env python3
"""A1 (fresh-eyes review of v2.0 draft 2), questions 1 (second part) and 2: the per-round-count limits of g5 from the
non-band check, (i) as published (R5), (ii) read with per-shot certificates, (iii) with a targeted sample added to
the cells that set them (decoder, 2026-10-13). Post hoc; reads existing outputs only; no decoding.

Setup as rev_d2_worstcase.py (imported from this folder): concordant flip rates per round count r (C00 and C11 at
r = 10, 30, 50; 6 coordinates), observed discordant flips, no tau3-only term ("C only", as in the paper's -4.5% to
+20.2%). Each coordinate takes its one-sided Clopper-Pearson bounds at level alpha; g5 and the count gap are
evaluated at all 64 vertices. R5's "each extreme at a joint 95%" uses alpha = 0.05/3 for the minimum (3 non-trivial
bounds) and 0.05/4 for the maximum (4).
  (i)   Reproduce R5: alpha = 0.05, 0.05/3, 0.05/4 (and 0.05/7, the whole box).
  (ii)  Certificate readings, per kappa: a sampled shot of a cell is uncertified if its tau2 decision is not
        certified by D2 at kappa or differs from tau1 (route D2), or also fails the tau1 -> tau2 change (route D2+L12).
        "sample": a cell's flips may only come from uncertified shots, bounded from the same sample: rate in
        [CP_lo(k), CP_up(u)]. "census": as if the certificate were known for every shot of the population and the
        uncertified share equalled the sample's: rate in [0, u/n], all uncertified shots flipping the adverse way.
  (iii) Add m shots to the cells that set the joint-95% minimum, with 0 new flips (or flips at the observed rate),
        and find the m that lifts the minimum above g3 = 1.66% and above 0; a cell sampled to its population size is
        a census. Costs from D1-0's measured MPS CPU per round (chi ladder) and the check's measured tau3 CPU per
        round on concordant shots. D1's own runs are not read.
usage: python3 -I -B a1_perround.py --res <results/decoder/willow> --check <supplementary/nonband_check>
           --final-r1 <final_round1/final_round1.json> --probe <D1-0 probe_out dir> --out OUT.json
"""
import argparse, itertools, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import beta as BETA
import rev_d2_worstcase as D2                                       # Model, md5; read-only

W = (10, 30, 50)
CT = ("C00", "C11")
G3 = 0.0166
KAPPAS = (2, 10, 100, 1000)
LN9 = math.log(9.0)


def cp_up(k, n, a):
    return 1.0 if n <= 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


def cp_lo(k, n, a):
    return 0.0 if n <= 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def sig(x):
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--final-r1", "--probe", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    assert D2.md5(lp) == meta["shot_lists_npz_md5"]
    S = np.load(lp)
    leaves = [str(x) for x in S["leaves"]]; info = meta["leaves"]
    n = len(S["idx"])
    lid, strat, typ = S["leaf_id"], S["stratum"], S["type"].astype(str)
    ml, rounds, llr1 = S["ml"].astype(np.int64), S["rounds"], S["llr_t1"].astype(np.float64)
    d2 = np.full(n, -1); d3 = np.full(n, -1); l2 = np.full(n, np.nan); dr2 = np.full(n, np.nan)
    done2 = np.zeros(n, bool); done3 = np.zeros(n, bool)
    for k in range(len(leaves)):
        z = np.load(os.path.join(a.check, "leaves", f"{k:02d}.npz")); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1); d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
        l2[r] = z["llr_t2"]; dr2[r] = z["drop_t2"]; done2[r] = z["done_t2"]; done3[r] = z["done_t3"]
    final = np.where(done3, d3, d2)
    flip = done2 & (final != ml)
    P1, P2 = sig(llr1), sig(l2)
    m1, m2 = np.abs(2 * P1 - 1), np.abs(2 * P2 - 1)
    # per-leaf tables and populations (as D2)
    tables = {}
    for rel in leaves:
        d = os.path.join(a.res, "decode", rel, "rl_s5")
        z = np.load(os.path.join(d, D2.DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        row = [i for i, x in enumerate(sh["shipped_names"]) if D2.LIBRA in str(x)]
        lib = sh["shipped"][row[0]].astype(np.int64); m = z["pred"][:, 0].astype(np.int64); o = sh["obs"].astype(np.int64)
        ea, eb = lib != o, m != o
        tables[rel] = {"n": int(len(o)), "n10": int((ea & ~eb).sum()), "n01": int((~ea & eb).sum()), "n11": int((ea & eb).sum())}
    pops = {x["leaf"]: x["population"] for x in info}
    rnd = {x["leaf"]: x["rounds"] for x in info}
    Dobs = {rel: {t: int(((lid == k) & (typ == t) & flip).sum()) for t in ("D10", "D01")} for k, rel in enumerate(leaves)}
    F1 = json.load(open(a.final_r1))
    exA = sorted(x for x in (F1.get("excluded_leaves") or []) if x in tables)
    # per cell: sampled n, flips k, uncertified u (two routes, per kappa), population N
    cells = {}
    for r in W:
        for t in CT:
            mm = (strat == 1) & (typ == t) & (rounds == r)
            c = {"n": int(mm.sum()), "k": int((mm & flip).sum()),
                 "N": int(sum(pops[rel][t] for rel in leaves if rnd[rel] == r)),
                 "N_rule_A": int(sum(pops[rel][t] for rel in leaves if rnd[rel] == r and rel not in exA)),
                 "C_low_sampled": int((mm & (np.abs(llr1) < LN9)).sum()), "u_D2": {}, "u_D2_L12": {}}
            for kap in KAPPAS:
                with np.errstate(invalid="ignore"):
                    cd = (d2 == ml) & (m2 > kap * dr2)
                    cl = cd & (m1 > kap * 2 * np.abs(P2 - P1))
                c["u_D2"][str(kap)] = int((mm & ~cd).sum()); c["u_D2_L12"][str(kap)] = int((mm & ~cl).sum())
            cells[(r, t)] = c
    coords = [(r, t) for r in W for t in CT]
    ADV = {"C00": True, "C11": False}                               # a flip of this type adds an ML error

    def F_of(rates):
        F = {}
        for rel in leaves:
            f = {"D10": float(Dobs[rel]["D10"]), "D01": float(Dobs[rel]["D01"])}
            for t in CT:
                f[t] = pops[rel][t] * rates[(rnd[rel], t)]
            F[rel] = f
        return F

    def box(M, bounds):
        """bounds[(r, t)] = (lo, hi). Extremes of g5 and of the count gap over the 64 vertices."""
        res = {"g5_min": (float("inf"), None), "g5_max": (-float("inf"), None),
               "cg_min": (float("inf"), None), "cg_max": (-float("inf"), None)}
        nan = 0
        for ch in itertools.product((0, 1), repeat=len(coords)):
            rates = {c: bounds[c][x] for c, x in zip(coords, ch)}
            F = F_of(rates)
            with np.errstate(all="ignore"):
                g = M.g5(F)["g"]
            cg = M.count_gap(F)["rel"]
            if not math.isfinite(g):                                # no series with error rates in (0, 0.5)
                nan += 1
                continue
            v = {cc: ("hi" if x else "lo") for cc, x in zip(coords, ch)}
            if g < res["g5_min"][0]: res["g5_min"] = (g, v)
            if g > res["g5_max"][0]: res["g5_max"] = (g, v)
            if cg < res["cg_min"][0]: res["cg_min"] = (cg, v)
            if cg > res["cg_max"][0]: res["cg_max"] = (cg, v)
        o = {k: {"value": v[0], "vertex": {f"r{r}|{t}": s for (r, t), s in v[1].items()} if v[1] else None} for k, v in res.items()}
        o["undefined_vertices"] = nan
        return o

    def cp_bounds(alpha, extra=None, rate_mode="zero"):
        """CP bounds per cell; extra[(r, t)] = m added shots (0 new flips, or flips at the observed rate)."""
        b = {}
        for c in coords:
            k, nn, N = cells[c]["k"], cells[c]["n"], cells[c]["N"]
            m = (extra or {}).get(c, 0)
            n2 = min(N, nn + m)
            k2 = k if rate_mode == "zero" else k * n2 / nn
            if n2 >= N:                                             # a census: the rate is known
                b[c] = (k2 / N, k2 / N)
            else:
                b[c] = (cp_lo(k2, n2, alpha), cp_up(k2, n2, alpha))
        return b

    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": D2.md5(os.path.abspath(__file__)),
           "imports": {"rev_d2_worstcase.py": D2.md5(os.path.join(HERE, "rev_d2_worstcase.py"))},
           "label": "A1 questions 1-2, post hoc, existing outputs only", "g3": G3, "rule_A_excluded": exA,
           "cells": {f"r{r}|{t}": c for (r, t), c in cells.items()}, "rules": {}}
    for rule, ex in (("A", exA), ("none", [])):
        M = D2.Model(leaves, info, tables, ex)
        R = out["rules"][rule] = {}
        R["estimate_per_round"] = M.g5(F_of({c: cells[c]["k"] / cells[c]["n"] for c in coords}))["g"]
        R["base"] = M.g5({})["g"]
        # (i) R5's boxes
        R["published"] = {lab: box(M, cp_bounds(al)) for lab, al in
                          (("each95", 0.05), ("alpha_over_3", 0.05 / 3), ("alpha_over_4", 0.05 / 4), ("whole_box_alpha_over_7", 0.05 / 7))}
        # drivers of the joint-95% minimum: set one coordinate at a time to its estimate
        bmin = cp_bounds(0.05 / 3); vmin = R["published"]["alpha_over_3"]["g5_min"]
        vert = {c: vmin["vertex"][f"r{c[0]}|{c[1]}"] for c in coords}
        base_rates = {c: bmin[c][1] if vert[c] == "hi" else bmin[c][0] for c in coords}
        drv = {}
        for c in coords:
            rr = dict(base_rates); rr[c] = cells[c]["k"] / cells[c]["n"]
            drv[f"r{c[0]}|{c[1]}"] = {"vertex": vert[c], "rate_at_vertex": base_rates[c],
                                      "flips_at_vertex": base_rates[c] * (cells[c]["N_rule_A"] if rule == "A" else cells[c]["N"]),
                                      "g5_if_this_cell_at_its_estimate": M.g5(F_of(rr))["g"]}
        R["drivers_of_joint95_min"] = {"g5_min": vmin["value"], "cells": drv}
        bmax = cp_bounds(0.05 / 4); vmax = R["published"]["alpha_over_4"]["g5_max"]
        vertx = {c: vmax["vertex"][f"r{c[0]}|{c[1]}"] for c in coords}
        base_x = {c: bmax[c][1] if vertx[c] == "hi" else bmax[c][0] for c in coords}
        drvx = {}
        for c in coords:
            rr = dict(base_x); rr[c] = cells[c]["k"] / cells[c]["n"]
            drvx[f"r{c[0]}|{c[1]}"] = {"vertex": vertx[c], "g5_if_this_cell_at_its_estimate": M.g5(F_of(rr))["g"]}
        R["drivers_of_joint95_max"] = {"g5_max": vmax["value"], "cells": drvx}
        # (ii) certificate readings
        R["certificate"] = {}
        for route in ("u_D2", "u_D2_L12"):
            for kap in KAPPAS:
                key = f"{route}|kappa{kap}"
                bs = {c: (cp_lo(cells[c]["k"], cells[c]["n"], 0.05 / 3), cp_up(cells[c][route][str(kap)], cells[c]["n"], 0.05 / 3))
                      for c in coords}
                bc = {c: (0.0, cells[c][route][str(kap)] / cells[c]["n"]) for c in coords}
                bs4 = {c: (cp_lo(cells[c]["k"], cells[c]["n"], 0.05 / 4), cp_up(cells[c][route][str(kap)], cells[c]["n"], 0.05 / 4))
                       for c in coords}
                b1, b2, b3 = box(M, bs), box(M, bs4), box(M, bc)
                R["certificate"][key] = {"sample_alpha_over_3_min": b1["g5_min"]["value"],
                                         "sample_alpha_over_4_max": b2["g5_max"]["value"],
                                         "census_projection": {k_: v["value"] for k_, v in b3.items() if k_ != "undefined_vertices"},
                                         "undefined_vertices": [b1["undefined_vertices"], b2["undefined_vertices"], b3["undefined_vertices"]],
                                         "uncertified": {f"r{c[0]}|{c[1]}": cells[c][route][str(kap)] for c in coords}}
        # (iii) targeted sample: the cells with a non-trivial 'hi' bound at the joint-95% minimum
        drive = [c for c in coords if vert[c] == "hi" and cells[c]["k"] < cells[c]["n"]]
        R["targeted"] = {"cells": [f"r{c[0]}|{c[1]}" for c in drive]}
        for mode in ("zero", "observed"):
            for thr_name, thr in (("g3", G3), ("zero", 0.0)):
                def gmin(m):
                    return box(M, cp_bounds(0.05 / 3, {c: m for c in drive}, mode))["g5_min"]["value"]
                cap = max(cells[c]["N"] - cells[c]["n"] for c in drive)
                if gmin(cap) < thr:
                    R["targeted"][f"{mode}|{thr_name}"] = {"m_per_cell": None, "note": "not reached even with a census of the cells"}
                    continue
                lo, hi = 0, 1
                while gmin(hi) < thr:
                    lo, hi = hi, min(cap, hi * 2)
                while hi - lo > max(1, lo // 100):
                    mid = (lo + hi) // 2
                    if gmin(mid) >= thr:
                        hi = mid
                    else:
                        lo = mid
                R["targeted"][f"{mode}|{thr_name}"] = {"m_per_cell": hi, "g5_min": gmin(hi),
                                                       "n_total_per_cell": {f"r{c[0]}|{c[1]}": min(cells[c]["N"], cells[c]["n"] + hi) for c in drive}}
        # the minimum as a function of m (zero new flips)
        R["targeted"]["curve_zero"] = {str(m): box(M, cp_bounds(0.05 / 3, {c: m for c in drive}))["g5_min"]["value"]
                                       for m in (0, 250, 500, 1000, 2000, 4000, 8000, 16000)}
    # costs (measured): D1-0 probe MPS CPU per round per chi; the check's tau2 and tau3 CPU per round
    chi_cost = {}
    for w in (0, 1):
        for ln in open(os.path.join(a.probe, f"mps_probe_w{w}.jsonl")):
            x = json.loads(ln)
            for chi, c in zip(x["chis"], x["cpu_s"]):
                chi_cost.setdefault(chi, []).append(c / x["rounds"])
    summ = json.load(open(os.path.join(a.check, "summary.json")))["stages"]
    out["costs"] = {"mps_s_per_round_by_chi_D1_0": {str(k): float(np.mean(v)) for k, v in sorted(chi_cost.items())},
                    "check_tau3_s_per_round_C_random": summ["C_t3_random"]["cpu_s"] / summ["C_t3_random"]["shot_rounds"],
                    "check_tau3_s_per_round_C_targeted": summ["C_t3_targeted"]["cpu_s"] / summ["C_t3_targeted"]["shot_rounds"],
                    "check_tau2_s_per_round_C": (summ["C_t2_A"]["cpu_s"] + summ["C_t2_B"]["cpu_s"]) /
                                                (summ["C_t2_A"]["shot_rounds"] + summ["C_t2_B"]["shot_rounds"]),
                    "check_tau2_s_per_round_D": summ["D_t2"]["cpu_s"] / summ["D_t2"]["shot_rounds"]}
    out["seconds"] = time.time() - t0
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(tmp, a.out)
    print(json.dumps({"seconds": out["seconds"],
                      "A_published": {k: (v["g5_min"]["value"], v["g5_max"]["value"], v["cg_min"]["value"])
                                      for k, v in out["rules"]["A"]["published"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
