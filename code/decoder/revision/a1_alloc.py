#!/usr/bin/env python3
"""A1 question 2: the cheapest targeted sample that lifts the joint-95% per-round lower limit of g5 above a threshold
(decoder, 2026-10-13). Post hoc; existing outputs only; no decoding.
Setup as a1_perround.py (the non-band check's concordant flips per round count, C only, rev_d2_worstcase.Model).
The minimum's vertex is fixed (g5 is monotone in each coordinate): r10 C11, r30 C00, r50 C00 at their upper bounds,
the rest at their lower bounds (0: no flips there). Adding m shots to a driving cell with 0 new flips ("zero") or new
flips at the cell's observed rate ("observed") changes its upper bound; a cell sampled to its population is a census
(rate known; no bound). Joint 95%: Bonferroni over the driving cells that are not a census.
Cost = shot-rounds (10 m10 + 30 m30 + 50 m50) x CPU per shot-round.
usage: python3 -I -B a1_alloc.py --res <results/decoder/willow> --check <nonband_check> --final-r1 <json> --out OUT.json
"""
import argparse, functools, itertools, json, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import beta as BETA
import rev_d2_worstcase as D2

W = (10, 30, 50)
CT = ("C00", "C11")
DRIVE = ((10, "C11"), (30, "C00"), (50, "C00"))
THRESH = {"zero": 0.0, "g3": 0.0166, "g3_upper95": 0.0212}
DESIGNS = {"union_all_targets": {(10, "C11"): 10 ** 6, (30, "C00"): 800, (50, "C00"): 600},
           "ruleA_zero_g3": {(10, "C11"): 300, (30, "C00"): 500, (50, "C00"): 100},
           "ruleA_zero_0": {(10, "C11"): 200, (30, "C00"): 50, (50, "C00"): 0},
           "r10C11_census_only": {(10, "C11"): 10 ** 6, (30, "C00"): 0, (50, "C00"): 0}}


@functools.lru_cache(maxsize=None)
def cp_up(k, n, a):
    return 1.0 if n <= 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


@functools.lru_cache(maxsize=None)
def cp_lo(k, n, a):
    return 0.0 if n <= 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--final-r1", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    assert D2.md5(lp) == meta["shot_lists_npz_md5"]
    S = np.load(lp)
    leaves = [str(x) for x in S["leaves"]]; info = meta["leaves"]
    n = len(S["idx"])
    lid, strat, typ, ml, rounds = S["leaf_id"], S["stratum"], S["type"].astype(str), S["ml"].astype(np.int64), S["rounds"]
    d2 = np.full(n, -1); d3 = np.full(n, -1); done2 = np.zeros(n, bool); done3 = np.zeros(n, bool)
    for k in range(len(leaves)):
        z = np.load(os.path.join(a.check, "leaves", f"{k:02d}.npz")); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1); d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
        done2[r] = z["done_t2"]; done3[r] = z["done_t3"]
    flip = done2 & (np.where(done3, d3, d2) != ml)
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
    exA = sorted(x for x in (json.load(open(a.final_r1)).get("excluded_leaves") or []) if x in tables)
    cells = {}
    for r in W:
        for t in CT:
            mm = (strat == 1) & (typ == t) & (rounds == r)
            cells[(r, t)] = {"n": int(mm.sum()), "k": int((mm & flip).sum()), "N": int(sum(pops[rel][t] for rel in leaves if rnd[rel] == r))}
    coords = [(r, t) for r in W for t in CT]
    vmin = {(10, "C00"): 0, (10, "C11"): 1, (30, "C00"): 1, (30, "C11"): 0, (50, "C00"): 1, (50, "C11"): 0}

    def F_of(rates):
        F = {}
        for rel in leaves:
            f = {"D10": float(Dobs[rel]["D10"]), "D01": float(Dobs[rel]["D01"])}
            for t in CT:
                f[t] = pops[rel][t] * rates[(rnd[rel], t)]
            F[rel] = f
        return F

    def bounds(extra, mode):
        """per coordinate (lo, hi, is_census, nontrivial_hi) after adding extra[c] shots."""
        out = {}
        for c in coords:
            k, nn, N = cells[c]["k"], cells[c]["n"], cells[c]["N"]
            m = extra.get(c, 0)
            n2 = min(N, nn + m)
            k2 = k if mode == "zero" else k * n2 / nn
            out[c] = (n2, k2, n2 >= N)
        return out

    def g5_at(M, extra, mode, vertex=vmin, full=False):
        b = bounds(extra, mode)
        free = [c for c in DRIVE if not b[c][2]]
        alpha = 0.05 / max(1, len(free))
        rng = {}
        for c in coords:
            n2, k2, cen = b[c]
            rng[c] = (k2 / n2, k2 / n2) if cen else (cp_lo(k2, n2, alpha), cp_up(k2, n2, alpha))
        if not full:
            return M.g5(F_of({c: rng[c][vertex[c]] for c in coords}))["g"], alpha
        best = None
        for ch in itertools.product((0, 1), repeat=len(coords)):
            g = M.g5(F_of({c: rng[c][x] for c, x in zip(coords, ch)}))["g"]
            if best is None or g < best[0]:
                best = (g, ch)
        return best[0], alpha, {f"r{c[0]}|{c[1]}": ("hi" if x else "lo") for c, x in zip(coords, best[1])}

    grid1 = sorted(set([0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500,
                        cells[(10, "C11")]["N"] - cells[(10, "C11")]["n"]]))
    g23 = [0, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000,
           10000, 15000, 20000, 30000, 40000]
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": D2.md5(os.path.abspath(__file__)),
           "imports": {"rev_d2_worstcase.py": D2.md5(os.path.join(HERE, "rev_d2_worstcase.py"))},
           "cells": {f"r{c[0]}|{c[1]}": v for c, v in cells.items()}, "thresholds": THRESH, "rules": {}}
    for rule, ex in (("A", exA), ("none", [])):
        M = D2.Model(leaves, info, tables, ex)
        R = out["rules"][rule] = {"check_published_alpha3_min": g5_at(M, {}, "zero")[0]}
        for mode in ("zero", "observed"):
            evals = []
            for m1 in grid1:
                for m2 in g23:
                    for m3 in g23:
                        ext = {(10, "C11"): m1, (30, "C00"): min(m2, cells[(30, "C00")]["N"] - cells[(30, "C00")]["n"]),
                               (50, "C00"): min(m3, cells[(50, "C00")]["N"] - cells[(50, "C00")]["n"])}
                        g, al = g5_at(M, ext, mode)
                        evals.append((g, 10 * ext[(10, "C11")] + 30 * ext[(30, "C00")] + 50 * ext[(50, "C00")], ext, al))
            for tn, tv in THRESH.items():
                ok = [e for e in evals if e[0] >= tv]
                if not ok:
                    R[f"{mode}|{tn}"] = None
                    continue
                best = min(ok, key=lambda e: e[1])
                gf, alf, vert = g5_at(M, best[2], mode, full=True)
                R[f"{mode}|{tn}"] = {"extra": {f"r{c[0]}|{c[1]}": v for c, v in best[2].items()},
                                     "n_after": {f"r{c[0]}|{c[1]}": min(cells[c]["N"], cells[c]["n"] + v) for c, v in best[2].items()},
                                     "shot_rounds": best[1], "g5_min_at_fixed_vertex": best[0], "alpha_each": best[3],
                                     "g5_min_full_box": gf, "full_box_vertex": vert}
                # the cheapest with r10|C11 as a census, and with r10|C11 untouched
                for lab, pred in (("with_r10C11_census", lambda e: e[2][(10, "C11")] == grid1[-1]),
                                  ("without_r10C11", lambda e: e[2][(10, "C11")] == 0)):
                    sub = [e for e in ok if pred(e)]
                    R[f"{mode}|{tn}|{lab}"] = (None if not sub else
                                               {"extra": {f"r{c[0]}|{c[1]}": v for c, v in min(sub, key=lambda e: e[1])[2].items()},
                                                "shot_rounds": min(sub, key=lambda e: e[1])[1]})
            # the limit with r10|C11 alone as a census (no other new shots)
            R[f"{mode}|r10C11_census_only"] = g5_at(M, {(10, "C11"): grid1[-1]}, mode, full=True)[0]
            # named designs: full box (min at alpha / non-census driving cells, max at alpha/4) and the count-gap floor
            for dname, ext in DESIGNS.items():
                ext = {c: min(v, cells[c]["N"] - cells[c]["n"]) for c, v in ext.items()}
                b = bounds(ext, mode)
                free = [c for c in DRIVE if not b[c][2]]
                res = {}
                for lab, alpha in (("min", 0.05 / max(1, len(free))), ("max", 0.05 / 4)):
                    rng = {}
                    for c in coords:
                        n2, k2, cen = b[c]
                        rng[c] = (k2 / n2, k2 / n2) if cen else (cp_lo(k2, n2, alpha), cp_up(k2, n2, alpha))
                    vals = []
                    for ch in itertools.product((0, 1), repeat=len(coords)):
                        F = F_of({c: rng[c][x] for c, x in zip(coords, ch)})
                        vals.append((M.g5(F)["g"], M.count_gap(F)["rel"]))
                    if lab == "min":
                        res["g5_min"] = min(v[0] for v in vals); res["count_gap_min"] = min(v[1] for v in vals); res["alpha_min"] = alpha
                    else:
                        res["g5_max"] = max(v[0] for v in vals); res["alpha_max"] = alpha
                res["shot_rounds"] = sum(c[0] * v for c, v in ext.items())
                res["n_after"] = {f"r{c[0]}|{c[1]}": min(cells[c]["N"], cells[c]["n"] + v) for c, v in ext.items()}
                R[f"{mode}|design|{dname}"] = res
    out["seconds"] = time.time() - t0
    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps(out["rules"], indent=1, default=float))


if __name__ == "__main__":
    main()
