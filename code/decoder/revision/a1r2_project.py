#!/usr/bin/env python3
"""A1-R2-P: a planning projection for A1-R2, not a result (decoder, 2026-10-13). Declared as A1-R2-P in
notes/decoder/willow_revision_declarations.md before it ran. It reads no new decoding: its only A1-R2 input is the
analysed a1r2_results.json (a1r2_analyze.py), and it changes one cell's counts by assumption.

Question (main, 2026-10-13): if long runs are allowed, would finishing A1-R2's declared design be worth running? The
sparse run stopped with 195 of r = 30 C00's 800 new rows undone, so that cell holds 1,361 shots, not the planned 1,556.

Steps:
  1. Load what a1r2_analyze.py loads for the box (the check's discordant flips, the leaf tables, the populations, rule A's
     exclusions), with the same code. Take the cells (k, n, N, census) from a1r2_results.json's cells_primary, and the
     census cell's per-leaf flips from its flipped rows (t3 differs from m).
  2. Reproduce: evaluate the box (a1r2_analyze.py's evaluate, copied unchanged) on those cells. Every primary number of
     a1r2_results.json (estimate, joint-95% minimum and maximum with alpha, vertex and count-gap floor, each-95% range)
     must come out equal, under rules A and none, or the script stops. --validate-only stops here.
  3. Project r = 30 C00 to n = 1,361 + 195 = 1,556 shots:
       zero_new_flips   k = 1 (the 195 rows have no flips);
       observed_rate    k = 1 x 1,556 / 1,361 (the cell's observed rate kept, as a1_alloc.py's "observed" mode:
                        fractional k, bounds from the same Beta quantiles);
       one_new_flip     k = 2 (for reference: flips are whole numbers).
     Every other cell is unchanged. The joint-95% minimum (alpha / m, m the non-trivial bounds at the vertex) and the
     rest of the box are evaluated as in step 2, for rules A and none.
usage: python3 -I -B a1r2_project.py --res R --check C --final-r1 F --a1r2 A1R2_RESULTS_JSON --out OUT.json [--validate-only]
"""
import argparse, copy, functools, itertools, json, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import beta as BETA
import rev_d2_worstcase as D2                                       # Model, md5, DN, LIBRA; read-only

W = (10, 30, 50)
CT = ("C00", "C11")
COORDS = [(r, t) for r in W for t in CT]
DRIVE = {"r10_C11": (10, "C11"), "r30_C00": (30, "C00"), "r50_C00": (50, "C00")}
VMIN = {(10, "C00"): 0, (10, "C11"): 1, (30, "C00"): 1, (30, "C11"): 0, (50, "C00"): 1, (50, "C11"): 0}
VMAX = {c: 1 - v for c, v in VMIN.items()}
A1R2_ANALYZE_MD5 = "85270c2fa4a8f85f5791588bab2f2bdf"
TARGET_R30_C00 = 1556                                               # the design's planned n (A1-R2, a1_alloc union design)


@functools.lru_cache(maxsize=None)
def cp_up(k, n, a):
    return 1.0 if n <= 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


@functools.lru_cache(maxsize=None)
def cp_lo(k, n, a):
    return 0.0 if n <= 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--final-r1", "--a1r2", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    P = json.load(open(a.a1r2))
    assert P["script_md5"] == A1R2_ANALYZE_MD5, "a1r2_results.json was not made by the declared a1r2_analyze.py"
    # ------------------------------------------------------------------ as a1r2_analyze.py (the box's inputs)
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    assert D2.md5(lp) == meta["shot_lists_npz_md5"] == P["inputs_md5"]["shot_lists.npz"]
    assert D2.md5(a.final_r1) == P["inputs_md5"]["final_round1.json"]
    S = np.load(lp)
    leaves = [str(x) for x in S["leaves"]]; info = meta["leaves"]
    n = len(S["idx"])
    lid, typ, ml = S["leaf_id"], S["type"].astype(str), S["ml"].astype(np.int64)
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
    assert exA == P["rule_A_excluded"]
    # ------------------------------------------------------------------ the analysed cells, from a1r2_results.json
    cells = {}
    for key, v in P["cells_primary"].items():
        r_, t_ = key.split("|")
        cells[(int(r_[1:]), t_)] = {"k": v["k"], "n": v["n"], "N": v["N"], "census": v["census"]}
    census = {}
    for c, v in cells.items():
        if v["census"]:
            name = next(nm for nm, cc in DRIVE.items() if cc == c)
            nr = P["new_rows"][name]
            assert nr["decoded"] == v["n"] == v["N"], "a census cell must be wholly A1-R2's rows"
            per_leaf = {}
            for fr in nr["flipped_rows"]:
                if fr["t3"] != fr["m"]:
                    per_leaf[fr["leaf"]] = per_leaf.get(fr["leaf"], 0) + 1
            assert sum(per_leaf.values()) == v["k"] == nr["flips_t3"]
            census[c] = per_leaf

    # ------------------------------------------------------------------ a1r2_analyze.py's evaluate, unchanged
    def evaluate(M, cells, census):
        def F_of(rates):
            F = {}
            for rel in leaves:
                f = {"D10": float(Dobs[rel]["D10"]), "D01": float(Dobs[rel]["D01"])}
                for t in CT:
                    c = (rnd[rel], t)
                    f[t] = float(census[c].get(rel, 0)) if c in census else pops[rel][t] * rates[c]
                F[rel] = f
            return F

        free = [c for c in COORDS if c not in census]

        def nontrivial(c, side):
            k_, n_ = cells[c]["k"], cells[c]["n"]
            return (side == 1 and k_ < n_) or (side == 0 and k_ > 0)

        def box(alpha):
            rng_ = {c: (cp_lo(cells[c]["k"], cells[c]["n"], alpha), cp_up(cells[c]["k"], cells[c]["n"], alpha)) for c in free}
            vals = []
            for ch in itertools.product((0, 1), repeat=len(free)):
                rates = {c: rng_[c][x] for c, x in zip(free, ch)}
                F = F_of(rates)
                vals.append((M.g5(F)["g"], M.count_gap(F)["rel"], {f"r{c[0]}|{c[1]}": ("hi" if x else "lo") for c, x in zip(free, ch)}))
            return vals

        out = {"estimate": M.g5(F_of({c: cells[c]["k"] / cells[c]["n"] for c in free}))["g"],
               "census_cells": [f"r{c[0]}|{c[1]}" for c in census]}
        mmin = sum(nontrivial(c, VMIN[c]) for c in free)
        mmax = sum(nontrivial(c, VMAX[c]) for c in free)
        out["n_bounds_min"], out["n_bounds_max"] = mmin, mmax
        vmin = box(0.05 / max(1, mmin)); vmax = box(0.05 / max(1, mmax)); v95 = box(0.05)
        bmin = min(vmin, key=lambda v: v[0]); bmax = max(vmax, key=lambda v: v[0])
        out["joint95_min"] = {"g5": bmin[0], "alpha": 0.05 / max(1, mmin), "vertex": bmin[2],
                              "count_gap_floor": min(v[1] for v in vmin)}
        out["joint95_max"] = {"g5": bmax[0], "alpha": 0.05 / max(1, mmax), "vertex": bmax[2]}
        out["each95"] = {"g5_min": min(v[0] for v in v95), "g5_max": max(v[0] for v in v95),
                         "count_gap_floor": min(v[1] for v in v95)}
        return out

    EX = {"A": exA, "none": []}

    def model(rule):                                                # a fresh Model per evaluation, as a1r2_analyze.py
        return D2.Model(leaves, info, tables, EX[rule])

    # ------------------------------------------------------------------ step 2: reproduce A1-R2's primary numbers
    repro = {rule: evaluate(model(rule), cells, census) for rule in EX}
    mism = []
    for rule in EX:
        got, want = json.loads(json.dumps(repro[rule], default=float)), P["primary"][rule]
        if got != want:
            mism.append(rule)
    print(json.dumps({"reproduced_primary": not mism, "mismatch_rules": mism}))
    if mism:
        sys.exit("the reproduction differs from a1r2_results.json; no projection made")
    if a.validate_only:
        return
    # ------------------------------------------------------------------ step 3: the projection
    c30 = (30, "C00")
    nr30 = P["new_rows"]["r30_C00"]
    k0, n0 = cells[c30]["k"], cells[c30]["n"]
    n_target = n0 + (nr30["selected"] - nr30["decoded"])
    assert n_target == TARGET_R30_C00, n_target
    scen = {"zero_new_flips": k0, "observed_rate": k0 * n_target / n0, "one_new_flip": k0 + 1}
    O = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": D2.md5(os.path.abspath(__file__)),
         "imports": {"rev_d2_worstcase.py": D2.md5(os.path.join(HERE, "rev_d2_worstcase.py"))},
         "label": "A1-R2-P: planning projection, not a result. It reads no new decoding. Not for the paper.",
         "a1r2_results_md5": D2.md5(a.a1r2), "reproduced_primary": True,
         "r30_C00_now": {"k": k0, "n": n0}, "r30_C00_projected_n": n_target, "scenarios": {}}
    for nm, k2 in scen.items():
        cl = copy.deepcopy(cells)
        cl[c30]["k"], cl[c30]["n"] = k2, n_target
        O["scenarios"][nm] = {"r30_C00": {"k": k2, "n": n_target},
                              "rules": {rule: evaluate(model(rule), cl, census) for rule in EX}}
    O["seconds"] = time.time() - t0
    with open(a.out + ".tmp", "w") as f:
        json.dump(O, f, indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    show = {nm: {"k": round(v["r30_C00"]["k"], 4), "n": v["r30_C00"]["n"],
                 **{rule: {"joint95_min": v["rules"][rule]["joint95_min"]["g5"], "alpha": v["rules"][rule]["joint95_min"]["alpha"],
                           "count_gap_floor": v["rules"][rule]["joint95_min"]["count_gap_floor"]} for rule in v["rules"]}}
            for nm, v in O["scenarios"].items()}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main()
