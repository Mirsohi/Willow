#!/usr/bin/env python3
"""A1-R2-X: a post hoc, descriptive addendum to A1-R2 (decoder, 2026-10-14). Declared as A1-R2-X in
notes/decoder/willow_revision_declarations.md before it ran. Stored outputs only; no decoding.

It answers main's request on the fresh-eyes check of draft 6 (notes/orchestrator/fresh_eyes_v2.0_draft6.md, B1, B2, B4).

1. Libra (B1, B2). A1-R2's joint-95% box (a1r2_results.json): its minimising and maximising vertices under rules A and
   none, rebuilt from the stored cells, vertex and alpha (the rebuilt g5 must equal the stored one exactly). At each:
   - g5 with its own shot-noise 95% CI: entry 7's corrected SE on the vertex's modified tables, by
     rev_d2_worstcase.Model.g5, the function that rebuilds the census rows (nonband_results.json) and their CIs;
     CI = g5 +- 1.96 SE;
   - Delta = ln(1 + g5) - ln(1 + g3), g3 = Libra's d = 3 gap on {10, 30, 50} at its estimate (final_round1.json), with
     SE(Delta) = sqrt((SE5 / (1 + g5))^2 + (SE3 / (1 + g3))^2) and the 95% CI Delta +- 1.96 SE, as final_round1.py
     computes the headline's 0.048 +- 0.012 (checked here).
2. Other decoders (B4). The same box (A1-R2's cells; the minimum's box at its alpha, the maximum's at its alpha; every
   vertex enumerated) moves ML(RL)'s decisions, so it applies to each decoder's paired gap to ML(RL) at d = 5:
   Harmony and correlated matching (shipped, RL prior), our correlated MWPM and MWPM (RL prior) on {10, 30, 50}, and
   Tesseract on {10, 30}, the d = 5 round counts it ran within the box's coverage. Rules A (the final analysis's
   excluded leaves) and none.
   - Flips on known shots (the check's discordant flips; A1-R2's r = 10 C11 census) use the decoder's own decision on
     that shot. A bounded rate moves ML's decision on that share of a leaf's non-band shots of the type (C00 or C11);
     the flipped shots are split between the decoder's right and wrong decisions in proportion to the decoder's errors
     among those shots of that leaf (the expected split). The decoder's own errors never change; ML's change as in
     Libra's box. For Libra this reproduces rev_d2_worstcase.Model exactly (checked at every vertex).
   - Per decoder and rule: g5 at its own minimising vertex (min over the minimum's box) and maximising vertex (max over
     the maximum's box), the shot-noise 95% CI at both, Delta at both against that decoder's g3 on the same round
     counts (scout's willow_v2d3_derived.json, c12_libra_windows, for the four on {10, 30, 50}; reservoir's
     exact_analysis.json, degeneracy_factor 10_30, for Tesseract, whose d = 3 decisions are the exact MPE's), with
     SE(Delta) as in 1; and the number of non-trivial bounds at each vertex, counting the coordinates in the window.
   - Reading (declared): the floor is the smaller rule's minimising-vertex g5. Above the upper 95% limit of the
     decoder's g3 on the same round counts: growth holds per round count. Between 0 and that limit: g5 is positive per
     round count, but growth is not established. At or below 0: the condition stays.
Checks (all must pass, or the script stops before writing): A1-R2's primary numbers reproduced exactly; the per-shot
tables reproduce rev_d2_worstcase's Libra tables and the check's populations, discordant flips and census flips; each
decoder's g5 with no flips reproduces scout's and reservoir's stored g5 and SE, and its Delta and SE(Delta); Libra's
headline Delta and SE reproduce final_round1.json. --validate-only runs the checks and prints only pass or fail.
usage: python3 -I -B a1r2x_analyze.py --res R --check C --final-r1 F --a1r2 A1R2_JSON --v2d3 V2D3_JSON --ea EA_JSON
                                       --out OUT.json [--validate-only]
"""
import argparse, functools, itertools, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import beta as BETA, norm as NORM
import rev_d2_worstcase as D2                                       # Model, series_fit, g_from_series; read-only

W3 = (10, 30, 50)
W2 = (10, 30)
CT = ("C00", "C11")
TYPES = ("D10", "D01", "C00", "C11")
COORDS = [(r, t) for r in W3 for t in CT]
DRIVE = {"r10_C11": (10, "C11"), "r30_C00": (30, "C00"), "r50_C00": (50, "C00")}
VMIN = {(10, "C00"): 0, (10, "C11"): 1, (30, "C00"): 1, (30, "C11"): 0, (50, "C00"): 1, (50, "C11"): 0}
VMAX = {c: 1 - v for c, v in VMIN.items()}
ML_UP = {"C00": True, "C11": False, "D10": True, "D01": False}   # a flip of this type makes ML wrong (True) or right
A1R2_ANALYZE_MD5 = "85270c2fa4a8f85f5791588bab2f2bdf"
G3_UPPER_A1R2 = 0.0212                                              # A1-R2's declared constant for Libra
Z = 1.96
DECODERS = {"harmony": ("shipped", "harmony_decoder_with_rl_optimized_prior", W3),
            "corrmatch": ("shipped", "correlated_matching_decoder_with_rl_optimized_prior", W3),
            "corr_mwpm": ("rl_s5", "corr_mwpm.npz", W3),
            "mwpm": ("rl_s5", "mwpm.npz", W3),
            "tesseract": ("rl_s5t", "tesseract.npz", W2)}
TOL = 1e-9


@functools.lru_cache(maxsize=None)
def cp_up(k, n, a):
    return 1.0 if n <= 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


@functools.lru_cache(maxsize=None)
def cp_lo(k, n, a):
    return 0.0 if n <= 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def growth(g5, se5, g3, se3):
    d = math.log(1 + g5) - math.log(1 + g3)
    se = math.sqrt((se5 / (1 + g5)) ** 2 + (se3 / (1 + g3)) ** 2)
    return {"Delta": d, "se": se, "ci95": [d - Z * se, d + Z * se]}


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--final-r1", "--a1r2", "--v2d3", "--ea", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
        if not ok:
            print(json.dumps({"failed_check": name, "detail": detail}, default=float))
            sys.exit(f"check failed: {name}")

    P = json.load(open(a.a1r2))
    check("a1r2_results_made_by_declared_script", P["script_md5"] == A1R2_ANALYZE_MD5)
    # ------------------------------------------------------------------ as a1r2_analyze.py: the box's inputs
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    check("shot_lists_md5", D2.md5(lp) == meta["shot_lists_npz_md5"] == P["inputs_md5"]["shot_lists.npz"])
    check("final_round1_md5", D2.md5(a.final_r1) == P["inputs_md5"]["final_round1.json"])
    S = np.load(lp)
    leaves = [str(x) for x in S["leaves"]]; info = meta["leaves"]
    n = len(S["idx"])
    lid, typ, ml_s, pos_s, strat = S["leaf_id"], S["type"].astype(str), S["ml"].astype(np.int64), S["pos"], S["stratum"]
    d2 = np.full(n, -1); d3 = np.full(n, -1); done2 = np.zeros(n, bool); done3 = np.zeros(n, bool)
    for k in range(len(leaves)):
        z = np.load(os.path.join(a.check, "leaves", f"{k:02d}.npz")); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1); d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
        done2[r] = z["done_t2"]; done3[r] = z["done_t3"]
    flip = done2 & (np.where(done3, d3, d2) != ml_s)
    tables = {}
    for rel_ in leaves:
        d = os.path.join(a.res, "decode", rel_, "rl_s5")
        z = np.load(os.path.join(d, D2.DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        row = [i for i, x in enumerate(sh["shipped_names"]) if D2.LIBRA in str(x)]
        lib = sh["shipped"][row[0]].astype(np.int64); m = z["pred"][:, 0].astype(np.int64); o = sh["obs"].astype(np.int64)
        ea, eb = lib != o, m != o
        tables[rel_] = {"n": int(len(o)), "n10": int((ea & ~eb).sum()), "n01": int((~ea & eb).sum()), "n11": int((ea & eb).sum())}
    pops = {x["leaf"]: x["population"] for x in info}
    rnd = {x["leaf"]: x["rounds"] for x in info}
    Dobs = {rel_: {t: int(((lid == k) & (typ == t) & flip).sum()) for t in ("D10", "D01")} for k, rel_ in enumerate(leaves)}
    exA = sorted(x for x in (json.load(open(a.final_r1)).get("excluded_leaves") or []) if x in tables)
    check("rule_A_exclusions", exA == P["rule_A_excluded"])
    EX = {"A": exA, "none": []}
    # ------------------------------------------------------------------ A1-R2's cells and census (as a1r2_project.py)
    cells = {}
    for key, v in P["cells_primary"].items():
        r_, t_ = key.split("|")
        cells[(int(r_[1:]), t_)] = {"k": v["k"], "n": v["n"], "N": v["N"], "census": v["census"]}
    census, census_shots = {}, {}
    for c, v in cells.items():
        if v["census"]:
            name = next(nm for nm, cc in DRIVE.items() if cc == c)
            nr = P["new_rows"][name]
            check(f"census_{name}_is_A1R2_rows", nr["decoded"] == v["n"] == v["N"])
            per_leaf = {}
            for fr in nr["flipped_rows"]:
                if fr["t3"] != fr["m"]:
                    per_leaf[fr["leaf"]] = per_leaf.get(fr["leaf"], 0) + 1
                    census_shots.setdefault(fr["leaf"], []).append((c[1], int(fr["pos"])))
            check(f"census_{name}_flips", sum(per_leaf.values()) == v["k"] == nr["flips_t3"])
            census[c] = per_leaf
    free = [c for c in COORDS if c not in census]

    # ------------------------------------------------------------------ a1r2_analyze.py's evaluate, unchanged
    def F_of(rates):
        F = {}
        for rel_ in leaves:
            f = {"D10": float(Dobs[rel_]["D10"]), "D01": float(Dobs[rel_]["D01"])}
            for t in CT:
                c = (rnd[rel_], t)
                f[t] = float(census[c].get(rel_, 0)) if c in census else pops[rel_][t] * rates[c]
            F[rel_] = f
        return F

    def evaluate(M, cells, census):
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

    def model(rule):
        return D2.Model(leaves, info, tables, EX[rule])

    for rule in EX:
        got = json.loads(json.dumps(evaluate(model(rule), cells, census), default=float))
        check(f"a1r2_primary_reproduced_{rule}", got == P["primary"][rule])

    def rates_at(vertex, alpha):
        out = {}
        for c in free:
            k_, n_ = cells[c]["k"], cells[c]["n"]
            out[c] = cp_up(k_, n_, alpha) if vertex[f"r{c[0]}|{c[1]}"] == "hi" else cp_lo(k_, n_, alpha)
        return out

    # ------------------------------------------------------------------ Libra's d = 3 gap and the headline (final_round1.json)
    FR = json.load(open(a.final_r1))
    g3L = FR["entry7_se"]["pooled_gaps_W5:d3|libra.rl vs ML.rl"]
    d5r = FR["decisions"]["d5_rules"]
    for rule in EX:
        M = model(rule); g0 = M.g5({})
        check(f"libra_base_g5_{rule}", rel(g0["g"], d5r[rule]["g5"]["g"]) <= TOL and rel(g0["se"], d5r[rule]["g5"]["se"]) <= TOL,
              [g0["g"], d5r[rule]["g5"]["g"], g0["se"], d5r[rule]["g5"]["se"]])
        pr = d5r[rule]["paired_ratio_libra_over_ML_rl"]
        gr = growth(d5r[rule]["g5"]["g"], d5r[rule]["g5"]["se"], g3L["g"], g3L["se"])
        check(f"libra_headline_Delta_{rule}", rel(gr["Delta"], -pr["ln_ratio"]) <= 1e-12 and rel(gr["se"], pr["se_ln"]) <= 1e-12)
        check(f"libra_g3_is_final_round1s_{rule}", pr["g3_W5"] == g3L["g"])

    # ------------------------------------------------------------------ per-shot outputs: every decoder, every d = 5 leaf
    known_D = {}                                                    # leaf -> list of (type, pos) of the check's D flips
    for i in np.where((strat == 0) & flip)[0]:
        known_D.setdefault(leaves[int(lid[i])], []).append((str(typ[i]), int(pos_s[i])))
    check("known_D_flips_are_Dobs", all(sum(1 for t, _ in known_D.get(rel_, []) if t == tt) == Dobs[rel_][tt]
                                         for rel_ in leaves for tt in ("D10", "D01")))
    data = {}                                                       # decoder -> leaf -> per-leaf summary
    names = ["libra"] + list(DECODERS)
    for rel_ in leaves:
        d = os.path.join(a.res, "decode", rel_, "rl_s5")
        z = np.load(os.path.join(d, D2.DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        idx = z["idx"]
        assert np.array_equal(idx, sh["idx"])
        obs = sh["obs"].astype(np.int64); ml = z["pred"][:, 0].astype(np.int64); red = z["redecoded"].astype(bool)
        sn = [str(x) for x in sh["shipped_names"]]
        P_ = {}
        for nm, (src, key, W) in [("libra", ("shipped", D2.LIBRA, W3))] + list(DECODERS.items()):
            if rnd[rel_] not in W:
                continue
            if src == "shipped":
                rows = [i for i, x in enumerate(sn) if key in x]
                assert len(rows) == 1, (rel_, nm, rows)
                P_[nm] = sh["shipped"][rows[0]].astype(np.int64)
            else:
                dz = os.path.join(a.res, "decode", rel_, src)
                q = np.load(os.path.join(dz, key))
                assert np.array_equal(q["idx"], idx), (rel_, nm, "idx")
                if src == "rl_s5t":
                    st = np.load(os.path.join(dz, "shots.npz"))
                    assert np.array_equal(st["idx"], idx) and np.array_equal(st["obs"].astype(np.int64), obs), (rel_, nm, "shots")
                P_[nm] = q["pred"][:, 0].astype(np.int64)
            assert set(np.unique(P_[nm]).tolist()) <= {0, 1}, (rel_, nm, "pred values")
        eM = ml != obs; disc = P_["libra"] != ml; nb = ~red
        cls = {"D10": disc & nb & ~eM, "D01": disc & nb & eM, "C00": ~disc & nb & ~eM, "C11": ~disc & nb & eM}
        for t in TYPES:
            assert int(cls[t].sum()) == pops[rel_][t], (rel_, t, "population")
        for t, p in known_D.get(rel_, []):
            assert cls[t][p], (rel_, t, p, "known D flip's class")
        for t, p in census_shots.get(rel_, []):
            assert cls[t][p], (rel_, t, p, "census flip's class")
        for nm, pred in P_.items():
            eX = pred != obs
            data.setdefault(nm, {})[rel_] = {
                "base": (int(len(obs)), int((eX & ~eM).sum()), int((~eX & eM).sum()), int((eX & eM).sum())),
                "N": {t: int(cls[t].sum()) for t in TYPES}, "Wr": {t: int((cls[t] & eX).sum()) for t in TYPES},
                "known": [(t, bool(eX[p])) for t, p in known_D.get(rel_, [])],
                "census": [(t, bool(eX[p])) for t, p in census_shots.get(rel_, [])]}
    check("libra_tables_reproduced", all(data["libra"][rel_]["base"] == (tables[rel_]["n"], tables[rel_]["n10"], tables[rel_]["n01"],
                                                                        tables[rel_]["n11"]) for rel_ in leaves))
    check("census_shots_per_leaf", all(len(data["libra"][rel_]["census"]) == sum(v.get(rel_, 0) for v in census.values())
                                        for rel_ in leaves))

    # ------------------------------------------------------------------ the decoder's box: modified tables and g5 on its window
    class DBox:
        def __init__(self, nm, W, rule):
            self.nm, self.W, self.D = nm, W, data[nm]
            excl = set(EX[rule])
            ser = {}
            for x in info:
                if x["leaf"] in excl or x["leaf"] not in self.D:
                    continue
                ser.setdefault((x["patch"], x["basis"]), []).append((x["rounds"], x["leaf"]))
            self.series = {k: sorted(v) for k, v in sorted(ser.items())}

        def tables(self, rates):
            out = {}
            for rel_, L in self.D.items():
                n_, n10, n01, n11 = L["base"]
                for t, xw in L["known"]:                            # the check's discordant flips, shot by shot
                    if ML_UP[t]:
                        if xw: n10 -= 1; n11 += 1
                        else: n01 += 1
                    else:
                        if xw: n11 -= 1; n10 += 1
                        else: n01 -= 1
                for t in CT:
                    c = (rnd[rel_], t)
                    if c in census:                                 # A1-R2's census, shot by shot
                        for tt, xw in L["census"]:
                            if tt != t:
                                continue
                            if ML_UP[t]:
                                if xw: n10 -= 1; n11 += 1
                                else: n01 += 1
                            else:
                                if xw: n11 -= 1; n10 += 1
                                else: n01 -= 1
                        continue
                    f = pops[rel_][t] * rates[c]
                    q = L["Wr"][t] / L["N"][t] if L["N"][t] else 0.0
                    fw, fr = f * q, f * (1 - q)
                    if ML_UP[t]:
                        n10 -= fw; n11 += fw; n01 += fr
                    else:
                        n11 -= fw; n10 += fw; n01 -= fr
                out[rel_] = (n_, n10, n01, n11)
            return out

        def g5(self, T):
            Sx, used = [], []
            for key, lst in self.series.items():
                pts = [(r,) + T[rel_] for r, rel_ in lst if r in self.W]
                if len(pts) < 2:
                    continue
                s = D2.series_fit(pts)
                if s is not None:
                    Sx.append(s); used.append("|".join(key))
            o = D2.g_from_series(Sx)
            o["series"] = used
            return o

    # Libra through the decoder path: every vertex of both boxes, both rules, equals Model's
    amin = {rule: P["primary"][rule]["joint95_min"]["alpha"] for rule in EX}
    amax = {rule: P["primary"][rule]["joint95_max"]["alpha"] for rule in EX}
    worst = 0.0
    for rule in EX:
        M = model(rule); B = DBox("libra", W3, rule)
        for alpha in (amin[rule], amax[rule]):
            for ch in itertools.product((0, 1), repeat=len(free)):
                rng_ = {c: (cp_lo(cells[c]["k"], cells[c]["n"], alpha), cp_up(cells[c]["k"], cells[c]["n"], alpha)) for c in free}
                rates = {c: rng_[c][x] for c, x in zip(free, ch)}
                g1 = M.g5(F_of(rates)); g2 = B.g5(B.tables(rates))
                worst = max(worst, rel(g2["g"], g1["g"]), rel(g2["se"], g1["se"]))
    check("libra_decoder_path_equals_Model", worst == 0.0, worst)

    # each decoder's g5 with no flips, and its Delta, against the stored values
    V = json.load(open(a.v2d3)); EA = json.load(open(a.ea))
    g3 = {"libra": {"g": g3L["g"], "se": g3L["se"], "ci95": g3L["ci95"], "source": "final_round1.json entry7_se "
                    "pooled_gaps_W5:d3|libra.rl vs ML.rl", "upper_used_for_reading": G3_UPPER_A1R2}}
    zq = float(NORM.ppf(0.975))
    for nm in ("harmony", "corrmatch", "corr_mwpm", "mwpm"):
        c12 = V["c12_libra_windows"][nm]
        x = c12["A"]
        check(f"{nm}_g3_same_under_both_rules", x["g3"] == c12["none"]["g3"] and x["g3_ci95"] == c12["none"]["g3_ci95"])
        se3 = (x["g3_ci95"][1] - x["g3_ci95"][0]) / (2 * zq)
        g3[nm] = {"g": x["g3"], "se": se3, "ci95": x["g3_ci95"], "source": "willow_v2d3_derived.json c12_libra_windows (SE from "
                  "its 95% CI, z = 1.959964)", "upper_used_for_reading": x["g3_ci95"][1]}
    ea3 = EA["degeneracy_factor"]["10_30|ruleA"]["g3"]
    check("tesseract_g3_same_under_both_rules", ea3 == EA["degeneracy_factor"]["10_30|no_exclusion"]["g3"])
    g3["tesseract"] = {"g": ea3["g"], "se": ea3["se_inflated"], "ci95": ea3["ci95"], "source": "exact_analysis.json "
                       "degeneracy_factor 10_30 g3 (the exact MPE's decisions, which are Tesseract's at d = 3)",
                       "upper_used_for_reading": ea3["ci95"][1]}

    def base_tables(B):
        return {rel_: L["base"] for rel_, L in B.D.items()}

    ref5 = {}
    for nm in ("harmony", "corrmatch", "corr_mwpm", "mwpm"):
        for rule in EX:
            x = V["c12_libra_windows"][nm][rule]
            ref5[(nm, rule)] = {"g": x["g5"], "se": (x["g5_ci95"][1] - x["g5_ci95"][0]) / (2 * zq), "Delta": x["Delta"],
                                "Delta_se": x["Delta_se"], "series": x["series5"]}
    for rule, key in (("A", "10_30|ruleA"), ("none", "10_30|no_exclusion")):
        x = EA["degeneracy_factor"][key]
        ref5[("tesseract", rule)] = {"g": x["g5"]["g"], "se": x["g5"]["se_inflated"], "Delta": x["growth_ln"],
                                     "Delta_se": x["growth_se"], "series": x["g5"]["series"]}
    for (nm, rule), rf in ref5.items():
        B = DBox(nm, DECODERS[nm][2], rule)
        g0 = B.g5(base_tables(B))
        gr = growth(g0["g"], g0["se"], g3[nm]["g"], g3[nm]["se"])
        ok = (rel(g0["g"], rf["g"]) <= TOL and rel(g0["se"], rf["se"]) <= TOL and len(g0["series"]) == rf["series"]
              and rel(gr["Delta"], rf["Delta"]) <= TOL and rel(gr["se"], rf["Delta_se"]) <= TOL)
        check(f"{nm}_{rule}_no_flip_g5_and_Delta_reproduced", ok,
              {"g": [g0["g"], rf["g"]], "se": [g0["se"], rf["se"]], "series": [len(g0["series"]), rf["series"]],
               "Delta": [gr["Delta"], rf["Delta"]], "Delta_se": [gr["se"], rf["Delta_se"]]})
    print(json.dumps({"checks_passed": len(checks), "all_pass": all(v["pass"] for v in checks.values())}))
    if a.validate_only:
        return

    # ------------------------------------------------------------------ 1. Libra at the stored vertices
    O = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": D2.md5(os.path.abspath(__file__)),
         "imports": {"rev_d2_worstcase.py": D2.md5(os.path.join(HERE, "rev_d2_worstcase.py"))},
         "label": "A1-R2-X: post hoc, descriptive addendum to A1-R2. Stored outputs only; no decoding.",
         "inputs_md5": {"a1r2_results.json": D2.md5(a.a1r2), "final_round1.json": D2.md5(a.final_r1),
                        "willow_v2d3_derived.json": D2.md5(a.v2d3), "exact_analysis.json": D2.md5(a.ea),
                        "shot_lists.npz": meta["shot_lists_npz_md5"]},
         "conventions": {"ci95": "estimate +- 1.96 SE", "Delta": "ln(1 + g5) - ln(1 + g3)",
                         "Delta_se": "sqrt((SE5 / (1 + g5))^2 + (SE3 / (1 + g3))^2)",
                         "alpha": {"min_box": amin, "max_box": amax}},
         "g3": g3, "checks": checks}

    def vertex_row(g, g3x):
        gr = growth(g["g"], g["se"], g3x["g"], g3x["se"])
        return {"g5": g["g"], "se": g["se"], "ci95": [g["g"] - Z * g["se"], g["g"] + Z * g["se"]], "n_series": len(g["series"]),
                "series": g["series"], "Delta": gr["Delta"], "Delta_se": gr["se"], "Delta_ci95": gr["ci95"]}

    L1 = {}
    for rule in EX:
        M = model(rule); R = {}
        for ext in ("joint95_min", "joint95_max"):
            st = P["primary"][rule][ext]
            g = M.g5(F_of(rates_at(st["vertex"], st["alpha"])))
            check(f"libra_{rule}_{ext}_vertex_rebuilt", g["g"] == st["g5"], [g["g"], st["g5"]])
            R[ext] = dict(vertex_row(g, g3["libra"]), alpha=st["alpha"], vertex=st["vertex"])
        g0 = M.g5({})
        R["no_flips"] = vertex_row(g0, g3["libra"])
        L1[rule] = R
    floorL = min(L1["A"]["joint95_min"]["g5"], L1["none"]["joint95_min"]["g5"])
    O["libra"] = {"rules": L1, "floor_smaller_rule": floorL, "reading_as_A1R2": (
        "growth holds per round count" if floorL > G3_UPPER_A1R2 else
        "g5 positive per round count; growth per round count not established" if floorL > 0 else "the condition stays")}

    # ------------------------------------------------------------------ 2. every decoder, its own vertices
    def nontriv(c, side):
        k_, n_ = cells[c]["k"], cells[c]["n"]
        return (side == "hi" and k_ < n_) or (side == "lo" and k_ > 0)

    DEC = {}
    for nm in ["libra"] + list(DECODERS):
        Wd = W3 if nm == "libra" else DECODERS[nm][2]
        R = {}
        for rule in EX:
            B = DBox(nm, Wd, rule)
            res = {}
            for ext, alpha, pick in (("min", amin[rule], min), ("max", amax[rule], max)):
                rng_ = {c: (cp_lo(cells[c]["k"], cells[c]["n"], alpha), cp_up(cells[c]["k"], cells[c]["n"], alpha)) for c in free}
                vals = []
                for ch in itertools.product((0, 1), repeat=len(free)):
                    rates = {c: rng_[c][x] for c, x in zip(free, ch)}
                    vals.append((B.g5(B.tables(rates)), {c: ("hi" if x else "lo") for c, x in zip(free, ch)}))
                best = pick(vals, key=lambda v: v[0]["g"])
                vtx = {f"r{c[0]}|{c[1]}": s for c, s in best[1].items() if c[0] in Wd}
                res[ext] = dict(vertex_row(best[0], g3[nm]), alpha=alpha, vertex_in_window=vtx,
                                nontrivial_bounds_in_window=sum(nontriv(c, s) for c, s in best[1].items() if c[0] in Wd),
                                same_orientation_as_libra=all(best[1][c] == ("hi" if (VMIN if ext == "min" else VMAX)[c] else "lo")
                                                              for c in free if c[0] in Wd),
                                g5_spread_over_box=[min(v[0]["g"] for v in vals), max(v[0]["g"] for v in vals)])
            res["no_flips"] = vertex_row(B.g5(base_tables(B)), g3[nm])
            R[rule] = res
        floor = min(R["A"]["min"]["g5"], R["none"]["min"]["g5"])
        up = g3[nm]["upper_used_for_reading"]
        DEC[nm] = {"window": list(Wd), "rules": R, "g3": g3[nm], "floor_smaller_rule": floor,
                   "floor_rule": "A" if R["A"]["min"]["g5"] <= R["none"]["min"]["g5"] else "none", "g3_upper95": up,
                   "reading": ("growth holds per round count" if floor > up else
                               "g5 positive per round count; growth not established" if floor > 0 else "the condition stays")}
    check("libra_decoder_path_min_max_equal_A1R2", all(DEC["libra"]["rules"][rule]["min"]["g5"] == P["primary"][rule]["joint95_min"]["g5"]
                                                        and DEC["libra"]["rules"][rule]["max"]["g5"] == P["primary"][rule]["joint95_max"]["g5"]
                                                        for rule in EX))
    O["decoders"] = DEC
    O["checks"] = checks
    O["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(O, f, indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    show = {}
    for nm, v in DEC.items():
        show[nm] = {"floor": v["floor_smaller_rule"], "floor_rule": v["floor_rule"], "g3": v["g3"]["g"], "g3_upper95": v["g3_upper95"],
                    "reading": v["reading"]}
        for rule in EX:
            r_ = v["rules"][rule]
            show[nm][rule] = {"min": [r_["min"]["g5"], r_["min"]["ci95"]], "max": r_["max"]["g5"],
                              "Delta_min_max": [r_["min"]["Delta"], r_["max"]["Delta"]],
                              "min_nontrivial": r_["min"]["nontrivial_bounds_in_window"],
                              "min_same_orientation": r_["min"]["same_orientation_as_libra"]}
    show["libra_at_stored_vertices"] = {rule: {ext: [L1[rule][ext]["g5"], L1[rule][ext]["ci95"], L1[rule][ext]["Delta"],
                                                     L1[rule][ext]["Delta_se"], L1[rule][ext]["Delta_ci95"]]
                                               for ext in ("joint95_min", "joint95_max")} for rule in EX}
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main()
