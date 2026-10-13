#!/usr/bin/env python3
"""D2 (review 1.3): a reproducible derivation of the non-band check's "95% worst cases" at d = 5 (decoder, 2026-10-13).
Post hoc revision analysis; no decoding. Reads the check's shot lists and outputs, round 1's ladder outputs and the final
analysis; writes one JSON and one markdown file.

1. Rebuild, with this script's own code, every row of nonband_results.json (rules A and none; base, estimate, the two
   95% worst cases and the two worst cases with the tau3-only term): per leaf and type the flips, the modified 2x2
   tables, the count gap, and g_5 by entry 7's estimator, implemented here from its definition (WLS slopes of
   y_M = ln(1 - 2 E_ML) and z = ln(1 - 2 E_A) - y_M on r, Birge factors, delta method with the slope covariance).
   Every row is checked against the published JSON, and g_5 against the frozen functions (p1_se_covariance.py,
   read-only).
2. State the ingredients: strata, sampling weights, the pooled rates and their one-sided 95% Clopper-Pearson bounds, the
   number of separate bounds each row uses and the Bonferroni level of their conjunction.
3. Count gap against per-cycle gap: the count gap is monotone in every flip, so its extremes over the box of rates that
   the bounds allow are the published rows. g_5 is not monotone in the base direction; its extremes over the same box
   are found by enumerating the box's vertices (pooled rates, as published), and, as a sensitivity, with the rates of
   the concordant types estimated per round count (enumeration, or a local search from the gradient-sign vertex when
   the box has more than 4096 vertices).
4. Sensitivities of the estimate: leaf-weighted (Horvitz-Thompson) expansion of the concordant flips instead of pooled
   rates; the change in g_5 per added ML error at each leaf (finite differences), which is how flips move the slopes.
usage: python3 -I -B rev_d2_worstcase.py --res <results/decoder/willow> --check <supplementary/nonband_check>
           --final-r1 <final_round1/final_round1.json> --published <nonband_check/analysis/nonband_results.json> --out DIR
           [--code <code/decoder>]
"""
import argparse, hashlib, itertools, json, math, os, sys, time

sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import beta as BETA

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)                                       # code/decoder
DN = "ml_tau2_1e-6-1e-8_0.1"
LIBRA = "libra_decoder_with_rl_optimized_prior"
W = (10, 30, 50)
TYPES = ("D10", "D01", "C00", "C11")
ADVERSE = {"D10": True, "D01": False, "C00": True, "C11": False}   # a flip of this type adds an ML error
CASES = ("base", "estimate", "worst_lower_g", "worst_higher_g", "worst_lower_g_with_tau3_only",
         "worst_higher_g_with_tau3_only")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cp_upper(k, n):
    return 1.0 if n == 0 or k >= n else float(BETA.ppf(0.95, k + 1, n - k))


def cp_lower(k, n):
    return 0.0 if n == 0 or k <= 0 else float(BETA.ppf(0.05, k, n - k + 1))


# ------------------------------------------------------------------ entry 7's estimator, from its definition
def wls(r, y, v):
    r, y, v = (np.asarray(x, float) for x in (r, y, v))
    w = 1 / v
    rb = (w * r).sum() / w.sum(); yb = (w * y).sum() / w.sum()
    Srr = (w * (r - rb) ** 2).sum()
    b = (w * (r - rb) * (y - yb)).sum() / Srr
    a = yb - b * rb
    dof = len(r) - 2
    f = math.sqrt(max(1.0, float((w * (y - a - b * r) ** 2).sum()) / dof)) if dof > 0 else 1.0
    return b, w * (r - rb) / Srr, f


def series_fit(pts):
    """pts: list of (r, n, n10, n01, n11) for one series (rounds in the window, after exclusions)."""
    R, z, vz, yM, vM, czy = [], [], [], [], [], []
    for r, n, n10, n01, n11 in pts:
        if not n:
            continue
        EA = (n10 + n11) / n; EM = (n01 + n11) / n; p11 = n11 / n
        if not (0 < EA < 0.5 and 0 < EM < 0.5):
            continue
        gA, gM = -2 / (1 - 2 * EA), -2 / (1 - 2 * EM)
        VA, VM, Cv = EA * (1 - EA) / n, EM * (1 - EM) / n, (p11 - EA * EM) / n
        R.append(r); z.append(math.log(1 - 2 * EA) - math.log(1 - 2 * EM))
        vz.append(max(gA ** 2 * VA + gM ** 2 * VM - 2 * gA * gM * Cv, 1e-18))
        yM.append(math.log(1 - 2 * EM)); vM.append(gM ** 2 * VM); czy.append(gA * gM * Cv - gM ** 2 * VM)
    if len(R) < 2:
        return None
    beta, a, fz = wls(R, z, vz)
    bM, c, fm = wls(R, yM, vM)
    return {"beta": beta, "bM": bM, "var_beta": float(np.sum(a ** 2 * np.array(vz))) * fz ** 2,
            "var_bM": float(np.sum(c ** 2 * np.array(vM))) * fm ** 2,
            "cov": float(np.sum(a * c * np.array(czy))) * fz * fm, "rounds": R}


def g_from_series(S):
    n = len(S)
    eM = np.array([(1 - math.exp(s["bM"])) / 2 for s in S])
    dl = np.array([(math.exp(s["bM"]) - math.exp(s["bM"] + s["beta"])) / 2 for s in S])
    D, E = dl.mean(), eM.mean()
    vD = vE = cDE = 0.0
    for s, d in zip(S, dl):
        db = -math.exp(s["bM"] + s["beta"]) / 2; dm = d; em = -math.exp(s["bM"]) / 2
        vD += (db ** 2 * s["var_beta"] + dm ** 2 * s["var_bM"] + 2 * db * dm * s["cov"]) / n ** 2
        vE += em ** 2 * s["var_bM"] / n ** 2
        cDE += (db * em * s["cov"] + dm * em * s["var_bM"]) / n ** 2
    g = D / E
    se = math.sqrt(vD / E ** 2 + D ** 2 * vE / E ** 4 - 2 * D * cDE / E ** 3)
    return {"g": g, "se": se, "ci95": [g - 1.96 * se, g + 1.96 * se], "n_series": n}


class Model:
    def __init__(self, leaves, info, tables, excl):
        self.leaves, self.excl = leaves, set(excl)
        self.tables = tables                                       # leaf -> dict n, n10, n01, n11
        ser = {}
        for x in info:
            if x["leaf"] in self.excl:
                continue
            ser.setdefault((x["patch"], x["basis"]), []).append((x["rounds"], x["leaf"]))
        self.series = {k: sorted(v) for k, v in sorted(ser.items())}
        self.used = [k for k, v in self.series.items() if len([r for r, _ in v if r in W]) >= 2]

    def modified(self, F):
        out = {}
        for rel, c in self.tables.items():
            f = F.get(rel, {})
            d10, d01, c00, c11 = (f.get(t, 0.0) for t in TYPES)
            out[rel] = (c["n"], c["n10"] - d10 + c11, c["n01"] - d01 + c00, c["n11"] + d10 - c11)
        return out

    def g5(self, F):
        T = self.modified(F)
        S, used = [], []
        for key, lst in self.series.items():
            pts = [(r,) + T[rel] for r, rel in lst if r in W]
            if len(pts) < 2:
                continue
            s = series_fit(pts)
            if s is not None:
                S.append(s); used.append("|".join(key))
        out = g_from_series(S)
        out["series"] = used
        return out

    def count_gap(self, F):
        T = self.modified(F)
        lib = sum(t[1] + t[3] for rel, t in T.items() if rel not in self.excl)
        ml = sum(t[2] + t[3] for rel, t in T.items() if rel not in self.excl)
        return {"errors_libra": lib, "errors_ML": ml, "gap_errors": lib - ml, "rel": lib / ml - 1}


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--final-r1", "--published", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--code", default=CODE, help="code/decoder (default: the parent of this script's folder)")
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    assert md5(lp) == meta["shot_lists_npz_md5"]
    S = {k: v for k, v in np.load(lp).items()}
    leaves = [str(x) for x in S["leaves"]]
    info = meta["leaves"]
    assert [x["leaf"] for x in info] == leaves
    n = len(S["idx"])
    lid, strat, typ = S["leaf_id"], S["stratum"], S["type"].astype(str)
    ml, obs = S["ml"].astype(np.int64), S["obs"].astype(np.int64)
    rounds = S["rounds"]
    # ---------------------------------------------------------------- the check's decisions per selected shot
    d2 = np.full(n, -1); d3 = np.full(n, -1); reason = np.zeros(n, np.int64)
    l2 = np.full(n, np.nan)
    for k in range(len(leaves)):
        z = np.load(os.path.join(a.check, "leaves", f"{k:02d}.npz"))
        r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1); d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
        reason[r] = z["t3_reason"]; l2[r] = z["llr_t2"]
    done2, done3 = d2 >= 0, d3 >= 0
    with np.errstate(over="ignore", invalid="ignore"):
        P1 = 1 / (1 + np.exp(-S["llr_t1"].astype(np.float64))); P2 = 1 / (1 + np.exp(-l2))
        # the check's rule for tau3: tau2 moved the decision, or left it in the band, or moved P by 0.05 or more
        tg = done2 & ((d2 != ml) | ~(np.abs(P2 - 0.5) >= 0.1) | ~(np.abs(P2 - P1) < 0.05))
    final = np.where(done3, d3, d2)
    flip = done2 & (final != ml)
    unrun = done2 & ~tg & ~done3                                # tau2 kept the decision; tau3 not run
    assert not (tg & ~done3).any() and done2.all(), "targeted shots without tau3, or shots without tau2"
    consistency = {"targeted_equals_reason1": bool(np.array_equal(tg & done3, (reason == 1) & done3)),
                   "random_subsample_untargeted": bool(not (tg & (reason == 2)).any())}
    # ---------------------------------------------------------------- per-leaf tables from the ladder outputs
    tables, pop_check = {}, []
    for k, rel in enumerate(leaves):
        d = os.path.join(a.res, "decode", rel, "rl_s5")
        z = np.load(os.path.join(d, DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        row = [i for i, x in enumerate(sh["shipped_names"]) if LIBRA in str(x)]
        assert len(row) == 1
        lib = sh["shipped"][row[0]].astype(np.int64); m = z["pred"][:, 0].astype(np.int64); o = sh["obs"].astype(np.int64)
        ea, eb = lib != o, m != o
        tables[rel] = {"n": int(len(o)), "n10": int((ea & ~eb).sum()), "n01": int((~ea & eb).sum()), "n11": int((ea & eb).sum())}
        nb = ~z["redecoded"].astype(bool); disc = lib != m
        pop = {"D10": int((nb & disc & ~eb).sum()), "D01": int((nb & disc & eb).sum()),
               "C00": int((nb & ~disc & ~eb).sum()), "C11": int((nb & ~disc & eb).sum())}
        pop_check.append(pop == info[k]["population"])
    assert all(pop_check), "non-band populations differ from shot_lists.json"
    pops = {x["leaf"]: x["population"] for x in info}
    rnd = {x["leaf"]: x["rounds"] for x in info}
    # ---------------------------------------------------------------- strata, weights, rates
    NC = {rel: pops[rel]["C00"] + pops[rel]["C11"] for rel in leaves}
    alloc = {rel: int(((lid == k) & (strat == 1)).sum()) for k, rel in enumerate(leaves)}
    nD = {rel: int(((lid == k) & (strat == 0)).sum()) for k, rel in enumerate(leaves)}
    assert all(nD[rel] == pops[rel]["D10"] + pops[rel]["D01"] for rel in leaves), "D is not a census"
    weights = {rel: {"D_census": nD[rel], "C_population": NC[rel], "C_sampled": alloc[rel],
                     "C_weight": NC[rel] / alloc[rel]} for rel in leaves}

    def rates(group_mask):
        """Pooled quantities of the concordant types and of the tau3-only subsample, over the rows in group_mask."""
        out = {}
        for t in ("C00", "C11"):
            m = group_mask & (strat == 1) & (typ == t)
            k, nn = int((m & flip).sum()), int(m.sum())
            out[t] = {"k": k, "n": nn, "rate": k / nn if nn else 0.0, "lower95": cp_lower(k, nn), "upper95": cp_upper(k, nn),
                      "unrun_share": float((m & unrun).sum() / nn) if nn else 0.0}
        for name, s in (("t3_D", 0), ("t3_C", 1)):
            m = group_mask & (strat == s) & (reason == 2) & done3
            k, nn = int((m & (d3 != d2)).sum()), int(m.sum())
            out[name] = {"k": k, "n": nn, "rate": k / nn if nn else 0.0, "upper95": cp_upper(k, nn)}
        return out

    pooled = rates(np.ones(n, bool))
    by_r = {r: rates(rounds == r) for r in W}
    # D: observed flips and unrun counts per leaf and type
    Dobs = {rel: {t: int(((lid == k) & (typ == t) & flip).sum()) for t in ("D10", "D01")} for k, rel in enumerate(leaves)}
    Dunrun = {rel: {t: int(((lid == k) & (typ == t) & unrun).sum()) for t in ("D10", "D01")} for k, rel in enumerate(leaves)}

    def flips(par, group_of=lambda rel: "all"):
        """par[group] = {"C00": rate, "C11": rate, "t3_D10": r, "t3_D01": r, "t3_C00": r, "t3_C11": r};
        returns F[leaf][type]. Concordant flips: N_leaf,t x (rate + unrun_share x tau3-only rate); discordant: observed
        flips + unrun shots x tau3-only rate."""
        F = {}
        for rel in leaves:
            g = group_of(rel); p = par[g]; q = by_r[rnd[rel]] if g != "all" else pooled
            f = {}
            for t in ("D10", "D01"):
                f[t] = Dobs[rel][t] + Dunrun[rel][t] * p.get("t3_" + t, 0.0)
            for t in ("C00", "C11"):
                f[t] = pops[rel][t] * min(1.0, p[t] + q[t]["unrun_share"] * p.get("t3_" + t, 0.0))
            F[rel] = f
        return F

    def corner(direction, with_t3, src):
        """The published scenarios: every bound in one direction. direction 'lower' = ML's errors up."""
        p = {}
        for t in ("C00", "C11"):
            up = ADVERSE[t] == (direction == "lower")
            p[t] = src[t]["upper95"] if up else src[t]["lower95"]
            if with_t3:
                p["t3_" + t] = src["t3_C"]["upper95"] if up else 0.0
        for t in ("D10", "D01"):
            up = ADVERSE[t] == (direction == "lower")
            p["t3_" + t] = src["t3_D"]["upper95"] if (with_t3 and up) else 0.0
        return p

    est = {"C00": pooled["C00"]["rate"], "C11": pooled["C11"]["rate"], "t3_D10": pooled["t3_D"]["rate"],
           "t3_D01": pooled["t3_D"]["rate"], "t3_C00": pooled["t3_C"]["rate"], "t3_C11": pooled["t3_C"]["rate"]}
    scen = {"base": None, "estimate": {"all": est},
            "worst_lower_g": {"all": corner("lower", False, pooled)}, "worst_higher_g": {"all": corner("higher", False, pooled)},
            "worst_lower_g_with_tau3_only": {"all": corner("lower", True, pooled)},
            "worst_higher_g_with_tau3_only": {"all": corner("higher", True, pooled)}}
    F1 = json.load(open(a.final_r1))
    exA = sorted(x for x in (F1.get("excluded_leaves") or []) if x in tables)
    pub = json.load(open(a.published))
    # frozen estimator, read-only, for the cross-check of g_5
    code = a.code
    sys.path.insert(0, code); sys.path.insert(0, os.path.join(code, "supplementary"))
    import p1_se_covariance as PC                                  # frozen; read-only

    def frozen_g5(F, excl):
        T = {}
        for rel, c in tables.items():
            f = F.get(rel, {}) if F else {}
            d10, d01, c00, c11 = (f.get(t, 0.0) for t in TYPES)
            T[rel] = {"n": c["n"], "n10": c["n10"] - d10 + c11, "n11": c["n11"] + d10 - c11, "n01": c["n01"] - d01 + c00}
        ser = {}
        for x in info:
            if x["leaf"] not in excl:
                ser.setdefault((x["patch"], x["basis"]), {})[x["rounds"]] = T[x["leaf"]]
        SS = []
        for key in sorted(ser):
            pts = ser[key]; w = tuple(r for r in W if r in pts)
            if len(w) < 2 or len(PC.points_arrays(pts, w)) < 2:
                continue
            SS.append(PC.series_params(pts, w))
        return PC.g_and_se(SS, "inflated", full=True, with_cov=True)

    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "post hoc revision analysis (D2); no decoding",
           "inputs_md5": {"shot_lists.npz": md5(lp), "shot_lists.json": md5(os.path.join(a.check, "shot_lists.json")),
                          "summary.json": md5(os.path.join(a.check, "summary.json")),
                          "final_round1.json": md5(a.final_r1), "nonband_results.json": md5(a.published),
                          "p1_se_covariance.py": md5(os.path.join(code, "supplementary", "p1_se_covariance.py")),
                          "leaves": {f"{k:02d}.npz": md5(os.path.join(a.check, "leaves", f"{k:02d}.npz")) for k in range(len(leaves))}},
           "consistency": consistency, "rule_A_excluded": exA,
           "strata": {"D": {"design": "census of the non-band discordant shots", "selected": int((strat == 0).sum()),
                            "population": {t: sum(pops[r][t] for r in leaves) for t in ("D10", "D01")}},
                      "C": {"design": "2,500 non-band concordant shots, allocated to the leaves in proportion to their counts "
                                      "(largest remainder), drawn without replacement within each leaf",
                            "selected": int((strat == 1).sum()),
                            "population": {t: sum(pops[r][t] for r in leaves) for t in ("C00", "C11")},
                            "weight_range": [min(w["C_weight"] for w in weights.values()), max(w["C_weight"] for w in weights.values())]}},
           "weights": weights, "pooled": pooled, "by_round": {str(r): v for r, v in by_r.items()},
           "D_observed_flips": Dobs, "D_unrun": Dunrun, "rules": {}}
    worst = 0.0
    for rule, ex in (("A", exA), ("none", [])):
        M = Model(leaves, info, tables, ex)
        rows = {}
        for case in CASES:
            F = flips(scen[case]) if scen[case] else {}
            g = M.g5(F); cg = M.count_gap(F)
            fz = frozen_g5(F, set(ex))
            P = pub["rules"][rule][case]
            diff = {"g": abs(g["g"] - P["g5"]["g"]), "se": abs(g["se"] - P["g5"]["se"]),
                    "count_gap_rel": abs(cg["rel"] - P["count_gap"]["rel"]),
                    "gap_errors": abs(cg["gap_errors"] - P["count_gap"]["gap_errors"]),
                    "flips_total": max(abs(sum(F.get(r, {}).get(t, 0.0) for r in leaves if r not in ex) - P["flips_total"][t])
                                       for t in TYPES),
                    "frozen_g": abs(g["g"] - fz["g"]), "frozen_se": abs(g["se"] - fz["se"])}
            worst = max(worst, max(diff.values()))
            rows[case] = {"g5": g, "count_gap": cg,
                          "flips_total": {t: float(sum(F.get(r, {}).get(t, 0.0) for r in leaves if r not in ex)) for t in TYPES},
                          "abs_diff_vs_published_and_frozen": diff}
        out["rules"][rule] = {"rows": rows}
    out["reproduction_max_abs_diff"] = worst
    # ---------------------------------------------------------------- bounds per row and their joint level
    def nontrivial(lo_or_hi, k, nn):
        return not ((lo_or_hi == "lower" and k == 0) or (lo_or_hi == "upper" and k >= nn))
    bounds = {}
    for case, d, t3 in (("worst_lower_g", "lower", False), ("worst_higher_g", "higher", False),
                        ("worst_lower_g_with_tau3_only", "lower", True), ("worst_higher_g_with_tau3_only", "higher", True)):
        used = []
        for t in ("C00", "C11"):
            side = "upper" if ADVERSE[t] == (d == "lower") else "lower"
            if nontrivial(side, pooled[t]["k"], pooled[t]["n"]):
                used.append(f"{t} rate, {side} ({pooled[t]['k']} of {pooled[t]['n']})")
        if t3:
            if any(ADVERSE[t] == (d == "lower") for t in ("C00", "C11")):
                used.append(f"tau3-only rate, concordant, upper ({pooled['t3_C']['k']} of {pooled['t3_C']['n']})")
            used.append(f"tau3-only rate, discordant, upper ({pooled['t3_D']['k']} of {pooled['t3_D']['n']})")
        m = len(used)
        bounds[case] = {"bounds": used, "number": m, "each_one_sided": 0.95, "bonferroni_joint_at_least": 1 - 0.05 * m}
    out["bounds_per_row"] = bounds
    # ---------------------------------------------------------------- extremes over the box (pooled and per round)
    def box_coords(with_t3):
        c = ["C00", "C11"]
        return c + (["t3_C00", "t3_C11", "t3_D10", "t3_D01"] if with_t3 else [])

    def coord_range(name, src):
        if name in ("C00", "C11"):
            return (src[name]["lower95"], src[name]["upper95"])
        if name.startswith("t3_C"):
            return (0.0, src["t3_C"]["upper95"])
        return (0.0, src["t3_D"]["upper95"])

    def box_extremes(M, groups, with_t3):
        """groups: list of group names ('all' or rounds). Returns min and max of g5 and of the count gap over the box."""
        coords = [(g, c) for g in groups for c in box_coords(with_t3)]
        src = {g: (pooled if g == "all" else by_r[g]) for g in groups}
        ranges = [coord_range(c, src[g]) for g, c in coords]
        gof = (lambda rel: "all") if groups == ["all"] else (lambda rel: rnd[rel])

        def evaluate(choice):
            par = {g: {} for g in groups}
            for (g, c), (lo, hi), ch in zip(coords, ranges, choice):
                par[g][c] = hi if ch else lo
            F = flips(par, gof)
            return M.g5(F)["g"], M.count_gap(F)["rel"], F
        res = {}
        nv = 2 ** len(coords)
        if nv <= 4096:
            vals = [(evaluate(ch)[:2], ch) for ch in itertools.product((0, 1), repeat=len(coords))]
            for key, idx, sign in (("g5_min", 0, 1), ("g5_max", 0, -1), ("count_gap_min", 1, 1), ("count_gap_max", 1, -1)):
                v, ch = min(vals, key=lambda x: sign * x[0][idx])
                res[key] = {"value": v[idx], "vertex": {f"{g}:{c}": ("hi" if x else "lo") for (g, c), x in zip(coords, ch)}}
            res["method"] = f"enumeration of {nv} vertices"
        else:
            base0 = tuple(0 for _ in coords)
            g0 = evaluate(base0)[0]
            grad = []
            for i in range(len(coords)):
                ch = list(base0); ch[i] = 1
                grad.append(evaluate(tuple(ch))[0] - g0)
            for key, sign in (("g5_min", 1), ("g5_max", -1)):
                ch = [1 if sign * gi < 0 else 0 for gi in grad]
                best = sign * evaluate(tuple(ch))[0]; steps = 0
                improved = True
                while improved:
                    improved = False
                    for i in range(len(ch)):
                        ch2 = list(ch); ch2[i] = 1 - ch2[i]
                        v = sign * evaluate(tuple(ch2))[0]
                        if v < best - 1e-15:
                            ch, best, improved = ch2, v, True; steps += 1
                res[key] = {"value": sign * best, "vertex": {f"{g}:{c}": ("hi" if x else "lo") for (g, c), x in zip(coords, ch)},
                            "local_search_steps": steps}
            for key, sign in (("count_gap_min", 1), ("count_gap_max", -1)):
                # the count gap is monotone in each coordinate: ML's errors up lowers it
                ch = tuple(1 if ((c in ("C00", "t3_C00", "t3_D10")) == (sign > 0)) else 0 for g, c in coords)
                res[key] = {"value": evaluate(ch)[1], "vertex": {f"{g}:{c}": ("hi" if x else "lo") for (g, c), x in zip(coords, ch)}}
            res["method"] = f"gradient-sign vertex and single-coordinate local search ({nv} vertices)"
        # monotonicity along each coordinate at the g5 extremes (interior points)
        worst_interior = 0.0
        for key, sign in (("g5_min", 1), ("g5_max", -1)):
            ch = [1 if res[key]["vertex"][f"{g}:{c}"] == "hi" else 0 for g, c in coords]
            for i, ((g, c), (lo, hi)) in enumerate(zip(coords, ranges)):
                for s in (0.25, 0.5, 0.75):
                    par = {gg: {} for gg in groups}
                    for j, ((g2, c2), (lo2, hi2)) in enumerate(zip(coords, ranges)):
                        par[g2][c2] = (lo + s * (hi - lo)) if j == i else (hi2 if ch[j] else lo2)
                    v = M.g5(flips(par, gof))["g"]
                    worst_interior = max(worst_interior, sign * (res[key]["value"] - v))
        res["interior_points_beat_vertex_by"] = worst_interior
        return res

    for rule, ex in (("A", exA), ("none", [])):
        M = Model(leaves, info, tables, ex)
        R = out["rules"][rule]
        R["box_pooled"] = {"without_tau3_only": box_extremes(M, ["all"], False), "with_tau3_only": box_extremes(M, ["all"], True)}
        R["box_by_round"] = {"without_tau3_only": box_extremes(M, list(W), False), "with_tau3_only": box_extremes(M, list(W), True)}
        # estimate with leaf weights (Horvitz-Thompson) for the concordant flips
        Fht = {}
        for k, rel in enumerate(leaves):
            f = {t: float(Dobs[rel][t]) for t in ("D10", "D01")}
            for t in ("C00", "C11"):
                kk = int(((lid == k) & (strat == 1) & (typ == t) & flip).sum())
                f[t] = kk * weights[rel]["C_weight"]
            Fht[rel] = f
        R["estimate_leaf_weighted"] = {"g5": M.g5(Fht), "count_gap": M.count_gap(Fht),
                                       "C_flips_by_leaf": {rel: {t: Fht[rel][t] for t in ("C00", "C11")}
                                                           for rel in leaves if Fht[rel]["C00"] + Fht[rel]["C11"] > 0}}
        # estimate with concordant rates per round count
        R["estimate_by_round"] = {}
        par = {r: {"C00": by_r[r]["C00"]["rate"], "C11": by_r[r]["C11"]["rate"]} for r in W}
        Fr = flips(par, lambda rel: rnd[rel])
        R["estimate_by_round"] = {"g5": M.g5(Fr), "count_gap": M.count_gap(Fr)}
        # the observed discordant flips alone (the same in every row), and their net change in ML's errors by round count
        FD = {rel: {"D10": float(Dobs[rel]["D10"]), "D01": float(Dobs[rel]["D01"])} for rel in leaves}
        R["D_flips_only"] = {"g5": M.g5(FD), "count_gap": M.count_gap(FD),
                             "net_ML_errors_by_round": {str(r): int(sum(Dobs[rel]["D10"] - Dobs[rel]["D01"] for rel in leaves
                                                                        if rnd[rel] == r and rel not in M.excl)) for r in W}}
        # change in g5 per added ML error, by leaf (finite difference of 10 errors added as C00 flips)
        g0 = M.g5({})["g"]
        sens = {}
        for rel in leaves:
            if rel in M.excl:
                continue
            sens[rel] = (M.g5({rel: {"C00": 10.0}})["g"] - g0) / 10
        byr = {str(r): [v for rel, v in sens.items() if rnd[rel] == r] for r in W}
        R["dg5_per_added_ML_error"] = {"by_leaf": sens, "mean_by_round": {r: float(np.mean(v)) for r, v in byr.items() if v},
                                       "range_by_round": {r: [min(v), max(v)] for r, v in byr.items() if v}}
    out["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "d2_worstcase.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    write_md(out, os.path.join(a.out, "d2_worstcase_tables.md"))
    print(json.dumps({"reproduction_max_abs_diff": worst, "seconds": out["seconds"],
                      "A_box": {k: {kk: vv["value"] for kk, vv in v.items() if isinstance(vv, dict) and "value" in vv}
                                for k, v in out["rules"]["A"]["box_pooled"].items()}}, default=float))


def write_md(o, path):
    pc = lambda x: f"{100 * x:+.2f}%"
    s = ("# D2: the non-band check's worst cases, derived (tables)\n\n"
         f"Made {o['made']} by `rev_d2_worstcase.py` (md5 {o['script_md5']}). Post hoc revision analysis; no decoding.\n\n"
         f"Every published row is reproduced: largest absolute difference {o['reproduction_max_abs_diff']:.2e} "
         "(g, SE, count gap, flips; and g against the frozen estimator).\n\n")
    s += "## Rows as published (reproduced)\n\n| rule | case | g_5 (95% CI) | count gap (errors) | flips D10, D01, C00, C11 |\n|---|---|---|---|---|\n"
    for rule in ("A", "none"):
        for case in CASES:
            r = o["rules"][rule]["rows"][case]; g = r["g5"]; cg = r["count_gap"]; f = r["flips_total"]
            s += (f"| {rule} | {case} | {pc(g['g'])} ({pc(g['ci95'][0])}, {pc(g['ci95'][1])}) | {pc(cg['rel'])} "
                  f"({cg['gap_errors']:+.0f}) | {f['D10']:.1f}, {f['D01']:.1f}, {f['C00']:.1f}, {f['C11']:.1f} |\n")
    s += "\n## Bounds behind each worst-case row\n\n| row | separate one-sided 95% bounds used | Bonferroni level of all at once |\n|---|---|---|\n"
    for case, b in o["bounds_per_row"].items():
        s += f"| {case} | {'; '.join(b['bounds'])} | at least {100 * b['bonferroni_joint_at_least']:.0f}% |\n"
    s += "\n## Extremes over the box the bounds allow\n\n| rule | rates | tau3-only term | g_5 min | g_5 max | count gap min | count gap max | method |\n|---|---|---|---|---|---|---|---|\n"
    for rule in ("A", "none"):
        for bx, lab in (("box_pooled", "pooled (as published)"), ("box_by_round", "per round count")):
            for k, lab2 in (("without_tau3_only", "no"), ("with_tau3_only", "yes")):
                b = o["rules"][rule][bx][k]
                s += (f"| {rule} | {lab} | {lab2} | {pc(b['g5_min']['value'])} | {pc(b['g5_max']['value'])} | "
                      f"{pc(b['count_gap_min']['value'])} | {pc(b['count_gap_max']['value'])} | {b['method']} |\n")
    s += "\n## Estimate under other ways of combining the strata\n\n| rule | estimate | g_5 | count gap |\n|---|---|---|---|\n"
    for rule in ("A", "none"):
        R = o["rules"][rule]
        for k, lab in (("rows", "pooled rates (published)"), ("estimate_leaf_weighted", "leaf weights (Horvitz-Thompson)"),
                       ("estimate_by_round", "rates per round count")):
            x = R[k]["estimate"] if k == "rows" else R[k]
            s += f"| {rule} | {lab} | {pc(x['g5']['g'])} | {pc(x['count_gap']['rel'])} |\n"
        x = R["D_flips_only"]
        net = ", ".join(f"r = {r}: {v:+d}" for r, v in x["net_ML_errors_by_round"].items())
        s += f"| {rule} | discordant flips only (net ML errors {net}) | {pc(x['g5']['g'])} | {pc(x['count_gap']['rel'])} |\n"
    s += "\n## How one added ML error moves g_5 (finite difference), by round count\n\n| rule | r | mean change in g_5 per added ML error | range over leaves |\n|---|---|---|---|\n"
    for rule in ("A", "none"):
        R = o["rules"][rule]["dg5_per_added_ML_error"]
        for r, v in R["mean_by_round"].items():
            lo, hi = R["range_by_round"][r]
            s += f"| {rule} | {r} | {100 * v:+.5f} points | {100 * lo:+.5f} to {100 * hi:+.5f} |\n"
    open(path, "w").write(s)


if __name__ == "__main__":
    main()
