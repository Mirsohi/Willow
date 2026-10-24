#!/usr/bin/env python3
"""D1-8 (review 1.3; fresh-eyes item B4 of draft 9): a post hoc, descriptive projection of the added sample that would
make D1's primary reading (rule A, per cell) bound g_5 above 0 (decoder, 2026-10-24). No decoding; no label or test.
Declared in notes/decoder/willow_revision_declarations.md (D1-8, D1-8b).

Machinery: D1-5b's synthetic test (every sampled shot takes the census decision a and is resolved) with D1-2's
posterior. The census inputs, build_F and the posterior loop are copied unchanged from rev_d1_analyze.py (ef15b054...):
a Jeffreys rate per cell and side, the census part fixed, 2,000 draws (seed 20261016 per cell, 20261017 pooled), g_5
plus its shot-noise SE times a normal draw. A scenario changes only each cell side's sample count n (disagreements 0);
the real-rows sensitivity adds shots to D1's real counts instead.
Validation first: from the stored counts, the posterior of d1_results.json (both rules, both levels) and of D1-5b's
synthetic 2,500-row case are reproduced exactly, or the script stops.
Cost: D1's measured CPU per row by stratum and round count (MPS: build + rungs; sparse: build + runs), scaled so that
the rows sum to the counters (25.538 and 2.48 core-hours).
usage: python3 -I -B rev_d1_project.py --res <results/decoder/willow> --sel <d1_mps/selection> --final-r1 <json>
           --check <supplementary/nonband_check> --d1 <d1_mps/analysis/d1_results.json>
           --synthetic <D1-5b's synthetic d1_results.json> --runs <d1_mps/runs> --out DIR [--validate-only]
"""
import argparse, glob, hashlib, importlib.util, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

DN = "ml_tau2_1e-6-1e-8_0.1"
LIBRA = "libra_decoder_with_rl_optimized_prior"
W = (10, 30, 50)
STRATA = ("B", "D", "C_low", "C_high")
TYPES = ("D10", "D01", "C00", "C11")
RIGHT_T = ("D10", "C00")
LN9 = math.log(9.0)
SIDES = ("right", "wrong")
MPS_COUNTERS_H = 25.538          # D1-7: worker 0 12.337 + worker 1 13.201, by the counters over all runs
SPARSE_COUNTERS_H = 2.48         # D1-7: the sparse runs, by the counters
FD_RATE = 1e-3                   # D1-8: the finite-difference rate that marks a side as adverse
SCAN_C = (250, 500, 750, 1000, 1500, 2000, 3000, 4000)
SCAN_REAL = SCAN_C + (6000, 8000)
SCAN_B = (500, 1000, 2000, 4000)
STEP = 50


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sig(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def pred_of_llr(llr):
    """decode_willow._ml_out's rule: pred = nan_to_num(P1, 0) > 0.5, with P1 = sigmoid(llr)."""
    return (np.nan_to_num(sig(llr), nan=0.0) > 0.5).astype(np.int64)


def tcode(lib_err, ref_err):
    return np.where(lib_err & ~ref_err, 0, np.where(~lib_err & ref_err, 1, np.where(~lib_err & ~ref_err, 2, 3)))


cell_key = lambda s_, r: f"{s_}|{r}"


def load_census(a_):
    """Copied from rev_d1_analyze.py (ef15b054...), the census part: per leaf the tables T, the populations per
    (stratum, type relative to a) popc, and the census flips Fc (a != m, by type relative to m)."""
    meta = json.load(open(os.path.join(a_.sel, "sample.json")))
    zp = os.path.join(a_.sel, "sample.npz")
    assert md5(zp) == meta["sample_npz_md5"], "sample.npz changed"
    leaves = [x["leaf"] for x in meta["leaves"]]
    info = [{"leaf": x["leaf"], "rounds": x["rounds"], "patch": x["leaf"].split("/")[1].replace("d5_at_", ""),
             "basis": x["leaf"].split("/")[2]} for x in meta["leaves"]]
    inputs = {"sample.npz": meta["sample_npz_md5"], "sample.json": md5(os.path.join(a_.sel, "sample.json")),
              "final_round1.json": md5(a_.final_r1), "rev_d2_worstcase.py": md5(os.path.join(HERE, "rev_d2_worstcase.py"))}
    # ------------------------------------------------------------ the non-band check: final decision of every D shot
    cm = json.load(open(os.path.join(a_.check, "shot_lists.json")))
    cp_ = os.path.join(a_.check, "shot_lists.npz")
    assert md5(cp_) == cm["shot_lists_npz_md5"], "shot_lists.npz changed"
    inputs["shot_lists.npz"] = cm["shot_lists_npz_md5"]; inputs["shot_lists.json"] = md5(os.path.join(a_.check, "shot_lists.json"))
    S = np.load(cp_)
    cleaves = [str(x) for x in S["leaves"]]
    nS = len(S["idx"])
    d2 = np.full(nS, -1); d3 = np.full(nS, -1); l2 = np.full(nS, np.nan); l3 = np.full(nS, np.nan)
    inputs["nonband_leaves"] = {}
    for k in range(len(cleaves)):
        p_ = os.path.join(a_.check, "leaves", f"{k:02d}.npz")
        inputs["nonband_leaves"][f"{k:02d}.npz"] = md5(p_)
        z = np.load(p_); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1)
        d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
        l2[r] = z["llr_t2"]; l3[r] = np.where(z["done_t3"], z["llr_t3"], np.nan)
    nbD = {}
    for i in np.where(S["stratum"] == 0)[0]:
        assert d2[i] >= 0, "a D shot without tau2 in the non-band check"
        fin = int(d3[i]) if d3[i] >= 0 else int(d2[i]); lf = float(l3[i]) if d3[i] >= 0 else float(l2[i])
        nbD[(cleaves[int(S["leaf_id"][i])], int(S["idx"][i]))] = (fin, lf, float(l2[i]), bool(d3[i] >= 0))
    # ------------------------------------------------------------ the paper's data, per leaf
    T, P, popc, Fc = {}, {}, {}, {}
    inputs["k2_extra_rung"] = {}
    for k, rel in enumerate(leaves):
        d = os.path.join(a_.res, "decode", rel, "rl_s5")
        for f in (DN + ".npz", "shots.npz", "summary.json"):
            assert md5(os.path.join(d, f)) == meta["leaves"][k]["files_md5"][f], f"{rel}/{f} changed since the selection"
        z = np.load(os.path.join(d, DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        row = [i for i, x in enumerate(sh["shipped_names"]) if LIBRA in str(x)]
        assert len(row) == 1
        lib = sh["shipped"][row[0]].astype(np.int64); m = z["pred"][:, 0].astype(np.int64); o = sh["obs"].astype(np.int64)
        idx = z["idx"].astype(np.int64)
        band = z["redecoded"].astype(bool); l1 = np.abs(z["llr_t1"].astype(np.float64)); disc = lib != m
        code = np.where(band, 0, np.where(disc, 1, np.where(l1 < LN9, 2, 3)))
        assert [int((code == si).sum()) for si in range(4)] == [meta["leaves"][k]["strata"][s] for s in STRATA]
        ep = os.path.join(a_.res, "supplementary", "k2_extra_rung", rel, "rl_s5", "extra_rung.npz")
        inputs["k2_extra_rung"][rel] = md5(ep)
        e = np.load(ep)
        posmap = {int(x): i for i, x in enumerate(idx)}
        t3 = np.full(len(m), np.nan)
        for x, v in zip(e["idx"], e["llr_t3"]):
            t3[posmap[int(x)]] = float(v)
        assert sorted(int(x) for x in e["idx"]) == sorted(int(x) for x in idx[band]), f"{rel}: K2 does not cover the band"
        aux = m.copy(); la = z["llr"].astype(np.float64).copy(); nb_t2 = np.full(len(m), np.nan)
        aux[band] = pred_of_llr(t3[band]); la[band] = t3[band]
        for p in np.where(code == 1)[0]:
            fin, lf, l2v, _ = nbD[(rel, int(idx[p]))]
            aux[p] = fin; la[p] = lf; nb_t2[p] = l2v
        ea, em, ex = lib != o, m != o, aux != o
        typ_m, typ_a = tcode(ea, em), tcode(ea, ex)
        T[rel] = {"n": int(len(o)), "n10": int((ea & ~em).sum()), "n01": int((~ea & em).sum()), "n11": int((ea & em).sum())}
        popc[k] = {"diff": {(s, t): int(((code == si) & (typ_a == ti)).sum()) for si, s in enumerate(STRATA) for ti, t in enumerate(TYPES)},
                   "plain": {(s, t): int(((code == si) & (typ_m == ti)).sum()) for si, s in enumerate(STRATA) for ti, t in enumerate(TYPES)}}
        Fc[k] = {t: int(((aux != m) & (typ_m == ti)).sum()) for ti, t in enumerate(TYPES)}
        P[k] = {"code": code}
    nD_pop = sum(int((P[k]["code"] == 1).sum()) for k in P)
    assert nD_pop == int((S["stratum"] == 0).sum()), "the non-band check's D census differs from the D stratum"
    F1 = json.load(open(a_.final_r1))
    exA = sorted(x for x in (F1.get("excluded_leaves") or []) if x in T)
    return meta, leaves, info, T, popc, Fc, exA, inputs


class Machinery:
    """pop_side, build_F and the posterior loop, copied from rev_d1_analyze.py (ef15b054...) for mode "diff"."""

    def __init__(self, rd2, leaves, info, T, popc, Fc, ex):
        self.leaves, self.info, self.popc, self.Fc = leaves, info, popc, Fc
        self.domain = set(leaves) - set(ex)
        self.M = rd2.Model(leaves, info, T, ex)
        self.Npc = self.pop_side(self.domain, "diff", cell_key)

    def pop_side(self, domain, mode, keyf):
        Np = {}
        for k, rel in enumerate(self.leaves):
            if rel not in domain:
                continue
            r = self.info[k]["rounds"]
            for s_ in STRATA:
                c = Np.setdefault(keyf(s_, r), {"right": 0, "wrong": 0})
                for t in TYPES:
                    c["right" if t in RIGHT_T else "wrong"] += self.popc[k][mode][(s_, t)]
        return Np

    def build_F(self, domain, mode, rate_fn):
        F = {}
        for k, rel in enumerate(self.leaves):
            if rel not in domain:
                continue
            r = self.info[k]["rounds"]
            f = {t: (float(self.Fc[k][t]) if mode == "diff" else 0.0) for t in TYPES}
            for s_ in STRATA:
                for t in TYPES:
                    N = self.popc[k][mode][(s_, t)]
                    if N:
                        f[t] += N * rate_fn(s_, r, "right" if t in RIGHT_T else "wrong")
            F[rel] = f
        return F

    def g5_at_rates(self, rates):
        """g_5 (and count gap) with fixed rates per (cell key, side); missing keys are 0."""
        F = self.build_F(self.domain, "diff", lambda s_, r, side: rates.get((cell_key(s_, r), side), 0.0))
        return self.M.g5(F), self.M.count_gap(F)

    def posterior(self, Gc, Gp, level, seed, draws=2000):
        domain, mode, M, Npc, build_F = self.domain, "diff", self.M, self.Npc, self.build_F
        qq = lambda v: ([float(x) for x in np.nanquantile(v, [0.025, 0.5, 0.975])]
                        if np.isfinite(np.asarray(v, float)).any() else [None, None, None])
        rng = np.random.default_rng(seed)
        dg, dgn, dcg, dcr = [], [], [], []
        for _ in range(draws):
            thp = {}
            for s_ in STRATA:
                for sd in SIDES:
                    c = Gp.get(s_, {}).get(sd, {"n": 0, "dis": 0})
                    thp[(s_, sd)] = (c["dis"], c["n"], rng.beta(c["dis"] + 0.5, c["n"] - c["dis"] + 0.5))
            th = {}
            for key in sorted(Npc):
                s_ = key.split("|")[0]
                for sd in SIDES:
                    c = Gc.get(key, {}).get(sd, {"n": 0, "dis": 0})
                    if level == "pooled" or c["n"] == 0:
                        th[(key, sd)] = thp[(s_, sd)][2]
                    else:
                        th[(key, sd)] = rng.beta(c["dis"] + 0.5, c["n"] - c["dis"] + 0.5)
            F = build_F(domain, mode, lambda s_, r, side, th=th: th[(cell_key(s_, r), side)])
            g = M.g5(F); cg = M.count_gap(F)
            dg.append(g["g"]); dgn.append(g["g"] + rng.normal() * g["se"]); dcg.append(cg["gap_errors"]); dcr.append(cg["rel"])
        return {"draws": draws, "seed": seed,
                "prior": "Jeffreys Beta(1/2, 1/2) per " + ("cell" if level == "cells" else "stratum") + " and side",
                "g5_correction_only_q025_q50_q975": qq(dg),
                "g5_with_shot_noise_q025_q50_q975": qq(dgn),
                "count_gap_errors_q025_q50_q975": qq(dcg), "count_gap_rel_q025_q50_q975": qq(dcr),
                "P_count_gap_le_0": float(np.mean(np.array(dcg) <= 0)),
                "P_g5_le_0_with_shot_noise": float(np.nanmean(np.array(dgn) <= 0)),
                "draws_with_nan_g5": int(np.sum(~np.isfinite(np.array(dg))))}


def stored_counts(R):
    """Cell and pooled counts as the frozen analysis stored them (rules.<rule>)."""
    Gc = {k: {sd: {"n": int(v[sd]["n"]), "dis": int(v[sd]["dis"])} for sd in SIDES} for k, v in R["estimate_diff"]["cells"].items()}
    Gp = {k: ({sd: {"n": int(v[sd]["n"]), "dis": int(v[sd]["dis"])} for sd in SIDES} if v else {})
          for k, v in R["groups_diff_pooled"].items()}
    return Gc, Gp


def pooled_of(Gc):
    Gp = {}
    for key, c in Gc.items():
        s_ = key.split("|")[0]
        g = Gp.setdefault(s_, {sd: {"n": 0, "dis": 0} for sd in SIDES})
        for sd in SIDES:
            g[sd]["n"] += c[sd]["n"]; g[sd]["dis"] += c[sd]["dis"]
    return Gp


def with_added(Gc, add, zero_dis):
    out = {}
    for key, c in Gc.items():
        out[key] = {sd: {"n": c[sd]["n"] + int(add.get((key, sd), 0)), "dis": 0 if zero_dis else c[sd]["dis"]} for sd in SIDES}
    return out


def largest_remainder(want, cap):
    """Integer allocation of sum(want) (rounded) with floor + largest remainders, never above cap."""
    total = int(round(sum(want.values())))
    base = {k: min(cap[k], int(math.floor(v))) for k, v in want.items()}
    rest = total - sum(base.values())
    order = sorted(want, key=lambda k: -(want[k] - math.floor(want[k])))
    while rest > 0:
        moved = False
        for k in order:
            if rest > 0 and base[k] < cap[k]:
                base[k] += 1; rest -= 1; moved = True
        if not moved:
            break
    return {k: v for k, v in base.items() if v}


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--sel", "--final-r1", "--check", "--d1", "--synthetic", "--runs", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--validate-only", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("rd2", os.path.join(HERE, "rev_d2_worstcase.py"))
    rd2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd2)
    meta, leaves, info, T, popc, Fc, exA, inputs = load_census(a_)
    D1 = json.load(open(a_.d1)); SY = json.load(open(a_.synthetic))
    inputs.update({"d1_results.json": md5(a_.d1), "synthetic_d1_results.json": md5(a_.synthetic)})
    assert D1["rule_A_excluded"] == exA and SY["rule_A_excluded"] == exA
    FROZEN = "ef15b0548df048d163005c8b08c995a1"
    assert D1["script_md5"] == FROZEN and SY["script_md5"] == FROZEN, "an output not made by the frozen analysis"
    for k_, v_ in D1["inputs_md5"].items():
        if k_ in inputs and k_ != "runs":
            assert inputs[k_] == v_, f"input {k_} differs from the one D1's analysis read"
    MA = {rule: Machinery(rd2, leaves, info, T, popc, Fc, ex) for rule, ex in (("A", exA), ("none", []))}
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "post hoc, descriptive projection (D1-8); D1-5b's synthetic machinery; no decoding, no label or test",
           "inputs_md5": inputs, "rule_A_excluded": exA}
    # ------------------------------------------------------------ validation: reproduce stored posteriors exactly
    val = {}
    for name, src in (("d1_results", D1), ("synthetic_2500_q_eq_a", SY)):
        worst, checked = 0.0, 0
        for rule in ("A", "none"):
            R = src["rules"][rule]; mc = MA[rule]
            for key, v in R["estimate_diff"]["cells"].items():
                for sd in SIDES:
                    assert v[sd]["N"] == mc.Npc[key][sd], f"{name} {rule} {key} {sd}: population differs"
            Gc, Gp = stored_counts(R)
            assert pooled_of(Gc) == {k: v for k, v in Gp.items() if v}, f"{name} {rule}: pooled counts are not the cells' sums"
            for level, seed in (("cells", 20261016), ("pooled", 20261017)):
                got = mc.posterior(Gc, Gp, level, seed)
                ref = R[f"posterior_diff_{level}"]
                assert set(got) == set(ref), f"{name} {rule} {level}: keys differ"
                for k, x in got.items():
                    y = ref[k]
                    xs, ys = (x, y) if isinstance(x, list) else ([x], [y])
                    for u, w in zip(xs, ys):
                        checked += 1
                        if isinstance(u, str) or isinstance(w, str):
                            assert u == w, f"{name} {rule} {level} {k}"
                            continue
                        d = abs(float(u) - float(w))
                        worst = max(worst, d)
        val[name] = {"fields_checked": checked, "max_abs_diff": worst}
        if worst != 0.0:
            print(json.dumps(val)); sys.exit(f"validation failed on {name}: max abs diff {worst}")
    syn_cells = {rule: stored_counts(SY["rules"][rule])[0] for rule in ("A", "none")}
    real_cells = {rule: stored_counts(D1["rules"][rule])[0] for rule in ("A", "none")}
    for rule in ("A", "none"):
        assert all(syn_cells[rule][k][sd]["n"] == real_cells[rule][k][sd]["n"] and syn_cells[rule][k][sd]["dis"] == 0
                   for k in real_cells[rule] for sd in SIDES), "the synthetic case's counts differ from D1's sample"
    out["validation"] = val
    out["current_counts"] = {rule: {k: {sd: dict(real_cells[rule][k][sd], N=MA[rule].Npc[k][sd]) for sd in SIDES}
                                    for k in sorted(real_cells[rule])} for rule in ("A", "none")}
    if a_.validate_only:
        out["seconds"] = time.time() - t0
        print(json.dumps({"validation": val, "seconds": out["seconds"]}))
        return
    # ------------------------------------------------------------ cost per shot, by stratum and round count
    Z = np.load(os.path.join(a_.sel, "sample.npz"))
    strat_of_row = {int(i): STRATA[int(s)] for i, s in enumerate(Z["stratum"])}
    rnd_of_row = {int(i): int(r) for i, r in enumerate(Z["rounds"])}

    def rows(pattern):
        o = {}
        for p in sorted(glob.glob(os.path.join(a_.runs, pattern))):
            inputs.setdefault("runs", {})[os.path.basename(p)] = md5(p)
            for ln in open(p):
                try:
                    x = json.loads(ln)
                except ValueError:
                    continue
                o[int(x["row"])] = x
        return o
    mps, spr = rows("mps_w*.jsonl"), rows("sparse_w*.jsonl")
    assert len(mps) == len(strat_of_row) == len(spr), "run files do not cover the selection"
    cpu_m = {i: float(x["build_cpu_s"]) + float(sum(x["cpu_s"])) for i, x in mps.items()}
    cpu_s = {i: float(x["build_cpu_s"]) + float(sum(v["cpu_s"] for v in x["runs"].values())) for i, x in spr.items()}
    fm = MPS_COUNTERS_H * 3600 / sum(cpu_m.values()); fs = SPARSE_COUNTERS_H * 3600 / sum(cpu_s.values())
    cost = {"mps_scale_to_counters": fm, "sparse_scale_to_counters": fs, "cells": {}}
    rc = out["current_counts"]["none"]
    for s_ in STRATA:
        for r in W:
            ids = [i for i in mps if strat_of_row[i] == s_ and rnd_of_row[i] == r]
            cost["cells"][cell_key(s_, r)] = {"rows": len(ids), "mps_core_s_per_shot": fm * float(np.mean([cpu_m[i] for i in ids])),
                                              "sparse_core_s_per_shot": fs * float(np.mean([cpu_s[i] for i in ids]))}
    # by side as well (a check only): side = a right or wrong for each row, from d1_results.json's shot records
    side_of = {int(s["row"]): ("right" if s["a"] == s["o"] else "wrong") for s in D1["shots"]}
    for key, c in cost["cells"].items():
        s_, r = key.split("|"); r = int(r)
        for sd in SIDES:
            ids = [i for i in mps if strat_of_row[i] == s_ and rnd_of_row[i] == r and side_of[i] == sd]
            c[f"mps_core_s_per_shot_{sd}_side"] = fm * float(np.mean([cpu_m[i] for i in ids])) if ids else None
            c[f"rows_{sd}_side"] = len(ids)
    out["cost_per_shot"] = cost

    def cost_of(add):
        m = sum(k * cost["cells"][key]["mps_core_s_per_shot"] for (key, sd), k in add.items()) / 3600
        s = sum(k * cost["cells"][key]["sparse_core_s_per_shot"] for (key, sd), k in add.items()) / 3600
        return {"mps": m, "sparse": s}

    # ------------------------------------------------------------ evaluation of one scenario
    def evaluate(name, family, add, base="synthetic", scale=None, seeds=((20261016, 20261017),), levels=("cells", "pooled")):
        res = {"name": name, "family": family, "base": base}
        if scale is None:
            res["added"] = {f"{k}:{sd}": v for (k, sd), v in sorted(add.items())}
            res["added_total"] = int(sum(add.values()))
            res["cost_core_hours"] = cost_of(add)
        res["rules"] = {}
        for rule in ("A", "none"):
            mc = MA[rule]
            src = syn_cells[rule] if base == "synthetic" else real_cells[rule]
            if scale is not None:
                Gc = {k: {sd: {"n": min(mc.Npc[k][sd], int(round(scale * src[k][sd]["n"]))), "dis": 0} for sd in SIDES} for k in src}
            else:
                Gc = with_added(src, add, zero_dis=(base == "synthetic"))
            for k in Gc:
                for sd in SIDES:
                    assert Gc[k][sd]["n"] <= mc.Npc[k][sd], f"{name}: {rule} {k} {sd} above its population"
            Gp = pooled_of(Gc)
            rr = {"counts": {f"{k}:{sd}": Gc[k][sd]["n"] for k in sorted(Gc) for sd in SIDES}}
            for level in levels:
                for sc, sp in seeds:
                    rr[level if (sc, sp) == (20261016, 20261017) else f"{level}_seed{sc if level == 'cells' else sp}"] = \
                        mc.posterior(Gc, Gp, level, sc if level == "cells" else sp)
            res["rules"][rule] = rr
        if scale is not None:
            add_n = {}
            for k in src:
                for sd in SIDES:
                    d = res["rules"]["none"]["counts"][f"{k}:{sd}"] - real_cells["none"][k][sd]["n"]
                    if d:
                        add_n[(k, sd)] = d
            res["added_total"] = int(sum(add_n.values())); res["cost_core_hours"] = cost_of(add_n)
            res["added"] = {f"{k}:{sd}": v for (k, sd), v in sorted(add_n.items())}
        res["lower_A_cells"] = res["rules"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"][0]
        res["lower_none_cells"] = res["rules"]["none"]["cells"]["g5_with_shot_noise_q025_q50_q975"][0]
        print(json.dumps({"name": name, "added_total": res["added_total"], "lower_A_cells": res["lower_A_cells"],
                          "lower_none_cells": res["lower_none_cells"], "t": round(time.time() - t0, 1)}), flush=True)
        return res

    scen = []
    scen.append(evaluate("current (D1-5b's synthetic case)", "current", {}))
    for f in (2, 4):
        scen.append(evaluate(f"(a) current allocation x{f}", "a", None, scale=float(f)))
    # (b) the ML-wrong sides of C_high and C_low, in proportion to their populations (rule A's leaves)
    mA = MA["A"]
    bsides = [(cell_key(s_, r), "wrong") for s_ in ("C_high", "C_low") for r in W]
    capA = {ks: mA.Npc[ks[0]][ks[1]] - real_cells["A"][ks[0]][ks[1]]["n"] for ks in bsides}
    popA = {ks: mA.Npc[ks[0]][ks[1]] for ks in bsides}
    for tot in SCAN_B:
        want = {ks: tot * popA[ks] / sum(popA.values()) for ks in bsides}
        scen.append(evaluate(f"(b) ML-wrong sides of C_high and C_low, +{tot}", "b", largest_remainder(want, capA)))
    # (c) adverse sides, n + 1 proportional to (w^2 / c)^(1/3), capped at the population
    g0 = mA.g5_at_rates({})[0]["g"]
    sens = {}
    for key in sorted(mA.Npc):
        for sd in SIDES:
            if mA.Npc[key][sd] == 0:
                continue
            g1 = mA.g5_at_rates({(key, sd): FD_RATE})[0]["g"]
            w = (g1 - g0) / FD_RATE
            sens[(key, sd)] = {"dg5_per_unit_rate": w, "adverse": bool(g1 - g0 < 0), "N": mA.Npc[key][sd],
                               "n_current": real_cells["A"][key][sd]["n"],
                               "u": (w ** 2 / cost["cells"][key]["mps_core_s_per_shot"]) ** (1 / 3)}
    out["sensitivity"] = {f"{k}:{sd}": v for (k, sd), v in sens.items()}
    out["g5_census_rule_A"] = g0
    adv = [ks for ks, v in sens.items() if v["adverse"]]

    def alloc_c(total):
        cap = {ks: sens[ks]["N"] - sens[ks]["n_current"] for ks in adv}

        def added(K):
            return {ks: min(cap[ks], max(0.0, K * sens[ks]["u"] - 1 - sens[ks]["n_current"])) for ks in adv}
        lo, hi = 0.0, 1.0
        while sum(added(hi).values()) < total and hi < 1e12:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            if sum(added(mid).values()) < total:
                lo = mid
            else:
                hi = mid
        return largest_remainder(added(hi), cap)

    def scan(family, sizes, base):
        found = None; prev = None; rs = []
        for tot in sizes:
            r_ = evaluate(f"({family}) adverse sides, +{tot}" + (" on D1's real rows" if base == "real" else ""), family,
                          alloc_c(tot), base=base)
            rs.append(r_)
            if r_["lower_A_cells"] is not None and r_["lower_A_cells"] > 0:
                found = tot; break
            prev = tot
        if found is None:
            return rs, None
        lo, hi = (prev or 0), found
        while hi - lo > STEP:
            mid = int(round((lo + hi) / 2 / STEP)) * STEP
            if mid in (lo, hi):
                break
            r_ = evaluate(f"({family}) adverse sides, +{mid}" + (" on D1's real rows" if base == "real" else ""), family,
                          alloc_c(mid), base=base)
            rs.append(r_)
            if r_["lower_A_cells"] is not None and r_["lower_A_cells"] > 0:
                hi = mid
            else:
                lo = mid
        return rs, hi

    rs, best = scan("c", SCAN_C, "synthetic"); scen += rs
    out["smallest_decisive_c"] = best
    if best is not None:
        seeds = tuple((20261016 + 100 * j, 20261017 + 100 * j) for j in range(1, 11))
        r_ = evaluate(f"(c) adverse sides, +{best}, 10 other seeds", "c_seeds", alloc_c(best),
                      seeds=((20261016, 20261017),) + seeds, levels=("cells",))
        lows = [r_["rules"]["A"][k]["g5_with_shot_noise_q025_q50_q975"][0] for k in r_["rules"]["A"] if k.startswith("cells_seed")]
        out["seeds_at_smallest_decisive_c"] = {"seeds": [s for s, _ in seeds], "lower_A_cells": lows,
                                               "above_0": sum(x > 0 for x in lows),
                                               "lower_none_cells": [r_["rules"]["none"][k]["g5_with_shot_noise_q025_q50_q975"][0]
                                                                    for k in r_["rules"]["none"] if k.startswith("cells_seed")]}
        scen.append(r_)
    rs, best_real = scan("real", SCAN_REAL, "real"); scen += rs
    out["smallest_decisive_on_real_rows"] = best_real
    out["scenarios"] = scen
    out["seconds"] = time.time() - t0
    os.makedirs(a_.out, exist_ok=True)
    json.dump(out, open(os.path.join(a_.out, "d1_projection.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(a_.out, "d1_projection.md"))
    print(json.dumps({"validation": val, "smallest_decisive_c": best, "smallest_decisive_on_real_rows": best_real,
                      "seconds": out["seconds"]}))


def write_md(o, path):
    pc = lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"
    L = ["# D1-8: projection of the added sample that would make D1's primary reading decisive", "",
         f"Made {o['made']} by `rev_d1_project.py` (md5 {o['script_md5']}). Post hoc, descriptive projection with D1-5b's "
         "synthetic machinery (every sampled shot takes the census decision a); no decoding, no label or test.", "",
         f"Validation: {json.dumps(o['validation'])}", "",
         "| scenario | added shots | MPS core-hours | sparse core-hours | rule A per cell: g_5 2.5/50/97.5% (shot noise) | "
         "rule none per cell | rule A pooled |", "|---|---|---|---|---|---|---|"]
    for s in o["scenarios"]:
        A, N = s["rules"]["A"], s["rules"]["none"]
        if "cells" not in A:
            continue
        pool = ", ".join(pc(x) for x in A["pooled"]["g5_with_shot_noise_q025_q50_q975"]) if "pooled" in A else "n/a"
        L.append(f"| {s['name']} | {s['added_total']:,} | {s['cost_core_hours']['mps']:.2f} | {s['cost_core_hours']['sparse']:.2f} | "
                 f"{', '.join(pc(x) for x in A['cells']['g5_with_shot_noise_q025_q50_q975'])} | "
                 f"{', '.join(pc(x) for x in N['cells']['g5_with_shot_noise_q025_q50_q975'])} | {pool} |")
    L += ["", f"Smallest decisive addition under (c): {o.get('smallest_decisive_c')}; on D1's real rows: "
          f"{o.get('smallest_decisive_on_real_rows')}.", ""]
    if "seeds_at_smallest_decisive_c" in o:
        L += [f"At that size, 10 other seeds: {json.dumps(o['seeds_at_smallest_decisive_c'])}", ""]
    L += ["## Sensitivity of g_5 to each cell side's rate (rule A, census, finite difference at 1e-3)", "",
          "| cell:side | N | n now | dg_5 per unit rate | adverse |", "|---|---|---|---|---|"]
    for k, v in o["sensitivity"].items():
        L.append(f"| {k} | {v['N']:,} | {v['n_current']} | {v['dg5_per_unit_rate']:+.4f} | {v['adverse']} |")
    L += ["", "## Cost per shot (core-seconds, scaled to the counters)", "", "```", json.dumps(o["cost_per_shot"], indent=1), "```", ""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
