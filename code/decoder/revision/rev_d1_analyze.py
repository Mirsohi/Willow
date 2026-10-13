#!/usr/bin/env python3
"""D1 (review 1.3), step 3: analysis of the independent check of the d = 5 ML(RL) reference (decoder, 2026-10-13).
Post hoc revision analysis. The plan is declared in notes/decoder/willow_revision_declarations.md (D1-1); this script's
md5 is declared there (D1-2) before any output of the D1 runs was read.

Inputs: the selection (rev_d1_select.py sample: sample.json, sample.npz), the decodes (rev_d1_run.py: <runs>/mps_w*.jsonl,
<runs>/sparse_w*.jsonl), round 1's ladder outputs and shots.npz (the paper's ML(RL) decision m, Libra's l, the outcome o),
K2's extra rung (tau3 on every band shot), the non-band check (tau2 on every non-band discordant shot, tau3 on some),
final_round1.json (rule A's exclusions), and rev_d2_worstcase.py (entry 7's estimator; imported from this folder).

Definitions.
  q        the MPS decision at the last rung run; "resolved" if the ladder's first stop condition held there.
  a        the auxiliary decision, known for every shot of the population (a census): B: K2's tau3; D: the non-band
           check's final decision (tau3 where it ran, tau2 elsewhere); C_low, C_high: m. ("plain" mode: a = m everywhere.)
  side     a right (a = o: a disagreement of q with a adds an ML error) or a wrong (removes one).
  type     (Libra, reference) right/wrong: D10 = Libra wrong, reference right; D01; C00; C11.
Estimates (rules A and none; rule A uses only the sampled shots of its 19 leaves, a domain of each cell's sample):
  ML's errors under the MPS reference, per leaf = census errors of a + sum over the leaf's strata and types of
  N(leaf, stratum, type relative to a) x rate(cell, side of the type), rate = the share of the cell's sampled shots on
  that side where q != a (all MPS-done shots, resolved or not). The flips change the leaf tables; the count gap and
  g_5 (rev_d2_worstcase.Model, entry 7's estimator, refitted) follow. The sampling SE of the change in ML's errors uses
  the per-side binomial variances with the finite-population factor.
Worst cases (sensitivity limits): per group (primary: a stratum pooled over r; sensitivity: a cell) and side, the
  disagreement rate at a one-sided Clopper-Pearson bound; unresolved sampled shots count as disagreements for an upper
  bound and as agreements for a lower bound. "lower" case = a-right rates at the upper bound and a-wrong rates at the
  lower bound (ML's errors up, Libra's gap down); "higher" = the reverse. Levels: separate (each bound 95%) and
  simultaneous (Bonferroni: alpha = 0.05 / (2 x groups) for the two cases, 0.05 / (4 x groups) for the g_5 box).
  g_5 box: extremes of g_5 over the box's vertices (pooled groups: 4 corners per group; cells: the 2 corners that move
  ML's errors the same way at the cell's r).
Also: agreement tables (q against m and against a), the stop-rule audit, margins, the sparse rebuild / tau / cap
checks, and MPS diagnostics.
usage: python3 -I -B rev_d1_analyze.py --res <results/decoder/willow> --sel <dir> --runs <dir> --final-r1 <json>
                                      --check <results/decoder/willow/supplementary/nonband_check> --out DIR
"""
import argparse, glob, hashlib, importlib.util, itertools, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.stats import beta as BETA

DN = "ml_tau2_1e-6-1e-8_0.1"
LIBRA = "libra_decoder_with_rl_optimized_prior"
W = (10, 30, 50)
STRATA = ("B", "D", "C_low", "C_high")
TYPES = ("D10", "D01", "C00", "C11")
RIGHT_T = ("D10", "C00")
LN9 = math.log(9.0)
SIDES = ("right", "wrong")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cp_up(k, n, a=0.05):
    return 1.0 if n == 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


def cp_lo(k, n, a=0.05):
    return 0.0 if n == 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def cp2(k, n):
    return [cp_lo(k, n, 0.025), cp_up(k, n, 0.025)]


def sig(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def pred_of_llr(llr):
    """decode_willow._ml_out's rule: pred = nan_to_num(P1, 0) > 0.5, with P1 = sigmoid(llr)."""
    return (np.nan_to_num(sig(llr), nan=0.0) > 0.5).astype(np.int64)


def tcode(lib_err, ref_err):
    return np.where(lib_err & ~ref_err, 0, np.where(~lib_err & ref_err, 1, np.where(~lib_err & ~ref_err, 2, 3)))


def f32eq(x, y):
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return bool(np.float32(x) == np.float32(y))


def read_lines(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            try:
                x = json.loads(ln)
            except ValueError:
                continue                                            # a torn last line from a kill
            out[int(x["row"])] = x
    return out


def q_stats(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    if not v:
        return None
    return {"n": len(v), "median": float(np.median(v)), "q90": float(np.quantile(v, 0.9)), "max": float(max(v))}


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--sel", "--runs", "--final-r1", "--check", "--out"):
        ap.add_argument(x, required=True)
    a_ = ap.parse_args()
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("rd2", os.path.join(HERE, "rev_d2_worstcase.py"))
    rd2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd2)
    meta = json.load(open(os.path.join(a_.sel, "sample.json")))
    zp = os.path.join(a_.sel, "sample.npz")
    assert md5(zp) == meta["sample_npz_md5"], "sample.npz changed"
    Z = np.load(zp)
    lid, pos, rnd, strat, sidx = Z["leaf_id"], Z["pos"], Z["rounds"], Z["stratum"], Z["idx"]
    order = Z["order"]
    n = len(lid)
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
        P[k] = {"m": m, "l": lib, "o": o, "a": aux, "llr": z["llr"].astype(np.float64), "la": la, "llr_t1": z["llr_t1"],
                "t3": t3, "nb_t2": nb_t2, "band": band, "code": code, "typ_m": typ_m, "typ_a": typ_a, "idx": idx}
    nD_pop = sum(int((P[k]["code"] == 1).sum()) for k in P)
    assert nD_pop == int((S["stratum"] == 0).sum()), "the non-band check's D census differs from the D stratum"
    F1 = json.load(open(a_.final_r1))
    exA = sorted(x for x in (F1.get("excluded_leaves") or []) if x in T)
    rules = (("A", exA), ("none", []))
    mps = read_lines(os.path.join(a_.runs, "mps_w*.jsonl"))
    spr = read_lines(os.path.join(a_.runs, "sparse_w*.jsonl"))
    inputs["runs"] = {os.path.basename(p): md5(p) for p in sorted(glob.glob(os.path.join(a_.runs, "*_w*.jsonl")))}
    # ------------------------------------------------------------ per sampled shot
    R_ = []
    for i in range(n):
        k = int(lid[i]); p = int(pos[i]); d = P[k]
        assert int(d["code"][p]) == int(strat[i]) and int(d["idx"][p]) == int(sidx[i])
        rec = {"row": i, "order": int(order[i]), "leaf_id": k, "leaf": leaves[k], "rounds": int(rnd[i]),
               "stratum": STRATA[int(strat[i])], "type_m": TYPES[int(d["typ_m"][p])], "type_a": TYPES[int(d["typ_a"][p])],
               "m": int(d["m"][p]), "l": int(d["l"][p]), "o": int(d["o"][p]), "a": int(d["a"][p]),
               "P_paper": float(sig(d["llr"][p])), "P_aux": float(sig(d["la"][p])), "P_t1": float(sig(d["llr_t1"][p])),
               "P_t3_K2": float(sig(d["t3"][p])) if np.isfinite(d["t3"][p]) else None}
        x = mps.get(i)
        if x is not None:
            Pl = [float(v) for v in x["P1"]]
            ff = x.get("first_fire")
            rec.update(mps_done=True, q=int(x["pred"]), resolved=bool(x["resolved"]), P_mps=Pl[-1], P_rungs=Pl,
                       chi=int(x["chis"][-1]), rungs=len(Pl), first_fire=ff, pred_first_fire=x.get("pred_first_fire"),
                       audit=bool(x.get("audit", False)), maxbond=max(x["maxbond"]), mps_cpu=float(sum(x["cpu_s"])),
                       dP_last=abs(Pl[-1] - Pl[-2]) if len(Pl) >= 2 else None)
        else:
            rec["mps_done"] = False
        y = spr.get(i)
        if y is not None:
            Rr = y["runs"]
            rec["sparse"] = {nm: {"pred": int(v["pred"]), "llr": float(v["llr"]), "maxk": int(v["maxk"]), "kmax": int(v["kmax"])}
                             for nm, v in Rr.items()}
            rec["rebuild_t1"] = f32eq(Rr["t1"]["llr"], d["llr_t1"][p]) if "t1" in Rr else None
            if d["band"][p] and "t2" in Rr:
                rec["rebuild_t2_band"] = f32eq(Rr["t2"]["llr"], d["llr"][p])
            if int(d["code"][p]) == 1 and "t2" in Rr:
                rec["rebuild_t2_nonband_check"] = f32eq(Rr["t2"]["llr"], d["nb_t2"][p])
        R_.append(rec)
    done = [s for s in R_ if s["mps_done"]]
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "post hoc revision analysis (D1); plan declared (D1-1) and script frozen (D1-2) before any result was read",
           "inputs_md5": inputs, "rule_A_excluded": exA,
           "sample": {"selected": n, "mps_done": len(done), "mps_resolved": sum(s["resolved"] for s in done),
                      "sparse_done": len(spr)},
           "cells": meta["cells"]}
    # ------------------------------------------------------------ 1. agreement tables
    def agree(sel, ref):
        nd = len(sel); nr = sum(s["resolved"] for s in sel)
        dis = [s for s in sel if s["q"] != s[ref]]
        add = sum(s[ref] == s["o"] for s in dis)
        return {"done": nd, "resolved": nr, "unresolved": nd - nr, "disagree": len(dis),
                "disagree_resolved": sum(s["resolved"] for s in dis), "adds_ML_error": add, "removes_ML_error": len(dis) - add,
                "rate": len(dis) / nd if nd else None, "ci95": cp2(len(dis), nd)}
    tabs = {}
    for ref, nm in (("m", "mps_vs_paper"), ("a", "mps_vs_aux")):
        t = {"overall": agree(done, ref)}
        for s_ in STRATA:
            t[s_] = agree([s for s in done if s["stratum"] == s_], ref)
            for r in W:
                t[f"{s_}|{r}"] = agree([s for s in done if s["stratum"] == s_ and s["rounds"] == r], ref)
        for r in W:
            t[f"all|{r}"] = agree([s for s in done if s["rounds"] == r], ref)
        tabs[nm] = t
    tabs["aux_vs_paper_in_sample"] = {s_: {"n": sum(1 for s in R_ if s["stratum"] == s_),
                                           "aux_differs": sum(1 for s in R_ if s["stratum"] == s_ and s["a"] != s["m"])}
                                      for s_ in STRATA}
    out["agreement"] = tabs
    # ------------------------------------------------------------ 2-3. implied gap: estimates and worst cases
    def side_of(s, ref):
        return "right" if s[ref] == s["o"] else "wrong"

    def stats(domain, mode, keyf):
        ref = "a" if mode == "diff" else "m"
        G = {}
        for s in done:
            if s["leaf"] not in domain:
                continue
            c = G.setdefault(keyf(s["stratum"], s["rounds"]), {sd: {"n": 0, "dis": 0, "dis_res": 0, "unres": 0} for sd in SIDES})
            c = c[side_of(s, ref)]
            dis = s["q"] != s[ref]
            c["n"] += 1; c["dis"] += int(dis)
            if s["resolved"]:
                c["dis_res"] += int(dis)
            else:
                c["unres"] += 1
        return G

    def pop_side(domain, mode, keyf):
        Np = {}
        for k, rel in enumerate(leaves):
            if rel not in domain:
                continue
            r = info[k]["rounds"]
            for s_ in STRATA:
                c = Np.setdefault(keyf(s_, r), {"right": 0, "wrong": 0})
                for t in TYPES:
                    c["right" if t in RIGHT_T else "wrong"] += popc[k][mode][(s_, t)]
        return Np

    def build_F(domain, mode, rate_fn):
        F = {}
        for k, rel in enumerate(leaves):
            if rel not in domain:
                continue
            r = info[k]["rounds"]
            f = {t: (float(Fc[k][t]) if mode == "diff" else 0.0) for t in TYPES}
            for s_ in STRATA:
                for t in TYPES:
                    N = popc[k][mode][(s_, t)]
                    if N:
                        f[t] += N * rate_fn(s_, r, "right" if t in RIGHT_T else "wrong")
            F[rel] = f
        return F

    def summ(M, F, domain):
        g = M.g5(F); cg = M.count_gap(F)
        return {"g5": g, "count_gap": cg,
                "flips": {t: float(sum(FF[t] for rel, FF in F.items() if rel in domain)) for t in TYPES}}

    cell_key = lambda s_, r: f"{s_}|{r}"
    pooled_key = lambda s_, r: s_
    out["rules"] = {}
    for rule, ex in rules:
        domain = set(leaves) - set(ex)
        M = rd2.Model(leaves, info, T, ex)
        RR = {"base": {"g5": M.g5({}), "count_gap": M.count_gap({})}}
        for mode in ("diff", "plain"):
            Gc = stats(domain, mode, cell_key); Npc = pop_side(domain, mode, cell_key)

            def est_rate(s_, r, side, Gc=Gc):
                c = Gc.get(cell_key(s_, r))
                if not c:
                    return 0.0
                if c[side]["n"]:
                    return c[side]["dis"] / c[side]["n"]
                nn = sum(c[x]["n"] for x in SIDES)
                return sum(c[x]["dis"] for x in SIDES) / nn if nn else 0.0
            F = build_F(domain, mode, est_rate)
            e = summ(M, F, domain)
            var = 0.0; flag = []
            for key, c in Gc.items():
                for side in SIDES:
                    nn, Nn = c[side]["n"], Npc[key][side]
                    if nn >= 2 and Nn:
                        p_ = c[side]["dis"] / nn
                        var += Nn ** 2 * p_ * (1 - p_) / nn * max(0.0, 1 - nn / Nn)
                    elif Nn:
                        flag.append(f"{key}:{side}")
            e["se_change_in_ML_errors"] = math.sqrt(var); e["cells_sides_without_variance"] = flag
            e["census_flips"] = ({t: float(sum(Fc[k][t] for k, rel in enumerate(leaves) if rel in domain)) for t in TYPES}
                                 if mode == "diff" else None)
            e["cells"] = {key: {sd: dict(c[sd], N=Npc[key][sd]) for sd in SIDES} for key, c in sorted(Gc.items())}
            e["cells_without_mps_rows_rate_0"] = sorted(k_ for k_ in Npc if k_ not in Gc)
            RR[f"estimate_{mode}"] = e
            if mode == "diff":
                # posterior intervals: rates ~ Beta(k + 1/2, n - k + 1/2) (Jeffreys), independently per (cell, side)
                # ("cells"; a side without sampled shots in a cell takes its stratum's pooled counts) or per (stratum,
                # side) applied at every r ("pooled": rates assumed not to depend on r); q as is (unresolved shots
                # included, as in the estimate); census part fixed; 2,000 draws each, fixed seed
                Gp = stats(domain, mode, pooled_key)
                qq = lambda v: ([float(x) for x in np.nanquantile(v, [0.025, 0.5, 0.975])]
                                if np.isfinite(np.asarray(v, float)).any() else [None, None, None])
                for level, seed in (("cells", 20261016), ("pooled", 20261017)):
                    rng = np.random.default_rng(seed)
                    dg, dgn, dcg, dcr = [], [], [], []
                    for _ in range(2000):
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
                    RR[f"posterior_diff_{level}"] = {"draws": 2000, "seed": seed,
                                                     "prior": "Jeffreys Beta(1/2, 1/2) per " + ("cell" if level == "cells" else "stratum") + " and side",
                                                     "g5_correction_only_q025_q50_q975": qq(dg),
                                                     "g5_with_shot_noise_q025_q50_q975": qq(dgn),
                                                     "count_gap_errors_q025_q50_q975": qq(dcg), "count_gap_rel_q025_q50_q975": qq(dcr),
                                                     "P_count_gap_le_0": float(np.mean(np.array(dcg) <= 0)),
                                                     "P_g5_le_0_with_shot_noise": float(np.nanmean(np.array(dgn) <= 0)),
                                                     "draws_with_nan_g5": int(np.sum(~np.isfinite(np.array(dg))))}
            # worst cases
            for gname, keyf in (("pooled", pooled_key), ("per_cell", cell_key)):
                G = stats(domain, mode, keyf)
                keys = sorted({keyf(s_, r) for s_ in STRATA for r in W})
                ng = len(keys)

                def bound(key, side, which, alpha, G=G):
                    c = G.get(key, {}).get(side, {"n": 0, "dis_res": 0, "unres": 0})
                    if which == "up":
                        return cp_up(c["dis_res"] + c["unres"], c["n"], alpha)
                    return cp_lo(c["dis_res"], c["n"], alpha)
                RR[f"groups_{mode}_{gname}"] = {key: G.get(key) for key in keys}
                for adj, al2 in (("separate", 0.05), ("simultaneous", 0.05 / (2 * ng))):
                    for case, (wr, ww) in (("lower", ("up", "lo")), ("higher", ("lo", "up"))):
                        rf = (lambda s_, r, side, keyf=keyf, wr=wr, ww=ww, al2=al2:
                              bound(keyf(s_, r), side, wr if side == "right" else ww, al2))
                        F = build_F(domain, mode, rf)
                        v = summ(M, F, domain); v.update(groups=ng, alpha_per_bound=al2)
                        RR[f"worst_{case}_{mode}_{gname}_{adj}"] = v
                if mode != "diff":
                    continue
                # g5 box over vertices (primary mode only)
                for adj, al4 in (("separate", 0.05), ("simultaneous", 0.05 / (4 * ng))):
                    B_ = {key: {side: {w: bound(key, side, w, al4) for w in ("lo", "up")} for side in SIDES} for key in keys}
                    corners = ([("lo", "lo"), ("lo", "up"), ("up", "lo"), ("up", "up")] if gname == "pooled"
                               else [("up", "lo"), ("lo", "up")])
                    best = {"min": (float("inf"), None), "max": (-float("inf"), None)}; nan_v = 0
                    for ch in itertools.product(range(len(corners)), repeat=ng):
                        cmap = dict(zip(keys, ch))
                        rf = (lambda s_, r, side, keyf=keyf, cmap=cmap, B_=B_, corners=corners:
                              B_[keyf(s_, r)][side][corners[cmap[keyf(s_, r)]][0 if side == "right" else 1]])
                        g = M.g5(build_F(domain, mode, rf))["g"]
                        if not math.isfinite(g):
                            nan_v += 1
                            continue
                        if g < best["min"][0]:
                            best["min"] = (g, ch)
                        if g > best["max"][0]:
                            best["max"] = (g, ch)
                    RR[f"box_{gname}_{adj}"] = {
                        "g5_min": best["min"][0] if best["min"][1] is not None else None,
                        "g5_max": best["max"][0] if best["max"][1] is not None else None,
                        "vertices": len(corners) ** ng, "vertices_with_nan_g5": nan_v, "alpha_per_bound": al4,
                        "argmin": {key: corners[c] for key, c in zip(keys, best["min"][1] or ())},
                        "argmax": {key: corners[c] for key, c in zip(keys, best["max"][1] or ())},
                        "corner_meaning": "(a-right rate bound, a-wrong rate bound); up = upper, lo = lower"}
        out["rules"][rule] = RR
    # ------------------------------------------------------------ 4. stop-rule audit, margins, MPS diagnostics
    aud = {}
    for s_ in STRATA + ("all",):
        A = [s for s in done if s["audit"] and (s_ == "all" or s["stratum"] == s_)]
        fired = [s for s in A if s["first_fire"] is not None]
        ch = [s for s in fired if s["pred_first_fire"] != s["q"]]
        aud[s_] = {"audited": len(A), "fired": len(fired), "decision_changed_after_first_fire": len(ch),
                   "ci95_change_rate": cp2(len(ch), len(fired)),
                   "first_fire_rung": {str(j): sum(1 for s in fired if s["first_fire"] == j) for j in range(1, 4)},
                   "abs_dP_first_fire_to_last": q_stats([abs(s["P_rungs"][s["first_fire"]] - s["P_mps"]) for s in fired])}
    out["audit"] = aud
    marg, diag = {}, {}
    for s_ in STRATA:
        sel = [s for s in done if s["stratum"] == s_]
        marg[s_] = {"n": len(sel),
                    "min_abs_P_paper_minus_half": min((abs(s["P_paper"] - 0.5) for s in sel if math.isfinite(s["P_paper"])), default=None),
                    "abs_P_mps_minus_paper": q_stats([abs(s["P_mps"] - s["P_paper"]) for s in sel]),
                    "abs_P_mps_minus_aux": q_stats([abs(s["P_mps"] - s["P_aux"]) for s in sel]),
                    "abs_P_mps_minus_half_min": min((abs(s["P_mps"] - 0.5) for s in sel if math.isfinite(s["P_mps"])), default=None),
                    "nan_P_mps": sum(1 for s in sel if not math.isfinite(s["P_mps"]))}
        for r in W:
            sel = [s for s in done if s["stratum"] == s_ and s["rounds"] == r]
            if not sel:
                continue
            diag[f"{s_}|{r}"] = {"n": len(sel), "resolved": sum(s["resolved"] for s in sel),
                                 "chi_reached": {str(c): sum(s["chi"] == c for s in sel) for c in sorted({s["chi"] for s in sel})},
                                 "audited": sum(s["audit"] for s in sel), "max_bond": max(s["maxbond"] for s in sel),
                                 "cpu_s_per_shot": float(np.mean([s["mps_cpu"] for s in sel])),
                                 "cpu_s_per_round": float(np.mean([s["mps_cpu"] / s["rounds"] for s in sel]))}
    out["margins"] = marg; out["mps_diagnostics"] = diag
    out["mps_cpu_core_hours"] = sum(s["mps_cpu"] for s in done) / 3600
    # ------------------------------------------------------------ 5. sparse on the same shots
    sp = [s for s in R_ if "sparse" in s]
    if sp:
        names = sorted({nm for s in sp for nm in s["sparse"]})
        st4 = {"n": len(sp), "rebuild_t1_identical": sum(bool(s.get("rebuild_t1")) for s in sp),
               "rebuild_t1_checked": sum(s.get("rebuild_t1") is not None for s in sp),
               "rebuild_t2_band_identical": sum(bool(s.get("rebuild_t2_band")) for s in sp),
               "rebuild_t2_band_checked": sum("rebuild_t2_band" in s for s in sp),
               "rebuild_t2_vs_nonband_check_identical": sum(bool(s.get("rebuild_t2_nonband_check")) for s in sp),
               "rebuild_t2_vs_nonband_check_checked": sum("rebuild_t2_nonband_check" in s for s in sp)}
        for s_ in STRATA + ("all",):
            sel = [s for s in sp if s_ == "all" or s["stratum"] == s_]
            row = {"n": len(sel)}
            for nm in names:
                hv = [s for s in sel if nm in s["sparse"]]
                if not hv:
                    continue
                both = [s for s in hv if s["mps_done"]]
                bres = [s for s in both if s["resolved"]]
                row[nm] = {"n": len(hv), "vs_paper_changes": sum(s["sparse"][nm]["pred"] != s["m"] for s in hv),
                           "vs_mps_disagree": sum(s["sparse"][nm]["pred"] != s["q"] for s in both), "mps_done": len(both),
                           "vs_mps_disagree_resolved": sum(s["sparse"][nm]["pred"] != s["q"] for s in bres), "mps_resolved": len(bres),
                           "cap_bound": sum(s["sparse"][nm]["maxk"] >= s["sparse"][nm]["kmax"] for s in hv),
                           "maxk_max": max(s["sparse"][nm]["maxk"] for s in hv)}
                if nm.startswith("t2_k"):
                    pr = [s for s in hv if "t2" in s["sparse"]]
                    dP = [abs(float(sig(s["sparse"][nm]["llr"])) - float(sig(s["sparse"]["t2"]["llr"]))) for s in pr]
                    row[nm].update(vs_t2_changes=sum(s["sparse"][nm]["pred"] != s["sparse"]["t2"]["pred"] for s in pr),
                                   vs_t2_abs_dP=q_stats(dP),
                                   changes_where_cap_bound=sum(s["sparse"][nm]["pred"] != s["sparse"]["t2"]["pred"] for s in pr
                                                               if s["sparse"][nm]["maxk"] >= s["sparse"][nm]["kmax"]))
            st4[s_] = row
        out["sparse"] = st4
    out["shots"] = R_
    out["seconds"] = time.time() - t0
    os.makedirs(a_.out, exist_ok=True)
    json.dump(out, open(os.path.join(a_.out, "d1_results.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(a_.out, "d1_results.md"))
    print(json.dumps({"sample": out["sample"], "seconds": out["seconds"]}, default=float))


def write_md(o, path):
    pc = lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"
    L = [f"# D1: boundary-MPS check of the d = 5 ML(RL) reference (results)", "",
         f"Made {o['made']} by `rev_d1_analyze.py` (md5 {o['script_md5']}). Post hoc revision analysis; plan declared (D1-1) "
         f"and script frozen (D1-2) before any result was read.", "",
         f"Sample: {o['sample']}. Rule A excludes {len(o['rule_A_excluded'])} leaves.", "",
         "## Agreement", ""]
    for nm, title in (("mps_vs_paper", "MPS against the paper's decision m"),
                      ("mps_vs_aux", "MPS against the census decision a (B: K2 tau3; D: non-band check; C: m)")):
        L += [f"### {title}", "", "| cell | done | resolved | disagree (adds / removes an ML error) | rate (95% CI) |",
              "|---|---|---|---|---|"]
        for key, v in o["agreement"][nm].items():
            fr = "n/a" if v["rate"] is None else f"{100 * v['rate']:.2f}% ({100 * v['ci95'][0]:.2f}, {100 * v['ci95'][1]:.2f})"
            L.append(f"| {key} | {v['done']} | {v['resolved']} | {v['disagree']} ({v['adds_ML_error']} / {v['removes_ML_error']}) | {fr} |")
        L.append("")
    L += ["## Implied Libra gap under the MPS reference", "",
          "| rule | case | g_5 (95% CI) | count gap (errors) |", "|---|---|---|---|"]
    for rule, R in o["rules"].items():
        for case, v in R.items():
            if case.startswith("groups_"):
                continue
            if case.startswith("posterior_"):
                L.append(f"| {rule} | {case}: 2.5/50/97.5% | g_5 {', '.join(pc(x) for x in v['g5_with_shot_noise_q025_q50_q975'])} "
                         f"(correction only: {', '.join(pc(x) for x in v['g5_correction_only_q025_q50_q975'])}) | "
                         f"{', '.join('n/a' if x is None else f'{x:+.0f}' for x in v['count_gap_errors_q025_q50_q975'])} |")
                continue
            if case.startswith("box_"):
                L.append(f"| {rule} | {case}: extremes of g_5 over {v['vertices']} vertices | {pc(v['g5_min'])} to {pc(v['g5_max'])} | |")
                continue
            g = v["g5"]; cg = v["count_gap"]
            extra = f"; SE of ML-error change {v['se_change_in_ML_errors']:.1f}" if "se_change_in_ML_errors" in v else ""
            L.append(f"| {rule} | {case} | {pc(g['g'])} ({pc(g['ci95'][0])}, {pc(g['ci95'][1])}) | "
                     f"{pc(cg['rel'])} ({cg['gap_errors']:+.0f}{extra}) |")
    L += ["", "## Stop-rule audit", "", "```", json.dumps(o["audit"], indent=1), "```", "",
          "## Margins", "", "```", json.dumps(o["margins"], indent=1), "```", ""]
    if "sparse" in o:
        L += ["## Sparse decoder on the same shots (rebuild, tau, cap)", "", "```", json.dumps(o["sparse"], indent=1), "```", ""]
    L += ["## MPS diagnostics", "", f"MPS CPU: {o['mps_cpu_core_hours']:.2f} core-hours.", "", "```",
          json.dumps(o["mps_diagnostics"], indent=1), "```", ""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
