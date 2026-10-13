#!/usr/bin/env python3
"""A1-R2 (route 2 of the A1 assessment): analysis of the targeted sample (decoder, 2026-10-13). Post hoc revision
analysis, declared as A1-R2 in notes/decoder/willow_revision_declarations.md before any of its decoding ran.

Inputs: round 1's ladder outputs (--res), the non-band check (--check), final_round1.json (--final-r1, rule A's
exclusions), the A1-R2 selection (--sel, a1r2_select.py sample) and its sparse run (--runs: sparse_w0.jsonl from
rev_d1_run.py sparse, runs t1, t2, t3), and optionally the MPS subsample (--mps-sel, --mps-runs: mps_w*.jsonl).
Steps:
  1. Rebuild: on every decoded row, t1's llr must equal the stored tau1 llr (float32, bitwise).
  2. New rows: the decision at t3 (tau = 1e-9, the deepest rung, as the check's final decision) against the paper's
     decision m: a flip if they differ. t2's flips are counted too, and a secondary reading uses t2 on the new rows.
  3. Cells. Each driving cell (r10 C11, r30 C00, r50 C00) holds the check's sampled shots (decision: tau3 where the check
     ran it, else tau2) and the new rows (t3); a check shot that A1-R2 decoded again takes the new decision. A cell whose
     every shot is decoded is a census: its flips enter per leaf, exactly, with no bound. r10 C00, r30 C11 and r50 C11
     are as in R5. Discordant flips are the check's observed flips (as R5; no tau3-only term: "C only").
  4. Per-round box (R5's; rev_d2_worstcase.Model, entry 7's estimator; rules A and none): each coordinate that is not a
     census at its one-sided Clopper-Pearson bound at level alpha; g5 and the count gap at every vertex. Joint 95%:
     alpha = 0.05 / m, with m the number of non-trivial bounds at the extreme's vertex (minimum: r10 C00 lower, r10 C11
     upper, r30 C00 upper, r30 C11 lower, r50 C00 upper, r50 C11 lower; maximum: the reverse); an upper bound is
     trivial when k = n, a lower bound when k = 0. With no new rows this reproduces R5 (-4.50% at alpha/3, +20.22% at
     alpha/4). Also: every bound at 95%, and the count-gap floor at the minimum's level.
  5. Outcome (declared): read on the smaller of the two rules' joint-95% minima, g:
     g > 2.12% (g3's upper 95% limit): the condition is lifted; 0 < g <= 2.12%: g5 is positive per round count, but its
     growth per round count is not established; g <= 0: the condition stays.
  6. MPS subsample: the MPS decision q (the last rung with a finite P, as D1-3) against t3 and against m, per cell; a
     sensitivity box with q in place of t3 wherever q exists.
usage: python3 -I -B a1r2_analyze.py --res R --check C --final-r1 F --sel S --runs RUNS [--mps-sel S2 --mps-runs R2] --out DIR
"""
import argparse, functools, itertools, json, math, os, sys, time

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
G3_UPPER = 0.0212


@functools.lru_cache(maxsize=None)
def cp_up(k, n, a):
    return 1.0 if n <= 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


@functools.lru_cache(maxsize=None)
def cp_lo(k, n, a):
    return 0.0 if n <= 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def f32eq(x, y):
    return bool(np.float32(x).tobytes() == np.float32(y).tobytes())


def read_lines(path):
    out = {}
    if path and os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if not ln:
                continue
            try:
                x = json.loads(ln)
            except ValueError:
                continue                                            # a torn last line
            out[int(x["row"])] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--final-r1", "--sel", "--runs", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--mps-sel"); ap.add_argument("--mps-runs")
    a = ap.parse_args()
    t0 = time.time()
    # ------------------------------------------------------------------ the check, the tables, rule A (as a1_alloc.py)
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    assert D2.md5(lp) == meta["shot_lists_npz_md5"]
    S = np.load(lp)
    leaves = [str(x) for x in S["leaves"]]; info = meta["leaves"]
    n = len(S["idx"])
    lid, strat, typ, ml, rounds, cpos = (S["leaf_id"], S["stratum"], S["type"].astype(str), S["ml"].astype(np.int64),
                                         S["rounds"], S["pos"])
    d2 = np.full(n, -1); d3 = np.full(n, -1); done2 = np.zeros(n, bool); done3 = np.zeros(n, bool)
    for k in range(len(leaves)):
        z = np.load(os.path.join(a.check, "leaves", f"{k:02d}.npz")); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1); d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
        done2[r] = z["done_t2"]; done3[r] = z["done_t3"]
    flip = done2 & (np.where(done3, d3, d2) != ml)
    tables, llr_t1_store = {}, {}
    for rel in leaves:
        d = os.path.join(a.res, "decode", rel, "rl_s5")
        z = np.load(os.path.join(d, D2.DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        row = [i for i, x in enumerate(sh["shipped_names"]) if D2.LIBRA in str(x)]
        lib = sh["shipped"][row[0]].astype(np.int64); m = z["pred"][:, 0].astype(np.int64); o = sh["obs"].astype(np.int64)
        ea, eb = lib != o, m != o
        tables[rel] = {"n": int(len(o)), "n10": int((ea & ~eb).sum()), "n01": int((~ea & eb).sum()), "n11": int((ea & eb).sum())}
        llr_t1_store[rel] = (z["llr_t1"], m)
    pops = {x["leaf"]: x["population"] for x in info}
    rnd = {x["leaf"]: x["rounds"] for x in info}
    Dobs = {rel: {t: int(((lid == k) & (typ == t) & flip).sum()) for t in ("D10", "D01")} for k, rel in enumerate(leaves)}
    exA = sorted(x for x in (json.load(open(a.final_r1)).get("excluded_leaves") or []) if x in tables)
    # ------------------------------------------------------------------ A1-R2's rows
    sm = json.load(open(os.path.join(a.sel, "sample.json")))
    zp = os.path.join(a.sel, "sample.npz")
    assert D2.md5(zp) == sm["sample_npz_md5"], "A1-R2 selection changed"
    Z = np.load(zp)
    sleaves = [x["leaf"] for x in sm["leaves"]]
    cell_names = sm["cell_names"]
    N_rows = len(Z["order"])
    sp_path = os.path.join(a.runs, "sparse_w0.jsonl")
    SP = read_lines(sp_path)
    rows = []
    rebuild_same = rebuild_diff = 0
    for i in range(N_rows):
        rel = sleaves[int(Z["leaf_id"][i])]; pos = int(Z["pos"][i])
        st_llr, st_m = llr_t1_store[rel]
        assert int(st_m[pos]) == int(Z["m"][i]), "selection's m differs from the ladder"
        x = SP.get(i)
        rec = {"row": i, "leaf": rel, "pos": pos, "cell": cell_names[int(Z["cell"][i])], "m": int(Z["m"][i]),
               "obs": int(Z["obs"][i]), "done": x is not None}
        if x is not None:
            R = x["runs"]
            rec.update(t1=int(R["t1"]["pred"]), t2=int(R["t2"]["pred"]), t3=int(R["t3"]["pred"]),
                       llr_t2=float(R["t2"]["llr"]), llr_t3=float(R["t3"]["llr"]),
                       cpu=float(sum(v["cpu_s"] for v in R.values())))
            same = f32eq(R["t1"]["llr"], st_llr[pos])
            rec["rebuild_t1"] = same
            rebuild_same += same; rebuild_diff += (not same)
        rows.append(rec)
    by_key = {(r_["leaf"], r_["pos"]): r_ for r_ in rows}
    # ------------------------------------------------------------------ MPS subsample (optional)
    mps = {}
    if a.mps_sel and a.mps_runs:
        mm = json.load(open(os.path.join(a.mps_sel, "sample.json")))
        mz = os.path.join(a.mps_sel, "sample.npz")
        assert D2.md5(mz) == mm["sample_npz_md5"]
        MZ = np.load(mz)
        assert mm["from_selection_npz_md5"] == sm["sample_npz_md5"], "MPS subsample made from another selection"
        lines = {}
        for w in range(int(mm["workers"])):
            lines.update(read_lines(os.path.join(a.mps_runs, f"mps_w{w}.jsonl")))
        O_mps_sel = {"n_rows": int(len(MZ["order"])), "n_changed": int(mm["n_changed"]), "workers": int(mm["workers"]),
                     "sample_npz_md5": mm["sample_npz_md5"], "rows_with_lines": len(lines)}
        for j in range(len(MZ["order"])):
            x = lines.get(j)
            if x is None:
                continue
            Pl = [float(p) for p in x["P1"]]
            fin = [q for q, p in enumerate(Pl) if math.isfinite(p)]
            src = int(MZ["src_row"][j])
            assert int(MZ["m"][j]) == int(Z["m"][src]) and int(MZ["pos"][j]) == int(Z["pos"][src])
            mps[src] = {"failed": not fin, "q": (int(Pl[fin[-1]] > 0.5) if fin else None),
                        "P": (Pl[fin[-1]] if fin else None), "resolved": bool(x.get("resolved")),
                        "chi": (int(x["chis"][fin[-1]]) if fin else None), "changed": int(MZ["changed"][j]),
                        "mps_random": int(MZ["mps_random"][j]), "audit": bool(x.get("audit")),
                        "rung_errors": len(x.get("errors") or []), "cpu": float(sum(x.get("cpu_s") or []))}

    # ------------------------------------------------------------------ cells
    def cells_for(decision):
        """decision(rec) -> the new decision of a decoded row, or None. Returns cells and the per-leaf census flips."""
        cells, census = {}, {}
        for (r, t) in COORDS:
            mask = (strat == 1) & (typ == t) & (rounds == r)
            name = next((nm for nm, c in DRIVE.items() if c == (r, t)), None)
            k_ = n_ = 0
            per_leaf = {}
            covered = set()
            if name is not None:
                for r_ in rows:
                    if r_["cell"] == name and r_["done"]:
                        dcs = decision(r_)
                        if dcs is None:
                            continue
                        f = int(dcs != r_["m"])
                        k_ += f; n_ += 1
                        per_leaf[r_["leaf"]] = per_leaf.get(r_["leaf"], 0) + f
                        covered.add((r_["leaf"], r_["pos"]))
            for i in np.where(mask)[0]:
                key = (leaves[int(lid[i])], int(cpos[i]))
                if key in covered:
                    continue                                        # decoded again by A1-R2: the new decision counts
                f = int(flip[i])
                k_ += f; n_ += 1
                per_leaf[key[0]] = per_leaf.get(key[0], 0) + f
                covered.add(key)
            N_ = int(sum(pops[rel][t] for rel in leaves if rnd[rel] == r))
            cells[(r, t)] = {"k": k_, "n": n_, "N": N_, "census": n_ == N_}
            if n_ == N_:
                census[(r, t)] = per_leaf
        return cells, census

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

    def outcome(res):
        g = min(res["A"]["joint95_min"]["g5"], res["none"]["joint95_min"]["g5"])
        if g > G3_UPPER:
            return g, "lifted: the joint-95% per-round minimum exceeds g3's upper limit (2.12%) under both rules"
        if g > 0:
            return g, "positive per round count; growth per round count not established"
        return g, "the condition stays"

    O = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": D2.md5(os.path.abspath(__file__)),
         "imports": {"rev_d2_worstcase.py": D2.md5(os.path.join(HERE, "rev_d2_worstcase.py"))},
         "label": "A1-R2, post hoc revision analysis (declared before its decoding ran)",
         "inputs_md5": {"sample.npz": sm["sample_npz_md5"], "sparse_w0.jsonl": (D2.md5(sp_path) if os.path.exists(sp_path) else None),
                        "shot_lists.npz": meta["shot_lists_npz_md5"], "final_round1.json": D2.md5(a.final_r1)},
         "rule_A_excluded": exA}
    # rows and flips per cell
    per = {}
    for nm in cell_names:
        R_ = [r_ for r_ in rows if r_["cell"] == nm]
        dn = [r_ for r_ in R_ if r_["done"]]
        per[nm] = {"selected": len(R_), "decoded": len(dn), "flips_t3": sum(r_["t3"] != r_["m"] for r_ in dn),
                   "flips_t2": sum(r_["t2"] != r_["m"] for r_ in dn), "t2_t3_differ": sum(r_["t2"] != r_["t3"] for r_ in dn),
                   "t1_differs_from_m": sum(r_["t1"] != r_["m"] for r_ in dn),
                   "flipped_rows": [{"row": r_["row"], "leaf": r_["leaf"], "pos": r_["pos"], "m": r_["m"], "t2": r_["t2"], "t3": r_["t3"],
                                     "llr_t2": r_["llr_t2"], "llr_t3": r_["llr_t3"]} for r_ in dn if r_["t3"] != r_["m"] or r_["t2"] != r_["m"]],
                   "cpu_s": sum(r_["cpu"] for r_ in dn)}
    O["new_rows"] = per
    O["rebuild_t1"] = {"identical": rebuild_same, "different": rebuild_diff}
    res = {}
    cells, census = cells_for(lambda r_: r_["t3"])
    O["cells_primary"] = {f"r{c[0]}|{c[1]}": v for c, v in cells.items()}
    for rule, ex in (("A", exA), ("none", [])):
        res[rule] = evaluate(D2.Model(leaves, info, tables, ex), cells, census)
    O["primary"] = res
    g, txt = outcome(res)
    O["outcome"] = {"g5_joint95_min_smaller_rule": g, "reading": txt}
    # secondary: t2 in place of t3 on the new rows (most of the check's rows carry t2 decisions)
    cells2, census2 = cells_for(lambda r_: r_["t2"])
    sens2 = {rule: evaluate(D2.Model(leaves, info, tables, ex), cells2, census2) for rule, ex in (("A", exA), ("none", []))}
    O["secondary_t2_decisions"] = {"cells": {f"r{c[0]}|{c[1]}": v for c, v in cells2.items()},
                                   "joint95_min": {rule: sens2[rule]["joint95_min"]["g5"] for rule in sens2},
                                   "outcome": outcome(sens2)}
    # MPS subsample
    if mps:
        mp = {}

        def stats(X_):
            ok = [(r_, q) for r_, q in X_ if not q["failed"] and r_["done"]]
            s = {"rows": len(X_), "failed": sum(q["failed"] for _, q in X_), "with_t3_and_q": len(ok),
                 "unresolved": sum((not q["resolved"]) for _, q in ok),
                 "q_differs_from_t3": sum(q["q"] != r_["t3"] for r_, q in ok),
                 "q_differs_from_m": sum(q["q"] != r_["m"] for r_, q in ok),
                 "rows_with_rung_errors": sum(q["rung_errors"] > 0 for _, q in X_),
                 "last_finite_chi": {str(c): sum(q["chi"] == c for _, q in ok) for c in sorted({q["chi"] for _, q in ok})},
                 "cpu_s": sum(q["cpu"] for _, q in X_)}
            for key in ("q_differs_from_t3", "q_differs_from_m"):
                s[key + "_cp95"] = [cp_lo(s[key], len(ok), 0.025), cp_up(s[key], len(ok), 0.025)]
            return s

        for nm in cell_names:
            X_ = [(r_, mps[r_["row"]]) for r_ in rows if r_["cell"] == nm and r_["row"] in mps]
            mp[nm] = {"changed_rows": stats([(r_, q) for r_, q in X_ if q["changed"]]),
                      "predrawn_random_rows": stats([(r_, q) for r_, q in X_ if q["mps_random"]])}
        O["mps_selection"] = O_mps_sel
        O["mps"] = mp
        O["mps_disagreements"] = [{"row": r_["row"], "cell": r_["cell"], "m": r_["m"], "t3": r_.get("t3"), **q}
                                  for r_ in rows if r_["row"] in mps for q in [mps[r_["row"]]]
                                  if not q["failed"] and (q["q"] != r_["m"] or q["q"] != r_.get("t3"))]
        cells_s, census_s = cells_for(lambda r_: (mps[r_["row"]]["q"] if (r_["row"] in mps and not mps[r_["row"]]["failed"]) else r_["t3"]))
        sens = {rule: evaluate(D2.Model(leaves, info, tables, ex), cells_s, census_s) for rule, ex in (("A", exA), ("none", []))}
        O["sensitivity_mps_decisions"] = {"cells": {f"r{c[0]}|{c[1]}": v for c, v in cells_s.items()},
                                          "joint95_min": {rule: sens[rule]["joint95_min"]["g5"] for rule in sens},
                                          "outcome": outcome(sens)}
    O["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "a1r2_results.json")
    with open(p + ".tmp", "w") as f:
        json.dump(O, f, indent=1, default=float)
    os.replace(p + ".tmp", p)
    show = {"rebuild_t1": O["rebuild_t1"], "new_rows": {k: {x: v[x] for x in ("selected", "decoded", "flips_t3", "flips_t2")} for k, v in per.items()},
            "joint95_min": {r_: res[r_]["joint95_min"]["g5"] for r_ in res}, "joint95_max": {r_: res[r_]["joint95_max"]["g5"] for r_ in res},
            "n_bounds": {r_: (res[r_]["n_bounds_min"], res[r_]["n_bounds_max"]) for r_ in res}, "outcome": O["outcome"],
            "secondary_t2": O["secondary_t2_decisions"]["joint95_min"]}
    if mps:
        show["mps"] = O["mps"]; show["sensitivity"] = O["sensitivity_mps_decisions"]["joint95_min"]
    print(json.dumps(show, indent=1, default=float))


if __name__ == "__main__":
    main()
