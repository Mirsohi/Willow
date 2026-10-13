#!/usr/bin/env python3
"""A1-R2 (route 2 of the A1 assessment of the fresh-eyes review of v2.0 draft 2): selection of the targeted sample and,
after its sparse run, of the MPS subsample (decoder, 2026-10-13). Post hoc revision analysis, declared as A1-R2 in
notes/decoder/willow_revision_declarations.md. Reads round 1's ladder outputs, the non-band check's shot lists and the
local extract (archive data: run with python3 -I, from a folder outside it). Writes the selected shots' detection events
in rev_d1_run.py's selection format, so the decoding never reads the archive or a stored decision.

Cells: concordant shots outside the band, by type, pooled over the 8 W5 leaves of a round count, exactly as the non-band
check defines them (outside the band: not re-decoded at tau2; concordant: Libra's shipped decision = ML(RL)'s decision m;
C00: m = the outcome o; C11: m != o).
  r10_C11  every shot (a census; N = 1,814), the check's 15 sampled shots included
  r30_C00  800 shots, a simple random sample without replacement of the cell's shots that the check did not sample
  r50_C00  600 shots, likewise
Within each cell, 150 of the selected shots are pre-drawn at random for the MPS subsample (field mps_random).
Processing order: the census first, in random order; then r30_C00 and r50_C00 in waves (the w-th shot of each, in
random order within a wave). One worker (worker 0).

  sample --root E --res R --check C --out DIR [--seed 20261017]
  mps --sel DIR --runs RUNS --out DIR2 [--workers 3] [--seed 20261017]
      the MPS subsample: every selected row whose t2 or t3 decision (RUNS/sparse_w0.jsonl) differs from m, plus the
      pre-drawn rows; changed rows first (random order), then the pre-drawn rows in waves over the cells; worker = order
      mod --workers.
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rev_d1_select as SEL                                        # leaves, load_leaf, strata_of, dets_for, pad; read-only

CELLS = (("r10_C11", 10, "C11", None), ("r30_C00", 30, "C00", 800), ("r50_C00", 50, "C00", 600))
N_MPS_RANDOM = 150


def md5(p):
    return SEL.md5(p)


def sample(a):
    t0 = time.time()
    made = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rng = np.random.default_rng(a.seed)
    L = SEL.leaves(a.res)
    cm = json.load(open(os.path.join(a.check, "shot_lists.json")))
    cp = os.path.join(a.check, "shot_lists.npz")
    assert md5(cp) == cm["shot_lists_npz_md5"], "check shot lists changed"
    CS = np.load(cp)
    cleaves = [str(x) for x in CS["leaves"]]
    cpop = {x["leaf"]: x["population"] for x in cm["leaves"]}
    sampled = set()                                                # (leaf, pos) of the check's concordant sample
    for i in range(len(CS["pos"])):
        if int(CS["stratum"][i]) == 1:
            sampled.add((cleaves[int(CS["leaf_id"][i])], int(CS["pos"][i])))
    cand = {c[0]: [] for c in CELLS}                               # cell -> list of (leaf k, pos)
    info, cache = [], {}
    for k, (rel, r) in enumerate(L):
        s, z, sh, lrow, files = SEL.load_leaf(a.res, rel)
        band = z["redecoded"].astype(bool)
        m = z["pred"][:, 0].astype(np.int64); lib = sh["shipped"][lrow].astype(np.int64); o = sh["obs"].astype(np.int64)
        conc = ~band & (lib == m)
        typ = {"C00": conc & (m == o), "C11": conc & (m != o)}
        assert int(typ["C00"].sum()) == cpop[rel]["C00"] and int(typ["C11"].sum()) == cpop[rel]["C11"], \
            f"{rel}: concordant populations differ from the check's"
        info.append({"leaf": rel, "rounds": r, "files_md5": files, "n": int(len(m)),
                     "C00": int(typ["C00"].sum()), "C11": int(typ["C11"].sum())})
        for name, rr, t, nn in CELLS:
            if r == rr:
                for pos in np.where(typ[t])[0]:
                    cand[name].append((k, int(pos)))
        cache[k] = (s, z, sh, lrow, m, lib, o)
    picked, cells = {}, {}
    for name, rr, t, nn in CELLS:
        lst = cand[name]
        if nn is None:                                             # census
            perm = rng.permutation(len(lst))
            picked[name] = [lst[i] for i in perm]
            cells[name] = {"N": len(lst), "n_check": sum(1 for k, p in lst if (L[k][0], p) in sampled),
                           "n_new": len(lst), "census": True}
        else:
            free = [x for x in lst if (L[x[0]][0], x[1]) not in sampled]
            n_check = len(lst) - len(free)
            perm = rng.permutation(len(free))[:nn]
            picked[name] = [free[i] for i in perm]
            cells[name] = {"N": len(lst), "n_check": n_check, "n_new": len(picked[name]), "census": False,
                           "n_after": n_check + len(picked[name])}
    mps_pick = {name: set(int(i) for i in rng.choice(len(picked[name]), size=N_MPS_RANDOM, replace=False))
                for name, _, _, _ in CELLS}
    order = [("r10_C11", j) for j in range(len(picked["r10_C11"]))]
    wmax = max(len(picked["r30_C00"]), len(picked["r50_C00"]))
    for w in range(wmax):
        for name in [("r30_C00", "r50_C00")[i] for i in rng.permutation(2)]:
            if w < len(picked[name]):
                order.append((name, w))
    cols = {x: [] for x in ("leaf_id", "pos", "idx", "rounds", "stratum", "order", "worker", "cell", "mps_random",
                            "in_check", "m", "libra", "obs", "llr_t1")}
    by_leaf = {}
    for o_, (name, j) in enumerate(order):
        k, pos = picked[name][j]
        by_leaf.setdefault(k, []).append((o_, pos))
    det_rows = [None] * len(order)
    leaf_meta = {}
    for k in sorted(by_leaf):
        s, z, sh, lrow, m, lib, o = cache[k]
        os_, poss = zip(*by_leaf[k])
        idx = z["idx"][list(poss)].astype(np.int64)
        det, oo, dem_rel, dem_md5 = SEL.dets_for(a.root, L[k][0], s, idx)
        assert np.array_equal(oo.astype(np.int64), o[list(poss)]), f"{L[k][0]}: observed flips differ from shots.npz"
        for j, ox in enumerate(os_):
            det_rows[ox] = det[j]
        leaf_meta[k] = {"dem": dem_rel, "dem_md5": dem_md5}
    cell_names = [c[0] for c in CELLS]
    codes = {}
    for o_, (name, j) in enumerate(order):
        k, pos = picked[name][j]
        s, z, sh, lrow, m, lib, o = cache[k]
        if k not in codes:
            codes[k] = SEL.strata_of(z, sh, lrow)
        code = codes[k]
        assert int(code[pos]) in (2, 3)
        cols["leaf_id"].append(k); cols["pos"].append(pos); cols["idx"].append(int(z["idx"][pos]))
        cols["rounds"].append(L[k][1]); cols["stratum"].append(int(code[pos])); cols["order"].append(o_)
        cols["worker"].append(0); cols["cell"].append(cell_names.index(name)); cols["mps_random"].append(int(j in mps_pick[name]))
        cols["in_check"].append(int((L[k][0], pos) in sampled)); cols["m"].append(int(m[pos])); cols["libra"].append(int(lib[pos]))
        cols["obs"].append(int(o[pos])); cols["llr_t1"].append(float(z["llr_t1"][pos]))
    dp, nd = SEL.pad(det_rows)
    arr = {x: np.array(v, np.float32 if x == "llr_t1" else np.int64) for x, v in cols.items()}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "sample.npz"), det=dp, ndet=nd, **arr)
    for k in range(len(L)):
        info[k].update(leaf_meta.get(k, {"dem": None, "dem_md5": None}))
    meta = {"made": made, "script_md5": md5(os.path.abspath(__file__)), "imports": {"rev_d1_select.py": md5(os.path.join(HERE, "rev_d1_select.py"))},
            "what": "A1-R2 targeted sample: census of r10 C11; r30 C00 and r50 C00 extended to 1,556 and 1,321 with the check's shots",
            "seed": a.seed, "cells": cells, "cell_names": cell_names, "n_mps_random_per_cell": N_MPS_RANDOM,
            "n_total": len(order), "shot_rounds": int(sum(arr["rounds"])), "leaves": info,
            "check_shot_lists_npz_md5": cm["shot_lists_npz_md5"],
            "sample_npz_md5": md5(os.path.join(a.out, "sample.npz")), "seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(a.out, "sample.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("cells", "n_total", "shot_rounds", "sample_npz_md5")}))


def mps(a):
    t0 = time.time()
    rng = np.random.default_rng(a.seed + 1)
    meta = json.load(open(os.path.join(a.sel, "sample.json")))
    zp = os.path.join(a.sel, "sample.npz")
    assert md5(zp) == meta["sample_npz_md5"], "sample.npz changed"
    Z = np.load(zp)
    n = len(Z["order"])
    done, t2, t3 = np.zeros(n, bool), np.full(n, -1), np.full(n, -1)
    lp = os.path.join(a.runs, "sparse_w0.jsonl")
    for ln in open(lp):
        ln = ln.strip()
        if not ln:
            continue
        x = json.loads(ln)
        i = int(x["row"]); R = x["runs"]
        done[i] = True; t2[i] = int(R["t2"]["pred"]); t3[i] = int(R["t3"]["pred"])
    m = Z["m"]
    changed = done & ((t2 != m) | (t3 != m))
    rand = Z["mps_random"].astype(bool)
    rows_changed = [int(i) for i in np.where(changed)[0]]
    rows_changed = [rows_changed[i] for i in rng.permutation(len(rows_changed))]
    per_cell = {c: [int(i) for i in np.where(rand & ~changed & (Z["cell"] == c))[0]] for c in range(len(meta["cell_names"]))}
    for c in per_cell:
        per_cell[c] = [per_cell[c][i] for i in rng.permutation(len(per_cell[c]))]
    order = list(rows_changed)
    wmax = max(len(v) for v in per_cell.values())
    for w in range(wmax):
        for c in rng.permutation(len(per_cell)):
            if w < len(per_cell[int(c)]):
                order.append(per_cell[int(c)][w])
    src = np.array(order, np.int64)
    out = {x: Z[x][src] for x in Z.files if x not in ("det", "order", "worker")}
    out["det"] = Z["det"][src]
    out["order"] = np.arange(len(src), dtype=np.int64)
    out["worker"] = out["order"] % a.workers
    out["src_row"] = src
    out["changed"] = changed[src].astype(np.int64)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "sample.npz"), **out)
    m2 = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
          "what": "A1-R2 MPS subsample: rows whose t2 or t3 decision differs from m, plus the pre-drawn rows",
          "from_selection_npz_md5": meta["sample_npz_md5"], "sparse_jsonl_md5": md5(lp), "workers": a.workers,
          "sparse_rows_done": int(done.sum()), "n_changed": len(rows_changed),
          "n_random": {meta["cell_names"][c]: len(v) for c, v in per_cell.items()}, "n_total": len(src),
          "shot_rounds": int(out["rounds"].sum()), "leaves": meta["leaves"], "cell_names": meta["cell_names"],
          "sample_npz_md5": md5(os.path.join(a.out, "sample.npz")), "seconds": time.time() - t0}
    json.dump(m2, open(os.path.join(a.out, "sample.json"), "w"), indent=1)
    print(json.dumps({k: m2[k] for k in ("sparse_rows_done", "n_changed", "n_random", "n_total", "shot_rounds", "sample_npz_md5")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sample", "mps"))
    ap.add_argument("--root"); ap.add_argument("--res"); ap.add_argument("--check")
    ap.add_argument("--sel"); ap.add_argument("--runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args()
    if a.mode == "sample":
        assert a.root and a.res and a.check
        sample(a)
    else:
        assert a.sel and a.runs
        mps(a)


if __name__ == "__main__":
    main()
