#!/usr/bin/env python3
"""D1-9 (review 1.3): the shot selection for a one-off, targeted extension of D1 (decoder, 2026-10-24).
Post hoc revision analysis. A DRAFT until D1-9 is declared; its md5 is declared in D1-9b before any real draw.

Reads round 1's ladder outputs (decode/<leaf>/rl_s5), K2's extra rung, the non-band check (for the census decision a),
final_round1.json (rule A's exclusions), D1's selection (sample.json, sample.npz) and the local extract (detection
events, through rev_d1_select.py). It reads no MPS or sparse result. Run it with python3 -I, from a folder outside the
extract.

The census decision a, as in rev_d1_analyze.py (D1-2): B: K2's tau3 (pred = sigmoid(llr) > 0.5, NaN -> 0); D: the
non-band check's final decision (tau3 where it ran, tau2 elsewhere); C_low, C_high: m. A shot is on side "right" if
a = o, else "wrong". A design side j is "stratum|r:side". Its population is the shots of rule A's leaves in that cell
(stratum, r) and on that side that D1 did not select.

The draw (rng = numpy default_rng(the design's seed)):
  1. For each side j of the design with n_j > 0, in the order of strata B, D, C_low, C_high, then r = 10, 30, 50, then
     right, wrong: the population in a fixed order (leaf, then position in the leaf), and rng.permutation(N_j)[:n_j].
     This is a simple random sample without replacement; its order is kept as the side's processing order.
  2. For each such side, in the same order: u_j = rng.random().
  3. One tie-break uniform per drawn row, rng.random(n_total), with the rows in the order of step 1.
  Processing order: the i-th drawn shot of side j (i = 0, 1, ...) has key (i + u_j) / n_j. Rows run by key, then by
  the tie-break. Row o of that order (o = 0, 1, ...) gets order = D1's row count + o and worker = worker_base +
  (o mod workers). Whether a row is done therefore depends only on the rows before it, and any prefix of the queue
  holds close to the same fraction of every side.
Outputs in --out: sample.npz and sample.json (D1's rows first, every field unchanged, then D1-9's rows in processing
order, in the format rev_d1_run.py and rev_d1_analyze.py read; extra fields part (0 = D1, 1 = D1-9), side_a (0 = right,
1 = wrong) and key (NaN for D1's rows)), and design.json (the design as applied).
usage: python3 -I -B rev_d1_extend_select.py --root <extract> --res <results/decoder/willow> --sel <D1's selection>
         --final-r1 <final_round1.json> --check <supplementary/nonband_check> --design DESIGN.json --out DIR
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

SELECT_MD5 = "e0518efcd03ea3b90b41e41ab5ab3df9"                 # rev_d1_select.py, D1's selection script (D1-1)
SIDES = ("right", "wrong")
D1_FIELDS = {"det", "ndet", "leaf_id", "pos", "idx", "rounds", "stratum", "order", "worker"}


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
    """rev_d1_analyze.py's rule (decode_willow._ml_out): pred = nan_to_num(P1, 0) > 0.5, with P1 = sigmoid(llr)."""
    return (np.nan_to_num(sig(llr), nan=0.0) > 0.5).astype(np.int64)


def load_select():
    p = os.path.join(HERE, "rev_d1_select.py")
    assert md5(p) == SELECT_MD5, "rev_d1_select.py differs from D1's"
    spec = importlib.util.spec_from_file_location("rev_d1_select", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_side(j, strata, rounds):
    cell, sd = j.rsplit(":", 1)
    st, r = cell.split("|")
    assert st in strata and int(r) in rounds and sd in SIDES, f"not a side: {j}"
    return st, int(r), sd


def canon(keys, strata, rounds):
    return sorted(keys, key=lambda j: (strata.index(parse_side(j, strata, rounds)[0]), parse_side(j, strata, rounds)[1],
                                       SIDES.index(parse_side(j, strata, rounds)[2])))


def draw(pools, sizes, seed, strata, rounds):
    """pools: side -> list of (leaf_id, pos) in the population's fixed order; sizes: side -> n.
    Returns (rows, q, key): rows = [(side, i, leaf_id, pos)] in step 1's order (i = the shot's place in its side's
    draw), q = the processing order (indices into rows) and key = each row's key."""
    rng = np.random.default_rng(seed)
    js = [j for j in canon([j for j, n in sizes.items() if n > 0], strata, rounds)]
    rows = []
    for j in js:
        P, n = pools[j], int(sizes[j])
        assert 0 < n <= len(P), f"{j}: {n} asked, {len(P)} unsampled"
        perm = rng.permutation(len(P))[:n]
        rows += [(j, i, int(P[int(p)][0]), int(P[int(p)][1])) for i, p in enumerate(perm)]
    u = {j: float(rng.random()) for j in js}
    tb = rng.random(len(rows))
    key = np.array([(i + u[j]) / int(sizes[j]) for j, i, _, _ in rows], np.float64)
    q = np.lexsort((tb, key))
    return rows, q, key


def census(RS, res, check, L, d1_leaves, inputs):
    """Per leaf: stratum code, side of a (0 right, 1 wrong), shot index and outcome, as rev_d1_analyze.py derives them."""
    cm = json.load(open(os.path.join(check, "shot_lists.json")))
    cp_ = os.path.join(check, "shot_lists.npz")
    assert md5(cp_) == cm["shot_lists_npz_md5"], "shot_lists.npz changed"
    inputs["shot_lists.npz"] = cm["shot_lists_npz_md5"]; inputs["shot_lists.json"] = md5(os.path.join(check, "shot_lists.json"))
    S = np.load(cp_)
    cleaves = [str(x) for x in S["leaves"]]
    nS = len(S["idx"])
    d2 = np.full(nS, -1); d3 = np.full(nS, -1)
    inputs["nonband_leaves"] = {}
    for k in range(len(cleaves)):
        p_ = os.path.join(check, "leaves", f"{k:02d}.npz")
        inputs["nonband_leaves"][f"{k:02d}.npz"] = md5(p_)
        z = np.load(p_); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"].astype(np.int64), -1)
        d3[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1)
    nbD = {}
    for i in np.where(S["stratum"] == 0)[0]:
        assert d2[i] >= 0, "a D shot without tau2 in the non-band check"
        nbD[(cleaves[int(S["leaf_id"][i])], int(S["idx"][i]))] = int(d3[i]) if d3[i] >= 0 else int(d2[i])
    out = []
    inputs["k2_extra_rung"] = {}
    for k, (rel, r) in enumerate(L):
        s, z, sh, lrow, files = RS.load_leaf(res, rel)
        assert files == d1_leaves[k]["files_md5"], f"{rel}: files changed since D1's selection"
        code = RS.strata_of(z, sh, lrow)
        assert {st: int((code == i).sum()) for i, st in enumerate(RS.STRATA)} == d1_leaves[k]["strata"], f"{rel}: strata"
        m = z["pred"][:, 0].astype(np.int64); o = sh["obs"].astype(np.int64); idx = z["idx"].astype(np.int64)
        band = z["redecoded"].astype(bool)
        ep = os.path.join(res, "supplementary", "k2_extra_rung", rel, "rl_s5", "extra_rung.npz")
        inputs["k2_extra_rung"][rel] = md5(ep)
        e = np.load(ep)
        posmap = {int(x): i for i, x in enumerate(idx)}
        t3 = np.full(len(m), np.nan)
        for x, v in zip(e["idx"], e["llr_t3"]):
            t3[posmap[int(x)]] = float(v)
        assert sorted(int(x) for x in e["idx"]) == sorted(int(x) for x in idx[band]), f"{rel}: K2 does not cover the band"
        a = m.copy()
        a[band] = pred_of_llr(t3[band])
        for p in np.where(code == 1)[0]:
            a[p] = nbD[(rel, int(idx[p]))]
        out.append({"s": s, "code": code, "side": np.where(a == o, 0, 1).astype(np.int64), "idx": idx,
                    "obs": sh["obs"].astype(np.uint8)})
    assert sum(int((c["code"] == 1).sum()) for c in out) == int((S["stratum"] == 0).sum()), \
        "the non-band check's D census differs from the D stratum"
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("--root", "--res", "--sel", "--final-r1", "--check", "--design", "--out"):
        ap.add_argument(x, required=True)
    a_ = ap.parse_args()
    t0 = time.time()
    made = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RS = load_select()
    STRATA, W = tuple(RS.STRATA), tuple(RS.W)
    assert os.path.realpath(a_.out) != os.path.realpath(a_.sel), "--out is D1's selection folder"
    for f in ("sample.npz", "sample.json", "design.json"):
        assert not os.path.exists(os.path.join(a_.out, f)), f"{a_.out}/{f} exists: not overwritten"
    design = json.load(open(a_.design))
    seed, nw, wb = int(design["seed"]), int(design["workers"]), int(design["worker_base"])
    assert nw >= 1
    sizes = {j: int(v["n"]) for j, v in design["sides"].items()}
    for j in sizes:
        parse_side(j, STRATA, W)
    # ------------------------------------------------------------ D1's selection, checked and kept as it is
    m1 = json.load(open(os.path.join(a_.sel, "sample.json")))
    z1p = os.path.join(a_.sel, "sample.npz")
    assert md5(z1p) == m1["sample_npz_md5"], "D1's sample.npz changed"
    Z1 = np.load(z1p)
    assert set(Z1.files) == D1_FIELDS, f"D1's sample.npz fields: {sorted(Z1.files)}"
    D1 = {f: Z1[f] for f in Z1.files}
    n1 = len(D1["leaf_id"])
    assert sorted(int(x) for x in D1["order"]) == list(range(n1)), "D1's orders are not 0 .. n - 1"
    assert int(D1["worker"].max()) < wb, "D1-9's workers would collide with D1's"
    L = RS.leaves(a_.res)
    assert [(x["leaf"], x["rounds"]) for x in m1["leaves"]] == L, "the leaves differ from D1's"
    inputs = {"d1_sample.json": md5(os.path.join(a_.sel, "sample.json")), "d1_sample.npz": m1["sample_npz_md5"],
              "final_round1.json": md5(a_.final_r1), "design": md5(a_.design), "rev_d1_select.py": SELECT_MD5}
    C = census(RS, a_.res, a_.check, L, m1["leaves"], inputs)
    for i in range(n1):
        k, p = int(D1["leaf_id"][i]), int(D1["pos"][i])
        assert int(C[k]["code"][p]) == int(D1["stratum"][i]) and int(C[k]["idx"][p]) == int(D1["idx"][i]), f"D1 row {i}"
    d1_set = {(int(k), int(p)) for k, p in zip(D1["leaf_id"], D1["pos"])}
    assert len(d1_set) == n1, "D1 has a repeated shot"
    F1 = json.load(open(a_.final_r1))
    rels = [rel for rel, _ in L]
    exA = sorted(x for x in (F1.get("excluded_leaves") or []) if x in rels)
    inA = [rel not in exA for rel in rels]
    # ------------------------------------------------------------ populations of the design's sides (rule A, unsampled)
    pools, sides_out = {}, {}
    for j in canon(list(sizes), STRATA, W):
        st, r, sd = parse_side(j, STRATA, W)
        si, sv = STRATA.index(st), SIDES.index(sd)
        allA, P = 0, []
        for k, (rel, rr) in enumerate(L):
            if not inA[k] or rr != r:
                continue
            hit = np.where((C[k]["code"] == si) & (C[k]["side"] == sv))[0]
            allA += len(hit)
            P += [(k, int(p)) for p in hit if (k, int(p)) not in d1_set]
        nd1 = allA - len(P)
        exp = design["sides"][j]
        assert int(exp["N"]) == allA and int(exp["n_d1"]) == nd1, \
            f"{j}: census N = {allA}, D1's n = {nd1}; the design says {exp['N']} and {exp['n_d1']}"
        assert sizes[j] <= len(P), f"{j}: {sizes[j]} asked, {len(P)} unsampled"
        pools[j] = P
        sides_out[j] = {"N_rule_A": allA, "n_d1": nd1, "unsampled": len(P), "n": sizes[j]}
    # ------------------------------------------------------------ the draw and the processing order
    rows, q, key = draw(pools, sizes, seed, STRATA, W)
    n9 = len(rows)
    seen = set()
    for j, i, k, p in rows:
        st, r, sd = parse_side(j, STRATA, W)
        assert inA[k] and L[k][1] == r, f"{j}: drawn outside rule A's leaves of its round count"
        assert int(C[k]["code"][p]) == STRATA.index(st) and int(C[k]["side"][p]) == SIDES.index(sd), f"{j}: off its cell or side"
        assert (k, p) not in d1_set and (k, p) not in seen, f"{j}: a shot of D1's sample, or drawn twice"
        seen.add((k, p))
    rows_q = [rows[int(t)] for t in q]
    key_q = key[q]
    # ------------------------------------------------------------ detection events (checked against shots.npz)
    det9 = [None] * n9
    by_leaf = {}
    for o_, (j, i, k, p) in enumerate(rows_q):
        by_leaf.setdefault(k, []).append((o_, p))
    leaves_meta = [dict(x) for x in m1["leaves"]]
    for k in sorted(by_leaf):
        os_, ps = zip(*by_leaf[k])
        idx = C[k]["idx"][list(ps)]
        det, ob, dem_rel, dem_md5 = RS.dets_for(a_.root, L[k][0], C[k]["s"], idx)
        assert np.array_equal(ob, C[k]["obs"][list(ps)]), f"{L[k][0]}: observed flips differ from shots.npz"
        for t, oo in enumerate(os_):
            det9[oo] = det[t]
        if leaves_meta[k].get("dem") is None:
            leaves_meta[k]["dem"], leaves_meta[k]["dem_md5"] = dem_rel, dem_md5
        assert (leaves_meta[k]["dem"], leaves_meta[k]["dem_md5"]) == (dem_rel, dem_md5), f"{L[k][0]}: DEM differs from D1's"
    # ------------------------------------------------------------ the combined selection
    w1 = D1["det"].shape[1]
    nd9 = np.array([r_.shape[0] for r_ in det9], np.int64)
    wid = max(w1, int(nd9.max()) if n9 else 0)
    det = np.zeros((n1 + n9, wid), np.uint8)
    det[:n1, :w1] = D1["det"]
    for t, r_ in enumerate(det9):
        det[n1 + t, :r_.shape[0]] = r_
    new = {"leaf_id": [k for _, _, k, _ in rows_q], "pos": [p for _, _, _, p in rows_q],
           "idx": [int(C[k]["idx"][p]) for _, _, k, p in rows_q],
           "rounds": [L[k][1] for _, _, k, _ in rows_q],
           "stratum": [STRATA.index(parse_side(j, STRATA, W)[0]) for j, _, _, _ in rows_q],
           "order": [n1 + o_ for o_ in range(n9)], "worker": [wb + (o_ % nw) for o_ in range(n9)]}
    arr = {"det": det, "ndet": np.concatenate([D1["ndet"].astype(np.int64), nd9])}
    for f, v in new.items():
        arr[f] = np.concatenate([D1[f].astype(np.int64), np.array(v, np.int64)])
    arr["part"] = np.concatenate([np.zeros(n1, np.int64), np.ones(n9, np.int64)])
    side1 = [int(C[int(k)]["side"][int(p)]) for k, p in zip(D1["leaf_id"], D1["pos"])]
    arr["side_a"] = np.array(side1 + [SIDES.index(parse_side(j, STRATA, W)[2]) for j, _, _, _ in rows_q], np.int64)
    arr["key"] = np.concatenate([np.full(n1, np.nan), key_q])
    os.makedirs(a_.out, exist_ok=True)
    zp = os.path.join(a_.out, "sample.npz")
    np.savez_compressed(zp, **arr)
    applied = {"design": design, "design_md5": inputs["design"], "order_base": n1,
               "workers_d1_9": list(range(wb, wb + nw)), "rule_A_excluded": exA, "sides": sides_out, "n_d1_9": n9}
    json.dump(applied, open(os.path.join(a_.out, "design.json"), "w"), indent=1)
    cells = {}
    for c in m1["cells"]:
        st, r = c.split("|")
        k9 = sum(1 for j, _, _, _ in rows_q if parse_side(j, STRATA, W)[:2] == (st, int(r)))
        cells[c] = {"N": m1["cells"][c]["N"], "n": m1["cells"][c]["n"] + k9, "n_d1": m1["cells"][c]["n"], "n_d1_9": k9}
    meta = {"made": made, "script_md5": md5(os.path.abspath(__file__)),
            "what": "D1 and D1-9: D1's selection, every row unchanged, then D1-9's draw in processing order",
            "d1_selection": {"sample_json_md5": inputs["d1_sample.json"], "sample_npz_md5": m1["sample_npz_md5"],
                             "script_md5": m1.get("script_md5"), "n_total": n1},
            "design": design, "design_md5": inputs["design"], "strata": m1["strata"], "rule_A_excluded": exA,
            "cells": cells, "sides_d1_9": sides_out, "n_total": n1 + n9, "n_d1": n1, "n_d1_9": n9, "order_base": n1,
            "workers_d1_9": list(range(wb, wb + nw)), "leaves": leaves_meta, "inputs_md5": inputs,
            "sample_npz_md5": md5(zp), "seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(a_.out, "sample.json"), "w"), indent=1)
    print(json.dumps({"n_d1": n1, "n_d1_9": n9, "sides": len(sides_out), "rule_A_excluded": len(exA),
                      "sample_npz_md5": meta["sample_npz_md5"], "seconds": round(meta["seconds"], 1)}))


if __name__ == "__main__":
    main()
