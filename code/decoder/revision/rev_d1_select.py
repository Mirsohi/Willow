#!/usr/bin/env python3
"""D1 (review 1.3), step 1: shot selection for the independent check of the d = 5 ML(RL) reference (decoder, 2026-10-13).
Post hoc revision analysis. Reads round 1's ladder outputs (decode/<leaf>/rl_s5: ladder npz, shots.npz, summary.json)
and the local extract (archive data: run with python3 -I, from a folder outside it). Writes the selected shots' detection
events, so that the decoding steps never read the archive or any stored decision.

Leaves: the 24 d = 5 leaves at r in {10, 30, 50} (4 patches, X and Z), every 5th shot (selection 0:50000:5).
Strata per shot (from the stored ladder, before any new decoding):
  B       the band: re-decoded at tau2 (|P_tau1 - 1/2| < 0.1, or a NaN posterior)
  D       outside the band, Libra's shipped decision differs from ML(RL)'s
  C_low   outside the band, they agree, |llr_tau1| < ln 9 (|P - 1/2| < 0.4)
  C_high  outside the band, they agree, |llr_tau1| >= ln 9 (confident)
Cells are stratum x round count, pooled over the 8 leaves of that round count.

  probe   shots OUTSIDE the population (shot index = 2 mod 5, never decoded by ML), --per-round per round count from one
          leaf drawn at random, for timing only:  probe --root E --res R --out DIR [--per-round 2] [--seed S]
  sample  a simple random sample without replacement of n shots in each cell (n from the design JSON), in a processing
          order of waves (the w-th shot of every cell, cells shuffled within each wave), each shot assigned to worker
          order mod W:  sample --root E --res R --design DESIGN.json --out DIR
Outputs: <out>/probe.npz + probe.json, or <out>/sample.npz + sample.json (detection events padded to the largest detector
count, with ndet per row; population counts per cell; md5 of every input; this script's md5).
"""
import argparse, glob, hashlib, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)                                       # code/decoder
import numpy as np

ARCH = "google_105Q_surface_code_d3_d5_d7"
DN = "ml_tau2_1e-6-1e-8_0.1"
LIBRA = "libra_decoder_with_rl_optimized_prior"
W = (10, 30, 50)
STRATA = ("B", "D", "C_low", "C_high")
LN9 = math.log(9.0)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def leaves(res):
    out = []
    for d in sorted(glob.glob(os.path.join(res, "decode", ARCH, "d5_at_*", "[XZ]", "r*", "rl_s5"))):
        rel = os.path.relpath(os.path.dirname(d), os.path.join(res, "decode"))
        r = int(rel.split("/r")[-1])
        if r in W:
            out.append((rel, r))
    assert len(out) == 24, f"{len(out)} leaves"
    return out


def load_leaf(res, rel):
    d = os.path.join(res, "decode", rel, "rl_s5")
    s = json.load(open(os.path.join(d, "summary.json")))
    assert s["status"] == "complete" and s["selection"] == "0:50000:5", rel
    z = np.load(os.path.join(d, DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
    assert np.array_equal(z["idx"], sh["idx"])
    row = [i for i, x in enumerate(sh["shipped_names"]) if LIBRA in str(x)]
    assert len(row) == 1
    files = {f: md5(os.path.join(d, f)) for f in (DN + ".npz", "shots.npz", "summary.json")}
    return s, z, sh, row[0], files


def strata_of(z, sh, lrow):
    band = z["redecoded"].astype(bool)
    ml = z["pred"][:, 0].astype(np.int64); lib = sh["shipped"][lrow].astype(np.int64)
    l1 = np.abs(z["llr_t1"].astype(np.float64))
    disc = lib != ml
    code = np.full(len(ml), -1, np.int64)
    code[band] = 0
    code[~band & disc] = 1
    code[~band & ~disc & (l1 < LN9)] = 2
    code[~band & ~disc & (l1 >= LN9)] = 3
    assert (code >= 0).all()
    return code


def dets_for(root, rel, s, idx):
    sys.path.insert(0, CODE)
    from willow_data import Experiment
    folder = os.path.join(root, rel)
    dem_path = os.path.join(folder, s["job"]["dem_file"])
    assert md5(dem_path) == s["dem_md5"], f"{rel}: DEM md5 differs from summary.json"
    exp = Experiment(folder)
    det = exp.detection_events(); obs = exp.obs_flips_actual()
    out = np.ascontiguousarray(det[idx]); o = obs[idx, 0].astype(np.uint8)
    del det, obs
    return out, o, os.path.relpath(dem_path, root), s["dem_md5"]


def pad(rows):
    m = max(r.shape[0] for r in rows)
    out = np.zeros((len(rows), m), np.uint8)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r
    return out, np.array([r.shape[0] for r in rows], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("probe", "sample"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--design")
    ap.add_argument("--per-round", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()
    t0 = time.time()
    L = leaves(a.res)
    os.makedirs(a.out, exist_ok=True)
    made = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.mode == "probe":
        rng = np.random.default_rng(a.seed + 1)
        rows, meta = [], {"made": made, "script_md5": md5(os.path.abspath(__file__)), "seed": a.seed + 1,
                          "what": "timing probe: shots outside the population (index = 2 mod 5), never decoded by ML",
                          "leaves": [], "shots": []}
        for r in W:
            cand = [rel for rel, rr in L if rr == r]
            rel = cand[int(rng.integers(len(cand)))]
            s, z, sh, lrow, files = load_leaf(a.res, rel)
            idx = np.sort(rng.choice(np.arange(2, 50000, 5), size=a.per_round, replace=False))
            assert not np.isin(idx, z["idx"]).any()
            det, _, dem_rel, dem_md5 = dets_for(a.root, rel, s, idx)
            k = len(meta["leaves"])
            meta["leaves"].append({"leaf": rel, "rounds": r, "dem": dem_rel, "dem_md5": dem_md5})
            for i, x in enumerate(idx):
                rows.append(det[i]); meta["shots"].append({"leaf_id": k, "idx": int(x), "rounds": r})
        dp, nd = pad(rows)
        np.savez_compressed(os.path.join(a.out, "probe.npz"), det=dp, ndet=nd,
                            leaf_id=np.array([x["leaf_id"] for x in meta["shots"]]),
                            idx=np.array([x["idx"] for x in meta["shots"]]), rounds=np.array([x["rounds"] for x in meta["shots"]]))
        meta["probe_npz_md5"] = md5(os.path.join(a.out, "probe.npz")); meta["seconds"] = time.time() - t0
        json.dump(meta, open(os.path.join(a.out, "probe.json"), "w"), indent=1)
        print(json.dumps({k: meta[k] for k in ("leaves", "probe_npz_md5")}))
        return
    design = json.load(open(a.design))
    rng = np.random.default_rng(design["seed"])
    cells = {}                      # (stratum, r) -> list of (leaf_id, pos)
    info, pop = [], {}
    cache = {}
    for k, (rel, r) in enumerate(L):
        s, z, sh, lrow, files = load_leaf(a.res, rel)
        code = strata_of(z, sh, lrow)
        info.append({"leaf": rel, "rounds": r, "files_md5": files, "n": int(len(code)),
                     "strata": {st: int((code == i).sum()) for i, st in enumerate(STRATA)}})
        for i, st in enumerate(STRATA):
            for pos in np.where(code == i)[0]:
                cells.setdefault((st, r), []).append((k, int(pos)))
        cache[k] = (s, z["idx"].astype(np.int64), sh["obs"].astype(np.uint8))
    picked = {}
    for st in STRATA:
        for r in W:
            lst = cells.get((st, r), [])
            n = int(design["n"][st][str(r)])
            perm = rng.permutation(len(lst))[:n]
            picked[(st, r)] = [lst[i] for i in perm]
            pop[f"{st}|{r}"] = {"N": len(lst), "n": len(picked[(st, r)])}
    order = []
    keys = [(st, r) for st in STRATA for r in W]
    wmax = max(len(v) for v in picked.values())
    for w in range(wmax):
        for j in rng.permutation(len(keys)):
            st, r = keys[j]
            if w < len(picked[(st, r)]):
                order.append((st, r) + picked[(st, r)][w])
    nw = int(design.get("workers", 2))
    rows, obs_check, cols = [], [], {k: [] for k in ("leaf_id", "pos", "idx", "rounds", "stratum", "order", "worker")}
    by_leaf = {}
    for o, (st, r, k, pos) in enumerate(order):
        by_leaf.setdefault(k, []).append((o, pos))
    det_rows = [None] * len(order)
    leaf_meta = {}
    for k in sorted(by_leaf):
        s, idx_all, obs_all = cache[k]
        os_, poss = zip(*by_leaf[k])
        idx = idx_all[list(poss)]
        det, o, dem_rel, dem_md5 = dets_for(a.root, L[k][0], s, idx)
        assert np.array_equal(o, obs_all[list(poss)]), f"{L[k][0]}: observed flips differ from shots.npz"
        for j, oo in enumerate(os_):
            det_rows[oo] = det[j]
        leaf_meta[k] = {"dem": dem_rel, "dem_md5": dem_md5}
    for o, (st, r, k, pos) in enumerate(order):
        cols["leaf_id"].append(k); cols["pos"].append(pos); cols["idx"].append(int(cache[k][1][pos]))
        cols["rounds"].append(r); cols["stratum"].append(STRATA.index(st)); cols["order"].append(o); cols["worker"].append(o % nw)
    dp, nd = pad(det_rows)
    arr = {k: np.array(v, np.int64) for k, v in cols.items()}
    np.savez_compressed(os.path.join(a.out, "sample.npz"), det=dp, ndet=nd, **arr)
    for k in range(len(L)):
        info[k].update(leaf_meta.get(k, {"dem": None, "dem_md5": None}))
    meta = {"made": made, "script_md5": md5(os.path.abspath(__file__)), "design": design, "design_md5": md5(a.design),
            "strata": {"B": "band", "D": "non-band, Libra != ML", "C_low": "non-band, agree, |llr_t1| < ln 9",
                       "C_high": "non-band, agree, |llr_t1| >= ln 9"},
            "cells": pop, "n_total": len(order), "leaves": info, "sample_npz_md5": md5(os.path.join(a.out, "sample.npz")),
            "seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(a.out, "sample.json"), "w"), indent=1)
    print(json.dumps({"n_total": len(order), "cells": pop, "sample_npz_md5": meta["sample_npz_md5"]}))


if __name__ == "__main__":
    main()
