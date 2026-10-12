#!/usr/bin/env python3
"""Post hoc robustness check, prompted by the internal review: are ML(RL)'s decisions outside the ladder's band stable
at d = 5? (notes/decoder/willow_prereg.md section 9 entry 13). Step 1 of 3: the shot lists, fixed by a seed before any
decoding.

The 24 d = 5 leaves of P1's window (r = 10, 30, 50; 4 patches, X and Z), round 1's ladder outputs (decode/<leaf>/rl_s5:
every 5th shot, 10,000 per leaf). A shot is in the band if the ladder re-decoded it at tau2 (|P_tau1 - 1/2| < 0.1 or a
NaN posterior); K2 covers those. Outside the band ML(RL)'s decision is tau1's. Strata, outside the band:
  D  Libra's shipped decision differs from ML(RL)'s: every shot (census);
  C  they agree: a random sample of --n-c shots, allocated to the leaves in proportion to their C counts (largest
     remainder), drawn without replacement within each leaf.
Every selected shot also gets a pre-drawn u ~ U(0, 1): tau3 runs on a shot that tau2 did not move if u < 0.1. C's
sample is split into two halves (pass A first) by a pre-drawn random order. Shots are ordered by leaf (plan order),
then stratum (D first), then index.

Output: <out>/shot_lists.npz (one row per selected shot) and <out>/shot_lists.json (seed, counts, population sizes by
leaf and type, the md5 of every input file, this script's md5).
  python3 -I -B nonband_select.py --res results/decoder/willow --out results/decoder/willow/supplementary/nonband_check
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
import numpy as np

SEED = 20261010
DEC = "ml_tau2:1e-6/1e-8:0.1"
DN = DEC.replace(":", "_").replace("/", "-")
LIBRA = "libra_decoder_with_rl_optimized_prior"


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def leaves(plan):
    out = [j for j in plan["jobs"] if j.get("kind") == "decode" and j.get("d") == 5 and j["rounds"] in (10, 30, 50)
           and DEC in j["decoders"].split(",")]
    return sorted(out, key=lambda j: (j["patch"], j["basis"], j["rounds"]))


def load(res, j):
    d = os.path.join(res, "decode", j["leaf"], j["dem_tag"])
    s = json.load(open(os.path.join(d, "summary.json")))
    assert s.get("status") == "complete", d
    z = np.load(os.path.join(d, DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
    assert np.array_equal(z["idx"], sh["idx"]), f"{d}: ladder and shots select different shots"
    rows = [i for i, x in enumerate(sh["shipped_names"]) if LIBRA in str(x)]
    assert len(rows) == 1, f"{d}: {len(rows)} Libra (RL) rows"
    L = {"idx": z["idx"].astype(np.int64), "ml": z["pred"][:, 0].astype(np.uint8), "red": z["redecoded"].astype(bool),
         "llr": z["llr"], "llr_t1": z["llr_t1"], "maxk_t1": z["maxk_t1"].astype(np.int64),
         "obs": sh["obs"].astype(np.uint8), "libra": sh["shipped"][rows[0]].astype(np.uint8)}
    nb = ~L["red"]
    # outside the band the ladder's final decision is tau1's
    assert np.array_equal(L["llr"][nb], L["llr_t1"][nb], equal_nan=True), f"{d}: final llr != tau1 llr outside the band"
    assert not np.isnan(L["llr_t1"][nb]).any(), f"{d}: a NaN tau1 posterior outside the band"
    files = {os.path.relpath(os.path.join(d, f), res): md5_file(os.path.join(d, f))
             for f in (DN + ".npz", "shots.npz", "summary.json")}
    return L, s, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-c", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    plan = json.load(open(os.path.join(a.res, "plan.json")))
    J = leaves(plan)
    assert len(J) == 24, f"{len(J)} leaves, not 24"
    rng = np.random.default_rng(a.seed)
    data, info, files = [], [], {"plan.json": md5_file(os.path.join(a.res, "plan.json"))}
    for j in J:
        L, s, f = load(a.res, j)
        files.update(f)
        data.append(L)
    NC = np.array([int(((L["libra"] == L["ml"]) & ~L["red"]).sum()) for L in data])
    q = a.n_c * NC / NC.sum()
    alloc = np.floor(q).astype(int)
    for i in np.argsort(-(q - alloc), kind="stable")[: a.n_c - alloc.sum()]:
        alloc[i] += 1
    assert alloc.sum() == a.n_c
    cols = {k: [] for k in ("leaf_id", "pos", "idx", "stratum", "type", "u", "c_pass", "llr_t1", "ml", "libra", "obs",
                            "maxk_t1", "rounds")}
    for li, (j, L) in enumerate(zip(J, data)):
        nb = ~L["red"]
        disc = L["libra"] != L["ml"]
        posD = np.where(disc & nb)[0]
        posCall = np.where(~disc & nb)[0]
        posC = np.sort(rng.choice(posCall, size=alloc[li], replace=False))
        pas = np.zeros(len(posC), np.int8)
        pas[rng.permutation(len(posC))[len(posC) // 2:]] = 1          # pass A (0) and pass B (1), half each
        pos = np.concatenate([posD, posC]); strat = np.r_[np.zeros(len(posD), np.int8), np.ones(len(posC), np.int8)]
        u = rng.random(len(pos))
        ml, lib, obs = L["ml"][pos], L["libra"][pos], L["obs"][pos]
        # type: D10 Libra wrong & ML right, D01 Libra right & ML wrong, C00 both right, C11 both wrong
        typ = np.where(strat == 0, np.where(ml == obs, "D10", "D01"), np.where(ml == obs, "C00", "C11"))
        for k, v in (("leaf_id", np.full(len(pos), li)), ("pos", pos), ("idx", L["idx"][pos]), ("stratum", strat),
                     ("type", typ), ("u", u), ("c_pass", np.r_[np.full(len(posD), -1, np.int8), pas]),
                     ("llr_t1", L["llr_t1"][pos]), ("ml", ml), ("libra", lib), ("obs", obs), ("maxk_t1", L["maxk_t1"][pos]),
                     ("rounds", np.full(len(pos), j["rounds"]))):
            cols[k].append(v)
        pop = {}
        for t, m in (("D10", disc & nb & (L["ml"] == L["obs"])), ("D01", disc & nb & (L["ml"] != L["obs"])),
                     ("C00", ~disc & nb & (L["ml"] == L["obs"])), ("C11", ~disc & nb & (L["ml"] != L["obs"]))):
            pop[t] = int(m.sum())
        info.append({"leaf": j["leaf"], "tag": j["dem_tag"], "patch": j["patch"], "basis": j["basis"], "rounds": j["rounds"],
                     "n": int(len(L["ml"])), "band": int(L["red"].sum()), "population": pop,
                     "selected": {"D": int(len(posD)), "C": int(len(posC))},
                     "errors_ML": int((L["ml"] != L["obs"]).sum()), "errors_libra": int((L["libra"] != L["obs"]).sum())})
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    arr["type"] = arr["type"].astype("U3")
    arr["leaves"] = np.array([j["leaf"] for j in J])
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "shot_lists.tmp.npz"), **arr)
    os.replace(os.path.join(a.out, "shot_lists.tmp.npz"), os.path.join(a.out, "shot_lists.npz"))
    tot = lambda k: int(sum(x["population"][k] for x in info))
    meta = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": a.seed, "n_c": a.n_c,
            "decoder": DEC, "label": "post hoc robustness check, prompted by the internal review",
            "prereg": "notes/decoder/willow_prereg.md section 9 entry 13",
            "selected": {"D": int((arr["stratum"] == 0).sum()), "C": int((arr["stratum"] == 1).sum()),
                         "C_pass_A": int((arr["c_pass"] == 0).sum()), "C_pass_B": int((arr["c_pass"] == 1).sum()),
                         "u_below_0.1": int((arr["u"] < 0.1).sum())},
            "population": {k: tot(k) for k in ("D10", "D01", "C00", "C11")},
            "errors_ML": int(sum(x["errors_ML"] for x in info)), "errors_libra": int(sum(x["errors_libra"] for x in info)),
            "leaves": info, "inputs_md5": files, "script_md5": md5_file(os.path.abspath(__file__)),
            "shot_lists_npz_md5": md5_file(os.path.join(a.out, "shot_lists.npz"))}
    with open(os.path.join(a.out, "shot_lists.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(a.out, "shot_lists.json.tmp"), os.path.join(a.out, "shot_lists.json"))
    print(json.dumps({k: meta[k] for k in ("selected", "population", "errors_ML", "errors_libra", "shot_lists_npz_md5")}))


if __name__ == "__main__":
    main()
