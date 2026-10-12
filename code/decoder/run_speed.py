"""Speed and accuracy certificates of the approximate (sparse-frontier) ML decoder at d = 3, 5, 7, including
deliberately hard shots (posterior near 1/2) and long memories.

  python3 run_speed.py d3  --rounds 10,50 --out <dir>      sparse vs EXACT (FastExactDecoder) at several tau
  python3 run_speed.py ladder --d 5 --rounds 10 --n 2000 --hard 40 --random 40 --taus 1e-4,...,1e-9 --out <dir>
                                                          tau ladder on hard-enriched shots; reference = smallest tau
For every (tau, shot): posterior P(L=1|s), log P(s), decision, frontier size, wall time and the per-shot
dropped-mass estimate D (sum over pruning steps of the dropped importance-weighted fraction). The
certificate rule tested is: decision certified at tau if |2P - 1| > kappa * D (kappa = 2).
Noise: SI1000(p), p = 3.5e-3 (detection fraction close to Willow's), Z memory.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from dem_utils import parse_dem
from mismatch import si_family
from ml_sparse import SparseFrontierDecoder
from ml_exact_fast import FastExactDecoder


def circuit(d, r, p):
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r)
    return si_family(ideal, p, 5 * p, p / 10)


def sparse_run(dm, det, tau, kmax=2_000_000):
    D = SparseFrontierDecoder(dm, tau=tau, kmax=kmax)
    D.class_probs(det[:1])
    P = np.zeros(len(det)); L = np.zeros(len(det)); K = np.zeros(len(det)); T = np.zeros(len(det)); Dr = np.zeros(len(det))
    for i in range(len(det)):
        t0 = time.time()
        post, lp = D.class_probs(det[i:i + 1])
        T[i] = time.time() - t0
        P[i] = post[0, 1]; L[i] = lp[0]; K[i] = D.last_maxk[0]; Dr[i] = D.last_drop[0]
    return P, L, K, T, Dr


def summarize(P, L, K, T, Dr, Pref, Lref, r, kappa=2.0):
    ok = np.isfinite(P) & np.isfinite(Pref)
    dP = np.abs(P - Pref); dL = np.abs(L - Lref)
    dec = (P > 0.5) != (Pref > 0.5)
    cert = np.abs(2 * P - 1) > kappa * Dr
    return {"n": int(len(P)), "nan": int((~ok).sum()), "max_abs_dP": float(np.nanmax(dP)), "median_abs_dP": float(np.nanmedian(dP)),
            "max_abs_dlogP": float(np.nanmax(dL)), "decisions_differ": int(np.sum(dec & ok)),
            "ms_per_round_per_shot_mean": float(1e3 * T.mean() / r), "ms_per_round_per_shot_max": float(1e3 * T.max() / r),
            "mean_K": float(K.mean()), "max_K": float(K.max()),
            "drop_median": float(np.median(Dr)), "drop_max": float(Dr.max()),
            "frac_dP_le_drop": float(np.mean(dP[ok] <= Dr[ok] + 1e-15)),
            "frac_dlogP_le_drop": float(np.mean(dL[ok] <= 2 * Dr[ok] + 1e-12)),
            "certified_frac": float(np.mean(cert)), "certified_but_wrong": int(np.sum(cert & dec & ok))}


def mode_d3(a):
    res = {"mode": "d3_vs_exact", "p": a.p, "runs": []}
    for r in [int(x) for x in a.rounds.split(",")]:
        c = circuit(3, r, a.p)
        dm = parse_dem(c.detector_error_model(decompose_errors=False))
        n = a.n if r <= 10 else max(500, a.n // 4)
        det = c.compile_detector_sampler(seed=a.seed + r).sample(n).astype(np.uint8)
        E = FastExactDecoder(dm)
        t0 = time.time(); pe, le = E.class_probs(det); te = time.time() - t0
        Pref, Lref = pe[:, 1], le
        hard = np.argsort(np.abs(Pref - 0.5))[:a.hard]
        rnd = np.random.default_rng(a.seed).choice(n, size=min(a.random, n), replace=False)
        for tau in [float(x) for x in a.taus.split(",")]:
            for name, idx in (("random", rnd), ("hard", hard)):
                P, L, K, T, Dr = sparse_run(dm, det[idx], tau)
                s = summarize(P, L, K, T, Dr, Pref[idx], Lref[idx], r)
                s.update(rounds=r, tau=tau, subset=name, exact_ms_per_round_per_shot=1e3 * te / n / r,
                         hard_max_abs_P_minus_half=float(np.max(np.abs(Pref[hard] - 0.5))))
                res["runs"].append(s)
                print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
                json.dump(res, open(os.path.join(a.out, f"speed_d3_p{a.p:g}.json"), "w"), indent=1)


def mode_ladder(a):
    r = int(a.rounds)
    c = circuit(a.d, r, a.p)
    dm = parse_dem(c.detector_error_model(decompose_errors=False))
    det = c.compile_detector_sampler(seed=a.seed + 100 * a.d + r).sample(a.n).astype(np.uint8)
    taus = [float(x) for x in a.taus.split(",")]
    fn = os.path.join(a.out, f"ladder_d{a.d}_r{r}_p{a.p:g}.json")
    # screening pass at the loosest tau on all shots -> hard (closest to 1/2) + random subsets
    t0 = time.time()
    P0, L0, K0, T0, D0 = sparse_run(dm, det, taus[0])
    res = {"mode": "ladder", "d": a.d, "rounds": r, "p": a.p, "n_screen": a.n, "num_faults": dm.num_faults,
           "num_detectors": dm.num_detectors, "screen_tau": taus[0], "screen_seconds": time.time() - t0,
           "screen_ms_per_round_per_shot": float(1e3 * T0.mean() / r), "screen_certified_frac": float(np.mean(np.abs(2 * P0 - 1) > 2 * D0)),
           "screen_drop_median": float(np.median(D0)), "screen_LER_vs_obs": None, "taus": taus, "per_tau": {}}
    hard = np.argsort(np.abs(P0 - 0.5))[:a.hard]
    rest = np.setdiff1d(np.arange(a.n), hard)
    rnd = np.random.default_rng(a.seed).choice(rest, size=min(a.random, len(rest)), replace=False)
    idx = np.concatenate([hard, rnd])
    res["subset"] = {"hard": hard.tolist(), "random": rnd.tolist()}
    out = {}
    for tau in taus:
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            print(f"stopping before tau={tau}: time budget", flush=True); break
        out[tau] = sparse_run(dm, det[idx], tau)
        res["per_tau"][str(tau)] = {"P": out[tau][0].tolist(), "logP": out[tau][1].tolist(), "K": out[tau][2].tolist(),
                                    "seconds": out[tau][3].tolist(), "drop": out[tau][4].tolist()}
        json.dump(res, open(fn, "w"))
        T = out[tau][3]
        print(f"d={a.d} r={r} tau={tau:g}: hard {1e3 * T[:len(hard)].mean() / r:.1f} ms/round/shot (max {1e3 * T[:len(hard)].max() / r:.0f}), "
              f"random {1e3 * T[len(hard):].mean() / r:.1f} ms/round/shot, mean K {out[tau][2].mean():.0f}", flush=True)
    done = list(out.keys())
    ref = done[-1]
    Pref, Lref = out[ref][0], out[ref][1]
    res["summary"] = {}
    for tau in done[:-1]:
        P, L, K, T, Dr = out[tau]
        res["summary"][str(tau)] = {nm: summarize(P[sl], L[sl], K[sl], T[sl], Dr[sl], Pref[sl], Lref[sl], r)
                                    for nm, sl in (("hard", slice(0, len(hard))), ("random", slice(len(hard), None)))}
        print(f"tau={tau:g} vs {ref:g}: " + json.dumps({nm: {k: v for k, v in s.items() if k in ("max_abs_dP", "decisions_differ", "drop_max", "frac_dP_le_drop", "certified_frac", "certified_but_wrong")} for nm, s in res["summary"][str(tau)].items()}), flush=True)
    res["reference_tau"] = ref
    json.dump(res, open(fn, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["d3", "ladder"])
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--rounds", default="10")
    ap.add_argument("--p", type=float, default=3.5e-3)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--hard", type=int, default=40)
    ap.add_argument("--random", type=int, default=40)
    ap.add_argument("--taus", default="1e-4,1e-5,1e-6,1e-7,1e-8")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--max_seconds", type=float, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mode_d3(a) if a.mode == "d3" else mode_ladder(a)
