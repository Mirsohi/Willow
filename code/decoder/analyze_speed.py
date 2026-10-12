"""Offline analysis of the tau ladders written by run_speed.py (results/decoder/speed/ladder_*.json):
per tau, cost and accuracy against the smallest-tau reference on hard and random shots, and the adaptive
per-shot ladder (AdaptiveSparseDecoder rule: move to the next tau while |2P - 1| <= kappa * D) replayed on
the stored per-shot results: final-decision errors vs the reference and mean cost.
Usage: python3 analyze_speed.py <results/decoder/speed> -> prints a table and writes speed_summary.json there.
"""
import json, os, sys
import numpy as np


def replay(P, D, T, kappa):
    """P, D, T: arrays (n_taus, n_shots). Returns final level per shot, final P, cost per shot."""
    nt, ns = P.shape
    level = np.zeros(ns, int); cost = T[0].copy(); fin = P[0].copy()
    for s in range(ns):
        k = 0
        while k < nt - 1 and not (abs(2 * P[k, s] - 1) > kappa * D[k, s]):
            k += 1
            cost[s] += T[k, s]
        level[s] = k; fin[s] = P[k, s]
    return level, fin, cost


def main(sd):
    out = {}
    for f in sorted(os.listdir(sd)):
        if not (f.startswith("ladder_") and f.endswith(".json")):
            continue
        L = json.load(open(os.path.join(sd, f)))
        d, r = L["d"], L["rounds"]
        taus = [t for t in L["per_tau"]]
        P = np.array([L["per_tau"][t]["P"] for t in taus]); D = np.array([L["per_tau"][t]["drop"] for t in taus])
        T = np.array([L["per_tau"][t]["seconds"] for t in taus]); K = np.array([L["per_tau"][t]["K"] for t in taus])
        nh = len(L["subset"]["hard"])
        ref = P[-1]
        res = {"d": d, "rounds": r, "taus": [float(t) for t in taus], "reference_tau": float(taus[-1]),
               "screen_ms_per_round_per_shot": L.get("screen_ms_per_round_per_shot"), "n_screen": L.get("n_screen"),
               "screen_certified_frac": L.get("screen_certified_frac"), "uniform": {}, "adaptive": {}}
        for nm, sl in (("hard", slice(0, nh)), ("random", slice(nh, None))):
            u = []
            for k, t in enumerate(taus):
                dP = np.abs(P[k, sl] - ref[sl])
                u.append({"tau": float(t), "ms_per_round_per_shot": float(1e3 * T[k, sl].mean() / r),
                          "max_ms_per_round_per_shot": float(1e3 * T[k, sl].max() / r), "mean_K": float(K[k, sl].mean()),
                          "max_abs_dP": float(dP.max()), "decisions_differ": int(np.sum((P[k, sl] > 0.5) != (ref[sl] > 0.5))),
                          "drop_max": float(D[k, sl].max()), "frac_dP_le_drop": float(np.mean(dP <= D[k, sl] + 1e-15))})
            res["uniform"][nm] = u
            for kappa in (1.0, 2.0, 4.0):
                lev, fin, cost = replay(P[:-1, sl], D[:-1, sl], T[:-1, sl], kappa) if len(taus) > 1 else (None, None, None)
                if lev is None:
                    continue
                res["adaptive"].setdefault(nm, {})[str(kappa)] = {
                    "levels": np.bincount(lev, minlength=len(taus) - 1).tolist(),
                    "decisions_differ_vs_ref": int(np.sum((fin > 0.5) != (ref[sl] > 0.5))),
                    "max_abs_dP": float(np.max(np.abs(fin - ref[sl]))),
                    "ms_per_round_per_shot": float(1e3 * cost.mean() / r)}
        out[f] = res
        print(f"\n{f}: d={d} r={r} reference tau={taus[-1]}")
        for nm in ("hard", "random"):
            for u in res["uniform"][nm]:
                print(f"  {nm:6s} tau={u['tau']:.0e}: {u['ms_per_round_per_shot']:8.2f} ms/round/shot (max {u['max_ms_per_round_per_shot']:.0f}), "
                      f"K {u['mean_K']:.0f}, max|dP| {u['max_abs_dP']:.1e}, decisions differ {u['decisions_differ']}, max D {u['drop_max']:.1e}, "
                      f"frac |dP|<=D {u['frac_dP_le_drop']:.2f}")
            for kappa, a in res["adaptive"].get(nm, {}).items():
                print(f"  {nm:6s} adaptive kappa={kappa}: levels {a['levels']}, differ {a['decisions_differ_vs_ref']}, "
                      f"max|dP| {a['max_abs_dP']:.1e}, {a['ms_per_round_per_shot']:.2f} ms/round/shot")
    json.dump(out, open(os.path.join(sd, "speed_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
