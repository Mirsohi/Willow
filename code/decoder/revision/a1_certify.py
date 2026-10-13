#!/usr/bin/env python3
"""A1 (fresh-eyes review of v2.0 draft 2), question 1: per-shot certificates at d = 5, from existing outputs only
(decoder, 2026-10-13). Post hoc; no decoding.

For each W5 stratum and round count it reports the share of shots whose decision is NOT certified at
kappa = 1, 2, 10, 100 and 1000, by two indicators:
  D  the sparse decoder's importance-weighted dropped-mass indicator: certified iff |2P - 1| > kappa * D (ml_sparse.py,
     AdaptiveSparseDecoder, which uses kappa = 2). D exists only where the non-band check re-decoded: tau2 on every
     selected shot, tau3 where it ran. Round 1's ladder did not store D.
  L  the ladder's own change: certified iff |2P - 1| > kappa * 2|P - P_prev|, with P_prev the next coarser rung
     (tau1 -> tau2 for the check's shots; tau2 -> tau3 for the band, from K2's extra rung).
The decision certified is the paper's: tau1 outside the band, tau2 in the band. Outside the band the D route
certifies it when tau2 is certified by D2 and agrees with tau1.
Strata (as in D1): B band; D discordant outside the band (the check's census); C concordant outside the band (the
check's sample of 2,500), also split at |llr_t1| = ln 9 into C_low and C_high.
Also reported: where tau3 ran, |P3 - P2| against D2 (does the indicator cover the next rung's change?); the margin to
spare, |2P2 - 1| / D2; and the ladder's measured CPU per shot and round at tau1 and tau2 (for costing a census).
usage: python3 -I -B a1_certify.py --res <results/decoder/willow> --check <supplementary/nonband_check>
           --k2 <supplementary/k2_extra_rung> --out OUT.json
"""
import argparse, hashlib, json, math, os, time
import numpy as np

DN = "ml_tau2_1e-6-1e-8_0.1"
LN9 = math.log(9.0)
KAPPAS = (1, 2, 10, 100, 1000)
W = (10, 30, 50)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sig(x):
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def share_unc(margin, err, kappas=KAPPAS):
    """margin = |2P - 1|, err = the indicator; uncertified iff not (margin > kappa * err); NaN counts as uncertified."""
    out = {}
    n = len(margin)
    for k in kappas:
        with np.errstate(invalid="ignore"):
            cert = margin > k * err
        u = int(n - np.sum(cert))
        out[str(k)] = {"uncertified": u, "share": u / n if n else None}
    return out


def q(v, qs=(0.0, 0.01, 0.1, 0.5)):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.quantile(v, x)) for x in qs] if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--check", "--k2", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(os.path.join(a.check, "shot_lists.json")))
    lp = os.path.join(a.check, "shot_lists.npz")
    assert md5(lp) == meta["shot_lists_npz_md5"]
    S = np.load(lp)
    leaves = [str(x) for x in S["leaves"]]
    n = len(S["idx"])
    strat, typ, rounds = S["stratum"], S["type"].astype(str), S["rounds"]
    ml = S["ml"].astype(np.int64); llr1 = S["llr_t1"].astype(np.float64)
    l2 = np.full(n, np.nan); d2 = np.full(n, np.nan); p2d = np.full(n, -1)
    l3 = np.full(n, np.nan); d3 = np.full(n, np.nan); p3d = np.full(n, -1); done3 = np.zeros(n, bool)
    reason = np.zeros(n, np.int64)
    inputs = {"shot_lists.npz": md5(lp)}
    for k in range(len(leaves)):
        p = os.path.join(a.check, "leaves", f"{k:02d}.npz")
        inputs[f"leaves/{k:02d}.npz"] = md5(p)
        z = np.load(p); r = z["row"]
        assert z["done_t2"].all()
        l2[r] = z["llr_t2"]; d2[r] = z["drop_t2"]; p2d[r] = z["pred_t2"]
        l3[r] = np.where(z["done_t3"], z["llr_t3"], np.nan); d3[r] = np.where(z["done_t3"], z["drop_t3"], np.nan)
        p3d[r] = np.where(z["done_t3"], z["pred_t3"].astype(np.int64), -1); done3[r] = z["done_t3"]; reason[r] = z["t3_reason"]
    P1, P2, P3 = sig(llr1), sig(l2), sig(l3)
    m1, m2 = np.abs(2 * P1 - 1), np.abs(2 * P2 - 1)
    sub = {"D": strat == 0, "C": strat == 1, "C_low": (strat == 1) & (np.abs(llr1) < LN9),
           "C_high": (strat == 1) & ~(np.abs(llr1) < LN9)}
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "A1 question 1, post hoc, existing outputs only", "kappas": list(KAPPAS), "strata": {}}
    for nm, msk in sub.items():
        for r in ("all",) + W:
            mm = msk & ((rounds == r) if r != "all" else True)
            if not mm.any():
                continue
            e = {"n": int(mm.sum()), "flips_tau1_to_tau2": int((p2d[mm] != ml[mm]).sum()),
                 "flips_tau1_to_final": int((np.where(done3, p3d, p2d)[mm] != ml[mm]).sum()),
                 # D route: tau2 certified by D2 and equal to tau1 (the paper's decision)
                 "paper_decision_by_D2": share_unc(np.where(p2d[mm] == ml[mm], m2[mm], -1.0), d2[mm]),
                 # tau2's own decision by D2 (whatever tau1 said)
                 "tau2_decision_by_D2": share_unc(m2[mm], d2[mm]),
                 # L route: tau1 by the tau1 -> tau2 change
                 "paper_decision_by_L12": share_unc(m1[mm], 2 * np.abs(P2[mm] - P1[mm])),
                 # both indicators
                 "paper_decision_by_D2_and_L12": {str(k): None for k in KAPPAS},
                 "margin_over_D2_q0_q01_q10_q50": q(m2[mm] / d2[mm]),
                 "D2_q0_q50_q90_q99_q100": q(d2[mm], (0.0, 0.5, 0.9, 0.99, 1.0))}
            for k in KAPPAS:
                with np.errstate(invalid="ignore"):
                    c = (p2d[mm] == ml[mm]) & (m2[mm] > k * d2[mm]) & (m1[mm] > k * 2 * np.abs(P2[mm] - P1[mm]))
                u = int(mm.sum() - c.sum())
                e["paper_decision_by_D2_and_L12"][str(k)] = {"uncertified": u, "share": u / int(mm.sum())}
            # where tau3 ran: does D2 cover the tau2 -> tau3 change?
            t3 = mm & done3
            if t3.any():
                rho = np.abs(P3[t3] - P2[t3]) / d2[t3]
                e["tau3"] = {"n": int(t3.sum()), "random_subsample": int((t3 & (reason == 2)).sum()),
                             "decision_changed_tau2_to_tau3": int((p3d[t3] != p2d[t3]).sum()),
                             "abs_P3_minus_P2_over_D2": {"max": float(np.nanmax(rho)), "q50_q90_q99": q(rho, (0.5, 0.9, 0.99)),
                                                         "share_above_1": float(np.mean(rho > 1)),
                                                         "share_above_0.5": float(np.mean(rho > 0.5))},
                             "tau2_decision_by_L23": share_unc(m2[t3], 2 * np.abs(P3[t3] - P2[t3])),
                             "tau3_decision_by_D3": share_unc(np.abs(2 * P3[t3] - 1), d3[t3])}
            out["strata"].setdefault(nm, {})[str(r)] = e
    # ---------------------------------------------------------------- band: tau2 (the paper's) against K2's tau3
    band = {}
    cpu = {}
    for rel in leaves:
        r = int(rel.rsplit("/r", 1)[1])
        d = os.path.join(a.res, "decode", rel, "rl_s5")
        z = np.load(os.path.join(d, DN + ".npz"))
        inputs[f"{rel}/{DN}.npz"] = md5(os.path.join(d, DN + ".npz"))
        red = z["redecoded"].astype(bool); idx = z["idx"].astype(np.int64)
        k2p = os.path.join(a.k2, rel, "rl_s5", "extra_rung.npz")
        inputs[f"k2/{rel}/extra_rung.npz"] = md5(k2p)
        e = np.load(k2p); t3 = dict(zip(e["idx"].astype(np.int64).tolist(), e["llr_t3"].astype(np.float64).tolist()))
        bi = idx[red]
        have = np.array([i in t3 for i in bi.tolist()], bool)
        P2b = sig(z["llr"][red][have]); P3b = sig(np.array([t3[i] for i in bi[have].tolist()]))
        b = band.setdefault(r, {"m2": [], "err": [], "chg": 0, "n": 0, "without_tau3": 0})
        b["without_tau3"] += int((~have).sum())
        b["m2"].append(np.abs(2 * P2b - 1)); b["err"].append(2 * np.abs(P3b - P2b))
        b["chg"] += int(((P3b > 0.5) != (P2b > 0.5)).sum()); b["n"] += int(have.sum())
        c = cpu.setdefault(r, {"sec_t1": 0.0, "shots": 0, "sec_t2": 0.0, "band": 0})
        c["sec_t1"] += float(np.sum(z["sec_t1"])); c["shots"] += int(len(idx))
        c["sec_t2"] += float(np.sum(z["sec_t2"])); c["band"] += int(red.sum())
    out["strata"]["B"] = {}
    allm, alle = [], []
    for r in W:
        m2b = np.concatenate(band[r]["m2"]); eb = np.concatenate(band[r]["err"])
        allm.append(m2b); alle.append(eb)
        out["strata"]["B"][str(r)] = {"n": band[r]["n"], "without_tau3": band[r]["without_tau3"],
                                      "decision_changed_tau2_to_tau3": band[r]["chg"],
                                      "paper_decision_by_L23": share_unc(m2b, eb)}
    out["strata"]["B"]["all"] = {"n": sum(band[r]["n"] for r in W),
                                 "decision_changed_tau2_to_tau3": sum(band[r]["chg"] for r in W),
                                 "paper_decision_by_L23": share_unc(np.concatenate(allm), np.concatenate(alle))}
    out["ladder_cpu"] = {str(r): {"tau1_s_per_shot": c["sec_t1"] / c["shots"],
                                  "tau1_s_per_shot_round": c["sec_t1"] / c["shots"] / r,
                                  "tau1_total_core_hours": c["sec_t1"] / 3600, "shots": c["shots"],
                                  "tau2_s_per_band_shot": c["sec_t2"] / c["band"] if c["band"] else None,
                                  "band_shots": c["band"]} for r, c in sorted(cpu.items())}
    out["inputs_md5"] = inputs
    out["seconds"] = time.time() - t0
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"seconds": out["seconds"], "ladder_cpu": out["ladder_cpu"]}, indent=1))


if __name__ == "__main__":
    main()
