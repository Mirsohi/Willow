"""EXPLORATORY baseline for round 2's F1: how much of D-hat's held-out calibration and log-loss gain does a single
temperature on ML(RL)'s posteriors give? (Not pre-registered; part of check 5 of main's brief, 2026-10-10.)

reservoir, 2026-10-10. Per series, one factor beta multiplies ML(RL)'s llr: P_beta(L = 1 | s) = sigmoid(beta llr_RL).
beta is fitted like D-hat, by maximising sum ln P_beta(L_obs | s) over the same fit shots (round 1's fit_idx, 5,000 per
leaf, 20,000 per series), from round 1's ML(RL) llr of every shot (decode/<leaf>/rl/ml_exact.npz). A temperature never
changes a decision (beta > 0), so its m is 0 by construction; it can only change calibration and log-loss. Also one
beta for all 18 series. Evaluated on the same 72 held-out halves as D-hat: mean -ln P(L_obs | s) with paired
differences (SE stratified by leaf), R and R_conf, and the share of ML(RL)'s excess of R removed.
Usage: nice -n 10 python3 -I r2_temper.py ARCHIVE DECODE FIT F1 OUT_DIR  -> OUT_DIR/r2_temper.json (qrc.outpaths)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy import optimize

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (BASES, CONF, PATCHES, SHOTS, W5, min_p_from_llr, nll_from_llr, read_b8_bit0,  # noqa: E402
                       rel_of)
from r2_check import mean_se, ratio_block, share_removed  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402


def fit_beta(llr, obs):
    s = np.where(obs.astype(bool), 1.0, -1.0) * llr
    f = lambda b: float(np.logaddexp(0.0, -b * s).sum())  # noqa: E731
    o = optimize.minimize_scalar(f, bounds=(0.05, 5.0), method="bounded", options={"xatol": 1e-10})
    # SE of beta from the observed information: sum s^2 sigma(b s) sigma(-b s)
    b = float(o.x)
    sg = 1.0 / (1.0 + np.exp(-b * s))
    info = float(np.sum(s * s * sg * (1 - sg)))
    return b, 1.0 / math.sqrt(info), f(1.0) - f(b)


def main(arch, dec, fit, f1, out_dir):
    t0 = time.time()
    data = []
    for p in PATCHES:
        for b in BASES:
            lf_fit, lf_ho = [], []
            for r in W5:
                rel = rel_of(p, b, r)
                fd = np.load(os.path.join(fit, rel, "rl_tsplit", "fit_data.npz"), allow_pickle=False)
                ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
                dd = np.load(os.path.join(f1, rel, "d_data.npz"), allow_pickle=False)
                obs = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)
                fi, ti = fd["fit_idx"].astype(np.int64), fd["test_idx"].astype(np.int64)
                if not np.array_equal(ml["idx"], np.arange(SHOTS)) or not np.array_equal(ti, dd["test_idx"]):
                    raise SystemExit(f"{rel}: decode order or held-out shots differ")
                llr = ml["llr"].astype(np.float64)
                lf_fit.append((llr[fi], obs[fi]))
                lf_ho.append({"rel": rel, "r": r, "llr": llr[ti], "obs": obs[ti].astype(np.int8),
                              "pred": ml["pred"].reshape(SHOTS, -1)[ti, 0].astype(np.int8),
                              "D_llr": dd["D_llr"].astype(np.float64), "D_pred": dd["D_pred"].astype(np.int8)})
            data.append((f"d3_at_{p}/{b}", lf_fit, lf_ho))
    res = {"label": "EXPLORATORY (reservoir, 2026-10-10): one temperature on ML(RL)'s posteriors, fitted on F1's fit "
                    "shots, against D-hat on the held-out halves", "series": {}}
    allf = (np.concatenate([x[0] for _, lf, _ in data for x in lf]), np.concatenate([x[1] for _, lf, _ in data for x in lf]))
    bg, bg_se, _ = fit_beta(*allf)
    res["beta_global"] = {"beta": bg, "se": bg_se, "fit_shots": int(allf[0].size)}
    cols = {k: [] for k in ("nll_RL", "nll_D", "nll_B", "nll_G", "e_RL", "e_D", "m_RL", "m_D", "m_B", "m_G", "leaf", "r")}
    li = 0
    for s, lf_fit, lf_ho in data:
        llr_f = np.concatenate([x[0] for x in lf_fit]); obs_f = np.concatenate([x[1] for x in lf_fit])
        bs, bs_se, gain = fit_beta(llr_f, obs_f)
        res["series"][s] = {"beta": bs, "se": bs_se, "in_sample_gain_per_shot": gain / llr_f.size}
        for h in lf_ho:
            o = h["obs"]
            cols["nll_RL"].append(nll_from_llr(h["llr"], o)); cols["nll_D"].append(nll_from_llr(h["D_llr"], o))
            cols["nll_B"].append(nll_from_llr(bs * h["llr"], o)); cols["nll_G"].append(nll_from_llr(bg * h["llr"], o))
            cols["e_RL"].append((h["pred"] != o).astype(np.int8)); cols["e_D"].append((h["D_pred"] != o).astype(np.int8))
            cols["m_RL"].append(min_p_from_llr(h["llr"])); cols["m_D"].append(min_p_from_llr(h["D_llr"]))
            cols["m_B"].append(min_p_from_llr(bs * h["llr"])); cols["m_G"].append(min_p_from_llr(bg * h["llr"]))
            cols["leaf"].append(np.full(o.size, li, np.int32)); cols["r"].append(np.full(o.size, h["r"], np.int16))
            li += 1
    C = {k: np.concatenate(v) for k, v in cols.items()}
    lo = C["leaf"]
    res["heldout"] = {"shots": int(lo.size),
                      "mean_nll": {k: float(C[f"nll_{k}"].mean()) for k in ("RL", "D", "B", "G")},
                      "diff_RL_minus_temper_series": mean_se(C["nll_RL"] - C["nll_B"], lo),
                      "diff_RL_minus_temper_global": mean_se(C["nll_RL"] - C["nll_G"], lo),
                      "diff_RL_minus_D": mean_se(C["nll_RL"] - C["nll_D"], lo),
                      "diff_temper_series_minus_D": mean_se(C["nll_B"] - C["nll_D"], lo)}
    h = res["heldout"]
    h["share_of_D_logloss_gain_from_temper_series"] = (h["diff_RL_minus_temper_series"]["mean"] /
                                                       h["diff_RL_minus_D"]["mean"])
    cal = {}
    for tag, e, m in (("RL", C["e_RL"], C["m_RL"]), ("D", C["e_D"], C["m_D"]), ("temper_series", C["e_RL"], C["m_B"]),
                      ("temper_global", C["e_RL"], C["m_G"])):
        Rx = ratio_block(e, m, lo, minus_one=False)
        cf = m < CONF
        cal[tag] = {"errors": int(e.sum()), "predicted": float(m.sum()), "R": Rx["value"], "R_se": Rx["se"],
                    "conf_shots": int(cf.sum()), "conf_errors": int(e[cf].sum()), "conf_predicted": float(m[cf].sum()),
                    "R_conf": float(e[cf].sum() / m[cf].sum())}
    h["calibration"] = cal
    h["share_of_RL_excess_removed"] = {
        "temper_series_R": share_removed(C["e_RL"], C["m_B"], C["e_RL"], C["m_RL"], lo),
        "D_R": share_removed(C["e_D"], C["m_D"], C["e_RL"], C["m_RL"], lo)}
    bet = np.array([v["beta"] for v in res["series"].values()])
    res["beta_series_summary"] = {"min": float(bet.min()), "median": float(np.median(bet)), "max": float(bet.max())}
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r2_temper.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("beta_global", "beta_series_summary", "heldout")}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
