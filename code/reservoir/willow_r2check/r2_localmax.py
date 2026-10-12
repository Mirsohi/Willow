"""Check D-hat's fit on whole series: the objective at the reported scales, and that moves along a few coordinates do not
raise it (check 3 of main's brief, 2026-10-10).

reservoir, 2026-10-10. The objective is PR2 section 2's: the mean over a series' fit shots (round 1's fit_idx, 5,000
per leaf, 20,000 per series) of ln P(L_obs | s), computed exactly with the decoder written here (r2_common:
p4_redecode.layered_ml), under p = p0 * exp(phi(class)) clipped as F1's model, with T-hat's classes rebuilt here.
Evaluated at phi-hat (fit_series.json), at phi = 0 (the RL DEM; fit_series.json history[0]) and at phi-hat +- h SE_i e_i
for a few coordinates i: the two classes with the smallest OPG SE, the class named by the last convergence check, and
the class with the largest |phi-hat| (then the next smallest SE, up to N_COORD). With l = n f the summed log-likelihood:
  d+ = l(phi-hat + h SE_i e_i) - l(phi-hat), d- likewise (both <= 0 at a local maximum);
  gradient g_i = (d+ - d-) / (2 h SE_i), curvature c_i = -(d+ + d-) / (h SE_i)^2, and the Newton step along i in SEs,
  (g_i / c_i) / SE_i (the fit's stop rule asks every component of its full step to be below 0.02 SE).
Checkpoint: every evaluation is saved at once to OUT_DIR/r2_localmax.ckpt.json (through qrc.outpaths, written to a
temporary name and renamed); a re-run with the same settings resumes from it, and it is removed when the run ends.
Usage: nice -n 10 python3 -I r2_localmax.py ARCHIVE FIT F1 OUT_DIR SERIES[,SERIES] [H N_COORD]
       SERIES as patch/basis, e.g. q8_5/Z,q4_9/X; defaults H = 0.5, N_COORD = 4
  -> OUT_DIR/r2_localmax.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import (W5, detection_events, layered_ml, leaf_model, nll_from_llr, read_b8_bit0, rel_of,  # noqa: E402
                       scaled, series_fit, SHOTS)
from qrc.outpaths import out_path  # noqa: E402


def load_series(arch, fit, patch, basis):
    leaves = []
    for r in W5:
        rel = rel_of(patch, basis, r)
        fj = json.load(open(os.path.join(fit, rel, "rl_tsplit", "fit.json")))
        fd = np.load(os.path.join(fit, rel, "rl_tsplit", "fit_data.npz"), allow_pickle=False)
        fi = fd["fit_idx"].astype(np.int64)
        M = leaf_model(arch, rel, r, fj["n_fit"], float(fj["fit_attempts"][-1]["min_class_events"]))
        obs = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)
        leaves.append({"rel": rel, "r": r, "M": M, "syn": detection_events(arch, rel, M["nd"], fi), "L": obs[fi]})
    return leaves


def objective(leaves, theta):
    """Mean ln P(L_obs | s) over the series' fit shots, the per-leaf means, and the number of clipped faults."""
    tot, n, per, ncl = 0.0, 0, {}, 0
    for lf in leaves:
        mech, nclip = scaled(lf["M"]["mech"], lf["M"]["cls"], theta)
        llr, _ = layered_ml(mech, lf["M"]["t_of"], lf["r"], lf["syn"])
        cll = -nll_from_llr(llr, lf["L"])
        if not np.all(np.isfinite(cll)):
            raise SystemExit(f"non-finite ln P(L|s) at {lf['rel']}")
        tot += float(cll.sum()); n += cll.size; per[lf["rel"]] = float(cll.mean()); ncl += nclip
    return tot / n, n, per, ncl


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main(arch, fit, f1, out_dir, series_arg, h=0.5, n_coord=4):
    t0 = time.time()
    ser = [tuple(s.split("/")) for s in series_arg.split(",")]
    ck_path = out_path(os.path.join(out_dir, "r2_localmax.ckpt.json"))
    cfg = {"series": series_arg, "h_in_se": h, "n_coord": n_coord}
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else None
    if ck is not None and ck.get("config") != cfg:
        raise SystemExit(f"checkpoint {ck_path} has other settings: {ck.get('config')}")
    evals = ck["evals"] if ck else {}
    resumed = len(evals)
    res = {"label": "reservoir, round-2 F1 check 3: the objective at D-hat and moves along single coordinates, 2026-10-10",
           "config": cfg, "series": {}}
    for patch, basis in ser:
        key = f"d3_at_{patch}/{basis}"
        fs = series_fit(f1, patch, basis)
        names, phi = fs["names"], np.array(fs["phi_hat"], float)
        se = np.array(fs["se_opg_ln"], float)
        order = [int(i) for i in np.argsort(se)]
        pick = order[:2] + [names.index(fs["last_convergence_check"]["argmax"]), int(np.argmax(np.abs(phi)))]
        coords = []
        for i in pick + order:
            if i not in coords:
                coords.append(i)
            if len(coords) == n_coord:
                break
        leaves = load_series(arch, fit, patch, basis)
        n_fit = sum(lf["syn"].shape[0] for lf in leaves)

        def ev(tag, ph):
            k = f"{key}|{tag}"
            if k not in evals:
                t1 = time.time()
                f, n, per, ncl = objective(leaves, dict(zip(names, np.exp(ph).tolist())))
                evals[k] = {"f": f, "n": n, "per_leaf": per, "clipped": ncl, "seconds": time.time() - t1}
                save(ck_path, {"config": cfg, "evals": evals})
                print(f"{k}: f = {f:.12f} ({evals[k]['seconds']:.0f} s)", flush=True)
            return evals[k]
        e0 = ev("hat", phi)
        e1 = ev("rl", np.zeros_like(phi))
        S = {"n_fit_shots": n_fit, "f_hat_here": e0["f"], "f_hat_decoder": fs["ll_per_shot"],
             "abs_diff_f_hat": abs(e0["f"] - fs["ll_per_shot"]), "f_rl_here": e1["f"],
             "f_rl_decoder": fs["history"][0]["ll"], "abs_diff_f_rl": abs(e1["f"] - fs["history"][0]["ll"]),
             "in_sample_gain_here": e0["f"] - e1["f"], "clipped_faults_at_hat": e0["clipped"], "coords": {}}
        worst = -np.inf
        for i in coords:
            hs = h * se[i]
            up, dn = phi.copy(), phi.copy()
            up[i] += hs; dn[i] -= hs
            fu, fd_ = ev(f"{names[i]}+", up)["f"], ev(f"{names[i]}-", dn)["f"]
            dp, dm = n_fit * (fu - e0["f"]), n_fit * (fd_ - e0["f"])
            g = (dp - dm) / (2 * hs)
            c = -(dp + dm) / hs ** 2
            newton = g / c / se[i] if c > 0 else None
            S["coords"][names[i]] = {"phi_hat": phi[i], "se_opg": se[i], "step": hs, "d_plus_sum_ll": dp,
                                     "d_minus_sum_ll": dm, "both_not_above": bool(dp <= 0 and dm <= 0),
                                     "gradient_sum_ll": g, "curvature": c, "curvature_times_se2": c * se[i] ** 2,
                                     "newton_step_in_se": newton}
            worst = max(worst, dp, dm)
        S["largest_increase_sum_ll"] = worst
        S["local_max_along_tested_coordinates"] = bool(worst <= 0)
        res["series"][key] = S
    res["evaluations_resumed_from_checkpoint"] = resumed
    res["evaluations"] = evals
    res["run_seconds_this_process"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r2_localmax.json")), "w") as f:
        json.dump(res, f, indent=1)
    os.remove(ck_path)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coords"} for k, v in res["series"].items()}, indent=1))
    print(json.dumps({k: v["coords"] for k, v in res["series"].items()}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (5, 7):
        raise SystemExit(__doc__)
    main(*a[:5], *((float(a[5]), int(a[6])) if len(a) == 7 else ()))
