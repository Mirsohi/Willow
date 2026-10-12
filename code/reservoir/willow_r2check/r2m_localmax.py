"""Check the mech fits on whole series: the objective at the reported shifts and at the fits' start, and that moves along
a few coordinates do not raise it. Part of the mech check of main's brief of 2026-10-10 ("the same checks as for F1");
written before the real mech fits.

reservoir, 2026-10-10. Objectives (PR2 section 8.2) over a series' fit shots (round 1's fit_idx, 5,000 per leaf,
20,000 per series), computed exactly with r2_common.layered_ml under the family rebuilt in r2m_common:
  T-hat_mech: l_T(delta) = sum ln P_delta(s), no penalty; its start is delta = 0;
  D-hat_mech: l_D(delta) = sum ln P_delta(L_obs | s) - (lambda / 2) sum delta^2 with lambda = 4 (the fit's own
              lambda is used and must be 4); its start is F1's D-hat mapped to the templates (r2m_common.start_from_f1).
Each evaluation gives both mean ln P(s) and mean ln P(L|s), so both are recorded. Evaluated at delta-hat (compared with
mech_fit_<T|D>.json: mean_ll, and objective = the penalised sum), at the start (compared with history[0]), and at
delta-hat +- h SE_i e_i for N_COORD free coordinates: the two with the smallest SE (the fit's own SEs), the last
convergence check's argmax, the largest |delta-hat|, then the next smallest SEs. d+- = l(delta-hat +- h SE_i e_i) -
l(delta-hat), both <= 0 at a local maximum along i; gradient (d+ - d-) / (2 h SE_i), curvature -(d+ + d-) / (h SE_i)^2,
and the Newton step in SEs (the fits stop when every step is below 0.02 SE).
Checkpoint: each evaluation is saved at once to OUT_DIR/r2m_localmax.ckpt.json (through qrc.outpaths, a temporary
name then a rename); a re-run with the same settings resumes from it. At the end it is removed, and removed again if
the shared store brings it back within 30 s.
--shots-per-leaf N, for smoke tests only, uses the first N fit shots of each leaf; the comparisons with decoder's numbers
are then meaningless, and the output says so.
Usage: nice -n 10 python3 -I r2m_localmax.py ARCHIVE FIT F1 MECH OUT_DIR --series q8_5/X[,q6_3/Z] [--fits T,D]
       [--h 0.5] [--n-coord 4] [--shots-per-leaf N]  -> OUT_DIR/r2m_localmax.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import W5, layered_ml, nll_from_llr, series_fit  # noqa: E402
from r2m_common import LAMBDA_D, fit_shots, mech_family, mech_faults, series_models, start_from_f1  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

FILES = {"T": "mech_fit_T.json", "D": "mech_fit_D.json"}


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def remove_for_good(path, wait=30.0):
    t = time.time()
    while True:
        if os.path.exists(path):
            os.remove(path)
        if time.time() - t > wait:
            break
        time.sleep(5.0)
    return not os.path.exists(path)


def evaluate(leaves, fam, delta):
    """Sums over the fit shots: ln P(s), ln P(L_obs | s); the shot count."""
    lps, cll, n = 0.0, 0.0, 0
    for lf, ix in zip(leaves, fam["idx"]):
        faults, _ = mech_faults(lf["M"], ix, delta)
        llr, logp = layered_ml(faults, lf["M"]["t_of"], lf["M"]["r"], lf["syn"])
        c = -nll_from_llr(llr, lf["L"])
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(logp))):
            raise SystemExit(f"non-finite log-likelihood at {lf['M']['rel']}")
        lps += float(np.sum(logp)); cll += float(np.sum(c)); n += c.size
    return {"sum_lnPs": lps, "sum_cll": cll, "n": n}


def objective(e, tag, delta, lam):
    if tag == "T":
        return e["sum_lnPs"]
    return e["sum_cll"] - 0.5 * lam * float(np.dot(delta, delta))


def main(a):
    t0 = time.time()
    ser = [tuple(s.split("/")) for s in a.series.split(",")]
    tags = a.fits.split(",")
    cfg = {"series": a.series, "fits": a.fits, "h_in_se": a.h, "n_coord": a.n_coord, "shots_per_leaf": a.shots_per_leaf}
    ck_path = out_path(os.path.join(a.out_dir, "r2m_localmax.ckpt.json"))
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else None
    if ck is not None and ck.get("config") != cfg:
        raise SystemExit(f"checkpoint {ck_path} has other settings: {ck.get('config')}")
    evals = ck["evals"] if ck else {}
    resumed = len(evals)
    res = {"label": "reservoir, mech check: the objectives at the fitted shifts, at the start, and moves along single "
                    "coordinates, 2026-10-10", "config": cfg, "series": {},
           "smoke_test_subsample": a.shots_per_leaf is not None}
    for patch, basis in ser:
        key = f"d3_at_{patch}/{basis}"
        sd = os.path.join(a.mech, f"d3_at_{patch}", basis)
        models = series_models(a.archive, a.fit, patch, basis)
        fam = mech_family(models)
        leaves = []
        for M in models:
            fi, det, L = fit_shots(a.archive, a.fit, M)
            if a.shots_per_leaf:
                det, L = det[:a.shots_per_leaf], L[:a.shots_per_leaf]
            leaves.append({"M": M, "syn": det, "L": L})
        fs1 = series_fit(a.f1, patch, basis)
        S = {"family": fam["info"], "fits": {}}
        for tag in tags:
            fj = json.load(open(os.path.join(sd, FILES[tag])))
            if fj["names"] != fam["names"]:
                raise SystemExit(f"{key} {tag}: parameter names differ from the family rebuilt here")
            lam = float(fj.get("lambda", 0.0)) if tag == "D" else 0.0
            if tag == "D" and lam != LAMBDA_D:
                raise SystemExit(f"{key} D: lambda {lam}, pre-registered {LAMBDA_D}")
            dhat = np.array(fj["delta_hat"], float)
            se = np.array([np.nan if v is None else v for v in fj["se"]], float)
            fixed = set(fj.get("fixed", []))
            free = [i for i, nm in enumerate(fj["names"]) if nm not in fixed and np.isfinite(se[i]) and se[i] > 0]
            start = (np.zeros_like(dhat) if tag == "T"
                     else start_from_f1(models, fam, fs1["names"], np.array(fs1["phi_hat"], float)))

            def ev(name, d):
                k = f"{key}|{tag}|{name}"
                if k not in evals:
                    t1 = time.time()
                    e = evaluate(leaves, fam, d)
                    e["objective"] = objective(e, tag, d, lam)
                    e["seconds"] = time.time() - t1
                    evals[k] = e
                    save(ck_path, {"config": cfg, "evals": evals})
                    print(f"{k}: objective {e['objective']:.6f}, mean ln P(s) {e['sum_lnPs'] / e['n']:.9f}, "
                          f"mean ln P(L|s) {e['sum_cll'] / e['n']:.12f} ({e['seconds']:.0f} s)", flush=True)
                return evals[k]
            e_hat, e_start = ev("hat", dhat), ev("start", start)
            n = e_hat["n"]
            mean_hat = (e_hat["sum_lnPs"] if tag == "T" else e_hat["sum_cll"]) / n
            mean_start = (e_start["sum_lnPs"] if tag == "T" else e_start["sum_cll"]) / n
            F = {"n_shots": n, "lambda": lam, "free_parameters": len(free),
                 "mean_ll_hat_here": mean_hat, "mean_ll_hat_decoder": fj.get("mean_ll"),
                 "abs_diff_mean_ll_hat": abs(mean_hat - fj["mean_ll"]) if fj.get("mean_ll") is not None else None,
                 "objective_hat_here": e_hat["objective"], "objective_hat_decoder": fj.get("objective"),
                 "abs_diff_objective_hat": (abs(e_hat["objective"] - fj["objective"])
                                            if fj.get("objective") is not None else None),
                 "mean_ll_start_here": mean_start,
                 "mean_ll_start_decoder": (fj.get("history") or [{}])[0].get("mean_ll"),
                 "objective_start_here": e_start["objective"],
                 "objective_start_decoder": (fj.get("history") or [{}])[0].get("objective"),
                 "other_mean_at_hat": {"lnPs": e_hat["sum_lnPs"] / n, "cll": e_hat["sum_cll"] / n},
                 "coords": {}}
            if F["mean_ll_start_decoder"] is not None:
                F["abs_diff_mean_ll_start"] = abs(mean_start - F["mean_ll_start_decoder"])
            order = sorted(free, key=lambda i: se[i])
            lc = (fj.get("last_convergence_check") or {}).get("argmax")
            pick = order[:2] + ([fj["names"].index(lc)] if lc in fj["names"] and fj["names"].index(lc) in free else [])
            pick += [max(free, key=lambda i: abs(dhat[i]))] if free else []
            coords = []
            for i in pick + order:
                if i not in coords:
                    coords.append(i)
                if len(coords) == a.n_coord:
                    break
            worst = -np.inf
            for i in coords:
                hs = a.h * se[i]
                up, dn = dhat.copy(), dhat.copy()
                up[i] += hs; dn[i] -= hs
                dp = ev(f"{fj['names'][i]}+", up)["objective"] - e_hat["objective"]
                dm = ev(f"{fj['names'][i]}-", dn)["objective"] - e_hat["objective"]
                g = (dp - dm) / (2 * hs)
                c = -(dp + dm) / hs ** 2
                F["coords"][fj["names"][i]] = {"delta_hat": dhat[i], "se": se[i], "step": hs, "d_plus": dp, "d_minus": dm,
                                               "both_not_above": bool(dp <= 0 and dm <= 0), "gradient": g,
                                               "curvature": c, "curvature_times_se2": c * se[i] ** 2,
                                               "newton_step_in_se": (g / c / se[i]) if c > 0 else None}
                worst = max(worst, dp, dm)
            F["largest_increase"] = worst if coords else None
            F["local_max_along_tested_coordinates"] = bool(coords) and bool(worst <= 0)
            S["fits"][tag] = F
        res["series"][key] = S
    res["evaluations_resumed_from_checkpoint"] = resumed
    res["evaluations"] = evals
    res["run_seconds_this_process"] = time.time() - t0
    with open(out_path(os.path.join(a.out_dir, "r2m_localmax.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    res["checkpoint_removed"] = remove_for_good(ck_path)
    print(json.dumps({k: {t: {kk: vv for kk, vv in v.items() if kk != "coords"} for t, v in s["fits"].items()}
                      for k, s in res["series"].items()}, indent=1, default=float))
    print("checkpoint removed:", res["checkpoint_removed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("archive", "fit", "f1", "mech", "out_dir"):
        ap.add_argument(k)
    ap.add_argument("--series", required=True)
    ap.add_argument("--fits", default="T,D")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--n-coord", type=int, default=4)
    ap.add_argument("--shots-per-leaf", type=int, default=None)
    main(ap.parse_args())
