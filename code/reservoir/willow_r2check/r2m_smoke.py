"""SMOKE TEST ONLY: fabricated mech-fit outputs for one series, to exercise r2m_check.py, r2m_redecode.py and
r2m_localmax.py before decoder's real mech fits exist. reservoir, 2026-10-10. Nothing here is a result.

Writes, under FAKE_DIR (a scratch folder outside the project's results), the files run_mech.py would write for one
series, in the same layout (<FAKE_DIR>/d3_at_<patch>/<basis>/mech_fit_T.json, mech_fit_D.json and
<FAKE_DIR>/d3_at_<patch>/<basis>/r<NN>/mech_data.npz, eval_mech.json):
  T-hat_mech: every shift 0, so its DEM is the RL DEM; its held-out predictions and llr are round 1's ML(RL), so the
              re-decode must agree to float32 rounding; mean_ll = round 1's ML(RL) mean ln P(s) of the fit shots.
  D-hat_mech: the shifts are F1's D-hat mapped to the templates (the start of PR2 8.2); its held-out predictions and
              llr are F1's ML(D-hat), so the re-decode agrees only approximately (a logit shift is not a scale, and a
              template can mix F1 classes); mean_ll = F1's in-sample ll (approximate).
  Records: converged after 2 steps, SE 0.05 for every parameter, the fingerprint recomputed here (so it must match).
Usage: python3 -I r2m_smoke.py ARCHIVE DECODE FIT F1 FAKE_DIR patch/basis
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_common import SHOTS, W5, read_b8_bit0, rel_of, series_fit  # noqa: E402
from r2m_common import (LAMBDA_D, fingerprint, fit_shots, leaf_name, mech_family, series_models,  # noqa: E402
                        start_from_f1)


def cells(x, y, o):
    ex, ey = x != o, y != o
    return {"n": int(o.size), "n10": int(np.sum(ex & ~ey)), "n01": int(np.sum(~ex & ey)), "n11": int(np.sum(ex & ey))}


def main(arch, dec, fit, f1, fake, pb):
    patch, basis = pb.split("/")
    if os.path.abspath(fake).startswith("/mnt/project-files"):
        raise SystemExit("smoke-test files go to a scratch folder, not the project")
    models = series_models(arch, fit, patch, basis)
    fam = mech_family(models)
    fs1 = series_fit(f1, patch, basis)
    lnames, dets, Ls, lp = [], [], [], []
    for r, M in zip(W5, models):
        fi, det, L = fit_shots(arch, fit, M)
        lnames.append(leaf_name(patch, basis, r)); dets.append(det); Ls.append(L)
        ml = np.load(os.path.join(dec, M["rel"], "rl", "ml_exact.npz"), allow_pickle=False)
        lp.append(ml["logp"][fi].astype(np.float64))
    fp = fingerprint(lnames, dets, Ls, fam["names"])
    K = len(fam["names"])
    dD = start_from_f1(models, fam, fs1["names"], np.array(fs1["phi_hat"], float))
    n = int(sum(L.size for L in Ls))
    m_T = float(np.concatenate(lp).mean())
    m_D = float(fs1["ll_per_shot"])
    common = {"names": fam["names"], "se": [0.05] * K, "se_opg": [0.05] * K, "secant": "bfgs", "fixed": [],
              "events": [], "converged": True, "stopped": False, "iterations": 2, "n_eval": 3, "n_shots": n,
              "family": fam["info"], "cpu_seconds_fit": 1.0, "cpu_seconds_fit_total": 1.0,
              "last_convergence_check": {"iteration": 2, "max_step_in_se": 0.01, "argmax": fam["names"][0],
                                         "secant": None},
              "series": f"google_105Q_surface_code_d3_d5_d7/d3_at_{patch}/{basis}", "leaves": lnames,
              "SMOKE_TEST": "fabricated by r2m_smoke.py; not a fit"}
    fT = dict(common, objective_type="marg", **{"lambda": 0.0}, information="centred OPG", delta_hat=[0.0] * K,
              mean_ll=m_T, objective=m_T * n, start="0",
              history=[{"objective": m_T * n, "mean_ll": m_T}, {"objective": m_T * n, "mean_ll": m_T}],
              fingerprint=fp + "|marg|0.0|info=opg|secant=bfgs")
    objD = m_D * n - 0.5 * LAMBDA_D * float(dD @ dD)
    fD = dict(common, objective_type="cond", **{"lambda": LAMBDA_D}, information="centred OPG + lambda",
              delta_hat=dD.tolist(), mean_ll=m_D, objective=objD, start="F1 D-hat mapped to templates",
              history=[{"objective": objD, "mean_ll": m_D}, {"objective": objD, "mean_ll": m_D}],
              fingerprint=fp + f"|cond|{LAMBDA_D}|info=opg|secant=bfgs")
    sd = os.path.join(fake, f"d3_at_{patch}", basis)
    os.makedirs(sd, exist_ok=True)
    json.dump(fT, open(os.path.join(sd, "mech_fit_T.json"), "w"), indent=1)
    json.dump(fD, open(os.path.join(sd, "mech_fit_D.json"), "w"), indent=1)
    for r, M in zip(W5, models):
        rel = rel_of(patch, basis, r)
        fd = np.load(os.path.join(fit, rel, "rl_tsplit", "fit_data.npz"), allow_pickle=False)
        dd = np.load(os.path.join(f1, rel, "d_data.npz"), allow_pickle=False)
        ti = fd["test_idx"].astype(np.int64)
        o = read_b8_bit0(os.path.join(arch, rel, "obs_flips_actual.b8"), SHOTS)[ti].astype(np.int8)
        S, T = fd["shipped_pred"].astype(np.int8), fd["fitted_pred"].astype(np.int8)
        pT, lT = S.astype(np.uint8), fd["shipped_llr"].astype(np.float32)
        pD, lD = dd["D_pred"].astype(np.uint8), dd["D_llr"].astype(np.float32)
        d = os.path.join(fake, rel)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "mech_data.npz"), test_idx=ti.astype(np.int32), obs_test=fd["obs_test"],
                            Tm_pred=pT, Tm_llr=lT, Dm_pred=pD, Dm_llr=lD)
        ev = {"leaf": leaf_name(patch, basis, r), "rounds": r, "n_test": int(ti.size),
              "errors": {"Tm": int(np.sum(pT != o)), "Dm": int(np.sum(pD != o)), "RL": int(np.sum(S != o)),
                         "T": int(np.sum(T != o))},
              "cells_RL_vs_Tm": cells(S, pT.astype(np.int8), o), "cells_RL_vs_Dm": cells(S, pD.astype(np.int8), o),
              "cells_T_vs_Tm": cells(T, pT.astype(np.int8), o), "cells_T_vs_Dm": cells(T, pD.astype(np.int8), o),
              "nan_posteriors": {"Tm": int(np.isnan(lT).sum()), "Dm": int(np.isnan(lD).sum())},
              "converged": {"Tm": True, "Dm": True}, "cpu_seconds_decode": 0.0, "SMOKE_TEST": True}
        json.dump(ev, open(os.path.join(d, "eval_mech.json"), "w"), indent=1)
    print(json.dumps({"series": pb, "parameters": K, "family": fam["info"], "fake_dir": fake}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
