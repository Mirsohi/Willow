"""Methods validation, part (c): likelihood fitting of the noise model recovers the true parameters, with error
bars; and what the fit does when the truth is outside the family.

Usage:
  python3 run_fit.py scenarios --scenarios S0,S1,S2,S3b,S3b+xt,S4,S5 --nfit 4000 --mismatch <results/decoder/mismatch> --out <dir>
  python3 run_fit.py coverage --reps 20 --nfit 1500 --out <dir>
Scenario mode regenerates exactly the shots of run_mismatch.py (same seeds; checked against its data.npz), fits
theta = (p2, pm, p1) of si_family by exact joint likelihood on the first nfit shots, starting from the decoding
model M = SI1000(p), and evaluates the fitted model on the held-out shots (exact ML decoding, calibration,
syndrome log-likelihood test against fresh samples from the fitted model, paired counts against ML(M) and ML(T)).
'S3b+xt' fits the 4-parameter family with the crosstalk rate as a free parameter; '<S>+dem' fits per-symptom-class
scale factors on the decoding model's own DEM (the family that applies to a shipped DEM on hardware).
Coverage mode: independent datasets from the S2 truth, fitted from the truth; z = (ln theta_hat - ln theta)/se.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from dem_utils import parse_dem
from ml_exact_fast import FastExactDecoder as ExactFrontierDecoder
from mismatch import si_family, calibration, paired
from fit_noise import CircuitFamily, DemClassScale, fit_bhhh, hessian_and_im_test, error_bars, class_calibration
from mismatch import dm_to_stim
from run_mismatch import scenario, wilson

MISMATCH_LIST = "S0,S1,S1b,S2,S3a,S3b,S4,S5".split(",")   # run_mismatch default order (seeds depend on it)


def truth_theta(name, p):
    return {"S0": [p, 5 * p, p / 10], "S1": [1.5 * p, 7.5 * p, 0.15 * p], "S1b": [0.7 * p, 3.5 * p, 0.07 * p],
            "S2": [0.7 * p, 15 * p, p / 10], "S3b+xt": [p, 5 * p, p / 10, 1e-3]}.get(name)


def tolist(x):
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def ml(dm, det):
    post, logp = ExactFrontierDecoder(dm).class_probs(det)
    return post[:, 1], logp


def fit_report(model, det_fit, L_fit, theta0, theta_true, hess=True):
    t0 = time.time()
    fr = fit_bhhh(model, det_fit, L_fit, theta0)
    out = {"names": list(model.names), "theta0": list(theta0), "theta_hat": fr["theta_hat"], "n_fit": fr["n"],
           "n_eval": fr["n_eval"], "history": fr["history"], "mean_score_at_hat": fr["mean_score_at_hat"],
           "ll_per_shot_at_hat": fr["eval"]["ll"], "nonfinite": fr["eval"]["nonfinite"]}
    if hess:
        H, im = hessian_and_im_test(model, fr["phi_hat"], det_fit, L_fit, fr["eval"]["grads"])
        eb = error_bars(fr["opg"], H)
        out.update(im_test=im, hessian=H, opg=fr["opg"], **{k: eb[k] for k in ("se_opg", "se_hessian", "se_sandwich")})
    else:
        out["se_opg"] = np.sqrt(np.diag(fr["cov_opg"]))
    if theta_true is not None:
        dl = np.log(fr["theta_hat"]) - np.log(theta_true)
        out["theta_true"] = list(theta_true)
        out["ln_ratio_hat_true"] = dl
        for k in ("se_opg", "se_hessian", "se_sandwich"):
            if k in out:
                out["z_" + k[3:]] = dl / out[k]
    out["class_calibration_fit_set"] = class_calibration(model, fr["eval"]["class_q"], fr["eval"]["p"])
    out["seconds_fit"] = time.time() - t0
    return out, fr


def run_scenarios(a):
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=a.d, rounds=a.rounds)
    p = a.p
    for name in a.scenarios.split(","):
        t0 = time.time()
        base_name = name.split("+")[0]
        si = MISMATCH_LIST.index(base_name)
        rng = np.random.default_rng(a.seed + 100 * si)
        sc = scenario(base_name, ideal, p, a.shots, rng, a.seed + 1000 * (si + 1))
        det, obs = sc["det"], sc["obs"].astype(np.int64)
        res = {"scenario": name, "d": a.d, "rounds": a.rounds, "p": p, "shots": len(obs), "n_fit": a.nfit}
        mm = os.path.join(a.mismatch, base_name, "data.npz")
        prev = np.load(mm) if os.path.exists(mm) else None
        if prev is not None:
            res["regenerated_matches_mismatch_run"] = bool(np.array_equal(prev["obs"].astype(np.int64), obs))
        fit, te = slice(0, a.nfit), slice(a.nfit, len(obs))
        make = None
        if name.endswith("+dem") or name.endswith("+demt"):
            # per-symptom-class scale factors on the decoding model's own DEM (the family usable on shipped DEMs);
            # '+demt': classes also split by first / last detector layer (reset and final-measurement mechanisms)
            model = DemClassScale(sc["M_dm"], time_split=name.endswith("+demt"))
            names = model.names
            theta0 = [1.0] * len(names)
        elif name == "S3b+xt":
            names = ("p2", "pm", "p1", "xt")
            make = lambda th: si_family(ideal, th[0], th[1], th[2], crosstalk=th[3])
            theta0 = [p, 5 * p, p / 10, 3e-4]
        else:
            names = ("p2", "pm", "p1")
            make = lambda th: si_family(ideal, *th)
            theta0 = [p, 5 * p, p / 10]
        if make is not None:
            model = CircuitFamily(make, theta0, names)
        print(f"{name}: {model.base.num_faults} faults, fitting {names} on {a.nfit} shots", flush=True)
        rep, fr = fit_report(model, det[fit], obs[fit], theta0, truth_theta(name, p))
        res["fit"] = rep
        # held-out evaluation of the fitted model
        th = fr["theta_hat"]
        dmF = model.dm_at(th)
        pF, lF = ml(dmF, det[te])
        predF = (pF > 0.5).astype(np.uint8)
        obs_te = obs[te].astype(np.uint8)
        eF = int(np.sum(predF != obs_te)); nte = len(obs_te)
        test = {"n": nte, "ML(fit)": {"errors": eF, "LER": eF / nte, "LER_1sigma": wilson(eF, nte)},
                "calibration_ML(fit)": calibration(pF, obs_te)}
        if prev is not None:
            for k in ("ML_M", "ML_T"):
                pr = prev["pred_" + k][te]
                e = int(np.sum(pr != obs_te))
                test[k.replace("_", "(") + ")"] = {"errors": e, "LER": e / nte, "LER_1sigma": wilson(e, nte)}
                test[f"paired_fit_vs_{k}"] = paired(predF, pr, obs_te)
            test["calibration_ML(M)"] = calibration(prev["post1_M"][te], obs_te)
            if "label_burst_config" in prev.files:
                b = prev["label_burst_config"][te] >= 0
                for nm, sel in [("burst", b), ("no_burst", ~b)]:
                    test["subset_" + nm] = {"n": int(sel.sum()), "LER_ML(fit)": float(np.mean(predF[sel] != obs_te[sel])),
                                            "LER_ML(M)": float(np.mean(prev["pred_ML_M"][te][sel] != obs_te[sel])),
                                            "LER_ML(T)": float(np.mean(prev["pred_ML_T"][te][sel] != obs_te[sel])),
                                            "calibration_ML(fit)": calibration(pF[sel], obs_te[sel])}
            if "label_drift_component" in prev.files:
                comp = prev["label_drift_component"][te]
                for c, pv in enumerate([1.5e-3, 3e-3, 4.5e-3]):
                    sel = comp == c
                    test[f"subset_p{pv:.1e}"] = {"n": int(sel.sum()), "LER_ML(fit)": float(np.mean(predF[sel] != obs_te[sel])),
                                                 "LER_ML(M)": float(np.mean(prev["pred_ML_M"][te][sel] != obs_te[sel])),
                                                 "LER_ML(T)": float(np.mean(prev["pred_ML_T"][te][sel] != obs_te[sel])),
                                                 "calibration_ML(fit)": calibration(pF[sel], obs_te[sel])}
        # syndrome log-likelihood test of the fitted model: held-out data vs fresh samples from the fitted model
        if make is not None:
            dref = make(th).compile_detector_sampler(seed=a.seed + 4242).sample(nte).astype(np.uint8)
        else:
            dref = dm_to_stim(dmF).compile_sampler(seed=a.seed + 4242).sample(nte)[0].astype(np.uint8)
        _, lref = ml(dmF, dref)
        ok = np.isfinite(lF)
        test["syndrome_loglik_fit"] = {"mean_data": float(lF[ok].mean()), "mean_ref": float(lref.mean()),
                                       "z": float((lF[ok].mean() - lref.mean()) / np.sqrt(lF[ok].var() / ok.sum() + lref.var() / len(lref))),
                                       "sd_data": float(lF[ok].std()), "sd_ref": float(lref.std()), "nonfinite": int((~ok).sum())}
        res["test"] = test
        res["seconds"] = time.time() - t0
        od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True)
        json.dump(tolist(res), open(os.path.join(od, "fit.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(od, "fit_data.npz"), post1_fit_test=pF, logp_fit_test=lF, logp_ref=lref,
                            scores_at_hat=fr["eval"]["grads"], class_q_at_hat=fr["eval"]["class_q"])
        f = rep
        zs = " ".join(f"{n}={v:.4g}(z={z:+.2f})" for n, v, z in zip(names, f["theta_hat"], f.get("z_sandwich", [np.nan] * len(names))))
        c = test["calibration_ML(fit)"]
        print(f"{name}: theta_hat {zs} | se_opg/se_sw={np.array2string(np.array(f['se_opg']) / np.array(f['se_sandwich']), precision=2)} "
              f"IM z={np.array2string(np.array(f['im_test']['z']), precision=1)} | test LER fit={test['ML(fit)']['LER']:.4f} "
              + (f"M={test['ML(M)']['LER']:.4f} T={test['ML(T)']['LER']:.4f} " if "ML(M)" in test else "")
              + f"| calib fit z_LER={c['z_LER']:.1f} z_xent={c['z_xent']:.1f} synd-LL z={test['syndrome_loglik_fit']['z']:.1f} | {res['seconds']:.0f}s",
              flush=True)


def run_coverage(a):
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=a.d, rounds=a.rounds)
    p = a.p
    th_true = np.array(truth_theta(a.truth, p))
    make = lambda th: si_family(ideal, *th)
    model = CircuitFamily(make, th_true, ("p2", "pm", "p1"))
    Tc = make(th_true)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"coverage_{a.truth}_r{a.rounds}_n{a.nfit}.json")
    reps = json.load(open(fn))["reps"] if os.path.exists(fn) else []
    for r in range(len(reps), a.reps):
        t0 = time.time()
        det, obs = Tc.compile_detector_sampler(seed=a.seed + 7000 + r).sample(a.nfit, separate_observables=True)
        rep, fr = fit_report(model, det.astype(np.uint8), obs[:, 0].astype(np.int64), th_true, th_true, hess=a.hessian)
        rep = {k: rep[k] for k in ("theta_hat", "se_opg", "z_opg", "n_eval", "ll_per_shot_at_hat") + (("se_sandwich", "z_sandwich", "se_hessian", "z_hessian") if a.hessian else ())}
        rep["seconds"] = time.time() - t0
        reps.append(tolist(rep))
        json.dump({"truth": th_true.tolist(), "rounds": a.rounds, "n_fit": a.nfit, "reps": reps}, open(fn, "w"), indent=1)
        print(f"rep {r}: theta_hat={np.array2string(np.array(rep['theta_hat']), precision=5)} z_opg={np.array2string(np.array(rep['z_opg']), precision=2)} {rep['seconds']:.0f}s", flush=True)
    Z = np.array([x["z_opg"] for x in reps])
    summ = {"mean_z": Z.mean(axis=0).tolist(), "sd_z": Z.std(axis=0, ddof=1).tolist(),
            "cover_1se": np.mean(np.abs(Z) < 1, axis=0).tolist(), "cover_2se": np.mean(np.abs(Z) < 2, axis=0).tolist(),
            "mean_rel_se": np.mean([x["se_opg"] for x in reps], axis=0).tolist(),
            "sd_ln_theta_hat": np.std(np.log([x["theta_hat"] for x in reps]), axis=0, ddof=1).tolist()}
    json.dump({"truth": th_true.tolist(), "rounds": a.rounds, "n_fit": a.nfit, "reps": reps, "summary": summ}, open(fn, "w"), indent=1)
    print("coverage summary:", json.dumps(summ), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scenarios", "coverage"])
    ap.add_argument("--scenarios", default="S0,S1,S2,S3b,S3b+xt,S4,S5")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--nfit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mismatch", default="")
    ap.add_argument("--truth", default="S2")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--hessian", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_scenarios(a) if a.mode == "scenarios" else run_coverage(a)
