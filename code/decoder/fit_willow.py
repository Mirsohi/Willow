"""Real-data model fitting (d = 3, exact): per-symptom-class scale factors on a shipped DEM, fitted by exact joint
likelihood on the fit half of a leaf's shots (split below), then evaluated on the held-out half: exact ML decoding
under the shipped and the fitted DEM (paired), calibration, and the syndrome log-likelihood mean and dispersion tests of both
DEMs against fresh samples from each. The fitted model is the "best available model" T-hat used as the truth proxy in
the decomposition (methods M3, M5; pre-registration notes/decoder/willow_prereg.md, test P4).

Queue usage: python3 -I -B fit_willow.py --root <extracted archive> --out <results>/fit --leaf <leaf> \
                 --dem-file decoding_results/<pathway>/error_model.dem --dem-tag rl --time-split --min-class-events 30
Older usage: --filter d3_at,/Z/,r10 --dem rl. Options: [--nfit 5000] [--shots 0:50000:1] [--hessian] [--seed 7]
Classes: the symptom classes of methods M4.5; --time-split also separates faults touching the first / last detector
layer. A class with fewer than --min-class-faults faults or fewer than --min-class-events expected fault events in the
fit sample is merged: first the layer tag is dropped, then the logical tag, then what is left is pooled ('rare'). A fit
that fails (singular information, non-finite scales) is retried with the threshold doubled, twice at most; if it still
fails, fit.json says status "fit_failed" and the leaf is left out of the decomposition.
Split: within each initialization bitstring (sweep-bit pattern), its occurrences alternate between the fit half and the
held-out half, so both halves hold every bitstring in the same proportion whatever the shot order (grouped or
interleaved); without sweep bits, even / odd positions. The fit sample is a seeded random subset of --nfit shots of the
fit half; every held-out shot is evaluated.
Undetectable (logical-only) faults, if any, are dropped and the syndrome-only likelihood is used for that leaf.
Outputs <out>/<leaf>/<dem tag>[_tsplit]/fit.json (written last) and fit_data.npz (held-out indices, predictions,
llr and ln P(s) under both DEMs, ln P(s) of the fresh reference samples).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from willow_data import find_experiments, Experiment
from dem_utils import parse_dem, DemMatrices
from fit_noise import DemClassScale, fit_bhhh, hessian_and_im_test, error_bars, class_calibration, symptom_class
from ml_exact_fast import FastExactDecoder
from mismatch import calibration, paired, dm_to_stim
from decode_willow import pick_dem, eps_per_cycle, parse_sel, md5_file
from analyze_synd_ll import ll_tests
from run_fit import tolist
import stim


def class_labels(dm, time_split, min_faults, min_events, n_fit):
    """Symptom classes (optionally split by first / last layer), merged until every class has at least min_faults
    faults and min_events expected fault events in the fit sample (sum of p_j times n_fit), so that no scale is poorly
    determined or runs to the boundary (validation S2+demt: a 1-fault class ran to zero; dry run 2026-10-08: a
    'diagonal+L' class with a few expected events went to zero and broke the fit). Merge order: drop the layer tag
    ('timelike+L@first' -> 'timelike+L'), then the logical tag ('timelike+L' -> 'timelike'); what is still too small is
    pooled into 'rare', and 'rare' itself, if too small, goes into the class with the most expected events."""
    lab = np.array([symptom_class(dm, j, time_split=time_split) for j in range(dm.num_faults)], dtype=object)
    merged = {}

    def stats(nm):
        sel = lab == nm
        return int(sel.sum()), float(dm.p[sel].sum() * n_fit)

    def small(nm):
        c, ev = stats(nm)
        return c < min_faults or ev < min_events

    def parent(nm):
        if "@" in nm:
            return nm.split("@")[0]
        if nm.endswith("+L"):
            return nm[:-2]
        return None
    while True:
        cand = [nm for nm in sorted(set(lab)) if small(nm) and parent(nm) is not None]
        if not cand:
            break
        nm = min(cand, key=lambda x: (stats(x)[1], x))
        c, ev = stats(nm)
        merged[nm] = {"n_faults": c, "expected_events": ev, "into": parent(nm)}
        lab[lab == nm] = parent(nm)
    rest = [nm for nm in sorted(set(lab)) if small(nm)]
    if rest and len(set(lab)) > 1:
        for nm in rest:
            c, ev = stats(nm)
            merged[nm] = {"n_faults": c, "expected_events": ev, "into": "rare"}
            lab[lab == nm] = "rare"
        if small("rare") and len(set(lab)) > 1:
            big = max((x for x in set(lab) if x != "rare"), key=lambda x: stats(x)[1])
            c, ev = stats("rare")
            merged["rare"] = {"n_faults": c, "expected_events": ev, "into": big}
            lab[lab == "rare"] = big
    return [str(x) for x in lab], merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dem", default="rl")
    ap.add_argument("--dem-file", default=None)
    ap.add_argument("--dem-tag", default=None)
    ap.add_argument("--filter", default="d3_at")
    ap.add_argument("--leaf", default=None, help="exact leaf path(s) relative to --root, comma-separated")
    ap.add_argument("--nfit", type=int, default=5000)
    ap.add_argument("--max-shots", type=int, default=50000)
    ap.add_argument("--shots", default=None, help="start:stop:step selection before the fit / held-out split")
    ap.add_argument("--hessian", action="store_true")
    ap.add_argument("--time-split", action="store_true", help="split symptom classes by first / last detector layer")
    ap.add_argument("--min-class-faults", type=int, default=0)
    ap.add_argument("--min-class-events", type=float, default=0.0, help="merge classes with fewer expected fault events in the fit sample")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.leaf:
        exps = [os.path.join(a.root, x) for x in a.leaf.split(",")]
    else:
        exps = [e for e in find_experiments(a.root) if all(s in os.path.relpath(e, a.root) for s in a.filter.split(",") if s)]
    print(f"{len(exps)} experiments", flush=True)
    for folder in exps:
        t0 = time.time()
        rel = os.path.relpath(folder, a.root)
        dtag = (a.dem_tag or a.dem).replace("/", "_")
        od = os.path.join(a.out, rel, dtag + ("_tsplit" if a.time_split else ""))
        if os.path.exists(os.path.join(od, "fit.json")):
            print(f"done already: {rel}", flush=True); continue
        exp = Experiment(folder)
        det_all, obs_all = exp.detection_events(), exp.obs_flips_actual()
        if det_all is None or obs_all is None:
            print(f"skip {rel}: no shots", flush=True); continue
        idx = parse_sel(a.shots, len(det_all), a.max_shots)
        n = len(idx)
        det = det_all[idx].astype(np.uint8); L = obs_all[idx, 0].astype(np.int64)
        if a.dem_file:
            dem, demname = stim.DetectorErrorModel.from_file(os.path.join(folder, a.dem_file)), a.dem_file
            dem_md5 = md5_file(os.path.join(folder, a.dem_file))
        else:
            (dem, demname), dem_md5 = pick_dem(exp, a.dem), None
        dm = parse_dem(dem)
        joint = True
        if dm.undetectable:
            dm = DemMatrices(dm.num_detectors, dm.num_observables, dm.dets, dm.obs, dm.p, dm.coords, [])
            joint = False
        sw = exp.sweep_bits()
        if sw is not None:
            from willow_stage import pattern_ids
            pid = pattern_ids(sw[idx])
            occ = np.zeros(n, np.int64)                          # occurrence number of each shot within its pattern
            for q in np.unique(pid):
                w = np.where(pid == q)[0]; occ[w] = np.arange(len(w))
            half, split = occ % 2, f"alternating within {int(pid.max()) + 1} sweep-bit patterns"
        else:
            half, split = np.arange(n) % 2, "even / odd positions (no sweep bits)"
        fit_all, te_pos = np.where(half == 0)[0], np.where(half == 1)[0]
        rng = np.random.default_rng(a.seed)
        fit_pos = np.sort(rng.choice(fit_all, size=min(a.nfit, len(fit_all)), replace=False)) if a.nfit else fit_all
        # fit; a failed fit (singular information, non-finite scales) is retried with the merge threshold doubled
        attempts, fr = [], None
        for k in range(3):
            me = a.min_class_events * 2 ** k if a.min_class_events > 0 else (0.0 if k == 0 else 15.0 * 2 ** k)
            labels, merged = class_labels(dm, a.time_split, a.min_class_faults, me, len(fit_pos))
            model = DemClassScale(dm, classes=labels)
            t1 = time.time()
            try:
                fr = fit_bhhh(model, det[fit_pos], L[fit_pos] if joint else None, [1.0] * len(model.names))
                ok = bool(np.all(np.isfinite(fr["theta_hat"])) and np.all(np.isfinite(fr["cov_opg"])))
                err = None if ok else "non-finite scales"
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as ex:
                ok, err = False, repr(ex)
            attempts.append({"min_class_events": me, "classes": len(model.names), "ok": ok, "error": err,
                             "seconds": time.time() - t1})
            if ok:
                break
            print(f"fit attempt {k + 1} failed ({err}); merging more", flush=True)
        t_fit = sum(x["seconds"] for x in attempts)
        if not attempts[-1]["ok"]:
            os.makedirs(od, exist_ok=True)
            json.dump(tolist({"experiment": rel, "status": "fit_failed", "selection": a.shots, "attempts": attempts,
                              "dem": demname, "dem_md5": dem_md5}), open(os.path.join(od, "fit.json.tmp"), "w"), indent=1)
            os.replace(os.path.join(od, "fit.json.tmp"), os.path.join(od, "fit.json"))
            print(f"{rel}: fit failed after {len(attempts)} attempts", flush=True)
            continue
        r = exp.meta.get("rounds") or 0
        res = {"experiment": rel, "dem": demname, "dem_md5": dem_md5, "dem_tag": dtag, "n": int(n), "selection": a.shots,
               "n_fit": int(len(fit_pos)), "n_test": int(len(te_pos)), "split": split, "fit_subsample_seed": a.seed,
               "joint_likelihood": joint, "rounds": r, "status": "complete", "fit_attempts": attempts,
               "classes": list(model.names), "merged_classes": merged,
               "n_faults_per_class": {nm: int(model.onehot[:, c].sum()) for c, nm in enumerate(model.names)},
               "expected_events_per_class": {nm: float(dm.p[model.onehot[:, c] > 0].sum() * len(fit_pos)) for c, nm in enumerate(model.names)},
               "scale_hat": fr["theta_hat"], "se_opg_ln": np.sqrt(np.abs(np.diag(fr["cov_opg"]))), "n_eval": fr["n_eval"],
               "history": fr["history"], "ll_per_shot_fit": fr["eval"]["ll"], "seconds_fit": t_fit,
               "class_calibration_at_hat": class_calibration(model, fr["eval"]["class_q"], fr["eval"]["p"])}
        if a.hessian:
            H, im = hessian_and_im_test(model, fr["phi_hat"], det[fit_pos], L[fit_pos] if joint else None, fr["eval"]["grads"])
            eb = error_bars(fr["opg"], H)
            res.update(se_hessian_ln=eb["se_hessian"], se_sandwich_ln=eb["se_sandwich"], im_test=im)
        dmF = model.dm_at(fr["theta_hat"])
        test, arrays = {}, {"test_idx": idx[te_pos].astype(np.int32), "fit_idx": idx[fit_pos].astype(np.int32)}
        Lt = L[te_pos].astype(np.uint8)
        for tag, D in (("shipped", dm), ("fitted", dmF)):
            dec = FastExactDecoder(D)
            post, logp = dec.class_probs(det[te_pos])
            P1 = post[:, 1]
            pred = (np.nan_to_num(P1, nan=0.0) > 0.5).astype(np.uint8)
            k = int(np.sum(pred != Lt))
            # syndrome log-likelihood tests on the held-out shots against fresh samples from the same DEM
            dref = dm_to_stim(D).compile_sampler(seed=a.seed).sample(len(te_pos))[0].astype(np.uint8)
            _, lref = dec.class_probs(dref)
            test[tag] = {"errors": k, "LER": k / len(te_pos), "eps_per_cycle": eps_per_cycle(k / len(te_pos), r) if r else None,
                         "calibration": calibration(P1, Lt), "nan_posteriors": int(np.isnan(P1).sum()),
                         "syndrome_loglik": ll_tests(logp, lref)}
            with np.errstate(divide="ignore", invalid="ignore"):
                arrays.update({f"{tag}_pred": pred, f"{tag}_llr": (np.log(post[:, 1]) - np.log(post[:, 0])).astype(np.float32),
                               f"{tag}_logp": logp, f"{tag}_lref": lref})
        test["paired_fitted_vs_shipped"] = paired(arrays["fitted_pred"], arrays["shipped_pred"], Lt)
        test["syndrome_loglik_fitted"] = test["fitted"]["syndrome_loglik"]      # Round-2 key, kept for old readers
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(os.path.join(od, "fit_data.npz"), obs_test=Lt, **arrays)
        res["test"] = test
        res["seconds"] = time.time() - t0
        json.dump(tolist(res), open(os.path.join(od, "fit.json.tmp"), "w"), indent=1)
        os.replace(os.path.join(od, "fit.json.tmp"), os.path.join(od, "fit.json"))
        print(f"{rel} [{demname}]: {len(model.names)} classes, scales "
              f"{', '.join(f'{str(k)}={float(v):.2f}' for k, v in zip(model.names, fr['theta_hat']))} | held-out LER shipped "
              f"{test['shipped']['LER']:.4f} fitted {test['fitted']['LER']:.4f} (paired {test['paired_fitted_vs_shipped']}) | "
              f"calib z shipped {test['shipped']['calibration']['z_LER']:.1f} fitted {test['fitted']['calibration']['z_LER']:.1f} | "
              f"synd-LL z fitted mean {test['fitted']['syndrome_loglik']['z_mean']:.1f} disp {test['fitted']['syndrome_loglik']['z_disp']:.1f}"
              f" | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
