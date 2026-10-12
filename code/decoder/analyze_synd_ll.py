"""Syndrome log-likelihood tests, mean and dispersion, for the methods validation (offline, from stored per-shot ln P(s)).

Mean test (as in run_mismatch / run_fit): mean ln P_M(s) on the data vs on fresh samples from M.
Dispersion test (new): Var ln P_M(s) on the data vs on the fresh samples. A mixture of shot-level conditions (bursts,
drift) that a fitted model absorbs on average still inflates the spread of ln P(s). z = ln(v_d / v_r) /
sqrt((k_d - 1)/n_d + (k_r - 1)/n_r), with k the (non-excess) kurtosis of each sample (delta method for ln of a variance).

Pre-fit (model M of results/decoder/mismatch/<S>/): the fresh samples from M are regenerated exactly as run_mismatch.py
draws them (same seeds), and decoded with the exact decoder. Post-fit (results/decoder/fit/<S>/fit_data.npz): the stored
held-out ln P_fit(s) and the stored fresh samples from the fitted model.
Usage: python3 analyze_synd_ll.py [--res ../../results/decoder]  -> <res>/fit/synd_ll_tests.json and a printed table.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def ll_tests(l_data, l_ref):
    d = l_data[np.isfinite(l_data)]; r = l_ref[np.isfinite(l_ref)]
    nd, nr = len(d), len(r)
    vd, vr = d.var(), r.var()
    kd = np.mean((d - d.mean()) ** 4) / vd ** 2; kr = np.mean((r - r.mean()) ** 4) / vr ** 2
    return {"n_data": int(nd), "n_ref": int(nr), "mean_data": float(d.mean()), "mean_ref": float(r.mean()),
            "z_mean": float((d.mean() - r.mean()) / np.sqrt(vd / nd + vr / nr)),
            "sd_data": float(np.sqrt(vd)), "sd_ref": float(np.sqrt(vr)), "var_ratio": float(vd / vr),
            "kurt_data": float(kd), "kurt_ref": float(kr),
            "z_disp": float(np.log(vd / vr) / np.sqrt((kd - 1) / nd + (kr - 1) / nr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "decoder"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--p", type=float, default=3e-3)
    a = ap.parse_args()
    import stim
    from run_mismatch import scenario
    from ml_exact_fast import FastExactDecoder
    order = "S0,S1,S1b,S2,S3a,S3b,S4,S5".split(",")    # run_mismatch default order (its seeds depend on the index)
    out = {"pre_fit": {}, "post_fit": {}}
    ref_cache = {}
    for si, name in enumerate(order):
        fn = os.path.join(a.res, "mismatch", name, "data.npz")
        if not os.path.exists(fn):
            continue
        z = np.load(fn); lM = z["logp_M"]; N = len(lM)
        summ = json.load(open(os.path.join(a.res, "mismatch", name, "summary.json")))
        d, r = summ["d"], summ["rounds"]
        key = "dem" if name == "S3a" else "circuit"
        if key not in ref_cache:
            ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r)
            sc = scenario(name, ideal, a.p, N, np.random.default_rng(a.seed + 100 * si), a.seed + 1000 * (si + 1))
            kind, obj = sc["M_sampler"]
            if kind == "circuit":
                dref = obj.compile_detector_sampler(seed=a.seed + 99).sample(N)
            else:
                dref = obj.compile_sampler(seed=a.seed + 99).sample(N)[0]
            _, lref = FastExactDecoder(sc["M_dm"]).class_probs(dref.astype(np.uint8))
            ref_cache[key] = lref
        t = ll_tests(lM, ref_cache[key])
        t["z_mean_stored"] = summ["syndrome_loglik_M"]["z"]
        out["pre_fit"][name] = t
    for name in sorted(os.listdir(os.path.join(a.res, "fit"))):
        fn = os.path.join(a.res, "fit", name, "fit_data.npz")
        if not os.path.exists(fn):
            continue
        z = np.load(fn)
        t = ll_tests(z["logp_fit_test"], z["logp_ref"])
        t["z_mean_stored"] = json.load(open(os.path.join(a.res, "fit", name, "fit.json")))["test"]["syndrome_loglik_fit"]["z"]
        out["post_fit"][name] = t
    json.dump(out, open(os.path.join(a.res, "fit", "synd_ll_tests.json"), "w"), indent=1)
    for stage in ("pre_fit", "post_fit"):
        print(f"\n{stage}: scenario | z_mean (stored) | sd data/ref | var ratio | kurt data/ref | z_disp")
        for k, t in out[stage].items():
            print(f"  {k:8s} | {t['z_mean']:+7.1f} ({t['z_mean_stored']:+7.1f}) | {t['sd_data']:.3f}/{t['sd_ref']:.3f} | "
                  f"{t['var_ratio']:.3f} | {t['kurt_data']:.2f}/{t['kurt_ref']:.2f} | {t['z_disp']:+.1f}")


if __name__ == "__main__":
    main()
