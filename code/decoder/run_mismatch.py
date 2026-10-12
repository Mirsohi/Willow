"""Methods validation, parts (a) cost of mismatch and (b) calibration statistics, on synthetic ground truth.

Usage: python3 run_mismatch.py --scenarios S0,S1,... --out ../../results/decoder/mismatch [--d 3 --rounds 10 --shots 20000]
Scenarios (decoding model M = SI1000(p) unless noted; T = truth that generates the shots):
  S0   T = M (control)
  S1   T = SI1000(1.5 p)                      global rate underestimated by M
  S1b  T = SI1000(0.7 p)                      global rate overestimated by M
  S2   T = si_family(0.7 p, 15 p, p/10)       component rates off (2q gates x0.7, measurement x3)
  S3a  T = SI1000(p), M = graphlike (independent-edge) approximation of T's DEM (matching-style prior)
  S3b  T = SI1000(p) + correlated XX/ZZ crosstalk (1e-3 per horizontal data pair per round)
  S4   T = SI1000(p) + leakage-like bursts: with prob 0.1 per shot one random data qubit is fully
       depolarised at the start of 3 consecutive rounds; ML(T) is the oracle that knows the burst
  S5   T = shot-to-shot drift: p_shot in {1.5, 3, 4.5}e-3 with equal weights; ML(T) = exact mixture ML
Per scenario: <out>/<S>/data.npz (obs, posteriors, log P(s), predictions, burst/drift labels) and summary.json
(LERs with Wilson 1-sigma, paired counts vs ML(T), calibration of ML(M) and ML(T), syndrome log-likelihood
test of M against fresh samples from M).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from dem_utils import parse_dem
from ml_exact import ExactFrontierDecoder
from mismatch import si_family, leak_variant, graphlike_dm, dm_to_stim, data_qubits, calibration, paired
import baselines


def wilson(k, n, z=1.0):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def ml(dm, det):
    D = ExactFrontierDecoder(dm)
    post, logp = D.class_probs(det)
    return post[:, 1], logp


def logsumexp2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def scenario(name, ideal, p, N, rng, seed):
    """Return dict: det, obs, T (label), M_dm, M_demu, M_demd, decode_T(det) -> (post1, logp), extra labels."""
    base = si_family(ideal, p, 5 * p, p / 10)
    M_demu = base.detector_error_model(decompose_errors=False)
    M_demd = base.detector_error_model(decompose_errors=True)
    M_dm = parse_dem(M_demu)
    out = {"M_dm": M_dm, "M_demu": M_demu, "M_demd": M_demd, "M_sampler": ("circuit", base), "labels": None}

    def single(Tc):
        det, obs = Tc.compile_detector_sampler(seed=seed).sample(N, separate_observables=True)
        Tdm = parse_dem(Tc.detector_error_model(decompose_errors=False))
        out.update(det=det.astype(np.uint8), obs=obs[:, 0].astype(np.uint8), decode_T=lambda d: ml(Tdm, d))

    if name == "S0":
        single(base)
    elif name == "S1":
        single(si_family(ideal, 1.5 * p, 7.5 * p, 0.15 * p))
    elif name == "S1b":
        single(si_family(ideal, 0.7 * p, 3.5 * p, 0.07 * p))
    elif name == "S2":
        single(si_family(ideal, 0.7 * p, 15 * p, p / 10))
    elif name == "S3a":
        single(base)
        g = graphlike_dm(M_demd)
        gs = dm_to_stim(g)
        out.update(M_dm=g, M_demu=gs, M_demd=gs, M_sampler=("dem", gs))
    elif name == "S3b":
        single(si_family(ideal, p, 5 * p, p / 10, crosstalk=1e-3))
    elif name == "S4":
        dq = data_qubits(ideal)
        r = sum(1 for inst in ideal.flattened() if inst.name == "MR")
        delta, qb = 3, 0.1
        burst = rng.random(N) < qb
        cfg = np.full(N, -1)
        configs = [(q, t0) for q in dq for t0 in range(0, r - delta + 1)]
        cfg[burst] = rng.integers(0, len(configs), int(burst.sum()))
        det = np.zeros((N, base.num_detectors), np.uint8); obs = np.zeros(N, np.uint8)
        circs = {}
        for c in np.unique(cfg):
            idx = np.where(cfg == c)[0]
            if c < 0:
                C = base
            else:
                q, t0 = configs[c]
                C = leak_variant(base, q, range(t0, t0 + delta))
            circs[c] = C
            dd, oo = C.compile_detector_sampler(seed=seed + 7 + int(c)).sample(len(idx), separate_observables=True)
            det[idx] = dd; obs[idx] = oo[:, 0]

        def decode_T(d):
            post1 = np.zeros(len(d)); logp = np.zeros(len(d))
            for c in np.unique(cfg):
                idx = np.where(cfg == c)[0]
                pp, ll = ml(parse_dem(circs[c].detector_error_model(decompose_errors=False)), d[idx])
                post1[idx] = pp; logp[idx] = ll
            return post1, logp
        out.update(det=det, obs=obs, decode_T=decode_T, labels={"burst_config": cfg, "configs": configs})
    elif name == "S5":
        ps = np.array([1.5e-3, 3e-3, 4.5e-3]); w = np.ones(3) / 3
        comp = rng.integers(0, 3, N)
        circs = [si_family(ideal, q, 5 * q, q / 10) for q in ps]
        det = np.zeros((N, base.num_detectors), np.uint8); obs = np.zeros(N, np.uint8)
        for c in range(3):
            idx = np.where(comp == c)[0]
            dd, oo = circs[c].compile_detector_sampler(seed=seed + 11 + c).sample(len(idx), separate_observables=True)
            det[idx] = dd; obs[idx] = oo[:, 0]
        dms = [parse_dem(C.detector_error_model(decompose_errors=False)) for C in circs]

        def decode_T(d):
            # exact mixture ML: P(s, L) = sum_c w_c P_c(s, L)
            l0 = np.full(len(d), -np.inf); l1 = np.full(len(d), -np.inf)
            for c in range(3):
                pp, ll = ml(dms[c], d)
                l0 = logsumexp2(l0, np.log(w[c]) + ll + np.log(np.clip(1 - pp, 1e-300, 1)))
                l1 = logsumexp2(l1, np.log(w[c]) + ll + np.log(np.clip(pp, 1e-300, 1)))
            lt = logsumexp2(l0, l1)
            return np.exp(l1 - lt), lt
        out.update(det=det, obs=obs, decode_T=decode_T, labels={"drift_component": comp, "p_values": ps.tolist()})
    else:
        raise ValueError(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="S0,S1,S1b,S2,S3a,S3b,S4,S5")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--baselines", default="mwpm,corr_mwpm,tesseract")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ideal = stim.Circuit.generated("surface_code:rotated_memory_z", distance=a.d, rounds=a.rounds)
    ref_cache = {}
    for si, name in enumerate(a.scenarios.split(",")):
        t0 = time.time()
        od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True)
        rng = np.random.default_rng(a.seed + 100 * si)
        sc = scenario(name, ideal, a.p, a.shots, rng, a.seed + 1000 * (si + 1))
        det, obs = sc["det"], sc["obs"]
        N = len(obs)
        res = {"scenario": name, "d": a.d, "rounds": a.rounds, "p": a.p, "shots": N, "p_det": float(det.mean()), "decoders": {}}
        pM, lM = ml(sc["M_dm"], det)
        pT, lT = sc["decode_T"](det)
        preds = {"ML(M)": (pM > 0.5).astype(np.uint8), "ML(T)": (pT > 0.5).astype(np.uint8)}
        for b in a.baselines.split(","):
            if not b:
                continue
            D = baselines.make(b, sc["M_demu"], sc["M_demd"])
            preds[b + "(M)"] = D.decode(det)[:, 0].astype(np.uint8)
        for k, v in preds.items():
            e = int(np.sum(v != obs))
            res["decoders"][k] = {"errors": e, "LER": e / N, "LER_1sigma": wilson(e, N),
                                  "paired_vs_ML(T)": paired(v, preds["ML(T)"], obs)}
        res["calibration_ML(M)"] = calibration(pM, obs)
        res["calibration_ML(T)"] = calibration(pT, obs)
        # syndrome-only goodness of fit: mean log P_M(s) on the data vs on fresh samples from M
        key = (sc["M_sampler"][0], id(sc["M_dm"]) if sc["M_sampler"][0] == "dem" else "base")
        if key not in ref_cache:
            kind, obj = sc["M_sampler"]
            if kind == "circuit":
                dref = obj.compile_detector_sampler(seed=a.seed + 99).sample(N)
            else:
                dref = obj.compile_sampler(seed=a.seed + 99).sample(N)[0]
            _, lref = ml(sc["M_dm"], dref.astype(np.uint8))
            ref_cache[key] = lref
        lref = ref_cache[key]
        ok = np.isfinite(lM)
        res["syndrome_loglik_M"] = {"mean_data": float(lM[ok].mean()), "mean_ref": float(lref.mean()),
                                    "per_detector_data": float(lM[ok].mean() / det.shape[1]),
                                    "z": float((lM[ok].mean() - lref.mean()) / np.sqrt(lM[ok].var() / ok.sum() + lref.var() / len(lref))),
                                    "nonfinite": int((~ok).sum())}
        res["syndrome_loglik_T_minus_M_per_shot"] = float(np.mean(lT[ok] - lM[ok]))
        if sc["labels"] is not None:
            lab = sc["labels"]
            if "burst_config" in lab:
                b = lab["burst_config"] >= 0
                for nm, sel in [("burst", b), ("no_burst", ~b)]:
                    res[f"subset_{nm}"] = {"n": int(sel.sum()),
                                           "LER_ML(M)": float(np.mean(preds["ML(M)"][sel] != obs[sel])),
                                           "LER_ML(T)": float(np.mean(preds["ML(T)"][sel] != obs[sel])),
                                           "calibration_ML(M)": calibration(pM[sel], obs[sel])}
            if "drift_component" in lab:
                for c, pv in enumerate(lab["p_values"]):
                    sel = lab["drift_component"] == c
                    res[f"subset_p{pv:.1e}"] = {"n": int(sel.sum()),
                                                "LER_ML(M)": float(np.mean(preds["ML(M)"][sel] != obs[sel])),
                                                "LER_ML(T)": float(np.mean(preds["ML(T)"][sel] != obs[sel])),
                                                "calibration_ML(M)": calibration(pM[sel], obs[sel])}
        res["seconds"] = time.time() - t0
        save = {"obs": obs, "post1_M": pM.astype(np.float64), "logp_M": lM, "post1_T": pT.astype(np.float64), "logp_T": lT}
        for k, v in preds.items():
            save["pred_" + k.replace("(", "_").replace(")", "")] = v
        if sc["labels"] is not None:
            for k, v in sc["labels"].items():
                if isinstance(v, np.ndarray):
                    save["label_" + k] = v
        np.savez_compressed(os.path.join(od, "data.npz"), **save)
        json.dump(res, open(os.path.join(od, "summary.json"), "w"), indent=1)
        cM, cT = res["calibration_ML(M)"], res["calibration_ML(T)"]
        print(f"{name}: p_det={res['p_det']:.4f} LER ML(T)={res['decoders']['ML(T)']['LER']:.4f} ML(M)={res['decoders']['ML(M)']['LER']:.4f} "
              + " ".join(f"{k}={v['LER']:.4f}" for k, v in res["decoders"].items() if k not in ("ML(T)", "ML(M)"))
              + f" | calib M: pred={cM['LER_predicted']:.4f} obs={cM['LER_observed']:.4f} z={cM['z_LER']:.1f} xent z={cM['z_xent']:.1f}"
              + f" | calib T: z={cT['z_LER']:.1f} | synd-LL z={res['syndrome_loglik_M']['z']:.1f} | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
