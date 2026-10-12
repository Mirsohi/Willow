"""Benchmark ML (tensor-network) decoding against baselines on simulated Willow-like memory circuits.

Example:
  OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice python3 bench_sim.py --d 3 --rounds 5 10 20 --p 3e-3 \
      --basis Z --shots 20000 --decoders ml_exact,mwpm,corr_mwpm,beliefmatch,tesseract,bposd \
      --out ../../results/decoder/sim
Per config, predictions are checkpointed to <out>/<tag>/<decoder>.npz (skipped if present) and a
summary is written to <out>/<tag>/summary.json. ML decoders also store the posterior P(flip|s) and log P(s).
Decoder names: ml_exact | ml_sparse[:tau[:kmax]] | ml_mps[:chi[:cutoff]] | ml_mps_norw[:chi] | ml_exact_zonly | mwpm | corr_mwpm |
               beliefmatch | tesseract | bposd
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from willow_sim import memory_circuit
from dem_utils import parse_dem, restrict_dem


def eps_per_cycle(E, r):
    E = np.clip(E, 0, 0.5 - 1e-12)
    return 0.5 * (1 - (1 - 2 * E) ** (1.0 / r))


def wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def zonly_dem(dm, basis):
    """Detectors of the same type as the memory basis = those present in round 0 of Stim's generated circuits."""
    pos0 = set((dm.coords[i][0], dm.coords[i][1]) for i in range(dm.num_detectors) if dm.coords[i][-1] == 0)
    keep = [i for i in range(dm.num_detectors) if (dm.coords[i][0], dm.coords[i][1]) in pos0]
    return restrict_dem(dm, keep), np.array(keep)


def run_decoder(name, circ, demu, demd, dm, det, basis):
    t0 = time.time()
    post = logp = None
    extra = {}
    if name == "ml_exact":
        from ml_exact import ExactFrontierDecoder
        D = ExactFrontierDecoder(dm)
        D.class_probs(det[:2])
        t0 = time.time()
        post, logp = D.class_probs(det)
        pred = post.argmax(1)[:, None].astype(np.uint8)
        extra["W"] = int(D.W)
    elif name == "ml_exact_zonly":
        from ml_exact import ExactFrontierDecoder
        dz, keep = zonly_dem(dm, basis)
        D = ExactFrontierDecoder(dz)
        D.class_probs(det[:2, keep])
        t0 = time.time()
        post, logp = D.class_probs(det[:, keep])
        pred = post.argmax(1)[:, None].astype(np.uint8)
        extra["W"] = int(D.W)
    elif name.startswith("ml_sparse"):
        from ml_sparse import SparseFrontierDecoder
        parts = name.split(":")
        tau = float(parts[1]) if len(parts) > 1 else 1e-8
        kmax = int(parts[2]) if len(parts) > 2 else 400000
        D = SparseFrontierDecoder(dm, tau=tau, kmax=kmax)
        D.class_probs(det[:2])
        t0 = time.time()
        post, logp = D.class_probs(det)
        pred = np.nan_to_num(post, nan=0.5).argmax(1)[:, None].astype(np.uint8)
        extra.update(meanK=float(D.last_maxk.mean()), maxK=int(D.last_maxk.max()),
                     kmax_hit=int((D.last_maxk >= kmax).sum()))
    elif name.startswith("ml_mps"):
        from ml_mps_fast import FastMPSDecoder
        parts = name.split(":")
        chi = int(parts[1]) if len(parts) > 1 else 32
        cutoff = float(parts[2]) if len(parts) > 2 else 1e-12
        D = FastMPSDecoder(dm, chi=chi, cutoff=cutoff, reweight=not parts[0].endswith("norw"))
        D.class_probs(det[:2])
        t0 = time.time()
        post, logp = D.class_probs(det)
        pred = np.nan_to_num(post, nan=0.5).argmax(1)[:, None].astype(np.uint8)
        extra["max_bond"] = int(D.last_maxbond.max())
    else:
        import baselines
        D = baselines.make(name, demu, demd)
        D.decode(det[:2])
        t0 = time.time()
        pred = D.decode(det)
    dt = time.time() - t0
    return pred, post, logp, dt, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--decoders", default="ml_exact,mwpm,corr_mwpm,beliefmatch,tesseract")
    ap.add_argument("--cap", default="bposd:5000,tesseract:20000",
                    help="per-decoder shot caps, e.g. bposd:5000")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    caps = {}
    for kv in a.cap.split(","):
        if kv:
            k, v = kv.split(":"); caps[k] = int(v)
    for r in a.rounds:
        tag = f"d{a.d}_r{r:03d}_{a.basis}_p{a.p:.0e}_n{a.shots}_s{a.seed}"
        od = os.path.join(a.out, tag)
        os.makedirs(od, exist_ok=True)
        circ = memory_circuit(a.d, r, a.basis, a.p)
        demu = circ.detector_error_model(decompose_errors=False)
        demd = circ.detector_error_model(decompose_errors=True)
        dm = parse_dem(demu)
        det, obs = circ.compile_detector_sampler(seed=a.seed + 1000 * r + a.d).sample(a.shots, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        summ_path = os.path.join(od, "summary.json")
        summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
        summ.update({"d": a.d, "rounds": r, "p": a.p, "basis": a.basis, "shots": a.shots, "seed": a.seed,
                     "p_det": float(det.mean()), "num_detectors": int(circ.num_detectors),
                     "num_faults_merged": int(dm.num_faults)})
        summ.setdefault("decoders", {})
        for name in a.decoders.split(","):
            fn = os.path.join(od, name.replace(":", "_") + ".npz")
            n = min(a.shots, caps.get(name.split(":")[0], a.shots))
            if os.path.exists(fn):
                z = np.load(fn)
                pred = z["pred"]; n = pred.shape[0]
                dt = float(z["seconds"])
                extra = json.loads(str(z["extra"])) if "extra" in z else {}
            else:
                pred, post, logp, dt, extra = run_decoder(name, circ, demu, demd, dm, det[:n], a.basis)
                save = {"pred": pred, "seconds": dt, "extra": json.dumps(extra)}
                if post is not None:
                    save["post1"] = post[:, 1].astype(np.float32)
                    save["logp"] = logp.astype(np.float64)
                np.savez_compressed(fn, **save)
            err = (pred[:, 0] != obs[:n, 0])
            k = int(err.sum())
            E = k / n
            lo, hi = wilson(k, n)
            summ["decoders"][name] = {"n": n, "errors": k, "LER": E, "LER_1sigma": [lo, hi],
                                      "eps_per_cycle": eps_per_cycle(E, r),
                                      "eps_1sigma": [eps_per_cycle(lo, r), eps_per_cycle(hi, r)],
                                      "ms_per_shot": 1e3 * dt / n, **extra}
            print(f"[{tag}] {name:16s} n={n:6d} LER={E:.5f} eps={eps_per_cycle(E, r):.3e} {1e3*dt/n:8.3f} ms/shot {extra}", flush=True)
            json.dump(summ, open(summ_path, "w"), indent=1)
        # paired comparisons against the first ML decoder present
        ref = [nm for nm in a.decoders.split(",") if nm.startswith("ml_")]
        if ref:
            zr = np.load(os.path.join(od, ref[0].replace(":", "_") + ".npz"))["pred"][:, 0]
            pc = {}
            for name in a.decoders.split(","):
                z = np.load(os.path.join(od, name.replace(":", "_") + ".npz"))["pred"][:, 0]
                m = min(len(z), len(zr))
                er_ref = zr[:m] != obs[:m, 0]; er_x = z[:m] != obs[:m, 0]
                pc[name] = {"n": int(m), "ref_wrong_other_right": int(np.sum(er_ref & ~er_x)),
                            "ref_right_other_wrong": int(np.sum(~er_ref & er_x))}
            summ["paired_vs_" + ref[0]] = pc
            json.dump(summ, open(summ_path, "w"), indent=1)


if __name__ == "__main__":
    main()
