"""Batch decoding of Google's released memory experiments with checkpointing (real-data pipeline).

Queue usage (one leaf, one DEM per call; this is what willow_queue.py runs):
  OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice python3 -I -B decode_willow.py --root <extracted archive> \
      --out <results>/decode --work <local scratch> --leaf <leaf path relative to root> \
      --dem-file decoding_results/<pathway>/error_model.dem --dem-tag rl \
      --decoders ml_exact,corr_mwpm,mwpm [--shots 0:50000:5] [--chunk 5000] [--synd-ref 10000] [--time-budget 3600]
Older usage (still works): --filter d3_at --dem si1000|rl|<substring of a shipped .dem path>|circuit_si1000.

For each experiment folder (one patch/basis/rounds = one leaf) and decoder, predictions are written chunk by chunk to
<work>/<leaf>/<dem tag>/<decoder>/chunk_XXXXXX.npz (with a spec.json; chunks of another shot selection or DEM are
discarded), so the job can be killed and restarted at any time; finished chunks are skipped. When every decoder is
done, the per-shot results are consolidated into <out>/<leaf>/<dem tag>/: <decoder>.npz (idx, pred, and for ML
decoders llr = ln P(L=1|s)/P(L=0|s) as float32 and logp = ln P(s)), shots.npz (idx, observed flips and every shipped
prediction), synd_ref_<decoder>.npz (ln P(s) of fresh DEM samples), and summary.json (written last; "status" says
complete / partial / abort). summary.json holds LER, eps per cycle, paired 2x2 counts between all decoders and the
shipped predictions, ML calibration, the syndrome log-likelihood mean and dispersion tests, and for ml_tau2 the
ladder statistics.
Detection events: the shipped file is used and checked against recomputation from measurements + sweep bits on all
selected shots (--recompute-check all; a mismatch aborts the leaf).
ML decoders: ml_exact (d=3; fast compact/batched kernel), ml_exact_ref (reference kernel), ml_sparse:<tau>[:<kmax>],
ml_tau2:<tau1>/<tau2>:<band> (the real-data protocol at d >= 5: every shot at tau1, re-decoded at tau2 when
|P - 1/2| < band), ml_adaptive:<tau1>/<tau2>/..[:kappa] (dropped-mass ladder, screening only), ml_mps:<chi>.
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from willow_data import find_experiments, Experiment
from dem_utils import parse_dem

KMAX_LADDER = 2_000_000


def eps_per_cycle(E, r):
    E = min(max(E, 0.0), 0.5 - 1e-12)
    return 0.5 * (1 - (1 - 2 * E) ** (1.0 / max(r, 1)))


def pick_dem(exp: Experiment, which: str):
    if which == "circuit_si1000":
        for name, path in exp.noisy_circuits.items():
            if "si1000" in name.lower() or "noisy" in name.lower():
                c = stim.Circuit.from_file(path)
                return c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True), name
        raise FileNotFoundError("no noisy circuit")
    cands = [k for k in exp.dems if which.lower() in k.lower()]
    if not cands:
        raise FileNotFoundError(f"no DEM matching {which}: have {sorted(exp.dems)}")
    cands.sort(key=len)
    return stim.DetectorErrorModel.from_file(exp.dems[cands[0]]), cands[0]


def _llr(post):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(post[:, 1]) - np.log(post[:, 0])).astype(np.float32)


def _ml_out(post, logp):
    P1 = post[:, 1]
    return {"pred": (np.nan_to_num(P1, nan=0.0) > 0.5)[:, None].astype(np.uint8), "llr": _llr(post), "logp": logp}


def make_decoder(name, dem):
    """Returns f(det_uint8) -> dict(pred (n,1) uint8 [, llr float32, logp float64, extra arrays...])."""
    if name in ("ml_exact", "ml_exact_ref"):
        # ml_exact = FastExactDecoder (compact slots + 16-shot batches; bitwise-identical to the reference
        # ExactFrontierDecoder on d=3 tests); ml_exact_ref = the reference implementation
        if name == "ml_exact":
            from ml_exact_fast import FastExactDecoder as ExactFrontierDecoder
        else:
            from ml_exact import ExactFrontierDecoder
        D = ExactFrontierDecoder(parse_dem(dem))
        return lambda det: _ml_out(*D.class_probs(det))
    if name.startswith("ml_sparse"):
        from ml_sparse import SparseFrontierDecoder
        parts = name.split(":")
        tau = float(parts[1]) if len(parts) > 1 else 1e-8
        kmax = int(parts[2]) if len(parts) > 2 else 200000
        D = SparseFrontierDecoder(parse_dem(dem), tau=tau, kmax=kmax)
        return lambda det: _ml_out(*D.class_probs(det))
    if name.startswith("ml_tau2"):
        # the real-data protocol (methods M1): every shot at tau1; shots with |P - 1/2| < band (or a NaN posterior)
        # are decoded again at tau2, whose answer is final. Both rungs are stored for the convergence report.
        from ml_sparse import SparseFrontierDecoder
        parts = name.split(":")
        t1, t2 = (float(x) for x in parts[1].split("/")) if len(parts) > 1 else (1e-6, 1e-8)
        band = float(parts[2]) if len(parts) > 2 else 0.1
        dm = parse_dem(dem)
        D1 = SparseFrontierDecoder(dm, tau=t1, kmax=KMAX_LADDER)
        D2 = SparseFrontierDecoder.__new__(SparseFrontierDecoder)
        D2.dm, D2.tau, D2.kmax, D2.prog = dm, t2, KMAX_LADDER, D1.prog
        def f(det):          # rung times are CPU seconds of this process (immune to sharing a core)
            t0 = time.process_time(); post, logp = D1.class_probs(det); s1 = time.process_time() - t0
            maxk1 = D1.last_maxk.copy()
            P1 = post[:, 1]
            with np.errstate(invalid="ignore"):
                unc = ~(np.abs(P1 - 0.5) >= band)
            out = _ml_out(post, logp)
            out.update(llr_t1=out["llr"].copy(), logp_t1=logp.copy(), redecoded=unc, maxk_t1=maxk1,
                       sec_t1=np.array(s1), sec_t2=np.array(0.0), maxk_t2=np.zeros(len(det), np.int64))
            idx = np.where(unc)[0]
            if len(idx):
                t0 = time.process_time(); p2, l2 = D2.class_probs(det[idx]); out["sec_t2"] = np.array(time.process_time() - t0)
                o2 = _ml_out(p2, l2)
                for k in ("pred", "llr", "logp"):
                    out[k][idx] = o2[k]
                out["maxk_t2"][idx] = D2.last_maxk
            return out
        return f
    if name.startswith("ml_adaptive"):
        # ml_adaptive:<tau1>/<tau2>/...[:kappa]  e.g. ml_adaptive:1e-4/1e-6/1e-8:2
        from ml_sparse import AdaptiveSparseDecoder
        parts = name.split(":")
        taus = [float(t) for t in parts[1].split("/")] if len(parts) > 1 else [1e-4, 1e-6, 1e-8]
        kappa = float(parts[2]) if len(parts) > 2 else 2.0
        D = AdaptiveSparseDecoder(parse_dem(dem), taus=taus, kappa=kappa)
        return lambda det: _ml_out(*D.class_probs(det))
    if name.startswith("ml_mps"):
        from ml_mps_fast import FastMPSDecoder
        parts = name.split(":")
        chi = int(parts[1]) if len(parts) > 1 else 16
        D = FastMPSDecoder(parse_dem(dem), chi=chi, cutoff=1e-14, reweight=True)
        return lambda det: _ml_out(*D.class_probs(det))
    import baselines
    if name in ("mwpm", "corr_mwpm", "beliefmatch"):
        # matching-based decoders need a graphlike (decomposed) DEM, e.g. a shipped matching prior
        import pymatching
        try:
            pymatching.Matching.from_detector_error_model(dem)
        except Exception:
            raise RuntimeError("DEM not graphlike; pass a decomposed DEM (e.g. shipped matching prior) for matching decoders")
    D = baselines.make(name, dem, dem)
    return lambda det: {"pred": D.decode(det)}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def parse_sel(s, n_total, max_shots):
    if s:
        st, sp, stp = (int(x) if x else None for x in (s.split(":") + ["", ""])[:3])
        idx = np.arange(n_total)[slice(st, sp, stp)]
    else:
        idx = np.arange(n_total)
    return idx[:max_shots]


def paired2x2(px, py, obs):
    ex = px != obs; ey = py != obs
    return {"x_wrong_y_right": int(np.sum(ex & ~ey)), "x_right_y_wrong": int(np.sum(~ex & ey)),
            "both_wrong": int(np.sum(ex & ey)), "n": int(len(obs))}


def ladder_stats(z, obs, r, chunk=None):
    """Convergence report of ml_tau2 (per leaf): rung-1 vs final decisions, posterior shifts, cost. The tau1 cost is the
    median over chunks of CPU ms per round per shot, so the JIT warm-up of a process's first chunk does not count."""
    red = z["redecoded"].astype(bool)
    P_t1 = 1 / (1 + np.exp(-z["llr_t1"].astype(np.float64)))
    P_f = 1 / (1 + np.exp(-z["llr"].astype(np.float64)))
    d1 = np.nan_to_num(P_t1, nan=0.0) > 0.5; df = z["pred"][:, 0].astype(bool)
    e1 = int(np.sum(d1 != obs)); ef = int(np.sum(df != obs))
    dP = np.abs(P_f[red] - P_t1[red])
    n = len(obs)
    return {"n": n, "n_redecoded": int(red.sum()), "share_redecoded": float(red.mean()) if n else None,
            "decisions_changed": int(np.sum(d1 != df)), "errors_tau1": e1, "errors_final": ef,
            "rel_change_in_errors": (ef - e1) / max(ef, 1),
            "max_abs_dP_redecoded": float(np.nanmax(dP)) if dP.size else 0.0,
            "n_dP_gt_0.05": int(np.sum(dP > 0.05)), "nan_final": int(np.isnan(P_f).sum()),
            "ms_per_round_per_shot_tau1": _median_rate(z["sec_t1"], n, r, chunk),
            "ms_per_round_per_shot_tau1_mean": 1e3 * float(np.sum(z["sec_t1"])) / max(n, 1) / max(r, 1),
            "ms_per_round_per_redecoded_shot_tau2": 1e3 * float(z["sec_t2"].sum()) / max(int(red.sum()), 1) / max(r, 1),
            "max_K_tau1": int(z["maxk_t1"].max()) if n else 0, "max_K_tau2": int(z["maxk_t2"].max()) if n else 0}


def _median_rate(sec, n, r, chunk):
    sec = np.atleast_1d(np.asarray(sec, float))
    if not chunk or len(sec) < 2:
        return 1e3 * float(sec.sum()) / max(n, 1) / max(r, 1)
    sizes = np.array([min(chunk, n - i * chunk) for i in range(len(sec))], float)
    return float(np.median(1e3 * sec / np.maximum(sizes, 1) / max(r, 1)))


def decode_leaf(a, folder, T0):
    exp = Experiment(folder)
    rel = os.path.relpath(folder, a.root)
    det_all, obs_all = exp.detection_events(), exp.obs_flips_actual()
    dtag = (a.dem_tag or (a.dem_file and a.dem_file.split("/")[-2]) or a.dem).replace("/", "_").replace(":", "_")
    so = os.path.join(a.out, rel, dtag)
    os.makedirs(so, exist_ok=True)
    r = exp.meta.get("rounds") or int(exp.meta.get("props", {}).get("rounds", 0) or 0)
    summ = {"experiment": rel, "meta": exp.meta, "status": "running", "dem_tag": dtag, "rounds": r,
            "job": {k: getattr(a, k) for k in ("decoders", "dem", "dem_file", "shots", "max_shots", "chunk", "synd_ref",
                                                "time_budget", "recompute_check")},
            "versions": {"stim": stim.__version__, "numpy": np.__version__,
                         "decode_willow_md5": md5_file(os.path.abspath(__file__))},
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    def finish(status, **kw):
        summ.update(kw); summ["status"] = status
        summ["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(summ, open(os.path.join(so, "summary.json.tmp"), "w"), indent=1, default=str)
        os.replace(os.path.join(so, "summary.json.tmp"), os.path.join(so, "summary.json"))
        print(f"{rel} [{dtag}]: {status}", flush=True)
    if det_all is None or obs_all is None:
        return finish("abort_missing_shots")
    n_total = len(det_all)
    idx = parse_sel(a.shots, n_total, a.max_shots)
    n = len(idx)
    det, obs = det_all[idx], obs_all[idx]
    summ.update(n_total=int(n_total), n=int(n), selection=a.shots or f"0:{n_total}:1", p_det=float(det.mean()))
    try:     # does the selection hold every initialization bitstring in its full-leaf proportion?
        sw = exp.sweep_bits()
        if sw is not None and len(sw) == n_total:
            from willow_stage import pattern_ids
            pid = pattern_ids(sw)
            full = np.bincount(pid) / n_total; share = np.bincount(pid[idx], minlength=len(full)) / max(n, 1)
            summ["selection_balance"] = {"patterns": int(len(full)), "patterns_selected": int(np.sum(share > 0)),
                                         "max_share_deviation": float(np.max(np.abs(share / full - 1)))}
    except Exception as ex:
        summ["selection_balance"] = {"error": repr(ex)}
    # detection events: shipped vs recomputed from measurements + sweep bits
    chk = {"mode": a.recompute_check}
    if a.recompute_check != "none":
        try:
            d2, o2 = exp.recompute_detection_events()
            if d2 is None:
                chk["result"] = "no measurements shipped"
            else:
                sel = idx if a.recompute_check == "all" else idx[:1000]
                dm_ = np.any(d2[sel] != det_all[sel], axis=1); om_ = np.any(o2[sel] != obs_all[sel], axis=1)
                chk.update(n_checked=int(len(sel)), det_mismatch_shots=int(dm_.sum()), obs_mismatch_shots=int(om_.sum()),
                           sweep_bits_used=exp.sweep_bits() is not None)
                del d2, o2
                if dm_.any() or om_.any():
                    summ["recompute_check"] = chk
                    return finish("abort_detection_mismatch")
        except Exception as ex:
            chk["result"] = f"could not recompute: {ex!r}"
    summ["recompute_check"] = chk
    # DEM
    try:
        if a.dem_file:
            p = os.path.join(folder, a.dem_file)
            dem, demname = stim.DetectorErrorModel.from_file(p), a.dem_file
            summ["dem_md5"] = md5_file(p)
        else:
            dem, demname = pick_dem(exp, a.dem)
    except Exception as ex:
        return finish("abort_no_dem", error=repr(ex))
    summ["dem"] = demname
    summ["dem_num_detectors"] = dem.num_detectors
    if dem.num_detectors != det.shape[1] or dem.num_observables != obs.shape[1]:
        return finish("abort_dem_shape", error=f"DEM has {dem.num_detectors} detectors / {dem.num_observables} observables, "
                                              f"data {det.shape[1]} / {obs.shape[1]}")
    # shipped predictions (e.g. Libra, Harmony, correlated matching), stored once per leaf and DEM tag
    shipped = {k: v[idx] for k, v in exp.predictions().items() if not isinstance(v, str) and len(v) == n_total}
    np.savez_compressed(os.path.join(so, "shots.npz"), idx=idx.astype(np.int32), obs=obs[:, 0].astype(np.uint8),
                        shipped_names=np.array(sorted(shipped)),
                        shipped=np.stack([shipped[k][:, 0] for k in sorted(shipped)]).astype(np.uint8) if shipped else np.zeros((0, n), np.uint8))
    spec = {"leaf": rel, "dem_md5": summ.get("dem_md5", demname), "selection": summ["selection"], "max_shots": a.max_shots,
            "chunk": a.chunk, "n": n}
    wroot = os.path.join(a.work or a.out, rel, dtag)
    preds, status, summ["decoders"] = {}, "complete", {}
    for name in a.decoders.split(","):
        dn = name.replace(":", "_").replace("/", "-")
        wd = os.path.join(wroot, dn)
        os.makedirs(wd, exist_ok=True)
        sp = os.path.join(wd, "spec.json")
        if os.path.exists(sp) and json.load(open(sp)) != dict(spec, decoder=name):
            for fn in os.listdir(wd):
                if fn.startswith("chunk_"):
                    os.remove(os.path.join(wd, fn))
        json.dump(dict(spec, decoder=name), open(sp, "w"))
        f, parts, secs = None, [], 0.0
        for c0 in range(0, n, a.chunk):
            fn = os.path.join(wd, f"chunk_{c0:06d}.npz")
            if os.path.exists(fn):
                z = np.load(fn); parts.append({k: z[k] for k in z.files}); secs += float(z["seconds"]); continue
            if a.time_budget and time.time() - T0 > a.time_budget:
                status = "partial"; break
            if f is None:
                try:
                    f = make_decoder(name, dem)
                except MemoryError as ex:
                    summ["decoders"][name] = {"skipped": repr(ex)}; break
                except RuntimeError as ex:
                    summ["decoders"][name] = {"skipped": repr(ex)}; break
            t0, w0 = time.process_time(), time.time()
            res = f(det[c0:c0 + a.chunk])
            res["seconds"] = np.array(time.process_time() - t0)         # CPU seconds (cost); wall time kept apart
            res["wall_seconds"] = np.array(time.time() - w0)
            np.savez_compressed(fn + ".tmp.npz", **res)
            os.replace(fn + ".tmp.npz", fn)
            parts.append(res); secs += float(res["seconds"])
        nd = sum(len(p["pred"]) for p in parts)
        if nd == 0:
            continue
        keys = [k for k in parts[0] if all(k in p for p in parts)]        # chunks written by older versions lack keys
        cat = {k: (np.concatenate([p[k] for p in parts]) if np.ndim(parts[0][k]) else np.array([float(p[k]) for p in parts]))
               for k in keys}
        cat["idx"] = idx[:nd].astype(np.int32)
        np.savez_compressed(os.path.join(so, dn + ".npz.tmp.npz"), **cat)
        os.replace(os.path.join(so, dn + ".npz.tmp.npz"), os.path.join(so, dn + ".npz"))
        pred = cat["pred"]; o = obs[:nd, 0]
        preds[name] = pred[:, 0]
        k = int(np.sum(pred[:, 0] != o))
        ds = {"n": int(nd), "errors": k, "LER": k / nd, "eps_per_cycle": eps_per_cycle(k / nd, r), "ms_per_shot": 1e3 * secs / nd,
              "cpu_ms_per_round_per_shot_median_chunk": _median_rate(cat["seconds"], nd, r, a.chunk)}
        if "llr" in cat:
            llr = cat["llr"].astype(np.float64)
            P1 = 1 / (1 + np.exp(-llr))
            ds["nan_posteriors"] = int(np.isnan(llr).sum())
            from mismatch import calibration
            ds["calibration"] = calibration(P1, o)
            ds["predicted_LER_if_DEM_true"] = ds["calibration"]["LER_predicted"]
            if "redecoded" in cat:
                ds["ladder"] = ladder_stats(cat, o, r, a.chunk)
            if a.synd_ref > 0 and nd == n:
                rf = os.path.join(wd, f"synd_ref_{a.synd_ref}.npz")
                if os.path.exists(rf):
                    lref = np.load(rf)["logp"]
                else:
                    if f is None:
                        f = make_decoder(name, dem)
                    dref = dem.compile_sampler(seed=20261007).sample(a.synd_ref)[0].astype(np.uint8)
                    lref = np.concatenate([f(dref[c:c + a.chunk])["logp"] for c in range(0, a.synd_ref, a.chunk)])
                    np.savez_compressed(rf, logp=lref)
                np.savez_compressed(os.path.join(so, f"synd_ref_{dn}.npz"), logp=lref)
                from analyze_synd_ll import ll_tests
                ds["syndrome_loglik"] = ll_tests(cat["logp"], lref)
                ds["syndrome_loglik"]["nonfinite_data"] = int((~np.isfinite(cat["logp"])).sum())
        summ["decoders"][name] = ds
        print(f"{rel} [{dtag}] {name}: n={nd} LER={k/nd:.5f} eps={eps_per_cycle(k/nd, r):.3e} {1e3*secs/nd:.2f} ms/shot", flush=True)
        if status == "partial":
            break
    for kname, v in zip(list(shipped), [shipped[k] for k in shipped]):
        e = float(np.mean(v[:, 0] != obs[:, 0]))
        summ["decoders"]["shipped:" + kname] = {"n": int(n), "errors": int(np.sum(v[:, 0] != obs[:, 0])), "LER": e,
                                                "eps_per_cycle": eps_per_cycle(e, r)}
        preds["shipped:" + kname] = v[:, 0]
    pc, names = {}, list(preds)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            m = min(len(preds[x]), len(preds[y]))
            pc[f"{x} vs {y}"] = paired2x2(preds[x][:m], preds[y][:m], obs[:m, 0])
    summ["paired"] = pc
    return finish(status, seconds=time.time() - T0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="local folder for chunk checkpoints (default: --out)")
    ap.add_argument("--decoders", default="ml_exact,mwpm,corr_mwpm")
    ap.add_argument("--dem", default="si1000", help="substring of a shipped .dem path, or circuit_si1000")
    ap.add_argument("--dem-file", default=None, help="exact DEM path relative to the leaf (overrides --dem)")
    ap.add_argument("--dem-tag", default=None, help="output subfolder name (default: derived from --dem)")
    ap.add_argument("--filter", default="", help="substring that experiment paths must contain (comma = all of)")
    ap.add_argument("--leaf", default=None, help="exact leaf path(s) relative to --root, comma-separated (overrides --filter)")
    ap.add_argument("--max-shots", type=int, default=50000)
    ap.add_argument("--shots", default=None, help="start:stop:step selection of shot indices (then capped by --max-shots)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--synd-ref", type=int, default=0, help="fresh DEM samples for the syndrome log-likelihood tests (0 = off)")
    ap.add_argument("--time-budget", type=float, default=0.0, help="seconds; no new chunk starts after this (status partial)")
    ap.add_argument("--recompute-check", default="all", choices=["all", "first1000", "none"])
    a = ap.parse_args()
    if a.leaf:
        exps = [os.path.join(a.root, x) for x in a.leaf.split(",")]
    else:
        exps = [e for e in find_experiments(a.root) if all(s in os.path.relpath(e, a.root) for s in a.filter.split(",") if s)]
    print(f"{len(exps)} experiments", flush=True)
    T0 = time.time()
    for folder in exps:
        decode_leaf(a, folder, T0)


if __name__ == "__main__":
    main()
