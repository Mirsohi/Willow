"""scout: independent exact maximum-likelihood decoder for small Stim DEMs, by bucket elimination over detectors.

Algorithm (different from a tensor-network contraction, used only to cross-check decoder's TN-ML numbers):
  Errors are processed in order of their largest detector index. A table T over the parities of "open" detectors
  (touched by a processed error and by an unprocessed one) plus the logical observable bit is kept as an ndarray with
  one axis of length 2 per bit (and a leading batch axis over shots). Applying error e with probability p:
      T <- (1-p) T + p flip(T, axes(e))      (flipping a length-2 axis == XOR of that bit)
  A detector is closed right after the last error touching it: T <- T[..., s_j, ...] (the observed syndrome bit).
  At the end T[b, l] = P(syndrome_b, L = l) up to a per-shot normalisation; decide argmax_l.
Exact for any DEM (hyperedges allowed); cost ~ (#errors) * 2^(max frontier) per batch. Practical for d=3 surface-code
memories (frontier ~ 2 rounds of detectors), not for d=5.
Only one logical observable (L0) is supported.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import stim


def dem_errors(dem: stim.DetectorErrorModel):
    """Flatten a DEM into a list of (p, detectors tuple, flips_L0 bool). Separators (^) are merged (XOR) as in Stim's
    semantics for decomposed errors (the decomposition is a hint only)."""
    errs = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = set(), False
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                if t.val != 0:
                    raise ValueError("only L0 supported")
                obs = not obs
        errs.append((p, tuple(sorted(dets)), obs))
    return errs


class DPDecoder:
    def __init__(self, dem: stim.DetectorErrorModel):
        self.n_det = dem.num_detectors
        errs = [e for e in dem_errors(dem) if e[0] > 0 and (e[1] or e[2])]
        # order errors by max detector index (errors touching only L0 go first)
        errs.sort(key=lambda e: (max(e[1]) if e[1] else -1, min(e[1]) if e[1] else -1))
        self.errs = errs
        last = {}
        for k, (_, dets, _) in enumerate(errs):
            for d in dets:
                last[d] = k
        self.close_after = {}
        for d, k in last.items():
            self.close_after.setdefault(k, []).append(d)
        self.untouched = [d for d in range(self.n_det) if d not in last]
        # dry run for the frontier width
        open_, width = set(), 0
        for k, (_, dets, _) in enumerate(errs):
            open_ |= set(dets)
            width = max(width, len(open_))
            for d in self.close_after.get(k, []):
                open_.discard(d)
        self.max_frontier = width

    def posterior(self, syndromes: np.ndarray, return_logp=False):
        """syndromes: (B, n_det) bool. Returns (B,) P(L0 = 1 | s) (and natural log P(s) if return_logp). Shots whose
        syndrome has a fired detector that no error can explain get NaN (logp = -inf)."""
        S = np.asarray(syndromes, dtype=np.int64)
        B = S.shape[0]
        T = np.ones((B, 2))  # axis 1 = L0 bit
        T[:, 1] = 0.0
        axes = ["L"]  # names of non-batch axes, in order
        bad = np.zeros(B, bool)
        logp = np.zeros(B)
        for d in self.untouched:
            bad |= S[:, d] == 1
        for k, (p, dets, obs) in enumerate(self.errs):
            for d in dets:
                if d not in axes:
                    T = np.stack([T, np.zeros_like(T)], axis=-1)
                    axes.append(d)
            ax = [1 + axes.index(d) for d in dets]
            if obs:
                ax.append(1 + axes.index("L"))
            T = (1.0 - p) * T + p * np.flip(T, axis=tuple(ax))
            for d in self.close_after.get(k, []):
                a = 1 + axes.index(d)
                T = np.moveaxis(T, a, 1)
                T = T[np.arange(B), S[:, d]]
                axes.remove(d)
                norm = T.reshape(B, -1).sum(1)
                with np.errstate(divide="ignore"):
                    logp += np.log(norm)
                norm[norm == 0] = 1.0
                T = T / norm.reshape((B,) + (1,) * (T.ndim - 1))
        assert axes == ["L"], axes
        tot = T.sum(1)
        post = np.where(tot > 0, T[:, 1] / np.where(tot > 0, tot, 1), np.nan)
        post[bad] = np.nan
        if return_logp:
            with np.errstate(divide="ignore"):
                logp = logp + np.log(tot)
            logp[bad] = -np.inf
            return post, logp
        return post

    def posterior_chunked(self, syndromes, max_entries=2 ** 23, return_logp=False):
        """Same as posterior, in shot chunks so that a table has at most ~max_entries doubles."""
        S = np.asarray(syndromes)
        per = 2 ** (self.max_frontier + 1)
        chunk = max(1, max_entries // per)
        outs = [self.posterior(S[i:i + chunk], return_logp) for i in range(0, len(S), chunk)]
        if return_logp:
            return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])
        return np.concatenate(outs)

    def decode(self, syndromes):
        return self.posterior_chunked(syndromes) > 0.5


if __name__ == "__main__":
    import argparse, time, json
    import pymatching
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--basis", default="z")
    a = ap.parse_args()
    circ = stim.Circuit.generated(f"surface_code:rotated_memory_{a.basis}", distance=a.d, rounds=a.rounds,
                                  after_clifford_depolarization=a.p, before_round_data_depolarization=a.p,
                                  before_measure_flip_probability=a.p, after_reset_flip_probability=a.p)
    dem = circ.detector_error_model(decompose_errors=False)
    dec = DPDecoder(dem)
    det, obs = circ.compile_detector_sampler(seed=a.seed).sample(a.shots, separate_observables=True)
    t = time.time()
    post = dec.posterior_chunked(det)
    tml = time.time() - t
    ml = post > 0.5
    m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    mw = m.decode_batch(det)[:, 0].astype(bool)
    o = obs[:, 0]
    out = dict(d=a.d, rounds=a.rounds, p=a.p, shots=a.shots, seed=a.seed, n_err=len(dec.errs),
               max_frontier=dec.max_frontier, ml_fail=int((ml != o).sum()), mwpm_fail=int((mw != o).sum()),
               nan=int(np.isnan(post).sum()), sec_per_shot_ml=tml / a.shots)
    print(json.dumps(out))
