"""Exact maximum-likelihood (coset) decoding of a detector error model.

ExactFrontierDecoder: variable elimination over faults in a sweep order (time is the long axis).
The state is the dense table F[x, c] over the parities x of the currently "open" detectors
(touched by some processed and some unprocessed fault) and the logical class c. A fault j with
symptom mask m_j and observable mask o_j acts as F <- (1-p_j) F + p_j F[x^m_j, c^o_j]; a detector is
"closed" after its last fault by projecting its parity bit onto the observed syndrome bit.
This is exactly the contraction of the DEM tensor network along a fixed elimination order; its cost
is O(#faults * 2^(W+n_obs)) per shot with W the maximal number of open detectors.

brute_force_posterior: enumeration over all 2^M fault configurations (validation only, M <= ~22).
"""
import numpy as np
import numba as nb
from dem_utils import DemMatrices, best_sweep_order, frontier_profile


def build_schedule(dm: DemMatrices, order=None):
    if order is None:
        order = best_sweep_order(dm)[0]
    first, last = {}, {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            first.setdefault(d, pos)
            last[d] = pos
    closes_after = {}
    for d, pos in last.items():
        closes_after.setdefault(pos, []).append(d)
    free = []
    slot_of = {}
    next_slot = 0
    # op arrays: type 0 = fault (mask, obsmask, p), type 1 = close (slot, detector)
    typ, a1, a2, pp = [], [], [], []
    for pos, j in enumerate(order):
        m = 0
        for d in dm.dets[j]:
            if d not in slot_of:
                if free:
                    s = free.pop()
                else:
                    s = next_slot; next_slot += 1
                slot_of[d] = s
            m |= (1 << slot_of[d])
        typ.append(0); a1.append(m); a2.append(int(dm.obs[j])); pp.append(float(dm.p[j]))
        for d in closes_after.get(pos, []):
            s = slot_of.pop(d)
            typ.append(1); a1.append(s); a2.append(d); pp.append(0.0)
            free.append(s)
    W = next_slot
    never = [d for d in range(dm.num_detectors) if d not in first]
    return (np.array(typ, np.int64), np.array(a1, np.int64), np.array(a2, np.int64),
            np.array(pp, np.float64), W, np.array(never, np.int64))


@nb.njit(cache=True, fastmath=False)
def _run_frontier(typ, a1, a2, pp, W, nobs, syndromes, out_logp):
    nshots = syndromes.shape[0]
    C = 1 << nobs
    size = (1 << W) * C
    out = np.zeros((nshots, C))
    F = np.zeros(size)
    for sh in range(nshots):
        F[:] = 0.0
        F[0] = 1.0
        logscale = 0.0
        for k in range(typ.shape[0]):
            if typ[k] == 0:
                M = a1[k] | (a2[k] << W)
                p = pp[k]; q = 1.0 - p
                hb = 0
                t = M
                while t > 1:
                    t >>= 1; hb += 1
                hbit = 1 << hb
                for x in range(size):
                    if x & hbit:
                        continue
                    y = x ^ M
                    a = F[x]; b = F[y]
                    F[x] = q * a + p * b
                    F[y] = p * a + q * b
            else:
                s = a1[k]
                bit = 1 << s
                synd = syndromes[sh, a2[k]]
                mx = 0.0
                for x in range(size):
                    if x & bit:
                        continue
                    if synd:
                        v = F[x | bit]
                    else:
                        v = F[x]
                    F[x] = v
                    F[x | bit] = 0.0
                    if v > mx:
                        mx = v
                if mx > 0.0:
                    inv = 1.0 / mx
                    for x in range(size):
                        F[x] *= inv
                    logscale += np.log(mx)
                else:
                    logscale = -np.inf
        for c in range(C):
            out[sh, c] = F[c << W]
        out_logp[sh] = logscale
    return out


class ExactFrontierDecoder:
    def __init__(self, dm: DemMatrices, order=None, max_bytes=1.5e9):
        self.dm = dm
        self.typ, self.a1, self.a2, self.pp, self.W, self.never = build_schedule(dm, order)
        self.nobs = dm.num_observables
        need = 8.0 * (2 ** self.W) * (2 ** self.nobs)
        if need > max_bytes:
            raise MemoryError(f"exact frontier needs {need/1e9:.1f} GB (W={self.W}, nobs={self.nobs}) > max_bytes; "
                              "use ml_sparse / ml_mps or raise max_bytes")

    def class_probs(self, syndromes):
        """Return (P[c|s] array (nshots, 2^nobs), log P(s)) for a uint8/bool array (nshots, num_detectors)."""
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        logp = np.zeros(syn.shape[0])
        out = _run_frontier(self.typ, self.a1, self.a2, self.pp, self.W, self.nobs, syn, logp)
        # syndrome-independent (undetectable) logical flips
        for om, pr in self.dm.undetectable:
            C = out.shape[1]
            idx = np.arange(C) ^ om
            out = (1 - pr) * out + pr * out[:, idx]
        if len(self.never):
            bad = syn[:, self.never].any(axis=1)
            out[bad] = np.nan
        tot = out.sum(axis=1)
        logp = logp + np.log(np.where(tot > 0, tot, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            post = out / tot[:, None]
        return post, logp

    def decode(self, syndromes):
        post, _ = self.class_probs(syndromes)
        return np.argmax(post, axis=1)


def brute_force_table(dm: DemMatrices):
    """Enumerate all 2^M fault configurations; return dict syndrome_bytes -> array over classes."""
    M = dm.num_faults
    assert M <= 24, "brute force only for tiny DEMs"
    C = 1 << dm.num_observables
    smask = np.zeros(M, dtype=np.int64)
    for j, ds in enumerate(dm.dets):
        for d in ds:
            smask[j] ^= (1 << d)
    table = {}
    configs = np.arange(1 << M, dtype=np.int64)
    syn = np.zeros(1 << M, dtype=np.int64)
    cls = np.zeros(1 << M, dtype=np.int64)
    logw = np.zeros(1 << M)
    for j in range(M):
        on = (configs >> j) & 1
        syn ^= np.where(on == 1, smask[j], 0)
        cls ^= np.where(on == 1, dm.obs[j], 0)
        logw += np.where(on == 1, np.log(dm.p[j]), np.log1p(-dm.p[j]))
    w = np.exp(logw)
    for om, pr in dm.undetectable:
        pass  # handled by caller
    keys = syn * C + cls
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=w)
    for k, v in zip(uniq, sums):
        s, c = divmod(int(k), C)
        table.setdefault(s, np.zeros(C))[c] += v
    return table
