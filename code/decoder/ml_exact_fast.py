"""Faster exact coset-ML decoder for small frontier width (d = 3): same contraction as
ml_exact.ExactFrontierDecoder, with two changes that leave the arithmetic exact:
 1. compact slots: open detectors always occupy the lowest bits (a closed slot is refilled by moving the top
    slot into it), so every operation touches only 2^(n_open + n_obs) entries instead of 2^(W + n_obs)
    (at d = 3 the mean of 2^n_open / 2^W over the schedule is about 0.34);
 2. shot batching: the table is F[index, lane] with B shots in contiguous lanes, so the inner loops vectorise
    (the schedule is shot-independent; only the projection at a detector close depends on the syndrome).
Index layout: index = (x << n_obs) | c, x = parities of open detectors, c = logical class.
"""
import numpy as np
import numba as nb
from dem_utils import DemMatrices, best_sweep_order


def build_compact_schedule(dm: DemMatrices, order=None):
    """ops: type 0 fault (a1 = index mask, pp = p), 1 close (a1 = bit position, a2 = detector),
    2 move (a1 = from bit, a2 = to bit). nbits[k] = index bits in use during op k."""
    if order is None:
        order = best_sweep_order(dm)[0]
    nobs = dm.num_observables
    first, last = {}, {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            first.setdefault(d, pos)
            last[d] = pos
    closes_after = {}
    for d, pos in last.items():
        closes_after.setdefault(pos, []).append(d)
    slot_of, det_in = {}, []
    typ, a1, a2, pp, nbits = [], [], [], [], []
    for pos, j in enumerate(order):
        m = 0
        for d in dm.dets[j]:
            if d not in slot_of:
                slot_of[d] = len(det_in); det_in.append(d)
            m |= 1 << slot_of[d]
        typ.append(0); a1.append((m << nobs) | int(dm.obs[j])); a2.append(0); pp.append(float(dm.p[j]))
        nbits.append(len(det_in) + nobs)
        for d in closes_after.get(pos, []):
            s = slot_of.pop(d)
            nb_ = len(det_in) + nobs
            typ.append(1); a1.append(s + nobs); a2.append(d); pp.append(0.0); nbits.append(nb_)
            top = len(det_in) - 1
            if s != top:
                dt = det_in[top]; slot_of[dt] = s; det_in[s] = dt
                typ.append(2); a1.append(top + nobs); a2.append(s + nobs); pp.append(0.0); nbits.append(nb_)
            det_in.pop()
    never = np.array([d for d in range(dm.num_detectors) if d not in first], np.int64)
    return (np.array(typ, np.int64), np.array(a1, np.int64), np.array(a2, np.int64), np.array(pp, np.float64),
            np.array(nbits, np.int64), never)


@nb.njit(cache=True, fastmath=False)
def _run_fast(typ, a1, a2, pp, nbits, nobs, syndromes, B, out, out_logp):
    nsh = syndromes.shape[0]
    C = 1 << nobs
    maxbits = 0
    for k in range(nbits.shape[0]):
        if nbits[k] > maxbits:
            maxbits = nbits[k]
    F = np.zeros((1 << maxbits, B))
    sv = np.zeros(B); mx = np.zeros(B); inv = np.zeros(B); logscale = np.zeros(B)
    for b0 in range(0, nsh, B):
        nl = min(B, nsh - b0)
        F[:, :] = 0.0
        for b in range(B):
            F[0, b] = 1.0
            logscale[b] = 0.0
        for k in range(typ.shape[0]):
            size = 1 << nbits[k]
            t = typ[k]
            if t == 0:
                M = a1[k]
                p = pp[k]; q = 1.0 - p
                hb = 0; tt = M
                while tt > 1:
                    tt >>= 1; hb += 1
                low = (1 << hb) - 1
                for xp in range(size >> 1):
                    x = ((xp >> hb) << (hb + 1)) | (xp & low)
                    y = x ^ M
                    for b in range(B):
                        u = F[x, b]; v = F[y, b]
                        F[x, b] = q * u + p * v
                        F[y, b] = p * u + q * v
            elif t == 1:
                bp = a1[k]; bit = 1 << bp; low = bit - 1; d = a2[k]
                for b in range(B):
                    sv[b] = syndromes[b0 + b, d] if b < nl else 0.0
                    mx[b] = 0.0
                for xp in range(size >> 1):
                    x = ((xp >> bp) << (bp + 1)) | (xp & low)
                    x1 = x | bit
                    for b in range(B):
                        s = sv[b]
                        v = s * F[x1, b] + (1.0 - s) * F[x, b]
                        F[x, b] = v
                        F[x1, b] = 0.0
                        if v > mx[b]:
                            mx[b] = v
                for b in range(B):
                    if mx[b] > 0.0:
                        inv[b] = 1.0 / mx[b]
                        logscale[b] += np.log(mx[b])
                    else:
                        inv[b] = 0.0
                        logscale[b] = -np.inf
                for x in range(size):
                    for b in range(B):
                        F[x, b] *= inv[b]
            else:
                # move slot at bit fb (top) into the just-closed bit tb < fb: only entries with bit tb = 0
                # are populated after the close, so visit x with bit fb = 1, bit tb = 0
                fb = a1[k]; tb = a2[k]
                fbit = 1 << fb; tbit = 1 << tb
                for xp in range(size >> 2):
                    v1 = ((xp >> tb) << (tb + 1)) | (xp & (tbit - 1))
                    x = ((v1 >> fb) << (fb + 1)) | (v1 & (fbit - 1)) | fbit
                    dst = (x ^ fbit) | tbit
                    for b in range(B):
                        F[dst, b] = F[x, b]
                        F[x, b] = 0.0
        for b in range(nl):
            for c in range(C):
                out[b0 + b, c] = F[c, b]
            out_logp[b0 + b] = logscale[b]


class FastExactDecoder:
    """Drop-in replacement for ml_exact.ExactFrontierDecoder (class_probs / decode)."""

    def __init__(self, dm: DemMatrices, order=None, batch=16, max_bytes=1.5e9):
        self.dm = dm
        self.typ, self.a1, self.a2, self.pp, self.nbits, self.never = build_compact_schedule(dm, order)
        self.nobs = dm.num_observables
        self.W = int(self.nbits.max()) - self.nobs
        self.B = batch
        need = 8.0 * (2 ** int(self.nbits.max())) * batch
        if need > max_bytes:
            raise MemoryError(f"exact frontier needs {need/1e9:.1f} GB (W={self.W}); use ml_sparse")
        self.work_ratio = float(np.mean(2.0 ** self.nbits[self.typ == 0]) / 2.0 ** self.nbits.max())

    def class_probs(self, syndromes):
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        C = 1 << self.nobs
        out = np.zeros((n, C)); logp = np.zeros(n)
        if n:
            _run_fast(self.typ, self.a1, self.a2, self.pp, self.nbits, self.nobs, syn, self.B, out, logp)
        for om, pr in self.dm.undetectable:
            out = (1 - pr) * out + pr * out[:, np.arange(C) ^ om]
        if len(self.never):
            out[syn[:, self.never].any(axis=1)] = np.nan
        tot = out.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            logp = logp + np.log(np.where(tot > 0, tot, np.nan))
            post = out / tot[:, None]
        return post, logp

    def decode(self, syndromes):
        return np.argmax(self.class_probs(syndromes)[0], axis=1)
