"""Forward-backward exact marginals (as ml_fb_exact.ForwardBackwardExact) on the compact, shot-batched layout
of ml_exact_fast: per-shot log P(s[, L]), every fault posterior q_j = P(e_j = 1 | s[, L]) and the score
d ln P / d theta = sum_j (q_j - p_j)/(p_j (1 - p_j)) dp_j/dtheta. Same results to rounding, several times faster.

Backward table in the z-encoding (z_i = s_i XOR parity of later faults on open detector i, z_L = L XOR later
logical parity), indexed exactly like the forward table at the same cut, so for each fault j
q_j = p_j sum_x F_pre(x) B_post(x ^ m_j) / [sum_x F_pre(x) ((1 - p_j) B_post(x) + p_j B_post(x ^ m_j))].
Reverse ops: move (bit tb -> fb), close (reopen detector with z = s_i), fault (q_j, butterfly, then project the
slots first opened by this fault onto z = 0, i.e. drop index bits >= nb_pre, and renormalise per lane).
"""
import numpy as np
import numba as nb
from dem_utils import DemMatrices, best_sweep_order


def build_fb_schedule(dm: DemMatrices, order=None):
    if order is None:
        order = best_sweep_order(dm)[0]
    nobs = dm.num_observables
    last = {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            last[d] = pos
    closes_after = {}
    for d, pos in last.items():
        closes_after.setdefault(pos, []).append(d)
    slot_of, det_in = {}, []
    typ, a1, a2, pp, nbits, nbpre, fid = [], [], [], [], [], [], []
    for pos, j in enumerate(order):
        pre = len(det_in) + nobs
        m = 0
        for d in dm.dets[j]:
            if d not in slot_of:
                slot_of[d] = len(det_in); det_in.append(d)
            m |= 1 << slot_of[d]
        typ.append(0); a1.append((m << nobs) | int(dm.obs[j])); a2.append(0); pp.append(float(dm.p[j]))
        nbits.append(len(det_in) + nobs); nbpre.append(pre); fid.append(j)
        for d in closes_after.get(pos, []):
            s = slot_of.pop(d)
            nb_ = len(det_in) + nobs
            typ.append(1); a1.append(s + nobs); a2.append(d); pp.append(0.0); nbits.append(nb_); nbpre.append(nb_); fid.append(-1)
            top = len(det_in) - 1
            if s != top:
                dt = det_in[top]; slot_of[dt] = s; det_in[s] = dt
                typ.append(2); a1.append(top + nobs); a2.append(s + nobs); pp.append(0.0); nbits.append(nb_); nbpre.append(nb_); fid.append(-1)
            det_in.pop()
    typ = np.array(typ, np.int64); nbits = np.array(nbits, np.int64)
    foff = np.zeros(len(typ), np.int64)
    tot = 0
    for k in range(len(typ)):
        if typ[k] == 0:
            foff[k] = tot; tot += 1 << nbits[k]
    return dict(typ=typ, a1=np.array(a1, np.int64), a2=np.array(a2, np.int64), pp=np.array(pp, np.float64),
                nbits=nbits, nbpre=np.array(nbpre, np.int64), fid=np.array(fid, np.int64), foff=foff, rows=tot)


@nb.njit(cache=True)
def _ins0(v, b):
    return ((v >> b) << (b + 1)) | (v & ((1 << b) - 1))


@nb.njit(cache=True)
def _run_fb_fast(typ, a1, a2, pp, nbits, nbpre, fid, foff, rows, nobs, syn, Ls, joint, J, B,
                 out_logp, qsum, grads, qall, store_q):
    nsh = syn.shape[0]
    C = 1 << nobs
    maxbits = 0
    for k in range(nbits.shape[0]):
        if nbits[k] > maxbits:
            maxbits = nbits[k]
    F = np.zeros((1 << maxbits, B)); Bt = np.zeros((1 << maxbits, B))
    Fst = np.empty((rows, B))
    sv = np.zeros(B); mx = np.zeros(B); inv = np.zeros(B); ls = np.zeros(B)
    valid = np.zeros(B, np.bool_)
    num = np.zeros(B); den0 = np.zeros(B)
    kp = J.shape[1]
    nops = typ.shape[0]
    for b0 in range(0, nsh, B):
        nl = min(B, nsh - b0)
        F[:, :] = 0.0
        for b in range(B):
            F[0, b] = 1.0; ls[b] = 0.0
        # ---------------- forward ----------------
        for k in range(nops):
            size = 1 << nbits[k]
            t = typ[k]
            if t == 0:
                o = foff[k]
                for x in range(size):
                    for b in range(B):
                        Fst[o + x, b] = F[x, b]
                M = a1[k]; p = pp[k]; q = 1.0 - p
                hb = 0; tt = M
                while tt > 1:
                    tt >>= 1; hb += 1
                for xp in range(size >> 1):
                    x = _ins0(xp, hb)
                    y = x ^ M
                    for b in range(B):
                        u = F[x, b]; v = F[y, b]
                        F[x, b] = q * u + p * v
                        F[y, b] = p * u + q * v
            elif t == 1:
                bp = a1[k]; bit = 1 << bp; d = a2[k]
                for b in range(B):
                    sv[b] = syn[b0 + b, d] if b < nl else 0.0
                    mx[b] = 0.0
                for xp in range(size >> 1):
                    x = _ins0(xp, bp); x1 = x | bit
                    for b in range(B):
                        s = sv[b]
                        v = s * F[x1, b] + (1.0 - s) * F[x, b]
                        F[x, b] = v; F[x1, b] = 0.0
                        if v > mx[b]:
                            mx[b] = v
                for b in range(B):
                    if mx[b] > 0.0:
                        inv[b] = 1.0 / mx[b]; ls[b] += np.log(mx[b])
                    else:
                        inv[b] = 0.0; ls[b] = -np.inf
                for x in range(size):
                    for b in range(B):
                        F[x, b] *= inv[b]
            else:
                fb = a1[k]; tb = a2[k]; fbit = 1 << fb; tbit = 1 << tb
                for xp in range(size >> 2):
                    x = _ins0(_ins0(xp, tb), fb) | fbit
                    dst = (x ^ fbit) | tbit
                    for b in range(B):
                        F[dst, b] = F[x, b]; F[x, b] = 0.0
        for b in range(B):
            valid[b] = False
            if b >= nl:
                continue
            if joint:
                tot = F[Ls[b0 + b], b]
            else:
                tot = 0.0
                for c in range(C):
                    tot += F[c, b]
            if tot > 0.0 and np.isfinite(ls[b]):
                out_logp[b0 + b] = ls[b] + np.log(tot); valid[b] = True
            else:
                out_logp[b0 + b] = -np.inf
        # ---------------- backward (z-encoding) ----------------
        Bt[:, :] = 0.0
        for b in range(B):
            if joint:
                Bt[Ls[b0 + b] if b < nl else 0, b] = 1.0
            else:
                for c in range(C):
                    Bt[c, b] = 1.0
        for k in range(nops - 1, -1, -1):
            size = 1 << nbits[k]
            t = typ[k]
            if t == 2:
                fb = a1[k]; tb = a2[k]; fbit = 1 << fb; tbit = 1 << tb
                for xp in range(size >> 2):
                    x = _ins0(_ins0(xp, tb), fb) | tbit
                    dst = (x ^ tbit) | fbit
                    for b in range(B):
                        Bt[dst, b] = Bt[x, b]; Bt[x, b] = 0.0
            elif t == 1:
                bp = a1[k]; bit = 1 << bp; d = a2[k]
                for b in range(B):
                    sv[b] = syn[b0 + b, d] if b < nl else 0.0
                for xp in range(size >> 1):
                    x = _ins0(xp, bp); x1 = x | bit
                    for b in range(B):
                        v = Bt[x, b]; s = sv[b]
                        Bt[x1, b] = s * v
                        Bt[x, b] = (1.0 - s) * v
            else:
                o = foff[k]; M = a1[k]; p = pp[k]; q = 1.0 - p
                for b in range(B):
                    num[b] = 0.0; den0[b] = 0.0
                for x in range(size):
                    y = x ^ M
                    for b in range(B):
                        f = Fst[o + x, b]
                        num[b] += f * Bt[y, b]
                        den0[b] += f * Bt[x, b]
                j = fid[k]
                for b in range(nl):
                    if not valid[b]:
                        continue
                    den = q * den0[b] + p * num[b]
                    qj = p * num[b] / den if den > 0.0 else 0.0
                    qsum[j] += qj
                    if store_q:
                        qall[b0 + b, j] = qj
                    if kp > 0:
                        fac = (qj - p) / (p * q)
                        for c in range(kp):
                            grads[b0 + b, c] += fac * J[j, c]
                hb = 0; tt = M
                while tt > 1:
                    tt >>= 1; hb += 1
                for xp in range(size >> 1):
                    x = _ins0(xp, hb)
                    y = x ^ M
                    for b in range(B):
                        u = Bt[x, b]; v = Bt[y, b]
                        Bt[x, b] = q * u + p * v
                        Bt[y, b] = p * u + q * v
                npre = 1 << nbpre[k]
                if npre < size:
                    for x in range(npre, size):
                        for b in range(B):
                            Bt[x, b] = 0.0
                    for b in range(B):
                        mx[b] = 0.0
                    for x in range(npre):
                        for b in range(B):
                            if Bt[x, b] > mx[b]:
                                mx[b] = Bt[x, b]
                    for b in range(B):
                        inv[b] = 1.0 / mx[b] if mx[b] > 0.0 else 0.0
                    for x in range(npre):
                        for b in range(B):
                            Bt[x, b] *= inv[b]


class ForwardBackwardFast:
    """Same interface as ml_fb_exact.ForwardBackwardExact."""

    def __init__(self, dm: DemMatrices, order=None, batch=16, max_bytes=1.5e9):
        assert not dm.undetectable, "undetectable faults not supported"
        self.dm = dm
        self.prog = build_fb_schedule(dm, order)
        self.B = batch
        need = 8.0 * self.prog["rows"] * batch
        if need > max_bytes:
            raise MemoryError(f"forward-backward needs {need/1e9:.2f} GB")

    def run(self, syndromes, L=None, J=None, store_q=False):
        P = self.prog; dm = self.dm
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        joint = L is not None
        Ls = np.ascontiguousarray(L if joint else np.zeros(n), dtype=np.int64)
        Jm = np.zeros((dm.num_faults, 0)) if J is None else np.ascontiguousarray(J, dtype=np.float64)
        logp = np.zeros(n); qsum = np.zeros(dm.num_faults); grads = np.zeros((n, Jm.shape[1]))
        qall = np.zeros((n, dm.num_faults) if store_q else (1, 1))
        if n:
            _run_fb_fast(P["typ"], P["a1"], P["a2"], P["pp"], P["nbits"], P["nbpre"], P["fid"], P["foff"], P["rows"],
                         dm.num_observables, syn, Ls, joint, Jm, self.B, logp, qsum, grads, qall, store_q)
        return {"logp": logp, "qsum": qsum, "grads": grads if J is not None else None, "q": qall if store_q else None}
