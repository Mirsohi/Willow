"""Exact forward-backward for the conditional likelihood ln P(L | s) (round 2; notes/decoder/willow_prereg_r2.md).

One forward pass and one backward pass give, for every shot, ln P(s, L = 0), ln P(s, L = 1) and the per-shot scores
of the joint ln P(s, L_obs) and of the marginal ln P(s) with respect to the parameters (columns of J = dp/dtheta).
The conditional score is their difference:

    d ln P(L_obs | s) / d theta = sum_j (E[x_j | s, L_obs] - E[x_j | s]) / (p_j (1 - p_j)) dp_j/dtheta.

The layout and the forward pass are those of the frozen ml_fb_fast (its schedule builder is imported unchanged). The
backward pass starts from the L = 0 terminal. With one observable, the backward table for the L = 1 terminal is the
L = 0 table with the logical bit (index bit 0) flipped at every step: every reverse operation (bit move, detector
reopen, fault butterfly, projection, per-lane renormalisation) commutes with that flip. Four sums per fault therefore
give q_j for both L:

    q_j^L = p_j n_L / ((1 - p_j) d_L + p_j n_L),  n_L = sum_x F(x) B(x ^ m_j ^ L),  d_L = sum_x F(x) B(x ^ L).

The marginal posterior is E[x_j | s] = sum_L P(L | s) q_j^L. Validated against two runs of the frozen kernel
(tests in test_fb_cond.py). Single observable only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/decoder (frozen, read-only)
import numpy as np
import numba as nb
from dem_utils import DemMatrices
from ml_fb_fast import build_fb_schedule


@nb.njit(cache=True)
def _ins0(v, b):
    return ((v >> b) << (b + 1)) | (v & ((1 << b) - 1))


@nb.njit(cache=True)
def _run_fb_cond(typ, a1, a2, pp, nbits, nbpre, fid, foff, rows, syn, Ls, J, B, lp0, lp1, gj, gm):
    nsh = syn.shape[0]
    maxbits = 0
    for k in range(nbits.shape[0]):
        if nbits[k] > maxbits:
            maxbits = nbits[k]
    F = np.zeros((1 << maxbits, B)); Bt = np.zeros((1 << maxbits, B))
    Fst = np.empty((rows, B))
    sv = np.zeros(B); mx = np.zeros(B); inv = np.zeros(B); ls = np.zeros(B)
    w1 = np.zeros(B); okb = np.zeros(B, np.bool_)
    n0 = np.zeros(B); d0 = np.zeros(B); n1 = np.zeros(B); d1 = np.zeros(B)
    kp = J.shape[1]
    nops = typ.shape[0]
    for b0 in range(0, nsh, B):
        nl = min(B, nsh - b0)
        F[:, :] = 0.0
        for b in range(B):
            F[0, b] = 1.0; ls[b] = 0.0
        # ---------------- forward (as ml_fb_fast) ----------------
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
            okb[b] = False; w1[b] = 0.0
            if b >= nl:
                continue
            f0 = F[0, b]; f1 = F[1, b]
            tot = f0 + f1
            if tot > 0.0 and np.isfinite(ls[b]):
                lp0[b0 + b] = ls[b] + np.log(f0) if f0 > 0.0 else -np.inf
                lp1[b0 + b] = ls[b] + np.log(f1) if f1 > 0.0 else -np.inf
                w1[b] = f1 / tot
                okb[b] = True
            else:
                lp0[b0 + b] = -np.inf; lp1[b0 + b] = -np.inf
        # ---------------- backward from the L = 0 terminal (z-encoding) ----------------
        Bt[:, :] = 0.0
        for b in range(B):
            Bt[0, b] = 1.0
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
                    n0[b] = 0.0; d0[b] = 0.0; n1[b] = 0.0; d1[b] = 0.0
                for x in range(size):
                    y = x ^ M
                    for b in range(B):
                        f = Fst[o + x, b]
                        n0[b] += f * Bt[y, b]
                        d0[b] += f * Bt[x, b]
                        n1[b] += f * Bt[y ^ 1, b]
                        d1[b] += f * Bt[x ^ 1, b]
                j = fid[k]
                if kp > 0:
                    for b in range(nl):
                        if not okb[b]:
                            continue
                        den = q * d0[b] + p * n0[b]
                        qa = p * n0[b] / den if den > 0.0 else p
                        den = q * d1[b] + p * n1[b]
                        qb = p * n1[b] / den if den > 0.0 else p
                        fa = (qa - p) / (p * q); fbq = (qb - p) / (p * q)
                        fo = fbq if Ls[b0 + b] == 1 else fa
                        fmg = (1.0 - w1[b]) * fa + w1[b] * fbq
                        for c in range(kp):
                            gj[b0 + b, c] += fo * J[j, c]
                            gm[b0 + b, c] += fmg * J[j, c]
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


class ForwardBackwardCond:
    """run(syndromes, L, J) -> lp0, lp1 (ln P(s, L = 0 / 1)), cll (ln P(L | s)), g_joint, g_marg, g_cond (n, k)."""

    def __init__(self, dm: DemMatrices, order=None, batch=16, max_bytes=1.5e9):
        assert not dm.undetectable, "undetectable faults not supported"
        assert dm.num_observables == 1, "single observable only"
        self.dm = dm
        self.prog = build_fb_schedule(dm, order)
        self.B = batch
        need = 8.0 * self.prog["rows"] * batch
        if need > max_bytes:
            raise MemoryError(f"forward-backward needs {need/1e9:.2f} GB")

    def run(self, syndromes, L, J):
        P = self.prog
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        Ls = np.ascontiguousarray(np.asarray(L).reshape(-1), dtype=np.int64)
        assert len(Ls) == n
        Jm = np.ascontiguousarray(J, dtype=np.float64)
        lp0 = np.zeros(n); lp1 = np.zeros(n)
        gj = np.zeros((n, Jm.shape[1])); gm = np.zeros((n, Jm.shape[1]))
        if n:
            _run_fb_cond(P["typ"], P["a1"], P["a2"], P["pp"], P["nbits"], P["nbpre"], P["fid"], P["foff"], P["rows"],
                         syn, Ls, Jm, self.B, lp0, lp1, gj, gm)
        with np.errstate(invalid="ignore"):
            lpo = np.where(Ls == 1, lp1, lp0)
            lps = np.logaddexp(lp0, lp1)
            cll = lpo - lps
        return {"lp0": lp0, "lp1": lp1, "lp_joint": lpo, "lp_marg": lps, "cll": cll,
                "g_joint": gj, "g_marg": gm, "g_cond": gj - gm}
