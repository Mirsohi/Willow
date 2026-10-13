"""Max-product twin of decoder's sparse-frontier ML contraction (code/decoder/ml_sparse.py), for R2 of the revision.

reservoir, 2026-10-13. The kernel below is a copy of ml_sparse._run_sparse with one switch, `mode`:
  mode 0 (sum-product): v(y) = q A G(y) + p B G(y ^ m), and the classes are summed at the end: decoder's ML, the same
         arithmetic in the same order, so it reproduces ml_sparse's posteriors bit for bit (checked by the tests);
  mode 1 (max-product): v(y) = max(q A G(y), p B G(y ^ m)), and the classes take the maximum at the end, so
         out[c] = max over fault sets in class c of P(fault set, syndrome) up to the common scale: the most probable
         error (MPE) in each class.
Everything else is shared: the program (decoder's compile_sparse_program and fault order, imported read-only), the
importance weights G = W F, the pruning rule (drop entries with G < tau * max G, keep at most kmax), and the shots.
So the two modes differ only in how the degenerate fault sets of a class are combined (sum or max): the controlled
change to degeneracy handling of R2. Undetectable faults (no detectors) are applied per class as a sum (mode 0) or
a max (mode 1). Faults with identical (detectors, observables) are merged by decoder's parse_dem; the R2 run counts
any such merges, since the MPE of a merged DEM is not the MPE of the original one when a merge occurs.

Outputs per shot: ln M_c (mode 1) or ln Z_c (mode 0) for c = 0, 1 (natural log, including the scale), from which
llr = ln(M_1 / M_0) and the class. NaN when every entry was pruned or a detector no fault touches is lit.
"""
from __future__ import annotations

import os
import sys

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEC = os.path.join(HERE, "..", "..", "decoder")
sys.path.insert(0, os.path.realpath(DEC))
from dem_utils import DemMatrices, parse_dem  # noqa: E402,F401  (decoder's, read-only)
from ml_sparse import compile_sparse_program  # noqa: E402  (decoder's, read-only)


@nb.njit(cache=True)
def _wt(x, s, pi):
    return (1.0 - pi) if x == s else pi


@nb.njit(cache=True)
def _run_semiring(op_type, op_a, op_b, op_p, op_ps, op_pe, e_slot, e_det, e_po, e_pn, W, nobs,
                  syndromes, tau, kmax, mode, out_logs, out_maxk):
    nsh = syndromes.shape[0]
    C = 1 << nobs
    out = np.zeros((nsh, C))
    cap = 1024
    hsize = 4096
    hkeys = np.full(hsize, -1, np.int64)
    hval = np.empty(hsize, np.int64)
    used = np.empty(cap, np.int64)
    keys = np.empty(cap, np.int64); vals = np.empty(cap)
    nkeys = np.empty(cap, np.int64); nvals = np.empty(cap)
    for sh in range(nsh):
        syn = syndromes[sh]
        n = 1; keys[0] = 0; vals[0] = 1.0
        logscale = 0.0
        maxk = 1
        dead = False
        for o in range(op_type.shape[0]):
            t = op_type[o]
            if t == 2:
                w0 = _wt(0, syn[op_b[o]], op_p[o])
                if w0 <= 0:
                    dead = True; break
                logscale += np.log(w0)
            elif t == 0:
                m = op_a[o] | (op_b[o] << W)
                p = op_p[o]; q = 1.0 - p
                ps = op_ps[o]; pe = op_pe[o]
                if 2 * n + 16 > cap:
                    ncap = max(2 * cap, 2 * n + 16)
                    k2 = np.empty(ncap, np.int64); v2 = np.empty(ncap)
                    k2[:n] = keys[:n]; v2[:n] = vals[:n]
                    keys = k2; vals = v2
                    nkeys = np.empty(ncap, np.int64); nvals = np.empty(ncap)
                    used = np.empty(ncap, np.int64)
                    cap = ncap
                if 4 * n > hsize:
                    while 4 * n > hsize:
                        hsize <<= 1
                    hkeys = np.full(hsize, -1, np.int64)
                    hval = np.empty(hsize, np.int64)
                mask = hsize - 1
                for i in range(n):
                    h = (keys[i] * 0x9E3779B97F4A7C15) >> 17
                    h &= mask
                    while hkeys[h] != -1:
                        h = (h + 1) & mask
                    hkeys[h] = keys[i]; hval[h] = i; used[i] = h
                nn = 0
                for pass_ in range(2):
                    for i in range(n):
                        y = keys[i] if pass_ == 0 else (keys[i] ^ m)
                        if pass_ == 1:
                            h = (y * 0x9E3779B97F4A7C15) >> 17
                            h &= mask
                            found = False
                            while hkeys[h] != -1:
                                if hkeys[h] == y:
                                    found = True; break
                                h = (h + 1) & mask
                            if found:
                                continue
                            gy = 0.0
                            gym = vals[i]
                        else:
                            gy = vals[i]
                            z = y ^ m
                            h = (z * 0x9E3779B97F4A7C15) >> 17
                            h &= mask
                            gym = 0.0
                            while hkeys[h] != -1:
                                if hkeys[h] == z:
                                    gym = vals[hval[h]]; break
                                h = (h + 1) & mask
                        A = 1.0; Bw = 1.0
                        for e in range(ps, pe):
                            sl = e_slot[e]; d = e_det[e]; s = syn[d]
                            xb = (y >> sl) & 1
                            wn = _wt(xb, s, e_pn[e])
                            A *= wn / _wt(xb, s, e_po[e])
                            Bw *= wn / _wt(1 - xb, s, e_po[e])
                        if mode == 0:
                            v = q * A * gy + p * Bw * gym
                        else:
                            v1 = q * A * gy
                            v2_ = p * Bw * gym
                            v = v1 if v1 >= v2_ else v2_
                        if v > 0.0:
                            if nn >= cap:
                                continue
                            nkeys[nn] = y; nvals[nn] = v; nn += 1
                for i in range(n):
                    hkeys[used[i]] = -1
                vmax = 0.0
                for i in range(nn):
                    if nvals[i] > vmax:
                        vmax = nvals[i]
                if vmax <= 0.0:
                    dead = True; break
                thr = tau * vmax
                n = 0
                for i in range(nn):
                    if nvals[i] >= thr:
                        keys[n] = nkeys[i]; vals[n] = nvals[i] / vmax; n += 1
                logscale += np.log(vmax)
                if n > kmax:
                    idx = np.argsort(-vals[:n])[:kmax]
                    tk = keys[idx].copy(); tv = vals[idx].copy()
                    n = kmax
                    keys[:n] = tk; vals[:n] = tv
                if n > maxk:
                    maxk = n
            else:
                sl = op_a[o]; s = syn[op_b[o]]
                bit = np.int64(1) << sl
                n2 = 0
                for i in range(n):
                    b = (keys[i] >> sl) & 1
                    if b == s:
                        keys[n2] = keys[i] & ~bit; vals[n2] = vals[i]; n2 += 1
                n = n2
                if n == 0:
                    dead = True; break
        if dead:
            for c in range(C):
                out[sh, c] = np.nan
            out_logs[sh] = -np.inf
            continue
        for i in range(n):
            c = keys[i] >> W
            if mode == 0:
                out[sh, c] += vals[i]
            elif vals[i] > out[sh, c]:
                out[sh, c] = vals[i]
        out_logs[sh] = logscale
        out_maxk[sh] = maxk
    return out


class SemiringDecoder:
    """mode 0: decoder's sparse-frontier ML (sum-product); mode 1: the same contraction in max-product (MPE)."""

    def __init__(self, dm, tau, kmax, mode, prog=None):
        self.dm, self.tau, self.kmax, self.mode = dm, float(tau), int(kmax), int(mode)
        self.prog = prog if prog is not None else compile_sparse_program(dm)

    def raw(self, syndromes):
        """(out, logs): per shot and class the scaled class value (Z_c for mode 0, M_c for mode 1) after the
        undetectable faults, NaN where the shot died or a never-touched detector is lit; ln X_c = ln out_c + logs."""
        P = self.prog
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        logs = np.zeros(n); mk = np.zeros(n, np.int64)
        out = _run_semiring(P["op_type"], P["op_a"], P["op_b"], P["op_p"], P["op_ps"], P["op_pe"],
                            P["e_slot"], P["e_det"], P["e_po"], P["e_pn"], P["W"], self.dm.num_observables,
                            syn, self.tau, self.kmax, self.mode, logs, mk)
        C = out.shape[1]
        for om, pr in self.dm.undetectable:
            other = out[:, np.arange(C) ^ om]
            out = (1 - pr) * out + pr * other if self.mode == 0 else np.maximum((1 - pr) * out, pr * other)
        if len(P["never"]):
            out[syn[:, P["never"]].any(axis=1)] = np.nan
        self.last_maxk = mk
        return out, logs


def summarize(out, logs):
    """decoder's conventions (decode_willow._ml_out): post = out / sum, llr = ln post_1 - ln post_0 as float32,
    pred = post_1 > 0.5 (NaN -> 0); plus ln X_c = ln out_c + logs in float64."""
    tot = out.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        post = out / tot[:, None]
        llr = (np.log(post[:, 1]) - np.log(post[:, 0])).astype(np.float32)
        lnx = np.log(out) + logs[:, None]
    pred = (np.nan_to_num(post[:, 1], nan=0.0) > 0.5).astype(np.uint8)
    return {"post1": post[:, 1], "llr": llr, "pred": pred, "lnx": lnx, "logp_like": logs + np.log(tot)}


def ladder(dm, syn, mode, t1=1e-6, t2=1e-8, band=0.1, kmax=2_000_000, prog=None):
    """decoder's ml_tau2 protocol (decode_willow.make_decoder) in either semiring: every shot at t1; shots with
    |P - 1/2| < band (or NaN) again at t2, whose answer is final; P = X_1 / (X_0 + X_1), X = Z (mode 0) or M (mode 1).
    In mode 0 this is decoder's ladder itself (same program, same arithmetic, same band rule)."""
    D1 = SemiringDecoder(dm, t1, kmax, mode, prog)
    D2 = SemiringDecoder(dm, t2, kmax, mode, D1.prog)
    o1, s1 = D1.raw(syn)
    r1 = summarize(o1, s1)
    k1 = D1.last_maxk.copy()
    with np.errstate(invalid="ignore"):
        unc = ~(np.abs(r1["post1"] - 0.5) >= band)
    res = {k: v.copy() for k, v in r1.items()}
    res.update(llr_t1=r1["llr"].copy(), lnx_t1=r1["lnx"].copy(), redecoded=unc, maxk_t1=k1,
               maxk_t2=np.zeros(len(syn), np.int64))
    idx = np.where(unc)[0]
    if len(idx):
        o2, s2 = D2.raw(syn[idx])
        r2 = summarize(o2, s2)
        for k in ("post1", "llr", "pred", "lnx", "logp_like"):
            res[k][idx] = r2[k]
        res["maxk_t2"][idx] = D2.last_maxk
    return res
