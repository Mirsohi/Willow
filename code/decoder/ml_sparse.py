"""Pruned sparse-frontier approximate ML decoder (importance-weighted), a second, independent
approximation of the same contraction as ml_exact (cross-check for the boundary-MPS decoder).

The exact frontier table F[x, c] (see ml_exact) is stored sparsely as (key, value) pairs over the
open-detector parities x and logical class c. Values are importance-weighted, G = W F with the same
product-form backward weights as ml_mps_fast (w_i(x) = 1 - pi_i if x == s_i else pi_i, pi_i =
P(odd number of the remaining faults on detector i fire)). After each fault, entries with
G < tau * max(G) are dropped and at most kmax entries are kept. tau -> 0, kmax -> inf is exact.
"""
import numpy as np
import numba as nb
from dem_utils import DemMatrices, best_sweep_order


def compile_sparse_program(dm: DemMatrices, order=None):
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
    # remaining-fault probabilities via suffix products (robust to p = 0.5)
    rem_lists = {d: [] for d in range(dm.num_detectors)}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            rem_lists[d].append(dm.p[j])
    rem_idx = {d: 0 for d in range(dm.num_detectors)}

    def pi_after(d, k):  # prob odd parity of faults rem_lists[d][k:]
        prod = 1.0
        for q in rem_lists[d][k:]:
            prod *= (1.0 - 2.0 * q)
        return 0.5 * (1.0 - prod)

    free, slot_of, nxt = [], {}, 0
    # per fault: mask, obsmask, p, entries (slot, det, pi_old, pi_new); per close: slot, det
    op_type, op_a, op_b, op_p, op_ps, op_pe = [], [], [], [], [], []
    e_slot, e_det, e_po, e_pn = [], [], [], []
    open_pi = []  # for opens: we record (slot, det, pi_init) as op type 2
    for pos, j in enumerate(order):
        m = 0
        for d in dm.dets[j]:
            if d not in slot_of:
                s = free.pop() if free else nxt
                if s == nxt:
                    nxt += 1
                slot_of[d] = s
                op_type.append(2); op_a.append(s); op_b.append(d); op_p.append(pi_after(d, 0)); op_ps.append(0); op_pe.append(0)
            m |= (1 << slot_of[d])
        st = len(e_slot)
        for d in dm.dets[j]:
            k = rem_idx[d]
            po = pi_after(d, k)
            pn = pi_after(d, k + 1)
            rem_idx[d] = k + 1
            e_slot.append(slot_of[d]); e_det.append(d); e_po.append(po); e_pn.append(pn)
        op_type.append(0); op_a.append(m); op_b.append(int(dm.obs[j])); op_p.append(float(dm.p[j]))
        op_ps.append(st); op_pe.append(len(e_slot))
        for d in closes_after.get(pos, []):
            s = slot_of.pop(d)
            free.append(s)
            op_type.append(1); op_a.append(s); op_b.append(d); op_p.append(0.0); op_ps.append(0); op_pe.append(0)
    W = nxt
    assert W + dm.num_observables <= 62, "frontier too wide for 64-bit keys"
    never = np.array([d for d in range(dm.num_detectors) if d not in first], dtype=np.int64)
    return dict(op_type=np.array(op_type, np.int64), op_a=np.array(op_a, np.int64), op_b=np.array(op_b, np.int64),
                op_p=np.array(op_p, np.float64), op_ps=np.array(op_ps, np.int64), op_pe=np.array(op_pe, np.int64),
                e_slot=np.array(e_slot, np.int64), e_det=np.array(e_det, np.int64),
                e_po=np.array(e_po, np.float64), e_pn=np.array(e_pn, np.float64), W=W, never=never)


@nb.njit(cache=True)
def _wt(x, s, pi):
    return (1.0 - pi) if x == s else pi


@nb.njit(cache=True)
def _run_sparse(op_type, op_a, op_b, op_p, op_ps, op_pe, e_slot, e_det, e_po, e_pn, W, nobs,
                syndromes, tau, kmax, out_logp, out_maxk, out_drop):
    nsh = syndromes.shape[0]
    C = 1 << nobs
    out = np.zeros((nsh, C))
    # buffers grow on demand (capacity >= 2n + 16 entries, hash table >= 4n slots, power of two); the
    # earlier fixed allocation of 4*kmax entries cost ~10-300 ms per call for large kmax
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
        drop = 0.0
        dead = False
        for o in range(op_type.shape[0]):
            t = op_type[o]
            if t == 2:
                # open detector: entries have bit 0 there; apply initial weight w(0)
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
                # hash existing keys (table is all -1 between faults; used slots are cleared below)
                mask = hsize - 1
                for i in range(n):
                    h = (keys[i] * 0x9E3779B97F4A7C15) >> 17
                    h &= mask
                    while hkeys[h] != -1:
                        h = (h + 1) & mask
                    hkeys[h] = keys[i]; hval[h] = i; used[i] = h
                nn = 0
                # new value at key y = q*A(y)*G(y) + p*B(y)*G(y^m)
                # visit each key y in union(keys, keys^m) once
                for pass_ in range(2):
                    for i in range(n):
                        y = keys[i] if pass_ == 0 else (keys[i] ^ m)
                        if pass_ == 1:
                            # skip if y already exists in keys (handled in pass 0)
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
                        A = 1.0; B = 1.0
                        for e in range(ps, pe):
                            sl = e_slot[e]; d = e_det[e]; s = syn[d]
                            xb = (y >> sl) & 1
                            wn = _wt(xb, s, e_pn[e])
                            A *= wn / _wt(xb, s, e_po[e])
                            B *= wn / _wt(1 - xb, s, e_po[e])
                        v = q * A * gy + p * B * gym
                        if v > 0.0:
                            if nn >= cap:
                                # emergency prune: should not happen when kmax is respected
                                continue
                            nkeys[nn] = y; nvals[nn] = v; nn += 1
                for i in range(n):
                    hkeys[used[i]] = -1
                # prune
                vmax = 0.0
                vtot = 0.0
                for i in range(nn):
                    vtot += nvals[i]
                    if nvals[i] > vmax:
                        vmax = nvals[i]
                if vmax <= 0.0:
                    dead = True; break
                thr = tau * vmax
                n = 0
                vkept = 0.0
                for i in range(nn):
                    if nvals[i] >= thr:
                        keys[n] = nkeys[i]; vals[n] = nvals[i] / vmax; n += 1
                        vkept += nvals[i]
                logscale += np.log(vmax)
                if n > kmax:
                    idx = np.argsort(-vals[:n])[:kmax]
                    tk = keys[idx].copy(); tv = vals[idx].copy()
                    n = kmax
                    keys[:n] = tk; vals[:n] = tv
                    vkept = 0.0
                    for i in range(n):
                        vkept += vals[i] * vmax
                # importance-weighted mass dropped at this step (relative): with exact backward weights the
                # sum over steps bounds the relative error of P(s) to first order; here it is an estimate
                drop += 1.0 - vkept / vtot
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
            out_logp[sh] = -np.inf
            continue
        for i in range(n):
            c = keys[i] >> W
            out[sh, c] += vals[i]
        out_logp[sh] = logscale
        out_maxk[sh] = maxk
        out_drop[sh] = drop
    return out


class SparseFrontierDecoder:
    def __init__(self, dm: DemMatrices, tau=1e-10, kmax=4096, order=None):
        self.dm, self.tau, self.kmax = dm, tau, kmax
        self.prog = compile_sparse_program(dm, order)

    def class_probs(self, syndromes):
        P = self.prog
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        logp = np.zeros(n); mk = np.zeros(n, np.int64); dr = np.zeros(n)
        out = _run_sparse(P["op_type"], P["op_a"], P["op_b"], P["op_p"], P["op_ps"], P["op_pe"],
                          P["e_slot"], P["e_det"], P["e_po"], P["e_pn"], P["W"], self.dm.num_observables,
                          syn, self.tau, self.kmax, logp, mk, dr)
        C = out.shape[1]
        for om, pr in self.dm.undetectable:
            out = (1 - pr) * out + pr * out[:, np.arange(C) ^ om]
        if len(P["never"]):
            out[syn[:, P["never"]].any(axis=1)] = np.nan
        tot = out.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            post = out / tot[:, None]
            logp = logp + np.log(tot)
        self.last_maxk = mk
        self.last_drop = dr  # per shot: sum over pruning steps of the dropped importance-weighted fraction
        return post, logp

    def decode(self, syndromes):
        return np.argmax(self.class_probs(syndromes)[0], axis=1)


class AdaptiveSparseDecoder:
    """Tau ladder per shot: decode everything at taus[0]; re-decode at the next tau only the shots whose decision is
    not certified by the dropped-mass indicator, i.e. |2P - 1| <= kappa * D (or NaN), until certified or the ladder
    ends. last_level[s] = index of the tau that produced shot s's answer. The rule is validated empirically
    (run_speed.py ladders), not a proof: D is exact only with exact backward weights."""

    def __init__(self, dm: DemMatrices, taus=(1e-4, 1e-6, 1e-8), kappa=2.0, kmax=2_000_000, order=None):
        self.decs = [SparseFrontierDecoder(dm, tau=t, kmax=kmax, order=order) for t in taus]
        self.taus, self.kappa = list(taus), kappa

    def class_probs(self, syndromes):
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        post, logp = self.decs[0].class_probs(syn)
        drop = self.decs[0].last_drop.copy()
        level = np.zeros(len(syn), np.int64)
        for li in range(1, len(self.decs)):
            P1 = post[:, 1]
            with np.errstate(invalid="ignore"):
                unc = ~(np.abs(2 * P1 - 1) > self.kappa * drop)
            idx = np.where(unc)[0]
            if len(idx) == 0:
                break
            p2, l2 = self.decs[li].class_probs(syn[idx])
            post[idx] = p2; logp[idx] = l2; drop[idx] = self.decs[li].last_drop; level[idx] = li
        self.last_level, self.last_drop = level, drop
        return post, logp

    def decode(self, syndromes):
        return np.argmax(np.nan_to_num(self.class_probs(syndromes)[0], nan=0.5), axis=1)
