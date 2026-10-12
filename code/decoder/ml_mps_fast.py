"""Fast (numba) boundary-MPS approximate ML decoder with optional importance reweighting.

Same contraction as ml_mps.MPSDecoder (see that file), compiled into a static, shot-independent program
(insert / fault / close operations with precomputed MPS site indices) executed by a numba kernel.

Importance reweighting (reweight=True): the stored boundary tensor is G = W F, where
F is the exact frontier table and W = prod_i w_i(x_i) is a product-form estimate of the backward
message of the *unprocessed* faults: w_i(x) = 1 - pi_i if x == s_i else pi_i, with
pi_i = P(an odd number of the remaining faults touching detector i fire). As faults are processed,
pi_i -> 0 and the weight becomes the syndrome projection. This changes nothing for the exact
contraction (chi -> inf) but makes SVD truncation weight states by how compatible they are with the
syndrome still to come, instead of by their current mass alone.
"""
import numpy as np
import numba as nb
from numba.typed import List
from dem_utils import DemMatrices, best_sweep_order
from ml_mps import spatial_snake_keys, choose_logical_positions


def compile_program(dm: DemMatrices, order=None, major="x"):
    if order is None:
        order = best_sweep_order(dm)[0]
    spos, npos = spatial_snake_keys(dm, major)
    lpos = choose_logical_positions(dm, spos, npos)
    nobs = dm.num_observables
    first, last = {}, {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            first.setdefault(d, pos)
            last[d] = pos
    closes_after = {}
    for d, pos in last.items():
        closes_after.setdefault(pos, []).append(d)
    never = np.array([d for d in range(dm.num_detectors) if d not in first], dtype=np.int64)
    key = {}
    for i in range(dm.num_detectors):
        c = dm.coords.get(i)
        t = c[-1] if c is not None and len(c) >= 3 else i
        key[i] = (spos[i], t)
    # remaining-fault product per detector: prod (1-2p) over unprocessed faults touching it
    rem = np.ones(dm.num_detectors)
    for j in order:
        for d in dm.dets[j]:
            rem[d] *= (1.0 - 2.0 * dm.p[j])
    labels = [("L", k) for k in range(nobs)]
    keys = [(lpos[k], -1e18) for k in range(nobs)]
    # ops: type, i0, i1, ptr_start, ptr_end, prob
    op_type, op_i0, op_i1, op_ps, op_pe, op_p = [], [], [], [], [], []
    # touched-site entries for fault ops: site index, detector (-1 for logical), flip flag, pi_old, pi_new
    e_site, e_det, e_pi_old, e_pi_new = [], [], [], []
    for pos, j in enumerate(order):
        ds = dm.dets[j]
        for d in ds:
            lab = ("D", d)
            if lab in labels:
                continue
            kd = key[d]
            ins = 0
            while ins < len(keys) and keys[ins] < kd:
                ins += 1
            labels.insert(ins, lab); keys.insert(ins, kd)
            op_type.append(0); op_i0.append(ins); op_i1.append(d); op_ps.append(0); op_pe.append(0)
            op_p.append(0.5 * (1.0 - rem[d]))  # pi at opening (all faults unprocessed)
        ents = []
        for d in ds:
            pi_old = 0.5 * (1.0 - rem[d])
            rem[d] /= (1.0 - 2.0 * dm.p[j])
            if d in last and last[d] == pos:
                rem[d] = 1.0  # exactly no remaining faults
            pi_new = 0.5 * (1.0 - rem[d])
            ents.append((labels.index(("D", d)), d, pi_old, pi_new))
        om = int(dm.obs[j])
        for k in range(nobs):
            if (om >> k) & 1:
                ents.append((labels.index(("L", k)), -1, 0.0, 0.0))
        ents.sort()
        st = len(e_site)
        for (si, d, po, pn) in ents:
            e_site.append(si); e_det.append(d); e_pi_old.append(po); e_pi_new.append(pn)
        op_type.append(1); op_i0.append(ents[0][0]); op_i1.append(ents[-1][0]); op_ps.append(st)
        op_pe.append(len(e_site)); op_p.append(float(dm.p[j]))
        for d in closes_after.get(pos, []):
            k = labels.index(("D", d))
            del labels[k]; del keys[k]
            op_type.append(2); op_i0.append(k); op_i1.append(d); op_ps.append(0); op_pe.append(0); op_p.append(0.0)
    final_logical_order = np.array([lab[1] for lab in labels], dtype=np.int64)
    prog = dict(op_type=np.array(op_type, np.int64), op_i0=np.array(op_i0, np.int64),
                op_i1=np.array(op_i1, np.int64), op_ps=np.array(op_ps, np.int64),
                op_pe=np.array(op_pe, np.int64), op_p=np.array(op_p, np.float64),
                e_site=np.array(e_site, np.int64), e_det=np.array(e_det, np.int64),
                e_pi_old=np.array(e_pi_old, np.float64), e_pi_new=np.array(e_pi_new, np.float64),
                never=never, final_logical_order=final_logical_order, nobs=nobs)
    return prog


@nb.njit(cache=True)
def _move_center(T, center, to):
    while center < to:
        A = T[center]
        Dl, Dr = A.shape[0], A.shape[2]
        Q, R = np.linalg.qr(np.ascontiguousarray(A.reshape(Dl * 2, Dr)))
        k = Q.shape[1]
        T[center] = np.ascontiguousarray(Q).reshape(Dl, 2, k)
        B = T[center + 1]
        Dl2, Dr2 = B.shape[0], B.shape[2]
        T[center + 1] = np.ascontiguousarray((R @ B.reshape(Dl2, 2 * Dr2)).reshape(k, 2, Dr2))
        center += 1
    while center > to:
        A = T[center]
        Dl, Dr = A.shape[0], A.shape[2]
        Q, R = np.linalg.qr(np.ascontiguousarray(A.reshape(Dl, 2 * Dr).T))
        k = Q.shape[1]
        T[center] = np.ascontiguousarray(Q.T).reshape(k, 2, Dr)
        B = T[center - 1]
        Dl2, Dr2 = B.shape[0], B.shape[2]
        T[center - 1] = np.ascontiguousarray((B.reshape(Dl2 * 2, Dr2) @ R.T).reshape(Dl2, 2, k))
        center -= 1
    return center


@nb.njit(cache=True)
def _w(x, s, pi):
    return (1.0 - pi) if x == s else pi


@nb.njit(cache=True)
def _run_mps(op_type, op_i0, op_i1, op_ps, op_pe, op_p, e_site, e_det, e_pi_old, e_pi_new,
             nobs, syndromes, chi, cutoff, reweight, out_logp, out_maxbond):
    nshots = syndromes.shape[0]
    C = 1 << nobs
    out = np.zeros((nshots, C))
    for sh in range(nshots):
        syn = syndromes[sh]
        T = List()
        for k in range(nobs):
            A = np.zeros((1, 2, 1)); A[0, 0, 0] = 1.0
            T.append(A)
        center = 0
        logscale = 0.0
        maxbond = 1
        dead = False
        for o in range(op_type.shape[0]):
            t = op_type[o]
            if t == 0:
                ins = op_i0[o]
                if ins == 0:
                    D = T[0].shape[0]
                elif ins == len(T):
                    D = T[len(T) - 1].shape[2]
                else:
                    D = T[ins].shape[0]
                A = np.zeros((D, 2, D))
                for a_ in range(D):
                    A[a_, 0, a_] = 1.0
                if reweight:
                    d = op_i1[o]
                    w0 = _w(0, syn[d], op_p[o])
                    if w0 <= 0.0:
                        dead = True
                        break
                    logscale += np.log(w0)
                T.insert(ins, A)
                if center >= ins:
                    center += 1
            elif t == 1:
                a = op_i0[o]; b = op_i1[o]
                p = op_p[o]; q = 1.0 - p
                ps = op_ps[o]; pe = op_pe[o]
                center = _move_center(T, center, a)
                # per-site factors for the identity (id) and flip (fl) branches
                ptr = ps
                if a == b:
                    A = T[a]
                    Dl, Dr = A.shape[0], A.shape[2]
                    d = e_det[ptr]
                    new = np.empty_like(A)
                    for x in range(2):
                        if d >= 0 and reweight:
                            s = syn[d]
                            fid = _w(x, s, e_pi_new[ptr]) / _w(x, s, e_pi_old[ptr])
                            ffl = _w(x, s, e_pi_new[ptr]) / _w(1 - x, s, e_pi_old[ptr])
                        else:
                            fid = 1.0; ffl = 1.0
                        for l in range(Dl):
                            for r in range(Dr):
                                new[l, x, r] = q * fid * A[l, x, r] + p * ffl * A[l, 1 - x, r]
                    T[a] = new
                else:
                    A = T[a]
                    Dl, Dr = A.shape[0], A.shape[2]
                    M = np.zeros((Dl, 2, 2 * Dr))
                    d = e_det[ptr]
                    for x in range(2):
                        if d >= 0 and reweight:
                            s = syn[d]
                            fid = _w(x, s, e_pi_new[ptr]) / _w(x, s, e_pi_old[ptr])
                            ffl = _w(x, s, e_pi_new[ptr]) / _w(1 - x, s, e_pi_old[ptr])
                        else:
                            fid = 1.0; ffl = 1.0
                        for l in range(Dl):
                            for r in range(Dr):
                                M[l, x, r] = q * fid * A[l, x, r]
                                M[l, x, Dr + r] = p * ffl * A[l, 1 - x, r]
                    ptr += 1
                    for k in range(a + 1, b + 1):
                        Dl, _, Dr2 = M.shape
                        mat = np.ascontiguousarray(M.reshape(Dl * 2, Dr2))
                        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
                        keep = 1
                        if S[0] > 0:
                            keep = 0
                            thr = cutoff * S[0]
                            for i_ in range(S.shape[0]):
                                if S[i_] > thr:
                                    keep += 1
                            if keep < 1:
                                keep = 1
                        if keep > chi:
                            keep = chi
                        if keep > maxbond:
                            maxbond = keep
                        T[k - 1] = np.ascontiguousarray(U[:, :keep]).reshape(Dl, 2, keep)
                        SV = np.empty((keep, Dr2))
                        for i_ in range(keep):
                            for j_ in range(Dr2):
                                SV[i_, j_] = S[i_] * Vh[i_, j_]
                        B = T[k]
                        Dk, Drk = B.shape[0], B.shape[2]
                        touched = (ptr < pe) and (e_site[ptr] == k)
                        if touched:
                            d = e_det[ptr]
                        else:
                            d = -2
                        # build branch tensors
                        Bid = np.empty((Dk, 2, Drk))
                        Bfl = np.empty((Dk, 2, Drk))
                        for x in range(2):
                            if touched and d >= 0 and reweight:
                                s = syn[d]
                                fid = _w(x, s, e_pi_new[ptr]) / _w(x, s, e_pi_old[ptr])
                                ffl = _w(x, s, e_pi_new[ptr]) / _w(1 - x, s, e_pi_old[ptr])
                            else:
                                fid = 1.0; ffl = 1.0
                            xs = (1 - x) if touched else x
                            for l in range(Dk):
                                for r in range(Drk):
                                    Bid[l, x, r] = fid * B[l, x, r]
                                    Bfl[l, x, r] = ffl * B[l, xs, r]
                        if touched:
                            ptr += 1
                        left = (np.ascontiguousarray(SV[:, :Dk]) @ Bid.reshape(Dk, 2 * Drk)).reshape(keep, 2, Drk)
                        right = (np.ascontiguousarray(SV[:, Dk:]) @ Bfl.reshape(Dk, 2 * Drk)).reshape(keep, 2, Drk)
                        if k < b:
                            M = np.empty((keep, 2, 2 * Drk))
                            M[:, :, :Drk] = left
                            M[:, :, Drk:] = right
                        else:
                            M = left + right
                    T[b] = M
                    center = b
                nrm = np.sqrt(np.sum(T[center] ** 2))
                if nrm > 0:
                    T[center] = T[center] / nrm
                    logscale += np.log(nrm)
                else:
                    dead = True
                    break
            else:
                k = op_i0[o]; d = op_i1[o]
                center = _move_center(T, center, k)
                s = syn[d]
                A = T[k]
                Mx = np.ascontiguousarray(A[:, s, :])
                T.pop(k)
                if k < len(T):
                    B = T[k]
                    Dk, Drk = B.shape[0], B.shape[2]
                    T[k] = np.ascontiguousarray((Mx @ B.reshape(Dk, 2 * Drk)).reshape(Mx.shape[0], 2, Drk))
                    center = k
                else:
                    B = T[k - 1]
                    Dl2 = B.shape[0]
                    T[k - 1] = np.ascontiguousarray((B.reshape(Dl2 * 2, B.shape[2]) @ Mx).reshape(Dl2, 2, Mx.shape[1]))
                    center = k - 1
                nrm = np.sqrt(np.sum(T[center] ** 2))
                if nrm > 0:
                    T[center] = T[center] / nrm
                    logscale += np.log(nrm)
                else:
                    dead = True
                    break
        if dead:
            for c in range(C):
                out[sh, c] = np.nan
            out_logp[sh] = -np.inf
            continue
        # contract logical sites (in site order) into a vector
        v = T[0].reshape(T[0].shape[1] * T[0].shape[2])
        v = T[0][0, :, :].copy().reshape(2 * T[0].shape[2])
        dimv = 2
        Dcur = T[0].shape[2]
        vec = T[0][0].copy()  # (2, D)
        for k in range(1, len(T)):
            B = T[k]
            Dk, Drk = B.shape[0], B.shape[2]
            vec = (vec @ B.reshape(Dk, 2 * Drk)).reshape(vec.shape[0] * 2, Drk)
        flat = vec[:, 0]
        for idx in range(C):
            out[sh, idx] = flat[idx]
        out_logp[sh] = logscale
        out_maxbond[sh] = maxbond
    return out


class FastMPSDecoder:
    def __init__(self, dm: DemMatrices, chi=32, cutoff=1e-12, reweight=True, order=None, major="x"):
        self.dm = dm
        self.chi, self.cutoff, self.reweight = chi, cutoff, reweight
        self.prog = compile_program(dm, order, major)

    def class_probs(self, syndromes):
        P = self.prog
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        logp = np.zeros(n); mb = np.zeros(n, dtype=np.int64)
        raw = _run_mps(P["op_type"], P["op_i0"], P["op_i1"], P["op_ps"], P["op_pe"], P["op_p"],
                       P["e_site"], P["e_det"], P["e_pi_old"], P["e_pi_new"], P["nobs"], syn,
                       self.chi, self.cutoff, self.reweight, logp, mb)
        nobs = P["nobs"]
        # map site-order bit pattern to class index
        lo = P["final_logical_order"]
        C = 1 << nobs
        out = np.zeros((n, C))
        for idx in range(C):
            c = 0
            for pos_i, k in enumerate(lo):
                bit = (idx >> (len(lo) - 1 - pos_i)) & 1
                c |= bit << int(k)
            out[:, c] = raw[:, idx]
        for om, pr in self.dm.undetectable:
            out = (1 - pr) * out + pr * out[:, np.arange(C) ^ om]
        if len(P["never"]):
            bad = syn[:, P["never"]].any(axis=1)
            out[bad] = np.nan
        out = np.maximum(out, 0.0)  # tiny negative round-off from SVD truncation
        tot = out.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            post = out / tot[:, None]
            logp = logp + np.log(tot)
        self.last_maxbond = mb
        return post, logp

    def decode(self, syndromes):
        return np.argmax(self.class_probs(syndromes)[0], axis=1)
