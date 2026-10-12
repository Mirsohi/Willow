"""Exact forward-backward contraction of the DEM tensor network: per-shot log-likelihood and the exact
posterior marginal of every fault, q_j(s) = P(e_j = 1 | s [, L]), for small frontier width (d = 3).

Forward pass = ml_exact (dense frontier F over open-detector parities x and logical class c), storing the
state just before every fault. Backward pass over the reversed schedule with the "z-encoding": an open
detector i carries z_i = s_i XOR (parity of the faults after the cut), the observable carries
z_L = L XOR (parity after the cut), so the backward table B is indexed like F and
P(s, L) = sum_{x,c} F_pre_j(x,c) [ (1-p_j) B_post_j(x,c) + p_j B_post_j(x^m_j, c^o_j) ]  for every fault j,
q_j = p_j sum F_pre_j B_post_j(x^m_j, c^o_j) / P(s, L).
Uses: d/dp_j ln P = (q_j - p_j) / (p_j (1 - p_j)) -> exact likelihood gradients for parametric noise models;
E_s[q_j] = p_j when the model is true (per-fault calibration); EM update p_j <- mean_s q_j.
"""
import numpy as np
import numba as nb
from dem_utils import DemMatrices
from ml_exact import build_schedule


def opens_at_faults(dm: DemMatrices, order):
    """For the schedule of build_schedule(dm, order): per fault position, the slots of detectors whose FIRST
    fault it is (they are 'closed' with z = 0 in the backward pass right after processing that fault)."""
    seen = set()
    first_pos = {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            if d not in seen:
                seen.add(d); first_pos[d] = pos
    return first_pos


def compile_fb(dm: DemMatrices, order=None):
    from dem_utils import best_sweep_order
    if order is None:
        order = best_sweep_order(dm)[0]
    typ, a1, a2, pp, W, never = build_schedule(dm, order)
    # recover slot assignment by replaying build_schedule's logic
    first, last = {}, {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            first.setdefault(d, pos); last[d] = pos
    closes_after = {}
    for d, pos in last.items():
        closes_after.setdefault(pos, []).append(d)
    free, slot_of, nxt = [], {}, 0
    open_slots = []  # per fault position: list of slots opened (first fault) at that position
    for pos, j in enumerate(order):
        opened = []
        for d in dm.dets[j]:
            if d not in slot_of:
                if free:
                    s = free.pop()
                else:
                    s = nxt; nxt += 1
                slot_of[d] = s; opened.append(s)
        open_slots.append(opened)
        for d in closes_after.get(pos, []):
            free.append(slot_of.pop(d))
    assert nxt == W
    # per fault op: fault index j (in dm order), and opened-slot mask
    fault_ids = np.array(order, np.int64)
    open_mask = np.array([sum(1 << s for s in sl) for sl in open_slots], np.int64)
    return dict(typ=typ, a1=a1, a2=a2, pp=pp, W=W, never=never, fault_ids=fault_ids, open_mask=open_mask)


@nb.njit(cache=True)
def _run_fb(typ, a1, a2, pp, W, nobs, fault_ids, open_mask, syndromes, Ls, joint, J, out_logp, qsum, grads, qall, store_q):
    nsh = syndromes.shape[0]
    C = 1 << nobs
    size = (1 << W) * C
    nf = fault_ids.shape[0]
    Fst = np.empty((nf, size))
    F = np.empty(size); B = np.empty(size)
    k_par = J.shape[1]
    nf_total = qsum.shape[0]
    for sh in range(nsh):
        # ---------------- forward ----------------
        F[:] = 0.0; F[0] = 1.0
        logscale = 0.0
        fi = 0
        for k in range(typ.shape[0]):
            if typ[k] == 0:
                for x in range(size):
                    Fst[fi, x] = F[x]
                fi += 1
                M = a1[k] | (a2[k] << W)
                p = pp[k]; q = 1.0 - p
                hb = 0; t = M
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
                s = a1[k]; bit = 1 << s
                synd = syndromes[sh, a2[k]]
                mx = 0.0
                for x in range(size):
                    if x & bit:
                        continue
                    v = F[x | bit] if synd else F[x]
                    F[x] = v; F[x | bit] = 0.0
                    if v > mx:
                        mx = v
                if mx > 0.0:
                    inv = 1.0 / mx
                    for x in range(size):
                        F[x] *= inv
                    logscale += np.log(mx)
                else:
                    logscale = -np.inf
        if joint:
            tot = F[Ls[sh] << W]
        else:
            tot = 0.0
            for c in range(C):
                tot += F[c << W]
        if not (tot > 0.0) or not np.isfinite(logscale):
            out_logp[sh] = -np.inf
            continue
        out_logp[sh] = logscale + np.log(tot)
        # ---------------- backward (z-encoding) ----------------
        B[:] = 0.0
        if joint:
            B[Ls[sh] << W] = 1.0
        else:
            for c in range(C):
                B[c << W] = 1.0
        fi = nf
        for k in range(typ.shape[0] - 1, -1, -1):
            if typ[k] == 1:
                # reverse of a close: open the slot with z = s_i
                s = a1[k]; bit = 1 << s
                if syndromes[sh, a2[k]]:
                    for x in range(size):
                        if x & bit:
                            continue
                        B[x | bit] = B[x]; B[x] = 0.0
            else:
                fi -= 1
                M = a1[k] | (a2[k] << W)
                p = pp[k]; q = 1.0 - p
                # marginal of this fault: F_pre (stored) with B_post (current)
                num = 0.0; den0 = 0.0
                for x in range(size):
                    fx = Fst[fi, x]
                    if fx == 0.0:
                        continue
                    num += fx * B[x ^ M]
                    den0 += fx * B[x]
                den = q * den0 + p * num
                qj = p * num / den if den > 0.0 else 0.0
                j = fault_ids[fi]
                qsum[j] += qj
                if store_q:
                    qall[sh, j] = qj
                if k_par > 0:
                    pj = pp[k]
                    fac = (qj - pj) / (pj * (1.0 - pj))
                    for c in range(k_par):
                        grads[sh, c] += fac * J[j, c]
                # apply the fault to B
                hb = 0; t = M
                while t > 1:
                    t >>= 1; hb += 1
                hbit = 1 << hb
                for x in range(size):
                    if x & hbit:
                        continue
                    y = x ^ M
                    a = B[x]; b = B[y]
                    B[x] = q * a + p * b
                    B[y] = p * a + q * b
                # detectors whose first fault this is: project their slot onto z = 0 and free it
                om = open_mask[fi]
                if om != 0:
                    mx = 0.0
                    for x in range(size):
                        if x & om:
                            B[x] = 0.0
                        elif B[x] > mx:
                            mx = B[x]
                    if mx > 0.0:
                        inv = 1.0 / mx
                        for x in range(size):
                            B[x] *= inv


class ForwardBackwardExact:
    """Exact per-shot log-likelihood and fault posterior marginals (dense frontier; small W only)."""

    def __init__(self, dm: DemMatrices, order=None, max_bytes=1.5e9):
        self.dm = dm
        self.prog = compile_fb(dm, order)
        C = 1 << dm.num_observables
        need = 8.0 * (2 ** self.prog["W"]) * C * (len(self.prog["fault_ids"]) + 2)
        if need > max_bytes:
            raise MemoryError(f"forward-backward needs {need/1e9:.2f} GB per shot (W={self.prog['W']})")

    def run(self, syndromes, L=None, J=None, store_q=False):
        """syndromes (n, ndet) uint8; L (n,) logical outcomes for the joint likelihood P(s, L) (None -> P(s));
        J (nfaults, k) Jacobian dp_j/dtheta_c (optional) -> per-shot gradient of ln P wrt theta.
        Returns dict(logp, qsum (sum over shots of q_j), grads (n, k) or None, q (n, nfaults) if store_q)."""
        P = self.prog; dm = self.dm
        syn = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = syn.shape[0]
        joint = L is not None
        Ls = np.ascontiguousarray(L if joint else np.zeros(n), dtype=np.int64)
        Jm = np.zeros((dm.num_faults, 0)) if J is None else np.ascontiguousarray(J, dtype=np.float64)
        logp = np.zeros(n); qsum = np.zeros(dm.num_faults); grads = np.zeros((n, Jm.shape[1]))
        qall = np.zeros((n, dm.num_faults) if store_q else (1, 1))
        _run_fb(P["typ"], P["a1"], P["a2"], P["pp"], P["W"], dm.num_observables, P["fault_ids"], P["open_mask"],
                syn, Ls, joint, Jm, logp, qsum, grads, qall, store_q)
        # undetectable logical-flip faults are not in the schedule; with the joint likelihood they are not
        # handled here (assert none) - circuit-level DEMs used in this project have none.
        assert not dm.undetectable or not joint, "undetectable faults not supported in joint mode"
        return {"logp": logp, "qsum": qsum, "grads": grads if J is not None else None, "q": qall if store_q else None}
