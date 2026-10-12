"""Approximate maximum-likelihood decoding by boundary-MPS contraction of the DEM tensor network.

Same elimination as ExactFrontierDecoder (faults processed in a space-time sweep; time is the long
axis), but the frontier table F[x_open, c] is stored as a matrix product state whose sites are the open
detectors (ordered by a static spatial snake key, then time) plus one site per logical observable.
  open detector  -> insert a site in state |0>
  fault j        -> apply the bond-dimension-2 MPO (1-p_j) I + p_j X^{m_j} over the span of its sites,
                    then re-compress the span with SVDs (max bond chi, relative cutoff `cutoff`)
  close detector -> project the site onto the observed syndrome bit and absorb it into a neighbour
At the end only the logical site(s) remain: their 2^n_obs entries are P(s, c) up to the tracked scale.
The approximation is controlled by chi (and cutoff); chi -> inf reproduces the exact contraction.
"""
import numpy as np
from dem_utils import DemMatrices, best_sweep_order


def spatial_snake_keys(dm: DemMatrices, major="x"):
    """Return dict detector -> spatial snake index (column-major snake over (x,y) positions)."""
    pos = {}
    for i in range(dm.num_detectors):
        c = dm.coords.get(i)
        pos[i] = (round(c[0], 3), round(c[1], 3)) if c is not None and len(c) >= 2 else (i, 0)
    xy = sorted(set(pos.values()))
    if major == "x":
        cols = sorted(set(p[0] for p in xy))
        order = []
        for ci, x in enumerate(cols):
            ys = sorted(p[1] for p in xy if p[0] == x)
            if ci % 2 == 1:
                ys = ys[::-1]
            order.extend((x, y) for y in ys)
    else:
        rows = sorted(set(p[1] for p in xy))
        order = []
        for ri, y in enumerate(rows):
            xs = sorted(p[0] for p in xy if p[1] == y)
            if ri % 2 == 1:
                xs = xs[::-1]
            order.extend((x, y) for x in xs)
    idx = {p: k for k, p in enumerate(order)}
    return {i: idx[pos[i]] for i in range(dm.num_detectors)}, len(order)


def choose_logical_positions(dm: DemMatrices, spos, npos):
    """For each observable choose an insertion position (half-integer) minimising the total span of the
    faults that flip it."""
    out = []
    for k in range(dm.num_observables):
        best = None
        js = [j for j in range(dm.num_faults) if (int(dm.obs[j]) >> k) & 1]
        for cand in np.arange(-0.5, npos, 1.0):
            tot = 0.0
            for j in js:
                ps = [spos[d] for d in dm.dets[j]] + [cand]
                tot += max(ps) - min(ps)
            if best is None or tot < best[0]:
                best = (tot, cand)
        out.append(best[1] if best else -0.5)
    return out


class MPSDecoder:
    def __init__(self, dm: DemMatrices, chi=32, cutoff=1e-12, order=None, major="x"):
        self.dm = dm
        self.chi = chi
        self.cutoff = cutoff
        if order is None:
            order = best_sweep_order(dm)[0]
        self.order = order
        self.spos, self.npos = spatial_snake_keys(dm, major)
        self.lpos = choose_logical_positions(dm, self.spos, self.npos)
        first, last = {}, {}
        for pos, j in enumerate(order):
            for d in dm.dets[j]:
                first.setdefault(d, pos)
                last[d] = pos
        self.closes_after = {}
        for d, pos in last.items():
            self.closes_after.setdefault(pos, []).append(d)
        self.never = [d for d in range(dm.num_detectors) if d not in first]
        # static site key: detectors (spatial pos, time), logical sites (lpos, -1)
        self.key = {}
        for i in range(dm.num_detectors):
            c = dm.coords.get(i)
            t = c[-1] if c is not None and len(c) >= 3 else i
            self.key[i] = (self.spos[i], t)
        self.stats = {"max_sites": 0, "max_bond": 0}

    # ---- MPS primitives (one shot) ----
    def decode_one(self, syndrome):
        dm = self.dm
        nobs = dm.num_observables
        # sites: list of labels; tensors: list of arrays (Dl,2,Dr); keys for ordering
        labels = []
        keys = []
        T = []
        for k in range(nobs):
            labels.append(("L", k)); keys.append((self.lpos[k], -1e9))
            A = np.zeros((1, 2, 1)); A[0, 0, 0] = 1.0
            T.append(A)
        logscale = 0.0
        chi, cutoff = self.chi, self.cutoff
        where = {("L", k): None for k in range(nobs)}
        center = 0

        def site_index(label):
            return labels.index(label)

        def move_center(to):
            nonlocal center
            while center < to:
                A = T[center]; Dl, _, Dr = A.shape
                Q, R = np.linalg.qr(A.reshape(Dl * 2, Dr))
                T[center] = Q.reshape(Dl, 2, Q.shape[1])
                T[center + 1] = np.tensordot(R, T[center + 1], axes=(1, 0))
                center += 1
            while center > to:
                A = T[center]; Dl, _, Dr = A.shape
                Q, R = np.linalg.qr(A.reshape(Dl, 2 * Dr).T)
                T[center] = Q.T.reshape(Q.shape[1], 2, Dr)
                T[center - 1] = np.tensordot(T[center - 1], R.T, axes=(2, 0))
                center -= 1

        for pos, j in enumerate(self.order):
            ds = dm.dets[j]
            # open new detectors
            for d in ds:
                lab = ("D", d)
                if lab in where:
                    continue
                kd = self.key[d]
                # insertion index: first site with key > kd
                ins = 0
                while ins < len(keys) and keys[ins] < kd:
                    ins += 1
                # bond dimension at insertion point
                if ins == 0:
                    D = T[0].shape[0]
                elif ins == len(T):
                    D = T[-1].shape[2]
                else:
                    D = T[ins].shape[0]
                A = np.zeros((D, 2, D)); A[:, 0, :] = np.eye(D)
                T.insert(ins, A); labels.insert(ins, lab); keys.insert(ins, kd)
                where[lab] = True
                if center >= ins:
                    center += 1
            # apply fault
            sites = [labels.index(("D", d)) for d in ds]
            om = int(dm.obs[j])
            for k in range(nobs):
                if (om >> k) & 1:
                    sites.append(labels.index(("L", k)))
            sites.sort()
            a, b = sites[0], sites[-1]
            sset = set(sites)
            p = dm.p[j]; q = 1.0 - p
            move_center(a)
            if a == b:
                A = T[a]
                T[a] = q * A + p * A[:, ::-1, :]
            else:
                # first site: concat along right bond: [q A, p XA]
                A = T[a]
                XA = A[:, ::-1, :]
                M = np.concatenate([q * A, p * XA], axis=2)  # (Dl,2,2Dr)
                for k in range(a + 1, b + 1):
                    Dl, _, Dr = M.shape
                    U, S, Vh = np.linalg.svd(M.reshape(Dl * 2, Dr), full_matrices=False)
                    keep = int(np.sum(S > cutoff * S[0])) if S[0] > 0 else 1
                    keep = max(1, min(keep, chi))
                    U = U[:, :keep]; S = S[:keep]; Vh = Vh[:keep]
                    T[k - 1] = U.reshape(Dl, 2, keep)
                    SV = S[:, None] * Vh  # (keep, 2*Dl_k) where Dl_k = T[k].shape[0]
                    B = T[k]
                    XB = B[:, ::-1, :] if k in sset else B
                    Dk = B.shape[0]
                    left = np.tensordot(SV[:, :Dk], B, axes=(1, 0))
                    right = np.tensordot(SV[:, Dk:], XB, axes=(1, 0))
                    if k < b:
                        M = np.concatenate([left, right], axis=2)
                    else:
                        M = left + right
                T[b] = M
                center = b
            # normalise
            nrm = np.linalg.norm(T[center])
            if nrm > 0:
                T[center] /= nrm
                logscale += np.log(nrm)
            # close detectors
            for d in self.closes_after.get(pos, []):
                lab = ("D", d)
                k = labels.index(lab)
                move_center(k)
                s = int(syndrome[d])
                Mx = T[k][:, s, :]  # (Dl, Dr)
                del T[k]; del labels[k]; del keys[k]; del where[lab]
                if k < len(T):
                    T[k] = np.tensordot(Mx, T[k], axes=(1, 0))
                    center = k
                else:
                    T[k - 1] = np.tensordot(T[k - 1], Mx, axes=(2, 0))
                    center = k - 1
                nrm = np.linalg.norm(T[center])
                if nrm > 0:
                    T[center] /= nrm
                    logscale += np.log(nrm)
                else:
                    return np.full(1 << nobs, np.nan), -np.inf
            self.stats["max_sites"] = max(self.stats["max_sites"], len(T))
            self.stats["max_bond"] = max(self.stats["max_bond"], max(t.shape[2] for t in T))
        # contract remaining logical sites
        v = T[0]
        for k in range(1, len(T)):
            v = np.tensordot(v, T[k], axes=(v.ndim - 1, 0))
        v = v.reshape(-1)
        # site order of logicals is by lpos; map to class index bits
        labs = [lab[1] for lab in labels]
        out = np.zeros(1 << nobs)
        for idx in range(1 << nobs):
            c = 0
            for pos_i, k in enumerate(labs):
                bit = (idx >> (len(labs) - 1 - pos_i)) & 1
                c |= bit << k
            out[c] = v[idx]
        for om, pr in self.dm.undetectable:
            out = (1 - pr) * out + pr * out[np.arange(len(out)) ^ om]
        tot = out.sum()
        return out / tot, logscale + np.log(tot)

    def class_probs(self, syndromes):
        post = np.zeros((syndromes.shape[0], 1 << self.dm.num_observables))
        logp = np.zeros(syndromes.shape[0])
        for i in range(syndromes.shape[0]):
            post[i], logp[i] = self.decode_one(syndromes[i])
        return post, logp
