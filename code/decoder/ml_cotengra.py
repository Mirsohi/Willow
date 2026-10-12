"""Independent exact ML contraction of the DEM tensor network with a cotengra-optimised path
(positive/direct formulation; used to cross-check ml_exact and ml_mps on small/medium instances).

Network: each fault j owns a hyper-index e_j (dim 2) with weight vector [1-p_j, p_j]. Each detector i is
a chain of rank-3 XOR tensors accumulating the parity of its incident e_j, terminated by the observed
syndrome bit s_i (a unit vector). The observable is a parity chain whose last index is left open, so the
contraction returns the vector [P(s, l=0), P(s, l=1)]. Same structure for every shot; only the terminal
syndrome vectors change, so one path is optimised and reused.
"""
import numpy as np
import cotengra as ctg

XOR3 = np.zeros((2, 2, 2))
for a in range(2):
    for b in range(2):
        XOR3[a, b, a ^ b] = 1.0


class CotengraExact:
    def __init__(self, dm, max_time=60, minimize="flops", seed=0, max_repeats=256):
        assert dm.num_observables == 1
        self.dm = dm
        inputs, arrays, kinds = [], [], []
        # every use of fault j (one per incident detector, plus the observable) gets its own index e{j}_{u};
        # a diagonal COPY tensor carrying the weight [1-p, p] ties them together (no hyper-indices).
        uses = {j: [] for j in range(dm.num_faults)}
        def use(j):
            name = f"e{j}_{len(uses[j])}"; uses[j].append(name); return name
        inc = {i: [] for i in range(dm.num_detectors)}
        for j, ds in enumerate(dm.dets):
            for d in ds:
                inc[d].append(j)
        self.term_pos = {}
        cnt = 0
        def chain(js, tag):
            nonlocal cnt
            cur = use(js[0])
            for j in js[1:]:
                nxt = f"c{cnt}"; cnt += 1
                inputs.append((cur, use(j), nxt)); arrays.append(XOR3); kinds.append(("x", tag))
                cur = nxt
            return cur
        for i in range(dm.num_detectors):
            if not inc[i]:
                continue
            last = chain(inc[i], i)
            self.term_pos[i] = len(inputs)
            inputs.append((last,)); arrays.append(np.array([1.0, 0.0])); kinds.append(("s", i))
        obs_js = [j for j in range(dm.num_faults) if dm.obs[j] & 1]
        if obs_js:
            out = chain(obs_js, "L")
        else:
            out = "Lfree"
            inputs.append((out,)); arrays.append(np.array([1.0, 0.0]))
        for j in range(dm.num_faults):
            k = len(uses[j])
            if k == 0:
                continue  # fault touches nothing: contributes a factor (1-p)+p = 1
            t = np.zeros((2,) * k); t[(0,) * k] = 1 - dm.p[j]; t[(1,) * k] = dm.p[j]
            inputs.append(tuple(uses[j])); arrays.append(t); kinds.append(("w", j))
        self.inputs, self.arrays, self.output = inputs, arrays, (out,)
        size_dict = {ix: 2 for inp in inputs for ix in inp}
        # optlib="random" avoids optuna (whose create_study() rejects cotengra's seed kwarg in this environment)
        opt = ctg.HyperOptimizer(minimize=minimize, max_time=max_time, max_repeats=max_repeats,
                                 parallel=False, progbar=False, optlib="random")
        self.tree = opt.search(inputs, self.output, size_dict)
        self.width = self.tree.contraction_width()
        self.cost = self.tree.contraction_cost()

    def class_probs(self, syndromes):
        out = np.zeros((syndromes.shape[0], 2))
        logp = np.zeros(syndromes.shape[0])
        e0, e1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        for k in range(syndromes.shape[0]):
            arrs = list(self.arrays)
            for i, pos in self.term_pos.items():
                arrs[pos] = e1 if syndromes[k, i] else e0
            v = self.tree.contract(arrs)
            tot = v.sum()
            out[k] = v / tot
            logp[k] = np.log(tot)
        return out, logp
