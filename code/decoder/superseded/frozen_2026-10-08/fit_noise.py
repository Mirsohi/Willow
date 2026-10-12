"""Maximum-likelihood fitting of a parametric noise model to syndrome (+ logical-outcome) data, with exact
likelihoods and exact gradients from the forward-backward contraction (ml_fb_fast = ml_fb_exact on a compact, batched layout), plus error bars and
misspecification diagnostics. Exact, so d = 3 only (any number of rounds).

Families (all share a fixed fault/symptom set, aligned by (detectors, observable mask)):
  CircuitFamily(make, theta0, names)  p_j(theta) from Stim's DEM of make(theta), e.g. si_family(ideal, p2, pm, p1);
                                      dp_j/dln(theta) by central differences of the DEM (smooth, exact rates)
  DemClassScale(dm0, classes)         p_j = p0_j * s_{class(j)}: per-class scale factors on a GIVEN DEM (e.g. a
                                      shipped Google DEM); dp_j/dln s_c = p_j [class(j) = c] (analytic)
Objective: sum_s ln P_theta(s, L_s) (joint, default) or sum_s ln P_theta(s) (syndrome only), in phi = ln theta.
Per-shot score: g_s = sum_j (q_j(s) - p_j)/(p_j (1 - p_j)) dp_j/dphi, q_j(s) = P_theta(e_j = 1 | s [, L]) exact.
Optimiser: Fisher scoring with the empirical score covariance (BHHH-type, centred; step halving); stops
when every step is < 0.02 standard errors. Errors: OPG (sum g g^T)^-1, Hessian (-H)^-1 (H by central differences of the analytic
score), sandwich H^-1 OPG H^-1 (valid under misspecification). Diagnostics: information-matrix test
(per-shot g_c^2 + H_cc has mean 0 under a correct model) and per-class calibration E_s[sum_class q_j] vs
sum_class p_j (equal under a correct model; at the MLE only k linear combinations are forced to agree).
"""
import numpy as np
from dem_utils import parse_dem, DemMatrices, best_sweep_order
from ml_fb_fast import ForwardBackwardFast  # validated against ml_fb_exact to 1e-15 (tests/test_fb_fast.py)


def symptom_class(dm: DemMatrices, j: int, time_split: bool = False) -> str:
    """boundary (1 detector), timelike (same stabiliser, consecutive rounds), spacelike (two stabilisers, same
    round), diagonal (two stabilisers, different rounds), hyperedge (>= 3 detectors); '+L' if it flips L.
    time_split: also tag faults touching the first ('@first') or the last ('@last') detector layer, which carry
    the reset / final-measurement mechanisms and scale differently from the bulk."""
    ds = dm.dets[j]
    tag = "+L" if int(dm.obs[j]) else ""
    if time_split:
        ts = [dm.coords[d][-1] for d in ds if d in dm.coords and len(dm.coords[d]) >= 3]
        if ts:
            allt = [c[-1] for c in dm.coords.values() if len(c) >= 3]
            t0, t1 = min(allt), max(allt)
            if abs(min(ts) - t0) < 1e-9:
                tag += "@first"
            elif abs(max(ts) - t1) < 1e-9:
                tag += "@last"
    if len(ds) == 1:
        return "boundary" + tag
    if len(ds) >= 3:
        return "hyperedge" + tag
    a, b = dm.coords.get(ds[0]), dm.coords.get(ds[1])
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return "edge" + tag
    if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
        return "timelike" + tag
    if abs(a[-1] - b[-1]) < 1e-9:
        return "spacelike" + tag
    return "diagonal" + tag


class ParametricDEM:
    """Base class: fixed symptom set (from dm0), p(theta) and dp/dln(theta) supplied by subclasses."""
    names = ()

    def _init_base(self, dm0: DemMatrices):
        self.base = dm0
        self.keys = [(d, int(o)) for d, o in zip(dm0.dets, dm0.obs)]
        self.index = {k: i for i, k in enumerate(self.keys)}
        assert len(self.index) == len(self.keys), "duplicate symptoms"
        self.order = best_sweep_order(dm0)[0]
        self.classes = np.array([symptom_class(dm0, j) for j in range(dm0.num_faults)])
        self.class_names = sorted(set(self.classes))
        self.onehot = np.zeros((dm0.num_faults, len(self.class_names)))
        for c, nm in enumerate(self.class_names):
            self.onehot[self.classes == nm, c] = 1.0

    def probs(self, theta):
        raise NotImplementedError

    def jacobian(self, theta):
        raise NotImplementedError

    def dm_at(self, theta):
        b = self.base
        return DemMatrices(b.num_detectors, b.num_observables, b.dets, b.obs, self.probs(theta), b.coords, [])

    def evaluate(self, theta, det, L=None, want_q=False):
        """Mean log-likelihood per shot, per-shot scores wrt ln(theta) (n, k), log P per shot, sum of q_j over
        shots and (want_q) per-shot class sums of q_j (n, n_classes)."""
        dm = self.dm_at(theta)
        fb = ForwardBackwardFast(dm, order=self.order)
        r = fb.run(det, L=L, J=self.jacobian(theta), store_q=want_q)
        ok = np.isfinite(r["logp"])
        out = {"ll": float(np.mean(r["logp"][ok])), "grads": r["grads"][ok], "logp": r["logp"], "qsum": r["qsum"],
               "nonfinite": int((~ok).sum()), "p": dm.p}
        if want_q:
            out["class_q"] = (r["q"] @ self.onehot)[ok]
        return out


class CircuitFamily(ParametricDEM):
    def __init__(self, make, theta0, names):
        """make(theta) -> noisy stim.Circuit. The symptom set must not change with theta (all rates > 0)."""
        self.make = make
        self.names = tuple(names)
        self._init_base(self._parse(theta0))

    def _parse(self, theta):
        return parse_dem(self.make(theta).detector_error_model(decompose_errors=False))

    def probs(self, theta):
        dm = self._parse(theta)
        p = np.zeros(len(self.keys))
        for d, o, q in zip(dm.dets, dm.obs, dm.p):
            k = (d, int(o))
            if k not in self.index:
                raise KeyError("symptom set changed with theta")
            p[self.index[k]] = q
        return p

    def jacobian(self, theta, rel=1e-4):
        J = np.zeros((len(self.keys), len(theta)))
        for c in range(len(theta)):
            tp = np.array(theta, float); tm = np.array(theta, float)
            tp[c] *= np.exp(rel); tm[c] *= np.exp(-rel)
            J[:, c] = (self.probs(tp) - self.probs(tm)) / (2 * rel)
        return J


class DemClassScale(ParametricDEM):
    def __init__(self, dm0: DemMatrices, classes=None, time_split=False):
        """theta_c = scale factor of class c (1 = the given DEM). classes: per-fault labels (default: symptom
        classes; time_split=True separates faults touching the first / last detector layer)."""
        self._init_base(dm0)
        if classes is None and time_split:
            classes = [symptom_class(dm0, j, time_split=True) for j in range(dm0.num_faults)]
        if classes is not None:
            self.classes = np.asarray(classes)
            self.class_names = sorted(set(self.classes))
            self.onehot = np.zeros((dm0.num_faults, len(self.class_names)))
            for c, nm in enumerate(self.class_names):
                self.onehot[self.classes == nm, c] = 1.0
        self.names = tuple(self.class_names)
        self.p0 = dm0.p.copy()

    def probs(self, theta):
        return np.clip(self.p0 * (self.onehot @ np.asarray(theta, float)), 1e-15, 0.5 - 1e-9)

    def jacobian(self, theta):
        return self.probs(theta)[:, None] * self.onehot


def fit_bhhh(model: ParametricDEM, det, L, theta0, max_iter=15, step_tol=0.02, verbose=True):
    """BHHH ascent in phi = ln theta. Stops when every component of the step is < step_tol standard errors."""
    phi = np.log(np.asarray(theta0, float))
    ev = model.evaluate(np.exp(phi), det, L, want_q=True)
    hist = [{"theta": np.exp(phi).tolist(), "ll_per_shot": ev["ll"]}]
    n_eval = 1
    for it in range(max_iter):
        G = ev["grads"]; n = G.shape[0]
        g = G.mean(axis=0)
        # Fisher scoring with the empirical covariance of the per-shot scores (centred OPG): equals the
        # Fisher information at theta when the model holds; the uncentred BHHH matrix shrinks steps far from
        # the optimum (it contains g g^T). Both coincide at the MLE, where the error bars are computed.
        Gc = G - g
        Iinfo = (Gc.T @ Gc) / n
        step = np.linalg.solve(Iinfo, g)
        se = np.sqrt(np.diag(np.linalg.inv(Iinfo)) / n)
        if np.all(np.abs(step) < step_tol * se):
            break
        lam = 1.0
        while True:
            phi_new = phi + lam * step
            ev_new = model.evaluate(np.exp(phi_new), det, L, want_q=True); n_eval += 1
            if ev_new["ll"] >= ev["ll"] - 1e-12 or lam < 1e-3:
                break
            lam *= 0.5
        gain = ev_new["ll"] - ev["ll"]
        phi, ev = phi_new, ev_new
        hist.append({"theta": np.exp(phi).tolist(), "ll_per_shot": ev["ll"], "step_in_se": (lam * step / se).tolist(),
                     "lambda": lam})
        if verbose:
            print(f"    iter {it}: theta={np.array2string(np.exp(phi), precision=5)} ll/shot={ev['ll']:.7f} "
                  f"gain={gain:.2e} max|step|/se={np.max(np.abs(lam * step / se)):.3g}", flush=True)
    G = ev["grads"]
    opg = G.T @ G
    return {"theta_hat": np.exp(phi), "phi_hat": phi, "opg": opg, "cov_opg": np.linalg.inv(opg), "history": hist,
            "eval": ev, "n": G.shape[0], "n_eval": n_eval, "mean_score_at_hat": G.mean(axis=0)}


def hessian_and_im_test(model: ParametricDEM, phi_hat, det, L, G_hat, h=0.02):
    """Total Hessian of ln L wrt phi (central differences of the analytic per-shot scores) and the diagonal
    information-matrix test: d_sc = g_sc^2 + H_s,cc has mean 0 under a correct model (z ignores the O(1/n)
    effect of estimating theta)."""
    k = len(phi_hat)
    n = G_hat.shape[0]
    Hs_diag = np.zeros((n, k))
    H = np.zeros((k, k))
    for c in range(k):
        e = np.zeros(k); e[c] = h
        gp = model.evaluate(np.exp(phi_hat + e), det, L)["grads"]
        gm = model.evaluate(np.exp(phi_hat - e), det, L)["grads"]
        dG = (gp - gm) / (2 * h)
        H[:, c] = dG.sum(axis=0)
        Hs_diag[:, c] = dG[:, c]
    H = 0.5 * (H + H.T)
    d = G_hat ** 2 + Hs_diag
    im = {"mean_d": d.mean(axis=0).tolist(), "z": (d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(n))).tolist(),
          "opg_diag_per_shot": (G_hat ** 2).mean(axis=0).tolist(), "minus_hess_diag_per_shot": (-Hs_diag).mean(axis=0).tolist()}
    return H, im


def _inv(A):
    """Inverse, falling back to the pseudo-inverse when A is singular (e.g. a class driven to a zero rate, whose
    Hessian row vanishes); the corresponding standard errors are then not meaningful and are reported as NaN."""
    try:
        return np.linalg.inv(A), False
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A), True


def error_bars(opg, H):
    cov_opg, s1 = _inv(opg)
    cov_h, s2 = _inv(-H)
    cov_sw = cov_h @ opg @ cov_h
    out = {"se_opg": np.sqrt(np.abs(np.diag(cov_opg))), "se_hessian": np.sqrt(np.abs(np.diag(cov_h))),
           "se_sandwich": np.sqrt(np.abs(np.diag(cov_sw))), "cov_opg": cov_opg, "cov_hessian": cov_h, "cov_sandwich": cov_sw,
           "singular": bool(s1 or s2)}
    if s1 or s2:
        dead = (np.abs(np.diag(H)) < 1e-12 * max(np.abs(np.diag(H)).max(), 1e-300)) | (np.diag(opg) <= 0)
        for k in ("se_opg", "se_hessian", "se_sandwich"):
            out[k] = np.where(dead, np.nan, out[k])
    return out


def class_calibration(model: ParametricDEM, class_q, p):
    """Per symptom class: mean_s sum_class q_j(s) vs sum_class p_j, with z from the per-shot spread."""
    n = class_q.shape[0]
    out = {}
    for c, nm in enumerate(model.class_names):
        sel = model.onehot[:, c] > 0
        e = float(np.sum(p[sel])); o = float(class_q[:, c].mean())
        sd = float(class_q[:, c].std(ddof=1) / np.sqrt(n))
        out[nm] = {"n_faults": int(sel.sum()), "prior_per_shot": e, "posterior_per_shot": o,
                   "ratio": o / e if e > 0 else float("nan"), "z": (o - e) / sd if sd > 0 else float("nan")}
    return out
