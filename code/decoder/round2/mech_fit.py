"""Round 2, amendment A1 (notes/decoder/willow_prereg_r2.md section 8): family "mech" and the fits T-hat_mech and D-hat_mech.

Family: logit p_j = logit p0_j + delta_t(j) on the shipped RL DEM. Template t(j) = (layer tag, observable flag,
geometry): layer tag "first" if the fault touches detector layer 0, "last" if it touches the final layer, else
"bulk"; geometry = the sorted (x, y, t - t_min) of its detectors (first three detector coordinates). One delta per
template, shared by every round and by the 4 W5 leaves of a series. Rare templates (fewer than 30 expected fault events
in the series' fit sample, sum of p0_j * n_fit over its faults) share one delta per F1 class name (T-hat's label of the
fault in its leaf); a shared group still under 30 joins one "rare" delta; if that is still under 30 it is fixed at 0.

Fits (pooled per series, T-hat's 5,000 fit shots per leaf):
  T-hat_mech: maximise sum ln P(s) (the frozen ml_fb_fast kernel without L), no penalty, start delta = 0.
  D-hat_mech: maximise sum ln P(L_obs | s) - (lambda / 2) sum delta^2, lambda = 4 (fb_cond kernel), start at D-hat's F1
              solution mapped to templates (mean ln theta-hat over each template's faults), or 0 if none is given.
Optimiser: Fisher scoring in summed log-likelihood units with the centred outer product of the per-shot scores (plus
lambda on the diagonal for D-hat_mech; info="expected" would use the conditional Fisher information instead); a step
larger than 1 is scaled to 1 in its largest component; step halving on the (penalised) objective; stop when every
component is below 0.02 of its SE (from the information), checked after every step; cap 15 iterations; singular
information: the parameter with the smallest information is fixed at 0 (logged).
Quasi-Newton (BFGS) updating as cond_fit.fit_series (PR2 section 7, 2026-10-10; secant="bfgs"); the SE of the stop
rule stays the (penalised) information's; cap 15 iterations as pre-registered in A1.
Checkpoints (ckpt=path prefix) as cond_fit.fit_series.
stop (optional callable), as cond_fit.fit_series: checked before every new evaluation; when it returns True the fit
returns at once with stopped=True and keeps its checkpoint (run_mech's CPU budget). Run control only.
"""
import math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/decoder (frozen, read-only)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dem_utils import DemMatrices
from ml_fb_fast import ForwardBackwardFast
from ml_exact_fast import FastExactDecoder
from fb_cond import ForwardBackwardCond
from cond_fit import cond_weights, fingerprint, save_ckpt, load_ckpt, StepMatrix, describe

MIN_EVENTS = 30.0
LAMBDA_D = 4.0


def template_keys(dm):
    allt = [c[2] for c in dm.coords.values() if len(c) >= 3]
    t0, t1 = min(allt), max(allt)
    keys = []
    for j, ds in enumerate(dm.dets):
        xyz = [tuple(float(v) for v in dm.coords[d][:3]) for d in ds]
        assert xyz and all(len(c) == 3 for c in xyz), "detector without coordinates"
        ts = [c[2] for c in xyz]
        tag = "first" if abs(min(ts) - t0) < 1e-9 else ("last" if abs(max(ts) - t1) < 1e-9 else "bulk")
        tm = min(ts)
        geo = tuple(sorted((round(c[0], 6), round(c[1], 6), round(c[2] - tm, 6)) for c in xyz))
        keys.append((tag, int(dm.obs[j] != 0), geo))
    return keys


def key_name(k):
    return f"{k[0]}|L{k[1]}|" + ";".join(f"{x:g},{y:g},{t:g}" for x, y, t in k[2])


class MechSeries:
    """Parameter map of one series. leaves: cond_fit.Leaf objects (their model holds the RL DEM, the order and T-hat's
    class labels in model.classes)."""

    def __init__(self, leaves, n_fit_per_leaf=None):
        self.leaves = leaves
        keys = [template_keys(lf.model.base) for lf in leaves]
        ev = {}
        for lf, ks in zip(leaves, keys):
            nfit = n_fit_per_leaf or len(lf.L_fit)
            for k, p in zip(ks, lf.model.p0):
                ev[k] = ev.get(k, 0.0) + p * nfit
        self.n_templates = len(ev)
        big = {k for k, v in ev.items() if v >= MIN_EVENTS}
        pname = []
        for lf, ks in zip(leaves, keys):
            pname.append([key_name(k) if k in big else "R:" + str(c) for k, c in zip(ks, lf.model.classes)])
        gev = {}
        for lf, names in zip(leaves, pname):
            for nm, p in zip(names, lf.model.p0):
                if nm.startswith("R:"):
                    gev[nm] = gev.get(nm, 0.0) + p * (n_fit_per_leaf or len(lf.L_fit))
        small = {nm for nm, v in gev.items() if v < MIN_EVENTS}
        pname = [[("R:rare" if nm in small else nm) for nm in names] for names in pname]
        rare_ev = sum(v for nm, v in gev.items() if nm in small)
        self.fixed_rare = bool(small) and bool(rare_ev < MIN_EVENTS)
        if self.fixed_rare:
            pname = [[(None if nm == "R:rare" else nm) for nm in names] for names in pname]
        self.names = sorted({nm for names in pname for nm in names if nm is not None})
        index = {nm: i for i, nm in enumerate(self.names)}
        self.idx = [np.array([index[nm] if nm is not None else -1 for nm in names], np.int64) for names in pname]
        self.info = {"templates": int(self.n_templates), "templates_ge_30_events": len(big), "rare_groups": sorted(gev),
                     "rare_groups_merged": sorted(small), "rare_fixed_at_0": self.fixed_rare, "parameters": len(self.names),
                     "template_flags_L": int(sum(1 for k in ev if k[1]))}
        self.keys = keys

    def start_from_classes(self, class_names, ln_theta):
        """Mean ln theta-hat of the F1 class of each parameter's faults (D-hat's start)."""
        m = dict(zip(class_names, ln_theta))
        s = np.zeros(len(self.names)); c = np.zeros(len(self.names))
        for lf, ix in zip(self.leaves, self.idx):
            for i, cl in zip(ix, lf.model.classes):
                if i >= 0 and cl in m:
                    s[i] += m[cl]; c[i] += 1
        return np.where(c > 0, s / np.maximum(c, 1), 0.0)

    def probs(self, l, delta):
        lf, ix = self.leaves[l], self.idx[l]
        p0 = lf.model.p0
        d = np.where(ix >= 0, delta[np.maximum(ix, 0)], 0.0)
        lo = np.log(p0) - np.log1p(-p0) + d
        return np.clip(1.0 / (1.0 + np.exp(-lo)), 1e-15, 0.5 - 1e-9)

    def leaf_cols(self, l):
        return np.unique(self.idx[l][self.idx[l] >= 0])

    def jac(self, l, p):
        ix = self.idx[l]; cols = self.leaf_cols(l)
        pos = {c: i for i, c in enumerate(cols)}
        J = np.zeros((len(p), len(cols)))
        for j in np.where(ix >= 0)[0]:
            J[j, pos[ix[j]]] = p[j] * (1 - p[j])
        return J, cols

    def dm(self, l, delta):
        b = self.leaves[l].model.base
        return DemMatrices(b.num_detectors, b.num_observables, b.dets, b.obs, self.probs(l, delta), b.coords, [])

    def evaluate(self, delta, objective):
        """objective 'marg': ln P(s) per shot (frozen kernel without L); 'cond': ln P(L_obs | s) (fb_cond)."""
        t0 = time.time()
        vals, Gs = [], []
        K = len(self.names)
        for l, lf in enumerate(self.leaves):
            p = self.probs(l, delta)
            J, cols = self.jac(l, p)
            dm = self.dm(l, delta)
            if objective == "marg":
                r = ForwardBackwardFast(dm, order=lf.model.order).run(lf.det_fit, L=None, J=J)
                v, g = r["logp"], r["grads"]
            else:
                r = ForwardBackwardCond(dm, order=lf.model.order).run(lf.det_fit, lf.L_fit, J)
                v, g = r["cll"], r["g_cond"]
            G = np.zeros((len(v), K)); G[:, cols] = g
            vals.append(v); Gs.append(G)
        v = np.concatenate(vals); G = np.concatenate(Gs)
        ok = np.isfinite(v)
        w = cond_weights(v[ok]) if objective == "cond" else np.zeros(0)
        return {"sum": float(v[ok].sum()), "mean": float(v[ok].mean()), "G": G[ok], "w": w, "nonfinite": int((~ok).sum()),
                "seconds": time.time() - t0}

    def decode_test(self, l, delta):
        lf = self.leaves[l]
        post, logp = FastExactDecoder(self.dm(l, delta), order=lf.model.order).class_probs(lf.det_test)
        pred = (np.nan_to_num(post[:, 1], nan=0.0) > 0.5).astype(np.uint8)
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = (np.log(post[:, 1]) - np.log(post[:, 0])).astype(np.float32)
        return pred, llr, logp


def fit_mech(ms, objective, lam=0.0, start=None, max_iter=15, step_tol=0.02, max_step=1.0, log=print, ckpt=None,
             secant="bfgs", info="opg", stop=None):
    K = len(ms.names)
    cpu0 = time.process_time(); cpu_prev = 0.0
    fp = fingerprint(ms.leaves, ms.names) + f"|{objective}|{lam}|info={info}|secant={secant}"
    got = load_ckpt(ckpt, fp) if ckpt else None
    prev = None

    def pen_obj(ev, d):
        return ev["sum"] - 0.5 * lam * float(d @ d)
    if got:
        st, ar = got
        delta = np.array(st["delta"], float); fixed = np.array(st["fixed"], bool)
        it, n_eval, hist, events, cpu_prev = st["it"], st["n_eval"], st["hist"], st["events"], st["cpu_seconds"]
        ev = {"sum": st["sum"], "mean": st["mean"], "nonfinite": st["nonfinite"], "seconds": st["seconds"],
              "G": ar["G"], "w": ar["w"]}
        if st.get("prev"):
            prev = {"d": np.array(st["prev"]["d"]), "g": np.array(st["prev"]["g"]), "free": np.array(st["prev"]["free"])}
        sm = StepMatrix(secant, st.get("step_matrix"), ar.get("Bq"))
        log(f"  resumed from checkpoint {ckpt} (iteration {it}, {n_eval} evaluations, saved {st['saved']})")
    else:
        delta = np.zeros(K) if start is None else np.array(start, float)
        fixed = np.zeros(K, bool)
        ev = ms.evaluate(delta, objective); n_eval = 1
        hist = [{"objective": pen_obj(ev, delta), "mean_ll": ev["mean"], "seconds": ev["seconds"]}]
        events, it = [], 0
        sm = StepMatrix(secant)
        log(f"    eval: mean {objective} ll = {ev['mean']:.7f}, penalised sum = {pen_obj(ev, delta):.4f} ({ev['seconds']:.0f} s)")
    obj = pen_obj(ev, delta)

    def checkpoint():
        if ckpt:
            pv = None if prev is None else {k: v.tolist() for k, v in prev.items()}
            save_ckpt(ckpt, {"fingerprint": fp, "names": ms.names, "delta": delta.tolist(), "fixed": fixed.tolist(),
                             "it": it, "n_eval": n_eval, "hist": hist, "events": events, "sum": ev["sum"], "prev": pv,
                             "step_matrix": sm.state(),
                             "mean": ev["mean"], "nonfinite": ev["nonfinite"], "seconds": ev["seconds"],
                             "cpu_seconds": cpu_prev + time.process_time() - cpu0},
                      dict({"G": ev["G"], "w": ev["w"]}, **({} if sm.B is None else {"Bq": sm.B})))
    if not got:
        checkpoint()

    def information(G, w, free):
        if objective == "cond" and info == "expected":
            Gw = G[:, free] * w[:, None]
            return Gw.T @ Gw + lam * np.eye(len(free))
        Gc = G[:, free] - G[:, free].mean(axis=0)
        return Gc.T @ Gc + lam * np.eye(len(free))
    converged, stopped, last_check = False, False, None
    while True:
        G = ev["G"]; g = G.sum(axis=0) - lam * delta
        free = np.where(~fixed)[0]
        if len(free) == 0:
            events.append({"iteration": it, "reason": "every parameter fixed"}); break
        I = information(G, ev["w"], free)
        sing = False
        sec = None
        try:
            cond = np.linalg.cond(I)
            if not np.isfinite(cond) or cond > 1e12:
                sing = True
            else:
                Iinv = np.linalg.inv(I)
                same = prev is not None and np.array_equal(prev["free"], free)
                s_, y_ = (delta[free] - prev["d"][free], prev["g"][free] - g[free]) if same else (None, None)
                B, sec = sm.matrix(I, s_, y_, it)
                step_f = np.linalg.solve(B, g[free]); se_f = np.sqrt(np.diag(Iinv))
                sing = not (np.all(np.isfinite(step_f)) and np.all(np.isfinite(se_f)))
        except np.linalg.LinAlgError:
            sing = True
        if sing:
            if stop is not None and stop():
                stopped = True
                log("  stopped before the next evaluation (stop condition; checkpoint kept)")
                break
            c = free[int(np.argmin(np.diag(I)))]
            fixed[c] = True; delta[c] = 0.0
            events.append({"iteration": it, "fixed_at_0": ms.names[c], "reason": "singular information"})
            log(f"  singular information: fixing {ms.names[c]} at 0")
            ev = ms.evaluate(delta, objective); n_eval += 1; obj = pen_obj(ev, delta)
            prev = None; sm.reset()
            checkpoint()
            continue
        rel = np.abs(step_f) / se_f
        last_check = {"iteration": it, "max_step_in_se": float(rel.max()), "argmax": ms.names[free[int(np.argmax(rel))]],
                      "secant": sec}
        log(f"  iter {it}: max |step|/se = {rel.max():.3g} ({ms.names[free[int(np.argmax(rel))]]}){describe(sec)}")
        if np.all(rel < step_tol):
            converged = True; break
        if it >= max_iter:
            break
        if stop is not None and stop():
            stopped = True
            log("  stopped before the next evaluation (stop condition; checkpoint kept)")
            break
        scale = min(1.0, max_step / float(np.max(np.abs(step_f))))
        lamb = scale
        while True:
            d_new = delta.copy(); d_new[free] += lamb * step_f
            ev_new = ms.evaluate(d_new, objective); n_eval += 1
            obj_new = pen_obj(ev_new, d_new)
            log(f"    eval: mean {objective} ll = {ev_new['mean']:.7f}, penalised sum = {obj_new:.4f} "
                f"({ev_new['seconds']:.0f} s, lambda {lamb:.3g})")
            if obj_new >= obj - 1e-9 or lamb < 1e-3 * scale:
                break
            lamb *= 0.5
        sm.stepped(scale)
        prev = {"d": delta.copy(), "g": g.copy(), "free": free.copy()}
        delta, ev, obj = d_new, ev_new, obj_new
        it += 1
        hist.append({"objective": obj, "mean_ll": ev["mean"], "lambda": lamb, "scale": scale,
                     "max_step_in_se": float(np.max(np.abs(lamb * step_f / se_f))), "seconds": ev["seconds"]})
        checkpoint()
    G = ev["G"]; free = np.where(~fixed)[0]
    se = np.full(K, np.nan); se_opg = np.full(K, np.nan)
    try:
        se[free] = np.sqrt(np.diag(np.linalg.inv(information(G, ev["w"], free))))
        se_opg[free] = np.sqrt(np.diag(np.linalg.inv(G[:, free].T @ G[:, free] + lam * np.eye(len(free)))))
    except np.linalg.LinAlgError:
        pass
    return {"objective_type": objective, "lambda": lam, "names": ms.names, "delta_hat": delta.tolist(),
            "se": se.tolist(), "se_opg": se_opg.tolist(),
            "information": ("conditional Fisher" if (objective == "cond" and info == "expected") else "centred OPG")
            + (" + lambda" if lam else ""), "secant": secant,
            "fixed": [ms.names[c] for c in np.where(fixed)[0]], "events": events,
            "converged": converged, "stopped": stopped, "iterations": it, "n_eval": n_eval, "history": hist, "objective": obj,
            "last_convergence_check": last_check, "mean_ll": ev["mean"], "n_shots": int(G.shape[0]), "family": ms.info,
            "cpu_seconds_fit_total": cpu_prev + time.process_time() - cpu0, "fingerprint": fp}
