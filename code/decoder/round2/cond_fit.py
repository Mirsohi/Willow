"""Round 2 (notes/decoder/willow_prereg_r2.md): the conditional-likelihood fit D-hat of family F1, pooled per series.

F1: the class scales of T-hat (p_j = p0_j * theta_c on the shipped RL DEM), with each leaf's classes rebuilt exactly
as in round 1 (fit_willow.class_labels with the first / last layer split and 30 expected events in 5,000 fit shots).
One scale per class name is shared by the leaves of a series; each leaf keeps its own assignment of faults to names.

Objective: sum over the series' fit shots of ln P_theta(L_obs | s), exact (fb_cond.ForwardBackwardCond).
Optimiser (pre-registered): Fisher scoring in ln theta with the centred outer product of the per-shot conditional scores
(info="opg"); start at theta = 1; step halving until the objective does not decrease; a step larger than 1 in ln theta
is scaled to 1 in its largest component; stop when every component is below 0.02 SE (checked after every step, also
after the last); cap 15 iterations; singular information: the free class with the smallest information is fixed at 1
and the iteration is repeated (logged).
Deviation (PR2 section 7, 2026-10-10, before any real-data fit; secant="bfgs", the default): quasi-Newton updating.
The step matrix starts as the information at the first iterate and is updated after every accepted step with the
Powell-damped BFGS formula (s = phi_k - phi_(k-1), y = g_(k-1) - g_k), so it learns the observed curvature; it restarts
from the information after a class is fixed. The SE of the stop rule is always the information's. run_f1 raises the
iteration cap to 25 (same entry). Options kept for the record: secant="hybrid" (plain scoring until the first uncapped
step or three capped steps in a row, then BFGS), "one" (only the last pair on top of the current information), False
(plain scoring); info="expected" (the conditional Fisher information given s, sum_s [P(L_obs|s) / P(L != L_obs|s)]
g_s g_s^T, exact from the same pass).
Checkpoints (ckpt=path prefix): after the first evaluation and after every accepted step the state and arrays are written
to one file, <ckpt>.ckpt.npz (the state as JSON inside; written to a temporary name, then renamed), so a restart that
loses the latest write leaves the previous complete checkpoint; a restart with the same leaves resumes from it exactly.
stop (optional callable): checked before every new evaluation; when it returns True the fit returns at once with
stopped=True and keeps its checkpoint (run_f1's CPU budget).
"""
import hashlib, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/decoder (frozen, read-only)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import stim
from dem_utils import parse_dem
from fit_noise import DemClassScale
from ml_exact_fast import FastExactDecoder
from fit_willow import class_labels
from fb_cond import ForwardBackwardCond

RL_DEM = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
W5 = (10, 13, 30, 50)


def load_dm(path):
    dm = parse_dem(stim.DetectorErrorModel.from_file(path))
    assert not dm.undetectable, f"{path}: undetectable faults"
    return dm


def t_classes(dm, n_fit=5000):
    """T-hat's per-fault class labels (fit_willow.class_labels, --time-split --min-class-events 30)."""
    labels, merged = class_labels(dm, True, 0, 30.0, n_fit)
    return labels, merged


class Leaf:
    """One leaf: the model (RL DEM + T-hat classes), fit shots and held-out shots."""

    def __init__(self, name, rounds, dm, labels, det_fit, L_fit, det_test=None, L_test=None, extra=None):
        self.name, self.rounds = name, rounds
        self.model = DemClassScale(dm, classes=labels)
        self.names = list(self.model.names)
        self.det_fit = np.ascontiguousarray(det_fit, dtype=np.uint8)
        self.L_fit = np.asarray(L_fit, dtype=np.int64).reshape(-1)
        self.det_test = None if det_test is None else np.ascontiguousarray(det_test, dtype=np.uint8)
        self.L_test = None if L_test is None else np.asarray(L_test, dtype=np.int64).reshape(-1)
        self.extra = extra or {}
        self.cols = None

    def bind(self, series_names):
        self.cols = np.array([series_names.index(nm) for nm in self.names], dtype=np.int64)

    def theta(self, phi_series):
        return np.exp(phi_series[self.cols])

    def evaluate(self, phi_series, K):
        th = self.theta(phi_series)
        dm = self.model.dm_at(th)
        fb = ForwardBackwardCond(dm, order=self.model.order)
        r = fb.run(self.det_fit, self.L_fit, J=self.model.jacobian(th))
        G = np.zeros((len(self.L_fit), K))
        G[:, self.cols] = r["g_cond"]
        return r["cll"], G

    def decode_test(self, phi_series):
        dm = self.model.dm_at(self.theta(phi_series))
        post, logp = FastExactDecoder(dm, order=self.model.order).class_probs(self.det_test)
        P1 = post[:, 1]
        pred = (np.nan_to_num(P1, nan=0.0) > 0.5).astype(np.uint8)
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = (np.log(post[:, 1]) - np.log(post[:, 0])).astype(np.float32)
        return pred, llr, logp


def series_names(leaves):
    names = sorted(set(nm for lf in leaves for nm in lf.names))
    for lf in leaves:
        lf.bind(names)
    return names


def cond_weights(cll):
    """sqrt(P(L_obs | s) / P(L != L_obs | s)) per shot: (w g)(w g)^T is the shot's conditional Fisher information."""
    po = np.exp(cll); pq = -np.expm1(cll)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt(po / pq)
    return np.where(np.isfinite(w) & (pq > 0), w, 0.0)


def objective(leaves, phi, K, log=None):
    t0 = time.time()
    clls, Gs = [], []
    for lf in leaves:
        c, G = lf.evaluate(phi, K)
        clls.append(c); Gs.append(G)
    cll = np.concatenate(clls); G = np.concatenate(Gs)
    ok = np.isfinite(cll)
    out = {"ll": float(np.mean(cll[ok])), "ll_sum": float(np.sum(cll[ok])), "G": G[ok], "w": cond_weights(cll[ok]),
           "nonfinite": int((~ok).sum()), "seconds": time.time() - t0}
    if log:
        log(f"    eval: mean ln P(L|s) = {out['ll']:.7f}  ({out['seconds']:.0f} s, {out['nonfinite']} non-finite)")
    return out


def secant_matrix(A, s, y):
    """Powell-damped BFGS update of the (positive definite) matrix A with the pair (s, y): B s = y_damped."""
    As = A @ s; sAs = float(s @ As); sy = float(s @ y)
    if not (sAs > 0 and np.all(np.isfinite(y))):
        return A, None
    t = 1.0
    if sy < 0.2 * sAs:
        t = 0.8 * sAs / (sAs - sy)
        y = t * y + (1 - t) * As; sy = float(s @ y)
    B = A + np.outer(y, y) / sy - np.outer(As, As) / sAs
    return B, {"curvature_ratio": float(s @ y) / sAs if t == 1.0 else None, "damping": t}


class StepMatrix:
    """The matrix of the scoring step on the free set (see the module docstring for the modes)."""

    def __init__(self, mode, state=None, B=None):
        self.mode, self.B, self.on, self.capped_run, self.last_capped = mode, None, False, 0, None
        if state:
            self.on, self.capped_run, self.last_capped = state["on"], state["capped_run"], state["last_capped"]
            self.B = B

    def matrix(self, I, s, y, it):
        """I: the information; (s, y): the last accepted step and the decrease of the score (None after a restart)."""
        ok = lambda M: np.all(np.isfinite(M)) and np.linalg.cond(M) < 1e12
        if not self.mode:
            return I, None
        if self.mode == "one":
            if s is None:
                return I, None
            B2, sec = secant_matrix(I, s, y)
            return (B2, sec) if sec is not None and ok(B2) else (I, None)
        if self.mode == "hybrid" and not self.on:
            if it >= 1 and (self.last_capped is False or self.capped_run >= 3):
                self.on, self.B = True, I
                return I, {"bfgs_start": it}
            return I, None
        if self.B is None or s is None or self.B.shape != I.shape:
            self.B = I
            return I, None
        B2, sec = secant_matrix(self.B, s, y)
        if sec is None or not ok(B2):
            return self.B, None
        self.B = B2
        return B2, sec

    def stepped(self, scale):
        capped = bool(scale < 1.0)
        self.capped_run = self.capped_run + 1 if capped else 0
        self.last_capped = capped

    def reset(self):
        self.B, self.on, self.capped_run, self.last_capped = None, False, 0, None

    def state(self):
        return {"on": self.on, "capped_run": self.capped_run, "last_capped": self.last_capped}


def describe(sec):
    if not sec:
        return ""
    if "bfgs_start" in sec:
        return ", BFGS starts"
    r = sec.get("curvature_ratio")
    return f", BFGS curvature ratio {r:.3g}" if r is not None else f", BFGS damping {sec['damping']:.3g}"


# ------------------------------------------------------------------ checkpoints
def fingerprint(leaves, names):
    h = hashlib.md5()
    for lf in leaves:
        h.update(lf.name.encode()); h.update(lf.det_fit.tobytes()); h.update(lf.L_fit.tobytes())
    h.update(json.dumps(list(names)).encode())
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def save_ckpt(prefix, state, arrays):
    """One file per checkpoint: the arrays plus the state (JSON bytes), written to a temporary name and renamed."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    state = dict(state, saved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    blob = np.frombuffer(json.dumps(tolist(state)).encode(), dtype=np.uint8)
    tmp = prefix + ".ckpt.tmp.npz"
    np.savez(tmp, ckpt_state=blob, **arrays)
    os.replace(tmp, prefix + ".ckpt.npz")


def read_ckpt_state(prefix):
    """The state of the checkpoint at prefix, or None (no file, or unreadable)."""
    try:
        with np.load(prefix + ".ckpt.npz") as z:
            return json.loads(z["ckpt_state"].tobytes().decode())
    except Exception:
        return None


def load_ckpt(prefix, fp):
    """The saved state and arrays if a complete checkpoint for these data exists, else None."""
    try:
        with np.load(prefix + ".ckpt.npz") as z:
            state = json.loads(z["ckpt_state"].tobytes().decode())
            if state.get("fingerprint") != fp:
                return None
            return state, {k: z[k] for k in z.files if k != "ckpt_state"}
    except Exception:
        return None


def remove_ckpt(prefix):
    for ext in (".ckpt.npz", ".ckpt.tmp.npz", ".json", ".npz", ".tmp.json", ".tmp.npz"):
        try:
            os.remove(prefix + ext)
        except OSError:
            pass


# ------------------------------------------------------------------ fit
def fit_series(leaves, max_iter=15, step_tol=0.02, max_step=1.0, log=print, info="opg", secant="bfgs", ckpt=None,
               stop=None):
    names = series_names(leaves)
    K = len(names)
    cpu0 = time.process_time(); cpu_prev = 0.0
    fp = fingerprint(leaves, names) + f"|info={info}|secant={secant}"
    got = load_ckpt(ckpt, fp) if ckpt else None
    prev = None
    if got:
        st, ar = got
        phi = np.array(st["phi"], float); fixed = np.array(st["fixed"], bool)
        it, n_eval, hist, events, cpu_prev = st["it"], st["n_eval"], st["hist"], st["events"], st["cpu_seconds"]
        ev = {"ll": st["ll"], "ll_sum": st["ll_sum"], "nonfinite": st["nonfinite"], "seconds": st["seconds"],
              "G": ar["G"], "w": ar["w"]}
        if st.get("prev"):
            prev = {"phi": np.array(st["prev"]["phi"]), "g": np.array(st["prev"]["g"]), "free": np.array(st["prev"]["free"])}
        sm = StepMatrix(secant, st.get("step_matrix"), ar.get("Bq"))
        log(f"  resumed from checkpoint {ckpt} (iteration {it}, {n_eval} evaluations, saved {st['saved']})")
    else:
        phi = np.zeros(K); fixed = np.zeros(K, bool)
        ev = objective(leaves, phi, K, log); n_eval = 1
        hist = [{"theta": np.exp(phi).tolist(), "ll": ev["ll"], "seconds": ev["seconds"]}]
        events, it = [], 0
        sm = StepMatrix(secant)

    def checkpoint():
        if ckpt:
            pv = None if prev is None else {k: v.tolist() for k, v in prev.items()}
            save_ckpt(ckpt, {"fingerprint": fp, "names": names, "phi": phi.tolist(), "fixed": fixed.tolist(), "it": it,
                             "n_eval": n_eval, "hist": hist, "events": events, "ll": ev["ll"], "ll_sum": ev["ll_sum"],
                             "nonfinite": ev["nonfinite"], "seconds": ev["seconds"], "prev": pv,
                             "step_matrix": sm.state(), "cpu_seconds": cpu_prev + time.process_time() - cpu0},
                      dict({"G": ev["G"], "w": ev["w"]}, **({} if sm.B is None else {"Bq": sm.B})))
    if not got:
        checkpoint()
    converged, stopped, last_check = False, False, None
    se = np.full(K, np.nan)
    while True:
        G = ev["G"]; n = G.shape[0]; g = G.mean(axis=0)
        free = np.where(~fixed)[0]
        if len(free) == 0:
            events.append({"iteration": it, "reason": "every class fixed"})
            break
        if info == "opg":
            Gc = G[:, free] - g[free]
        else:
            Gc = G[:, free] * ev["w"][:, None]
        Iinfo = (Gc.T @ Gc) / n
        sing = False
        sec = None
        try:
            cond = np.linalg.cond(Iinfo)
            if not np.isfinite(cond) or cond > 1e12:
                sing = True
            else:
                same = prev is not None and np.array_equal(prev["free"], free)
                s_, y_ = (phi[free] - prev["phi"][free], prev["g"][free] - g[free]) if same else (None, None)
                B, sec = sm.matrix(Iinfo, s_, y_, it)
                step_f = np.linalg.solve(B, g[free])
                se_f = np.sqrt(np.diag(np.linalg.inv(Iinfo)) / n)
                if not (np.all(np.isfinite(step_f)) and np.all(np.isfinite(se_f))):
                    sing = True
        except np.linalg.LinAlgError:
            sing = True
        if sing:
            if stop is not None and stop():
                stopped = True
                log("  stopped before the next evaluation (stop condition; checkpoint kept)")
                break
            c = free[int(np.argmin(np.diag(Iinfo)))]
            fixed[c] = True; phi[c] = 0.0
            events.append({"iteration": it, "fixed_at_1": names[c], "reason": "singular information"})
            log(f"  singular information: fixing {names[c]} at 1")
            ev = objective(leaves, phi, K, log); n_eval += 1
            prev = None; sm.reset()
            checkpoint()
            continue
        se[:] = np.nan; se[free] = se_f
        rel = np.abs(step_f) / se_f
        last_check = {"iteration": it, "max_step_in_se": float(rel.max()), "argmax": names[free[int(np.argmax(rel))]],
                      "secant": sec}
        log(f"  iter {it}: max |step|/se = {rel.max():.3g} ({names[free[int(np.argmax(rel))]]}){describe(sec)}")
        if np.all(rel < step_tol):
            converged = True
            break
        if it >= max_iter:
            break
        if stop is not None and stop():
            stopped = True
            log("  stopped before the next evaluation (stop condition; checkpoint kept)")
            break
        scale = min(1.0, max_step / float(np.max(np.abs(step_f))))
        lam = scale
        while True:
            phi_new = phi.copy(); phi_new[free] += lam * step_f
            ev_new = objective(leaves, phi_new, K, log); n_eval += 1
            if ev_new["ll"] >= ev["ll"] - 1e-12 or lam < 1e-3 * scale:
                break
            lam *= 0.5
        sm.stepped(scale)
        prev = {"phi": phi.copy(), "g": g.copy(), "free": free.copy()}
        phi, ev = phi_new, ev_new
        it += 1
        hist.append({"theta": np.exp(phi).tolist(), "ll": ev["ll"], "lambda": lam, "scale": scale,
                     "step_in_se": (lam * step_f / se_f).tolist(), "seconds": ev["seconds"], "secant": sec})
        log(f"  step {it}: theta = {np.array2string(np.exp(phi), precision=3)}  ll = {ev['ll']:.7f}  lambda = {lam:.3g}")
        checkpoint()
    G = ev["G"]
    free = np.where(~fixed)[0]
    se_opg = np.full(K, np.nan); se_exp = np.full(K, np.nan)
    try:
        se_opg[free] = np.sqrt(np.diag(np.linalg.inv(G[:, free].T @ G[:, free])))
        Gw = G[:, free] * ev["w"][:, None]
        se_exp[free] = np.sqrt(np.diag(np.linalg.inv(Gw.T @ Gw)))
    except np.linalg.LinAlgError:
        pass
    return {"names": names, "theta_hat": np.exp(phi).tolist(), "phi_hat": phi.tolist(), "se_opg_ln": se_opg.tolist(),
            "se_exp_ln": se_exp.tolist(), "information": info, "secant": secant,
            "fixed": [names[c] for c in np.where(fixed)[0]],
            "events": events, "converged": converged, "stopped": stopped, "iterations": it, "n_eval": n_eval,
            "history": hist, "last_convergence_check": last_check, "ll_per_shot": ev["ll"], "n_shots": int(G.shape[0]),
            "nonfinite": ev["nonfinite"], "mean_score_at_hat": G.mean(axis=0).tolist(),
            "cpu_seconds_fit_total": cpu_prev + time.process_time() - cpu0, "fingerprint": fp}


def tolist(x):
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return tolist(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tolist(obj), f, indent=1)
    os.replace(tmp, path)
