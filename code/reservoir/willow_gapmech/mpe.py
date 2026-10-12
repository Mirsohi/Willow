"""Exact most-probable-error (MPE) class per shot at d = 3: the layered contraction of a detector error model in the
max-product semiring, next to the same contraction in the sum-product semiring (exact ML). Item 3 of main's brief
(10 Oct 2026); EXPLORATORY, post hoc; new code, independent of code/decoder/ (nothing there is imported or edited).

reservoir, 2026-10-10. For a DEM of independent fault mechanisms j (probability p_j, detector set D_j, observable flag
L_j; mechanisms with identical (D_j, L_j) merged first, p = p1 + p2 - 2 p1 p2, as Tesseract and matching do) and a
syndrome s, per observable class l:
    ML:   Z_l(s) = sum over fault sets E with syndrome s and observable parity l of prod_j p_j^e_j (1 - p_j)^(1 - e_j)
    MPE:  M_l(s) = max over the same fault sets of the same product
ML decides argmax_l Z_l; an exact most-probable-error decoder decides the class of the single most probable fault set,
argmax_l M_l. "ML against exact MPE" isolates what summing over fault sets (degeneracy) buys under the model.

Contraction (the layered decoder of willow_check/p4_redecode.py, generalised): every fault of these DEMs touches the
detectors of one time layer t or of two adjacent layers t, t + 1 (checked). F_t(a, b, l) is the semiring sum over the
subsets of the faults whose lowest layer is t, indexed by their parity pattern a on layer t, b on layer t + 1 and the
observable parity l; it is built once per DEM by adding the faults one at a time. W(a, l) carries the earlier layers.
Closing layer t on the observed pattern s_t:
    W'(b, l) = (+)_{a, l'} W(a ^ s_t, l') (x) F_t(a, b, l ^ l'),
with (+, x) = (max, +) on log-weights here (numba kernel), and (+, x) = (+, *) with per-shot rescaling for ML
(p4_redecode.layered_ml, imported from reservoir's own willow_check). After the last layer M_l = W(empty pattern, l).
Exact for any DEM whose faults span at most two adjacent layers; the cost is 2^(n_t + n_(t+1)) per layer and shot
(n_t = detectors in layer t: 4 or 8 at d = 3; 12 or 24 at d = 5, so d = 5 is out of reach for this dense form).

Tesseract (the tesseract_decoder package, as decoder's runs use it) minimises cost(E) = sum_{j in E} ln((1 - p_j)/p_j),
i.e. maximises ln P(E) = C0 - cost(E) with C0 = sum_j ln(1 - p_j); so its solution is an exact MPE iff
cost_T = C0 - max(ln M_0, ln M_1) (checked per shot below, "cost gap").

Subcommands:
  test   OUT_DIR DEM                  brute-force checks on small DEMs (random, and fault subsets of a real d = 3 DEM):
                                      max- and sum-product against enumeration of every fault subset -> mpe_tests.json
  timing ARCHIVE DECODE OUT_DIR PLAN  a timing test (PLAN: a JSON file, see run_mpe_timing.sh): per d = 3 leaf, exact
                                      MPE and ML on the first N shots of the every-5th subset; ML llr against decoder's
                                      stored exact llr; Tesseract (decoder's settings, and wider settings) against the
                                      exact MPE (class and cost gap); per d = 5 leaf, Tesseract with decoder's settings
                                      and wider settings on a few shots. Every step timed (process CPU seconds); each
                                      wide run stops at its CPU budget. Decision counts are by-products of the timing
                                      sample, not results -> mpe_timing.json
Usage: nice -n 10 python3 -I mpe.py test OUT_DIR DEM | timing ARCHIVE DECODE OUT_DIR PLAN
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numba
import numpy as np
import stim

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_check"))
sys.path.insert(0, os.path.join(HERE, ".."))
from p4_check import mechanisms  # noqa: E402  (XOR of all targets; decomposition marks handled as stim does)
from p4_redecode import layered_ml  # noqa: E402  (exact sum-product, probabilities with per-layer rescaling)
from qrc.outpaths import out_path  # noqa: E402

ARCH = "google_105Q_surface_code_d3_d5_d7"
CM_RL = "correlated_matching_decoder_with_rl_optimized_prior"
TIE = 1e-9


def merge_mech(mech):
    """Merge mechanisms with identical (detectors, observable flag): independent XOR, p = p1 + p2 - 2 p1 p2."""
    acc, order = {}, []
    for p, ds, l in mech:
        k = (tuple(ds), int(l))
        if k in acc:
            q = acc[k]
            acc[k] = q + p - 2 * q * p
        else:
            acc[k] = float(p)
            order.append(k)
    return [(acc[k], k[0], k[1]) for k in order], len(mech) - len(order)


def layer_plan(mech, t_of, r):
    """Faults grouped by their lowest layer, with their bit masks on layers t and t + 1."""
    nd = len(t_of)
    layers = [np.flatnonzero(t_of == t) for t in range(r + 1)]
    pos = np.zeros(nd, np.int64)
    for t in range(r + 1):
        pos[layers[t]] = np.arange(layers[t].size)
    by_lo = [[] for _ in range(r + 1)]
    for p, ds, l in mech:
        ts = sorted({int(t_of[d]) for d in ds})
        if not ts or ts[-1] - ts[0] > 1:
            raise SystemExit(f"fault spans more than two layers or has no detector: {ds}")
        t = ts[0]
        ma = sum(1 << int(pos[d]) for d in ds if t_of[d] == t)
        mb = sum(1 << int(pos[d]) for d in ds if t_of[d] == t + 1)
        by_lo[t].append((float(p), ma, mb, int(l)))
    return layers, by_lo


def build_F_max(faults, na, nb):
    """ln of the max over fault subsets, by (pattern on layer t, pattern on layer t + 1, observable parity)."""
    F = np.full((1 << na, 1 << nb, 2), -np.inf)
    F[0, 0, 0] = 0.0
    ia, ib = np.arange(1 << na), np.arange(1 << nb)
    for p, ma, mb, l in faults:
        G = F[ia ^ ma][:, ib ^ mb]
        if l:
            G = G[:, :, ::-1]
        F = np.maximum(F + math.log1p(-p), G + math.log(p))
    return np.ascontiguousarray(F)


@numba.njit(cache=False)
def close_max(W, st, F):
    """out[s, b, l] = max_{a, l'} W[s, a ^ st[s], l'] + F[a, b, l ^ l'] (ln domain)."""
    S, A = W.shape[0], W.shape[1]
    B = F.shape[1]
    out = np.full((S, B, 2), -np.inf)
    for s in range(S):
        x = st[s]
        for a in range(A):
            w0 = W[s, a ^ x, 0]
            w1 = W[s, a ^ x, 1]
            if w0 == -np.inf and w1 == -np.inf:
                continue
            for b in range(B):
                f0 = F[a, b, 0]
                f1 = F[a, b, 1]
                v = w0 + f0
                u = w1 + f1
                if u > v:
                    v = u
                if v > out[s, b, 0]:
                    out[s, b, 0] = v
                v = w0 + f1
                u = w1 + f0
                if u > v:
                    v = u
                if v > out[s, b, 1]:
                    out[s, b, 1] = v
    return out


def prepare_max(mech, t_of, r):
    layers, by_lo = layer_plan(mech, t_of, r)
    Fs = []
    for t in range(r + 1):
        na = layers[t].size
        nb = layers[t + 1].size if t < r else 0
        Fs.append(build_F_max(by_lo[t], na, nb))
    return layers, Fs


def layered_mpe(prep, syn):
    """ln M_0, ln M_1 per shot (max over fault sets of each observable class)."""
    layers, Fs = prep
    S = syn.shape[0]
    W = np.full((S, 1 << layers[0].size, 2), -np.inf)
    W[:, 0, 0] = 0.0
    for t, F in enumerate(Fs):
        na = layers[t].size
        st = (syn[:, layers[t]].astype(np.int64) << np.arange(na)).sum(1) if na else np.zeros(S, np.int64)
        W = close_max(W, st, F)
    return W[:, 0, 0].copy(), W[:, 0, 1].copy()


def log_Z_from_ml(llr, logp):
    """ln Z_0, ln Z_1 from layered_ml's llr = ln Z_1 - ln Z_0 and logp = ln(Z_0 + Z_1) (handles llr = +-inf)."""
    return logp - np.logaddexp(0.0, llr), logp - np.logaddexp(0.0, -llr)


def ml_exact(mech, t_of, r, syn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return layered_ml(mech, t_of, r, syn)


# ------------------------------------------------------------------------------------------------ brute-force tests
def brute(mech):
    """Every fault subset: key = 2 * (syndrome as an int) + observable parity -> (max, logsumexp) of ln P."""
    syn = np.zeros(1, np.int64)
    L = np.zeros(1, np.int64)
    lp = np.zeros(1)
    for p, ds, l in mech:
        m = sum(1 << int(d) for d in ds)
        syn = np.concatenate([syn, syn ^ m])
        L = np.concatenate([L, L ^ l])
        lp = np.concatenate([lp + math.log1p(-p), lp + math.log(p)])
    key = syn * 2 + L
    o = np.argsort(key, kind="stable")
    key, lp = key[o], lp[o]
    u, start = np.unique(key, return_index=True)
    mx = np.maximum.reduceat(lp, start)
    ends = np.append(start[1:], lp.size)
    tot = np.array([np.logaddexp.reduce(lp[a:b]) for a, b in zip(start, ends)])
    return {int(k): (float(a), float(b)) for k, a, b in zip(u, mx, tot)}


def check_case(mech, t_of, r, nd, rng, max_syn=400):
    table = brute(mech)
    syns = sorted({k >> 1 for k in table})
    if len(syns) > max_syn:
        syns = sorted(rng.choice(syns, max_syn, replace=False).tolist())
    syn = np.array([[(s >> i) & 1 for i in range(nd)] for s in syns], np.uint8)
    m0, m1 = layered_mpe(prepare_max(mech, t_of, r), syn)
    llr, logp = ml_exact(mech, t_of, r, syn)
    z0, z1 = log_Z_from_ml(llr, logp)
    none = (-np.inf, -np.inf)
    err_m = err_z = 0.0
    dec_m = dec_z = n_dm = n_dz = differ = 0
    for i, s in enumerate(syns):
        for l, mv, zv in ((0, m0[i], z0[i]), (1, m1[i], z1[i])):
            ref = table.get(2 * s + l)
            if ref is None:
                err_m = max(err_m, 0.0 if mv == -np.inf else np.inf)
                err_z = max(err_z, 0.0 if zv == -np.inf else np.inf)
                continue
            err_m = max(err_m, abs(mv - ref[0]))
            err_z = max(err_z, abs(zv - ref[1]))
        b0, b1 = table.get(2 * s, none), table.get(2 * s + 1, none)
        if abs(b1[0] - b0[0]) > TIE:
            n_dm += 1
            dec_m += int((m1[i] > m0[i]) != (b1[0] > b0[0]))
        if abs(b1[1] - b0[1]) > TIE:
            n_dz += 1
            dec_z += int((llr[i] > 0) != (b1[1] > b0[1]))
        if abs(b1[0] - b0[0]) > TIE and abs(b1[1] - b0[1]) > TIE:
            differ += int((b1[0] > b0[0]) != (b1[1] > b0[1]))
    return {"faults": len(mech), "detectors": nd, "layers": r + 1, "syndromes_checked": len(syns),
            "max_abs_err_ln_max": err_m, "max_abs_err_ln_sum": err_z, "mpe_decisions_checked": n_dm,
            "mpe_decision_mismatches": dec_m, "ml_decisions_checked": n_dz, "ml_decision_mismatches": dec_z,
            "syndromes_where_brute_mpe_and_ml_classes_differ": differ}


def random_case(rng, n_layers, per_layer, n_faults):
    nd = n_layers * per_layer
    t_of = np.repeat(np.arange(n_layers), per_layer)
    perm = rng.permutation(nd)            # detector ids not sorted by layer, as in real DEMs
    t_of = t_of[np.argsort(perm)]
    mech = []
    for _ in range(n_faults):
        t = int(rng.integers(0, n_layers))
        span = [t] if t == n_layers - 1 or rng.random() < 0.4 else [t, t + 1]
        cand = [d for d in range(nd) if t_of[d] in span]
        k = int(rng.integers(1, min(4, len(cand)) + 1))
        ds = tuple(sorted(rng.choice(cand, k, replace=False).tolist()))
        mech.append((float(rng.uniform(0.01, 0.4)), ds, int(rng.random() < 0.35)))
    return mech, t_of, n_layers - 1, nd


def real_subcase(dem, rng, layers_keep=3, n_faults=18, radius=1.5):
    """A local piece of a real DEM: the faults inside layers 0 .. layers_keep - 1 whose detectors all lie within
    `radius` (in the x, y coordinates) of a random detector, with at least one observable-flipping fault; at most
    n_faults of them (observable-flipping ones kept first). Overlapping local faults give degenerate classes."""
    co = dem.get_detector_coordinates()
    t_all = np.array([int(round(co[i][2])) for i in range(dem.num_detectors)])
    xy = np.array([[co[i][0], co[i][1]] for i in range(dem.num_detectors)])
    keep = np.flatnonzero(t_all < layers_keep)
    allm = [(p, ds, l) for p, ds, l in mechanisms(dem) if ds and all(t_all[d] < layers_keep for d in ds)]
    for _ in range(1000):
        c = int(rng.choice(keep))
        sub = {int(d) for d in keep if math.hypot(*(xy[d] - xy[c])) <= radius}
        pool = [m for m in allm if set(m[1]) <= sub]
        if len(pool) >= 10 and any(m[2] for m in pool):
            break
    else:
        raise SystemExit("no local pool with an observable-flipping fault")
    lf = [i for i, m in enumerate(pool) if m[2]]
    of = [i for i, m in enumerate(pool) if not m[2]]
    rng.shuffle(lf)
    rng.shuffle(of)
    pick = sorted((lf[:3] + of)[:n_faults] if len(pool) > n_faults else range(len(pool)))
    dets = sorted({d for i in pick for d in pool[i][1]})
    new = {d: k for k, d in enumerate(dets)}
    mech = [(pool[i][0], tuple(new[d] for d in pool[i][1]), pool[i][2]) for i in pick]
    # probabilities raised x10 (capped at 0.45) so that many syndromes occur: structure kept, values not
    mech = [(min(0.45, 10 * p), ds, l) for p, ds, l in mech]
    t_sub = t_all[dets]
    lay = sorted(set(t_sub.tolist()))
    t_sub = np.array([lay.index(t) for t in t_sub])      # renumber the layers present as 0, 1, ...
    if any(b - a > 1 for a, b in zip(lay, lay[1:])):
        raise SystemExit("local pool skips a layer")
    return mech, t_sub, len(lay) - 1, len(dets), len(pool)


def cmd_test(out_dir, dem_path):
    t0, c0 = time.time(), time.process_time()
    rng = np.random.default_rng(20261010)
    res = {"label": "EXPLORATORY (reservoir, 2026-10-10): brute-force checks of mpe.py's max- and sum-product layered "
                    "contraction against enumeration of every fault subset", "random": [], "real_subsets": []}
    for n_layers, per_layer, n_faults in [(3, 2, 10), (3, 3, 14), (4, 2, 14), (4, 3, 16), (5, 2, 16), (3, 4, 18),
                                          (5, 3, 18), (6, 2, 18)]:
        for _ in range(3):
            mech, t_of, r, nd = random_case(rng, n_layers, per_layer, n_faults)
            res["random"].append(check_case(mech, t_of, r, nd, rng))
    dem = stim.DetectorErrorModel.from_file(dem_path)
    for _ in range(8):
        mech, t_of, r, nd, pool = real_subcase(dem, rng)
        o = check_case(mech, t_of, r, nd, rng)
        o["local_pool_faults"] = pool
        o["observable_flipping_faults"] = int(sum(m[2] for m in mech))
        res["real_subsets"].append(o)
    allc = res["random"] + res["real_subsets"]
    res["summary"] = {k: (max if k.startswith("max") else sum)(c[k] for c in allc)
                      for k in ["max_abs_err_ln_max", "max_abs_err_ln_sum", "syndromes_checked",
                                "mpe_decisions_checked", "mpe_decision_mismatches", "ml_decisions_checked",
                                "ml_decision_mismatches", "syndromes_where_brute_mpe_and_ml_classes_differ"]}
    res["summary"]["cases"] = len(allc)
    res["real_dem"] = dem_path
    res["cpu_s"], res["run_seconds"] = time.process_time() - c0, time.time() - t0
    with open(out_path(os.path.join(out_dir, "mpe_tests.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["summary"], indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ timing test
def load_leaf(arch, d, series, r, n, skip=0):
    patch, basis = series.split("/")
    rel = f"d{d}_at_{patch}/{basis}/r{r:02d}"
    a_dir = os.path.join(arch, rel)
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    from willow_check import read_obs_b8
    act, _ = read_obs_b8(os.path.join(a_dir, "obs_flips_actual.b8"))
    shots = np.arange(0, act.size, 5)[skip:skip + n]
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(act.size, -1)
    syn = np.unpackbits(raw[shots], axis=1, bitorder="little")[:, :nd]
    return rel, dem, t_of, act[shots], shots, syn


def stored(dec, rel, sub, fn, shots, key="pred"):
    p = os.path.join(dec, ARCH, rel, sub, fn)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    pos = {int(i): k for k, i in enumerate(z["idx"])}
    if any(int(s) not in pos for s in shots):
        return None
    v = z[key]
    v = v.reshape(z["idx"].size, -1)[:, 0] if key == "pred" else v.astype(np.float64)
    return v[[pos[int(s)] for s in shots]]


def run_tesseract(dem, syn, kw, budget_cpu):
    """Shot by shot, to get each solution's cost and stop at the CPU budget."""
    from tesseract_decoder import tesseract
    cfg = tesseract.TesseractConfig(dem=dem, **kw)
    tc = time.process_time()
    dd = cfg.compile_decoder()
    compile_cpu = time.process_time() - tc
    pred, cost, low, per = [], [], [], []
    t_start = time.process_time()
    for i in range(syn.shape[0]):
        t1 = time.process_time()
        errs = dd.decode_to_errors(syn[i].astype(bool))
        per.append(time.process_time() - t1)
        obs = dd.get_observables_from_errors(errs)
        pred.append(int(bool(obs[0])) if len(obs) else 0)
        cost.append(float(dd.cost_from_errors(errs)))
        low.append(bool(dd.low_confidence_flag))
        if time.process_time() - t_start > budget_cpu:
            break
    per = np.array(per)
    return {"pred": np.array(pred, np.uint8), "cost": np.array(cost), "low": np.array(low), "per_shot_cpu": per,
            "compile_cpu_s": compile_cpu}


def tess_summary(t, n_req):
    per = t["per_shot_cpu"]
    return {"shots_decoded": int(per.size), "shots_requested": n_req, "cpu_s": float(per.sum()),
            "compile_cpu_s": t["compile_cpu_s"], "ms_per_shot_mean": float(1e3 * per.mean()) if per.size else None,
            "ms_per_shot_median": float(1e3 * np.median(per)) if per.size else None,
            "ms_per_shot_max": float(1e3 * per.max()) if per.size else None,
            "low_confidence_shots": int(t["low"].sum())}


def tess_kw(T, tag):
    kw = dict(T[tag]["kw"])
    if kw.get("det_beam") == "INF":
        from tesseract_decoder import tesseract
        kw["det_beam"] = tesseract.INF_DET_BEAM
    return kw


def cmd_timing(arch, dec, out_dir, plan_path):
    plan = json.load(open(plan_path))
    t0, c0 = time.time(), time.process_time()
    res = {"label": "EXPLORATORY timing test (reservoir, 2026-10-10): exact MPE (max-product) and exact ML "
                    "(sum-product) at d = 3; Tesseract settings against the exact MPE at d = 3 and against each other "
                    "at d = 5. Decision counts are by-products of a small timing sample, not results",
           "plan": plan, "numba": numba.__version__, "d3": [], "d5": []}
    tc = time.process_time()
    close_max(np.zeros((1, 2, 2)), np.zeros(1, np.int64), np.zeros((2, 2, 2)))
    res["numba_compile_cpu_s"] = time.process_time() - tc
    T = plan["tesseract"]
    for leaf in plan["d3"]:
        rel, dem, t_of, obs, shots, syn = load_leaf(arch, 3, leaf["series"], leaf["r"], leaf["n"])
        mech0 = mechanisms(dem)
        mech, merged = merge_mech(mech0)
        r = leaf["r"]
        tc = time.process_time()
        prep = prepare_max(mech, t_of, r)
        cpu_prep = time.process_time() - tc
        tc = time.process_time()
        m0, m1 = layered_mpe(prep, syn)
        cpu_mpe = time.process_time() - tc
        tc = time.process_time()
        llr, logp = ml_exact(mech, t_of, r, syn)
        cpu_ml = time.process_time() - tc
        C0 = float(sum(math.log1p(-p) for p, _, _ in mech))
        cost_min = C0 - np.maximum(m0, m1)
        mpe = (m1 > m0).astype(np.uint8)
        ml = (llr > 0).astype(np.uint8)
        L = {"leaf": rel, "r": r, "n_det": int(dem.num_detectors), "mechanisms": len(mech0),
             "merged_duplicates": merged, "shots": int(shots.size), "shot_rule": "first N of the every-5th subset",
             "cpu_s_prepare_F_max": cpu_prep, "cpu_s_mpe": cpu_mpe, "cpu_s_ml": cpu_ml,
             "ms_per_shot_mpe": 1e3 * cpu_mpe / shots.size, "ms_per_shot_ml": 1e3 * cpu_ml / shots.size,
             "mpe_ties": int(np.sum(np.abs(m1 - m0) < TIE)),
             "errors": {"ml": int(np.sum(ml != obs)), "mpe": int(np.sum(mpe != obs))},
             "mpe_ne_ml": int(np.sum(mpe != ml)),
             "mpe_wrong_ml_right": int(np.sum((mpe != obs) & (ml == obs))),
             "mpe_right_ml_wrong": int(np.sum((mpe == obs) & (ml != obs)))}
        ref_llr = stored(dec, rel, "rl", "ml_exact.npz", shots, key="llr")
        if ref_llr is not None:
            L["ml_llr_max_abs_diff_vs_decoder_ml_exact"] = float(np.max(np.abs(llr - ref_llr)))
            L["ml_decisions_differ_from_decoder"] = int(np.sum(ml != (ref_llr > 0)))
        tst = stored(dec, rel, "rl_s5", "tesseract.npz", shots)
        if tst is not None:
            tst = tst.astype(np.uint8)
            L["stored_tesseract"] = {"ne_exact_mpe": int(np.sum(tst != mpe)), "ne_ml": int(np.sum(tst != ml)),
                                     "errors": int(np.sum(tst != obs))}
        for tag in leaf["tesseract"]:
            n_t = min(leaf.get(f"n_{tag}", shots.size), shots.size)
            t = run_tesseract(dem, syn[:n_t], tess_kw(T, tag), T[tag]["budget_cpu_s"])
            k = t["pred"].size
            gap = t["cost"] - cost_min[:k]
            o = tess_summary(t, n_t)
            o.update({"class_ne_exact_mpe": int(np.sum(t["pred"] != mpe[:k])),
                      "class_ne_ml": int(np.sum(t["pred"] != ml[:k])),
                      "cost_gap_vs_exact_min": {"max": float(gap.max()) if k else None,
                                                "min": float(gap.min()) if k else None,
                                                "shots_gap_gt_1e-6": int(np.sum(gap > 1e-6)),
                                                "shots_gap_lt_-1e-6": int(np.sum(gap < -1e-6))},
                      "class_ne_exact_mpe_where_gap_le_1e-6": int(np.sum((t["pred"] != mpe[:k]) & (gap <= 1e-6))),
                      "errors": int(np.sum(t["pred"] != obs[:k])),
                      "errors_ml_same_shots": int(np.sum(ml[:k] != obs[:k])),
                      "errors_mpe_same_shots": int(np.sum(mpe[:k] != obs[:k]))})
            if tst is not None:
                o["class_ne_stored_tesseract"] = int(np.sum(t["pred"] != tst[:k]))
            L[f"tesseract_{tag}"] = o
        res["d3"].append(L)
        print(json.dumps(L), flush=True)
    for leaf in plan["d5"]:
        rel, dem, t_of, obs, shots, syn = load_leaf(arch, 5, leaf["series"], leaf["r"], leaf["n"])
        mech0 = mechanisms(dem)
        _, merged = merge_mech(mech0)
        ml = stored(dec, rel, "rl_s5", "ml_tau2_1e-6-1e-8_0.1.npz", shots)
        tst = stored(dec, rel, "rl_s5t", "tesseract.npz", shots)
        L = {"leaf": rel, "r": leaf["r"], "n_det": int(dem.num_detectors), "mechanisms": len(mech0),
             "merged_duplicates": merged, "shots": int(shots.size), "shot_rule": "first N of the every-5th subset"}
        runs = {}
        for tag in leaf["tesseract"]:
            n_t = min(leaf.get(f"n_{tag}", shots.size), shots.size)
            t = run_tesseract(dem, syn[:n_t], tess_kw(T, tag), T[tag]["budget_cpu_s"])
            runs[tag] = t
            k = t["pred"].size
            o = tess_summary(t, n_t)
            o["errors"] = int(np.sum(t["pred"] != obs[:k]))
            if ml is not None:
                o["class_ne_ml_tau2"] = int(np.sum(t["pred"] != ml[:k].astype(np.uint8)))
                o["errors_ml_same_shots"] = int(np.sum(ml[:k].astype(np.uint8) != obs[:k]))
            if tst is not None:
                o["class_ne_stored_tesseract"] = int(np.sum(t["pred"] != tst[:k].astype(np.uint8)))
            L[f"tesseract_{tag}"] = o
        tags = list(runs)
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                k = min(runs[a]["pred"].size, runs[b]["pred"].size)
                dc = runs[a]["cost"][:k] - runs[b]["cost"][:k]
                L[f"{a}_vs_{b}"] = {"shots": k, "class_ne": int(np.sum(runs[a]["pred"][:k] != runs[b]["pred"][:k])),
                                    "cost_a_minus_b_max": float(dc.max()) if k else None,
                                    "cost_a_minus_b_min": float(dc.min()) if k else None,
                                    "shots_cost_a_gt_b_1e-6": int(np.sum(dc > 1e-6)),
                                    "shots_cost_a_lt_b_1e-6": int(np.sum(dc < -1e-6))}
        res["d5"].append(L)
        print(json.dumps(L), flush=True)
    res["cpu_s_total"] = time.process_time() - c0
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "mpe_timing.json")), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "test" and len(a) == 3:
        cmd_test(a[1], a[2])
    elif a and a[0] == "timing" and len(a) == 5:
        cmd_timing(a[1], a[2], a[3], a[4])
    else:
        raise SystemExit(__doc__)
