"""EXPLORATORY: the (7, 7) anomaly in the X-basis final layer (reservoir, 2026-10-09; brief from main after the P4
check). Not pre-registered; no multiplicity correction unless stated.

Background (f2_candidates.json, fit halves): shots with a single detection event, at the last-layer detector j of
measure qubit (7, 7), fail far more often than ML(RL) predicts in both patches that have that detector (q6_7 X 26
failures against 5.7 predicted, q8_9 X 14 against 0.23). The RL DEM's only one-fault explanation of such an event is
the single-detector fault b at j, which does not flip the observable. Hypothesis: a "twin", a fault with the same
footprint {j} and the opposite observable flag, is missing from the DEM.

Exact method. Every fault of these DEMs lies in one detector layer or two adjacent ones. The layered exact decoder of
willow_check/p4_redecode.py is run over layers 0..r-1 only. It leaves, per shot, the last-layer message
W(a, l) = P(observed detectors below layer r, last-layer pattern a from the faults that start below r, observable l).
Any change confined to the faults inside the last layer (remove b, flip its flag, add the twin) is then exact and
cheap: P(s, L) = sum_a sum_l' W(s_r XOR a, l') F_r(a, L XOR l'), F_r the joint distribution of the last-layer faults.
With A_x(L) = P(x, L) under the DEM without b, for x = s and x = s XOR j, the DEM with b at p0 (flag l) plus the twin
at q (flag 1 - l) gives exactly
  P(s, L) = (1 - p0)(1 - q) A_s(L) + p0 q A_s(L XOR 1) + p0 (1 - q) A_sj(L XOR l) + (1 - p0) q A_sj(L XOR 1 XOR l).
q = 0, p0 = p_b is the RL DEM; it reproduces the stored ML(RL) decodes on every shot processed (checked).

Per shot with an event at j ("j-shot"):
  hinge  the ML(RL) decision changes when b's observable flag is flipped: the decision hinges on how the event at j is
         explained;
  pi     the posterior probability under the RL DEM that b fired;
  N      the number of detection events in the shot.
The twin's probability q is the maximum of the exact conditional log-likelihood, the sum of ln P_q(L_obs | s) over
the j-shots of the fit halves (a series' 4 W5 leaves pooled), with b kept at p_b (the brief's model: add the twin);
95% profile interval (drop 1.92). Variant B (sensitivity): b at p_b - q, so that the total weight of the footprint {j}
stays at p_b. Shots without an event at j feel the twin only at second order (b and the twin together make an
undetectable logical flip of probability about p_b q); measured exactly on 2,000 random held-out shots per leaf.

Parts:
 1. Extension (fit halves): j-shots of the target series by N and hinge, ML(RL) failures against predicted
    (sum of min(P, 1 - P)). The same for every last-layer detector of the 18 series ("map", twin_map.csv); controls are
    the other detectors, in particular those of the neighbouring measure qubits (5,7), (7,5), (7,9), (9,7) (X type) and
    (6,6), (6,8), (8,6), (8,8) (Z type). Z basis: (7, 7) is X type and has no last-layer detector there. Correlated
    flips of the data qubits (7,8) and (6,9), present in both target patches, give exactly the twin at (7, 7) in the X
    basis; in the Z basis of q8_9 they give the unconnected pair (6,10)-(8,8) in the last layer (in q6_7 Z they cancel).
    That pair's excess (mech_f2's estimator, s2 from f2_candidates.json) is measured per leaf and block, in the last
    layer (q8_9 Z) and in the bulk (q8_9 X and Z, rounds pooled).
 2. Persistence (fit halves): per W5 leaf and per block (10 blocks of 5,000 shots in file order, one initialisation
    bitstring each): hinge failures against predicted, q per leaf and per leaf and block, a likelihood-ratio test of a
    q per leaf and block against one q, and the event rate at j against the DEM's marginal.
 3. Twin test: q fitted on the fit halves, then the held-out j-shots of the same leaves, which no earlier study of
    mine examined at (7, 7): ML(RL) against ML(RL + twin): errors, predicted failures, R, R_conf, conditional log-loss
    (paired), decision changes. The same procedure at the other last-layer detectors of the two patches as controls.
    summary: the predicted failures the twin adds on the held-out halves, as a share of the P3 excess over all 72
    held-out halves (willow_check/p4_comparison.json).
The fit half is the even rank within each initialisation bitstring (the complement of P4's held-out test_idx, checked
against each fit_data.npz).
Usage: nice -n 19 python3 -I mech_twin.py ARCHIVE DECODE FIT OUT_DIR
  -> OUT_DIR/twin.json, twin_map.csv, twin_shots.npz (through qrc.outpaths)
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import sys
import time

import numpy as np
import stim
from scipy import optimize, stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_check"))
sys.path.insert(0, os.path.join(HERE, ".."))  # code/reservoir (qrc.outpaths)
from p4_check import BASES, CM_RL, PATCHES, SHOTS, W5, block_codes, mechanisms, read_b8_bit0  # noqa: E402
from qrc.outpaths import in_path, out_path  # noqa: E402

LABEL = "EXPLORATORY, not pre-registered (reservoir, 2026-10-09): the (7, 7) final-layer anomaly and a twin fault"
TARGET = (7.0, 7.0)
TARGET_SERIES = [("q6_7", "X"), ("q8_9", "X")]
NEIGHBOURS = {(5.0, 7.0), (7.0, 5.0), (7.0, 9.0), (9.0, 7.0), (6.0, 6.0), (6.0, 8.0), (8.0, 6.0), (8.0, 8.0)}
PAIR = ((6.0, 10.0), (8.0, 8.0))
PAIR_SERIES = [("q8_9", "Z"), ("q8_9", "X")]
CONF = 3e-3
NSAMPLE = 2000
SEED = 20261009
BATCH = 4000
BLOCK = 5000
QMAX = 0.3
N_BINS = [("1", 1, 1), ("2-3", 2, 3), ("4-9", 4, 9), (">=10", 10, 10 ** 9)]


def poisson_upper(k, lam):
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    return float(stats.poisson.sf(k - 1, lam))


# ---------------------------------------------------------------- exact layered decoder up to the last layer
def setup(dem, r):
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    layers = [np.flatnonzero(t_of == t) for t in range(r + 1)]
    pos = np.zeros(nd, np.int64)
    for t in range(r + 1):
        pos[layers[t]] = np.arange(layers[t].size)
    stab = [tuple(co[i][-3:-1]) if t_of[i] == r else tuple(co[i][:2]) for i in range(nd)]
    return t_of, layers, pos, stab


def joint(faults, na, nb):
    """Joint parity distribution of independent faults (p, bits in this layer, bits in the next, flips L) over
    (pattern a, next-layer pattern b, observable), as an array (2^na, 2^nb, 2). Bit k of a pattern is axis na-1-k of
    the (2,)*na view, so a fault is a flip of some axes."""
    F = np.zeros((2,) * (na + nb + 1))
    F[(0,) * (na + nb + 1)] = 1.0
    for p, ba, bb, l in faults:
        axes = tuple([na - 1 - k for k in ba] + [na + nb - 1 - k for k in bb] + ([na + nb] if l else []))
        F = (1 - p) * F + p * np.flip(F, axis=axes)
    return F.reshape(1 << na, 1 << nb, 2)


def build(mech, t_of, layers, pos, r):
    by_lo = [[] for _ in range(r + 1)]
    for k, (p, ds, l) in enumerate(mech):
        ts = sorted({int(t_of[d]) for d in ds})
        if not ts or ts[-1] - ts[0] > 1:
            raise SystemExit(f"fault spans more than two layers or has no detector: {ds}")
        t = ts[0]
        by_lo[t].append((p, [int(pos[d]) for d in ds if t_of[d] == t], [int(pos[d]) for d in ds if t_of[d] == t + 1],
                         l, k))
    Fs = [joint([f[:4] for f in by_lo[t]], layers[t].size, layers[t + 1].size) for t in range(r)]
    return Fs, by_lo


def forward(Fs, layers, r, syn):
    """Last-layer message W (S, 2^n_r, 2), normalised per shot, and its log scale."""
    S = syn.shape[0]
    W = np.zeros((S, 1 << layers[0].size, 2))
    W[:, 0, 0] = 1.0
    logscale = np.zeros(S)
    ar = np.arange(S)[:, None]
    for t in range(r):
        F = Fs[t]
        na, nb = layers[t].size, F.shape[1]
        s_t = (syn[:, layers[t]].astype(np.int64) << np.arange(na)).sum(1)
        Wt = W[ar, np.arange(1 << na)[None, :] ^ s_t[:, None], :]
        M = np.block([[F[:, :, 0], F[:, :, 1]], [F[:, :, 1], F[:, :, 0]]])
        Y = np.concatenate([Wt[:, :, 0], Wt[:, :, 1]], axis=1) @ M
        Wn = np.stack([Y[:, :nb], Y[:, nb:]], axis=2)
        mx = Wn.reshape(S, -1).max(1)
        if np.any(mx <= 0):
            raise SystemExit("a shot has zero probability under the DEM")
        W = Wn / mx[:, None, None]
        logscale += np.log(mx)
    return W, logscale


def close_last(W, s_r, Fr):
    """P(s, L) (scaled like W) for last-layer patterns s_r and a last-layer distribution Fr (2^n_r, 2)."""
    S = W.shape[0]
    Wt = W[np.arange(S)[:, None], np.arange(W.shape[1])[None, :] ^ s_r[:, None], :]
    P0 = Wt[:, :, 0] @ Fr[:, 0] + Wt[:, :, 1] @ Fr[:, 1]
    P1 = Wt[:, :, 0] @ Fr[:, 1] + Wt[:, :, 1] @ Fr[:, 0]
    return np.stack([P0, P1], 1)


def model(As, Asj, p0, q, l):
    """P(s, L), L = 0, 1, with b at p0 (flag l) and the twin at q (flag 1 - l); As, Asj (S, 2) without b."""
    P = np.empty_like(As)
    for L in (0, 1):
        P[:, L] = ((1 - p0) * (1 - q) * As[:, L] + p0 * q * As[:, L ^ 1] + p0 * (1 - q) * Asj[:, L ^ l]
                   + (1 - p0) * q * Asj[:, L ^ 1 ^ l])
    return P


def loglik(q, As, Asj, pb, l, obs, variant="A"):
    p0 = pb if variant == "A" else pb - q
    P = model(As, Asj, p0, q, l)
    return float(np.sum(np.log(P[np.arange(len(obs)), obs]) - np.log(P.sum(1))))


def fit_q(As, Asj, pb, l, obs, variant="A"):
    """MLE of q >= 0 with its 95% profile interval and the likelihood-ratio statistic against q = 0."""
    if len(obs) == 0:
        return {"q": None}
    hi = QMAX if variant == "A" else float(np.min(pb))
    f = lambda q: -loglik(q, As, Asj, pb, l, obs, variant)  # noqa: E731
    res = optimize.minimize_scalar(f, bounds=(0.0, hi), method="bounded", options={"xatol": 1e-8})
    qh, lmax, l0 = float(res.x), -float(res.fun), -f(0.0)
    if l0 >= lmax:
        qh, lmax = 0.0, l0
    cut = lmax - 1.92
    lo = 0.0 if -f(0.0) >= cut else optimize.brentq(lambda q: -f(q) - cut, 0.0, qh)
    up = hi if -f(hi) >= cut else optimize.brentq(lambda q: -f(q) - cut, qh, hi)
    lr = max(0.0, 2 * (lmax - l0))
    return {"q": qh, "ci95": [lo, up], "lr_vs_0": lr, "p_lr": 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 0.5,
            "loglik_max": lmax, "loglik_0": l0, "shots": int(len(obs))}


# ---------------------------------------------------------------- one leaf
def leaf(arch, dec, fit, patch, basis, r, is_target, rng):
    rel = os.path.join(f"d3_at_{patch}", basis, f"r{r:02d}")
    a_dir = os.path.join(arch, rel)
    dem = stim.DetectorErrorModel.from_file(os.path.join(a_dir, "decoding_results", CM_RL, "error_model.dem"))
    nd = dem.num_detectors
    t_of, layers, pos, stab = setup(dem, r)
    mech = mechanisms(dem)
    t0 = time.time()
    Fs, by_lo = build(mech, t_of, layers, pos, r)
    # split: even rank within each initialisation bitstring = fit half; odd = held out (P4's test_idx)
    code = block_codes(os.path.join(a_dir, "sweep_bits.b8"), SHOTS)
    rank = np.zeros(SHOTS, np.int64)
    for c in np.unique(code):
        p_ = np.flatnonzero(code == c)
        rank[p_] = np.arange(p_.size)
    half = (rank % 2).astype(np.int8)
    test_idx = np.load(os.path.join(fit, rel, "rl_tsplit", "fit_data.npz"))["test_idx"].astype(np.int64)
    if not np.array_equal(np.sort(test_idx), np.flatnonzero(half == 1)):
        raise SystemExit(f"held-out rule differs from test_idx at {rel}")
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(SHOTS, -1)
    E = np.unpackbits(raw, axis=1, bitorder="little")[:, :nd]
    obs = read_b8_bit0(os.path.join(a_dir, "obs_flips_actual.b8"), SHOTS).astype(np.int64)
    ml = np.load(os.path.join(dec, rel, "rl", "ml_exact.npz"), allow_pickle=False)
    N = E.sum(1)
    J = layers[r]
    last_any = E[:, J].any(1)
    proc = (half == 0) & last_any
    jt = None
    if is_target:
        jt = int([j for j in J if stab[j] == TARGET][0])
        proc |= (half == 1) & last_any
        nonj = np.flatnonzero((half == 1) & (E[:, jt] == 0))
        samp = np.sort(rng.choice(nonj, size=min(NSAMPLE, nonj.size), replace=False))
    else:
        samp = np.zeros(0, np.int64)
    shots = np.union1d(np.flatnonzero(proc), samp)
    # forward pass, in batches
    Ws, Ls = [], []
    for a in range(0, shots.size, BATCH):
        W, ls = forward(Fs, layers, r, E[shots[a:a + BATCH]])
        Ws.append(W)
        Ls.append(ls)
    W = np.concatenate(Ws) if Ws else np.zeros((0, 1 << J.size, 2))
    logscale = np.concatenate(Ls) if Ls else np.zeros(0)
    s_r = (E[shots][:, J].astype(np.int64) << np.arange(J.size)).sum(1)
    last_faults = by_lo[r]
    Fr_full = joint([f[:4] for f in last_faults], J.size, 0)[:, 0, :]
    Pfull = close_last(W, s_r, Fr_full)
    llr_chk = np.log(Pfull[:, 1]) - np.log(Pfull[:, 0])
    logp_chk = np.log(Pfull.sum(1)) + logscale
    check = {"max_abs_diff_llr": float(np.max(np.abs(llr_chk - ml["llr"][shots]))) if shots.size else 0.0,
             "max_abs_diff_logp": float(np.max(np.abs(logp_chk - ml["logp"][shots]))) if shots.size else 0.0,
             "pred_equal": int(np.sum((llr_chk > 0) == (ml["pred"][shots, 0] == 1))), "shots": int(shots.size)}
    out = {"rows": {}, "check": check, "seconds": 0.0, "nd": nd}
    blk = (np.arange(SHOTS) // BLOCK).astype(np.int8)
    for jj, j in enumerate(J):
        bs = [f for f in last_faults if f[1] == [int(pos[j])] and not f[2]]
        if len(bs) > 1:
            raise SystemExit(f"two single-detector faults at {rel} {stab[j]}")
        pb, lb = (bs[0][0], int(bs[0][3])) if bs else (0.0, 0)
        Fr_minus = joint([f[:4] for f in last_faults if not (bs and f[4] == bs[0][4])], J.size, 0)[:, 0, :]
        mask = 1 << int(pos[j])
        sel = np.flatnonzero(E[shots, j] == 1)
        if j == jt:
            sel = np.union1d(sel, np.flatnonzero(np.isin(shots, samp)))
        sh = shots[sel]
        As = close_last(W[sel], s_r[sel], Fr_minus)
        Asj = close_last(W[sel], s_r[sel] ^ mask, Fr_minus)
        P0 = model(As, Asj, pb, 0.0, lb)
        if sel.size and np.max(np.abs(P0 / Pfull[sel] - 1)) > 1e-9:
            raise SystemExit(f"b-removal identity fails at {rel} {stab[j]}")
        # DEM marginal of detector j: (1 - prod over faults touching j of (1 - 2p)) / 2
        prod = 1.0
        for p, ds, l in mech:
            if j in ds:
                prod *= 1 - 2 * p
        out["rows"][stab[j]] = {
            "j": int(j), "pb": pb, "lb": lb, "has_b": bool(bs), "shot": sh, "half": half[sh], "block": blk[sh],
            "N": N[sh], "obs": obs[sh], "As": As, "Asj": Asj, "sample": np.isin(sh, samp) & (E[sh, j] == 0),
            "rate_data_fit": float(E[half == 0, j].mean()), "rate_dem": (1 - prod) / 2,
            "rate_data_fit_by_block": [float(E[(half == 0) & (blk == b), j].mean()) for b in range(SHOTS // BLOCK)]}
    out["seconds"] = time.time() - t0
    out["E_fit"] = E[half == 0] if (patch, basis) in PAIR_SERIES else None
    out["blk_fit"] = blk[half == 0]
    out["stab"], out["t_of"] = stab, t_of
    return out


# ---------------------------------------------------------------- summaries
def derived(x):
    """Per-shot RL quantities for one detector's rows."""
    P0 = model(x["As"], x["Asj"], x["pb_r"], 0.0, x["lb"])
    Pf = model(x["As"], x["Asj"], x["pb_r"], 0.0, 1 - x["lb"])
    tot = P0.sum(1)
    m0 = P0.min(1) / tot
    d0 = (P0[:, 1] > P0[:, 0]).astype(np.int64)
    hinge = d0 != (Pf[:, 1] > Pf[:, 0])
    pi = x["pb_r"] * x["Asj"].sum(1) / tot
    return m0, d0, hinge, pi


def cal(fail, m, sel):
    k, pr = int(fail[sel].sum()), float(m[sel].sum())
    return {"shots": int(sel.sum()), "failures": k, "predicted": pr, "R": k / pr if pr > 0 else None,
            "p_poisson_upper": poisson_upper(k, pr)}


def main(arch, dec, fit, out_dir):
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    f2 = json.load(open(in_path(os.path.join(out_dir, "f2_candidates.json"))))
    data = {}  # (patch, basis) -> {stab: concatenated rows}, plus checks
    checks, pair_raw = [], {}
    for patch in PATCHES:
        for basis in BASES:
            is_t = (patch, basis) in TARGET_SERIES
            acc = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in W5:
                L = leaf(arch, dec, fit, patch, basis, r, is_t, rng)
                checks.append({"leaf": f"{patch} {basis} r{r}", **L["check"], "seconds": L["seconds"]})
                print(f"{patch} {basis} r={r}: {L['check']['shots']} shots in {L['seconds']:.0f} s, max |dllr| "
                      f"{L['check']['max_abs_diff_llr']:.1e}, max |dlogp| {L['check']['max_abs_diff_logp']:.1e}",
                      flush=True)
                for st, x in L["rows"].items():
                    a = acc[st]
                    for k in ("shot", "half", "block", "N", "obs", "As", "Asj", "sample"):
                        a[k].append(x[k])
                    a["r"].append(np.full(len(x["shot"]), r, np.int16))
                    a["pb_by_r"].append(x["pb"])
                    a["lb"].append(x["lb"])
                    a["rate_data_fit"].append(x["rate_data_fit"])
                    a["rate_dem"].append(x["rate_dem"])
                    a["rate_by_block"].append(x["rate_data_fit_by_block"])
                if (patch, basis) in PAIR_SERIES:
                    pair_raw[(patch, basis, r)] = (L["E_fit"], L["blk_fit"], L["stab"], L["t_of"])
            ser = {}
            for st, a in acc.items():
                if len(set(a["lb"])) != 1:
                    raise SystemExit(f"{patch} {basis} {st}: b's observable flag differs across r")
                x = {k: np.concatenate(a[k]) for k in ("shot", "half", "block", "N", "obs", "As", "Asj", "sample", "r")}
                x["pb_r"] = np.concatenate([np.full(len(s), p) for s, p in zip(a["shot"], a["pb_by_r"])])
                x["pb"], x["lb"] = float(np.mean(a["pb_by_r"])), int(a["lb"][0])
                x["rate_data_fit"], x["rate_dem"] = a["rate_data_fit"], a["rate_dem"]
                x["rate_by_block"] = a["rate_by_block"]
                ser[st] = x
            data[(patch, basis)] = ser
    res = {"label": LABEL, "definitions": __doc__.split("Usage:")[0].strip(), "decoder_checks": {
        "leaves": len(checks), "max_abs_diff_llr_vs_stored_float32": max(c["max_abs_diff_llr"] for c in checks),
        "max_abs_diff_logp_vs_stored": max(c["max_abs_diff_logp"] for c in checks),
        "predictions_equal": f"{sum(c['pred_equal'] for c in checks)} of {sum(c['shots'] for c in checks)}",
        "shots_forward_passed": sum(c["shots"] for c in checks), "seconds": sum(c["seconds"] for c in checks)}}

    # ---- map: every last-layer detector of the 18 series, fit halves
    rows = []
    fits = {}
    for (patch, basis), ser in data.items():
        for st, x in sorted(ser.items()):
            fsel = (x["half"] == 0) & ~x["sample"]
            m0, d0, hinge, pi = derived(x)
            fail = d0 != x["obs"]
            y = {k: v[fsel] for k, v in x.items() if isinstance(v, np.ndarray) and len(v) == len(fsel)}
            fq = fit_q(y["As"], y["Asj"], y["pb_r"], x["lb"], y["obs"])
            fits[(patch, basis, st)] = fq
            n1 = fsel & (x["N"] == 1)
            hs = fsel & hinge
            row = {"patch": patch, "basis": basis, "stab_x": st[0], "stab_y": st[1],
                   "neighbour_of_7_7": st in NEIGHBOURS, "target": st == TARGET and basis == "X",
                   "b_prob": x["pb"], "b_flips_L": x["lb"], "rate_data_fit": float(np.mean(x["rate_data_fit"])),
                   "rate_dem": float(np.mean(x["rate_dem"])), "j_shots": int(fsel.sum()),
                   "N1_shots": int(n1.sum()), "N1_failures": int(fail[n1].sum()), "N1_predicted": float(m0[n1].sum()),
                   "hinge_shots": int(hs.sum()), "hinge_failures": int(fail[hs].sum()),
                   "hinge_predicted": float(m0[hs].sum()),
                   "hinge_N2plus_shots": int((hs & (x["N"] >= 2)).sum()),
                   "hinge_N2plus_failures": int(fail[hs & (x["N"] >= 2)].sum()),
                   "hinge_N2plus_predicted": float(m0[hs & (x["N"] >= 2)].sum()),
                   "q_hat": fq["q"], "q_lo": fq["ci95"][0], "q_hi": fq["ci95"][1], "lr_vs_0": fq["lr_vs_0"],
                   "p_lr": fq["p_lr"]}
            row["hinge_R"] = row["hinge_failures"] / row["hinge_predicted"] if row["hinge_predicted"] > 0 else None
            row["hinge_p_poisson_upper"] = poisson_upper(row["hinge_failures"], row["hinge_predicted"])
            rows.append(row)
    ps = np.array([x["p_lr"] for x in rows])
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    run = 0.0
    for rank_, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank_) * ps[i]))
        holm[i] = run
    for x, h in zip(rows, holm):
        x["p_lr_holm72"] = float(h)
    rows.sort(key=lambda x: -x["lr_vs_0"])
    with open(out_path(os.path.join(out_dir, "twin_map.csv")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res["map"] = {"cells": len(rows), "note": ("one row per series and last-layer stabilizer, fit halves; q_hat is the "
                                               "twin's MLE (b kept), LR its likelihood-ratio statistic against q = 0, "
                                               "p_lr = 0.5 P(chi2_1 > LR); Holm over the 72 cells"),
                  "top10_by_lr": [{k: x[k] for k in ("patch", "basis", "stab_x", "stab_y", "q_hat", "q_lo", "q_hi",
                                                     "lr_vs_0", "p_lr_holm72", "hinge_failures", "hinge_predicted",
                                                     "N1_failures", "N1_predicted")} for x in rows[:10]],
                  "cells_with_holm_p_below_0.05": [f"{x['patch']} {x['basis']} ({x['stab_x']:g},{x['stab_y']:g})"
                                                   for x in rows if x["p_lr_holm72"] < 0.05],
                  "neighbours_of_7_7": [{k: x[k] for k in ("patch", "basis", "stab_x", "stab_y", "q_hat", "q_lo",
                                                           "q_hi", "lr_vs_0", "hinge_failures", "hinge_predicted")}
                                        for x in rows if x["neighbour_of_7_7"]],
                  "all_other_cells_hinge": {
                      "failures": sum(x["hinge_failures"] for x in rows if not x["target"]),
                      "predicted": sum(x["hinge_predicted"] for x in rows if not x["target"])}}
    oth = res["map"]["all_other_cells_hinge"]
    oth["R"] = oth["failures"] / oth["predicted"]

    # ---- target series: extension, persistence, twin test
    res["target"] = {}
    shots_out = {}
    for patch, basis in TARGET_SERIES:
        x = data[(patch, basis)][TARGET]
        m0, d0, hinge, pi = derived(x)
        fail = d0 != x["obs"]
        fsel = (x["half"] == 0) & ~x["sample"]
        hsel = (x["half"] == 1) & ~x["sample"]
        T = {"b_prob": x["pb"], "b_flips_L": x["lb"], "twin_flips_L": 1 - x["lb"],
             "rate_data_fit_by_r": dict(zip(map(str, W5), x["rate_data_fit"])),
             "rate_dem_by_r": dict(zip(map(str, W5), x["rate_dem"]))}
        # 1. extension (fit halves)
        ext = {}
        for name, lo, hi in N_BINS:
            nb_ = fsel & (x["N"] >= lo) & (x["N"] <= hi)
            ext[f"N {name}"] = {"hinge": cal(fail, m0, nb_ & hinge), "not_hinge": cal(fail, m0, nb_ & ~hinge)}
        ext["hinge N>=2"] = cal(fail, m0, fsel & hinge & (x["N"] >= 2))
        ext["hinge N>=2 by pi"] = {"pi<0.5": cal(fail, m0, fsel & hinge & (x["N"] >= 2) & (pi < 0.5)),
                                  "pi>=0.5": cal(fail, m0, fsel & hinge & (x["N"] >= 2) & (pi >= 0.5))}
        ext["all j-shots"] = cal(fail, m0, fsel)
        ext["hinge share of j-shots"] = float(hinge[fsel].mean())
        T["extension_fit_halves"] = ext
        # 2. persistence (fit halves)
        per = {"by_r": {}, "by_block_pooled_over_r": {}}
        for r in W5:
            s_ = fsel & (x["r"] == r)
            fq = fit_q(x["As"][s_], x["Asj"][s_], x["pb_r"][s_], x["lb"], x["obs"][s_])
            per["by_r"][str(r)] = {"hinge": cal(fail, m0, s_ & hinge), "q": fq["q"], "q_ci95": fq.get("ci95"),
                                  "lr_vs_0": fq.get("lr_vs_0")}
        for b in range(SHOTS // BLOCK):
            s_ = fsel & (x["block"] == b)
            per["by_block_pooled_over_r"][str(b)] = {"hinge": cal(fail, m0, s_ & hinge),
                                                    "j_shots": int(s_.sum())}
        lmax_cells, cells = 0.0, 0
        qcell = np.full((len(W5), SHOTS // BLOCK), np.nan)
        lrcell = np.full((len(W5), SHOTS // BLOCK), np.nan)
        for ir, r in enumerate(W5):
            for b in range(SHOTS // BLOCK):
                s_ = fsel & (x["r"] == r) & (x["block"] == b)
                if s_.sum():
                    fc = fit_q(x["As"][s_], x["Asj"][s_], x["pb_r"][s_], x["lb"], x["obs"][s_])
                    lmax_cells += fc["loglik_max"]
                    qcell[ir, b], lrcell[ir, b] = fc["q"], fc["lr_vs_0"]
                    cells += 1
        per["q_by_leaf_and_block"] = {"rows_r": W5, "cols_block": list(range(SHOTS // BLOCK)),
                                      "q": qcell.tolist(), "lr_vs_0": lrcell.tolist()}
        lr_r = 0.0
        for r in W5:
            s_ = fsel & (x["r"] == r)
            lr_r += fit_q(x["As"][s_], x["Asj"][s_], x["pb_r"][s_], x["lb"], x["obs"][s_])["loglik_max"]
        fq_all = fit_q(x["As"][fsel], x["Asj"][fsel], x["pb_r"][fsel], x["lb"], x["obs"][fsel])
        lr_cells = 2 * (lmax_cells - fq_all["loglik_max"])
        lr_rr = 2 * (lr_r - fq_all["loglik_max"])
        per["heterogeneity"] = {
            "q_per_leaf_vs_one_q": {"lr": lr_rr, "df": 3, "p": float(stats.chi2.sf(lr_rr, 3))},
            "q_per_leaf_and_block_vs_one_q": {"lr": lr_cells, "df": cells - 1, "p": float(stats.chi2.sf(lr_cells,
                                                                                                    cells - 1))},
            "note": "chi2 reference approximate (q >= 0 boundary; small cells)"}
        rb = np.array(x["rate_by_block"])  # (4 leaves, 10 blocks)
        per["event_rate_at_j_fit_halves"] = {
            "by_r_data": dict(zip(map(str, W5), x["rate_data_fit"])), "by_r_dem": dict(zip(map(str, W5),
                                                                                        x["rate_dem"])),
            "by_block_mean_over_r": rb.mean(0).tolist(), "block_sd_over_mean": float(rb.mean(0).std() / rb.mean())}
        T["persistence_fit_halves"] = per
        # 3. twin fit and held-out test
        fA = fq_all
        fB = fit_q(x["As"][fsel], x["Asj"][fsel], x["pb_r"][fsel], x["lb"], x["obs"][fsel], variant="B")
        T["twin_fit_fit_halves"] = {"A_add_twin": fA, "B_split_b": fB, "q_over_b_A": fA["q"] / x["pb"]}
        PT = model(x["As"], x["Asj"], x["pb_r"], fA["q"], x["lb"])
        totT = PT.sum(1)
        m1 = PT.min(1) / totT
        d1 = (PT[:, 1] > PT[:, 0]).astype(np.int64)
        fail1 = d1 != x["obs"]
        P0 = model(x["As"], x["Asj"], x["pb_r"], 0.0, x["lb"])
        ll0 = -(np.log(P0[np.arange(len(x["obs"])), x["obs"]]) - np.log(P0.sum(1)))
        ll1 = -(np.log(PT[np.arange(len(x["obs"])), x["obs"]]) - np.log(totT))
        PB = model(x["As"], x["Asj"], x["pb_r"] - fB["q"], fB["q"], x["lb"])
        llB = -(np.log(PB[np.arange(len(x["obs"])), x["obs"]]) - np.log(PB.sum(1)))
        failB = (PB[:, 1] > PB[:, 0]).astype(np.int64) != x["obs"]
        mB = PB.min(1) / PB.sum(1)

        def test(sel):
            dl = ll1[sel] - ll0[sel]
            ch = sel & (d1 != d0)
            c0, c1 = sel & (m0 < CONF), sel & (m1 < CONF)
            return {"shots": int(sel.sum()), "errors_RL": int(fail[sel].sum()), "errors_twin": int(fail1[sel].sum()),
                    "predicted_RL": float(m0[sel].sum()), "predicted_twin": float(m1[sel].sum()),
                    "R_RL": float(fail[sel].sum() / m0[sel].sum()) if m0[sel].sum() > 0 else None,
                    "R_twin": float(fail1[sel].sum() / m1[sel].sum()) if m1[sel].sum() > 0 else None,
                    "p_poisson_upper_RL": poisson_upper(int(fail[sel].sum()), float(m0[sel].sum())),
                    "p_poisson_upper_twin": poisson_upper(int(fail1[sel].sum()), float(m1[sel].sum())),
                    "conf_RL": [int(fail[c0].sum()), float(m0[c0].sum())],
                    "conf_twin": [int(fail1[c1].sum()), float(m1[c1].sum())],
                    "logloss_RL": float(ll0[sel].sum()), "logloss_twin": float(ll1[sel].sum()),
                    "logloss_diff_twin_minus_RL": float(dl.sum()),
                    "logloss_diff_se": float(math.sqrt(dl.size) * dl.std(ddof=1)) if dl.size > 1 else None,
                    "logloss_diff_B_minus_RL": float((llB[sel] - ll0[sel]).sum()),
                    "errors_B": int(failB[sel].sum()), "predicted_B": float(mB[sel].sum()),
                    "decisions_changed": int(ch.sum()), "changed_twin_right": int((ch & ~fail1).sum())}

        T["heldout_test"] = {
            "q_used": fA["q"],
            "all j-shots": test(hsel), "N=1": test(hsel & (x["N"] == 1)),
            "hinge N>=2": test(hsel & hinge & (x["N"] >= 2)), "not hinge": test(hsel & ~hinge),
            "by_r": {str(r): test(hsel & (x["r"] == r)) for r in W5}}
        smp = x["sample"]
        dpost = np.abs(PT[smp, 1] / totT[smp] - P0[smp, 1] / P0[smp].sum(1))
        T["second_order_sample_heldout_no_event_at_j"] = {
            "shots": int(smp.sum()), "max_abs_change_posterior": float(dpost.max()) if smp.any() else None,
            "mean_change_predicted_failure_per_shot": float((m1[smp] - m0[smp]).mean()) if smp.any() else None,
            "decisions_changed": int((d1[smp] != d0[smp]).sum()),
            "errors_RL": int(fail[smp].sum()), "errors_twin": int(fail1[smp].sum()),
            "B_mean_change_predicted_failure_per_shot": float((mB[smp] - m0[smp]).mean()) if smp.any() else None,
            "B_decisions_changed": int((failB[smp] != fail[smp]).sum()),
            "note": "random held-out shots without an event at j, 2,000 per leaf: the twin's effect there"}
        # controls: the same fit and test at the other last-layer detectors of this patch
        ctl = {}
        for st, y in data[(patch, basis)].items():
            if st == TARGET:
                continue
            fs_ = (y["half"] == 0) & ~y["sample"]
            hs_ = (y["half"] == 1) & ~y["sample"]
            fq = fit_q(y["As"][fs_], y["Asj"][fs_], y["pb_r"][fs_], y["lb"], y["obs"][fs_])
            Pa = model(y["As"], y["Asj"], y["pb_r"], 0.0, y["lb"])
            Pb_ = model(y["As"], y["Asj"], y["pb_r"], fq["q"], y["lb"])
            ia_ = np.arange(len(y["obs"]))
            dl = (-(np.log(Pb_[ia_, y["obs"]]) - np.log(Pb_.sum(1))) + (np.log(Pa[ia_, y["obs"]])
                                                                         - np.log(Pa.sum(1))))[hs_]
            e0 = ((Pa[:, 1] > Pa[:, 0]) != y["obs"])[hs_].sum()
            e1 = ((Pb_[:, 1] > Pb_[:, 0]) != y["obs"])[hs_].sum()
            ctl[f"({st[0]:g},{st[1]:g})"] = {"q_fit": fq["q"], "q_ci95": fq["ci95"], "lr_vs_0_fit": fq["lr_vs_0"],
                                            "heldout_shots": int(hs_.sum()), "heldout_errors_RL": int(e0),
                                            "heldout_errors_twin": int(e1),
                                            "heldout_logloss_diff": float(dl.sum()),
                                            "heldout_logloss_diff_se": float(math.sqrt(dl.size) * dl.std(ddof=1))
                                            if dl.size > 1 else None}
        T["controls_same_patch"] = ctl
        res["target"][f"{patch} {basis}"] = T
        shots_out[f"{patch}_{basis}"] = {k: x[k] for k in ("shot", "r", "half", "block", "N", "obs", "As", "Asj",
                                                         "sample", "pb_r")}
        shots_out[f"{patch}_{basis}"]["lb"] = np.array([x["lb"]])
    # both target patches together (one physical location)
    tot = {"fit": [0.0, 0.0], "test": collections.Counter()}
    res["target_pooled_heldout"] = {}
    for k in ("errors_RL", "errors_twin", "predicted_RL", "predicted_twin", "logloss_diff_twin_minus_RL",
              "decisions_changed", "changed_twin_right", "shots"):
        res["target_pooled_heldout"][k] = sum(res["target"][s]["heldout_test"]["all j-shots"][k]
                                              for s in res["target"])
    se2 = sum(res["target"][s]["heldout_test"]["all j-shots"]["logloss_diff_se"] ** 2 for s in res["target"])
    res["target_pooled_heldout"]["logloss_diff_se"] = math.sqrt(se2)
    # share of the P3 excess on the held-out halves (72 leaves) that the twin accounts for (in predicted failures)
    p4c = json.load(open(in_path(os.path.join(out_dir, "..", "willow_check", "p4_comparison.json"))))
    rl = p4c["heldout_calibration_recomputed"]["mine"]["RL"]
    excess = rl["errors"] - rl["predicted"]
    d_j = res["target_pooled_heldout"]["predicted_twin"] - res["target_pooled_heldout"]["predicted_RL"]
    d_nonj = {v: 0.0 for v in ("A", "B")}
    for s_, T in res["target"].items():
        nj = T["heldout_test"]["all j-shots"]["shots"]
        so = T["second_order_sample_heldout_no_event_at_j"]
        d_nonj["A"] += so["mean_change_predicted_failure_per_shot"] * (len(W5) * SHOTS // 2 - nj)
        d_nonj["B"] += so["B_mean_change_predicted_failure_per_shot"] * (len(W5) * SHOTS // 2 - nj)
    rows_t = [x for x in rows if x["target"]]
    rows_o = [x for x in rows if not x["target"]]
    res["summary"] = {
        "target_cells_lr": {f"{x['patch']} {x['basis']}": x["lr_vs_0"] for x in rows_t},
        "largest_lr_elsewhere": max(x["lr_vs_0"] for x in rows_o),
        "largest_q_over_b_elsewhere": max(x["q_hat"] / x["b_prob"] for x in rows_o if x["b_prob"] > 0),
        "target_q_over_b": {f"{x['patch']} {x['basis']}": x["q_hat"] / x["b_prob"] for x in rows_t},
        "cells_holm_below_0.05": len(res["map"]["cells_with_holm_p_below_0.05"]),
        "p3_heldout_excess_72_leaves": {"errors": rl["errors"], "predicted": rl["predicted"], "excess": excess},
        "twin_added_predicted_failures_heldout": {"j_shots": d_j, "other_shots_A_extrapolated": d_nonj["A"],
                                                  "other_shots_B_extrapolated": d_nonj["B"]},
        "share_of_p3_excess": {"A": (d_j + d_nonj["A"]) / excess,
                               "j_shots_only": d_j / excess},
        "note": ("predicted failures the twin adds on the held-out halves of the 8 leaves, as a share of the P3 "
                 "excess (ML(RL) errors minus predicted) over all 72 held-out halves; the shots without an event "
                 "at j are extrapolated from the 2,000-shot samples (A)")}

    # ---- Z basis and bulk: the pair (6,10)-(8,8) per leaf and block (fit halves), mech_f2's estimator
    pr = {}
    for (patch, basis, r), (Ef, bf, stab, t_of) in sorted(pair_raw.items()):
        s2 = f2["series"][f"{patch} {basis}"]["s2_by_r"][str(r)]
        ia = [i for i in range(len(stab)) if stab[i] == PAIR[0]]
        ib = [i for i in range(len(stab)) if stab[i] == PAIR[1]]
        prs = [(a, b) for a in ia for b in ib if t_of[a] == t_of[b]]
        out = {}
        for region in ("last", "bulk"):
            sel_p = [(a, b) for a, b in prs if (t_of[a] == r) == (region == "last") and 0 < t_of[a] <= r]
            if not sel_p:
                continue
            o = {}
            for bname, rows_b in [("all", np.ones(len(bf), bool))] + [(str(b), bf == b) for b in range(SHOTS // BLOCK)]:
                X = Ef[rows_b].astype(np.float64)
                n = X.shape[0]
                qs, ses = [], []
                for a, b in sel_p:
                    ma, mb = X[:, a].mean(), X[:, b].mean()
                    cov = (X[:, a] * X[:, b]).mean() - ma * mb
                    den = (1 - 2 * ma) * (1 - 2 * mb)
                    qs.append((cov - s2 * ma * mb) / den)
                    ses.append(math.sqrt(ma * (1 - ma) * mb * (1 - mb) / n) / den)
                o[bname] = [float(np.mean(qs)), float(math.sqrt(np.sum(np.square(ses))) / len(ses))]
            qb = np.array([o[str(b)][0] for b in range(SHOTS // BLOCK)])
            sb = np.array([o[str(b)][1] for b in range(SHOTS // BLOCK)])
            chi2 = float(np.sum(((qb - o["all"][0]) / sb) ** 2))
            out[region] = {"q_se_all": o["all"], "q_se_by_block": {b: o[str(b)] for b in range(SHOTS // BLOCK)},
                           "pairs_pooled": len(sel_p), "chi2_blocks": chi2, "df": 9,
                           "p_blocks": float(stats.chi2.sf(chi2, 9))}
        pr[f"{patch} {basis} r{r}"] = out
    res["pair_6_10__8_8"] = {
        "note": ("excess q of the unconnected pair (6,10)-(8,8) (same round), fit halves, per leaf and block; the "
                 "Z-type footprint of correlated flips of data qubits (7,8) and (6,9). Last layer: q8_9 Z only "
                 "(in the X basis these are not last-layer detectors). Bulk: rounds pooled, per round."),
        "leaves": pr}
    res["run_seconds"] = time.time() - t_start
    with open(out_path(os.path.join(out_dir, "twin.json")), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    flat = {}
    for s, d in shots_out.items():
        for k, v in d.items():
            flat[f"{s}__{k}"] = v
    with open(out_path(os.path.join(out_dir, "twin_shots.npz")), "wb") as f:
        np.savez_compressed(f, **flat)
    print(json.dumps({k: res[k] for k in ("decoder_checks", "target_pooled_heldout", "summary")}, indent=1,
                     default=str))
    for s, T in res["target"].items():
        print(s, json.dumps({k: T[k] for k in ("twin_fit_fit_halves", "heldout_test")}, indent=1, default=str)[:6000])
    print(json.dumps(res["map"]["top10_by_lr"], indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:5])
