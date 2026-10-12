"""Synthetic ground truth for validating the decoder-vs-model decomposition (methods validation).

A "true" noise model T generates the shots; the decoder uses a model M that differs from T in a controlled
way. Tools here:
  si_family(ideal, p2, pm, p1, crosstalk=0)  generalised SI1000 noise (3 rates) + optional correlated crosstalk
  leak_variant(noisy, qubit, rounds)          leakage-like burst: the data qubit is fully depolarised at the start
                                              of each listed round (DEPOLARIZE1(0.75) = random Pauli)
  graphlike_dm(dem_decomposed)                independent-edge ("matching") approximation of a DEM: each '^'
                                              component becomes its own independent fault
  dm_to_stim(dm)                              DemMatrices -> stim.DetectorErrorModel (for sampling from M)
  calibration(...)                            predicted vs observed logical error, reliability bins,
                                              excess logical cross-entropy, all with z-scores under H0: M = T
SI1000(p) == si_family(p2=p, pm=5p, p1=p/10): 2q gates DEPOLARIZE2(p2); 1q gates and gate-layer idles
DEPOLARIZE1(p1); resonator idle (idle during measure/reset layers) DEPOLARIZE1(20 p1); measurement flip pm;
reset flip 0.4 pm.
"""
import numpy as np
import stim
from willow_sim import ONE_Q, TWO_Q, MEAS, RESET, MR, FLIP, ANNOT
from dem_utils import DemMatrices, parse_dem


def si_family(circuit: stim.Circuit, p2: float, pm: float, p1: float, crosstalk: float = 0.0,
              crosstalk_pairs=None) -> stim.Circuit:
    """Generalised SI1000 noise on a noiseless circuit (moment by moment). If crosstalk > 0, after the last
    two-qubit-gate moment of every round, correlated X_aX_b and Z_aZ_b errors (each with probability
    `crosstalk`) are applied to every pair in crosstalk_pairs (default: horizontally adjacent data qubits)."""
    circuit = circuit.flattened()
    all_qubits = set(range(circuit.num_qubits))
    if crosstalk > 0 and crosstalk_pairs is None:
        crosstalk_pairs = horizontal_data_pairs(circuit)
    out = stim.Circuit()
    moments, cur = [], []
    for inst in circuit:
        if inst.name == "TICK":
            moments.append(cur); cur = []
        else:
            cur.append(inst)
    moments.append(cur)
    kinds = []
    for mom in moments:
        names = {i.name for i in mom if i.name not in ANNOT}
        kinds.append("2q" if names & TWO_Q else ("mr" if names & (MEAS | RESET | MR) else ("1q" if names else "none")))
    for k, mom in enumerate(moments):
        touched, has_mr = set(), False
        for inst in mom:
            if inst.name in ANNOT:
                continue
            touched.update(t.value for t in inst.targets_copy())
            if inst.name in MEAS or inst.name in RESET or inst.name in MR:
                has_mr = True
        for inst in mom:
            name = inst.name
            if name in ANNOT:
                out.append(inst); continue
            qs = [t.value for t in inst.targets_copy()]
            if name in MEAS:
                out.append(FLIP[name], qs, pm); out.append(inst)
            elif name in RESET:
                out.append(inst); out.append(FLIP[name], qs, 0.4 * pm)
            elif name in MR:
                out.append(FLIP[name], qs, pm); out.append(inst); out.append(FLIP[name], qs, 0.4 * pm)
            elif name in TWO_Q:
                out.append(inst); out.append("DEPOLARIZE2", qs, p2)
            elif name in ONE_Q:
                out.append(inst); out.append("DEPOLARIZE1", qs, p1)
            else:
                raise ValueError(f"unhandled instruction {name}")
        idle = sorted(all_qubits - touched)
        if idle and any(inst.name not in ANNOT for inst in mom):
            out.append("DEPOLARIZE1", idle, (20 * p1) if has_mr else p1)
        # crosstalk after the last 2q moment of a round (next non-empty moment is not 2q)
        if crosstalk > 0 and kinds[k] == "2q":
            nxt = next((kinds[j] for j in range(k + 1, len(kinds)) if kinds[j] != "none"), None)
            if nxt != "2q":
                for a, b in crosstalk_pairs:
                    out.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], crosstalk)
                    out.append("CORRELATED_ERROR", [stim.target_z(a), stim.target_z(b)], crosstalk)
        if k < len(moments) - 1:
            out.append("TICK")
    return out


def data_qubits(circuit: stim.Circuit):
    """Qubits measured by the final (non-reset) measurement layer."""
    last = None
    for inst in circuit.flattened():
        if inst.name in ("M", "MX", "MZ"):
            last = inst
    return [t.value for t in last.targets_copy()]


def horizontal_data_pairs(circuit: stim.Circuit):
    co = circuit.get_final_qubit_coordinates()
    dq = data_qubits(circuit)
    pairs = []
    for a in dq:
        for b in dq:
            if a < b and abs(co[a][1] - co[b][1]) < 1e-9 and abs(abs(co[a][0] - co[b][0]) - 2) < 1e-9:
                pairs.append((a, b))
    return pairs


def leak_variant(noisy: stim.Circuit, qubit: int, rounds, strength: float = 0.75) -> stim.Circuit:
    """Insert DEPOLARIZE1(strength) on `qubit` at the start of each round in `rounds` (round t starts after the
    t-th measure-reset layer; round 0 after the initial reset layer)."""
    rounds = set(rounds)
    out = stim.Circuit()
    pending, nxt_round, n_mr, started = False, 0, 0, False
    for inst in noisy.flattened():
        out.append(inst)
        if inst.name == "TICK":
            if pending:
                if nxt_round in rounds:
                    out.append("DEPOLARIZE1", [qubit], strength)
                pending = False
        elif inst.name in MR:
            n_mr += 1; pending = True; nxt_round = n_mr
        elif inst.name in RESET and not started:
            started = True; pending = True; nxt_round = 0
    return out


def graphlike_dm(dem_decomposed: stim.DetectorErrorModel) -> DemMatrices:
    """Independent-edge approximation: each '^'-separated component of a decomposed DEM error becomes an
    independent fault with the error's probability; identical symptoms are then merged."""
    lines = []
    for inst in dem_decomposed.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comp = []
        for t in inst.targets_copy():
            if t.is_separator():
                lines.append((p, comp)); comp = []
            else:
                comp.append(t)
        lines.append((p, comp))
    d = stim.DetectorErrorModel()
    for p, comp in lines:
        if comp:
            d.append("error", p, comp)
    # carry detector coordinates over
    for inst in dem_decomposed.flattened():
        if inst.type == "detector":
            d.append(inst)
    return parse_dem(d)


def dm_to_stim(dm: DemMatrices) -> stim.DetectorErrorModel:
    d = stim.DetectorErrorModel()
    for j, ds in enumerate(dm.dets):
        tg = [stim.target_relative_detector_id(x) for x in ds]
        o = int(dm.obs[j]); k = 0
        while o:
            if o & 1:
                tg.append(stim.target_logical_observable_id(k))
            o >>= 1; k += 1
        d.append("error", float(dm.p[j]), tg)
    for om, pr in dm.undetectable:
        tg = []; k = 0; o = int(om)
        while o:
            if o & 1:
                tg.append(stim.target_logical_observable_id(k))
            o >>= 1; k += 1
        d.append("error", float(pr), tg)
    # make sure the detector/observable counts match
    for i in range(dm.num_detectors):
        d.append("detector", list(dm.coords.get(i, ())), [stim.target_relative_detector_id(i)])
    for k in range(dm.num_observables):
        d.append("logical_observable", [], [stim.target_logical_observable_id(k)])
    return d


RELIABILITY_EDGES = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.15, 0.3, 0.5000001])


def calibration(post1: np.ndarray, obs: np.ndarray):
    """Calibration of model-M posteriors P_M(L=1|s) against the observed logical outcomes.
    Under H0 (M = truth): E[#errors of ML(M)] = sum_s m_s with m_s = min(P, 1-P) and Var = sum m_s (1-m_s);
    E[-ln P_M(L_obs|s)] = sum_s H2(P_s), Var = sum P0 P1 ln(P1/P0)^2.
    Returns predicted/observed LER with z, ratio, excess cross-entropy (nats/shot) with z, reliability bins."""
    P = np.clip(np.nan_to_num(post1, nan=0.5), 1e-300, 1 - 1e-16)
    o = obs.astype(bool)
    pred = P > 0.5
    m = np.minimum(P, 1 - P)
    n = len(P)
    err = (pred != o)
    E_pred, E_obs = m.mean(), err.mean()
    sd = np.sqrt(np.sum(m * (1 - m))) / n
    P0, P1 = 1 - P, P
    nll = -np.log(np.where(o, P1, P0))
    H = -(P0 * np.log(P0) + P1 * np.log(P1))
    ce_sd = np.sqrt(np.sum(P0 * P1 * np.log(P1 / P0) ** 2)) / n
    bins = []
    for lo, hi in zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:]):
        sel = (m >= lo) & (m < hi)
        k = int(sel.sum())
        if k == 0:
            continue
        mm = float(m[sel].mean()); fo = float(err[sel].mean())
        bins.append({"lo": float(lo), "hi": float(min(hi, 0.5)), "n": k, "mean_predicted": mm, "observed": fo,
                     "observed_errors": int(err[sel].sum()), "expected_errors": float(m[sel].sum()),
                     "sd_expected_errors": float(np.sqrt(np.sum(m[sel] * (1 - m[sel]))))})
    return {"n": int(n), "LER_predicted": float(E_pred), "LER_observed": float(E_obs),
            "ratio_obs_over_pred": float(E_obs / E_pred) if E_pred > 0 else float("nan"),
            "z_LER": float((E_obs - E_pred) / sd) if sd > 0 else float("nan"),
            "xent_observed": float(nll.mean()), "xent_expected": float(H.mean()),
            "excess_xent": float(nll.mean() - H.mean()), "z_xent": float((nll.mean() - H.mean()) / ce_sd) if ce_sd > 0 else float("nan"),
            "reliability": bins}


def paired(pred_a, pred_b, obs):
    ea = pred_a != obs; eb = pred_b != obs
    return {"a_wrong_b_right": int(np.sum(ea & ~eb)), "a_right_b_wrong": int(np.sum(~ea & eb))}
