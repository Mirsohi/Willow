"""Willow-like simulated surface-code memory circuits (SI1000 noise) for decoder benchmarks.

SI1000(p) noise model (Gidney, Newman, Fowler, Broughton 2021, "A fault-tolerant honeycomb memory"):
  - 2-qubit Clifford: DEPOLARIZE2(p) after
  - 1-qubit Clifford: DEPOLARIZE1(p/10) after
  - reset: X_ERROR(2p) after
  - measurement: X_ERROR(5p) before
  - idle qubits in a gate layer: DEPOLARIZE1(p/10)
  - idle qubits in a measure/reset layer ("resonator idle"): DEPOLARIZE1(2p)
Applied here to Stim's generated rotated-surface-code memory circuit (CX-based) as a proxy for
Google's CZ-based circuit. Real-data runs will use Google's own circuits/DEMs.
"""
import stim

ONE_Q = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "X", "Y", "Z", "I", "H_YZ", "H_XY"}
TWO_Q = {"CX", "CNOT", "CZ", "CY", "SWAP", "ISWAP", "ISWAP_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG", "XCX", "XCZ", "YCX", "YCZ", "ZCX", "ZCY", "ZCZ"}
MEAS = {"M", "MZ", "MX", "MY"}
RESET = {"R", "RZ", "RX", "RY"}
MR = {"MR", "MRZ", "MRX", "MRY"}
# basis-appropriate flip channel for measurement/reset errors
FLIP = {"M": "X_ERROR", "MZ": "X_ERROR", "R": "X_ERROR", "RZ": "X_ERROR", "MR": "X_ERROR", "MRZ": "X_ERROR",
        "MX": "Z_ERROR", "RX": "Z_ERROR", "MRX": "Z_ERROR", "MY": "X_ERROR", "RY": "X_ERROR", "MRY": "X_ERROR"}
ANNOT = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}


def _flatten(circuit: stim.Circuit) -> stim.Circuit:
    return circuit.flattened()


def si1000(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Return a copy of a noiseless circuit with SI1000(p) noise added, moment by moment (TICK-delimited)."""
    circuit = _flatten(circuit)
    all_qubits = set(range(circuit.num_qubits))
    out = stim.Circuit()
    moment = []

    def flush(moment):
        if not moment:
            return
        touched = set()
        has_mr = False
        for inst in moment:
            name = inst.name
            if name in ANNOT:
                continue
            qs = [t.value for t in inst.targets_copy()]
            touched.update(qs)
            if name in MEAS or name in RESET or name in MR:
                has_mr = True
        # emit
        for inst in moment:
            name = inst.name
            if name in ANNOT:
                out.append(inst)
                continue
            qs = [t.value for t in inst.targets_copy()]
            if name in MEAS:
                out.append(FLIP[name], qs, 5 * p)
                out.append(inst)
            elif name in RESET:
                out.append(inst)
                out.append(FLIP[name], qs, 2 * p)
            elif name in MR:
                out.append(FLIP[name], qs, 5 * p)
                out.append(inst)
                out.append(FLIP[name], qs, 2 * p)
            elif name in TWO_Q:
                out.append(inst)
                out.append("DEPOLARIZE2", qs, p)
            elif name in ONE_Q:
                out.append(inst)
                out.append("DEPOLARIZE1", qs, p / 10)
            else:
                raise ValueError(f"unhandled instruction {name}")
        idle = sorted(all_qubits - touched)
        if idle and any(inst.name not in ANNOT for inst in moment):
            out.append("DEPOLARIZE1", idle, (2 * p) if has_mr else (p / 10))

    for inst in circuit:
        if inst.name == "TICK":
            flush(moment)
            moment = []
            out.append("TICK")
        else:
            moment.append(inst)
    flush(moment)
    return out


def memory_circuit(d: int, rounds: int, basis: str = "Z", p: float = 1e-3) -> stim.Circuit:
    """Rotated surface-code memory experiment with SI1000(p) noise. basis in {'X','Z'}."""
    code = f"surface_code:rotated_memory_{basis.lower()}"
    ideal = stim.Circuit.generated(code, distance=d, rounds=rounds)
    return si1000(ideal, p)


if __name__ == "__main__":
    import sys
    import numpy as np
    for p in [1e-3, 1.5e-3, 2e-3, 2.5e-3]:
        for d in [3, 5, 7]:
            c = memory_circuit(d, 20, "Z", p)
            s = c.compile_detector_sampler(seed=1)
            det = s.sample(20000)
            dem = c.detector_error_model(decompose_errors=False)
            print(f"p={p:.4f} d={d} pdet(mean over all detectors)={det.mean():.4f} n_det={c.num_detectors} n_err={dem.num_errors}")
            sys.stdout.flush()
