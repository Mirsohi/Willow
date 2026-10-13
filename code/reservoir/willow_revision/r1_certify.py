"""R1 of the revision (review 1.4): a stronger exact-MPE certificate of decoder's stored Tesseract at d = 5, above all at
r = 30; and the integer-programming resolution of R2's disagreements. POST HOC revision analysis, not a pre-registered
test; declared in notes/reservoir/willow_revision_declarations.md before its results were read.

reservoir, 2026-10-13. The method is stage 2's (code/reservoir/willow_gapmech/exact_run.py, imported unchanged: the
RL DEM's mechanisms merged as in mpe.py, min sum_j w_j e_j subject to H e = s (mod 2) with w_j = ln((1 - p_j)/p_j), HiGHS
with relative MIP gap 0 and a 60-s wall-clock limit per program, the solution checked against the syndrome; when the
optimum's class differs from the stored Tesseract class, a second program restricted to the stored class decides between
a tie (within COST_TOL = 1e-6) and a miss; a bound that settles neither counts as a miss).

Sample (plan): per d = 5 leaf (8 per round count) the candidates are the stored Tesseract shots (every 5th, 10,000)
less stage 2's sampled shots; a random order of them is drawn with numpy's default_rng(20261013), in the order r = 30,
13, 10, patches q4_7, q6_5, q6_9, q8_7, bases X, Z. Phase 1 takes the first k_r of each leaf's order (k_30 = 133,
k_13 = 240, k_10 = 336), so that with stage 2's 5, 50 and 50 per leaf the totals reach n_r = 1,104, 2,320 and 3,088:
the smallest equal per-leaf allocations at which a zero-miss one-sided 95% Clopper-Pearson bound u = 1 - 0.05^(1/n)
is at most half of the observed per-shot MPE - ML difference D_r (stage 2's leverage, rule A). Phase 2 (r = 30 only,
run only if the revision's compute cap leaves room, decided without looking at any R1 result) takes the next 138 per
leaf (totals 276 per leaf, n = 2,208: u <= D_30 / 4). Shots are run interleaved across the leaves of a round count
(the i-th shot of every leaf before the (i+1)-th), so a run stopped by its CPU guard leaves a balanced sample.

Subcommands (python3 -I -B; the archive is downloaded data):
  plan DECODE STAGE2_JSON PLAN_OUT        the shot lists (deterministic; see above)
  run ARCHIVE DECODE PLAN PHASE RECORDS GUARD_S
                                          PHASE 1 or 2; appends one JSON line per shot to RECORDS (the checkpoint: shots
                                          already there are skipped); stops before a shot once this process's CPU time
                                          passes GUARD_S. Reads no outcome.
  resolve ARCHIVE DECODE LIST RECORDS GUARD_S
                                          LIST: JSON list of {"leaf", "idx", "classes": {name: class}}; per shot the
                                          proven optimum and, for every listed class that differs from it, the
                                          class-restricted optimum (tie or miss for that name), in the listed order.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_gapmech"))
from exact_run import (BASES, COST_TOL, P5, dem_of, load_syn, mechanisms, merge_mech, milp_setup, rel_of,  # noqa: E402
                       solve, stored_pred)
from mpe import ARCH  # noqa: E402

SEED = 20261013
TIME_LIMIT = 60
PHASE1 = {30: 133, 13: 240, 10: 336}
PHASE2 = {30: 138}


def u_zero(n):
    return 1.0 - 0.05 ** (1.0 / n)


def cmd_plan(dec, stage2_json, plan_out):
    s2 = json.load(open(stage2_json))["records"]
    taken = {}
    for x in s2:
        taken.setdefault(x["leaf"], set()).add(int(x["idx"]))
    rng = np.random.default_rng(SEED)
    leaves = []
    for r in (30, 13, 10):
        for p in P5:
            for b in BASES:
                rel = rel_of(5, p, b, r)
                z = np.load(os.path.join(dec, ARCH, rel, "rl_s5t", "tesseract.npz"), allow_pickle=False)
                cand = np.sort(z["idx"].astype(np.int64))
                every5 = bool(np.array_equal(cand, np.arange(0, 50000, 5)))
                cand = np.array([c for c in cand if int(c) not in taken.get(rel, set())], np.int64)
                order = rng.permutation(cand)
                k1, k2 = PHASE1[r], PHASE2.get(r, 0)
                leaves.append({"leaf": rel, "r": r, "candidates_every_5th": every5,
                               "stage2_shots": len(taken.get(rel, ())), "candidates_left": int(cand.size),
                               "phase1": [int(s) for s in order[:k1]], "phase2": [int(s) for s in order[k1:k1 + k2]]})
    tot = {r: {"stage2": sum(x["stage2_shots"] for x in leaves if x["r"] == r),
               "phase1": sum(len(x["phase1"]) for x in leaves if x["r"] == r),
               "phase2": sum(len(x["phase2"]) for x in leaves if x["r"] == r)} for r in (30, 13, 10)}
    for r, t in tot.items():
        t["n_after_phase1"] = t["stage2"] + t["phase1"]
        t["u_zero_after_phase1"] = u_zero(t["n_after_phase1"])
        if t["phase2"]:
            t["n_after_phase2"] = t["n_after_phase1"] + t["phase2"]
            t["u_zero_after_phase2"] = u_zero(t["n_after_phase2"])
    plan = {"note": "R1 plan (reservoir, 2026-10-13): extension of stage 2's sample, drawn without replacement from the "
                    "stored Tesseract shots not in stage 2; default_rng(20261013), in the listed order",
            "seed": SEED, "time_limit_s": TIME_LIMIT, "cost_tol": COST_TOL, "totals": tot, "leaves": leaves}
    with open(plan_out, "w") as f:
        json.dump(plan, f, indent=1)
    print(json.dumps(tot, indent=1), flush=True)


class LeafSetup:
    def __init__(self, arch, dec, rel, shots):
        dem, _ = dem_of(arch, rel)
        self.nd = dem.num_detectors
        mech, self.merged = merge_mech(mechanisms(dem))
        self.setup = milp_setup(mech, self.nd)
        shots = np.array(sorted(set(int(s) for s in shots)), np.int64)
        _, raw = load_syn(arch, rel, shots)
        syn = np.unpackbits(raw, axis=1, bitorder="little")[:, :self.nd]
        self.row = {int(s): i for i, s in enumerate(shots)}
        self.syn = syn
        self.tess = dict(zip(shots.tolist(), stored_pred(dec, rel, "rl_s5t", "tesseract.npz", shots).tolist()))
        self.ml = dict(zip(shots.tolist(), stored_pred(dec, rel, "rl_s5", "ml_tau2_1e-6-1e-8_0.1.npz", shots).tolist()))


def verdict(setup, s, m, cls):
    """stage 2's rule for the class cls against the proven optimum m."""
    if m["cls"] == cls:
        return "same_class", None
    q = solve(setup, s, TIME_LIMIT, cls=cls)
    if q["valid"] and q["cost"] <= m["cost"] + COST_TOL:
        return "tie", q
    if q["status"] == 0 and q["valid"]:
        return "miss", q
    if q["dual_bound"] is not None and q["dual_bound"] > m["cost"] + COST_TOL:
        return "miss", q
    return "miss_undetermined", q


def done_keys(records):
    out = set()
    if os.path.exists(records):
        with open(records) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        x = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a line cut by a restart; the shot is run again
                    out.add((x["leaf"], int(x["idx"])))
    return out


def append(records, rec):
    with open(records, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cmd_run(arch, dec, plan_path, phase, records, guard):
    phase, guard = int(phase), float(guard)
    c0, w0 = time.process_time(), time.time()
    plan = json.load(open(plan_path))
    key = f"phase{phase}"

    def log(msg):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)

    done = done_keys(records)
    log(f"start {key}: {len(done)} shots already in {records}; guard {guard:.0f} CPU-s; nice {os.nice(0)}")
    for r in (30, 13, 10):
        L = [x for x in plan["leaves"] if x["r"] == r and x[key]]
        if not L:
            continue
        todo = [(x["leaf"], s) for i in range(max(len(x[key]) for x in L)) for x in L if i < len(x[key])
                for s in [x[key][i]] if (x["leaf"], s) not in done]
        if not todo:
            continue
        if time.process_time() - c0 > guard:
            log("guard reached; stopping")
            return
        setups = {x["leaf"]: LeafSetup(arch, dec, x["leaf"], x[key]) for x in L}
        n_done = 0
        for rel, s in todo:
            if time.process_time() - c0 > guard:
                log(f"guard reached at r = {r} after {n_done} shots this run; stopping")
                return
            S = setups[rel]
            syn = S.syn[S.row[s]]
            m = solve(S.setup, syn, TIME_LIMIT)
            rec = {"leaf": rel, "r": r, "idx": int(s), "phase": phase, "milp": m,
                   "stored_tesseract_class": int(S.tess[s]), "ml_class": int(S.ml[s]), "merged": S.merged}
            rec["proven"] = bool(m["status"] == 0 and m["valid"])
            if rec["proven"]:
                v, q = verdict(S.setup, syn, m, rec["stored_tesseract_class"])
                if q is not None:
                    rec["milp_in_stored_class"] = q
            else:
                v = "unproven"
            rec["verdict"] = v
            rec["miss"] = v in ("miss", "miss_undetermined")
            append(records, rec)
            n_done += 1
            if n_done % 200 == 0:
                log(f"r = {r}: {n_done} of {len(todo)} shots this run (process {time.process_time() - c0:.0f} CPU-s)")
        log(f"r = {r} done: {n_done} shots this run (process {time.process_time() - c0:.0f} CPU-s, "
            f"wall {time.time() - w0:.0f} s)")
    log(f"{key} done: process {time.process_time() - c0:.0f} CPU-s, wall {time.time() - w0:.0f} s")


def cmd_resolve(arch, dec, list_path, records, guard):
    guard = float(guard)
    c0 = time.process_time()
    items = json.load(open(list_path))
    done = done_keys(records)
    by_leaf = {}
    for it in items:
        by_leaf.setdefault(it["leaf"], []).append(int(it["idx"]))
    setups = {}
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} resolve: {len(items)} shots, "
          f"{len(done)} done already; guard {guard:.0f} CPU-s", flush=True)
    for it in items:
        rel, s = it["leaf"], int(it["idx"])
        if (rel, s) in done:
            continue
        if time.process_time() - c0 > guard:
            print("guard reached; stopping", flush=True)
            return
        if rel not in setups:
            setups[rel] = LeafSetup(arch, dec, rel, by_leaf[rel])
        S = setups[rel]
        syn = S.syn[S.row[s]]
        m = solve(S.setup, syn, TIME_LIMIT)
        rec = {"leaf": rel, "idx": s, "milp": m, "classes": it["classes"], "proven": bool(m["status"] == 0 and m["valid"]),
               "verdicts": {}}
        if rec["proven"]:
            per_class = {}
            for c in sorted(set(int(c) for c in it["classes"].values())):
                v, q = verdict(S.setup, syn, m, c)
                per_class[c] = v
                if q is not None:
                    rec[f"milp_in_class_{c}"] = q
            rec["verdicts"] = {name: per_class[int(c)] for name, c in it["classes"].items()}
        append(records, rec)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} resolve done: "
          f"{time.process_time() - c0:.0f} CPU-s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "plan" and len(a) == 4:
        cmd_plan(*a[1:])
    elif a and a[0] == "run" and len(a) == 7:
        cmd_run(*a[1:])
    elif a and a[0] == "resolve" and len(a) == 6:
        cmd_resolve(*a[1:])
    else:
        raise SystemExit(__doc__)
