"""The exact test of the gap-growth mechanism, as approved by main (10 Oct 2026, after the timing test): stage 1, the
exact most-probable-error (MPE) class of every Tesseract shot at d = 3; stage 2, an integer-programming certificate of
decoder's stored Tesseract at d = 5 on a fixed random sample. EXPLORATORY (the stages are fixed in advance, the question
came after the Tesseract split was seen). Design and rules: notes/reservoir/gapmech_design.md, written before the run.

reservoir, 2026-10-10. New code; imports reservoir's mpe.py and mpe_milp.py (and through them p4_check.py,
p4_redecode.py, qrc/outpaths.py), the tesseract_decoder package (decoder's settings) and scipy's HiGHS.

  plan  DECODE PLAN_OUT        draws stage 2's sample (seed 20261010) from the stored Tesseract shots of each d = 5
                               leaf: 50 per leaf at r = 10 and 13, 5 per leaf at r = 30 (8 series each; 840 shots),
                               and writes the plan (stage 1's leaves, stage 2's shot lists, limits, CPU guard)
  run   ARCHIVE DECODE OUT_DIR PLAN
                               stage 1 then stage 2, one process; stops before a leaf or a shot once the process CPU
                               time passes the plan's guard (5000 s: main's 1.5 core-hour cap less a margin)
Stage 1, per d = 3 leaf (18 series x r in {10, 13, 30, 50}): the RL DEM's mechanisms (merged as in mpe.py), the exact
max-product contraction on Tesseract's shots (its stored idx, every 5th shot), and
  OUT_DIR/exact/<leaf>.npz  idx (int32), mpe (uint8: 1 if ln M_1 > ln M_0), ln_m0, ln_m1 (float64)
  OUT_DIR/exact_stage1.json per-leaf counts: shots, ties (|ln M_1 - ln M_0| < 1e-9), MPE against the stored Tesseract,
                            stored ML and the outcome; CPU
Stage 2, per sampled d = 5 shot: the proven-optimal MPE by integer programming (HiGHS, wall-clock limit per shot,
solution checked against the syndrome); where its class differs from the stored Tesseract class, a second integer program
restricted to the stored class decides between a tie (an optimum in the stored class too, within COST_TOL) and a miss
(the stored class's optimum is above the overall optimum; a bound that settles neither counts as a miss). Tesseract is
rerun with decoder's settings as a diagnostic only (class against the stored one, cost against the optimum, flag).
  OUT_DIR/exact_stage2.json per-shot records; per round and in all: verdicts (same_class, tie, miss, miss_undetermined,
                            unproven), misses among the proven shots with the one-sided 95% Clopper-Pearson upper bound,
                            and the conservative bound that counts unproven shots as misses
Usage: python3 -I exact_run.py plan DECODE PLAN_OUT | run ARCHIVE DECODE OUT_DIR PLAN   (outputs through qrc.outpaths)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpe import (ARCH, CM_RL, TIE, layered_mpe, merge_mech, prepare_max, run_tesseract,  # noqa: E402
                 tess_kw)
from mpe_milp import milp_setup  # noqa: E402
from p4_check import mechanisms  # noqa: E402  (path set by mpe)
from qrc.outpaths import out_path  # noqa: E402

P3 = ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"]
P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
BASES = ["X", "Z"]
SEED = 20261010
DECODER_TESSERACT = {"det_beam": 20, "beam_climbing": True, "no_revisit_dets": True, "pqlimit": 1000000,
                     "num_det_orders": 20}
COST_TOL = 1e-6  # ln-probability units: solutions within this of each other count as equally probable (HiGHS's
#                  default absolute MIP gap is 1e-6)


def rel_of(d, patch, basis, r):
    return f"d{d}_at_{patch}/{basis}/r{r:02d}"


def cmd_plan(dec, plan_out):
    rng = np.random.default_rng(SEED)
    leaves2 = []
    for r, n in ((10, 50), (13, 50), (30, 5)):
        for p in P5:
            for b in BASES:
                rel = rel_of(5, p, b, r)
                z = np.load(os.path.join(dec, ARCH, rel, "rl_s5t", "tesseract.npz"), allow_pickle=False)
                cand = np.sort(z["idx"].astype(np.int64))
                pick = np.sort(rng.choice(cand, n, replace=False))
                leaves2.append({"leaf": rel, "r": r, "candidates": int(cand.size),
                                "candidates_are_every_5th": bool(np.array_equal(cand, np.arange(0, 50000, 5))),
                                "shots": [int(s) for s in pick]})
    plan = {"note": "Exact-test plan (reservoir, 2026-10-10), approved by main. Stage 1: exact MPE (mpe.py) on the "
                    "stored Tesseract shots (every 5th) of the 72 d = 3 W5 leaves. Stage 2: integer-programming "
                    "certificate (mpe_milp.py) of the stored d = 5 Tesseract on these shots, drawn with numpy's "
                    "default_rng(20261010) in the order listed (r = 10, 13, 30; patches q4_7, q6_5, q6_9, q8_7; X, Z).",
            "seed": SEED,
            "stage1": {"leaves": [rel_of(3, p, b, r) for p in P3 for b in BASES for r in (10, 13, 30, 50)],
                       "shots": "the stored Tesseract idx of each leaf (rl_s5/tesseract.npz), checked to be every 5th",
                       "batch": 2500},
            "stage2": {"leaves": leaves2, "time_limit_s": 60, "time_limit_is": "HiGHS wall clock, per integer program",
                       "cost_tol": COST_TOL, "tesseract": DECODER_TESSERACT},
            "cpu_guard_s": 5000,
            "cpu_guard_note": "process CPU time, checked before each stage-1 leaf and each stage-2 shot; main's cap is "
                              "1.5 core-hours (5400 s) for the run, its tests and the analysis together"}
    with open(out_path(plan_out), "w") as f:
        json.dump(plan, f, indent=1)
    print(f"plan: {sum(len(x['shots']) for x in leaves2)} stage-2 shots; written {plan_out}", flush=True)


def load_syn(arch, rel, shots):
    from willow_check import read_obs_b8
    a_dir = os.path.join(arch, rel)
    act, _ = read_obs_b8(os.path.join(a_dir, "obs_flips_actual.b8"))
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(act.size, -1)
    return act, raw[shots]


def dem_of(arch, rel):
    import stim
    dem = stim.DetectorErrorModel.from_file(os.path.join(arch, rel, "decoding_results", CM_RL, "error_model.dem"))
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(dem.num_detectors)])
    return dem, t_of


def stored_pred(dec, rel, sub, fn, shots):
    z = np.load(os.path.join(dec, ARCH, rel, sub, fn), allow_pickle=False)
    pos = {int(i): k for k, i in enumerate(z["idx"])}
    return z["pred"].reshape(z["idx"].size, -1)[:, 0][[pos[int(s)] for s in shots]].astype(np.uint8)


def solve(setup, s, time_limit, cls=None):
    """min w.e subject to H e = s (mod 2) and, when cls is given, L.e = cls (mod 2); HiGHS with relative MIP gap 0.
    The returned solution is checked against the syndrome (and the class) after rounding."""
    A, c, bounds, w, L = setup
    nm, nd = w.size, A.shape[0]
    H = A[:, :nm]
    lb, ub, rhs = bounds.lb, bounds.ub, s.astype(float)
    if cls is not None:
        A = sp.vstack([sp.hstack([A, sp.csr_matrix((nd, 1))]),
                       sp.csr_matrix(np.concatenate([L.astype(float), np.zeros(nd), [-2.0]])[None, :])], format="csr")
        c = np.concatenate([c, [0.0]])
        lb = np.concatenate([lb, [0.0]])
        ub = np.concatenate([ub, [float(L.sum() // 2)]])
        rhs = np.concatenate([rhs, [float(cls)]])
    t1 = time.process_time()
    r = milp(c, integrality=np.ones(c.size), bounds=Bounds(lb, ub), constraints=LinearConstraint(A, rhs, rhs),
             options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False})
    db = getattr(r, "mip_dual_bound", None)
    out = {"status": int(r.status), "cpu_s": time.process_time() - t1,
           "dual_bound": float(db) if db is not None and np.isfinite(db) else None}
    if r.x is not None:
        e = np.round(r.x[:nm]).astype(np.int64)
        syn_ok = bool(np.array_equal(np.rint(H @ e).astype(np.int64) % 2, s.astype(np.int64) % 2))
        out.update(cost=float(w @ e), cls=int(L @ e % 2), weight=int(e.sum()))
        out["valid"] = syn_ok and (cls is None or out["cls"] == cls)
    else:
        out.update(cost=None, cls=-1, weight=None, valid=False)
    return out


def stage1(arch, dec, out_dir, plan, c0, guard, log):
    res = {"leaves": [], "stopped_by_guard": False}
    for rel in plan["stage1"]["leaves"]:
        if time.process_time() - c0 > guard:
            res["stopped_by_guard"] = True
            break
        t1 = time.process_time()
        r = int(rel.split("/r")[-1])
        z = np.load(os.path.join(dec, ARCH, rel, "rl_s5", "tesseract.npz"), allow_pickle=False)
        idx = z["idx"].astype(np.int64)
        every5 = bool(np.array_equal(idx, np.arange(0, 50000, 5)))
        tes = z["pred"].reshape(idx.size, -1)[:, 0].astype(np.uint8)
        dem, t_of = dem_of(arch, rel)
        nd = dem.num_detectors
        mech, merged = merge_mech(mechanisms(dem))
        prep = prepare_max(mech, t_of, r)
        act, raw = load_syn(arch, rel, idx)
        m0 = np.empty(idx.size)
        m1 = np.empty(idx.size)
        B = plan["stage1"]["batch"]
        for a in range(0, idx.size, B):
            syn = np.unpackbits(raw[a:a + B], axis=1, bitorder="little")[:, :nd]
            m0[a:a + B], m1[a:a + B] = layered_mpe(prep, syn)
        mpe = (m1 > m0).astype(np.uint8)
        tie = np.abs(m1 - m0) < TIE
        ml = stored_pred(dec, rel, "rl", "ml_exact.npz", idx)
        obs = act[idx]
        np.savez_compressed(out_path(os.path.join(out_dir, "exact", rel.replace("/", "_") + ".npz")),
                            idx=idx.astype(np.int32), mpe=mpe, ln_m0=m0, ln_m1=m1)
        ne_t = (mpe != tes) & ~tie
        L = {"leaf": rel, "shots": int(idx.size), "shots_every_5th": every5, "n_det": int(nd),
             "mechanisms": len(mech), "merged_duplicates": merged, "finite": bool(np.all(np.isfinite(m0) |
                                                                                         np.isfinite(m1))),
             "ties": int(tie.sum()), "tie_idx": [int(i) for i in idx[tie]],
             "mpe_ne_stored_tesseract": int(ne_t.sum()), "mpe_ne_stored_tesseract_idx": [int(i) for i in idx[ne_t]],
             "mpe_ne_ml": int(np.sum(mpe != ml)), "errors": {"mpe": int(np.sum(mpe != obs)),
                                                             "tesseract": int(np.sum(tes != obs)),
                                                             "ml": int(np.sum(ml != obs))},
             "cpu_s": time.process_time() - t1}
        res["leaves"].append(L)
        log(f"stage 1 {rel}: T != MPE {L['mpe_ne_stored_tesseract']}, ties {L['ties']}, MPE != ML {L['mpe_ne_ml']}, "
            f"cpu {L['cpu_s']:.1f} s (total {time.process_time() - c0:.0f} s)")
        with open(out_path(os.path.join(out_dir, "exact_stage1.json")), "w") as f:
            json.dump(res, f, indent=1)
    tot = res["leaves"]
    res["summary"] = {"leaves": len(tot), "shots": sum(x["shots"] for x in tot), "ties": sum(x["ties"] for x in tot),
                      "mpe_ne_stored_tesseract": sum(x["mpe_ne_stored_tesseract"] for x in tot),
                      "mpe_ne_ml": sum(x["mpe_ne_ml"] for x in tot),
                      "all_every_5th": all(x["shots_every_5th"] for x in tot),
                      "cpu_s": sum(x["cpu_s"] for x in tot)}
    with open(out_path(os.path.join(out_dir, "exact_stage1.json")), "w") as f:
        json.dump(res, f, indent=1)
    return res


def stage2(arch, dec, out_dir, plan, c0, guard, log):
    P = plan["stage2"]
    recs, stopped = [], False
    for leaf in P["leaves"]:
        if stopped:
            break
        rel, r = leaf["leaf"], leaf["r"]
        shots = np.array(leaf["shots"], np.int64)
        dem, t_of = dem_of(arch, rel)
        nd = dem.num_detectors
        mech, merged = merge_mech(mechanisms(dem))
        setup = milp_setup(mech, nd)
        act, raw = load_syn(arch, rel, shots)
        syn = np.unpackbits(raw, axis=1, bitorder="little")[:, :nd]
        stored = stored_pred(dec, rel, "rl_s5t", "tesseract.npz", shots)
        ml = stored_pred(dec, rel, "rl_s5", "ml_tau2_1e-6-1e-8_0.1.npz", shots)
        if time.process_time() - c0 > guard:
            stopped = True
            break
        t = run_tesseract(dem, syn, tess_kw({"x": {"kw": P["tesseract"]}}, "x"), 1e9)  # one compile per leaf
        for i, s in enumerate(shots):
            if time.process_time() - c0 > guard:
                stopped = True
                break
            m = solve(setup, syn[i], P["time_limit_s"])
            rec = {"leaf": rel, "r": r, "idx": int(s), "milp": m, "stored_tesseract_class": int(stored[i]),
                   "tesseract_rerun": {"cls": int(t["pred"][i]), "cost": float(t["cost"][i]),
                                       "low_confidence": bool(t["low"][i]), "cpu_s": float(t["per_shot_cpu"][i])},
                   "ml_class": int(ml[i]), "outcome": int(act[s])}
            proven = m["status"] == 0 and m["valid"]
            rec["proven"] = proven
            rec["rerun_equals_stored"] = rec["tesseract_rerun"]["cls"] == rec["stored_tesseract_class"]
            rec["rerun_cost_minus_optimum"] = rec["tesseract_rerun"]["cost"] - m["cost"] if proven else None
            # Verdict on the stored class (the rerun is a diagnostic only): same_class if the proven optimum lies in
            # it; otherwise the optimum within the stored class decides: tie if within COST_TOL of the optimum, miss if
            # above it; a stored-class bound that settles neither counts as a miss (miss_undetermined).
            if not proven:
                v = "unproven"
            elif m["cls"] == rec["stored_tesseract_class"]:
                v = "same_class"
            else:
                q = solve(setup, syn[i], P["time_limit_s"], cls=rec["stored_tesseract_class"])
                rec["milp_in_stored_class"] = q
                if q["valid"] and q["cost"] <= m["cost"] + COST_TOL:
                    v = "tie"
                elif q["status"] == 0 and q["valid"]:
                    v = "miss"
                elif q["dual_bound"] is not None and q["dual_bound"] > m["cost"] + COST_TOL:
                    v = "miss"
                else:
                    v = "miss_undetermined"
            rec["verdict"] = v
            rec["miss"] = v in ("miss", "miss_undetermined")
            recs.append(rec)
        done = [x for x in recs if x["leaf"] == rel]
        log(f"stage 2 {rel}: {len(done)} shots, proven {sum(x['proven'] for x in done)}, misses "
            f"{sum(x['miss'] for x in done)}, ties {sum(x['verdict'] == 'tie' for x in done)}, rerun != stored "
            f"{sum(not x['rerun_equals_stored'] for x in done)} (total cpu {time.process_time() - c0:.0f} s)")
        with open(out_path(os.path.join(out_dir, "exact_stage2.json")), "w") as f:
            json.dump({"records": recs, "stopped_by_guard": stopped}, f, indent=1)
    out = {"records": recs, "stopped_by_guard": stopped, "cost_tol": COST_TOL, "summary": {}}
    cp = lambda k, n: float(stats.beta.ppf(0.95, k + 1, n - k)) if n and k < n else (1.0 if n else None)  # noqa: E731
    for tag, sel in (("all", lambda x: True), ("r10", lambda x: x["r"] == 10), ("r13", lambda x: x["r"] == 13),
                     ("r30", lambda x: x["r"] == 30)):
        R = [x for x in recs if sel(x)]
        n = sum(x["proven"] for x in R)
        k = sum(x["miss"] for x in R)
        u = len(R) - n
        cnt = {v: sum(x["verdict"] == v for x in R) for v in ("same_class", "tie", "miss", "miss_undetermined",
                                                              "unproven")}
        out["summary"][tag] = {
            "shots_run": len(R), "proven": n, "unproven": u, "misses": k, "verdicts": cnt,
            "miss_rate": k / n if n else None, "miss_rate_upper95_one_sided_clopper_pearson": cp(k, n),
            "conservative_unproven_as_misses": {"k": k + u, "n": len(R), "rate": (k + u) / len(R) if R else None,
                                                "upper95_one_sided_clopper_pearson": cp(k + u, len(R))},
            "rerun_ne_stored": sum(not x["rerun_equals_stored"] for x in R),
            "rerun_cost_above_optimum": sum(x["rerun_cost_minus_optimum"] is not None
                                            and x["rerun_cost_minus_optimum"] > COST_TOL for x in R),
            "rerun_low_confidence": sum(x["tesseract_rerun"]["low_confidence"] for x in R),
            "stored_wrong": sum(x["stored_tesseract_class"] != x["outcome"] for x in R),
            "ml_wrong": sum(x["ml_class"] != x["outcome"] for x in R),
            "milp_cpu_s": sum(x["milp"]["cpu_s"] + x.get("milp_in_stored_class", {}).get("cpu_s", 0.0) for x in R),
            "tesseract_cpu_s": sum(x["tesseract_rerun"]["cpu_s"] for x in R)}
    with open(out_path(os.path.join(out_dir, "exact_stage2.json")), "w") as f:
        json.dump(out, f, indent=1)
    return out


def cmd_run(arch, dec, out_dir, plan_path):
    plan = json.load(open(plan_path))
    t0, c0 = time.time(), time.process_time()
    guard = plan["cpu_guard_s"]

    def log(msg):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)

    log(f"start; guard {guard} CPU-s; nice {os.nice(0)}")
    s1 = stage1(arch, dec, out_dir, plan, c0, guard, log)
    log(f"stage 1 done: {json.dumps(s1['summary'])}")
    s2 = stage2(arch, dec, out_dir, plan, c0, guard, log)
    log(f"stage 2 done: {json.dumps(s2['summary']['all'])}")
    log(f"end: cpu {time.process_time() - c0:.0f} s, wall {time.time() - t0:.0f} s, "
        f"stopped by guard: {s1['stopped_by_guard'] or s2['stopped_by_guard']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "plan" and len(a) == 3:
        cmd_plan(a[1], a[2])
    elif a and a[0] == "run" and len(a) == 5:
        cmd_run(a[1], a[2], a[3], a[4])
    else:
        raise SystemExit(__doc__)
