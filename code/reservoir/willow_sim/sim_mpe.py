"""Most-probable-error (MPE) decoders on the simulated leaves (notes/reservoir/sim_design.md, approved by main
10 Oct 2026, stage A). EXPLORATORY: post hoc question, pre-specified design.

reservoir, 2026-10-10. New code; imports reservoir's mpe.py, mpe_milp.py and exact_run.py (willow_gapmech), and through
them p4_check.py, p4_redecode.py and qrc/outpaths.py; the tesseract_decoder package and scipy's HiGHS.

  d3 PLAN LEAF    exact max-product MPE class of every shot (mpe.py, as stage 1 of the exact test): ln M_0, ln M_1,
                  mpe = 1 if ln M_1 > ln M_0, ties |ln M_1 - ln M_0| < 1e-9 counted (they stay at class 0)
                  -> <mpe dir>/<leaf>.npz (idx, mpe, ln_m0, ln_m1) and <leaf>.json
  d5 PLAN LEAF    per shot: Tesseract with an infinite beam (det_beam INF, no beam climbing, no_revisit_dets,
                  pqlimit 1e6, 1 order); shots with the low-confidence flag are decoded again with decoder's settings
                  (det_beam 20, beam climbing, 20 orders). The integer program (exact_run.solve: HiGHS, relative MIP
                  gap 0, the plan's wall-clock limit per program, solution checked against the syndrome) certifies
                    every shot where Tesseract's class differs from ML's (decode_willow.py's ladder, dem tag rl_sim),
                    every other shot still flagged low-confidence after the fallback, and
                    a random sample of the shots where Tesseract equals ML (the plan's count per leaf; a generator
                    seeded per leaf from the plan).
                  Verdicts as in stage 2 of the exact test: same_class; tie (an optimum within COST_TOL also in
                  Tesseract's class, by a class-restricted program); miss (the restricted optimum, or its proven bound,
                  lies above the overall optimum); miss_undetermined (neither settled; counted as a miss); unproven (no
                  proven optimum in the limit). The corrected MPE class is the proven optimum's class on misses (both
                  kinds) and Tesseract's class otherwise.
                  -> <mpe dir>/<leaf>.npz (idx, tess, tess_cost, low_inf, low_final, fallback, ml, mpe, cert: 0 none,
                     1 discordant, 2 concordant sample, 3 low-confidence; verdict: index in VERDICTS) and <leaf>.json
                     (records of every certified shot, counts by kind, CPU)
Checkpoints: Tesseract's per-shot results (<leaf>_tess.npz) and the certificate records so far (<leaf>_cert.json,
rewritten every 10 shots) are reused when the step is restarted.
Usage: python3 -I sim_mpe.py d3|d5 PLAN LEAF
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import stim
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_gapmech"))
from mpe import TIE, layered_mpe, merge_mech, prepare_max, run_tesseract, tess_kw  # noqa: E402
from mpe_milp import milp_setup  # noqa: E402
from exact_run import COST_TOL, DECODER_TESSERACT, solve  # noqa: E402
from p4_check import mechanisms  # noqa: E402  (path set by mpe)

ARCH = "google_105Q_surface_code_d3_d5_d7"
DEM_REL = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
INF_BEAM = {"det_beam": "INF", "beam_climbing": False, "no_revisit_dets": True, "pqlimit": 1000000,
            "num_det_orders": 1}
VERDICTS = ["none", "same_class", "tie", "miss", "miss_undetermined", "unproven"]
ML5 = "ml_tau2_1e-6-1e-8_0.1.npz"


def leaf_of(plan, rel):
    for L in plan["leaves"]:
        if L["leaf"] == rel:
            return L
    raise SystemExit(f"{rel} not in plan")


def load(plan, rel):
    leaf = os.path.join(plan["paths"]["sim_root"], ARCH, rel)
    dem = stim.DetectorErrorModel.from_file(os.path.join(leaf, DEM_REL))
    nd = dem.num_detectors
    co = dem.get_detector_coordinates()
    t_of = np.array([int(round(co[i][2])) for i in range(nd)])
    n = json.load(open(os.path.join(leaf, "sample.json")))["shots"]
    raw = np.fromfile(os.path.join(leaf, "detection_events.b8"), dtype=np.uint8)
    nb = (nd + 7) // 8
    if raw.size != n * nb:
        raise SystemExit(f"{rel}: detection_events.b8 has {raw.size} bytes, expected {n} x {nb}")
    return dem, nd, t_of, raw.reshape(n, nb)


def unpack(raw, nd):
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :nd]


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def run_d3(plan, rel):
    t0 = time.process_time()
    out = os.path.join(plan["paths"]["mpe_dir"], rel.replace("/", "_"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dem, nd, t_of, raw = load(plan, rel)
    r = int(rel.split("/r")[-1])
    mech, merged = merge_mech(mechanisms(dem))
    prep = prepare_max(mech, t_of, r)
    n = raw.shape[0]
    m0, m1 = np.empty(n), np.empty(n)
    B = 2500
    for a in range(0, n, B):
        m0[a:a + B], m1[a:a + B] = layered_mpe(prep, unpack(raw[a:a + B], nd))
    mpe = (m1 > m0).astype(np.uint8)
    tie = np.abs(m1 - m0) < TIE
    np.savez_compressed(out + ".tmp.npz", idx=np.arange(n, dtype=np.int32), mpe=mpe, ln_m0=m0, ln_m1=m1)
    os.replace(out + ".tmp.npz", out + ".npz")
    rec = {"leaf": rel, "shots": n, "n_det": nd, "mechanisms": len(mech), "merged_duplicates": merged,
           "finite": bool(np.all(np.isfinite(m0) | np.isfinite(m1))), "ties": int(tie.sum()),
           "tie_idx": [int(i) for i in np.where(tie)[0]], "mpe_class1": int(mpe.sum()), "cpu_s": time.process_time() - t0}
    save_json(out + ".json", rec)
    print(f"mpe d3 {rel}: {n} shots, ties {rec['ties']}, cpu {rec['cpu_s']:.1f} s", flush=True)


def certify(setup, s, t_cls, limit):
    m = solve(setup, s, limit)
    rec = {"milp": m, "proven": bool(m["status"] == 0 and m["valid"])}
    if not rec["proven"]:
        v = "unproven"
    elif m["cls"] == t_cls:
        v = "same_class"
    else:
        q = solve(setup, s, limit, cls=t_cls)
        rec["milp_in_tesseract_class"] = q
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
    return rec


def run_d5(plan, rel):
    c0 = time.process_time()
    P = plan["mpe_d5"]
    L = leaf_of(plan, rel)
    out = os.path.join(plan["paths"]["mpe_dir"], rel.replace("/", "_"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dem, nd, t_of, raw = load(plan, rel)
    n = raw.shape[0]
    syn = unpack(raw, nd)
    # ML: decode_willow.py's ladder on the same leaf (all shots, in order)
    z = np.load(os.path.join(plan["paths"]["decode_out"], ARCH, rel, "rl_sim", ML5), allow_pickle=False)
    if not np.array_equal(z["idx"].astype(np.int64), np.arange(n)):
        raise SystemExit(f"{rel}: ML idx is not 0..n-1")
    ml = z["pred"].reshape(n, -1)[:, 0].astype(np.uint8)
    # Tesseract, infinite beam, then decoder's settings on the low-confidence shots
    tp = out + "_tess.npz"
    if os.path.exists(tp):
        t = dict(np.load(tp, allow_pickle=False))
    else:
        t1 = time.process_time()
        a = run_tesseract(dem, syn, tess_kw({"x": {"kw": INF_BEAM}}, "x"), 1e12)
        if a["pred"].size != n:
            raise SystemExit(f"{rel}: Tesseract stopped early")
        t = {"tess": a["pred"].copy(), "tess_cost": a["cost"].copy(), "low_inf": a["low"].copy(),
             "low_final": a["low"].copy(), "fallback": np.zeros(n, bool), "cpu_inf": a["per_shot_cpu"],
             "compile_cpu_inf": np.array(a["compile_cpu_s"]), "cpu_fallback": np.zeros(n)}
        lo = np.where(a["low"])[0]
        if lo.size:
            b = run_tesseract(dem, syn[lo], tess_kw({"x": {"kw": DECODER_TESSERACT}}, "x"), 1e12)
            t["fallback"][lo] = True
            t["cpu_fallback"][lo] = b["per_shot_cpu"]
            ok = ~b["low"]
            t["tess"][lo[ok]] = b["pred"][ok]
            t["tess_cost"][lo[ok]] = b["cost"][ok]
            t["low_final"][lo] = b["low"]
        t["cpu_tesseract_s"] = np.array(time.process_time() - t1)
        np.savez_compressed(tp + ".tmp.npz", **t)
        os.replace(tp + ".tmp.npz", tp)
    tess = t["tess"].astype(np.uint8)
    low = t["low_final"].astype(bool)
    disc = np.where(tess != ml)[0]
    lowc = np.where((tess == ml) & low)[0]
    conc_all = np.where((tess == ml) & ~low)[0]
    rng = np.random.default_rng(np.random.SeedSequence(P["concordant_seed"]).spawn(P["leaves_d5_r10"])[L["d5_index"]])
    k = min(P["concordant_per_leaf"], conc_all.size)
    conc = np.sort(rng.choice(conc_all, k, replace=False)) if k else np.zeros(0, np.int64)
    # certificate, with a resumable record file
    mech, merged = merge_mech(mechanisms(dem))
    setup = milp_setup(mech, nd)
    cp = out + "_cert.json"
    recs = json.load(open(cp))["records"] if os.path.exists(cp) else []
    done = {x["idx"] for x in recs}
    todo = ([(int(i), "discordant") for i in disc] + [(int(i), "low_confidence") for i in lowc]
            + [(int(i), "concordant_sample") for i in conc])
    for j, (i, kind) in enumerate(todo):
        if i in done:
            continue
        rec = certify(setup, syn[i], int(tess[i]), P["time_limit_s"])
        rec.update(idx=i, kind=kind, tesseract_class=int(tess[i]), tesseract_cost=float(t["tess_cost"][i]),
                   low_confidence=bool(low[i]), fallback=bool(t["fallback"][i]), ml_class=int(ml[i]))
        rec["tesseract_cost_minus_optimum"] = (rec["tesseract_cost"] - rec["milp"]["cost"]) if rec["proven"] else None
        recs.append(rec)
        if len(recs) % 10 == 0:
            save_json(cp, {"records": recs})
    save_json(cp, {"records": recs})
    # corrected MPE class
    mpe = tess.copy()
    cert = np.zeros(n, np.uint8)
    verdict = np.zeros(n, np.uint8)
    for x in recs:
        i = x["idx"]
        cert[i] = {"discordant": 1, "concordant_sample": 2, "low_confidence": 3}[x["kind"]]
        verdict[i] = VERDICTS.index(x["verdict"])
        if x["miss"]:
            mpe[i] = x["milp"]["cls"]
    np.savez_compressed(out + ".tmp.npz", idx=np.arange(n, dtype=np.int32), tess=tess, tess_cost=t["tess_cost"],
                        low_inf=t["low_inf"].astype(bool), low_final=low, fallback=t["fallback"].astype(bool), ml=ml,
                        mpe=mpe, cert=cert, verdict=verdict)
    os.replace(out + ".tmp.npz", out + ".npz")
    cpb = lambda kk, nn: float(stats.beta.ppf(0.95, kk + 1, nn - kk)) if nn and kk < nn else (1.0 if nn else None)  # noqa: E731
    summ = {}
    for kind in ("discordant", "low_confidence", "concordant_sample"):
        R = [x for x in recs if x["kind"] == kind]
        npr = sum(x["proven"] for x in R)
        km = sum(x["miss"] for x in R)
        summ[kind] = {"shots": len(R), "proven": npr, "misses": km,
                      "verdicts": {v: sum(x["verdict"] == v for x in R) for v in VERDICTS[1:]},
                      "miss_rate_upper95_one_sided_clopper_pearson": cpb(km, npr),
                      "milp_cpu_s": sum(x["milp"]["cpu_s"] + x.get("milp_in_tesseract_class", {}).get("cpu_s", 0.0)
                                        for x in R),
                      "tesseract_cost_above_optimum": sum(x["tesseract_cost_minus_optimum"] is not None
                                                          and x["tesseract_cost_minus_optimum"] > COST_TOL for x in R)}
    res = {"leaf": rel, "shots": n, "n_det": nd, "mechanisms": len(mech), "merged_duplicates": merged,
           "tesseract": {"inf_beam": INF_BEAM, "fallback": DECODER_TESSERACT,
                         "low_confidence_inf_beam": int(t["low_inf"].sum()), "fallback_shots": int(t["fallback"].sum()),
                         "low_confidence_after_fallback": int(low.sum()),
                         "cpu_s": float(t["cpu_tesseract_s"]), "ms_per_shot_inf_mean": 1e3 * float(np.mean(t["cpu_inf"])),
                         "compile_cpu_s": float(t["compile_cpu_inf"])},
           "tesseract_ne_ml": int(np.sum(tess != ml)), "concordant_available": int(conc_all.size),
           "certificate": summ, "time_limit_s": P["time_limit_s"], "cost_tol": COST_TOL,
           "mpe_corrected_shots": int(np.sum(mpe != tess)), "mpe_ne_ml": int(np.sum(mpe != ml)),
           "records": recs, "cpu_s_this_process": time.process_time() - c0}
    save_json(out + ".json", res)
    print(f"mpe d5 {rel}: T != ML {res['tesseract_ne_ml']}, low {res['tesseract']['low_confidence_inf_beam']} -> "
          f"{res['tesseract']['low_confidence_after_fallback']}, discordant misses {summ['discordant']['misses']} of "
          f"{summ['discordant']['proven']} proven, concordant misses {summ['concordant_sample']['misses']} of "
          f"{summ['concordant_sample']['proven']}, cpu {res['cpu_s_this_process']:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 3 or a[0] not in ("d3", "d5"):
        raise SystemExit(__doc__)
    plan = json.load(open(a[1]))
    (run_d3 if a[0] == "d3" else run_d5)(plan, a[2])
