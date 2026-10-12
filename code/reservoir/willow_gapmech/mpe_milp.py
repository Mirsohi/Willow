"""An independent exact most-probable-error decoder for any distance: the integer program
    min_e  sum_j w_j e_j   subject to  H e - 2 k = s,  e_j in {0, 1},  0 <= k_i <= floor(deg_i / 2) integer,
with w_j = ln((1 - p_j) / p_j) (mechanisms merged as in mpe.py), solved by HiGHS (scipy.optimize.milp) to a proven
optimum (relative MIP gap 0). Used to time, and to certify on a few shots, Tesseract settings as an exact MPE decoder at
d = 5, where the dense max-product contraction of mpe.py is out of reach; at d = 3 it is checked against mpe.py's exact
max-product. Part of item 3 of main's brief (10 Oct 2026); EXPLORATORY, a timing test only (counts are by-products).

reservoir, 2026-10-10. New code; imports mpe.py (reservoir's) and the tesseract_decoder package (as decoder's runs use it).
Usage: nice -n 10 python3 -I mpe_milp.py ARCHIVE DECODE OUT_DIR PLAN  -> OUT_DIR/mpe_milp.json (through qrc.outpaths)
PLAN (JSON): {"time_limit_s": .., "budget_cpu_s": .., "tesseract": {tag: {"kw": {..}}},
              "leaves": [{"d": 3 or 5, "series": "q6_5/Z", "r": 10, "n": 20, "tesseract": [tags]}]}
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpe import layered_mpe, load_leaf, merge_mech, prepare_max, run_tesseract, tess_kw  # noqa: E402
from p4_check import mechanisms  # noqa: E402  (path set by mpe)
from qrc.outpaths import out_path  # noqa: E402


def milp_setup(mech, nd):
    rows = [d for _, ds, _ in mech for d in ds]
    cols = [j for j, (_, ds, _) in enumerate(mech) for _ in ds]
    H = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nd, len(mech)))
    deg = np.asarray(H.sum(1)).ravel()
    A = sp.hstack([H, -2.0 * sp.identity(nd, format="csr")], format="csr")
    w = np.array([math.log((1 - p) / p) for p, _, _ in mech])
    c = np.concatenate([w, np.zeros(nd)])
    bounds = Bounds(np.zeros(len(mech) + nd), np.concatenate([np.ones(len(mech)), np.floor(deg / 2)]))
    L = np.array([l for _, _, l in mech], np.int64)
    return A, c, bounds, w, L


def run_milp(setup, syn, time_limit, budget_cpu):
    A, c, bounds, w, L = setup
    nm = w.size
    integ = np.ones(c.size)
    pred, cost, status, per = [], [], [], []
    t_start = time.process_time()
    for i in range(syn.shape[0]):
        s = syn[i].astype(float)
        t1 = time.process_time()
        r = milp(c, integrality=integ, bounds=bounds, constraints=LinearConstraint(A, s, s),
                 options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False})
        per.append(time.process_time() - t1)
        status.append(int(r.status))
        if r.x is not None:
            e = np.round(r.x[:nm]).astype(np.int64)
            pred.append(int(L @ e % 2))
            cost.append(float(w @ e))
        else:
            pred.append(-1)
            cost.append(float("nan"))
        if time.process_time() - t_start > budget_cpu:
            break
    return {"pred": np.array(pred), "cost": np.array(cost), "status": np.array(status), "per": np.array(per)}


def summary(m):
    per = m["per"]
    return {"shots_decoded": int(per.size), "cpu_s": float(per.sum()),
            "ms_per_shot_mean": float(1e3 * per.mean()) if per.size else None,
            "ms_per_shot_median": float(1e3 * np.median(per)) if per.size else None,
            "ms_per_shot_max": float(1e3 * per.max()) if per.size else None,
            "proven_optimal": int(np.sum(m["status"] == 0)), "not_optimal": int(np.sum(m["status"] != 0))}


def main(arch, dec, out_dir, plan_path):
    plan = json.load(open(plan_path))
    t0, c0 = time.time(), time.process_time()
    res = {"label": "EXPLORATORY timing test (reservoir, 2026-10-10): exact MPE by integer programming (HiGHS) against "
                    "mpe.py's exact max-product (d = 3) and Tesseract settings (d = 3 and 5). Counts are by-products",
           "plan": plan, "scipy": scipy.__version__, "leaves": []}
    for leaf in plan["leaves"]:
        d, r = leaf["d"], leaf["r"]
        rel, dem, t_of, obs, shots, syn = load_leaf(arch, d, leaf["series"], r, leaf["n"])
        mech, merged = merge_mech(mechanisms(dem))
        nd = dem.num_detectors
        tc = time.process_time()
        setup = milp_setup(mech, nd)
        cpu_setup = time.process_time() - tc
        m = run_milp(setup, syn, plan["time_limit_s"], plan["budget_cpu_s"])
        k = m["pred"].size
        ok = m["status"] == 0
        L = {"leaf": rel, "d": d, "r": r, "n_det": int(nd), "mechanisms": len(mech), "merged_duplicates": merged,
             "shots_requested": int(shots.size), "cpu_s_setup": cpu_setup, "milp": summary(m),
             "errors_milp": int(np.sum((m["pred"] != obs[:k]) & ok))}
        C0 = float(sum(math.log1p(-p) for p, _, _ in mech))
        if d == 3:
            m0, m1 = layered_mpe(prepare_max(mech, t_of, r), syn[:k])
            cmin = C0 - np.maximum(m0, m1)
            mpe = (m1 > m0).astype(np.int64)
            gap = m["cost"] - cmin
            L["milp_vs_exact_max_product"] = {
                "class_ne": int(np.sum((m["pred"] != mpe) & ok)),
                "cost_gap_max_abs": float(np.nanmax(np.abs(gap[ok]))) if ok.any() else None}
        for tag in leaf.get("tesseract", []):
            t = run_tesseract(dem, syn[:k], tess_kw(plan["tesseract"], tag), 1e9)
            dc = t["cost"] - m["cost"]
            L[f"tesseract_{tag}_vs_milp"] = {
                "ms_per_shot_tesseract": float(1e3 * t["per_shot_cpu"].mean()),
                "low_confidence_shots": int(t["low"].sum()),
                "class_ne": int(np.sum((t["pred"] != m["pred"]) & ok)),
                "shots_tesseract_cost_above_milp_1e-6": int(np.sum((dc > 1e-6) & ok)),
                "shots_tesseract_cost_below_milp_1e-6": int(np.sum((dc < -1e-6) & ok)),
                "max_cost_excess": float(np.nanmax(dc[ok])) if ok.any() else None}
        res["leaves"].append(L)
        print(json.dumps(L), flush=True)
    res["cpu_s_total"] = time.process_time() - c0
    res["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "mpe_milp.json")), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 4:
        raise SystemExit(__doc__)
    main(*a)
