"""Round-2 F1 check, a descriptive split (post hoc, not pre-registered): m_D by basis and by basis and r, with the
held-out log-loss gain and R by basis, from the per-leaf cells of r2_leaves.csv (written by r2_check.py).

reservoir, 2026-10-10. m_D = K_RL / K_D - 1 over the selected leaves, with the SE of r2_check.py (P4's stratified delta
method: per shot z = e_RL - (1 + m) e_D, its variance within each leaf with n - 1, n var summed over the leaves,
SE = sqrt(sum) / K_D), computed exactly from each leaf's 2x2 cells. On all 72 leaves it must reproduce r2_check.json's
primary value and SE (else exit). The log-loss gain (n-weighted mean of the per-leaf nll_RL - nll_D) and R (sum K over
the sum of predicted failures, predicted = K / R per leaf) are point values: their SEs need the per-shot data.
Usage: python3 -I r2_basis.py OUT_DIR   (reads OUT_DIR/r2_leaves.csv and r2_check.json) -> OUT_DIR/r2_basis.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/reservoir (qrc.outpaths)
from qrc.outpaths import out_path  # noqa: E402


def group(rows):
    KR = sum(int(r["K_RL"]) for r in rows)
    KD = sum(int(r["K_D"]) for r in rows)
    m = KR / KD - 1.0
    v = 0.0
    for r in rows:
        n, b, c, w = (int(r[k]) for k in ("n_test", "RL_wrong_D_right", "RL_right_D_wrong", "both_wrong"))
        if int(r["K_RL"]) != b + w or int(r["K_D"]) != c + w:
            raise SystemExit(f"{r['patch']} {r['basis']} r{r['r']}: cells do not add up")
        s1 = b - c * (1.0 + m) - w * m
        s2 = b + c * (1.0 + m) ** 2 + w * m * m
        v += n * (s2 - s1 * s1 / n) / (n - 1)
    se = math.sqrt(v) / KD
    n = sum(int(r["n_test"]) for r in rows)
    pRL = sum(int(r["K_RL"]) / float(r["R_RL"]) for r in rows)
    pD = sum(int(r["K_D"]) / float(r["R_D"]) for r in rows)
    RRL, RD = KR / pRL, KD / pD
    return {"leaves": len(rows), "shots": n, "K_RL": KR, "K_D": KD, "m_D": m, "se": se, "z": m / se,
            "ci95": [m - 1.959963984540054 * se, m + 1.959963984540054 * se],
            "RL_wrong_D_right": sum(int(r["RL_wrong_D_right"]) for r in rows),
            "RL_right_D_wrong": sum(int(r["RL_right_D_wrong"]) for r in rows),
            "logloss_RL": sum(int(r["n_test"]) * float(r["nll_RL"]) for r in rows) / n,
            "logloss_gain_RL_minus_D": sum(int(r["n_test"]) * (float(r["nll_RL"]) - float(r["nll_D"])) for r in rows) / n,
            "R_RL": RRL, "R_D": RD, "share_of_RL_excess_removed": 1.0 - (RD - 1.0) / (RRL - 1.0),
            "events_per_shot": {k: sum(int(r["n_test"]) * float(r[f"events_{k}"]) for r in rows) / n
                                for k in ("data", "RL", "T", "D")}}


def main(out_dir):
    rows = list(csv.DictReader(open(os.path.join(out_dir, "r2_leaves.csv"))))
    prim = json.load(open(os.path.join(out_dir, "r2_check.json")))["primary_m_D"]
    res = {"label": "reservoir, round-2 F1 check: descriptive split by basis and r (post hoc, not pre-registered), "
                    "from r2_leaves.csv, 2026-10-10"}
    a = group(rows)
    if abs(a["m_D"] - prim["value"]) > 1e-12 or abs(a["se"] / prim["se"] - 1.0) > 1e-9:
        raise SystemExit(f"all leaves: {a['m_D']} {a['se']} against r2_check.json {prim['value']} {prim['se']}")
    res["all"] = a
    res["by_basis"] = {b: group([r for r in rows if r["basis"] == b]) for b in ("X", "Z")}
    res["by_basis_and_r"] = {f"{b}/r{rr}": {k: v for k, v in group([r for r in rows if r["basis"] == b and r["r"] == rr]).items()
                                            if k in ("leaves", "K_RL", "K_D", "m_D", "se", "z", "logloss_gain_RL_minus_D")}
                             for b in ("X", "Z") for rr in ("10", "13", "30", "50")}
    X, Z = res["by_basis"]["X"], res["by_basis"]["Z"]
    d = Z["m_D"] - X["m_D"]
    res["Z_minus_X"] = {"m_D": d, "se": math.hypot(X["se"], Z["se"]), "z": d / math.hypot(X["se"], Z["se"]),
                        "note": "independent leaves, so the SEs add in quadrature; one post hoc contrast"}
    with open(out_path(os.path.join(out_dir, "r2_basis.json")), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("by_basis", "Z_minus_X")}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
