"""R5, step 2: compare reservoir's independent reproduction (r5_worstcase.json, made without reading D2's script or
outputs) with decoder's D2 outputs (d2_worstcase.json), number by number. POST HOC; read-only.

reservoir, 2026-10-13. Everything D2's JSON reports that R5 also computes is compared: the 12 published rows (g_5, its
SE and CI, the count gap, the expected flips), the pooled and per-round rates with their bounds and unrun shares, the
C weights, the extremes of the pooled and per-round boxes (values and the vertices that attain them), the estimate under
other combinations of the strata, D's flips alone, and the change of g_5 per added ML error.
Usage: python3 -I -B r5_compare_d2.py R5_JSON D2_JSON OUT_DIR -> OUT_DIR/r5_vs_d2.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.outpaths import out_path  # noqa: E402

ROWS = ("base", "estimate", "worst_lower_g", "worst_higher_g", "worst_lower_g_with_tau3_only",
        "worst_higher_g_with_tau3_only")
TYPES = ("D10", "D01", "C00", "C11")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def vertex_d2(v):
    """D2's vertex names ("10:t3_C00": "hi", "all:C11": "lo") in R5's form ("r10|t3C00": "up", "C11": "lo")."""
    out = {}
    for k, w in v.items():
        where, key = k.split(":")
        key = key.replace("t3_", "t3")
        out[key if where == "all" else f"r{where}|{key}"] = "up" if w == "hi" else w
    return out


def main(r5_json, d2_json, out_dir):
    R = json.load(open(r5_json))
    D = json.load(open(d2_json))
    cmp, vert = [], []

    def add(group, name, mine, theirs):
        cmp.append({"group": group, "quantity": name, "r5": mine, "d2": theirs, "abs_diff": abs(mine - theirs)})

    for rule in ("A", "none"):
        for row in ROWS:
            m, d = R["rows"][f"{rule}|{row}"], D["rules"][rule]["rows"][row]
            q = f"{rule}|{row}"
            add("published rows", q + "|g5", m["g5"], d["g5"]["g"])
            add("published rows", q + "|se", m["se"], d["g5"]["se"])
            add("published rows", q + "|ci95_lo", m["ci95"][0], d["g5"]["ci95"][0])
            add("published rows", q + "|ci95_hi", m["ci95"][1], d["g5"]["ci95"][1])
            add("published rows", q + "|count_gap", m["count_gap"], d["count_gap"]["rel"])
            add("published rows", q + "|count_gap_errors", m["count_gap_errors"], d["count_gap"]["gap_errors"])
            add("published rows", q + "|errors_ML", m["k_ml"], d["count_gap"]["errors_ML"])
            for t in TYPES:
                add("published rows", q + f"|flips_{t}", m.get("flips_total", {}).get(t, 0.0), d["flips_total"][t])
        for box, mk, dk in (("pooled", f"C_only|{rule}", "without_tau3_only"),
                            ("pooled", f"with_tau3_only|unrun_C_pooled_share|{rule}", "with_tau3_only"),
                            ("by_round", f"C_only|each95|{rule}", "without_tau3_only"),
                            ("by_round", f"with_tau3_only|unrun_C_round_share|each95|{rule}", "with_tau3_only")):
            m = R["vertex_extremes_pooled"][mk] if box == "pooled" else R["per_round_boxes"][mk]
            d = D["rules"][rule][f"box_{box}"][dk]
            for s in ("g5_min", "g5_max", "count_gap_min", "count_gap_max"):
                add(f"box extremes ({box})", f"{rule}|{dk}|{s}", m[s], d[s]["value"])
            for s, ms in (("g5_min", "argmin_g5"), ("g5_max", "argmax_g5")):
                mine = {k: v for k, v in m[ms].items()}
                theirs = vertex_d2(d[s]["vertex"])
                same = all(mine.get(k) == v for k, v in theirs.items())
                vert.append({"box": f"{rule}|{box}|{dk}|{s}", "same_vertex": same, "d2_method": d.get("method"),
                             "r5_method": f"exhaustive enumeration of {m.get('vertices', m.get('points'))} vertices"})
        e = R["estimate_by_combination"][rule]
        for name, mk, dk in (("horvitz_thompson", "per_leaf_horvitz_thompson", "estimate_leaf_weighted"),
                             ("per_round", "per_type_per_round", "estimate_by_round"),
                             ("d_flips_only", "d_flips_only", "D_flips_only")):
            add("other combinations", f"{rule}|{name}|g5", e[mk]["g5"], D["rules"][rule][dk]["g5"]["g"])
            add("other combinations", f"{rule}|{name}|count_gap", e[mk]["count_gap"],
                D["rules"][rule][dk]["count_gap"]["rel"])
        for r, v in D["rules"][rule]["D_flips_only"]["net_ML_errors_by_round"].items():
            add("other combinations", f"{rule}|d_flips_net_ml_errors|r{r}", e["d_flips_net_ml_errors_by_round"][r], v)
        fd, dd = R["dg5_per_added_ml_error"][f"h10|{rule}"], D["rules"][rule]["dg5_per_added_ML_error"]
        for r in ("10", "30", "50"):
            add("dg5 per added ML error", f"{rule}|r{r}|mean", fd[r]["mean_points"] / 100, dd["mean_by_round"][r])
            add("dg5 per added ML error", f"{rule}|r{r}|min", fd[r]["min_points"] / 100, dd["range_by_round"][r][0])
            add("dg5 per added ML error", f"{rule}|r{r}|max", fd[r]["max_points"] / 100, dd["range_by_round"][r][1])
    P, DP = R["pooled_rates_and_cp95"], D["pooled"]
    for mk, dk in (("C00", "C00"), ("C11", "C11"), ("t3D", "t3_D"), ("t3C", "t3_C")):
        add("rates", f"pooled|{dk}|k", P[mk]["flips"], DP[dk]["k"])
        add("rates", f"pooled|{dk}|n", P[mk]["n"], DP[dk]["n"])
        add("rates", f"pooled|{dk}|upper95", P[mk]["up"], DP[dk]["upper95"])
        if "lower95" in DP[dk]:
            add("rates", f"pooled|{dk}|lower95", P[mk]["lo"], DP[dk]["lower95"])
        if "unrun_share" in DP[dk]:
            add("rates", f"pooled|{dk}|unrun_share", R["C_unrun_share"]["pooled"][mk], DP[dk]["unrun_share"])
    for r in ("10", "30", "50"):
        PR, DR = R["per_round_rates_cp95"][r], D["by_round"][r]
        for mk, dk in (("C00", "C00"), ("C11", "C11"), ("t3D", "t3_D"), ("t3C", "t3_C")):
            add("rates", f"r{r}|{dk}|k", PR[mk]["flips"], DR[dk]["k"])
            add("rates", f"r{r}|{dk}|n", PR[mk]["n"], DR[dk]["n"])
            add("rates", f"r{r}|{dk}|upper95", PR[mk]["up"], DR[dk]["upper95"])
            if "lower95" in DR[dk]:
                add("rates", f"r{r}|{dk}|lower95", PR[mk]["lo"], DR[dk]["lower95"])
            if "unrun_share" in DR[dk]:
                add("rates", f"r{r}|{dk}|unrun_share", R["C_unrun_share"]["by_round"][r][mk], DR[dk]["unrun_share"])
    add("rates", "C_weight_min", R["C_weight_range"][0], D["strata"]["C"]["weight_range"][0])
    add("rates", "C_weight_max", R["C_weight_range"][1], D["strata"]["C"]["weight_range"][1])
    # the normal quantile behind each side's 95% CI, implied by its g, SE and CI
    z = {"r5": sorted({round((R["rows"][f"{u}|{w}"]["ci95"][1] - R["rows"][f"{u}|{w}"]["g5"]) /
                             R["rows"][f"{u}|{w}"]["se"], 9) for u in ("A", "none") for w in ROWS}),
         "d2": sorted({round((D["rules"][u]["rows"][w]["g5"]["ci95"][1] - D["rules"][u]["rows"][w]["g5"]["g"]) /
                             D["rules"][u]["rows"][w]["g5"]["se"], 9) for u in ("A", "none") for w in ROWS})}
    groups = {}
    for c in cmp:
        g = groups.setdefault(c["group"], {"quantities": 0, "max_abs_diff": 0.0})
        g["quantities"] += 1
        g["max_abs_diff"] = max(g["max_abs_diff"], c["abs_diff"])
    res = {"label": "reservoir, revision R5: R5 (independent) against D2, number by number. POST HOC; read-only",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "inputs_md5": {"r5_worstcase.json": md5(r5_json), "d2_worstcase.json": md5(d2_json)},
           "d2_script_md5_as_reported": D.get("script_md5"), "summary_by_group": groups, "implied_z95": z,
           "notes": ["R5's per-error change of g_5 is compared at a step of 10 errors (divided by 10), the step that "
                     "reproduces D2's table exactly; at a step of 1 error the values differ by at most 3e-6 "
                     "(0.0003 points) per leaf",
                     "CI differences, if any, follow from the normal quantile each side uses (implied_z95)"],
           "quantities": len(cmp), "max_abs_diff": max(c["abs_diff"] for c in cmp),
           "over_1e-9": [c for c in cmp if c["abs_diff"] > 1e-9], "extreme_vertices": vert,
           "extreme_vertices_all_same": all(v["same_vertex"] for v in vert), "comparisons": cmp}
    with open(out_path(os.path.join(out_dir, "r5_vs_d2.json")), "w") as f:
        json.dump(res, f, indent=1)
    for g, v in groups.items():
        print(f"{g:28s} {v['quantities']:4d} quantities, max |diff| {v['max_abs_diff']:.2e}")
    print(f"extreme vertices identical: {res['extreme_vertices_all_same']} ({len(vert)} compared); "
          f"over 1e-9: {len(res['over_1e-9'])}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
