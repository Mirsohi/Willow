"""Scout: post hoc sensitivity of round 2's T-hat_mech (exploratory mech fits, amendment A1) to PR2 section 8.2's
exclusion of the 6 series whose fits stopped at the 25-step cap. Descriptive arithmetic, no decoding.

Input: reservoir's per-leaf cells, results/reservoir/willow_r2check/r2m_leaves.csv (RB 2026-10-11 20:41Z).
For each leaf, cells_RL_vs_<fit> holds the paired counts n, n10 (RL wrong, the fit right), n01 (RL right, the fit
wrong) and n11 (both wrong).
m = K_RL / K_fit - 1, with K_RL = n10 + n11 and K_fit = n01 + n11 pooled over the leaves; the SE is the delta
method on the per-shot values a - c b (c = K_RL / K_fit), summed over leaves (shots independent between leaves).
It reproduces explore_analysis.json's held-out m for T-hat_mech (12 series) and D-hat_mech (18 series).

Usage: python3 -I willow_tm_posthoc.py <r2m_leaves.csv> <out.json>
"""
import csv
import hashlib
import json
import math
import sys

CAPPED = {"q4_5|X", "q4_9|Z", "q6_11|Z", "q6_3|Z", "q6_7|X", "q8_5|X"}
Z95 = 1.959964


def stat(rows, fam):
    k_rl = k_x = 0
    cells = []
    for r in rows:
        c = json.loads(r["cells_RL_vs_" + fam])
        n, n10, n01, n11 = c["n"], c["n10"], c["n01"], c["n11"]
        k_rl += n10 + n11
        k_x += n01 + n11
        cells.append((n, n10, n01, n11))
    cc = k_rl / k_x
    var = 0.0
    for n, n10, n01, n11 in cells:
        e1 = (n10 * 1 + n01 * (-cc) + n11 * (1 - cc)) / n
        e2 = (n10 * 1 + n01 * cc * cc + n11 * (1 - cc) ** 2) / n
        var += n * (e2 - e1 * e1)
    m = cc - 1
    se = math.sqrt(var) / k_x
    return {"m": m, "se": se, "ci95": [m - Z95 * se, m + Z95 * se], "leaves": len(rows),
            "errors_RL": k_rl, "errors_fit": k_x}


def main():
    path, out = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(path)))
    key = lambda r: r["patch"] + "|" + r["basis"]
    pre = [r for r in rows if key(r) not in CAPPED]
    cap = [r for r in rows if key(r) in CAPPED]
    res = {
        "label": "scout, post hoc sensitivity of T-hat_mech to PR2 section 8.2's exclusion (exploratory fits; "
                 "descriptive arithmetic)",
        "input": {"path": path, "md5": hashlib.md5(open(path, "rb").read()).hexdigest()},
        "capped_series": sorted(CAPPED),
        "Tm_12_converged": stat(pre, "Tm"),
        "Tm_12_converged_without_r13": stat([r for r in pre if r["r"] != "13"], "Tm"),
        "Tm_all_18": stat(rows, "Tm"),
        "Tm_all_18_without_r13": stat([r for r in rows if r["r"] != "13"], "Tm"),
        "Tm_6_capped": stat(cap, "Tm"),
        "Tm_per_capped_series": {s: stat([r for r in rows if key(r) == s], "Tm") for s in sorted(CAPPED)},
        "Dm_all_18": stat(rows, "Dm"),
    }
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for k in ("Tm_12_converged", "Tm_all_18", "Tm_all_18_without_r13", "Tm_6_capped", "Dm_all_18"):
        v = res[k]
        print("%-24s m %+.3f%% [%+.3f, %+.3f] leaves %d" % (k, 100 * v["m"], 100 * v["ci95"][0],
                                                          100 * v["ci95"][1], v["leaves"]))
    for s, v in res["Tm_per_capped_series"].items():
        print("  %-10s m %+.2f%% [%+.2f, %+.2f]" % (s, 100 * v["m"], 100 * v["ci95"][0], 100 * v["ci95"][1]))


if __name__ == "__main__":
    main()
