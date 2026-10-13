"""D4: corrected standard errors for manuscript Table 13 (d = 3, every r >= 10) and the cross-prior pairs of the same
section, from the final analysis (decoder, 2026-10-13; post hoc revision, read-only).
Frozen-formula values: final_round1/analysis/analysis.json "pooled_gaps" (rel, se_rel, ci95; what v1.1 prints).
Corrected values: final_round1.json "entry7_se" ("pooled_gaps:<key>": g, se = entry 7's full delta-method SE with the
covariance of the two slope fits, Birge-inflated as the frozen formula is; ci95 = g +- 1.96 se; z, p).
usage: python3 -I rev_d4_table13.py --final <final_round1.json> --analysis <analysis.json> --out <dir>
"""
import argparse, hashlib, json, os, time

ROWS = [  # (row label as in Table 13, key)
    ("Libra (shipped), RL", "d3|libra.rl vs ML.rl"),
    ("Tesseract (ours), RL (every 5th shot, r in {10, 13, 30, 50})", "d3|tesseract.rl vs ML.rl"),
    ("Harmony (shipped), RL", "d3|harmony.rl vs ML.rl"),
    ("corr. matching (shipped), RL", "d3|corrmatch.rl vs ML.rl"),
    ("BeliefMatching (ours), RL (every 5th shot, r in {10, 13, 30, 50})", "d3|beliefmatch.rl vs ML.rl"),
    ("corr. MWPM (ours), RL", "d3|corr_mwpm.rl vs ML.rl"),
    ("MWPM (ours), RL", "d3|mwpm.rl vs ML.rl"),
    ("ML (exact), SI1000 against ML(RL) (P2)", "d3|ML.si1000 vs ML.rl"),
    ("Tesseract (ours), SI1000 (every 5th shot, r in {10, 13, 30, 50})", "d3|tesseract.si1000 vs ML.si1000"),
    ("Harmony (shipped), SI1000", "d3|harmony.si1000 vs ML.si1000"),
    ("BeliefMatching (ours), SI1000 (every 5th shot, r in {10, 13, 30, 50})", "d3|beliefmatch.si1000 vs ML.si1000"),
    ("corr. matching (shipped), SI1000", "d3|corrmatch.si1000 vs ML.si1000"),
    ("corr. MWPM (ours), SI1000", "d3|corr_mwpm.si1000 vs ML.si1000"),
    ("MWPM (ours), SI1000", "d3|mwpm.si1000 vs ML.si1000"),
]
CROSS = [  # section 5.2, "Prior against inference" (pairs across priors)
    ("Libra (RL) against ML (SI1000)", "d3|libra.rl vs ML.si1000"),
    ("Harmony (RL) against ML (SI1000)", "d3|harmony.rl vs ML.si1000"),
    ("corr. matching (RL) against ML (SI1000)", "d3|corrmatch.rl vs ML.si1000"),
    ("Tesseract (RL) against ML (SI1000)", "d3|tesseract.rl vs ML.si1000"),
]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def row(label, key, A, F):
    a = A["pooled_gaps"][key]
    e = F["entry7_se"]["pooled_gaps:" + key]
    assert abs(a["rel"] - e["g"]) < 1e-12, key
    return dict(row=label, key=key, n_series=e["n_series"], g_pct=100 * e["g"],
                frozen_se_pct=100 * a["se_rel"], frozen_ci95_pct=[100 * v for v in a["ci95"]], frozen_p=a["p"],
                corrected_se_pct=100 * e["se"], corrected_ci95_pct=[100 * v for v in e["ci95"]],
                corrected_z=e["z"], corrected_p=e["p"], se_ratio=e["se"] / a["se_rel"])


def fmt(r):
    f = lambda v: f"{v:+.2f}"
    return (f"| {r['row']} | {f(r['g_pct'])} | {r['frozen_se_pct']:.3f} ({f(r['frozen_ci95_pct'][0])}, "
            f"{f(r['frozen_ci95_pct'][1])}); p {r['frozen_p']:.2g} | **{r['corrected_se_pct']:.3f} "
            f"({f(r['corrected_ci95_pct'][0])}, {f(r['corrected_ci95_pct'][1])}); p {r['corrected_p']:.2g}** | "
            f"{r['se_ratio']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F, A = json.load(open(a.final)), json.load(open(a.analysis))
    assert A["windows"]["eps"]["3"] == "r >= 10"
    rows = [row(l, k, A, F) for l, k in ROWS]
    cross = [row(l, k, A, F) for l, k in CROSS]
    o = dict(made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_md5=md5(os.path.abspath(__file__)),
             inputs_md5={"final_round1.json": md5(a.final), "analysis.json": md5(a.analysis)},
             what="Table 13 gaps (d = 3, every r >= 10): frozen-formula SE as printed in v1.1 against entry 7's corrected SE "
                  "used in the final analysis; and the cross-prior pairs of section 5.2",
             table13=rows, cross_prior=cross)
    os.makedirs(a.out, exist_ok=True)
    json.dump(o, open(os.path.join(a.out, "table13_corrected_se.json"), "w"), indent=1)
    hdr = ("| row | g (%) | frozen formula: SE (%), 95% CI; p | corrected (final analysis): SE (%), 95% CI; p | SE ratio |\n"
           "|---|---|---|---|---|")
    md = [f"# Table 13: corrected standard errors (decoder, D4)", "",
          f"Made {o['made']} by `rev_d4_table13.py` (md5 {o['script_md5']}). Inputs: `final_round1.json` "
          f"{o['inputs_md5']['final_round1.json']}, `final_round1/analysis/analysis.json` {o['inputs_md5']['analysis.json']}.",
          "", "d = 3, every r >= 10, 18 series per row. g is the paired gap to ML with the same prior, in per cent (positive: "
          "ML better). Corrected SE: entry 7's full delta method with the covariance of the two slope fits, Birge-inflated "
          "like the frozen formula (`final_round1.json` `entry7_se`); 95% CI = g ± 1.96 SE. v1.1's Table 13 prints the "
          "frozen formula in every gap cell except Libra's (P1), which already has the corrected CI; its P2 cell has the "
          "frozen CI. p is two-sided (z = g/SE). A p printed as 0 is below 1e-300.", "", hdr]
    md += [fmt(r) for r in rows]
    md += ["", "Section 5.2, \"Prior against inference\" (pairs across priors, same shots):", "", hdr]
    md += [fmt(r) for r in cross]
    open(os.path.join(a.out, "table13_corrected_se.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
