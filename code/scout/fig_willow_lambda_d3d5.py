"""Fig. 2 of the Willow paper, v2.0 (scout, 2026-10-13): mean logical error per cycle against code distance at
d = 3 and 5 only, for ML(RL) and Libra (RL), with the NN's reported values for context.

Why a new figure: the referee review of v1.1 (point 3.3) asked that Fig. 2 drop the exploratory d = 7 probe point and
the fit through d = 3, 5 and 7, and use the d = 3/5 estimate in the legend. v1.x's Fig. 2 was drawn by decoder's
analysis script from the frozen analysis view (ML's legend: the exploratory three-distance fit). This script draws
d = 3 and 5 only, from decoder's final round-1 analysis under rule A, and reads every value; nothing is refitted.

Inputs (read only):
  results/decoder/willow/final_round1/final_round1.json, key decisions.lambda_3_5.<entity> for ML.rl and libra.rl:
    eps["3"], eps["5"] = [mean eps over series, SE] on W5 (Libra: its round counts {10, 30, 50}); Lambda, se = the
    pre-registered Lambda_3/5 = eps_3 / eps_5 with its SE (rule A: ML(RL)'s d = 5 fits leave the 7 leaves that still
    fail K2; Libra's own fits keep every leaf).
  code/decoder/willow_reported.json, key table_S1_fig1d.nn: the NN's reported eps at d = 3 and 5 [Ref. 1, Table S1],
    over all round counts; unpaired with ours.
Error bars: 95% CIs, eps +- 1.96 SE (as v1.x's Fig. 2).
Palette and style: as fig_willow_gaps.py (categorical slots in fixed order; validated with the dataviz checks).
Usage: python3 -I fig_willow_lambda_d3d5.py <experimental-root>/ <out-prefix>  ->  <out-prefix>.{png,pdf} and a printout.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root, prefix = sys.argv[1], sys.argv[2]
FRP = root + "results/decoder/willow/final_round1/final_round1.json"
REPP = root + "code/decoder/willow_reported.json"
FR = json.load(open(FRP))
LAM = FR["decisions"]["lambda_3_5"]
NN = json.load(open(REPP))["table_S1_fig1d"]["nn"]
Z95 = 1.959963984540054
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

SERIES = [("ML.rl", "ML(RL), ours", "o", SLOT[0], -0.07, "-"),
          ("libra.rl", "Libra (RL), shipped", "s", SLOT[1], +0.07, "-")]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
rows = []
for key, lab, mk, col, dx, ls in SERIES:
    v = LAM[key]
    assert v["distances"] == [3, 5], v["distances"]
    ds = [3, 5]
    y = [v["eps"][str(d)][0] for d in ds]
    e = [Z95 * v["eps"][str(d)][1] for d in ds]
    ax.errorbar([d + dx for d in ds], y, yerr=e, fmt=mk + ls, ms=5.5, lw=1.6, color=col, mec=SURF, mew=0.8,
                elinewidth=1.1, capsize=0, zorder=3,
                label="%s: Λ$_{3/5}$ = %.2f ± %.2f" % (lab, v["Lambda"], v["se"]))
    rows.append((lab, v))
yn = [NN["eps3"][0], NN["eps5"][0]]
en = [Z95 * NN["eps3"][1], Z95 * NN["eps5"][1]]
ax.errorbar([3, 5], yn, yerr=en, fmt="D--", ms=4.5, lw=1.0, color=SLOT[2], mec=SURF, mew=0.6, elinewidth=1.0,
            capsize=0, zorder=2, label="NN, reported [1] (all r; unpaired)")
ax.set_yscale("log")
ax.set_xlim(2.5, 5.5)
ax.set_xticks([3, 5])
ax.set_xticklabels(["3", "5"])
ax.set_ylim(2.4e-3, 9e-3)
ax.set_yticks([3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3])
ax.set_yticklabels(["3", "4", "5", "6", "7", "8"])
ax.minorticks_off()
ax.set_xlabel("code distance d")
ax.set_ylabel("mean error per cycle, 10$^{-3}$ (95% CI)")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left", fontsize=7, handletextpad=0.4)
ax.set_title("Error per cycle at d = 3 and 5 (W5; rule A)", loc="left", fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(prefix + ".png", dpi=220)
fig.savefig(prefix + ".pdf", metadata={"CreationDate": None})
print("source", FRP.replace(root, ""), FR["made"], "rule", FR["rule"], "k2", FR["k2"]["verdicts"])
for lab, v in rows:
    print("%-22s eps3 %.4e (%.1e)  eps5 %.4e (%.1e)  Lambda_3/5 %.4f +- %.4f  n_series %s" % (
        lab, v["eps"]["3"][0], v["eps"]["3"][1], v["eps"]["5"][0], v["eps"]["5"][1], v["Lambda"], v["se"],
        v["n_series"]))
print("NN reported: eps3 %s  eps5 %s  (source %s, %s)" % (NN["eps3"], NN["eps5"], REPP.replace(root, ""),
                                                          "table_S1_fig1d.nn"))
