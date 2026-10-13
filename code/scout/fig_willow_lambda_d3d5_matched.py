"""Fig. 2 of the Willow paper, v2.0 draft 3 (scout, 2026-10-13): mean logical error per cycle at d = 3 and 5 for
ML(RL) and Libra on matched inputs, with the NN's reported values for context.

Why a new figure: the fresh-eyes reading of draft 2 (point B1) found that draft 2's Fig. 2
(fig_willow_lambda_d3d5.py, 79fddc6f9cbb62677a27fb5e7fbe0700) drew the two decoders on different inputs: Libra on its
round counts {10, 30, 50} with every leaf (8 series at d = 5), ML(RL) on W5 = {10, 13, 30, 50} under rule A, which keeps
the two-point outlier series q6_5 X. Its legend's Lambda pair (2.14 and 2.08) therefore divided to 0.975, which the
text itself says not to do. This figure draws both decoders on the same shots: the 18 d = 3 series and the 7 d = 5
series of Libra's paired gap, on r = 10, 30 and 50, with rule A at d = 5 (the leaves that still fail K2 leave both
decoders' fits). Each point is the decoder's own (unpaired) fit, the mean eps over series; the legend gives each
decoder's Lambda_3/5 = eps3/eps5 on these inputs (Table S8 gives their SEs), and the panel gives the paired ratio,
which carries the comparison's uncertainty because the two decoders share their shots.

Inputs (read only):
  results/scout/revision/willow_v2d3_derived.json, key matched_fig2.A: eps and SE per decoder and distance
    (SE = sqrt(sum of the series' squared SEs)/n, the final analysis's convention), Lambda_ml and Lambda_libra;
    and matched_fig2.A.paired_ratio (copied there from final_round1.json decisions.d5_rules.A).
  code/decoder/willow_reported.json, key table_S1_fig1d.nn: the NN's reported eps at d = 3 and 5 [Ref. 1, Table S1],
    over all round counts; unpaired with ours.
Error bars: 95% CIs, eps +- 1.96 SE. Palette and style: as fig_willow_lambda_d3d5.py.
Usage: python3 -I fig_willow_lambda_d3d5_matched.py <experimental-root>/ <derived.json> <out-prefix>
       -> <out-prefix>.{png,pdf} and a printout.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root, derived, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
REPP = root + "code/decoder/willow_reported.json"
M = json.load(open(derived))["matched_fig2"]["A"]
NN = json.load(open(REPP))["table_S1_fig1d"]["nn"]
PR = M["paired_ratio"]
Z95 = 1.959963984540054
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
assert M["3"]["series"] == 18 and M["5"]["series"] == 7, (M["3"]["series"], M["5"]["series"])
SERIES = [("ml", "ML(RL), ours", "o", SLOT[0], -0.07), ("libra", "Libra, shipped", "s", SLOT[1], +0.07)]
fig, ax = plt.subplots(figsize=(4.4, 3.3))
for who, lab, mk, col, dx in SERIES:
    y = [M["3"]["eps_" + who], M["5"]["eps_" + who]]
    e = [Z95 * M["3"]["se_" + who], Z95 * M["5"]["se_" + who]]
    ax.errorbar([3 + dx, 5 + dx], y, yerr=e, fmt=mk + "-", ms=5.5, lw=1.6, color=col, mec=SURF, mew=0.8,
                elinewidth=1.1, capsize=0, zorder=3, label="%s: Λ$_{3/5}$ = %.2f" % (lab, M["Lambda_" + who]))
yn = [NN["eps3"][0], NN["eps5"][0]]
en = [Z95 * NN["eps3"][1], Z95 * NN["eps5"][1]]
ax.errorbar([3, 5], yn, yerr=en, fmt="D--", ms=4.5, lw=1.0, color=SLOT[2], mec=SURF, mew=0.6, elinewidth=1.0,
            capsize=0, zorder=2, label="NN, reported [1] (all r; unpaired)")
ax.annotate("d = 5 ML is approximate (§5.1.4)", xy=(5 - 0.07, M["5"]["eps_ml"]), xytext=(3.25, 2.6e-3),
            fontsize=7, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
ax.text(0.98, 0.97, "paired Λ$_{Libra}$/Λ$_{ML}$ = %.3f [%.3f, %.3f]" % (PR["ratio"], PR["ci95"][0], PR["ci95"][1]),
        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
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
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=7, handletextpad=0.4)
ax.set_title("Error per cycle, ML(RL) and Libra on the same shots\n(r = 10, 30, 50; d = 5 under rule A)", loc="left",
             fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(prefix + ".png", dpi=220)
fig.savefig(prefix + ".pdf", metadata={"CreationDate": None})
for who, lab, *_ in SERIES:
    print("%-16s eps3 %.4e (SE %.2e)  eps5 %.4e (SE %.2e)  Lambda_3/5 %.4f +- %.4f" % (
        lab, M["3"]["eps_" + who], M["3"]["se_" + who], M["5"]["eps_" + who], M["5"]["se_" + who],
        M["Lambda_" + who], M["Lambda_" + who + "_se"]))
print("series: d3 %d, d5 %d (%s)" % (M["3"]["series"], M["5"]["series"], ", ".join(M["5"]["series_names"])))
print("paired ratio %.4f [%.4f, %.4f]" % (PR["ratio"], PR["ci95"][0], PR["ci95"][1]))
print("NN reported: eps3 %s  eps5 %s  (source %s, table_S1_fig1d.nn)" % (NN["eps3"], NN["eps5"],
                                                                        REPP.replace(root, "")))
