"""Fig. 1 of the Willow paper (scout, 2026-10-11): every decoder's paired gap to ML(RL) on W5 at d = 3 and 5, RL priors.

Main's decision 8 (paper.md): the final figure is W5 = {10, 13, 30, 50} at both distances (Libra: its rounds
{10, 30, 50}), RL priors only, drawn from decoder's final round-1 analysis. Every value is read from
results/decoder/willow/final_round1/final_round1.json (rule A; entry 7's SE), keys
entry7_se["pooled_gaps_W5:d<d>|<decoder>.rl vs ML.rl"]; nothing is refitted here.

Panel (a): pooled relative gap g_d = mean(eps_A - eps_ML) / mean(eps_ML) with its 95% CI (normal, entry 7's SE),
  d = 3 (open) and d = 5 (filled), log axis. BeliefMatching has d = 3 only.
Panel (b): the paired Lambda ratio Lambda_A / Lambda_ML(RL) = (1 + g_3) / (1 + g_5), 95% CI on the log scale with the
  two distances' SEs combined as independent (the convention of reservoir's matched table and of
  code/scout/willow_derived_final.py). Descriptive for every decoder but Libra (main's decision 4).
Tesseract (dagger): its d = 5 rounds are {10, 13, 30} while its d = 3 value is on W5, so its ratio mixes windows
  (reservoir's matched table, on {10, 13, 30} at both distances, gives 0.9601 against 0.9605 here).
Palette and style: as fig_rec.py (dataviz reference instance, categorical slots in fixed order).
Usage: python3 -I fig_willow_gaps.py <experimental-root>/ <out-prefix> [--r10]
  ->  <out-prefix>.{png,pdf} and a printout. Without --r10 the figure is manuscript v1.1's Fig. 1, unchanged.

--r10 (added 2026-10-13 for manuscript v1.2) adds panel (c) below (a) and (b), on its own axis:
Panel (c): PR3's post hoc r = 10 follow-up (willow_prereg_r3.md section 10, amendments 10a and 10b), descriptive.
  Every decoder's paired gap to ML(RL) at r = 10 only, at d = 3 (18 leaves), d = 5 (7 leaves, rule A) and d = 7
  (X/r10 and Z/r10, 24,000 shots each), with 95% CIs (g +- 1.96 SE), log axis. Read from
  results/decoder/willow/pr3/r10_posthoc/analysis/r10_results.json, keys gaps[<decoder>][<d>]["g"] and ["ci95"];
  nothing is refitted here. One-round-count values with SPAM included: they compare across d at r = 10 only, not
  with panel (a) or (b).
"""
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root, prefix = sys.argv[1], sys.argv[2]
R10 = "--r10" in sys.argv[3:]
FRP = root + "results/decoder/willow/final_round1/final_round1.json"
FR = json.load(open(FRP))
E7 = FR["entry7_se"]
Z95 = 1.959963984540054
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
ROWS = [("libra", "Libra"), ("tesseract", "Tesseract †"), ("harmony", "Harmony"), ("corrmatch", "corr. matching"),
        ("corr_mwpm", "corr. MWPM"), ("mwpm", "MWPM"), ("beliefmatch", "BeliefMatching")]


def entry(d, a):
    k = "pooled_gaps_W5:d%d|%s.rl vs ML.rl" % (d, a)
    return E7.get(k)


rows = []
for a, lab in ROWS:
    e3, e5 = entry(3, a), entry(5, a)
    r = {"key": a, "label": lab, "g3": e3["g"], "ci3": e3["ci95"], "n3": e3["n_series"]}
    if e5 is not None:
        g3, s3, g5, s5 = e3["g"], e3["se"], e5["g"], e5["se"]
        lr = (1 + g3) / (1 + g5)
        se = math.sqrt((s3 / (1 + g3)) ** 2 + (s5 / (1 + g5)) ** 2)
        r.update(g5=g5, ci5=e5["ci95"], n5=e5["n_series"], ratio=lr,
                 ratio_ci=[lr * math.exp(-Z95 * se), lr * math.exp(Z95 * se)])
    rows.append(r)

if R10:
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1], sharey=ax)
    bx.tick_params(labelleft=False)
    cx = fig.add_subplot(gs[1, :])
else:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.65, 1.0]})
n = len(rows)
off = 0.13
for i, r in enumerate(rows):
    y = n - 1 - i
    if "g5" in r:
        ax.plot([100 * r["g3"], 100 * r["g5"]], [y + off, y - off], color=AXIS, lw=1.0, zorder=1)
    ax.errorbar(100 * r["g3"], y + off, xerr=[[100 * (r["g3"] - r["ci3"][0])], [100 * (r["ci3"][1] - r["g3"])]],
                fmt="o", ms=4.5, mfc=SURF, mec=SLOT[0], mew=1.3, ecolor=SLOT[0], elinewidth=1.1, capsize=0, zorder=3)
    if "g5" in r:
        ax.errorbar(100 * r["g5"], y - off, xerr=[[100 * (r["g5"] - r["ci5"][0])], [100 * (r["ci5"][1] - r["g5"])]],
                    fmt="o", ms=4.5, mfc=SLOT[1], mec=SLOT[1], ecolor=SLOT[1], elinewidth=1.1, capsize=0, zorder=3)
        lo, hi = r["ratio_ci"]
        bx.errorbar(r["ratio"], y, xerr=[[r["ratio"] - lo], [hi - r["ratio"]]], fmt="o", ms=4.5, mfc=INK2, mec=INK2,
                    ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
        bx.annotate("%.3f" % r["ratio"], (r["ratio"], y), xytext=(0, 5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7, color=INK2)
    else:
        bx.annotate("d = 3 only", (0.6, y), xytext=(2, 0), textcoords="offset points", va="center", fontsize=7,
                    color=MUTED)
ax.set_xscale("log")
ax.set_xlim(0.8, 160)
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_yticks(range(n))
ax.set_yticklabels([r["label"] for r in rows][::-1], color=INK)
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel("paired gap to ML(RL), % (95% CI)")
ax.grid(axis="y", visible=False)
ax.plot([], [], "o", ms=4.5, mfc=SURF, mec=SLOT[0], mew=1.3, label="d = 3 (18 series)")
ax.plot([], [], "o", ms=4.5, mfc=SLOT[1], mec=SLOT[1], label="d = 5 (rule A; 7–8 series)")
ax.legend(loc="lower right", fontsize=7, handletextpad=0.3)
ax.set_title("(a) gap per round window W5", loc="left", fontsize=8, color=INK)
bx.axvline(1.0, color=MUTED, lw=0.9, ls="--", zorder=1)
bx.set_xlim(0.6, 1.06)
bx.set_xlabel("Λ(decoder) / Λ(ML(RL)), 95% CI")
bx.grid(axis="y", visible=False)
bx.set_title("(b) paired Λ ratio", loc="left", fontsize=8, color=INK)
if R10:
    R10P = root + "results/decoder/willow/pr3/r10_posthoc/analysis/r10_results.json"
    RR = json.load(open(R10P))
    C_ROWS = [("libra.rl", "Libra"), ("harmony.rl", "Harmony"), ("harmony.si1000", "Harmony, SI1000"),
              ("corrmatch.rl", "corr. matching"), ("corrmatch.si1000", "corr. matching, SI1000"),
              ("corr_mwpm.rl", "corr. MWPM"), ("mwpm.rl", "MWPM")]
    STY = {"3": dict(mfc=SURF, mec=SLOT[0], mew=1.3, ecolor=SLOT[0]),
           "5": dict(mfc=SLOT[1], mec=SLOT[1], ecolor=SLOT[1]),
           "7": dict(mfc=SLOT[2], mec=SLOT[2], ecolor=SLOT[2])}
    nc = len(C_ROWS)
    offc = 0.22
    crows = []
    for i, (k, lab) in enumerate(C_ROWS):
        y = nc - 1 - i
        G = RR["gaps"][k]
        pts = [(d, G[d]["g"], G[d]["ci95"], G[d]["n_series"]) for d in ("3", "5", "7")]
        ys = [y + offc, y, y - offc]
        cx.plot([100 * p[1] for p in pts], ys, color=AXIS, lw=1.0, zorder=1)
        for (d, g, ci, ns), yy in zip(pts, ys):
            cx.errorbar(100 * g, yy, xerr=[[100 * (g - ci[0])], [100 * (ci[1] - g)]], fmt="o", ms=4.0,
                        elinewidth=1.1, capsize=0, zorder=3, **STY[d])
        crows.append((lab, pts, RR["growth"][k]))
    cx.set_xscale("log")
    cx.set_xlim(0.8, 200)
    cx.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    cx.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    cx.set_yticks(range(nc))
    cx.set_yticklabels([lab for _, lab in C_ROWS][::-1], color=INK)
    cx.set_ylim(-0.6, nc - 0.4)
    cx.set_xlabel("paired gap to ML(RL) at r = 10 only, SPAM included, % (95% CI)")
    cx.grid(axis="y", visible=False)
    cx.plot([], [], "o", ms=4.0, mfc=SURF, mec=SLOT[0], mew=1.3, label="d = 3 (18 leaves)")
    cx.plot([], [], "o", ms=4.0, mfc=SLOT[1], mec=SLOT[1], label="d = 5 (7 leaves; rule A)")
    cx.plot([], [], "o", ms=4.0, mfc=SLOT[2], mec=SLOT[2], label="d = 7 (2 leaves: X and Z)")
    cx.legend(loc="upper right", fontsize=7, handletextpad=0.3)
    cx.set_title("(c) post hoc, descriptive: r = 10 only, d = 3, 5 and 7 (not comparable with (a) or (b))", loc="left",
                 fontsize=8, color=INK)
fig.tight_layout(w_pad=1.2)
fig.savefig(prefix + ".png", dpi=220)
fig.savefig(prefix + ".pdf", metadata={"CreationDate": None})
print("source", FRP.replace(root, ""), FR["made"], "rule", FR["rule"], "k2", FR["k2"]["verdicts"])
for r in rows:
    print("%-15s g3 %6.2f%% [%6.2f, %6.2f] (%d)" % (r["label"], 100 * r["g3"], 100 * r["ci3"][0], 100 * r["ci3"][1],
                                                  r["n3"]), end="")
    if "g5" in r:
        print("  g5 %7.2f%% [%6.2f, %6.2f] (%d)  ratio %.4f [%.4f, %.4f]" % (
            100 * r["g5"], 100 * r["ci5"][0], 100 * r["ci5"][1], r["n5"], r["ratio"], *r["ratio_ci"]))
    else:
        print("  (d = 3 only)")
if R10:
    print("panel (c): source", R10P.replace(root, ""), RR["made"], "all_checks_ok", RR["all_checks_ok"])
    for lab, pts, gr in crows:
        print("%-23s" % lab + "  ".join("g%s %7.2f%% [%6.2f, %6.2f] (%d)" % (d, 100 * g, 100 * ci[0], 100 * ci[1], ns)
                                         for d, g, ci, ns in pts),
              "  g5/g3 %.2f [%.2f, %.2f]  g7/g5 %.2f [%.2f, %.2f]" % (
                  gr["g5/g3"]["ratio"], *gr["g5/g3"]["ci95"], gr["g7/g5"]["ratio"], *gr["g7/g5"]["ci95"]))
