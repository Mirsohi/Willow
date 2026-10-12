"""Methods-validation figures for the Willow ML paper (static PNG + PDF), from results/decoder/{mismatch,fit,speed}.
Usage: python3 make_figures.py [--res <results/decoder>] [--out <results/decoder/figures>]
Palette: categorical slots blue, orange, aqua (+ violet for a fourth statistic) of the validated reference palette
(all-pairs CVD and normal-vision checks pass for these four), with distinct marker shapes as the secondary channel;
ink-coloured text; hairline grid.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
SCEN = ["S0", "S1", "S1b", "S2", "S3a", "S3b", "S4", "S5"]
LABEL = {"S0": "S0  control (T = M)", "S1": "S1  all rates x1.5", "S1b": "S1b  all rates x0.7",
         "S2": "S2  2q x0.7, meas. x3", "S3a": "S3a  graphlike (matching) prior", "S3b": "S3b  + XX/ZZ crosstalk",
         "S4": "S4  + leakage-like bursts", "S5": "S5  shot-to-shot drift", "S3b+xt": "S3b  4-param. fit incl. crosstalk"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.8, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
                         "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def paired_se(b, c, n):
    """SE of (LER_a - LER_b) from discordant counts b (a wrong, b right), c (a right, b wrong)."""
    return np.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n


def fig_mismatch(res, out):
    S = [s for s in SCEN if os.path.exists(os.path.join(res, "mismatch", s, "summary.json"))]
    if not S:
        return
    R = {s: json.load(open(os.path.join(res, "mismatch", s, "summary.json"))) for s in S}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(S))[::-1]
    # (a) excess LER over ML under the truth
    ax = axes[0]
    series = [("ML(M)", "exact ML, model M", BLUE, "o", -0.2), ("tesseract(M)", "Tesseract, model M", AQUA, "s", 0.0),
              ("corr_mwpm(M)", "correlated MWPM, model M", ORANGE, "D", 0.2)]
    for key, lab, col, mk, dy in series:
        xs, es = [], []
        for s in S:
            d = R[s]["decoders"]; n = R[s]["shots"]; lt = d["ML(T)"]["LER"]
            pv = d[key]["paired_vs_ML(T)"]
            xs.append(100 * (d[key]["LER"] - lt) / lt)
            es.append(100 * 2 * paired_se(pv["a_wrong_b_right"], pv["a_right_b_wrong"], n) / lt)
        ax.errorbar(xs, y + dy, xerr=es, fmt=mk, color=col, ms=6, mec="white", mew=1.0, elinewidth=1.2, capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[s] for s in S])
    ax.set_xlabel("excess logical error rate over exact ML under the truth T (%)")
    ax.set_title("(a) cost of mismatch vs decoder suboptimality", loc="left", fontsize=9.5, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    # (b) calibration ratio observed/predicted LER, model M vs truth T
    ax = axes[1]
    for key, lab, col, mk, dy in [("calibration_ML(M)", "predicted by model M", BLUE, "o", 0.12),
                                  ("calibration_ML(T)", "predicted by truth T (control)", INK2, "o", -0.12)]:
        xs, lo, hi = [], [], []
        for s in S:
            c = R[s][key]
            r = c["ratio_obs_over_pred"]
            # 2-sigma band of the ratio from the H0 standard deviation of the observed LER, sqrt(sum m(1-m))/n
            s_pred = (c["LER_observed"] - c["LER_predicted"]) / c["z_LER"] if c["z_LER"] else np.nan
            xs.append(r); lo.append(r - 2 * s_pred / c["LER_predicted"]); hi.append(r + 2 * s_pred / c["LER_predicted"])
        xs, lo, hi = map(np.array, (xs, lo, hi))
        ax.errorbar(xs, y + dy, xerr=[xs - lo, hi - xs], fmt=mk, color=col, ms=6 if col == BLUE else 4.5, mec="white", mew=1.0,
                    elinewidth=1.2, capsize=0, label=lab)
    ax.axvline(1, color=INK2, lw=0.8)
    for s, yy in zip(S, y):
        z = R[s]["syndrome_loglik_M"]["z"]
        ax.text(1.36, yy, f"{z:+.1f}", va="center", ha="right", fontsize=8, color=INK)
    ax.text(1.36, y[0] + 0.75, "synd. log-lik. z", ha="right", fontsize=8, color=INK2)
    ax.set_xlim(0.62, 1.38)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("observed / predicted logical error rate (2 sigma)")
    ax.set_title("(b) calibration of the decoding model", loc="left", fontsize=9.5, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, handletextpad=0.3)
    n = R[S[0]]["shots"]
    fig.suptitle(f"Synthetic ground truth: d = 3, {R[S[0]]['rounds']} rounds, model M = SI1000({R[S[0]]['p']:g}), {n:,} shots per scenario",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "fig_mismatch_cost_calibration")


def fig_detection(res, out):
    S = [s for s in SCEN if os.path.exists(os.path.join(res, "mismatch", s, "summary.json"))]
    if not S:
        return
    R = {s: json.load(open(os.path.join(res, "mismatch", s, "summary.json"))) for s in S}
    tfn = os.path.join(res, "fit", "synd_ll_tests.json")       # dispersion test (analyze_synd_ll.py), optional
    disp = json.load(open(tfn))["pre_fit"] if os.path.exists(tfn) else {}
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    y = np.arange(len(S))[::-1]
    ax.axvspan(-2, 2, color=BAND, zorder=0)
    stats = [("z_LER", "logical error rate calibration", BLUE, "o"), ("z_xent", "logical cross-entropy", AQUA, "s"),
             ("sll", "syndrome log-likelihood: mean", ORANGE, "D")]
    if disp:
        stats.append(("disp", "syndrome log-likelihood: dispersion", VIOLET, "^"))
    offs = np.linspace(0.27, -0.27, len(stats)) if len(stats) == 4 else [0.18, 0.0, -0.18]
    for (key, lab, col, mk), dy in zip(stats, offs):
        if key == "disp":
            ss = [s for s in S if s in disp]
            ax.plot([disp[s]["z_disp"] for s in ss], y[[S.index(s) for s in ss]] + dy, mk, color=col, ms=6, mec="white",
                    mew=1.0, label=lab, ls="none")
            continue
        xs = [R[s]["syndrome_loglik_M"]["z"] if key == "sll" else R[s]["calibration_ML(M)"][key] for s in S]
        ax.plot(xs, y + dy, mk, color=col, ms=6, mec="white", mew=1.0, label=lab, ls="none")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks([-100, -10, -2, 0, 2, 10, 100]); ax.set_xticklabels(["-100", "-10", "-2", "0", "2", "10", "100"])
    ax.set_yticks(y); ax.set_yticklabels([LABEL[s] for s in S])
    ax.set_xlabel("test statistic z under H0: model M = truth (band: |z| < 2)")
    ax.legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.4, -0.15), ncol=2, handletextpad=0.3)
    ax.set_title("Which statistic detects which mismatch (d = 3, r = 10, 20,000 shots each)", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, out, "fig_mismatch_detection")


def fig_fit(res, out):
    names_all = ["p2", "pm", "p1", "xt"]
    pretty = {"p2": "two-qubit gate p2", "pm": "measurement pm", "p1": "one-qubit/idle p1", "xt": "crosstalk xt"}
    fits = {}
    for s_ in ["S0", "S1", "S2", "S3b+xt", "S3b", "S4", "S5", "S2+dem", "S2+demt", "S3a+dem"]:
        fn = os.path.join(res, "fit", s_, "fit.json")
        if os.path.exists(fn):
            fits[s_] = json.load(open(fn))
    cov = None
    cdir = os.path.join(res, "fit", "coverage")
    if os.path.isdir(cdir):
        cf = sorted(f for f in os.listdir(cdir) if f.endswith(".json"))
        if cf:
            cov = json.load(open(os.path.join(cdir, cf[0])))
    if not fits and not cov:
        return
    infam = [s_ for s_ in ["S0", "S1", "S2", "S3b+xt"] if s_ in fits]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1.15]})
    cols = {"p2": BLUE, "pm": ORANGE, "p1": AQUA, "xt": VIOLET}
    mks = {"p2": "o", "pm": "s", "p1": "D", "xt": "^"}
    # (a) recovery: theta_hat / theta_true with 2 sigma (sandwich) bars; open marker = starting model M
    ax = axes[0]
    y0 = np.arange(len(infam))[::-1]
    offs = {"p2": 0.27, "pm": 0.09, "p1": -0.09, "xt": -0.27}
    for nm in names_all:
        xs, lo, hi, ys, starts = [], [], [], [], []
        for s_, yy in zip(infam, y0):
            f = fits[s_]["fit"]
            if nm not in f["names"]:
                continue
            i = f["names"].index(nm)
            ratio = f["theta_hat"][i] / f["theta_true"][i]
            se = f["se_sandwich"][i]
            xs.append(ratio); lo.append(ratio * np.exp(-2 * se)); hi.append(ratio * np.exp(2 * se))
            starts.append(f["theta0"][i] / f["theta_true"][i]); ys.append(yy + offs[nm])
        if xs:
            xs, lo, hi = map(np.array, (xs, lo, hi))
            ax.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt=mks[nm], color=cols[nm], ms=6, mec="white", mew=1.0,
                        elinewidth=1.2, capsize=0, label=pretty[nm])
            ax.plot(starts, ys, mks[nm], mfc="white", mec=cols[nm], ms=5, mew=1.2, ls="none")
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2]); ax.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_xlim(0.25, 2.3)
    ax.set_yticks(y0); ax.set_yticklabels([LABEL[s_] for s_ in infam])
    ax.set_xlabel("fitted / true (2 sigma, sandwich); open marker: start = model M")
    ax.set_title("(a) parameters recovered (truth in the family)", loc="left", fontsize=9.5, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, fontsize=7.5, handletextpad=0.3)
    # (b) coverage
    ax = axes[1]
    if cov:
        Z = np.array([r["z_opg"] for r in cov["reps"]])
        grid = np.linspace(-4, 4, 200)
        bins = np.linspace(-4, 4, 17)
        for i, nm in enumerate(["p2", "pm", "p1"]):
            h, e = np.histogram(Z[:, i], bins=bins)
            ctr = 0.5 * (e[1:] + e[:-1])
            ax.step(ctr, h / (len(Z) * (e[1] - e[0])), where="mid", color=cols[nm], lw=1.6,
                    label=f"{nm}: mean {Z[:, i].mean():+.2f}, sd {Z[:, i].std(ddof=1):.2f}")
        ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2, label="N(0, 1)")
        cv = np.mean(np.abs(Z) < 2, axis=0)
        ax.set_title(f"(b) coverage: {len(Z)} fits of {cov['n_fit']} shots", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("z = (ln fit - ln truth) / se (OPG)\n|z| < 2: " + ", ".join(f"{nm} {100*c:.0f}%" for nm, c in zip(["p2", "pm", "p1"], cv)))
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
        ax.set_ylim(0, 0.85)
    # (c) held-out decoding: excess LER over ML(T) for ML(M) and ML(fitted); post-fit syndrome log-likelihood z
    ax = axes[2]
    sc = [s_ for s_ in ["S1", "S2", "S2+dem", "S2+demt", "S3a+dem", "S3b", "S3b+xt", "S4", "S5"] if s_ in fits and "ML(T)" in fits[s_]["test"]]
    y2 = np.arange(len(sc))[::-1]
    for key, lab, col, mk, dy in [("ML(M)", "exact ML, model M", BLUE, "o", 0.13), ("ML(fit)", "exact ML, fitted model", ORANGE, "D", -0.13)]:
        xs, es = [], []
        for s_ in sc:
            t = fits[s_]["test"]; n = t["n"]; lt = t["ML(T)"]["LER"]
            xs.append(100 * (t[key]["LER"] - lt) / lt)
            if key == "ML(fit)":
                pv = t["paired_fit_vs_ML_T"]
                es.append(100 * 2 * paired_se(pv["a_wrong_b_right"], pv["a_right_b_wrong"], n) / lt)
            else:
                base = s_.split("+")[0]
                z = np.load(os.path.join(res, "mismatch", base, "data.npz"))
                nf = fits[s_]["n_fit"]
                o = z["obs"][nf:]; pm_ = z["pred_ML_M"][nf:]; pt = z["pred_ML_T"][nf:]
                b = int(np.sum((pm_ != o) & (pt == o))); c = int(np.sum((pm_ == o) & (pt != o)))
                es.append(100 * 2 * paired_se(b, c, len(o)) / lt)
        ax.errorbar(xs, y2 + dy, xerr=es, fmt=mk, color=col, ms=6, mec="white", mew=1.0, elinewidth=1.2, capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    xmax = ax.get_xlim()[1]
    tfn = os.path.join(res, "fit", "synd_ll_tests.json")
    post = json.load(open(tfn))["post_fit"] if os.path.exists(tfn) else {}
    for s_, yy in zip(sc, y2):
        z = fits[s_]["test"]["syndrome_loglik_fit"]["z"]
        txt = f"{z:+.1f}" + (f" / {post[s_]['z_disp']:+.1f}" if s_ in post else "")
        ax.text(1.02, yy, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=INK)
    ax.text(1.02, 1.01, "post-fit synd. LL z\nmean / dispersion" if post else "post-fit\nsynd. z", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=INK2)
    lab = {"S1": "S1 x1.5", "S2": "S2 (3-param.)", "S2+dem": "S2 (DEM class scales)", "S2+demt": "S2 (class x time-layer scales)",
           "S3a+dem": "S3a (DEM class scales)",
           "S3b": "S3b (3-param.)", "S3b+xt": "S3b (4-param. incl. xt)", "S4": "S4 bursts (3-param.)", "S5": "S5 drift (3-param.)"}
    ax.set_yticks(y2); ax.set_yticklabels([lab[s_] for s_ in sc])
    ax.set_xlabel("held-out excess LER over exact ML under T (%), 2 sigma")
    ax.set_title("(c) held-out decoding", loc="left", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, handletextpad=0.3)
    fig.tight_layout()
    save(fig, out, "fig_fit_recovery")


def fig_speed(res, out):
    """Sparse-frontier decoder along a tau ladder, hard shots (posterior closest to 1/2): (a) cost, (b) max |dP| and
    (c) fraction of decisions that differ from the reference (exact ML at d = 3, the smallest tau at d = 5, 7)."""
    sd = os.path.join(res, "speed")
    if not os.path.isdir(sd):
        return
    series = []      # (d, r, taus, ms_hard, ms_random, maxdP_hard (aligned to taus, nan = reference), frac_diff_hard, n_hard)
    d3 = [f for f in os.listdir(sd) if f.startswith("speed_d3")]
    ex3 = {}
    if d3:
        R = json.load(open(os.path.join(sd, d3[0])))
        for r in sorted({x["rounds"] for x in R["runs"]}):
            rows = {sub: sorted([x for x in R["runs"] if x["rounds"] == r and x["subset"] == sub], key=lambda x: -x["tau"])
                    for sub in ("hard", "random")}
            h = rows["hard"]
            series.append((3, r, [x["tau"] for x in h], [x["ms_per_round_per_shot_mean"] for x in h],
                           [x["ms_per_round_per_shot_mean"] for x in rows["random"]], [x["max_abs_dP"] for x in h],
                           [x["decisions_differ"] / x["n"] for x in h], h[0]["n"]))
            ex3[r] = h[0]["exact_ms_per_round_per_shot"]
    for f in sorted(os.listdir(sd)):
        if not (f.startswith("ladder_") and f.endswith(".json")):
            continue
        L = json.load(open(os.path.join(sd, f)))
        if "summary" not in L:
            continue
        d, r = L["d"], L["rounds"]
        taus = [float(t) for t in L["per_tau"].keys()]
        nh = len(L["subset"]["hard"])
        msh = [1e3 * np.mean(np.array(L["per_tau"][k]["seconds"])[:nh]) / r for k in L["per_tau"]]
        msr = [1e3 * np.mean(np.array(L["per_tau"][k]["seconds"])[nh:]) / r for k in L["per_tau"]]
        dp = [L["summary"][k]["hard"]["max_abs_dP"] if k in L["summary"] else np.nan for k in L["per_tau"]]
        fd = [L["summary"][k]["hard"]["decisions_differ"] / nh if k in L["summary"] else np.nan for k in L["per_tau"]]
        series.append((d, r, taus, msh, msr, dp, fd, nh))
    if not series:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    cols = {3: BLUE, 5: ORANGE, 7: AQUA}
    mks = {3: "o", 5: "s", 7: "D"}
    lstyle = {10: "-", 25: "-.", 50: "--"}
    for d, r, taus, msh, msr, dp, fd, nh in sorted(series):
        ls = lstyle.get(r, ":")
        lab = f"d = {d}, r = {r} ({nh} hard shots)"
        kw = dict(color=cols[d], ls=ls, lw=1.6, ms=5.5, mec="white", mew=0.8)
        axes[0].plot(taus, msh, mks[d], label=lab, **kw)
        t2 = [t for t, v in zip(taus, dp) if np.isfinite(v)]
        axes[1].plot(t2, [max(v, 1e-9) for v in dp if np.isfinite(v)], mks[d], **kw)
        axes[2].plot(t2, [100 * v for v in fd if np.isfinite(v)], mks[d], **kw)
    for r, v in sorted(ex3.items()):
        axes[0].axhline(v, color=BLUE, lw=0.8, ls=":")
    if ex3:
        axes[0].text(1.5e-8, max(ex3.values()) * 1.12, "exact ML, d = 3 (fast kernel)", fontsize=7.5, color=INK2,
                     va="bottom", ha="right")
    for a in axes:
        a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("pruning threshold tau")
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].set_ylabel("ms per round per shot (1 core)")
    axes[1].set_ylabel("max |P(L|s) - P_ref(L|s)|, hard shots")
    axes[2].set_ylabel("hard shots with a different decision (%)")
    axes[0].set_title("(a) cost on hard shots", loc="left", fontsize=9.5, color=INK)
    axes[1].set_title("(b) posterior error", loc="left", fontsize=9.5, color=INK)
    axes[2].set_title("(c) decision stability", loc="left", fontsize=9.5, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), fontsize=7.5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, handlelength=3.2,
               columnspacing=1.2)
    save(fig, out, "fig_speed_certificates")


def fig_reliability(res, out):
    S = [x for x in ["S0", "S2", "S3a"] if os.path.exists(os.path.join(res, "mismatch", x, "summary.json"))]
    if not S:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    cols = {"S0": INK2, "S2": BLUE, "S3a": ORANGE}
    mks = {"S0": "o", "S2": "s", "S3a": "D"}
    lo, hi = 1e-4, 0.6
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
    for sname in S:
        R = json.load(open(os.path.join(res, "mismatch", sname, "summary.json")))
        b = [x for x in R["calibration_ML(M)"]["reliability"] if x["n"] >= 50 and x["mean_predicted"] >= lo and x["observed"] > 0]
        xm = np.array([x["mean_predicted"] for x in b]); yo = np.array([x["observed"] for x in b])
        sd = np.array([x["sd_expected_errors"] / x["n"] for x in b])
        # 2-sigma band of the observed frequency under H0 around the prediction, drawn on the observed point
        ylo = np.maximum(yo - 2 * sd, lo * 1.05)
        ax.errorbar(xm, yo, yerr=[yo - ylo, 2 * sd], fmt=mks[sname] + "-", color=cols[sname], ms=6, mec="white", mew=1.0, lw=1.5,
                    elinewidth=1.0, capsize=0, label=LABEL[sname])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted error probability of ML(M), min(P, 1-P) (bin mean)\n(bins with >= 50 shots and >= 1 error)")
    ax.set_ylabel("observed error frequency of ML(M)")
    ax.set_title("Reliability of the decoding model (2 sigma under H0)", loc="left", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_reliability")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(here, "..", "..", "results", "decoder"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.res, "figures")
    os.makedirs(out, exist_ok=True)
    style()
    fig_mismatch(a.res, out)
    fig_detection(a.res, out)
    fig_reliability(a.res, out)
    fig_fit(a.res, out)
    fig_speed(a.res, out)
