"""Tables, figures and pre-registered decisions from the real-data outputs of decode_willow.py and fit_willow.py.
Implements notes/decoder/willow_prereg.md. Works on partial results: a series (patch, basis) counts only when every
round count the plan scheduled for it is done; incomplete series are listed and left out of the pooled statistics.

  python3 -I -B analyze_willow.py --res <results>/willow --out <results>/willow/analysis [--plan <res>/plan.json]
                                  [--reported willow_reported.json] [--no-figures]

Reads <res>/decode/<leaf>/<dem tag>/{summary.json, shots.npz, <decoder>.npz, synd_ref_<decoder>.npz},
<res>/fit/<leaf>/rl_tsplit/{fit.json, fit_data.npz} and <res>/plan.json (which rounds each series should have).
Writes into --out: analysis.json (everything), prereg_results.json, report.md, CSV tables (leaf_table, series_eps,
paired_gaps, calibration_leaves) and figures fig_willow_*.{png,pdf}.

Entities: ML.<prior> (exact ML at d = 3, the tau ladder at d >= 5, with the shipped DEM of that prior), our baselines
<decoder>.<prior> (same DEM file), and Google's shipped predictions <family>.<prior> (libra, harmony, corrmatch).
eps per series: weighted least squares of y_r = ln(1 - 2 E_r) on r over the window, eps = (1 - e^slope) / 2, standard
error inflated by sqrt(max(1, chi2_red)) (Birge). Paired gaps: the same fit on z_r = y_r(A) - y_r(ML) with
delta-method variances from the per-leaf 2x2 tables (unit of pairing: the shot), then the mean over series (series are
independent experiments): g = mean(eps_A - eps_ML) / mean(eps_ML).
"""
import argparse, csv, json, math, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

W5 = (10, 13, 30, 50)
EDGES = np.array([0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.4, 0.5 + 1e-12])
FAMILY = (("libra", "libra"), ("harmony", "harmony"), ("correlated_matching", "corrmatch"))
NICE = {"ML": "ML", "libra": "Libra", "harmony": "Harmony", "corrmatch": "corr. matching (shipped)",
        "corr_mwpm": "corr. MWPM (ours)", "mwpm": "MWPM (ours)", "tesseract": "Tesseract", "beliefmatch": "BeliefMatching"}
PRIOR = {"rl": "RL", "si1000": "SI1000", "other": "other"}
COL = {"ML.rl": "#2a78d6", "ML.si1000": "#4a3aa7", "libra.rl": "#eb6834", "nn": "#1baf7a"}
INK, INK2, GRID, BAND, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec", "#fcfcfb"
MARGIN = 0.03          # K1 equivalence margin and the practical-relevance threshold of the bar
CAL_MARGIN = 0.05      # P3: calibrated if the 90% CI of obs / pred lies in [0.95, 1.05]
K2_COST_MS = 25.0      # K2: tau1 cost per round per shot on the first d = 5 leaf


def nice(e):
    f, p = e.split(".", 1) if "." in e else (e, "")
    return f"{NICE.get(f, f)} ({PRIOR.get(p, p)})" if p else NICE.get(f, f)


def prior_of(s):
    s = s.lower()
    return "si1000" if "si1000" in s else ("rl" if ("rl_optimized" in s or "_rl_" in s or s.endswith("_rl")) else "other")


def shipped_entity(path):
    parts = path.replace("\\", "/").split("/")
    pw = parts[-2] if len(parts) > 1 else parts[0]
    for key, fam in FAMILY:
        if pw.startswith(key):
            return f"{fam}.{prior_of(pw)}"
    return f"{pw}.{prior_of(pw)}"


def entity_of(dec, prior):
    return ("ML." if dec.startswith("ml_") else dec.split(":")[0] + ".") + prior


def eps_from(E, r):
    E = min(max(E, 0.0), 0.5 - 1e-12)
    return 0.5 * (1 - (1 - 2 * E) ** (1.0 / max(r, 1)))


def pval(z):
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------------------------------------------------------- loading
def load_leaf(res, rel, tags):
    """All decode folders of one leaf -> observed flips, per-entity predictions (global shot index; -1 = not decoded),
    ML posteriors, and per-folder info."""
    obs, ents, ml, info, shipped = None, {}, {}, {"folders": {}, "problems": []}, set()
    for tag in sorted(tags):
        fd = os.path.join(res, "decode", rel, tag)
        try:
            s = json.load(open(os.path.join(fd, "summary.json")))
        except Exception as ex:
            info["problems"].append(f"{tag}: unreadable summary ({ex!r})"); continue
        info["folders"][tag] = s.get("status")
        info.setdefault("meta", s.get("meta"))
        info["rounds"] = s.get("rounds") or (s.get("meta") or {}).get("rounds")
        info.setdefault("recompute", {})[tag] = s.get("recompute_check")
        info.setdefault("balance", {})[tag] = s.get("selection_balance")
        if not os.path.exists(os.path.join(fd, "shots.npz")) or s.get("n_total") is None:
            continue
        N = s["n_total"]
        if obs is None:
            obs = np.full(N, -1, np.int8)
        z = np.load(os.path.join(fd, "shots.npz"))
        idx = z["idx"]
        if np.any((obs[idx] >= 0) & (obs[idx] != z["obs"])):
            info["problems"].append(f"{tag}: observed flips differ between folders")
        obs[idx] = z["obs"]
        for nm, row in zip(z["shipped_names"], z["shipped"]):
            e = shipped_entity(str(nm)); shipped.add(e)
            ents.setdefault(e, np.full(N, -1, np.int8))[idx] = row
        prior = tag.split("_")[0]
        for dec, ds in (s.get("decoders") or {}).items():
            if dec.startswith("shipped:") or "skipped" in ds:
                continue
            dn = dec.replace(":", "_").replace("/", "-")
            p = os.path.join(fd, dn + ".npz")
            if not os.path.exists(p):
                continue
            zz = np.load(p)
            e = entity_of(dec, prior)
            ents.setdefault(e, np.full(N, -1, np.int8))[zz["idx"]] = zz["pred"][:, 0]
            if "llr" in zz.files:
                m = ml.setdefault(e, {"llr": np.full(N, np.nan, np.float32), "logp": np.full(N, np.nan),
                                      "have": np.zeros(N, bool)})
                m["llr"][zz["idx"]] = zz["llr"]; m["logp"][zz["idx"]] = zz["logp"]; m["have"][zz["idx"]] = True
                rp = os.path.join(fd, f"synd_ref_{dn}.npz")
                if os.path.exists(rp):
                    m["lref"] = np.load(rp)["logp"]
                if "ladder" in ds:
                    m["ladder"] = dict(ds["ladder"], decoder=dec)
                m["ms_per_shot"] = ds.get("ms_per_shot")
    return obs, ents, ml, info, shipped


def count2(a, b, obs):
    m = (a >= 0) & (b >= 0) & (obs >= 0)
    ea = a[m] != obs[m]; eb = b[m] != obs[m]
    return {"n": int(m.sum()), "n10": int(np.sum(ea & ~eb)), "n01": int(np.sum(~ea & eb)), "n11": int(np.sum(ea & eb))}


def calib_sums(llr, obs, have):
    sel = have & (obs >= 0)
    L = llr[sel].astype(np.float64); o = obs[sel].astype(bool)
    nan = np.isnan(L)
    L = np.where(nan, 0.0, L)                       # a NaN posterior counts as P = 1/2 (decoded as "no flip")
    P1 = 1 / (1 + np.exp(-L)); m = 1 / (1 + np.exp(np.abs(L)))
    err = np.where(nan, False, P1 > 0.5) != o
    P1c = np.clip(P1, 1e-300, 1 - 1e-16); P0c = np.clip(1 - P1, 1e-300, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nll = -np.where(o, np.log(P1c), np.log(P0c))
        H = -(P0c * np.log(P0c) + P1c * np.log(P1c))
        v = P0c * P1c * np.log(P1c / P0c) ** 2
    b = np.digitize(m, EDGES) - 1
    K = len(EDGES) - 1
    bins = {"n": np.bincount(b, minlength=K).tolist(), "pred": np.bincount(b, weights=m, minlength=K).tolist(),
            "var": np.bincount(b, weights=m * (1 - m), minlength=K).tolist(),
            "obs": np.bincount(b, weights=err.astype(float), minlength=K).tolist()}
    return {"n": int(sel.sum()), "k": int(err.sum()), "sum_m": float(m.sum()), "sum_mvar": float(np.sum(m * (1 - m))),
            "sum_nll": float(np.nansum(nll)), "sum_H": float(np.nansum(H)), "sum_xvar": float(np.nansum(v)),
            "nan": int(nan.sum()), "bins": bins}


# ---------------------------------------------------------------- statistics
def wls(r, y, var):
    r, y, var = (np.asarray(x, float) for x in (r, y, var))
    w = 1 / var
    W = w.sum(); rb = (w * r).sum() / W; yb = (w * y).sum() / W
    Srr = (w * (r - rb) ** 2).sum()
    if Srr <= 0:
        return None
    b = (w * (r - rb) * (y - yb)).sum() / Srr
    a = yb - b * rb
    chi2 = float((w * (y - a - b * r) ** 2).sum()); dof = len(r) - 2
    red = chi2 / dof if dof > 0 else float("nan")
    se = math.sqrt(1 / Srr) * (math.sqrt(max(1.0, red)) if dof > 0 else 1.0)
    return {"slope": float(b), "intercept": float(a), "se": se, "chi2_red": red, "npts": int(len(r))}


def series_eps(points, window):
    """points {r: (n, k)} -> eps fit over the window (None if fewer than 2 usable points)."""
    rs, ys, vs, dropped = [], [], [], []
    for r in sorted(points):
        if r not in window:
            continue
        n, k = points[r]
        E = k / n if n else float("nan")
        if not n or not (0 < E < 0.5):
            dropped.append(r); continue
        rs.append(r); ys.append(math.log(1 - 2 * E)); vs.append(4 * E * (1 - E) / (n * (1 - 2 * E) ** 2))
    if len(rs) < 2:
        return None
    f = wls(rs, ys, vs)
    if f is None:
        return None
    eb = math.exp(f["slope"])
    f.update(eps=(1 - eb) / 2, se_eps=eb * f["se"] / 2, rounds=rs, dropped=dropped,
             complete=bool(window) and set(window) <= set(points))
    return f


def paired_series(points, window):
    """points {r: 2x2 counts of (A, ML)} -> paired eps difference of A relative to ML."""
    rs, zs, vz, yM, vM, dropped = [], [], [], [], [], []
    for r in sorted(points):
        if r not in window:
            continue
        c = points[r]; n = c["n"]
        if not n:
            dropped.append(r); continue
        EA = (c["n10"] + c["n11"]) / n; EM = (c["n01"] + c["n11"]) / n; p11 = c["n11"] / n
        if not (0 < EA < 0.5 and 0 < EM < 0.5):
            dropped.append(r); continue
        gA, gM = -2 / (1 - 2 * EA), -2 / (1 - 2 * EM)
        VA, VM, Cv = EA * (1 - EA) / n, EM * (1 - EM) / n, (p11 - EA * EM) / n
        rs.append(r); zs.append(math.log(1 - 2 * EA) - math.log(1 - 2 * EM))
        vz.append(max(gA ** 2 * VA + gM ** 2 * VM - 2 * gA * gM * Cv, 1e-18))
        yM.append(math.log(1 - 2 * EM)); vM.append(gM ** 2 * VM)
    if len(rs) < 2:
        return None
    fz, fm = wls(rs, zs, vz), wls(rs, yM, vM)
    if fz is None or fm is None:
        return None
    bM, beta = fm["slope"], fz["slope"]
    eM = (1 - math.exp(bM)) / 2; eA = (1 - math.exp(bM + beta)) / 2
    se_d = math.exp(bM + beta) * fz["se"] / 2
    se_eM = math.exp(bM) * fm["se"] / 2
    return {"eps_ML": eM, "eps_A": eA, "delta": eA - eM, "se_delta": se_d, "se_eps_ML": se_eM, "rel": (eA - eM) / eM,
            "se_rel": math.sqrt((se_d / eM) ** 2 + ((eA - eM) * se_eM / eM ** 2) ** 2), "chi2_red": fz["chi2_red"],
            "rounds": rs, "dropped": dropped, "complete": bool(window) and set(window) <= set(points)}


def ci_block(dct, rel, se):
    dct.update(rel=rel, se_rel=se, z=rel / se if se > 0 else float("nan"), p=pval(rel / se) if se > 0 else float("nan"),
               ci95=[rel - 1.96 * se, rel + 1.96 * se], ci90=[rel - 1.645 * se, rel + 1.645 * se])
    return dct


def pool_gap(rows):
    """Mean over series: g = mean(delta) / mean(eps_ML); SE from independent series (delta method)."""
    S = len(rows)
    if S == 0:
        return None
    d = float(np.mean([x["delta"] for x in rows])); eM = float(np.mean([x["eps_ML"] for x in rows]))
    sd = math.sqrt(sum(x["se_delta"] ** 2 for x in rows)) / S; se_eM = math.sqrt(sum(x["se_eps_ML"] ** 2 for x in rows)) / S
    rel = d / eM
    se_rel = math.sqrt((sd / eM) ** 2 + (d * se_eM / eM ** 2) ** 2)
    return ci_block({"n_series": S, "mean_eps_ML": eM, "mean_delta": d, "se_delta": sd}, rel, se_rel)


def ratio_from_counts(cells):
    """(sum x - sum y) / sum y over leaves from 2x2 counts of (x, y) = (A wrong, B wrong); delta-method SE with the
    shot as the unit (stratified by leaf)."""
    n = sum(c["n"] for c in cells)
    X = sum(c["n10"] + c["n11"] for c in cells); Y = sum(c["n01"] + c["n11"] for c in cells)
    if n == 0 or Y == 0:
        return None
    R = X / Y
    var = 0.0
    u = np.array([1.0, -R, 1.0 - R, 0.0])           # per shot u = x - R y for (1,0), (0,1), (1,1), (0,0)
    for c in cells:
        if not c["n"]:
            continue
        w = np.array([c["n10"], c["n01"], c["n11"], c["n"] - c["n10"] - c["n01"] - c["n11"]], float)
        mu = (u * w).sum() / c["n"]
        var += ((u - mu) ** 2 * w).sum()
    return ci_block({"n": n, "errors_A": X, "errors_B": Y, "leaves": len(cells)}, R - 1, math.sqrt(var) / Y)


def mean_eps(fits):
    m = float(np.mean([f["eps"] for f in fits]))
    return m, math.sqrt(sum(f["se_eps"] ** 2 for f in fits)) / len(fits)


def lam_fit(eps_by_d):
    """eps_by_d {d: (eps, se)} -> Lambda = exp(-slope) of ln eps on (d + 1) / 2 (Google's definition)."""
    ds = sorted(eps_by_d)
    if len(ds) < 2:
        return None
    x = np.array([(d + 1) / 2 for d in ds]); y = np.array([math.log(eps_by_d[d][0]) for d in ds])
    v = np.array([(eps_by_d[d][1] / eps_by_d[d][0]) ** 2 for d in ds])
    if len(ds) == 2:
        m = (y[1] - y[0]) / (x[1] - x[0]); se = math.sqrt(v.sum()) / (x[1] - x[0])
    else:
        f = wls(x, y, v); m, se = f["slope"], f["se"]
    L = math.exp(-m)
    return {"distances": ds, "Lambda": L, "se": L * se, "eps": {str(k): list(v) for k, v in eps_by_d.items()}}


def holm(tests):
    """tests {name: p} -> {name: Holm-adjusted p}; None entries are left out."""
    items = sorted((p, k) for k, p in tests.items() if p is not None and math.isfinite(p))
    out, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 200, "figure.facecolor": SURF, "axes.facecolor": SURF})
    return plt


def rounds_axis(ax):
    """Log-scaled rounds axis with plain tick labels."""
    import matplotlib.ticker as mt
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ticks = [t for t in (1, 3, 10, 30, 100, 250) if lo <= t <= hi]
    ax.xaxis.set_major_locator(mt.FixedLocator(ticks)); ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xticklabels([str(t) for t in ticks]); ax.set_xlim(lo, hi)
    ax.set_xlabel("rounds r")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor=fig.get_facecolor())


def figures(R, out, reported):
    plt = setup_mpl()
    made = []
    # (1) eps per series: ML with both DEMs, Libra, and the reported NN where the patch labels map to Table S5
    for d in (3, 5):
        keys = [k for k in R["series_eps"] if k.startswith(f"d{d}|")]
        if not keys:
            continue
        ser = sorted({"|".join(k.split("|")[:3]) for k in keys})
        fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(ser) + 1.3))
        spec = [("ML.rl", "o", True), ("ML.si1000", "o", False), ("libra.rl", "s", True)]
        for j, (e, mk, filled) in enumerate(spec):
            xs, es, ys = [], [], []
            for i, s in enumerate(ser):
                f = R["series_eps"].get(f"{s}|{e}")
                if f:
                    xs.append(f["eps"]); es.append(1.96 * f["se_eps"]); ys.append(i + (j - 1) * 0.2)
            if xs:
                c = COL[e]
                ax.errorbar(xs, ys, xerr=es, fmt=mk, ms=5.5, color=c, mfc=c if filled else SURF, mec=c, mew=1.2,
                            elinewidth=1.0, capsize=0, label=nice(e))
        nn = R.get("nn_by_series", {})
        have = [(i, s) for i, s in enumerate(ser) if s in nn]
        if have:
            ax.errorbar([nn[s][0] for _, s in have], [i + 0.4 for i, _ in have], xerr=[1.96 * nn[s][1] for _, s in have],
                        fmt="D", ms=4.5, color=COL["nn"], mec=SURF, mew=0.6, elinewidth=1.0, capsize=0,
                        label="neural network (reported, Table S5)")
        ax.set_yticks(range(len(ser))); ax.set_yticklabels([s.split("|", 1)[1].replace("|", "  ") for s in ser])
        ax.invert_yaxis(); ax.set_xlabel("logical error per cycle eps (95% CI)")
        ax.set_title(f"d = {d}: eps per patch and basis (window {R['windows']['eps'].get(str(d))}; Libra: its rounds in it)",
                     loc="left", fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5 / max(len(ser), 1) - 0.1), ncol=2, fontsize=7.5)
        save(fig, out, f"fig_willow_eps_d{d}"); plt.close(fig); made.append(f"fig_willow_eps_d{d}")
    # (2) paired gaps to ML with the same prior (and ML(SI1000) vs ML(RL)), pooled per distance
    pg = {k: v for k, v in R["pooled_gaps"].items() if v}
    def same_prior(pair):
        a, b = pair.split(" vs ")
        return a.split(".")[-1] == b.split(".")[-1] or (a == "ML.si1000" and b == "ML.rl")
    labs = sorted({k.split("|", 1)[1] for k in pg if same_prior(k.split("|", 1)[1])},
                  key=lambda s: (s.split(" vs ")[1] != "ML.rl", s.split(" vs ")[1], s))
    if labs:
        fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(labs) + 1.4))
        ax.axvspan(-100 * MARGIN, 100 * MARGIN, color=BAND, zorder=0, label=f"±{100 * MARGIN:.0f}% margin")
        for j, d in enumerate((3, 5)):
            xs, lo, hi, ys = [], [], [], []
            for i, l in enumerate(labs):
                p = pg.get(f"d{d}|{l}")
                if p:
                    xs.append(100 * p["rel"]); lo.append(100 * (p["rel"] - p["ci95"][0])); hi.append(100 * (p["ci95"][1] - p["rel"]))
                    ys.append(i + (j - 0.5) * 0.28)
            if xs:
                ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o" if d == 3 else "s", ms=5, color=INK if d == 3 else INK2,
                            mfc=INK if d == 3 else SURF, mew=1.1, elinewidth=1.0, capsize=0, label=f"d = {d}")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels([f"{nice(l.split(' vs ')[0])}  vs  {nice(l.split(' vs ')[1])}" for l in labs], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("eps gap to ML, % of eps(ML) (95% CI; positive = ML better)")
        ax.set_title("Paired decoder gaps (same shots, mean over series)", loc="left", fontsize=9)
        ax.legend(loc="lower right", fontsize=7.5)
        save(fig, out, "fig_willow_paired_gaps"); plt.close(fig); made.append("fig_willow_paired_gaps")
    # (3) calibration: observed / predicted errors per leaf against r, and the pooled reliability diagram (d = 3)
    cal = R["calibration"]
    if any(x["d"] == 3 for x in cal["leaves"]):
        fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2))
        ax = axs[0]
        ax.axhspan(1 - CAL_MARGIN, 1 + CAL_MARGIN, color=BAND, zorder=0)
        ax.axhline(1, color=INK2, lw=0.8)
        for e, mk in (("ML.rl", "o"), ("ML.si1000", "s")):
            pts = [(x["rounds"], x["ratio"]) for x in cal["leaves"] if x["entity"] == e and x["d"] == 3 and math.isfinite(x["ratio"])]
            if pts:
                rr, ra = zip(*pts)
                jit = np.exp(np.random.default_rng(0).uniform(-0.04, 0.04, len(rr)))
                ax.plot(np.array(rr) * jit, ra, mk, ms=4, color=COL[e], mec=SURF, mew=0.6, label=nice(e), alpha=0.9)
        rounds_axis(ax); ax.set_ylabel("observed / predicted logical errors")
        ax.set_title("(a) calibration per leaf, d = 3", loc="left", fontsize=9); ax.legend(fontsize=7.5)
        ax = axs[1]
        ax.plot([1e-4, 0.5], [1e-4, 0.5], color=INK2, lw=0.8)
        for e, mk in (("ML.rl", "o"), ("ML.si1000", "s")):
            b = cal["pooled_bins"].get(f"d3|{e}")
            if not b:
                continue
            n = np.array(b["n"], float); pr = np.array(b["pred"]); ob = np.array(b["obs"]); vv = np.array(b["var"])
            ok = (n >= 50) & (ob >= 1)
            if ok.any():
                ax.errorbar(pr[ok] / n[ok], ob[ok] / n[ok], yerr=2 * np.sqrt(vv[ok]) / n[ok], fmt=mk + "-", ms=4.5, lw=1.4,
                            color=COL[e], mec=SURF, mew=0.6, elinewidth=0.9, capsize=0, label=nice(e))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("predicted error probability min(P, 1 - P)")
        ax.set_ylabel("observed error frequency")
        ax.set_title("(b) reliability, d = 3, r in W5 (bars: 2 sd if calibrated)", loc="left", fontsize=9)
        ax.legend(fontsize=7.5)
        fig.tight_layout(); save(fig, out, "fig_willow_calibration"); plt.close(fig); made.append("fig_willow_calibration")
    # (4) syndrome log-likelihood tests per leaf (data against fresh samples of the same DEM)
    syn = [x for x in R["syndrome"] if x["d"] == 3]
    if syn:
        fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
        for ax, key, ttl in ((axs[0], "z_mean", "(a) mean of ln P(s), z"), (axs[1], "z_disp", "(b) dispersion of ln P(s), z")):
            ax.axhspan(-2, 2, color=BAND, zorder=0)
            for e, mk in (("ML.rl", "o"), ("ML.si1000", "s")):
                pts = [(x["rounds"], x[key]) for x in syn if x["entity"] == e and x.get(key) is not None and math.isfinite(x[key])]
                if pts:
                    rr, zz = zip(*pts)
                    ax.plot(rr, zz, mk, ms=4, color=COL[e], mec=SURF, mew=0.6, label=nice(e))
            rounds_axis(ax); ax.set_yscale("symlog", linthresh=2)
            lo, hi = ax.get_ylim()
            ticks = [t for t in (-1000, -100, -10, -2, 0, 2, 10, 100, 1000) if lo <= t <= hi]
            ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}".replace("-", "−") for t in ticks]); ax.set_ylim(lo, hi)
            ax.set_title(ttl + ", d = 3", loc="left", fontsize=9)
        axs[0].set_ylabel("z (band: |z| < 2)"); axs[0].legend(fontsize=7.5)
        fig.tight_layout(); save(fig, out, "fig_willow_syndrome_ll"); plt.close(fig); made.append("fig_willow_syndrome_ll")
    # (5) Lambda on the common window W5
    lam = {k: v for k, v in R["lambda"].items() if v and k in ("ML.rl", "libra.rl")}
    if lam:
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        for e, mk in (("libra.rl", "s"), ("ML.rl", "o")):
            v = lam.get(e)
            if not v:
                continue
            ds = v["distances"]
            ax.errorbar(ds, [v["eps"][str(d)][0] for d in ds], yerr=[1.96 * v["eps"][str(d)][1] for d in ds], fmt=mk + "-",
                        ms=5, lw=1.4, color=COL[e], mec=SURF, mew=0.6,
                        label=f"{nice(e)}: Λ = {v['Lambda']:.2f} ± {v['se']:.2f}")
        rep = reported["table_S1_fig1d"]["nn"]
        ax.errorbar([3, 5, 7], [rep["eps3"][0], rep["eps5"][0], rep["eps7"][0]],
                    yerr=[1.96 * rep[k][1] for k in ("eps3", "eps5", "eps7")], fmt="D--", ms=4.5, lw=1.0, color=COL["nn"],
                    mec=SURF, mew=0.6, label=f"NN (reported, all r): Λ = {rep['Lambda'][0]:.2f}")
        ax.set_yscale("log"); ax.set_xticks([3, 5, 7]); ax.set_xlabel("code distance d"); ax.set_ylabel("mean eps")
        ax.set_title("Error suppression (ours: r in W5)", loc="left", fontsize=9); ax.legend(fontsize=7)
        save(fig, out, "fig_willow_lambda"); plt.close(fig); made.append("fig_willow_lambda")
    # (6) decomposition on the held-out shots of the fitted leaves
    dc = R["decomposition"].get("pooled")
    if dc:
        nl = dc.get("n_leaves_with_libra", 0)
        rows = [(f"P4 mismatch: ML(RL DEM) vs ML(fitted), {dc['n_leaves']} leaves", dc.get("mismatch")),
                (f"mismatch, {nl} leaves with Libra", dc.get("mismatch_libra_leaves")),
                (f"suboptimality: Libra vs ML(RL DEM), {nl} leaves", dc.get("suboptimality")),
                (f"total: Libra vs ML(fitted), {nl} leaves", dc.get("total"))]
        rows = [(l, v) for l, v in rows if v]
        fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(rows) + 1.2))
        ax.axvspan(-100 * MARGIN, 100 * MARGIN, color=BAND, zorder=0)
        for i, (l, v) in enumerate(rows):
            ax.errorbar([100 * v["rel"]], [i], xerr=[[100 * (v["rel"] - v["ci95"][0])], [100 * (v["ci95"][1] - v["rel"])]],
                        fmt="o", ms=6, color=INK, elinewidth=1.1, capsize=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for l, _ in rows], fontsize=7.5); ax.invert_yaxis()
        ax.set_xlabel("held-out logical errors, % of ML(fitted DEM) (95% CI)")
        ax.set_title("Decomposition on held-out shots, d = 3, r in W5", loc="left", fontsize=9)
        save(fig, out, "fig_willow_decomposition"); plt.close(fig); made.append("fig_willow_decomposition")
    return made


# ---------------------------------------------------------------- main
def planned_rounds(plan):
    """{(d, patch, basis, entity or 'shipped'): set of rounds} from the decode jobs of the plan. A shipped series
    (libra.rl, ...) is planned at the leaves whose job lists that pathway under "shipped" (the real archive has no
    Libra predictions at r = 1 and 13); plans without these lists fall back to every decoded leaf ('shipped')."""
    pr = {}
    for j in plan["jobs"]:
        if j.get("kind") != "decode":
            continue
        key = (j["d"], j["patch"], j["basis"])
        prior = j["dem_tag"].split("_")[0]
        for dec in j["decoders"].split(","):
            pr.setdefault(key + (entity_of(dec, prior),), set()).add(j["rounds"])
        pr.setdefault(key + ("shipped",), set()).add(j["rounds"])
        for pw in j.get("shipped") or []:
            pr.setdefault(key + (shipped_entity(pw),), set()).add(j["rounds"])
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=None, help="default <res>/plan.json")
    ap.add_argument("--reported", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "willow_reported.json"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reported = json.load(open(a.reported))
    t0 = time.time()
    plan_p = a.plan or os.path.join(a.res, "plan.json")
    plan = json.load(open(plan_p)) if os.path.exists(plan_p) else None
    PR = planned_rounds(plan) if plan else None
    dec_root = os.path.join(a.res, "decode")
    leaves = {}
    for dp, dn, fn in os.walk(dec_root):
        if "summary.json" in fn:
            rel, tag = os.path.split(os.path.relpath(dp, dec_root))
            leaves.setdefault(rel, []).append(tag)
    R = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan": os.path.abspath(plan_p) if plan else None,
         "leaves": {}, "series_eps": {}, "series_eps_W5": {}, "pooled_gaps": {}, "pooled_gaps_W5": {}, "pooled_counts": {},
         "paired_series": {}, "paired_series_W5": {}, "calibration": {"leaves": [], "pooled": {}, "pooled_bins": {}},
         "syndrome": [], "decomposition": {"leaves": []}, "ladder": [],
         "integrity": {"problems": [] if plan else ["no plan.json: completeness judged on what is present"]},
         # d = 3 leaves where Libra's RL DEM file has other content than the file ML used (plan, willow_stage.py)
         "libra_dem_differs": sorted({j["leaf"] for j in (plan or {}).get("jobs", []) if j.get("d") == 3 and
                                      ((j.get("dems") or {}).get("rl") or {}).get("libra_same_content", True) is not True})}
    ent_pts, pair_pts, cal_acc, shipped_all = {}, {}, {}, set()
    from analyze_synd_ll import ll_tests
    for rel in sorted(leaves):
        obs, ents, ml, info, shipped = load_leaf(a.res, rel, leaves[rel])
        shipped_all |= shipped
        meta = info.get("meta") or {}
        d, basis, patch, r = meta.get("distance"), meta.get("basis"), meta.get("patch"), info.get("rounds")
        if obs is None or d is None:
            R["integrity"]["problems"].append(f"{rel}: no usable folder ({info['folders']})"); continue
        ser = f"d{d}|{patch}|{basis}"
        L = {"d": d, "patch": patch, "basis": basis, "rounds": r, "folders": info["folders"], "entities": {},
             "recompute": info.get("recompute"), "balance": info.get("balance"), "problems": info["problems"]}
        for e, p in ents.items():
            sel = (p >= 0) & (obs >= 0)
            n, k = int(sel.sum()), int(np.sum(p[sel] != obs[sel]))
            if n:
                L["entities"][e] = {"n": n, "errors": k, "LER": k / n, "eps_1pt": eps_from(k / n, r)}
                ent_pts.setdefault(f"{ser}|{e}", {})[r] = (n, k)
        names = sorted(ents)
        for A in names:
            for B in names:
                if A != B and B.startswith("ML."):
                    c = count2(ents[A], ents[B], obs)
                    if c["n"]:
                        pair_pts.setdefault(f"{ser}|{A} vs {B}", {})[r] = c
        for e, m in ml.items():
            cs = calib_sums(m["llr"], obs, m["have"])
            row = {"leaf": rel, "d": d, "patch": patch, "basis": basis, "rounds": r, "entity": e, "n": cs["n"],
                   "errors": cs["k"], "predicted": cs["sum_m"],
                   "ratio": cs["k"] / cs["sum_m"] if cs["sum_m"] > 0 else float("nan"),
                   "z_LER": (cs["k"] - cs["sum_m"]) / math.sqrt(cs["sum_mvar"]) if cs["sum_mvar"] > 0 else float("nan"),
                   "z_xent": (cs["sum_nll"] - cs["sum_H"]) / math.sqrt(cs["sum_xvar"]) if cs["sum_xvar"] > 0 else float("nan"),
                   "nan_posteriors": cs["nan"]}
            R["calibration"]["leaves"].append(row)
            cal_acc.setdefault((d, e, r), []).append(cs)
            ident = {k: row[k] for k in ("leaf", "d", "patch", "basis", "rounds", "entity")}
            if "lref" in m:
                R["syndrome"].append(dict(ident, **ll_tests(m["logp"][m["have"]], m["lref"])))
            if "ladder" in m:
                R["ladder"].append(dict(ident, **m["ladder"]))
        # decomposition on the held-out shots of the fitted leaf (fit on half the shots with the RL DEM)
        fp = os.path.join(a.res, "fit", rel, "rl_tsplit")
        if os.path.exists(os.path.join(fp, "fit_data.npz")) and os.path.exists(os.path.join(fp, "fit.json")):
            z = np.load(os.path.join(fp, "fit_data.npz")); fj = json.load(open(os.path.join(fp, "fit.json")))
            ti = z["test_idx"]
            o = obs[ti]
            if np.any((o >= 0) & (o != z["obs_test"])):
                L["problems"].append("fit: observed flips differ from the decode folders")
            T = np.full(len(obs), -1, np.int8); T[ti] = z["fitted_pred"]
            S_ = np.full(len(obs), -1, np.int8); S_[ti] = z["shipped_pred"]
            ft = fj["test"]
            dl = {"leaf": rel, "d": d, "patch": patch, "basis": basis, "rounds": r, "n_test": int(len(ti)),
                  "n_classes": len(fj["classes"]), "scales": dict(zip(fj["classes"], fj["scale_hat"])), "split": fj.get("split"),
                  "LER_shipped": ft["shipped"]["LER"], "LER_fitted": ft["fitted"]["LER"],
                  "calibration": {t: {k: ft[t]["calibration"][k] for k in ("ratio_obs_over_pred", "z_LER", "z_xent")}
                                  for t in ("shipped", "fitted")},
                  "syndrome_loglik": {t: {k: ft[t]["syndrome_loglik"][k] for k in ("z_mean", "z_disp")} for t in ("shipped", "fitted")},
                  "cells": {"mismatch": count2(S_, T, obs),
                            "total": count2(ents["libra.rl"], T, obs) if "libra.rl" in ents else None,
                            "suboptimality": count2(ents["libra.rl"], S_, obs) if "libra.rl" in ents else None}}
            if "ML.rl" in ents:
                have = ents["ML.rl"][ti] >= 0
                dl["ML_rl_disagrees_with_fit_shipped"] = int(np.sum(ents["ML.rl"][ti][have] != z["shipped_pred"][have]))
                if dl["ML_rl_disagrees_with_fit_shipped"]:
                    L["problems"].append(f"fit: {dl['ML_rl_disagrees_with_fit_shipped']} held-out ML(RL) decisions differ from the decode job")
            R["decomposition"]["leaves"].append(dl)
        R["leaves"][rel] = L
    # ---- planned rounds, windows, completeness
    if PR is not None:
        series_planned = {k[:3] for k in PR}
    else:
        PR = {}
        for key, pts in ent_pts.items():
            dd, pa, ba, e = key.split("|"); kk = (int(dd[1:]), pa, ba)
            PR.setdefault(kk + (e,), set()).update(pts)
            PR.setdefault(kk + ("shipped",), set()).update(pts)
        series_planned = {k[:3] for k in PR}
    expected_series = {d: len({s for s in series_planned if s[0] == d}) for d in (3, 5, 7)}

    def planned(skey, e):
        dd, pa, ba = skey.split("|"); kk = (int(dd[1:]), pa, ba)
        return PR.get(kk + (e,)) or (PR.get(kk + ("shipped",)) if e in shipped_all else None) or set()

    R["windows"] = {"eps": {"3": "r >= 10", "5": "r in W5", "7": "probe rounds (r >= 10)"}, "calibration": "r in W5",
                    "lambda": "r in W5 at every distance", "expected_series": expected_series}
    # ---- eps per series (primary window: the planned rounds with r >= 10; and W5)
    for key, pts in ent_pts.items():
        skey, e = key.rsplit("|", 1)
        pl = planned(skey, e)
        f = series_eps(pts, tuple(sorted(x for x in pl if x >= 10)))
        if f:
            R["series_eps"][key] = f
        f5 = series_eps(pts, tuple(x for x in W5 if x in pl))
        if f5:
            R["series_eps_W5"][key] = f5
    means = {}
    for key, f in R["series_eps"].items():
        if f["complete"]:
            means.setdefault(f"{key.split('|')[0]}|{key.rsplit('|', 1)[1]}", []).append(f)
    R["mean_eps"] = {}
    for k, v in means.items():
        m, se = mean_eps(v)
        R["mean_eps"][k] = {"n_series": len(v), "expected_series": expected_series[int(k[1])], "eps": m, "se": se}
    # ---- paired gaps (window: rounds planned for both entities with r >= 10; and W5)
    for key, pts in pair_pts.items():
        skey, pair = key.rsplit("|", 1)
        A, B = pair.split(" vs ")
        both = planned(skey, A) & planned(skey, B)
        f = paired_series(pts, tuple(sorted(x for x in both if x >= 10)))
        if f:
            R["paired_series"][key] = f
        f5 = paired_series(pts, tuple(x for x in W5 if x in both))
        if f5:
            R["paired_series_W5"][key] = f5
    for src, dst in (("paired_series", "pooled_gaps"), ("paired_series_W5", "pooled_gaps_W5")):
        groups = {}
        for key, f in R[src].items():
            if f["complete"]:
                groups.setdefault(f"{key.split('|')[0]}|{key.rsplit('|', 1)[1]}", []).append(f)
        for k, rows in groups.items():
            p = pool_gap(rows)
            if p:
                p["expected_series"] = expected_series[int(k[1])]
                R[dst][k] = p
    # leaf-level secondary: pooled error counts over every paired leaf with r >= 10
    cells = {}
    for key, pts in pair_pts.items():
        cells.setdefault(f"{key.split('|')[0]}|{key.rsplit('|', 1)[1]}", []).extend(c for r, c in pts.items() if r >= 10)
    for k, cs in cells.items():
        v = ratio_from_counts(cs)
        if v:
            R["pooled_counts"][k] = v
    # ---- calibration pooled over r in W5
    K = len(EDGES) - 1
    for (d, e, r), lst in cal_acc.items():
        if r not in W5:
            continue
        acc = R["calibration"]["pooled"].setdefault(f"d{d}|{e}", {"n": 0, "k": 0, "sum_m": 0.0, "sum_mvar": 0.0, "var_k": 0.0,
                                                                   "sum_nll": 0.0, "sum_H": 0.0, "sum_xvar": 0.0, "leaves": 0})
        bins = R["calibration"]["pooled_bins"].setdefault(f"d{d}|{e}", {"n": [0] * K, "pred": [0.0] * K, "var": [0.0] * K,
                                                                         "obs": [0.0] * K})
        for cs in lst:
            for k2 in ("n", "k", "sum_m", "sum_mvar", "sum_nll", "sum_H", "sum_xvar"):
                acc[k2] += cs[k2]
            acc["var_k"] += cs["k"] * (1 - cs["k"] / max(cs["n"], 1)); acc["leaves"] += 1
            for k2 in ("n", "pred", "var", "obs"):
                bins[k2] = [x + y for x, y in zip(bins[k2], cs["bins"][k2])]
    for k, acc in R["calibration"]["pooled"].items():
        ratio = acc["k"] / acc["sum_m"] if acc["sum_m"] else float("nan")
        se = math.sqrt(acc["var_k"]) / acc["sum_m"] if acc["sum_m"] else float("nan")
        acc.update(ratio=ratio, se_ratio=se, ci90=[ratio - 1.645 * se, ratio + 1.645 * se],
                   ci95=[ratio - 1.96 * se, ratio + 1.96 * se],
                   z_LER=(acc["k"] - acc["sum_m"]) / math.sqrt(acc["sum_mvar"]) if acc["sum_mvar"] else float("nan"),
                   excess_xent_per_shot=(acc["sum_nll"] - acc["sum_H"]) / max(acc["n"], 1),
                   z_xent=(acc["sum_nll"] - acc["sum_H"]) / math.sqrt(acc["sum_xvar"]) if acc["sum_xvar"] else float("nan"),
                   expected_leaves=sum(1 for kk, rr in PR.items() if kk[0] == int(k[1]) and kk[3] == k.split("|")[1]
                                       for x in rr if x in W5))
    # ---- syndrome tests pooled (Stouffer over the leaves with r in W5)
    R["syndrome_pooled"] = {}
    for d in (3, 5):
        for e in ("ML.rl", "ML.si1000"):
            rows = [x for x in R["syndrome"] if x["d"] == d and x["entity"] == e and x["rounds"] in W5]
            if rows:
                R["syndrome_pooled"][f"d{d}|{e}"] = {"leaves": len(rows),
                    "stouffer_z_mean": float(np.sum([x["z_mean"] for x in rows]) / math.sqrt(len(rows))),
                    "stouffer_z_disp": float(np.sum([x["z_disp"] for x in rows]) / math.sqrt(len(rows))),
                    "median_var_ratio": float(np.median([x["var_ratio"] for x in rows]))}
    # ---- decomposition pooled (held-out halves of the fitted d = 3 leaves)
    dls = [x for x in R["decomposition"]["leaves"] if x["d"] == 3 and x["rounds"] in W5]
    expected_fits = sum(1 for j in (plan or {}).get("jobs", []) if j.get("kind") == "fit") or None
    failed = []
    for dp, dn, fn in os.walk(os.path.join(a.res, "fit")):
        if "fit.json" in fn:
            try:
                if json.load(open(os.path.join(dp, "fit.json"))).get("status") == "fit_failed":
                    failed.append(os.path.relpath(dp, os.path.join(a.res, "fit")))
            except Exception:
                failed.append(os.path.relpath(dp, os.path.join(a.res, "fit")) + " (unreadable)")
    R["decomposition"]["fits_failed"] = failed
    if dls:
        pooled = {"n_leaves": len(dls), "expected_leaves": expected_fits, "fits_failed": len(failed)}
        pooled["mismatch"] = ratio_from_counts([x["cells"]["mismatch"] for x in dls])      # P4: every fitted leaf
        lib = [x for x in dls if x["cells"].get("total") and x["cells"].get("suboptimality")]
        pooled["n_leaves_with_libra"] = len(lib)
        if lib:   # the identity total = suboptimality + mismatch, on the leaves with Libra predictions only
            tot = ratio_from_counts([x["cells"]["total"] for x in lib])
            mis_l = ratio_from_counts([x["cells"]["mismatch"] for x in lib])
            sub = ratio_from_counts([x["cells"]["suboptimality"] for x in lib])
            if tot and mis_l and sub:
                pooled["mismatch_libra_leaves"], pooled["total"] = mis_l, tot
                se = sub["se_rel"] * sub["errors_B"] / tot["errors_B"]     # (E_Libra - E_RL) / E_fitted, E_fitted fixed
                pooled["suboptimality"] = ci_block({"relative_to_ML_RL": sub}, tot["rel"] - mis_l["rel"], se)
        R["decomposition"]["pooled"] = pooled
    # ---- sensitivity: P1 and P4 without r = 13 (the RL prior was trained on the 13-cycle calibration data)
    sens, rows, had13 = {}, [], False
    for key, pts in pair_pts.items():
        if key.startswith("d3|") and key.endswith("|libra.rl vs ML.rl"):
            skey = key.rsplit("|", 1)[0]
            both = planned(skey, "libra.rl") & planned(skey, "ML.rl")
            had13 |= 13 in both
            f = paired_series(pts, tuple(sorted(x for x in both if x >= 10 and x != 13)))
            if f and f["complete"]:
                rows.append(f)
    if rows:
        sens["P1_libra_vs_ML_rl_d3"] = dict(pool_gap(rows), expected_series=expected_series[3])
    d13 = [x for x in R["decomposition"]["leaves"] if x["d"] == 3 and x["rounds"] in W5 and x["rounds"] != 13]
    if d13:
        sens["P4_mismatch_d3"] = ratio_from_counts([x["cells"]["mismatch"] for x in d13])
    R["sensitivity_excl_r13"] = sens
    R["sensitivity_excl_r13_note"] = ("" if had13 or not rows else
                                      "P1 is unchanged: there are no Libra predictions at r = 13, so P1 never used r = 13.")
    # ---- Lambda on W5 at every distance, and the paired Lambda ratio of Libra to ML(RL)
    R["lambda"] = {}
    for e in ("ML.rl", "ML.si1000", "libra.rl", "harmony.rl", "corrmatch.rl", "corr_mwpm.rl", "mwpm.rl"):
        eps = {}
        for d in (3, 5, 7):
            fs = [f for k, f in R["series_eps_W5"].items() if k.startswith(f"d{d}|") and k.endswith("|" + e) and f["complete"]]
            if d == 7:
                fs = [f for k, f in R["series_eps"].items() if k.startswith("d7|") and k.endswith("|" + e) and f["complete"]]
            if fs and len(fs) == expected_series[d]:
                eps[d] = mean_eps(fs)
        lf = lam_fit(eps)
        if lf:
            lf["window"] = "r in W5 (d = 7: probe rounds)"
            R["lambda"][e] = lf
    g3, g5 = R["pooled_gaps_W5"].get("d3|libra.rl vs ML.rl"), R["pooled_gaps_W5"].get("d5|libra.rl vs ML.rl")
    if g3 and g5 and g3["n_series"] == g3["expected_series"] and g5["n_series"] == g5["expected_series"]:
        lr = math.log(1 + g3["rel"]) - math.log(1 + g5["rel"])          # ln Lambda_Libra - ln Lambda_ML (d = 3 / 5)
        se = math.sqrt((g3["se_rel"] / (1 + g3["rel"])) ** 2 + (g5["se_rel"] / (1 + g5["rel"])) ** 2)
        R["lambda_paired_libra_over_ML_rl"] = {"ratio": math.exp(lr), "ln_ratio": lr, "se_ln": se,
                                               "ci95": [math.exp(lr - 1.96 * se), math.exp(lr + 1.96 * se)]}
    # ---- NN per series (Table S5) when the patch labels carry the grid label "a+bj"
    R["nn_by_series"] = {}
    for key in R["series_eps"]:
        d, patch, basis = key.split("|")[:3]
        m = re.search(r"(\d+)\+(\d+)j", patch or "")
        if m:
            v = reported["table_S5_nn_eps"].get(f"{d} {m.group(1)}+{m.group(2)}j", {}).get(basis)
            if v:
                R["nn_by_series"][f"{d}|{patch}|{basis}"] = v
    R["prereg"] = decisions(R, reported, ent_pts, expected_series, planned)
    json.dump(R, open(os.path.join(a.out, "analysis.json"), "w"), indent=1,
              default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    write_tables(R, a.out)
    made = [] if a.no_figures else figures(R, a.out, reported)
    print(f"analysis: {len(R['leaves'])} leaves, {len(R['series_eps'])} series fits, {len(R['pooled_gaps'])} pooled gaps, "
          f"figures {made}, {time.time() - t0:.0f} s")
    for k, v in R["prereg"].items():
        if isinstance(v, dict) and "decision" in v:
            print(f"  {k}: {v['decision']}")


def decisions(R, rep, ent_pts, expected_series, planned):
    out = {}
    S1 = rep["table_S1_fig1d"]
    # G0 integrity
    probs = list(R["integrity"]["problems"])
    mism, aborted = [], []
    for rel, L in R["leaves"].items():
        probs += [f"{rel}: {p}" for p in L["problems"]]
        for tag, rc in (L.get("recompute") or {}).items():
            if rc and (rc.get("det_mismatch_shots") or rc.get("obs_mismatch_shots")):
                mism.append(f"{rel} [{tag}]")
        for tag, st in L["folders"].items():
            if str(st).startswith("abort"):
                aborted.append(f"{rel} [{tag}]: {st}")
        for tag, b in (L.get("balance") or {}).items():
            if b and b.get("max_share_deviation", 0) > 0.1:
                probs.append(f"{rel} [{tag}]: selection unbalanced over bitstrings ({b['max_share_deviation']:.2f})")
    nan_bad = [x["leaf"] for x in R["calibration"]["leaves"] if x["n"] and x["nan_posteriors"] / x["n"] > 1e-4]
    found = {str(d): len({(v["patch"], v["basis"]) for v in R["leaves"].values() if v["d"] == d}) for d in (3, 5, 7)}
    d3_leaves = [v for v in R["leaves"].values() if v["d"] == 3]
    excl = len(set(x.split(" [")[0] for x in mism + aborted))
    g0_fail = bool(mism or nan_bad or probs) or bool(d3_leaves and excl > 0.1 * len(d3_leaves))
    out["G0_integrity"] = {"series_found": found, "series_planned": {str(k): v for k, v in expected_series.items()},
                           "recompute_mismatch": mism, "aborted": aborted, "nan_share_above_1e-4": nan_bad,
                           "problems": probs[:50], "decision": "pass" if not g0_fail else "FAIL (see lists)"}
    # G1: our fit of the shipped Libra predictions against Table S1 (d = 3: r >= 10, fallback all r; d = 5: W5)
    g1 = {}
    for d, windows in ((3, (("r>=10", lambda s: tuple(sorted(x for x in s if x >= 10))), ("all r", lambda s: tuple(sorted(s))))),
                       (5, (("W5", lambda s: tuple(x for x in W5 if x in s)),))):
        for wname, wf in windows:
            fs = []
            for k, pts in ent_pts.items():
                if k.startswith(f"d{d}|") and k.endswith("|libra.rl"):
                    f = series_eps(pts, wf(planned(k.rsplit("|", 1)[0], "libra.rl")))
                    if f and f["complete"]:
                        fs.append(f)
            if not fs or len(fs) < expected_series[d]:
                continue
            m, se = mean_eps(fs)
            rv, rs = S1["libra"][f"eps{d}"]
            z = (m - rv) / math.sqrt(se ** 2 + rs ** 2)
            g1.setdefault(str(d), {})[wname] = {"ours": m, "se": se, "reported": rv, "reported_se": rs, "z": z, "pass": abs(z) < 2}
    w3 = next((w for w in ("r>=10", "all r") if g1.get("3", {}).get(w, {}).get("pass")), None)
    w5 = "W5" if g1.get("5", {}).get("W5", {}).get("pass") else None
    out["G1_reproduce_libra"] = {"by_distance": g1, "window_d3": w3, "window_d5": w5,
                                 "decision": (f"pass at d = 3 ({w3})" if w3 else
                                              ("not evaluable (d = 3 Libra series incomplete)" if "3" not in g1 else
                                               "FAIL at d = 3: comparisons with published numbers (P5) suspended"))
                                             + ("" if "5" not in g1 else ("; d = 5 pass (W5)" if w5 else "; d = 5 FAIL"))}
    def full(p):
        return p is not None and p["n_series"] == p["expected_series"]
    # P1 decoder suboptimality (confirmatory): Libra vs ML(RL), d = 3, r >= 10
    p1 = R["pooled_gaps"].get("d3|libra.rl vs ML.rl")
    p1d5 = R["pooled_gaps"].get("d5|libra.rl vs ML.rl")
    # P4 mismatch cost (confirmatory): ML(RL) vs ML(fitted), held-out shots of the fitted d = 3 leaves
    dc = R["decomposition"].get("pooled")
    p4 = dc.get("mismatch") if dc else None
    # P4 uses every converged fit; it is evaluated when at least 90% of the planned fits are in
    p4_full = p4 is not None and (dc.get("expected_leaves") is None or dc["n_leaves"] >= 0.9 * dc["expected_leaves"])
    adj = holm({"P1": p1["p"] if full(p1) else None, "P4": p4["p"] if p4_full else None})
    def verdict(p, padj, pos, neg):
        if padj is None:
            return "not evaluable (incomplete)"
        if padj < 0.05:
            return pos if p["rel"] > 0 else neg
        return "no significant difference"
    ldiff = R.get("libra_dem_differs") or []
    out["P1_libra_vs_ML_rl_d3"] = {"pooled": p1, "p_holm": adj.get("P1"),
                                   "decision": (f"PROVISIONAL (Libra's RL DEM differs in content at {len(ldiff)} d = 3 leaves; "
                                                f"P1 needs ML with Libra's file, prereg section 2): " if ldiff else "")
                                               + verdict(p1, adj.get("P1"), "ML(RL) better than Libra", "Libra better than ML(RL)"),
                                   "d5_replication": None if not full(p1d5) else {
                                       "rel": p1d5["rel"], "ci90": p1d5["ci90"],
                                       "replicated": bool(p1 and (p1d5["rel"] > 0) == (p1["rel"] > 0) and
                                                          (p1d5["ci90"][0] > 0 or p1d5["ci90"][1] < 0))}}
    out["P4_mismatch_cost_d3"] = {"pooled": p4, "p_holm": adj.get("P4"), "n_leaves": dc["n_leaves"] if dc else 0,
                                  "expected_leaves": dc.get("expected_leaves") if dc else None,
                                  "fits_failed": R["decomposition"].get("fits_failed", []),
                                  "decision": verdict(p4, adj.get("P4"), "ML(fitted DEM) better than ML(RL DEM)",
                                                      "ML(RL DEM) better than ML(fitted DEM)")}
    # P2 (secondary): prior under ML
    p2 = R["pooled_gaps"].get("d3|ML.si1000 vs ML.rl")
    out["P2_ML_si1000_vs_ML_rl_d3"] = {"pooled": p2, "decision": "not evaluable (incomplete)" if not full(p2) else
                                       ("RL DEM better under ML (95% CI > 0)" if p2["ci95"][0] > 0 else
                                        ("SI1000 DEM better under ML (95% CI < 0)" if p2["ci95"][1] < 0 else "no significant difference"))}
    # P3 (secondary, enters K1): calibration of ML(RL) posteriors, d = 3, r in W5
    p3 = R["calibration"]["pooled"].get("d3|ML.rl")
    p3_full = p3 is not None and p3["leaves"] >= p3.get("expected_leaves", 0)
    out["P3_calibration_ML_rl_d3"] = {"pooled": p3, "decision": "not evaluable (incomplete)" if not p3_full else
                                      (f"calibrated within {100 * CAL_MARGIN:.0f}% (90% CI inside [{1 - CAL_MARGIN}, {1 + CAL_MARGIN}])"
                                       if 1 - CAL_MARGIN <= p3["ci90"][0] and p3["ci90"][1] <= 1 + CAL_MARGIN
                                       else "not shown calibrated within 5%")}
    # P5 (secondary, approximate): ML(RL) against the reported NN mean eps, in the window that passed G1
    p5 = {}
    for d, w in ((3, w3), (5, w5)):
        if not w:
            continue
        fs = []
        for k, pts in ent_pts.items():
            if k.startswith(f"d{d}|") and k.endswith("|ML.rl"):
                skey = k.rsplit("|", 1)[0]     # the G1 window: the rounds of this series that have Libra predictions
                s = planned(skey, "ML.rl") & planned(skey, "libra.rl")
                win = (tuple(sorted(x for x in s if x >= 10)) if w == "r>=10" else
                       (tuple(sorted(s)) if w == "all r" else tuple(x for x in W5 if x in s)))
                f = series_eps(pts, win)
                if f and f["complete"]:
                    fs.append(f)
        if fs and len(fs) == expected_series[d]:
            m, se = mean_eps(fs)
            nv, ns = S1["nn"][f"eps{d}"]
            p5[str(d)] = ci_block({"eps_ML_rl": m, "se": se, "eps_NN": nv, "se_NN": ns, "window": w},
                                  (m - nv) / nv, math.sqrt(se ** 2 + ns ** 2) / nv)
    q = p5.get("3")
    out["P5_ML_rl_vs_NN"] = {"pooled": p5, "decision": "not evaluable" if not q else
                             ("NN better than ML(RL) (95% CI > 0)" if q["ci95"][0] > 0 else
                              ("ML(RL) better than NN (95% CI < 0)" if q["ci95"][1] < 0 else "no significant difference"))}
    # K1 nothing to decompose: Libra = ML(RL) and ML(RL) = ML(fitted) within 3%, RL DEM calibrated within 5%
    inside = lambda ci, lo, hi: lo <= ci[0] and ci[1] <= hi
    k1 = {"P1_within_3pct": inside(p1["ci90"], -MARGIN, MARGIN) if full(p1) else None,
          "P4_within_3pct": inside(p4["ci90"], -MARGIN, MARGIN) if p4_full else None,
          "P3_within_5pct": inside(p3["ci90"], 1 - CAL_MARGIN, 1 + CAL_MARGIN) if p3_full else None}
    if any(v is None for v in k1.values()):
        k1["decision"] = "not evaluable yet"
    elif all(k1.values()):
        k1["decision"] = "KILL: nothing to decompose (Libra is ML-optimal for its prior; the prior is calibrated and not improved by the fit)"
    elif k1["P1_within_3pct"] and k1["P4_within_3pct"]:
        k1["decision"] = "no kill; both terms within 3% but the RL DEM is miscalibrated: pivot to the model-check result"
    else:
        k1["decision"] = "no kill"
    out["K1_nothing_to_decompose"] = k1
    # K2 d = 5 cost and convergence (per leaf: error count within max(3, 1%) between rungs, max |dP| below the band)
    lad = [x for x in R["ladder"] if x["d"] == 5]
    if lad:
        first = sorted(lad, key=lambda x: (x["rounds"], x["leaf"]))[0]
        cost_ok = first["ms_per_round_per_shot_tau1"] <= K2_COST_MS
        def band(x):
            try:
                return float(x["decoder"].split(":")[2])
            except Exception:
                return 0.1
        bad = [x["leaf"] for x in lad if abs(x["errors_final"] - x["errors_tau1"]) > max(3, 0.01 * x["errors_final"])
               or x["max_abs_dP_redecoded"] >= band(x)]
        out["K2_d5_cost_convergence"] = {"first_leaf": first["leaf"], "ms_per_round_per_shot_tau1": first["ms_per_round_per_shot_tau1"],
                                         "cost_ok": cost_ok, "leaves": len(lad), "leaves_failing": bad,
                                         "decision": ("KILL d = 5 ML: too expensive" if not cost_ok else
                                                      ("KILL d = 5 ML unless one extra rung fixes the failing leaves"
                                                       if len(bad) > 0.25 * len(lad) else
                                                       ("pass" if not bad else "pass; failing leaves excluded and flagged")))}
    else:
        out["K2_d5_cost_convergence"] = {"decision": "not evaluable yet"}
    # the bar
    ok0 = out["G0_integrity"]["decision"] == "pass"
    t1 = full(p1) and adj.get("P1") is not None and adj["P1"] < 0.05 and p1["rel"] >= MARGIN
    t4 = p4_full and adj.get("P4") is not None and adj["P4"] < 0.05 and p4["rel"] >= MARGIN
    if adj.get("P1") is None and adj.get("P4") is None:
        bar = "not evaluable yet"
    elif not (t1 or t4):
        bar = "does not clear the bar"
    elif not ok0:
        bar = "blocked: G0 failed"
    elif not w3:
        bar = "blocked until G1 is explained: P1/P4 would clear the bar, but our Libra eps does not reproduce Table S1"
    elif t1 and not t4 and ldiff:
        bar = "provisional: only P1 clears the bar, and P1 needs ML with Libra's own DEM file (prereg section 2)"
    else:
        bar = "clears the bar at d = 3"
    out["bar"] = {"G0": ok0, "G1_d3": bool(w3), "P1_significant_and_ge_3pct": bool(t1), "P4_significant_and_ge_3pct": bool(t4),
                  "decision": bar}
    return out


def fmt(x, k=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{k}g}"


def md_table(head, rows):
    cell = lambda x: str(x).replace("|", " · ")
    s = "| " + " | ".join(cell(h) for h in head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        s += "| " + " | ".join(cell(x) for x in r) + " |\n"
    return s


def same_prior(pair):
    a, b = pair.split(" vs ")
    return a.split(".")[-1] == b.split(".")[-1] or (a == "ML.si1000" and b == "ML.rl")


def write_csv(path, head, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(rows)


def pct(v):
    return f"{100 * v['rel']:.2f}", f"[{100 * v['ci95'][0]:.2f}, {100 * v['ci95'][1]:.2f}]"


def write_tables(R, out):
    s = "# Willow real-data analysis (generated by analyze_willow.py)\n\n"
    s += f"Made {R['made']}. Leaves analysed: {len(R['leaves'])}. Plan: {R['plan']}. Windows: {R['windows']}.\n\n"
    s += "## Pre-registered decisions (notes/decoder/willow_prereg.md)\n\n"
    for k, v in R["prereg"].items():
        s += f"- **{k}**: {v.get('decision')}\n"
    s += "\n## Mean eps per distance (complete series only)\n\n"
    s += md_table(["distance, entity", "series", "of", "mean eps", "se"],
                  [(k, v["n_series"], v["expected_series"], fmt(v["eps"], 4), fmt(v["se"], 2)) for k, v in sorted(R["mean_eps"].items())])
    s += "\nGap tables list A against ML with the same DEM, and ML(SI1000) against ML(RL); every other pair is in analysis.json\n"
    s += "and paired_gaps.csv.\n"
    for name, key in (("r >= 10 at d = 3, W5 at d = 5", "pooled_gaps"), ("W5 at every distance", "pooled_gaps_W5")):
        s += f"\n## Pooled paired eps gaps to ML, {name} (positive = ML better)\n\n"
        s += md_table(["distance, A vs ML", "series", "of", "gap %", "95% CI %", "p"],
                      [(k, v["n_series"], v["expected_series"], *pct(v), fmt(v["p"], 2)) for k, v in sorted(R[key].items())
                       if same_prior(k.split("|", 1)[1])])
    s += "\n## Pooled error counts, every paired leaf with r >= 10 (secondary; positive = ML better)\n\n"
    s += md_table(["distance, A vs ML", "leaves", "errors A", "errors ML", "rel %", "95% CI %"],
                  [(k, v["leaves"], v["errors_A"], v["errors_B"], *pct(v)) for k, v in sorted(R["pooled_counts"].items())
                   if same_prior(k.split("|", 1)[1])])
    s += "\n## Calibration of ML posteriors, pooled over r in W5\n\n"
    s += md_table(["distance, DEM", "leaves", "obs/pred", "90% CI", "z_LER", "excess xent/shot", "z_xent"],
                  [(k, v["leaves"], fmt(v["ratio"], 4), f"[{fmt(v['ci90'][0], 4)}, {fmt(v['ci90'][1], 4)}]", fmt(v["z_LER"]),
                    fmt(v["excess_xent_per_shot"]), fmt(v["z_xent"])) for k, v in sorted(R["calibration"]["pooled"].items())])
    if R["syndrome_pooled"]:
        s += "\n## Syndrome log-likelihood tests (Stouffer over leaves, r in W5)\n\n"
        s += md_table(["distance, DEM", "leaves", "z mean", "z dispersion", "median var ratio"],
                      [(k, v["leaves"], fmt(v["stouffer_z_mean"]), fmt(v["stouffer_z_disp"]), fmt(v["median_var_ratio"]))
                       for k, v in sorted(R["syndrome_pooled"].items())])
    if R["lambda"]:
        s += "\n## Lambda (r in W5 at every distance)\n\n"
        s += md_table(["entity", "distances", "Lambda", "se"],
                      [(k, v["distances"], fmt(v["Lambda"], 4), fmt(v["se"], 2)) for k, v in R["lambda"].items()])
        if R.get("lambda_paired_libra_over_ML_rl"):
            v = R["lambda_paired_libra_over_ML_rl"]
            s += f"\nPaired Lambda_Libra / Lambda_ML(RL), d = 3 / 5: {v['ratio']:.4f}, 95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]\n"
    dc = R["decomposition"].get("pooled")
    if dc:
        s += (f"\n## Decomposition on held-out shots ({dc['n_leaves']} fitted leaves, {dc.get('n_leaves_with_libra', 0)} of them "
              f"with Libra predictions; relative to ML(fitted DEM))\n\n")
        lab = {"mismatch": "mismatch m (P4), all fitted leaves", "mismatch_libra_leaves": "mismatch, leaves with Libra",
               "suboptimality": "suboptimality (Libra vs ML(RL)), leaves with Libra", "total": "total (Libra vs ML(fitted)), leaves with Libra"}
        s += md_table(["term", "relative", "95% CI"], [(lab.get(k, k), f"{100 * v['rel']:.2f}%", f"[{100 * v['ci95'][0]:.2f}, {100 * v['ci95'][1]:.2f}]%")
                                                       for k, v in dc.items() if isinstance(v, dict) and "rel" in v])
        s += "\n" + md_table(["leaf", "classes", "held-out LER shipped", "fitted", "calib ratio shipped", "fitted",
                              "synd z mean shipped", "fitted"],
                             [(x["leaf"].split("/", 1)[-1], x["n_classes"], fmt(x["LER_shipped"], 4), fmt(x["LER_fitted"], 4),
                               fmt(x["calibration"]["shipped"]["ratio_obs_over_pred"]), fmt(x["calibration"]["fitted"]["ratio_obs_over_pred"]),
                               fmt(x["syndrome_loglik"]["shipped"]["z_mean"]), fmt(x["syndrome_loglik"]["fitted"]["z_mean"]))
                              for x in R["decomposition"]["leaves"]])
    if R.get("sensitivity_excl_r13"):
        s += "\n## Sensitivity: without r = 13 (RL prior trained on 13-cycle calibration data)\n\n"
        if R.get("sensitivity_excl_r13_note"):
            s += R["sensitivity_excl_r13_note"] + "\n\n"
        s += md_table(["test", "rel %", "95% CI %"], [(k, *pct(v)) for k, v in R["sensitivity_excl_r13"].items() if v])
    if R["ladder"]:
        s += "\n## Tau ladder (d >= 5)\n\n"
        s += md_table(["leaf", "n", "re-decoded", "decisions changed", "errors tau1 -> final", "max |dP|", "ms/round/shot tau1"],
                      [(x["leaf"].split("/", 1)[-1], x["n"], x["n_redecoded"], x["decisions_changed"],
                        f"{x['errors_tau1']} -> {x['errors_final']}", fmt(x["max_abs_dP_redecoded"]),
                        fmt(x["ms_per_round_per_shot_tau1"])) for x in R["ladder"]])
    s += "\n## Eps per series and entity (primary window)\n\n"
    s += md_table(["series, entity", "eps", "se", "chi2_red", "points", "complete"],
                  [(k, fmt(v["eps"], 4), fmt(v["se_eps"], 2), fmt(v["chi2_red"]), v["npts"], v["complete"])
                   for k, v in sorted(R["series_eps"].items())])
    open(os.path.join(out, "report.md"), "w").write(s)
    json.dump(R["prereg"], open(os.path.join(out, "prereg_results.json"), "w"), indent=1, default=str)
    write_csv(os.path.join(out, "leaf_table.csv"), ["leaf", "d", "patch", "basis", "rounds", "entity", "n", "errors", "LER", "eps_one_point"],
              [(rel, L["d"], L["patch"], L["basis"], L["rounds"], e, v["n"], v["errors"], v["LER"], v["eps_1pt"])
               for rel, L in sorted(R["leaves"].items()) for e, v in sorted(L["entities"].items())])
    write_csv(os.path.join(out, "series_eps.csv"), ["series_entity", "window", "eps", "se_eps", "chi2_red", "npts", "rounds", "complete"],
              [(k, w, v["eps"], v["se_eps"], v["chi2_red"], v["npts"], " ".join(map(str, v["rounds"])), v["complete"])
               for w, src in (("primary", "series_eps"), ("W5", "series_eps_W5")) for k, v in sorted(R[src].items())])
    write_csv(os.path.join(out, "paired_gaps.csv"), ["series_pair", "window", "eps_A", "eps_ML", "rel", "se_rel", "chi2_red", "complete"],
              [(k, w, v["eps_A"], v["eps_ML"], v["rel"], v["se_rel"], v["chi2_red"], v["complete"])
               for w, src in (("primary", "paired_series"), ("W5", "paired_series_W5")) for k, v in sorted(R[src].items())])
    write_csv(os.path.join(out, "calibration_leaves.csv"), ["leaf", "entity", "rounds", "n", "errors", "predicted", "ratio", "z_LER",
                                                            "z_xent", "nan_posteriors"],
              [(x["leaf"], x["entity"], x["rounds"], x["n"], x["errors"], x["predicted"], x["ratio"], x["z_LER"], x["z_xent"],
                x["nan_posteriors"]) for x in R["calibration"]["leaves"]])


if __name__ == "__main__":
    main()
