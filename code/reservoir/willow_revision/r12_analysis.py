"""R1's certificate and R2's analysis of the revision (review 1.4). POST HOC revision analyses, not pre-registered
tests; declared in notes/reservoir/willow_revision_declarations.md (R1, R2 at 15:53Z; this script's md5 in a later
section) before they were run on R1's and R2's outputs.

reservoir, 2026-10-13. Reads decoder's stored outputs, stage 1/2 of the exact test (willow_gapmech), R1's records
(r1_certify.py run), R2's max-product outputs (r2_semiring.py run) and R2's resolved disagreements (r1_certify.py
resolve); any of the last three may be missing, and the parts that need them are then skipped. The estimators are the
published ones, imported unchanged from exact_analysis.py / gapmech.py / d5_check.py (prereg section 4, entry 7's SE).

R1 (r1_certificate.json): per r and for {10, 13, 30}, for stage 2, phase 1, phase 2 and their union: shots, proven,
misses (miss + miss_undetermined), ties, unproven, the one-sided 95% Clopper-Pearson bound and the conservative bound
(unproven as misses); u/D with D the per-shot Tesseract - ML difference on Tesseract's shots (rule A and no exclusion);
the worst case the bound permits: every leaf's MPE error count moved by ceil(u_r n_leaf) in one direction, g5(MPE|ML)
and Lambda_MPE/Lambda_ML recomputed (d = 3 exact, unchanged), on {10, 13, 30} and {10, 30}; the same with stage 2's
bound alone; the class of R2's max-product ladder against the proven optimum on the certified shots.
R2 (r2_analysis.json):
  (a) the controlled degeneracy factor Lambda_MPE/Lambda_ML: d = 5 MPE = R2's max-product ladder (mpe_tau) or the
      stored Tesseract on the same shots; d = 3 MPE = stage 1's exact MPE; windows {10, 13, 30}, {10, 30}, {10, 30, 50};
      rule A and no exclusion;
  (b) mpe_tau against Tesseract per shot (class agreement per r; the resolved disagreements);
  (c) Libra's share ln(Lambda_MPE/Lambda_ML) / ln(Lambda_Libra/Lambda_ML) with exact_analysis's bootstrap (B = 10,000,
      seed 20261010, a fresh generator per MPE variant and window, rule A then no exclusion), with Tesseract (the
      published numbers) and with mpe_tau, on {10, 30} and (mpe_tau) {10, 30, 50};
  (d) shot-level concordance of Libra's and the MPE's failures: among the shots ML gets right, b_A = Libra wrong and
      b_M = MPE wrong; among those ML gets wrong, c_A = Libra right and c_M = MPE right; shares, Jaccard index and the
      Mantel-Haenszel odds ratio across leaves with the Robins-Breslow-Greenland 95% CI; and exact_analysis's net-loss
      categories; as a reference for how much of that concordance any two decoders share, the same statistics for
      Libra against Harmony, correlated matching, correlated MWPM and MWPM (rule A at d = 5);
  (e) the ladder at d = 3 on every 25th shot: mpe_tau against the exact MPE, ml_tau (mode 0) against exact ML; the
      largest retained-state counts of the d = 5 max-product runs;
  (f) parse_dem's merge count of every DEM R2 used.
Usage: python3 -I -B r12_analysis.py ARCHIVE DECODE K2_DIR STAGE_DIR R1_RECORDS R2_WORK RESOLVE_RECORDS OUT_DIR
       (a missing file or folder counts as absent; outputs through qrc.outpaths)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_gapmech"))
from exact_analysis import boot_arrays, bootstrap, categories, mask_of, ratio, series_fits  # noqa: E402
from gapmech import MISSING, gsum, growth, read_leaf  # noqa: E402
from d5_check import BASES, P3, P5, k2_verdicts, paired, rel_of  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

W5 = [10, 13, 30, 50]
WINDOWS = {"10_13_30": [10, 13, 30], "10_30": [10, 30], "10_30_50": [10, 30, 50]}
B_BOOT, SEED_BOOT = 10000, 20261010
Z95 = 1.959963984540054


def cp_upper(k, n):
    return float(stats.beta.ppf(0.95, k + 1, n - k)) if n and k < n else (1.0 if n else None)


def read_jsonl(p):
    out = []
    if p and os.path.isfile(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def load_leaves(arch, dec, stage_dir, r2_work, chk):
    leaves, r2meta = {}, {}
    for d, pats in ((3, P3), (5, P5)):
        for p in pats:
            for b in BASES:
                for r in W5:
                    rec = read_leaf(arch, dec, d, p, b, r, chk)
                    rel = rel_of(d, p, b, r)
                    n = rec["n"]
                    pos = (lambda i: i) if d == 3 else (lambda i: i // 5)
                    if d == 3:
                        f = os.path.join(stage_dir, "exact", rel.replace("/", "_") + ".npz")
                        if os.path.exists(f):
                            z = np.load(f, allow_pickle=False)
                            full = np.full(n, MISSING, np.uint8)
                            full[pos(z["idx"].astype(np.int64))] = z["mpe"]
                            rec["preds"]["mpe_exact"] = full
                    wd = os.path.join(r2_work, rel.replace("/", "_")) if r2_work else None
                    for m, name in ((1, "mpe_tau"), (0, "ml_tau")):
                        f = os.path.join(wd, f"m{m}.npz") if wd else None
                        if f and os.path.exists(f):
                            z = np.load(f, allow_pickle=False)
                            ii = z["idx"].astype(np.int64)
                            if d == 5 and np.any(ii % 5):
                                raise SystemExit(f"{rel}: R2 shots not on ML's every-5th grid")
                            full = np.full(n, MISSING, np.uint8)
                            full[pos(ii)] = z["pred"].astype(np.uint8)
                            rec["preds"][name] = full
                            rec.setdefault("r2", {})[name] = {"shots": int(ii.size),
                                                              "maxk_t1": int(z["maxk_t1"].max()),
                                                              "maxk_t2": int(z["maxk_t2"].max()),
                                                              "redecoded": int(z["redecoded"].sum()),
                                                              "nan": int(np.isnan(z["llr"]).sum()),
                                                              "cpu_s": float(z["sec_t1"].sum() + z["sec_t2"].sum())}
                    if wd and os.path.exists(os.path.join(wd, "meta.json")):
                        r2meta[rel] = json.load(open(os.path.join(wd, "meta.json")))
                    leaves[(d, p, b, r)] = rec
    return leaves, r2meta


def set_mpe(leaves, d5_variant):
    """preds['mpe']: stage 1's exact MPE at d = 3; the named variant at d = 5 (removed where it is absent)."""
    for (d, p, b, r), rec in leaves.items():
        src = "mpe_exact" if d == 3 else d5_variant
        if src in rec["preds"]:
            rec["preds"]["mpe"] = rec["preds"][src]
        else:
            rec["preds"].pop("mpe", None)


def window_ok(leaves, need, rounds):
    return all(all(z in leaves[(d, p, b, r)]["preds"] for z in need)
               for d, pats in ((3, P3), (5, P5)) for p in pats for b in BASES for r in rounds)


# ---------------------------------------------------------------- R1
def r1_certificate(leaves, failing, stage_dir, r1_records):
    s2 = json.load(open(os.path.join(stage_dir, "exact_stage2.json")))["records"]
    for x in s2:
        x.setdefault("phase", 0)
    r1 = read_jsonl(r1_records)
    keys = set()
    recs = []
    for x in s2 + r1:
        k = (x["leaf"], int(x["idx"]))
        if k in keys:
            continue
        keys.add(k)
        recs.append(x)
    out = {"records": {"stage2": len(s2), "r1": len(r1), "union_unique": len(recs)}, "summary": {}}

    def summ(R):
        n = sum(x["proven"] for x in R)
        k = sum(x["miss"] for x in R)
        u = len(R) - n
        return {"shots_run": len(R), "proven": n, "unproven": u, "misses": k,
                "verdicts": {v: sum(x["verdict"] == v for x in R) for v in ("same_class", "tie", "miss",
                                                                         "miss_undetermined", "unproven")},
                "miss_rate_upper95_one_sided_clopper_pearson": cp_upper(k, n),
                "conservative_unproven_as_misses": {"k": k + u, "n": len(R), "upper95": cp_upper(k + u, len(R))},
                "milp_cpu_s": sum(x["milp"]["cpu_s"] + x.get("milp_in_stored_class", {}).get("cpu_s", 0.0) for x in R)}
    for tag, sel in (("stage2", lambda x: x["phase"] == 0), ("phase1", lambda x: x["phase"] == 1),
                     ("phase2", lambda x: x["phase"] == 2), ("union", lambda x: True)):
        out["summary"][tag] = {f"r{r}": summ([x for x in recs if sel(x) and x["r"] == r]) for r in (10, 13, 30)}
        out["summary"][tag]["all"] = summ([x for x in recs if sel(x)])
    # leverage: D per r on Tesseract's shots
    lev = {}
    for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
        for r in (10, 13, 30):
            n = km = kl = 0
            for (d, p, b, rr), rec in leaves.items():
                if d != 5 or rr != r or rel_of(d, p, b, rr) in excl or "tesseract" not in rec["preds"]:
                    continue
                m = mask_of(rec, ["tesseract"])
                n += int(m.sum())
                km += int(np.sum((rec["preds"]["tesseract"] != rec["obs"]) & m))
                kl += int(np.sum((rec["ml"] != rec["obs"]) & m))
            D = (km - kl) / n
            lev[f"{kname}|r{r}"] = {"shots": n, "k_mpe": km, "k_ml": kl, "D_per_shot": D}
            for tag in ("stage2", "union"):
                u = out["summary"][tag][f"r{r}"]["miss_rate_upper95_one_sided_clopper_pearson"]
                uc = out["summary"][tag][f"r{r}"]["conservative_unproven_as_misses"]["upper95"]
                lev[f"{kname}|r{r}"][f"u_{tag}"] = u
                lev[f"{kname}|r{r}"][f"u_over_D_{tag}"] = u / D if (u is not None and D) else None
                lev[f"{kname}|r{r}"][f"u_conservative_{tag}"] = uc
                lev[f"{kname}|r{r}"][f"sign_of_D_kept_{tag}"] = bool(u is not None and u < D)
    out["leverage"] = lev
    # worst case
    wc = {}
    for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
        for wname in ("10_13_30", "10_30"):
            rounds = WINDOWS[wname]
            f3 = series_fits(leaves, "mpe_exact", "ml", ["mpe_exact"], 3, rounds, set())
            g3 = gsum(list(f3.values()))
            o = {"g3_exact": g3["g"], "series_d3": len(f3)}
            for tag in ("stage2", "union"):
                u_r = {r: out["summary"][tag][f"r{r}"]["miss_rate_upper95_one_sided_clopper_pearson"] for r in rounds}
                if any(v is None for v in u_r.values()):
                    continue
                res = {}
                for sgn, lab in ((0, "nominal"), (-1, "mpe_better"), (+1, "mpe_worse")):
                    prs = []
                    for p in P5:
                        for b in BASES:
                            L = []
                            for r in rounds:
                                rec = leaves[(5, p, b, r)]
                                if rel_of(5, p, b, r) in excl or "tesseract" not in rec["preds"]:
                                    continue
                                m = mask_of(rec, ["tesseract"])
                                aw = (rec["preds"]["tesseract"] != rec["obs"]) & m
                                mw = (rec["ml"] != rec["obs"]) & m
                                n = int(m.sum())
                                kA, kM, n11 = int(aw.sum()), int(mw.sum()), int(np.sum(aw & mw))
                                dl = math.ceil(u_r[r] * n) * sgn
                                kA2 = max(kA + dl, 0)
                                L.append((r, kA2, kM, min(n11, kA2), n))
                            if len(L) < 2:
                                continue
                            prs.append(paired([x[0] for x in L], [x[1] for x in L], [x[2] for x in L],
                                              [x[3] for x in L], [x[4] for x in L]))
                    g5 = gsum(prs)
                    res[lab] = {"g5": g5["g"], "series_d5": len(prs),
                                "lambda_mpe_over_ml": (1 + g3["g"]) / (1 + g5["g"]),
                                "growth_kept": bool(g5["g"] > g3["g"]), "g5_positive": bool(g5["g"] > 0)}
                    if sgn == 0:
                        res[lab].update(growth(g3, g5))
                o[tag] = {"u_per_r": u_r, **res}
            wc[f"{wname}|{kname}"] = o
    out["worst_case"] = wc
    # R2's max-product class on the certified shots
    agree = {"shots": 0, "with_mpe_tau": 0, "mpe_tau_equals_optimum_class": 0, "mpe_tau_differs": [],
             "tesseract_equals_optimum_class": 0}
    for x in recs:
        if not x["proven"]:
            continue
        r, rel = x["r"], x["leaf"]
        p, b = rel.split("/")[0][len("d5_at_"):], rel.split("/")[1]
        rec = leaves[(5, p, b, r)]
        agree["shots"] += 1
        agree["tesseract_equals_optimum_class"] += int(x["stored_tesseract_class"] == x["milp"]["cls"])
        if "mpe_tau" not in rec["preds"]:
            continue
        c = int(rec["preds"]["mpe_tau"][int(x["idx"]) // 5])
        if c == MISSING:
            continue
        agree["with_mpe_tau"] += 1
        if c == x["milp"]["cls"]:
            agree["mpe_tau_equals_optimum_class"] += 1
        else:
            agree["mpe_tau_differs"].append({"leaf": rel, "idx": int(x["idx"]), "mpe_tau": c, "optimum": x["milp"]["cls"],
                                             "tesseract": x["stored_tesseract_class"], "verdict": x["verdict"]})
    out["mpe_tau_on_certified_shots"] = agree
    return out


# ---------------------------------------------------------------- R2
def mh_or(tables):
    """Mantel-Haenszel odds ratio over 2x2 tables (n11, n10, n01, n00) with the Robins-Breslow-Greenland CI."""
    R = S = PR = PSQR = QS = 0.0
    for n11, n10, n01, n00 in tables:
        N = n11 + n10 + n01 + n00
        if N == 0:
            continue
        Ri, Si = n11 * n00 / N, n10 * n01 / N
        Pi, Qi = (n11 + n00) / N, (n10 + n01) / N
        R += Ri
        S += Si
        PR += Pi * Ri
        PSQR += Pi * Si + Qi * Ri
        QS += Qi * Si
    if R == 0 or S == 0:
        return {"or_mh": None, "ci95": None}
    orv = R / S
    var = PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    return {"or_mh": orv, "ln_se": se, "ci95": [orv * math.exp(-Z95 * se), orv * math.exp(Z95 * se)]}


def concordance(leaves, a, d, rounds, excl, b2="mpe"):
    """Libra (a) against the MPE, or against another decoder b2 as a reference for how much concordance is generic."""
    T_b, T_c = [], []
    tot = {"shots": 0, "ml_right": 0, "ml_wrong": 0, "bA": 0, "bM": 0, "bA_and_bM": 0, "cA": 0, "cM": 0,
           "cA_and_cM": 0, "a_wrong": 0, "m_wrong": 0, "both_wrong": 0, "leaves": 0}
    for (dd, p, b, r), rec in leaves.items():
        if dd != d or r not in rounds or rel_of(d, p, b, r) in excl or a not in rec["preds"] or b2 not in rec["preds"]:
            continue
        m = mask_of(rec, [a, b2])
        A = rec["preds"][a][m] != rec["obs"][m]
        M = rec["preds"][b2][m] != rec["obs"][m]
        L = rec["ml"][m] != rec["obs"][m]
        tot["leaves"] += 1
        tot["shots"] += int(m.sum())
        tot["ml_right"] += int((~L).sum())
        tot["ml_wrong"] += int(L.sum())
        tot["a_wrong"] += int(A.sum())
        tot["m_wrong"] += int(M.sum())
        tot["both_wrong"] += int((A & M).sum())
        bA, bM = A & ~L, M & ~L
        cA, cM = ~A & L, ~M & L
        for k, v in (("bA", bA), ("bM", bM), ("bA_and_bM", bA & bM), ("cA", cA), ("cM", cM), ("cA_and_cM", cA & cM)):
            tot[k] += int(v.sum())
        nr = ~L
        T_b.append((int((bA & bM).sum()), int((bA & ~bM).sum()), int((~bA & bM & nr).sum()),
                    int((~bA & ~bM & nr).sum())))
        T_c.append((int((cA & cM).sum()), int((cA & ~cM).sum()), int((~cA & cM & L).sum()),
                    int((~cA & ~cM & L).sum())))
    if tot["leaves"] == 0:
        return None
    sh = lambda x, y: x / y if y else None  # noqa: E731
    tot.update(share_of_bA_also_bM=sh(tot["bA_and_bM"], tot["bA"]), share_of_bM_also_bA=sh(tot["bA_and_bM"], tot["bM"]),
               jaccard_b=sh(tot["bA_and_bM"], tot["bA"] + tot["bM"] - tot["bA_and_bM"]),
               share_of_cA_also_cM=sh(tot["cA_and_cM"], tot["cA"]), share_of_cM_also_cA=sh(tot["cA_and_cM"], tot["cM"]),
               jaccard_c=sh(tot["cA_and_cM"], tot["cA"] + tot["cM"] - tot["cA_and_cM"]),
               net_A=tot["bA"] - tot["cA"], net_MPE=tot["bM"] - tot["cM"],
               mh_b_among_ml_right=mh_or(T_b), mh_c_among_ml_wrong=mh_or(T_c))
    return tot


def degeneracy_factor(leaves, excl, rounds):
    f3 = series_fits(leaves, "mpe", "ml", ["mpe"], 3, rounds, set())
    f5 = series_fits(leaves, "mpe", "ml", ["mpe"], 5, rounds, excl)
    return ratio(f3, f5)


def share_block(leaves, a, rounds, failing):
    rng = np.random.default_rng(SEED_BOOT)
    res = {}
    for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
        fits3, fits5 = {}, {}
        for t, x, y in (("A|ML", a, "ml"), ("MPE|ML", "mpe", "ml"), ("A|MPE", a, "mpe")):
            fits3[t] = series_fits(leaves, x, y, [a, "mpe"], 3, rounds, set())
            fits5[t] = series_fits(leaves, x, y, [a, "mpe"], 5, rounds, excl)
        o = {t: ratio(fits3[t], fits5[t]) for t in fits3}
        keys3 = sorted(set.intersection(*(set(fits3[t]) for t in fits3)))
        keys5 = sorted(set.intersection(*(set(fits5[t]) for t in fits5)))
        if all("lambda_ratio" in o[t] for t in o):
            la, lm = math.log(o["A|ML"]["lambda_ratio"]), math.log(o["MPE|ML"]["lambda_ratio"])
            bt = bootstrap(boot_arrays(fits3, fits5, keys3, keys5), rng, B_BOOT)
            o["share"] = {"estimate": lm / la if la else None, "ci95_bootstrap": bt["share_ci95"],
                          "bootstrap_nonfinite": bt["share_nonfinite"]}
            o["bootstrap_lambda_ci95"] = bt["lambda_ci95"]
        o["series_common"] = {"d3": len(keys3), "d5": len(keys5)}
        o["categories"] = {"d3": categories(leaves, a, 3, rounds, set()), "d5": categories(leaves, a, 5, rounds, excl)}
        res[kname] = o
    return res


def r2_analysis(leaves, failing, r2meta, resolve_records):
    out = {"factor": {}, "share_libra": {}, "concordance": {}, "concordance_reference": {}, "mpe_tau_vs_tesseract": {},
           "d3_ladder_check": {}, "retained_max": {}, "merges": {}, "r2_cpu_s": {}}
    for wname in ("10_30", "10_30_50"):
        for b2 in ("harmony", "corrmatch", "corr_mwpm", "mwpm"):
            for d in (3, 5):
                c = concordance(leaves, "libra", d, WINDOWS[wname], failing if d == 5 else set(), b2=b2)
                if c:
                    out["concordance_reference"][f"{b2}|{wname}|d{d}|ruleA"] = c
    have_tau = any("mpe_tau" in rec["preds"] for (d, *_), rec in leaves.items() if d == 5)
    variants = ["tesseract"] + (["mpe_tau"] if have_tau else [])
    for v in variants:
        set_mpe(leaves, v)
        for wname, rounds in WINDOWS.items():
            if not window_ok(leaves, ["mpe"], rounds):
                continue
            for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
                out["factor"][f"{v}|{wname}|{kname}"] = degeneracy_factor(leaves, excl, rounds)
            if wname in ("10_30", "10_30_50") and window_ok(leaves, ["libra", "mpe"], rounds):
                out["share_libra"][f"{v}|{wname}"] = share_block(leaves, "libra", rounds, failing)
            if wname == "10_13_30":
                continue  # Libra has no r = 13: its concordance window is {10, 30} (below)
            for d in (3, 5):
                for kname, excl in (("ruleA", failing), ("no_exclusion", set())):
                    if d == 3 and kname == "no_exclusion":
                        continue
                    c = concordance(leaves, "libra", d, rounds, excl if d == 5 else set())
                    if c:
                        out["concordance"][f"{v}|{wname}|d{d}|{kname}"] = c
    # (b) mpe_tau against Tesseract
    if have_tau:
        for r in (10, 13, 30):
            n = dif = 0
            per = {}
            for (d, p, b, rr), rec in leaves.items():
                if d != 5 or rr != r or "mpe_tau" not in rec["preds"] or "tesseract" not in rec["preds"]:
                    continue
                m = mask_of(rec, ["mpe_tau", "tesseract"])
                k = int(np.sum(rec["preds"]["mpe_tau"][m] != rec["preds"]["tesseract"][m]))
                n += int(m.sum())
                dif += k
                per[rel_of(5, p, b, r)] = k
            out["mpe_tau_vs_tesseract"][f"r{r}"] = {"shots": n, "class_differs": dif, "per_leaf": per}
        res = read_jsonl(resolve_records)
        if res:
            agg = {}
            for x in res:
                r = int(x["leaf"].rsplit("/r", 1)[1])
                a = agg.setdefault(f"r{r}", {"resolved": 0, "proven": 0, "tesseract": {}, "mpe_tau": {}})
                a["resolved"] += 1
                a["proven"] += int(x["proven"])
                for name in ("tesseract", "mpe_tau"):
                    v = x["verdicts"].get(name)
                    if v:
                        a[name][v] = a[name].get(v, 0) + 1
            out["mpe_tau_vs_tesseract"]["resolved"] = agg
    # (e) d = 3 ladder check, d = 5 retained counts
    for r in W5:
        o = {"shots": 0, "mpe_tau_vs_exact_mpe_differs": 0, "ml_tau_vs_exact_ml_differs": 0}
        for (d, p, b, rr), rec in leaves.items():
            if d != 3 or rr != r or "mpe_tau" not in rec["preds"]:
                continue
            m = mask_of(rec, ["mpe_tau", "mpe_exact"])
            o["shots"] += int(m.sum())
            o["mpe_tau_vs_exact_mpe_differs"] += int(np.sum(rec["preds"]["mpe_tau"][m] != rec["preds"]["mpe_exact"][m]))
            if "ml_tau" in rec["preds"]:
                m0 = mask_of(rec, ["ml_tau"])
                o["ml_tau_vs_exact_ml_differs"] += int(np.sum(rec["preds"]["ml_tau"][m0] != rec["ml"][m0]))
        if o["shots"]:
            out["d3_ladder_check"][f"r{r}"] = o
    for (d, p, b, r), rec in leaves.items():
        if "r2" in rec:
            for name, s in rec["r2"].items():
                key = f"d{d}|{name}"
                cur = out["retained_max"].setdefault(key, {"maxk_t1": 0, "maxk_t2": 0, "shots": 0,
                                                                   "redecoded": 0, "nan": 0})
                cur["maxk_t1"] = max(cur["maxk_t1"], s["maxk_t1"])
                cur["maxk_t2"] = max(cur["maxk_t2"], s["maxk_t2"])
                for k in ("shots", "redecoded", "nan"):
                    cur[k] += s[k]
                out["r2_cpu_s"][key] = out["r2_cpu_s"].get(key, 0.0) + s["cpu_s"]
    out["merges"] = {rel: m.get("merges") for rel, m in sorted(r2meta.items())}
    out["merges_total"] = sum(v or 0 for v in out["merges"].values())
    return out


def main(arch, dec, k2_dir, stage_dir, r1_records, r2_work, resolve_records, out_dir):
    t0 = time.time()
    chk = {k: 0 for k in ("ml_pred_vs_llr_mismatch", "ml_llr_nan", "det_padding_nonzero", "shipped_dem_not_rl",
                          "own_run_dem_not_rl", "own_run_subset_not_every_5th_or_all", "ml_dem_not_rl")}
    chk["ml_idx_rule_ok"] = True
    k2 = k2_verdicts(k2_dir)
    failing = {k for k, v in k2.items() if v["verdict"] != "pass"}
    r2_work = r2_work if os.path.isdir(r2_work) else None
    leaves, r2meta = load_leaves(arch, dec, stage_dir, r2_work, chk)
    print(f"leaves read ({time.time() - t0:.0f} s)", flush=True)
    head = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": chk,
            "k2": {"verdicts": len(k2), "still_fail": sorted(failing)},
            "inputs": {"r1_records": r1_records if os.path.isfile(r1_records) else None, "r2_work": r2_work,
                       "resolve_records": resolve_records if os.path.isfile(resolve_records) else None}}
    c1 = r1_certificate(leaves, failing, stage_dir, r1_records)
    c1.update(head, label="reservoir, revision R1 (review 1.4): exact-MPE certificate of the stored Tesseract at d = 5. "
                          "POST HOC; declared before it was run")
    with open(out_path(os.path.join(out_dir, "r1_certificate.json")), "w") as f:
        json.dump(c1, f, indent=1)
    print(f"R1 done ({time.time() - t0:.0f} s)", flush=True)
    c2 = r2_analysis(leaves, failing, r2meta, resolve_records)
    c2.update(head, label="reservoir, revision R2 (review 1.4): controlled change to degeneracy handling "
                          "(max-product against sum-product in decoder's sparse contraction). POST HOC; declared "
                          "before it was run")
    c2["run_seconds"] = time.time() - t0
    with open(out_path(os.path.join(out_dir, "r2_analysis.json")), "w") as f:
        json.dump(c2, f, indent=1)
    print(f"R2 done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 9:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
