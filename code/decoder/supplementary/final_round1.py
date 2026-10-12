#!/usr/bin/env python3
"""Round 1's final pre-registered analysis (notes/decoder/willow_final_analysis_plan.md; notes/decoder/willow_prereg.md
section 9 entry 12).

The frozen analyze_willow.py (md5 52fb23ff8e20c0b29b7e19191f38f0d4) runs unchanged, on a view of its inputs that applies
K2's verdicts. Two of its input functions are wrapped here; nothing in it is edited.
  - load_leaf: an excluded leaf loses its ML(RL) entity (decisions, posteriors, ladder row). Every 2x2 table of a
    decoder against an ML entity is also recorded, for entry 7's SE below.
  - planned_rounds: an excluded leaf's round is not planned for ML(RL).
Exclusion rule A (the plan's section 3): the excluded leaf's round leaves every fit that uses ML(RL): its own series
fits and every paired fit against it, the count ratios, and the d = 5 calibration and syndrome pools. The series keeps
its other rounds and stays in the means, as section 9 entry 8 does for a round with E >= 1/2. If K2's kill rule fires
(more than a quarter of the d = 5 leaves still fail), every d = 5 leaf loses its ML(RL) entity, and d = 5 is reported
with the baselines only (section 6).

Then, from the recorded tables: entry 7's SE (p1_se_covariance.py's full delta method, read-only) for every pooled g
(P1, P2, g_3 and g_5 on W5, the other-decoder rows); P1's Holm-adjusted p with that SE; K2's tally; entry 8's list of
rounds with E >= 1/2. A reproduction check compares every fit that no exclusion touches with the frozen run (--frozen).

The d = 5 statistics (section 9 entry 12, part 1) are recomputed here with the frozen functions, from the frozen run's
leaf counts and the recorded 2x2 tables, under rule A (chosen) and, as sensitivity rows, rule B (a series that lost a
leaf in the statistic's window leaves it) and no exclusion: g_5 (rounds 10, 30, 50, entry 7's SE), Lambda_ML(RL) at
d = 3/5 (W5), the paired ratio Lambda_Libra/Lambda_ML(RL) = (1 + g_3)/(1 + g_5), and P5 at d = 5. A series left with
fewer than 2 rounds in a window leaves that statistic and is listed; the statistic uses the remaining series. Each
value is checked against the frozen code's own where that exists (rule A: the view's run; no exclusion: --frozen).
Lambda_3/5 is the two-point value from d = 3 and 5 (section 5: d = 7 is exploratory); the frozen code's fit over
d = 3, 5 and 7 is reported beside it as exploratory.

Outputs in --out: analysis/ (the frozen code's outputs on the view: analysis.json, prereg_results.json, report.md,
tables, figures), final_round1.json and final_round1.md.

Run (one process; about 25 CPU minutes on all 852 jobs, as the frozen run):
  python3 -I -B code/decoder/supplementary/final_round1.py --res results/decoder/willow \\
      --k2 results/decoder/willow/supplementary/k2_extra_rung/k2_tally.json \\
      --frozen results/decoder/willow/analysis/analysis.json --out results/decoder/willow/final_round1
--allow-incomplete runs on an incomplete tally (tests only; the output says so).
"""
import argparse, hashlib, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("FINAL_R1_CODE") or os.path.dirname(HERE)    # code/decoder (tests may point elsewhere)
SUPP = os.path.join(CODE, "supplementary")
sys.path.insert(0, CODE)                       # the frozen modules
sys.path.insert(0, SUPP)                       # p1_se_covariance
import numpy as np
import analyze_willow as AW                    # frozen; two input functions wrapped, nothing edited
import p1_se_covariance as PC                  # section 9 entry 7; used read-only

ML5 = "ML.rl"                                  # the d = 5 ML entity: the tau ladder with the shipped RL DEM
ENTRY8 = {("d3|q6_7|X|mwpm.rl", 190), ("d3|q6_7|X|mwpm.rl", 250), ("d3|q8_9|X|mwpm.rl", 210),
          ("d3|q8_9|Z|mwpm.rl", 250), ("d3|q6_7|X|corrmatch.si1000", 250), ("d3|q8_9|X|corrmatch.si1000", 250),
          ("d3|q8_9|X|corr_mwpm.rl", 250), ("d3|q8_9|X|harmony.si1000", 250)}   # the 8 cases entry 8 listed


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rel_diff(a, b):
    if a is None or b is None:
        return float("inf") if (a is None) != (b is None) else 0.0
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--k2", required=True, help="k2_tally.json")
    ap.add_argument("--frozen", required=True, help="analysis.json of the frozen run on all jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=None, help="default <res>/plan.json")
    ap.add_argument("--rule", default="A", choices=["A"], help="exclusion rule (the plan's section 3)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true", help="tests only")
    a = ap.parse_args()
    t0 = time.time()
    plan_p = a.plan or os.path.join(a.res, "plan.json")
    plan = json.load(open(plan_p))
    tally = json.load(open(a.k2))
    if not tally.get("complete") and not a.allow_incomplete:
        sys.exit(f"{a.k2} is not complete; K2 has no final tally yet")
    expect = int(tally.get("expected_leaves") or 32)
    still = sorted(tally.get("still_failing") or [])
    kill = len(still) > expect / 4
    jobs = {j["leaf"]: j for j in plan["jobs"] if j.get("kind") == "decode"}
    d5_ml = sorted({rel for rel, j in jobs.items() if j["d"] == 5 and "ml_" in j["decoders"]})
    excl = set(d5_ml) if kill else set(still)
    unknown = sorted(x for x in excl if x not in jobs)
    if unknown:
        sys.exit(f"excluded leaves not in the plan: {unknown}")
    tables, removed = {}, []
    load0, planned0 = AW.load_leaf, AW.planned_rounds
    PR0 = planned0(plan)                         # the plan's rounds before any exclusion (rule B below)

    def load_leaf_view(res, rel, tags):
        obs, ents, ml, info, shipped = load0(res, rel, tags)
        meta = info.get("meta") or {}
        d, r = meta.get("distance"), info.get("rounds")
        if obs is not None and d is not None:            # every table, before an exclusion removes the ML entity
            for A in ents:
                for B in ents:
                    if A != B and B.startswith("ML."):
                        c = AW.count2(ents[A], ents[B], obs)
                        if c["n"]:
                            tables.setdefault((d, meta.get("patch"), meta.get("basis"), A, B), {})[r] = c
        if rel in excl and ML5 in ents:
            ents.pop(ML5); ml.pop(ML5, None); removed.append(rel)
        return obs, ents, ml, info, shipped

    def planned_rounds_view(pl):
        PR = planned0(pl)
        for rel in excl:
            j = jobs[rel]
            s = PR.get((j["d"], j["patch"], j["basis"], ML5))
            if s:
                s.discard(j["rounds"])
        return PR

    AW.load_leaf, AW.planned_rounds = load_leaf_view, planned_rounds_view
    out_an = os.path.join(a.out, "analysis")
    os.makedirs(out_an, exist_ok=True)
    argv0 = sys.argv
    sys.argv = ["analyze_willow.py", "--res", a.res, "--out", out_an, "--plan", plan_p] + (["--no-figures"] if a.no_figures else [])
    try:
        AW.main()
    finally:
        sys.argv = argv0
        AW.load_leaf, AW.planned_rounds = load0, planned0
    t_main = time.time() - t0
    R = json.load(open(os.path.join(out_an, "analysis.json")))
    F = json.load(open(a.frozen))

    # ---- reproduction check: every fit that no exclusion touches equals the frozen run's
    def touched(key):
        dd, patch, basis = key.split("|")[:3]
        if dd != "d5" or ML5 not in key:
            return False
        return any(jobs[x]["patch"] == patch and jobs[x]["basis"] == basis for x in excl)
    worst, n_cmp, missing = 0.0, 0, []
    for src, fields in (("series_eps", ("eps", "se_eps", "chi2_red")), ("series_eps_W5", ("eps", "se_eps", "chi2_red")),
                        ("paired_series", ("eps_ML", "eps_A", "delta", "se_delta", "rel", "se_rel")),
                        ("paired_series_W5", ("eps_ML", "eps_A", "delta", "se_delta", "rel", "se_rel"))):
        for key, f0 in F.get(src, {}).items():
            if touched(key):
                continue
            f1 = R.get(src, {}).get(key)
            if f1 is None:
                missing.append(f"{src}:{key}"); continue
            for q in fields:
                worst = max(worst, rel_diff(f1.get(q), f0.get(q))); n_cmp += 1
    check = {"compared_values": n_cmp, "max_relative_difference": worst, "missing": missing[:20], "n_missing": len(missing),
             "pass": worst <= 1e-12 and not missing}

    # ---- entry 7: the full delta-method SE of every pooled g, from the recorded tables
    def entry7(src, pooled_key):
        dd, pair = pooled_key.split("|", 1)
        A, B = pair.split(" vs ")
        S, keys, worst_s = [], [], 0.0
        for key, f in sorted(R[src].items()):
            if not (key.startswith(dd + "|") and key.endswith("|" + pair) and f["complete"]):
                continue
            _, patch, basis = key.split("|")[:3]
            pts = tables[(int(dd[1:]), patch, basis, A, B)]
            window = tuple(sorted(set(f["rounds"]) | set(f["dropped"])))
            sp = PC.series_params(pts, window)
            worst_s = max(worst_s, rel_diff(sp["frozen"]["rel"], f["rel"]))
            S.append(sp); keys.append(key)
        if not S:
            return None
        cs = PC.g_and_se(S, "inflated", full=True, with_cov=True)
        un = PC.g_and_se(S, "uninflated", full=True, with_cov=True)
        return {"n_series": len(S), "g": cs["g"], "se": cs["se"], "z": cs["z"], "p": cs["p"], "ci95": cs["ci95"],
                "ci90": cs["ci90"], "se_uninflated": un["se"], "corr_mean_delta_mean_eps_ML": cs["corr_mean_delta_mean_eps_ML"],
                "series_rel_check": worst_s, "series": keys}
    se7 = {}
    for src, dst in (("paired_series", "pooled_gaps"), ("paired_series_W5", "pooled_gaps_W5")):
        for k, p in sorted(R[dst].items()):
            e = entry7(src, k)
            if e:
                e.update(frozen_formula_se=p["se_rel"], expected_series=p["expected_series"],
                         g_check=rel_diff(e["g"], p["rel"]))
                se7[f"{dst}:{k}"] = e

    # ---- decisions with entry 7's SE
    P = R["prereg"]
    dec = {}
    p1 = se7.get("pooled_gaps:d3|libra.rl vs ML.rl")
    p4 = P.get("P4_mismatch_cost_d3", {}).get("pooled")
    if p1:
        dc = R["decomposition"].get("pooled") or {}
        p4_full = p4 is not None and (dc.get("expected_leaves") is None or dc["n_leaves"] >= 0.9 * dc["expected_leaves"])
        adj = AW.holm({"P1": p1["p"] if p1["n_series"] == p1["expected_series"] else None,
                       "P4": p4["p"] if p4_full else None})
        dec["P1"] = {"g": p1["g"], "se": p1["se"], "ci90": p1["ci90"], "ci95": p1["ci95"], "p": p1["p"],
                     "p_holm": adj.get("P1"), "p4_p_holm": adj.get("P4"),
                     "within_3pct_ci90": -AW.MARGIN <= p1["ci90"][0] and p1["ci90"][1] <= AW.MARGIN,
                     "frozen_decision": P["P1_libra_vs_ML_rl_d3"]["decision"]}
    p2 = se7.get("pooled_gaps:d3|ML.si1000 vs ML.rl")
    if p2:
        dec["P2"] = {"g": p2["g"], "se": p2["se"], "ci95": p2["ci95"],
                     "decision": "RL DEM better under ML (95% CI > 0)" if p2["ci95"][0] > 0 else
                     ("SI1000 DEM better under ML (95% CI < 0)" if p2["ci95"][1] < 0 else "no significant difference")}
    dec["G1"] = P.get("G1_reproduce_libra", {}).get("decision")

    # ---- d = 5 under rule A (chosen), with rule B and no exclusion as sensitivity rows (section 9 entry 12)
    reported = json.load(open(os.path.join(CODE, "willow_reported.json")))
    nn5 = reported["table_S1_fig1d"]["nn"]["eps5"]
    excl_r = {}
    for x in excl:
        if jobs[x]["d"] == 5:
            excl_r.setdefault((jobs[x]["patch"], jobs[x]["basis"]), set()).add(jobs[x]["rounds"])

    def planned_in_w5(d, patch, basis, *ents):
        s_ = set.intersection(*[(PR0.get((d, patch, basis, e)) or set()) for e in ents])
        return tuple(x for x in AW.W5 if x in s_)

    def rule_window(patch, basis, W, rule, applies=True):
        """(the rounds the series uses, the excluded rounds in W); None if the series leaves the statistic (rule B)."""
        lost = sorted(excl_r.get((patch, basis), set()) & set(W)) if applies else []
        if rule == "none" or not lost:
            return tuple(W), lost
        if rule == "B":
            return None, lost
        return tuple(x for x in W if x not in lost), lost

    def leaf_points(entity, d):
        out = {}
        for L in F["leaves"].values():
            x = (L.get("entities") or {}).get(entity)
            if L.get("d") == d and x and x.get("n"):
                out.setdefault((L["patch"], L["basis"]), {})[L["rounds"]] = (x["n"], x["errors"])
        return out

    def mean_eps_rule(entity, d, wents, rule):
        """Mean eps over an entity's series at distance d, W5 restricted to the rounds planned for wents."""
        fits, used, left = [], [], []
        for (patch, basis), pts in sorted(leaf_points(entity, d).items()):
            W = planned_in_w5(d, patch, basis, *wents)
            Wr, lost = rule_window(patch, basis, W, rule, applies=(entity == ML5 and d == 5))
            f = AW.series_eps(pts, Wr) if Wr is not None else None
            if f is None or not f["complete"]:
                left.append({"series": f"{patch}|{basis}", "excluded_rounds": lost,
                             "why": "rule B" if Wr is None else "fewer than 2 usable rounds, or a leaf missing"})
                continue
            fits.append(f); used.append(f"{patch}|{basis}")
        if not fits:
            return None
        m, se = AW.mean_eps(fits)
        return {"eps": m, "se": se, "n_series": len(fits), "series": used, "left_out": left}

    def g5_rule(rule):
        S, used, left = [], [], []
        for (d_, patch, basis, A, B), pts in sorted(tables.items()):
            if d_ != 5 or A != "libra.rl" or B != ML5:
                continue
            W = planned_in_w5(5, patch, basis, A, B)
            Wr, lost = rule_window(patch, basis, W, rule)
            if Wr is None or not set(Wr) <= set(pts) or len(PC.points_arrays(pts, Wr)) < 2:
                left.append({"series": f"{patch}|{basis}", "excluded_rounds": lost,
                             "why": "rule B" if Wr is None else "fewer than 2 usable rounds, or a leaf missing"})
                continue
            S.append(PC.series_params(pts, Wr)); used.append(f"{patch}|{basis}")
        if not S:
            return None
        cs = PC.g_and_se(S, "inflated", full=True, with_cov=True)
        ff = PC.g_and_se(S, "inflated", full=False, with_cov=False)
        return {"g": cs["g"], "se": cs["se"], "z": cs["z"], "p": cs["p"], "ci90": cs["ci90"], "ci95": cs["ci95"],
                "se_frozen_formula": ff["se"], "n_series": len(S), "series": used, "left_out": left}

    d5, chk5 = {}, {}
    g3w = se7.get("pooled_gaps_W5:d3|libra.rl vs ML.rl")
    if not kill:
        e3 = mean_eps_rule(ML5, 3, (ML5,), "none")
        for rule in ("A", "B", "none"):
            row = {"g5": g5_rule(rule)}
            e5 = mean_eps_rule(ML5, 5, (ML5,), rule)
            if e3 and e5:
                row["Lambda_ML_rl_3_5"] = dict(AW.lam_fit({3: (e3["eps"], e3["se"]), 5: (e5["eps"], e5["se"])}),
                                               eps5=e5, eps3={"eps": e3["eps"], "se": e3["se"], "n_series": e3["n_series"]})
            g = row["g5"]
            if g and g3w:
                lr = math.log(1 + g3w["g"]) - math.log(1 + g["g"])
                se = math.sqrt((g3w["se"] / (1 + g3w["g"])) ** 2 + (g["se"] / (1 + g["g"])) ** 2)
                row["paired_ratio_libra_over_ML_rl"] = {"ratio": math.exp(lr), "ln_ratio": lr, "se_ln": se,
                                                        "ci95": [math.exp(lr - 1.96 * se), math.exp(lr + 1.96 * se)],
                                                        "g3_W5": g3w["g"], "g5": g["g"]}
            p5 = mean_eps_rule(ML5, 5, (ML5, "libra.rl"), rule)
            if p5:
                row["P5_d5"] = AW.ci_block({"eps_ML_rl": p5["eps"], "se": p5["se"], "eps_NN": nn5[0], "se_NN": nn5[1],
                                            "n_series": p5["n_series"], "series": p5["series"], "left_out": p5["left_out"]},
                                           (p5["eps"] - nn5[0]) / nn5[0], math.sqrt(p5["se"] ** 2 + nn5[1] ** 2) / nn5[0])
            d5[rule] = row
        # checks against the frozen code's own values (rule A: the run on the view; no exclusion: --frozen)
        a5 = se7.get("pooled_gaps_W5:d5|libra.rl vs ML.rl")
        if a5 and d5["A"]["g5"]:
            chk5["g5_A_vs_view"] = max(rel_diff(d5["A"]["g5"]["g"], a5["g"]), rel_diff(d5["A"]["g5"]["se"], a5["se"]))
        fz = F.get("pooled_gaps_W5", {}).get("d5|libra.rl vs ML.rl")
        if fz and d5["none"]["g5"]:
            chk5["g5_none_vs_frozen"] = rel_diff(d5["none"]["g5"]["g"], fz["rel"])
            chk5["g5_none_frozen_formula_se_vs_frozen"] = rel_diff(d5["none"]["g5"]["se_frozen_formula"], fz["se_rel"])
        for rule, src in (("A", P), ("none", F.get("prereg", {}))):
            q = (src.get("P5_ML_rl_vs_NN", {}).get("pooled") or {}).get("5")
            if q and d5[rule].get("P5_d5"):
                chk5[f"P5_d5_{rule}_vs_frozen_code"] = rel_diff(d5[rule]["P5_d5"]["rel"], q["rel"])
        fl = F.get("lambda", {}).get(ML5)
        if fl and d5["none"].get("Lambda_ML_rl_3_5"):
            chk5["eps3_vs_frozen"] = rel_diff(e3["eps"], fl["eps"]["3"][0])
            chk5["eps5_none_vs_frozen"] = rel_diff(d5["none"]["Lambda_ML_rl_3_5"]["eps5"]["eps"], fl["eps"]["5"][0])
        fr = F.get("lambda_paired_libra_over_ML_rl")
        if fr and d5["none"].get("paired_ratio_libra_over_ML_rl"):
            chk5["paired_ratio_none_vs_frozen_point"] = rel_diff(d5["none"]["paired_ratio_libra_over_ML_rl"]["ratio"],
                                                                 fr["ratio"])
    dec["d5_rules"] = d5
    dec["d5_checks"] = chk5
    if kill:
        dec["d5_replication"] = {"decision": "not evaluated: K2 killed d = 5 ML (section 6)"}
    elif d5["A"]["g5"] and p1:
        g5 = d5["A"]["g5"]
        rep_ = (g5["g"] > 0) == (p1["g"] > 0) and (g5["ci90"][0] > 0 or g5["ci90"][1] < 0)
        dec["d5_replication"] = {"rule": "A", "g5": g5["g"], "se": g5["se"], "ci90": g5["ci90"], "ci95": g5["ci95"],
                                 "n_series": g5["n_series"], "expected_series": len(g5["series"]) + len(g5["left_out"]),
                                 "left_out": g5["left_out"], "replicated": bool(rep_),
                                 "same_verdict_under_B_and_none": all(
                                     (d5[q]["g5"] is not None) and
                                     (((d5[q]["g5"]["g"] > 0) == (p1["g"] > 0) and
                                       (d5[q]["g5"]["ci90"][0] > 0 or d5[q]["g5"]["ci90"][1] < 0)) == bool(rep_))
                                     for q in ("B", "none"))}
    # ---- Lambda_3/5 per decoder (rule A; two points, W5) and the frozen code's fit over d = 3, 5, 7 (exploratory)
    lam35 = {}
    for e in ("ML.rl", "ML.si1000", "libra.rl", "harmony.rl", "corrmatch.rl", "corr_mwpm.rl", "mwpm.rl"):
        if e == ML5 and kill:
            continue
        e3_, e5_ = mean_eps_rule(e, 3, (e,), "A"), mean_eps_rule(e, 5, (e,), "A")
        if e3_ and e5_:
            lam35[e] = dict(AW.lam_fit({3: (e3_["eps"], e3_["se"]), 5: (e5_["eps"], e5_["se"])}),
                            n_series={"3": e3_["n_series"], "5": e5_["n_series"]},
                            left_out={"3": e3_["left_out"], "5": e5_["left_out"]})
    dec["lambda_3_5"] = lam35
    dec["lambda_frozen_fit_exploratory"] = {k: {"Lambda": v["Lambda"], "se": v["se"], "distances": v["distances"]}
                                            for k, v in R["lambda"].items()}
    dec["calibration_d5_ML_rl"] = R["calibration"]["pooled"].get("d5|" + ML5)

    # ---- entry 8: rounds with E >= 1/2, left out of a fit
    e8 = []
    for src in ("series_eps", "series_eps_W5", "paired_series", "paired_series_W5"):
        for key, f in sorted(R[src].items()):
            for r in f.get("dropped") or []:
                e8.append({"fit": src, "key": key, "r": r, "listed_in_entry_8": (key, r) in ENTRY8})
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "code_md5": {"final_round1.py": md5_file(os.path.abspath(__file__)),
                        "analyze_willow.py": md5_file(os.path.join(CODE, "analyze_willow.py")),
                        "p1_se_covariance.py": md5_file(os.path.join(SUPP, "p1_se_covariance.py"))},
           "inputs_md5": {"k2_tally.json": md5_file(a.k2), "plan.json": md5_file(plan_p), "frozen analysis.json": md5_file(a.frozen)},
           "test_only_incomplete_k2": bool(not tally.get("complete")),
           "rule": a.rule, "k2": {"complete": tally.get("complete"), "verdicts": tally.get("verdicts"),
                                  "still_failing": still, "without_verdict": tally.get("without_verdict"),
                                  "kill": kill, "kill_rule": f"more than {expect / 4:g} of {expect} still fail",
                                  "cpu_core_hours_extra_rungs": tally.get("cpu_core_hours_extra_rungs")},
           "excluded_leaves": sorted(excl), "ml_removed_at": sorted(set(removed)),
           "reproduction_check": check, "entry7_se": se7, "decisions": dec, "entry8_dropped_rounds": e8,
           "frozen_prereg_on_view": {k: v.get("decision") for k, v in P.items() if isinstance(v, dict)},
           "seconds": {"frozen_main": t_main, "total": time.time() - t0}}
    json.dump(out, open(os.path.join(a.out, "final_round1.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(a.out, "final_round1.md"))
    print(json.dumps({k: out[k] for k in ("k2", "excluded_leaves", "reproduction_check", "decisions")}, indent=1, default=float))


def write_md(o, path):
    pc = lambda x: f"{100 * x:.3f}%"
    ci = lambda c: f"[{100 * c[0]:.3f}, {100 * c[1]:.3f}]%"
    s = "# Round 1: final pre-registered analysis (generated by final_round1.py)\n\n"
    if o["test_only_incomplete_k2"]:
        s += "**TEST RUN on an incomplete K2 tally. Not the final analysis.**\n\n"
    k2 = o["k2"]
    s += (f"Made {o['made']}. Rule {o['rule']} (section 9 entry 12). K2: {k2['verdicts']}; still failing "
          f"{len(k2['still_failing'])} ({', '.join(x.split('/', 1)[1] for x in k2['still_failing'])}); "
          f"kill: {k2['kill']} ({k2['kill_rule']}).\n\n")
    c = o["reproduction_check"]
    s += (f"Reproduction check against the frozen run: {c['compared_values']} values, largest relative difference "
          f"{c['max_relative_difference']:.1e}, {c['n_missing']} missing: {'pass' if c['pass'] else 'FAIL'}.\n\n")
    d = o["decisions"]
    s += "## Decisions with entry 7's SE\n\n"
    if "P1" in d:
        x = d["P1"]
        s += f"- P1: g = {pc(x['g'])}, SE {pc(x['se'])}, 90% CI {ci(x['ci90'])}, Holm p = {x['p_holm']:.2g}.\n"
    if "d5_replication" in d:
        x = d["d5_replication"]
        if "decision" in x:
            s += f"- d = 5 replication: {x['decision']}\n"
        else:
            lo = "; ".join(f"{y['series']} ({y['why']})" for y in x["left_out"]) or "none"
            s += (f"- d = 5 replication (rule A): g_5 = {pc(x['g5'])}, SE {pc(x['se'])}, 90% CI {ci(x['ci90'])}, "
                  f"{x['n_series']} of {x['expected_series']} series (left out: {lo}): replicated = {x['replicated']}. "
                  f"Same verdict under rule B and with no exclusion: {x['same_verdict_under_B_and_none']}.\n")
    if "P2" in d:
        x = d["P2"]
        s += f"- P2: g = {pc(x['g'])}, SE {pc(x['se'])}, 95% CI {ci(x['ci95'])}: {x['decision']}.\n"
    s += f"- G1: {d.get('G1')}\n"
    d5 = d.get("d5_rules") or {}
    if d5:
        s += "\n## d = 5: rule A (chosen) with rule B and no exclusion as sensitivity rows\n\n"
        s += "| statistic | rule A (chosen) | rule B | no exclusion |\n|---|---|---|---|\n"
        def cell(rule, kind):
            row = d5.get(rule) or {}
            if kind == "g5":
                x = row.get("g5")
                return "n/a" if not x else f"{pc(x['g'])} ± {pc(x['se'])}, 95% CI {ci(x['ci95'])}, {x['n_series']} series"
            if kind == "lam":
                x = row.get("Lambda_ML_rl_3_5")
                return "n/a" if not x else f"{x['Lambda']:.4f} ± {x['se']:.4f}, {x['eps5']['n_series']} series at d = 5"
            if kind == "ratio":
                x = row.get("paired_ratio_libra_over_ML_rl")
                return "n/a" if not x else f"{x['ratio']:.4f}, 95% CI [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"
            x = row.get("P5_d5")
            return "n/a" if not x else (f"{pc(x['rel'])} ± {pc(x['se_rel'])}, 95% CI {ci(x['ci95'])}, "
                                        f"{x['n_series']} series")
        for kind, label in (("g5", "g_5, Libra vs ML(RL), rounds 10, 30, 50 (entry 7's SE)"),
                            ("lam", "Λ_ML(RL), d = 3/5, W5"),
                            ("ratio", "Λ_Libra / Λ_ML(RL) = (1 + g_3)/(1 + g_5)"),
                            ("P5", "P5 at d = 5: ML(RL) against the NN (Table S1)")):
            s += f"| {label} | {cell('A', kind)} | {cell('B', kind)} | {cell('none', kind)} |\n"
        for rule in ("A", "B"):
            for kind in ("g5", "Lambda_ML_rl_3_5", "P5_d5"):
                x = (d5.get(rule) or {}).get(kind) or {}
                lo = x.get("left_out") if kind != "Lambda_ML_rl_3_5" else (x.get("eps5") or {}).get("left_out")
                if lo:
                    s += (f"\nRule {rule}, {kind}: left out " +
                          "; ".join(f"{y['series']} (excluded rounds {y['excluded_rounds']}; {y['why']})" for y in lo) + ".")
        s += "\n\nChecks against the frozen code's own values (relative differences): " + \
             ", ".join(f"{k} {v:.1e}" for k, v in (d.get("d5_checks") or {}).items()) + "\n"
    if d.get("lambda_3_5"):
        s += "\n## Λ_3/5 per decoder (rule A; W5; two points, section 5)\n\n| entity | Λ_3/5 | se | series (d = 3, 5) | frozen code's fit over d = 3, 5, 7 (exploratory) |\n|---|---|---|---|---|\n"
        for k, v in d["lambda_3_5"].items():
            fz = (d.get("lambda_frozen_fit_exploratory") or {}).get(k)
            fzs = "n/a" if not fz else "{:.4f} ± {:.4f} (d = {})".format(fz["Lambda"], fz["se"], fz["distances"])
            s += f"| {k} | {v['Lambda']:.4f} | {v['se']:.4f} | {v['n_series']['3']}, {v['n_series']['5']} | {fzs} |\n"
    s += "\n## Pooled gaps: frozen-formula SE and entry 7's SE (rule A)\n\n| gap | series | g | SE (frozen formula) | SE (entry 7) | 95% CI |\n|---|---|---|---|---|---|\n"
    for k, x in o["entry7_se"].items():
        s += f"| {k} | {x['n_series']} of {x['expected_series']} | {pc(x['g'])} | {pc(x['frozen_formula_se'])} | {pc(x['se'])} | {ci(x['ci95'])} |\n"
    new8 = [x for x in o["entry8_dropped_rounds"] if not x["listed_in_entry_8"]]
    s += f"\n## Entry 8: rounds with E >= 1/2 left out of a fit\n\n{len(o['entry8_dropped_rounds'])} in all, {len(new8)} not listed in entry 8:\n\n"
    for x in new8:
        s += f"- {x['fit']}: {x['key']}, r = {x['r']}\n"
    s += ("\nThe frozen code's own outputs on the rule-A view are in `analysis/`. Its K2 line reads only the ladder; "
          "K2's verdict is the tally above. Its Λ table is the fit over d = 3, 5 and 7, shown above as exploratory.\n")
    open(path, "w").write(s)


if __name__ == "__main__":
    main()
