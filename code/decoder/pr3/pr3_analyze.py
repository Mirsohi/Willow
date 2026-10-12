#!/usr/bin/env python3
"""PR3's analysis (notes/decoder/willow_prereg_r3.md sections 4 and 8), fixed before any PR3 decision exists.

Inputs, all read-only: the main run's parts (<res>/parts/p<j>/decode/<leaf>/<tag>/: the ladder's final decisions, the
shots with Libra's shipped predictions, our corr. MWPM and MWPM), its pr3_tally.json (verdicts, kill), round 1's
final_round1.json (g_5 under rule A, entry 7's SE), round 1's frozen analysis.json (leaf counts at d = 3 and 5, for Λ),
plan.json, and d7_numbers.json (our Libra eps7 over r >= 10 on all shots, from the archive).

Sample: every shot the main run decoded, except the r = 10 probe shots (index divisible by 25). Rule A: a leaf that
still fails after its extra rung leaves its series' fits; a series left with fewer than 2 rounds leaves g_7 (listed).
If d = 7 ML is killed (2 or more leaves still fail), H7 and the d = 7 ML rows are not evaluated.

H7, the primary test (one test, alpha = 0.05, two-sided): Delta = ln(1 + g_7) - ln(1 + g_5), with g_d the paired gap of
Libra to ML(RL) on the rounds {10, 30, 50} (section 4 of willow_prereg.md: per-series paired fits, mean over series;
entry 7's SE via p1_se_covariance.py) and SE(Delta) = sqrt(SE_7^2 + SE_5^2), SE_d = SE(g_d)/(1 + g_d). "The gap keeps
growing" if Delta > 0 and p < 0.05.
Secondary (95% CIs, descriptive): g_7 against 0; the count ratio over the leaves; ML(RL) against the NN at d = 7, (a)
paired (eps_ML,7 = eps_Libra,7 / (1 + g_7), Table S1, and per basis Table S5) and (b) the direct fit of ML(RL); Lambda
of ML(RL) and Libra over d = 3, 5, 7 on {10, 30, 50}; our corr. MWPM and MWPM against ML(RL) at d = 7. The ML(RL)
decisions are the ladder's final (tau2) decisions, as in round 1 and K2. The NN rows use round 1's P5 formula
(analyze_willow.py: rel = eps_ML / eps_NN - 1, SE = sqrt(SE_ML^2 + SE_NN^2) / eps_NN); the error of eps_ML,7 in (a)
combines our Libra eps7's SE and g_7's SE as independent. Checks: the parts cover the design's shots exactly; the
sample's ML(RL) error counts equal the tally's; g_7 equals the frozen pool_gap()'s value on the same tables, and its
frozen-formula SE equals pool_gap()'s SE. The d = 7 values (ML(RL) and Libra) come from the PR3 sample; the d = 3 and 5
values from round 1's leaves in analysis.json, with rule A's exclusions at d = 5 (final_round1.json) for ML(RL).
  python3 -I -B pr3_analyze.py --res results/decoder/willow/pr3/main --out results/decoder/willow/pr3/analysis \
      --final-r1 results/decoder/willow/final_round1/final_round1.json \
      --frozen results/decoder/willow/analysis/analysis.json --plan results/decoder/willow/plan.json \
      --d7 results/decoder/willow/pr3_planning/d7_numbers.json [--allow-incomplete]
"""
import argparse, hashlib, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("PR3_CODE") or os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "supplementary"))
sys.path.insert(0, HERE)
import numpy as np
import analyze_willow as AW               # frozen; read-only
import p1_se_covariance as PC             # section 9 entry 7; read-only
import pr3_common as C

W = (10, 30, 50)
LADDER_DN = C.LADDER.replace(":", "_").replace("/", "-")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_leaf(res, tag, lf, r, design):
    """The leaf's 5 parts, concatenated, on the PR3 sample: obs and the decisions of ML(RL), Libra, corr. MWPM, MWPM."""
    out = {k: [] for k in ("idx", "obs", "ML", "libra", "corr_mwpm", "mwpm")}
    want = []
    for j in range(C.PARTS):
        st, sp, stp = (int(x) for x in C.part_slice(design, j).split(":"))
        want.append(np.arange(st, sp, stp))
        d = os.path.join(res, "parts", f"p{j}", "decode", lf, tag)
        s = json.load(open(os.path.join(d, "summary.json")))
        assert s["status"] == "complete", f"{d}: not complete"
        z = np.load(os.path.join(d, LADDER_DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        assert np.array_equal(z["idx"], sh["idx"]), f"{d}: ladder and shots select different shots"
        rows = [i for i, x in enumerate(sh["shipped_names"]) if AW.shipped_entity(str(x)) == "libra.rl"]
        assert len(rows) == 1, f"{d}: {len(rows)} shipped Libra (RL) rows"
        out["idx"].append(sh["idx"].astype(np.int64)); out["obs"].append(sh["obs"].astype(np.int8))
        out["ML"].append(z["pred"][:, 0].astype(np.int8))
        out["libra"].append(sh["shipped"][rows[0]].astype(np.int8))
        for b in ("corr_mwpm", "mwpm"):
            zb = np.load(os.path.join(d, b + ".npz"))
            assert np.array_equal(zb["idx"], sh["idx"]), f"{d}: {b} selects different shots"
            out[b].append(zb["pred"][:, 0].astype(np.int8))
    cat = {k: np.concatenate(v) for k, v in out.items()}
    assert np.array_equal(cat["idx"], np.concatenate(want)), f"{lf}: the parts do not cover the {design} design's shots"
    m = C.in_sample(cat["idx"], r)
    return {k: v[m] for k, v in cat.items()}, int(len(m)), int(m.sum())


def entry7_g(tables, windows):
    """g with entry 7's SE over the series that keep at least 2 rounds; tables {series: {r: 2x2}}."""
    S, used, left = [], [], []
    for ser in sorted(tables):
        w = windows.get(ser, ())
        pts = tables[ser]
        if len(w) < 2 or not set(w) <= set(pts) or len(PC.points_arrays(pts, w)) < 2:
            left.append(ser); continue
        S.append(PC.series_params(pts, w)); used.append(ser)
    if not S:
        return None
    cs = PC.g_and_se(S, "inflated", full=True, with_cov=True)
    ff = PC.g_and_se(S, "inflated", full=False, with_cov=False)
    pg = AW.pool_gap([s["frozen"] for s in S])           # the frozen estimator on the same tables
    rd = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    return {"g": cs["g"], "se": cs["se"], "z": cs["z"], "p": cs["p"], "ci95": cs["ci95"], "ci90": cs["ci90"],
            "se_frozen_formula": ff["se"], "n_series": len(S), "series": used, "left_out": left,
            "per_series_rel": {u: s["frozen"]["rel"] for u, s in zip(used, S)},
            "check_vs_frozen_pool_gap": {"g": rd(cs["g"], pg["rel"]), "se_frozen_formula": rd(ff["se"], pg["se_rel"])}}


def mean_eps_on(points, windows):
    fits, used, left = [], [], []
    for ser in sorted(points):
        f = AW.series_eps(points[ser], windows.get(ser, ()))
        if f is None or not f["complete"]:
            left.append(ser); continue
        fits.append(f); used.append(ser)
    if not fits:
        return None
    m, se = AW.mean_eps(fits)
    return {"eps": m, "se": se, "n_series": len(fits), "series": used, "left_out": left}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="default rl_<the tally's design>")
    ap.add_argument("--tally", default=None, help="default <res>/pr3_tally.json")
    ap.add_argument("--final-r1", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--d7", required=True)
    ap.add_argument("--reported", default=os.path.join(CODE, "willow_reported.json"))
    ap.add_argument("--allow-incomplete", action="store_true", help="tests only")
    a = ap.parse_args()
    tally_p = a.tally or os.path.join(a.res, "pr3_tally.json")
    tally = json.load(open(tally_p))
    if not tally.get("complete") and not a.allow_incomplete:
        sys.exit("PR3's tally is not complete")
    F1 = json.load(open(a.final_r1))
    if F1.get("test_only_incomplete_k2") and not a.allow_incomplete:
        sys.exit("final_round1.json comes from an incomplete K2 tally")
    rep = json.load(open(a.reported))
    d7 = json.load(open(a.d7))
    design = tally["design"]
    tag = a.tag or f"rl_{design}"
    kill7 = bool(tally.get("kill"))
    excl7 = set(tally.get("still_failing") or [])
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "test_only": bool(a.allow_incomplete),
           "design": design, "tag": tag,
           "inputs_md5": {"pr3_tally.json": md5(tally_p), "final_round1.json": md5(a.final_r1),
                          "analysis.json": md5(a.frozen), "plan.json": md5(a.plan), "d7_numbers.json": md5(a.d7)},
           "code_md5": {"pr3_analyze.py": md5(os.path.abspath(__file__)), "pr3_common.py": md5(os.path.join(HERE, "pr3_common.py")),
                        "analyze_willow.py": md5(os.path.join(CODE, "analyze_willow.py")),
                        "p1_se_covariance.py": md5(os.path.join(CODE, "supplementary", "p1_se_covariance.py"))},
           "d7_kill": kill7, "d7_excluded_leaves": sorted(excl7), "leaves": {}}
    # ---- per leaf, on the PR3 sample
    tables = {e: {} for e in ("libra", "corr_mwpm", "mwpm")}       # {entity: {basis: {r: 2x2 vs ML}}}
    ml_pts, lib_pts, cells = {}, {}, []
    for b in C.BASES:
        for r in C.ROUNDS:
            lf = C.leaf(b, r)
            L, n_dec, n_s = load_leaf(a.res, tag, lf, r, design)
            tv = (tally.get("verdicts") or {}).get(lf, {})
            row = {"n_decoded": n_dec, "n_sample": n_s, "verdict": tv.get("verdict"),
                   "ladder": tv.get("ladder"), "extra": tv.get("extra")}
            for e in ("ML", "libra", "corr_mwpm", "mwpm"):
                row[f"errors_{e}"] = int(np.sum(L[e] != L["obs"]))
            k_t = (tv.get("ladder") or {}).get("errors_tau2")
            assert k_t is None or k_t == row["errors_ML"], f"{lf}: ML errors {row['errors_ML']} != the tally's {k_t}"
            assert tv.get("n_sample") in (None, n_s), f"{lf}: sample {n_s} != the tally's {tv.get('n_sample')}"
            lib_pts.setdefault(b, {})[r] = (n_s, row["errors_libra"])
            if lf not in excl7:
                ml_pts.setdefault(b, {})[r] = (n_s, row["errors_ML"])
                for e in tables:
                    c = AW.count2(L[e], L["ML"], L["obs"])
                    tables[e].setdefault(b, {})[r] = c
                    if e == "libra":
                        row["libra_vs_ML_2x2"] = c; cells.append(c)
            out["leaves"][lf] = row
    win = {b: tuple(r for r in W if r in tables["libra"].get(b, {})) for b in C.BASES}
    res = {}
    if kill7:
        res["H7"] = {"decision": "not evaluated: d = 7 ML killed (2 or more leaves still fail)"}
    else:
        g7 = entry7_g(tables["libra"], win)
        res["g7"] = g7
        g5 = (((F1.get("decisions") or {}).get("d5_rules") or {}).get("A") or {}).get("g5")
        if F1.get("k2", {}).get("kill") or not g5:
            res["H7"] = {"decision": "not evaluable: no g_5 (d = 5 ML killed or round 1's final analysis missing)"}
        elif not g7:
            res["H7"] = {"decision": "not evaluable: no d = 7 series keeps 2 rounds"}
        else:
            D = math.log(1 + g7["g"]) - math.log(1 + g5["g"])
            se = math.sqrt((g7["se"] / (1 + g7["g"])) ** 2 + (g5["se"] / (1 + g5["g"])) ** 2)
            z = D / se; p = AW.pval(z)
            res["H7"] = {"g7": g7["g"], "se7": g7["se"], "g5": g5["g"], "se5": g5["se"], "Delta": D, "se_Delta": se,
                         "z": z, "p": p, "ci95_Delta": [D - 1.96 * se, D + 1.96 * se],
                         "paired_Lambda_ratio_5_7": math.exp(-D),
                         "decision": ("the gap keeps growing (Delta > 0, p < 0.05)" if D > 0 and p < 0.05 else
                                      ("the gap shrinks (Delta < 0, p < 0.05)" if D < 0 and p < 0.05 else
                                       "no significant change from d = 5 to d = 7"))}
        # secondary
        res["count_ratio_libra_vs_ML"] = AW.ratio_from_counts(cells)
        for e in ("corr_mwpm", "mwpm"):
            res[f"g7_{e}_vs_ML"] = entry7_g(tables[e], win)
        nn7 = rep["table_S1_fig1d"]["nn"]["eps7"]; s5 = rep["table_S5_nn_eps"]["d7 0+0j"]
        lib7 = d7["G1_d7_libra"]["r>=10"]
        if g7:
            eL, sL = lib7["ours"], lib7["se"]
            eM = eL / (1 + g7["g"]); seM = math.sqrt((sL / (1 + g7["g"])) ** 2 + (eL * g7["se"] / (1 + g7["g"]) ** 2) ** 2)
            res["NN_paired_S1"] = AW.ci_block({"eps_ML_rl_7": eM, "se": seM, "eps_NN": nn7[0], "se_NN": nn7[1],
                                               "eps_libra_7_r>=10": eL, "se_libra_7": sL}, eM / nn7[0] - 1,
                                              math.sqrt(seM ** 2 + nn7[1] ** 2) / nn7[0])
            per = {}
            for b in C.BASES:
                gb = entry7_g({b: tables["libra"].get(b, {})}, win)
                fb = d7["fits"]["libra_decoder_with_rl_optimized_prior"]["r>=10"][b]
                if gb:
                    eMb = fb["eps"] / (1 + gb["g"])
                    seMb = math.sqrt((fb["se"] / (1 + gb["g"])) ** 2 + (fb["eps"] * gb["se"] / (1 + gb["g"]) ** 2) ** 2)
                    nb = s5[b]
                    per[b] = AW.ci_block({"g7_basis": gb["g"], "se_g7_basis": gb["se"], "eps_libra_7_r>=10": fb["eps"],
                                          "eps_ML_rl_7": eMb, "se": seMb, "eps_NN": nb[0], "se_NN": nb[1]}, eMb / nb[0] - 1,
                                         math.sqrt(seMb ** 2 + nb[1] ** 2) / nb[0])
            res["NN_paired_S5_per_basis"] = per
        dM = mean_eps_on(ml_pts, {b: tuple(r for r in W if r in ml_pts.get(b, {})) for b in C.BASES})
        if dM:
            res["NN_direct_S1"] = AW.ci_block(dict(dM, eps_NN=nn7[0], se_NN=nn7[1]), dM["eps"] / nn7[0] - 1,
                                              math.sqrt(dM["se"] ** 2 + nn7[1] ** 2) / nn7[0])
        # Lambda over d = 3, 5, 7 on {10, 30, 50}
        FZ = json.load(open(a.frozen)); plan = json.load(open(a.plan))
        PR = AW.planned_rounds(plan)
        ex5 = set(F1.get("excluded_leaves") or [])
        ex5r = {}
        for j in plan["jobs"]:
            if j.get("kind") == "decode" and j["leaf"] in ex5 and j["d"] == 5:
                ex5r.setdefault((j["patch"], j["basis"]), set()).add(j["rounds"])
        lam = {}
        for ent, d7pts in (("ML.rl", ml_pts), ("libra.rl", lib_pts)):
            eps = {}
            for d in (3, 5):
                pts, wins = {}, {}
                for L in FZ["leaves"].values():
                    x = (L.get("entities") or {}).get(ent)
                    if L.get("d") == d and x and x.get("n"):
                        pts.setdefault((L["patch"], L["basis"]), {})[L["rounds"]] = (x["n"], x["errors"])
                for ser in pts:
                    pl = set(PR.get((d,) + ser + (ent,)) or set()) & set(PR.get((d,) + ser + ("libra.rl",)) or set())
                    drop = ex5r.get(ser, set()) if (d == 5 and ent == "ML.rl") else set()
                    wins[ser] = tuple(r for r in W if r in pl and r not in drop)
                me = mean_eps_on(pts, wins)
                if me:
                    eps[d] = (me["eps"], me["se"])
            me7 = mean_eps_on(d7pts, {b: tuple(r for r in W if r in d7pts.get(b, {})) for b in C.BASES})
            if me7:
                eps[7] = (me7["eps"], me7["se"])
            lam[ent] = {"eps": {str(k): list(v) for k, v in eps.items()},
                        "Lambda_3_5_7": AW.lam_fit(eps) if len(eps) >= 2 else None,
                        "Lambda_5_7": AW.lam_fit({k: eps[k] for k in (5, 7) if k in eps}) if 5 in eps and 7 in eps else None}
        res["Lambda_on_10_30_50"] = lam
    out["results"] = res
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "pr3_results.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(a.out, "pr3_results.md"))
    print(json.dumps({"H7": res.get("H7"), "g7": (res.get("g7") or {}).get("g")}, indent=1, default=float))


def write_md(o, path):
    pc = lambda x: f"{100 * x:+.3f}%"
    ci = lambda c: f"[{100 * c[0]:+.3f}, {100 * c[1]:+.3f}]%"
    R = o["results"]
    s = "# PR3: does the Libra–ML gap keep growing at d = 7? (generated by pr3_analyze.py)\n\n"
    if o["test_only"]:
        s += "**TEST RUN. Not the PR3 result.**\n\n"
    s += f"Made {o['made']}. d = 7 kill: {o['d7_kill']}; excluded leaves: {o['d7_excluded_leaves'] or 'none'}.\n\n"
    h = R.get("H7", {})
    s += "## H7 (primary)\n\n"
    if "Delta" in h:
        s += (f"- g_7 = {pc(h['g7'])} (SE {100 * h['se7']:.3f}%), g_5 = {pc(h['g5'])} (SE {100 * h['se5']:.3f}%).\n"
              f"- Delta = ln(1 + g_7) - ln(1 + g_5) = {h['Delta']:+.4f} ± {h['se_Delta']:.4f}, 95% CI "
              f"[{h['ci95_Delta'][0]:+.4f}, {h['ci95_Delta'][1]:+.4f}], z = {h['z']:+.2f}, p = {h['p']:.3g}.\n"
              f"- Paired Λ_Libra/Λ_ML(RL), d = 5/7: {h['paired_Lambda_ratio_5_7']:.4f}.\n- **{h['decision']}**\n")
    else:
        s += f"- {h.get('decision')}\n"
    g7 = R.get("g7")
    if g7:
        s += (f"\n## Secondary\n\n- g_7 against 0: {pc(g7['g'])}, SE {100 * g7['se']:.3f}% (frozen formula "
              f"{100 * g7['se_frozen_formula']:.3f}%), 95% CI {ci(g7['ci95'])}; {g7['n_series']} series"
              + (f", left out {g7['left_out']}" if g7["left_out"] else "") + ".\n")
    cr = R.get("count_ratio_libra_vs_ML")
    if cr:
        s += f"- Count ratio, Libra vs ML(RL), {cr['leaves']} leaves: {pc(cr['rel'])}, 95% CI {ci(cr['ci95'])}.\n"
    x = R.get("NN_paired_S1")
    if x:
        s += f"- ML(RL) vs NN (a, paired, Table S1): eps_ML,7 = {x['eps_ML_rl_7']:.4e} ± {x['se']:.1e} vs {x['eps_NN']:.2e}: {pc(x['rel'])}, 95% CI {ci(x['ci95'])}.\n"
    for b, x in (R.get("NN_paired_S5_per_basis") or {}).items():
        s += f"  - {b} (Table S5): eps_ML,7 = {x['eps_ML_rl_7']:.4e} vs {x['eps_NN']:.2e}: {pc(x['rel'])}, 95% CI {ci(x['ci95'])}.\n"
    x = R.get("NN_direct_S1")
    if x:
        s += f"- ML(RL) vs NN (b, direct fit on {{10, 30, 50}}): {x['eps']:.4e} ± {x['se']:.1e}: {pc(x['rel'])}, 95% CI {ci(x['ci95'])}.\n"
    for e in ("corr_mwpm", "mwpm"):
        x = R.get(f"g7_{e}_vs_ML")
        if x:
            s += f"- {e} vs ML(RL) at d = 7: {pc(x['g'])}, 95% CI {ci(x['ci95'])}.\n"
    for ent, v in (R.get("Lambda_on_10_30_50") or {}).items():
        l3, l2 = v.get("Lambda_3_5_7"), v.get("Lambda_5_7")
        s += (f"- Λ {ent} on {{10, 30, 50}}: d = 3, 5, 7: " + (f"{l3['Lambda']:.3f} ± {l3['se']:.3f}" if l3 else "n/a")
              + "; d = 5/7: " + (f"{l2['Lambda']:.3f} ± {l2['se']:.3f}" if l2 else "n/a") + ".\n")
    s += "\n## Leaves (PR3 sample)\n\n| leaf | sample | ML errors | Libra errors | verdict |\n|---|---|---|---|---|\n"
    for lf, x in o["leaves"].items():
        s += f"| {lf.split('/', 1)[1]} | {x['n_sample']} | {x['errors_ML']} | {x['errors_libra']} | {x['verdict']} |\n"
    open(path, "w").write(s)


if __name__ == "__main__":
    main()
