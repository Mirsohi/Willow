"""Decoder's check of the frozen d = 5 P1 replication (g_5): 2x2 tables rebuilt with the frozen functions, entry 7's
covariance SE, K2 exclusion scenarios, the count ratio, discordance, and the ladder / extra-rung error-count changes.
Read-only; writes one json to this scratch folder."""
import itertools, json, math, os, sys, time
sys.dont_write_bytecode = True
X = "/mnt/project-files/tensor-networks/deep-dives/experimental"
sys.path.insert(0, X + "/code/decoder"); sys.path.insert(0, X + "/code/decoder/supplementary")
import numpy as np
import analyze_willow as AW
import p1_se_covariance as PC
W = X + "/results/decoder/willow"
EXP = "google_105Q_surface_code_d3_d5_d7"
t0 = time.time()
plan = json.load(open(W + "/plan.json")); PR = AW.planned_rounds(plan)
pts, leafinfo = {}, {}
for patch in ("q4_7", "q6_5", "q6_9", "q8_7"):
    for basis in ("X", "Z"):
        for r in (10, 13, 30, 50):
            rel = f"{EXP}/d5_at_{patch}/{basis}/r{r}"
            obs, ents, ml, info, shipped = AW.load_leaf(W, rel, ["rl_s5"])
            meta = info.get("meta") or {}
            lad = (ml.get("ML.rl") or {}).get("ladder")
            li = {"patch_meta": meta.get("patch"), "n_ml": int(np.sum((ents["ML.rl"] >= 0) & (obs >= 0))),
                  "k_ml": int(np.sum((ents["ML.rl"] >= 0) & (obs >= 0) & (ents["ML.rl"] != obs)))}
            if lad:
                li.update({k: lad.get(k) for k in ("errors_tau1", "errors_final", "n_redecoded", "decisions_changed",
                                                   "max_abs_dP_redecoded", "ms_per_round_per_shot_tau1",
                                                   "ms_per_round_per_redecoded_shot_tau2", "n")})
            if "libra.rl" in ents:
                c = AW.count2(ents["libra.rl"], ents["ML.rl"], obs)
                pts.setdefault((patch, basis), {})[r] = c
                li["c_libra_ml"] = c
            leafinfo[f"d5_at_{patch}/{basis}/r{r}"] = li
print("loaded", round(time.time() - t0), "s", flush=True)
mp = leafinfo["d5_at_q4_7/X/r10"]["patch_meta"]
def win(patch, basis):
    key = (5, mp if mp == patch else patch, basis)
    both = (PR.get(key + ("libra.rl",)) or set()) & (PR.get(key + ("ML.rl",)) or set())
    return tuple(sorted(x for x in both if x >= 10))

def gap(excl=(), drop_series=False):
    """g_5 with the leaves in excl ('patch/basis/rR') removed: rule A keeps the series on its other rounds (entry 8's
    way); drop_series=True drops every series that lost a leaf (rule B). A series left with < 2 rounds has no fit."""
    rows, S, cells, lost = [], [], [], []
    for (patch, basis), P in sorted(pts.items()):
        w = win(patch, basis)
        gone = [r for r in w if f"{patch}/{basis}/r{r}" in excl]
        if gone and drop_series:
            lost.append(f"{patch}/{basis}"); continue
        w2 = tuple(r for r in w if r not in gone)
        P2 = {r: c for r, c in P.items() if r in w2}
        f = AW.paired_series(P2, w2)
        if not f or not f["complete"]:
            lost.append(f"{patch}/{basis}"); continue
        rows.append(f); S.append(PC.series_params(P2, w2)); cells += [P2[r] for r in w2]
    pg = AW.pool_gap(rows)
    cs = PC.g_and_se(S, "inflated", full=True, with_cov=True)
    cr = AW.ratio_from_counts(cells)
    return {"n_series": len(rows), "lost_series": lost, "g": pg["rel"], "se_frozen": pg["se_rel"], "se_cov": cs["se"],
            "ci95_cov": cs["ci95"], "ci90_cov": cs["ci90"], "count_ratio": cr["rel"], "count_se": cr["se_rel"],
            "leaves": len(cells)}

out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "windows": {f"{p}/{b}": win(p, b) for p, b in pts}}
A = json.load(open(W + "/analysis/analysis.json"))
base = gap()
fz = A["pooled_gaps"]["d5|libra.rl vs ML.rl"]
out["base"] = base
out["frozen_check"] = {"frozen_g": fz["rel"], "frozen_se": fz["se_rel"], "rebuilt_g": base["g"], "rebuilt_se": base["se_frozen"],
                       "rel_diff_g": abs(base["g"] - fz["rel"]) / fz["rel"],
                       "frozen_count_ratio": A["pooled_counts"]["d5|libra.rl vs ML.rl"]["rel"], "rebuilt_count_ratio": base["count_ratio"]}
per = {}
for (patch, basis), P in sorted(pts.items()):
    f = AW.paired_series(P, win(patch, basis))
    k0 = A["paired_series"].get(f"d5|{mp if mp == patch else patch}|{basis}|libra.rl vs ML.rl")
    per[f"{patch}/{basis}"] = {"rel": f["rel"], "se_rel": f["se_rel"], "chi2_red": f["chi2_red"],
                               "frozen_rel": k0["rel"] if k0 else None}
out["per_series"] = per
fail_now = ["q4_7/Z/r13", "q6_9/X/r13", "q6_5/X/r30", "q8_7/Z/r10"]
out["excl_now_ruleA"] = gap(fail_now)
out["excl_now_ruleB"] = gap(fail_now, drop_series=True)
allw = [f"{p}/{b}/r{r}" for (p, b) in sorted(pts) for r in win(p, b)]
loo = {lf: gap([lf])["g"] for lf in allw}
out["leave_one_out"] = {"min": min(loo.values()), "max": max(loo.values()), "by_leaf": loo}
pending = ["q8_7/Z/r30", "q8_7/X/r30", "q6_9/Z/r30", "q4_7/Z/r50", "q6_5/Z/r50", "q4_7/X/r50", "q8_7/Z/r50",
           "q6_5/X/r50", "q8_7/X/r50", "q6_9/X/r50", "q6_9/Z/r50"]
sc = {}
for k in range(0, 5):
    gs, ns = [], []
    for extra in itertools.combinations(pending, k):
        res = gap(fail_now + list(extra))
        gs.append(res["g"]); ns.append(res["n_series"])
    sc[str(k)] = {"scenarios": len(gs), "g_min": min(gs), "g_max": max(gs), "min_series": min(ns)}
out["scenarios_more_failures_ruleA"] = sc
# discordance
disc = []
for (patch, basis), P in pts.items():
    for r, c in P.items():
        kM = c["n01"] + c["n11"]
        disc.append({"leaf": f"{patch}/{basis}/r{r}", "kM": kM, "kA": c["n10"] + c["n11"], "n10": c["n10"], "n01": c["n01"],
                     "f01": c["n01"] / kM, "D_over_kM": (c["n10"] + c["n01"]) / kM})
tot = {k: sum(x[k] for x in disc) for k in ("kM", "kA", "n10", "n01")}
out["discordance"] = {"pooled": dict(tot, f01=tot["n01"] / tot["kM"], D_over_kM=(tot["n10"] + tot["n01"]) / tot["kM"]),
                      "f01_range": [min(x["f01"] for x in disc), max(x["f01"] for x in disc)]}
# truncation: ladder (tau1 -> tau2 on the band shots) and extra rung (tau2 -> tau3)
lad = [v for v in leafinfo.values() if v.get("errors_tau1") is not None]
out["ladder_tau1_to_tau2"] = {"leaves": len(lad), "sum_errors_tau1": sum(v["errors_tau1"] for v in lad),
                              "sum_errors_final": sum(v["errors_final"] for v in lad),
                              "down": sum(v["errors_final"] < v["errors_tau1"] for v in lad),
                              "same": sum(v["errors_final"] == v["errors_tau1"] for v in lad),
                              "up": sum(v["errors_final"] > v["errors_tau1"] for v in lad),
                              "sum_decisions_changed": sum(v.get("decisions_changed") or 0 for v in lad),
                              "band_share_by_r": {str(r): [round(v["n_redecoded"] / v["n"], 4) for k, v in leafinfo.items()
                                                            if k.endswith(f"/r{r}") and v.get("n")] for r in (10, 13, 30, 50)},
                              "tau1_ms_by_r": {str(r): [round(v["ms_per_round_per_shot_tau1"], 2) for k, v in leafinfo.items()
                                                        if k.endswith(f"/r{r}") and v.get("ms_per_round_per_shot_tau1")] for r in (10, 13, 30, 50)},
                              "tau2_ms_by_r": {str(r): [round(v["ms_per_round_per_redecoded_shot_tau2"], 1) for k, v in leafinfo.items()
                                                        if k.endswith(f"/r{r}") and v.get("ms_per_round_per_redecoded_shot_tau2")] for r in (10, 13, 30, 50)}}
K = W + "/supplementary/k2_extra_rung/" + EXP
ex = []
for dp, dn, fn in os.walk(K):
    if "extra_rung.json" in fn:
        e = json.load(open(os.path.join(dp, "extra_rung.json")))
        t3 = e.get("tau3") or {}
        if "errors_with_tau3" in t3:
            ex.append({"leaf": e["leaf"].split("/", 1)[1], "k2": t3["errors_ladder_final"], "k3": t3["errors_with_tau3"],
                       "changed": t3.get("decisions_changed_tau2_tau3"), "verdict": e["verdict"][:10],
                       "ms3": t3.get("ms_per_round_per_band_shot_tau3")})
out["extra_rung_tau2_to_tau3"] = {"leaves": len(ex), "sum_k2": sum(x["k2"] for x in ex), "sum_k3": sum(x["k3"] for x in ex),
                                  "down": sum(x["k3"] < x["k2"] for x in ex), "same": sum(x["k3"] == x["k2"] for x in ex),
                                  "up": sum(x["k3"] > x["k2"] for x in ex),
                                  "sum_decisions_changed": sum(x["changed"] or 0 for x in ex), "rows": ex}
out["leafinfo"] = leafinfo
out["seconds"] = time.time() - t0
json.dump(out, open("/home/claude/work/decoder/pr3/d5_gap_check.json", "w"), indent=1, default=float)
show = {k: out[k] for k in ("frozen_check", "base", "excl_now_ruleA", "excl_now_ruleB", "scenarios_more_failures_ruleA", "discordance")}
show["loo"] = {k: out["leave_one_out"][k] for k in ("min", "max")}
show["ladder"] = {k: v for k, v in out["ladder_tau1_to_tau2"].items()}
show["extra"] = {k: v for k, v in out["extra_rung_tau2_to_tau3"].items() if k != "rows"}
print(json.dumps(show, indent=1, default=float))
