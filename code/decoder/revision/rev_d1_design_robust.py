#!/usr/bin/env python3
"""D1-8e (review 1.3; after D1-8c, 24 Oct): D1-8c's search with the one-disagreement condition built in (decoder,
2026-10-24). A post hoc, descriptive projection, synthetic only: no decoding, no label or test. Declared in
notes/decoder/willow_revision_declarations.md (D1-8e, D1-8f).

Everything is as in rev_d1_design.py (D1-8c, 2239e6b0...) except the greedy's objective and the budgets:
  each step minimises S = W + 4 x max(0, 0.005 - L1) per core-hour, with W the surrogate's 95% width and L1 the
  surrogate's lower end after one disagreement on the adverse side that lowers it most (the lower end itself if there is
  no such side). Variant "tested": L1 over the adverse sides that receive shots (main's test); variant "all": over every
  adverse side. Budgets 2, 4, 6 and 8 core-hours (MPS + sparse).
Machinery: D1-8's. rev_d1_project.py (98fb097e..., imported unchanged) supplies the frozen analysis's census inputs and
posterior loop (rev_d1_analyze.py ef15b054...). Added shots come from rule A's 19 leaves (rule none gains the same shots),
never more than a side's unsampled shots under rule A. The surrogate: g_5 ~ g_5(census) + sum_j h_j(theta_j) + se_0 z, as
in D1-8c (20,000 fixed uniforms per side, seed 20261018). The designs are evaluated with the full machinery, synthetic and
on D1's real rows, with one disagreement on each adverse side that receives shots. The recommendation applies main's rule
to D1-8c's designs (read from d1_design.json) and these together.
Validation first: D1-8's stored (c) +250 scenarios (synthetic and real rows) are reproduced exactly, or the script stops.
usage: python3 -I -B rev_d1_design_robust.py --res R --sel SEL --final-r1 J --check C --d1 D1.json --synthetic SY.json
           --projection d1_projection.json --design-c d1_design.json --out DIR [--validate-only]
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.special import betaincinv

PROJECT_MD5 = "98fb097ea179a7b71e64ad420b32ba26"     # rev_d1_project.py (D1-8b)
RD2_MD5 = "03e905c11c3cb45e45eecceb81812bfb"         # rev_d2_worstcase.py (frozen)
FROZEN = "ef15b0548df048d163005c8b08c995a1"          # rev_d1_analyze.py (frozen)
DESIGN_C_SCRIPT_MD5 = "2239e6b0fb2466c72d8737ba28edb69a"   # rev_d1_design.py (D1-8d)
BUDGETS_H = (2.0, 4.0, 6.0, 8.0)
MAIN_BUDGETS_H = (2.0, 4.0, 6.0)
QUANTUM_S = 36.0
DRAWS_S = 20000
SEED_S = 20261018
HEADLINE_W = 0.0909 - 0.0423                         # the headline's 95% CI, [4.23, 9.09]%: 4.86 points
WORKERS = 3
PENALTY, MARGIN = 4.0, 0.005
GRID = np.concatenate([[0.0], np.logspace(-6, 0, 49)])
SIDES = ("right", "wrong")
REF_SYN = "(c) adverse sides, +250"
REF_REAL = "(real) adverse sides, +250 on D1's real rows"
CURRENT = "current (D1-5b's synthetic case)"
KQ = (int(math.floor((DRAWS_S - 1) * 0.025)), int(math.floor((DRAWS_S - 1) * 0.975)))
FQ = ((DRAWS_S - 1) * 0.025 - KQ[0], (DRAWS_S - 1) * 0.975 - KQ[1])


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def jkey(s):
    k, sd = s.rsplit(":", 1)
    return (k, sd)


def jname(j):
    return f"{j[0]}:{j[1]}"


def p_none(k, n, d):
    """D1's posterior chance of no disagreement among d new shots: E[(1 - theta)^d], theta ~ Beta(k + 1/2, n - k + 1/2)."""
    a, b = k + 0.5, n - k + 0.5
    return math.exp(math.lgamma(b + d) + math.lgamma(a + b) - math.lgamma(b) - math.lgamma(a + b + d))


def lo(x):
    """numpy's linear 2.5% quantile, by partition."""
    p = np.partition(x, (KQ[0], KQ[0] + 1))
    return float(p[KQ[0]] + FQ[0] * (p[KQ[0] + 1] - p[KQ[0]]))


def lo_hi(x):
    p = np.partition(x, (KQ[0], KQ[0] + 1, KQ[1], KQ[1] + 1))
    return (float(p[KQ[0]] + FQ[0] * (p[KQ[0] + 1] - p[KQ[0]])), float(p[KQ[1]] + FQ[1] * (p[KQ[1] + 1] - p[KQ[1]])))


def compare(got, ref, where):
    worst, checked = 0.0, 0
    assert set(got) == set(ref), f"{where}: keys differ"
    for k, x in got.items():
        y = ref[k]
        xs, ys = (x, y) if isinstance(x, list) else ([x], [y])
        assert len(xs) == len(ys), f"{where} {k}"
        for u, w in zip(xs, ys):
            checked += 1
            if isinstance(u, str) or isinstance(w, str) or u is None or w is None:
                assert u == w, f"{where} {k}"
                continue
            worst = max(worst, abs(float(u) - float(w)))
    return worst, checked


def main():
    ap = argparse.ArgumentParser()
    for x in ("--res", "--sel", "--final-r1", "--check", "--d1", "--synthetic", "--projection", "--design-c", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--validate-only", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time()
    pj_path, rd2_path = os.path.join(HERE, "rev_d1_project.py"), os.path.join(HERE, "rev_d2_worstcase.py")
    assert md5(pj_path) == PROJECT_MD5, "rev_d1_project.py differs from D1-8b's"
    assert md5(rd2_path) == RD2_MD5, "rev_d2_worstcase.py differs from the frozen one"
    PJ = load_module("rev_d1_project", pj_path)
    rd2 = load_module("rd2", rd2_path)
    meta, leaves, info, T, popc, Fc, exA, inputs = PJ.load_census(a_)
    D1 = json.load(open(a_.d1)); SY = json.load(open(a_.synthetic)); P = json.load(open(a_.projection))
    DC = json.load(open(a_.design_c))
    inputs.update({"d1_results.json": md5(a_.d1), "synthetic_d1_results.json": md5(a_.synthetic),
                   "d1_projection.json": md5(a_.projection), "d1_design.json": md5(a_.design_c),
                   "rev_d1_project.py": PROJECT_MD5})
    assert D1["script_md5"] == FROZEN and SY["script_md5"] == FROZEN, "an output not made by the frozen analysis"
    assert D1["rule_A_excluded"] == exA and SY["rule_A_excluded"] == exA and P["rule_A_excluded"] == exA
    assert P["script_md5"] == PROJECT_MD5, "d1_projection.json was not made by D1-8's script"
    assert DC["script_md5"] == DESIGN_C_SCRIPT_MD5, "d1_design.json was not made by D1-8c's script"
    for k_, v_ in P["inputs_md5"].items():
        if k_ in inputs:
            assert inputs[k_] == v_, f"input {k_} differs from the one D1-8 read"
    for k_, v_ in DC["inputs_md5"].items():
        if k_ in inputs:
            assert inputs[k_] == v_, f"input {k_} differs from the one D1-8c read"
    MA = {rule: PJ.Machinery(rd2, leaves, info, T, popc, Fc, ex) for rule, ex in (("A", exA), ("none", []))}
    syn_cells = {rule: PJ.stored_counts(SY["rules"][rule])[0] for rule in ("A", "none")}
    real_cells = {rule: PJ.stored_counts(D1["rules"][rule])[0] for rule in ("A", "none")}
    for rule in ("A", "none"):
        assert all(syn_cells[rule][k][sd]["n"] == real_cells[rule][k][sd]["n"] and syn_cells[rule][k][sd]["dis"] == 0
                   for k in real_cells[rule] for sd in SIDES), "the synthetic case's counts differ from D1's sample"
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "post hoc, descriptive projection (D1-8e); D1-5b's synthetic machinery; no decoding, no label or test",
           "inputs_md5": inputs, "rule_A_excluded": exA,
           "objective": f"S = W + {PENALTY:g} x max(0, {MARGIN:g} - L1) per core-hour"}

    def evaluate(add, base, rules=("A", "none"), levels=("cells", "pooled"), extra_dis=None):
        res = {}
        for rule in rules:
            mc = MA[rule]
            src = syn_cells[rule] if base == "synthetic" else real_cells[rule]
            Gc = PJ.with_added(src, add, zero_dis=(base == "synthetic"))
            for (key, sd), v in (extra_dis or {}).items():
                Gc[key][sd]["dis"] += v
            for k in Gc:
                for sd in SIDES:
                    assert Gc[k][sd]["n"] <= mc.Npc[k][sd], f"{rule} {k} {sd} above its population"
                    assert 0 <= Gc[k][sd]["dis"] <= Gc[k][sd]["n"]
            Gp = PJ.pooled_of(Gc)
            rr = {"counts": {f"{k}:{sd}": Gc[k][sd]["n"] for k in sorted(Gc) for sd in SIDES}}
            for level in levels:
                rr[level] = mc.posterior(Gc, Gp, level, 20261016 if level == "cells" else 20261017)
            res[rule] = rr
        return res

    # ------------------------------------------------------------ validation: D1-8's (c) +250, exactly
    scen = {s["name"]: s for s in P["scenarios"]}
    val = {}
    for nm, base in ((REF_SYN, "synthetic"), (REF_REAL, "real")):
        s = scen[nm]
        assert s["base"] == base
        add = {jkey(k): int(v) for k, v in s["added"].items()}
        got = evaluate(add, base)
        worst, checked = 0.0, 0
        for rule in ("A", "none"):
            assert got[rule]["counts"] == s["rules"][rule]["counts"], f"{nm} {rule}: counts differ"
            for level in ("cells", "pooled"):
                w_, c_ = compare(got[rule][level], s["rules"][rule][level], f"{nm} {rule} {level}")
                worst = max(worst, w_); checked += c_
        val[nm] = {"fields_checked": checked, "max_abs_diff": worst}
        print(json.dumps({"validation": nm, **val[nm], "t": round(time.time() - t0, 1)}), flush=True)
        if worst != 0.0:
            sys.exit(f"validation failed on {nm}: max abs diff {worst}")
    out["validation"] = val
    if a_.validate_only:
        print(json.dumps({"validation": val, "seconds": time.time() - t0}))
        return

    # ------------------------------------------------------------ sides, costs, sensitivities (from D1-8)
    mA, mN = MA["A"], MA["none"]
    sides = [(key, sd) for key in sorted(mA.Npc) for sd in SIDES if mA.Npc[key][sd] > 0]
    cps = P["cost_per_shot"]["cells"]
    sens = {jkey(k): v for k, v in P["sensitivity"].items()}
    adverse = [j for j in sides if sens[j]["adverse"]]
    adv_set = set(adverse)

    def c_mps(j, basis):
        c = cps[j[0]]; cell = c["mps_core_s_per_shot"]; side = c.get(f"mps_core_s_per_shot_{j[1]}_side")
        return max(cell, side) if (basis == "conservative" and side is not None) else cell

    def c_spr(j):
        return cps[j[0]]["sparse_core_s_per_shot"]

    cost_s = {j: c_mps(j, "conservative") + c_spr(j) for j in sides}

    def cost_h(add, basis="conservative"):
        m = sum(v * c_mps(j, basis) for j, v in add.items()) / 3600
        s = sum(v * c_spr(j) for j, v in add.items()) / 3600
        return {"mps": m, "sparse": s, "total": m + s}

    n0 = {j: syn_cells["A"][j[0]][j[1]]["n"] for j in sides}
    cap = {j: mA.Npc[j[0]][j[1]] - n0[j] for j in sides}

    # ------------------------------------------------------------ the surrogate (as D1-8c)
    g0d = mA.g5_at_rates({})[0]; g0, se0 = float(g0d["g"]), float(g0d["se"])
    resp = {}
    for j in sides:
        hv = []
        for th in GRID:
            g = mA.g5_at_rates({j: float(th)})[0]["g"]
            hv.append(float(g) - g0 if (g is not None and math.isfinite(float(g))) else float("nan"))
        hv = np.array(hv)
        bad = np.where(~np.isfinite(hv))[0]
        last = len(hv) - 1 if len(bad) == 0 else int(bad[0]) - 1
        assert last >= 1, f"{j}: g_5 is not finite on the grid"
        resp[j] = (GRID[:last + 1].copy(), hv[:last + 1].copy())
    rngS = np.random.default_rng(SEED_S)
    U = {j: rngS.random(DRAWS_S) for j in sides}
    Zs = rngS.standard_normal(DRAWS_S)
    print(json.dumps({"surrogate": "response curves", "g0": g0, "se0": se0, "t": round(time.time() - t0, 1)}), flush=True)

    def term(j, n, k):
        th = betaincinv(k + 0.5, n - k + 0.5, U[j])
        x, y = resp[j]
        return np.interp(np.minimum(th, x[-1]), x, y)

    def surrogate_q(nmap, kmap):
        S = g0 + se0 * Zs
        for j in sides:
            S = S + term(j, nmap[j], kmap[j])
        return [float(v) for v in np.quantile(S, [0.025, 0.5, 0.975])]

    # ------------------------------------------------------------ greedy search with the one-disagreement condition
    step = {j: max(1, int(round(QUANTUM_S / cost_s[j]))) for j in sides}
    B = sorted(BUDGETS_H)

    def run_greedy(variant):
        n = dict(n0); added = {j: 0 for j in sides}
        H = {j: term(j, n[j], 0) for j in sides}
        H1 = {j: term(j, n[j], 1) for j in adverse}
        S = g0 + se0 * Zs + sum(H.values())

        def score(Sv, j_new=None, Hn=None, H1n=None, d=0):
            L, U_ = lo_hi(Sv)
            L1 = L
            for i in adverse:
                tested = variant == "all" or (added[i] + (d if i == j_new else 0)) > 0
                if not tested:
                    continue
                if i == j_new:
                    L1 = min(L1, lo(Sv - Hn + H1n))
                else:
                    L1 = min(L1, lo(Sv - H[i] + H1[i]))
            W_ = U_ - L
            return W_ + PENALTY * max(0.0, MARGIN - L1), W_, L1, L

        cur = score(S)
        start = cur

        def make_cand(j):
            d = min(step[j], cap[j] - added[j])
            if d <= 0:
                return None
            return d, term(j, n[j] + d, 0), (term(j, n[j] + d, 1) if j in adv_set else None)

        cand = {j: make_cand(j) for j in sides}
        spent, snaps, path = 0.0, {}, []
        bi = 0
        while bi < len(B):
            best = None
            for j in sides:
                c = cand[j]
                if c is None:
                    continue
                d, Hn, H1n = c
                S2 = S - H[j] + Hn
                sc = score(S2, j, Hn, H1n, d)
                gain = (cur[0] - sc[0]) / (d * cost_s[j] / 3600)
                if best is None or gain > best[0]:
                    best = (gain, j, d, Hn, H1n, S2, sc)
            if best is None:
                break
            gain, j, d, Hn, H1n, S2, sc = best
            dc = d * cost_s[j] / 3600
            while bi < len(B) and spent + dc > B[bi] + 1e-12:
                snaps[B[bi]] = {"added": {jname(x): v for x, v in added.items() if v}, "cost_h": spent,
                                "surrogate": {"S": cur[0], "W": cur[1], "L1": cur[2], "L": cur[3]}}
                bi += 1
            if bi >= len(B):
                break
            S = S2; H[j] = Hn
            if j in adv_set:
                H1[j] = H1n
            n[j] += d; added[j] += d; spent += dc; cur = sc
            path.append({"side": jname(j), "shots": d, "cost_h": spent, "S": sc[0], "W": sc[1], "L1": sc[2], "gain_per_h": gain})
            cand[j] = make_cand(j)
        for b in B[bi:]:
            snaps[b] = {"added": {jname(x): v for x, v in added.items() if v}, "cost_h": spent,
                        "surrogate": {"S": cur[0], "W": cur[1], "L1": cur[2], "L": cur[3]}}
        return snaps, path, start

    greedy = {}
    for variant in ("tested", "all"):
        snaps, path, start = run_greedy(variant)
        greedy[variant] = {"steps": len(path), "start": {"S": start[0], "W": start[1], "L1": start[2], "L": start[3]},
                           "path_every_0_25h": [p for i, p in enumerate(path) if i == len(path) - 1
                                                or int(p["cost_h"] / 0.25) != int(path[i + 1]["cost_h"] / 0.25)],
                           "snapshots": {f"{b:g}": v for b, v in snaps.items()}}
        print(json.dumps({"greedy": variant, "steps": len(path),
                          "snapshots": {f"{b:g}": (round(v["cost_h"], 3), round(v["surrogate"]["W"], 5), round(v["surrogate"]["L1"], 5))
                                        for b, v in snaps.items()}, "t": round(time.time() - t0, 1)}), flush=True)
    out["greedy"] = {"quantum_core_s": QUANTUM_S, "penalty": PENALTY, "margin": MARGIN, **greedy}

    # ------------------------------------------------------------ full evaluation of the designs
    designs = []
    for variant in ("tested", "all"):
        for b in B:
            designs.append((f"{variant}: budget {b:g} core-hours", variant, b,
                            {jkey(k): v for k, v in greedy[variant]["snapshots"][f"{b:g}"]["added"].items()}))
    W_cur = (lambda q: q[2] - q[0])(scen[CURRENT]["rules"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"])
    res_d = []
    for name, variant, b, add in designs:
        for j, v in add.items():
            assert 0 < v <= cap[j], f"{name}: {j} gets {v} of {cap[j]} unsampled"
        d = {"name": name, "variant": variant, "budget_h": b, "within_main_budgets": b in MAIN_BUDGETS_H,
             "added": {jname(j): v for j, v in sorted(add.items())}, "added_total": int(sum(add.values())),
             "cost_core_hours": cost_h(add), "cost_core_hours_cell_means": cost_h(add, "cell")}
        d["wall_hours_on_3_workers"] = d["cost_core_hours"]["total"] / WORKERS
        nm_ = {j: n0[j] + add.get(j, 0) for j in sides}
        d["surrogate_q025_q50_q975"] = surrogate_q(nm_, {j: 0 for j in sides})
        scale = {j: mN.Npc[j[0]][j[1]] / mA.Npc[j[0]][j[1]] for j in add}
        d["if_drawn_from_all_24_leaves"] = {"shots": float(sum(v * scale[j] for j, v in add.items())),
                                            "cost_core_hours": cost_h({j: v * scale[j] for j, v in add.items()})}
        for base in ("synthetic", "real"):
            r = evaluate(add, base)
            d[base] = {rule: {lv: r[rule][lv] for lv in ("cells", "pooled")} for rule in ("A", "none")}
            q = r["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"]
            d[base]["A_cells_width"] = q[2] - q[0]
            d[base]["A_cells_width_x_headline"] = (q[2] - q[0]) / HEADLINE_W
        adv = [j for j in sorted(add) if j in adv_set]
        one = []
        for j in adv:
            e = {"side": jname(j), "added": add[j]}
            for base in ("synthetic", "real"):
                r = evaluate(add, base, rules=("A",), levels=("cells",), extra_dis={j: 1})
                e[f"A_cells_{base}_q025_q50_q975"] = r["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"]
            kr, nr = real_cells["A"][j[0]][j[1]]["dis"], real_cells["A"][j[0]][j[1]]["n"]
            e["P_at_least_one"] = 1 - p_none(kr, nr, add[j])
            one.append(e)
        d["one_disagreement"] = {"by_side": one}
        if one:
            wv = min(one, key=lambda e: e["A_cells_synthetic_q025_q50_q975"][0])
            d["one_disagreement"]["worst"] = {"side": wv["side"], "lower_synthetic": wv["A_cells_synthetic_q025_q50_q975"][0],
                                              "lower_real": wv["A_cells_real_q025_q50_q975"][0],
                                              "P_at_least_one": wv["P_at_least_one"]}
            pn = 1.0
            for j in adv:
                pn *= p_none(real_cells["A"][j[0]][j[1]]["dis"], real_cells["A"][j[0]][j[1]]["n"], add[j])
            d["one_disagreement"]["P_at_least_one_on_any_adverse_side"] = 1 - pn
        w_ = d["synthetic"]["A_cells_width"]
        d["narrowing_per_core_hour"] = (W_cur - w_) / d["cost_core_hours"]["total"]
        low = d["synthetic"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"][0]
        d["survives_one_disagreement"] = bool(low > 0 and (not one or d["one_disagreement"]["worst"]["lower_synthetic"] > 0))
        d["width_at_most_twice_headline"] = bool(w_ <= 2 * HEADLINE_W)
        res_d.append(d)
        print(json.dumps({"design": name, "added": d["added_total"], "cost_h": round(d["cost_core_hours"]["total"], 3),
                          "A_cells_syn": [round(x, 5) for x in d["synthetic"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"]],
                          "width_x": round(d["synthetic"]["A_cells_width_x_headline"], 3),
                          "worst": d["one_disagreement"].get("worst"), "survives": d["survives_one_disagreement"],
                          "t": round(time.time() - t0, 1)}), flush=True)
    out["designs"] = res_d
    out["current_synthetic_width"] = W_cur
    # ------------------------------------------------------------ recommendation: main's rule over D1-8c's and these
    pool = []
    for d in DC["designs"]:
        pool.append({"name": "D1-8c " + d["name"], "kind": d["kind"],
                     "budget_h": (float(d["name"].split()[1]) if d["kind"] == "budget" else None),
                     "survives": d["survives_one_disagreement"], "width": d["synthetic"]["A_cells_width"],
                     "narrowing_per_core_hour": d["narrowing_per_core_hour"], "cost_h": d["cost_core_hours"]["total"]})
    for d in res_d:
        pool.append({"name": "D1-8e " + d["name"], "kind": "budget", "budget_h": d["budget_h"],
                     "survives": d["survives_one_disagreement"], "width": d["synthetic"]["A_cells_width"],
                     "narrowing_per_core_hour": d["narrowing_per_core_hour"], "cost_h": d["cost_core_hours"]["total"]})
    rec, why = None, None
    for b in sorted({p["budget_h"] for p in pool if p["budget_h"] is not None}):
        ok = [p for p in pool if p["budget_h"] == b and p["survives"] and p["width"] <= 2 * HEADLINE_W]
        if ok:
            pick = min(ok, key=lambda p: p["width"])
            rec = pick["name"]
            why = ("smallest budget with a design that survives one disagreement and is at most twice the headline's width"
                   + ("" if b in MAIN_BUDGETS_H else "; beyond main's budgets of 2, 4 and 6, since none within them qualifies"))
            break
    if rec is None:
        ok = [p for p in pool if p["survives"]]
        if ok:
            pick = max(ok, key=lambda p: p["narrowing_per_core_hour"])
            rec, why = pick["name"], "no design qualifies; best narrowing per core-hour among the designs that survive"
    out["recommendation_pool"] = pool
    out["recommendation"] = {"design": rec, "why": why}
    out["seconds"] = time.time() - t0
    os.makedirs(a_.out, exist_ok=True)
    json.dump(out, open(os.path.join(a_.out, "d1_design_robust.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(a_.out, "d1_design_robust.md"))
    print(json.dumps({"recommendation": out["recommendation"], "seconds": out["seconds"]}))


def write_md(o, path):
    pc = lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"
    q3 = lambda q: ", ".join(pc(x) for x in q)
    L = ["# D1-8e: D1-8c's search with the one-disagreement condition built in", "",
         f"Made {o['made']} by `rev_d1_design_robust.py` (md5 {o['script_md5']}). Post hoc, descriptive projection with "
         f"D1-5b's synthetic machinery; no decoding, no label or test. Objective: {o['objective']}.", "",
         f"Validation: {json.dumps(o['validation'])}", "",
         f"Recommendation (main's rule over D1-8c's and D1-8e's designs): {json.dumps(o['recommendation'])}", "",
         "| design | added shots | MPS h | sparse h | total h | wall h, 3 workers | rule A per cell, synthetic: 2.5/50/97.5% | "
         "width (x headline) | rule A per cell, real rows | rule none per cell, synthetic | rule none per cell, real | "
         "rule A pooled, synthetic | rule A pooled, real | worst single disagreement: side, lower end synthetic / real, P(>= 1) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in o["designs"]:
        c = d["cost_core_hours"]; s, r = d["synthetic"], d["real"]
        wv = d["one_disagreement"].get("worst")
        wtxt = (f"{wv['side']}, {pc(wv['lower_synthetic'])} / {pc(wv['lower_real'])}, {wv['P_at_least_one']:.3f}" if wv else "none")
        L.append(f"| {d['name']} | {d['added_total']:,} | {c['mps']:.2f} | {c['sparse']:.2f} | {c['total']:.2f} | "
                 f"{d['wall_hours_on_3_workers']:.2f} | {q3(s['A']['cells']['g5_with_shot_noise_q025_q50_q975'])} | "
                 f"{100 * s['A_cells_width']:.2f} ({s['A_cells_width_x_headline']:.2f}) | "
                 f"{q3(r['A']['cells']['g5_with_shot_noise_q025_q50_q975'])} | "
                 f"{q3(s['none']['cells']['g5_with_shot_noise_q025_q50_q975'])} | "
                 f"{q3(r['none']['cells']['g5_with_shot_noise_q025_q50_q975'])} | "
                 f"{q3(s['A']['pooled']['g5_with_shot_noise_q025_q50_q975'])} | "
                 f"{q3(r['A']['pooled']['g5_with_shot_noise_q025_q50_q975'])} | {wtxt} |")
    L += ["", "## Allocations (added shots by cell side)", ""]
    for d in o["designs"]:
        L.append(f"- {d['name']}: {json.dumps(d['added'])}")
    L += ["", "## One disagreement on each adverse side that receives shots (rule A, per cell, lower end)", ""]
    for d in o["designs"]:
        for e in d["one_disagreement"]["by_side"]:
            L.append(f"- {d['name']}, {e['side']} (+{e['added']}): synthetic {pc(e['A_cells_synthetic_q025_q50_q975'][0])}, "
                     f"real {pc(e['A_cells_real_q025_q50_q975'][0])}, P(>= 1) {e['P_at_least_one']:.4f}")
    L += ["", "## The pool for main's rule", "", "| design | budget | survives | width (points) | narrowing per core-hour |",
          "|---|---|---|---|---|"]
    for p in o["recommendation_pool"]:
        L.append(f"| {p['name']} | {p['budget_h']} | {p['survives']} | {100 * p['width']:.2f} | {100 * p['narrowing_per_core_hour']:.2f} |")
    L += [""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
