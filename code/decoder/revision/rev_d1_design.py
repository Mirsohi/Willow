#!/usr/bin/env python3
"""D1-8c (review 1.3; main's message of about 11:30Z on 24 Oct): a post hoc, descriptive projection of the added shots
that narrow D1's primary interval (rule A, per cell: g_5's 95% interval with shot noise) most per core-hour, at total
budgets of 2, 4 and 6 core-hours (MPS + sparse) (decoder, 2026-10-24). Synthetic only: no decoding, no label or test.
Declared in notes/decoder/willow_revision_declarations.md (D1-8c, D1-8d).

Machinery: D1-8's. rev_d1_project.py (98fb097e..., imported unchanged) supplies the frozen analysis's census inputs and
posterior loop (rev_d1_analyze.py ef15b054...): a Jeffreys rate per cell and side, the census part fixed, 2,000 draws
(seed 20261016 per cell, 20261017 pooled), g_5 plus its shot-noise SE times a normal draw. Added shots come from rule A's
19 leaves (rule none gains the same shots), never more than a side's unsampled shots under rule A.
Search: a surrogate chooses the allocations, and only that. g_5 ~ g_5(census) + sum_j h_j(theta_j) + se_0 z, with h_j the
full model's response to side j's rate alone (rule A) on a grid of rates, interpolated linearly; theta_j from the same
Jeffreys posteriors by common random numbers (20,000 fixed uniforms per side, seed 20261018); se_0 the census shot-noise
SE. A greedy search adds about 0.01 core-hours of shots per step to the side that narrows the surrogate's 95% interval
most per core-hour. The chosen allocations, and D1-8's (c) +250 as the reference, are evaluated with the full machinery,
synthetic and on D1's real rows, and with one disagreement on each adverse side that receives shots.
Validation first: D1-8's stored (c) +250 scenarios (synthetic and real rows) are reproduced exactly, or the script stops.
usage: python3 -I -B rev_d1_design.py --res R --sel SEL --final-r1 J --check C --d1 D1.json --synthetic SY.json
           --projection d1_projection.json --out DIR [--validate-only]
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.special import betaincinv

PROJECT_MD5 = "98fb097ea179a7b71e64ad420b32ba26"     # rev_d1_project.py (D1-8b)
RD2_MD5 = "03e905c11c3cb45e45eecceb81812bfb"         # rev_d2_worstcase.py (frozen)
FROZEN = "ef15b0548df048d163005c8b08c995a1"          # rev_d1_analyze.py (frozen)
BUDGETS_H = (2.0, 4.0, 6.0)
QUANTUM_S = 36.0                                     # about 0.01 core-hours of shots per greedy step
DRAWS_S = 20000
SEED_S = 20261018
HEADLINE_W = 0.0909 - 0.0423                         # the headline's 95% CI, [4.23, 9.09]%: 4.86 points
WORKERS = 3
GRID = np.concatenate([[0.0], np.logspace(-6, 0, 49)])
SIDES = ("right", "wrong")
REF_SYN = "(c) adverse sides, +250"
REF_REAL = "(real) adverse sides, +250 on D1's real rows"
CURRENT = "current (D1-5b's synthetic case)"


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
    for x in ("--res", "--sel", "--final-r1", "--check", "--d1", "--synthetic", "--projection", "--out"):
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
    inputs.update({"d1_results.json": md5(a_.d1), "synthetic_d1_results.json": md5(a_.synthetic),
                   "d1_projection.json": md5(a_.projection), "rev_d1_project.py": PROJECT_MD5})
    assert D1["script_md5"] == FROZEN and SY["script_md5"] == FROZEN, "an output not made by the frozen analysis"
    assert D1["rule_A_excluded"] == exA and SY["rule_A_excluded"] == exA and P["rule_A_excluded"] == exA
    assert P["script_md5"] == PROJECT_MD5, "d1_projection.json was not made by D1-8's script"
    for k_, v_ in P["inputs_md5"].items():
        if k_ in inputs:
            assert inputs[k_] == v_, f"input {k_} differs from the one D1-8 read"
    MA = {rule: PJ.Machinery(rd2, leaves, info, T, popc, Fc, ex) for rule, ex in (("A", exA), ("none", []))}
    syn_cells = {rule: PJ.stored_counts(SY["rules"][rule])[0] for rule in ("A", "none")}
    real_cells = {rule: PJ.stored_counts(D1["rules"][rule])[0] for rule in ("A", "none")}
    for rule in ("A", "none"):
        assert all(syn_cells[rule][k][sd]["n"] == real_cells[rule][k][sd]["n"] and syn_cells[rule][k][sd]["dis"] == 0
                   for k in real_cells[rule] for sd in SIDES), "the synthetic case's counts differ from D1's sample"
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "post hoc, descriptive projection (D1-8c); D1-5b's synthetic machinery; no decoding, no label or test",
           "inputs_md5": inputs, "rule_A_excluded": exA}

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

    # ------------------------------------------------------------ the surrogate
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

    # check of the surrogate against the full machinery (D1-8's stored scenarios and D1 itself)
    chk = []
    rows_ = [("D1 as measured (real rows)", "real", {j: real_cells["A"][j[0]][j[1]]["n"] for j in sides},
              D1["rules"]["A"]["posterior_diff_cells"]["g5_with_shot_noise_q025_q50_q975"])]
    for s in P["scenarios"]:
        if s["family"] == "c_seeds":
            continue
        cnt = s["rules"]["A"]["counts"]
        rows_.append((s["name"], s["base"], {j: int(cnt[jname(j)]) for j in sides},
                      s["rules"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"]))
    for nm, base, nmap, fq in rows_:
        kmap = {j: (real_cells["A"][j[0]][j[1]]["dis"] if base == "real" else 0) for j in sides}
        sq = surrogate_q(nmap, kmap)
        chk.append({"name": nm, "base": base, "surrogate_q025_q50_q975": sq, "full_q025_q50_q975": fq,
                    "diff": [a - b for a, b in zip(sq, fq)], "width_surrogate": sq[2] - sq[0], "width_full": fq[2] - fq[0]})
    out["surrogate_check"] = chk
    print(json.dumps({"surrogate_check_max_abs_diff": max(abs(d) for c in chk for d in c["diff"]),
                      "t": round(time.time() - t0, 1)}), flush=True)

    # ------------------------------------------------------------ greedy search (synthetic, rule A per cell)
    k0 = {j: 0 for j in sides}
    n = dict(n0); added = {j: 0 for j in sides}
    H = {j: term(j, n[j], 0) for j in sides}
    S = g0 + se0 * Zs + sum(H.values())

    def width(x):
        q = np.quantile(x, [0.025, 0.975])
        return float(q[1] - q[0])

    W = width(S)
    W_start = W
    step = {j: max(1, int(round(QUANTUM_S / cost_s[j]))) for j in sides}

    def make_cand(j):
        d = min(step[j], cap[j] - added[j])
        return None if d <= 0 else (d, term(j, n[j] + d, k0[j]))

    cand = {j: make_cand(j) for j in sides}
    spent, snaps, path = 0.0, {}, []
    B = sorted(BUDGETS_H); bi = 0
    while bi < len(B):
        best = None
        for j in sides:
            c = cand[j]
            if c is None:
                continue
            d, Hn = c
            Wn = width(S - H[j] + Hn)
            gain = (W - Wn) / (d * cost_s[j] / 3600)
            if best is None or gain > best[0]:
                best = (gain, j, d, Hn, Wn)
        if best is None:
            break
        gain, j, d, Hn, Wn = best
        dc = d * cost_s[j] / 3600
        while bi < len(B) and spent + dc > B[bi] + 1e-12:
            snaps[B[bi]] = {"added": {jname(x): v for x, v in added.items() if v}, "cost_h": spent, "surrogate_width": W}
            bi += 1
        if bi >= len(B):
            break
        S = S - H[j] + Hn; H[j] = Hn; W = Wn; n[j] += d; added[j] += d; spent += dc
        path.append({"side": jname(j), "shots": d, "cost_h": spent, "surrogate_width": W, "gain_per_h": gain})
        cand[j] = make_cand(j)
    for b in B[bi:]:
        snaps[b] = {"added": {jname(x): v for x, v in added.items() if v}, "cost_h": spent, "surrogate_width": W}
    out["greedy"] = {"quantum_core_s": QUANTUM_S, "steps": len(path), "surrogate_width_start": W_start,
                     "path_every_0_25h": [p for i, p in enumerate(path)
                                          if i == len(path) - 1 or int(p["cost_h"] / 0.25) != int(path[i + 1]["cost_h"] / 0.25)],
                     "snapshots": {f"{b:g}": v for b, v in snaps.items()}}
    print(json.dumps({"greedy_steps": len(path), "snapshots": {f"{b:g}": (round(v["cost_h"], 3), round(v["surrogate_width"], 5))
                                                               for b, v in snaps.items()}, "t": round(time.time() - t0, 1)}), flush=True)

    # ------------------------------------------------------------ full evaluation of the designs
    designs = [(f"budget {b:g} core-hours", "budget", {jkey(k): v for k, v in snaps[b]["added"].items()}) for b in B]
    designs.append(("(c) +250 (D1-8's reference)", "reference", {jkey(k): int(v) for k, v in scen[REF_SYN]["added"].items()}))
    W_cur = (lambda q: q[2] - q[0])(scen[CURRENT]["rules"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"])
    res_d = []
    for name, kind, add in designs:
        for j, v in add.items():
            assert 0 < v <= cap[j], f"{name}: {j} gets {v} of {cap[j]} unsampled"
        d = {"name": name, "kind": kind, "added": {jname(j): v for j, v in sorted(add.items())},
             "added_total": int(sum(add.values())), "cost_core_hours": cost_h(add),
             "cost_core_hours_cell_means": cost_h(add, "cell")}
        d["wall_hours_on_3_workers"] = d["cost_core_hours"]["total"] / WORKERS
        nm_ = {j: n0[j] + add.get(j, 0) for j in sides}
        d["surrogate_q025_q50_q975"] = surrogate_q(nm_, k0)
        # if the shots were drawn from all 24 leaves instead, for the same expected rule-A shots
        scale = {j: mN.Npc[j[0]][j[1]] / mA.Npc[j[0]][j[1]] for j in add}
        d["if_drawn_from_all_24_leaves"] = {"shots": float(sum(v * scale[j] for j, v in add.items())),
                                            "cost_core_hours": cost_h({j: v * scale[j] for j, v in add.items()})}
        for base in ("synthetic", "real"):
            r = evaluate(add, base)
            d[base] = {rule: {lv: r[rule][lv] for lv in ("cells", "pooled")} for rule in ("A", "none")}
            q = r["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"]
            d[base]["A_cells_width"] = q[2] - q[0]
            d[base]["A_cells_width_x_headline"] = (q[2] - q[0]) / HEADLINE_W
        adv = [j for j in sorted(add) if sens[j]["adverse"]]
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
        surv = bool(one) and d["one_disagreement"]["worst"]["lower_synthetic"] > 0
        d["survives_one_disagreement"] = surv
        d["width_at_most_twice_headline"] = bool(w_ <= 2 * HEADLINE_W)
        res_d.append(d)
        print(json.dumps({"design": name, "added": d["added_total"], "cost_h": round(d["cost_core_hours"]["total"], 3),
                          "A_cells_syn": [round(x, 5) for x in d["synthetic"]["A"]["cells"]["g5_with_shot_noise_q025_q50_q975"]],
                          "width_x": round(d["synthetic"]["A_cells_width_x_headline"], 3), "survives": surv,
                          "t": round(time.time() - t0, 1)}), flush=True)
    out["designs"] = res_d
    out["current_synthetic_width"] = W_cur
    # ------------------------------------------------------------ recommendation (main's rule)
    rec, why = None, None
    for d in res_d:
        if d["kind"] == "budget" and d["survives_one_disagreement"] and d["width_at_most_twice_headline"]:
            rec, why = d["name"], "smallest budget that survives one disagreement with a width at most twice the headline's"
            break
    if rec is None:
        ok = [d for d in res_d if d["survives_one_disagreement"]]
        if ok:
            b = max(ok, key=lambda d: d["narrowing_per_core_hour"])
            rec, why = b["name"], "no budget qualifies; best narrowing per core-hour among the designs that survive"
    out["recommendation"] = {"design": rec, "why": why}
    out["seconds"] = time.time() - t0
    os.makedirs(a_.out, exist_ok=True)
    json.dump(out, open(os.path.join(a_.out, "d1_design.json"), "w"), indent=1, default=float)
    write_md(out, os.path.join(a_.out, "d1_design.md"))
    print(json.dumps({"recommendation": out["recommendation"], "seconds": out["seconds"]}))


def write_md(o, path):
    pc = lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"
    q3 = lambda q: ", ".join(pc(x) for x in q)
    L = ["# D1-8c: the allocation that narrows D1's primary interval most per core-hour", "",
         f"Made {o['made']} by `rev_d1_design.py` (md5 {o['script_md5']}). Post hoc, descriptive projection with D1-5b's "
         "synthetic machinery; no decoding, no label or test.", "",
         f"Validation: {json.dumps(o['validation'])}", "",
         f"Recommendation: {json.dumps(o['recommendation'])}", "",
         "| design | added shots | MPS h | sparse h | total h | wall h, 3 workers | rule A per cell, synthetic: 2.5/50/97.5% | "
         "width (x headline) | rule A per cell, real rows | rule none per cell, synthetic | rule none per cell, real | "
         "rule A pooled, synthetic | rule A pooled, real | worst single disagreement: side, lower end synthetic / real, P(>= 1) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in o["designs"]:
        c = d["cost_core_hours"]; s, r = d["synthetic"], d["real"]
        wv = d["one_disagreement"].get("worst")
        wtxt = (f"{wv['side']}, {pc(wv['lower_synthetic'])} / {pc(wv['lower_real'])}, {wv['P_at_least_one']:.3f}" if wv else "n/a")
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
    L += ["", "## Check of the surrogate against the full machinery (rule A, per cell)", "",
          "| scenario | base | surrogate 2.5/50/97.5% | full | difference (points) |", "|---|---|---|---|---|"]
    for c in o["surrogate_check"]:
        L.append(f"| {c['name']} | {c['base']} | {q3(c['surrogate_q025_q50_q975'])} | {q3(c['full_q025_q50_q975'])} | "
                 f"{', '.join(f'{100 * x:+.2f}' for x in c['diff'])} |")
    L += ["", f"Greedy: {o['greedy']['steps']} steps of about {o['greedy']['quantum_core_s']:.0f} core-s; snapshots "
          f"{json.dumps(o['greedy']['snapshots'])}", ""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
