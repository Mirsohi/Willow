#!/usr/bin/env python3
"""Post hoc robustness check, prompted by the internal review: are ML(RL)'s decisions outside the ladder's band stable
at d = 5? (notes/decoder/willow_prereg.md section 9 entry 13). Step 3 of 3: the analysis, fixed before the decoding.

Inputs: the shot lists (nonband_select.py), the check's outputs (nonband_check.py), round 1's ladder outputs (the 2x2
tables of Libra against ML(RL) per leaf, as count2 builds them), and round 1's final analysis (final_round1.json: rule
A's excluded leaves and g_5, which the base must reproduce).

Per selected shot: d1 = ML(RL)'s round-1 decision (tau1's outside the band), d2 at tau2, d3 at tau3 where it ran; the
final decision is d3 where tau3 ran, else d2. A flip is a final decision that differs from d1; its direction is "ML right
-> wrong" if d1 was right, else "wrong -> right". Types: D10 (Libra wrong, ML right), D01 (Libra right, ML wrong), C00
(both right), C11 (both wrong). A flip moves a D10 shot to n11, a D01 shot to n00, a C00 shot to n01 and a C11 shot to n10
of count2(Libra, ML).

Reported: flips tau1 -> tau2 and tau1 -> tau3 by stratum and type with their direction (and by |llr_tau1| bin); the
implied change in g_5 under rule A (entry 7's estimator, rounds {10, 30, 50} as in final_round1.py) and in the pooled
count gap, with no exclusion as the sensitivity row:
  estimate: D's flips as observed; C's at the pooled sample rate of each type (C00, C11) times each leaf's count; for
    shots that tau2 did not move and tau3 did not visit, the tau3-only change rate of the random 10% subsample
    (pooled within D and within C);
  95% worst case (one-sided, each direction), at the ladder's own level: a shot that tau2 did not move keeps its
    decision; D's flips as observed; C's at the Clopper-Pearson bound of each type's sample rate; all flips in the
    direction that lowers g (ML right -> wrong at the upper bounds, the other direction at the lower bounds), and the
    mirror case that raises g; a targeted shot whose tau3 did not run counts in the adverse direction;
  the same worst case with the tau3-only term at its bound (the random 10% subsample's Clopper-Pearson bound applied
    to every shot that tau2 did not move and tau3 did not visit): reported as a further, looser row.
  python3 -I -B nonband_analyze.py --res results/decoder/willow --lists <nonband_check> --check <nonband_check> \
      --final-r1 results/decoder/willow/final_round1/final_round1.json --out <nonband_check>/analysis
"""
import argparse, hashlib, json, math, os, sys, time

sys.dont_write_bytecode = True
CODE = os.environ.get("NONBAND_CODE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/decoder
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "supplementary"))
import numpy as np
from scipy.stats import beta
import analyze_willow as AW                  # frozen; read-only
import p1_se_covariance as PC                # entry 7; read-only

LABEL = "post hoc robustness check, prompted by the internal review"
DEC = "ml_tau2:1e-6/1e-8:0.1"
DN = DEC.replace(":", "_").replace("/", "-")
LIBRA = "libra_decoder_with_rl_optimized_prior"
W = (10, 30, 50)
TYPES = ("D10", "D01", "C00", "C11")
ADVERSE = {"D10": True, "D01": False, "C00": True, "C11": False}       # a flip of this type raises ML's errors
BINS = (0.405, 1.0, 2.0, 3.0, 5.0, float("inf"))


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cp(k, n, side):
    """One-sided 95% Clopper-Pearson bound of a proportion."""
    if n == 0:
        return 1.0 if side == "upper" else 0.0
    if side == "upper":
        return 1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k))
    return 0.0 if k <= 0 else float(beta.ppf(0.05, k, n - k + 1))


def base_tables(res, leaves):
    T = {}
    for rel in leaves:
        d = os.path.join(res, "decode", rel, "rl_s5")
        z = np.load(os.path.join(d, DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
        rows = [i for i, x in enumerate(sh["shipped_names"]) if LIBRA in str(x)]
        assert len(rows) == 1
        T[rel] = AW.count2(sh["shipped"][rows[0]].astype(np.int8), z["pred"][:, 0].astype(np.int8), sh["obs"].astype(np.int8))
    return T


def g5(tables, meta, excl):
    """Entry 7's g over the 8 series, rounds {10, 30, 50} minus rule A's excluded rounds (excl: set of leaf paths)."""
    ser = {}
    for x in meta["leaves"]:
        if x["leaf"] in excl:
            continue
        ser.setdefault((x["patch"], x["basis"]), {})[x["rounds"]] = tables[x["leaf"]]
    S, used, left = [], [], []
    for key in sorted(ser):
        pts = ser[key]; w = tuple(r for r in W if r in pts)
        if len(w) < 2 or len(PC.points_arrays(pts, w)) < 2:
            left.append("|".join(key)); continue
        S.append(PC.series_params(pts, w)); used.append("|".join(key))
    if not S:
        return None
    cs = PC.g_and_se(S, "inflated", full=True, with_cov=True)
    return {"g": cs["g"], "se": cs["se"], "ci95": cs["ci95"], "n_series": len(S), "series": used, "left_out": left}


def count_gap(tables, excl):
    lib = sum(c["n10"] + c["n11"] for k, c in tables.items() if k not in excl)
    ml = sum(c["n01"] + c["n11"] for k, c in tables.items() if k not in excl)
    return {"errors_libra": lib, "errors_ML": ml, "gap_errors": lib - ml, "rel": lib / ml - 1}


def modified(T, F):
    """F[leaf][type] = flips; returns new tables (fractional counts allowed)."""
    out = {}
    for rel, c in T.items():
        f = F.get(rel, {})
        d10, d01, c00, c11 = (f.get(t, 0.0) for t in TYPES)
        out[rel] = {"n": c["n"], "n10": c["n10"] - d10 + c11, "n11": c["n11"] + d10 - c11, "n01": c["n01"] - d01 + c00}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--final-r1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="tests only")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.lists, "shot_lists.json")))
    lp = os.path.join(a.lists, "shot_lists.npz")
    assert md5_file(lp) == meta["shot_lists_npz_md5"]
    S = {k: v for k, v in np.load(lp).items()}
    summ = json.load(open(os.path.join(a.check, "summary.json")))
    if summ.get("status") != "complete" and not a.allow_incomplete:
        print(f"note: the check's status is {summ.get('status')!r}; the analysis uses what was decoded", flush=True)
    if summ.get("test_only") and not a.allow_incomplete:
        sys.exit("the check's outputs come from a test run")
    F1 = json.load(open(a.final_r1))
    if F1.get("test_only_incomplete_k2") and not a.allow_incomplete:
        sys.exit("final_round1.json comes from an incomplete K2 tally")
    leaves = [str(x) for x in S["leaves"]]
    n = len(S["idx"])
    # per row: decisions at tau2 and tau3
    d2 = np.full(n, 255, np.uint8); d3 = np.full(n, 255, np.uint8); reason = np.zeros(n, np.int8)
    l2 = np.full(n, np.nan); l3 = np.full(n, np.nan)
    for k, rel in enumerate(leaves):
        p = os.path.join(a.check, "leaves", f"{k:02d}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p); r = z["row"]
        d2[r] = np.where(z["done_t2"], z["pred_t2"], 255); d3[r] = np.where(z["done_t3"], z["pred_t3"], 255)
        reason[r] = z["t3_reason"]; l2[r] = z["llr_t2"]; l3[r] = z["llr_t3"]
    ml, obs, typ, strat = S["ml"], S["obs"], S["type"], S["stratum"]
    done2, done3 = d2 != 255, d3 != 255
    with np.errstate(over="ignore", invalid="ignore"):
        P1 = 1 / (1 + np.exp(-S["llr_t1"].astype(np.float64)))
        P2 = 1 / (1 + np.exp(-l2))
        tg = done2 & ((d2 != ml) | ~(np.abs(P2 - 0.5) >= 0.1) | ~(np.abs(P2 - P1) < 0.05))
    final = np.where(done3, d3, d2)
    right1 = ml == obs

    def flips(mask, dec):
        m = mask & (dec != 255)
        f = m & (dec != ml)
        return {"n": int(m.sum()), "flips": int(f.sum()), "right_to_wrong": int((f & right1).sum()),
                "wrong_to_right": int((f & ~right1).sum())}
    tab = {}
    for name, sm in (("D", strat == 0), ("C", strat == 1)):
        tab[name] = {"tau1_to_tau2": flips(sm, d2), "tau1_to_tau3": flips(sm, d3),
                     "tau1_to_tau3_targeted": flips(sm & (reason == 1), d3), "tau1_to_tau3_random": flips(sm & (reason == 2), d3),
                     "tau1_to_final": flips(sm, final),
                     "tau2_to_tau3_changes": int((sm & done3 & done2 & (d3 != d2)).sum()),
                     "targeted": int((sm & tg).sum()), "targeted_without_tau3": int((sm & tg & ~done3).sum()),
                     "not_done_tau2": int((sm & ~done2).sum())}
        for t in TYPES:
            if t[0] == name:
                tm = sm & (typ == t)
                tab[name][t] = {"tau1_to_tau2": flips(tm, d2), "tau1_to_tau3": flips(tm, d3), "tau1_to_final": flips(tm, final)}
        absl = np.abs(S["llr_t1"].astype(np.float64))
        tab[name]["by_abs_llr_tau1"] = {f"[{lo:g}, {hi:g})": flips(sm & (absl >= lo) & ((absl < hi) | (hi == float("inf"))), final)
                                        for lo, hi in zip(BINS[:-1], BINS[1:])}
    # tau3-only change rates among the random subsample of shots that tau2 did not move
    rate3 = {}
    for name, sm in (("D", strat == 0), ("C", strat == 1)):
        m = sm & (reason == 2) & done3 & done2
        k3 = int((m & (d3 != d2)).sum()); n3 = int(m.sum())
        unrun = sm & done2 & ~tg & ~done3
        rate3[name] = {"k": k3, "n": n3, "rate": k3 / n3 if n3 else 0.0, "upper95": cp(k3, n3, "upper"),
                       "unrun": int(unrun.sum())}
    # per leaf and type: estimate and worst cases
    pops = {x["leaf"]: x["population"] for x in meta["leaves"]}
    Cm = strat == 1
    ctype = {}
    for t in ("C00", "C11"):
        m = Cm & (typ == t) & done2
        k = int((m & (final != ml)).sum()); nn = int(m.sum())
        ctype[t] = {"k": k, "n": nn, "rate": k / nn if nn else 0.0, "upper95": cp(k, nn, "upper"),
                    "lower95": cp(k, nn, "lower"), "unrun_share": float((m & ~tg & ~done3).sum() / nn) if nn else 0.0}
    Dk = {}
    for t in ("D10", "D01"):                                      # D's observed flip rate per type (for shots without tau2)
        m = (strat == 0) & (typ == t) & done2 & ~(tg & ~done3)
        Dk[t] = float((m & (final != ml)).sum() / m.sum()) if m.sum() else 0.0
    Fe, Fadv, Ffav, Fadv3, Ffav3 = {}, {}, {}, {}, {}
    for kk, rel in enumerate(leaves):
        lm = S["leaf_id"] == kk
        e, adv, fav, adv3, fav3 = {}, {}, {}, {}, {}
        for t in ("D10", "D01"):
            m = lm & (typ == t)
            known = m & done2 & ~(tg & ~done3)                      # final decision known (tau3 where needed)
            obs_f = int((known & (final != ml)).sum())
            pending = int((m & tg & ~done3).sum())                  # targeted, tau3 not run
            unrun = int((m & done2 & ~tg & ~done3).sum())
            notdone = int((m & ~done2).sum())
            e[t] = obs_f + int((m & tg & ~done3 & (d2 != ml)).sum()) + unrun * rate3["D"]["rate"] + notdone * Dk[t]
            hi = obs_f + pending + notdone
            hi3 = hi + unrun * rate3["D"]["upper95"]
            lo = obs_f
            adv[t], fav[t] = (hi, lo) if ADVERSE[t] else (lo, hi)
            adv3[t], fav3[t] = (hi3, lo) if ADVERSE[t] else (lo, hi3)
        for t in ("C00", "C11"):
            N = pops[rel][t]; c = ctype[t]
            e[t] = N * (c["rate"] + c["unrun_share"] * rate3["C"]["rate"])
            hi = N * c["upper95"]
            hi3 = N * min(1.0, c["upper95"] + c["unrun_share"] * rate3["C"]["upper95"])
            lo = N * c["lower95"]
            adv[t], fav[t] = (hi, lo) if ADVERSE[t] else (lo, hi)
            adv3[t], fav3[t] = (hi3, lo) if ADVERSE[t] else (lo, hi3)
        Fe[rel], Fadv[rel], Ffav[rel], Fadv3[rel], Ffav3[rel] = e, adv, fav, adv3, fav3
    T = base_tables(a.res, leaves)
    excl_all = set(F1.get("excluded_leaves") or [])
    killed = bool((F1.get("k2") or {}).get("kill"))
    exA = {x for x in excl_all if x in T}
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": LABEL, "test_only": bool(a.allow_incomplete),
           "check_status": summ.get("status"), "check_cpu_core_hours": summ.get("cpu_core_hours"),
           "rebuild": summ.get("rebuild"), "selected": meta["selected"], "population": meta["population"],
           "flips": tab, "tau3_only_rates": rate3, "C_type_rates": ctype, "rule_A_excluded_in_window": sorted(exA),
           "inputs_md5": {"shot_lists.npz": meta["shot_lists_npz_md5"], "final_round1.json": md5_file(a.final_r1),
                          "check summary.json": md5_file(os.path.join(a.check, "summary.json"))},
           "code_md5": {"nonband_analyze.py": md5_file(os.path.abspath(__file__)),
                        "analyze_willow.py": md5_file(os.path.join(CODE, "analyze_willow.py")),
                        "p1_se_covariance.py": md5_file(os.path.join(CODE, "supplementary", "p1_se_covariance.py"))}}
    rows = {}
    for rule, ex in (("A", exA), ("none", set())):
        if rule == "A" and killed:
            rows[rule] = {"note": "K2 killed d = 5 ML: rule A has no g_5"}
            continue
        r = {}
        for case, F in (("base", {}), ("estimate", Fe), ("worst_lower_g", Fadv), ("worst_higher_g", Ffav),
                        ("worst_lower_g_with_tau3_only", Fadv3), ("worst_higher_g_with_tau3_only", Ffav3)):
            Tm = modified(T, F)
            r[case] = {"g5": g5(Tm, meta, ex), "count_gap": count_gap(Tm, ex),
                       "flips_total": {t: float(sum(F.get(rel, {}).get(t, 0.0) for rel in leaves if rel not in ex)) for t in TYPES}}
        b = r["base"]["g5"]
        for case in ("estimate", "worst_lower_g", "worst_higher_g", "worst_lower_g_with_tau3_only",
                     "worst_higher_g_with_tau3_only"):
            if b and r[case]["g5"]:
                r[case]["delta_g5"] = r[case]["g5"]["g"] - b["g"]
            r[case]["delta_count_gap_rel"] = r[case]["count_gap"]["rel"] - r["base"]["count_gap"]["rel"]
        rows[rule] = r
    # the base must reproduce round 1's final analysis
    chk = {}
    for rule in ("A", "none"):
        f = ((F1.get("decisions") or {}).get("d5_rules") or {}).get(rule, {}).get("g5")
        b = (rows.get(rule) or {}).get("base", {}).get("g5")
        if f and b:
            chk[rule] = {"g": float(abs(b["g"] - f["g"])), "se": float(abs(b["se"] - f["se"]))}
    out["base_vs_final_round1"] = chk
    out["rules"] = rows
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "nonband_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    write_md(out, os.path.join(a.out, "nonband_results.md"))
    print(json.dumps({"check": chk, "D": tab["D"]["tau1_to_final"], "C": tab["C"]["tau1_to_final"],
                      "A": {c: (rows.get("A", {}).get(c, {}).get("g5") or {}).get("g") for c in ("base", "estimate", "worst_lower_g", "worst_higher_g")}},
                     default=float))


def write_md(o, path):
    pc = lambda x: f"{100 * x:+.2f}%"
    s = f"# Non-band truncation check at d = 5 ({o['label']})\n\n"
    if o["test_only"]:
        s += "**TEST RUN. Not the result.**\n\n"
    s += (f"Made {o['made']} by nonband_analyze.py. Check status: {o['check_status']}; CPU {o['check_cpu_core_hours']:.2f} "
          f"core-hours. Selected: {o['selected']['D']} discordant (D, all) and {o['selected']['C']} concordant (C, random) "
          f"non-band shots of the 24 leaves at r = 10, 30, 50. Rule A leaves out {o['rule_A_excluded_in_window'] or 'none'}.\n\n")
    s += "## Flips (ML(RL)'s decision against its round-1 tau1 decision)\n\n| stratum | step | shots | flips | right -> wrong | wrong -> right |\n|---|---|---|---|---|---|\n"
    for st in ("D", "C"):
        for step in ("tau1_to_tau2", "tau1_to_tau3_targeted", "tau1_to_tau3_random", "tau1_to_tau3", "tau1_to_final"):
            x = o["flips"][st][step]
            s += f"| {st} | {step} | {x['n']} | {x['flips']} | {x['right_to_wrong']} | {x['wrong_to_right']} |\n"
    s += "\n| type | step | shots | flips |\n|---|---|---|---|\n"
    for st in ("D", "C"):
        for t in TYPES:
            if t in o["flips"][st]:
                for step in ("tau1_to_tau2", "tau1_to_tau3", "tau1_to_final"):
                    x = o["flips"][st][t][step]
                    s += f"| {t} | {step} | {x['n']} | {x['flips']} |\n"
    s += "\n## Implied change (estimate and 95% worst case)\n\n| rule | case | g_5 | change in g_5 | count gap | change |\n|---|---|---|---|---|---|\n"
    for rule, r in o["rules"].items():
        if "base" not in r:
            s += f"| {rule} | {r.get('note')} | | | | |\n"; continue
        for case in ("base", "estimate", "worst_lower_g", "worst_higher_g", "worst_lower_g_with_tau3_only",
                     "worst_higher_g_with_tau3_only"):
            g = r[case]["g5"]; cg = r[case]["count_gap"]
            s += (f"| {rule} | {case} | {pc(g['g']) if g else 'n/a'} | {pc(r[case]['delta_g5']) if 'delta_g5' in r[case] else ''} | "
                  f"{pc(cg['rel'])} ({cg['gap_errors']:+.0f} errors) | {pc(r[case]['delta_count_gap_rel']) if case != 'base' else ''} |\n")
    s += f"\nBase against final_round1.json (absolute differences): {o['base_vs_final_round1']}\n"
    open(path, "w").write(s)


if __name__ == "__main__":
    main()
