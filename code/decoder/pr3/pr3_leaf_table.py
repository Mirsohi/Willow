#!/usr/bin/env python3
"""PR3's leaf table after the kill (decoder, 2026-10-12; notes/decoder/willow_prereg_r3.md section 8).

Why this script. PR3's runner exited 0 at 2026-10-12 15:46:02Z: d = 7 ML was killed (2 or more leaves still fail
after the extra rung), and its pr3_tally.json is not complete, because the kill starts no new part. Under the kill,
H7 and the d = 7 rows are not evaluated, and only the leaf table is filled in (section 8). The logged analysis,
pr3_analyze.py (md5 1b343f366fc1200876bea7c070c0e8ee), refuses a tally that is not complete and writes nothing; it is
not changed. This separate script writes the leaf table only. It evaluates no H7 and no d = 7 row, and it compares
nothing with Libra.

Per leaf: the tally's verdict; the ladder and extra parts that ran; the rules' values between tau1 and tau2 and
between tau2 and tau3 (rule 1: the error counts k and |k_new - k_old|; rule 2': the band shots whose decision changed;
both against max(3, 1% of k_new)); the largest |dP| over the band shots (descriptive); NaN posteriors; the band shots
whose tau3 frontier reached kmax; CPU from the runner's ledger (cpu_s + lost_cpu_s, the latter being the CPU of a run
lost in a restart). All on the PR3 sample, with the formulas of pr3_runner.py's judge(). For a leaf whose parts are
all done, the values must equal the tally's (checked). A leaf with only some extra parts done gets its tau2 -> tau3
values on those parts only, labelled partial: descriptive, not a verdict.
  python3 -I -B pr3_leaf_table.py --res results/decoder/willow/pr3/main \
      --ledger results/decoder/willow/live_logs/pr3/ledger.json --out results/decoder/willow/pr3/leaf_table
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

GUARD_CORE_H = 116.8


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sigmoid(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-x.astype(np.float64)))


def decide(llr):
    return (np.nan_to_num(sigmoid(llr), nan=0.0) > 0.5).astype(np.uint8)


def nanmax_or(x, empty):
    return float(np.nanmax(x)) if x.size and not np.all(np.isnan(x)) else empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="the main run's results folder (results/decoder/willow/pr3/main)")
    ap.add_argument("--ledger", required=True, help="the runner's final ledger.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tally", default=None, help="default <res>/pr3_tally.json")
    ap.add_argument("--code", default=HERE, help="folder with pr3_common.py (default: this script's folder)")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import pr3_common as C
    tally_p = a.tally or os.path.join(a.res, "pr3_tally.json")
    T = json.load(open(tally_p))
    LG = json.load(open(a.ledger))
    design = T["design"]
    tag = f"rl_{design}"
    LADDER_DN = C.LADDER.replace(":", "_").replace("/", "-")
    units = LG["units"]

    def ldir(lf, j):
        return os.path.join(a.res, "parts", f"p{j}", "decode", lf, tag)

    def edir(lf, j):
        return os.path.join(a.res, "parts", f"p{j}", "extra_rung", lf, tag)

    def ladder_done(lf, j):                     # as pr3_runner.Runner.ladder_done
        try:
            s = json.load(open(os.path.join(ldir(lf, j), "summary.json")))
        except Exception:
            return False
        return (s.get("status") == "complete" and C.LADDER in (s.get("decoders") or {})
                and s.get("selection") == C.part_slice(design, j))

    def extra_done(lf, j):                      # as pr3_runner.Runner.extra_done
        try:
            e = json.load(open(os.path.join(edir(lf, j), "extra_rung.json")))
        except Exception:
            return False
        return "tau3" in e and e.get("ladder_npz_md5") == md5(os.path.join(ldir(lf, j), LADDER_DN + ".npz"))

    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "what": "PR3 leaf table after the kill; not pr3_analyze.py's output (it refuses an incomplete tally)",
           "design": design, "tag": tag,
           "tally": {"made": T.get("made"), "status": T.get("status"), "kill": T.get("kill"),
                     "complete": T.get("complete"), "still_failing": T.get("still_failing")},
           "inputs_md5": {"pr3_tally.json": md5(tally_p), "ledger.json": md5(a.ledger)},
           "code_md5": {"pr3_leaf_table.py": md5(os.path.abspath(__file__)),
                        "pr3_common.py": md5(os.path.join(a.code, "pr3_common.py"))},
           "checks": [], "leaves": {}}
    total_cpu = total_lost = 0.0
    for k, x in units.items():
        total_cpu += (x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0)
        total_lost += x.get("lost_cpu_s") or 0.0
    for b in C.BASES:
        for r in C.ROUNDS:
            lf = C.leaf(b, r)
            tv = (T.get("verdicts") or {}).get(lf, {})
            row = {"verdict": tv.get("verdict"), "parts": {}}
            cpu = {"ladder": 0.0, "extra": 0.0, "lost_in_restart": 0.0}
            for j in range(C.PARTS):
                pj = {}
                for kind in ("ladder", "extra"):
                    x = units.get(f"{kind}|{lf}|{j}")
                    done = ladder_done(lf, j) if kind == "ladder" else extra_done(lf, j)
                    ent = {"done": done, "ledger_state": (x or {}).get("state", "never started")}
                    if x:
                        c = (x.get("cpu_s") or 0.0) + (x.get("lost_cpu_s") or 0.0)
                        ent.update(cpu_core_h=c / 3600, lost_cpu_core_h=(x.get("lost_cpu_s") or 0.0) / 3600,
                                   started=x.get("started"), ended=x.get("ended"),
                                   peak_rss_gib=(x.get("peak_rss") or 0) / 2 ** 30)
                        cpu[kind] += c; cpu["lost_in_restart"] += x.get("lost_cpu_s") or 0.0
                    if kind == "extra" and done:
                        e = json.load(open(os.path.join(edir(lf, j), "extra_rung.json")))
                        t3, ck = e.get("tau3", {}), e.get("check", {})
                        ent.update(n_band=t3.get("n_band"), decisions_changed_tau2_tau3=t3.get("decisions_changed_tau2_tau3"),
                                   max_K_tau3=t3.get("max_K_tau3"), kmax=e.get("kmax"),
                                   ms_per_round_per_band_shot_tau3=t3.get("ms_per_round_per_band_shot_tau3"),
                                   rebuild_check={"n_checked": ck.get("n_checked"), "identical": ck.get("identical"),
                                                  "max_abs_llr_diff": ck.get("max_abs_llr_diff")})
                    pj[kind] = ent
                row["parts"][f"p{j}"] = pj
            row["ladder_parts_done"] = [j for j in range(C.PARTS) if row["parts"][f"p{j}"]["ladder"]["done"]]
            row["extra_parts_done"] = [j for j in range(C.PARTS) if row["parts"][f"p{j}"]["extra"]["done"]]
            row["cpu_core_h"] = {k_: v / 3600 for k_, v in cpu.items()}
            row["cpu_core_h"]["total"] = (cpu["ladder"] + cpu["extra"]) / 3600
            Ld = row["ladder_parts_done"]
            if Ld:
                idx, obs, df, llr, llr1, red, part_of = [], [], [], [], [], [], []
                for j in Ld:
                    z = np.load(os.path.join(ldir(lf, j), LADDER_DN + ".npz"))
                    sh = np.load(os.path.join(ldir(lf, j), "shots.npz"))
                    assert np.array_equal(z["idx"], sh["idx"]), f"{lf} p{j}: ladder and shots.npz select different shots"
                    idx.append(z["idx"].astype(np.int64)); obs.append(sh["obs"].astype(np.uint8))
                    df.append(z["pred"][:, 0]); llr.append(z["llr"]); llr1.append(z["llr_t1"])
                    red.append(z["redecoded"].astype(bool)); part_of.append(np.full(len(z["idx"]), j))
                cat = np.concatenate
                idx, obs, df, llr, llr1, red, part_of = (cat(idx), cat(obs), cat(df).astype(np.uint8), cat(llr),
                                                         cat(llr1), cat(red), cat(part_of))
                m = C.in_sample(idx, r)
                d1 = decide(llr1)
                k1, kf = int(np.sum((d1 != obs) & m)), int(np.sum((df != obs) & m))
                ch = int(np.sum((d1 != df) & m))
                dP = np.abs(sigmoid(llr) - sigmoid(llr1))[red & m]
                lim = max(3, 0.01 * kf)
                lad = {"parts": Ld, "complete": len(Ld) == C.PARTS, "n_decoded": int(len(idx)), "n_sample": int(m.sum()),
                       "band_shots_sample": int((red & m).sum()), "errors_tau1": k1, "errors_tau2": kf,
                       "abs_change_errors": abs(kf - k1), "rule1": abs(kf - k1) <= lim, "decisions_changed": ch,
                       "rule2prime": ch <= lim, "limit": lim, "max_abs_dP_descriptive": nanmax_or(dP, 0.0),
                       "nan_tau2": int(np.isnan(llr[m]).sum())}
                lad["passes"] = lad["rule1"] and lad["rule2prime"]
                row["tau1_tau2"] = lad
                if lad["complete"] and tv.get("ladder"):
                    tl = tv["ladder"]
                    same = all(tl.get(k_) == lad[k_] for k_ in ("errors_tau1", "errors_tau2", "rule1", "decisions_changed",
                                                                "rule2prime", "nan_tau2", "passes")) \
                        and abs(tl["limit"] - lad["limit"]) < 1e-9 \
                        and abs(tl["max_abs_dP_descriptive"] - lad["max_abs_dP_descriptive"]) < 1e-12
                    out["checks"].append({"leaf": lf, "what": "tau1 -> tau2 values equal the tally's", "ok": bool(same)})
                    assert same, f"{lf}: tau1 -> tau2 values differ from the tally's: {lad} vs {tl}"
                Ed = row["extra_parts_done"]
                if Ed:
                    assert len(Ld) == C.PARTS, f"{lf}: extra parts done without all ladder parts"
                    pos = {int(x): i for i, x in enumerate(idx)}
                    bi, l3, mk, kmax = [], [], [], None
                    for j in Ed:
                        e = np.load(os.path.join(edir(lf, j), "extra_rung.npz"))
                        bj = np.array([pos[int(x)] for x in e["idx"]], np.int64)
                        want = np.where(red & (part_of == j))[0]
                        assert np.array_equal(np.sort(bj), want), f"{lf} p{j}: extra-rung shots differ from the part's band"
                        bi.append(bj); l3.append(e["llr_t3"]); mk.append(e["maxk_t3"])
                        kmax = json.load(open(os.path.join(edir(lf, j), "extra_rung.json"))).get("kmax")
                    bi, l3, mk = cat(bi), cat(l3), cat(mk)
                    keep = m[bi]
                    d3 = decide(l3)
                    bk = bi[keep]
                    k3 = kf - int(np.sum(df[bk] != obs[bk])) + int(np.sum(d3[keep] != obs[bk]))
                    ch3 = int(np.sum(d3[keep] != df[bk])); nan3 = int(np.isnan(l3[keep]).sum())
                    lim3 = max(3, 0.01 * k3)
                    dP3 = np.abs(sigmoid(l3) - sigmoid(llr[bi]))[keep]
                    full = len(Ed) == C.PARTS
                    ex = {"parts": Ed, "complete": full, "band_shots_sample_covered": int(keep.sum()),
                          "band_shots_sample_leaf": lad["band_shots_sample"],
                          "errors_tau2": kf, "errors_tau3" if full else "errors_tau3_on_these_parts_tau2_elsewhere": k3,
                          "abs_change_errors": abs(k3 - kf), "decisions_changed": ch3, "limit": lim3,
                          "nan_tau3": nan3,
                          "max_abs_dP_descriptive": nanmax_or(dP3, None) if nan3 < dP3.size else None,
                          "kmax": kmax, "band_shots_at_kmax_tau3": int(np.sum(mk[keep] >= kmax)) if kmax else None}
                    if full:
                        ex.update(rule1=abs(k3 - kf) <= lim3, rule2prime=ch3 <= lim3 and nan3 == 0)
                        ex["passes"] = ex["rule1"] and ex["rule2prime"]
                        te = tv.get("extra") or {}
                        same = all(te.get(k_) == v for k_, v in (("errors_tau2", kf), ("errors_tau3", k3),
                                                                  ("decisions_changed", ch3), ("nan_tau3", nan3),
                                                                  ("rule1", ex["rule1"]), ("rule2prime", ex["rule2prime"]),
                                                                  ("passes", ex["passes"]))) \
                            and abs(te.get("limit", -1) - lim3) < 1e-9 \
                            and abs((te.get("max_abs_dP_descriptive") or 0) - (ex["max_abs_dP_descriptive"] or 0)) < 1e-12
                        out["checks"].append({"leaf": lf, "what": "tau2 -> tau3 values equal the tally's", "ok": bool(same)})
                        assert same, f"{lf}: tau2 -> tau3 values differ from the tally's: {ex} vs {te}"
                    else:
                        rest = lad["band_shots_sample"] - int(keep.sum())     # band shots of the parts without tau3
                        ub = max(3, 0.01 * (k3 + rest))                       # the largest limit the leaf could reach
                        ex.update(band_shots_sample_not_covered=rest, limit_upper_bound=ub,
                                  rule2prime_fails_whatever_the_other_parts=ch3 > ub)
                        ex["note"] = ("partial: tau3 ran on these parts only; the counts are descriptive and give no verdict; "
                                      "the limit is shown for reference, from the error count with tau3 on these parts; "
                                      "limit_upper_bound assumes every band shot of the other parts adds an error")
                    row["tau2_tau3"] = ex
            out["leaves"][lf] = row
    out["totals"] = {"cpu_core_h": total_cpu / 3600, "of_which_lost_cpu_core_h": total_lost / 3600,
                     "guard_core_h": GUARD_CORE_H, "share_of_guard": total_cpu / 3600 / GUARD_CORE_H,
                     "runner_projection_at_exit": LG.get("projection"),
                     "pilot_core_h_not_counted": 0.24}
    out["no_leaf_complete_and_passing"] = not any(str(v.get("verdict", "")).startswith("pass")
                                                  for v in (T.get("verdicts") or {}).values())
    rc = [p["extra"]["rebuild_check"] for x in out["leaves"].values() for p in x["parts"].values()
          if p["extra"].get("done")]
    out["extra_parts_rebuild_checks"] = {"n_parts": len(rc), "all_identical": all(c.get("identical") for c in rc),
                                         "max_abs_llr_diff": max((c.get("max_abs_llr_diff") or 0.0) for c in rc) if rc else None}
    os.makedirs(a.out, exist_ok=True)
    jp, mp = os.path.join(a.out, "pr3_leaf_table.json"), os.path.join(a.out, "pr3_leaf_table.md")
    with open(jp + ".tmp", "w") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(jp + ".tmp", jp)
    with open(mp + ".tmp", "w") as f:
        f.write(write_md(out))
    os.replace(mp + ".tmp", mp)
    print(json.dumps({"checks": out["checks"], "totals": out["totals"]}, indent=1, default=float))


def write_md(o):
    f2 = lambda x: "–" if x is None else f"{x:.2f}"
    s = ("# PR3 leaf table after the kill (generated by pr3_leaf_table.py)\n\n"
         f"Made {o['made']}. Tally of {o['tally']['made']}: {o['tally']['status']}; kill {o['tally']['kill']}, complete "
         f"{o['tally']['complete']}; still failing: {', '.join(x.split('/', 2)[-1] for x in o['tally']['still_failing'] or [])}.\n\n"
         "Under the kill, H7 and the d = 7 rows are not evaluated, and only this table is filled in "
         "(`willow_prereg_r3.md` §8). `pr3_analyze.py` refuses an incomplete tally and wrote nothing; this table comes "
         "from a separate script. Rules per leaf on the PR3 sample: rule 1, |k_new − k_old| ≤ max(3, 1% of k_new); "
         "rule 2′, decisions changed ≤ the same limit. |ΔP| is descriptive. A ladder part runs τ1 on all its shots and "
         "τ2 on its band shots (with corr. MWPM and MWPM); an extra part runs τ3 on its band shots after a τ2 rebuild "
         "check on 10 of them.\n\n")
    s += ("## Verdicts, parts and CPU\n\n| leaf | verdict | ladder parts done | extra parts done | band shots | CPU, "
          "core-hours (ladder + extra) |\n|---|---|---|---|---|---|\n")
    for lf, x in o["leaves"].items():
        lad = x.get("tau1_tau2") or {}
        s += (f"| {lf.split('/', 2)[-1]} | {x['verdict']} | {len(x['ladder_parts_done'])} of 5 "
              f"{x['ladder_parts_done'] or ''} | {len(x['extra_parts_done'])} of 5 {x['extra_parts_done'] or ''} | "
              f"{lad.get('band_shots_sample', '–')} | {x['cpu_core_h']['total']:.2f} ({x['cpu_core_h']['ladder']:.2f} + "
              f"{x['cpu_core_h']['extra']:.2f}) |\n")
    t = o["totals"]
    s += (f"\nPR3's CPU: {t['cpu_core_h']:.2f} core-hours against the {t['guard_core_h']} guard "
          f"({100 * t['share_of_guard']:.1f}%). It includes {t['of_which_lost_cpu_core_h']:.2f} core-hours that the runner "
          "booked as `lost_cpu_s`: the earlier runs of the 4 parts restarted after the container restart of 2026-10-12 "
          "01:38Z, most of whose work was kept in chunks. The pilot's "
          f"{t['pilot_core_h_not_counted']} core-hours are not counted against the guard.\n\n")
    s += ("## Rules between rungs\n\n| leaf | rung | k (old → new) | rule 1 | decisions changed / limit | rule 2′ | "
          "max abs ΔP | NaN | band shots at kmax (τ3) |\n|---|---|---|---|---|---|---|---|---|\n")
    for lf, x in o["leaves"].items():
        n = lf.split('/', 2)[-1]
        lad, ex = x.get("tau1_tau2"), x.get("tau2_tau3")
        if not lad:
            s += f"| {n} | – | no part ran | | | | | | |\n"
            continue
        s += (f"| {n} | τ1 → τ2 | {lad['errors_tau1']} → {lad['errors_tau2']} | {'pass' if lad['rule1'] else 'fail'} | "
              f"{lad['decisions_changed']} / {lad['limit']:.2f} | {'pass' if lad['rule2prime'] else 'fail'} | "
              f"{lad['max_abs_dP_descriptive']:.3f} | {lad['nan_tau2']} | |\n")
        if ex:
            k3 = ex.get("errors_tau3", ex.get("errors_tau3_on_these_parts_tau2_elsewhere"))
            if ex["complete"]:
                s += (f"| {n} | τ2 → τ3 | {ex['errors_tau2']} → {k3} | {'pass' if ex['rule1'] else 'fail'} | "
                      f"{ex['decisions_changed']} / {ex['limit']:.2f} | {'pass' if ex['rule2prime'] else 'fail'} | "
                      f"{f2(ex['max_abs_dP_descriptive']) if ex['max_abs_dP_descriptive'] is None else format(ex['max_abs_dP_descriptive'], '.3f')} | "
                      f"{ex['nan_tau3']} | {ex['band_shots_at_kmax_tau3']} of {ex['band_shots_sample_covered']} |\n")
            else:
                s += (f"| {n} | τ2 → τ3, partial (parts {ex['parts']}; {ex['band_shots_sample_covered']} of "
                      f"{ex['band_shots_sample_leaf']} band shots) | {ex['errors_tau2']} → {k3} (τ3 on these parts) | no verdict | "
                      f"{ex['decisions_changed']} / ({ex['limit']:.2f}, for reference) | no verdict | "
                      f"{format(ex['max_abs_dP_descriptive'], '.3f') if ex['max_abs_dP_descriptive'] is not None else '–'} | "
                      f"{ex['nan_tau3']} | {ex['band_shots_at_kmax_tau3']} of {ex['band_shots_sample_covered']} |\n")
    for lf, x in o["leaves"].items():
        ex = x.get("tau2_tau3") or {}
        if ex and not ex["complete"]:
            s += (f"\n{lf.split('/', 2)[-1]}, partial τ2 → τ3: {ex['decisions_changed']} decisions changed on "
                  f"{ex['band_shots_sample_covered']} of {ex['band_shots_sample_leaf']} band shots. The largest limit the "
                  f"leaf could reach is {ex['limit_upper_bound']:.2f} (every other band shot an error), so rule 2′ "
                  + ("fails whatever the other parts give." if ex["rule2prime_fails_whatever_the_other_parts"]
                     else "is not decided by these parts.") + "\n")
    rc = o["extra_parts_rebuild_checks"]
    s += (f"\nτ2 rebuild checks of the {rc['n_parts']} extra parts: all identical {rc['all_identical']} "
          f"(largest |Δllr| {rc['max_abs_llr_diff']}).\n")
    s += ("\nChecks: " + "; ".join(f"{c['leaf'].split('/', 2)[-1]} {c['what']}: {c['ok']}" for c in o["checks"]) + ".\n")
    if o["no_leaf_complete_and_passing"]:
        s += "\nNo leaf has a complete, passing set of decisions.\n"
    return s


if __name__ == "__main__":
    main()
