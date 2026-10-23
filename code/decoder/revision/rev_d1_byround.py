#!/usr/bin/env python3
"""D1-6: D1's counts by round count, descriptively (decoder, 2026-10-23). Post hoc revision analysis.

Reads only d1_results.json, the output of the frozen rev_d1_analyze.py (ef15b054...), and gives, for each round count
r = 10, 30 and 50 and each stratum (B, D, C_low, C_high):
  - MPS against the paper's decision m: the rows MPS finished (n: a finite rung, unresolved rows included), the
    disagreements q != m, split by whether the change adds or removes an ML error (m right and q wrong adds one), the
    rate with its exact 95% Clopper-Pearson interval, the unresolved rows among n, the failed rows (no finite rung,
    D1-3) and the selected rows not decoded;
  - MPS against the census decision a of D1-1 (B: K2's tau3; D: the non-band check's final decision; C_low, C_high: m):
    the same;
  - the sparse ladder against m: D1's tau2 decisions (run "t2", tau = 1e-8) and tau3 decisions (run "t3", tau = 1e-9),
    both at kmax 2,000,000, that differ from m, with the same split and interval. D1 runs tau3 only on D and C_low
    (D1-1's skips), so tau3 rows exist only there.
Per r it also gives each quantity's population-weighted rate over the strata, sum_s N(s,r) k(s,r)/n(s,r) / sum_s N(s,r),
with the design's cell sizes N: a point estimate only. A stratum without rows is left out, and the strata used are
listed. For tau3 they are D and C_low.
Every count is recomputed from the per-shot records ("shots") and checked against the frozen analysis's own tables
(agreement.mps_vs_paper and mps_vs_aux per cell, sample.mps_failed, sparse.<stratum>.<run>); the script stops if any
differs. It adds no label or test.
usage: python3 -I -B rev_d1_byround.py D1_RESULTS_JSON --out OUT.json      (also writes OUT.md)
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
from scipy.stats import beta as BETA

W = (10, 30, 50)
STRATA = ("B", "D", "C_low", "C_high")
RUNS = (("t2", "tau2 = 1e-8"), ("t3", "tau3 = 1e-9"))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cp_up(k, n, a=0.05):                    # as in rev_d1_analyze.py
    return 1.0 if n == 0 or k >= n else float(BETA.ppf(1 - a, k + 1, n - k))


def cp_lo(k, n, a=0.05):
    return 0.0 if n == 0 or k <= 0 else float(BETA.ppf(a, k, n - k + 1))


def cp2(k, n):
    return [cp_lo(k, n, 0.025), cp_up(k, n, 0.025)]


def tab(rows, dec, ref):
    """Disagreements of decision dec(s) with s[ref] over the rows where dec(s) exists."""
    have = [s for s in rows if dec(s) is not None]
    dis = [s for s in have if dec(s) != s[ref]]
    add = sum(1 for s in dis if s[ref] == s["o"])
    n, k = len(have), len(dis)
    return {"n": n, "disagree": k, "adds_ML_error": add, "removes_ML_error": k - add,
            "rate": (k / n) if n else None, "ci95": cp2(k, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    t0 = time.time()
    res = json.load(open(a_.results))
    shots = res["shots"]
    q_of = lambda s: s["q"] if s.get("mps_done") else None
    checks = {"mps_cells_match": 0, "failed_cells_match": 0, "sparse_strata_match": 0}
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "label": "D1-6: descriptive counts by round count; no label or test (post hoc revision analysis)",
           "input": {"path": os.path.abspath(a_.results), "md5": md5(a_.results),
                     "analysis_script_md5": res.get("script_md5"), "analysis_made": res.get("made")},
           "sample": {"selected": len(shots), "mps_done": sum(1 for s in shots if s.get("mps_done")),
                      "mps_failed": sum(1 for s in shots if s.get("mps_failed")),
                      "with_sparse": sum(1 for s in shots if "sparse" in s)},
           "by_round": {}}
    sp_tot = {}
    for r in W:
        cells, wt = {}, {}
        for st in STRATA:
            key = f"{st}|{r}"
            cell = [s for s in shots if s["stratum"] == st and int(s["rounds"]) == r]
            dn = [s for s in cell if s.get("mps_done")]
            failed = sum(1 for s in cell if s.get("mps_failed"))
            c = {"N": int(res["cells"][key]["N"]), "selected": len(cell), "mps_done": len(dn), "failed": failed,
                 "not_decoded": len(cell) - len(dn) - failed}
            assert c["selected"] == int(res["cells"][key]["n"]), f"{key}: selected rows differ from the design"
            fz = int(res["sample"]["mps_failed"].get(key, 0))
            assert fz == failed, f"{key}: failed rows differ ({failed} vs {fz})"
            checks["failed_cells_match"] += 1
            for ref, nm in (("m", "mps_vs_paper"), ("a", "mps_vs_aux")):
                t = tab(dn, q_of, ref)
                t["unresolved"] = sum(1 for s in dn if not s["resolved"])
                t["disagree_unresolved"] = sum(1 for s in dn if not s["resolved"] and s["q"] != s[ref])
                f = res["agreement"][nm][key]
                mine = (t["n"], t["disagree"], t["adds_ML_error"], t["removes_ML_error"], t["unresolved"], t["ci95"])
                theirs = (f["done"], f["disagree"], f["adds_ML_error"], f["removes_ML_error"], f["unresolved"], f["ci95"])
                assert mine == theirs, f"{key} {nm}: recount {mine} differs from the analysis {theirs}"
                assert t["disagree"] - t["disagree_unresolved"] == f["disagree_resolved"], f"{key} {nm}: resolved split differs"
                checks["mps_cells_match"] += 1
                c[nm] = t
            sp = [s for s in cell if "sparse" in s]
            for nm, _ in RUNS:
                t = tab(sp, lambda s, nm=nm: (int(s["sparse"][nm]["pred"]) if nm in s["sparse"] else None), "m")
                c[f"sparse_{nm}_vs_paper"] = t
                acc = sp_tot.setdefault((st, nm), [0, 0])
                acc[0] += t["n"]; acc[1] += t["disagree"]
            cells[st] = c
        for nm in ("mps_vs_paper", "mps_vs_aux", "sparse_t2_vs_paper", "sparse_t3_vs_paper"):
            num = den = 0.0
            used = []
            for st in STRATA:
                t = cells[st][nm]
                if not t["n"]:
                    continue
                num += cells[st]["N"] * t["rate"]; den += cells[st]["N"]; used.append(st)
            wt[nm] = {"rate": (num / den) if den else None, "strata": used, "N": int(den)}
        out["by_round"][str(r)] = {"cells": cells, "population_weighted": wt}
    for st in STRATA:                                   # the analysis pools the sparse runs over r, per stratum
        for nm, _ in RUNS:
            n_, k_ = sp_tot.get((st, nm), (0, 0))
            f = (res.get("sparse") or {}).get(st, {}).get(nm)
            if f is None:
                assert n_ == 0, f"{st} {nm}: rows exist but the analysis lists none"
            else:
                assert (f["n"], f["vs_paper_changes"]) == (n_, k_), f"{st} {nm}: sparse recount differs"
            checks["sparse_strata_match"] += 1
    out["checks"] = checks
    out["seconds"] = time.time() - t0
    with open(a_.out + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(a_.out + ".tmp", a_.out)
    md = os.path.splitext(a_.out)[0] + ".md"
    with open(md + ".tmp", "w") as fh:
        fh.write(write_md(out))
    os.replace(md + ".tmp", md)
    print(json.dumps({"checks": checks, "sample": out["sample"], "seconds": out["seconds"]}))


def write_md(o):
    pc = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"
    ci = lambda v: "n/a" if v[0] is None else f"{100 * v[0]:.2f}–{100 * v[1]:.2f}%"
    L = ["# D1-6: D1's counts by round count", "",
         f"Descriptive; no label or test. Source: `d1_results.json` (md5 {o['input']['md5']}), from "
         f"`rev_d1_analyze.py` ({o['input']['analysis_script_md5']}). Script md5 {o['script_md5']}.", "",
         "Rates are k / n with exact 95% Clopper–Pearson intervals. \"adds / removes\" splits each disagreement by "
         "whether it adds or removes an ML error. n counts the rows with a finite MPS rung (unresolved included).", ""]
    titles = (("mps_vs_paper", "MPS q against the paper's decision m"),
              ("mps_vs_aux", "MPS q against the census decision a (B: K2's τ₃; D: the non-band check; C: m)"))
    for nm, title in titles:
        L += [f"## {title}", "", "| r | stratum | N | selected | n | q ≠ ref (adds / removes) | rate (95% CI) | unresolved "
              "(of them disagreeing) | failed | not decoded |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r, v in o["by_round"].items():
            for st in STRATA:
                c = v["cells"][st]; t = c[nm]
                L.append(f"| {r} | {st} | {c['N']:,} | {c['selected']} | {t['n']} | {t['disagree']} ({t['adds_ML_error']} / "
                         f"{t['removes_ML_error']}) | {pc(t['rate'])} ({ci(t['ci95'])}) | {t['unresolved']} "
                         f"({t['disagree_unresolved']}) | {c['failed']} | {c['not_decoded']} |")
        L.append("")
    for nm, lab in RUNS:
        L += [f"## The sparse ladder's {lab} decision against m", "",
              "| r | stratum | n | flips (adds / removes) | rate (95% CI) |", "|---|---|---|---|---|"]
        for r, v in o["by_round"].items():
            for st in STRATA:
                t = v["cells"][st][f"sparse_{nm}_vs_paper"]
                if not t["n"]:
                    L.append(f"| {r} | {st} | 0 | not run (D1-1) | n/a |")
                    continue
                L.append(f"| {r} | {st} | {t['n']} | {t['disagree']} ({t['adds_ML_error']} / {t['removes_ML_error']}) | "
                         f"{pc(t['rate'])} ({ci(t['ci95'])}) |")
        L.append("")
    L += ["## Population-weighted rates per round count (point estimates)", "",
          "| r | q ≠ m | q ≠ a | τ₂ ≠ m | τ₃ ≠ m (strata) |", "|---|---|---|---|---|"]
    for r, v in o["by_round"].items():
        w = v["population_weighted"]
        L.append(f"| {r} | {pc(w['mps_vs_paper']['rate'])} | {pc(w['mps_vs_aux']['rate'])} | {pc(w['sparse_t2_vs_paper']['rate'])} | "
                 f"{pc(w['sparse_t3_vs_paper']['rate'])} ({', '.join(w['sparse_t3_vs_paper']['strata'])}) |")
    L += ["", f"Checks against the analysis's own tables: {o['checks']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
