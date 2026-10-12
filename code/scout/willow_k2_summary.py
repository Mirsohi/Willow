"""K2's final per-leaf summary at d = 5, from decoder's 32 extra_rung.json files (scout, 2026-10-11).

Descriptive arithmetic on posted outputs; no decoding. For methods §4.3 (the per-leaf ladder summary and K2's outcome)
and results §5.1.5 (truncation from tau_2 to tau_3 over all 32 extra rungs).
Inputs (read only): results/decoder/willow/supplementary/k2_extra_rung/k2_tally.json and
  results/decoder/willow/supplementary/k2_extra_rung/google_105Q_surface_code_d3_d5_d7/d5_at_*/*/r*/rl_s5/extra_rung.json
Usage: python3 -I willow_k2_summary.py <experimental-root>/ <out.json>
"""
import glob
import hashlib
import json
import sys

root, out = sys.argv[1], sys.argv[2]
K2 = root + "results/decoder/willow/supplementary/k2_extra_rung/"
tally = json.load(open(K2 + "k2_tally.json"))
files = sorted(glob.glob(K2 + "google_105Q_surface_code_d3_d5_d7/d5_at_*/*/r*/rl_s5/extra_rung.json"))
rows = []
for f in files:
    d = json.load(open(f))
    t, lad = d["tau3"], d["ladder"]
    rows.append({"leaf": d["leaf"].split("/", 1)[1], "verdict": d["verdict"], "passes": d["verdict"] == "pass",
                 "rule1_errors": t["rule1_errors"], "rule2_dP": t["rule2_dP"],
                 "max_abs_dP_tau2_tau3": t["max_abs_dP_tau2_tau3"], "n_dP_ge_band": t["n_dP_ge_band"],
                 "n_band": t["n_band"], "errors_ladder_final": t["errors_ladder_final"],
                 "errors_with_tau3": t["errors_with_tau3"], "decisions_changed_tau2_tau3": t["decisions_changed_tau2_tau3"],
                 "nan_tau3": t["nan_tau3"], "ladder_errors_tau1": lad["errors_tau1"],
                 "ladder_errors_final": lad["errors_final"], "tau2_recheck_identical": d["check"]["identical"],
                 "file_md5": hashlib.md5(open(f, "rb").read()).hexdigest()})
P = [r for r in rows if r["passes"]]
F = [r for r in rows if not r["passes"]]
failing_tally = sorted(s.split("google_105Q_surface_code_d3_d5_d7/")[1] for s in tally["still_failing"])
res = {"made_by": "scout, code/scout/willow_k2_summary.py", "label": "measured (decoder's files); summary by scout",
       "k2_tally": {"made": tally["made"], "complete": tally["complete"], "kill": tally["kill"],
                    "verdicts": tally["verdicts"], "cpu_core_hours_extra_rungs": tally["cpu_core_hours_extra_rungs"],
                    "md5": hashlib.md5(open(K2 + "k2_tally.json", "rb").read()).hexdigest()},
       "n_files": len(rows), "n_pass": len(P), "n_still_failing": len(F),
       "failing_equals_tally": sorted(r["leaf"] for r in F) == failing_tally,
       "every_leaf_passes_rule1": all(r["rule1_errors"] for r in rows),
       "failing_fail_rule2_only": all((not r["rule2_dP"]) and r["rule1_errors"] for r in F),
       "failing_n_dP_ge_band": sorted({r["n_dP_ge_band"] for r in F}),
       "failing_max_abs_dP_range": [min(r["max_abs_dP_tau2_tau3"] for r in F), max(r["max_abs_dP_tau2_tau3"] for r in F)],
       "passing_max_abs_dP_max": max(r["max_abs_dP_tau2_tau3"] for r in P),
       "max_abs_k3_minus_k2": max(abs(r["errors_with_tau3"] - r["errors_ladder_final"]) for r in rows),
       "band_shots_tau3": sum(r["n_band"] for r in rows),
       "decisions_changed_tau2_tau3": sum(r["decisions_changed_tau2_tau3"] for r in rows),
       "errors_ladder_final": sum(r["errors_ladder_final"] for r in rows),
       "errors_with_tau3": sum(r["errors_with_tau3"] for r in rows),
       "ladder_errors_tau1": sum(r["ladder_errors_tau1"] for r in rows),
       "nan_tau3": sum(r["nan_tau3"] for r in rows),
       "all_tau2_rechecks_identical": all(r["tau2_recheck_identical"] for r in rows),
       "leaves": rows}
res["relative_change_tau2_to_tau3"] = res["errors_with_tau3"] / res["errors_ladder_final"] - 1
res["relative_change_tau1_to_ladder"] = res["errors_ladder_final"] / res["ladder_errors_tau1"] - 1
json.dump(res, open(out, "w"), indent=1)
print({k: v for k, v in res.items() if k != "leaves"})
