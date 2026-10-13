#!/usr/bin/env python3
"""D1-5b and D1-5c: the reporting labels of D1's result (decoder, 2026-10-13). Declared in D1-5b and D1-5c of
notes/decoder/willow_revision_declarations.md. Changes nothing in D1's design, runner or analysis.

d1_results.json (rev_d1_analyze.py) has no interval for the count gap with shot noise. This script therefore replays
the frozen analysis on the same inputs, with the same code and seeds, and records each posterior draw's count gap and
its shot-noise SE. The replay must reproduce d1_results.json exactly (apart from its time stamps), or the script stops.
The replay's posterior draws are then the analysis's own draws; their g5 and count-gap quantiles are checked against
d1_results.json too.

Definitions (for a rule R, A or none, and a level L, cells or pooled; from d1_results.json, key rules.R):
  paper's g5         base.g5.g
  corrected g5       estimate_diff.g5.g (D1-1's difference estimator); corrected count gap: estimate_diff.count_gap.gap_errors
  correction         corrected g5 minus the paper's g5, with only the MPS sample's uncertainty: the posterior draws of g5
                     without hardware shot noise (posterior_diff_L.g5_correction_only_q025_q50_q975; rates drawn per cell
                     (L = cells) or per stratum (L = pooled) and side; census part fixed) minus the paper's g5. Its 95%
                     interval: [q025 - paper's g5, q975 - paper's g5].
  contradiction      that interval excludes 0 (the paper's g5 lies below q025 or above q975).
  g5 bounded above 0 the 2.5% quantile of posterior_diff_L.g5_with_shot_noise_q025_q50_q975 is above 0.
  count gap with shot noise (D1-5c), per draw: the draw's count gap (Libra's errors minus ML's, over the rule's leaves)
                     plus z x its shot-noise SE, z standard normal (seed: the posterior's seed + 1,000,000). The SE is the
                     paired multinomial one, sqrt(sum over leaves of n10 + n01 - (n10 - n01)^2 / n) on the draw's tables.
                     Its 95% interval: the 2.5% and 97.5% quantiles over the 2,000 draws. "above 0": 2.5% quantile > 0;
                     "at or below 0": 97.5% quantile <= 0; otherwise it straddles 0.
  label (mutually exclusive, in this order):
    contradiction:    "partial" if g5 is bounded above 0 and the count-gap interval lies above 0; else "disagreement".
    no contradiction: count-gap interval at or below 0: "disagreement"; straddles 0: "inconclusive";
                      above 0: "agreement" if g5 is bounded above 0, else "inconclusive".
    A missing quantile (every draw NaN) gives "undetermined".
  primary            rule A, level cells. Beside it: level pooled ("under the pooled reading"), and rule none.
  descriptors        whether the corrected g5 lies in [4.23, 9.09]% (the headline's 95% CI); P(count gap <= 0).
  per stratum        agreement.mps_vs_paper[B, D, C_low, C_high] (pooled over r): agreements (q = m) out of n (MPS shots
                     with a finite rung, unresolved included), with exact 95% bounds: 1 minus the Clopper-Pearson interval
                     of the disagreement rate, reversed; unresolved and failed counts beside.
usage: python3 -I -B rev_d1_labels.py RESULTS_JSON --res R --sel S --runs RUNS --final-r1 F --check C --replay-out DIR
                                      [--out labels.json]
       (the analysis arguments must be the ones that made RESULTS_JSON; DIR receives the replay's output)
"""
import argparse, contextlib, hashlib, importlib.util, json, math, os, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

HEADLINE_CI = (0.0423, 0.0909)
STRATA = ("B", "D", "C_low", "C_high")
SEEDS = {20261016: "cells", 20261017: "pooled"}                    # rev_d1_analyze.py's posterior seeds
NDRAWS = 2000
GAP_SEED_OFFSET = 1000000


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Capture:
    """Records, per (rule, level), the first NDRAWS g5 and count-gap calls after the posterior's generator is made."""

    def __init__(self):
        self.level = None; self.data = {}; self.n_g = 0; self.n_c = 0

    def start(self, level):
        self.level = level; self.n_g = 0; self.n_c = 0

    def g5(self, model, out):
        if self.level is not None and self.n_g < NDRAWS:
            key = ("A" if model.excl else "none", self.level)
            self.data.setdefault(key, {"g": [], "gap": [], "se": []})["g"].append(out["g"])
            self.n_g += 1

    def gap(self, model, out, se):
        if self.level is not None and self.n_c < NDRAWS:
            key = ("A" if model.excl else "none", self.level)
            d = self.data.setdefault(key, {"g": [], "gap": [], "se": []})
            d["gap"].append(out["gap_errors"]); d["se"].append(se)
            self.n_c += 1
            if self.n_c == NDRAWS:
                self.level = None


def replay(analysis_path, argv, cap):
    orig_sffl = importlib.util.spec_from_file_location
    orig_rng = np.random.default_rng

    def patch_model(Model):
        g5_0, cg_0 = Model.g5, Model.count_gap

        def g5(self, F):
            out = g5_0(self, F)
            cap.g5(self, out)
            return out

        def count_gap(self, F):
            out = cg_0(self, F)
            T = self.modified(F)
            v = sum(t[1] + t[2] - (t[1] - t[2]) ** 2 / t[0] for rel, t in T.items() if rel not in self.excl and t[0] > 0)
            cap.gap(self, out, math.sqrt(max(v, 0.0)))
            return out
        Model.g5, Model.count_gap = g5, count_gap

    def sffl(name, path, *a, **k):
        spec = orig_sffl(name, path, *a, **k)
        if name == "rd2":
            ex0 = spec.loader.exec_module

            def exec_module(mod):
                ex0(mod)
                patch_model(mod.Model)
            spec.loader.exec_module = exec_module
        return spec

    def rng(seed=None, *a, **k):
        if seed in SEEDS:
            cap.start(SEEDS[seed])
        return orig_rng(seed, *a, **k)

    spec = orig_sffl("d1a_replay", analysis_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    old_argv = sys.argv
    importlib.util.spec_from_file_location = sffl
    np.random.default_rng = rng
    try:
        sys.argv = [analysis_path] + argv
        with contextlib.redirect_stdout(sys.stderr):
            mod.main()
    finally:
        sys.argv = old_argv
        importlib.util.spec_from_file_location = orig_sffl
        np.random.default_rng = orig_rng


def strip(d):
    """The results without their time stamps, as canonical JSON text (NaN-safe: NaN != NaN as floats)."""
    return json.dumps({k: v for k, v in d.items() if k not in ("made", "seconds")}, sort_keys=True, default=float)


def qq(v):
    v = np.asarray(v, float)
    return [float(x) for x in np.nanquantile(v, [0.025, 0.5, 0.975])] if np.isfinite(v).any() else [None, None, None]


def label(R, level, gapq):
    g_paper = float(R["base"]["g5"]["g"])
    e = R["estimate_diff"]
    g_corr = float(e["g5"]["g"]); gap = float(e["count_gap"]["gap_errors"])
    p = R[f"posterior_diff_{level}"]
    c = p["g5_correction_only_q025_q50_q975"]; s = p["g5_with_shot_noise_q025_q50_q975"]
    out = {"paper_g5": g_paper, "corrected_g5": g_corr, "corrected_count_gap_errors": gap,
           "correction_point": g_corr - g_paper,
           "correction_ci95": None if c[0] is None else [c[0] - g_paper, c[2] - g_paper],
           "corrected_g5_ci95_with_shot_noise": None if s[0] is None else [s[0], s[2]],
           "count_gap_ci95_with_shot_noise": None if gapq[0] is None else [gapq[0], gapq[2]],
           "P_count_gap_le_0": p.get("P_count_gap_le_0"), "P_g5_le_0_with_shot_noise": p.get("P_g5_le_0_with_shot_noise"),
           "draws_with_nan_g5": p.get("draws_with_nan_g5"),
           "corrected_g5_in_headline_ci": HEADLINE_CI[0] <= g_corr <= HEADLINE_CI[1]}
    if c[0] is None or s[0] is None or gapq[0] is None:
        out.update(contradiction=None, g5_bounded_above_0=None, count_gap_interval=None, label="undetermined")
        return out
    contra = (g_paper < c[0]) or (g_paper > c[2])
    above = s[0] > 0
    gi = "above 0" if gapq[0] > 0 else ("at or below 0" if gapq[2] <= 0 else "straddles 0")
    if contra:
        lab = "partial" if (above and gi == "above 0") else "disagreement"
    elif gi == "at or below 0":
        lab = "disagreement"
    elif gi == "straddles 0":
        lab = "inconclusive"
    else:
        lab = "agreement" if above else "inconclusive"
    out.update(contradiction=contra, g5_bounded_above_0=above, count_gap_interval=gi, label=lab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    for x in ("--res", "--sel", "--runs", "--final-r1", "--check", "--replay-out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--analysis", default=os.path.join(HERE, "rev_d1_analyze.py"))
    ap.add_argument("--out")
    a = ap.parse_args()
    d = json.load(open(a.results))
    assert md5(a.analysis) == d["script_md5"], "the analysis script differs from the one that made the results"
    assert os.path.abspath(a.replay_out) != os.path.abspath(os.path.dirname(a.results)), "the replay would overwrite the results"
    cap = Capture()
    replay(a.analysis, ["--res", a.res, "--sel", a.sel, "--runs", a.runs, "--final-r1", a.final_r1, "--check", a.check,
                        "--out", a.replay_out], cap)
    r = json.load(open(os.path.join(a.replay_out, "d1_results.json")))
    assert strip(r) == strip(d), "the replay does not reproduce d1_results.json"
    O = {"script_md5": md5(os.path.abspath(__file__)), "results_md5": md5(a.results),
         "analysis_script_md5": d.get("script_md5"), "sample": d.get("sample"),
         "replay": {"reproduces_results": True}, "primary": {"rule": "A", "level": "cells"}, "labels": {}}
    for rule in ("A", "none"):
        for level in ("cells", "pooled"):
            D = cap.data.get((rule, level))
            assert D and len(D["g"]) == NDRAWS and len(D["gap"]) == NDRAWS, f"draws not captured for {rule}|{level}"
            P = d["rules"][rule][f"posterior_diff_{level}"]
            assert qq(D["g"]) == P["g5_correction_only_q025_q50_q975"], f"{rule}|{level}: g5 draws differ"
            assert qq(D["gap"]) == P["count_gap_errors_q025_q50_q975"], f"{rule}|{level}: count-gap draws differ"
            seed = [s for s, lv in SEEDS.items() if lv == level][0] + GAP_SEED_OFFSET
            z = np.random.default_rng(seed).standard_normal(NDRAWS)
            gn = np.asarray(D["gap"], float) + z * np.asarray(D["se"], float)
            L = label(d["rules"][rule], level, qq(gn))
            L["count_gap_shot_noise_se_q50"] = float(np.median(D["se"]))
            L["count_gap_seed"] = seed
            O["labels"][f"{rule}|{level}"] = L
    ag = {}
    for s_ in STRATA:
        v = d["agreement"]["mps_vs_paper"][s_]
        n, k = int(v["done"]), int(v["disagree"])
        lo, hi = v["ci95"]
        ag[s_] = {"agreements": n - k, "n": n, "ci95": [1 - hi, 1 - lo] if n else None,
                  "unresolved": int(v["unresolved"]), "disagree_adds_ML_error": int(v["adds_ML_error"]),
                  "disagree_removes_ML_error": int(v["removes_ML_error"])}
    O["agreement_per_stratum"] = ag
    O["failed_by_cell"] = (d.get("sample") or {}).get("mps_failed")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(O, f, indent=1)
    print(json.dumps(O, indent=1))


if __name__ == "__main__":
    main()
