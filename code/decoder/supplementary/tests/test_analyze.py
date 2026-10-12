"""Synthetic check of nonband_analyze.py: fabricated check outputs with known flips on the real shot lists; the counts,
the estimate, the worst cases and the modified count gap are recomputed independently; the base g_5 must equal
final_round1's (test run, same exclusions)."""
import json, math, os, subprocess, sys, tempfile
import numpy as np
X = "/mnt/project-files/tensor-networks/deep-dives/experimental"
sys.path.insert(0, X + "/code/decoder"); sys.path.insert(0, X + "/code/decoder/supplementary")
from scipy.stats import beta

LISTS = "/home/claude/work/decoder/nonband/test_lists"
F1 = "/home/claude/work/decoder/final/test_out3/final_round1.json"
T = tempfile.mkdtemp(dir="/home/claude/work/decoder/nonband")
S = {k: v for k, v in np.load(os.path.join(LISTS, "shot_lists.npz")).items()}
meta = json.load(open(os.path.join(LISTS, "shot_lists.json")))
rng = np.random.default_rng(5)
n = len(S["idx"])
pred2 = S["ml"].copy(); pred3 = np.full(n, 255, np.uint8); reason = np.zeros(n, np.int8)
typ, ml, u = S["type"], S["ml"], S["u"]
flip = np.zeros(n, bool)
for t, k in (("D10", 10), ("D01", 5), ("C00", 2)):
    flip[rng.choice(np.where(typ == t)[0], k, replace=False)] = True
pred2[flip] = 1 - ml[flip]
pred3[flip] = pred2[flip]; reason[flip] = 1                                 # targeted, confirmed at tau3
rnd = (~flip) & (u < 0.1)
pred3[rnd] = pred2[rnd]; reason[rnd] = 2
d_rnd = np.where(rnd & (S["stratum"] == 0))[0]
only3 = d_rnd[0]                                                            # one tau3-only change in D's random subsample
pred3[only3] = 1 - pred2[only3]
llr2 = S["llr_t1"].astype(np.float32).copy()
llr2[flip] = -llr2[flip]                                                    # consistent with the flipped decision
os.makedirs(os.path.join(T, "leaves"))
for k in range(24):
    r = np.where(S["leaf_id"] == k)[0]
    np.savez_compressed(os.path.join(T, "leaves", f"{k:02d}.npz"), row=r, llr_t1=S["llr_t1"][r], llr_t2=llr2[r],
                        pred_t2=pred2[r], done_t2=np.ones(len(r), bool), llr_t3=np.where(pred3[r] != 255, llr2[r], np.nan),
                        pred_t3=pred3[r], done_t3=pred3[r] != 255, t3_reason=reason[r])
json.dump({"status": "complete", "cpu_core_hours": 0.0, "rebuild": {}, "test_only": True}, open(os.path.join(T, "summary.json"), "w"))
out = os.path.join(T, "analysis")
p = subprocess.run([sys.executable, "-I", "-B", "/home/claude/work/decoder/nonband/nonband_analyze.py", "--res",
                    X + "/results/decoder/willow", "--lists", LISTS, "--check", T, "--final-r1", F1, "--out", out,
                    "--allow-incomplete"], env=dict(os.environ, NONBAND_CODE=X + "/code/decoder"), capture_output=True, text=True)
print(p.stdout[-2000:], p.stderr[-3000:])
assert p.returncode == 0
o = json.load(open(os.path.join(out, "nonband_results.json")))
D = o["flips"]["D"]["tau1_to_final"]; C = o["flips"]["C"]["tau1_to_final"]
right1 = ml == S["obs"]
exp_D = int(((S["stratum"] == 0) & (flip | (np.arange(n) == only3))).sum())
print("D flips", D, "expected", exp_D, "| C flips", C)
assert D["flips"] == exp_D and C["flips"] == 2
assert D["right_to_wrong"] == 10 + int(typ[only3] == "D10") and C["right_to_wrong"] == 2
r3 = o["tau3_only_rates"]["D"]
n3 = len(d_rnd)
assert r3["k"] == 1 and r3["n"] == n3, r3
# estimate: D observed + unrun x rate; C00: N x 2 / n_C00
unrunD = int(((S["stratum"] == 0) & ~flip & ~rnd).sum())
nC00 = int((typ == "C00").sum()); NC00 = meta["population"]["C00"]
estA = o["rules"]["none"]["estimate"]["flips_total"]
print("estimate totals (none)", estA)
assert abs(estA["D10"] + estA["D01"] - (exp_D + unrunD / n3)) < 1e-9
assert abs(estA["C00"] - NC00 * 2 / nC00) < 1e-6 and estA["C11"] == 0
# worst case lowering g: D10 observed + unrun_D10 x upper; C00: N x upper(2, nC00)
up3 = float(beta.ppf(0.95, 2, n3 - 1))
unrunD10 = int(((typ == "D10") & ~flip & ~rnd).sum())
obsD10 = int(((typ == "D10") & flip).sum()) + int(typ[only3] == "D10")
wl = o["rules"]["none"]["worst_lower_g"]["flips_total"]
print("worst lower-g totals (none)", wl)
assert abs(wl["D10"] - obsD10) < 1e-6 and abs(wl["D01"] - (5 + int(typ[only3] == "D01"))) < 1e-9
assert abs(wl["C00"] - NC00 * float(beta.ppf(0.95, 3, nC00 - 2))) < 1e-6 and wl["C11"] == 0
wl3 = o["rules"]["none"]["worst_lower_g_with_tau3_only"]["flips_total"]
print("worst lower-g with the tau3-only bound (none)", wl3)
assert abs(wl3["D10"] - (obsD10 + unrunD10 * up3)) < 1e-6
# modified count gap (no exclusion): ML errors change by D10 + C00 - D01 - C11
b = o["rules"]["none"]["base"]["count_gap"]; e = o["rules"]["none"]["estimate"]["count_gap"]
dML = estA["D10"] + estA["C00"] - estA["D01"] - estA["C11"]
assert abs(e["errors_ML"] - (b["errors_ML"] + dML)) < 1e-6 and e["errors_libra"] == b["errors_libra"]
print("count gap base", b, "\n          estimate", e)
print("base vs final_round1:", o["base_vs_final_round1"])
assert all(v["g"] < 1e-12 and v["se"] < 1e-12 for v in o["base_vs_final_round1"].values()) and len(o["base_vs_final_round1"]) == 2
for rule in ("A", "none"):
    r = o["rules"][rule]
    print(rule, {c: (round(100 * r[c]["g5"]["g"], 3), round(100 * r[c].get("delta_g5", 0), 3)) for c in ("base", "estimate", "worst_lower_g", "worst_higher_g", "worst_lower_g_with_tau3_only")})
print(open(os.path.join(out, "nonband_results.md")).read()[:3000])
print("ALL OK", T)
