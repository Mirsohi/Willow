"""Synthetic check of pr3_analyze.py: fake part outputs with a known gap, three tallies (all pass, one leaf excluded,
kill), and independent recomputation of the counts, g_7 (by the frozen estimators), H7, the count ratio and the NN row."""
import json, math, os, subprocess, sys, tempfile
sys.path.insert(0, "/home/claude/work/decoder/pr3")
X = "/mnt/project-files/tensor-networks/deep-dives/experimental"
CODE = X + "/code/decoder"
sys.path.insert(0, CODE); sys.path.insert(0, CODE + "/supplementary")
import numpy as np
import pr3_common as C
import analyze_willow as AW
import p1_se_covariance as PC

rng = np.random.default_rng(7)
T = tempfile.mkdtemp(dir="/home/claude/work/decoder/pr3/tests")
RES = os.path.join(T, "res")
F1 = "/home/claude/work/decoder/final/test_out3/final_round1.json"
FZ = X + "/results/decoder/willow/analysis/analysis.json"
PLAN = X + "/results/decoder/willow/plan.json"
D7 = X + "/results/decoder/willow/pr3_planning/d7_numbers.json"
DN = C.LADDER.replace(":", "_").replace("/", "-")
NAMES = ["decoding_results/correlated_matching_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
         "decoding_results/harmony_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
         "decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8"]
G_TRUE = 0.10
EPS_A = {"X": 0.00186, "Z": 0.00156}
truth = {}
for b in C.BASES:
    for r in C.ROUNDS:
        lf = C.leaf(b, r)
        eA = EPS_A[b]; eM = eA / (1 + G_TRUE)
        EA = (1 - (1 - 2 * eA) ** r) / 2; EM = (1 - (1 - 2 * eM) ** r) / 2
        q = 0.107
        t = {"ML": 0, "libra": 0, "corr_mwpm": 0, "mwpm": 0, "n": 0, "n10": 0, "n01": 0, "n11": 0}
        for j in range(C.PARTS):
            st, sp, stp = (int(x) for x in C.part_slice("25k", j).split(":"))
            idx = np.arange(st, sp, stp).astype(np.int32); n = len(idx)
            obs = rng.integers(0, 2, n).astype(np.uint8)
            u = rng.random(n)
            wM = u < EM
            wA = (u < EM * (1 - q)) | ((u >= EM) & (u < EM + EA - EM * (1 - q)))
            wC = rng.random(n) < 1.5 * EM; wW = rng.random(n) < 2.0 * EM
            d = os.path.join(RES, "parts", f"p{j}", "decode", lf, "rl_25k"); os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, DN + ".npz"), idx=idx, pred=(obs ^ wM).astype(np.uint8)[:, None])
            np.savez_compressed(os.path.join(d, "corr_mwpm.npz"), idx=idx, pred=(obs ^ wC).astype(np.uint8)[:, None])
            np.savez_compressed(os.path.join(d, "mwpm.npz"), idx=idx, pred=(obs ^ wW).astype(np.uint8)[:, None])
            shipped = np.stack([(obs ^ (rng.random(n) < 0.02)).astype(np.uint8), (obs ^ (rng.random(n) < 0.02)).astype(np.uint8),
                                (obs ^ wA).astype(np.uint8)])
            np.savez_compressed(os.path.join(d, "shots.npz"), idx=idx, obs=obs, shipped_names=np.array(NAMES), shipped=shipped)
            json.dump({"status": "complete", "rounds": r}, open(os.path.join(d, "summary.json"), "w"))
            m = C.in_sample(idx, r)
            t["ML"] += int((wM & m).sum()); t["libra"] += int((wA & m).sum())
            t["corr_mwpm"] += int((wC & m).sum()); t["mwpm"] += int((wW & m).sum()); t["n"] += int(m.sum())
            t["n10"] += int((wA & ~wM & m).sum()); t["n01"] += int((~wA & wM & m).sum()); t["n11"] += int((wA & wM & m).sum())
        truth[lf] = t


def tally(still):
    v = {}
    for lf, t in truth.items():
        v[lf] = {"verdict": "still fails: excluded and flagged" if lf in still else "pass (ladder)",
                 "n_sample": t["n"], "ladder": {"errors_tau2": t["ML"]}}
    return {"design": "25k", "verdicts": v, "still_failing": sorted(still), "kill": len(still) >= 2,
            "complete": True}


def run(name, still):
    tp = os.path.join(T, f"tally_{name}.json")
    json.dump(tally(still), open(tp, "w"))
    out = os.path.join(T, f"out_{name}")
    env = dict(os.environ, PR3_CODE=CODE)
    p = subprocess.run([sys.executable, "-I", "-B", "/home/claude/work/decoder/pr3/pr3_analyze.py", "--res", RES,
                        "--out", out, "--tally", tp, "--final-r1", F1, "--frozen", FZ, "--plan", PLAN, "--d7", D7,
                        "--allow-incomplete"], env=env, capture_output=True, text=True)
    print(f"--- {name}: exit {p.returncode}\n{p.stdout[-1500:]}{p.stderr[-3000:]}")
    assert p.returncode == 0
    return json.load(open(os.path.join(out, "pr3_results.json")))


def check(o, still):
    R = o["results"]
    for lf, t in truth.items():
        x = o["leaves"][lf]
        for e in ("ML", "libra", "corr_mwpm", "mwpm"):
            assert x[f"errors_{e}"] == t[e], (lf, e, x[f"errors_{e}"], t[e])
        assert x["n_sample"] == t["n"]
    # g_7 independently: frozen paired_series + pool_gap on the truth tables, and entry 7's SE via p1_se_covariance
    tabs = {b: {r: {k: truth[C.leaf(b, r)][k] for k in ("n", "n10", "n01", "n11")} for r in C.ROUNDS
                if C.leaf(b, r) not in still} for b in C.BASES}
    S = [PC.series_params(tabs[b], tuple(sorted(tabs[b]))) for b in C.BASES if len(tabs[b]) >= 2]
    cs = PC.g_and_se(S, "inflated", full=True, with_cov=True)
    pg = AW.pool_gap([AW.paired_series(tabs[b], tuple(sorted(tabs[b]))) for b in C.BASES if len(tabs[b]) >= 2])
    g7 = R["g7"]
    print(f"g7 {g7['g']:.5f} se {g7['se']:.5f} (independent {cs['g']:.5f} {cs['se']:.5f}; pool_gap {pg['rel']:.5f} "
          f"{pg['se_rel']:.5f}; true {G_TRUE}) check {g7['check_vs_frozen_pool_gap']}")
    assert abs(g7["g"] - cs["g"]) < 1e-12 and abs(g7["se"] - cs["se"]) < 1e-12 and abs(g7["g"] - pg["rel"]) < 1e-12
    assert abs(g7["se_frozen_formula"] - pg["se_rel"]) < 1e-12
    F = json.load(open(F1))["decisions"]["d5_rules"]["A"]["g5"]
    D = math.log(1 + cs["g"]) - math.log(1 + F["g"])
    se = math.sqrt((cs["se"] / (1 + cs["g"])) ** 2 + (F["se"] / (1 + F["g"])) ** 2)
    h = R["H7"]
    print(f"H7 Delta {h['Delta']:+.5f} se {h['se_Delta']:.5f} p {h['p']:.4g} -> {h['decision']} (independent {D:+.5f} {se:.5f}"
          f" p {math.erfc(abs(D / se) / math.sqrt(2)):.4g})")
    assert abs(h["Delta"] - D) < 1e-12 and abs(h["se_Delta"] - se) < 1e-12
    assert abs(h["p"] - math.erfc(abs(D / se) / math.sqrt(2))) < 1e-12
    keep = [lf for lf in truth if lf not in still]
    XA = sum(truth[lf]["libra"] for lf in keep); YM = sum(truth[lf]["ML"] for lf in keep)
    cr = R["count_ratio_libra_vs_ML"]
    print(f"count ratio {cr['rel']:.5f} over {cr['leaves']} leaves (independent {XA / YM - 1:.5f})")
    assert abs(cr["rel"] - (XA / YM - 1)) < 1e-12 and cr["leaves"] == len(keep)
    d7 = json.load(open(D7))["G1_d7_libra"]["r>=10"]
    eM = d7["ours"] / (1 + cs["g"])
    nn = R["NN_paired_S1"]
    print(f"NN (a) eps_ML7 {nn['eps_ML_rl_7']:.4e} rel {nn['rel']:+.4f} CI {nn['ci95']} (independent eps {eM:.4e})")
    assert abs(nn["eps_ML_rl_7"] - eM) < 1e-15
    lam = R["Lambda_on_10_30_50"]
    for ent, v in lam.items():
        print(ent, {k: (round(x[0], 6), round(x[1], 7)) for k, x in v["eps"].items()},
              "L357", v["Lambda_3_5_7"] and round(v["Lambda_3_5_7"]["Lambda"], 3), "L57", v["Lambda_5_7"] and round(v["Lambda_5_7"]["Lambda"], 3))
    # ML eps at d = 7 independently (series_eps over the non-excluded rounds)
    fits = []
    for b in C.BASES:
        pts = {r: (truth[C.leaf(b, r)]["n"], truth[C.leaf(b, r)]["ML"]) for r in C.ROUNDS if C.leaf(b, r) not in still}
        f = AW.series_eps(pts, tuple(sorted(pts)))
        if f:
            fits.append(f)
    m7, s7 = AW.mean_eps(fits)
    assert abs(lam["ML.rl"]["eps"]["7"][0] - m7) < 1e-15 and abs(lam["ML.rl"]["eps"]["7"][1] - s7) < 1e-15
    print("checks passed")


o = run("pass", set())
check(o, set())
ex = {C.leaf("X", 30)}
o2 = run("excl", ex)
assert o2["d7_excluded_leaves"] == sorted(ex)
check(o2, ex)
print("g7 series windows (excl):", o2["results"]["g7"]["series"], o2["results"]["g7"]["left_out"])
o3 = run("kill", {C.leaf("X", 30), C.leaf("Z", 50)})
print("kill:", o3["results"])
assert "not evaluated" in o3["results"]["H7"]["decision"]
# a tally that is not complete must be refused without --allow-incomplete
t = tally(set()); t["complete"] = False
json.dump(t, open(os.path.join(T, "tally_inc.json"), "w"))
p = subprocess.run([sys.executable, "-I", "-B", "/home/claude/work/decoder/pr3/pr3_analyze.py", "--res", RES, "--out",
                    os.path.join(T, "out_inc"), "--tally", os.path.join(T, "tally_inc.json"), "--final-r1", F1, "--frozen", FZ,
                    "--plan", PLAN, "--d7", D7], env=dict(os.environ, PR3_CODE=CODE), capture_output=True, text=True)
print("incomplete tally:", p.returncode, p.stderr.strip()[-200:])
assert p.returncode != 0
print(open(os.path.join(T, "out_excl", "pr3_results.md")).read())
print("ALL OK", T)
