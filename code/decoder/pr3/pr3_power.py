"""PR3 planning: Birge inflation of SE(g) at d = 5 (rebuilt tables), and cost / power of candidate d = 7 designs."""
import json, math, sys
sys.dont_write_bytecode = True
X = "/mnt/project-files/tensor-networks/deep-dives/experimental"
sys.path.insert(0, X + "/code/decoder"); sys.path.insert(0, X + "/code/decoder/supplementary")
import analyze_willow as AW
import p1_se_covariance as PC
W = X + "/results/decoder/willow"
EXP = "google_105Q_surface_code_d3_d5_d7"
pts = {}
for patch in ("q4_7", "q6_5", "q6_9", "q8_7"):
    for basis in ("X", "Z"):
        for r in (10, 30, 50):
            obs, ents, ml, info, shipped = AW.load_leaf(W, f"{EXP}/d5_at_{patch}/{basis}/r{r}", ["rl_s5"])
            pts.setdefault((patch, basis), {})[r] = AW.count2(ents["libra.rl"], ents["ML.rl"], obs)
S = [PC.series_params(P, (10, 30, 50)) for P in pts.values()]
inf = PC.g_and_se(S, "inflated", full=True, with_cov=True)["se"]
uni = PC.g_and_se(S, "uninflated", full=True, with_cov=True)["se"]
print(f"d5: SE inflated {inf:.5f}, uninflated {uni:.5f}, ratio {inf / uni:.3f}")
d7 = json.load(open("/home/claude/work/decoder/pr3/d7_numbers.json"))["designs"]
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
power = lambda z: Phi(z - 1.96) + Phi(-z - 1.96)
SE5 = 0.01247                    # g_5 with the current K2 exclusions (rule A), entry 7's SE (measured 19:40Z)
G5 = 0.0617
rows = []
for N in (10000, 25000, 50000):
    for infl in (1.0, inf / uni, 1.5):
        s6 = d7[f"10,30,50|N={N}|g=0.06"]["se_slope_uninflated"] * infl
        s11 = d7[f"10,30,50|N={N}|g=0.11"]["se_slope_uninflated"] * infl
        sd = math.sqrt(s11 ** 2 + SE5 ** 2)
        rows.append({"N": N, "inflation": round(infl, 2), "se_g7_at_6pct": s6, "power_g7_6pct_vs_0": power(0.06 / s6),
                     "se_g7_at_11pct": s11, "power_growth_11_vs_g5": power((0.11 - G5) / sd),
                     "power_growth_9_vs_g5": power((0.09 - G5) / math.sqrt((s6 * 1.05) ** 2 + SE5 ** 2)),
                     "count_se_6pct": d7[f"10,30,50|N={N}|g=0.06"]["se_count"], "ML_errors_6pct": d7[f"10,30,50|N={N}|g=0.06"]["ML_errors"]})
for x in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
# cost model (estimate): tau1 45 ms per round per shot at r = 10 (measured 42.4-46.6 at r = 10-13), x(1 + 0.0075 (r - 10));
# band share 0.5% at r = 10 (measured 0.40-0.65%), x(1 + 0.1125 (r - 10)) as at d = 5; tau2 = 1e-7 400 ms per round per band
# shot (measured 370-400), x(1 + 0.005 (r - 10)); extra rung tau3 = 1e-8 at 4.5 x tau2 (d = 5: 3.5-4.5)
def cost(N, rounds=(10, 30, 50), bases=2, t1=0.045, share=0.005, t2=0.40, k3=4.5):
    c1 = c2 = 0.0
    for r in rounds:
        c1 += N * r * t1 * (1 + 0.0075 * (r - 10))
        c2 += N * share * (1 + 0.1125 * (r - 10)) * r * t2 * (1 + 0.005 * (r - 10))
    return {"tau1": bases * c1 / 3600, "tau2": bases * c2 / 3600, "tau3": bases * k3 * c2 / 3600}
for N in (10000, 25000, 50000):
    c = cost(N)
    print(N, {k: round(v, 1) for k, v in c.items()}, "total", round(sum(c.values()), 1), "without tau3", round(c["tau1"] + c["tau2"], 1))
allr = (10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250)
c = cost(50000, allr)
print("all 26 Libra leaves, full shots:", {k: round(v) for k, v in c.items()})
json.dump({"inflation_d5": inf / uni, "rows": rows, "costs": {N: cost(N) for N in (10000, 25000, 50000)},
           "cost_all_26_full": c}, open("/home/claude/work/decoder/pr3/pr3_power.json", "w"), indent=1)
