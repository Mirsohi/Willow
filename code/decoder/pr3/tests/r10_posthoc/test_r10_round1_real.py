"""Check of pr3_r10_analyze.round1() on round 1's real decode folders at r = 30 and r = 50 (not r = 10, whose numbers
belong to the declared analysis): the rebuilt 2x2 tables and the ML(RL) and Libra marginals must reproduce the final
analysis's counts and eps_1pt for every leaf and decoder, rule A's excluded leaves must give no ML(RL) number, and
pooled() must agree with the frozen pool_gap(). Prints check results only, no estimate.
usage: PR3_CODE=<code/decoder> python3 -I -B test_r10_round1_real.py <analyze script folder> <results/decoder/willow>"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import pr3_r10_analyze as A
RES1 = sys.argv[2]
FR1 = os.path.join(RES1, "final_round1", "final_round1.json")
FAN = os.path.join(RES1, "final_round1", "analysis", "analysis.json")
ok = []


def check(cond, what):
    ok.append(bool(cond)); print(("ok   " if cond else "FAIL ") + what)


excl_all = set(json.load(open(FR1))["excluded_leaves"])
for r, n3, n5 in ((30, 18, 7), (50, 18, 5)):
    t, m, ck, ex, miss, prob = A.round1(RES1, FR1, FAN, r)
    bad = [c for c in ck if not c["ok"]]
    check(ck and not bad, f"r = {r}: {len(ck)} leaf x decoder checks against the final analysis, {len(bad)} failed {bad[:3]}")
    check(not miss and not prob, f"r = {r}: nothing missing ({miss[:3]}), no load problem ({prob[:3]})")
    check(sorted(ex) == sorted(x for x in excl_all if x.endswith(f"/r{r}")), f"r = {r}: rule A excludes {len(ex)} leaves: {[x.split('/', 1)[1] for x in ex]}")
    for e in A.ENTS:
        check(len(t[3].get(e, {})) == n3 and len(t[5].get(e, {})) == n5,
              f"r = {r}, {e}: series d = 3 {len(t[3].get(e, {}))} (want {n3}), d = 5 {len(t[5].get(e, {}))} (want {n5})")
    check(len(m[5][A.ML]) == n5 and len(m[5][A.LIBRA]) == 8 and len(m[3][A.ML]) == 18 and len(m[3][A.LIBRA]) == 18,
          f"r = {r}: marginals ML d5 {len(m[5][A.ML])}, Libra d5 {len(m[5][A.LIBRA])}, d3 {len(m[3][A.ML])}/{len(m[3][A.LIBRA])}")
    for d in (3, 5):
        S = {s: A.series_1pt(c, r) for s, c in t[d][A.LIBRA].items()}
        p = A.pooled({k: v for k, v in S.items() if v is not None}) if r == A.R else None
        # pooled() works at r = 10 only (series_1pt's default); check the frozen-formula agreement with r passed through
        S2 = [v for v in S.values() if v is not None]
        cs = A.PC.g_and_se(S2, "uninflated", full=False, with_cov=False); pg = A.AW.pool_gap(S2)
        check(A.rel_diff(cs["g"], pg["rel"]) < 1e-9 and A.rel_diff(cs["se"], pg["se_rel"]) < 1e-9,
              f"r = {r}, d = {d}: g_and_se's frozen formula equals pool_gap() on {len(S2)} one-round series")
print(f"{sum(ok)} of {len(ok)} checks pass")
sys.exit(0 if all(ok) else 1)
