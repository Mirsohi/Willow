"""Expected SE of the P1 statistic g for the assumed real layout, from expected 2x2 counts (no sampling).
Assumptions (all assumed, none measured on Willow): eps_ML per cycle, a SPAM factor c in E_r = (1 - c (1-2 eps)^r)/2,
Libra worse by a relative eps gap gtrue, and a discordance b = q * E_ML (ML wrong, Libra right) per leaf."""
import sys, itertools
sys.path.insert(0, "/mnt/project-files/tensor-networks/deep-dives/experimental/code/decoder")
import analyze_willow as A

# P1 uses the d = 3 rounds r >= 10 that have Libra predictions: the real archive has none at r = 1 and 13 (layout read
# from the zip's central directory by the fetch session, 2026-10-08), so 13 round counts per series
R = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
def leaf(eps, r, c=0.98):
    return (1 - c * (1 - 2 * eps) ** r) / 2
out = []
for eps_ml, gtrue, q, n, S in itertools.product([6.5e-3, 7.0e-3], [0.0, 0.03], [0.1, 0.3], [50000], [18]):
    rows = []
    for s in range(S):
        pts = {}
        for r in R:
            EM = leaf(eps_ml, r); EA = leaf(eps_ml * (1 + gtrue), r)
            b = q * EM; a = b + (EA - EM)            # a = P(A wrong, ML right), b = P(A right, ML wrong)
            n11 = EM - b
            pts[r] = {"n": n, "n10": round(a * n), "n01": round(b * n), "n11": round(n11 * n)}
        rows.append(A.paired_series(pts, tuple(x for x in R if x >= 10)))
    p = A.pool_gap(rows)
    out.append((eps_ml, gtrue, q, round(100 * p["rel"], 2), round(100 * p["se_rel"], 3)))
print("eps_ML, true g, discordance q, g_hat %, SE(g) %")
for o in out:
    print(o)
