"""Share of Libra's d = 5 net loss to ML(RL) carried by decisions K2 never re-checks (scout, 2026-10-10).

Descriptive, post hoc, no decoding. Supports the exact description of K2 asked for in main's answers to scout's
internal review: K2 re-decodes only the band shots (|P - 1/2| < 0.1 at tau_1), so every other ML decision at d = 5
comes from tau_1 = 1e-6 alone.

On the 24 d = 5 leaves of Libra's rounds {10, 30, 50} (8 series, every 5th shot, 10,000 per leaf), from the stored
ladder output (`ml_tau2_1e-6-1e-8_0.1.npz`: pred, llr, llr_t1, redecoded) and the shipped Libra predictions:
  net loss = #(Libra wrong, ML right) - #(Libra right, ML wrong), split by whether the ladder re-decoded the shot,
  and by the final |llr| (< 0.405, the band's edge; 0.405-0.5; >= 0.5), for all 24 leaves and under rule A
  (leaves still failing K2 left out).
Usage: OMP_NUM_THREADS=1 nice -n 10 python3 -I willow_outside_band_share.py <experimental-root>/ <out.json>
"""
import json
import sys

import numpy as np

root, out = sys.argv[1], sys.argv[2]
DEC = root + "results/decoder/willow/decode/google_105Q_surface_code_d3_d5_d7/"
P5 = ["q4_7", "q6_5", "q6_9", "q8_7"]
ROUNDS = [10, 30, 50]
LADDER = "ml_tau2_1e-6-1e-8_0.1.npz"
k2 = json.load(open(root + "results/decoder/willow/supplementary/k2_extra_rung/summary.json"))
failing = {s.split("google_105Q_surface_code_d3_d5_d7/")[1] for s in k2["still_failing"]}

rows = []
for p in P5:
    for b in "XZ":
        for r in ROUNDS:
            base = "%sd5_at_%s/%s/r%02d/rl_s5/" % (DEC, p, b, r)
            sh = np.load(base + "shots.npz", allow_pickle=False)
            ml = np.load(base + LADDER, allow_pickle=False)
            assert np.array_equal(sh["idx"], ml["idx"])
            names = list(sh["shipped_names"])
            li = [i for i, s in enumerate(names) if "libra_decoder_with_rl_optimized_prior" in s]
            assert len(li) == 1
            obs = sh["obs"].astype(np.uint8)
            pm = ml["pred"][:, 0].astype(np.uint8)
            assert np.array_equal(pm, (ml["llr"] > 0).astype(np.uint8))
            red = ml["redecoded"].astype(bool)
            band_t1 = np.abs(ml["llr_t1"]) < np.log(0.6 / 0.4)
            pl = sh["shipped"][li[0]].astype(np.uint8)
            lw, mw = pl != obs, pm != obs
            a = np.abs(ml["llr"])
            row = {"leaf": "d5_at_%s/%s/r%02d" % (p, b, r), "n": int(obs.size), "k_ml": int(mw.sum()),
                   "k_libra": int(lw.sum()), "redecoded": int(red.sum()),
                   "redecoded_equals_band_at_tau1_or_nan": bool(np.array_equal(red, band_t1 | np.isnan(ml["llr_t1"]))),
                   "k2_still_failing": ("d5_at_%s/%s/r%02d" % (p, b, r)) in failing}
            for name, m in (("redecoded", red), ("tau1_only", ~red), ("llr_lt_0.405", a < 0.405),
                            ("llr_0.405_to_0.5", (a >= 0.405) & (a < 0.5)), ("llr_ge_0.5", a >= 0.5)):
                row["n10_" + name] = int(np.sum(lw & ~mw & m))
                row["n01_" + name] = int(np.sum(~lw & mw & m))
            rows.append(row)


def pool(R):
    o = {"leaves": len(R), "shots": sum(x["n"] for x in R), "k_ml": sum(x["k_ml"] for x in R),
         "k_libra": sum(x["k_libra"] for x in R), "redecoded_shots": sum(x["redecoded"] for x in R)}
    net = sum(x["n10_tau1_only"] - x["n01_tau1_only"] + x["n10_redecoded"] - x["n01_redecoded"] for x in R)
    o["net_loss"] = net
    for name in ("redecoded", "tau1_only", "llr_lt_0.405", "llr_0.405_to_0.5", "llr_ge_0.5"):
        n10 = sum(x["n10_" + name] for x in R)
        n01 = sum(x["n01_" + name] for x in R)
        o[name] = {"n10": n10, "n01": n01, "net": n10 - n01, "share_of_net_loss": (n10 - n01) / net}
    return o


res = {"made_by": "scout, code/scout/willow_outside_band_share.py",
       "label": "descriptive, post hoc; no decoding; d = 5 interim (K2 summary made %s)" % k2["made"],
       "k2_still_failing": sorted(failing),
       "all_24_leaves": pool(rows),
       "rule_A": pool([x for x in rows if not x["k2_still_failing"]]),
       "checks": {"redecoded_equals_band_at_tau1_or_nan_all": all(x["redecoded_equals_band_at_tau1_or_nan"]
                                                                  for x in rows)},
       "leaves": rows}
json.dump(res, open(out, "w"), indent=1)
for k in ("all_24_leaves", "rule_A"):
    o = res[k]
    print(k, "leaves", o["leaves"], "shots", o["shots"], "redecoded", o["redecoded_shots"], "net", o["net_loss"],
          {n: (o[n]["net"], round(o[n]["share_of_net_loss"], 3)) for n in
           ("redecoded", "tau1_only", "llr_lt_0.405", "llr_0.405_to_0.5", "llr_ge_0.5")})
print(res["checks"])
