"""Fake unit for test_r10_analyze.py: stands in for decode_willow.py (a ladder part) and pr3_extra_rung.py (an extra
part) and writes every file they write that the driver and pr3_r10_analyze.py read: summary.json, shots.npz with the 5
shipped rows, the ladder's npz, corr_mwpm.npz and mwpm.npz; extra_rung.npz and extra_rung.json. Its name ends in
decode_willow.py's tail so that the runner counts it as a heavy process ("..._decode_willow.py").
Data: ML(RL) wrong on about 1.7% of shots; about 0.5% of shots in the tau2 band; the other decoders are ML(RL)'s final
decisions with extra errors (A-only) and fixes (ML-only). Modes: per basis "pass" (tau2 = tau1: passes the ladder),
"extra" (tau2 moves the band: fails the ladder; tau3 = tau2: passes the extra rung), "fail" (tau2 and tau3 both move
the band: still fails).
usage: python3 fake_full_decode_willow.py '<json: kind, dir, leaf, part, slice, ladder_npz, modes {X: .., Z: ..}>'"""
import hashlib, json, os, sys, time
import numpy as np

a = json.loads(sys.argv[1])
DN = "ml_tau2_1e-6-1e-7_0.1"
LADDER = "ml_tau2:1e-6/1e-7:0.1"
SHIPPED = ["decoding_results/correlated_matching_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
           "decoding_results/correlated_matching_decoder_with_si1000_prior/obs_flips_predicted.b8",
           "decoding_results/harmony_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
           "decoding_results/harmony_decoder_with_si1000_prior/obs_flips_predicted.b8",
           "decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8"]
# (A-only error rate on shots ML gets right, ML-only rate on shots ML gets wrong)
RATES = {"corrmatch.rl": (0.006, 0.04), "corrmatch.si1000": (0.008, 0.04), "harmony.rl": (0.003, 0.05),
         "harmony.si1000": (0.004, 0.05), "libra.rl": (0.002, 0.05), "corr_mwpm": (0.005, 0.04), "mwpm": (0.012, 0.03)}
SH_ENT = ["corrmatch.rl", "corrmatch.si1000", "harmony.rl", "harmony.si1000", "libra.rl"]
basis = a["leaf"].split("/")[-2]
mode = a["modes"][basis]
time.sleep(0.3)
if a["kind"] == "ladder":
    rng = np.random.default_rng(1000 * (basis == "Z") + a["part"])
    st, sp, stp = (int(x) for x in a["slice"].split(":"))
    idx = np.arange(st, sp, stp).astype(np.int32); n = len(idx)
    llr1 = rng.normal(0, 60, n).astype(np.float32)
    obs = ((llr1 > 0) ^ (rng.random(n) < 0.017)).astype(np.uint8)
    red = np.abs(1 / (1 + np.exp(-llr1.astype(np.float64))) - 0.5) < 0.1
    llr = llr1.copy()
    if mode in ("extra", "fail"):
        llr[red] = llr1[red] + rng.normal(0, 2.0, red.sum()).astype(np.float32)
    df = (llr > 0).astype(np.uint8)
    ml_wrong = df != obs
    preds = {}
    for e, (p, q) in RATES.items():
        u = rng.random(n)
        flip = np.where(ml_wrong, u < q, u < p)
        preds[e] = (df ^ flip).astype(np.uint8)
    os.makedirs(a["dir"], exist_ok=True)
    np.savez_compressed(os.path.join(a["dir"], DN + ".npz"), idx=idx, pred=df[:, None], llr=llr, llr_t1=llr1,
                        logp=np.zeros(n), logp_t1=np.zeros(n), redecoded=red, maxk_t1=np.full(n, 100), seconds=1.0)
    np.savez_compressed(os.path.join(a["dir"], "shots.npz"), idx=idx, obs=obs, shipped_names=np.array(SHIPPED),
                        shipped=np.stack([preds[e] for e in SH_ENT]))
    for b in ("corr_mwpm", "mwpm"):
        np.savez_compressed(os.path.join(a["dir"], b + ".npz"), idx=idx, pred=preds[b][:, None], seconds=1.0)
    dec = {"corr_mwpm": {}, "mwpm": {}, LADDER: {}}
    dec.update({"shipped:" + s: {} for s in SHIPPED})
    json.dump({"status": "complete", "decoders": dec, "selection": a["slice"], "n_total": 50000, "rounds": 10},
              open(os.path.join(a["dir"], "summary.json"), "w"))
else:
    z = np.load(a["ladder_npz"])
    rng = np.random.default_rng(5000 + 1000 * (basis == "Z") + a["part"])
    b = np.where(z["redecoded"])[0]
    l3 = z["llr"][b].copy()
    if mode == "fail":
        l3 = l3 + rng.normal(0, 2.0, len(b)).astype(np.float32)
    mk = rng.integers(100, 1000, len(b))
    if len(b):
        mk[0] = 2000000                         # one band shot per part at kmax
    os.makedirs(a["dir"], exist_ok=True)
    np.savez_compressed(os.path.join(a["dir"], "extra_rung.npz"), idx=z["idx"][b], llr_t3=l3, logp_t3=np.zeros(len(b)),
                        maxk_t3=mk)
    zmd5 = hashlib.md5(open(a["ladder_npz"], "rb").read()).hexdigest()
    ch = int(np.sum((l3 > 0) != (z["llr"][b] > 0)))
    json.dump({"tau3": {"n_band": int(len(b)), "decisions_changed_tau2_tau3": ch, "max_K_tau3": int(mk.max()) if len(b) else 0,
                        "ms_per_round_per_band_shot_tau3": 1.0},
               "check": {"n_checked": int(min(10, len(b))), "identical": True, "max_abs_llr_diff": 0.0},
               "verdict": "fake", "ladder_npz_md5": zmd5, "kmax": 2000000},
              open(os.path.join(a["dir"], "extra_rung.json"), "w"))
