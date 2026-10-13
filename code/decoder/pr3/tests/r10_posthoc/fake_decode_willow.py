"""Fake unit for test_r10_driver.py (stands in for decode_willow.py and pr3_extra_rung.py; its name ends in
decode_willow.py so that the runner counts it as a heavy process). Writes the files the runner reads, from a seed.
usage: python3 fake_decode_willow.py '<json: kind, dir, leaf, part, slice, ladder_npz>'"""
import hashlib, json, os, sys, time
import numpy as np

a = json.loads(sys.argv[1])
DN = "ml_tau2_1e-6-1e-7_0.1"
basis = a["leaf"].split("/")[-2]
rng = np.random.default_rng(1000 * (basis == "Z") + a["part"])
st, sp, stp = (int(x) for x in a["slice"].split(":"))
idx = np.arange(st, sp, stp).astype(np.int32); n = len(idx)
llr1 = rng.normal(0, 60, n).astype(np.float32)       # about 0.5% of the shots in the band, as at r = 10
obs = ((llr1 > 0) ^ (rng.random(n) < 0.015)).astype(np.uint8)   # about 1.5% errors, so the limit is small
red = np.abs(1 / (1 + np.exp(-llr1)) - 0.5) < 0.1
llr = llr1.copy()
if basis == "X":                                   # X: tau2 moves the band shots a lot (fails rule 2'), tau3 nothing
    llr[red] = llr1[red] + rng.normal(0, 2.0, red.sum()).astype(np.float32)
time.sleep(0.6)
if a["kind"] == "ladder":
    os.makedirs(a["dir"], exist_ok=True)
    np.savez_compressed(os.path.join(a["dir"], DN + ".npz"), idx=idx, pred=(llr > 0).astype(np.uint8)[:, None], llr=llr,
                        llr_t1=llr1, redecoded=red)
    np.savez_compressed(os.path.join(a["dir"], "shots.npz"), idx=idx, obs=obs)
    json.dump({"status": "complete", "decoders": {"ml_tau2:1e-6/1e-7:0.1": {}}, "selection": a["slice"]},
              open(os.path.join(a["dir"], "summary.json"), "w"))
else:
    b = np.where(red)[0]
    os.makedirs(a["dir"], exist_ok=True)
    np.savez_compressed(os.path.join(a["dir"], "extra_rung.npz"), idx=idx[b], llr_t3=llr[b])
    zmd5 = hashlib.md5(open(a["ladder_npz"], "rb").read()).hexdigest()
    json.dump({"tau3": {}, "ladder_npz_md5": zmd5}, open(os.path.join(a["dir"], "extra_rung.json"), "w"))
