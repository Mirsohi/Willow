"""Fake tau4 part for test_r10_tau4.py: writes tau4_rung.json and .npz as pr3_r10_tau4.run_rung() does, from the stored
tau3 of the part. Modes: "pass" (tau4 = tau3), "fail" (tau4 = tau3 + N(0, 2) on every band shot), "burn" (spin for
--seconds of CPU first, then as "pass"), "mem" (allocate 600 MB and wait), "crash" (exit 1), "checkfail" (the tau3
check fails: a "not run" json, no npz, exit 4).
usage: python3 fake_tau4_worker.py '<json: odir, edir, mode, part, seconds>'"""
import hashlib, json, os, sys, time
import numpy as np

a = json.loads(sys.argv[1])
mode = a["mode"]
if mode == "crash":
    sys.exit(1)
if mode == "mem":
    x = np.ones(600 * 2 ** 20 // 8)
    time.sleep(30)
    sys.exit(0)
if mode == "burn":
    t0 = time.process_time()
    while time.process_time() - t0 < a.get("seconds", 20):
        pass
e = np.load(os.path.join(a["edir"], "extra_rung.npz"))
l3 = e["llr_t3"]
emd5 = hashlib.md5(open(os.path.join(a["edir"], "extra_rung.npz"), "rb").read()).hexdigest()
os.makedirs(a["odir"], exist_ok=True)
check = {"tau3": 1e-8, "kmax": 2_000_000, "n_checked": int(min(10, len(l3))), "n_band": int(len(l3)),
         "identical": mode != "checkfail"}
if mode == "checkfail":
    json.dump({"check": check, "extra_rung_npz_md5": emd5, "first": None,
               "verdict": "not run: the rebuilt tau3 rung does not reproduce the stored tau3"},
              open(os.path.join(a["odir"], "tau4_rung.json"), "w"))
    sys.exit(4)
rng = np.random.default_rng(77 + a["part"])
l4 = l3 + (rng.normal(0, 2.0, len(l3)).astype(np.float32) if mode == "fail" else 0)
mk = rng.integers(100, 1000, len(l3))
np.savez_compressed(os.path.join(a["odir"], "tau4_rung.npz"), idx=e["idx"], llr_t4=l4.astype(np.float32),
                    logp_t4=np.zeros(len(l3)), maxk_t4=mk)
json.dump({"tau4": {"tau4": 1e-9, "n_band": int(len(l3)), "nan_tau4": 0, "max_K_tau4": int(mk.max()) if len(mk) else 0,
                    "n_at_kmax_tau4": 0},
           "check": check, "extra_rung_npz_md5": emd5, "first": None, "kmax_t4": 8_000_000, "verdict": "done"},
          open(os.path.join(a["odir"], "tau4_rung.json"), "w"))
