"""Check of pr3_r10_tau4.run_rung() with the real decoder on round 1's data: a d = 5 leaf with K2's extra rung
(d5_at_q4_7/X/r10; ladder ml_tau2:1e-6/1e-8:0.1, so the stored tau3 is 1e-9 and the new rung is 1e-10). Not r = 10 of
d = 7. About half a core-minute.
  1. The tau3 rebuild check on 3 band shots reproduces K2's stored llr_t3 bitwise; tau4 runs on the first 3 band shots
     (first=3, a test option) and writes finite posteriors below the term cap.
  2. A rerun takes every chunk from the work folder and gives the same arrays.
  3. With one stored tau3 llr changed by 1e-3, the check fails: exit 4, verdict "not run", no npz.
usage: PR3_CODE=<code/decoder> python3 -I -B test_r10_tau4_worker_real.py <tau4 script folder> <results/decoder/willow>
       <round-1 extract> <scratch>"""
import json, os, shutil, sys, tempfile, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import pr3_r10_tau4 as T4
RES1, ROOT, SCR = sys.argv[2:5]
REL = "google_105Q_surface_code_d3_d5_d7/d5_at_q4_7/X/r10"
LDIR = os.path.join(RES1, "decode", REL, "rl_s5")
EDIR = os.path.join(RES1, "supplementary", "k2_extra_rung", REL, "rl_s5")
NAME = "ml_tau2:1e-6/1e-8:0.1"
ok = []


def check(cond, what):
    ok.append(bool(cond)); print(("ok   " if cond else "FAIL ") + what, flush=True)


log = lambda m: print("    " + m, flush=True)
T = tempfile.mkdtemp(dir=SCR)
c0 = time.process_time()
code = T4.run_rung(ROOT, LDIR, EDIR, os.path.join(T, "o"), os.path.join(T, "w"), REL, NAME, log, check_shots=3, first=3)
j = json.load(open(os.path.join(T, "o", "tau4_rung.json")))
z = np.load(os.path.join(T, "o", "tau4_rung.npz"))
check(code == 0 and j["check"]["identical"] and j["check"]["n_checked"] == 3 and j["check"]["tau3"] == 1e-9,
      f"exit {code}; tau3 check on {j['check']['n_checked']} band shots identical: {j['check']['identical']}")
check(len(z["llr_t4"]) == 3 and np.all(np.isfinite(z["llr_t4"])) and int(z["maxk_t4"].max()) < T4.KMAX_T4
      and j["tau4"]["tau4"] == 1e-10,
      f"tau4 = {j['tau4']['tau4']:g} on 3 band shots: finite, max K {int(z['maxk_t4'].max())} < {T4.KMAX_T4}; "
      f"{j['tau4']['cpu_seconds_tau4']:.1f} CPU s")
e = np.load(os.path.join(EDIR, "extra_rung.npz"))
check(np.array_equal(z["idx"], e["idx"][:3]), "tau4's shots are the first 3 band shots of the stored tau3")
print(f"    stored tau3 llr {e['llr_t3'][:3]}, tau4 llr {z['llr_t4']}")
code2 = T4.run_rung(ROOT, LDIR, EDIR, os.path.join(T, "o2"), os.path.join(T, "w"), REL, NAME, log, check_shots=3, first=3)
z2 = np.load(os.path.join(T, "o2", "tau4_rung.npz"))
j2 = json.load(open(os.path.join(T, "o2", "tau4_rung.json")))
check(code2 == 0 and np.array_equal(z2["llr_t4"], z["llr_t4"]) and abs(j2["tau4"]["cpu_seconds_tau4"] - j["tau4"]["cpu_seconds_tau4"]) < 1e-9,
      "rerun: every chunk reused, same arrays")
bad = os.path.join(T, "bad_extra"); os.makedirs(bad)
shutil.copy(os.path.join(EDIR, "extra_rung.json"), bad)
arr = {k: e[k] for k in e.files}
arr["llr_t3"] = arr["llr_t3"].copy(); arr["llr_t3"][1] += np.float32(1e-3)
np.savez_compressed(os.path.join(bad, "extra_rung.npz"), **arr)
code3 = T4.run_rung(ROOT, LDIR, bad, os.path.join(T, "o3"), os.path.join(T, "w3"), REL, NAME, log, check_shots=3, first=3)
j3 = json.load(open(os.path.join(T, "o3", "tau4_rung.json")))
check(code3 == 4 and j3["verdict"].startswith("not run") and not os.path.exists(os.path.join(T, "o3", "tau4_rung.npz")),
      f"a changed stored tau3: exit {code3}, verdict {j3['verdict']!r}, no npz")
print(f"    CPU of this test: {time.process_time() - c0:.1f} s")
print(f"{sum(ok)} of {len(ok)} checks pass")
sys.exit(0 if all(ok) else 1)
