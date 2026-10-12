"""Real-profile resource check on simulated leaves (fake data; nothing here is Willow data).
Generates d3 q4_5 X r50 and d5 q4_7 X r10 with 50,000 shots each, then runs the real-profile jobs the queue would run,
each under the queue's limits (RLIMIT_AS 3.5 GB, nice, one thread, CPU 0), and records wall time, CPU time and peak RSS.
usage: python3 -I -B tests/scale_check.py <workdir> [d3_main,d3_subset,fit,d5_ladder]   (results: <workdir>/scale_results.json)"""
import json, os, resource, subprocess, sys, time
C = "/mnt/project-files/tensor-networks/deep-dives/experimental/code/decoder"
sys.path.insert(0, os.path.join(C, "tests")); sys.path.insert(0, C)
import numpy as np
import make_fake_willow3 as F

wd = sys.argv[1]
src = os.path.join(wd, "src"); out = os.path.join(wd, "out"); work = os.path.join(wd, "work")
for p in (src, out, work):
    os.makedirs(p, exist_ok=True)
leaves = [(3, "q4_5", "X", 50), (5, "q4_7", "X", 10)]
rng = np.random.default_rng(7)
for k, (d, patch, basis, r) in enumerate(leaves):
    f = os.path.join(src, F.TOP, f"d{d}_at_{patch}", basis, f"r{r:02d}")
    if not os.path.exists(os.path.join(f, "metadata.json")):
        t0 = time.time(); F.leaf(src, d, patch, basis, r, 50000, rng, 5000 + 100 * k)
        print(f"generated d{d} r{r} in {time.time() - t0:.0f} s", flush=True)
DEM = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
L3 = f"{F.TOP}/d3_at_q4_5/X/r50"; L5 = f"{F.TOP}/d5_at_q4_7/X/r10"
py = [sys.executable, "-I", "-B"]
jobs = {
    "d3_main": py + [f"{C}/decode_willow.py", "--root", src, "--out", f"{out}/decode", "--work", f"{work}/chunks", "--leaf", L3,
                     "--dem-file", DEM, "--dem-tag", "rl", "--decoders", "ml_exact,corr_mwpm,mwpm", "--shots", "0:50000:1",
                     "--max-shots", "100000000", "--chunk", "5000", "--synd-ref", "10000"],
    "d3_subset": py + [f"{C}/decode_willow.py", "--root", src, "--out", f"{out}/decode", "--work", f"{work}/chunks", "--leaf", L3,
                       "--dem-file", DEM, "--dem-tag", "rl_s5", "--decoders", "tesseract,beliefmatch", "--shots", "0:50000:5",
                       "--max-shots", "100000000", "--chunk", "5000", "--synd-ref", "0"],
    "fit": py + [f"{C}/fit_willow.py", "--root", src, "--out", f"{out}/fit", "--leaf", L3, "--dem-file", DEM, "--dem-tag", "rl",
                 "--time-split", "--min-class-events", "30", "--nfit", "5000", "--shots", "0:50000:1", "--max-shots", "100000000"],
    "d5_ladder": py + [f"{C}/decode_willow.py", "--root", src, "--out", f"{out}/decode", "--work", f"{work}/chunks", "--leaf", L5,
                       "--dem-file", DEM, "--dem-tag", "rl_s5", "--decoders", "corr_mwpm,mwpm,ml_tau2:1e-6/1e-8:0.1",
                       "--shots", "0:50000:5", "--max-shots", "100000000", "--chunk", "500", "--synd-ref", "2000"],
}
which = sys.argv[2].split(",") if len(sys.argv) > 2 else list(jobs)
mem = int(3.5 * 2 ** 30)
def pre():
    os.nice(5); resource.setrlimit(resource.RLIMIT_AS, (mem, mem)); os.sched_setaffinity(0, [0])
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
           NUMBA_CACHE_DIR="/home/claude/work/decoder/numba_cache")
res = {}
rp = os.path.join(wd, "scale_results.json")
if os.path.exists(rp):
    res = json.load(open(rp))
for name in which:
    t0 = time.time()
    with open(os.path.join(wd, f"{name}.log"), "w") as fh:
        p = subprocess.Popen(jobs[name], stdout=fh, stderr=subprocess.STDOUT, env=env, preexec_fn=pre)
        _, status, ru = os.wait4(p.pid, 0)
    res[name] = {"rc": os.waitstatus_to_exitcode(status), "wall_s": round(time.time() - t0, 1),
                 "cpu_s": round(ru.ru_utime + ru.ru_stime, 1), "max_rss_mb": round(ru.ru_maxrss / 1024, 1)}
    print(name, res[name], flush=True)
    json.dump(res, open(rp, "w"), indent=1)
