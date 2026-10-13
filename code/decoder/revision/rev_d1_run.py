#!/usr/bin/env python3
"""D1 (review 1.3), step 2: decoding for the independent check of the d = 5 ML(RL) reference (decoder, 2026-10-13).
Post hoc revision analysis. Reads only the selected shots' detection events (rev_d1_select.py) and the DEM files they
name (md5 checked); it never reads a stored decision or an observed outcome. One thread; resumable.

  mps     boundary MPS (ml_mps_fast.FastMPSDecoder, reweighted, cutoff 1e-14, the frozen module) on a chi ladder. Per shot
          the ladder runs --chis in order and stops after rung k >= 1 when
              the decision equals rung k-1's and |P_k - P_(k-1)| <= --stop-frac x |P_k - 1/2|   (status "resolved"),
          or when |P_k - P_(k-1)| < 1e-4, or at the top chi (status "unresolved" unless the first condition holds).
          With --all-chis every rung runs (the timing probe).
  sparse  the frozen sparse-frontier decoder, rebuilt as decode_willow builds the ladder (tau1 = 1e-6, kmax =
          KMAX_LADDER, deeper rungs sharing tau1's compiled program), at every (name, tau, kmax) of --sparse-runs.
Each worker takes the rows with worker == --worker (sample) or row mod --workers (probe), in processing order, and
appends one JSON line per shot to <out>/<mode>_w<k>.jsonl (flushed and fsynced). A restart skips the rows already there.
CPU guard: before a shot, this worker's CPU so far (all runs, from <out>/<mode>_w<k>.state.json) plus the shot's
projected CPU (--proj-s-per-round x rounds) must stay within --cpu-cap-hours; otherwise it writes status "stopped: cpu
cap" and exits 3. Exit 0 when every row of the worker is done.
  NUMBA_CACHE_DIR=<scratch> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 10 python3 -I -B rev_d1_run.py mps \
      --root <extract> --sel <dir> [--probe] --out <dir> --worker 0 --workers 2 --cpu-cap-hours H --chis 8,16,32,64 \
      --stop-frac 0.5 --proj-s-per-round S
"""
import argparse, hashlib, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("D1_CODE") or os.path.dirname(HERE)          # code/decoder
sys.path.insert(0, CODE)
import numpy as np
import stim
from dem_utils import parse_dem                                    # frozen; read-only
from ml_mps_fast import FastMPSDecoder                             # frozen; read-only
from ml_sparse import SparseFrontierDecoder                        # frozen; read-only
from decode_willow import KMAX_LADDER, _ml_out                     # frozen; read-only

T1 = 1e-6


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def append_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("mps", "sparse"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sel", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu-cap-hours", type=float, required=True)
    ap.add_argument("--chis", default="8,16,32,64")
    ap.add_argument("--stop-frac", type=float, default=0.5)
    ap.add_argument("--all-chis", action="store_true")
    ap.add_argument("--sparse-runs", default="t1:1e-6:2000000,t2:1e-8:2000000,t2_k25000:1e-8:25000,t2_k6250:1e-8:6250,"
                                             "t3:1e-9:2000000")
    ap.add_argument("--proj-s-per-round", type=float, required=True)
    a = ap.parse_args()
    name = "probe" if a.probe else "sample"
    meta = json.load(open(os.path.join(a.sel, name + ".json")))
    zp = os.path.join(a.sel, name + ".npz")
    assert md5(zp) == meta[name + "_npz_md5"], "selection file changed"
    Z = np.load(zp)
    det, ndet, lid, rounds = Z["det"], Z["ndet"], Z["leaf_id"], Z["rounds"]
    n = len(lid)
    if a.probe:
        mine = [i for i in range(n) if i % a.workers == a.worker]
        leaves = meta["leaves"]
    else:
        order = Z["order"]; wk = Z["worker"]
        mine = [int(i) for i in np.argsort(order) if wk[i] == a.worker]
        leaves = meta["leaves"]
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.mode}{'_probe' if a.probe else ''}_w{a.worker}"
    lines_p = os.path.join(a.out, tag + ".jsonl"); state_p = os.path.join(a.out, tag + ".state.json")
    done = set()
    last_total = 0.0
    if os.path.exists(lines_p):
        for ln in open(lines_p):
            ln = ln.strip()
            if ln:
                try:
                    x = json.loads(ln)
                    done.add(int(x["row"])); last_total = max(last_total, float(x["cpu_total_s"]))
                except (ValueError, KeyError):
                    pass                                            # a torn last line from a kill: redo that row
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"cpu_prev_runs_s": 0.0, "runs": []}
    run = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": md5(os.path.abspath(__file__)),
           "args": vars(a), "selection_md5": meta[name + "_npz_md5"]}
    st["runs"].append(run)
    # CPU of earlier runs: the largest of the last clean exit, the last periodic save and the last result line (a killed
    # run loses at most the CPU after its last line)
    base_cpu = max(float(st.get("cpu_prev_runs_s", 0.0)), float(st.get("cpu_prev_runs_s_at_last_save", 0.0)), last_total)
    cap_s = a.cpu_cap_hours * 3600.0
    chis = [int(x) for x in a.chis.split(",")]
    sruns = []
    for item in a.sparse_runs.split(","):
        nm, tau, km = item.split(":")
        sruns.append((nm, float(tau), int(km)))
    decs = {}

    def build(k):
        if k in decs:
            return decs[k]
        L = leaves[k]
        dem_path = os.path.join(a.root, L["dem"])
        assert md5(dem_path) == L["dem_md5"], f"{L['leaf']}: DEM md5 differs"
        c0 = time.process_time()
        dm = parse_dem(stim.DetectorErrorModel.from_file(dem_path))
        if a.mode == "mps":
            D = FastMPSDecoder(dm, chi=chis[0], cutoff=1e-14, reweight=True)
        else:
            D1 = SparseFrontierDecoder(dm, tau=T1, kmax=KMAX_LADDER)
            D = {}
            for nm, tau, km in sruns:
                Dx = SparseFrontierDecoder.__new__(SparseFrontierDecoder)
                Dx.dm, Dx.tau, Dx.kmax, Dx.prog = dm, tau, km, D1.prog
                D[nm] = Dx
        decs[k] = (D, dm.num_detectors, time.process_time() - c0)
        return decs[k]

    def save_state(status):
        st["cpu_prev_runs_s_at_last_save"] = base_cpu + time.process_time()
        st["status"] = status; st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["rows_done"] = len(done); st["rows_mine"] = len(mine)
        write_json(state_p, st)

    status = "complete"
    for i in mine:
        if i in done:
            continue
        r = int(rounds[i])
        used = base_cpu + time.process_time()
        if used + a.proj_s_per_round * r > cap_s:
            status = "stopped: cpu cap"
            break
        D, nd, build_s = build(int(lid[i]))
        assert int(ndet[i]) == nd, "detector count differs from the DEM"
        syn = np.ascontiguousarray(det[i:i + 1, :nd])
        rec = {"row": int(i), "leaf_id": int(lid[i]), "rounds": r, "build_cpu_s": build_s}
        if a.mode == "mps":
            P, LP, MB, CPU, CH = [], [], [], [], []
            resolved = False
            for j, chi in enumerate(chis):
                D.chi = chi
                c0 = time.process_time()
                post, logp = D.class_probs(syn)
                CPU.append(time.process_time() - c0)
                p1 = float(post[0, 1]); P.append(p1); LP.append(float(logp[0])); MB.append(int(D.last_maxbond[0])); CH.append(chi)
                if j >= 1 and not a.all_chis:
                    p0 = P[-2]
                    if math.isfinite(p1) and math.isfinite(p0):
                        d1, d0 = p1 > 0.5, p0 > 0.5
                        dp = abs(p1 - p0)
                        resolved = (d1 == d0) and dp <= a.stop_frac * abs(p1 - 0.5)
                        if resolved or dp < 1e-4:
                            break
            if a.all_chis and len(P) >= 2 and math.isfinite(P[-1]) and math.isfinite(P[-2]):
                resolved = ((P[-1] > 0.5) == (P[-2] > 0.5)) and abs(P[-1] - P[-2]) <= a.stop_frac * abs(P[-1] - 0.5)
            rec.update(chis=CH, P1=P, logp=LP, maxbond=MB, cpu_s=CPU, resolved=bool(resolved),
                       pred=int(np.nan_to_num(P[-1], nan=0.0) > 0.5))
        else:
            out = {}
            for nm, tau, km in sruns:
                c0 = time.process_time()
                post, logp = D[nm].class_probs(syn)
                sec = time.process_time() - c0
                o = _ml_out(post, logp)
                out[nm] = {"llr": float(o["llr"][0]), "logp": float(o["logp"][0]), "pred": int(o["pred"][0, 0]),
                           "maxk": int(D[nm].last_maxk[0]), "drop": float(D[nm].last_drop[0]), "cpu_s": sec,
                           "tau": tau, "kmax": km}
            rec["runs"] = out
        rec["cpu_total_s"] = base_cpu + time.process_time()
        append_line(lines_p, rec)
        done.add(i)
        if len(done) % 5 == 0:
            save_state("running")
    st["cpu_prev_runs_s"] = base_cpu + time.process_time()
    save_state(status)
    print(json.dumps({"tag": tag, "status": status, "rows_done": len(done), "rows_mine": len(mine),
                      "cpu_h": st["cpu_prev_runs_s"] / 3600}))
    sys.exit(0 if status == "complete" else 3)


if __name__ == "__main__":
    main()
