#!/usr/bin/env python3
"""Post hoc robustness check, prompted by the internal review: are ML(RL)'s decisions outside the ladder's band stable
at d = 5? (notes/decoder/willow_prereg.md section 9 entry 13). Step 2 of 3: the decoding.

Reads the shot lists of nonband_select.py (md5 checked against shot_lists.json) and, per leaf, round 1's ladder summary
and the local extract (run with python3 -I). The decoder is rebuilt as decode_willow.make_decoder builds ml_tau2 and as
k2_extra_rung.py rebuilds it: the DEM file of summary.json (md5 checked), SparseFrontierDecoder at tau1 = 1e-6 with
kmax = decode_willow.KMAX_LADDER, the deeper rungs sharing tau1's compiled program; tau2 = 1e-8, tau3 = tau2 / 10.
Rebuild check, per leaf before anything else: the leaf's first --check-shots selected shots re-decoded at tau1 must give
the stored tau1 llr exactly (float32, bitwise); a leaf that fails is skipped and reported.

Stages, in this order (approved: tau2 on D first):
  1 D_t2           tau2 on every D shot
  2 D_t3_targeted  tau3 on the D shots that tau2 moved: decision changed, |P_tau2 - 1/2| < 0.1, |P_tau2 - P_tau1| >= 0.05,
                   or a NaN posterior at tau2
  3 C_t2_A         tau2 on pass A of C's sample (half, pre-drawn)
  4 C_t2_B         tau2 on pass B
  5 C_t3_targeted  tau3 on the C shots that tau2 moved
  6 D_t3_random    tau3 on the D shots not moved, with u < 0.1 (pre-drawn)
  7 C_t3_random    the same for C
Each stage visits the leaves in order and decodes in batches of --batch shots; results are saved after every batch, so a
restart resumes at the next batch. CPU guard (--cpu-budget-hours, 4.0): before a batch, the CPU used by this check so
far (all runs) plus the batch's projected CPU (this stage's measured CPU per shot-round, else the previous stage's, x4.2
from tau2 to tau3, else 0.12 s per shot-round at tau2) must stay within the budget; otherwise the check stops before the
batch, writes its summary with status "stopped: cpu budget" and exits 3.
Output (<out>): leaves/<k>.npz per leaf (rows of the shot lists: llr/logp/maxk/drop at tau2 and tau3, done flags, the
tau3 reason: 1 targeted, 2 random), state.json (progress, CPU, rebuild checks), summary.json (status "complete" at the
end). Exit 0 complete, 3 stopped by the guard, 4 a leaf failed its rebuild check (the others are done).
  NUMBA_CACHE_DIR=<scratch> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -I -B nonband_check.py --root <extract> \
      --res results/decoder/willow --lists results/decoder/willow/supplementary/nonband_check \
      --out results/decoder/willow/supplementary/nonband_check [--cpu-budget-hours 4.0]
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
CODE = os.environ.get("NONBAND_CODE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/decoder
sys.path.insert(0, CODE)
import numpy as np
import stim
from dem_utils import parse_dem
from ml_sparse import SparseFrontierDecoder
from willow_data import Experiment
from decode_willow import KMAX_LADDER, _ml_out           # frozen; read-only

T1, T2 = 1e-6, 1e-8
T3 = T2 / 10
BAND = 0.1
STAGES = ("D_t2", "D_t3_targeted", "C_t2_A", "C_t2_B", "C_t3_targeted", "D_t3_random", "C_t3_random")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def sig(llr):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-np.asarray(llr, np.float64)))


def targeted(R, ml):
    """Rows that tau2 moved (needs done_t2): decision changed, |P_tau2 - 1/2| < 0.1, |P_tau2 - P_tau1| >= 0.05, or NaN."""
    P2, P1 = sig(R["llr_t2"]), sig(R["llr_t1"])
    with np.errstate(invalid="ignore"):
        return R["done_t2"] & ((R["pred_t2"] != ml) | ~(np.abs(P2 - 0.5) >= BAND) | ~(np.abs(P2 - P1) < 0.05))


def rung_of(stage):
    return "t2" if stage.endswith(("_t2", "_t2_A", "_t2_B")) else "t3"


def projected_rate(rate, stage, rung):
    """CPU seconds per shot-round for the next batch: this stage's measured rate; else the latest stage of the same
    rung; else the latest of the other rung, x4.2 (tau2 -> tau3) or /4.2; else 0.12 (tau2) or 0.5 (tau3)."""
    if stage in rate and rate[stage][1] > 0:
        return rate[stage][0] / rate[stage][1]
    done = [s for s in STAGES if s in rate and rate[s][1] > 0]
    same = [s for s in done if rung_of(s) == rung]
    if same:
        return rate[same[-1]][0] / rate[same[-1]][1]
    if done:
        x = rate[done[-1]][0] / rate[done[-1]][1]
        return x * 4.2 if rung == "t3" else x / 4.2
    return 0.12 if rung == "t2" else 0.5


class Leaf:
    def __init__(self, a, k, rel, rows, S):
        self.a, self.k, self.rel, self.rows = a, k, rel, rows          # rows: indices into the shot lists
        self.path = os.path.join(a.out, "leaves", f"{k:02d}.npz")
        n = len(rows)
        if os.path.exists(self.path):
            z = np.load(self.path)
            self.R = {f: z[f] for f in z.files}
            assert np.array_equal(self.R["row"], rows), f"{rel}: stored rows differ from the shot lists"
        else:
            self.R = {"row": rows, "llr_t1": S["llr_t1"][rows]}
            for t in ("t2", "t3"):
                self.R[f"llr_{t}"] = np.full(n, np.nan, np.float32); self.R[f"logp_{t}"] = np.full(n, np.nan)
                self.R[f"maxk_{t}"] = np.full(n, -1, np.int64); self.R[f"drop_{t}"] = np.full(n, np.nan)
                self.R[f"done_{t}"] = np.zeros(n, bool); self.R[f"pred_{t}"] = np.full(n, 255, np.uint8)
            self.R["t3_reason"] = np.zeros(n, np.int8)
        self.S = S
        self.dec = None

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **self.R)
        os.replace(tmp, self.path)

    def build(self, log):
        if self.dec is not None:
            return
        res = self.a.res
        d = os.path.join(res, "decode", self.rel, "rl_s5")
        summ = json.load(open(os.path.join(d, "summary.json")))
        folder = os.path.join(self.a.root, self.rel)
        dem_path = os.path.join(folder, summ["job"]["dem_file"])
        assert md5_file(dem_path) == summ["dem_md5"], f"{self.rel}: DEM md5 differs from summary.json"
        exp = Experiment(folder)
        det = exp.detection_events()
        obs = exp.obs_flips_actual()
        idx = self.S["idx"][self.rows]
        assert np.array_equal(obs[idx, 0].astype(np.uint8), self.S["obs"][self.rows]), f"{self.rel}: observed flips differ"
        self.det = np.ascontiguousarray(det[idx])
        del det, obs
        dm = parse_dem(stim.DetectorErrorModel.from_file(dem_path))
        D1 = SparseFrontierDecoder(dm, tau=T1, kmax=KMAX_LADDER)

        def deeper(tau):
            D = SparseFrontierDecoder.__new__(SparseFrontierDecoder)
            D.dm, D.tau, D.kmax, D.prog = dm, tau, KMAX_LADDER, D1.prog
            return D
        self.dec = {"t1": D1, "t2": deeper(T2), "t3": deeper(T3)}
        self.rounds = int(summ.get("rounds") or 0)

    def free(self):
        self.dec = None; self.det = None

    def decode(self, rung, sel):
        D = self.dec[rung]
        c0 = time.process_time()
        post, logp = D.class_probs(self.det[sel])
        sec = time.process_time() - c0
        o = _ml_out(post, logp)
        return o, D.last_maxk.copy(), D.last_drop.copy(), sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-budget-hours", type=float, default=4.0)
    ap.add_argument("--check-shots", type=int, default=20)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--leaves", default=None, help="tests only: comma-separated leaf numbers")
    ap.add_argument("--max-rows", type=int, default=None, help="tests only: rows per leaf and stratum")
    a = ap.parse_args()
    me = md5_file(os.path.abspath(__file__))
    meta = json.load(open(os.path.join(a.lists, "shot_lists.json")))
    lp = os.path.join(a.lists, "shot_lists.npz")
    assert md5_file(lp) == meta["shot_lists_npz_md5"], "shot_lists.npz differs from shot_lists.json"
    z = np.load(lp)
    S = {f: z[f] for f in z.files}
    os.makedirs(a.out, exist_ok=True)
    st_p = os.path.join(a.out, "state.json")
    st = json.load(open(st_p)) if os.path.exists(st_p) else {"cpu_s": 0.0, "stages": {}, "rebuild": {}, "runs": []}
    st["runs"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": me,
                       "args": {k: v for k, v in vars(a).items()}})
    budget = a.cpu_budget_hours * 3600.0
    cpu0 = st["cpu_s"] - time.process_time()          # st["cpu_s"] = earlier runs + this process (all its CPU)
    logf = open(os.path.join(a.out, "nonband_check.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    def save_state():
        st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(st_p, st)
    log(f"nonband_check (md5 {me}): {len(S['idx'])} rows, budget {a.cpu_budget_hours} core-hours, used so far "
        f"{st['cpu_s'] / 3600:.3f}")
    want = set(int(x) for x in a.leaves.split(",")) if a.leaves else None
    leaves = []
    for k, rel in enumerate(S["leaves"]):
        if want is not None and k not in want:
            continue
        rows = np.where(S["leaf_id"] == k)[0]
        if a.max_rows:
            rows = np.r_[rows[S["stratum"][rows] == 0][: a.max_rows], rows[S["stratum"][rows] == 1][: a.max_rows]]
        leaves.append(Leaf(a, k, str(rel), rows, S))
    rate = {}                                  # stage -> [cpu_s, shot-rounds]
    for s in STAGES:
        x = st["stages"].get(s) or {}
        if x.get("shot_rounds"):
            rate[s] = [x["cpu_s"], x["shot_rounds"]]
    stopped, failed_rebuild = None, []
    for si, stage in enumerate(STAGES):
        if stopped:
            break
        sd = st["stages"].setdefault(stage, {"cpu_s": 0.0, "shot_rounds": 0, "shots": 0, "done": False})
        if sd.get("done"):
            continue
        rung = rung_of(stage)
        log(f"stage {stage} ({rung})")
        for L in leaves:
            if stopped:
                break
            if st["rebuild"].get(L.rel, {}).get("identical") is False:
                continue
            R, ml = L.R, S["ml"][L.rows]
            strat, cpass, u = S["stratum"][L.rows], S["c_pass"][L.rows], S["u"][L.rows]
            if stage == "D_t2":
                need = (strat == 0) & ~R["done_t2"]
            elif stage == "C_t2_A":
                need = (strat == 1) & (cpass == 0) & ~R["done_t2"]
            elif stage == "C_t2_B":
                need = (strat == 1) & (cpass == 1) & ~R["done_t2"]
            else:
                tg = targeted(R, ml)
                want_s = 0 if stage.startswith("D_") else 1
                if stage.endswith("targeted"):
                    need = (strat == want_s) & tg & ~R["done_t3"]
                else:
                    need = (strat == want_s) & R["done_t2"] & ~tg & (u < 0.1) & ~R["done_t3"]
            pos = np.where(need)[0]
            if len(pos) == 0:
                continue
            L.build(log)
            if L.rel not in st["rebuild"]:
                nc = min(a.check_shots, len(L.rows))
                o, mk, dr, sec = L.decode("t1", np.arange(nc))
                same = bool(np.array_equal(o["llr"], R["llr_t1"][:nc], equal_nan=True))
                st["rebuild"][L.rel] = {"n_checked": int(nc), "identical": same, "cpu_seconds": sec,
                                        "max_abs_llr_diff": float(np.nanmax(np.abs(o["llr"].astype(np.float64)
                                                                                    - R["llr_t1"][:nc]))) if nc else 0.0}
                st["cpu_s"] = cpu0 + time.process_time()
                save_state()
                log(f"  {L.rel}: rebuild check at tau1 on {nc} shots: {'identical' if same else 'FAILED'} ({sec:.1f} CPU s)")
                if not same:
                    failed_rebuild.append(L.rel); L.free(); continue
            for b0 in range(0, len(pos), a.batch):
                sel = pos[b0:b0 + a.batch]
                sr = len(sel) * L.rounds
                per = projected_rate(rate, stage, rung)
                st["cpu_s"] = cpu0 + time.process_time()
                if st["cpu_s"] + per * sr > budget:
                    stopped = (f"cpu budget: {st['cpu_s'] / 3600:.3f} core-hours used, the next batch projects "
                               f"{per * sr / 3600:.3f} more, budget {a.cpu_budget_hours}")
                    break
                o, mk, dr, sec = L.decode(rung, sel)
                R[f"llr_{rung}"][sel] = o["llr"]; R[f"logp_{rung}"][sel] = o["logp"]; R[f"pred_{rung}"][sel] = o["pred"][:, 0]
                R[f"maxk_{rung}"][sel] = mk; R[f"drop_{rung}"][sel] = dr; R[f"done_{rung}"][sel] = True
                if rung == "t3":
                    R["t3_reason"][sel] = 1 if stage.endswith("targeted") else 2
                L.save()
                st["cpu_s"] = cpu0 + time.process_time(); sd["cpu_s"] += sec; sd["shot_rounds"] += sr; sd["shots"] += len(sel)
                rate[stage] = [sd["cpu_s"], sd["shot_rounds"]]
                save_state()
            log(f"  {L.rel}: {len(pos)} shots at {rung}; check CPU so far {st['cpu_s'] / 3600:.3f} core-hours")
            L.free()
        if not stopped:
            sd["done"] = True
            save_state()
    st["cpu_s"] = cpu0 + time.process_time()
    save_state()
    summ = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": me,
            "label": "post hoc robustness check, prompted by the internal review",
            "prereg": "notes/decoder/willow_prereg.md section 9 entry 13",
            "status": ("stopped: " + stopped) if stopped else "complete", "cpu_core_hours": st["cpu_s"] / 3600,
            "budget_core_hours": a.cpu_budget_hours, "stages": st["stages"], "rebuild": st["rebuild"],
            "rebuild_failed": failed_rebuild, "test_only": bool(a.leaves or a.max_rows),
            "shot_lists_npz_md5": meta["shot_lists_npz_md5"]}
    write_json(os.path.join(a.out, "summary.json"), summ)
    log(f"summary: {summ['status']}; {summ['cpu_core_hours']:.3f} core-hours; rebuild failures {failed_rebuild}")
    sys.exit(3 if stopped else (4 if failed_rebuild else 0))


if __name__ == "__main__":
    main()
