#!/usr/bin/env python3
"""PR3's main run (notes/decoder/willow_prereg_r3.md section 8): schedules the d = 7 decoding on the free cores.

Units. Each of the 6 leaves (d7_at_q6_7, X and Z, r = 10, 30, 50) is decoded as 5 parts of 10,000 consecutive shot
indices with the design's step (pr3_common.part_slice):
  - ladder part: the frozen decode_willow.py, unchanged (--decoders corr_mwpm,mwpm,ml_tau2:1e-6/1e-7:0.1, the RL DEM
    file, --chunk 250, --synd-ref 0), output <res>/parts/p<j>/decode/<leaf>/<tag>/;
  - extra part: pr3_extra_rung.py (k2_extra_rung.extra_rung, unchanged; tau3 = 1e-8 on the part's band shots after a
    tau2 rebuild check on 10 of them), output <res>/parts/p<j>/extra_rung/<leaf>/<tag>/. Extra parts run only for a
    leaf that fails rule 1 or rule 2' between tau1 and tau2.
Verdicts are per leaf over its 5 parts, on the PR3 sample (the r = 10 probe shots, index divisible by 25, are left
out): rule 1, |k_new - k_old| <= max(3, 1% of k_new); rule 2', decisions changed <= max(3, 1% of k_new); a NaN
posterior at tau3 fails. A leaf that fails after its extra rung is excluded and flagged; 2 or more such leaves kill
d = 7 ML (no new unit starts after that). The largest |dP| is reported for each rung, as a description.

Slots. At most --max-heavy (4) heavy processes run on the machine, whoever started them. Heavy: a process whose command
line names run_mech.py, k2_extra_rung.py, decode_willow.py, pr3_extra_rung.py, run_insample.py, final_round1.py,
analyze_willow.py, analyze_explore.py, nonband_check.py, nonband_analyze.py or reservoir's mech-check scripts
(r2m_check.py, r2m_redecode.py, r2m_localmax.py), or any process that used at least half a core over the last 60 s.
Slots held for others:
  - K2: while k2_runner.py is alive, its unfinished leaves hold up to its --parallel (2) slots;
  - round 2's in-sample decodes: run_insample.py --part i/2 decodes the series of run_mech.py --part i/2, so from the
    end of mech part i, 1 slot is held for in-sample part i until it has been seen running (an in-sample part seen
    while its mech part runs does not count; --part 0/1 counts for both);
  - after the mech fits (no run_mech.py alive; nothing seen before then counts): 1 slot for reservoir's mech check,
    until its last step (r2m_localmax.py) has been seen and every r2m_ script has been gone for 10 minutes, or any r2m_
    script has been seen and they have all been gone for 60 minutes;
  - once K2's tally is complete: 1 slot for round 1's final analysis (final_round1.py), until its final_round1.json
    exists;
  - after the mech fits, once its shot lists exist: 1 slot for the post hoc non-band check (nonband_check.py, run by
    post_mech_chain.py after analyze_explore.py), until it has been seen and then gone for 10 minutes, or its summary
    is final ("complete" or "stopped: ...").
<state>/control.json can release a hold ("insample": "released", "reservoir": "released", "final_round1":
"released", "nonband": "released"), hold more slots ("hold": n) or pause the runner ("pause": true).
A unit starts only if MemAvailable stays at least --min-free-gib (3) above 1.5 x the pilot's peak RSS for its stage.
Every unit runs in its own session under an RSS cap (--cap-gib, 4): above it, its process group is killed.

Guard. CPU is counted per unit (wait4 for children; /proc for a unit found running after a restart). The projection is
the CPU used plus, for every unit not finished, its projected cost (the measured mean CPU of finished units of the same
kind and r, else the pilot's rates; a running unit counts at least its CPU so far + 60 s; extra parts are counted for
every leaf whose ladder has not been judged). If the projection passes --guard-core-hours, no new unit starts, the
running ones finish, and the runner exits 6. Other exits: 0 all done (or killed: see pr3_tally.json), 5 a unit failed,
7 a unit hit the memory cap.

Restartable: finished units are found from their outputs; a unit process still running from an earlier runner keeps
its slot and is not started again.
  NUMBA_CACHE_DIR=<scratch> python3 -I -B pr3_runner.py --root <extract> --res results/decoder/willow/pr3/main \
      --work <scratch>/work --state <scratch>/state --logs <scratch>/logs --pilot results/decoder/willow/pr3/pilot/pilot.json \
      --design 25k --guard-core-hours <1.3 x projection> [--dry-run]
"""
import argparse, json, math, os, signal, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pr3_common as C

CODE = os.environ.get("PR3_CODE") or os.path.dirname(HERE)
LADDER_DN = C.LADDER.replace(":", "_").replace("/", "-")
RESERVOIR_CHECK = (b"r2m_check.py", b"r2m_redecode.py", b"r2m_localmax.py")   # reservoir's mech check (run_r2m.sh)
HEAVY_NAMES = (b"run_mech.py", b"k2_extra_rung.py", b"decode_willow.py", b"pr3_extra_rung.py", b"run_insample.py",
               b"final_round1.py", b"analyze_willow.py", b"analyze_explore.py", b"nonband_check.py",
               b"nonband_analyze.py") + RESERVOIR_CHECK
HOLDS_VERSION = 4
NONBAND = "results/decoder/willow/supplementary/nonband_check"     # willow_prereg.md section 9 entry 13
K2_RES = "results/decoder/willow/supplementary/k2_extra_rung"
FINAL_R1 = "results/decoder/willow/final_round1/final_round1.json"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().split(b"\0")
    except OSError:
        return []


def sigmoid(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1 / (1 + np.exp(-x.astype(np.float64)))


class Procs:
    """Every process's CPU over a sliding window, for the heavy count."""
    def __init__(self):
        self.hist = {}

    def scan(self, me):
        t = time.time(); seen = {}
        for p in os.listdir("/proc"):
            if not p.isdigit() or int(p) in me:
                continue
            s = C._stat(int(p))
            if s is None:
                continue
            seen[int(p)] = s
            h = self.hist.setdefault(int(p), [])
            h.append((t, s["cpu"]))
            while len(h) > 2 and t - h[1][0] >= 60:
                h.pop(0)
        for p in list(self.hist):
            if p not in seen:
                del self.hist[p]
        out = []
        for p, s in seen.items():
            cl = cmdline(p)
            if not cl or not cl[0]:
                continue
            named = any(c.endswith(n) for c in cl for n in HEAVY_NAMES)
            h = self.hist[p]
            busy = len(h) >= 2 and h[-1][0] - h[0][0] >= 30 and (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]) >= 0.5
            if named or busy:
                out.append({"pid": p, "pgid": s["pgid"], "cpu": s["cpu"], "rss": s["rss"], "cmd": cl})
        return out


def has(cl, *frags):
    return any(any(f in c for f in frags) for c in cl)


def arg_of(cl, flag):
    for i, c in enumerate(cl[:-1]):
        if c == flag:
            return cl[i + 1].decode(errors="replace")
    return None


class Runner:
    def __init__(self, a):
        self.a = a
        self.tag = f"rl_{a.design}"
        self.pilot = json.load(open(a.pilot))
        self.m = self.pilot["measured_rates"]
        st1 = self.pilot["stage1_ladder"]["peak_rss"]; st2 = (self.pilot.get("stage2_extra") or {}).get("peak_rss", st1)
        self.need = {"ladder": max(1.5 * st1, C.GiB), "extra": max(1.5 * st2, C.GiB)}
        self.units = {}
        for b in C.BASES:
            for r in C.ROUNDS:
                lf = C.leaf(b, r)
                for j in range(C.PARTS):
                    self.units[("ladder", lf, j)] = {"kind": "ladder", "leaf": lf, "r": r, "part": j, "state": "pending"}
                    self.units[("extra", lf, j)] = {"kind": "extra", "leaf": lf, "r": r, "part": j, "state": "pending"}
        os.makedirs(a.state, exist_ok=True); os.makedirs(a.logs, exist_ok=True); os.makedirs(a.work, exist_ok=True)
        self.ledger_p = os.path.join(a.state, "ledger.json")
        self.ledger = json.load(open(self.ledger_p)) if os.path.exists(self.ledger_p) else {
            "made": now(), "design": a.design, "units": {},
            "holds": {"insample": {"state": "held", "last_seen": None}, "reservoir": {"state": "held", "last_seen": None}}}
        if self.ledger.get("design") != a.design:
            sys.exit(f"the ledger is for design {self.ledger.get('design')}, not {a.design}")
        if self.ledger.get("holds_version") != HOLDS_VERSION:
            # versions 1-2 counted processes seen while the mech fits ran and matched any command line naming
            # reservoir (their reservoir state is void); version 4 tracks the in-sample decodes per part. None of these
            # runs had started when version 4 was deployed.
            old = {k: self.ledger["holds"].get(k) for k in ("insample", "reservoir")}
            self.ledger["holds"]["insample"] = {"state": "held", "last_seen": None, "parts_seen": []}
            if self.ledger.get("holds_version") is None:
                self.ledger["holds"]["reservoir"] = {"state": "held", "last_seen": None}
            self.ledger["holds_version"] = HOLDS_VERSION
            self.ledger.setdefault("notes", []).append(f"{now()} holds moved to version {HOLDS_VERSION}; were {old}")
        self.kids = {}            # pid -> unit key
        self.procs = Procs()
        self.verdicts, self._vkey, self._band = {}, {}, {}

    # ------------------------------------------------------------------ paths and commands
    def part_dir(self, j):
        return os.path.join(self.a.res, "parts", f"p{j}")

    def ladder_dir(self, lf, j):
        return os.path.join(self.part_dir(j), "decode", lf, self.tag)

    def extra_json(self, lf, j):
        return os.path.join(self.part_dir(j), "extra_rung", lf, self.tag, "extra_rung.json")

    def cmd(self, u):
        py = [sys.executable, "-I", "-B"]
        if u["kind"] == "ladder":
            return py + [os.path.join(CODE, "decode_willow.py"), "--root", self.a.root,
                         "--out", os.path.join(self.part_dir(u["part"]), "decode"),
                         "--work", os.path.join(self.a.work, f"p{u['part']}"), "--leaf", u["leaf"],
                         "--dem-file", C.DEM_FILE, "--dem-tag", self.tag, "--decoders", C.DECODERS,
                         "--shots", C.part_slice(self.a.design, u["part"]), "--chunk", "250", "--synd-ref", "0"]
        return py + [os.path.join(HERE, "pr3_extra_rung.py"), "--root", self.a.root, "--res", self.part_dir(u["part"]),
                     "--out", os.path.join(self.part_dir(u["part"]), "extra_rung"), "--leaf", u["leaf"],
                     "--tag", self.tag, "--check-shots", str(C.CHECK_SHOTS_PER_PART)]

    def ukey(self, u):
        return f"{u['kind']}|{u['leaf']}|{u['part']}"

    # ------------------------------------------------------------------ state from the outputs
    def ladder_done(self, lf, j):
        try:
            s = json.load(open(os.path.join(self.ladder_dir(lf, j), "summary.json")))
        except Exception:
            return False
        return (s.get("status") == "complete" and C.LADDER in (s.get("decoders") or {})
                and s.get("selection") == C.part_slice(self.a.design, j))

    def extra_done(self, lf, j):
        try:
            e = json.load(open(self.extra_json(lf, j)))
        except Exception:
            return False
        return "tau3" in e and e.get("ladder_npz_md5") == md5(os.path.join(self.ladder_dir(lf, j), LADDER_DN + ".npz"))

    def refresh(self):
        for k, u in self.units.items():
            if u["state"] in ("running", "orphan"):
                continue
            done = self.ladder_done(u["leaf"], u["part"]) if u["kind"] == "ladder" else self.extra_done(u["leaf"], u["part"])
            if done:
                u["state"] = "done"
        self.judge()

    # ------------------------------------------------------------------ verdicts on the PR3 sample
    def load_ladder(self, lf, r):
        idx, obs, pred, llr, llr1, red = [], [], [], [], [], []
        for j in range(C.PARTS):
            d = self.ladder_dir(lf, j)
            z = np.load(os.path.join(d, LADDER_DN + ".npz")); sh = np.load(os.path.join(d, "shots.npz"))
            assert np.array_equal(z["idx"], sh["idx"]), f"{lf} p{j}: ladder and shots.npz select different shots"
            idx.append(z["idx"].astype(np.int64)); obs.append(sh["obs"].astype(np.uint8)); pred.append(z["pred"][:, 0])
            llr.append(z["llr"]); llr1.append(z["llr_t1"]); red.append(z["redecoded"].astype(bool))
        cat = lambda v: np.concatenate(v)
        return cat(idx), cat(obs), cat(pred).astype(np.uint8), cat(llr), cat(llr1), cat(red)

    def judge(self):
        for b in C.BASES:
            for r in C.ROUNDS:
                lf = C.leaf(b, r)
                key = tuple(self.units[(k, lf, j)]["state"] for k in ("ladder", "extra") for j in range(C.PARTS))
                if self._vkey.get(lf) == key and lf in self.verdicts:
                    continue
                self._vkey[lf] = key
                if not all(self.units[("ladder", lf, j)]["state"] == "done" for j in range(C.PARTS)):
                    self.verdicts[lf] = {"verdict": "ladder running or pending"}; continue
                idx, obs, df, llr, llr1, red = self.load_ladder(lf, r)
                m = C.in_sample(idx, r)
                d1 = (np.nan_to_num(sigmoid(llr1), nan=0.0) > 0.5).astype(np.uint8)
                k1, kf = int(np.sum((d1 != obs) & m)), int(np.sum((df != obs) & m))
                ch = int(np.sum((d1 != df) & m))
                dP = np.abs(sigmoid(llr) - sigmoid(llr1))[red & m]
                lim = max(3, 0.01 * kf)
                v = {"n_sample": int(m.sum()), "n_decoded": int(len(idx)), "band_shots_sample": int((red & m).sum()),
                     "ladder": {"errors_tau1": k1, "errors_tau2": kf, "rule1": abs(kf - k1) <= lim,
                                "decisions_changed": ch, "rule2prime": ch <= lim, "limit": lim,
                                "max_abs_dP_descriptive": float(np.nanmax(dP)) if dP.size else 0.0,
                                "nan_tau2": int(np.isnan(llr[m]).sum())}}
                v["ladder"]["passes"] = v["ladder"]["rule1"] and v["ladder"]["rule2prime"]
                if v["ladder"]["passes"]:
                    v["verdict"] = "pass (ladder)"; self.verdicts[lf] = v; continue
                if not all(self.units[("extra", lf, j)]["state"] == "done" for j in range(C.PARTS)):
                    v["verdict"] = "fails the ladder rules; extra rung pending"; self.verdicts[lf] = v; continue
                pos = {int(x): i for i, x in enumerate(idx)}
                bi, l3 = [], []
                for j in range(C.PARTS):
                    e = np.load(os.path.join(os.path.dirname(self.extra_json(lf, j)), "extra_rung.npz"))
                    bi.extend(pos[int(x)] for x in e["idx"]); l3.append(e["llr_t3"])
                bi = np.array(bi, np.int64); l3 = np.concatenate(l3) if l3 else np.zeros(0, np.float32)
                assert np.array_equal(np.sort(bi), np.where(red)[0]), f"{lf}: extra-rung shots differ from the band"
                keep = m[bi]
                d3 = (np.nan_to_num(sigmoid(l3), nan=0.0) > 0.5).astype(np.uint8)
                bk = bi[keep]
                k3 = kf - int(np.sum(df[bk] != obs[bk])) + int(np.sum(d3[keep] != obs[bk]))
                ch3 = int(np.sum(d3[keep] != df[bk])); nan3 = int(np.isnan(l3[keep]).sum())
                lim3 = max(3, 0.01 * k3)
                dP3 = np.abs(sigmoid(l3) - sigmoid(llr[bi]))[keep]
                v["extra"] = {"errors_tau2": kf, "errors_tau3": k3, "rule1": abs(k3 - kf) <= lim3,
                              "decisions_changed": ch3, "rule2prime": ch3 <= lim3 and nan3 == 0, "limit": lim3,
                              "nan_tau3": nan3, "max_abs_dP_descriptive": float(np.nanmax(dP3)) if dP3.size and nan3 < dP3.size else None}
                v["extra"]["passes"] = v["extra"]["rule1"] and v["extra"]["rule2prime"]
                v["verdict"] = "pass (extra rung)" if v["extra"]["passes"] else "still fails: excluded and flagged"
                self.verdicts[lf] = v

    def killed(self):
        return sum(1 for v in self.verdicts.values() if v["verdict"].startswith("still fails")) >= 2

    def needs_extra(self, lf):
        v = self.verdicts.get(lf, {})
        return v.get("verdict", "").startswith("fails the ladder rules")

    # ------------------------------------------------------------------ cost
    def rates(self):
        r_ = {"c1": {}, "s": {}, "c2": {}}
        m = self.m
        for v in ("c1", "s", "c2"):
            r_[v] = {10: m[f"{v}_10"], 50: m[f"{v}_50"]}; r_[v][30] = (r_[v][10] + r_[v][50]) / 2
        q = m["c3_50"] / m["c2_50"]
        r_["c3"] = {10: r_["c2"][10] * q, 30: r_["c2"][30] * q, 50: m["c3_50"]}
        return r_

    def band_of(self, u):
        k = (u["leaf"], u["part"])
        if k in self._band:
            return self._band[k]
        if self.units[("ladder",) + k]["state"] != "done":
            return None
        try:
            z = np.load(os.path.join(self.ladder_dir(u["leaf"], u["part"]), LADDER_DN + ".npz"))
            self._band[k] = int(z["redecoded"].sum())
            return self._band[k]
        except Exception:
            return None

    def projected_s(self, u):
        L = self.ledger["units"]
        same = [x for k, x in L.items() if x.get("state") == "done" and x.get("kind") == u["kind"] and x.get("r") == u["r"]
                and x.get("cpu_s")]
        R = self.rates(); r = u["r"]; n = C.DESIGNS[self.a.design]["n"] // C.PARTS
        if u["kind"] == "ladder":
            if same:
                return float(np.mean([x["cpu_s"] for x in same]))
            return (n * r * R["c1"][r] + n * R["s"][r] * r * R["c2"][r] + n * self.m["cb_50"] * r / 50) / 1e3 \
                + self.m["ovh_ladder_s"]
        band = self.band_of(u)
        if band is None:
            band = n * R["s"][r]
        per = [x["cpu_s"] / max(x.get("band", 1), 1) for x in same if x.get("band")]
        if per:
            return band * float(np.mean(per))
        return (band * r * R["c3"][r] + min(band, C.CHECK_SHOTS_PER_PART) * r * R["c2"][r]) / 1e3 + self.m["ovh_extra_s"]

    def projection(self):
        used = run = pend = 0.0
        for k, u in self.units.items():
            rec = self.ledger["units"].get(self.ukey(u), {})
            if u["state"] == "done":
                used += rec.get("cpu_s", 0.0) + rec.get("lost_cpu_s", 0.0)
            elif u["state"] in ("running", "orphan"):
                run += max(self.projected_s(u), rec.get("cpu_now", 0.0) + 60) + rec.get("lost_cpu_s", 0.0)
            elif u["kind"] == "ladder" or self.needs_extra(u["leaf"]) or \
                    self.verdicts.get(u["leaf"], {}).get("verdict") == "ladder running or pending":
                pend += self.projected_s(u) + rec.get("lost_cpu_s", 0.0)
        return {"used_h": used / 3600, "running_h": run / 3600, "pending_h": pend / 3600, "total_h": (used + run + pend) / 3600}

    # ------------------------------------------------------------------ slots
    def k2_hold(self, heavy):
        alive = any(has(p["cmd"], b"k2_runner.py") for p in self.procs_all)
        if not alive:
            return 0
        root = os.path.join(self.a.project, K2_RES)
        done = 0
        for dp, dn, fn in os.walk(root):
            if "extra_rung.json" in fn:
                done += 1
        unfinished = max(0, 32 - done)
        running = sum(1 for p in heavy if has(p["cmd"], b"k2_extra_rung.py"))
        return max(0, min(2, unfinished) - running)

    def holds(self, heavy):
        ctl = {}
        try:
            ctl = json.load(open(os.path.join(self.a.state, "control.json")))
        except Exception:
            pass
        H = self.ledger["holds"]
        mech = any(has(p["cmd"], b"run_mech.py") for p in heavy)
        out = {"k2": self.k2_hold(heavy)}
        # round 1's final analysis: 1 slot once K2's tally is complete, until its output exists
        H.setdefault("final_round1", {"state": "held", "last_seen": None})
        if ctl.get("final_round1") == "released" or os.path.exists(os.path.join(self.a.project, FINAL_R1)):
            if H["final_round1"]["state"] != "released":
                self.log("hold for final_round1 released")
            H["final_round1"]["state"] = "released"
        out["final_round1"] = 0
        if H["final_round1"]["state"] != "released":
            try:
                k2c = json.load(open(os.path.join(self.a.project, K2_RES, "k2_tally.json"))).get("complete")
            except Exception:
                k2c = False
            if k2c and not any(has(p["cmd"], b"final_round1.py") for p in heavy):
                out["final_round1"] = 1
        out["manual"] = max(0, int(ctl.get("hold", 0) or 0))
        t = time.time()
        for name in ("insample", "reservoir"):
            if ctl.get(name) == "released" and H[name]["state"] != "released":
                H[name]["state"] = "released"; self.log(f"hold for {name} released by control.json")
        # in-sample decodes: part i is held from the end of mech part i until it has been seen
        h = H["insample"]; h.setdefault("parts_seen", [])
        mech_parts = {arg_of(p["cmd"], b"--part") for p in heavy if has(p["cmd"], b"run_mech.py")}
        ins = [p for p in heavy if has(p["cmd"], b"run_insample.py")]
        if h["state"] != "released":
            for p in ins:
                part = arg_of(p["cmd"], b"--part")
                if part is None and arg_of(p["cmd"], b"--series") is not None:
                    continue
                for c in (("0/2", "1/2") if part in (None, "0/1") else (part,)):
                    if c in ("0/2", "1/2") and c not in mech_parts and c not in h["parts_seen"]:
                        h["parts_seen"].append(c); self.log(f"in-sample part {c} seen")
            if ins:
                h["last_seen"] = t
            if set(h["parts_seen"]) >= {"0/2", "1/2"} and not ins and t - (h["last_seen"] or t) > 120:
                h["state"] = "released"; self.log("hold for insample released: both parts ran and have been gone for 120 s")
        out["insample"] = 0 if h["state"] == "released" else \
            sum(1 for c in ("0/2", "1/2") if c not in mech_parts and c not in h["parts_seen"])
        # reservoir's mech check: 1 slot once no mech process is alive; nothing seen before then counts
        h = H["reservoir"]
        if h["state"] == "released" or mech:
            out["reservoir"] = 0
        else:
            n = sum(1 for p in heavy if has(p["cmd"], *RESERVOIR_CHECK))
            if n:
                h["last_seen"] = t; h["state"] = "seen"
                if any(has(p["cmd"], b"r2m_localmax.py") for p in heavy):
                    h["last_step_seen"] = True
            elif h["state"] == "seen" and h["last_seen"] is not None:
                quiet = 600 if h.get("last_step_seen") else 3600
                if t - h["last_seen"] > quiet:
                    h["state"] = "released"
                    self.log(f"hold for reservoir released: it ran and has been gone for {quiet} s")
            out["reservoir"] = 0 if h["state"] == "released" else max(0, 1 - n)
        # the post hoc non-band check: 1 slot once no mech process is alive and its shot lists exist
        h = H.setdefault("nonband", {"state": "held", "last_seen": None})
        if ctl.get("nonband") == "released" and h["state"] != "released":
            h["state"] = "released"; self.log("hold for nonband released by control.json")
        if h["state"] != "released":
            try:
                ns = json.load(open(os.path.join(self.a.project, NONBAND, "summary.json")))
                fin = not ns.get("test_only") and str(ns.get("status", "")).startswith(("complete", "stopped"))
            except Exception:
                fin = False
            if fin:
                h["state"] = "released"; self.log("hold for nonband released: the check's summary is final")
        armed = os.path.exists(os.path.join(self.a.project, NONBAND, "shot_lists.json"))
        if h["state"] == "released" or mech or not armed:
            out["nonband"] = 0
        else:
            n = sum(1 for p in heavy if has(p["cmd"], b"nonband_check.py"))
            if n:
                h["last_seen"] = t; h["state"] = "seen"
            elif h["state"] == "seen" and h["last_seen"] is not None and t - h["last_seen"] > 600:
                h["state"] = "released"; self.log("hold for nonband released: it ran and has been gone for 600 s")
            out["nonband"] = 0 if h["state"] == "released" else max(0, 1 - n)
        return out, bool(ctl.get("pause"))

    # ------------------------------------------------------------------ main loop
    def log(self, msg):
        print(f"{now()} {msg}", flush=True)

    def save(self):
        self.ledger["updated"] = now()
        self.ledger["verdicts"] = self.verdicts
        self.ledger["projection"] = self.projection()
        C.write_json(self.ledger_p, self.ledger)

    def tally(self, status):
        fails = [lf for lf, v in self.verdicts.items() if v["verdict"].startswith("still fails")]
        t = {"made": now(), "design": self.a.design, "status": status, "verdicts": self.verdicts,
             "still_failing": fails, "kill": len(fails) >= 2, "kill_rule": "2 or more of the 6 leaves still fail",
             "complete": all(v["verdict"].startswith(("pass", "still fails")) for v in self.verdicts.values())
                         and len(self.verdicts) == 6,
             "projection": self.projection(), "guard_core_h": self.a.guard_core_hours,
             "code_md5": {f: md5(os.path.join(HERE, f)) for f in ("pr3_runner.py", "pr3_common.py", "pr3_extra_rung.py")}}
        C.write_json(os.path.join(self.a.res, "pr3_tally.json"), t)

    def adopt_orphans(self, heavy):
        for p in heavy:
            cl = p["cmd"]
            if p["pid"] in self.kids:
                continue
            kind = "ladder" if has(cl, b"decode_willow.py") else ("extra" if has(cl, b"pr3_extra_rung.py") else None)
            if not kind:
                continue
            lf = arg_of(cl, b"--leaf")
            out = arg_of(cl, b"--out") if kind == "ladder" else arg_of(cl, b"--res")
            if not lf or not out or not out.startswith(os.path.join(self.a.res, "parts")):
                continue
            try:
                j = int(out.split(os.path.join(self.a.res, "parts", "p"))[1].split("/")[0])
            except (IndexError, ValueError):
                continue
            u = self.units.get((kind, lf, j))
            if u and u["state"] not in ("running", "orphan"):
                u["state"] = "orphan"; u["pid"] = p["pid"]
                self.ledger["units"].setdefault(self.ukey(u), {}).update(kind=kind, r=u["r"], state="orphan", pid=p["pid"])
                self.log(f"found {kind} {lf} p{j} running (pid {p['pid']}, from an earlier runner); it keeps its slot")

    def start(self, u):
        rec = self.ledger["units"].setdefault(self.ukey(u), {"kind": u["kind"], "r": u["r"]})
        if rec.get("state") in ("running", "orphan") and rec.get("cpu_now"):
            rec["lost_cpu_s"] = rec.get("lost_cpu_s", 0.0) + rec["cpu_now"]       # a run lost in a restart
        env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        lp = os.path.join(self.a.logs, f"{u['kind']}_{u['leaf'].split('/')[-2]}_{u['leaf'].split('/')[-1]}_p{u['part']}.out")
        import subprocess
        with open(lp, "ab") as lf:
            p = subprocess.Popen(self.cmd(u), env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        u["state"] = "running"; u["pid"] = p.pid
        self.kids[p.pid] = u
        rec.update(state="running", pid=p.pid, started=now(), cpu_now=0.0, peak_rss=0, projected_s=self.projected_s(u))
        if u["kind"] == "extra":
            rec["band"] = self.band_of(u)
        self.log(f"start {u['kind']} {u['leaf']} p{u['part']} (pid {p.pid}; projected {rec['projected_s'] / 3600:.2f} "
                 f"core-hours)")

    def run(self):
        a = self.a
        self.log(f"pr3_runner (md5 {md5(os.path.abspath(__file__))}) design {a.design}, guard {a.guard_core_hours:.1f} "
                 f"core-hours, at most {a.max_heavy} heavy processes, {a.min_free_gib} GiB kept free, cap {a.cap_gib} GiB")
        self.refresh()
        stop, last_report = None, 0.0
        while True:
            me = {os.getpid()}
            self.procs_all = []
            heavy = self.procs.scan(me)
            self.procs_all = [{"cmd": cmdline(int(p))} for p in os.listdir("/proc") if p.isdigit()]
            self.adopt_orphans(heavy)
            # reap children; watch their memory
            for pid in list(self.kids):
                u = self.kids[pid]; rec = self.ledger["units"][self.ukey(u)]
                rss, cpu = C.group_usage(pid)
                rec["cpu_now"] = max(rec.get("cpu_now", 0.0), cpu); rec["peak_rss"] = max(rec.get("peak_rss", 0), rss)
                if rss > a.cap_gib * C.GiB and not rec.get("capped"):
                    rec["capped"] = True
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.log(f"{u['kind']} {u['leaf']} p{u['part']}: RSS {rss / C.GiB:.2f} GiB above the cap; killed")
                got, status, ru = os.wait4(pid, os.WNOHANG)
                if not got:
                    continue
                code = os.waitstatus_to_exitcode(status)
                del self.kids[pid]
                rec.update(cpu_s=ru.ru_utime + ru.ru_stime, ended=now(), exit=code,
                           peak_rss=max(rec.get("peak_rss", 0), ru.ru_maxrss * 1024))
                ok = (self.ladder_done(u["leaf"], u["part"]) if u["kind"] == "ladder" else self.extra_done(u["leaf"], u["part"]))
                if code == 0 and ok:
                    u["state"] = rec["state"] = "done"
                    self.log(f"done {u['kind']} {u['leaf']} p{u['part']}: {rec['cpu_s'] / 3600:.2f} core-hours, "
                             f"peak {rec['peak_rss'] / C.GiB:.2f} GiB")
                else:
                    u["state"] = rec["state"] = "failed"
                    why = "memory cap" if rec.get("capped") else f"exit {code}"
                    self.log(f"FAILED {u['kind']} {u['leaf']} p{u['part']}: {why}")
                    stop = stop or ((7, "a unit hit the memory cap") if rec.get("capped") else (5, f"a unit failed ({why})"))
            # orphans: CPU from /proc until they end
            for k, u in self.units.items():
                if u["state"] == "orphan":
                    rec = self.ledger["units"][self.ukey(u)]
                    s = C._stat(u["pid"])
                    if s and has(cmdline(u["pid"]), b"decode_willow.py", b"pr3_extra_rung.py"):
                        rec["cpu_now"] = max(rec.get("cpu_now", 0.0), s["cpu"])
                    else:
                        done = self.ladder_done(u["leaf"], u["part"]) if u["kind"] == "ladder" else self.extra_done(u["leaf"], u["part"])
                        u["state"] = rec["state"] = "done" if done else "pending"
                        if done:
                            rec["cpu_s"] = rec.get("cpu_now", 0.0)
                        self.log(f"the earlier runner's {u['kind']} {u['leaf']} p{u['part']} ended: {u['state']}")
            self.refresh()
            P = self.projection()
            if P["total_h"] > a.guard_core_hours and not stop:
                stop = (6, f"projection {P['total_h']:.1f} core-hours is past the guard {a.guard_core_hours:.1f}")
            if self.killed() and not stop:
                stop = (0, "d = 7 ML killed: 2 or more leaves still fail after the extra rung")
            hold, pause = self.holds(heavy)
            n_heavy = len(heavy)
            free = a.max_heavy - n_heavy - sum(hold.values())
            todo = [u for u in self.units.values() if u["state"] == "pending" and
                    (u["kind"] == "ladder" or self.needs_extra(u["leaf"]))]
            todo.sort(key=lambda u: -self.projected_s(u))
            if a.dry_run:
                self.log(f"dry run: {n_heavy} heavy now, holds {hold}, free {free}, MemAvailable "
                         f"{C.mem_available() / C.GiB:.1f} GiB; {len(todo)} units ready; projection {P}")
                for u in todo[:12]:
                    self.log(f"  next: {u['kind']} {u['leaf']} p{u['part']} {self.projected_s(u) / 3600:.2f} core-hours "
                             f"slice {C.part_slice(a.design, u['part'])}")
                return 0
            if stop or pause:
                if stop and not getattr(self, "_stop_logged", False):
                    self.log(f"STOP ({stop[0]}): {stop[1]}. No new unit starts; waiting for {len(self.kids)} running.")
                    self._stop_logged = True
            else:
                while todo and free > 0:
                    u = todo[0]
                    if C.mem_available() - self.need[u["kind"]] < a.min_free_gib * C.GiB:
                        break
                    self.start(u); todo.pop(0); free -= 1
            if time.time() - last_report > 1800:
                self.log(f"{n_heavy} heavy processes, holds {hold}, PR3 running {len(self.kids)}; projection "
                         f"{P['total_h']:.1f} core-hours (used {P['used_h']:.1f}); verdicts "
                         + ", ".join(f"{lf.split('/')[-2]}/{lf.split('/')[-1]}: {v['verdict']}" for lf, v in sorted(self.verdicts.items())))
                last_report = time.time()
            self.save()
            orph = any(u["state"] == "orphan" for u in self.units.values())
            alldone = all(u["state"] == "done" or (u["kind"] == "extra" and not self.needs_extra(u["leaf"])
                                                   and self.verdicts.get(u["leaf"], {}).get("verdict", "").startswith(("pass", "still")))
                          for u in self.units.values())
            if (stop or alldone) and not self.kids and not orph:
                self.tally("done" if alldone else f"stopped: {stop[1]}")
                code = 0 if (alldone or stop[0] == 0) else stop[0]
                self.log(f"exit {code}: " + ("all units done" if alldone else stop[1]))
                return code
            sig = json.dumps([self.verdicts, {self.ukey(u): u["state"] for u in self.units.values()}], sort_keys=True, default=str)
            if sig != getattr(self, "_sig", None) or time.time() - getattr(self, "_tally_t", 0) > 600:
                self.tally("running"); self._sig, self._tally_t = sig, time.time()
            time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--project", default="/mnt/project-files/tensor-networks/deep-dives/experimental")
    ap.add_argument("--work", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--design", required=True, choices=sorted(C.DESIGNS))
    ap.add_argument("--guard-core-hours", type=float, required=True)
    ap.add_argument("--max-heavy", type=int, default=4)
    ap.add_argument("--min-free-gib", type=float, default=3.0)
    ap.add_argument("--cap-gib", type=float, default=4.0)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(Runner(a).run())


if __name__ == "__main__":
    main()
