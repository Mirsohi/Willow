#!/usr/bin/env python3
"""Shared pieces of PR3 (notes/decoder/willow_prereg_r3.md): the leaves and shot selections, a run under an RSS cap
with the CPU time of the run, and the cost projection that was fixed before the pilot ran.

Nothing here decodes. The decoding is the frozen decode_willow.py (the tau ladder) and k2_extra_rung.extra_rung
(the extra rung), both unchanged.
"""
import json, math, os, signal, subprocess, time

PAGE = os.sysconf("SC_PAGE_SIZE")
TICK = os.sysconf("SC_CLK_TCK")
EXP = "google_105Q_surface_code_d3_d5_d7"
PATCH = "d7_at_q6_7"
ROUNDS = (10, 30, 50)
BASES = ("X", "Z")
LADDER = "ml_tau2:1e-6/1e-7:0.1"
DECODERS = "corr_mwpm,mwpm," + LADDER
DEM_FILE = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
# designs: every step-th shot from shot 0; the r = 10 probe shots (index divisible by 25) are decoded with their part
# but left out of every PR3 statistic (their result is known from the round-1 probes)
DESIGNS = {"25k": {"step": 2, "n": 25000}, "10k": {"step": 5, "n": 10000}}
PROBE_STEP, PROBE_ROUNDS = 25, (10,)
PARTS = 5                       # each leaf is decoded as 5 parts of 10,000 consecutive shot indices
CHECK_SHOTS_PER_PART = 10       # the extra rung's tau2 rebuild check: 10 band shots per part, 50 per leaf
GiB = 1 << 30


def leaf(basis, r):
    return f"{EXP}/{PATCH}/{basis}/r{r}"


def part_slice(design, j):
    """--shots of part j (0..PARTS-1): indices [10,000 j, 10,000 (j + 1)) with the design's step."""
    st = DESIGNS[design]["step"]
    w = 50000 // PARTS
    return f"{j * w}:{(j + 1) * w}:{st}"


def in_sample(idx, r):
    """Mask of the shots that enter PR3's statistics (the r = 10 probe shots are left out)."""
    import numpy as np
    idx = np.asarray(idx)
    return (idx % PROBE_STEP != 0) if r in PROBE_ROUNDS else np.ones(len(idx), bool)


# ---------------------------------------------------------------- processes: RSS and CPU
def _stat(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        rest = s[s.rindex(")") + 2:].split()
        return {"ppid": int(rest[1]), "pgid": int(rest[2]), "cpu": (int(rest[11]) + int(rest[12])) / TICK,
                "rss": int(rest[21]) * PAGE}
    except (OSError, ValueError, IndexError):
        return None


def group_usage(pgid):
    """(RSS bytes, CPU seconds so far) summed over the live processes of a process group."""
    rss = cpu = 0
    for p in os.listdir("/proc"):
        if p.isdigit():
            s = _stat(int(p))
            if s and s["pgid"] == pgid:
                rss += s["rss"]; cpu += s["cpu"]
    return rss, cpu


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def run_capped(cmd, env, log_path, cap_bytes, poll=0.5, on_poll=None):
    """Run cmd in its own session; kill the whole group if its RSS passes cap_bytes. Returns a dict with the exit
    code, CPU seconds (from wait4), peak RSS, and whether the cap was hit."""
    with open(log_path, "ab") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    peak, capped, t0 = 0, False, time.time()
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            p.returncode = os.waitstatus_to_exitcode(status)
            return {"exit": p.returncode, "cpu_s": ru.ru_utime + ru.ru_stime, "peak_rss": max(peak, ru.ru_maxrss * 1024),
                    "capped": capped, "wall_s": time.time() - t0, "pid": p.pid}
        rss, _ = group_usage(p.pid)
        peak = max(peak, rss)
        if rss > cap_bytes and not capped:
            capped = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if on_poll:
            on_poll(p.pid, rss, peak)
        time.sleep(poll)


# ---------------------------------------------------------------- the cost projection (fixed before the pilot)
def projection(m, design, extra_on_all=True):
    """Core-hours for the 6 leaves of a design, from measured rates m (milliseconds per round per shot):
      c1_10, c1_50   tau1, every shot (the probes at r = 10; the pilot at r = 50)
      s_10, s_50     band share (re-decoded at tau2)
      c2_10, c2_50   tau2, per band shot
      c3_50          tau3 = 1e-8, per band shot (the pilot); c3 at r = 10 and 30 is c2 there times c3_50 / c2_50
      cb_50          corr. MWPM + MWPM, ms per shot at r = 50 (scaled by r / 50)
      ovh_ladder_s, ovh_extra_s   CPU seconds per process outside the decoders (start-up, I/O, the recompute check)
    r = 30 is the mean of r = 10 and r = 50 for c1, s and c2. Every part also re-decodes CHECK_SHOTS_PER_PART band
    shots at tau2 before its extra rung. Returns {"total": ..., "rows": [...]}"""
    N = DESIGNS[design]["n"]
    c1 = {10: m["c1_10"], 50: m["c1_50"]}
    s = {10: m["s_10"], 50: m["s_50"]}
    c2 = {10: m["c2_10"], 50: m["c2_50"]}
    for v in (c1, s, c2):
        v[30] = (v[10] + v[50]) / 2
    q = m["c3_50"] / m["c2_50"]
    c3 = {10: c2[10] * q, 30: c2[30] * q, 50: m["c3_50"]}
    rows, tot = [], 0.0
    for b in BASES:
        for r in ROUNDS:
            band = s[r] * N
            lad = (N * r * c1[r] + band * r * c2[r] + N * m["cb_50"] * r / 50) / 1e3 + PARTS * m["ovh_ladder_s"]
            ext = 0.0
            if extra_on_all:
                chk = min(band, PARTS * CHECK_SHOTS_PER_PART)
                ext = (band * r * c3[r] + chk * r * c2[r]) / 1e3 + PARTS * m["ovh_extra_s"]
            rows.append({"leaf": leaf(b, r), "band_shots": band, "ladder_h": lad / 3600, "extra_h": ext / 3600})
            tot += (lad + ext) / 3600
    return {"design": design, "total": tot, "ladder_h": sum(x["ladder_h"] for x in rows),
            "extra_h": sum(x["extra_h"] for x in rows), "rows": rows}


def choose_design(m):
    """The orchestrator's rule (2026-10-10 ~20:00Z): 25,000 per leaf if its projection, with the extra rung on all 6
    leaves, is at most 200 core-hours; otherwise 10,000 if at most 100; otherwise stop and report."""
    p25, p10 = projection(m, "25k"), projection(m, "10k")
    if p25["total"] <= 200:
        ch = "25k"
    elif p10["total"] <= 100:
        ch = "10k"
    else:
        ch = None
    out = {"projection_25k": p25, "projection_10k": p10, "design": ch}
    if ch:
        out["cpu_guard_core_h"] = 1.3 * out[f"projection_{ch}"]["total"]
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
