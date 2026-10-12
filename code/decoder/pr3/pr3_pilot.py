#!/usr/bin/env python3
"""PR3 stage 0, the pilot (notes/decoder/willow_prereg_r3.md section 5; approved by the orchestrator 2026-10-10 ~20:00Z).

200 shots of d7_at_q6_7/X/r50 (every 10th shot from shot 0, so they belong to either design's sample) through the
ladder tau1 = 1e-6, tau2 = 1e-7 (band 0.1) and the extra rung tau3 = 1e-8 on the band shots. It measures cost, band
share and memory, and nothing else: its extract has no shipped predictions, so no comparison with Libra can be made.
  1. the frozen decode_willow.py, unchanged, under an RSS cap of 4 GiB (a watchdog kills the process group above it);
  2. pr3_extra_rung.py (k2_extra_rung.extra_rung, unchanged) on the pilot's band shots, under the same cap;
  3. the projection fixed before the pilot (pr3_common.projection) for both designs, and the orchestrator's rule:
     25,000 per leaf if <= 200 core-hours, else 10,000 if <= 100, else stop. r = 10 rates come from the round-1 probes.
Output: <out>/pilot.json (+ <out>/decode/..., <out>/extra_rung/...). Exit 0 if a design is chosen, 2 if none is,
3 if a stage failed or hit the memory cap.
  python3 -I -B pr3_pilot.py --root <pilot extract> --out results/decoder/willow/pr3/pilot --work <scratch> \
      --logs <scratch logs> --probes results/decoder/willow/decode
"""
import argparse, json, math, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pr3_common as C

CODE = os.environ.get("PR3_CODE") or os.path.dirname(HERE)
SHOTS, TAG, CHUNK, CAP = "0:2000:10", "rl_pilot", 50, 4 * C.GiB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--probes", required=True, help="round 1's decode/ folder (the d = 7 probes, rl_s25)")
    a = ap.parse_args()
    os.makedirs(a.logs, exist_ok=True)
    lf = C.leaf("X", 50)
    if any("predicted" in f for _, _, fs in os.walk(os.path.join(a.root, lf)) for f in fs):
        sys.exit("the pilot extract holds shipped predictions; the pilot must not have them")
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rec = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "leaf": lf, "shots": SHOTS, "chunk": CHUNK,
           "cap_bytes": CAP, "mem_available_start": C.mem_available()}
    t0 = time.time()
    cmd = [sys.executable, "-I", "-B", os.path.join(CODE, "decode_willow.py"), "--root", a.root,
           "--out", os.path.join(a.out, "decode"), "--work", a.work, "--leaf", lf, "--dem-file", C.DEM_FILE,
           "--dem-tag", TAG, "--decoders", C.DECODERS, "--shots", SHOTS, "--chunk", str(CHUNK), "--synd-ref", "0"]
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} stage 1: {' '.join(cmd)}", flush=True)
    r1 = C.run_capped(cmd, env, os.path.join(a.logs, "pilot_decode.out"), CAP)
    rec["stage1_ladder"] = r1
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} stage 1 done: {r1}", flush=True)
    fd = os.path.join(a.out, "decode", lf, TAG)
    try:
        S = json.load(open(os.path.join(fd, "summary.json")))
    except Exception as ex:
        S = {"status": f"unreadable: {ex!r}"}
    if r1["exit"] != 0 or r1["capped"] or S.get("status") != "complete":
        rec["result"] = "stage 1 failed" + (" (memory cap hit)" if r1["capped"] else "")
        C.write_json(os.path.join(a.out, "pilot.json"), rec); sys.exit(3)
    L = S["decoders"][C.LADDER]["ladder"]
    n, r = L["n"], int(S["rounds"])
    nb = L["n_redecoded"]
    base_ms = sum(S["decoders"][x]["ms_per_shot"] for x in ("corr_mwpm", "mwpm"))
    rec["ladder"] = {k: L[k] for k in ("n", "n_redecoded", "share_redecoded", "ms_per_round_per_shot_tau1",
                                       "ms_per_round_per_shot_tau1_mean", "ms_per_round_per_redecoded_shot_tau2",
                                       "max_K_tau1", "max_K_tau2", "nan_final")}
    rec["baselines_ms_per_shot"] = base_ms
    if nb:
        cmd2 = [sys.executable, "-I", "-B", os.path.join(HERE, "pr3_extra_rung.py"), "--root", a.root, "--res", a.out,
                "--out", os.path.join(a.out, "extra_rung"), "--leaf", lf, "--tag", TAG, "--check-shots", "50"]
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} stage 2: {' '.join(cmd2)}", flush=True)
        r2 = C.run_capped(cmd2, env, os.path.join(a.logs, "pilot_extra.out"), CAP)
        rec["stage2_extra"] = r2
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} stage 2 done: {r2}", flush=True)
        try:
            E = json.load(open(os.path.join(a.out, "extra_rung", lf, TAG, "extra_rung.json")))
        except Exception as ex:
            E = {"error": repr(ex)}
        if r2["exit"] != 0 or r2["capped"] or "tau3" not in E:
            rec["result"] = "stage 2 failed" + (" (memory cap hit)" if r2["capped"] else "")
            C.write_json(os.path.join(a.out, "pilot.json"), rec); sys.exit(3)
        T3 = E["tau3"]
        rec["extra"] = {"check": E["check"], "tau3": {k: T3[k] for k in ("n_band", "ms_per_round_per_band_shot_tau3",
                                                                        "cpu_seconds_tau3", "max_K_tau3", "nan_tau3")}}
    # rates at r = 10 from the round-1 probes (2,000 shots each, X and Z, the same ladder and DEM)
    pr = []
    for b in C.BASES:
        s = json.load(open(os.path.join(a.probes, C.leaf(b, 10), "rl_s25", "summary.json")))
        pr.append(s["decoders"][C.LADDER]["ladder"])
    c1_10 = sum(x["ms_per_round_per_shot_tau1"] for x in pr) / 2
    s_10 = sum(x["n_redecoded"] for x in pr) / sum(x["n"] for x in pr)
    c2_10 = sum(x["ms_per_round_per_redecoded_shot_tau2"] for x in pr) / 2
    m = {"c1_10": c1_10, "s_10": s_10, "c2_10": c2_10, "c1_50": L["ms_per_round_per_shot_tau1"], "cb_50": base_ms}
    fallback = []
    if nb:
        m["s_50"] = nb / n
        m["c2_50"] = L["ms_per_round_per_redecoded_shot_tau2"]
        m["c3_50"] = rec["extra"]["tau3"]["ms_per_round_per_band_shot_tau3"]
        dec_cpu = (m["c1_50"] * n * r + m["c2_50"] * nb * r + base_ms * n) / 1e3
        m["ovh_extra_s"] = max(0.0, rec["stage2_extra"]["cpu_s"] - rec["extra"]["tau3"]["cpu_seconds_tau3"]
                               - rec["extra"]["check"]["cpu_seconds"])
    else:                       # fixed before the pilot: the model's values when the pilot has no band shot
        m["s_50"], m["c2_50"] = 0.0275, 1.2 * c2_10
        m["c3_50"] = 4.5 * m["c2_50"]
        dec_cpu = (m["c1_50"] * n * r + base_ms * n) / 1e3
        m["ovh_extra_s"] = 30.0
        fallback.append("no band shot in the pilot: s_50 = 2.75%, c2_50 = 1.2 c2_10, c3_50 = 4.5 c2_50, 30 s overhead")
    m["ovh_ladder_s"] = max(0.0, r1["cpu_s"] - dec_cpu)
    rec["measured_rates"] = m
    rec["fallbacks"] = fallback
    rec["decision"] = C.choose_design(m)
    # descriptive: the band share at r = 50 at its 90% upper Poisson bound
    if nb:
        from math import exp, factorial
        lam = nb
        while sum(exp(-lam) * lam ** k / factorial(k) for k in range(nb + 1)) > 0.10:
            lam += 0.01
        mu = dict(m, s_50=lam / n)
        rec["descriptive_s50_upper90"] = {"s_50": lam / n, "projection_25k": C.projection(mu, "25k")["total"],
                                          "projection_10k": C.projection(mu, "10k")["total"]}
    rec["cpu_core_hours_pilot"] = (r1["cpu_s"] + rec.get("stage2_extra", {}).get("cpu_s", 0.0)) / 3600
    rec["peak_rss_bytes"] = max(r1["peak_rss"], rec.get("stage2_extra", {}).get("peak_rss", 0))
    rec["wall_s"] = time.time() - t0
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["code_md5"] = {f: C_md5(os.path.join(HERE, f)) for f in ("pr3_common.py", "pr3_pilot.py", "pr3_extra_rung.py")}
    rec["code_md5"]["decode_willow.py"] = C_md5(os.path.join(CODE, "decode_willow.py"))
    rec["result"] = "design chosen" if rec["decision"]["design"] else "no design within the rule: stop and report"
    C.write_json(os.path.join(a.out, "pilot.json"), rec)
    d = rec["decision"]
    print(json.dumps({"rates": m, "projection_25k": d["projection_25k"]["total"], "projection_10k": d["projection_10k"]["total"],
                      "design": d["design"], "guard": d.get("cpu_guard_core_h"), "pilot_core_h": rec["cpu_core_hours_pilot"],
                      "peak_rss_GiB": rec["peak_rss_bytes"] / C.GiB}, indent=1, default=float), flush=True)
    sys.exit(0 if d["design"] else 2)


def C_md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
