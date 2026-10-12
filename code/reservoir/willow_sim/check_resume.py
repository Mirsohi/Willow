"""Resume check of the simulation's stage B after the outage of 11 Oct 2026 (notes/reservoir/sim_design.md, amendment
of 11 Oct 2026; orchestrator's request of 13:24Z: check the stage-B outputs in local scratch before reusing them).
EXPLORATORY run, pre-specified design; this script only reads.

reservoir, 2026-10-11. For every stage-B leaf of the plan:
  sample   sample.json (seed and shots as in the plan; DEM md5 = the real leaf's RL DEM; circuit md5 = the real leaf's),
           the md5 of every file against sample.json, the shapes read back through stim, and a fresh draw from stim's
           DEM sampler with the plan's seed compared bit for bit (stim only promises the same draw for the same version
           and SIMD width, so a difference there with every md5 intact is reported, not counted as damage);
  done     for each decode step that state.json marks done: summary.json complete, decode_willow.py md5 and DEM md5 as
           now, the job's flags as in the plan, n per decoder; shots.npz (idx 0..n-1, observable flips = the archive's);
           each decoder's npz (idx 0..n-1, pred of shape (n, 1), errors and NaN posteriors as in summary.json); the
           chunk checkpoints (spec.json as decode_willow.py writes it, every chunk present and loadable with the right
           length, their concatenation = the consolidated npz);
  partial  for a step not done, the chunk checkpoints it left, by the same rules (decode_willow.py skips a finished
           chunk whose spec.json matches, so these are what a rerun reuses);
  stage A  the scratch files of stage A that the A + B analysis reads again, against the copies made in
           results/reservoir/willow_sim/ at 02:13:40Z (md5), and the md5 of stage A's simulated shot files.
Usage: python3 -I -B check_resume.py PLAN STATE OUT_JSON   (writes the report; prints a summary)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import stim

sys.dont_write_bytecode = True
ARCH = "google_105Q_surface_code_d3_d5_d7"
DEM_REL = "decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def dn_of(name):
    return name.replace(":", "_").replace("/", "-")


def check_sample(plan, L):
    P = plan["paths"]
    dst = os.path.join(P["sim_root"], ARCH, L["leaf"])
    src = os.path.join(P["real_arch"], L["leaf"])
    prob, info = [], {}
    rp = os.path.join(dst, "sample.json")
    if not os.path.exists(rp):
        return ["sample.json missing"], None, info
    rec = json.load(open(rp))
    if rec.get("seed") != L["seed"]:
        prob.append("seed differs from the plan")
    if rec.get("shots") != L["shots"]:
        prob.append("shots differ from the plan")
    real_dem = md5(os.path.join(src, DEM_REL))
    if rec.get("dem_md5") != real_dem:
        prob.append("DEM md5 differs from the real leaf's RL DEM")
    if rec.get("md5", {}).get("circuit_ideal.stim") != md5(os.path.join(src, "circuit_ideal.stim")):
        prob.append("circuit md5 differs from the real leaf's")
    for f in ("circuit_ideal.stim", DEM_REL, "detection_events.b8", "obs_flips_actual.b8"):
        p = os.path.join(dst, f)
        if not os.path.exists(p):
            prob.append(f"{f} missing")
        elif md5(p) != rec.get("md5", {}).get(f):
            prob.append(f"{f}: md5 differs from sample.json")
    if prob:
        return prob, None, info
    dem = stim.DetectorErrorModel.from_file(os.path.join(dst, DEM_REL))
    det = stim.read_shot_data_file(path=os.path.join(dst, "detection_events.b8"), format="b8",
                                   num_detectors=dem.num_detectors)
    obs = stim.read_shot_data_file(path=os.path.join(dst, "obs_flips_actual.b8"), format="b8", num_observables=1)
    if det.shape != (L["shots"], dem.num_detectors) or obs.shape != (L["shots"], 1):
        prob.append(f"shapes {det.shape} / {obs.shape}")
    if rec.get("num_detectors") != dem.num_detectors:
        prob.append("detector count differs from sample.json")
    d2, o2, _ = dem.compile_sampler(seed=L["seed"]).sample(shots=L["shots"])
    info["fresh_draw_equal"] = bool(np.array_equal(d2, det) and np.array_equal(o2, obs))
    info["stim_now"], info["stim_then"] = stim.__version__, rec.get("stim")
    return prob, {"obs": obs[:, 0].astype(np.uint8), "dem_md5": real_dem, "num_detectors": dem.num_detectors}, info


def job_of(plan, kind, key, L):
    if kind == "decode":
        D = plan["decode"][key]
        return L["shots"], "rl_sim", D["decoders"], D["chunk"]
    R = plan["refs"][f"r{L['r']}"][kind]
    return R["shots"], R["dem_tag"], R["decoder"], R["chunk"]


def check_chunks(wd, spec, n, chunk, require_all):
    """Returns (problems, chunk files that are valid, concatenated pred or None)."""
    prob, good = [], []
    sp = os.path.join(wd, "spec.json")
    if not os.path.isdir(wd):
        return (["work folder missing"] if require_all else []), good, None
    if not os.path.exists(sp):
        return ["spec.json missing"], good, None
    if json.load(open(sp)) != spec:
        prob.append("spec.json differs from the plan's job (decode_willow.py would discard these chunks)")
    want = [f"chunk_{c0:06d}.npz" for c0 in range(0, n, chunk)]
    have = sorted(f for f in os.listdir(wd) if f.startswith("chunk_"))
    extra = [f for f in have if f not in want]
    if extra:
        prob.append(f"unexpected chunk files {extra}")
    preds = []
    for c0, f in zip(range(0, n, chunk), want):
        p = os.path.join(wd, f)
        if not os.path.exists(p):
            if require_all:
                prob.append(f"{f} missing")
            continue
        try:
            z = np.load(p)
            pred = z["pred"]
            ok = pred.shape == (min(chunk, n - c0), 1) and "seconds" in z.files and np.isfinite(float(z["seconds"]))
            if "llr" in z.files:
                ok = ok and z["llr"].shape == (min(chunk, n - c0),)
        except Exception as ex:  # noqa: BLE001
            prob.append(f"{f}: unreadable ({ex!r})")
            continue
        if not ok:
            prob.append(f"{f}: wrong shape or fields")
            continue
        good.append(f)
        preds.append(pred)
    full = np.concatenate(preds) if preds and len(good) == len(want) else None
    return prob, good, full


def check_decode(plan, st, L, samp, dw_md5, done):
    P = plan["paths"]
    n, tag, decs, chunk = job_of(plan, st["kind"], st.get("key"), L)
    rel = f"{ARCH}/{L['leaf']}"
    so = os.path.join(P["decode_out"], rel, tag)
    wroot = os.path.join(P["decode_work"], rel, tag)
    spec0 = {"leaf": rel, "dem_md5": samp["dem_md5"], "selection": f"0:{n}:1", "max_shots": n, "chunk": chunk, "n": n}
    out = {"n": n, "tag": tag, "problems": [], "chunks": {}}
    obs = samp["obs"][:n]
    if done:
        sp = os.path.join(so, "summary.json")
        if not os.path.exists(sp):
            out["problems"].append("summary.json missing")
            return out
        s = json.load(open(sp))
        job = s.get("job", {})
        checks = {"status complete": s.get("status") == "complete", "experiment": s.get("experiment") == rel,
                  "decode_willow.py md5": s.get("versions", {}).get("decode_willow_md5") == dw_md5,
                  "DEM md5": s.get("dem_md5") == samp["dem_md5"], "dem tag": s.get("dem_tag") == tag,
                  "job decoders": job.get("decoders") == decs, "job dem_file": job.get("dem_file") == DEM_REL,
                  "job shots": job.get("shots") == f"0:{n}:1", "job max_shots": job.get("max_shots") == n,
                  "job chunk": job.get("chunk") == chunk, "job synd_ref": job.get("synd_ref") == 0,
                  "job recompute_check": job.get("recompute_check") == "none",
                  "job time_budget": not job.get("time_budget"),
                  "n": s.get("n") == n and s.get("n_total") == L["shots"],
                  "detectors": s.get("dem_num_detectors") == samp["num_detectors"],
                  "recompute mode": s.get("recompute_check", {}).get("mode") == "none"}
        out["problems"] += [f"summary: {k}" for k, v in checks.items() if not v]
        try:
            z = np.load(os.path.join(so, "shots.npz"))
            if not np.array_equal(z["idx"], np.arange(n)) or not np.array_equal(z["obs"], obs):
                out["problems"].append("shots.npz: idx or observable flips differ from the archive")
        except Exception as ex:  # noqa: BLE001
            out["problems"].append(f"shots.npz unreadable ({ex!r})")
    for dec in decs.split(","):
        dn = dn_of(dec)
        prob, good, full = check_chunks(os.path.join(wroot, dn), dict(spec0, decoder=dec), n, chunk, done)
        out["chunks"][dec] = {"valid": len(good), "expected": len(range(0, n, chunk)), "problems": prob}
        out["problems"] += [f"{dec} chunks: {p}" for p in prob]
        npz = os.path.join(so, dn + ".npz")
        if not done:
            if os.path.exists(npz) and full is not None:
                try:
                    zc = np.load(npz)
                    out["chunks"][dec]["consolidated_equals_chunks"] = bool(
                        np.array_equal(zc["pred"], full) and np.array_equal(zc["idx"], np.arange(n)))
                except Exception as ex:  # noqa: BLE001
                    out["chunks"][dec]["consolidated_equals_chunks"] = repr(ex)
            continue
        ds = s.get("decoders", {}).get(dec, {})
        if ds.get("n") != n:
            out["problems"].append(f"{dec}: summary n {ds.get('n')}")
        try:
            z = np.load(npz)
            pred = z["pred"]
            if not np.array_equal(z["idx"], np.arange(n)) or pred.shape != (n, 1):
                out["problems"].append(f"{dec}: idx or pred shape")
            elif int(np.sum(pred[:, 0] != obs)) != ds.get("errors"):
                out["problems"].append(f"{dec}: errors {int(np.sum(pred[:, 0] != obs))} vs summary {ds.get('errors')}")
            if full is not None and not np.array_equal(full, pred):
                out["problems"].append(f"{dec}: consolidated npz differs from its chunks")
            if "llr" in z.files and int(np.isnan(z["llr"]).sum()) != ds.get("nan_posteriors", 0):
                out["problems"].append(f"{dec}: NaN posteriors differ from summary")
            out.setdefault("errors", {})[dec] = ds.get("errors")
        except Exception as ex:  # noqa: BLE001
            out["problems"].append(f"{dec}: npz unreadable ({ex!r})")
    if not done and os.path.isdir(so):
        out["outputs_present"] = sorted(os.listdir(so))
    return out


def check_stage_a(plan):
    P = plan["paths"]
    res = P["results"]
    bad, n = [], 0
    for base, dirs, files in os.walk(os.path.join(res, "decode")):
        for fn in files:
            r = os.path.relpath(os.path.join(base, fn), res)
            n += 1
            if md5(os.path.join(res, r)) != md5(os.path.join(P["decode_out"], os.path.relpath(r, "decode"))):
                bad.append(r)
    for fn in os.listdir(os.path.join(res, "mpe")):
        n += 1
        if md5(os.path.join(res, "mpe", fn)) != md5(os.path.join(P["mpe_dir"], fn)):
            bad.append(f"mpe/{fn}")
    for a, b in (("sim_analysis_A.json", os.path.join(P["analysis_dir"], "sim_analysis_A.json")),
                 ("sim_plan.json", PLAN[0])):
        n += 1
        if md5(os.path.join(res, a)) != md5(b):
            bad.append(a)
    samples = json.load(open(os.path.join(res, "samples_A.json")))
    shots_bad = []
    for rec in samples:
        dst = os.path.join(P["sim_root"], ARCH, rec["leaf"])
        if json.load(open(os.path.join(dst, "sample.json"))) != rec:
            shots_bad.append(f"{rec['leaf']}: sample.json")
        for f in ("detection_events.b8", "obs_flips_actual.b8", DEM_REL):
            if md5(os.path.join(dst, f)) != rec["md5"][f]:
                shots_bad.append(f"{rec['leaf']}: {f}")
    return {"files_compared": n, "mismatch": bad, "sample_leaves": len(samples), "sample_problems": shots_bad}


PLAN = [""]


def main(plan_path, state_path, out_json):
    PLAN[0] = os.path.abspath(plan_path)
    plan = json.load(open(plan_path))
    state = json.load(open(state_path))
    dw_md5 = md5(plan["paths"]["decode_willow"])
    rep = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan_md5": md5(plan_path),
           "state_md5": md5(state_path), "decode_willow_md5": dw_md5, "stim": stim.__version__,
           "samples": {}, "steps": {}}
    samp = {}
    for L in [x for x in plan["leaves"] if x["stage"] == "B"]:
        prob, s, info = check_sample(plan, L)
        rep["samples"][L["leaf"]] = {"problems": prob, **info}
        samp[L["leaf"]] = s
    for L in [x for x in plan["leaves"] if x["stage"] == "B"]:
        kinds = [("decode", f"B{L['d']}")] + ([("t2", "B"), ("t3", "B")] if L["d"] == 5 else [])
        for kind, key in kinds:
            sid = f"B/{kind}/{L['leaf']}"
            status = state["steps"].get(sid, {}).get("status")
            if samp[L["leaf"]] is None:
                rep["steps"][sid] = {"state": status, "problems": ["sample failed its check"]}
                continue
            r = check_decode(plan, {"kind": kind, "key": key}, L, samp[L["leaf"]], dw_md5, status == "done")
            r["state"] = status
            rep["steps"][sid] = r
    rep["stage_A"] = check_stage_a(plan)
    rep["rerun_done_steps"] = sorted(k for k, v in rep["steps"].items() if v["state"] == "done" and v["problems"])
    rep["bad_partial_steps"] = sorted(k for k, v in rep["steps"].items() if v["state"] != "done" and v["problems"])
    rep["bad_samples"] = sorted(k for k, v in rep["samples"].items() if v["problems"])
    rep["fresh_draw_unequal"] = sorted(k for k, v in rep["samples"].items() if v.get("fresh_draw_equal") is False)
    rep["reused_partial_chunks"] = {k: {d: c["valid"] for d, c in v["chunks"].items() if c["valid"]}
                                    for k, v in rep["steps"].items()
                                    if v["state"] != "done" and any(c["valid"] for c in v.get("chunks", {}).values())}
    with open(out_json + ".tmp", "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    nd = sum(1 for v in rep["steps"].values() if v["state"] == "done")
    print(f"samples checked {len(rep['samples'])}, bad {rep['bad_samples']}; fresh draw unequal {rep['fresh_draw_unequal']}")
    print(f"done decode steps checked {nd}, to rerun {rep['rerun_done_steps']}")
    print(f"steps not done with bad leftovers {rep['bad_partial_steps']}; partial chunks reused "
          f"{rep['reused_partial_chunks']}")
    print(f"stage A: {rep['stage_A']['files_compared']} files vs results, mismatches {rep['stage_A']['mismatch']}; "
          f"{rep['stage_A']['sample_leaves']} sample leaves, problems {rep['stage_A']['sample_problems']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 3:
        raise SystemExit(__doc__)
    main(*a)
