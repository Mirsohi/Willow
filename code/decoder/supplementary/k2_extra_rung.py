#!/usr/bin/env python3
"""K2's extra rung for failing d = 5 ladder leaves (notes/decoder/willow_prereg.md section 6; section 9 entry 10).

Section 6: per d = 5 leaf, the ladder ml_tau2:<tau1>/<tau2>:<band> converges if
  (1) the error count moves between the rungs by at most max(3, 1% of the final count), and
  (2) the largest |dP| over the re-decoded shots stays below the band.
A failing leaf gets one extra rung; the orchestrator chose "tau2/10 on the band shots" (2026-10-10). For each finished
d = 5 ladder leaf that fails (1) or (2), this script
  1. reads the ladder's stored output (<res>/decode/<leaf>/<tag>/<ml_tau2 ...>.npz: idx, redecoded, llr = final, pred),
     shots.npz (observed flips) and summary.json, and the leaf's shipped detection events from the local extract
     (run with python3 -I);
  2. rebuilds the ladder's decoder as decode_willow.make_decoder does: the same DEM file (md5 checked against
     summary.json), the tau1 decoder's compiled program shared by the deeper rungs, kmax = decode_willow.KMAX_LADDER;
  3. checks the rebuild: the first --check-shots band shots (default 50) re-decoded at tau2 must give the stored final
     llr exactly (float32, bitwise);
  4. decodes the band shots (the shots the ladder re-decoded at tau2) at tau3 = tau2 / 10, and only those;
  5. applies the two rules between the last two rungs (tau2 -> tau3) on the band shots:
     (1) |k3 - k2| <= max(3, 0.01 k3), with k2 the leaf's error count under the ladder's final decisions and k3 the count
         with the band shots' decisions taken from tau3;
     (2) max |P_tau3 - P_tau2| over the band shots < band (a NaN at tau3 fails it).
     A leaf passes if both hold. Otherwise it still fails: section 6 excludes and flags it (applied at the final
     analysis).
The ladder's stored predictions stay the leaf's predictions in the frozen analysis; for a passing leaf the extra rung
certifies them (its error count moves by at most max(3, 1%)). Nothing in decode/ or code/decoder is written.

Output (restartable: a leaf whose extra_rung.json matches the ladder output's md5 is skipped):
  <out>/<leaf>/<tag>/extra_rung.json   the rules, the check, the tau3 statistics, CPU cost, verdict
  <out>/<leaf>/<tag>/extra_rung.npz    idx (shot index in the leaf) of the band shots, llr_t3, logp_t3, maxk_t3
  <out>/summary.json                   one row per finished d = 5 ladder leaf: ladder rules, extra rung, verdict
(d = 7 is exploratory and has no convergence rule, so its ladder leaves are left out.)
Run (one process; NUMBA_CACHE_DIR keeps numba's cache out of code/decoder):
  NUMBA_CACHE_DIR=<local scratch> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 10 python3 -I -B k2_extra_rung.py \
      --root <local extract> --res results/decoder/willow --out results/decoder/willow/supplementary/k2_extra_rung \
      [--leaf <leaf rel path>[,...]] [--check-shots 50] [--dry-run]
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # code/decoder (frozen, read-only)
import numpy as np
import stim
from dem_utils import parse_dem
from ml_sparse import SparseFrontierDecoder
from willow_data import Experiment
from decode_willow import KMAX_LADDER, _ml_out           # frozen; read-only


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def ladder_leaves(res):
    """Every finished d = 5 ladder leaf: (leaf rel, tag, decoder name, summary)."""
    out = []
    root = os.path.join(res, "decode")
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if not d.startswith(("d3_", "d7_")))        # only d = 5 patches
        if "summary.json" not in fn:
            continue
        try:
            s = json.load(open(os.path.join(dp, "summary.json")))
        except Exception:
            continue
        if s.get("status") != "complete" or int(s.get("meta", {}).get("distance", 0) or 0) != 5:
            continue
        for name, ds in (s.get("decoders") or {}).items():
            if name.startswith("ml_tau2") and isinstance(ds, dict) and "ladder" in ds:
                out.append((s["experiment"], s["dem_tag"], name, s))
    return sorted(out, key=lambda x: (x[0], x[1]))


def rungs(name):
    parts = name.split(":")
    t1, t2 = (float(x) for x in parts[1].split("/")) if len(parts) > 1 else (1e-6, 1e-8)
    band = float(parts[2]) if len(parts) > 2 else 0.1
    return t1, t2, band


def ladder_rules(lad, band):
    e1, ef = lad["errors_tau1"], lad["errors_final"]
    r1 = abs(ef - e1) <= max(3, 0.01 * ef)
    r2 = lad["max_abs_dP_redecoded"] < band
    return {"errors_tau1": e1, "errors_final": ef, "rule1_errors": bool(r1), "max_abs_dP": lad["max_abs_dP_redecoded"],
            "rule2_dP": bool(r2), "passes": bool(r1 and r2)}


def extra_rung(a, rel, tag, name, summ, log):
    t1, t2, band = rungs(name)
    t3 = t2 / 10
    dn = name.replace(":", "_").replace("/", "-")
    fd = os.path.join(a.res, "decode", rel, tag)
    zpath = os.path.join(fd, dn + ".npz")
    z = np.load(zpath)
    sh = np.load(os.path.join(fd, "shots.npz"))
    idx = z["idx"].astype(np.int64)
    red = z["redecoded"].astype(bool)
    assert np.array_equal(idx, sh["idx"].astype(np.int64)), f"{rel}: ladder and shots.npz select different shots"
    obs = sh["obs"].astype(np.uint8)
    folder = os.path.join(a.root, rel)
    dem_path = os.path.join(folder, summ["job"]["dem_file"])
    assert md5_file(dem_path) == summ["dem_md5"], f"{rel}: DEM md5 differs from summary.json"
    exp = Experiment(folder)
    det_all, obs_all = exp.detection_events(), exp.obs_flips_actual()
    assert np.array_equal(obs_all[idx, 0].astype(np.uint8), obs), f"{rel}: observed flips differ from shots.npz"
    b = np.where(red)[0]
    detb = det_all[idx[b]]
    dm = parse_dem(stim.DetectorErrorModel.from_file(dem_path))
    D1 = SparseFrontierDecoder(dm, tau=t1, kmax=KMAX_LADDER)

    def deeper(tau):
        D = SparseFrontierDecoder.__new__(SparseFrontierDecoder)
        D.dm, D.tau, D.kmax, D.prog = dm, tau, KMAX_LADDER, D1.prog
        return D
    r = int(summ.get("rounds") or 0)
    nc = min(len(b), a.check_shots)
    log(f"{rel} [{tag}]: {len(idx)} shots, {len(b)} band shots, r = {r}; tau2 check at {t2:g} on {nc} of them")
    c0 = time.process_time()
    o2 = _ml_out(*deeper(t2).class_probs(detb[:nc]))
    sec2 = time.process_time() - c0
    stored = z["llr"][b]
    same = bool(np.array_equal(o2["llr"], stored[:nc], equal_nan=True))
    check = {"tau2": t2, "n_checked": int(nc), "n_band": int(len(b)), "identical": same,
             "max_abs_llr_diff": float(np.nanmax(np.abs(o2["llr"].astype(np.float64) - stored[:nc]))) if nc else 0.0,
             "cpu_seconds": sec2}
    if not same:
        log(f"  tau2 check FAILED (max |llr diff| {check['max_abs_llr_diff']:.3g}); tau3 not run")
        return {"check": check, "verdict": "not run: the rebuilt tau2 rung does not reproduce the ladder"}, None
    log(f"  tau2 check passed ({sec2:.0f} CPU s); tau3 = {t3:g} on the band shots")
    D3 = deeper(t3)
    c0 = time.process_time()
    post3, logp3 = D3.class_probs(detb)
    sec3 = time.process_time() - c0
    o3 = _ml_out(post3, logp3)
    P2 = 1 / (1 + np.exp(-stored.astype(np.float64)))
    P3 = 1 / (1 + np.exp(-o3["llr"].astype(np.float64)))
    dP = np.abs(P3 - P2)
    pred = z["pred"][:, 0].astype(np.uint8)
    k2 = int(np.sum(pred != obs))
    d3 = o3["pred"][:, 0].astype(np.uint8)
    k3 = k2 - int(np.sum(pred[b] != obs[b])) + int(np.sum(d3 != obs[b]))
    nan3 = int(np.isnan(P3).sum())
    r1 = abs(k3 - k2) <= max(3, 0.01 * k3)
    r2 = nan3 == 0 and (float(np.max(dP)) if len(b) else 0.0) < band
    res = {"tau3": t3, "n_band": int(len(b)), "errors_ladder_final": k2, "errors_with_tau3": k3,
           "rule1_errors": bool(r1), "rule1_limit": max(3, 0.01 * k3),
           "max_abs_dP_tau2_tau3": float(np.nanmax(dP)) if len(b) and nan3 < len(b) else None,
           "rule2_dP": bool(r2), "band": band, "n_dP_ge_band": int(np.sum(dP >= band)),
           "n_dP_gt_0.05": int(np.sum(dP > 0.05)), "decisions_changed_tau2_tau3": int(np.sum(d3 != pred[b])),
           "nan_tau3": nan3, "max_K_tau3": int(D3.last_maxk.max()) if len(b) else 0,
           "cpu_seconds_tau3": sec3, "ms_per_round_per_band_shot_tau3": 1e3 * sec3 / max(len(b), 1) / max(r, 1)}
    verdict = "pass" if (r1 and r2) else "still fails: excluded and flagged (section 6)"
    log(f"  tau3: errors {k2} -> {k3} (rule 1 {'pass' if r1 else 'FAIL'}), max |dP| {res['max_abs_dP_tau2_tau3']} "
        f"(rule 2 {'pass' if r2 else 'FAIL'}), {sec3:.0f} CPU s, max K {res['max_K_tau3']}: {verdict}")
    arrays = {"idx": idx[b].astype(np.int32), "llr_t3": o3["llr"], "logp_t3": o3["logp"], "maxk_t3": D3.last_maxk}
    return {"check": check, "tau3": res, "verdict": verdict}, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local extract of the archive")
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf", default=None, help="comma-separated leaf paths (default: every finished ladder leaf)")
    ap.add_argument("--dry-run", action="store_true", help="list the leaves and their ladder rules only")
    ap.add_argument("--check-shots", type=int, default=50, help="band shots re-decoded at tau2 for the rebuild check")
    a = ap.parse_args()
    me = md5_file(os.path.abspath(__file__))

    def log(msg):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)
    rows = []
    want = set(a.leaf.split(",")) if a.leaf else None
    for rel, tag, name, summ in ladder_leaves(a.res):
        t1, t2, band = rungs(name)
        lad = summ["decoders"][name]["ladder"]
        row = {"leaf": rel, "tag": tag, "decoder": name, "rounds": summ.get("rounds"), "n": lad["n"],
               "ladder": ladder_rules(lad, band)}
        if row["ladder"]["passes"]:
            row["verdict"] = "pass (ladder)"
            rows.append(row); continue
        od = os.path.join(a.out, rel, tag)
        pj = os.path.join(od, "extra_rung.json")
        zmd5 = md5_file(os.path.join(a.res, "decode", rel, tag, name.replace(":", "_").replace("/", "-") + ".npz"))
        old = json.load(open(pj)) if os.path.exists(pj) else None
        if old and old.get("ladder_npz_md5") == zmd5:
            row.update(extra=old.get("tau3"), check=old.get("check"), verdict=old["verdict"])
            rows.append(row); continue
        if a.dry_run or (want is not None and rel not in want):
            row["verdict"] = "fails the ladder rules; extra rung pending"
            rows.append(row); continue
        rec, arrays = extra_rung(a, rel, tag, name, summ, log)
        rec.update(leaf=rel, tag=tag, decoder=name, ladder=row["ladder"], ladder_npz_md5=zmd5, script_md5=me,
                   kmax=KMAX_LADDER, made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if arrays is not None:
            os.makedirs(od, exist_ok=True)
            np.savez_compressed(os.path.join(od, "extra_rung.tmp.npz"), **arrays)
            os.replace(os.path.join(od, "extra_rung.tmp.npz"), os.path.join(od, "extra_rung.npz"))
        write_json(pj, rec)
        row.update(extra=rec.get("tau3"), check=rec["check"], verdict=rec["verdict"])
        rows.append(row)
    n_d5 = sum(1 for x in rows if "/d5_" in x["leaf"])
    failing = [x["leaf"] for x in rows if x["verdict"].startswith("still fails")]
    pending = [x["leaf"] for x in rows if "pending" in x["verdict"] or x["verdict"].startswith("not run")]
    summary = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_md5": me,
               "prereg": "notes/decoder/willow_prereg.md section 6; section 9 entry 10",
               "ladder_leaves_finished": len(rows), "d5_leaves_finished": n_d5, "still_failing": failing,
               "pending_or_not_run": pending, "rows": rows}
    if not a.dry_run:
        write_json(os.path.join(a.out, "summary.json"), summary)
    for x in rows:
        log(f"{x['leaf']} [{x['tag']}]: {x['verdict']}")
    log(f"{len(rows)} ladder leaves; still failing {len(failing)}; pending {len(pending)}")


if __name__ == "__main__":
    main()
