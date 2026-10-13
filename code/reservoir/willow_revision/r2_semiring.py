"""R2 of the revision (review 1.4): a controlled change to degeneracy handling. POST HOC revision analysis, not a
pre-registered test; declared in notes/reservoir/willow_revision_declarations.md before its results were read.

reservoir, 2026-10-13. The change: decoder's sparse-frontier contraction of the RL DEM (code/decoder/ml_sparse.py: its
program, fault order, importance weights, pruning tau and cap kmax, and the ladder tau1 = 1e-6 -> tau2 = 1e-8 for
|P - 1/2| < 0.1) run in max-product instead of sum-product (mpe_sparse.py). Same program, same approximation, same
shots: only the way the fault sets of a logical class are combined changes (the maximum instead of the sum), so the
decision moves from the ML class to the class of the most probable error (MPE) under the same pruning. Mode 0 is the
sum-product (decoder's ML ladder itself), mode 1 the max-product.

Subcommands (python3 -I -B; the archive is downloaded data):
  test ARCHIVE DECODE STAGE1_DIR OUT_JSON [N]
      before the declaration; reads no outcome. (a) d = 5, one leaf per round count (q4_7/X): mode 0 on the first N
      stored ML shots against decoder's stored ladder, bit for bit (llr_t1, llr, logp, pred, redecoded), and the cost
      of modes 0 and 1; (b) d = 3, one leaf per round count (q4_5/X): mode 1 on the first N stage-1 shots against the
      exact MPE of stage 1 (class; ln M_1 - ln M_0) and mode 0 against decoder's exact ML (class; llr); the cost.
  run ARCHIVE DECODE STAGE1_DIR WORK SPEC GUARD_S
      SPEC: a JSON file with a list of jobs {"leaf": rel, "modes": [1] or [0, 1], "shots": "stored" | "every25" |
      "stage1"}, run in the listed order. "stored" is the stored ML idx (every 5th shot at d = 5), "every25" its
      shots with idx % 25 == 0, "stage1" the idx of stage 1's exact MPE (every 5th shot at d = 3). Per job, mode and
      chunk of 500 shots the ladder's outputs are saved as WORK/<leaf_>/m<mode>_c<first>.npz (the checkpoint: present
      chunks are skipped, so a restarted run resumes); once all chunks of a job and mode are present they are joined
      into WORK/<leaf_>/m<mode>.npz and the chunk files removed; WORK/<leaf_>/meta.json holds the DEM md5 (checked
      against decoder's summary for the stored run), parse_dem's merge count and the CPU. Stops before a chunk once
      this process's CPU time passes GUARD_S. Reads no outcome: the analysis does that.
Per shot and mode: pred (1 if X_1 > X_0 at the final rung, ties and NaN to 0, decoder's rule), llr = ln X_1 - ln X_0
(float32), lnx = ln X_c (float64; X = Z, the class probability, in mode 0 and M, the most probable fault set's
probability, in mode 1, on parse_dem's merged faults), the first rung's llr_t1 and lnx_t1, redecoded, maxk_t1, maxk_t2;
per chunk the CPU seconds of each rung.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from mpe_sparse import SemiringDecoder, parse_dem, summarize  # noqa: E402
from qrc.outpaths import out_path  # noqa: E402

ARCH = "google_105Q_surface_code_d3_d5_d7"
CM_RL = "correlated_matching_decoder_with_rl_optimized_prior"
ML5 = ("rl_s5", "ml_tau2_1e-6-1e-8_0.1.npz")
ML3 = ("rl", "ml_exact.npz")
T1, T2, BAND, KMAX = 1e-6, 1e-8, 0.1, 2_000_000
CHUNK = 500


def md5(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def dem_of(arch, rel):
    import stim
    p = os.path.join(arch, rel, "decoding_results", CM_RL, "error_model.dem")
    return stim.DetectorErrorModel.from_file(p), md5(p)


def merges(dem):
    """Error instructions that parse_dem merges into an earlier one with the same (detectors, observables) key."""
    seen, n = set(), 0
    for ins in dem.flattened():
        if ins.type != "error" or ins.args_copy()[0] == 0:
            continue
        ds, om = set(), 0
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                ds ^= {t.val}
            elif t.is_logical_observable_id():
                om ^= 1 << t.val
        k = (tuple(sorted(ds)), om)
        if k == ((), 0):
            continue
        n += k in seen
        seen.add(k)
    return n


def syndromes(arch, rel, idx, nd):
    a_dir = os.path.join(arch, rel)
    nsh = os.path.getsize(os.path.join(a_dir, "obs_flips_actual.b8"))  # one byte per shot for one observable
    raw = np.fromfile(os.path.join(a_dir, "detection_events.b8"), dtype=np.uint8).reshape(nsh, -1)
    return np.unpackbits(raw[idx], axis=1, bitorder="little")[:, :nd]


def shots_of(dec, stage1, rel, how):
    if how == "stage1":
        return np.load(os.path.join(stage1, "exact", rel.replace("/", "_") + ".npz"))["idx"].astype(np.int64)
    sub, fn = ML5 if rel.startswith("d5") else ML3
    idx = np.load(os.path.join(dec, ARCH, rel, sub, fn))["idx"].astype(np.int64)
    if how == "stored":
        return idx
    if how == "every25":
        return idx[idx % 25 == 0]
    raise SystemExit(f"unknown shot rule {how}")


def ladder(D1, D2, syn):
    """decoder's ml_tau2 protocol in D1's semiring; returns the outputs and the CPU seconds of each rung."""
    t0 = time.process_time()
    o1, s1 = D1.raw(syn)
    r1 = summarize(o1, s1)
    c1 = time.process_time() - t0
    res = {"pred": r1["pred"].copy(), "llr": r1["llr"].copy(), "lnx": r1["lnx"].copy(),
           "logp_like": r1["logp_like"].copy(), "llr_t1": r1["llr"].copy(), "lnx_t1": r1["lnx"].copy(),
           "maxk_t1": D1.last_maxk.copy(), "maxk_t2": np.zeros(len(syn), np.int64)}
    with np.errstate(invalid="ignore"):
        unc = ~(np.abs(r1["post1"] - 0.5) >= BAND)
    res["redecoded"] = unc
    idx = np.where(unc)[0]
    t0 = time.process_time()
    if len(idx):
        o2, s2 = D2.raw(syn[idx])
        r2 = summarize(o2, s2)
        for k in ("pred", "llr", "lnx", "logp_like"):
            res[k][idx] = r2[k]
        res["maxk_t2"][idx] = D2.last_maxk
    res["sec_t1"] = np.array(c1)
    res["sec_t2"] = np.array(time.process_time() - t0)
    return res


def decoders(dm, mode, prog=None):
    D1 = SemiringDecoder(dm, T1, KMAX, mode, prog)
    return D1, SemiringDecoder(dm, T2, KMAX, mode, D1.prog)


def cmd_test(arch, dec, stage1, out_json, n=100):
    n = int(n)
    out = {"label": "R2 pre-declaration test (reservoir, 2026-10-13): bitwise check of mode 0 against decoder's "
                    "stored ladder, mode 1 against stage 1's exact MPE at d = 3, and cost; no outcome read",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": n, "d5": {}, "d3": {}}
    for r in (10, 13, 30, 50):
        rel = f"d5_at_q4_7/X/r{r:02d}"
        dem, dmd5 = dem_of(arch, rel)
        dm = parse_dem(dem)
        z = np.load(os.path.join(dec, ARCH, rel, *ML5))
        idx = z["idx"][:n].astype(np.int64)
        syn = syndromes(arch, rel, idx, dem.num_detectors)
        D1, D2 = decoders(dm, 0)
        a = ladder(D1, D2, syn)
        E1, E2 = decoders(dm, 1, D1.prog)
        b = ladder(E1, E2, syn)
        eq = lambda x, y: bool(np.array_equal(x, y, equal_nan=True))  # noqa: E731
        out["d5"][rel] = {
            "dem_md5": dmd5, "merges": merges(dem), "shots": int(idx.size),
            "mode0_vs_stored_bitwise": {"llr_t1": eq(a["llr_t1"], z["llr_t1"][:n]), "llr": eq(a["llr"], z["llr"][:n]),
                                        "logp": eq(a["logp_like"], z["logp"][:n]),
                                        "pred": eq(a["pred"], z["pred"][:n, 0]),
                                        "redecoded": eq(a["redecoded"], z["redecoded"][:n]),
                                        "maxk_t1": eq(a["maxk_t1"], z["maxk_t1"][:n])},
            "mode0_cpu_ms_per_shot": 1e3 * float(a["sec_t1"] + a["sec_t2"]) / n,
            "mode1_cpu_ms_per_shot": 1e3 * float(b["sec_t1"] + b["sec_t2"]) / n,
            "mode1_redecoded": int(b["redecoded"].sum()), "mode1_nan": int(np.isnan(b["llr"]).sum()),
            "maxk_t1_mean": {"mode0": float(a["maxk_t1"].mean()), "mode1": float(b["maxk_t1"].mean())}}
        print(rel, json.dumps(out["d5"][rel]), flush=True)
    for r in (10, 13, 30, 50):
        rel = f"d3_at_q4_5/X/r{r:02d}"
        dem, dmd5 = dem_of(arch, rel)
        dm = parse_dem(dem)
        s1 = np.load(os.path.join(stage1, "exact", rel.replace("/", "_") + ".npz"))
        idx = s1["idx"][:n].astype(np.int64)
        syn = syndromes(arch, rel, idx, dem.num_detectors)
        D1, D2 = decoders(dm, 0)
        a = ladder(D1, D2, syn)
        E1, E2 = decoders(dm, 1, D1.prog)
        b = ladder(E1, E2, syn)
        ml = np.load(os.path.join(dec, ARCH, rel, *ML3))
        mpos = {int(i): k for k, i in enumerate(ml["idx"])}
        sel = np.array([mpos[int(i)] for i in idx])
        ex_llr = s1["ln_m1"][:n] - s1["ln_m0"][:n]
        fin = np.isfinite(ex_llr) & np.isfinite(b["llr"])
        mfin = np.isfinite(ml["llr"][sel]) & np.isfinite(a["llr"])
        out["d3"][rel] = {
            "dem_md5": dmd5, "merges": merges(dem), "shots": int(idx.size),
            "mode1_vs_exact_mpe": {"class_differs": int(np.sum(b["pred"] != s1["mpe"][:n])),
                                   "max_abs_llr_diff": float(np.max(np.abs(b["llr"][fin] - ex_llr[fin])))
                                   if fin.any() else None,
                                   "max_abs_lnM0_diff": float(np.nanmax(np.abs(b["lnx"][:, 0] - s1["ln_m0"][:n])))},
            "mode0_vs_exact_ml": {"class_differs": int(np.sum(a["pred"] != ml["pred"][sel].reshape(-1))),
                                  "max_abs_llr_diff": float(np.max(np.abs(a["llr"][mfin] - ml["llr"][sel][mfin])))
                                  if mfin.any() else None},
            "mode0_cpu_ms_per_shot": 1e3 * float(a["sec_t1"] + a["sec_t2"]) / n,
            "mode1_cpu_ms_per_shot": 1e3 * float(b["sec_t1"] + b["sec_t2"]) / n}
        print(rel, json.dumps(out["d3"][rel]), flush=True)
    with open(out_path(out_json), "w") as f:
        json.dump(out, f, indent=1)


def cmd_run(arch, dec, stage1, work, spec, guard):
    guard = float(guard)
    c0, w0 = time.process_time(), time.time()
    jobs = json.load(open(spec))

    def log(msg):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)

    log(f"start: {len(jobs)} jobs, guard {guard:.0f} CPU-s, nice {os.nice(0)}")
    for job in jobs:
        rel, modes, how = job["leaf"], job["modes"], job["shots"]
        wd = os.path.join(work, rel.replace("/", "_"))
        os.makedirs(wd, exist_ok=True)
        if all(os.path.exists(os.path.join(wd, f"m{m}.npz")) for m in modes):
            continue
        if time.process_time() - c0 > guard:
            log("guard reached; stopping")
            return
        dem, dmd5 = dem_of(arch, rel)
        sub = (ML5 if rel.startswith("d5") else ML3)[0]
        dec_md5 = json.load(open(os.path.join(dec, ARCH, rel, sub, "summary.json"))).get("dem_md5")
        if dec_md5 != dmd5:
            raise SystemExit(f"{rel}: DEM md5 {dmd5} is not decoder's {dec_md5}")
        dm = parse_dem(dem)
        idx = shots_of(dec, stage1, rel, how)
        syn = syndromes(arch, rel, idx, dem.num_detectors)
        mp = os.path.join(wd, "meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {"leaf": rel, "dem_md5": dmd5, "merges": merges(dem),
                                                               "shots_rule": how, "shots": int(idx.size), "cpu_s": {}}
        prog = None
        for m in modes:
            if os.path.exists(os.path.join(wd, f"m{m}.npz")):
                continue
            D1, D2 = decoders(dm, m, prog)
            prog = D1.prog
            for c in range(0, idx.size, CHUNK):
                fn = os.path.join(wd, f"m{m}_c{c:05d}.npz")
                if os.path.exists(fn):
                    continue
                if time.process_time() - c0 > guard:
                    log(f"guard reached in {rel} mode {m}; stopping")
                    json.dump(meta, open(mp, "w"), indent=1)
                    return
                res = ladder(D1, D2, syn[c:c + CHUNK])
                res["idx"] = idx[c:c + CHUNK].astype(np.int32)
                np.savez_compressed(fn + ".tmp.npz", **res)
                os.replace(fn + ".tmp.npz", fn)
                meta["cpu_s"][f"m{m}"] = meta["cpu_s"].get(f"m{m}", 0.0) + float(res["sec_t1"] + res["sec_t2"])
                json.dump(meta, open(mp + ".tmp", "w"), indent=1)
                os.replace(mp + ".tmp", mp)
            parts = [np.load(os.path.join(wd, f"m{m}_c{c:05d}.npz")) for c in range(0, idx.size, CHUNK)]
            keys = [k for k in parts[0].files if not k.startswith("sec_")]
            joined = {k: np.concatenate([p[k] for p in parts]) for k in keys}
            joined["sec_t1"] = np.array([float(p["sec_t1"]) for p in parts])
            joined["sec_t2"] = np.array([float(p["sec_t2"]) for p in parts])
            if not np.array_equal(joined["idx"], idx):
                raise SystemExit(f"{rel}: joined idx differs from the shot list")
            fn = os.path.join(wd, f"m{m}.npz")
            np.savez_compressed(fn + ".tmp.npz", **joined)
            os.replace(fn + ".tmp.npz", fn)
            for c in range(0, idx.size, CHUNK):
                os.remove(os.path.join(wd, f"m{m}_c{c:05d}.npz"))
            log(f"{rel} mode {m}: {idx.size} shots, {meta['cpu_s'][f'm{m}']:.0f} CPU-s "
                f"(process {time.process_time() - c0:.0f} CPU-s, wall {time.time() - w0:.0f} s)")
        json.dump(meta, open(mp, "w"), indent=1)
    log(f"all jobs done: process {time.process_time() - c0:.0f} CPU-s, wall {time.time() - w0:.0f} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "test" and len(a) in (5, 6):
        cmd_test(*a[1:])
    elif a and a[0] == "run" and len(a) == 7:
        cmd_run(*a[1:])
    else:
        raise SystemExit(__doc__)
