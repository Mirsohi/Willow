"""Fake Willow 105Q archive for dry runs of the real-data pipeline (stage -> inventory -> plan -> queue -> fit ->
analysis). It mimics the layout relayed by scout (third-party reader code, 2026-10-07; not yet seen by us):

  google_105Q_surface_code_d3_d5_d7/README.md
  google_105Q_surface_code_d3_d5_d7/d{d}_at_q{row}_{col}/{X,Z}/r{rounds:02d}/
      circuit_ideal.stim (with sweep-controlled data-qubit initialisation, like Google's random initial bitstrings),
      circuit_noisy_si1000.stim, measurements.b8, sweep_bits.b8, detection_events.b8, obs_flips_actual.b8,
      metadata.json, decoding_results/<pathway>/{error_model.dem, obs_flips_predicted.b8}
  pathways: {correlated_matching,harmony}_decoder_with_{si1000,rl_optimized}_prior, libra_decoder_with_rl_optimized_prior
  google_105Q_surface_code_d3_d5_d7/README/{README.md, patches.png, logicals.png}
As in the real archive (layout read from its central directory by the fetch session, 2026-10-08, relayed to us), the Libra
folder is missing at r = 1 and 13. Libra's error_model.dem holds the RL DEM with its instructions in another order (same
model for ML, different md5), to exercise the plan's DEM content comparison.

Noise (all simulated, nothing here is real data): the shots come from a "truth" = SI1000-family circuit noise with
p = 3.5e-3 and measurement flips x1.3, in 10 blocks of shots with 10 random initial bitstrings (sweep bits). The shipped
DEMs: si1000 = SI1000(3e-3) (rates off), rl = the truth's DEM with every rate multiplied by a random factor in
[0.85, 1.18] (same structure, different rates, as Arms et al. report for Google's two DEMs). Shipped predictions:
correlated matching on each DEM, "harmony" = plain MWPM stand-in on each DEM, "libra" = correlated matching on the
truth's DEM. Detection events and observable flips are computed from measurements + sweep bits with the ideal circuit.

Usage: python3 make_fake_willow3.py <workdir> [--shots3 2000] [--shots5 1000] [--shots7 500]
  -> <workdir>/src/google_105Q_surface_code_d3_d5_d7/...      (the tree)
     <workdir>/shared/google_105Q_surface_code_d3_d5_d7.zip  (+ zenodo_record.json, <zip>.listing.tsv, MANIFEST.json,
                                                              in the format fetch_willow.sh writes)
     <workdir>/shared/evil.zip                               (an archive with an unsafe member, for the safety test)
"""
import argparse, hashlib, json, os, shutil, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import stim
from mismatch import si_family

TOP = "google_105Q_surface_code_d3_d5_d7"
PATCHES = {3: ["q4_5", "q6_7"], 5: ["q4_7"], 7: ["q6_7"]}
ROUNDS = {3: [1, 10, 13, 30, 50], 5: [1, 10, 13, 30, 50], 7: [10, 30]}
NO_LIBRA_ROUNDS = (1, 13)


def tiny_png():
    import struct, zlib
    def chunk(t, b):
        return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80")) + chunk(b"IEND", b""))


def insert_before_first_tick(circ_text, line):
    s = circ_text.splitlines()
    i = s.index("TICK")
    return "\n".join(s[:i] + ([line] if line else []) + s[i:]) + "\n"


def data_qubits(ideal):
    last = [l for l in str(ideal).splitlines() if l.startswith("M ") or l.startswith("MX ")][-1]
    return [int(t) for t in last.split()[1:]]


def rescale_dem(dem, rng, lo, hi):
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            p = min(0.49, inst.args_copy()[0] * float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
            out.append("error", p, inst.targets_copy())
        else:
            out.append(inst)
    return out


def leaf(root, d, patch, basis, r, nshots, rng, seed):
    f = os.path.join(root, TOP, f"d{d}_at_{patch}", basis, f"r{r:02d}")
    os.makedirs(f, exist_ok=True)
    ideal0 = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=r)
    dq = data_qubits(ideal0)
    gate, pauli = ("CX", "X") if basis == "Z" else ("CZ", "Z")
    sweep_line = f"{gate} " + " ".join(f"sweep[{k}] {q}" for k, q in enumerate(dq))
    ideal = stim.Circuit(insert_before_first_tick(str(ideal0), sweep_line))
    p = 3.5e-3
    truth0 = si_family(ideal0, p, 5 * p * 1.3, p / 10)
    si0 = si_family(ideal0, 3e-3, 15e-3, 3e-4)
    noisy_si = stim.Circuit(insert_before_first_tick(str(si0), sweep_line))
    truth_sw = stim.Circuit(insert_before_first_tick(str(truth0), sweep_line))
    dem_si = noisy_si.detector_error_model(decompose_errors=True)
    dem_truth = truth_sw.detector_error_model(decompose_errors=True)
    dem_rl = rescale_dem(dem_truth, rng, 0.85, 1.18)
    # shots: 10 blocks with 10 random initial bitstrings, as in Google's 10 x 5000 layout
    blocks = np.array_split(np.arange(nshots), 10)
    M, SW = [], []
    for k, b in enumerate(blocks):
        bits = rng.integers(0, 2, len(dq)).astype(bool)
        expl = f"{pauli} " + " ".join(str(q) for q, x in zip(dq, bits) if x) if bits.any() else ""
        c = stim.Circuit(insert_before_first_tick(str(truth0), expl))
        M.append(c.compile_sampler(seed=seed + k).sample(len(b)))
        SW.append(np.tile(bits, (len(b), 1)))
    m = np.concatenate(M); sw = np.concatenate(SW)
    det, obs = ideal.compile_m2d_converter().convert(measurements=m, sweep_bits=sw, separate_observables=True)
    open(os.path.join(f, "circuit_ideal.stim"), "w").write(str(ideal))
    open(os.path.join(f, "circuit_noisy_si1000.stim"), "w").write(str(noisy_si))
    stim.write_shot_data_file(data=m, path=os.path.join(f, "measurements.b8"), format="b8", num_measurements=ideal.num_measurements)
    stim.write_shot_data_file(data=sw, path=os.path.join(f, "sweep_bits.b8"), format="b8", num_measurements=ideal.num_sweep_bits)
    stim.write_shot_data_file(data=det, path=os.path.join(f, "detection_events.b8"), format="b8", num_detectors=ideal.num_detectors)
    stim.write_shot_data_file(data=obs, path=os.path.join(f, "obs_flips_actual.b8"), format="b8", num_observables=1)
    json.dump({"distance": d, "basis": basis, "rounds": r, "patch": patch, "shots": int(nshots), "fake": True},
              open(os.path.join(f, "metadata.json"), "w"))
    import pymatching
    def corr(dem):
        mm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        return mm.decode_batch(det, enable_correlations=True).astype(bool)
    def plain(dem):
        return pymatching.Matching.from_detector_error_model(dem).decode_batch(det).astype(bool)
    paths = {"correlated_matching_decoder_with_si1000_prior": (dem_si, corr), "harmony_decoder_with_si1000_prior": (dem_si, plain),
             "correlated_matching_decoder_with_rl_optimized_prior": (dem_rl, corr), "harmony_decoder_with_rl_optimized_prior": (dem_rl, plain)}
    if r not in NO_LIBRA_ROUNDS:
        paths["libra_decoder_with_rl_optimized_prior"] = (dem_rl, None)
    for pw, (dem, fn) in paths.items():
        g = os.path.join(f, "decoding_results", pw)
        os.makedirs(g, exist_ok=True)
        if pw.startswith("libra"):      # same model, other instruction order
            lines = str(dem).splitlines()
            text = "\n".join([l for l in lines if not l.startswith("error")] + [l for l in lines if l.startswith("error")][::-1]) + "\n"
        else:
            text = str(dem)
        open(os.path.join(g, "error_model.dem"), "w").write(text)
        pred = corr(dem_truth) if fn is None else fn(dem)
        stim.write_shot_data_file(data=pred, path=os.path.join(g, "obs_flips_predicted.b8"), format="b8", num_observables=1)
    return float(det.mean())


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--shots3", type=int, default=2000)
    ap.add_argument("--shots5", type=int, default=1000)
    ap.add_argument("--shots7", type=int, default=500)
    a = ap.parse_args()
    src = os.path.join(a.workdir, "src"); sh = os.path.join(a.workdir, "shared")
    shutil.rmtree(src, ignore_errors=True); shutil.rmtree(sh, ignore_errors=True)
    os.makedirs(os.path.join(src, TOP)); os.makedirs(sh)
    os.makedirs(os.path.join(src, TOP, "README"))
    open(os.path.join(src, TOP, "README", "README.md"), "w").write("FAKE archive for pipeline dry runs (simulated data only).\n")
    for fn in ("patches.png", "logicals.png"):
        open(os.path.join(src, TOP, "README", fn), "wb").write(tiny_png())
    rng = np.random.default_rng(20261008)
    nsh = {3: a.shots3, 5: a.shots5, 7: a.shots7}
    seed = 1000
    for d in (3, 5, 7):
        for patch in PATCHES[d]:
            for basis in "XZ":
                for r in ROUNDS[d]:
                    pd = leaf(src, d, patch, basis, r, nsh[d], rng, seed); seed += 100
                    print(f"d{d} {patch} {basis} r{r}: p_det {pd:.4f}", flush=True)
    zname = TOP + ".zip"
    zp = os.path.join(sh, zname)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(os.path.join(src, TOP)):
            dirnames.sort()
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, src))
    size, h = os.path.getsize(zp), md5(zp)
    with zipfile.ZipFile(zp) as z, open(os.path.join(sh, zname + ".listing.tsv"), "w") as out:
        for i in z.infolist():
            out.write(f"{i.filename}\t{i.file_size}\n")
    rec = {"id": 0, "doi": "fake", "metadata": {"title": "FAKE Willow archive (dry run)", "version": "fake"},
           "files": [{"key": zname, "size": size, "checksum": "md5:" + h}]}
    json.dump(rec, open(os.path.join(sh, "zenodo_record.json"), "w"), indent=1)
    man = {"zenodo_record": "fake", "fetched_at": "fake",
           "files": [{"key": zname, "size": size, "md5": h, "url": "fake", "verified": "fake archive"}],
           "inventory": {"note": "fake"}}
    json.dump(man, open(os.path.join(sh, "MANIFEST.json"), "w"), indent=1)
    with zipfile.ZipFile(os.path.join(sh, "evil.zip"), "w") as z:
        z.writestr(TOP + "/README.md", "ok\n")
        z.writestr("../evil.txt", "this member escapes the extraction directory\n")
    print(f"{zp}: {size / 1e6:.1f} MB, md5 {h}")


if __name__ == "__main__":
    main()
