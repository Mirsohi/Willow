"""Simulated Willow leaves with the RL DEM as the true noise (notes/reservoir/sim_design.md, approved by main
10 Oct 2026, stages A + B). EXPLORATORY: post hoc question, pre-specified design.

reservoir, 2026-10-10. For each leaf of the plan, a folder with the real archive's layout is written in scratch,
<sim root>/google_105Q_surface_code_d3_d5_d7/<leaf>/:
  circuit_ideal.stim   copied from the real leaf (decode_willow.py reads the detector count and the round count from it)
  decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem   the real leaf's RL DEM, copied
  detection_events.b8, obs_flips_actual.b8   n shots from stim's DEM sampler on that DEM (seed from the plan)
  sample.json          seed, shots, md5 of every file, detection-event and observable-flip rates
No measurements, sweep bits or shipped predictions are written, so decode_willow.py decodes the simulated leaf exactly
as a real one, with --recompute-check none. A leaf whose sample.json matches (seed, shots, DEM md5) is left as it is.
The real archive is downloaded data: it is only read (stim's parsers), and this runs with python3 -I.
Usage: python3 -I sim_sample.py PLAN STAGE D   (samples every leaf of that stage and distance in the plan)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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


def sample_leaf(real_arch, sim_root, rel, n, seed):
    src = os.path.join(real_arch, rel)
    dst = os.path.join(sim_root, ARCH, rel)
    rec_path = os.path.join(dst, "sample.json")
    dem_md5 = md5(os.path.join(src, DEM_REL))
    if os.path.exists(rec_path):
        old = json.load(open(rec_path))
        if old.get("seed") == seed and old.get("shots") == n and old.get("dem_md5") == dem_md5:
            return old
    os.makedirs(os.path.join(dst, os.path.dirname(DEM_REL)), exist_ok=True)
    t0 = time.process_time()
    shutil.copyfile(os.path.join(src, "circuit_ideal.stim"), os.path.join(dst, "circuit_ideal.stim"))
    shutil.copyfile(os.path.join(src, DEM_REL), os.path.join(dst, DEM_REL))
    dem = stim.DetectorErrorModel.from_file(os.path.join(dst, DEM_REL))
    circ = stim.Circuit.from_file(os.path.join(dst, "circuit_ideal.stim"))
    if circ.num_detectors != dem.num_detectors or circ.num_observables != dem.num_observables or dem.num_observables != 1:
        raise SystemExit(f"{rel}: circuit and DEM disagree ({circ.num_detectors}/{dem.num_detectors} detectors, "
                         f"{circ.num_observables}/{dem.num_observables} observables)")
    det, obs, _ = dem.compile_sampler(seed=seed).sample(shots=n)
    for name, data, kw in (("detection_events.b8", det, {"num_detectors": dem.num_detectors}),
                           ("obs_flips_actual.b8", obs, {"num_observables": dem.num_observables})):
        tmp = os.path.join(dst, name + ".tmp")
        stim.write_shot_data_file(data=data, path=tmp, format="b8", **kw)
        os.replace(tmp, os.path.join(dst, name))
    # read back through stim's reader, as decode_willow.py will
    det2 = stim.read_shot_data_file(path=os.path.join(dst, "detection_events.b8"), format="b8",
                                    num_detectors=dem.num_detectors)
    obs2 = stim.read_shot_data_file(path=os.path.join(dst, "obs_flips_actual.b8"), format="b8", num_observables=1)
    if not (np.array_equal(det2, det) and np.array_equal(obs2, obs)):
        raise SystemExit(f"{rel}: b8 read-back mismatch")
    rec = {"leaf": rel, "seed": seed, "shots": n, "dem_md5": dem_md5,
           "num_detectors": dem.num_detectors, "p_det": float(det.mean()), "obs_flip_rate": float(obs.mean()),
           "md5": {f: md5(os.path.join(dst, f)) for f in ("circuit_ideal.stim", DEM_REL, "detection_events.b8",
                                                             "obs_flips_actual.b8")},
           "stim": stim.__version__, "cpu_s": time.process_time() - t0}
    with open(rec_path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(rec_path + ".tmp", rec_path)
    return rec


def main(plan_path, stage, d):
    plan = json.load(open(plan_path))
    for L in plan["leaves"]:
        if L["stage"] == stage and L["d"] == int(d):
            rec = sample_leaf(plan["paths"]["real_arch"], plan["paths"]["sim_root"], L["leaf"], L["shots"], L["seed"])
            print(f"sampled {L['leaf']}: {rec['shots']} shots, p_det {rec['p_det']:.4f}, obs {rec['obs_flip_rate']:.4f}",
                  flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 3:
        raise SystemExit(__doc__)
    main(*a)
