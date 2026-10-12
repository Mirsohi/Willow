"""Loaders for Google's public QEC datasets (Willow 105Q surface code, Zenodo 13273331; also the 2023
Sycamore release, Zenodo 6804040), written before the data was readable here, so they are deliberately
format-tolerant. Run `python3 willow_data.py inventory <root> [out.json]` first when the data arrives.

Reported layout of the 105Q zip (third-party reader code relayed by scout, 2026-10-07; not yet seen by us):
  google_105Q_surface_code_d3_d5_d7/d{d}_at_q{row}_{col}/{X,Z}/r{rounds:02d}/ with circuit_ideal.stim,
  circuit_noisy_si1000.stim, detection_events.b8, obs_flips_actual.b8, measurements.b8, sweep_bits.b8,
  metadata.json and decoding_results/<pathway>/{error_model.dem, obs_flips_predicted.b8}, pathways
  {correlated_matching,harmony}_decoder_with_{si1000,rl_optimized}_prior and libra_decoder_with_rl_optimized_prior.
  Round counts (Sim-to-Real 2609.04557 + Willow SI): 15 values incl. 1, 10, 13, 30, 50, 70, ..., up to 250.
  If only the zip is available, extract it to local disk first (5.7 GB).
Earlier expectation (from the papers and the 2023 release):
  one folder per experiment (patch, basis, rounds) containing
    circuit_ideal.stim, circuit_noisy*.stim (SI1000-decorated), *.dem (SI1000 / RL priors),
    measurements.b8|.01, sweep.b8|.01, detection_events.b8|.01, obs_flips_actual.01,
    obs_flips_predicted_by_*.01 or decoding_results/<decoder>/... , properties.yml|metadata.json
Detection events can be (re)computed from measurements + sweep bits with the ideal circuit
(stim m2d converter), which is what the Sim-to-Real study did; we do both and cross-check.
"""
import os, re, sys, json, glob
import numpy as np
import stim

SHOT_EXT = (".b8", ".01", ".r8", ".ptb64", ".hits", ".dets")


def _fmt(path):
    ext = os.path.splitext(path)[1]
    return {".b8": "b8", ".01": "01", ".r8": "r8", ".ptb64": "ptb64", ".hits": "hits", ".dets": "dets"}[ext]


def find_experiments(root):
    """Return sorted list of folders that contain at least one .stim circuit."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.endswith(".stim") for f in filenames):
            out.append(dirpath)
    return sorted(out)


def parse_name(path):
    """Best-effort metadata from a folder path, e.g. .../d3_at_q4_7/X/r10 or surface_code_bX_d3_r05_center_3_5.
    The deepest (last) match wins, so dataset-root names like ..._d3_d5_d7 do not leak in."""
    meta = {}
    s = path.replace("\\", "/")
    m = re.findall(r"(?:^|[/_])d(\d+)(?=[/_]|$)", s)
    if m: meta["distance"] = int(m[-1])
    m = re.findall(r"(?:^|[/_])b([XZ])(?=[/_]|$)", s) or re.findall(r"/([XZ])(?=/|$)", s)
    if m: meta["basis"] = m[-1]
    m = re.findall(r"(?:^|[/_])r(\d+)(?=[/_]|$)", s)
    if m: meta["rounds"] = int(m[-1])
    m = re.findall(r"(?:_at_|center_)([^/]+)", s)
    if m: meta["patch"] = m[-1]
    return meta


def read_properties(folder):
    meta = {}
    for name in ["properties.yml", "properties.yaml", "metadata.yml", "metadata.yaml"]:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            for line in open(p):
                if ":" in line and not line.strip().startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
    for name in ["metadata.json", "properties.json"]:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            try:
                meta.update(json.load(open(p)))
            except Exception as e:
                meta["_json_error"] = str(e)
    return meta


class Experiment:
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(os.listdir(folder))
        self.meta = parse_name(folder)
        self.meta.update({"props": read_properties(folder)})
        self.circuit_ideal = None
        stims = [f for f in self.files if f.endswith(".stim")]
        ideal = [f for f in stims if "ideal" in f] or stims
        self.circuit_ideal = stim.Circuit.from_file(os.path.join(folder, ideal[0]))
        self.noisy_circuits = {f: os.path.join(folder, f) for f in stims if "noisy" in f or "si1000" in f}
        self.dems = {}
        for f in glob.glob(os.path.join(folder, "**", "*.dem"), recursive=True):
            self.dems[os.path.relpath(f, folder)] = f
        self.num_detectors = self.circuit_ideal.num_detectors
        self.num_observables = self.circuit_ideal.num_observables
        self.num_measurements = self.circuit_ideal.num_measurements
        self.num_sweep_bits = self.circuit_ideal.num_sweep_bits

    def _find(self, stem):
        for ext in SHOT_EXT:
            p = os.path.join(self.folder, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def detection_events(self):
        p = self._find("detection_events")
        if p is None:
            return None
        return stim.read_shot_data_file(path=p, format=_fmt(p), num_detectors=self.num_detectors).astype(np.uint8)

    def obs_flips_actual(self):
        p = self._find("obs_flips_actual")
        if p is None:
            return None
        return stim.read_shot_data_file(path=p, format=_fmt(p), num_observables=self.num_observables).astype(np.uint8)

    def measurements(self):
        p = self._find("measurements")
        if p is None:
            return None
        return stim.read_shot_data_file(path=p, format=_fmt(p), num_measurements=self.num_measurements).astype(bool)

    def sweep_bits(self):
        p = self._find("sweep_bits") or self._find("sweep")
        if p is None or self.num_sweep_bits == 0:
            return None
        return stim.read_shot_data_file(path=p, format=_fmt(p), num_measurements=self.num_sweep_bits).astype(bool)

    def recompute_detection_events(self):
        """From measurement record (+ sweep bits) using the ideal circuit's DETECTOR/OBSERVABLE definitions."""
        m = self.measurements()
        if m is None:
            return None, None
        conv = self.circuit_ideal.compile_m2d_converter()
        sw = self.sweep_bits()
        kw = {"sweep_bits": sw} if sw is not None else {}
        det, obs = conv.convert(measurements=m, separate_observables=True, **kw)
        return det.astype(np.uint8), obs.astype(np.uint8)

    def predictions(self):
        """All shipped per-shot decoder predictions found under the folder: name -> (nshots, nobs) uint8."""
        out = {}
        for f in glob.glob(os.path.join(self.folder, "**", "*"), recursive=True):
            b = os.path.basename(f)
            if os.path.isfile(f) and "predicted" in b and os.path.splitext(b)[1] in SHOT_EXT:
                try:
                    out[os.path.relpath(f, self.folder)] = stim.read_shot_data_file(
                        path=f, format=_fmt(f), num_observables=self.num_observables).astype(np.uint8)
                except Exception as e:
                    out[os.path.relpath(f, self.folder)] = f"ERR {e}"
        return out

    def summary(self):
        return {"folder": self.folder, "files": self.files, "meta": self.meta,
                "num_detectors": self.num_detectors, "num_observables": self.num_observables,
                "num_measurements": self.num_measurements, "num_sweep_bits": self.num_sweep_bits,
                "dems": sorted(self.dems), "noisy_circuits": sorted(self.noisy_circuits)}


def inventory(root, out_json=None, check_shots=True, max_check=3):
    exps = find_experiments(root)
    rows = []
    for k, f in enumerate(exps):
        try:
            e = Experiment(f)
            row = e.summary()
            if check_shots and k < max_check:
                det = e.detection_events(); obs = e.obs_flips_actual()
                row["n_shots_det"] = None if det is None else int(det.shape[0])
                row["n_shots_obs"] = None if obs is None else int(obs.shape[0])
                if det is not None:
                    row["p_det"] = float(det.mean())
                d2, o2 = e.recompute_detection_events()
                if d2 is not None and det is not None:
                    row["recomputed_det_matches"] = bool(np.array_equal(d2, det))
                    row["recomputed_obs_matches"] = None if obs is None else bool(np.array_equal(o2, obs))
                row["predictions"] = {kk: (vv if isinstance(vv, str) else [int(vv.shape[0]), float(vv.mean())])
                                      for kk, vv in e.predictions().items()}
        except Exception as ex:
            row = {"folder": f, "error": repr(ex)}
        rows.append(row)
        print(json.dumps({k2: row.get(k2) for k2 in ["folder", "meta", "num_detectors", "dems", "n_shots_det", "p_det", "recomputed_det_matches", "error"]}), flush=True)
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=1, default=str)
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inventory":
        inventory(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)
