# Fake dataset mimicking the REPORTED Willow 105Q layout (scout 12:27Z), to test loaders + pipeline end to end.
import sys, os, json, shutil
sys.path.insert(0, '/mnt/project-files/tensor-networks/deep-dives/experimental/code/decoder')
import numpy as np, stim
from willow_sim import si1000
base = "/home/claude/work/decoder/fake_willow2"
shutil.rmtree(base, ignore_errors=True)
root = os.path.join(base, "google_105Q_surface_code_d3_d5_d7")
rng = np.random.default_rng(0)
for d, patch in [(3, "q4_5"), (5, "q6_5")]:
    for basis in "XZ":
        for r in [1, 10]:
            f = os.path.join(root, f"d{d}_at_{patch}", basis, f"r{r:02d}")
            os.makedirs(f, exist_ok=True)
            ideal = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=r)
            noisy = si1000(ideal, 3e-3)
            open(os.path.join(f, "circuit_ideal.stim"), "w").write(str(ideal))
            open(os.path.join(f, "circuit_noisy_si1000.stim"), "w").write(str(noisy))
            dem = noisy.detector_error_model(decompose_errors=True)
            m = noisy.compile_sampler(seed=1).sample(3000)
            det, obs = noisy.compile_m2d_converter().convert(measurements=m, separate_observables=True)
            for pw, scale in [("correlated_matching_decoder_with_si1000_prior", 1.0), ("correlated_matching_decoder_with_rl_optimized_prior", 1.3),
                              ("libra_decoder_with_rl_optimized_prior", 1.3)]:
                g = os.path.join(f, "decoding_results", pw); os.makedirs(g, exist_ok=True)
                open(os.path.join(g, "error_model.dem"), "w").write(str(dem if scale == 1.0 else stim.DetectorErrorModel(
                    "\n".join(l if not l.startswith("error(") else f"error({min(0.5, float(l[6:l.index(')')]) * scale)})" + l[l.index(')') + 1:] for l in str(dem).splitlines()))))
                stim.write_shot_data_file(data=obs ^ (rng.random(obs.shape) < 0.05), path=os.path.join(g, "obs_flips_predicted.b8"), format="b8", num_observables=1)
            stim.write_shot_data_file(data=m, path=os.path.join(f, "measurements.b8"), format="b8", num_measurements=noisy.num_measurements)
            stim.write_shot_data_file(data=det, path=os.path.join(f, "detection_events.b8"), format="b8", num_detectors=noisy.num_detectors)
            stim.write_shot_data_file(data=obs, path=os.path.join(f, "obs_flips_actual.b8"), format="b8", num_observables=1)
            json.dump({"distance": d, "basis": basis, "rounds": r, "fake": True}, open(os.path.join(f, "metadata.json"), "w"))
print("ok")
