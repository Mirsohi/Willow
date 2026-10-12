# code/decoder: maximum-likelihood (tensor-network) decoding of surface-code memory data

All scripts: set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`
and run long jobs with `nice`. Numba caches compiled kernels; set `NUMBA_CACHE_DIR` to a local folder
(e.g. `/home/claude/work/decoder/numba_cache`) so the shared folder is not used for caches.
Python deps: stim, numpy, numba, scipy, pymatching (>=2.3 for correlated matching), beliefmatching,
ldpc (BP+OSD), tesseract-decoder, sinter (pulled in by beliefmatching/tesseract); cotengra only for the cross-check.

## Library
| file | what it is |
|---|---|
| `willow_sim.py` | SI1000(p) noise on Stim's rotated-surface-code memory circuits (Willow-like proxy at p ~ 3e-3: p_det ~ 7%). `python3 willow_sim.py` prints p_det |
| `dem_utils.py` | DEM parsing (exact symptoms: `^` components XOR-ed, duplicate symptoms merged), marginalisation onto a detector subset, sweep orders (stim-style snakes and, since 2026-10-09, sweeps along the 45° rotated axes of Willow's grid; the narrowest is used), frontier profiles |
| `ml_exact.py` | **exact** coset-ML decoder: variable elimination of the DEM tensor network along a space-time sweep (dense frontier table of 2^W x 2^nobs, numba). Returns P(class \| s) and log P(s). Refuses (MemoryError) when the table would exceed `max_bytes` (1.5 GB). Also brute-force enumeration for tiny DEMs |
| `ml_sparse.py` | **approximate** ML: the same sweep with a pruned sparse frontier (entries < tau x max dropped, at most kmax kept) and importance weights from the not-yet-swept faults (product-form backward messages). tau -> 0 is exact. The workhorse for d >= 5 |
| `ml_mps_fast.py` | approximate ML by boundary-MPS contraction (bond dimension chi, SVD cutoff, optional importance reweighting), numba. Independent chi-controlled cross-check |
| `ml_mps.py` | slow pure-numpy reference of the MPS decoder |
| `ml_cotengra.py` | **unfinished**: independent exact contraction (positive network, COPY/XOR tensors, cotengra path). Path search works, but `tree.contract` raises inside cotengra in this environment; not used. The exact decoder is instead cross-checked by brute force and by scout's independent code (`code/scout/ml_dp.py`) |
| `ml_exact_fast.py` | **exact** ML, fast kernel (Round 2): compact slot relabelling (the table spans only the open detectors) + 16-shot SIMD batches; bitwise-identical to `ml_exact`, 7-8x faster at d=3 (0.034-0.046 ms/round/shot). `decode_willow.py` uses it for `ml_exact` |
| `ml_fb_exact.py` | exact **forward-backward**: per-shot log P(s[,L]) and every fault posterior q_j = P(e_j=1\|s[,L]) (z-encoded backward table), plus likelihood scores via a Jacobian dp/dtheta |
| `ml_fb_fast.py` | same on the compact/batched layout (4x faster; validated against `ml_fb_exact` to 1e-15) |
| `mismatch.py` | synthetic ground truth: `si_family(ideal, p2, pm, p1, crosstalk)` (SI1000 generalisation), leakage-like bursts, graphlike DEM approximation, DEM->stim, calibration statistics (predicted vs observed LER, excess cross-entropy, reliability bins, z-scores) |
| `fit_noise.py` | likelihood fitting of noise-model families (circuit family; per-class scales on a given DEM) with exact scores, Fisher scoring, OPG/Hessian/sandwich errors, information-matrix test, per-class posterior calibration |
| `baselines.py` | MWPM, correlated MWPM (PyMatching 2), BeliefMatching, BP+OSD (ldpc), Tesseract wrappers with one interface |
| `willow_data.py` | format-tolerant loaders for Google's released data (circuits, DEMs, b8/01 shot files, recomputation of detection events from measurements + sweep bits, shipped decoder predictions). `python3 willow_data.py inventory <root> out.json` |

## Drivers
| file | how to run |
|---|---|
| `bench_sim.py` | `python3 bench_sim.py --d 3 --rounds 5 10 25 --p 3e-3 --basis Z --shots 20000 --decoders ml_exact,mwpm,corr_mwpm,beliefmatch,tesseract,bposd --out ../../results/decoder/sim`. ML options: `ml_exact`, `ml_sparse:<tau>[:<kmax>]`, `ml_mps:<chi>`, `ml_exact_zonly`. Per-decoder checkpoints + `summary.json` (LER, Wilson 1-sigma, eps per cycle, ms/shot, paired counts vs the first ML decoder) |
| `run_mismatch.py` | methods validation (a)+(b): `python3 run_mismatch.py --rounds 10 --shots 20000 --out ../../results/decoder/mismatch` (scenarios S0-S5, see docstring) |
| `run_fit.py` | methods validation (c): `python3 run_fit.py scenarios --nfit 4000 --mismatch ../../results/decoder/mismatch --out ../../results/decoder/fit` (scenario names: `S0`...`S5`, `S3b+xt` 4-parameter family, `<S>+dem` per-class scales on M's DEM, `<S>+demt` with classes split by first/last detector layer); `python3 run_fit.py coverage --truth S2 --reps 40 --nfit 1500 --out ../../results/decoder/fit/coverage` |
| `dem_class_truth.py` | reference values for the class-scale fits (true per-class ratios and their spread): `python3 dem_class_truth.py` -> `results/decoder/fit/dem_class_truth.json` |
| `run_speed.py` | sparse-decoder cost and certificates: `python3 run_speed.py d3 --rounds 10,50 --out ../../results/decoder/speed` (vs exact); `python3 run_speed.py ladder --d 5 --rounds 10 --out ...` (tau ladder on hard + random shots) |
| `analyze_synd_ll.py` | syndrome log-likelihood tests (mean and dispersion) before and after fitting, from stored per-shot ln P(s): `python3 analyze_synd_ll.py` -> `results/decoder/fit/synd_ll_tests.json` |
| `analyze_speed.py` | tau-ladder summaries and the adaptive-ladder replay: `python3 analyze_speed.py ../../results/decoder/speed` -> `speed_summary.json` |
| `fit_willow.py` | real data: per-class scale fit of a shipped DEM on the fit half of a leaf, held-out test on the other half; the Willow pipeline below runs it per leaf (queue usage in its docstring). Round-2 usage: `python3 -I -B fit_willow.py --root <dataset root> --filter d3_at,/Z/,r10 --dem rl --out ../../results/decoder/willow_fit` |
| `make_figures.py` | methods figures into `results/decoder/figures/` |
| `fetch_willow.sh` | (written by the orchestrator, 2026-10-08) fetches Zenodo record 13273331 in a session whose environment allows zenodo.org: resumable download, size and md5 checks against the record, copy into `shared/data/willow/` (zip, `zenodo_record.json`, `<zip>.listing.tsv`, `MANIFEST.json`), local extraction (refuses unsafe member paths; only `d3_at` members if disk is short) and the loader inventory -> `results/decoder/willow_inventory.json`. `bash fetch_willow.sh [local work dir]`; brief: `notes/orchestrator/willow-fetch-brief.md`; exit codes in its header |
| `decode_willow.py` | real-data batch decoding with chunk checkpoints (restartable); the Willow pipeline below runs it one leaf and DEM per call (`--leaf --dem-file --dem-tag --shots --work`, see its docstring). Older usage: `python3 decode_willow.py --root <dataset root> --out ../../results/decoder/willow --decoders ml_exact,mwpm,corr_mwpm,tesseract --dem <substring of a shipped .dem name, or circuit_si1000> --filter d3_at --max-shots 50000 --chunk 2000`. `--filter` matches the path relative to `--root`. Checks shipped detection events against recomputation, pairs every decoder with the shipped predictions (e.g. the NN); for ML decoders also writes the calibration statistics and, with `--synd-ref N`, the syndrome log-likelihood test against N fresh samples from the DEM. Filter with `d3_at` (the archive's top folder name contains d3_d5_d7) |

## Willow real-data pipeline (staged 2026-10-08; real run started 2026-10-08 18:13Z, restarted 2026-10-09 11:14Z after two bug fixes)
Pre-registration of the tests, thresholds and kill criteria: `notes/decoder/willow_prereg.md` (it records the md5 of the
frozen scripts below). The archive is untrusted: every Python that reads it runs with `-I -B`, the scripts live here
and never in the data directory, and `willow_stage.py stage` refuses unsafe members before extracting anything.

| file | role |
|---|---|
| `queue_real.sh` | the driver. `bash queue_real.sh [--workers 4] [--cpus 0,1,2,3] [--stages d3,fit,d5,d7] [--steps stage,inventory,plan,extract,queue,analysis]`. Real mode reads `shared/data/willow/` (exits 2 if `MANIFEST.json` is missing), works in `/home/claude/work/decoder/willow_run/` and writes `results/decoder/willow/` (`inventory.json`, `plan.json`, `decode/`, `fit/`, `queue_state.json`, `analysis/`). `--dry-run <dir>` runs every step on a fake archive in `<dir>/shared` with small shot counts and touches nothing shared. Every step is restartable: re-running skips finished work and resumes interrupted decodes from their chunk checkpoints |
| `willow_stage.py` | `stage`: local copy of the zip with md5 against `MANIFEST.json` and the Zenodo record; if the shared folder holds the zip as slices `<zip>.001`, `.002`, ... (the shared store does not reliably keep files above 256 MiB; the real zip came as 22 slices of at most 256 MiB), it is rebuilt from them, with each slice's size and md5 (from `MANIFEST.json` "slices") checked and then the whole zip's (a missing slice exits 2, any mismatch 3); member check (absolute or `..` paths, backslashes, control characters, symlinks, duplicates, compression ratio > 1000), extraction of the d=3 leaves plus the small files of all leaves (`--plan --stages d5,d7` later extracts only the leaves those stages need), disk check. `inventory`: the shared loader inventory if it covers every leaf of the member listing, else our own; adds shot counts from the listed file sizes. `plan`: the job list from the inventory: both shipped DEMs per leaf (files with the same md5 or the same content for ML collapse; a Libra RL file with other content is flagged, see the pre-registration, section 2), the shipped prediction pathways of each leaf (the real archive has no Libra at r = 1 and 13), core-hours per job, cheapest unit first, subset strides checked for initialization-bitstring balance. `estimate`: the same plan for the reported layout |
| `willow_queue.py` | the runner: N workers, one process group per job, one thread, nice 5, 3.5 GB address-space limit, optional CPU set; done markers are the result files, failed jobs are retried once, SIGTERM stops every job before it exits; `python3 -I -B willow_queue.py status --plan <plan> --out <res>` |
| `decode_willow.py` | per leaf and DEM: ML + baselines with chunk checkpoints, recompute check of detection events on every selected shot, shipped predictions, calibration, syndrome log-likelihood tests, tau-ladder statistics, CPU cost per round per shot (median over chunks), bitstring balance of the selection -> `decode/<leaf>/<dem tag>/` |
| `fit_willow.py` | per d=3 leaf (r in W5): class-scale fit of the RL DEM on 5,000 shots of the fit half (halves alternate within each initialization bitstring), exact decodes of the held-out half under the shipped and the fitted DEM; sparse classes are merged, a failed fit is retried and otherwise marked `fit_failed` -> `fit/<leaf>/rl_tsplit/` |
| `analyze_willow.py` | tables (markdown, CSV, JSON), figures (`fig_willow_*.png/pdf`) and the pre-registered decisions (`prereg_results.json`) from whatever is finished; a shipped decoder's series runs over the rounds where its predictions exist: `python3 -I -B analyze_willow.py --res <res> --out <res>/analysis --reported willow_reported.json` |
| `willow_reported.json` | the published numbers we compare with (Table S1, Table S5; reported) |
| `queue_real_r2.sh` | history: the Round-2 one-core queue, superseded by `queue_real.sh` |

**Start** (after the board/main.md notice): `bash code/decoder/queue_real.sh --steps stage,inventory,plan,extract`
(not timed at real size: rebuilding the 5.7 GB zip from its slices with the md5 checks, the d = 3 extraction, the
plan's DEM comparisons), then read the dataset README (`README/README.md` inside the
archive; the log prints its local path) and the plan notes, then `bash code/decoder/queue_real.sh --steps
queue,analysis`. Re-running the same command resumes. If the plan notes start with "LIBRA DEM DIFFERS", P1 needs ML
with Libra's own RL file before it can count (pre-registration, section 2); the analysis marks P1 provisional until then.

**Changes after the data landed** (pre-registration, section 9; frozen copies in `superseded/frozen_2026-10-08/`):
`willow_stage.py` reads the zip from slices (entry 1); `dem_utils.py` sweeps along Willow's rotated grid and takes the
round from the third detector coordinate (entry 3: cost only, d = 3 ML 0.25-0.29 -> 0.035 ms per round per shot,
identical predictions on 50,000 real shots); `fit_noise.py` takes the round from the third coordinate, so "@first"
is the first detector layer only (entry 4). The real data had no Libra-file exception (all DEM files of a prior are
byte-identical).
New files after the data landed, outside the frozen set, in `supplementary/` (both import the frozen
`analyze_willow.py` read-only; run with `nice python3 -I -B`, arguments in each docstring):
`p1_se_covariance.py` gives the full delta-method SE of g with the covariance of the two slope fits (entry 7);
`p3_heldout_fitted.py` is EXPLORATORY and compares the calibration of ML(T̂) and ML(RL) on the held-out halves
(entry 6). Outputs in `results/decoder/willow/supplementary/`.
K2's extra rung (entries 10 and 11), also in `supplementary/`: `k2_extra_rung.py` decodes a failing d = 5 ladder
leaf's band shots at τ3 = τ2/10 and applies section 6's rules (it imports the frozen decoders read-only);
`k2_runner.py` only schedules it, cheapest first, within the orchestrator's stop (16 core-hours, 28 since 12:15Z, 34
since 15:10Z, with a guard on the projection for all the leaves), and writes `k2_tally.json` once all 32 leaves have
verdicts. It runs `--parallel` leaves at a time: one until 19:33Z, 2 since (entry 11, addendum 3). Outputs in `results/decoder/willow/supplementary/k2_extra_rung/`.
Round 1's final analysis (entry 12), also in `supplementary/`: `final_round1.py` runs the frozen `analyze_willow.py`,
unchanged, on a view of its inputs that applies K2's exclusions. Rule A (chosen 2026-10-10): a still-failing leaf's
round leaves the ML(RL) fits. It adds:
- entry 7's SE for every g;
- the d = 5 statistics under rule A, with rule B and no exclusion as sensitivity rows;
- Λ_3/5 per decoder;
- entry 8's list.

It runs once `k2_tally.json` is complete; output `results/decoder/willow/final_round1/`.
`final_round1_trigger.py` waits for that complete tally and then starts `final_round1.py` (RELAUNCH §2.5).
Post hoc robustness check, prompted by the internal review (entry 13, logged 2026-10-10 22:04Z), also in
`supplementary/`: are ML(RL)'s decisions outside the ladder's band stable at d = 5?
- `nonband_select.py` fixes the shot lists by seed: every non-band shot where Libra and ML(RL) disagree, and a random
  2,500 where they agree.
- `nonband_check.py` decodes them at τ2, then τ3 (targeted and a random 10%), with a 4.0 core-hour guard.
- `nonband_analyze.py` reports the flips and the implied change in g_5 (rule A; no exclusion as sensitivity) and in
  the count gap, as an estimate and a 95% worst case.

Output `results/decoder/willow/supplementary/nonband_check/`. `post_mech_chain.py` only schedules: after the mech
fits it starts the in-sample decodes, `analyze_explore.py` and the check, in main's order (RELAUNCH §2.6).

**Willow round 2** (`round2/`, new files; pre-registration `notes/decoder/willow_prereg_r2.md`, PR2). They import the
frozen modules read-only. Run with `NUMBA_CACHE_DIR=<local scratch> python3 -I -B`, one thread per process; arguments
are in each docstring. Since 2026-10-10 10:47Z (PR2 §7): at most 3 processes while the queue runs (2 for mech, 1 for
K2), 4 after it, at normal priority. The queue finished at 19:06Z; since 19:33Z, 2 for mech and 2 for K2.
- `fb_cond.py`: exact forward–backward for ln P(L | s) and its score in one pass. The L = 1 backward table is the L = 0
  table with the logical bit flipped. Tested by `test_fb_cond.py` against the frozen `ml_fb_fast` and against finite
  differences.
- `cond_fit.py`: D̂, the F1 class scales fitted by Σ ln P(L|s), pooled per series. Fisher scoring with the centred OPG
  and BFGS updating (PR2 §7, 2026-10-10 01:19Z). After every accepted step it checkpoints to one file,
  `<prefix>.ckpt.npz` (PR2 §7, 02:44Z).
- `mech_fit.py`: the "mech" family (one log-odds shift per template) and the T̂_mech (ln P(s)) and D̂_mech
  (ln P(L|s) with a ridge, λ = 4) fits (PR2 §8), with F1's checkpoints and `stop` hook.
- Drivers:
  - `run_f1.py`: real F1 fits and held-out decodes. `--cpu-budget-hours 12` is the orchestrator's guard.
  - `run_mech.py`: real mech fits and held-out decodes. `--max-iter 25` is each fit's cap (15 in PR2 §8.2, raised at
    15:10Z; PR2 §7 entry 15:14Z), and `--cpu-budget-hours 30` is the orchestrator's guard (25 until 15:10Z).
  - `run_insample.py`: in-sample decodes of the fit shots under RL, T̂, D̂, T̂_mech and D̂_mech (PR2 §8.4).
- `RELAUNCH.md`: the commands that relaunch the active round-2 runs after a container restart. They resume from the
  checkpoints.
- Analysis:
  - `analyze_r2.py`: F1's pre-registered primary m_D and its secondaries.
  - `analyze_explore.py`: mech held out, and in-sample for all four fits; exploratory.
- Validation on simulated data:
  - `sim_validate.py`: F1, V0 and V1.
  - `sim_validate_mech.py`: mech, V0 and V2.
  - `test_optimizer.py`: the optimiser comparison behind the PR2 §7 deviation.

**PR3, d = 7** (`pr3/`; pre-registration `notes/decoder/willow_prereg_r3.md`, approved 2026-10-10 about 20:00Z, with
the choices in its §8). The decoding uses the frozen `decode_willow.py` and `k2_extra_rung.extra_rung`, both unchanged.
Run with `NUMBA_CACHE_DIR=<local scratch> python3 -I -B`; arguments are in each docstring.
- `pr3_common.py`: the leaves, shot selections and parts, a run under an RSS cap with its CPU time, and the cost
  projection fixed before the pilot.
- `pr3_extract.py`: copies the 6 d = 7 leaves from the archive into a new local extract (zip read-only), and the
  pilot's leaf without shipped predictions.
- `pr3_extra_rung.py`: the extra rung (τ3 = 1e-8) on one d = 7 ladder output, through `k2_extra_rung.extra_rung`.
- `pr3_pilot.py`: stage 0, 200 shots of d7/X/r50 through τ1, τ2 and τ3, and the projection and design by the
  orchestrator's rule. Output `results/decoder/willow/pr3/pilot/`; finished 2026-10-10 20:35Z.
- `pr3_runner.py`: the main run, 25,000 shots per leaf in 5 parts per leaf. It keeps at most 4 heavy processes on the
  machine and holds slots for K2, the in-sample decodes and reservoir's mech check. Leaf verdicts use rule 1 and
  rule 2′. Output `results/decoder/willow/pr3/main/` (`pr3_tally.json`); it started 2026-10-10 20:39Z and was
  restarted at 20:46Z (version 2), 21:14Z (version 3) and 21:18Z (version 4: holds; PR3 §8, runner updates 2–3).
  Version 5 (a slot for the non-band check; runner update 4) is on disk since 22:05Z; the restart was refused, and
  version 4 keeps running.
- `pr3_analyze.py`: H7 and the secondary rows, once the tally is complete and round 1's final analysis exists. Output
  `results/decoder/willow/pr3/analysis/` (`pr3_results.json`, `.md`). Fixed (md5 in PR3 §8) before any part ran.

PR3 planning, read-only and with no decoding. Each script writes one json into `/home/claude/work/decoder/pr3/`, with
copies in `results/decoder/willow/pr3_planning/`.
- `d5_gap_check.py`: rebuilds the d = 5 2x2 tables (Libra vs ML(RL), r in {10, 30, 50}) with the frozen functions and
  checks them against `analysis/analysis.json`; entry 7's SE of g_5; K2 exclusion scenarios; error-count changes
  between the ladder's rungs and the extra rung.
- `d7_numbers.py`: the shipped decoders' error counts on all 50,000 shots of the d = 7 leaves, read from the archive
  (zipfile, read only); Libra's eps7 against Table S1; the expected precision of g_7 on expected 2x2 tables.
- `pr3_power.py`: the Birge inflation at d = 5, and power and cost of the PR3 designs.

**Stages and cost** (`python3 -I -B willow_stage.py estimate`; the layout the fetch session read from the zip's central
directory: 9 / 4 / 1 patches at d = 3 / 5 / 7, X and Z, 15 round counts 1, 10, 13, 30, 50, 70, ..., 250, 50,000 shots per
folder, no Libra at r = 1 and 13; CPU unit costs below). The plan made from the real inventory prints the same table for
the real archive. Local disk: the 5.7 GB zip, the d = 3 leaves (4.1 GB of the 12.8 GB uncompressed) and the small files
of every leaf, then the W5 d = 5 leaves and the two d = 7 probe leaves (both a small share of their 5.7 and 2.9 GB);
`stage` checks free space before it writes.

| stage | what | jobs | core-h | h on 4 workers |
|---|---|---|---|---|
| d3 | 270 leaves x {SI1000, RL} DEM: `ml_exact`, `corr_mwpm`, `mwpm` on all shots + 10^4 syndrome reference samples; Tesseract and BeliefMatching on every 5th shot for r <= 50 | 720 | 48.9 | 12.2 |
| fit | 72 leaves (r in W5) x RL DEM: class-scale fit + held-out decodes | 72 | 5.7 | 1.4 |
| d5 | 32 leaves (r in W5) x RL DEM, every 5th shot: `ml_tau2:1e-6/1e-8:0.1`, `corr_mwpm`, `mwpm` + 2,000 reference samples; Tesseract for r <= 30 | 56 | 31.9 | 8.0 |
| d7 | probe: X and Z at the two smallest r >= 10, every 25th shot, `ml_tau2:1e-6/1e-7:0.1`; each job capped at 1.5 core-h | 4 | 2.6 (cap 6) | 0.7 |
| total | | 852 | 89 | 22 |

Unit costs (CPU ms per round per shot, one core): `ml_exact` 0.035 (Round-2 timing 0.023-0.037 at r = 10-250);
`corr_mwpm`, `mwpm` < 0.002 (measured; 0.002 d²/9 used); Tesseract 0.27 / 0.36 / 0.58 / 0.76 at d = 3 and r = 10 / 13 /
30 / 50, and 5.2 / 8.6 at d = 5 and r = 10 / 30; BeliefMatching 0.15 / 0.21 / 0.32 / 0.40 at d = 3 (medians over the
dry-run leaves on the fake archive); tau ladder 8.5 at d = 5 (dry run: 6.1-13.7 on 40-shot leaves) and 103 at d = 7
(dry run: 64-293 on 10-shot leaves; the probe is capped, so this sets how many shots it reaches, not what it costs);
fit: 5 forward-backward passes (0.3 each) on the 5,000-shot fit sample plus 4 exact decodes of the held-out half.

**Check at real scale** (2026-10-08; simulated 50,000-shot leaves from `tests/make_fake_willow3.py`; one core, the
queue's 3.5 GB address-space limit; `tests/scale_check.py`, results in `results/decoder/willow_dryrun/scale_check.json`). Measured CPU time against the
estimate, and peak memory:

| job | CPU s (estimate) | peak MB |
|---|---|---|
| d=3 r=50 main: `ml_exact`, `corr_mwpm`, `mwpm` on 50,000 shots + 10,000 reference samples | 88 (115) | 233 |
| d=3 r=50 subset: Tesseract + BeliefMatching on 10,000 shots | 572 (580) | 176 |
| d=3 r=50 fit: 21 classes, 5,000 fit shots, 25,000 held-out shots | 335 (550) | 930 |
| d=5 r=10 ladder: `ml_tau2:1e-6/1e-8:0.1`, `corr_mwpm`, `mwpm` on 10,000 shots + 2,000 reference samples | 842 (1022) | 245 |

The d = 5 ladder there re-decoded 0.9% of the shots at τ2, changed 2 decisions (errors 245 -> 247) and moved no
posterior by more than 0.045, so it passes the K2 convergence rule; τ1 cost 6.0 ms per round per shot.

Four workers together stay under 4 GB.

**Order** (the plan's; `--stages` only selects): d=3 with the RL DEM (done after 24.4 core-h, about 6 h on 4 workers),
the fits (30.1 core-h), d=3 with the SI1000 DEM (54.5), d=5 (86.4), d=7 (89.1). The bar and the kill criterion K1 need
G1, P1, P3 and P4, which are all in after about 7.5 h, so a kill can stop the run before most of its cost. The first
complete d=3 unit (one patch, basis and DEM, all round counts) costs 1.4 core-h, so complete tables arrive within the
first hour.

**K2 remediation** (pre-registered): a d=5 leaf that fails the ladder convergence check is decoded once more into a
separate tag that sorts after the original, e.g. `python3 -I -B decode_willow.py --root <extract> --out <res>/decode
--work <local>/q/chunks --leaf <leaf> --dem-file <file> --dem-tag rl_s5x --decoders corr_mwpm,mwpm,ml_tau2:1e-6/1e-9:0.1
--shots <same selection> --chunk 500 --synd-ref 2000`; `analyze_willow.py` reads the later tag last, so its ML results
replace the first ones for that leaf.

**Dry runs** (2026-10-08, fake archives from `tests/make_fake_willow3.py`, 114 jobs each, 4 workers pinned to one core):
`results/decoder/willow_dryrun/run2/` and `run3/` (analysis with report, figures and decisions; plan, inventory, queue
state, logs; decode and fit outputs as a tarball). Run 2 was stopped and resumed in the middle and had a partial shared
inventory, so the pipeline made its own. Run 3 used an archive shaped like the real one (no Libra at r = 1 and 13, a
`README/` folder, Libra's DEM file with the same content but another md5) and a complete shared inventory.

## Tests (`tests/`), outputs in `results/decoder/feasibility/`
| file | what it checks | how to run |
| `make_fake_willow2.py` | small fake archive in the reported layout (d = 3 and 5, r = 1 and 10) for the loader and the Round-2 queue | `python3 -I -B tests/make_fake_willow2.py` (writes `/home/claude/work/decoder/fake_willow2`) |
| `make_fake_willow3.py` | fake archive for the pipeline dry runs: d = 3 / 5 / 7 leaves, both bases, sweep-bit bitstrings, both shipped DEMs and the shipped predictions (no Libra at r = 1 and 13), a `README/` folder, the zip with listing, record and MANIFEST in `fetch_willow.sh`'s format, and an unsafe zip | `python3 -I -B tests/make_fake_willow3.py <dir>`, then `bash queue_real.sh --dry-run <dir>` |
| `scale_check.py` | the real-profile jobs on simulated 50,000-shot leaves under the queue's limits: CPU time and peak memory | `python3 -I -B tests/scale_check.py <dir>` |
| `power_g.py` | expected SE of the P1 statistic g for the reported layout (pre-registration, section 7) | `python3 -I -B tests/power_g.py` |
| `test_stage_slices.py` | `willow_stage.py stage` on the fake zip cut into slices: rebuild with and without the MANIFEST slice list, the whole-zip path, a verified local copy without the slices, and failures for a corrupted, truncated or missing slice and a wrong whole-zip md5 (exit codes, nothing left behind) | `python3 -I -B tests/test_stage_slices.py <fake workdir> <test dir>` (prints PASS per case) |
|---|---|---|
| `test_exact.py` | `ml_exact` vs brute force on 12 tiny DEMs | `python3 tests/test_exact.py` (prints PASS) |
| `test_approx_d3.py` | sparse (tau scan) and MPS (chi scan, with/without reweighting) vs exact, d=3 r=5, 300 shots | `python3 tests/test_approx_d3.py` |
| `test_sparse_d5.py` | d=5 r=5: Z-only exact vs sparse; joint sparse tau scan; baselines | `python3 tests/test_sparse_d5.py` |
| `test_exact_d5r2.py` | exact joint d=5 r=2 (W=23) on hard + random shots vs sparse tau scan | `python3 tests/test_exact_d5r2.py <nshots> <out.npz>` |
| `conv_subset.py` | small-tau convergence on the hardest shots of a `bench_sim` run | see docstring |
| `probe_d7.py` | d=7 feasibility: W, frontier size K, time per shot, tau convergence | `python3 tests/probe_d7.py <rounds> <shots> <budget_s> <out.json>` |
| `test_cotengra.py` | `ml_cotengra` vs `ml_exact` at d=3 (currently fails, see `ml_cotengra.py`) | `python3 tests/test_cotengra.py` |
| `test_mps_d5.py` | chi-controlled MPS vs sparse tau=1e-10 on the hardest d=5 shots | `python3 tests/test_mps_d5.py <conv_d5_r5.json> <n> <budget_s> <out.json>` |
| `probe_d7b.py` | d=7 r=10: cost, hard-shot tau convergence, p sensitivity | `python3 tests/probe_d7b.py <rounds> <N> <n_hard> <n_rand> <budget_s> <out.json>` |
| `test_fb_exact.py` | forward-backward marginals vs brute force, log P vs `ml_exact`, scores vs finite differences | `python3 tests/test_fb_exact.py` (PASS) |
| `test_fb_fast.py` | `ml_fb_fast` vs `ml_fb_exact` (log P, q, scores; joint and marginal) + timing | `python3 tests/test_fb_fast.py` (PASS) |
| `test_exact_fast.py` | `ml_exact_fast` vs brute force and vs `ml_exact` (d=3, r=10/30) + timing | `python3 tests/test_exact_fast.py` (PASS) |
| `test_adaptive_d3.py` | adaptive tau ladder (`AdaptiveSparseDecoder`) vs exact ML on random d=3 shots (r=10, 50): decision flips, cost | `python3 tests/test_adaptive_d3.py` (log: `results/decoder/speed/adaptive_d3_vs_exact.log`) |

