# code/reservoir: independent checks, mechanism tests and simulations for the Willow study

These are the Willow sections of the workspace's `code/reservoir/README.md`. The rest of that folder is the
reservoir-computing (QRC) project and is not in this repository, except `qrc/outpaths.py`, the path helper
every script here imports. Paths such as `results/...` are relative to the workspace root.

## Willow P1 check (`willow_check/`, 9 October 2026)
An independent check of decoder's first confirmatory Willow result (`notes/decoder/willow_prereg.md` §4–5, P1: Libra
against exact ML with the RL DEM at d = 3, 18 series × the 13 round counts r ≥ 10 with Libra predictions). The archive
is downloaded data, so every command runs as `nice -n 19 python3 -I` with 1 thread. Here
`A=/home/claude/work/decoder/willow_run/extract/google_105Q_surface_code_d3_d5_d7` (local to the container),
`D=results/decoder/willow/decode/google_105Q_surface_code_d3_d5_d7` and `O=results/reservoir/willow_check`, as absolute
paths. Writes go through `qrc/outpaths.py` as above.

| command | writes to `$O/` |
|---|---|
| `willow_check.py $A $D $O` | `leaves.csv` (counts, 2×2 table and checks per leaf), `series.csv`, `willow_check.json`: items (a)–(e) |
| `bootstrap_g.py $O` | `bootstrap_g.json`: shot bootstrap of the eps means and g (B = 2000, seed 12345) |
| `variants.py $A $D $O` | `variants.json`: ML eps on 13 and 14 round counts; SE of g under 16 conventions |
| `ties.py $A $D $O` | `ties.json`: near-tie shots of exact ML (∣LLR∣ below 1e-9 to 1e-2) |
| `compare_decoder.py $O results/decoder/willow/analysis_m2` | `comparison.json`: against decoder's analysis files, read only after the numbers above were stored |

Checked on 9 October 2026 (20:16–20:19Z) with `QRC_OUT` set to a scratch folder, after the last code change. The first
four commands took 56, 24, 7 and 48 s. Their outputs are identical to the stored files (`willow_check.json` apart from
its run time). `comparison.json` was written to scratch and then stored with `QRC_OVERWRITE=1`; the two are identical.

## Willow P3 mechanisms (`willow_mech/`, 9 October 2026; EXPLORATORY)
Why is the RL DEM miscalibrated at d = 3 (P3: ML(RL) makes 22% more errors than its posteriors predict)? Exploratory,
not pre-registered; no multiplicity correction. Brief from main, 9 Oct (after the P1 check). The archive is downloaded
data, so every command runs as `nice -n 19 python3 -I` with 1 thread. `A` and `D` are as in the P1 check above,
`L=/home/claude/work/reservoir/mech/leaves` and `S=.../mech/streaks` are scratch folders (outside the project tree, so
never redirected), and `O=results/reservoir/willow_mech`. decoder's `results/decoder/willow/fit/` is not read (except
by `mech_f2.py` and `mech_twin.py`, written after main's 22:17Z brief allowed it, which read only each leaf's held-out
`test_idx`).

| command | writes | what |
|---|---|---|
| `mech_leaves.py $A $D $L all 1000000 100000` | `$L/<patch>_<basis>_r<NNN>.npz` (270 files, 235 MB, not stored) | per shot: event count N, largest events in one round B, first/last-layer events F/L, two bulk windows, m = min(P, 1 − P), ML(RL) and Libra failures, ln P(s), block; per leaf: RL-DEM sample histograms (10⁶ samples for W5 leaves, 10⁵ otherwise) |
| `mech_pairs.py $A $O $S 200000` | `$O/pairs_leaves.csv`, `pairs.json`; `$S/*.npz` | W5: pair probabilities by class (data vs DEM samples), covariance of pairs that share no mechanism, streaks S, unconnected coincidences Ku, Ks2 |
| `mech_scaled.py $A $L $O 1000000` | `$O/scaled_leaves.csv`, `scaled.json` | W5: data against the RL DEM with all probabilities × c, c matched to the data's mean event count |
| `mech_summary.py $L $O $S` | `$O/events_leaves.csv`, `events_by_r.csv`, `blocks.csv`, `mech.json` | tasks 1–3: event counts by r, overconfident failures (and calibration of simple syndromes), blocks, shot order |
| `mech_hot.py $A $O` | `$O/hot_q8_5_Z_r030.json` | the hot blocks 8–9 of q8_5 Z r = 30: rates and pair classes by stabilizer |
| `mech_f2.py $A $D $F $O` | `$O/f2_candidates.json` | candidates for decoder's round-2 family F2, nothing fitted, on the fit halves only (`F` as in the P4 check below): short-range pairs that share no RL-DEM mechanism (excess q, z, observable flag, detector pairs per leaf); single first/last-layer events by stabilizer and by class |
| `mech_twin.py $A $D $F $O` | `$O/twin.json`, `twin_map.csv`, `twin_shots.npz` | the (7, 7) final-layer anomaly (brief from main after the P4 check): exact last-layer messages from the layered decoder of `willow_check/p4_redecode.py`, so that changing the last-layer faults (remove b, add the twin) is exact per shot; hinge shots by N; a map of the twin's fit over all 72 last-layer cells of the 18 W5 series; persistence by r and block; the twin's probability fitted on the fit halves, then tested on the held-out j-shots (errors, R, log-loss), with controls; the Z-basis pair (6,10)–(8,8) |

Run times: 881, 424, 179, 16 and 2 s. Checks: `mech.json` reproduces P3's R = 1.2170428668904176 (509,651 errors,
418,761.749 predicted) and decoder's detection probabilities (to 3e-14). Two leaves regenerated into a fresh folder
equal the stored npz arrays; pairs, scaled and summary were first run into scratch (`QRC_OUT`), then stored with
`QRC_OVERWRITE=1` (21:16–21:30Z): all nine stored files are identical (md5) to the scratch runs, and so are the 72
streak files. Block order is file order; whether it is acquisition order is not known. `mech_f2.py` (22:51Z, 67 s) was
run into scratch and then stored (22:52Z); the two are identical (md5 eee84d015c5df7bfb17147ad7e9ae52e).
`mech_twin.py` (md5 49fe304b3f46c93ca7ada8cdc4201e5f; reads `f2_candidates.json`) ran into scratch at 23:31–23:40Z (553 s).
The store run was killed by the 23:43Z container restart before writing anything and was re-run at 23:47–23:55Z (463 s).
The stored `twin_map.csv` and `twin_shots.npz` are identical (md5) to the scratch run, and `twin.json` differs from it only
in its two run-time fields. On all 303,912 shots it decodes, the forward pass reproduces decoder's stored ML(RL)
predictions exactly, the stored llr to 4.8e-7 (float32) and ln P(s) to 2.8e-13.

## Willow P4 check (`willow_check/`, 9 October 2026)
An independent check of decoder's milestone-3 result P4 (`notes/decoder/willow_prereg.md` §2, §4–5: ML(RL) against
ML(T̂) on the held-out halves of the 72 W5 leaves, T̂ the RL DEM with class scales fitted on 5,000 shots per leaf) and of
the decomposition on the 54 leaves with Libra predictions. Brief from main, 9 Oct (22:17Z decision). `A`, `D` and `O`
are as in the P1 check, `F=results/decoder/willow/fit/google_105Q_surface_code_d3_d5_d7`; every command runs as
`nice -n 19 python3 -I` with 1 thread.

| command | writes to `$O/` |
|---|---|
| `p4_check.py $A $D $F $O` | `p4_leaves.csv` (per leaf: held-out errors of ML(RL), ML(T̂) and Libra, 2×2 cells, checks), `p4_check.json`: (a) counts, (b) m with the §4 SE, (c) the decomposition in counts, (d) checks; r = 13 sensitivity, Holm with P1 |
| `p4_redecode.py $A $F $O` | `p4_redecode.json`: 2,000 held-out shots of q8_5 X r = 10 (every 12th) re-decoded under the RL DEM and under T̂ by an exact layered ML decoder written here |
| `compare_p4.py $O results/decoder/willow/analysis_m3 results/decoder/willow/supplementary $F` | `p4_comparison.json`: against decoder's files, read only after the first two outputs were stored; also the held-out R and R_conf of both decoders and the BHHH histories |

The held-out split is worked out from the files: within each initialisation bitstring (the 9 sweep bits), the shots of
odd rank are held out; `fit_idx` lies in the even ranks. T̂'s classes are rebuilt from the RL DEM by the §2 / §9-entry-4
rule and equal the fit's record on all 72 leaves. The first two commands ran into scratch (`QRC_OUT`, 22:29Z) and were
stored with `QRC_OVERWRITE=1` (22:30–22:31Z, 57 and 10 s); the stored files equal the scratch runs apart from their
run-time fields. Their md5s were recorded at 22:31:00Z, before decoder's P4 material was read: `p4_check.json`
01385bc81079a39fa3793a645081f7cb, `p4_leaves.csv` 2e0da24cdbb2eb5f6b468a595cd823aa, `p4_redecode.json`
66869be6ee43fa0f247e11af32e5a0e3. `p4_comparison.json` (ff67b3812be302c6c2945a605939b069) equals its scratch run.

## Willow round-2 F1 check (`willow_r2check/`, 10 October 2026)
An independent check of decoder's round-2 result F1 (`notes/decoder/willow_prereg_r2.md` §2–4 and §7: ML(D̂) against
ML(RL) on the held-out halves of the 72 W5 leaves; D̂ is the RL DEM with one scale per T̂ class, shared by the four
leaves of a series and fitted by maximising Σ ln P(L | s) over round 1's fit shots). Brief from main, 10 Oct. `A`, `D`
and `F` are as in the P4 check, `F1=results/decoder/willow/round2/f1/google_105Q_surface_code_d3_d5_d7`,
`AN=results/decoder/willow/round2/f1_analysis.json` and `O=results/reservoir/willow_r2check`, as absolute paths. Every
command runs as `nice -n 10 python3 -I` with 1 thread, one at a time (scratch wrapper
`/home/claude/work/reservoir/r2check/run_r2.sh`). Nothing in `code/decoder/` is imported. `r2_common.py` holds the
shared parts: the leaf model (the shipped RL DEM, T̂'s classes rebuilt by `willow_check/p4_check.classes`, D̂'s scales
applied and clipped as F1's model does), detection events from the archive, and the exact decoder
(`willow_check/p4_redecode.layered_ml`).

| command | writes to `$O/` |
|---|---|
| `r2_check.py $A $D $F $F1 $AN $O` | `r2_leaves.csv` (per leaf: errors, 2×2 cells, log-loss, R, events per shot, checks) and `r2_check.json`: m_D with P4's stratified SE from `D_pred`, round 1's ML(RL) predictions and the archive's observables (all, by r, without r = 13, per series); T̂ against D̂ and P4 again; log-loss differences; R, R_conf and the share of RL's excess removed (delta method); ln P(s) and event counts under each DEM; in-sample against held-out gain; split, fingerprint, DEM, class and record checks; a comparison with `f1_analysis.json` |
| `r2_redecode.py $A $D $F $F1 $O` | `r2_redecode.json`: 28 random held-out shots per leaf (2,016; seed 20261010) re-decoded under D̂ and under the RL DEM, against `D_llr`, `D_logp`, `D_pred` and round 1's ML(RL) |
| `r2_localmax.py $A $F $F1 $O q8_5/X,q6_3/Z` | `r2_localmax.json`: the mean ln P(L \| s) on each series' 20,000 fit shots at φ̂, at φ = 0 and at φ̂ ± 0.5 SE along 4 coordinates; `r2_localmax.ckpt.json` while it runs |
| `r2_temper.py $A $D $F $F1 $O` | `r2_temper.json` (EXPLORATORY): one temperature per series (and one for all) on ML(RL)'s llr, fitted on the same fit shots, against D̂ on the held-out halves |
| `r2_basis.py $O` | `r2_basis.json` (descriptive, post hoc): m_D by basis and by basis and r, from the cells in `r2_leaves.csv`; log-loss gain and R by basis |

Run notes. `r2_check.py`, `r2_redecode.py`, `r2_temper.py` and `r2_basis.py` were run into scratch (`QRC_OUT`) and
stored with `QRC_OVERWRITE=1` (10:28–10:52Z; `r2_check.py`'s scratch run with its final code at 10:49Z, after the
store run). Run times: 79–97, 826–1,087, 57–58 and under 1 s. The stored files equal the scratch runs apart from their
run-time fields (`r2_leaves.csv` and `r2_basis.json` identical by md5). `r2_localmax.py` (718 s, 10:16–10:28Z) ran once,
straight into the store, with its checkpoint in `$O`; a copy of the checkpoint that reappeared on the shared store after
the script had removed it was deleted by hand. Its objective at φ̂ and at φ = 0 equals decoder's `fit_series.json` to
1e-13. md5s: `r2_check.json` 58a685b668b6a4ceb33ff5251e8c27c3, `r2_leaves.csv` aecb38b1f73e21d71f59aad2227ec96d,
`r2_redecode.json` 805a8b942bc5c8ce5b27d50561430851, `r2_localmax.json` 8f2b45b7863cc6e5849a465829b54e8a,
`r2_temper.json` 332f2c011072c1ad3429fc7c20f91cd5, `r2_basis.json` 9dc990c2b406f9823966dd60a7095049.

## Willow round-2 follow-ups (`willow_r2check/`, 10 October 2026)
Brief from main, 10 Oct, after the F1 check: (1) why F1's gain is in Z only, from existing outputs only (EXPLORATORY:
descriptive and post hoc); (2) an independent check of decoder's mech fits (PR2 §8, A1 with §8.4), written now and run
only when main asks. Paths as in the F1 check, plus
`M=results/decoder/willow/round2/mech/google_105Q_surface_code_d3_d5_d7`,
`I=results/decoder/willow/round2/insample/google_105Q_surface_code_d3_d5_d7` and
`AN=results/decoder/willow/round2/explore_analysis.json`, as absolute paths. Same rules: `nice -n 10 python3 -I`,
1 thread, one process at a time. Nothing in `code/decoder/` is imported.

| command | writes to `$O/` |
|---|---|
| `r2_zx.py $A $D $F $F1 $O` (EXPLORATORY) | reads `r2_check.json`, `r2_leaves.csv` and `r2_temper.json` in `$O`. `r2_zx.json`: ML(D̂)'s changed decisions against ML(RL) (b = RL wrong and D̂ right, c = the reverse; hit rate b/(b + c); contribution (b − c)/K_D to m_D, additive over any split of the shots) by basis, by RL's and D̂'s min(P, 1 − P), by direction, by last- and first-layer events and timelike pairs, and on the (7,7) twin series q6_7/X and q8_9/X by detector (7,7) at the last layer; D̂'s scales by class, X against Z paired by patch; the logical tilt and the bulk scale by basis; per-series correlations (with `r2_temper.json`'s β) and weighted fits of m_D on basis and bulk scale; leave one patch out. `r2_zx_cells.csv`: the cells per leaf and factor |

Mech check, prepared and smoke-tested on fabricated inputs on 10 Oct and run on decoder's outputs on 11 Oct (see
"Mech check run" below):

| command | writes to `$O/` |
|---|---|
| `r2m_check.py $A $D $F $F1 $M $I $AN $O [--series p/B,...]` | `r2m_leaves.csv` and `r2m_check.json`: m for T̂_mech and D̂_mech against ML(RL) with P4's stratified SE, from the per-shot predictions in `mech_data.npz`, round 1's ML(RL) and the archive's observables (all, and without r = 13); log-loss differences against RL, T̂, D̂ and each other; R, R_conf and the share of RL's excess removed; split checks (held-out shots = round 1's and F1's, the odd-rank rule; fit shots in the even ranks and disjoint from them); record checks against the family rebuilt here (names, family, fingerprint recomputed, convergence, last step below 0.02 SE, the 25-step cap of PR2's amendment of 2026-10-10 15:14Z with `max_iter` recorded as 25 (15 before 11 Oct, §8.2's cap), converged exactly when the last step is below 0.02 SE, a fit not converged having used all its steps, objective and mean_ll equal to the last history entry, finite SEs); `eval_mech.json` errors and cells recomputed; in-sample against held-out where decoder's in-sample files exist (D̂_mech: mean_ll minus RL's in-sample ln P(L \| s), F1's history[0]; T̂_mech: mean_ll minus round 1's ML(RL) ln P(s) on the fit shots, which must also equal its start); a comparison with `explore_analysis.json`. `$I` and `$AN` may be `-`. Series without fits are listed; fits not converged are left out, per fit |
| `r2m_redecode.py $A $D $F $M $O [--n-per-leaf 28] [--seed 20261010] [--control 4]` | `r2m_redecode.json`: 28 random held-out shots per leaf (2,016) re-decoded exactly under T̂_mech and D̂_mech, against the stored llr and predictions; held-out ln P(s) minus RL's on those shots; a control with every shift at 0 on 4 shots of each series' r = 10 leaf, against round 1's ML(RL) llr and ln P(s) |
| `r2m_localmax.py $A $F $F1 $M $O --series q8_5/X,q6_3/Z [--fits T,D] [--n-coord 4] [--h 0.5]` | `r2m_localmax.json`: each fit's objective on the series' 20,000 fit shots (T̂_mech: Σ ln P(s); D̂_mech: Σ ln P(L \| s) − (λ/2) Σ δ², λ = 4) at δ̂ and at its start, against decoder's record, and at δ̂ ± 0.5 SE along 4 coordinates; `r2m_localmax.ckpt.json` while it runs (a re-run resumes from it) |
| `r2m_smoke.py $A $D $F $F1 FAKE_DIR q10_7/X` | SMOKE TEST ONLY: fabricated mech outputs for one series in decoder's layout, in a scratch folder (it refuses to write under `/mnt/project-files`). Nothing it writes is a result |

`r2m_common.py` rebuilds PR2 §8.2's family independently: one template per (first/last/bulk, L flag, the first three
detector coordinates of the fault), templates with fewer than 30 expected events pooled by F1 class, then into
`R:rare`, then fixed at 0; decoder's parameter names; the fit shots and decoder's fingerprint; F1's D̂ mapped to
D̂_mech's start (the mean ln θ̂ over each parameter's faults); and the mech DEM (logit shift, clipped to
[1e-15, 0.5 − 1e-9]). On d3_at_q10_7/X the rebuilt family (190 templates, 136 with at least 30 expected events, the
same 11 rare groups, 144 parameters, 29 templates flipping L) equals the family line of decoder's `sim_mech_V0.log`
(10:47:25), which is built on the same series.

Run plan (when main asks; one process at a time; wrapper
`/home/claude/work/reservoir/r2check/run_r2m.sh scratch|store check|redecode|localmax [args]`): `r2m_check.py` into
scratch, then into the store, and compare (about 1.5 min each); `r2m_redecode.py` once, into the store (about 19 min:
F1's re-decode, with the same number of exact decodes, took 18); `r2m_localmax.py` once, into the store, on q8_5/X and
q6_3/Z (the series used for F1, fixed before the fits exist), T and D, 4 coordinates: 40 evaluations of 35–45 s,
about 27 min. Estimates: about 50 min of wall time in all, so at most about 0.85 core-hours; on the busy machine the
CPU time was 46% of the wall time in the last smoke test.

Smoke tests (11:20–11:40Z; fabricated q10_7/X outputs written by `r2m_smoke.py` to
`/home/claude/work/reservoir/r2check/mech_fake/`; every output went to scratch, none to `$O`): `r2m_check.py` ran end
to end, every split, record and fingerprint check passed, and T̂_mech = ML(RL) gave m = 0 exactly. `r2m_redecode.py`:
T̂_mech (every shift 0) matched round 1's llr to 3.8e-7, the control matched to 2.2e-7 in llr and 1.3e-13 in
ln P(s), and D̂_mech (F1's D̂ mapped to the templates, an approximation by design) matched F1's llr to 0.013 with every
decision equal. `r2m_localmax.py` (50 fit shots per leaf, flagged as a smoke test) ran end to end and removed its
checkpoint. Fixes from the smoke tests: z and p are null when an SE is 0; a variance that rounds below 0 is set to 0;
the control runs on the r = 10 leaves only; a minimum over no shots is written as null.

Run notes. `r2_zx.py` ran into scratch at 11:11:43Z (79 s) and into the store at 11:13:02Z (81 s). The stored
`r2_zx.json` equals the scratch run apart from `run_seconds`, both are strict JSON, and `r2_zx_cells.csv` is identical
by md5. md5s: `r2_zx.json` fbbfad27ea4641e309809f6ccb95c777, `r2_zx_cells.csv` cf3e858df8a4d0367646dfa25f146d0a;
scripts `r2_zx.py` 48f2cc1618e60f16f7971ffa044aa629, `r2m_common.py` 92a328af586f23ef0b7510a0b544fe8c,
`r2m_check.py` c3e27ac7addf2e84d6b605066e944919 (as smoke-tested; as run on 11 Oct: 9656b5f83257aefa30e415facddd4f64, see
below), `r2m_redecode.py` 74d3eda1ad7e22b8000fc5be77ebf4e9, `r2m_localmax.py` 578aa4ecee0714a0e497bcfd17135790,
`r2m_smoke.py` 847e1e90188deef2d8379f0aa8e76c87.

Mech check run (11 Oct 2026, main's request of 19:40Z; nice 10 with the process's own autogroup at nice 10, one
process at a time, detached through `/home/claude/work/reservoir/r2check/mech_run/launch.sh`, wrapper and scripts
otherwise as planned). Inputs: decoder's 36 mech fit records and 72 held-out outputs as at 19:39Z; the 72 in-sample
files after both in-sample parts ended (19:44:59Z); `explore_analysis.json` as rewritten by the chain at 19:50:45Z
(md5 81ddad1c25d707f287b69e2dca7883e4; the earlier one, 0a64c0a1…, is the chain's `explore_analysis.before_chain.json`).
- Order (run control only): `r2m_redecode.py` first, 19:48:35–20:11:20Z, as it reads neither the in-sample files nor
  the analysis; then `r2m_check.py` into scratch and into the store; `r2m_localmax.py` last, 20:19:37–20:37:09Z
  (PR3's runner holds a slot until localmax has been seen and every r2m_ script has been gone 10 minutes).
- Two changes to `r2m_check.py`, neither touching an estimate. (1) Before the run: the iteration cap 15 → 25 (PR2's
  amendment of 2026-10-10 15:14Z, made after this script was written at 11:25Z), with three added record checks (the
  recorded max_iter is 25; converged exactly when the last check is below 0.02 SE; a fit not converged used all its
  steps); md5 aee2257ce3e34d2aaa5e05c953c53f99. (2) After the first scratch and store runs: with a series left out,
  the held-out per-series block and its heterogeneity statistic paired the wrong series' shots with each name,
  because the series index counted all 18 series. It now counts the series kept. Only T̂_mech's
  `m_X_per_series` and `m_X_series_heterogeneity` changed (25 values; D̂_mech keeps every series, and the pooled, by-r,
  in-sample and per-series gain numbers do not use this index); md5 9656b5f83257aefa30e415facddd4f64. Scratch and store
  were then rerun.
- Scratch against store (fixed version): `r2m_check.json` identical apart from `run_seconds`, `r2m_leaves.csv`
  identical by md5.
- Outputs: `r2m_check.json` c4f851459d18cef6ebb21be3577cfac9, `r2m_leaves.csv` fedaddb3354d5a390b4cf647a3d2f04a,
  `r2m_redecode.json` 78b041a64c74ad5f1b0dc43ffb72394e, `r2m_localmax.json` c0eca57be89cbac391b227f8a56e37c2.
- Compute (measured with bash `times` unless stated): each `r2m_check.py` run about 20 s of CPU (87–90 s wall); the
  first scratch run's CPU and the redecode's were not recorded (the launcher's first version printed the wrong times;
  the redecode is bounded by its 1,363 s of wall time); `r2m_localmax.py` 964 s. In all at most about 0.67 core-hours,
  within the plan's 0.85.

X against Z in the RL DEMs (`r2_xzprior.py`; descriptive, post hoc; asked by scout, board 11:53Z, and main):
`r2_xzprior.py $A $F $F1 $O` parses DEMs and counts, with no decoding. `r2_xzprior.json`: per leaf, T̂'s class totals
before and after the rare-class merge and D̂'s totals, expected and observed detection events per round and per bulk
detector, expected and observed logical-flip probability; per patch and r, the same-symptom X/Z comparison for the RL
DEMs and, as a control, the SI1000-prior DEMs, the lines that differ between the X and Z ideal circuits, and ln Z/X of
each class total; per patch, D̂'s bulk scale Z − X against the prior's differences (Spearman, Pearson and OLS over the
9 patches); whether each RL bulk template has one rate in every round and at every r. `r2_xzprior_classes.csv`: the
class totals per leaf. Scratch run with the final code 12:09:18Z and store 12:10:26Z, 51 s each (13 s of CPU); the
stored files equal the scratch run apart from `run_seconds` (the CSV identical by md5). md5s: `r2_xzprior.py`
00d2aa09424e95d7ba58b5b1434a580d, `r2_xzprior.json` 7b8d030506bdfa1b6fab56dc3639398d, `r2_xzprior_classes.csv`
cb6f1deeb89dd7e7e21ecc304695c5ab.

## Willow d = 5 replication and Λ check (`willow_d5check/`, 10 October 2026)
An independent check of P1's d = 5 replication and of Λ (`notes/decoder/willow_prereg.md` §4–6 and §9 entries 7, 10
and 11), from the per-shot predictions, with no decoding. Brief from main, 10 Oct, after the queue's final analysis step
(decoder's `analysis/`, written 19:30Z):
- the d = 5 gap with the pre-registered window and pairing;
- how it changes without the leaves that K2 fails or has not decided;
- whether one patch or one r drives it;
- Λ_ML(RL), Λ_Libra and their paired ratio;
- anything that could make d = 5 look better for ML than it is.

It is interim: K2's extra-rung verdicts are read as found at run time.

Paths, all absolute:
- `A` and `D` as in the P1 check;
- `DD=results/decoder/willow/decode`;
- `K=results/decoder/willow/supplementary/k2_extra_rung`;
- `AN=results/decoder/willow/analysis`;
- `G=results/decoder/willow/pr3_planning/d5_gap_check.json`;
- `O=results/reservoir/willow_d5check`.

Every command runs as `nice -n 10 python3 -I`, 1 thread, one at a time. Nothing in `code/decoder/` is imported.
`d5_check.py` imports `read_obs_b8` and `y_var` from `willow_check/willow_check.py`, and copies its `wls`, `eps_fit`
and `paired`. The copies add one guard: a fit with 2 rounds has no χ², so it gets no inflation.

| command | writes to `$O/` |
|---|---|
| `d5_check.py $A $DD $K $O` | `d5_leaves.csv`, per W5 leaf at d = 3 and 5: counts; τ1 and ladder counts; the band; the 2×2 table with Libra; Libra on the other shots; K2's verdict and τ3 counts; checks. `d5_series.csv`: series eps and per-series paired gaps, for each set. `d5_check.json`: g_5 with the corrected (entry 7), frozen-formula and uninflated SEs; pooled counts; the sensitivity sets (without the K2-failing leaves, K2-pass only, by K2 status, leave one patch or r out, pooled by r, patch, basis and series); Λ_3/5 for ML(RL), on W5 and on Libra's rounds, and for Libra; the paired ratio (1 + g_3)/(1 + g_5) and the unpaired one; g_3 and the pooled d = 3 counts on {10, 30, 50}; flags (the subset, truncation, a 1% shift of every ML count) |
| `compare_d5.py $AN $O $O $G` | `compare_d5.json`: a comparison with decoder's analysis files and its `d5_gap_check.json`, both read only after the outputs above were stored |
| `band_extrap.py $D $O` | `band_extrap.json` (EXPLORATORY, post hoc). K2 re-decodes only the band shots. This counts the out-of-band decisions that would flip if each out-of-band τ1 LLR were off by a shift drawn from its own leaf's band shots (τ1 → τ2), and gives the pooled gap if every such flip added an ML error |

Run notes.
- **`d5_check.py`** ran into scratch (`QRC_OUT`; K2 read at 19:42:52Z), then into the store (19:43:06Z; 33 s of wall
  time, about 1 s of CPU).
  - The stored files equal the scratch run apart from `k2/read_at` and `run_seconds`. The CSVs are identical by md5.
  - K2 at the store run (19:43:38Z): 18 pass, 4 still fail, 10 without a verdict. This was unchanged at 19:54Z.
  - The docstring mentions K2's summary list "made at 18:37:33Z"; the run read the 19:35:41Z version.
- **`band_extrap.py`** ran into scratch, then into the store (19:53Z, 8 s). The two are equal apart from
  `run_seconds`.
- **`compare_d5.py`** was first stored at 19:53:37Z without its 4th argument. It was re-run with the 4th argument, into
  scratch and then into the store (19:56:26Z, under 1 s). The stored file is the second run, identical to its scratch
  run.

md5s:

| file | md5 |
|---|---|
| `d5_check.py` | 0209e908597f243117c71ff60c76fd01 |
| `compare_d5.py` | 227feef9cb274864eb712f126d12264a |
| `band_extrap.py` | 2162eb729399bc97a04d75b0af50b325 |
| `d5_check.json` | 73f7604bf3dd556b3a3a676c1fa15ce1 |
| `d5_leaves.csv` | 6a85ab8eace44845394334d91bb873ce |
| `d5_series.csv` | dbfa0315aa00b4045162188e1ff9e78b |
| `compare_d5.json` | 9d2b3411eff6dfdfbedddbc6b3df9626 |
| `band_extrap.json` | c55c52ef8f06f6cf2c4f14512f0b7722 |

## Willow: why the gap to ML grows with d (`willow_gapmech/`, 10 October 2026; EXPLORATORY, post hoc)
Brief from main, 10 Oct, after the d = 5 check: every decoder's gap to ML(RL) grows from d = 3 to d = 5 in the frozen
analysis (`results/decoder/willow/analysis/report.md`, "Pooled paired eps gaps to ML, W5 at every distance"). Why?
1. A matched table, from the per-shot predictions: g_3 and g_5 on the same rounds at both distances, entry 7's SE, the
   paired Λ ratio (1 + g_3)/(1 + g_5) with a CI, growth per series; K2 rule A (main, 19:58Z) and no exclusion.
2. The degeneracy hypothesis on existing outputs: the discordant shots (decoder wrong and ML right, and the reverse) by
   ML's margin, syndrome weight, r and basis; their overlap with Tesseract's (a most-probable-error search); each
   decoder's Λ deficit split through Tesseract on the same shots.
3. An exact test: the d = 3 layered contraction in the max-product semiring gives each shot's exact most-probable-error
   (MPE) class. Brute-force tests, a timing test (≤ 0.2 core-hours), an integer-programming MPE for d = 5 checks, and a
   design. Nothing is decoded beyond the timing test until main approves.

Paths, all absolute: `A` (archive extract) as in the P1 check; `DD=results/decoder/willow/decode`;
`K=results/decoder/willow/supplementary/k2_extra_rung`; `O=results/reservoir/willow_gapmech`; `C=code/reservoir/willow_gapmech`;
`DEM=$A/d3_at_q6_7/X/r10/decoding_results/correlated_matching_decoder_with_rl_optimized_prior/error_model.dem`.

Every command runs as `nice -n 10 python3 -I` in a shell at autogroup nice 10 (`echo 10 > /proc/self/autogroup`), 1
thread, one at a time. Nothing in `code/decoder/` is imported or edited. `gapmech.py` imports `paired`, `g_stat` and
`k2_verdicts` from `willow_d5check/d5_check.py` (prereg §4 with entry 7's SE) and `read_obs_b8` from
`willow_check/willow_check.py`; `gapfactor.py` imports `gapmech.py`; `mpe.py` imports `mechanisms` from
`willow_check/p4_check.py` and the exact sum-product `layered_ml` from `willow_check/p4_redecode.py`; `mpe.py` and
`mpe_milp.py` call the `tesseract_decoder` package as decoder's runs do (`code/decoder/baselines.py`, read only).

| command | writes to `$O/` |
|---|---|
| `gapmech.py $A $DD $K $O` | `gap_table.json`: coverage per decoder; g_3, g_5 (entry 7's SE), Λ ratio and CI, growth z, per-series statistics (Mann–Whitney) and g by basis, for the matched rounds and for rounds {10, 30}, each under rule A and with no exclusion; checks. `gap_series.csv`: the per-series gaps. `discord.json`, `discord_bins.csv`: the discordant shots (b = decoder wrong and ML right, c = the reverse) by ML margin bin, syndrome-weight quartile within the leaf, r and basis, with ML's predicted share of b; the overlap of each decoder's discordant shots with Tesseract's |
| `gapfactor.py $A $DD $K $O` | `gap_factor.json`: on the shots and rounds where the decoder, Tesseract and ML all have predictions, g(A\|ML), g(T\|ML) and g(A\|T) at both distances and their Λ ratios; Λ_A/Λ_ML ≈ (Λ_A/Λ_T)(Λ_T/Λ_ML) |
| `mpe.py test $O $DEM` | `mpe_tests.json`: the max- and sum-product contractions against enumeration of every fault subset (24 random layered DEMs, 8 local pieces of a real d = 3 DEM) |
| `mpe.py timing $A $DD $O $C/mpe_timing_plan.json` | `mpe_timing.json`: per d = 3 leaf, exact MPE and ML on the first N every-5th shots (CPU time, ML llr against decoder's stored exact llr, MPE against the stored Tesseract predictions); Tesseract settings against the exact MPE (class and cost gap); per d = 5 leaf, Tesseract settings against each other |
| `mpe_milp.py $A $DD $O $C/mpe_milp_plan.json` | `mpe_milp.json`: exact MPE by integer programming (HiGHS, proven optimum) against mpe.py at d = 3 and against Tesseract settings at d = 3 and 5, with CPU times |

Run notes.
- **`gapmech.py`** ran into scratch (`QRC_OUT`; K2 read at 20:18:04Z), then into the store (K2 read at 20:32:33Z;
  166 s of wall time, 13.6 s of CPU). The stored files equal the scratch run apart from `k2/read_at` and the run
  times; the CSVs are identical by md5. K2: 22 verdicts, 4 still failing (d5 q4_7/Z/r13, q6_5/X/r30, q6_9/X/r13,
  q8_7/Z/r10), unchanged between the two runs.
- **`mpe.py test`** ran into scratch twice (the second after the real-DEM cases were changed to local pieces with
  observable-flipping faults, so that they contain degenerate classes), then into the store (20:36:45Z; 2.9 s of CPU).
- **`mpe.py timing`** ran once, into the store (20:36:49–20:47:18Z; 147 s of CPU, 628 s of wall time at autogroup
  nice 10 with the 4 cores full), after a smoke test on a 50-shot plan into a scratch folder (5.7 s of CPU).
- **`mpe_milp.py`** ran once, into the store (20:48:32–21:02:01Z; 228 s of CPU), after a smoke test on 3 + 3 shots
  into a scratch folder (4.8 s of CPU). HiGHS's time limit is wall-clock time, so at nice 10 under full load it
  allows less CPU than its nominal 30 s.
- **`gapfactor.py`** ran once, into the store (K2 read at 21:02:54Z: 23 verdicts, the same 4 still failing; 3.6 s of
  CPU).
- Compute for the whole brief: about 430 CPU-seconds (0.12 core-hours), of which the timing test (smoke tests,
  `mpe.py timing`, `mpe_milp.py`) used 387 s (0.11 core-hours). One process at a time.
- Versions: tesseract-decoder 0.1.1.dev20260910235247 (as in decoder's runs), numba 0.68.0, stim 1.16.0, scipy 1.18.1
  (HiGHS), numpy 2.5.3.

md5s:

| file | md5 |
|---|---|
| `gapmech.py` | 6922658fdcafa628f3c129b2113a6290 |
| `gapfactor.py` | 949213e8f9c338981dbe35476009d8c5 |
| `mpe.py` | 75c29fc03b73cb6017e3745c1837f3f8 |
| `mpe_milp.py` | 13e1473b28763c13f5015d97ed35bc89 |
| `mpe_timing_plan.json` | da0846058fe3452511b11eb8c059b786 |
| `mpe_milp_plan.json` | 74d7f088df89077cd5fb296a1cd0fc6a |
| `gap_table.json` | 45cd61385ed75f3524d831f21a97505c |
| `gap_series.csv` | 4a767166bc79c0d5d4de3af053718be9 |
| `discord.json` | a0a5a3a171cdfd871957ec705e72f7c6 |
| `discord_bins.csv` | 15f083311c45b09439794953c6d78d7d |
| `gap_factor.json` | 3b9052946d159be6f7fa6098e98f32b8 |
| `mpe_tests.json` | 723f25f2f1990b2892d0505f3bac780e |
| `mpe_timing.json` | cd18d31325bf83dfdc67d5beb5b39fdb |
| `mpe_milp.json` | db203aee5d8e2c5542132e9d2c92fe26 |

### The exact test (main's 21:11Z decision; EXPLORATORY)
The design was logged before stage 1 in `notes/reservoir/gapmech_design.md` (md5 18aa42e9…). It holds the shot lists, main's reading rule and every md5. Main's conditions for the run:
- one process at normal priority (no autogroup change, no nice);
- 1 thread;
- a cap of 1.5 core-hours for the run, its tests and the analysis.

| command | writes to `$O/` |
|---|---|
| `exact_run.py plan $DD $C/exact_plan.json` | the plan: stage 1's 72 d = 3 leaves; stage 2's 840 d = 5 shots, drawn with default_rng(20261010) from the stored Tesseract shots (50 per leaf at r = 10 and 13, 5 at r = 30); limits; CPU guard 5000 s |
| `exact_run.py run $A $DD $O $C/exact_plan.json` | `exact/<leaf>.npz` (72 files: idx, exact MPE class, ln M_0, ln M_1 on the every-5th shots) and `exact_stage1.json` (per leaf: ties, MPE ≠ stored Tesseract, MPE ≠ ML, errors, CPU); `exact_stage2.json` (per shot: the proven optimum by HiGHS, a class-restricted program where the classes differ, and the verdict on the stored class — same class, tie, miss, undetermined or unproven; the Tesseract rerun as a diagnostic; per round and overall: misses among proven shots with the one-sided 95% Clopper–Pearson bound) |
| `exact_analysis.py $A $DD $K $O` | `exact_analysis.json`. For each decoder: the paired gaps A\|ML, MPE\|ML and A\|MPE at d = 3 and 5 (entry 7's SE); Λ_MPE/Λ_ML, Λ_A/Λ_MPE and Λ_A/Λ_ML with analytic log-scale CIs; the degeneracy share with a bootstrap CI over series (B = 10,000, seed 20261010); main's reading; shot categories; the d = 5 certificate; the leverage u/D. Also the degeneracy factor alone on {10, 13, 30} and {10, 30}, under rule A and with no exclusion |

`exact_run.py` imports `mpe.py` (max-product, Tesseract runner) and `mpe_milp.py` (integer-program setup). `exact_analysis.py` imports `gapmech.py` and `d5_check.py`.

Run notes:
- **Tests,** into scratch, about 100 CPU-s in total:
  - a tiny run (1 d = 3 leaf; 5 d = 5 shots);
  - the class-restricted program against max-product at d = 3 (equal within 1e-6);
  - the synthetic analysis with the MPE set to the stored Tesseract at d = 3, which reproduces `gap_factor.json` exactly.
- **`exact_run.py run`** (`run_exact.sh`, scratch): 21:32:14–22:25:45Z, 2942 CPU-s and 3212 s of wall time at nice 0, about 90% of a core. The guard did not stop it.
  - Stage 1 took 1876 CPU-s.
  - Stage 2 took 984 CPU-s of integer programs; the mean per shot was 0.67, 0.97 and 8.3 s at r = 10, 13 and 30, the longest 26.5 s.
  - It also took 78 CPU-s of Tesseract reruns.
  - The log is `$O/exact_run.log`.
- **`exact_analysis.py`:** 22:26:33–22:27:22Z, 4.6 CPU-s. K2 was read at 22:26:41Z: 25 verdicts, with the same 4 still failing. The log is `$O/exact_analysis.log`.
- **Compute for the exact test:** about 3,050 CPU-s (0.85 core-hours), inside main's 1.5 core-hour cap.
- **Final K2 tally (11 Oct, orchestrator's request of 17:36Z).** `gapmech.py` and `exact_analysis.py` were re-run unchanged (md5s above) at autogroup nice 10, 1 thread, by `run_final_k2.sh` (b049ec15…), with `QRC_OUT` in scratch.
  - K2 was read at 17:38:56Z and 17:41:36Z: 32 verdicts, 7 still failing (the 4 above plus d5 q4_7/X/r50, q6_5/X/r50, q6_9/Z/r50).
  - CPU: 13.0 s and 4.9 s.
  - The outputs were copied to `results/reservoir/willow_gapmech/final_k2/`: gap_table.json a0570403…, gap_series.csv 979a4466…, discord.json 36370854…, discord_bins.csv 11d45410…, exact_analysis.json 4d64d5cc…, and the script and log.
  - The interim files above are kept.

md5s:

| file | md5 |
|---|---|
| `exact_run.py` | 1ea47826a4fe8f9cf363524dffb20a70 |
| `exact_analysis.py` | f57c590dae071c050e19806fb584e417 |
| `exact_plan.json` | c73a930a79391652f96010e3b080b366 |
| `exact_stage1.json` | 8543be6604c4d285a48ddbd91c38d491 |
| `exact_stage2.json` | 218cf77aabf9249bc1828f1546f64cf9 |
| `exact_analysis.json` | 20747cbfa99019e4337fee5d9335358f |
| `exact_run.log` | 14a203c178295dbeaaa80db357910618 |
| `exact_analysis.log` | 32100f28c938a05ee15d6993c78f1286 |
| `exact/*.npz` (md5 of the output of `md5sum *.npz` run in `exact/`) | 5c57549fa5a864f58f0074e6ed0d3f28 |

## Willow simulation with the RL DEM as the true noise (`willow_sim/`, 10 October 2026; EXPLORATORY)
Design: `notes/reservoir/sim_design.md` (approved by main at 22:48Z as stages A + B) and its addendum of 23:10Z, which holds the final run plan, seeds and md5s, and was logged before the run. Post hoc question; pre-specified design.

- **`sim_run.py plan OUT [tiny]` / `run PLAN`.**
  - `plan` writes the run plan.
  - `run` runs it one child at a time (python3 -I -B, 1 thread, numba cache in scratch), with:
    - a CPU guard (6.0 core-hours; per-step projection and a hard stop);
    - stage B only if it starts before the plan's deadline (05:00Z) and fits the guard;
    - one retry per step, and resumable state.
  - After each stage it runs `sim_analysis.py` and copies the consolidated outputs to `results/reservoir/willow_sim/` through `qrc.outpaths` (QRC_OVERWRITE=1 for the stored run).
- **`sim_sample.py PLAN STAGE D`.** Writes a simulated archive in scratch with the real layout: `circuit_ideal.stim` and the RL DEM copied from the real leaf, and `detection_events.b8` and `obs_flips_actual.b8` from stim's DEM sampler. Seeds come from SeedSequence(20261011).
- **decoder's `decode_willow.py`** is run unchanged, through its CLI only, on the simulated archive. Dem tags:
  - `rl_sim`: ML (`ml_exact` at d = 3, the τ ladder at d = 5), `mwpm`, `corr_mwpm`;
  - `rl_sim_t2`, `rl_sim_t3`: `ml_sparse` at τ = 1e-8 and 1e-9 on the first shots of each d = 5 leaf.
- **`sim_mpe.py d3|d5 PLAN LEAF`.**
  - d = 3: exact max-product MPE (`mpe.py`).
  - d = 5: Tesseract with an infinite beam, with decoder's settings as fallback on low-confidence shots, and the integer-program certificate (`exact_run.solve`). The certificate covers every shot where Tesseract ≠ ML, every shot still low-confidence, and 250 random agreeing shots per leaf. Misses are corrected to the proven class.
- **`sim_analysis.py PLAN A|AB OUT`.**
  - Q1/Q2: fixed-r Λ ratios ρ_A(3)/ρ_A(5) with stratified delta-method CIs, and the same estimator on the stored real outputs (rule A primary), with s_A.
  - Q3: R_sim with the d = 3 gate; β and β′ with a bootstrap stratified by leaf.
  - The readings of the design's §5.
- **Run:** `run_sim.sh` (scratch `/home/claude/work/reservoir/willow_sim/`), started 23:10:28Z at nice 0. It was stopped at 23:14:32Z, after 10 steps, to make failed steps non-fatal, and restarted from its state at 23:16:44Z (`notes/reservoir/sim_design.md`, amendment 23:16Z). Stage A ended at 02:13:40Z and stage B started then.
  - The /mnt/project-files outage (EIO from about 02:32Z) left 22 of stage B's 46 steps done. The other 24 were recorded as failed, and the run ended at 02:32:40Z.
  - After the container restart, `check_resume.py` found the scratch outputs complete and consistent. Stage B resumed at 13:39:03Z at nice 10 (the design's amendment of 2026-10-11 13:38Z) and ended at 15:50:43Z with 0 failed steps. CPU: 13,913 s (3.86 core-hours) in all; guard 6.0.
- **`check_resume.py PLAN STATE OUT_JSON`** (11 October 2026). A read-only check of stage B's scratch outputs before the resume:
  - samples: md5s, and a fresh draw from the plan's seeds compared bit for bit;
  - decodes marked done: summary, job flags, counts, and chunks against the consolidated npz;
  - the chunk checkpoints a rerun would reuse;
  - stage A's scratch files against their results copies.
  Report: `results/reservoir/willow_sim/resume_check_B.json`.
- **`beta_real.py DECODE CHECK SIM OUT_DIR`** (11 October 2026; EXPLORATORY, outcome-free, read-only). β and β′ on the
  real d = 5 shots, as pre-specified in the design's block of 2026-10-10 23:19Z (main's item 3a), from round 1's stored
  ladder (`decode/<leaf>/rl_s5/ml_tau2_1e-6-1e-8_0.1.npz`) and decoder's non-band check
  (`supplementary/nonband_check/`: shot lists, `leaves/<k>.npz` with τ₂ and τ₃, `summary.json`, which must say
  "complete"). Per r (10, 30; 50 descriptive), pooled over the 8 leaves:
  - β_real = Σ m_ladder / Σ m_τ2 − 1. The τ2 total is exact on the band and on the D census, and a combined ratio
    estimate on the C sample.
  - β′_real = Σ_N m_τ2 / Σ_N m_τ3 − 1, with both sums Horvitz–Thompson over the rows where τ3 ran.
  - CIs: percentile bootstrap within leaf and stratum, B = 10,000, default_rng(20261013). The census parts are fixed;
    for β′ the D rows that τ2 did not move are resampled with their random-τ3 flag (a reading fixed before any
    estimate; `notes/reservoir/log.md`).
  - Checks: the check is complete and its md5s are as recorded; outside the band the ladder llr equals τ1 bitwise; the
    rows match the ladder; the strata counts equal the populations; the τ3 reasons equal the check's rule, recomputed.
  - Output: `results/reservoir/willow_sim/beta_real.json`. Run: `/home/claude/work/reservoir/beta_real/run_beta_real.sh
    scratch|store` (nice 10). The scratch and store runs are identical apart from timings, and an independent
    recomputation (scratch `crosscheck.py`) gives the same point estimates.

| file | md5 |
|---|---|
| `beta_real.py` | 005237db9e4e52e218cf80b7965f6e26 (11 Oct 22:22Z) |
| `sim_run.py` | 717ac13943f08d5c63e8eac20967e77e (since 23:16Z; the first launch used f7342444…, see the design's amendment) |
| `sim_sample.py` | 144e2df6ff6a5dc3cd5580da47e73bae |
| `sim_mpe.py` | 8edc7f46b5513458029bfca7de7b8673 |
| `sim_analysis.py` | e43952143390e966322acf0cb31fe94e |
| `sim_plan.json` (scratch; copied to results) | 07968771a7295e4e29057663e517186c |
| `check_resume.py` | 1ccc1bb8ea17ef37f305d72de01a994a (11 Oct, before the resume) |
| `run_sim.sh` (scratch) | e608fdd46aedb27618d412054e6b8655 (from the resume, nice 10; e92c110e… before) |

## Willow revision after the 13 October review (`willow_revision/`, 13 October 2026; POST HOC)

These are main's tasks R1–R5 of `notes/orchestrator/willow_revision_plan.md`. They are post hoc revision analyses,
not pre-registered tests. Each compute task is declared in `notes/reservoir/willow_revision_declarations.md` before any
of its results is read. Run every script with `python3 -I -B`, at nice 10 with one thread
(`/home/claude/work/reservoir/revision/runjob.sh`); set `NUMBA_CACHE_DIR` to a scratch folder. Outputs go through
`qrc/outpaths.py` to `results/reservoir/willow_revision/`.

| file | task | what it does | md5 |
|---|---|---|---|
| `check_inputs.py` | all | integrity of the local inputs after the 15:07Z restart: the archive extract against the zip by CRC, the simulation against its md5 records, scratch copies against the store; read-only | 7575a21e0c88595055d122a4ea5ba1c7 |
| `r3_patch.py` | R3 (1.5) | patch-level sensitivity of the final analysis (per-patch d = 5 gaps; leave one patch or region out; intervals with series and patches as clusters; patch jackknife; X/Z correlation within patches) and Table 7's paired Λ ratio against separately fitted Λs (leaf-cell bootstrap, B = 10,000); reproduces `gap_table.json` and scout's series-clustered GRF | 55ed7c18f6576d84bc338ea938bca59e |
| `mpe_sparse.py` | R2 (1.4) | max-product twin of decoder's sparse-frontier contraction (`code/decoder/ml_sparse.py`, imported read-only): mode 0 is decoder's sum-product, bit for bit; mode 1 takes the maximum over the fault sets of a class (the MPE under the same pruning) | 6a6aa029c18be30bcd00c7abe5f93395 |
| `r2_semiring.py` | R2 | runs decoder's τ ladder (1e-6 → 1e-8, band 0.1, cap 2,000,000) in either mode on the stored ML shots; pre-test (bitwise check against the stored ladder; exact MPE and ML at d = 3); chunked checkpoints; reads no outcome | bed6331cf4625246aa9778232989d763 |
| `r1_certify.py` | R1 (1.4) | extends stage 2's integer-programming certificate of the stored Tesseract at d = 5 (plan: default_rng(20261013), equal per leaf, phase 1 to u ≤ D/2 at r = 10, 13, 30; phase 2 to u ≤ D/4 at r = 30); `resolve` settles R2's disagreements with the proven MPE; imports `willow_gapmech/exact_run.py` unchanged | 775fa211a4e7f8c969de99f02b8f4d40 |
| `r12_analysis.py` | R1, R2 | R1's certificate with its worst case; R2's controlled degeneracy factor, Libra's share (window stated), shot-level concordance of Libra's and the MPE's failures (with other decoders as a reference), and the ladder checks | 85abb09c961ccac8fd3c0b164a428221 |
| `r4_syndrome_fit.py` | R4 (1.6) | S-hat: decoder's F1 fit (`code/decoder/round2/run_f1.py`, imported read-only) with the syndrome log-likelihood as the only change of objective; same classes, pooling, optimizer, fit and held-out shots | 6532e200c24abf48d2cd42aaa39b330a |
| `r4_analysis.py` | R4 | the matched comparison of D-hat and S-hat on the held-out halves, with decoder's round-2 definitions (reproduces `f1_analysis.json` for D-hat exactly) | 1132f591d01c19f74b3b6ea1e690129d |
| `r5_worstcase.py` | R5 (1.3) | independent reproduction of the non-band check's worst cases (deviation 13) and of D2's derivation: the 12 published rows from the raw check outputs, the readings of the unrun term, the pooled and per-round boxes (per-round with the τ₃-only term: all 2^18 vertices, vectorised and checked against `d5_check`), Bonferroni levels, other combinations of the strata, the change of g₅ per added ML error; written without reading decoder's scripts | 14be2136c82c5cc2dec5500f45586cba |
| `r5_compare_d2.py` | R5 | compares `r5_worstcase.json` with decoder's `d2_worstcase.json`, number by number (266 quantities and the 16 extreme vertices) | e055e44683945793295d8bdd9535d56e |
| `r2_resolve_list.py` | R2 | the resolve list as declared: d = 5 shots at r = 10, 13, 30 where R2's max-product class differs from the stored Tesseract class, sorted by (leaf, idx), then one permutation with default_rng(20261013); reads no outcome | 87aeaef06d20eb2d007649bcf38989d2 |
