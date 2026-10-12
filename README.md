# Willow: Google's surface-code decoders against maximum likelihood

Code for the manuscript "Willow's decoders against maximum likelihood: the inference loss grows from distance 3 to 5" (in preparation, October 2026).

It decodes Google's public Willow surface-code data ([Google Quantum AI, arXiv:2408.13687](https://arxiv.org/abs/2408.13687); [Zenodo record 13273331](https://zenodo.org/records/13273331)) with exact and approximate maximum-likelihood (tensor-network) decoders under the noise models shipped with the data. It then compares Google's shipped decoders (Libra, Harmony, correlated matching) and other baselines with them, paired on the same shots.

## Layout

| folder | what it holds |
|---|---|
| `code/decoder/` | The ML decoders (exact variable elimination, a sparse-frontier approximation, boundary MPS), the baseline wrappers, the data loaders, the pre-registered real-data pipeline (`queue_real.sh`), the round-2 refits (`round2/`), the distance-7 run (`pr3/`), the supplementary analyses (`supplementary/`) and the tests. Its README maps every script. |
| `code/reservoir/` | Independent checks of the decoder analyses, the mechanism and degeneracy tests, and the simulation with Google's prior as the true noise (`willow_*` folders). `qrc/outpaths.py` is the small path helper they import. |
| `code/scout/` | The scripts behind the manuscript's derived numbers and Fig. 1, and an independent exact-ML cross-check (`ml_dp.py`). |

The folder names are the roles of the AI research agents that wrote the code.

## Requirements

Python 3 with stim, numpy, scipy, numba, pymatching (2.3 or later, for correlated matching), beliefmatching, ldpc, tesseract-decoder and sinter, plus matplotlib for the figures. Run one thread per process (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`) and point `NUMBA_CACHE_DIR` at a local folder.

## Paths and data

The scripts were written for a workspace where `code/` sits next to `results/`, `notes/` and `shared/data/willow/`. Most paths are relative to that root, so they work from a checkout with the same layout. A few scripts have absolute defaults (`/mnt/project-files/...`, `/home/claude/work/...`); pass those paths explicitly elsewhere.

`code/decoder/fetch_willow.sh` downloads the dataset with md5 checks. The archive is treated as untrusted, so every script that reads it runs with `python3 -I -B`.

## Pre-registration

The confirmatory tests were registered in three documents before their results existed. Those documents record the md5 of the frozen scripts, and frozen copies of the round-1 pipeline are in `code/decoder/superseded/frozen_2026-10-08/`. The pre-registrations, the results and the manuscript are not in this repository.
