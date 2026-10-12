#!/bin/bash
# usage: queue_real.sh <dataset root>   (start only after the Willow data has landed and the inventory looks right)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/claude/work/decoder/numba_cache PYTHONDONTWRITEBYTECODE=1
ROOT="$1"
C=/mnt/project-files/tensor-networks/deep-dives/experimental/code/decoder
R=/mnt/project-files/tensor-networks/deep-dives/experimental/results/decoder
cd /home/claude/work/decoder
ulimit -v 4000000
echo "start inventory $(date -u +%T)"
nice timeout 1200 python3 -I -B $C/willow_data.py inventory "$ROOT" $R/willow_inventory.json > real_inventory.log 2>&1
for DEM in si1000 rl; do
  echo "start d3 $DEM $(date -u +%T)"
  nice timeout 14400 python3 -I -B -u $C/decode_willow.py --root "$ROOT" --out $R/willow --decoders ml_exact,corr_mwpm,tesseract --dem $DEM --filter d3_at --max-shots 50000 --chunk 5000 --synd-ref 10000 > real_d3_$DEM.log 2>&1
done
echo "done $(date -u +%T)"
