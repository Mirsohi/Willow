#!/bin/bash
# queue_real.sh: the Willow real-data run, staged, restartable, 4 workers (replaces the 1-core Round-2 queue,
# kept as queue_real_r2.sh). Start it only after board/main.md says the data have landed in shared/data/willow/.
#
# usage: bash queue_real.sh [--dry-run <fake workdir>] [--workers 4] [--cpus 0,1,2,3] [--stages d3,fit,d5,d7]
#                           [--steps stage,inventory,plan,extract,queue,analysis]
#   real:     data from shared/data/willow/ (MANIFEST.json, zip, listing), local work in /home/claude/work/decoder/willow_run,
#             results in results/decoder/willow/ (decode/, fit/, inventory.json, plan.json, queue_state.json, analysis/)
#   dry run:  <fake workdir>/shared (from tests/make_fake_willow3.py), local work <fake workdir>/run,
#             results <fake workdir>/results, small shot counts (plan profile "dry"); nothing touches the shared folder
# Steps (each is restartable; re-running the script skips what is done):
#   stage      copy the zip to local disk, md5 against MANIFEST.json, member check, extract d=3 + metadata of all leaves
#   inventory  the shared loader inventory if it covers every leaf, else our own on the local extraction
#   plan       job list from the inventory (d3 all leaves x both shipped DEMs; fit at W5; d5 W5 every 5th shot;
#              d7 cost-capped probe), cheapest unit first, with core-hour estimates
#   extract    the d=5 / d=7 leaves the plan needs (only those)
#   queue      willow_queue.py with N workers (one thread each, nice 5, 3.5 GB address space each)
#   analysis   analyze_willow.py: tables, figures, pre-registered decisions (works on partial results too)
# Any Python that reads the archive runs with -I; all scripts live in code/decoder/, never in the data directory.
# First real run: --steps stage,inventory,plan,extract, then read the dataset README and the plan notes (missing
# shipped predictions, DEM files with other content), then --steps queue,analysis.
set -uo pipefail
X=/mnt/project-files/tensor-networks/deep-dives/experimental
C=$X/code/decoder
DRY=""; WORKERS=4; CPUS=""; STAGES=d3,fit,d5,d7; STEPS=stage,inventory,plan,extract,queue,analysis
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --cpus) CPUS=$2; shift 2;;
    --stages) STAGES=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
if [ -n "$DRY" ]; then
  SHARED=$DRY/shared; LOCAL=$DRY/run; RES=$DRY/results; PROFILE=dry; SHINV=$DRY/shared_inventory.json
else
  SHARED=$X/shared/data/willow; LOCAL=/home/claude/work/decoder/willow_run; RES=$X/results/decoder/willow; PROFILE=real
  SHINV=$X/results/decoder/willow_inventory.json
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/home/claude/work/decoder/numba_cache}
PY="nice -n 5 python3 -I -B"
mkdir -p "$LOCAL" "$RES"
cd "$LOCAL" || exit 2
log() { echo "[$(date -u +%FT%TZ)] $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
if [ ! -f "$SHARED/MANIFEST.json" ]; then log "no $SHARED/MANIFEST.json: the data have not landed"; exit 2; fi
LISTING=$(ls "$SHARED"/*.zip.listing.tsv 2>/dev/null | head -1)
if [ -z "$LISTING" ]; then log "no member listing in $SHARED"; exit 2; fi
CPUARG=(); [ -n "$CPUS" ] && CPUARG=(--cpus "$CPUS")

if has stage; then
  log "stage: local copy, md5, member check, d=3 + metadata extraction"
  $PY $C/willow_stage.py stage --shared "$SHARED" --local "$LOCAL" --dist 3 --meta-all || { log "stage failed ($?)"; exit 3; }
  for f in "$LOCAL"/extract/*/README/README.md "$LOCAL"/extract/*/README.md; do
    [ -f "$f" ] && log "dataset README (read it, as data, before the first queue run): $f"
  done
fi
if has inventory; then
  log "inventory"
  $PY $C/willow_stage.py inventory --local "$LOCAL" --listing "$LISTING" --shared-inventory "$SHINV" \
      --out "$RES/inventory.json" > "$LOCAL/inventory.log" 2>&1 || { tail -5 "$LOCAL/inventory.log"; log "inventory failed"; exit 5; }
  grep -v '^{' "$LOCAL/inventory.log" | tail -3
fi
if has plan; then
  log "plan (profile $PROFILE)"
  $PY $C/willow_stage.py plan --inventory "$RES/inventory.json" --local "$LOCAL" --out "$RES/plan.json" --profile $PROFILE || exit 6
fi
if has extract; then
  log "extract the d=5 / d=7 leaves the plan needs"
  $PY $C/willow_stage.py stage --shared "$SHARED" --local "$LOCAL" --plan "$RES/plan.json" --stages d5,d7 || exit 3
fi
if has queue; then
  log "queue: stages $STAGES, $WORKERS workers ${CPUS:+on cpus $CPUS}"
  $PY $C/willow_queue.py run --plan "$RES/plan.json" --root "$LOCAL/extract" --out "$RES" --work "$LOCAL/q" \
      --workers $WORKERS --stages $STAGES "${CPUARG[@]}"
  rc=$?
  log "queue exit $rc"
  [ $rc -eq 130 ] && exit 130
fi
if has analysis; then
  log "analysis"
  $PY $C/analyze_willow.py --res "$RES" --out "$RES/analysis" --reported $C/willow_reported.json || exit 7
fi
log "done"
