#!/bin/bash
# run_logged.sh LOG CMD [ARGS...]
# Runs CMD with its stdout and stderr appended to LOG, and appends a line with the UTC time (date -u) before it
# starts and after it ends, with its exit code. Used to launch D1's MPS workers detached (amendment D1-4), only if
# Kourosh allows runs past the 2-hour background limit. No other effect. (decoder, 2026-10-13)
set -u
log="$1"
shift
echo "[run_logged] start $(date -u +%Y-%m-%dT%H:%M:%SZ) pid $$ cmd: $*" >> "$log"
"$@" >> "$log" 2>&1
rc=$?
echo "[run_logged] end $(date -u +%Y-%m-%dT%H:%M:%SZ) pid $$ exit $rc" >> "$log"
exit "$rc"
