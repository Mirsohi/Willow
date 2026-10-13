# Round 2: relaunch after a container restart (decoder)

For @main. Updated 2026-10-13 03:41Z by decoder. The queue finished at 19:06:16Z, so there is no queue to restart.
After a restart, run every block under **Active now** except those marked "skip on a restart", and nothing else.

**Status at 03:41Z on 2026-10-13: the r = 10 follow-up's τ4 rung (§2.10) runs. A restart relaunches §2.10 only.**
Every other block under **Active now** is marked "skip on a restart".
- §2.10 runs τ4 = 1e-9 on X/r10 and Z/r10, which still fail after τ3 (`willow_prereg_r3.md` §10a and §10b).
  The 13 core-hour cap covers §2.9 and §2.10 together.
- §2.9's driver exited 0 at 03:40:35Z: X/r10 and Z/r10 both still fail after τ3 (rule 2′).
  It used 6.57 core-hours. Do not relaunch it.
- PR3's runner (§2.4) exited 0 at 15:46:02Z: d = 7 ML was killed, because X/r50 and Z/r50 still fail after the
  extra rung (`results/decoder/willow/pr3/main/pr3_tally.json`, md5 1fae8f45…). It used 82.85 core-hours against
  its 116.8 guard. Do not relaunch it.
- The drain helper (§2.7) was ended at 15:53Z, with nothing left to pause. The log shipper (§2.8) was stopped with
  its STOP file and exited at 15:58:39Z, after the final runner log and ledger were shipped.
- Finished earlier, so skip them too: K2's runner (§2.1, 17:09:59Z on 11 Oct), round 1's final analysis (§2.5,
  17:30Z), the mech fits (§2.3, 19:36:22Z) and the chain (§2.6, 22:13:35Z).

General rules:
- **Resuming.** Each run resumes from what it has already written, and it skips finished work.
- **Repeats are harmless.** Running a block twice, or after its run has finished, does no harm. The process finds
  nothing to do and exits within a minute.
- **Limit.** Since the queue finished, decoder runs at most 4 working processes (main's [notice] post headed 19:07Z):
  now 2 for mech and 2 for K2. K2's runner only waits for its children, so K2 counts as its `--parallel` (2). Until
  19:06Z the limit was 3. PR3's runner counts every heavy process on the machine, whoever started it, and keeps the
  total at 4. Run the check in §1 first.
- **Priority.** Since 10:27Z decoder's runs use normal priority, as main's [decision] post headed 10:23Z sets: no
  nice, and the autogroup nice stays 0. The blocks under History show the nice-10 commands used before.
- **Logs.** Each run appends to its shared log through a file it keeps open. After the 2026-10-11 restart, the shared
  logs that were open at the 02:29Z fault were missing or stale, so the local `.out` named in each block (the same
  lines) is the full record. Snapshots of those are beside the shared logs: `*.local_stdout_20261011T1351Z.log`.

## 1. Check what of decoder's is still running

```
ps -eo pid,etime,cmd | grep -E 'code/decoder/(round2|supplementary|pr3)/|decode_willow.py|pr3_drain.py|logship.py' | grep -v grep
```

A printed line means that run is still alive. Do not start a second copy of it.

## 2. Active now

### 2.1 K2's runner: the extra rung on the remaining d = 5 leaves (round 1, `willow_prereg.md` §9 entries 10 and 11): finished at 17:09:59Z on 11 Oct

**Finished; skip on a restart.** All 32 leaves have verdicts: 25 pass and 7 still fail, so no kill (`k2_tally.json`, md5
a16b5c57…). Do not run this block again: a runner that finds all 32 verdicts writes `k2_tally.json` again, and that md5
is cited.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
K=/home/claude/work/decoder/k2
mkdir -p $K/logs $K/numba_cache
cd $K
NUMBA_CACHE_DIR=$K/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/supplementary/k2_runner.py \
  --root /home/claude/work/decoder/willow_run/extract --res $X/results/decoder/willow \
  --out $X/results/decoder/willow/supplementary/k2_extra_rung --logs $K/logs --ledger $K/runner_ledger.json \
  --budget-hours 34 --parallel 2 >> $K/logs/k2_runner.out 2>&1 < /dev/null &
```

**How it runs**
- The runner only waits. It runs up to 2 children at a time (`--parallel 2`, since 19:33Z; one before),
  `k2_extra_rung.py --leaf <leaf>`. Whenever a slot is free it starts the cheapest pending leaf (fewest rounds, then
  fewest band shots). All 32 ladders have finished.
- Every `k2_extra_rung.py` process already running holds a slot, for example a leaf left by an earlier runner, and its
  leaf is not started again.
- Before each leaf it logs its projection for all the extra rungs, by the method of decoder's board post headed
  11:24:53Z. If the projection is past 34 core-hours, it logs that at once, starts no new leaf, lets the running
  leaves finish and exits with status 6.
- It exits when all 32 d = 5 leaves have verdicts, after writing `k2_tally.json` with the kill rule applied.

**How it resumes**
- A leaf that already has its `extra_rung.json` is skipped.
- An interrupted leaf starts again from the beginning: about 1–2 CPU hours at r = 30 and 1.2–4.1 at r = 50
  (estimate). Only r = 30 and 50 leaves remain.
- The CPU ledger is kept twice, in `results/.../k2_extra_rung/runner_ledger.json` and in
  `/home/claude/work/decoder/k2/runner_ledger.json`. The runner merges the two.

**Checking on it**
- **Progress:** `tail -n 3 $X/results/decoder/willow/supplementary/k2_extra_rung/k2_runner.log`
- **Finished:** that log ends with "all 32 leaves have verdicts".

**Exit statuses** (each is logged at once; the runner exits when the running leaves have finished, since no new leaf
starts)
- **3, "K2 budget reached":** the next leaf would take the extra rungs past 34 core-hours, your limit (16 until
  12:15Z, then 28 until 15:10Z). Do not relaunch; decoder reports.
- **6, "K2 projection past the budget":** the projection for all the extra rungs is past 34 core-hours. Do not
  relaunch; decoder reports.
- **4:** a leaf has no verdict (its rebuild check failed). Decoder looks at it.
- **5:** a child failed. Decoder looks at it.

### 2.3 The real mech fits, T̂_mech and D̂_mech, with their held-out decodes (PR2 §8.2): finished at 19:36:22Z on 11 Oct

**Finished; skip on a restart.** Part 1 ended at 16:26:09Z and part 0 at 19:36:22Z on 11 Oct: 18 of 18 series done,
27.47 of the 30 core-hours used (part 0's log).

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
R=/home/claude/work/decoder/round2
mkdir -p $R/logs $R/numba_cache
cd $R
for P in 0 1; do
  NUMBA_CACHE_DIR=$R/numba_cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  setsid nohup python3 -I -B $X/code/decoder/round2/run_mech.py \
    --extract /home/claude/work/decoder/willow_run/extract --res $X/results/decoder/willow \
    --f1 $X/results/decoder/willow/round2/f1 --out $X/results/decoder/willow/round2/mech --part $P/2 \
    --max-iter 25 --cpu-budget-hours 30 >> $R/logs/run_mech_$P.out 2>&1 < /dev/null &
done
```

**How it runs**
- Part 0 runs the 9 X series and part 1 the 9 Z series, each process in its own session.
- For each series: the T̂_mech fit (ln P(s), from 0), the D̂_mech fit (ln P(L|s) with the ridge, from the series' F1 D̂),
  each capped at 25 steps, then the held-out decodes of its 4 leaves under both. (§8.2's cap was 15. Main raised it to
  25 at 15:10Z, before any mech result; PR2 §7 entry 15:14Z.)
- The guard counts the mech CPU so far: the fits, including any fit in progress, and the held-out decodes. At 30
  core-hours, main's budget (25 until 15:10Z), the process stops before its next evaluation, keeps its checkpoint and
  exits.
- After each series the log gives the CPU so far and the projected total (mean per finished series × 18).

**How it resumes**
- A series that has `mech_fit_T.json`, `mech_fit_D.json` and the 4 leaves' `eval_mech.json` logs "done already".
- A fit in progress resumes from its checkpoint, `results/decoder/willow/round2/mech/<series>/mech_ckpt_<T|D>.ckpt.npz`.
  A restart loses at most the evaluation in progress, 2–4 CPU minutes.
- A finished fit is read back from its json. The held-out decodes of the series in progress start again, about 3 CPU
  minutes.

**Checking on it**
- **Progress:** `tail -n 3 $X/results/decoder/willow/round2/mech/run_mech_part*of2.log`
- **Finished:** part 0's log ends with `d3_at_q8_9/X: done; ...`, and part 1's with `d3_at_q8_9/Z: done; ...`.

**Exit status 3, "CPU budget reached":** mech has used 30 core-hours, main's budget. Do not relaunch, and do not raise
the budget; decoder reports.

### 2.4 PR3's main run: the runner (`notes/decoder/willow_prereg_r3.md` §8), up to 4 working processes: finished at 15:46:02Z on 12 Oct, d = 7 ML killed

**Finished; skip on a restart.** It exited 0 at 15:46:02Z on 12 Oct: d = 7 ML killed, as X/r50 and Z/r50 still fail
after the extra rung (`pr3_tally.json`, md5 1fae8f45…; its ledger, md5 7c28d6c1…). Do not run this block again: a
relaunched runner starts no part, since the kill holds, but it rewrites `pr3_tally.json` and its ledger, whose md5s
are cited. The leaf table is in `results/decoder/willow/pr3/leaf_table/`. The rest of this section is kept for the
record.

**While it ran: after a restart, clear the drain's pause first (§2.7).**
The runner reads `$P/main_state/control.json` at every poll
(15 s), and it starts no new part while that file's `"pause"` holds a true value: `true`, a non-zero number or any
non-empty string, even `"false"`. So if the file holds `"pause": true`, remove that key before or when you run the
block below, or the runner stays paused. Keep the file a JSON object: the runner ignores a file it cannot parse, and
with it every hold and release. This removes only the `"pause"` key, and does nothing if the file is missing:
```
C=/home/claude/work/decoder/pr3/main_state/control.json
[ -f $C ] && python3 -I -c 'import json,os,sys; p=sys.argv[1]; d=json.load(open(p)); d.pop("pause",None); open(p+".tmp","w").write(json.dumps(d)); os.replace(p+".tmp",p); print(d)' $C
```

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
P=/home/claude/work/decoder/pr3
D=/home/claude/work/decoder/pr3_data
mkdir -p $P/logs/main $P/numba_cache $P/main_work $P/main_state
cd $P
NUMBA_CACHE_DIR=$P/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/pr3/pr3_runner.py --root $D/extract \
  --res $X/results/decoder/willow/pr3/main --work $P/main_work --state $P/main_state --logs $P/logs/main \
  --pilot $X/results/decoder/willow/pr3/pilot/pilot.json --design 25k --guard-core-hours 116.8 \
  >> $P/logs/pr3_runner.out 2>&1 < /dev/null &
```

**How it runs**
- `pr3_runner.py` (md5 4bd2a21a…; before it 457c017a…, 48b2f729…, 24b31825… and b42595c3…) only schedules.
  **Version 5 has run since 2026-10-11 13:48Z**, started after the container restart. Version 4 (457c017a…, from
  21:18Z) died in the 02:29Z fault. It had every hold below except the non-band check's, because its restart as
  version 5 was refused by the environment at 22:06Z (main's decision at 22:14Z: option (a)).
- The runner starts parts as child processes:
  - ladder parts: the frozen `decode_willow.py`;
  - extra-rung parts: `pr3_extra_rung.py`.
- There are 30 ladder parts and up to 30 extra parts, about 90 core-hours in all, most expensive first.
- **Slots.** It keeps the heavy processes on the machine at 4 at most. It leaves:
  - up to 2 slots to K2 while K2's runner is alive;
  - 1 slot to each in-sample part (`run_insample.py --part i/2`) from the end of the matching mech part until that
    in-sample part has started;
  - after the mech fits, 1 slot to reservoir's mech check (its `r2m_*.py` scripts, run by `run_r2m.sh`) until it has
    run and ended. Nothing seen before the mech fits end counts;
  - once K2's tally is complete, 1 slot to round 1's final analysis (§2.5), until its output exists;
  - after the mech fits, 1 slot to the post hoc non-band check (§2.6), until it has run and ended or its summary is
    final.
- **Memory and the guard.** A part starts only if 3 GiB of memory stays free. Each part runs under a 4 GiB cap. The
  guard is 116.8 core-hours.
- **Control.** `$P/main_state/control.json` can release a hold or pause the runner:
  - `{"reservoir": "released"}` frees the slot held for reservoir's check, if that check will not run;
  - `{"insample": "released"}` frees the in-sample slots;
  - `{"final_round1": "released"}` frees the final analysis's slot;
  - `{"nonband": "released"}` frees the non-band check's slot;
  - `{"hold": 1}` holds one more slot, for a job not listed here;
  - `{"pause": true}` stops new parts from starting.

**How it resumes**
- Run the block again. Finished parts are found from their outputs, and ladder chunks in `$P/main_work` resume.
- After a crash: the ledger and each chunk are written whole (a temporary file, then a rename), and so are a ladder
  part's outputs except `shots.npz`, which a restarted part rewrites. The runner books a restarted part's earlier CPU
  in full as `lost_cpu_s`, though its chunks are kept, so the projection runs high until that part ends.
- A part still running from an earlier runner keeps its slot and is not started twice.
- If `$D/extract` is missing, re-create it with `pr3_extract.py` into new folders (§2.4's History entry has the
  command), and pass the new folder as `--root`.

**Checking on it**
- `tail -n 5 $P/logs/pr3_runner.out` (a status line every 30 minutes).
- `results/decoder/willow/pr3/main/pr3_tally.json`: the leaf verdicts and the projection.
- **Exit statuses:**
  - 0: all parts done, or d = 7 ML killed (see `pr3_tally.json`);
  - 5: a part failed;
  - 6: the projection passed the 116.8 core-hour guard. Do not relaunch; decoder reports to main;
  - 7: a part hit the memory cap.

### 2.5 Round 1's final analysis: waits for K2's complete tally, then runs once (`willow_prereg.md` §9 entry 12): finished at 17:30Z on 11 Oct

**Finished; skip on a restart.** It ran on K2's complete tally and wrote `final_round1.json` (md5 6ff5d375…) and
`final_round1.md` at 17:30:19Z on 11 Oct.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
F=/home/claude/work/decoder/final
mkdir -p $F/logs
cd $F
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/supplementary/final_round1_trigger.py \
  --k2 $X/results/decoder/willow/supplementary/k2_extra_rung/k2_tally.json --out $X/results/decoder/willow/final_round1 -- \
  --res $X/results/decoder/willow --k2 $X/results/decoder/willow/supplementary/k2_extra_rung/k2_tally.json \
  --frozen $X/results/decoder/willow/analysis/analysis.json --out $X/results/decoder/willow/final_round1 \
  >> $F/logs/final_round1.out 2>&1 < /dev/null &
```

**How it runs**
- `final_round1_trigger.py` (md5 00664d73…) waits, light, until `k2_tally.json` says `"complete": true`.
- It then becomes `final_round1.py` (md5 5f66be5b…): about 25 CPU minutes, one process.
- It writes `results/decoder/willow/final_round1/`: `final_round1.json`, `final_round1.md` and `analysis/`.
- It never runs on an incomplete tally. If K2 stops early (its exit 3, 5 or 6), the orchestrator decides.

**How it resumes**
- Run the block again. If `final_round1.json` already exists from a complete tally, it exits at once.
- If `final_round1.py` was killed while running, the rerun repeats it from the start, about 25 CPU minutes.

**Checking on it**
- `tail -n 5 $F/logs/final_round1.out`.
- Finished: `results/decoder/willow/final_round1/final_round1.md` exists. Decoder (or main) posts its decisions.

### 2.6 After the mech fits: the chain (in-sample decodes, `analyze_explore.py`, the post hoc non-band check): finished at 22:13:35Z on 11 Oct

**Finished; skip on a restart.** All 5 steps are done (`state.json`), and `chain.log` ends with "chain ends (exit 0)" at
22:13:35Z on 11 Oct.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
C=/home/claude/work/decoder/chain
mkdir -p $C/logs
cd $C
setsid nohup python3 -I -B $X/code/decoder/supplementary/post_mech_chain.py \
  --extract /home/claude/work/decoder/willow_run/extract \
  --out $X/results/decoder/willow/supplementary/post_mech_chain \
  >> $C/logs/post_mech_chain.out 2>&1 < /dev/null &
```

**How it runs**
- `post_mech_chain.py` (md5 0f03e524…) polls once a minute and uses almost no CPU. It starts each step as a process
  in its own session, in main's order (decisions relayed at 20:00Z and 21:40Z):
  1. In-sample part i (`run_insample.py --part i/2`, PR2 §8.4) as soon as mech part i has ended with all its series
     complete. The command is the History block "In-sample decodes under D̂" with `--part i/2` and no nice. About 50
     minutes per part [assumed: the D̂ run took 47 minutes for 18 series × 1 decoder; a part is 9 series × 2].
  2. `analyze_explore.py`, once both in-sample parts are done (minutes). It rewrites
     `results/decoder/willow/round2/explore_analysis.json`, which reservoir's mech check reads. The earlier file is
     first copied to the chain's folder as `explore_analysis.before_chain.json`.
  3. The post hoc non-band check (`nonband_check.py`; `willow_prereg.md` §9 entry 13): about 3 core-hours, with a hard
     stop at 4.0 (exit 3).
  4. Its analysis (`nonband_analyze.py`, minutes), once round 1's final analysis exists (§2.5).
- PR3's runner holds a slot for each in-sample part and for the check (§2.4).
- **A mech part that ends incomplete** (for example at the mech budget, exit 3) leaves its in-sample part "blocked".
  The chain logs it and goes on to the check; decoder or main decides. If that mech part later completes while the
  chain still runs, the chain starts its in-sample part.
- A step that fails is not retried by the same chain process. Running this block again retries it.

**How it resumes**
- Run the block again. The state is in `results/decoder/willow/supplementary/post_mech_chain/state.json`.
  - A running step is adopted, not started twice. A finished step is recognised from its outputs.
  - A second chain with the same `--out` exits at once.
- `run_insample.py` keeps the finished decoders of each leaf. `nonband_check.py` resumes at its next batch of 25 shots.

**Checking on it**
- `tail -n 5 $X/results/decoder/willow/supplementary/post_mech_chain/chain.log`
- In-sample: `tail -n 2 $X/results/decoder/willow/round2/insample/run_insample_part*of2.log`
- Check: `tail -n 3 $X/results/decoder/willow/supplementary/nonband_check/nonband_check.log`; `summary.json` there gives
  its status.
- **Finished:** `chain.log` ends with "chain ends (exit 0)". Exit 1 means a step failed or is blocked; `state.json`
  says which.

### 2.7 PR3's drain helper: a timed pause before the expected outage (main's decision relayed 2026-10-11 18:00Z), light: stopped at 15:53Z on 12 Oct

**Stopped; skip on a restart.** PR3's runner has exited, so there is nothing to pause. Decoder ended the helper with
SIGTERM at 15:53Z on 12 Oct, since it has no stop file of its own. It never wrote a pause: no `control.json` exists.
The rest of this section is kept for the record.

```
cd /home/claude/work/decoder/pr3/drain && NUMBA_CACHE_DIR=$PWD/numba_cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 setsid nohup python3 -I -B pr3_drain.py --wake 2026-10-12T20:15:00Z --deadline 2026-10-13T01:15:00Z --clear-at 2026-10-13T02:00:00Z >> /home/claude/work/decoder/pr3/logs/pr3_drain.out 2>&1 < /dev/null &
```

**While the runner ran: on a restart, clear its pause first (§2.4), then run this block with its three times moved.**
The times above are
for the container that started at 01:38:13Z on 12 Oct (main's decision relayed at about 01:40Z). For a new container,
set the deadline 23 h 37 min after it started, the folder check (`--clear-at`) 45 min after the deadline, and the
wake 5 h before the deadline. Its start time: `date -u -d @$(awk '/^btime/{print $2}' /proc/stat)`. These times
assume the shared folder fails 24 h after a container starts, which rests on one observation (the fault at 02:29Z
on 11 Oct). The helper is not needed once PR3's runner has exited. If it dies while its container still runs, run
the block again unchanged: it goes on from its `drain_state.json`.

**How it runs**
- `pr3_drain.py` (md5 9e8a511e…; local only, in `/home/claude/work/decoder/pr3/drain/`) reads and sleeps. Its one
  write for the runner is `{"pause": true}` in `$P/main_state/control.json`, made once, with the other keys kept (a
  temporary file, then a rename).
- Until the wake time (20:15Z on 12 Oct) it logs a preview every hour.
- From then, every 60 s, it reads the runner's state with the runner's own code (local copies with the same md5s,
  imported read-only). It finds the most expensive unit the runner could start next: a pending ladder part, an extra
  part of a leaf that failed the ladder rules, or an extra part of a leaf whose ladder parts have all started.
- It writes the pause at T = the deadline (01:15Z on 13 Oct) − 1.1 × that unit's wall time (the runner's CPU
  projection over the measured CPU per wall second, at most 0.98). If the state cannot be read for 10 minutes, or the
  last T it read has passed, it writes the pause at once.
- The pause only stops new parts: the running parts finish.
- At the folder check (02:00Z on 13 Oct) it checks that the shared folder is readable, in child processes with
  timeouts. If it is, the helper removes its own pause and exits. If not, it leaves the pause and exits.
- It never removes a pause it did not write, and it writes only once: deleting the key undoes its pause.

**Checking on it**
- `tail -n 3 /home/claude/work/decoder/pr3/logs/pr3_drain.out`: a preview each hour until the wake time; then a
  line when the next unit changes, or every 10 minutes.
- `cat /home/claude/work/decoder/pr3/main_state/control.json`: no file, or no `"pause"` key, means not paused.

### 2.8 The log shipper: copies decoder's local logs to the shared folder every 10 minutes (main's decision relayed 2026-10-11 18:00Z), light, nice 10: stopped at 15:58:39Z on 12 Oct

**Stopped; skip on a restart.** Decoder created its STOP file at 15:53Z on 12 Oct, after the final runner log and
ledger were shipped (15:48:38Z). Its last cycle, at 15:58:38Z, copied the drain's log and its own; it then exited.
Leave the STOP file in place: a shipper started by mistake then runs one cycle and exits. The rest of this section
is kept for the record.

```
cd /home/claude/work/decoder/ship && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 setsid nohup nice -n 10 python3 -I -B logship.py >> /home/claude/work/decoder/ship/logship.out 2>&1 < /dev/null &
python3 -c "import time; time.sleep(3)"
for P in $(ps -eo pid=,args= | awk '$2=="python3" && /logship\.py/ && !/--worker/ {print $1}'); do echo 10 > /proc/$P/autogroup; done
```

**While it ran: run this block on a restart.**
If `/home/claude/work/decoder/ship/STOP` exists, delete it first, or the shipper runs one
cycle and exits. A leftover `logship.lock` does no harm: the lock is an flock, released when its holder dies.

**How it runs**
- `logship.py` (md5 67fc3322…; local only, in `/home/claude/work/decoder/ship/`) sleeps. Every 600 s it starts one
  worker process with a 300-s timeout. A worker still running then is killed and never waited for, and no new one
  starts while it is stuck.
- The worker copies each local log that changed, and PR3's `ledger.json`, to `results/decoder/willow/live_logs/<group>/`:
  first as `.<name>.shiptmp`, then renamed. It then rewrites the index, `live_logs/SHIPPED.json`: for each copy, its
  source, bytes, md5, the source's time and the copy's time.
- The sources: PR3's runner and drain logs and its `ledger.json` (group `pr3`), PR3's part logs (`pr3/main`), and the
  logs of mech, the chain, K2, the final analysis and the shipper itself.
- A file over 200 MiB is not copied, so every shared file stays under 256 MiB.
- One shipper per folder: a second one exits at once. Its state is `ship_state.json` beside it.
- Its tests, on local folders only, are in `ship/tests/`: 25 of 25 checks pass (`run_ship_tests.out`).

**Stopping it:** `touch /home/claude/work/decoder/ship/STOP`. It runs one last cycle and exits.

**Checking on it**
- `tail -n 2 /home/claude/work/decoder/ship/logship.out`: one line per cycle.
- The copies and their md5s: `results/decoder/willow/live_logs/SHIPPED.json`.

### 2.9 PR3's post hoc r = 10 follow-up (`notes/decoder/willow_prereg_r3.md` §10), up to 4 working processes: started at 01:43:01Z, finished at 03:40:35Z on 13 Oct

**Skip on a restart.** The driver exited 0 at 03:40:35Z, with every part done (6.57 core-hours). A relaunched
driver would start no part, but it would rewrite `r10_tally.json`, whose md5 the results cite.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
P=/home/claude/work/decoder/pr3_r10
D=/home/claude/work/decoder/pr3_data
mkdir -p $P/logs/parts $P/work $P/state
cd $P
NUMBA_CACHE_DIR=/home/claude/work/decoder/pr3/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/pr3/pr3_r10_posthoc.py --root $D/extract \
  --res $X/results/decoder/willow/pr3/r10_posthoc --work $P/work --state $P/state --logs $P/logs/parts \
  --pilot $X/results/decoder/willow/pr3/pilot/pilot.json --design 25k --guard-core-hours 13 \
  >> $P/logs/pr3_r10_posthoc.out 2>&1 < /dev/null &
```

**How it runs**
- `pr3_r10_posthoc.py` (md5 f6e263ca…) runs PR3's runner class (`pr3_runner.py`, md5 4bd2a21a…, unchanged) on the
  two r = 10 leaves only. It sets `pr3_common.ROUNDS = (10,)` in its own process.
- Units: 10 ladder parts (the frozen `decode_willow.py`), then 5 extra parts (`pr3_extra_rung.py`) for a leaf that
  fails its ladder rules. Shots, decoders, rules and the extra rung are PR3's (§2.4).
- It holds no slot and has no kill rule. It keeps the heavy processes on the machine at 4 at most. A part starts only
  if 3 GiB of memory stays free, and each part runs under a 4 GiB cap.
- The guard is 13 core-hours. The projection at the start was 7.2 core-hours (estimate).
- `$P/state/control.json` with `{"pause": true}` stops new parts from starting. Remove the key to resume.
- No drain helper and no log shipper. Decoder copies the logs and the ledger to `r10_posthoc/logs/` at the end.

**How it resumes**
- Run the block again. Finished parts are found from their outputs, and ladder chunks in `$P/work` resume.
- A part still running from an earlier driver keeps its slot and is not started twice.
- If `$D/extract` is missing, re-create it as in §2.4.

**Checking on it**
- `tail -n 5 $P/logs/pr3_r10_posthoc.out` (a status line every 30 minutes).
- `results/decoder/willow/pr3/r10_posthoc/r10_tally.json`: the two verdicts and the projection.
- **Exit statuses:**
  - 0: all parts done (see `r10_tally.json`);
  - 5: a part failed;
  - 6: the projection passed the 13 core-hour guard. Do not relaunch; decoder reports to main;
  - 7: a part hit the memory cap.

### 2.10 The r = 10 follow-up's τ4 rung (`notes/decoder/willow_prereg_r3.md` §10a and §10b), up to 4 working processes: started at about 03:41Z on 13 Oct

**Run this block on a restart.** First run the check in §1: if `pr3_r10_tau4.py` is still alive, do not start a
second copy. §2.9's driver must have exited.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
P=/home/claude/work/decoder/pr3_r10
D=/home/claude/work/decoder/pr3_data
mkdir -p $P/logs/tau4 $P/work_tau4 $P/state_tau4
cd $P
NUMBA_CACHE_DIR=/home/claude/work/decoder/pr3/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/pr3/pr3_r10_tau4.py --root $D/extract \
  --res $X/results/decoder/willow/pr3/r10_posthoc --ledger $P/state/ledger.json --work $P/work_tau4 \
  --state $P/state_tau4 --logs $P/logs/tau4 --design 25k --guard-core-hours 13 \
  >> $P/logs/pr3_r10_tau4.out 2>&1 < /dev/null &
```

**How it runs**
- `pr3_r10_tau4.py` (md5 b9ac5207…) runs one more rung, τ4 = 1e-9, on the band shots of X/r10 and Z/r10, still failing
  after τ3. 5 parts per leaf, each part one process. The rules between τ3 and τ4 are 10a's.
- Each part first decodes 10 band shots again at τ3, which must match the stored τ3 bitwise.
- The cap is 13 core-hours in total, the driver's 6.57 included. The projection at the start was 11.12 (the driver's 6.57, τ4 on X/r10 2.00 and on Z/r10 2.54)
  core-hours (estimate). If the total CPU reaches 13, every τ4 part is killed.
- At most 4 heavy processes on the machine. A part starts only if 3 GiB of memory stays free, and each part runs
  under a 4 GiB cap.

**How it resumes**
- Run the block again. Finished parts are found from their outputs, and τ4 chunks in `$P/work_tau4` resume. The CPU a
  lost part had used is booked as lost.
- It refuses to start while a τ4 part from an earlier run is alive: wait for it to end, or stop it.

**Checking on it**
- `tail -n 5 $P/logs/pr3_r10_tau4.out`.
- `results/decoder/willow/pr3/r10_posthoc/r10_tau4_tally.json`: the status, the projection and the verdicts.
- **Exit statuses:**
  - 0: all τ4 parts done (see `r10_tau4_tally.json`);
  - 2: the driver or an earlier τ4 part is still running;
  - 4: a τ3 rebuild check failed;
  - 5: a part crashed. Run the block again to retry it;
  - 6: the cap. Do not relaunch; decoder reports to main;
  - 7: a part hit the memory cap.

## 3. Later stages

Not active yet. Decoder adds each stage's block under "Active now" before the stage starts:
- (the in-sample decodes under T̂_mech and D̂_mech and the post hoc non-band check now wait in §2.6, and round 1's
  final analysis in §2.5.)

## 4. History

| Time (2026-10-10) | Event |
|---|---|
| 02:28Z | F1 started. |
| 02:29Z | F1 lost in the restart, before its first checkpoint. |
| 02:43Z | F1 relaunched. Checkpoints are now a single file, and the 12-core-hour guard is built in. |
| 05:38Z | K2's extra rung started: one d = 5 leaf at a time, at nice 10. |
| 09:08Z | F1 finished: 18 of 18 series converged, 8.49 core-hours. |
| 09:12Z | In-sample decodes under D̂ started. |
| 09:32Z | K2's extra rung on d5_at_q6_5/Z/r13 started. |
| 09:46Z | This file now lists the active runs. F1 moved here. |
| 09:59Z | In-sample decodes under D̂ finished: 72 of 72 leaves. Moved here. |
| 10:13Z | K2's extra rung on d5_at_q6_5/Z/r13 finished (pass). Moved here. Nothing of decoder's is running. |
| 10:27Z | Orchestrator's decisions: K2 for all remaining leaves, the mech budget raised to 25 core-hours, up to 3 processes at normal priority. K2's extra rung on d5_at_q6_9/X/r10 started. |
| 10:35Z | K2's extra rung on d5_at_q6_9/X/r10 finished (pass). Moved here. |
| 10:39Z | K2's runner started (§2.1); `willow_prereg.md` §9 entry 11. |
| 10:47Z | Mech validation started (§2.2); PR2 §7 entry 10:47Z (budget 25 core-hours, guard). |
| 11:48Z | Mech validation V0 finished: PASS. V2 continues. |
| 12:25Z | Main's [decision] post headed 12:15Z raised K2's stop from 16 to 28 core-hours. K2's runner restarted with `--budget-hours 28`, a projection guard and a wait for a running leaf (`k2_runner.py` md5 cfd65967…; `willow_prereg.md` §9 entry 11, addendum). The leaf in progress, d5_at_q6_5/Z/r30, was left to finish. |
| 14:08Z | Mech validation V2 finished: PASS. Mech validation moved here. |
| 14:09Z | The real mech fits started (§2.3), 2 processes, with the 25-core-hour guard (PR2 §7 entry 10:47Z). |
| 15:11Z | The mech fits paused at their checkpoints, both T̂_mech fits of d3_at_q10_7 at iteration 13, for main's decision on the cap. |
| 15:15Z | Main's [decision] post headed 15:10Z: the mech fits' cap from 15 to 25 steps (PR2 §7 entry 15:14Z), the mech budget from 25 to 30 core-hours, K2's stop from 28 to 34 (`willow_prereg.md` §9 entry 11, addendum 2). The mech fits relaunched with `--max-iter 25 --cpu-budget-hours 30` (`run_mech.py` md5 18ea250a…), resuming from their checkpoints. K2's runner restarted with `--budget-hours 34`; the leaf in progress, d5_at_q6_5/X/r30, was left to finish. |
| 19:33Z | The queue finished at 19:06:16Z (main's [notice] post headed 19:07Z): no queue to restart, and decoder may use 4 processes. K2's runner restarted with `--parallel 2` (`k2_runner.py` md5 6e57ff0b…; `willow_prereg.md` §9 entry 11, addendum 3); the leaf in progress, d5_at_q8_7/Z/r30, was left to finish and holds one slot. |
| 20:19Z | Main's [decision] post headed 19:58Z: round 1's exclusion rule A (`willow_prereg.md` §9 entry 12) and PR3 approved (`willow_prereg_r3.md` §8). PR3's pilot started (§2.4) as a fifth process, under a 4 GiB cap. |
| 20:39Z | PR3's pilot finished at 20:35Z: 89.9 core-hours projected for 25,000 shots per leaf, so 25,000 by main's rule; guard 116.8 core-hours (`willow_prereg_r3.md` §8). Pilot moved here. PR3's runner started (§2.4); it waits for a free slot. |
| 20:46Z | PR3's runner restarted with md5 24b31825… (a slot held for round 1's final analysis, manual holds, fewer tally writes; no part had started). Round 1's final analysis now waits for K2's complete tally (§2.5; `willow_prereg.md` §9 entry 12). |
| 21:14Z | PR3's runner restarted with md5 48b2f729… (`willow_prereg_r3.md` §8, runner update 2). Version 2 released reservoir's slot at 21:07Z, while the mech fits still ran, after a reservoir analysis process had run; the holds now start only when the mech fits end, and reservoir's follows only its `r2m_*.py` scripts. No part had started. |
| 21:18Z | PR3's runner restarted with md5 457c017a… (`willow_prereg_r3.md` §8, runner update 3): each in-sample part is held a slot from the end of its mech part, so the first mech part to end does not free a slot for PR3. No part had started. |
| 22:05Z | Main's post hoc non-band check logged (`willow_prereg.md` §9 entry 13; shot lists md5 56bdb30b…). PR3's runner version 5 (md5 4bd2a21a…; `willow_prereg_r3.md` §8, runner update 4: 1 slot held for the check after the mech fits) put on disk. |
| 22:06Z | The restart of PR3's runner as version 5 was refused by the environment. Version 4 (457c017a…) keeps running, without a slot for the check. |
| 22:14Z | Main's decision: version 4 keeps running as it is; the non-band check may run as a 5th heavy process for its roughly 3 core-hours (option (a)). Version 5 takes over only after a container restart (§2.4). No change to the chain or to entry 13. |
| 22:10Z | The chain started (§2.6, pid 6191, md5 0f03e524…): it waits for the mech fits. |
| 2026-10-11 02:29Z | The shared folder began returning I/O errors. Both mech parts, the chain and PR3's runner (version 4) crashed on it. K2's runner, its 2 leaves (d5_at_q8_7/X/r50, d5_at_q6_9/X/r50) and the trigger stayed alive. |
| 2026-10-11 03:12Z | K2's 2 leaves paused (SIGSTOP by main's helper, approved by Kourosh). |
| 2026-10-11 about 13:21Z | Container restart: every process died, and the local disk survived. K2's 2 paused leaves were lost, 2.10 core-hours, excluded from K2's stop (`willow_prereg.md` §9 entry 11, addendum 4). |
| 2026-10-11 13:41Z | Both extracts verified, nothing rebuilt: `willow_stage.py stage` (zip md5, member check, size check: nothing to extract), a CRC-32 check of all 6,015 extracted files against the zip, and PR3's 102 + 12 files against their manifest md5s. |
| 2026-10-11 13:43Z | Mech relaunched (§2.3). Part 1 resumed d3_at_q8_5/Z's T̂_mech fit from its checkpoint (iteration 3); part 0 restarted d3_at_q6_7/X's D̂_mech fit, which had no checkpoint. |
| 2026-10-11 13:44Z | The chain relaunched (§2.6). |
| 2026-10-11 13:47Z | Entry 11, addendum 4 logged; K2's runner relaunched (§2.1) for the 3 leaves without a verdict. |
| 2026-10-11 13:48Z | PR3's runner relaunched as version 5 (§2.4), after K2's runner so that K2's 2 slots were already held. |
| 2026-10-11 13:49Z | The trigger relaunched (§2.5). |
| 2026-10-11 17:09:59Z to 22:13:35Z | Finished: K2's runner (17:09:59Z, all 32 verdicts), round 1's final analysis (17:30Z), the mech fits (part 0 at 19:36:22Z) and the chain (22:13:35Z). Their blocks stay in §2, marked "skip on a restart". |
| 2026-10-12 00:19Z | A fresh decoder agent took over, on Kourosh's approval of 00:05Z. The drain helper's block added (§2.7), with the note in §2.4 on clearing its pause. |
| 2026-10-12 00:26Z | The drain helper started at 00:19:56Z (pid 27679; §2.7). Its first preview: the next unit is the ladder part X/r30 p0, 2.15 core-hours, so the pause time is 10:35:11Z. The log shipper, tested on local folders (25 of 25 checks), started at 00:25:30Z (pid 27903, nice 10; §2.8). |
| 2026-10-12 01:38:13Z | Container restart, unannounced, 12 h 17 min after the one at 13:21Z on 11 Oct: every process died; the local disk and the shared folder survived. PR3's 4 running parts (Z/r50 p2–p4, X/r30 p0) lost about 0.25 core-hours not yet checkpointed; their other chunks were kept. |
| 2026-10-12 01:52Z | Relaunched after checks (PR3's extracts against their manifest, the 4 parts' chunks and specs, PR3's shared outputs and `live_logs`: all intact, no temporary files). PR3's runner at 01:47:38Z (pid 448) resumed its 4 parts from their chunks; the shipper started at 01:48:36Z (pid 515); the drain at 01:50:31Z (pid 603), with its times moved to this container (deadline 01:15Z on 13 Oct): pause time 22:50:11Z. |
| 2026-10-12 15:46:02Z | PR3's runner exited 0: d = 7 ML killed, since X/r50 and Z/r50 still fail after the extra rung (`pr3_tally.json`, md5 1fae8f45…). 82.85 core-hours used against the 116.8 guard. X/r10 and Z/r10 had run no part; X/r30 and Z/r30 had 3 and 1 of their 5 extra parts done. |
| 2026-10-12 16:07Z | Decoder ended the drain helper (SIGTERM at 15:53Z; it never wrote a pause) and stopped the shipper with its STOP file (exit at 15:58:39Z, after the final runner log and ledger were shipped at 15:48:38Z; the shared copy of its own log lacks only its last line). The leaf table written by `pr3_leaf_table.py` (md5 a3f2d7b1…) to `results/decoder/willow/pr3/leaf_table/`. §2.4, §2.7 and §2.8 marked "skip on a restart": a restart relaunches nothing. |
| 2026-10-13 01:42Z | After the container restart of about 00:36Z, nothing needed relaunching (main). Kourosh chose the r = 10 runs at 00:58:22Z. `willow_prereg_r3.md` §10 logged at 01:41Z, before any part started. Block §2.9 added, and the r = 10 driver started (guard 13 core-hours). |
| 2026-10-13 03:41Z | The r = 10 driver exited 0 at 03:40:35Z, after 6.57 core-hours. X/r10 and Z/r10 both still fail after τ3 (rule 2′). Under `willow_prereg_r3.md` §10a, τ4 runs on both: the projection is 11.12 core-hours against the cap of 13. §2.9 marked finished. Block §2.10 added, and τ4 started. |

### PR3 stage 0, the pilot (`notes/decoder/willow_prereg_r3.md` §8): finished at 20:35Z (design 25,000; guard 116.8 core-hours)

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
P=/home/claude/work/decoder/pr3
D=/home/claude/work/decoder/pr3_data
mkdir -p $P/logs $P/numba_cache $P/pilot_work
cd $P
NUMBA_CACHE_DIR=$P/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/pr3/pr3_pilot.py --root $D/extract_pilot \
  --out $X/results/decoder/willow/pr3/pilot --work $P/pilot_work --logs $P/logs \
  --probes $X/results/decoder/willow/decode >> $P/logs/pilot.out 2>&1 < /dev/null &
```

**How it runs**
- `pr3_pilot.py` (md5 824796c2…) runs two stages, each under a 4 GiB RSS cap enforced by its watchdog:
  - the frozen `decode_willow.py` on 200 shots of d7_at_q6_7/X/r50;
  - `pr3_extra_rung.py` on the band shots.
- It then writes `results/decoder/willow/pr3/pilot/pilot.json` with the projection and the design.
- About 0.5 core-hours, so 30–40 minutes as a fifth process.
- Its extract, `$D/extract_pilot`, holds no shipped prediction, so the pilot cannot compare with Libra.

**How it resumes**
- Run the block again. Finished chunks (in `$P/pilot_work`) and a finished extra rung are skipped. The rates come from
  the stored chunks, so they are unaffected; only the CPU outside the decoders is then undercounted.
- If `$D/extract_pilot` is missing, first run `pr3_extract.py` into new, empty folders:
  ```
  python3 -I -B $X/code/decoder/pr3/pr3_extract.py --zip /home/claude/work/decoder/willow_run/zip/google_105Q_surface_code_d3_d5_d7.zip --out <new> --pilot-out <new> --round1 /home/claude/work/decoder/willow_run/extract --manifest <new>.json
  ```

**Checking on it**
- `tail -n 5 $P/logs/pilot.out`
- **Exit statuses:**
  - 0: a design was chosen (`pilot.json`);
  - 2: no design is within main's rule (decoder reports);
  - 3: a stage failed or hit the memory cap (decoder reports).
- After it ends, the block moves to History. Do not relaunch it then.

### F1, the real fits (PR2 §8.3): finished at 09:08Z

This block is kept for the record. Running it again does no harm, because each process finds all of its series done and
exits within a minute.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
R=/home/claude/work/decoder/round2
mkdir -p $R/logs $R/numba_cache
cd $R
for P in 0 1; do
  NUMBA_CACHE_DIR=$R/numba_cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  setsid nohup python3 -I -B $X/code/decoder/round2/run_f1.py \
    --extract /home/claude/work/decoder/willow_run/extract --res $X/results/decoder/willow \
    --out $X/results/decoder/willow/round2/f1 --part $P/2 --cpu-budget-hours 12 \
    >> $R/logs/run_f1_$P.out 2>&1 < /dev/null &
done
```

- Part 0 ran the 9 X series and part 1 the 9 Z series, each process in its own session.
- Each fit checkpointed after every step to `results/decoder/willow/round2/f1/<series>/fit_ckpt.ckpt.npz`. A finished fit
  removes its checkpoint.
- "CPU budget reached" in the log meant that F1 had used 12 core-hours. The process stopped and exited with status 3,
  and was not to be relaunched with a larger budget. F1 finished at 8.49 core-hours, so the guard never tripped.

### In-sample decodes under D̂ (PR2 §8.4, descriptive): finished at 09:59Z

Kept for the record. Running it again does no harm: every series logs "done already".

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
R=/home/claude/work/decoder/round2
mkdir -p $R/logs $R/numba_cache
cd $R
NUMBA_CACHE_DIR=$R/numba_cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup nice -n 10 python3 -I -B $X/code/decoder/round2/run_insample.py \
  --extract /home/claude/work/decoder/willow_run/extract --res $X/results/decoder/willow \
  --f1 $X/results/decoder/willow/round2/f1 --mech $X/results/decoder/willow/round2/mech \
  --out $X/results/decoder/willow/round2/insample --part 0/1 \
  >> $R/logs/run_insample_D.out 2>&1 < /dev/null &
python3 -c "import time; time.sleep(3)"
for P in $(ps -eo pid=,comm=,args= | awk '$2=="python3" && /run_insample\.py/ {print $1}'); do echo 10 > /proc/$P/autogroup; done
```

**How it resumes**
- It works through the 18 series in order. A finished series logs "done already".
- Each leaf keeps the decoders already stored in its `insample_data.npz` (RL and T̂ since 02:04Z) and adds only the
  missing one, D̂.
- A leaf's file is written to a temporary name and then renamed. A restart therefore loses at most the leaf in
  progress, which takes under a minute.

**Checking on it**
- **Progress:** `tail -n 2 $X/results/decoder/willow/round2/insample/run_insample_part0of1.log`
- **Finished:** the last log line ends in `d3_at_q8_9/Z: done`, or in `done already` after a repeat run.

### K2's extra rung on d5_at_q6_5/Z/r13 (round 1, `willow_prereg.md` §9 entry 10): finished at 10:13Z

Kept for the record. Running it again does no harm: the leaf has its `extra_rung.json` and is skipped.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
K=/home/claude/work/decoder/k2
mkdir -p $K/logs $K/numba_cache
cd $K
LEAF=google_105Q_surface_code_d3_d5_d7/d5_at_q6_5/Z/r13
NUMBA_CACHE_DIR=$K/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup nice -n 10 python3 -I -B $X/code/decoder/supplementary/k2_extra_rung.py \
  --root /home/claude/work/decoder/willow_run/extract --res $X/results/decoder/willow \
  --out $X/results/decoder/willow/supplementary/k2_extra_rung --leaf $LEAF \
  >> $K/logs/k2_d5_at_q6_5_Z_r13.out 2>&1 < /dev/null &
python3 -c "import time; time.sleep(3)"
for P in $(ps -eo pid=,comm=,args= | awk '$2=="python3" && /k2_extra_rung\.py/ {print $1}'); do echo 10 > /proc/$P/autogroup; done
```

**Always keep `--leaf`.** Without it, the script runs every pending leaf. That includes the r = 30 and 50 leaves, which
are waiting for your decision (about 13 core-hours, estimate).

**How it resumes**
- A leaf that already has an `extra_rung.json` for the current ladder file (same md5) is skipped.
- An interrupted leaf starts again from the beginning: its τ2 check, then τ3 on the band shots. At r = 13 that is about
  4 CPU minutes, or about 35 minutes of wall time at nice 10.

**Checking on it**
- **Progress:** `tail -n 3 /home/claude/work/decoder/k2/logs/k2_d5_at_q6_5_Z_r13.out`
- **Finished:** that log has a `tau3:` line, followed by the list of all ladder leaves with their verdicts.

### K2's extra rung on d5_at_q6_9/X/r10 (round 1, `willow_prereg.md` §9 entry 10): finished at 10:35Z

Kept for the record. Running it again does no harm: the leaf has its `extra_rung.json` and is skipped.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
K=/home/claude/work/decoder/k2
mkdir -p $K/logs $K/numba_cache
cd $K
LEAF=google_105Q_surface_code_d3_d5_d7/d5_at_q6_9/X/r10
NUMBA_CACHE_DIR=$K/numba_cache OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
setsid nohup python3 -I -B $X/code/decoder/supplementary/k2_extra_rung.py \
  --root /home/claude/work/decoder/willow_run/extract --res $X/results/decoder/willow \
  --out $X/results/decoder/willow/supplementary/k2_extra_rung --leaf $LEAF \
  >> $K/logs/k2_d5_at_q6_9_X_r10.out 2>&1 < /dev/null &
```

- **Always keep `--leaf`.** Without it, the script runs every pending leaf at once, in one process, in its own order.
- **How it resumes.** A leaf that already has an `extra_rung.json` for the current ladder file (same md5) is skipped. An
  interrupted leaf starts again from the beginning, which costs a few CPU minutes at r = 10.
- **Finished:** `/home/claude/work/decoder/k2/logs/k2_d5_at_q6_9_X_r10.out` has a `tau3:` line, followed by the list
  of all ladder leaves with their verdicts.

### Mech validation on simulated data, V0 and V2 (PR2 §8.2): finished at 14:08Z (V0 PASS, V2 PASS)

Kept for the record. Running it again does no harm: both scenarios have their json and are skipped.

```
X=/mnt/project-files/tensor-networks/deep-dives/experimental
R=/home/claude/work/decoder/round2
V=$X/results/decoder/willow/round2/validation
mkdir -p $R/logs $R/numba_cache
cd $R
for S in V0 V2; do
  [ -f $V/sim_mech_$S.json ] && continue
  NUMBA_CACHE_DIR=$R/numba_cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  setsid nohup python3 -I -B $X/code/decoder/round2/sim_validate_mech.py \
    --extract /home/claude/work/decoder/willow_run/extract --scenario $S --out $V \
    >> $R/logs/sim_mech_$S.out 2>&1 < /dev/null &
done
```

**How it resumes**
- The `[ -f ... ] && continue` line skips a scenario that has already finished.
- The simulated shots are generated again, identically, from fixed seeds.
- Each fit resumes from its checkpoint, `validation/ckpt_sim_mech_<V0|V2>_<T|F1|D>.ckpt.npz`. The final held-out
  decoding, a few minutes, starts again.

**Checking on it**
- **Progress:** `tail -n 2 $X/results/decoder/willow/round2/validation/sim_mech_V*.log`
- **Finished:** `validation/sim_mech_V0.json` and `sim_mech_V2.json` exist, and each log ends in PASS or FAIL.
