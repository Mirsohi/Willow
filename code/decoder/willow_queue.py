"""Restartable multi-worker runner for the real-data job list made by `willow_stage.py plan`.

  python3 -I -B willow_queue.py run --plan plan.json --root <extracted archive> --out <results dir> --work <local dir>
         [--stages d3,fit,d5,d7] [--workers 4] [--cpus 0,1,2,3] [--nice 5] [--mem-gb 3.5] [--retries 1]
  python3 -I -B willow_queue.py status --plan plan.json --out <results dir>

Jobs run in plan order (see willow_stage.py plan); --stages only selects stages (run the queue once per stage to change
the order). At most --workers jobs run at a time, each as its own process group with one thread, `nice`, an address-space limit and (optionally) a CPU set; a
dry run on one core uses --cpus 0. A job is done when its result marker exists: decode jobs <out>/decode/<leaf>/<dem
tag>/summary.json (status complete, partial = time budget used up, or abort_* = a data problem, reported as such),
fit jobs <out>/fit/<leaf>/<dem tag>_tsplit/fit.json. Restarting skips done jobs; an interrupted decode job resumes
from its chunk checkpoints in <work>/chunks. Failed jobs (non-zero exit or no marker) are retried --retries times.
State: <work>/queue_state.json, copied to <out>/queue_state.json about once a minute; logs: <work>/logs/.
SIGTERM / SIGINT stop every running job (whole process groups) before the runner exits, so nothing is left running.
"""
import argparse, json, os, resource, signal, subprocess, sys, time

CODE = os.path.dirname(os.path.abspath(__file__))
STOP = False


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(*a):
    print(time.strftime("[%H:%M:%S]", time.gmtime()), *a, flush=True)


def marker(job, out):
    if job["kind"] == "fit":
        return os.path.join(out, "fit", job["leaf"], job["dem_tag"] + "_tsplit", "fit.json")
    return os.path.join(out, "decode", job["leaf"], job["dem_tag"], "summary.json")


def is_done(job, out):
    m = marker(job, out)
    if not os.path.exists(m):
        return False, None
    try:
        s = json.load(open(m))
    except Exception:
        return False, None
    if job["kind"] == "fit":     # a marker from another shot selection does not count
        return s.get("selection") == job["shots"], s.get("status", "complete")
    st = s.get("status")
    if (s.get("job") or {}).get("decoders") != job["decoders"]:
        return False, None
    if str(st).startswith("abort"):            # a data problem: reported, not retried (the selection may be unset)
        return True, st
    if s.get("selection") != job["shots"]:
        return False, None
    return st in ("complete", "partial"), st


def command(job, a):
    py = [sys.executable, "-I", "-B"]
    if job["kind"] == "fit":
        return py + [os.path.join(CODE, "fit_willow.py"), "--root", a.root, "--out", os.path.join(a.out, "fit"),
                     "--leaf", job["leaf"], "--dem-file", job["dem_file"], "--dem-tag", job["dem_tag"], "--time-split",
                     "--min-class-events", str(job.get("min_class_events", 30)), "--nfit", str(job["nfit"]), "--shots", job["shots"],
                     "--max-shots", "100000000"]
    c = py + [os.path.join(CODE, "decode_willow.py"), "--root", a.root, "--out", os.path.join(a.out, "decode"),
              "--work", os.path.join(a.work, "chunks"), "--leaf", job["leaf"], "--dem-file", job["dem_file"],
              "--dem-tag", job["dem_tag"], "--decoders", job["decoders"], "--shots", job["shots"],
              "--max-shots", "100000000", "--chunk", str(job["chunk"]), "--synd-ref", str(job["synd_ref"])]
    if job.get("budget_core_h"):
        c += ["--time-budget", str(3600.0 * job["budget_core_h"])]
    return c


def leaf_ready(job, root):
    f = os.path.join(root, job["leaf"])
    return os.path.isdir(f) and any(n.startswith("detection_events") for n in os.listdir(f))


def save_state(state, a, force=False):
    p = os.path.join(a.work, "queue_state.json")
    json.dump(state, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    if force or time.time() - state.get("_copied", 0) > 60:
        state["_copied"] = time.time()
        q = os.path.join(a.out, "queue_state.json")
        try:
            json.dump(state, open(q + ".tmp", "w"), indent=1)
            os.replace(q + ".tmp", q)
        except OSError as ex:
            log(f"could not copy the state to {q}: {ex}")


def run(a):
    global STOP
    plan = json.load(open(a.plan))
    stages = a.stages.split(",")
    jobs = sorted((j for j in plan["jobs"] if j["stage"] in stages), key=lambda j: j["order"])
    os.makedirs(os.path.join(a.work, "logs"), exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    lock = os.path.join(a.work, "queue.lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            log(f"another queue (pid {pid}) holds {lock}; exiting"); sys.exit(3)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    open(lock, "w").write(str(os.getpid()))
    sp = os.path.join(a.work, "queue_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    state.update({"_plan": os.path.abspath(a.plan), "_started": now(), "_workers": a.workers, "_stages": stages})
    state.setdefault("jobs", {})
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(a.work, "numba_cache"))
    cpus = [int(x) for x in a.cpus.split(",")] if a.cpus else None
    mem = int(a.mem_gb * 2 ** 30)

    def pre():
        os.nice(a.nice)
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        if cpus is not None:
            os.sched_setaffinity(0, cpus)

    def on_signal(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    pending = []
    for j in jobs:
        done, st = is_done(j, a.out)
        rec = state["jobs"].setdefault(j["id"], {"attempts": 0})
        if done:
            rec.update(status="done", result=st)
        else:
            rec["status"] = "pending"; pending.append(j)
    est_total = sum(j["est_core_h"] for j in pending)
    log(f"{len(jobs)} jobs in stages {stages}: {len(jobs) - len(pending)} done, {len(pending)} to run "
        f"(about {est_total:.2f} core-h, {est_total / max(a.workers, 1):.2f} h on {a.workers} workers)")
    running = {}
    t_start = time.time()
    try:
        while (pending or running) and not STOP:
            while pending and len(running) < a.workers and not STOP:
                j = pending.pop(0)
                rec = state["jobs"][j["id"]]
                if not leaf_ready(j, a.root):
                    rec.update(status="missing_data"); log(f"skip #{j['order']} {j['leaf']}: shot files not extracted"); continue
                lp = os.path.join(a.work, "logs", f"{j['order']:05d}_{j['stage']}.log")
                fh = open(lp, "a")
                fh.write(f"\n=== {now()} {' '.join(command(j, a))}\n"); fh.flush()
                p = subprocess.Popen(command(j, a), stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=a.work,
                                     preexec_fn=pre, start_new_session=True)
                rec.update(status="running", pid=p.pid, start=now(), log=lp, attempts=rec.get("attempts", 0) + 1)
                running[p.pid] = (p, j, fh, time.time())
                log(f"start #{j['order']} {j['stage']} {j['unit']} {j['leaf'].split('/', 1)[-1]} [{j['dem_tag']}] "
                    f"{j['decoders']} (est {j['est_core_h'] * 60:.1f} core-min)")
                save_state(state, a)
            time.sleep(a.poll)
            for pid in list(running):
                p, j, fh, t0 = running[pid]
                rc = p.poll()
                if rc is None:
                    continue
                fh.close(); del running[pid]
                rec = state["jobs"][j["id"]]
                done, st = is_done(j, a.out)
                rec.update(end=now(), rc=rc, seconds=round(time.time() - t0, 1), result=st)
                if rc == 0 and done:
                    rec["status"] = "done"
                    log(f"done  #{j['order']} {j['leaf'].split('/', 1)[-1]} [{j['dem_tag']}] {st} in {time.time() - t0:.0f} s")
                else:
                    rec["status"] = "failed"
                    log(f"FAIL  #{j['order']} {j['leaf'].split('/', 1)[-1]} [{j['dem_tag']}] rc={rc}; log {rec['log']}")
                    if rec["attempts"] <= a.retries:
                        pending.insert(0, j); rec["status"] = "pending"
                save_state(state, a)
    finally:
        for pid, (p, j, fh, t0) in running.items():
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 15
        for pid, (p, j, fh, t0) in running.items():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            fh.close()
            state["jobs"][j["id"]].update(status="interrupted", end=now())
        n = {}
        for rec in state["jobs"].values():
            n[rec["status"]] = n.get(rec["status"], 0) + 1
        state["_finished"] = now(); state["_counts"] = n
        state["_wall_h"] = round((time.time() - t_start) / 3600, 3)
        save_state(state, a, force=True)
        try:
            os.remove(lock)
        except OSError:
            pass
        log(f"queue stopped{' (signal)' if STOP else ''}: {n}")
    if STOP:
        sys.exit(130)
    sys.exit(0 if not any(r["status"] == "failed" for r in state["jobs"].values() if isinstance(r, dict)) else 1)


def status(a):
    plan = json.load(open(a.plan))
    by = {}
    for j in plan["jobs"]:
        done, st = is_done(j, a.out)
        s = by.setdefault(j["stage"], {"jobs": 0, "done": 0, "est_core_h_left": 0.0, "results": {}})
        s["jobs"] += 1
        if done:
            s["done"] += 1; s["results"][st] = s["results"].get(st, 0) + 1
        else:
            s["est_core_h_left"] += j["est_core_h"]
    for k, v in by.items():
        v["est_core_h_left"] = round(v["est_core_h_left"], 2)
        print(k, json.dumps(v))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True); r.add_argument("--root", required=True)
    r.add_argument("--out", required=True); r.add_argument("--work", required=True)
    r.add_argument("--stages", default="d3,fit,d5,d7"); r.add_argument("--workers", type=int, default=4)
    r.add_argument("--cpus", default=None); r.add_argument("--nice", type=int, default=5)
    r.add_argument("--mem-gb", type=float, default=3.5); r.add_argument("--retries", type=int, default=1)
    r.add_argument("--poll", type=float, default=1.0)
    s = sub.add_parser("status"); s.add_argument("--plan", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        a.root, a.out, a.work = (os.path.abspath(x) for x in (a.root, a.out, a.work))
        run(a)
    else:
        status(a)


if __name__ == "__main__":
    main()
