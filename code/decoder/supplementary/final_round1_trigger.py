#!/usr/bin/env python3
"""Waits until K2's tally is complete, then runs round 1's final analysis once (notes/decoder/willow_prereg.md
section 9 entry 12; notes/decoder/willow_final_analysis_plan.md section 5: "It runs after k2_tally.json is complete").

Every --poll seconds it reads <k2 tally>; when its "complete" is true, it replaces itself (exec) with
  python3 -I -B final_round1.py <the arguments after "--">
If <out>/final_round1.json already exists from a run on a complete tally, it exits without running anything. It never
runs final_round1.py on an incomplete tally (K2 stopping early leaves the tally incomplete: the orchestrator decides).
  python3 -I -B final_round1_trigger.py --k2 <k2_tally.json> --out <final_round1 out> [--poll 120] -- <final_round1 args>
"""
import argparse, json, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("give final_round1.py's arguments after --")
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--k2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll", type=float, default=120.0)
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1:]
    done = os.path.join(a.out, "final_round1.json")
    try:
        if not json.load(open(done)).get("test_only_incomplete_k2", True):
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {done} exists (complete tally); nothing to do",
                  flush=True)
            return
    except (OSError, ValueError):
        pass
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} waiting for a complete {a.k2}", flush=True)
    while True:
        try:
            t = json.load(open(a.k2))
            if t.get("complete") is True:
                break
        except (OSError, ValueError):
            pass
        time.sleep(a.poll)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} K2's tally is complete (kill: {t.get('kill')}); "
          f"running final_round1.py", flush=True)
    script = os.path.join(HERE, "final_round1.py")
    os.execv(sys.executable, [sys.executable, "-I", "-B", script] + rest)


if __name__ == "__main__":
    main()
