#!/usr/bin/env python3
"""PR3's extra rung (notes/decoder/willow_prereg_r3.md section 3) on one d = 7 ladder output: one part of a leaf, or
the pilot. It calls k2_extra_rung.extra_rung, unchanged (code/decoder/supplementary/k2_extra_rung.py; section 9 entry
10 of willow_prereg.md), which
  - checks the rebuilt decoder: the first --check-shots band shots re-decoded at tau2 must give the stored llr bitwise;
  - decodes the band shots (the shots the ladder re-decoded at tau2) at tau3 = tau2 / 10 = 1e-8;
  - reports rule 1 and rule 2 between tau2 and tau3 on this output's shots.
k2_extra_rung.py's own main() lists d = 5 leaves only, so this script names the d = 7 output directly. It adds the
numbers of rule 2' (PR3): the band shots whose decision changes between tau2 and tau3. PR3's verdicts are made per
leaf over its parts, on the PR3 sample, by pr3_runner.py; this script's per-part rules are descriptive.

Output: <out>/<leaf>/<tag>/extra_rung.json and extra_rung.npz (as k2_extra_rung.py writes them).
  NUMBA_CACHE_DIR=<scratch> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python3 -I -B pr3_extra_rung.py --root <extract> \
      --res <ladder results root: <res>/decode/<leaf>/<tag>/> --out <out> --leaf <leaf> --tag <tag> [--check-shots 10]
"""
import argparse, hashlib, json, os, sys, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("PR3_CODE") or os.path.dirname(HERE)                  # code/decoder
sys.path.insert(0, os.path.join(CODE, "supplementary"))
sys.path.insert(0, CODE)
import numpy as np
import k2_extra_rung as K2X                                                  # unchanged; read-only

LADDER = "ml_tau2:1e-6/1e-7:0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--check-shots", type=int, default=10)
    a = ap.parse_args()

    def log(msg):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)
    fd = os.path.join(a.res, "decode", a.leaf, a.tag)
    summ = json.load(open(os.path.join(fd, "summary.json")))
    if summ.get("status") != "complete" or LADDER not in (summ.get("decoders") or {}):
        sys.exit(f"{fd}: the ladder is not complete")
    dn = LADDER.replace(":", "_").replace("/", "-")
    zmd5 = K2X.md5_file(os.path.join(fd, dn + ".npz"))
    od = os.path.join(a.out, a.leaf, a.tag)
    pj = os.path.join(od, "extra_rung.json")
    if os.path.exists(pj):
        old = json.load(open(pj))
        if old.get("ladder_npz_md5") == zmd5 and "tau3" in old:
            log(f"{a.leaf} [{a.tag}]: already done ({old['verdict']})"); return
    c0 = time.process_time()
    rec, arrays = K2X.extra_rung(a, a.leaf, a.tag, LADDER, summ, log)
    if arrays is not None:
        t = rec["tau3"]
        lim = max(3, 0.01 * t["errors_with_tau3"])
        rec["rule2prime_part"] = {"decisions_changed_tau2_tau3": t["decisions_changed_tau2_tau3"], "limit": lim,
                                  "pass": t["decisions_changed_tau2_tau3"] <= lim,
                                  "note": "this output's shots; PR3's verdict is per leaf on the PR3 sample"}
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(os.path.join(od, "extra_rung.tmp.npz"), **arrays)
        os.replace(os.path.join(od, "extra_rung.tmp.npz"), os.path.join(od, "extra_rung.npz"))
    rec.update(leaf=a.leaf, tag=a.tag, decoder=LADDER, ladder_npz_md5=zmd5, kmax=K2X.KMAX_LADDER,
               script_md5={"pr3_extra_rung.py": K2X.md5_file(os.path.abspath(__file__)),
                           "k2_extra_rung.py": K2X.md5_file(K2X.__file__)},
               cpu_seconds_process=time.process_time() - c0, made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    K2X.write_json(pj, rec)
    log(f"{a.leaf} [{a.tag}]: {rec['verdict']}")
    if arrays is None:
        sys.exit(4)


if __name__ == "__main__":
    main()
