#!/usr/bin/env python3
"""PR3's post hoc r = 10 follow-up (notes/decoder/willow_prereg_r3.md section 10, logged before any r = 10 part
started): decodes the two r = 10 leaves of d7_at_q6_7 (X and Z), which never started before PR3's kill, with PR3's
design, decoders and rules. Kourosh chose this run after the kill (2026-10-13 00:58Z). The kill stands, H7 stays
"not evaluated", and everything from this run is post hoc and descriptive.

Why a new driver. PR3's runner (pr3_runner.py) starts no unit once d = 7 ML is killed, and it schedules all 6 leaves.
This driver imports it unchanged and runs its Runner on the r = 10 leaves only:
  - pr3_common.ROUNDS is set to (10,) in this process before the Runner is built, so its units, verdicts and projection
    cover X/r10 and Z/r10 only: 10 ladder parts, and 5 extra parts for a leaf that fails rule 1 or rule 2';
  - holds() holds no slot: every run it held slots for has ended. A pause in <state>/control.json still works;
  - killed() is always False: PR3's kill stands, and these two leaves are judged but kill nothing;
  - tally() writes <res>/r10_tally.json, marked post hoc, without the kill fields.
Everything else is the runner's own code: the unit commands (the frozen decode_willow.py for a ladder part,
pr3_extra_rung.py for an extra part), the shots (25,000 per leaf as 5 parts; the 1,000 probe shots of each leaf are
decoded but left out of every statistic), rule 1 and rule 2', the extra rung, the 4 GiB RSS cap, the 3 GiB memory
margin, at most 4 heavy processes, the CPU ledger, restarts, and the guard (--guard-core-hours 13, the cap of
Kourosh's choice): if the projection passes it, no new unit starts and the driver exits 6.
Exits as the runner's: 0 all done, 5 a unit failed, 6 the guard, 7 the memory cap.

  NUMBA_CACHE_DIR=<scratch> python3 -I -B pr3_r10_posthoc.py --root <extract> --res results/decoder/willow/pr3/r10_posthoc \
      --work <scratch>/work --state <scratch>/state --logs <scratch>/logs \
      --pilot results/decoder/willow/pr3/pilot/pilot.json --design 25k --guard-core-hours 13 [--dry-run]
"""
import argparse, json, os, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pr3_common as C

C.ROUNDS = (10,)                       # this process only: the Runner below sees the two r = 10 leaves
import pr3_runner as PRR               # unchanged; read-only

WHAT = ("post hoc r = 10 follow-up (willow_prereg_r3.md section 10): X/r10 and Z/r10 only; PR3's kill stands and "
        "H7 stays not evaluated; no kill rule here")


class R10(PRR.Runner):
    def holds(self, heavy):
        ctl = {}
        try:
            ctl = json.load(open(os.path.join(self.a.state, "control.json")))
        except Exception:
            pass
        return {}, bool(ctl.get("pause")) if isinstance(ctl, dict) else False

    def killed(self):
        return False

    def tally(self, status):
        fails = [lf for lf, v in self.verdicts.items() if v["verdict"].startswith("still fails")]
        t = {"made": PRR.now(), "what": WHAT, "design": self.a.design, "status": status, "verdicts": self.verdicts,
             "still_failing": fails,
             "complete": len(self.verdicts) == len(C.BASES) * len(C.ROUNDS)
                         and all(v["verdict"].startswith(("pass", "still fails")) for v in self.verdicts.values()),
             "projection": self.projection(), "guard_core_h": self.a.guard_core_hours,
             "code_md5": dict({f: PRR.md5(os.path.join(HERE, f))
                               for f in ("pr3_r10_posthoc.py", "pr3_runner.py", "pr3_common.py", "pr3_extra_rung.py")},
                              **{f: PRR.md5(os.path.join(PRR.CODE, f))
                                 for f in ("decode_willow.py", os.path.join("supplementary", "k2_extra_rung.py"))})}
        C.write_json(os.path.join(self.a.res, "r10_tally.json"), t)

    def run(self):
        self.log(f"pr3_r10_posthoc (md5 {PRR.md5(os.path.abspath(__file__))}): {WHAT}; leaves "
                 + ", ".join(sorted({u['leaf'].split('/', 2)[-1] for u in self.units.values()}))
                 + f"; {sum(1 for u in self.units.values() if u['kind'] == 'ladder')} ladder parts")
        return super().run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--project", default="/mnt/project-files/tensor-networks/deep-dives/experimental")
    ap.add_argument("--work", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--design", required=True, choices=sorted(C.DESIGNS))
    ap.add_argument("--guard-core-hours", type=float, required=True)
    ap.add_argument("--max-heavy", type=int, default=4)
    ap.add_argument("--min-free-gib", type=float, default=3.0)
    ap.add_argument("--cap-gib", type=float, default=4.0)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.normpath(a.res).endswith(os.path.join("pr3", "main")):
        sys.exit("--res must not be PR3's main results folder: this run writes its own (r10_posthoc)")
    sys.exit(R10(a).run())


if __name__ == "__main__":
    main()
