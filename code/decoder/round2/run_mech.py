#!/usr/bin/env python3
"""Round 2, Amendment A1 (notes/decoder/willow_prereg_r2.md section 8.2) on real data: per series, fit T-hat_mech
(sum ln P(s)) and D-hat_mech (sum ln P(L|s) - (4/2) sum delta^2, started at the series' F1 D-hat) on the round-1 fit
shots of its 4 W5 leaves (pooled), then decode the held-out halves under each. The leaves are loaded and checked exactly
as in run_f1.py (DEM md5, classes, held-out outcomes). Needs the F1 output of the series (fit_series.json).

Outputs (restartable per series and per fit):
  <out>/<series>/mech_fit_T.json, mech_fit_D.json       fits (shifts, SEs, convergence, history, family, timings)
  <out>/<leaf>/mech_data.npz                             test_idx, obs_test, Tm_pred, Tm_llr, Dm_pred, Dm_llr (float32)
  <out>/<leaf>/eval_mech.json                            held-out counts and paired cells against ML(RL) and ML(T-hat)
Cap (--max-iter, default 25): both fits stop after at most this many steps. Section 8.2 pre-registered 15; the
orchestrator raised it to 25 on 2026-10-10 before any mech result existed (PR2 section 7, entry 15:14Z). A fit resumed
from its checkpoint continues under the cap it is given.
CPU budget (--cpu-budget-hours H; the orchestrator's 25 core-hours, 30 since 15:14Z, for the real fits and their
held-out decodes, 2026-10-10): the mech CPU so far is the sum, over every series under --out, of the fits'
cpu_seconds_fit (mech_fit_T.json, mech_fit_D.json) or, for a fit in progress, its checkpoint's CPU, plus the held-out
decode CPU in the leaves' eval_mech.json. It is checked before each series and before every new fit evaluation
(mech_fit's stop); when it reaches H the process logs "CPU budget reached", keeps its checkpoint and exits with status
3. A series whose fits are done is decoded without a check (about 3 CPU minutes). After each series the log gives the
CPU so far and a projection: the mean CPU of the finished series times the number of series. Run control only.
Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B run_mech.py --extract <root> --res results/decoder/willow \
         --f1 results/decoder/willow/round2/f1 --out results/decoder/willow/round2/mech --part 0/2 \
         --max-iter 25 --cpu-budget-hours 30   (see RELAUNCH.md)
"""
import argparse, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import write_json, remove_ckpt, read_ckpt_state, W5
from run_f1 import load_leaf, all_series
from mech_fit import MechSeries, fit_mech, LAMBDA_D
import analyze_willow as AW            # frozen: count2

MAX_ITER = 25    # both mech fits' cap: 15 in section 8.2, raised to 25 by the orchestrator (PR2 section 7, 15:14Z)


def mech_cpu_seconds(out, cache):
    """The mech CPU so far (see the module docstring); cache keeps each item's last readable value."""
    for dp, dn, fn in os.walk(out):
        for tag in ("T", "D"):
            if f"mech_fit_{tag}.json" in fn:
                try:
                    cache[(dp, tag)] = float(json.load(open(os.path.join(dp, f"mech_fit_{tag}.json")))["cpu_seconds_fit"])
                except Exception:
                    pass
            elif f"mech_ckpt_{tag}.ckpt.npz" in fn:
                st = read_ckpt_state(os.path.join(dp, f"mech_ckpt_{tag}"))
                if st is not None:
                    cache[(dp, tag)] = float(st["cpu_seconds"])
        if "eval_mech.json" in fn:
            try:
                cache[(dp, "decode")] = float(json.load(open(os.path.join(dp, "eval_mech.json")))["cpu_seconds_decode"])
            except Exception:
                pass
    return sum(cache.values())


def series_done(a, series):
    sd = os.path.join(a.out, series)
    return (os.path.exists(os.path.join(sd, "mech_fit_T.json")) and os.path.exists(os.path.join(sd, "mech_fit_D.json"))
            and all(os.path.exists(os.path.join(a.out, f"{series}/r{r}", "eval_mech.json")) for r in W5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--f1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--series", default=None)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER, help="cap of both mech fits (PR2 section 7, 15:14Z: 25)")
    ap.add_argument("--cpu-budget-hours", type=float, default=None,
                    help="stop when the mech CPU so far (fits and held-out decodes) reaches this")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    i, n = (int(x) for x in a.part.split("/"))
    ser = a.series.split(",") if a.series else all_series(a.res)[i::n]
    logf = open(os.path.join(a.out, f"run_mech_part{i}of{n}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    budget = None if a.cpu_budget_hours is None else 3600.0 * a.cpu_budget_hours
    cache = {}
    n_all = len(all_series(a.res))
    over = lambda: budget is not None and mech_cpu_seconds(a.out, cache) >= budget

    def stop_budget(where):
        log(f"CPU budget reached: mech has used {mech_cpu_seconds(a.out, cache) / 3600:.2f} of {a.cpu_budget_hours} "
            f"core-hours; stopping {where} (checkpoints kept). Tell decoder; do not raise the budget without @main.")
        sys.exit(3)
    log(f"part {a.part}: {len(ser)} series; cap {a.max_iter} steps per fit; CPU budget {a.cpu_budget_hours} core-hours")
    for series in ser:
        sd = os.path.join(a.out, series)
        fT_path, fD_path = os.path.join(sd, "mech_fit_T.json"), os.path.join(sd, "mech_fit_D.json")
        if series_done(a, series):
            log(f"{series}: done already"); continue
        if over():
            stop_budget(f"before {series}")
        f1 = json.load(open(os.path.join(a.f1, series, "fit_series.json")))
        log(f"{series}: loading (F1 D-hat converged={f1['converged']})")
        leaves = [load_leaf(a, series, r, log) for r in W5]
        ms = MechSeries(leaves)
        log(f"  family: {json.dumps(ms.info)}")
        fits = {}
        for tag, path, obj, lam in (("T", fT_path, "marg", 0.0), ("D", fD_path, "cond", LAMBDA_D)):
            if os.path.exists(path):
                fits[tag] = json.load(open(path))
                assert fits[tag]["names"] == ms.names, f"{series}: stored {tag} fit has other parameters"
                log(f"  {tag}_mech: read from {path}"); continue
            t0 = time.time()
            start = None if tag == "T" else ms.start_from_classes(f1["names"], np.array(f1["phi_hat"]))
            ck = os.path.join(sd, f"mech_ckpt_{tag}")
            fr = fit_mech(ms, obj, lam=lam, start=start, log=log, ckpt=ck, stop=over, max_iter=a.max_iter)
            if fr.get("stopped"):
                stop_budget(f"during {series} {tag}_mech")
            fr.update(series=series, leaves=[lf.name for lf in leaves], cpu_seconds_fit=fr["cpu_seconds_fit_total"],
                      wall_seconds_fit_this_process=time.time() - t0,
                      prereg="notes/decoder/willow_prereg_r2.md section 8.2", max_iter=a.max_iter,
                      start=("0" if start is None else "F1 D-hat mapped to templates"))
            write_json(path, fr)
            remove_ckpt(ck)
            fits[tag] = fr
            log(f"  {tag}_mech: converged={fr['converged']}, {fr['iterations']} steps, {fr['n_eval']} evaluations, "
                f"{fr['cpu_seconds_fit']:.0f} CPU s")
        dT, dD = np.array(fits["T"]["delta_hat"]), np.array(fits["D"]["delta_hat"])
        for l, lf in enumerate(leaves):
            c1 = time.process_time()
            pT, lT, _ = ms.decode_test(l, dT)
            pD, lD, _ = ms.decode_test(l, dD)
            o = lf.extra["obs_test"].astype(np.int8)
            S = lf.extra["shipped_pred"].astype(np.int8); T = lf.extra["fitted_pred"].astype(np.int8)
            ev = {"leaf": lf.name, "rounds": lf.rounds, "n_test": int(len(o)), "series": series,
                  "errors": {"Tm": int(np.sum(pT != o)), "Dm": int(np.sum(pD != o)), "RL": int(np.sum(S != o)),
                             "T": int(np.sum(T != o))},
                  "cells_RL_vs_Tm": AW.count2(S, pT.astype(np.int8), o), "cells_RL_vs_Dm": AW.count2(S, pD.astype(np.int8), o),
                  "cells_T_vs_Tm": AW.count2(T, pT.astype(np.int8), o), "cells_T_vs_Dm": AW.count2(T, pD.astype(np.int8), o),
                  "nan_posteriors": {"Tm": int(np.isnan(lT).sum()), "Dm": int(np.isnan(lD).sum())},
                  "converged": {"Tm": fits["T"]["converged"], "Dm": fits["D"]["converged"]},
                  "cpu_seconds_decode": time.process_time() - c1}
            d = os.path.join(a.out, lf.name)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "mech_data.tmp.npz"), test_idx=lf.extra["test_idx"].astype(np.int32),
                                obs_test=lf.extra["obs_test"], Tm_pred=pT, Tm_llr=lT, Dm_pred=pD, Dm_llr=lD)
            os.replace(os.path.join(d, "mech_data.tmp.npz"), os.path.join(d, "mech_data.npz"))
            write_json(os.path.join(d, "eval_mech.json"), ev)
            log(f"  {lf.name}: held-out errors Tm {ev['errors']['Tm']}, Dm {ev['errors']['Dm']}, RL {ev['errors']['RL']}, "
                f"T {ev['errors']['T']}")
        allser = all_series(a.res)
        fin = [s for s in allser if series_done(a, s)]
        used = mech_cpu_seconds(a.out, cache)
        per = [mech_cpu_seconds(os.path.join(a.out, s), {}) for s in fin]
        mean = sum(per) / len(per) if per else float("nan")
        log(f"{series}: done; mech CPU so far {used / 3600:.2f} core-hours; {len(fin)} of {n_all} series finished, "
            f"{mean / 3600:.2f} core-hours each on average; projected total {mean * n_all / 3600:.1f} core-hours "
            f"(budget {a.cpu_budget_hours})")


if __name__ == "__main__":
    main()
