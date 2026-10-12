#!/usr/bin/env python3
"""Round 2, PR2 section 8.4: in-sample evaluation (descriptive only). Decodes each W5 leaf's 5,000 fit shots (round 1's
fit_idx, used by T-hat, D-hat, T-hat_mech and D-hat_mech alike) with exact ML under
  RL (the shipped DEM), T (T-hat: round-1 fit.json scale_hat), D (D-hat: F1 fit_series.json),
  Tm, Dm (T-hat_mech, D-hat_mech: mech_fit_T.json, mech_fit_D.json),
for the fits that exist. Reads only the fit halves and the fitted parameters. Leaves are loaded and checked as in
run_f1.py. Restartable: decoders already in a leaf's insample_data.npz are kept, missing ones are added.

Output: <out>/<leaf>/insample_data.npz with fit_idx, obs_fit and <tag>_pred, <tag>_llr (float32) per decoder.
Run: NUMBA_CACHE_DIR=<scratch> python3 -I -B run_insample.py --extract <root> --res results/decoder/willow \
         --f1 results/decoder/willow/round2/f1 --mech results/decoder/willow/round2/mech \
         --out results/decoder/willow/round2/insample --part 0/2
"""
import argparse, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cond_fit import W5, series_names
from run_f1 import load_leaf, all_series
from mech_fit import MechSeries
from ml_exact_fast import FastExactDecoder

TAGS = ("RL", "T", "D", "Tm", "Dm")


def decode(dm, order, det):
    post, _ = FastExactDecoder(dm, order=order).class_probs(det)
    pred = (np.nan_to_num(post[:, 1], nan=0.0) > 0.5).astype(np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        llr = (np.log(post[:, 1]) - np.log(post[:, 0])).astype(np.float32)
    return pred, llr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--f1", required=True)
    ap.add_argument("--mech", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--series", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    i, n = (int(x) for x in a.part.split("/"))
    ser = a.series.split(",") if a.series else all_series(a.res)[i::n]
    logf = open(os.path.join(a.out, f"run_insample_part{i}of{n}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    for series in ser:
        fits = {}
        p = os.path.join(a.f1, series, "fit_series.json")
        if os.path.exists(p):
            fits["D"] = json.load(open(p))
        for tag, fn in (("Tm", "mech_fit_T.json"), ("Dm", "mech_fit_D.json")):
            p = os.path.join(a.mech, series, fn)
            if os.path.exists(p):
                fits[tag] = json.load(open(p))
        want = ["RL", "T"] + [t for t in ("D", "Tm", "Dm") if t in fits]
        have = {}
        for r in W5:
            p = os.path.join(a.out, f"{series}/r{r}", "insample_data.npz")
            have[r] = set(t for t in TAGS if os.path.exists(p) and f"{t}_pred" in np.load(p).files)
        if all(set(want) <= have[r] for r in W5):
            log(f"{series}: done already ({', '.join(want)})"); continue
        log(f"{series}: decoding fit shots under {', '.join(want)}")
        leaves = [load_leaf(a, series, r, log) for r in W5]
        ms = MechSeries(leaves) if ("Tm" in fits or "Dm" in fits) else None
        if "D" in fits:
            names = series_names(leaves)
            assert names == fits["D"]["names"], f"{series}: F1 class names differ"
        for tag in ("Tm", "Dm"):
            if tag in fits:
                assert fits[tag]["names"] == ms.names, f"{series}: {tag} parameters differ"
        for l, lf in enumerate(leaves):
            r = lf.rounds
            d = os.path.join(a.out, lf.name)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "insample_data.npz")
            fi = np.load(os.path.join(a.res, "fit", lf.name, "rl_tsplit", "fit_data.npz"))["fit_idx"].astype(np.int32)
            arrs = dict(np.load(p)) if os.path.exists(p) else {}
            if arrs:
                assert np.array_equal(arrs["fit_idx"], fi), f"{lf.name}: stored fit_idx differs"
            arrs["fit_idx"] = fi
            arrs["obs_fit"] = lf.L_fit.astype(np.uint8)
            c0 = time.process_time()
            for tag in want:
                if f"{tag}_pred" in arrs:
                    continue
                if tag == "RL":
                    dm = lf.model.dm_at(np.ones(len(lf.names)))
                elif tag == "T":
                    fj = json.load(open(os.path.join(a.res, "fit", lf.name, "rl_tsplit", "fit.json")))
                    th = dict(zip(fj["classes"], fj["scale_hat"]))
                    dm = lf.model.dm_at(np.array([th[nm] for nm in lf.names]))
                elif tag == "D":
                    dm = lf.model.dm_at(lf.theta(np.array(fits["D"]["phi_hat"])))
                else:
                    dm = ms.dm(l, np.array(fits[tag]["delta_hat"]))
                arrs[f"{tag}_pred"], arrs[f"{tag}_llr"] = decode(dm, lf.model.order, lf.det_fit)
            np.savez_compressed(os.path.join(d, "insample_data.tmp.npz"), **arrs)
            os.replace(os.path.join(d, "insample_data.tmp.npz"), p)
            errs = {t: int(np.sum(arrs[f"{t}_pred"] != arrs["obs_fit"])) for t in TAGS if f"{t}_pred" in arrs}
            log(f"  {lf.name}: fit-shot errors {errs} ({time.process_time() - c0:.0f} CPU s)")
        log(f"{series}: done")


if __name__ == "__main__":
    main()
