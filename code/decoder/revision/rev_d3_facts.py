"""D3 facts for the review's technical clarifications (1.7), read-only (decoder, 2026-10-13; post hoc revision).
(a, b) Census of fault-mechanism merging in the shipped DEMs: for every DEM file the analyses decode with, the number
    of error instructions, of instructions with decomposition marks ('^'), of empty symptoms after XOR, of distinct
    (detectors, observables) symptoms, of merges (instructions that join an earlier identical symptom), and of
    detector sets that carry more than one observable mask (mechanisms a detector-only merge would wrongly join).
    For any merged group, the shift in its max-product (MPE) weight is reported: merging replaces the group's
    best odd-parity configuration max_i p_i prod_{j!=i}(1-p_j) (two members) by the summed odd-parity weight.
    Plus Tesseract's merge rule on toy DEMs, through the installed package's own function, and the number of errors
    that function leaves on every shipped DEM (equal to the instruction count when nothing merges).
(c) Census of the sparse decoder's retained-state cap at d = 5: per-shot maxima of retained entries (maxk) at tau1,
    tau2 (the ladder), tau3 (K2's extra rung) and the non-band check's tau2/tau3, against kmax.
usage: python3 -I rev_d3_facts.py --extract E [--extract7 E7] --res R --out OUT.json
"""
import argparse, collections, glob, hashlib, json, math, os, sys, time
import numpy as np
import stim

ARCH = "google_105Q_surface_code_d3_d5_d7"
KMAX = 2_000_000          # decode_willow.KMAX_LADDER, recorded as "kmax" in every K2 extra_rung.json


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def census(path):
    dem = stim.DetectorErrorModel.from_file(path).flattened()
    n_ins = n_sep = n_empty = 0
    groups = collections.defaultdict(list)
    order = []
    for ins in dem:
        if ins.type != "error":
            continue
        n_ins += 1
        p = ins.args_copy()[0]
        ds, om, sep = set(), 0, False
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                ds ^= {t.val}
            elif t.is_logical_observable_id():
                om ^= 1 << t.val
            elif t.is_separator():
                sep = True
        n_sep += sep
        key = (tuple(sorted(ds)), om)
        if key == ((), 0) or p == 0:
            n_empty += 1
            continue
        if key not in groups:
            order.append(key)
        groups[key].append(p)
    merges = sum(len(v) - 1 for v in groups.values())
    by_d = collections.defaultdict(set)
    for (ds, om) in groups:
        by_d[ds].add(om)
    multi_l = sum(1 for v in by_d.values() if len(v) > 1)
    # MPE weight shift of merged groups (log of summed odd weight over best single odd configuration)
    shift = 0.0
    for v in groups.values():
        if len(v) > 1:
            q = 0.0
            for p in v:
                q = q + p - 2 * q * p
            best = max(p * np.prod([1 - x for j, x in enumerate(v) if j != i]) for i, p in enumerate(v))
            shift = max(shift, math.log(q / best))
    tes = tesseract_count(path)
    return dict(instructions=n_ins, with_separator=n_sep, empty_or_zero=n_empty, distinct_symptoms=len(groups),
                tesseract_errors_after_merge=tes,
                merges=merges, detector_sets_with_several_L=multi_l, max_mpe_log_weight_shift=shift)


def tesseract_count(path):
    """Error instructions left after Tesseract's own merge (the step its decoder runs by default) on a shipped DEM."""
    from tesseract_decoder import common
    out = common.merge_indistinguishable_errors(stim.DetectorErrorModel.from_file(path))
    return sum(1 for ins in out.flattened() if ins.type == "error")


def tesseract_rule():
    from tesseract_decoder import common
    txt = ("error(0.1) D0 D1 L0\nerror(0.2) D0 D1 L0\nerror(0.05) D0 D1\nerror(0.3) D2 ^ D2 D3\n"
           "error(0.01) D3\n")
    out = common.merge_indistinguishable_errors(stim.DetectorErrorModel(txt))
    return dict(input=txt.strip().split("\n"), merged=str(out).strip().split("\n"),
                expected_if_detectors_and_observables=["D0 D1 L0 p = 0.1*0.8 + 0.2*0.9 = 0.26", "D0 D1 kept apart (other L)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--extract7", default=None)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    root = os.path.join(a.extract, ARCH)
    dems = []
    for leaf in sorted(glob.glob(os.path.join(root, "d[35]_at_*", "[XZ]", "r*"))):
        d = int(os.path.basename(os.path.dirname(os.path.dirname(leaf)))[1])
        r = int(os.path.basename(leaf)[1:])
        for prior, sub in (("rl", "correlated_matching_decoder_with_rl_optimized_prior"),
                           ("si1000", "correlated_matching_decoder_with_si1000_prior")):
            p = os.path.join(leaf, "decoding_results", sub, "error_model.dem")
            if os.path.isfile(p) and (d == 3 or r in (10, 30, 50)):
                dems.append((d, r, prior, p))
    if a.extract7:
        for p in sorted(glob.glob(os.path.join(a.extract7, ARCH, "d7_at_*", "[XZ]", "r*", "decoding_results",
                                               "correlated_matching_decoder_with_rl_optimized_prior", "error_model.dem"))):
            dems.append((7, int(p.split("/r")[-1].split("/")[0]), "rl", p))
    rows = []
    seen = {}
    for d, r, prior, p in dems:
        h = md5(p)
        if h not in seen:
            seen[h] = census(p)
        rows.append(dict(d=d, r=r, prior=prior, leaf=os.path.relpath(os.path.dirname(os.path.dirname(os.path.dirname(p))), a.extract if d < 7 else a.extract7),
                         md5=h, **seen[h]))
    agg = {}
    for d in (3, 5, 7):
        for prior in ("rl", "si1000"):
            rr = [x for x in rows if x["d"] == d and x["prior"] == prior]
            if rr:
                agg[f"d{d}_{prior}"] = dict(files=len(rr), distinct_files=len({x["md5"] for x in rr}),
                                            instructions=sum(x["instructions"] for x in rr),
                                            with_separator=sum(x["with_separator"] for x in rr),
                                            empty_or_zero=sum(x["empty_or_zero"] for x in rr),
                                            merges=sum(x["merges"] for x in rr),
                                            detector_sets_with_several_L=sum(x["detector_sets_with_several_L"] for x in rr),
                                            max_mpe_log_weight_shift=max(x["max_mpe_log_weight_shift"] for x in rr),
                                            tesseract_errors_after_merge=sum(x["tesseract_errors_after_merge"] for x in rr))
    # (c) retained-state cap at d = 5
    dec = os.path.join(a.res, "decode")
    k2 = os.path.join(a.res, "supplementary", "k2_extra_rung")
    cap = []
    for lad in sorted(glob.glob(os.path.join(dec, ARCH, "d5_at_*", "[XZ]", "r*", "rl_s5", "ml_tau2_1e-6-1e-8_0.1.npz"))):
        leaf = os.path.relpath(os.path.dirname(os.path.dirname(lad)), dec)
        Z = np.load(lad)
        e = json.load(open(os.path.join(k2, leaf, "rl_s5", "extra_rung.json")))
        E = np.load(os.path.join(k2, leaf, "rl_s5", "extra_rung.npz"))
        red = Z["redecoded"]
        cap.append(dict(leaf=leaf, n=int(len(red)), band=int(red.sum()), kmax_ladder=KMAX, kmax_k2=int(e["kmax"]),
                        maxk_t1=int(Z["maxk_t1"].max()), maxk_t2=int(Z["maxk_t2"][red].max()) if red.any() else 0,
                        maxk_t3=int(E["maxk_t3"].max()) if len(E["maxk_t3"]) else 0,
                        at_cap_t1=int((Z["maxk_t1"] >= KMAX).sum()), at_cap_t2=int((Z["maxk_t2"] >= KMAX).sum()),
                        at_cap_t3=int((E["maxk_t3"] >= int(e["kmax"])).sum())))
    nb = []
    for f in sorted(glob.glob(os.path.join(a.res, "supplementary", "nonband_check", "leaves", "*.npz"))):
        Z = np.load(f)
        nb.append(dict(file=os.path.basename(f), n=int(len(Z["row"])), maxk_t2=int(Z["maxk_t2"].max()),
                       maxk_t3=int(Z["maxk_t3"].max()), at_cap=int(((Z["maxk_t2"] >= KMAX) | (Z["maxk_t3"] >= KMAX)).sum())))
    capsum = dict(leaves=len(cap), shots=sum(c["n"] for c in cap), band_shots=sum(c["band"] for c in cap),
                  max_maxk_t1=max(c["maxk_t1"] for c in cap), max_maxk_t2=max(c["maxk_t2"] for c in cap),
                  max_maxk_t3=max(c["maxk_t3"] for c in cap), shots_at_cap=sum(c["at_cap_t1"] + c["at_cap_t2"] + c["at_cap_t3"] for c in cap),
                  nonband_files=len(nb), nonband_shots=sum(x["n"] for x in nb),
                  nonband_max_maxk_t2=max(x["maxk_t2"] for x in nb) if nb else None,
                  nonband_max_maxk_t3=max(x["maxk_t3"] for x in nb) if nb else None,
                  nonband_at_cap=sum(x["at_cap"] for x in nb))
    out = dict(made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_md5=md5(os.path.abspath(__file__)),
               merging=dict(aggregate=agg, tesseract_rule=tesseract_rule(), per_file=rows),
               cap=dict(summary=capsum, per_leaf=cap, nonband=nb), seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(aggregate=agg, cap=capsum, tesseract=out["merging"]["tesseract_rule"]["merged"]), indent=1))


if __name__ == "__main__":
    main()
