"""D3 addendum: the merge census of rev_d3_facts.py on the d = 5 DEM files at r = 13 (decoder, 2026-10-13; post hoc
revision analysis, read-only, no decoding).
D3's census took d = 5 at r in {10, 30, 50} only (rev_d3_facts.py:104). The ML ladder (rl_s5), Tesseract (rl_s5t) and
the exact-MPE check also decode the d = 5 RL files at r = 13. This runs the same functions, with the same rule (a merge
needs the same detectors AND the same observables), on every d = 5 DEM file at r = 13: the 8 RL files, plus the 8
SI1000 files for completeness (D3 also counted the SI1000 files at d = 5).
Before the census, each file is checked against values recorded earlier:
  - size and CRC-32 against the archive zip's central directory;
  - for RL, md5 against "dem_md5" in the decode summaries that used it (rl_s5, and rl_s5t where present).
The archive is untrusted data: run with python3 -I from a folder outside it, paths as arguments.
usage: python3 -I rev_d3_r13.py --extract E --zip Z --res R --out OUT.json
"""
import argparse, glob, json, os, sys, time, zipfile, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_d3_facts import ARCH, census, md5  # noqa: E402

PRIORS = (("rl", "correlated_matching_decoder_with_rl_optimized_prior"),
          ("si1000", "correlated_matching_decoder_with_si1000_prior"))


def crc(p, blk=1 << 22):
    c = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    for x in ("--extract", "--zip", "--res", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    t0 = time.time()
    me = os.path.abspath(__file__)
    info = {i.filename: i for i in zipfile.ZipFile(a.zip).infolist()}
    dec = os.path.join(a.res, "decode")
    leaves = sorted(glob.glob(os.path.join(a.extract, ARCH, "d5_at_*", "[XZ]", "r13")))
    rows, checks = [], []
    for leaf in leaves:
        rel = os.path.relpath(leaf, a.extract)
        for prior, sub in PRIORS:
            p = os.path.join(leaf, "decoding_results", sub, "error_model.dem")
            if not os.path.isfile(p):
                continue
            mem = os.path.relpath(p, a.extract)
            zi = info.get(mem)
            ok_zip = zi is not None and os.path.getsize(p) == zi.file_size and crc(p) == zi.CRC
            checks.append(dict(what=f"{mem}: size and CRC-32 equal the zip's central directory", ok=bool(ok_zip)))
            h = md5(p)
            used_by = []
            if prior == "rl":
                for tag in ("rl_s5", "rl_s5t"):
                    sp = os.path.join(dec, rel, tag, "summary.json")
                    if os.path.isfile(sp):
                        S = json.load(open(sp))
                        ok = S.get("dem_md5") == h and S.get("dem") == os.path.join("decoding_results", sub, "error_model.dem")
                        checks.append(dict(what=f"{rel}/{tag}/summary.json: dem and dem_md5 equal this file", ok=bool(ok),
                                           want=S.get("dem_md5"), got=h))
                        used_by.append(tag)
            rows.append(dict(d=5, r=13, prior=prior, leaf=rel, md5=h, used_by_decodes=used_by, **census(p)))
    agg = {}
    for prior, _ in PRIORS:
        rr = [x for x in rows if x["prior"] == prior]
        if rr:
            agg[f"d5_r13_{prior}"] = dict(
                files=len(rr), distinct_files=len({x["md5"] for x in rr}),
                **{k: sum(x[k] for x in rr) for k in ("instructions", "with_separator", "empty_or_zero", "distinct_symptoms",
                                                       "merges", "detector_sets_with_several_L",
                                                       "tesseract_errors_after_merge")},
                max_mpe_log_weight_shift=max(x["max_mpe_log_weight_shift"] for x in rr))
    out = dict(made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_md5=md5(me),
               imports={"rev_d3_facts.py": md5(os.path.join(os.path.dirname(me), "rev_d3_facts.py"))},
               zip=dict(path=os.path.basename(a.zip), size=os.path.getsize(a.zip)),
               all_checks_ok=all(c["ok"] for c in checks), checks=checks, aggregate=agg, per_file=rows,
               seconds=round(time.time() - t0, 1))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(dict(all_checks_ok=out["all_checks_ok"], n_checks=len(checks), aggregate=agg,
                          seconds=out["seconds"]), indent=1))


if __name__ == "__main__":
    main()
