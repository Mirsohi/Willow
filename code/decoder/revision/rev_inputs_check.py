"""Integrity check of the d = 5 inputs of the revision analyses (D1-D3), read-only (decoder, 2026-10-13).
Run after a container restart, before anything is designed or decoded. Every check compares with a value recorded
earlier (an md5 or the zip's own CRC-32):
 1. the local archive zip against shared/data/willow/MANIFEST.json (md5 of the whole zip);
 2. every extracted file of the 24 W5 d = 5 leaves (4 patches, X and Z, r = 10, 30, 50) against the zip's central
    directory (size and CRC-32);
 3. per leaf: the RL DEM used for ML (the decode summary's "dem", md5 "dem_md5") in the extract; the ladder npz
    (per-shot ML(RL) llr, decisions, band flags, retained-state maxima) against K2's extra_rung.json
    "ladder_npz_md5"; K2's extra-rung outputs present;
 4. k2_tally.json against final_round1.json "inputs_md5";
 5. the non-band check: shot_lists.npz against its summary.json; summary.json and analysis/nonband_results.{md,json}
    against the md5s main posted (board/main.md, 2026-10-12).
The archive is untrusted data: run with python3 -I from a folder outside it, paths as arguments.
usage: python3 -I rev_inputs_check.py --zip Z --manifest M --extract E --res R --out OUT.json
"""
import argparse, hashlib, json, os, sys, time, zipfile, zlib

ARCH = "google_105Q_surface_code_d3_d5_d7"
PATCHES = ("d5_at_q4_7", "d5_at_q6_5", "d5_at_q6_9", "d5_at_q8_7")
W5 = (10, 30, 50)
LADDER = "ml_tau2_1e-6-1e-8_0.1.npz"
NONBAND_POSTED = {"summary.json": "b6315257b340ce1fe38a71d0536e5083",
                  "analysis/nonband_results.md": "9cac1e96fdfb0004e53c94f99c8182cf",
                  "analysis/nonband_results.json": "8ad9031918f4d80228c1b392ff8d9e18"}


def md5(p, blk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def crc(p, blk=1 << 22):
    c = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blk), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def leaves():
    return [f"{ARCH}/{p}/{b}/r{r}" for p in PATCHES for b in ("X", "Z") for r in W5]


def main():
    ap = argparse.ArgumentParser()
    for a in ("--zip", "--manifest", "--extract", "--res", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    t0 = time.time()
    checks = []

    def chk(what, ok, **kw):
        checks.append(dict(what=what, ok=bool(ok), **kw))
        if not ok:
            print("FAIL", what, kw, flush=True)

    # 1. zip md5
    man = json.load(open(a.manifest))
    want = None
    for k in ("zips", "files", "entries"):
        for e in (man.get(k) or []):
            if isinstance(e, dict) and str(e.get("name", e.get("key", ""))).endswith(".zip") and e.get("md5"):
                want = e["md5"]
    if want is None and isinstance(man.get("md5"), str):
        want = man["md5"]
    got = md5(a.zip)
    chk("archive zip md5 equals MANIFEST.json", want is not None and got == want, got=got, want=want)
    # 2. extracted W5 d = 5 files against the central directory
    z = zipfile.ZipFile(a.zip)
    info = {i.filename: i for i in z.infolist()}
    nfiles = nbytes = 0
    for lf in leaves():
        mem = [i for n, i in info.items() if n.startswith(lf + "/") and not n.endswith("/")]
        bad = []
        for i in mem:
            p = os.path.join(a.extract, i.filename)
            if not os.path.isfile(p) or os.path.getsize(p) != i.file_size or crc(p) != i.CRC:
                bad.append(i.filename[len(lf) + 1:])
            nfiles += 1; nbytes += i.file_size
        chk(f"{lf}: {len(mem)} extracted files equal the zip (size, CRC-32)", len(mem) > 0 and not bad, bad=bad)
    # 3. per leaf: DEM, ladder npz, K2 outputs
    dec = os.path.join(a.res, "decode")
    k2 = os.path.join(a.res, "supplementary", "k2_extra_rung")
    for lf in leaves():
        S = json.load(open(os.path.join(dec, lf, "rl_s5", "summary.json")))
        demp = os.path.join(a.extract, lf, S["dem"])
        chk(f"{lf}: RL DEM in the extract has the decode's dem_md5", os.path.isfile(demp) and md5(demp) == S["dem_md5"],
            dem=S["dem"], want=S["dem_md5"])
        chk(f"{lf}: decode summary complete, selection 0:50000:5", S.get("status") == "complete" and S.get("selection") == "0:50000:5")
        E = json.load(open(os.path.join(k2, lf, "rl_s5", "extra_rung.json")))
        lad = os.path.join(dec, lf, "rl_s5", LADDER)
        g = md5(lad)
        chk(f"{lf}: ladder npz md5 equals K2's ladder_npz_md5", g == E["ladder_npz_md5"], got=g, want=E["ladder_npz_md5"])
        for f in ("shots.npz",):
            chk(f"{lf}: {f} present", os.path.isfile(os.path.join(dec, lf, "rl_s5", f)))
        chk(f"{lf}: K2 extra_rung.npz present", os.path.isfile(os.path.join(k2, lf, "rl_s5", "extra_rung.npz")))
    # 4. k2 tally against the final analysis
    F = json.load(open(os.path.join(a.res, "final_round1", "final_round1.json")))
    g = md5(os.path.join(k2, "k2_tally.json"))
    chk("k2_tally.json md5 equals final_round1.json inputs_md5", g == F["inputs_md5"]["k2_tally.json"], got=g)
    # 5. non-band check
    nb = os.path.join(a.res, "supplementary", "nonband_check")
    NS = json.load(open(os.path.join(nb, "summary.json")))
    g = md5(os.path.join(nb, "shot_lists.npz"))
    chk("non-band shot_lists.npz md5 equals its summary.json", g == NS["shot_lists_npz_md5"], got=g)
    for rel, w in NONBAND_POSTED.items():
        g = md5(os.path.join(nb, rel))
        chk(f"non-band {rel} md5 equals main's post", g == w, got=g, want=w)
    out = dict(made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_md5=md5(os.path.abspath(__file__)),
               zip=a.zip, extract=a.extract, res=a.res, n_checks=len(checks), all_ok=all(c["ok"] for c in checks),
               extracted_files_checked=nfiles, extracted_bytes_checked=nbytes, seconds=time.time() - t0, checks=checks)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("made", "n_checks", "all_ok", "extracted_files_checked", "extracted_bytes_checked", "seconds")}))
    sys.exit(0 if out["all_ok"] else 1)


if __name__ == "__main__":
    main()
