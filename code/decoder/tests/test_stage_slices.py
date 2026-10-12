"""Test of `willow_stage.py stage` on a zip delivered in slices (fake archive; nothing here is Willow data).

usage: python3 -I -B tests/test_stage_slices.py <fake workdir from make_fake_willow3.py> <test dir> [slice bytes]

Splits <fake workdir>/shared/<zip> into slices <zip>.001, .002, ... and runs the stage step (fresh local folder each
time) on these shared folders:
  ok             slices + MANIFEST["slices"] (name, size, md5)       -> exit 0, zip and extraction identical to the whole-zip run
  nolist         slices, no MANIFEST["slices"] (sorted <zip>.NNN)    -> exit 0
  local_first    ok again with its local zip in place and the slices gone -> exit 0 ("local copy present")
  whole          the whole zip, no slices (the old path)             -> exit 0
  corrupt        one byte of slice 2 flipped, MANIFEST md5s           -> exit 3 at slice 2
  corrupt_nolist the same without MANIFEST["slices"]                 -> exit 3 on the whole-zip md5
  truncated      slice 2 short by 100 bytes, MANIFEST sizes          -> exit 3 before copying
  missing        slice 3 absent, MANIFEST lists it                   -> exit 2 before copying
  missing_nolist slice 3 absent, no list (gap in the numbering)      -> exit 2 before copying
  missing_last   the last slice absent, no list                      -> exit 3 (slices add up to less than the zip)
  bad_whole_md5  slices fine, MANIFEST's whole-zip md5 wrong         -> exit 3 after the rebuild
Every failing case must leave no zip, no .part file and nothing extracted. Prints one line per case and PASS / FAIL.
"""
import hashlib, json, os, shutil, subprocess, sys

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def tree(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = md5(p)
    return out


def main():
    fake, td = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000
    src = os.path.join(fake, "shared")
    man0 = json.load(open(os.path.join(src, "MANIFEST.json")))
    key = next(f["key"] for f in man0["files"] if f["key"].endswith(".zip"))
    shutil.rmtree(td, ignore_errors=True); os.makedirs(td)
    data = open(os.path.join(src, key), "rb").read()
    slices = [data[i:i + step] for i in range(0, len(data), step)]
    assert len(slices) >= 4, "use a smaller slice size"
    names = [f"{key}.{k + 1:03d}" for k in range(len(slices))]
    entries = [{"name": n, "size": len(b), "md5": hashlib.md5(b).hexdigest()} for n, b in zip(names, slices)]

    def make(case, with_list=True, drop=(), edit=None, whole=False, man_edit=None):
        d = os.path.join(td, case, "shared"); os.makedirs(d)
        for fn in os.listdir(src):
            if fn.endswith(".listing.tsv") or fn == "zenodo_record.json":
                shutil.copy(os.path.join(src, fn), d)
        man = json.loads(json.dumps(man0))
        if whole:
            shutil.copy(os.path.join(src, key), d)
        else:
            for k, (n, b) in enumerate(zip(names, slices)):
                if k in drop:
                    continue
                if edit and k == edit[0]:
                    b = edit[1](b)
                open(os.path.join(d, n), "wb").write(b)
            if with_list:
                man["slices"] = entries
        if man_edit:
            man_edit(man)
        json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"), indent=1)
        return d

    def flip(b):
        b = bytearray(b); b[len(b) // 2] ^= 0xFF; return bytes(b)

    def wrong_md5(man):
        for f in man["files"]:
            if f["key"] == key:
                f["md5"] = "0" * 32

    last = len(slices) - 1
    cases = [("ok", dict(), 0), ("nolist", dict(with_list=False), 0), ("whole", dict(whole=True), 0),
             ("corrupt", dict(edit=(1, flip)), 3), ("corrupt_nolist", dict(with_list=False, edit=(1, flip)), 3),
             ("truncated", dict(edit=(1, lambda b: b[:-100])), 3), ("missing", dict(drop=(2,)), 2),
             ("missing_nolist", dict(with_list=False, drop=(2,)), 2), ("missing_last", dict(with_list=False, drop=(last,)), 3),
             ("bad_whole_md5", dict(man_edit=wrong_md5), 3)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    whole_tree, ok_all = None, True

    def run(case, shared, local):
        r = subprocess.run([sys.executable, "-I", "-B", os.path.join(C, "willow_stage.py"), "stage", "--shared", shared,
                            "--local", local, "--dist", "3", "--meta-all"], capture_output=True, text=True, env=env)
        open(os.path.join(td, case, "stage.log"), "w").write(r.stdout + r.stderr)
        return r.returncode, (r.stdout + r.stderr).strip().splitlines()

    print(f"{key}: {len(data)} bytes in {len(slices)} slices of at most {step} bytes; md5 {md5(os.path.join(src, key))}")
    results = {}
    for case, kw, want in cases:
        shared = make(case, **kw)
        local = os.path.join(td, case, "local")
        rc, lines = run(case, shared, local)
        zp = os.path.join(local, "zip", key)
        left = sorted(os.listdir(os.path.join(local, "zip"))) if os.path.isdir(os.path.join(local, "zip")) else []
        ext = tree(os.path.join(local, "extract")) if os.path.isdir(os.path.join(local, "extract")) else {}
        good = rc == want
        note = ""
        if want == 0:
            good &= os.path.exists(zp) and md5(zp) == md5(os.path.join(src, key)) and not any(x.endswith(".part") for x in left)
            results[case] = ext
            if case == "whole":
                whole_tree = ext
        else:
            good &= not left and not ext        # no zip, no .part, nothing extracted
            note = next((l for l in lines if "FAILURE" in l or "MISSING" in l or "missing" in l), lines[-1] if lines else "")
        ok_all &= good
        print(f"{'PASS' if good else 'FAIL'} {case:15s} exit {rc} (want {want}) {note[:150]}")
    # the extraction from slices equals the extraction from the whole zip
    same = results.get("ok") == whole_tree == results.get("nolist") and bool(whole_tree)
    ok_all &= same
    print(f"{'PASS' if same else 'FAIL'} extraction from slices identical to the whole-zip extraction ({len(whole_tree or {})} files)")
    # local copy first: the slices may be gone once the zip is on local disk
    sh = os.path.join(td, "ok", "shared")
    for n in names:
        os.remove(os.path.join(sh, n))
    os.makedirs(os.path.join(td, "local_first"))
    rc, lines = run("local_first", sh, os.path.join(td, "ok", "local"))
    good = rc == 0 and any("local copy present" in l for l in lines)
    ok_all &= good
    print(f"{'PASS' if good else 'FAIL'} local_first     exit {rc} (want 0) {next((l for l in lines if 'local copy' in l), '')[:150]}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
