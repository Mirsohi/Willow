#!/usr/bin/env python3
"""PR3: copy the 6 d = 7 leaves (d7_at_q6_7, X and Z, r = 10, 30, 50) from the archive into a new local extract, and
the pilot's leaf (X, r = 50) without any shipped predictions into a second one, so that the pilot cannot compare
with Libra. I/O only; the zip is opened read-only (run with python3 -I). Every member path is checked to stay under
the leaf's folder. Writes a manifest of md5s and checks the r = 10 files against round 1's extract.

  python3 -I -B pr3_extract.py --zip <archive.zip> --out <new extract> --pilot-out <new pilot extract> \
      --round1 <round-1 extract> --manifest <json>
"""
import argparse, hashlib, json, os, sys, zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr3_common import BASES, ROUNDS, leaf, write_json


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def safe_target(root, name):
    t = os.path.normpath(os.path.join(root, name))
    if os.path.isabs(name) or ".." in name.split("/") or not t.startswith(os.path.abspath(root) + os.sep):
        raise ValueError(f"unsafe member path {name!r}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot-out", required=True)
    ap.add_argument("--round1", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    for d in (a.out, a.pilot_out):
        if os.path.exists(d) and os.listdir(d):
            sys.exit(f"{d} exists and is not empty; use a new folder")
    Z = zipfile.ZipFile(a.zip)
    want = {leaf(b, r) for b in BASES for r in ROUNDS}
    pilot_leaf = leaf("X", 50)
    man = {"zip": a.zip, "zip_size": os.path.getsize(a.zip), "leaves": {}, "pilot": {}, "round1_check": {}}
    for info in Z.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        lf = next((w for w in want if name.startswith(w + "/")), None)
        if lf is None:
            continue
        data = Z.read(info)
        if len(data) != info.file_size:
            sys.exit(f"{name}: size {len(data)} != {info.file_size}")
        h = md5_bytes(data)
        t = safe_target(a.out, name)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        with open(t, "wb") as f:
            f.write(data)
        man["leaves"].setdefault(lf, {})[name[len(lf) + 1:]] = {"size": len(data), "md5": h}
        if lf == pilot_leaf and "predicted" not in os.path.basename(name):
            t2 = safe_target(a.pilot_out, name)
            os.makedirs(os.path.dirname(t2), exist_ok=True)
            with open(t2, "wb") as f:
                f.write(data)
            man["pilot"][name[len(lf) + 1:]] = {"size": len(data), "md5": h}
        r1 = os.path.join(a.round1, name)
        if os.path.isfile(r1):
            man["round1_check"][name] = md5_file(r1) == h
    missing = sorted(want - set(man["leaves"]))
    man["missing_leaves"] = missing
    man["round1_check_all_equal"] = all(man["round1_check"].values())
    man["files"] = sum(len(v) for v in man["leaves"].values())
    write_json(a.manifest, man)
    print(json.dumps({"files": man["files"], "missing": missing, "pilot_files": len(man["pilot"]),
                      "round1_files_compared": len(man["round1_check"]), "round1_equal": man["round1_check_all_equal"]}))
    if missing or not man["round1_check_all_equal"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
