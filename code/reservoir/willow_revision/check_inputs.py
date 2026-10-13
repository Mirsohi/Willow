"""Integrity of the local inputs for the revision (R1-R5), after the container restart of 2026-10-13 ~15:07Z.

reservoir, 2026-10-13. Read-only; decodes nothing. Run with python3 -I (it reads downloaded data).

Checks:
  extract    decoder's local extract of the Willow archive against the zip it came from: for every zip member that
             has a file on disk, the size and the CRC-32 of the file must equal the member's (zipfile's own record).
             Only d = 3 and d = 5 leaves are read (the revision uses no d = 7 data).
  sim        reservoir's simulated archive (willow_sim) against the md5s recorded in samples_A.json / samples_B.json.
  sim_out    the simulation's local decode/ and mpe/ outputs against their stored copies in results (md5), where a
             stored copy exists.
  scratch    local scratch copies of earlier results (gapmech, final_k2, r2check, beta_real) against the stored
             files of the same name (md5); timing fields make some JSON differ, so a difference is reported, not fatal.
  loads      every local .npz and .json under the folders above opens and parses.

Usage: python3 -I check_inputs.py ZIP EXTRACT_ROOT RESERVOIR_WORK PROJECT_ROOT -> OUT_DIR/inputs_check.json
       (written through qrc.outpaths: QRC_OUT for scratch, QRC_OVERWRITE=1 for the store)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.outpaths import out_path  # noqa: E402


def md5(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def crc(p, bs=1 << 22):
    c = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def check_extract(zp, root):
    t = time.time()
    out = {"zip": zp, "zip_size": os.path.getsize(zp), "members": 0, "on_disk": 0, "checked": 0, "size_mismatch": [],
           "crc_mismatch": [], "skipped_d7": 0, "bytes_checked": 0}
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            out["members"] += 1
            p = os.path.join(root, info.filename)
            if not os.path.isfile(p):
                continue
            out["on_disk"] += 1
            if "/d7_at_" in "/" + info.filename:
                out["skipped_d7"] += 1
                continue
            if os.path.getsize(p) != info.file_size:
                out["size_mismatch"].append(info.filename)
                continue
            if crc(p) != info.CRC:
                out["crc_mismatch"].append(info.filename)
            out["checked"] += 1
            out["bytes_checked"] += info.file_size
    out["ok"] = not out["size_mismatch"] and not out["crc_mismatch"] and out["checked"] > 0
    out["seconds"] = round(time.time() - t, 1)
    return out


def check_sim(work, proj):
    res = {"records": 0, "files": 0, "mismatch": [], "missing": []}
    arch = os.path.join(work, "willow_sim", "archive", "google_105Q_surface_code_d3_d5_d7")
    for name in ("samples_A.json", "samples_B.json"):
        recs = json.load(open(os.path.join(proj, "results", "reservoir", "willow_sim", name)))
        for r in recs:
            res["records"] += 1
            d = os.path.join(arch, r["leaf"])
            for f, m in r["md5"].items():
                p = os.path.join(d, f)
                if not os.path.isfile(p):
                    res["missing"].append(os.path.join(r["leaf"], f))
                    continue
                res["files"] += 1
                if md5(p) != m:
                    res["mismatch"].append(os.path.join(r["leaf"], f))
    res["ok"] = res["files"] > 0 and not res["mismatch"] and not res["missing"]
    return res


def compare_tree(local, stored):
    res = {"local": local, "stored": stored, "files": 0, "equal": 0, "differ": [], "no_stored_copy": 0}
    for p in sorted(glob.glob(os.path.join(local, "**", "*"), recursive=True)):
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, local)
        q = os.path.join(stored, rel)
        res["files"] += 1
        if not os.path.isfile(q):
            res["no_stored_copy"] += 1
            continue
        if md5(p) == md5(q):
            res["equal"] += 1
        else:
            res["differ"].append(rel)
    return res


def check_loads(folders):
    import numpy as np
    res = {"npz": 0, "json": 0, "failed": []}
    for fd in folders:
        for p in sorted(glob.glob(os.path.join(fd, "**", "*"), recursive=True)):
            if p.endswith(".npz"):
                try:
                    with np.load(p, allow_pickle=False) as z:
                        for k in z.files:
                            z[k]
                    res["npz"] += 1
                except Exception as e:  # noqa: BLE001
                    res["failed"].append([p, repr(e)[:200]])
            elif p.endswith(".json"):
                try:
                    json.load(open(p))
                    res["json"] += 1
                except Exception as e:  # noqa: BLE001
                    res["failed"].append([p, repr(e)[:200]])
    res["ok"] = not res["failed"]
    return res


def main(zp, root, work, proj):
    t0 = time.time()
    rr = os.path.join(proj, "results", "reservoir")
    out = {"label": "reservoir, revision: integrity of local inputs after the 2026-10-13 restart; read-only",
           "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["sim"] = check_sim(work, proj)
    print("sim", out["sim"]["ok"], out["sim"]["files"], flush=True)
    out["sim_out"] = {
        "decode": compare_tree(os.path.join(work, "willow_sim", "decode"), os.path.join(rr, "willow_sim", "decode")),
        "mpe": compare_tree(os.path.join(work, "willow_sim", "mpe"), os.path.join(rr, "willow_sim", "mpe")),
    }
    out["scratch"] = {
        "gapmech": compare_tree(os.path.join(work, "gapmech", "scratch_out", "willow_gapmech"),
                                os.path.join(rr, "willow_gapmech")),
        "final_k2": compare_tree(os.path.join(work, "final_k2", "out", "willow_gapmech"),
                                 os.path.join(rr, "willow_gapmech", "final_k2")),
        "r2check": compare_tree(os.path.join(work, "r2check", "scratch_out", "willow_r2check"),
                                os.path.join(rr, "willow_r2check")),
        "beta_real": compare_tree(os.path.join(work, "beta_real", "scratch_out", "willow_sim"),
                                  os.path.join(rr, "willow_sim")),
    }
    print("compare done", flush=True)
    out["loads"] = check_loads([os.path.join(work, s) for s in ("gapmech", "final_k2", "r2check", "beta_real",
                                                                  "willow_sim")])
    print("loads", out["loads"]["ok"], out["loads"]["npz"], out["loads"]["json"], flush=True)
    out["extract"] = check_extract(zp, root)
    print("extract", out["extract"]["ok"], out["extract"]["checked"], flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    p = out_path(os.path.join(rr, "willow_revision", "inputs_check.json"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
