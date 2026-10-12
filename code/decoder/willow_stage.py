"""Stage Google's Willow archive (Zenodo 13273331) for the real-data queue, then build the job list.
The archive is untrusted data: run this with `python3 -I -B`, keep it outside the data directory, never execute
anything from the archive. One core is enough for every step here.

  stage     --shared <shared/data/willow> --local <dir> [--dist 3] [--meta-all] [--plan plan.json --stages d5,d7]
            1. copy each zip listed in MANIFEST.json to <dir>/zip/ (skipped if already there with the right md5), checking
               size and md5 against MANIFEST.json and zenodo_record.json while copying. If the shared folder holds the
               zip as slices <key>.001, .002, ... instead (the shared store takes at most 5 GiB per file), rebuild it:
               the slices come from MANIFEST["slices"] (name, size, md5), else from the files <key>.NNN in number
               order; every slice's size is checked first and its md5 while it is copied, then the whole zip's size and
               md5 against MANIFEST.json and the record. A missing slice exits 2 before anything is copied; any size or
               md5 mismatch exits 3 and leaves no partial zip;
            2. check every member before extracting anything; refuse the archive (exit 4) on an absolute path, a '..'
               component, a backslash or drive letter, control characters, a symlink, a duplicate name, a compression
               ratio above 1000, or when the selection would not fit in 80% of the free disk;
            3. extract into <dir>/extract/: every member of the leaves of distance --dist (default 3) and the top-level
               files; with --meta-all also the small files (circuits, DEMs, metadata; no shot data) of every other leaf,
               so that the inventory covers all experiments; with --plan, the leaves the given stages need.
            A leaf = a folder that holds a .stim circuit. Writes <dir>/stage_report.json. Re-running skips finished work.
  inventory --local <dir> --listing <zip.listing.tsv> [--shared-inventory results/decoder/willow_inventory.json] --out <json>
            uses the shared inventory when it covers every leaf of the listing; otherwise (missing, or made on a subset)
            runs the loader inventory (`willow_data.inventory`) on <dir>/extract. Adds to every row its path relative to
            the archive and the shot counts implied by the listed file sizes.
  plan      --inventory <json> --local <dir> --out plan.json [--profile real|dry]
            builds the job list from the inventory (not from guessed paths): stages d3 (all leaves, both shipped DEMs,
            exact ML + baselines), d5 (W5 = r in {10, 13, 30, 50}, every 5th shot, tau ladder), d7 (cost-capped probe),
            fit (d=3 class-scale fits at W5). Order: d=3 RL DEM, fit, d=3 SI1000 DEM, d=5, d=7 (every input of the
            kill criterion K1 comes first), each block cheapest unit first (unit = distance, patch, basis, DEM).
            Shot subsets are strided; the stride is the smallest one >= the nominal step whose selection holds every
            initialization bitstring (sweep-bit pattern) of a d=3 leaf in its full-leaf proportion within 10%
            (shots may be grouped or interleaved by bitstring; nothing is known about the order before the data land).
  estimate  [--profile real] [--rounds 1,10,13,30,50,70,...,250] [--shots 50000] [--out plan_estimate.json]
            the same job list for the expected layout (9 d=3, 4 d=5 and 1 d=7 patches, X and Z, 15 round counts, all
            assumed), to get the core-hour estimate per stage before the data land.
Exit codes: 2 missing input (also a missing slice), 3 size or md5 failure, 4 unsafe archive or not enough local disk,
5 inventory failure.
"""
import argparse, collections, hashlib, json, math, os, re, shutil, stat, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHOT_STEMS = ("measurements", "detection_events", "obs_flips_actual", "sweep_bits", "sweep", "obs_flips_predicted")
SHOT_EXT = (".b8", ".01", ".r8", ".ptb64", ".hits", ".dets")
W5 = (10, 13, 30, 50)


def log(*a):
    print(time.strftime("[%H:%M:%S]", time.gmtime()), *a, flush=True)


def md5_file(path, bufsize=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(bufsize), b""):
            h.update(blk)
    return h.hexdigest()


def shipped_pathways(row):
    """Pathways (decoding_results/<pathway>/) with a shipped prediction file in this leaf, from the loader inventory,
    the member listing, or an explicit 'shipped' list."""
    keys = set((row.get("predictions") or {})) | set((row.get("shot_counts_from_listing") or {}))
    pws = set(row.get("shipped") or [])
    for k in keys:
        parts = k.replace("\\", "/").split("/")
        if len(parts) >= 2 and "predicted" in parts[-1]:
            pws.add(parts[-2])
    return sorted(pws)


def dem_content_equal(paths, rtol=1e-9):
    """True if the DEM files define the same fault distribution for ML decoding: the same merged exact symptoms
    (decomposition marks, instruction order and repeat blocks do not matter; dem_utils.parse_dem) with probabilities
    equal to rtol. None if a file cannot be read."""
    try:
        import stim
        from dem_utils import parse_dem
        ref = None
        for p in paths:
            dm = parse_dem(stim.DetectorErrorModel.from_file(p))
            c = {(tuple(ds), int(o)): float(q) for ds, o, q in zip(dm.dets, dm.obs, dm.p)}
            c.update({((), int(o)): float(q) for o, q in dm.undetectable})
            c = ((dm.num_detectors, dm.num_observables), c)
            if ref is None:
                ref = c
                continue
            if c[0] != ref[0] or c[1].keys() != ref[1].keys():
                return False
            if any(abs(v - ref[1][k]) > rtol * max(abs(v), abs(ref[1][k])) for k, v in c[1].items()):
                return False
        return True
    except Exception:
        return None


def is_shot_file(name):
    b = os.path.basename(name)
    stem, ext = os.path.splitext(b)
    return ext in SHOT_EXT and any(stem.startswith(s) for s in SHOT_STEMS)


def read_listing(path):
    rows = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        name, size = line.rsplit("\t", 1)
        rows.append((name, int(size)))
    return rows


def leaves_of(names):
    """Leaf folders (relative to the archive root) = folders that directly contain a .stim file."""
    return sorted({os.path.dirname(n) for n in names if n.endswith(".stim") and not n.endswith("/")})


def leaf_of(name, leafset):
    """The leaf a member belongs to (longest leaf prefix), or None for top-level files."""
    d = os.path.dirname(name)
    while d:
        if d in leafset:
            return d
        d = os.path.dirname(d)
    return None


def leaf_meta(rel):
    from willow_data import parse_name
    return parse_name("/" + rel)


# ---------------------------------------------------------------- stage
def unsafe_reason(info, names_seen):
    n = info.filename
    if not n or "\x00" in n or any(ord(c) < 32 for c in n):
        return "control character"
    if n.startswith("/") or n.startswith("\\") or re.match(r"^[A-Za-z]:", n):
        return "absolute path"
    if "\\" in n:
        return "backslash"
    parts = n.rstrip("/").split("/")
    if any(p in ("..",) for p in parts):
        return "'..' component"
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK:
        return "symlink"
    if n in names_seen:
        return "duplicate name"
    if info.compress_size > 0 and info.file_size / info.compress_size > 1000:
        return f"compression ratio {info.file_size / info.compress_size:.0f}"
    if info.compress_size == 0 and info.file_size > 0:
        return "zero compressed size"
    return None


def copy_checked(src, dst, size, md5s):
    """Copy src -> dst (via .part), computing the md5 while copying; True if size and md5 match."""
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        got = md5_file(dst)
        if all(got == m for m in md5s if m):
            log(f"local copy present, md5 OK ({got})")
            return True
        log("local copy has the wrong md5; copying again"); os.remove(dst)
    h = hashlib.md5()
    t0 = time.time()
    with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        for blk in iter(lambda: fi.read(1 << 22), b""):
            h.update(blk); fo.write(blk)
    got = h.hexdigest()
    have = os.path.getsize(dst + ".part")
    log(f"copied {have / 1e9:.2f} GB in {time.time() - t0:.0f} s, md5 {got}")
    if have != size or not all(got == m for m in md5s if m):
        log(f"SIZE/MD5 FAILURE: size {have} vs {size}, md5 {got} vs {md5s}")
        os.remove(dst + ".part")
        return False
    os.replace(dst + ".part", dst)
    return True


def strip_md5(v):
    v = str(v or "").strip().lower()
    return v.split(":", 1)[1] if v.startswith("md5:") else v


def local_copy_ok(dst, size, md5s):
    """True if dst already holds the verified zip (then nothing in the shared folder is needed); a wrong copy is removed."""
    if not (os.path.exists(dst) and os.path.getsize(dst) == size):
        return False
    got = md5_file(dst)
    if all(got == m for m in md5s):
        log(f"local copy present, md5 OK ({got})")
        return True
    log(f"local copy has the wrong md5 ({got}); making it again")
    os.remove(dst)
    return False


def check_disk(dst, size):
    free = shutil.disk_usage(os.path.dirname(dst)).free
    if size > free - 2e9:
        log(f"not enough local disk for the zip: {size / 1e9:.2f} GB, {free / 1e9:.1f} GB free"); sys.exit(4)


def find_slices(shared, key, man):
    """The slices of <key> in their order, as [(name, size or None, md5 or None)], with how they were found and a
    problem message when one is missing. The slices are MANIFEST["slices"] entries (name, size, md5) whose name is
    <key>.<number>; if MANIFEST lists none, the files <key>.NNN in the shared folder. Numbered slices must run
    consecutively from 1 (or 0)."""
    listed = []
    for s in man.get("slices") or []:
        if not isinstance(s, dict):
            continue
        name = os.path.basename(str(s.get("name") or s.get("key") or s.get("file") or s.get("path") or ""))
        if name.startswith(key + "."):
            sz = s.get("size", s.get("bytes"))
            listed.append((name, int(sz) if sz is not None else None, strip_md5(s.get("md5") or s.get("checksum")) or None))
    how = "MANIFEST.json slices"
    if not listed:
        how = f"files {key}.NNN"
        listed = [(n, None, None) for n in os.listdir(shared) if re.fullmatch(re.escape(key) + r"\.\d{3,}", n)]
    if not listed:
        return [], how, None
    nums = [re.search(r"\.(\d+)$", n) for n, _, _ in listed]
    if all(nums):
        nums = [int(m.group(1)) for m in nums]
        listed = [x for _, x in sorted(zip(nums, listed), key=lambda t: t[0])]
        nums = sorted(nums)
        if nums[0] not in (0, 1) or nums != list(range(nums[0], nums[0] + len(nums))):
            return listed, how, f"slice numbers {nums} do not run consecutively from 1"
    missing = [n for n, _, _ in listed if not os.path.isfile(os.path.join(shared, n))]
    if missing:
        return listed, how, f"{len(missing)} of {len(listed)} slices are not in {shared}: {missing}"
    return listed, how, None


def build_from_slices(shared, slices, dst, size, md5s):
    """Concatenate the slices into dst (via .part). Before copying: every slice's size against MANIFEST and the sum
    against the zip's size. While copying: each slice's md5 against MANIFEST (when given). After: the whole file's size
    and md5 against md5s (MANIFEST.json and the Zenodo record). True if everything matches; nothing is left on failure."""
    tot = 0
    for name, ssize, _ in slices:
        have = os.path.getsize(os.path.join(shared, name))
        if ssize is not None and have != ssize:
            log(f"SLICE SIZE FAILURE: {name} has {have} bytes, MANIFEST says {ssize}"); return False
        tot += have
    if tot != size:
        log(f"SIZE FAILURE: the {len(slices)} slices hold {tot} bytes, the zip has {size} (a slice is missing or incomplete)")
        return False
    part, h, t0, ok = dst + ".part", hashlib.md5(), time.time(), True
    with open(part, "wb") as fo:
        for k, (name, ssize, smd5) in enumerate(slices):
            hs, n = hashlib.md5(), 0
            with open(os.path.join(shared, name), "rb") as fi:
                for blk in iter(lambda: fi.read(1 << 22), b""):
                    hs.update(blk); h.update(blk); fo.write(blk); n += len(blk)
            got = hs.hexdigest()
            if (ssize is not None and n != ssize) or (smd5 and got != smd5):
                log(f"SLICE FAILURE: {name}: read {n} bytes with md5 {got}; MANIFEST says {ssize} bytes, md5 {smd5}")
                ok = False
                break
            log(f"   slice {k + 1}/{len(slices)} {name}: {n} bytes, md5 {got}"
                + (" = MANIFEST" if smd5 else " (MANIFEST gives no md5 for it)"))
    have = os.path.getsize(part)
    got = h.hexdigest()
    if ok:
        log(f"rebuilt {have / 1e9:.2f} GB from {len(slices)} slices in {time.time() - t0:.0f} s, md5 {got}")
        if have != size or not all(got == m for m in md5s):
            log(f"SIZE/MD5 FAILURE of the rebuilt zip: size {have} vs {size}, md5 {got} vs {md5s}")
            ok = False
    if not ok:
        os.remove(part)
        return False
    os.replace(part, dst)
    return True


def inside(path, root):
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    return rp == rr or rp.startswith(rr + os.sep)


def extract_member(z, info, root):
    target = os.path.join(root, *info.filename.rstrip("/").split("/"))
    if not inside(target, root):
        raise RuntimeError(f"member escapes the extraction root: {info.filename}")
    if info.is_dir():
        os.makedirs(target, exist_ok=True); return 0
    if os.path.exists(target) and os.path.getsize(target) == info.file_size:
        return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not inside(os.path.dirname(target), root) or os.path.islink(target):
        raise RuntimeError(f"member escapes the extraction root: {info.filename}")
    with z.open(info) as fi, open(target + ".part", "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    os.chmod(target + ".part", 0o644)
    os.replace(target + ".part", target)
    return info.file_size


def cmd_stage(a):
    man_p = os.path.join(a.shared, "MANIFEST.json")
    if not os.path.exists(man_p):
        log(f"no {man_p}: the data have not landed"); sys.exit(2)
    man = json.load(open(man_p))
    rec_md5 = {}
    rp = os.path.join(a.shared, "zenodo_record.json")
    if os.path.exists(rp):
        rec = json.load(open(rp))
        fs = rec.get("files") or []
        if isinstance(fs, dict):
            e = fs.get("entries") or []
            fs = list(e.values()) if isinstance(e, dict) else e
        for f in fs:
            ck = str(f.get("checksum") or "")
            rec_md5[f.get("key") or f.get("filename")] = ck.split(":", 1)[1] if ck.startswith("md5:") else ""
    os.makedirs(os.path.join(a.local, "zip"), exist_ok=True)
    root = os.path.join(a.local, "extract")
    os.makedirs(root, exist_ok=True)
    want_leaves = set()
    if a.plan:
        plan = json.load(open(a.plan))
        stages = set(a.stages.split(",")) if a.stages else None
        want_leaves = {j["leaf"] for j in plan["jobs"] if (stages is None or j["stage"] in stages) and j.get("leaf")}
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zips": []}
    for f in man["files"]:
        key = f["key"]
        if not key.endswith(".zip"):
            continue
        src = os.path.join(a.shared, key); dst = os.path.join(a.local, "zip", key)
        size = int(f["size"])
        want = [m for m in (strip_md5(f.get("md5")), strip_md5(rec_md5.get(key))) if m]
        if not want:
            log(f"no md5 for {key} in MANIFEST.json or zenodo_record.json"); sys.exit(3)
        if local_copy_ok(dst, size, want):
            source = "local copy"
        elif os.path.exists(src):
            # the whole zip in the shared folder
            if os.path.getsize(src) != size:
                log(f"shared copy of {key} has size {os.path.getsize(src)}, MANIFEST says {size}"); sys.exit(3)
            check_disk(dst, size)
            log(f"1. copy {key} ({size / 1e9:.2f} GB) to local disk and check md5")
            if not copy_checked(src, dst, size, want):
                sys.exit(3)
            source = "whole zip"
        else:
            # the zip in slices <key>.001, .002, ... (the shared store takes at most 5 GiB per file)
            slices, how, problem = find_slices(a.shared, key, man)
            if not slices:
                log(f"missing {src}, and no slices of it in {a.shared}"); sys.exit(2)
            if problem:
                log(f"MISSING SLICE: {problem}; nothing copied"); sys.exit(2)
            check_disk(dst, size)
            log(f"1. rebuild {key} ({size / 1e9:.2f} GB) on local disk from {len(slices)} slices ({how}), "
                f"checking each slice, then the whole md5")
            if not build_from_slices(a.shared, slices, dst, size, want):
                sys.exit(3)
            source = f"{len(slices)} slices ({how})"
        report["zip_source"] = source
        z = zipfile.ZipFile(dst)
        infos = z.infolist()
        log(f"2. member check: {len(infos)} members")
        seen, bad = set(), []
        for i in infos:
            r = unsafe_reason(i, seen)
            seen.add(i.filename)
            if r:
                bad.append((i.filename, r))
        if bad:
            log(f"UNSAFE ARCHIVE, nothing extracted: {len(bad)} members, e.g. {bad[:5]}")
            report["zips"].append({"key": key, "unsafe": bad[:50]})
            json.dump(report, open(os.path.join(a.local, "stage_report.json"), "w"), indent=1)
            sys.exit(4)
        names = [i.filename for i in infos if not i.is_dir()]
        leafset = set(leaves_of(names))
        sel = []
        for i in infos:
            if i.is_dir():
                continue
            lf = leaf_of(i.filename, leafset)
            if lf is None:
                sel.append(i); continue                       # top-level files (README etc.)
            if a.plan:
                if lf in want_leaves:
                    sel.append(i)
                continue
            d = leaf_meta(lf).get("distance")
            if d == a.dist or (a.meta_all and not is_shot_file(i.filename)):
                sel.append(i)
        need = sum(i.file_size for i in sel if not (os.path.exists(os.path.join(root, i.filename))
                                                      and os.path.getsize(os.path.join(root, i.filename)) == i.file_size))
        free = shutil.disk_usage(root).free
        log(f"3. extract {len(sel)} of {len(names)} files, {need / 1e9:.2f} GB still to write, {free / 1e9:.1f} GB free")
        if need > 0.8 * free:
            log("not enough local disk for this selection"); sys.exit(4)
        t0 = time.time(); wrote = 0
        for k, i in enumerate(sel):
            wrote += extract_member(z, i, root)
            if k % 2000 == 1999:
                log(f"   {k + 1} files, {wrote / 1e9:.2f} GB written")
        log(f"   done: {wrote / 1e9:.2f} GB written in {time.time() - t0:.0f} s")
        by = collections.Counter()
        for lf in leafset:
            m = leaf_meta(lf)
            by[(m.get("distance"), all(os.path.exists(os.path.join(root, n)) for n in names
                                         if n.startswith(lf + "/") and is_shot_file(n)))] += 1
        report["zips"].append({"key": key, "md5_ok": True, "members": len(infos), "leaves": len(leafset),
                               "extracted_files": len(sel), "bytes_written": wrote,
                               "leaves_by_distance_and_shots_extracted": {f"d{k[0]}_{'full' if k[1] else 'meta'}": v
                                                                          for k, v in sorted(by.items(), key=str)}})
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["extract_root"] = os.path.abspath(root)
    json.dump(report, open(os.path.join(a.local, "stage_report.json"), "w"), indent=1)
    log("stage report:", json.dumps(report["zips"][-1].get("leaves_by_distance_and_shots_extracted", {})))


# ---------------------------------------------------------------- inventory
def rel_of(folder, leafset):
    f = folder.replace("\\", "/").rstrip("/")
    for lf in leafset:
        if f == lf or f.endswith("/" + lf):
            return lf
    return None


def cmd_inventory(a):
    listing = read_listing(a.listing)
    names = [n for n, _ in listing]
    sizes = dict(listing)
    leafset = set(leaves_of(names))
    rows, source = None, None
    if a.shared_inventory and os.path.exists(a.shared_inventory):
        try:
            rows0 = json.load(open(a.shared_inventory))
            covered = {rel_of(r.get("folder", ""), leafset) for r in rows0} - {None}
            errs = [r for r in rows0 if r.get("error")]
            if covered == leafset and not errs:
                rows, source = rows0, f"shared inventory {a.shared_inventory} (covers all {len(leafset)} leaves)"
            else:
                log(f"shared inventory covers {len(covered)} of {len(leafset)} leaves ({len(errs)} errors): made on a subset, "
                    f"running our own")
        except Exception as ex:
            log(f"shared inventory unreadable ({ex}); running our own")
    if rows is None:
        from willow_data import inventory
        root = os.path.join(a.local, "extract")
        t0 = time.time()
        rows = inventory(root, None)
        source = f"loader inventory of {root} ({time.time() - t0:.0f} s)"
        covered = {rel_of(r.get("folder", ""), leafset) for r in rows} - {None}
        if covered != leafset:
            log(f"our inventory covers {len(covered)} of {len(leafset)} leaves: extract with --meta-all first"); sys.exit(5)
    out = []
    for r in rows:
        rel = rel_of(r.get("folder", ""), leafset)
        r = dict(r); r["rel"] = rel
        if rel is not None and "error" not in r:
            nd, no = r["num_detectors"], r["num_observables"]
            sh = {}
            for n in names:
                if n.startswith(rel + "/") and is_shot_file(n):
                    sub = n[len(rel) + 1:]
                    stem = os.path.basename(n).split(".")[0]
                    width = {"detection_events": nd, "obs_flips_actual": no, "obs_flips_predicted": no,
                             "measurements": r["num_measurements"], "sweep_bits": r["num_sweep_bits"], "sweep": r["num_sweep_bits"]}.get(stem)
                    sh[sub] = sizes[n] / ((width + 7) // 8) if (n.endswith(".b8") and width) else None
            r["shot_counts_from_listing"] = sh
            # the shot count comes from the two files every analysis needs; other files only raise warnings
            key = [sh.get(k) for k in sh if os.path.basename(k).split(".")[0] in ("detection_events", "obs_flips_actual")
                   and "/" not in k]
            ok = len(key) == 2 and key[0] == key[1] and key[0] is not None and float(key[0]).is_integer()
            r["n_shots"] = int(key[0]) if ok else None
            if not ok:
                r["shot_count_problem"] = {k: v for k, v in sh.items() if "/" not in k}
            others = sorted({k for k, v in sh.items() if ok and v is not None and v != key[0]})
            if others:
                r["shot_count_warning"] = others
        out.append(r)
    meta = {"source": source, "leaves_in_listing": len(leafset), "rows": len(out),
            "errors": [r["folder"] for r in out if r.get("error")][:20],
            "inconsistent_shot_counts": [r["rel"] for r in out if r.get("shot_count_problem")][:20],
            "shot_count_warnings": [(r["rel"], r["shot_count_warning"]) for r in out if r.get("shot_count_warning")][:20],
            "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"meta": meta, "rows": out}, open(a.out, "w"), indent=1, default=str)
    by = collections.Counter((r.get("meta") or {}).get("distance") for r in out)
    log(f"inventory: {len(out)} leaves {dict(by)} from {source}; errors {len(meta['errors'])}, "
        f"inconsistent shot counts {len(meta['inconsistent_shot_counts'])}")
    if meta["errors"] or meta["inconsistent_shot_counts"]:
        sys.exit(5)


# ---------------------------------------------------------------- plan
# CPU ms per round per shot on one core (code/decoder/README.md, "Stages and cost"). ml_exact: Round-2 timing,
# 0.023-0.037 at r = 10-250 (a 50,000-shot d=3 r=50 job took 76% of this estimate on 2026-10-08). Tesseract and
# BeliefMatching at d = 3, Tesseract at d = 5: medians of the 2026-10-08 dry run on the fake archive (r <= 50;
# r = 50 confirmed on 10,000 shots); the d = 3 points at r >= 130 are the Round-2 ones scaled by 1.25 (assumed).
def cost_ms_round_shot(dec, d, r):
    if dec in ("corr_mwpm", "mwpm"):
        return 0.002 * d * d / 9
    if dec == "ml_exact":
        return 0.035
    if dec == "tesseract":
        pts = {3: [(1, 0.02), (10, 0.27), (13, 0.36), (30, 0.58), (50, 0.76), (130, 1.2), (250, 1.9)]}.get(
            d, [(10, 5.2), (30, 8.6)])
        return float(interp(r, pts))
    if dec == "beliefmatch":
        return float(interp(r, [(1, 0.02), (10, 0.15), (13, 0.21), (30, 0.32), (50, 0.40), (130, 0.50),
                                (250, 0.53)]) * (d * d / 9))
    if dec.startswith("ml_tau2"):
        return {5: 8.5, 7: 103.0}.get(d, 0.5)
    raise ValueError(dec)


def interp(x, pts):
    xs, ys = zip(*pts)
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def prior_of(pathway):
    s = pathway.lower()
    if "si1000" in s:
        return "si1000"
    if "rl_optimized" in s or "_rl_" in s or s.endswith("_rl"):
        return "rl"
    return "other"


PROFILES = {
    "real": dict(d3_shots=None, d3_synd=10000, d3_chunk=5000, sub_step=5, sub_rmax=50, d5_step=5, d5_synd=2000,
                 d5_chunk=500, d7_step=25, d7_synd=0, d7_chunk=250, d7_budget_core_h=6.0, fit_nfit=5000, fit_max=None,
                 fit_rounds=W5, fit_min_class_events=30),
    "dry": dict(d3_shots=400, d3_synd=200, d3_chunk=200, sub_step=5, sub_rmax=50, d5_step=25, d5_synd=20, d5_chunk=20,
                d7_step=50, d7_synd=0, d7_chunk=5, d7_budget_core_h=0.05, fit_nfit=200, fit_max=400, fit_rounds=W5,
                fit_min_class_events=30)}


def pattern_ids(sweep):
    """Initialization bitstring (sweep-bit row) of every shot -> small integer id."""
    import numpy as np
    _, inv = np.unique(np.asarray(sweep, dtype=np.uint8), axis=0, return_inverse=True)
    return np.asarray(inv).reshape(-1)


def choose_step(pid, base, tol=0.1, span=60):
    """Smallest stride s >= base whose selection 0::s holds every pattern within (1 +- tol) of its full-leaf share."""
    import numpy as np
    n = len(pid)
    full = np.bincount(pid) / n
    for s in range(base, base + span):
        sel = pid[::s]
        if len(sel) == 0:
            break
        share = np.bincount(sel, minlength=len(full)) / len(sel)
        if np.all(share > 0) and np.all(np.abs(share / full - 1) <= tol):
            return s, {"step": s, "nominal": base, "patterns": int(len(full)), "selected": int(len(sel)),
                       "max_share_deviation": float(np.max(np.abs(share / full - 1)))}
    return base, {"step": base, "nominal": base, "patterns": int(len(full)), "unbalanced": True}


def balance_steps(rows, root, P, notes):
    """Read the sweep bits of the first d=3 leaf with shot data on disk and pick balanced strides."""
    from willow_data import Experiment
    steps = {k: P[k] for k in ("sub_step", "d5_step", "d7_step")}
    info = {"leaf": None, "steps": {}}
    for r in sorted(rows, key=lambda r: (r["meta"].get("distance") != 3, r["rel"])):
        if r["meta"].get("distance") != 3:
            continue
        f = os.path.join(root, r["rel"])
        if not os.path.isdir(f) or not any(n.startswith("sweep_bits") or n.startswith("sweep.") for n in os.listdir(f)):
            continue
        try:
            sw = Experiment(f).sweep_bits()
        except Exception as ex:
            notes.append(f"{r['rel']}: sweep bits unreadable ({ex!r})"); continue
        if sw is None:
            continue
        pid = pattern_ids(sw)
        info["leaf"] = r["rel"]
        info["order"] = {"patterns": int(pid.max() + 1), "runs": int(1 + (pid[1:] != pid[:-1]).sum()),
                         "first_20": pid[:20].tolist()}
        for k in steps:
            steps[k], info["steps"][k] = choose_step(pid, P[k])
            if info["steps"][k].get("unbalanced"):
                notes.append(f"no balanced stride near {P[k]} for {k}; kept {P[k]}")
        break
    if info["leaf"] is None:
        notes.append("no d=3 sweep bits on disk: subset strides not checked for bitstring balance")
    return steps, info


def build_plan(rows, P, root, hash_dems=True, steps=None):
    steps = steps or {k: P[k] for k in ("sub_step", "d5_step", "d7_step")}
    jobs, units, notes = [], collections.defaultdict(list), []
    libra_differs, shipped_count, leaf_shipped = [], collections.Counter(), []
    d7_rounds = sorted({x["meta"].get("rounds") for x in rows if x["meta"].get("distance") == 7} - {None})
    probe_r = [x for x in d7_rounds if x >= 10][:2]
    for r in rows:
        m = r["meta"]; rel = r["rel"]
        d, basis, rounds, patch = m.get("distance"), m.get("basis"), m.get("rounds"), m.get("patch")
        props = (m.get("props") or {})
        for k, v in (("distance", d), ("basis", basis), ("rounds", rounds)):
            if k in props and str(props[k]) != str(v):
                notes.append(f"{rel}: folder name says {k}={v}, metadata says {props[k]}")
        n = r.get("n_shots")
        # shipped DEMs: group by prior; identical files collapse (md5 of the local copy); files that differ only in
        # decomposition marks or formatting are the same model for ML (compared by content)
        dems = collections.defaultdict(list)
        for dp in r.get("dems", []):
            pw = dp.replace("\\", "/").split("/")[-2] if "/" in dp.replace("\\", "/") else dp
            full = os.path.join(root, rel, dp)
            h = md5_file(full) if (hash_dems and os.path.exists(full)) else None
            dems[prior_of(pw)].append((dp, pw, h))
        dem_choice = {}
        for pr, lst in dems.items():
            hs = {h for _, _, h in lst}
            canon = sorted(lst, key=lambda t: (0 if t[1].startswith("correlated_matching") else 1, t[0]))[0]
            dem_choice[pr] = {"file": canon[0], "md5": canon[2], "pathways": [t[1] for t in lst], "distinct_files": len(hs)}
            if len(hs) > 1 and None not in hs:
                first = {}
                for dp, pw, h in lst:
                    first.setdefault(h, os.path.join(root, rel, dp))
                same = dem_content_equal(list(first.values()))
                dem_choice[pr]["same_content"] = same
                notes.append(f"{rel}: prior {pr} has {len(hs)} distinct DEM files, "
                             f"{ {True: 'same content for ML', False: 'DIFFERENT content'}.get(same, 'content not comparable')}: "
                             f"{[(t[1], t[2]) for t in lst]}")
            if None in hs and hash_dems:
                notes.append(f"{rel}: DEM files of prior {pr} not extracted locally (md5 unknown)")
            # pre-registration section 2: P1 compares Libra with ML on Libra's own prior file
            lib = [t for t in lst if t[1].startswith("libra")]
            if pr == "rl" and lib and lib[0][2] != canon[2]:
                same = dem_content_equal([os.path.join(root, rel, canon[0]), os.path.join(root, rel, lib[0][0])])
                dem_choice[pr]["libra_same_content"] = same
                if same is not True:
                    libra_differs.append(rel)
        shipped = shipped_pathways(r)
        for pw in shipped:
            shipped_count[pw] += 1
        base = dict(leaf=rel, d=d, basis=basis, rounds=rounds, patch=patch, n_shots=n, dems=dem_choice, shipped=shipped)
        leaf_shipped.append((rel, rounds, set(shipped)))
        def add(stage, dtag, decoders, shots, chunk, synd, unit_extra="", budget=None, kind="decode", extra=None):
            if dtag.split("_")[0] not in dem_choice:
                notes.append(f"{rel}: no shipped DEM for prior {dtag}; job skipped"); return
            sel = shots if shots else f"0:{n}:1"
            st, sp, stp = (int(x) for x in sel.split(":"))
            nsel = len(range(st, min(sp, n or 0), stp))
            rr = max(rounds, 1)
            if kind == "fit":    # ~5 forward-backward passes on the fit sample + 4 exact decodes of the test half
                cost = (5 * 0.3 * min(nsel // 2, P["fit_nfit"]) + 4 * 0.035 * nsel / 2) * rr / 3.6e6
            else:
                cost = sum(cost_ms_round_shot(x.split(":")[0], d, rr) for x in decoders.split(",")) * rr * nsel / 3.6e6
                ml = [x.split(":")[0] for x in decoders.split(",") if x.startswith("ml_")]
                cost += sum(cost_ms_round_shot(x, d, rr) for x in ml) * rr * synd / 3.6e6
            if budget:
                cost = min(cost, budget)
            j = dict(base, stage=stage, kind=kind, dem_tag=dtag, dem_file=dem_choice[dtag.split("_")[0]]["file"],
                     decoders=decoders, shots=sel, n_selected=nsel, chunk=chunk, synd_ref=synd, budget_core_h=budget,
                     est_core_h=round(cost, 4), unit=f"{stage}|d{d}|{patch}|{basis}|{dtag.split('_')[0]}{unit_extra}")
            if extra:
                j.update(extra)
            j["id"] = f"{stage}|{rel}|{dtag}|{decoders}|{sel}"
            jobs.append(j); units[j["unit"]].append(j)
        if n is None:
            notes.append(f"{rel}: shot count unknown; leaf skipped"); continue
        # dry runs thin the d=3 leaves by a stride, never by a prefix (shots may be grouped by bitstring)
        t3 = max(1, n // P["d3_shots"]) if P["d3_shots"] else 1
        tf = max(1, n // P["fit_max"]) if P["fit_max"] else 1
        if d == 3:
            for pr in ("si1000", "rl"):
                add("d3", pr, "ml_exact,corr_mwpm,mwpm", f"0:{n}:{t3}", P["d3_chunk"], P["d3_synd"])
                if rounds <= P["sub_rmax"]:
                    add("d3", pr + f"_s{steps['sub_step']}", "tesseract,beliefmatch", f"0:{n}:{t3 * steps['sub_step']}",
                        P["d3_chunk"], 0)
            if rounds in P["fit_rounds"]:
                add("fit", "rl", "fit_willow", f"0:{n}:{tf}", 0, 0, kind="fit",
                    extra={"nfit": P["fit_nfit"], "time_split": True, "min_class_events": P["fit_min_class_events"]})
        elif d == 5 and rounds in W5:
            s5 = steps["d5_step"]
            add("d5", f"rl_s{s5}", "corr_mwpm,mwpm,ml_tau2:1e-6/1e-8:0.1", f"0:{n}:{s5}", P["d5_chunk"], P["d5_synd"])
            if rounds <= 30:
                add("d5", f"rl_s{s5}t", "tesseract", f"0:{n}:{s5}", P["d5_chunk"], 0)
        elif d == 7 and rounds in probe_r:
            s7 = steps["d7_step"]
            add("d7", f"rl_s{s7}", "corr_mwpm,mwpm,ml_tau2:1e-6/1e-7:0.1", f"0:{n}:{s7}",
                P["d7_chunk"], P["d7_synd"], budget=P["d7_budget_core_h"] / 4)
    # shipped predictions some leaves lack (the real archive has no Libra predictions at r = 1 and 13, as reported by
    # the fetch session); the analysis takes each shipped series' rounds from the jobs' "shipped" lists
    for pw in sorted(shipped_count):
        miss = [rr for rel, rr, s in leaf_shipped if pw not in s]
        if miss:
            notes.append(f"shipped predictions of {pw} absent at {len(miss)} of {len(leaf_shipped)} leaves "
                         f"(r = {sorted(set(miss))})")
    if libra_differs:
        notes.insert(0, f"LIBRA DEM DIFFERS: at {len(libra_differs)} leaves Libra's RL DEM file has other content than the "
                        f"file ML uses (e.g. {libra_differs[:2]}); pre-registration section 2: P1 needs ML with Libra's file")
    # order: d=3 with the RL DEM (P1, P3, G1), the fits (P4), d=3 with the SI1000 DEM (P2), d=5, d=7, so that every
    # input of the kill criterion K1 is in after about a third of the run; inside a block the cheapest unit first,
    # inside a unit the cheapest job first
    def block(j):
        return {"d3": 0 if j["dem_tag"].split("_")[0] == "rl" else 2, "fit": 1, "d5": 3, "d7": 4}[j["stage"]]
    ucost = {u: sum(j["est_core_h"] for j in js) for u, js in units.items()}
    jobs.sort(key=lambda j: (block(j), ucost[j["unit"]], j["unit"], j["est_core_h"], j["rounds"], j["id"]))
    for k, j in enumerate(jobs):
        j["order"] = k
    stages = collections.OrderedDict()
    for j in jobs:
        s = stages.setdefault(j["stage"], {"jobs": 0, "units": set(), "est_core_h": 0.0, "first_unit_core_h": None})
        s["jobs"] += 1; s["units"].add(j["unit"]); s["est_core_h"] += j["est_core_h"]
    for k, v in stages.items():
        first = next(j["unit"] for j in jobs if j["stage"] == k)
        v["first_unit_core_h"] = round(ucost[first], 3)
    summ = {k: {"jobs": v["jobs"], "units": len(v["units"]), "est_core_h": round(v["est_core_h"], 2),
                "est_wall_h_4_workers": round(v["est_core_h"] / 4, 2), "first_unit_core_h": v["first_unit_core_h"]}
            for k, v in stages.items()}
    return jobs, summ, notes


def cmd_plan(a):
    inv = json.load(open(a.inventory))
    rows = [r for r in inv["rows"] if r.get("rel") and not r.get("error")]
    root = os.path.join(a.local, "extract")
    P = PROFILES[a.profile]
    notes0 = []
    steps, balance = balance_steps(rows, root, P, notes0)
    jobs, summ, notes = build_plan(rows, P, root, steps=steps)
    notes = notes0 + notes
    plan = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "profile": a.profile, "params": P,
            "subset_steps": steps, "bitstring_balance": balance,
            "inventory": os.path.abspath(a.inventory), "inventory_source": inv["meta"]["source"],
            "stages": summ, "notes": notes, "jobs": jobs}
    json.dump(plan, open(a.out, "w"), indent=1, default=list)
    log(f"subset strides {steps} (bitstring balance checked on {balance.get('leaf')})")
    for k, v in summ.items():
        log(f"stage {k}: {v['jobs']} jobs in {v['units']} units, about {v['est_core_h']} core-h ({v['est_wall_h_4_workers']} h on 4 workers)")
    agg = [nt for nt in notes if nt.startswith(("LIBRA", "shipped predictions"))]
    rest = [nt for nt in notes if nt not in agg]
    for nt in agg + rest[:15]:
        log("note:", nt)
    if len(rest) > 15:
        log(f"({len(rest) - 15} more notes in {a.out})")


PATHWAYS = ["correlated_matching_decoder_with_si1000_prior", "correlated_matching_decoder_with_rl_optimized_prior",
            "harmony_decoder_with_si1000_prior", "harmony_decoder_with_rl_optimized_prior",
            "libra_decoder_with_rl_optimized_prior"]


# the layout the fetch session read from the zip's central directory on 2026-10-08 (reported to us; not yet seen here):
# 14 patches, X and Z, 15 round counts, 50,000 shots per folder; Libra's folder is missing at r = 1 and 13 everywhere
PATCHES = {3: ["q10_7", "q2_7", "q4_5", "q4_9", "q6_11", "q6_3", "q6_7", "q8_5", "q8_9"],
           5: ["q4_7", "q6_5", "q6_9", "q8_7"], 7: ["q6_7"]}
NO_LIBRA_ROUNDS = (1, 13)


def cmd_estimate(a):
    """Job list and core-hours for the expected archive layout (PATCHES); nothing is read from the archive."""
    rounds = [int(x) for x in a.rounds.split(",")]
    rows = []
    for d, patches in PATCHES.items():
        for patch in patches:
            for b in ("X", "Z"):
                for r in rounds:
                    rel = f"google_105Q_surface_code_d3_d5_d7/d{d}_at_{patch}/{b}/r{r:02d}"
                    pws = [pw for pw in PATHWAYS if not (pw.startswith("libra") and r in NO_LIBRA_ROUNDS)]
                    rows.append({"rel": rel, "meta": {"distance": d, "basis": b, "rounds": r, "patch": patch},
                                 "n_shots": a.shots, "dems": [f"decoding_results/{pw}/error_model.dem" for pw in pws],
                                 "shipped": pws})
    P = PROFILES[a.profile]
    jobs, summ, notes = build_plan(rows, P, "/nonexistent", hash_dems=False)
    tot = sum(v["est_core_h"] for v in summ.values())
    out = {"made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "assumed_layout": {
               "patches": PATCHES, "bases": ["X", "Z"], "rounds": rounds, "shots_per_leaf": a.shots,
               "pathways": PATHWAYS, "libra_missing_at_rounds": list(NO_LIBRA_ROUNDS)},
           "profile": a.profile, "params": P, "stages": summ, "total_core_h": round(tot, 1),
           "total_wall_h_4_workers": round(tot / 4, 1), "notes": notes, "jobs": jobs}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=list)
    for k, v in summ.items():
        log(f"stage {k}: {v['jobs']} jobs in {v['units']} units, about {v['est_core_h']} core-h "
            f"({v['est_wall_h_4_workers']} h on 4 workers; first unit {v['first_unit_core_h']} core-h)")
    log(f"total about {tot:.1f} core-h, {tot / 4:.1f} h on 4 workers")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--shared", required=True); s.add_argument("--local", required=True)
    s.add_argument("--dist", type=int, default=3); s.add_argument("--meta-all", action="store_true")
    s.add_argument("--plan"); s.add_argument("--stages")
    s = sub.add_parser("inventory"); s.add_argument("--local", required=True); s.add_argument("--listing", required=True)
    s.add_argument("--shared-inventory"); s.add_argument("--out", required=True)
    s = sub.add_parser("plan"); s.add_argument("--inventory", required=True); s.add_argument("--local", required=True)
    s.add_argument("--out", required=True); s.add_argument("--profile", default="real", choices=["real", "dry"])
    s = sub.add_parser("estimate"); s.add_argument("--profile", default="real", choices=["real", "dry"])
    s.add_argument("--rounds", default="1,10,13,30,50,70,90,110,130,150,170,190,210,230,250")
    s.add_argument("--shots", type=int, default=50000); s.add_argument("--out", default=None)
    a = ap.parse_args()
    {"stage": cmd_stage, "inventory": cmd_inventory, "plan": cmd_plan, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
