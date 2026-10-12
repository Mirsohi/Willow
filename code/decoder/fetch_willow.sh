#!/bin/bash
# fetch_willow.sh: fetch Google's Willow surface-code data (Zenodo record 13273331), check it, and put it in the
# project's shared folder. Needs a cloud environment whose network policy allows zenodo.org.
#
# usage: bash fetch_willow.sh [local work dir]      (default /home/claude/willow_fetch)
#   WILLOW_RECORD=<id>   another Zenodo record (default 13273331)
#   WILLOW_FILES=<regex> which files of the record to fetch (default: only the 105-qubit surface-code zip; the record
#                        also holds 72-qubit zips of 30, 12 and 65 GB)
#   WILLOW_SLICE=<bytes> files larger than this go to the shared folder as numbered slices (default 256 MiB)
#   WILLOW_SUBSET=<str>  if the archive is too big to extract locally, extract only members containing this (default d3_at)
#   WILLOW_ZENODO=<url>  base URL, only for testing against a local mock server (default https://zenodo.org)
#
# What it does, in order (each step is logged; it is safe to re-run, finished steps are skipped):
#   1. record metadata -> shared/data/willow/zenodo_record.json (title, version, DOI, licence, file list, md5s)
#   2. download with resume, then size and md5 check against the record
#   3. copy into shared/data/willow/, as 256 MiB slices <zip>.001, .002, ... On 2026-10-08 the project store never
#      accepted the 5.3 GiB file, nor 1 GiB slices, while 256 MiB slices uploaded in 12-32 s: its per-file limit is
#      somewhere between 256 MiB and 1 GiB. A write always succeeds locally even when the upload fails, and a read-back
#      in this container only sees its own mount cache, so neither proves the file is stored. Upload completion shows
#      only in /dev/shm/rclone-vfscache/vfsMeta/<fs>/<path> ("Dirty": false) or in a read from another container
#      (code/decoder/willow_stage.py checks every slice's md5 when it stages). MANIFEST.json lists the slices.
#   4. archive listing -> shared/data/willow/<zip>.listing.tsv (member path, size), so nobody needs to re-list the zip
#   5. local extraction (refuses unsafe member paths) and the decoder's loader inventory of the real archive
#      -> results/decoder/willow_inventory.json (first check that the format matches code/decoder/willow_data.py)
# Exit codes: 2 zenodo.org unreachable, 3 checksum or size failure, 4 unsafe archive, 5 inventory failed.
set -uo pipefail
REC=${WILLOW_RECORD:-13273331}
PAT=${WILLOW_FILES:-^google_105Q_surface_code_d3_d5_d7\.zip$}
SLICE=${WILLOW_SLICE:-268435456}
SUBSET=${WILLOW_SUBSET:-d3_at}
ZEN=${WILLOW_ZENODO:-https://zenodo.org}
X=${WILLOW_PROJECT:-/mnt/project-files/tensor-networks/deep-dives/experimental}   # override only for testing
S=$X/shared/data/willow
C=$X/code/decoder
R=$X/results/decoder

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# download_one <key> <url> <size> <md5> <dest dir>: resumable download with size and md5 checks
download_one() {
  local key=$1 url=$2 size=$3 md5=$4 dir=$5 f have got attempt
  f="$dir/$key"
  for attempt in $(seq 1 30); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" -gt "$size" ]; then log "$key: local file larger than the record says, starting again"; rm -f "$f"; have=0; fi
    if [ "$have" -eq "$size" ]; then break; fi
    log "$key: attempt $attempt, have $have of $size bytes"
    curl -sS -L --fail --connect-timeout 30 --speed-limit 10000 --speed-time 120 -C - -o "$f" "$url" \
      || log "$key: curl exit $? (will resume)"
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -eq "$size" ] && break
    sleep $(( attempt < 6 ? attempt * 10 : 60 ))
  done
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" -ne "$size" ]; then log "$key: size $have != $size after 30 attempts"; return 3; fi
  got=$(md5sum "$f" | cut -d' ' -f1)
  if [ -n "$md5" ] && [ "$got" != "$md5" ]; then
    log "$key: md5 $got != record $md5; deleting the local copy"; rm -f "$f"; return 3
  fi
  log "$key: size and md5 OK ($got)"
}

main() {
  local W=${1:-/home/claude/willow_fetch}
  mkdir -p "$W/dl" "$W/extract" "$S" || exit 1
  log "1. record $REC metadata"
  if ! curl -sS -L --fail --retry 5 --retry-delay 5 -o "$W/record.json" "$ZEN/api/records/$REC"; then
    log "cannot reach zenodo.org: this session's environment does not allow it (only sessions started after the change do)"
    exit 2
  fi
  python3 -I - "$W/record.json" "$W/files.tsv" "$REC" "$PAT" "$ZEN" <<'EOF' || exit 3
import json, re, sys
r = json.load(open(sys.argv[1])); rec = sys.argv[3]; pat = re.compile(sys.argv[4]); zen = sys.argv[5]
fs = r.get("files") or []
if isinstance(fs, dict):                      # InvenioRDM style {"entries": {...} or [...]}
    e = fs.get("entries") or []
    fs = list(e.values()) if isinstance(e, dict) else e
m = r.get("metadata") or {}
lic = m.get("license")
print("title:", m.get("title") or r.get("title"))
print("version:", m.get("version"), "| published:", m.get("publication_date"), "| doi:", r.get("doi") or m.get("doi"))
print("licence:", lic.get("id") if isinstance(lic, dict) else lic, "| latest version:", (r.get("links") or {}).get("latest"))
n = 0
with open(sys.argv[2], "w") as out:
    for f in fs:
        key = f.get("key") or f.get("filename"); size = int(f.get("size") or f.get("filesize") or 0)
        ck = str(f.get("checksum") or "")
        md5 = ck.split(":", 1)[1] if ck.startswith("md5:") else (ck if re.fullmatch(r"[0-9a-f]{32}", ck) else "")
        take = bool(pat.search(key))
        print(f"  {'FETCH' if take else 'skip '} {key}  {size/1e9:.3f} GB  {ck}")
        if take:
            url = f"{zen}/records/{rec}/files/{key}?download=1"
            out.write(f"{key}\t{url}\t{size}\t{md5}\n"); n += 1
if n == 0:
    print("no file matches", pat.pattern); sys.exit(1)
EOF
  cp "$W/record.json" "$S/zenodo_record.json"

  log "2. download"
  local key url size md5
  while IFS=$'\t' read -r key url size md5; do
    download_one "$key" "$url" "$size" "$md5" "$W/dl" || exit 3
  done < "$W/files.tsv"

  log "3. copy into $S (files over $SLICE bytes as numbered slices)"
  mkdir -p "$W/slices"; : > "$W/slices.tsv"
  while IFS=$'\t' read -r key url size md5; do
    local parts=() p n want
    if [ "$size" -gt "$SLICE" ]; then
      if ! ls "$W/slices/$key".[0-9][0-9][0-9] > /dev/null 2>&1; then
        split -b "$SLICE" -a 3 --numeric-suffixes=1 "$W/dl/$key" "$W/slices/$key." || { log "split of $key failed"; exit 3; }
      fi
      parts=("$W/slices/$key".[0-9][0-9][0-9])
      local got; got=$(cat "${parts[@]}" | md5sum | cut -d' ' -f1)
      if [ -n "$md5" ] && [ "$got" != "$md5" ]; then log "slices of $key do not rebuild to md5 $md5"; rm -f "${parts[@]}"; exit 3; fi
    else
      parts=("$W/dl/$key")
    fi
    for p in "${parts[@]}"; do
      n=$(basename "$p"); want=$(stat -c %s "$p")
      if [ "$(stat -c %s "$S/$n" 2>/dev/null || echo 0)" -ne "$want" ]; then
        cp "$p" "$S/$n.part" && mv -f "$S/$n.part" "$S/$n" || { log "copy of $n failed"; exit 3; }
        log "written $n ($want bytes)"
      else
        log "$n already in the shared folder"
      fi
      printf '%s\t%s\t%s\t%s\n' "$key" "$n" "$want" "$(md5sum "$p" | cut -d' ' -f1)" >> "$W/slices.tsv"
    done
  done < "$W/files.tsv"

  log "4. archive listing, 5. local extraction"
  python3 -I - "$W/files.tsv" "$W/dl" "$W/extract" "$S" "$SUBSET" <<'EOF' || exit 4
import os, shutil, sys, zipfile
tsv, dl, ex, S, subset = sys.argv[1:6]
for line in open(tsv):
    key = line.split("\t")[0]
    if not key.endswith(".zip"):
        continue
    z = zipfile.ZipFile(os.path.join(dl, key))
    infos = z.infolist()
    bad = [i.filename for i in infos if i.filename.startswith("/") or ".." in i.filename.replace("\\", "/").split("/")]
    if bad:
        print("unsafe member paths, not extracting:", bad[:5]); sys.exit(1)
    with open(os.path.join(S, key + ".listing.tsv"), "w") as out:
        for i in infos:
            out.write(f"{i.filename}\t{i.file_size}\n")
    tot = sum(i.file_size for i in infos)
    free = shutil.disk_usage(ex).free
    print(f"{key}: {len(infos)} members, {tot/1e9:.2f} GB uncompressed, {free/1e9:.1f} GB free here")
    if tot < 0.8 * free:
        members = infos
    else:
        members = [i for i in infos if subset in i.filename]
        print(f"not enough local disk for everything: extracting the {len(members)} members containing {subset!r}")
    z.extractall(ex, members=members)
EOF
  python3 -c "import stim, numpy" 2>/dev/null || python3 -m pip install -q stim numpy
  log "loader inventory of the real archive"
  if ! timeout 3600 python3 -I -B "$C/willow_data.py" inventory "$W/extract" "$R/willow_inventory.json" > "$W/inventory.log" 2>&1; then
    log "inventory failed; last lines:"; tail -20 "$W/inventory.log"; exit 5
  fi
  python3 -I - "$W/files.tsv" "$S" "$REC" "$R/willow_inventory.json" "$W/slices.tsv" <<'EOF'
import collections, datetime, json, os, sys
tsv, S, rec, inv, sl = sys.argv[1:6]
parts = collections.defaultdict(list)
for line in open(sl):
    key, name, size, md5 = line.rstrip("\n").split("\t")
    parts[key].append({"name": name, "size": int(size), "md5": md5})
files = []
for line in open(tsv):
    key, url, size, md5 = line.rstrip("\n").split("\t")
    f = {"key": key, "size": int(size), "md5": md5, "url": url,
         "verified": "size and md5 of the download match the Zenodo record; the shared copy still needs a read-back "
                     "from another session (a read-back here only sees this container's mount cache)"}
    if len(parts[key]) > 1 or (parts[key] and parts[key][0]["name"] != key):
        f["slices"] = parts[key]
        f["rebuild"] = "cat " + " ".join(p["name"] for p in parts[key]) + f" > {key}  (then check the md5 above)"
    files.append(f)
rows = json.load(open(inv))
by = collections.Counter((r.get("meta") or {}).get("distance") for r in rows)
errs = [r.get("folder") for r in rows if r.get("error")]
mism = [r.get("folder") for r in rows if r.get("recomputed_det_matches") is False]
man = {"zenodo_record": rec, "slices": [p for f in files for p in f.get("slices", [])], "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "files": files, "inventory": {"experiments": len(rows), "by_distance": {str(k): v for k, v in by.items()},
       "errors": errs[:20], "n_errors": len(errs), "recompute_mismatch": mism[:20]}}
json.dump(man, open(os.path.join(S, "MANIFEST.json"), "w"), indent=1)
print(json.dumps(man["inventory"], indent=1))
EOF
  log "done: the data are in $S (the zip or its slices, listing, MANIFEST.json); inventory in $R/willow_inventory.json"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
