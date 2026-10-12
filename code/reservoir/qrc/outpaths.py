"""Where re-runs write, so that rebuilding a number never replaces the stored result (decoder's audit of v1.0, gap 3).

Every script in code/reservoir writes through out_path() and reads its result-file inputs through in_path():

  default           a file stored at results/reservoir/<sub> is written to results/reservoir/rebuild/<sub> instead
                    (same layout). Other files under the project root, i.e. the dataset shared/data/tomato/tomato498.npz,
                    go to results/reservoir/rebuild/<path from the project root>.
  QRC_OUT=<dir>     the same layout under <dir> instead of results/reservoir/rebuild.
  QRC_OVERWRITE=1   write to the stored location, replacing the stored file (how the stored results were made).

Reads look in the output location first: in_path(p) returns the copy that this rebuild wrote, if there is one, and the
stored file otherwise. A single command therefore checks one step against the stored inputs, and a chain of commands
uses its own intermediate files (resume checkpoints included). Delete results/reservoir/rebuild/ to start a rebuild
afresh. Paths outside the project tree (scratch checkpoints, an explicit /tmp output) are never redirected.
Each redirected write and each read of a rebuilt copy is reported once on stderr.
"""
from __future__ import annotations

import glob as _glob
import os
import sys

ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RES = os.path.join(ROOT, "results", "reservoir")
DEFAULT_OUT = os.path.join(RES, "rebuild")
_said: set = set()


def _say(msg: str) -> None:
    if msg not in _said:
        _said.add(msg)
        print(f"[outpaths] {msg}", file=sys.stderr, flush=True)


def _rel(p: str) -> str:
    a = os.path.realpath(p)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def _inside(p: str, root: str) -> bool:
    return p == root or p.startswith(root + os.sep)


def overwrite() -> bool:
    """True when QRC_OVERWRITE=1: outputs replace the stored files."""
    return os.environ.get("QRC_OVERWRITE", "") == "1"


def out_root() -> str:
    """Root of this run's outputs: results/reservoir itself only under QRC_OVERWRITE=1."""
    if overwrite():
        return RES
    d = os.environ.get("QRC_OUT", "")
    d = os.path.realpath(d) if d else DEFAULT_OUT
    if d == RES:
        raise SystemExit("QRC_OUT names the stored results folder; set QRC_OVERWRITE=1 to replace stored files")
    return d


def _map(p: str) -> str:
    """Write location of path p (pure)."""
    a = os.path.realpath(p)
    if overwrite():
        return a
    root = out_root()
    if _inside(a, root):
        return a
    if _inside(a, RES):
        return os.path.join(root, os.path.relpath(a, RES))
    if _inside(a, ROOT):
        return os.path.join(root, os.path.relpath(a, ROOT))
    return a


def out_path(p: str, makedirs: bool = True) -> str:
    """Where this run writes the file that is stored at p (creates the folder)."""
    q = _map(p)
    if makedirs:
        os.makedirs(os.path.dirname(q) or ".", exist_ok=True)
    if q != os.path.realpath(p):
        _say(f"writing {_rel(q)} (stored {_rel(p)} left untouched; QRC_OVERWRITE=1 replaces it)")
    return q


def in_path(p: str) -> str:
    """The rebuilt copy of p if this rebuild has written one, else p itself."""
    q = _map(p)
    if q != os.path.realpath(p) and os.path.exists(q):
        _say(f"reading rebuilt {_rel(q)} instead of stored {_rel(p)}")
        return q
    return p


def in_glob(pattern: str) -> list:
    """Stored files matching pattern, each replaced by its rebuilt copy where one exists, plus rebuilt-only files."""
    found = {_map(p): p for p in _glob.glob(pattern)}
    mp = _map(pattern)
    if mp != os.path.realpath(pattern):
        for q in _glob.glob(mp):
            if q in found:
                _say(f"reading rebuilt {_rel(q)} instead of stored {_rel(found[q])}")
            else:
                _say(f"reading rebuilt {_rel(q)}")
            found[q] = q
    return [found[k] for k in sorted(found)]  # in the order of the stored names
