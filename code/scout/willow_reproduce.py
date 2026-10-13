#!/usr/bin/env python3
"""willow_reproduce.py -- the reproducibility entry point of the Willow paper (manuscript v2.0).

Run it from the workspace root, the folder that holds code/, results/, notes/, board/ and manuscript/:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        nice -n 10 python3 -I -B code/scout/willow_reproduce.py [--out reproduce_out/] [--draft DIR] [--skip L1,L2,...]

It runs no decoder. From the stored result files alone it re-derives what the text reports, in four layers:

  L1  re-runs scout's eight derivation scripts (code/scout/willow_*.py) on the stored result files and compares
      each new output, key by key, with the stored output in results/scout/ (floats to a relative 1e-9; the keys
      'made' and 'seconds' and the spelling of input paths are ignored).
  L2  redraws Figs. 1, 2 and S1 with scout's figure scripts (Fig. 2 from the L1 output) and Figs. 3, 4, S2 and S3
      with decoder's figures() (code/decoder/analyze_willow.py, imported read-only) from the stored analysis, and
      compares each with the draft's PNG pixel by pixel and byte by byte (the shared folder's caBX chunk removed).
  L3  regenerates the numbers of Tables 2-5 (main text) and S7-S8 (supplement): every number in a table cell must
      equal, at the precision printed, a value under the result-file keys that the row's specification names.
  L4  checks every number of the ledger (numbers.md) against the file and key that its row cites, in the L1
      outputs (scout's derived files) or in the stored result files (decoder's and reservoir's).

Output, one line per number, then a summary:
  PASS        the number equals a regenerated or stored value at its printed precision (L3: under the row's keys;
              L4: 'key' at the cited key, 'file' elsewhere in a cited file, 'text' in a cited report or note,
              '|x|' up to sign, '1-x' as one minus a cited value, 'computed' as a sum, difference or ratio of
              cited values that the script registers by row: DERIVED for L4, the COMPUTED, RATIO, CI95 and MINMAX
              specifications for L3)
  NEAR        within one unit of the last printed digit but not equal by rounding (to review)
  FAIL        L1/L2: an output differs; L3: no value under the row's keys gives the number
  UNRESOLVED  L4: no cited value gives the number (a derivation the entry point does not redo, or an error)
  NOT COVERED L4: a reported, assumed or rule value, or a row whose sources are not result files
Small integers (below 100, without decimals: counts of series, round counts, levels) are context, counted but not
checked, as are dates, times, section numbers, md5 fragments and identifiers.
INPUT lines compare the md5 of each data file named in the ledger's key tables with the md5 the ledger lists
(PASS or DIFFERS; notes and boards change by design and are not held to theirs).

Writes into --out: the regenerated files, reproduce_report.json, reproduce_log.txt and reproduce_inputs.md (every
input file read under the root, with its size and md5; code files listed separately). Exit status 1 if L1, L2 or
L3 has a FAIL, else 0. Needs Python 3.10+, numpy, scipy, matplotlib and Pillow (made with 3.13, numpy 2.5.3, scipy
1.18.1, matplotlib 3.11.2, Pillow 12.3.0; figures are byte-identical only with the same matplotlib and fonts).
Scout, 2026-10-13; post hoc tooling, no new analysis.
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# The absolute root the scripts were first run from; a script that names it is run with it replaced by "./".
ORIGINAL_ROOT = "/mnt/project-files/tensor-networks/deep-dives/experimental/"
W = "results/decoder/willow/"
DRAFT = "manuscript/willow/v2.0_draft/"
THIS = os.path.abspath(__file__)

# ------------------------------------------------------------------------------------------------ child mode

def _child_record_hook(rec):
    def hook(event, args):
        try:
            if event == "open":
                path, mode, flags = args
                if isinstance(path, int) or path is None:
                    return
                p = os.fsdecode(path)
                if isinstance(mode, str):
                    w = any(c in mode for c in "wax+")
                else:
                    w = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)) if isinstance(flags, int) else False
                rec["write" if w else "read"].add(os.path.abspath(p))
            elif event in ("os.listdir", "os.scandir"):
                p = args[0] if args else "."
                if isinstance(p, (str, bytes, os.PathLike)):
                    rec["listdir"].add(os.path.abspath(os.fsdecode(p)))
        except Exception:
            pass
    return hook


def child_main(argv):
    """--child RECORD SCRIPT ARGS... | --child-decoder-figs RECORD ANALYSIS_JSON REPORTED_JSON OUTDIR"""
    mode, record = argv[0], argv[1]
    rec = {"read": set(), "write": set(), "listdir": set()}
    sys.addaudithook(_child_record_hook(rec))

    def dump():
        with open(record, "w") as f:
            json.dump({k: sorted(v) for k, v in rec.items()}, f, indent=0)

    import atexit
    atexit.register(dump)
    if mode == "--child":
        script, args = argv[2], argv[3:]
        src = open(script, encoding="utf-8").read()
        if ORIGINAL_ROOT in src:
            src = src.replace(ORIGINAL_ROOT, "./")
        sys.argv = [script] + list(args)
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        g = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}
        exec(compile(src, script, "exec"), g)
    else:
        import importlib.util
        fa, rep, outdir = argv[2], argv[3], argv[4]
        os.makedirs(outdir, exist_ok=True)
        spec = importlib.util.spec_from_file_location("analyze_willow_ro", "code/decoder/analyze_willow.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        made = mod.figures(json.load(open(fa)), outdir, json.load(open(rep)))
        print("decoder figures:", made)


# ------------------------------------------------------------------------------------------------ utilities

LOG = []


def say(line=""):
    print(line, flush=True)
    LOG.append(line)


def md5_file(p, cache={}):
    if p not in cache:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        cache[p] = h.hexdigest()
    return cache[p]


def png_core(raw):
    """PNG bytes without caBX (C2PA) chunks, which the shared folder adds to stored PNGs."""
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return raw
    out, i = [raw[:8]], 8
    while i < len(raw):
        n = struct.unpack(">I", raw[i:i + 4])[0]
        if raw[i + 4:i + 8] != b"caBX":
            out.append(raw[i:i + 12 + n])
        i += 12 + n
    return b"".join(out)


def relpath(p, root):
    a = os.path.abspath(p)
    return os.path.relpath(a, root) if a.startswith(root.rstrip("/") + "/") else a


def norm_str(s):
    if ORIGINAL_ROOT in s:
        s = s.replace(ORIGINAL_ROOT, "")
    while s.startswith("./"):
        s = s[2:]
    return s


IGNORE_KEYS = {"made", "seconds"}


def deep_compare(a, b, path="", diffs=None, stats=None):
    """Compare two JSON trees; floats to rel 1e-9 / abs 1e-12; strings with input paths normalised."""
    if diffs is None:
        diffs, stats = [], {"numbers": 0, "max_rel": 0.0}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE_KEYS:
                continue
            if k not in a or k not in b:
                diffs.append("%s.%s: key only in %s" % (path, k, "stored" if k in a else "regenerated"))
                continue
            deep_compare(a[k], b[k], path + "." + str(k), diffs, stats)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append("%s: list length %d against %d" % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            deep_compare(x, y, "%s[%d]" % (path, i), diffs, stats)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append("%s: %r against %r" % (path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        stats["numbers"] += 1
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return diffs, stats
        if a != b:
            d = abs(a - b)
            rel = d / max(abs(a), abs(b))
            stats["max_rel"] = max(stats["max_rel"], rel)
            if not (rel <= 1e-9 or d <= 1e-12):
                diffs.append("%s: %r against %r" % (path, a, b))
    elif isinstance(a, str) and isinstance(b, str):
        if norm_str(a) != norm_str(b):
            diffs.append("%s: %r against %r" % (path, a[:80], b[:80]))
    elif a != b:
        diffs.append("%s: %r against %r" % (path, str(a)[:80], str(b)[:80]))
    return diffs, stats


# ------------------------------------------------------------------------------------------------ numbers in text

SUPMAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
NUM_RE = re.compile(r"(?P<sign>[+−-])?(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?P<frac>\.\d+)?"
                    r"(?:\((?P<se>\d+)\))?(?:[eE](?P<epow>[+−-]?\d+))?"
                    r"(?P<exp>\s?×\s?10(?P<pow>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?")
STRIP_RES = [re.compile(p) for p in (
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?",              # dates
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[–-]\s?\d{1,2}:\d{2}(?::\d{2})?)?\s?Z?",  # times
    r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,32}\b…?",          # md5 fragments
    r"arXiv:\S+", r"\b\d{4}\.\d{4,5}(?:v\d+)?\b",                         # arXiv ids
    r"§\s?[\dA-Z][\w.]*(?:\s?[–-]\s?§?[\d.]+)?",                          # sections
    r"¶\s?\d+(?:\s?[–-]\s?¶?\d+)?",
    r"\blines?\s\d+(?:\s?[–-]\s?\d+)?(?:(?:,|\sand)\s\d+(?:\s?[–-]\s?\d+)?)*",
    r"\b20[12]\d\b(?![,.]\d)",                                             # years
    r"\[\d+(?:,\s?\d+)*\]",                                                # citations [1], [9]
)]
CONTEXT_BEFORE = re.compile(r"(?:\b[rdχBkτ]\s?(?:=|≤|≥|<|>|∈)\s?|\{\s?|\bk_max\s?=\s?)$")


class Tok:
    __slots__ = ("text", "value", "half", "kind", "pos")

    def __init__(self, text, value, half, kind, pos):
        self.text, self.value, self.half, self.kind, self.pos = text, value, half, kind, pos

    def __repr__(self):
        return "Tok(%s=%r±%g,%s)" % (self.text, self.value, self.half, self.kind)


def tokens(text):
    """Numbers in a text cell: Tok with kind 'num', 'se' (the SE of a(b) notation) or 'context'."""
    t = re.sub("[ᵖᵗˡʷᵃ‡†]", " ", text.replace("\\|", "|"))
    for r in STRIP_RES:
        t = r.sub(lambda m: " " * len(m.group(0)), t)
    out = []
    for m in NUM_RE.finditer(t):
        s, e = m.start("int"), m.end()
        before = t[s - 1] if s > 0 else " "
        sign = m.group("sign")
        if sign and m.start() > 0 and (t[m.start() - 1].isalnum() or t[m.start() - 1] in ")]_"):
            sign = None  # a binary minus or hyphen, not a sign
        after = t[e] if e < len(t) else " "
        after2 = t[e + 1] if e + 1 < len(t) else " "
        if before.isalnum() or before in "_.§¶" or (before == "-" and s > 1 and t[s - 2].isalnum()):
            continue
        if after.isalnum() or after == "_" or (after == "." and after2.isdigit()) or after in "⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹":
            continue
        ip, fr = m.group("int").replace(",", ""), m.group("frac") or ""
        k = len(fr) - 1 if fr else 0
        mant = float(ip + fr)
        p10 = 0
        if m.group("pow"):
            p10 = int(m.group("pow").translate(SUPMAP))
        elif m.group("epow"):
            p10 = int(m.group("epow").replace("−", "-"))
        neg = sign in ("-", "−")
        val = (-mant if neg else mant) * 10.0 ** p10
        half = 0.5 * 10.0 ** (-k) * 10.0 ** p10
        is_int = not fr and not m.group("pow") and not m.group("epow") and not m.group("se")
        kind = "num"
        if is_int and mant < 100:
            kind = "context"
        elif CONTEXT_BEFORE.search(t[max(0, m.start() - 8):m.start()]):
            kind = "context"
        out.append(Tok(m.group(0).strip(), val, half, kind, m.start()))
        if m.group("se"):
            se = int(m.group("se")) * 10.0 ** (-k) * 10.0 ** p10
            out.append(Tok("(" + m.group("se") + ")", se, half, "se", m.start()))
    return out


# ------------------------------------------------------------------------------------------------ sources

KEYMAP = {
    # decoder (folder results/decoder/willow/)
    "FR": [W + "final_round1/final_round1.json"], "FRM": [W + "final_round1/final_round1.md"],
    "FPR": [W + "final_round1/analysis/prereg_results.json"], "FRR": [W + "final_round1/analysis/report.md"],
    "FA": [W + "final_round1/analysis/analysis.json"], "FCL": [W + "final_round1/analysis/calibration_leaves.csv"],
    "K2T": [W + "supplementary/k2_extra_rung/k2_tally.json"],
    "NB": [W + "supplementary/nonband_check/summary.json", W + "supplementary/nonband_check/analysis/nonband_results.json",
           W + "supplementary/nonband_check/analysis/nonband_results.md"],
    "PRF": [W + "analysis/prereg_results.json"], "RF": [W + "analysis/report.md"], "AJF": [W + "analysis/analysis.json"],
    "CLF": [W + "analysis/calibration_leaves.csv"], "GC": [W + "pr3_planning/d5_gap_check.json"],
    "D7N": [W + "pr3_planning/d7_numbers.json"], "PW": [W + "pr3_planning/pr3_power.json"],
    "K2S": [W + "supplementary/k2_extra_rung/summary.json", W + "supplementary/k2_extra_rung/k2_runner.log"],
    "PIL": [W + "pr3/pilot/pilot.json"], "PT": [W + "pr3/main/pr3_tally.json"],
    "LT": [W + "pr3/leaf_table/pr3_leaf_table.json", W + "pr3/leaf_table/pr3_leaf_table.md"],
    "PRA": ["code/decoder/pr3/pr3_analyze.py"], "QS": [W + "queue_state.json"],
    "SEC": [W + "supplementary/p1_se_covariance.json"], "P3F": [W + "supplementary/p3_heldout_fitted.json"],
    "AM3": [W + "analysis_m3/prereg_results.json"], "AM3r": [W + "analysis_m3/report.md"],
    "AM3j": [W + "analysis_m3/analysis.json"], "AM4": [W + "analysis_m4/prereg_results.json"],
    "AM4r": [W + "analysis_m4/report.md"], "AM4j": [W + "analysis_m4/analysis.json"],
    "PG3": [W + "analysis_m3/paired_gaps.csv"], "CL3": [W + "analysis_m3/calibration_leaves.csv"],
    "F1": [W + "round2/f1_analysis.json"], "EX": [W + "round2/explore_analysis.json"],
    "VM0": [W + "round2/validation/sim_mech_V0.json"], "VM2": [W + "round2/validation/sim_mech_V2.json"],
    "R10": [W + "pr3/r10_posthoc/analysis/r10_results.json"], "R10M": [W + "pr3/r10_posthoc/analysis/r10_results.md"],
    "LT10": [W + "pr3/r10_posthoc/leaf_table/pr3_leaf_table_with_r10.json",
             W + "pr3/r10_posthoc/leaf_table/pr3_leaf_table_with_r10.md"],
    "D2": [W + "revision/d2_worstcase/d2_worstcase.json"],
    "D3R13": [W + "revision/d3_facts/d3_r13_census.json"], "D3D7": [W + "revision/d3_facts/d3_d7r13_census.json"],
    "D3F": [W + "revision/d3_facts/d3_facts.json", W + "revision/d3_facts/d3_facts.md"], "REP": ["code/decoder/willow_reported.json"],
    # reservoir
    "GTF": ["results/reservoir/willow_gapmech/final_k2/gap_table.json", "results/reservoir/willow_gapmech/final_k2/gap_series.csv"],
    "EAF": ["results/reservoir/willow_gapmech/final_k2/exact_analysis.json"],
    "SAB": ["results/reservoir/willow_sim/sim_analysis_AB.json"], "BR": ["results/reservoir/willow_sim/beta_real.json"],
    "D5": ["results/reservoir/willow_d5check/d5_check.json"], "CD5": ["results/reservoir/willow_d5check/compare_d5.json"],
    "BX": ["results/reservoir/willow_d5check/band_extrap.json"], "GT": ["results/reservoir/willow_gapmech/gap_table.json"],
    "EA": ["results/reservoir/willow_gapmech/exact_analysis.json"], "ES1": ["results/reservoir/willow_gapmech/exact_stage1.json"],
    "ES2": ["results/reservoir/willow_gapmech/exact_stage2.json"], "SA": ["results/reservoir/willow_sim/sim_analysis_A.json"],
    "R2M": ["results/reservoir/willow_r2check/r2m_check.json", "results/reservoir/willow_r2check/r2m_leaves.csv",
            "results/reservoir/willow_r2check/r2m_redecode.json", "results/reservoir/willow_r2check/r2m_localmax.json"],
    "RC": ["results/reservoir/willow_r2check/r2_check.json"], "RT": ["results/reservoir/willow_r2check/r2_temper.json"],
    "RBs": ["results/reservoir/willow_r2check/r2_basis.json"], "RZ": ["results/reservoir/willow_r2check/r2_zx.json"],
    "MJ": ["results/reservoir/willow_mech/mech.json"], "EB": ["results/reservoir/willow_mech/events_by_r.csv"],
    "SJ": ["results/reservoir/willow_mech/scaled.json"], "PJ": ["results/reservoir/willow_mech/pairs.json"],
    "HJ": ["results/reservoir/willow_mech/hot_q8_5_Z_r030.json"], "TW": ["results/reservoir/willow_mech/twin.json"],
    "R3": ["results/reservoir/willow_revision/r3_patch.json"], "R5": ["results/reservoir/willow_revision/r5_worstcase.json"],
    "R1C": ["results/reservoir/willow_revision/r1_phase1/r1_certificate.json"],
    "R1P": ["results/reservoir/willow_revision/r1_phase1/r1_posthoc.json"],
    # scout: regenerated in L1
    "K2F": ["results/scout/willow_k2_summary.json"], "SDF": ["results/scout/willow_derived_final.json"],
    "GRF": ["results/scout/willow_growth_re_final.json"], "OBF": ["results/scout/willow_outside_band_share_final.json"],
    "TMP": ["results/scout/willow_tm_posthoc.json"], "LD": ["results/scout/willow_libra_vs_mlD.json"],
    "CR1": ["results/scout/willow_compute_round1.json"], "V2D3": ["results/scout/revision/willow_v2d3_derived.json"],
    # scout: records of earlier states (stored, not regenerated)
    "SD": ["results/scout/willow_derived_d3d5.json"], "GR": ["results/scout/willow_growth_re.json"],
    "OB": ["results/scout/willow_outside_band_share.json"],
    # texts: notes, boards, plans
    "PP": ["notes/decoder/willow_prereg.md"], "PR2": ["notes/decoder/willow_prereg_r2.md"],
    "PR3": ["notes/decoder/willow_prereg_r3.md"], "plan": ["notes/decoder/willow_final_analysis_plan.md"],
    "GD": ["notes/reservoir/gapmech_design.md"], "SDG": ["notes/reservoir/sim_design.md"],
    "A1A": ["notes/decoder/willow_a1_assessment.md"], "FEAS": ["notes/decoder/feasibility.md"],
    "DB": ["board/decoder.md"], "MB": ["board/main.md"], "RB": ["board/reservoir.md"],
}
# names of the revision's analyses, which are also ledger row ids; in a source cell they name the analysis
ANALYSIS_NAMES = {"R1", "R2", "R3", "R4", "R5", "D1", "D2", "D3", "D4", "A1", "B9", "B10", "B11"}
LITERATURE = re.compile(r"^(?:S1|W-?n|W\d+t?|W\d+[a-z]?)$")
TEXT_EXT = (".md", ".txt", ".log", ".py")

# L1: scout's derivation scripts, each with its arguments ({OUT} is the output file) and its stored output
REGEN = [
    ("K2F", "code/scout/willow_k2_summary.py", ["./", "{OUT}"]),
    ("SDF", "code/scout/willow_derived_final.py", ["./", "{OUT}"]),
    ("GRF", "code/scout/willow_growth_re.py", ["./", "{OUT}"]),
    ("OBF", "code/scout/willow_outside_band_share.py", ["./", "{OUT}"]),
    ("TMP", "code/scout/willow_tm_posthoc.py", ["results/reservoir/willow_r2check/r2m_leaves.csv", "{OUT}"]),
    ("CR1", "code/scout/willow_compute_round1.py", ["{OUT}"]),
    ("LD", "code/scout/willow_libra_vs_mlD.py", ["./", "{OUT}"]),
    ("V2D3", "code/scout/willow_v2d3_derived.py", ["./", "{OUT}"]),
]
# L2: figures (draft file, how it is drawn, the regenerated file)
FIGS = [
    ("fig1_gaps_d3d5.png", "code/scout/fig_willow_gaps.py", ["./", "{FIGS}/fig_willow_gaps_d3d5"], "fig_willow_gaps_d3d5.png"),
    ("figS1_r10.png", "code/scout/fig_willow_gaps.py", ["./", "{FIGS}/fig_willow_gaps_r10only", "--r10only"],
     "fig_willow_gaps_r10only.png"),
    ("fig2_lambda_d3d5.png", "code/scout/fig_willow_lambda_d3d5_matched.py",
     ["./", "{V2D3}", "{FIGS}/fig_willow_lambda_d3d5_matched"], "fig_willow_lambda_d3d5_matched.png"),
]
DECODER_FIGS = [("fig3_calibration.png", "fig_willow_calibration.png"),
                ("fig4_decomposition.png", "fig_willow_decomposition.png"),
                ("figS2_syndrome_ll.png", "fig_willow_syndrome_ll.png"), ("figS3_eps_d3.png", "fig_willow_eps_d3.png")]


class Sources:
    """Loads result files (JSON trees, CSV rows, text numbers) once; scout's keys come from the L1 outputs."""

    def __init__(self, root, regen_dir=None):
        self.root, self.regen = root, {}
        self.cache, self.leaf_cache, self.missing = {}, {}, set()
        if regen_dir:
            for key, _s, _a in REGEN:
                p = os.path.join(regen_dir, os.path.basename(KEYMAP[key][0]))
                if os.path.exists(p):
                    self.regen[KEYMAP[key][0]] = p

    def path(self, rel):
        return self.regen.get(rel, rel)

    def kind(self, rel):
        return "text" if rel.endswith(TEXT_EXT) else ("csv" if rel.endswith(".csv") else "json")

    def load(self, rel):
        if rel in self.cache:
            return self.cache[rel]
        p = self.path(rel)
        obj = None
        try:
            k = self.kind(rel)
            if k == "json":
                obj = json.load(open(p))
            elif k == "csv":
                obj = {"rows": [dict(r) for r in csv.DictReader(open(p, newline=""))]}
            else:
                obj = open(p, encoding="utf-8", errors="replace").read()
        except (OSError, ValueError):
            self.missing.add(rel)
        self.cache[rel] = obj
        return obj

    def leaves(self, rel, obj=None, prefix=()):
        """(paths, values) of every number in obj (default: the whole file); CSV cells are parsed as numbers."""
        import numpy as np
        ck = (rel, prefix) if obj is None else None
        if ck and ck in self.leaf_cache:
            return self.leaf_cache[ck]
        if obj is None:
            obj = self.load(rel)
        paths, vals = [], []

        def walk(o, p):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, p + (k,))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, p + (i,))
            elif isinstance(o, bool) or o is None:
                return
            elif isinstance(o, (int, float)):
                if not (isinstance(o, float) and (math.isnan(o) or math.isinf(o))):
                    paths.append(p)
                    vals.append(float(o))
            elif isinstance(o, str) and self.kind(rel) == "csv":
                try:
                    v = float(o)
                    if not (math.isnan(v) or math.isinf(v)):
                        paths.append(p)
                        vals.append(v)
                except ValueError:
                    pass

        if isinstance(obj, str):
            for tk in tokens(obj):
                paths.append(("text@%d" % tk.pos,))
                vals.append(tk.value)
        elif obj is not None:
            walk(obj, prefix)
        res = (paths, np.array(vals, dtype=float))
        if ck:
            self.leaf_cache[ck] = res
        return res


def fmt_path(p):
    return ".".join(str(x) for x in p)


def resolve(node, segs, path=()):
    """Every (path, subtree) that segs reach from node. A segment is a key, '*', a glob, ('ALT', [...]) or an index;
    a key that is missing is retried joined with the following segments (keys that contain dots)."""
    if not segs:
        return [(path, node)]
    s, rest = segs[0], segs[1:]
    out = []
    if isinstance(node, dict):
        if isinstance(s, tuple):
            for a in s[1]:
                if a in node:
                    out += resolve(node[a], rest, path + (a,))
            return out
        if s == "*":
            for k, v in node.items():
                out += resolve(v, rest, path + (k,))
            return out
        if s in node:
            return resolve(node[s], rest, path + (s,))
        for n in range(1, len(rest) + 1):
            if not all(isinstance(r, str) for r in rest[:n]):
                break
            j = ".".join([s] + list(rest[:n]))
            if j in node:
                return resolve(node[j], rest[n:], path + (j,))
        if any(c in s for c in "*?"):
            for k, v in node.items():
                if fnmatch.fnmatchcase(str(k), s):
                    out += resolve(v, rest, path + (k,))
        return out
    if isinstance(node, list):
        if s == "*":
            for i, v in enumerate(node):
                out += resolve(v, rest, path + (i,))
        elif isinstance(s, str) and s.isdigit() and int(s) < len(node):
            out = resolve(node[int(s)], rest, path + (int(s),))
        return out
    return []


def item_segments(item):
    """'decisions.d5_rules.A.g5' -> segments; quoted keys are atomic; {a,b} alternatives; <x> and [] are '*'."""
    segs, cur, i = [], "", 0
    item = item.replace("\\|", "|").replace("`", "")
    while i < len(item):
        ch = item[i]
        if ch in "\"“”":
            j = i + 1
            while j < len(item) and item[j] not in "\"“”":
                j += 1
            if cur:
                segs.append(cur)
                cur = ""
            q = item[i + 1:j].strip()
            if q.startswith("…"):
                q = "*" + q[1:].strip()
            elif q.endswith("…"):
                q = q[:-1].strip() + "*"
            segs.append(q)
            i = j + 1
            continue
        if ch == "{":
            j = item.find("}", i)
            j = len(item) if j < 0 else j
            if cur:
                segs.append(cur)
                cur = ""
            segs.append(("ALT", [a.strip().strip('"') for a in item[i + 1:j].split(",") if a.strip()]))
            i = j + 1
            continue
        if ch == "<":
            j = item.find(">", i)
            j = len(item) if j < 0 else j
            cur += "*"
            i = j + 1
            continue
        if ch == "[":
            j = item.find("]", i)
            j = len(item) if j < 0 else j
            if cur:
                segs.append(cur)
                cur = ""
            inner = item[i + 1:j].strip().strip('"')
            segs.append(inner if inner else "*")
            i = j + 1
            continue
        if ch in ". \t":
            if cur:
                segs.append(cur)
                cur = ""
            i += 1
            continue
        if ch == "…":
            cur += "*"
            i += 1
            continue
        cur += ch
        i += 1
    if cur:
        segs.append(cur)
    return [s for s in segs if s not in ("", "*") or s == "*"]


def split_top(s, seps=",;"):
    """Split at separators outside quotes, braces and brackets."""
    out, cur, depth, q = [], "", 0, False
    for ch in s:
        if ch in "\"“”":
            q = not q
        elif not q and ch in "{[(":
            depth += 1
        elif not q and ch in "}])":
            depth = max(0, depth - 1)
        if ch in seps and not q and depth == 0:
            out.append((cur, ch))
            cur = ""
        else:
            cur += ch
    out.append((cur, ""))
    return out


class Spec:
    """A resolved source: a file, the subtrees the cited key paths reach (empty: the whole file), its class."""

    def __init__(self, key, rel, subtrees, text_scope=None, whole=False):
        self.key, self.rel, self.subtrees, self.text_scope, self.whole = key, rel, subtrees, text_scope, whole


def resolve_key_items(src, key, items_text):
    """[KEY item, item; item] -> Specs. Relative items are tried at every ancestor of the previous item's path."""
    specs = []
    files = KEYMAP.get(key, [])
    for rel in files:
        kind = src.kind(rel)
        obj = src.load(rel)
        if obj is None:
            continue
        if kind == "text":
            scope = None
            if key in ("DB", "MB", "RB"):
                scope = board_scope(obj, items_text)
            specs.append(Spec(key, rel, [], scope))
            continue
        if kind == "csv" or not items_text.strip():
            specs.append(Spec(key, rel, [], whole=True))
            continue
        found, prev = [], []
        parent_prefix = []
        for clause, sep in split_top(items_text, ";"):
            prev = []
            parent_prefix = []
            for it, _s in split_top(clause, ","):
                it = it.strip().strip(".").strip()
                if not it:
                    continue
                # 'X: a, b' -> X is a parent for the items that follow
                m = re.match(r"^([^\"“”:]+?):\s*(.*)$", it)
                if m and not it.startswith('"'):
                    parent_prefix = item_segments(m.group(1))
                    it = m.group(2).strip()
                    if not it:
                        continue
                segs = item_segments(it)
                segs = [x for x in segs if not (isinstance(x, str) and re.fullmatch(r"[a-z]+ ?", x) and x in
                                                 ("and", "the", "of", "from", "in", "equal", "as", "at", "on"))]
                if not segs:
                    continue
                hits = []
                bases = []
                if parent_prefix:
                    bases.append(parent_prefix)
                bases.append([])
                for n in range(len(prev), 0, -1):
                    bases.append(prev[:n])
                for b in bases:
                    hits = resolve(obj, list(b) + segs)
                    if hits:
                        prev = list(b) + segs
                        break
                if not hits:
                    # drop trailing descriptive words one at a time
                    for cut in range(len(segs) - 1, 0, -1):
                        for b in bases:
                            hits = resolve(obj, list(b) + segs[:cut])
                            if hits:
                                prev = list(b) + segs[:cut]
                                break
                        if hits:
                            break
                found += hits
        specs.append(Spec(key, rel, found))
    return specs


BOARD_HDR = re.compile(r"^### (\S+)", re.M)


def board_scope(text, items_text):
    """The posts of a board whose header time is cited (e.g. 17:47:21Z or 2026-10-12 00:35:28Z), else None."""
    times = re.findall(r"(\d{1,2}:\d{2}(?::\d{2})?)Z", items_text)
    if not times:
        return None
    heads = [(m.start(), m.group(1)) for m in BOARD_HDR.finditer(text)]
    parts = []
    for i, (st, h) in enumerate(heads):
        if any(t in h for t in times):
            en = heads[i + 1][0] if i + 1 < len(heads) else len(text)
            parts.append(text[st:en])
    return "\n".join(parts) if parts else None


def parse_source_cell(src, cell, rowsrc=None, depth=0):
    """Specs for a ledger source cell, plus flags: literature cited, row references followed."""
    specs, lit, notes = [], False, []
    cell = cell.replace("\\|", "|")
    groups = []
    depth_b, start = 0, None
    for i, ch in enumerate(cell):
        if ch == "[":
            if depth_b == 0:
                start = i
            depth_b += 1
        elif ch == "]" and depth_b:
            depth_b -= 1
            if depth_b == 0:
                groups.append((start, i))
    for gi, (a, b) in enumerate(groups):
        inner = cell[a + 1:b].strip()
        m = re.match(r"^([A-Za-z][A-Za-z0-9-]*)(?:\s+(.*))?$", inner, re.S)
        if not m:
            continue
        key, rest = m.group(1), (m.group(2) or "")
        if LITERATURE.match(key):
            lit = True
            continue
        if key not in KEYMAP:
            continue
        if not rest.strip():
            # '[V2D3] `a.b.{c,d}` (`x`, `y`)': the backticked paths after the group, up to the next group or ';'
            nxt = groups[gi + 1][0] if gi + 1 < len(groups) else len(cell)
            tail = cell[b + 1:nxt].split(";")[0]
            ticks = re.findall(r"`([^`]+)`", tail)
            if ticks:
                rest = ticks[0] + (": " + ", ".join(ticks[1:]) if len(ticks) > 1 else "")
                if len(ticks) > 1 and "(" in tail:
                    # `a.b` (`x`, `y`): x and y are children of a.b
                    rest = ticks[0] + ": " + ", ".join(ticks[1:])
        specs += resolve_key_items(src, key, rest)
    # backticked file paths outside the groups; a bare name is in the folder of the last full path before it
    lastdir = W
    for t in re.findall(r"`([^`\s]+\.(?:json|csv|md|txt))`", cell):
        if re.match(r"^(results|notes|code|board|shared|manuscript)/", t):
            rel = t
            lastdir = os.path.dirname(t) + "/"
        else:
            rel = lastdir + t if os.path.exists(lastdir + t) else W + t
        if os.path.exists(src.path(rel)) and src.load(rel) is not None:
            specs.append(Spec("file", rel, [], whole=True))
    # earlier drafts cited as the source of a carried number
    for ver in re.findall(r"\bv(1\.[0-2])\b", cell):
        rel = "manuscript/willow/v%s/manuscript.md" % ver
        if os.path.exists(rel):
            specs.append(Spec("v" + ver, rel, [], whole=True))
    # rows cited as sources ('from R20, R20a'; 'A15'; 'M20; Q10')
    if rowsrc is not None and depth < 2:
        bare = re.sub(r"\[[^\]]*\]", " ", cell)
        bare = re.sub(r"`[^`]*`", " ", bare)
        rids = []
        for m in re.finditer(r"\b([A-Z]{1,2})(\d+)([a-z]?)(?:\s?[–-]\s?\1?(\d+)([a-z]?))?\b", bare):
            pre, a, sa, b, sb = m.groups()
            if b and not sa and not sb and int(b) > int(a) and int(b) - int(a) < 30:
                rids += ["%s%d" % (pre, k) for k in range(int(a), int(b) + 1)]
            else:
                rids.append(pre + a + sa)
        for rid in rids:
            if rid in ANALYSIS_NAMES:
                continue
            if rid in rowsrc:
                s2, l2, _n = parse_source_cell(src, rowsrc[rid], rowsrc, depth + 1)
                specs += s2
                lit = lit or l2
                notes.append(rid)
    if re.search(r"\[W\d|\[S1\]|\breported\b", cell):
        lit = True
    return specs, lit, notes


# ------------------------------------------------------------------------------------------------ matching

SCALES = (1.0, 100.0, 1e3, 1e4, 1e5, 1e6)


class _Level:
    """Sorted candidate arrays for one set of leaves: direct (v x scale), |v x scale|, and 1 - v, v - 1."""

    def __init__(self, name, paths, vals, specs, whole=False):
        import numpy as np
        self.name, self.paths, self.vals, self.specs, self.whole = name, paths, vals, specs, whole  # specs[i]: leaf i's Spec
        n = len(vals)
        self.n = n
        if n == 0:
            self.arrs = []
            return
        direct = np.concatenate([vals * s for s in SCALES])
        der = np.concatenate([(1 - vals), (1 - vals) * 100.0, (vals - 1), (vals - 1) * 100.0])
        self.arrs = []
        for tag, arr in (("", direct), ("|x|", np.abs(direct)), ("derived", der)):
            o = np.argsort(arr, kind="stable")
            self.arrs.append((tag, arr[o], o))

    def find(self, value, tol, tag_wanted):
        import numpy as np
        for tag, arr, o in self.arrs:
            if tag != tag_wanted:
                continue
            lo = np.searchsorted(arr, value - tol, "left")
            hi = np.searchsorted(arr, value + tol, "right")
            if hi > lo:
                # the first scale (x1, x100, x1e3, ...) that gives the number, then the closest value
                js = o[lo:hi].astype(int)
                blks = js // self.n
                first = blks == blks.min()
                cand = np.arange(lo, hi)[first]
                k = int(cand[int(np.argmin(np.abs(arr[cand] - value)))])
                j = int(o[k])
                i = j % self.n
                blk = j // self.n
                if tag == "derived":
                    tr, sc = ("1-x", 1.0) if blk == 0 else ("1-x", 100.0) if blk == 1 else ("x-1", 1.0) if blk == 2 else ("x-1", 100.0)
                else:
                    tr, sc = "", SCALES[blk]
                return self.specs[i], self.paths[i], float(self.vals[i]), sc, tr
        return None


_FILE_LEVELS = {}


class Candidates:
    """The values a row may draw on: at the cited keys ('key'), anywhere in a cited file ('file'), and the numbers
    of a cited text ('text'). A file cited without a key path counts as cited at its root."""

    def __init__(self, src, specs):
        import numpy as np
        self.keys, self.files, tv, tsp = [], [], [], []
        seen = set()
        for sp in specs:
            obj = src.load(sp.rel)
            if obj is None:
                continue
            if isinstance(obj, str):
                txt = sp.text_scope if sp.text_scope else obj
                for tk in tokens(txt):
                    tv.append(tk.value)
                    tsp.append(sp)
                continue
            subs = sp.subtrees if sp.subtrees else ([((), obj)] if sp.whole else [])
            kp, kv = [], []
            for p, sub in subs:
                ps, vs = src.leaves(sp.rel, sub, p)
                kp += ps
                kv += list(vs)
            if kp:
                self.keys.append(_Level("key", kp, np.array(kv, dtype=float), [sp] * len(kp), whole=not sp.subtrees))
            if sp.rel not in seen:
                seen.add(sp.rel)
                if sp.rel not in _FILE_LEVELS:
                    ps, vs = src.leaves(sp.rel)
                    _FILE_LEVELS[sp.rel] = _Level("file", ps, vs, [Spec(sp.key, sp.rel, [])] * len(ps))
                self.files.append(_FILE_LEVELS[sp.rel])
        self.text = None
        if tv:
            a = np.array(tv)
            o = np.argsort(a, kind="stable")
            self.text = (a[o], [tsp[i] for i in o])

    def match(self, tk, allow_file=True, allow_text=True):
        """-> (status, detail): status 'PASS', 'NEAR' or None."""
        import numpy as np
        tol = tk.half * (1 + 1e-9) + 1e-12 * abs(tk.value)
        levels = self.keys + (self.files if allow_file else [])
        for tag in ("", "|x|", "derived"):
            if tag == "|x|" and tk.value < 0:
                continue
            for lv in levels:
                hit = lv.find(tk.value, tol, tag)
                if hit:
                    sp, pp, raw, sc, tr = hit
                    how = lv.name + ("" if not tag else " " + (tr if tag == "derived" else tag))
                    return "PASS", "%s [%s %s] %s%.10g ×%g" % (how, sp.key, fmt_path(pp), (tr + " of ") if tr else "",
                                                              raw, sc)
        if allow_text and self.text is not None:
            arr, sps = self.text
            for sc in SCALES:
                lo = np.searchsorted(arr * sc, tk.value - tol, "left")
                hi = np.searchsorted(arr * sc, tk.value + tol, "right")
                if hi > lo:
                    sp = sps[lo]
                    return "PASS", "text [%s %s]%s" % (sp.key, sp.rel, "" if sc == 1 else " ×%g" % sc)
        for lv in self.keys:
            if lv.whole:
                continue
            hit = lv.find(tk.value, 2 * tol, "")
            if hit:
                sp, pp, raw, sc, tr = hit
                return "NEAR", "%s [%s %s] %.10g ×%g" % (lv.name, sp.key, fmt_path(pp), raw, sc)
        return None, ""


# ------------------------------------------------------------------------------------------------ L3: tables

# Row specifications: (first-cell prefix, [(key, path with '>' between segments; '*' and globs allowed)], columns
# to skip). The numbers of the other cells must come from these subtrees.
def _t3_row(dec, d3=None, d5=None):
    return [("SDF", "gaps_W5_entry7>%s" % dec), ("V2D3", "table3_abs_excess>%s" % dec),
            ("R3", "decoders>%s>ruleA>table7" % dec)] + (d3 or []) + (d5 or [])


TABLES = [
    ("manuscript", "Table 2.", [0, 1, 2], [
        ("P1 (PR1", [("SEC", "full_delta_method_inflated")]),
        ("P4 (PR1", [("AM3", "P4_mismatch_cost_d3>pooled")]),
        ("P2 (PR1", [("FR", "entry7_se>pooled_gaps:d3|ML.si1000 vs ML.rl")]),
        ("P3 (PR1", [("AM3", "P3_calibration_ML_rl_d3>pooled")]),
        ("P5 (PR1", [("AM3", "P5_ML_rl_vs_NN>pooled>3")]),
        ("d = 5 replication", [("FR", "decisions>d5_rules>A>g5")]),
        ("K2 (PR1)", [("K2F", "")]),
        ("F1 primary", [("F1", "primary_m_D")]),
    ]),
    ("manuscript", "Table 3.", [0], [
        ("Libra", [("V2D3", "g3_W5"), ("FR", "decisions>d5_rules>A>g5"), ("FR", "decisions>d5_rules>A>paired_ratio_libra_over_ML_rl"),
                   ("V2D3", "table3_abs_excess>libra"), ("R3", "decoders>libra>ruleA>table7")]),
        ("Harmony", _t3_row("harmony")),
        ("corr. matching", _t3_row("corrmatch")),
        ("corr. MWPM", _t3_row("corr_mwpm")),
        ("MWPM", _t3_row("mwpm")),
        ("Tesseract", _t3_row("tesseract")),
        ("BeliefMatching", [("FR", "entry7_se>pooled_gaps_W5:d3|beliefmatch.rl vs ML.rl"), ("V2D3", "table3_abs_excess>beliefmatch")]),
    ]),
    ("manuscript", "Table 4.", [0, 1], [
        ("headline", [("FR", "decisions>d5_rules>A>g5"), ("V2D3", "growth_rows>headline_rule_A"),
                      ("FA", "pooled_counts>d5|libra.rl vs ML.rl")]),
        ("frozen SE formula", [("V2D3", "growth_rows>frozen_se_formula_rule_A"),
                               ("CI95z1.96", "FR:decisions>d5_rules>A>g5>g", "FR:decisions>d5_rules>A>g5>se_frozen_formula")]),
        ("rule B", [("FR", "decisions>d5_rules>B>g5"), ("V2D3", "growth_rows>rule_B")]),
        ("no exclusion (sensitivity", [("FR", "decisions>d5_rules>none>g5"), ("V2D3", "growth_rows>no_exclusion")]),
        ("clustered by series", [("R3", "decoders>libra>ruleA>series_cluster"), ("GRF", "A>growth_cluster_se")]),
        ("random effects", [("GRF", "A>random_effects_d5"), ("GRF", "A>growth_random_effects_hksj")]),
        ("one leaf left out", [("V2D3", "leave_one_out_final>*>one_d5_leaf>g5_range", "g₅"),
                               ("V2D3", "leave_one_out_final>*>one_d5_leaf>Delta_range", "growth")]),
        ("one round count left out", [("V2D3", "leave_one_out_final>*>one_round_count_both_distances>g5_range", "g₅"),
                                      ("V2D3", "leave_one_out_final>*>one_round_count_both_distances>Delta_range", "growth")]),
        ("one patch left out", [("MINMAX", "R3:decoders>libra>ruleA>leave_one_patch_out_d5>*>g5", "g₅"),
                                ("MINMAX", "R3:decoders>libra>no_exclusion>leave_one_patch_out_d5>*>g5", "g₅"),
                                ("MINMAX", "R3:decoders>libra>ruleA>leave_one_patch_out_d5>*>G", "growth"),
                                ("MINMAX", "R3:decoders>libra>ruleA>leave_one_patch_out_d5>*>lambda_ratio", "growth")]),
        ("clustered by patch", [("R3", "decoders>libra>ruleA>patch_cluster")]),
        ("K2 (pre-specified)", [("K2F", ""), ("COMPUTED", "count_gap_if_ml_1pct_low")]),
        ("census outside the band", [("V2D3", "census_growth>A")]),
        ("census, no exclusion", [("V2D3", "census_growth>none")]),
        ("census, flip rates per round count", [("V2D3", "census_growth>C_only|*|A")]),
        ("census, how the strata", [("V2D3", "census_growth>A")]),
        ("independent contraction", []),
    ]),
    ("manuscript", "Table 5.", [0, 1], [
        ("RL, shipped", [("V2D3", "b9>RL")]),
        ("RL with one temperature", [("V2D3", "b9>temperature_series")]),
        ("T̂, class scales", [("V2D3", "b9>T_hat"), ("AM3", "P4_mismatch_cost_d3>pooled")]),
        ("D̂, class scales", [("V2D3", "b9>D_hat"), ("F1", "primary_m_D")]),
        ("T̂_mech", [("V2D3", "b9>T_hat_mech"), ("EX", "heldout>Tm>m_X"), ("TMP", "Tm_all_18")]),
        ("D̂_mech", [("V2D3", "b9>D_hat_mech"), ("EX", "heldout>Dm>m_X")]),
    ]),
    ("supplement", "Table S7.", [0, 5], [
        ("Libra (shipped)", [("V2D3", "g3_W5"), ("FR", "decisions>d5_rules>A>g5"),
                             ("FR", "decisions>d5_rules>A>paired_ratio_libra_over_ML_rl"), ("SDF", "libra>A")]),
        ("Tesseract (ours)", [("SDF", "gaps_W5_entry7>tesseract")]),
        ("Harmony (shipped)", [("SDF", "gaps_W5_entry7>harmony")]),
        ("corr. matching (shipped)", [("SDF", "gaps_W5_entry7>corrmatch")]),
        ("corr. MWPM (ours)", [("SDF", "gaps_W5_entry7>corr_mwpm")]),
        ("MWPM (ours)", [("SDF", "gaps_W5_entry7>mwpm")]),
        ("BeliefMatching (ours)", [("FR", "entry7_se>pooled_gaps_W5:d3|beliefmatch.rl vs ML.rl")]),
        ("Tesseract, without", [("GRF", "every_decoder_entry7_refit_and_without_q6_5_X>tesseract"), ("SDF", "gaps_W5_entry7>tesseract"),
                 ("RATIO", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>tesseract>ruleA_without_q6_5_X>g5", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>tesseract>g3"),
                 ("CI95", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>tesseract>g3", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>tesseract>g3_se_entry7")]),
        ("Harmony, without", [("GRF", "every_decoder_entry7_refit_and_without_q6_5_X>harmony"), ("SDF", "gaps_W5_entry7>harmony"),
                 ("RATIO", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>harmony>ruleA_without_q6_5_X>g5", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>harmony>g3"),
                 ("CI95", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>harmony>g3", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>harmony>g3_se_entry7")]),
        ("corr. matching, without", [("GRF", "every_decoder_entry7_refit_and_without_q6_5_X>corrmatch"), ("SDF", "gaps_W5_entry7>corrmatch"),
                 ("RATIO", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corrmatch>ruleA_without_q6_5_X>g5", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corrmatch>g3"),
                 ("CI95", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corrmatch>g3", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corrmatch>g3_se_entry7")]),
        ("corr. MWPM, without", [("GRF", "every_decoder_entry7_refit_and_without_q6_5_X>corr_mwpm"), ("SDF", "gaps_W5_entry7>corr_mwpm"),
                 ("RATIO", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corr_mwpm>ruleA_without_q6_5_X>g5", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corr_mwpm>g3"),
                 ("CI95", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corr_mwpm>g3", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>corr_mwpm>g3_se_entry7")]),
        ("MWPM, without", [("GRF", "every_decoder_entry7_refit_and_without_q6_5_X>mwpm"), ("SDF", "gaps_W5_entry7>mwpm"),
                 ("RATIO", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>mwpm>ruleA_without_q6_5_X>g5", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>mwpm>g3"),
                 ("CI95", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>mwpm>g3", "GRF:every_decoder_entry7_refit_and_without_q6_5_X>mwpm>g3_se_entry7")]),
        ("Harmony, on", [("V2D3", "c12_libra_windows>harmony>A"),
                 ("RATIO", "V2D3:c12_libra_windows>harmony>A>g5", "V2D3:c12_libra_windows>harmony>A>g3")]),
        ("corr. matching, on", [("V2D3", "c12_libra_windows>corrmatch>A"),
                 ("RATIO", "V2D3:c12_libra_windows>corrmatch>A>g5", "V2D3:c12_libra_windows>corrmatch>A>g3")]),
        ("corr. MWPM, on", [("V2D3", "c12_libra_windows>corr_mwpm>A"),
                 ("RATIO", "V2D3:c12_libra_windows>corr_mwpm>A>g5", "V2D3:c12_libra_windows>corr_mwpm>A>g3")]),
        ("MWPM, on", [("V2D3", "c12_libra_windows>mwpm>A"),
                 ("RATIO", "V2D3:c12_libra_windows>mwpm>A>g5", "V2D3:c12_libra_windows>mwpm>A>g3")]),
    ]),
    ("supplement", "Table S8.", [0], [
        ("ML (RL)", [("FR", "decisions>lambda_3_5>ML.rl")]),
        ("Libra", [("FR", "decisions>lambda_3_5>libra.rl"), ("R3", "decoders>libra>ruleA>conditional"),
                   ("R3", "decoders>libra>ruleA>patch_cluster"), ("R3", "decoders>libra>ruleA>table7")]),
        ("Harmony", [("FR", "decisions>lambda_3_5>harmony.rl"), ("R3", "decoders>harmony>ruleA>conditional"),
                 ("R3", "decoders>harmony>ruleA>patch_cluster"), ("R3", "decoders>harmony>ruleA>table7")]),
        ("corr. MWPM", [("FR", "decisions>lambda_3_5>corr_mwpm.rl"), ("R3", "decoders>corr_mwpm>ruleA>conditional"),
                 ("R3", "decoders>corr_mwpm>ruleA>patch_cluster"), ("R3", "decoders>corr_mwpm>ruleA>table7")]),
        ("corr. matching", [("FR", "decisions>lambda_3_5>corrmatch.rl"), ("R3", "decoders>corrmatch>ruleA>conditional"),
                 ("R3", "decoders>corrmatch>ruleA>patch_cluster"), ("R3", "decoders>corrmatch>ruleA>table7")]),
        ("MWPM", [("FR", "decisions>lambda_3_5>mwpm.rl"), ("R3", "decoders>mwpm>ruleA>conditional"),
                 ("R3", "decoders>mwpm>ruleA>patch_cluster"), ("R3", "decoders>mwpm>ruleA>table7")]),
        ("Tesseract", [("FA", "mean_eps>d3|tesseract.rl"), ("SDF", "tesseract_d5"), ("SDF", "lambda_3_5>tesseract_windows_differ"),
                       ("SDF", "gaps_W5_entry7>tesseract"), ("R3", "decoders>tesseract>ruleA>patch_cluster"),
                       ("R3", "decoders>tesseract>ruleA>table7")]),
        ("NN (reported)", None),
    ]),
]


def computed_values(src):
    """Values a table states that are one step of arithmetic on stored values (named, with their inputs)."""
    out = {}
    pc = src.load(KEYMAP["FA"][0])["pooled_counts"]["d5|libra.rl vs ML.rl"]
    out["count_gap_if_ml_1pct_low"] = [(pc["errors_A"] / (1.01 * pc["errors_B"]) - 1,
                                        "FA pooled_counts.d5|libra.rl vs ML.rl: errors_A/(1.01 errors_B) - 1")]
    return out


def _leaf(src, ref):
    key, path = ref.split(":", 1)
    hits = resolve(src.load(KEYMAP[key][0]), [x for x in path.split(">") if x])
    if len(hits) != 1 or not isinstance(hits[0][1], (int, float)):
        raise KeyError(ref)
    return float(hits[0][1])


def special_values(src, spec, comp):
    """('COMPUTED', name) | ('RATIO', a, b) -> a/b | ('CI95', g, se) -> g -/+ 1.959964 se ('CI95z1.96': decoder's
    z = 1.96) | ('MINMAX', path) -> the least and largest value a wildcard path reaches; a, b, g, se as KEY:path."""
    if spec[0] == "COMPUTED":
        return comp[spec[1]]
    if spec[0] == "RATIO":
        a, b = _leaf(src, spec[1]), _leaf(src, spec[2])
        return [(a / b, "%s / %s" % (spec[1], spec[2]))]
    if spec[0] == "MINMAX":
        key, path = spec[1].split(":", 1)
        hits = [float(v) for _p, v in resolve(src.load(KEYMAP[key][0]), [x for x in path.split(">") if x])
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not hits:
            raise KeyError(spec[1])
        return [(min(hits), "min over %s (%d values)" % (spec[1], len(hits))), (max(hits), "max over %s (%d values)" % (spec[1], len(hits)))]
    if spec[0] in ("CI95", "CI95z1.96"):
        g, se = _leaf(src, spec[1]), _leaf(src, spec[2])
        z = 1.96 if spec[0] == "CI95z1.96" else 1.959964
        return [(g - z * se, "%s - %g %s" % (spec[1], z, spec[2])), (g + z * se, "%s + %g %s" % (spec[1], z, spec[2]))]
    return []


def parse_md_tables(text):
    """{caption prefix 'Table N.': (header, rows)}; cells split at unescaped '|'."""
    lines = text.split("\n")
    out = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^: \*\*(Table S?\d+\.)\*\*", ln)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("|"):
            j += 1
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", lines[j].strip())[1:-1]]
            rows.append(cells)
            j += 1
        if len(rows) >= 2:
            out[m.group(1)] = (rows[0], [r for r in rows[2:]])
    return out


def check_tables(src, draft, results):
    import numpy as np  # noqa: F401
    comp = computed_values(src)
    docs = {d: parse_md_tables(open(os.path.join(draft, d + ".md"), encoding="utf-8").read())
            for d in ("manuscript", "supplement")}
    for doc, cap, skip, rowspecs in TABLES:
        if cap not in docs[doc]:
            results.append(("L3", "FAIL", cap, "table not found in %s.md" % doc))
            say("FAIL  L3 %s not found in %s.md" % (cap, doc))
            continue
        header, rows = docs[doc][cap]
        used = set()
        for prefix, keys in rowspecs:
            cand_rows = [r for r in rows if r[0].replace("*", "").startswith(prefix) and id(r) not in used]
            if not cand_rows:
                results.append(("L3", "FAIL", cap, "row '%s' not found" % prefix))
                say("FAIL  L3 %s row '%s' not found" % (cap, prefix))
                continue
            row = cand_rows[0]
            used.add(id(row))
            label = "%s %s" % (cap.rstrip("."), prefix)
            if keys is None:
                for ci, cell in enumerate(row):
                    if ci in skip:
                        continue
                    for tk in tokens(cell):
                        if tk.kind != "context":
                            results.append(("L3", "NOT COVERED", label, tk.text))
                            say("NOT COVERED  L3 %s | %s: %s (reported)" % (label, header[ci], tk.text))
                continue
            SPECIAL = ("COMPUTED", "RATIO", "CI95", "CI95z1.96", "MINMAX")
            built = {}

            def cands_for(col):
                """Candidates and computed values of the specs that apply to a column (a spec may name one)."""
                sel = []
                for spec in keys:
                    colf = spec[3] if spec[0] in SPECIAL and len(spec) > 3 else (spec[2] if spec[0] not in SPECIAL and len(spec) > 2 else None)
                    if spec[0] == "MINMAX":
                        colf = spec[2] if len(spec) > 2 else None
                    if colf is None or col.startswith(colf):
                        sel.append(spec)
                tkey = tuple(sel)
                if tkey in built:
                    return built[tkey]
                specs, extra = [], []
                for spec in sel:
                    if spec[0] in SPECIAL:
                        try:
                            extra += special_values(src, spec, comp)
                        except (KeyError, ZeroDivisionError, TypeError) as e:
                            say("FAIL  L3 %s: computed value %s not formed (%s)" % (label, spec, e))
                            results.append(("L3", "FAIL", label, "computed %s" % (spec,)))
                        continue
                    key, path = spec[0], spec[1]
                    segs = [x for x in path.split(">") if x] if path else []
                    obj = src.load(KEYMAP[key][0])
                    hits = resolve(obj, segs) if segs else [((), obj)]
                    if not hits:
                        say("FAIL  L3 %s: key %s %s not found" % (label, key, path))
                        results.append(("L3", "FAIL", label, "key %s %s not found" % (key, path)))
                    specs.append(Spec(key, KEYMAP[key][0], hits))
                built[tkey] = (Candidates(src, specs), extra)
                return built[tkey]

            for ci, cell in enumerate(row):
                if ci in skip or ci >= len(header):
                    continue
                cands, extra = cands_for(header[ci])
                for tk in tokens(cell):
                    if tk.kind == "context":
                        results.append(("L3", "context", label, tk.text))
                        continue
                    st, det = cands.match(tk, allow_file=False, allow_text=False)
                    if st != "PASS":
                        for val, frm in extra:
                            for sc in (1.0, 100.0):
                                if abs(val * sc - tk.value) <= tk.half * (1 + 1e-9):
                                    st, det = "PASS", "computed [%s] %.10g ×%g" % (frm, val, sc)
                    st = st or "FAIL"
                    results.append(("L3", st, label, "%s | %s" % (header[ci], tk.text), det.split(" [")[0] if st == "PASS" else ""))
                    say("%-5s L3 %s | %s: %s  %s" % (st, label, header[ci], tk.text, det))
        # rows of the table that no specification covers: their numbers are not checked
        for r in rows:
            if id(r) in used:
                continue
            nums = [tk.text for ci, cell in enumerate(r) if ci not in skip for tk in tokens(cell) if tk.kind != "context"]
            if nums:
                results.append(("L3", "FAIL", "%s %s" % (cap.rstrip("."), r[0][:30]), "row without a specification"))
                say("FAIL  L3 %s %s: row without a specification (%s)" % (cap.rstrip("."), r[0][:40], ", ".join(nums)))


# ------------------------------------------------------------------------------------------------ L4: the ledger

NUM_COLS = ("in the text", "value; label", "number in the text", "number", "value",
            "number(s) as the text gives them")
SRC_COLS = ("source", "file and key")
NOT_COVERED_LABELS = re.compile(r"\b(reported|assumed|rule|rules|design|estimate|estimates|pre-registered|fact|counted)\b")


def ledger_rows(text):
    """Yield (section, row id, {header: cell}) for every number row of the ledger; key tables are returned apart."""
    sec, header, keyrows = "", None, []
    rows = []
    for ln in text.split("\n"):
        if ln.startswith("#"):
            sec = ln.lstrip("#").strip()
            header = None
            continue
        if not ln.startswith("|"):
            header = None if not ln.strip() else header
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip())[1:-1]]
        if header is None:
            header = cells
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        d = dict(zip(header, cells))
        if header[0] == "key":
            keyrows.append(d)
            continue
        rid = d.get("row") or d.get("number") or cells[0]
        rows.append((sec, rid if "row" in d else "", d))
    return rows, keyrows


def check_inputs_md5(src, keyrows, results):
    """The ledger's key tables list each result file's md5 when read; compare with the files as found."""
    for d in keyrows:
        keys = re.findall(r"\[([A-Za-z][A-Za-z0-9-]*)\]", d.get("key", ""))
        if len(keys) != 1 or keys[0] not in KEYMAP:
            continue
        listed = set(re.findall(r"\b[0-9a-f]{32}\b", d.get("md5", "")))
        if not listed:
            continue
        for rel in KEYMAP[keys[0]]:
            p = rel  # the stored file as deposited (relative to the root), never its L1 regeneration
            if not os.path.exists(p) or os.path.basename(rel) not in d.get("file", "") + d.get("key", ""):
                continue  # only the files the key table names
            h = md5_file(p)
            ok = h in listed
            if not ok and src.kind(rel) != "json" and src.kind(rel) != "csv":
                continue  # notes, boards and reports change by design; only data files are held to their md5
            st = "PASS" if ok else "DIFFERS"
            results.append(("INPUT", st, keys[0], rel))
            say("%-7s INPUT [%s] %s md5 %s%s" % (st, keys[0], rel, h, "" if ok else " (the ledger lists %s)" % ", ".join(sorted(listed))))


class _Get:
    """Read one value, or all values a wildcard path reaches, from a key's file (paths with '>' between segments)."""

    def __init__(self, src):
        self.src = src

    def all(self, key, path, idx=0):
        obj = self.src.load(KEYMAP[key][idx])
        return [float(v) for _p, v in resolve(obj, [x for x in path.split(">") if x])
                if isinstance(v, (int, float)) and not isinstance(v, bool)]

    def one(self, key, path, idx=0):
        vs = self.all(key, path, idx)
        if len(vs) != 1:
            raise KeyError("%s %s: %d values" % (key, path, len(vs)))
        return vs[0]


def _obf_basis(g, basis, rule_a):
    lv = g.src.load(KEYMAP["OBF"][0])["leaves"]
    sel = [x for x in lv if "/%s/" % basis in "/" + x["leaf"] + "/" and not (rule_a and x["k2_still_failing"])]
    return sum(x["k_libra"] for x in sel) / sum(x["k_ml"] for x in sel) - 1


def _cl3(g, above):
    rows = [r for r in g.src.load(KEYMAP["CL3"][0])["rows"] if r["entity"] == "ML.rl"]
    return float(sum(1 for r in rows if float(r["ratio"]) > 1)) if above else float(len(rows))


def _ratio_abs_excess(g, dec):
    return (g.one("V2D3", "table3_abs_excess>%s>5>mean_delta" % dec) / g.one("V2D3", "table3_abs_excess>%s>3>mean_delta" % dec))


# Numbers the ledger gives as one step of arithmetic on stored values ("our arithmetic"): the row (its id, or the
# start of its first cell), what is computed, and how. L4 checks the row's numbers against these as well.
DERIVED = {
    "A15": [("F1 primary_m_D.pooled errors_A - errors_B",
             lambda g: g.one("F1", "primary_m_D>pooled>errors_A") - g.one("F1", "primary_m_D>pooled>errors_B")),
            ("F1 primary_m_D.pooled (errors_A - errors_B)/errors_A",
             lambda g: 1 - g.one("F1", "primary_m_D>pooled>errors_B") / g.one("F1", "primary_m_D>pooled>errors_A"))],
    "R50": [("OBF leaves, X, rule A (K2-passing leaves): sum k_libra / sum k_ml - 1", lambda g: _obf_basis(g, "X", True)),
            ("OBF leaves, Z, rule A: sum k_libra / sum k_ml - 1", lambda g: _obf_basis(g, "Z", True))],
    "R51": [("OBF rule_A redecoded.n10 + tau1_only.n10",
             lambda g: g.one("OBF", "rule_A>redecoded>n10") + g.one("OBF", "rule_A>tau1_only>n10")),
            ("OBF rule_A redecoded.n01 + tau1_only.n01",
             lambda g: g.one("OBF", "rule_A>redecoded>n01") + g.one("OBF", "rule_A>tau1_only>n01")),
            ("OBF all_24_leaves redecoded.n10 + tau1_only.n10",
             lambda g: g.one("OBF", "all_24_leaves>redecoded>n10") + g.one("OBF", "all_24_leaves>tau1_only>n10")),
            ("OBF all_24_leaves redecoded.n01 + tau1_only.n01",
             lambda g: g.one("OBF", "all_24_leaves>redecoded>n01") + g.one("OBF", "all_24_leaves>tau1_only>n01"))],
    "4a: decision changes": [("RC primary_m_D RL_wrong_D_right + RL_right_D_wrong",
                              lambda g: g.one("RC", "primary_m_D>RL_wrong_D_right") + g.one("RC", "primary_m_D>RL_right_D_wrong"))],
    "4a: 270 of 270": [("CL3: ML.rl leaves", lambda g: _cl3(g, False)), ("CL3: ML.rl leaves with R > 1", lambda g: _cl3(g, True))],
    "N23": [("NB population C00 + C11", lambda g: g.one("NB", "population>C00", 1) + g.one("NB", "population>C11", 1))],
    "N24": ([("BR per_r.%s.beta.components NC_sum_m_t2_estimated - NC_sum_m_t1" % r,
              (lambda r: lambda g: g.one("BR", "per_r>%s>beta>components>NC_sum_m_t2_estimated" % r)
               - g.one("BR", "per_r>%s>beta>components>NC_sum_m_t1" % r))(r)) for r in ("10", "30")]
            + [("BR per_r.%s.beta.components sum_m_t2_estimated - sum_m_ladder" % r,
                (lambda r: lambda g: g.one("BR", "per_r>%s>beta>components>sum_m_t2_estimated" % r)
                 - g.one("BR", "per_r>%s>beta>components>sum_m_ladder" % r))(r)) for r in ("10", "30")]
            + [("BR per_r.%s.counts D_rows + C_rows" % r,
                (lambda r: lambda g: g.one("BR", "per_r>%s>counts>D_rows" % r) + g.one("BR", "per_r>%s>counts>C_rows" % r))(r))
               for r in ("10", "30", "50")]),
    "M15": [("NB summary stages D_t3_targeted.shots + C_t3_targeted.shots",
             lambda g: g.one("NB", "stages>D_t3_targeted>shots") + g.one("NB", "stages>C_t3_targeted>shots")),
            ("NB summary stages D_t3_random.shots + C_t3_random.shots",
             lambda g: g.one("NB", "stages>D_t3_random>shots") + g.one("NB", "stages>C_t3_random>shots")),
            ("NB summary stages: the four t3 stages' shots",
             lambda g: sum(g.one("NB", "stages>%s>shots" % s) for s in ("D_t3_targeted", "C_t3_targeted", "D_t3_random", "C_t3_random")))],
    "V12": [("V2D3 table3_abs_excess.%s: mean_delta at d = 5 / d = 3" % d, (lambda d: lambda g: _ratio_abs_excess(g, d))(d))
            for d in ("libra", "harmony", "corrmatch", "corr_mwpm", "mwpm")],
    "V22": [("D3R13 aggregate.d5_r13_rl.instructions + D3F merging.aggregate.d5_rl.instructions",
             lambda g: g.one("D3R13", "aggregate>d5_r13_rl>instructions") + g.one("D3F", "merging>aggregate>d5_rl>instructions")),
            ("D3D7 aggregate.d7_r13_rl.instructions + D3F merging.aggregate.d7_rl.instructions",
             lambda g: g.one("D3D7", "aggregate>d7_r13_rl>instructions") + g.one("D3F", "merging>aggregate>d7_rl>instructions"))],
    "N25": [("SDF calibration_R_by_distance.d5_W5.R (the pooled R at d = 5)", lambda g: g.one("SDF", "calibration_R_by_distance>d5_W5>R"))],
    "D20": [("SDF calibration_R_by_distance.d5_W5.R (the pooled R at d = 5)", lambda g: g.one("SDF", "calibration_R_by_distance>d5_W5>R"))],
    "R47": [("FA pooled_counts.d5|libra.rl vs ML.rl: errors_A / (1.01 errors_B) - 1",
             lambda g: g.one("FA", "pooled_counts>d5|libra.rl vs ML.rl>errors_A") / (1.01 * g.one("FA", "pooled_counts>d5|libra.rl vs ML.rl>errors_B")) - 1)],
    "4a: P1's SE raised": [("SEC full_delta_method_inflated.se / AM3 P1_libra_vs_ML_rl_d3.pooled.se_rel - 1",
                           lambda g: g.one("SEC", "full_delta_method_inflated>se") / g.one("AM3", "P1_libra_vs_ML_rl_d3>pooled>se_rel") - 1)],
}


def derived_for(g, rname):
    out = []
    for k, items in DERIVED.items():
        if rname == k or rname.startswith(k):
            for desc, fn in items:
                try:
                    out.append((fn(g), desc))
                except Exception as e:  # noqa: BLE001 - a missing key is reported, not fatal
                    out.append((None, "%s: not formed (%s)" % (desc, e)))
    return out


def check_ledger(src, ledger_text, results):
    rows, keyrows = ledger_rows(ledger_text)
    check_inputs_md5(src, keyrows, results)
    getter = _Get(src)
    rowsrc = {}
    for _sec, rid, d in rows:
        if rid:
            rowsrc[rid] = " ; ".join(d.get(c, "") for c in SRC_COLS if c in d)
    for sec, rid, d in rows:
        srccell = " ; ".join(d.get(c, "") for c in SRC_COLS if c in d)
        label = d.get("label", "")
        if "value; label" in d and ";" in d["value; label"]:
            label = label + " " + d["value; label"].rsplit(";", 1)[1]
        try:
            specs, lit, via = parse_source_cell(src, srccell, rowsrc)
        except Exception as e:  # a malformed cell must not stop the run
            specs, lit, via = [], False, []
            say("ERROR L4 %s: source cell not parsed (%s)" % (rid or sec[:20], e))
        cands = Candidates(src, specs) if specs else None
        nc_row = bool(NOT_COVERED_LABELS.search(label)) or lit
        rname = rid or ("%s: %s" % (sec.split(".")[0], (d.get("number") or "")[:40]))
        for col in NUM_COLS:
            if col not in d:
                continue
            cell = d[col]
            if col == "value; label" and ";" in cell:
                cell = cell.rsplit(";", 1)[0]
            for tk in tokens(cell):
                if tk.kind == "context":
                    results.append(("L4", "context", rname, tk.text))
                    continue
                st, det = (cands.match(tk) if cands else (None, ""))
                if st != "PASS":
                    for val, desc in derived_for(getter, rname):
                        if val is None:
                            continue
                        for sc in (1.0, 100.0):
                            if abs(val * sc - tk.value) <= tk.half * (1 + 1e-9) + 1e-12 * abs(tk.value):
                                st, det = "PASS", "computed [%s] %.10g ×%g" % (desc, val, sc)
                                break
                        if st == "PASS":
                            break
                if st is None:
                    if not specs:
                        st, det = "NOT COVERED", "no result file cited" + (" (reported)" if lit else "")
                    elif nc_row:
                        st, det = "NOT COVERED", "label: %s" % (label.strip()[:60] or "literature")
                    else:
                        st, det = "UNRESOLVED", "not at the cited keys: %s" % ", ".join(sorted({s.key for s in specs}))
                results.append(("L4", st, rname, tk.text, det.split(" [")[0] if st == "PASS" else ""))
                say("%-11s L4 %s: %s  %s" % (st, rname, tk.text, det))


# ------------------------------------------------------------------------------------------------ L1, L2

def run_child(args, record, logf, root):
    cmd = [sys.executable, "-I", "-B", THIS] + args[:1] + [record] + args[1:]
    t0 = time.time()
    with open(logf, "w") as lf:
        r = subprocess.run(cmd, cwd=root, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def layer1(root, out, results, reads):
    regen_dir = os.path.join(out, "regenerated")
    os.makedirs(regen_dir, exist_ok=True)
    for key, script, args in REGEN:
        rel_out = KEYMAP[key][0]
        newf = os.path.join(regen_dir, os.path.basename(rel_out))
        a = [x.replace("{OUT}", newf) for x in args]
        rec = os.path.join(out, "records", key + ".json")
        os.makedirs(os.path.dirname(rec), exist_ok=True)
        rc, sec = run_child(["--child", script] + a, rec, os.path.join(out, "logs", key + ".log"), root)
        reads[key] = rec
        if rc != 0 or not os.path.exists(newf):
            results.append(("L1", "FAIL", key, "script exited %d" % rc))
            say("FAIL  L1 [%s] %s exited %d after %.0f s (log: %s)" % (key, script, rc, sec, os.path.join(out, "logs", key + ".log")))
            continue
        diffs, stats = deep_compare(json.load(open(rel_out)), json.load(open(newf)))
        if diffs:
            results.append(("L1", "FAIL", key, "%d differences" % len(diffs)))
            say("FAIL  L1 [%s] %s: %d differences against %s (first: %s)" % (key, script, len(diffs), rel_out, "; ".join(diffs[:3])))
        else:
            results.append(("L1", "PASS", key, "%d numbers" % stats["numbers"]))
            say("PASS  L1 [%s] %s -> %s equals %s: %d numbers, largest relative difference %.1e (%.0f s)"
                % (key, script, os.path.basename(newf), rel_out, stats["numbers"], stats["max_rel"], sec))
    return regen_dir


def compare_png(new, draft_png, label, results):
    import numpy as np
    from PIL import Image
    if not os.path.exists(new):
        results.append(("L2", "FAIL", label, "not drawn"))
        say("FAIL  L2 %s: not drawn" % label)
        return
    a = np.asarray(Image.open(draft_png).convert("RGBA"))
    b = np.asarray(Image.open(new).convert("RGBA"))
    core = hashlib.md5(png_core(open(draft_png, "rb").read())).hexdigest()
    newmd5 = hashlib.md5(open(new, "rb").read()).hexdigest()
    if a.shape != b.shape:
        results.append(("L2", "FAIL", label, "size %s against %s" % (b.shape, a.shape)))
        say("FAIL  L2 %s: size %s against the draft's %s" % (label, b.shape[:2], a.shape[:2]))
        return
    nd = int(np.any(a != b, axis=2).sum())
    if nd == 0:
        results.append(("L2", "PASS", label, "pixels identical" + ("; bytes identical" if core == newmd5 else "")))
        say("PASS  L2 %s: %dx%d pixels identical; bytes %s (md5 %s, the draft's without caBX %s)"
            % (label, a.shape[1], a.shape[0], "identical" if core == newmd5 else "differ", newmd5, core))
    else:
        results.append(("L2", "FAIL", label, "%d pixels differ" % nd))
        say("FAIL  L2 %s: %d of %d pixels differ (largest channel difference %d)"
            % (label, nd, a.shape[0] * a.shape[1], int(np.abs(a.astype(int) - b.astype(int)).max())))


def layer2(root, out, draft, regen_dir, results, reads):
    figs = os.path.join(out, "figs")
    os.makedirs(figs, exist_ok=True)
    v2d3 = os.path.join(regen_dir, os.path.basename(KEYMAP["V2D3"][0])) if regen_dir else KEYMAP["V2D3"][0]
    if not os.path.exists(v2d3):
        v2d3 = KEYMAP["V2D3"][0]
    for i, (dname, script, args, made) in enumerate(FIGS):
        a = [x.replace("{FIGS}", figs).replace("{V2D3}", v2d3) for x in args]
        rec = os.path.join(out, "records", "fig%d.json" % i)
        rc, sec = run_child(["--child", script] + a, rec, os.path.join(out, "logs", "fig%d.log" % i), root)
        reads["fig%d" % i] = rec
        if rc != 0:
            say("FAIL  L2 %s: %s exited %d" % (dname, script, rc))
        compare_png(os.path.join(figs, made), os.path.join(draft, "figs", dname), "%s (%s)" % (dname, script), results)
    dfig = os.path.join(figs, "decoder")
    rec = os.path.join(out, "records", "decoder_figs.json")
    rc, sec = run_child(["--child-decoder-figs", KEYMAP["FA"][0], KEYMAP["REP"][0], dfig], rec,
                        os.path.join(out, "logs", "decoder_figs.log"), root)
    reads["decoder_figs"] = rec
    if rc != 0:
        say("FAIL  L2 decoder figures(): exited %d (log %s)" % (rc, os.path.join(out, "logs", "decoder_figs.log")))
    for dname, made in DECODER_FIGS:
        compare_png(os.path.join(dfig, made), os.path.join(draft, "figs", dname),
                    "%s (code/decoder/analyze_willow.py figures())" % dname, results)


# ------------------------------------------------------------------------------------------------ manifest

def write_manifest(root, out, reads, parent_reads, path):
    files = {}
    for key, rec in reads.items():
        try:
            r = json.load(open(rec))
        except (OSError, ValueError):
            continue
        for p in r.get("read", []):
            files.setdefault(p, set()).add(key)
    for p in parent_reads:
        files.setdefault(p, set()).add("L2-L4")
    rootabs = os.path.abspath(root)
    outabs = os.path.abspath(out)
    rows = []
    for p, users in files.items():
        if not p.startswith(rootabs + "/") or p.startswith(outabs + "/") or p == THIS:
            continue
        if not os.path.isfile(p) or "/__pycache__/" in p:
            continue
        rel = os.path.relpath(p, rootabs)
        rows.append((rel, os.path.getsize(p), md5_file(p), ", ".join(sorted(users))))
    rows.sort()
    code = [r for r in rows if r[0].startswith("code/")]
    data = [r for r in rows if not r[0].startswith("code/")]
    L = ["# Inputs of `code/scout/willow_reproduce.py`", "",
         "Every file under the workspace root that one run of the entry point read (recorded by an audit hook on",
         "file opens, in the entry point and in each script it ran), with its size and md5 at the run. A deposit that",
         "keeps these files at these relative paths reproduces the run. Made %s by `code/scout/willow_reproduce.py`"
         % time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
         "(md5 %s). 'Read by' names the L1 output key, the figure job, or L2-L4 for the entry point itself (the" % md5_file(THIS),
         "draft's figures, its tables and the ledger, and the files they cite). Boards and notes grow by design, so a",
         "later copy of one can differ from the md5 here; the draft's files are those of the draft that was checked;",
         "every result file and every code file must match.", "",
         "## Data, results and texts (%d files, %.1f MB)" % (len(data), sum(r[1] for r in data) / 1e6), "",
         "| path | bytes | md5 | read by |", "|---|---|---|---|"]
    L += ["| `%s` | %d | %s | %s |" % r for r in data]
    L += ["", "## Code (%d files)" % (len(code) + 1), "", "| path | bytes | md5 | read by |", "|---|---|---|---|"]
    L += ["| `code/scout/willow_reproduce.py` | %d | %s | the entry point |" % (os.path.getsize(THIS), md5_file(THIS))]
    L += ["| `%s` | %d | %s | %s |" % r for r in code]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return len(data), len(code)


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="reproduce_out/")
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--ledger", default=None, help="default <draft>/numbers.md")
    ap.add_argument("--skip", default="", help="layers to skip, e.g. L1,L2 (L3 and L4 then read the stored outputs)")
    ap.add_argument("--manifest", default=None, help="default <out>/reproduce_inputs.md")
    a = ap.parse_args()
    root = os.getcwd()
    if not (os.path.isdir("code/scout") and os.path.isdir("results") and os.path.isdir(a.draft)):
        sys.exit("run from the workspace root (it must hold code/scout/, results/ and %s)" % a.draft)
    try:
        if os.nice(0) < 10:
            os.nice(10 - os.nice(0))
    except OSError:
        pass
    skip = {s.strip().upper() for s in a.skip.split(",") if s.strip()}
    out = a.out
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    parent_reads = set()
    rootabs = os.path.abspath(root)

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1]
            if isinstance(mode, str) and not any(c in mode for c in "wax+"):
                p = os.path.abspath(os.fsdecode(args[0]))
                if p.startswith(rootabs + "/"):
                    parent_reads.add(p)
    sys.addaudithook(hook)
    t0 = time.time()
    results, reads = [], {}
    say("willow_reproduce.py: root %s, draft %s, out %s, started %s" % (root, a.draft, out,
                                                                       time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())))
    regen_dir = None
    if "L1" not in skip:
        say("\n== L1: scout's derivation scripts, re-run on the stored result files")
        regen_dir = layer1(root, out, results, reads)
    if "L2" not in skip:
        say("\n== L2: figures")
        layer2(root, out, a.draft, regen_dir, results, reads)
    src = Sources(root, regen_dir)
    if "L3" not in skip:
        say("\n== L3: Tables 2-5 (main text) and S7-S8 (supplement)")
        check_tables(src, a.draft, results)
    if "L4" not in skip:
        say("\n== L4: the ledger (%s), row by row" % (a.ledger or os.path.join(a.draft, "numbers.md")))
        check_ledger(src, open(a.ledger or os.path.join(a.draft, "numbers.md"), encoding="utf-8").read(), results)
    # summary
    from collections import Counter
    say("\n== Summary")
    c = Counter((r[0], r[1]) for r in results)
    for layer in ("INPUT", "L1", "L2", "L3", "L4"):
        parts = ["%s %d" % (st, n) for (ly, st), n in sorted(c.items()) if ly == layer]
        if parts:
            say("%-5s %s" % (layer, ", ".join(parts)))
    for layer in ("L3", "L4"):
        how = Counter(r[4] for r in results if r[0] == layer and r[1] == "PASS" and len(r) > 4)
        if how:
            say("%s PASS by strength: %s" % (layer, ", ".join("%s %d" % (k or "?", n) for k, n in how.most_common())))
    if src.missing:
        say("files cited but not found: %s" % ", ".join(sorted(src.missing)))
    nd, ncode = write_manifest(root, out, reads, parent_reads, a.manifest or os.path.join(out, "reproduce_inputs.md"))
    say("manifest: %d data files and %d code files -> %s" % (nd, ncode + 1, a.manifest or os.path.join(out, "reproduce_inputs.md")))
    bad = sum(n for (ly, st), n in c.items() if ly in ("L1", "L2", "L3") and st == "FAIL")
    say("done in %.0f s; %s" % (time.time() - t0, "no FAIL in L1-L3" if not bad else "%d FAIL in L1-L3" % bad))
    json.dump({"results": results, "counts": {"%s %s" % k: v for k, v in c.items()}, "seconds": time.time() - t0},
              open(os.path.join(out, "reproduce_report.json"), "w"), indent=0)
    with open(os.path.join(out, "reproduce_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--child", "--child-decoder-figs"):
        child_main(sys.argv[1:])
    else:
        sys.exit(main())
