"""Synthetic end-to-end check of pr3_r10_analyze.py. For three scenarios the driver pr3_r10_posthoc.py runs the
frozen runner's loop with fake units (fake_full_decode_willow.py) that write every file the analysis reads; a fake
round 1 (d = 3: 9 patches x 2 bases, "rl"; d = 5: 4 patches x 2 bases, "rl_s5"; one d = 5 Z leaf excluded under rule
A), its final analysis.json and final_round1.json are written here; PR3's real leaf table is read. The analysis runs as
a subprocess; its numbers are compared with direct computations from the generated arrays.
  A: X fails the ladder and passes the extra rung, Z passes the ladder -> X and Z.
  B: X still fails after the extra rung, Z passes -> Z only (d = 3: 9 leaves, d = 5: 3, d = 7: 1).
  C: both still fail -> leaf rows only.
Also: a tampered tally fails a check (exit 1); an incomplete tally is refused; fieller() against a brute-force
inversion; the delta-method SE of g_3 and g_7 against a bootstrap made here.
usage: PR3_CODE=<code/decoder> python3 -I -B test_r10_analyze.py <code/decoder/pr3> <analyze script folder> <scratch>
       <real pr3_leaf_table.json>"""
import json, math, os, shutil, subprocess, sys, tempfile, types
CODE_PR3, ACODE, SCR, LT_REAL = sys.argv[1:5]
sys.path.insert(0, CODE_PR3)
import numpy as np
import pr3_r10_posthoc as D
C, PRR = D.C, D.PRR
sys.path.insert(0, ACODE)
import pr3_r10_analyze as A
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_full_decode_willow.py")
ASCRIPT = os.path.join(ACODE, "pr3_r10_analyze.py")
EXP = C.EXP
ok = []


def check(cond, what):
    ok.append(bool(cond)); print(("ok   " if cond else "FAIL ") + what, flush=True)


def eps(E, r=10):
    return 0.5 * (1 - (1 - 2 * E) ** (1 / r))


def run_driver(T, modes):
    pilot = {"measured_rates": {"c1_10": 43.0072098317, "s_10": 0.005, "c2_10": 383.1532430062512, "c1_50": 51.2932625268,
                                "cb_50": 1.047663489999997, "s_50": 0.02, "c2_50": 325.92149488, "c3_50": 1035.233558995,
                                "ovh_extra_s": 4.195043083999984, "ovh_ladder_s": 0.0},
             "stage1_ladder": {"peak_rss": 536133632}, "stage2_extra": {"peak_rss": 349929472}}
    json.dump(pilot, open(os.path.join(T, "pilot.json"), "w"))
    a = types.SimpleNamespace(root="/nonexistent", res=os.path.join(T, "res"), project="/nonexistent",
                              work=os.path.join(T, "w"), state=os.path.join(T, "s"), logs=os.path.join(T, "l"),
                              pilot=os.path.join(T, "pilot.json"), design="25k", guard_core_hours=13.0, max_heavy=4,
                              min_free_gib=3, cap_gib=4, poll=0.3, dry_run=False)
    R = D.R10(a)

    def cmd(u):
        lf, j = u["leaf"], u["part"]
        d = R.ladder_dir(lf, j) if u["kind"] == "ladder" else os.path.dirname(R.extra_json(lf, j))
        arg = {"kind": u["kind"], "dir": d, "leaf": lf, "part": j, "slice": C.part_slice("25k", j),
               "ladder_npz": os.path.join(R.ladder_dir(lf, j), PRR.LADDER_DN + ".npz"), "modes": modes}
        return [sys.executable, "-I", "-B", FAKE, json.dumps(arg)]
    R.cmd = cmd
    return R.run(), a


SHIPPED = ["decoding_results/correlated_matching_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
           "decoding_results/correlated_matching_decoder_with_si1000_prior/obs_flips_predicted.b8",
           "decoding_results/harmony_decoder_with_rl_optimized_prior/obs_flips_predicted.b8",
           "decoding_results/harmony_decoder_with_si1000_prior/obs_flips_predicted.b8",
           "decoding_results/libra_decoder_with_rl_optimized_prior/obs_flips_predicted.b8"]
SH_ENT = ["corrmatch.rl", "corrmatch.si1000", "harmony.rl", "harmony.si1000", "libra.rl"]
RATES = {3: {"corrmatch.rl": (0.004, 0.06), "corrmatch.si1000": (0.005, 0.06), "harmony.rl": (0.003, 0.07),
             "harmony.si1000": (0.003, 0.06), "libra.rl": (0.002, 0.08), "corr_mwpm.rl": (0.004, 0.05), "mwpm.rl": (0.01, 0.04)},
         5: {"corrmatch.rl": (0.005, 0.05), "corrmatch.si1000": (0.006, 0.05), "harmony.rl": (0.003, 0.06),
             "harmony.si1000": (0.004, 0.06), "libra.rl": (0.002, 0.06), "corr_mwpm.rl": (0.005, 0.04), "mwpm.rl": (0.012, 0.03)}}


def make_round1(T):
    """Fake round-1 decode folders, analysis.json and final_round1.json; returns (root, final_r1, final_an, truth)."""
    root = os.path.join(T, "round1")
    leaves, truth = {}, {}
    rng = np.random.default_rng(11)
    spec = [(3, f"q{i}_1", "rl", "ml_exact", 5000, "0:5000:1", 0.055) for i in range(1, 10)] + \
           [(5, p, "rl_s5", "ml_tau2:1e-6/1e-8:0.1", 10000, "0:10000:5", 0.03) for p in ("q2_7", "q4_7", "q6_5", "q8_7")]
    excl = []
    for d, patch, tag, mlname, N, sel, pml in spec:
        for b in ("X", "Z"):
            rel = f"{EXP}/d{d}_at_{patch}/{b}/r10"
            st, sp, stp = (int(x) for x in sel.split(":"))
            idx = np.arange(st, sp, stp); n = len(idx)
            obs = (rng.random(n) < 0.5).astype(np.uint8)
            ml = obs ^ (rng.random(n) < pml)
            preds = {"ML.rl": ml.astype(np.uint8)}
            for e, (p, q) in RATES[d].items():
                u = rng.random(n)
                preds[e] = (ml ^ np.where(ml != obs, u < q, u < p)).astype(np.uint8)
            fd = os.path.join(root, "decode", rel, tag); os.makedirs(fd)
            mdn = mlname.replace(":", "_").replace("/", "-")
            np.savez_compressed(os.path.join(fd, mdn + ".npz"), idx=idx, pred=preds["ML.rl"][:, None],
                                llr=np.where(preds["ML.rl"] == 1, 5.0, -5.0).astype(np.float32), logp=np.zeros(n))
            for bb in ("corr_mwpm", "mwpm"):
                np.savez_compressed(os.path.join(fd, bb + ".npz"), idx=idx, pred=preds[bb + ".rl"][:, None])
            np.savez_compressed(os.path.join(fd, "shots.npz"), idx=idx, obs=obs, shipped_names=np.array(SHIPPED),
                                shipped=np.stack([preds[e] for e in SH_ENT]))
            dec = {mlname: {}, "corr_mwpm": {}, "mwpm": {}}
            dec.update({"shipped:" + s: {} for s in SHIPPED})
            json.dump({"status": "complete", "n_total": N, "selection": sel, "rounds": 10, "decoders": dec},
                      open(os.path.join(fd, "summary.json"), "w"))
            ents = {e: {"n": n, "errors": int((p_ != obs).sum()), "LER": float((p_ != obs).mean()),
                        "eps_1pt": eps(float((p_ != obs).mean()))} for e, p_ in preds.items()}
            if d == 5 and patch == "q8_7" and b == "Z":
                excl.append(rel); ents.pop("ML.rl")
            leaves[rel] = {"d": d, "patch": patch, "basis": b, "rounds": 10, "folders": [tag], "entities": ents}
            truth[rel] = {"obs": obs, "preds": preds, "d": d, "basis": b, "patch": patch}
    # a leaf at another round count, which must be ignored
    leaves[f"{EXP}/d3_at_q1_1/X/r30"] = {"d": 3, "patch": "q1_1", "basis": "X", "rounds": 30, "folders": ["rl"], "entities": {}}
    excl.append(f"{EXP}/d5_at_q4_7/X/r50")
    fan = os.path.join(T, "analysis.json"); json.dump({"leaves": leaves}, open(fan, "w"))
    fr1 = os.path.join(T, "final_round1.json"); json.dump({"excluded_leaves": excl}, open(fr1, "w"))
    return root, fr1, fan, truth, excl


def d7_truth(res):
    """Errors and 2x2 tables per basis on the PR3 sample, read directly from the parts."""
    out = {}
    for b in ("X", "Z"):
        lf = C.leaf(b, 10)
        cols = {k: [] for k in ["idx", "obs", "ML.rl"] + SH_ENT + ["corr_mwpm.rl", "mwpm.rl"]}
        for j in range(5):
            dd = os.path.join(res, "parts", f"p{j}", "decode", lf, "rl_25k")
            z = np.load(os.path.join(dd, PRR.LADDER_DN + ".npz")); sh = np.load(os.path.join(dd, "shots.npz"))
            cols["idx"].append(sh["idx"]); cols["obs"].append(sh["obs"]); cols["ML.rl"].append(z["pred"][:, 0])
            for e, row in zip(SH_ENT, sh["shipped"]):
                cols[e].append(row)
            for bb in ("corr_mwpm", "mwpm"):
                cols[bb + ".rl"].append(np.load(os.path.join(dd, bb + ".npz"))["pred"][:, 0])
        c = {k: np.concatenate(v) for k, v in cols.items()}
        m = c["idx"] % 25 != 0
        out[b] = {k: v[m] for k, v in c.items()}
    return out


def g_direct(tabs):
    """tabs: list of (obs, A, ML) arrays -> g = mean(eA - eM) / mean(eM)."""
    eA = [eps(float((a_ != o).mean())) for o, a_, m_ in tabs]; eM = [eps(float((m_ != o).mean())) for o, a_, m_ in tabs]
    return (np.mean(eA) - np.mean(eM)) / np.mean(eM)


def boot_g(tabs, reps, seed):
    rng = np.random.default_rng(seed)
    gs = []
    for _ in range(reps):
        tt = []
        for o, a_, m_ in tabs:
            i = rng.integers(0, len(o), len(o))
            tt.append((o[i], a_[i], m_[i]))
        gs.append(g_direct(tt))
    return float(np.std(gs, ddof=1))


def run_analysis(T, a, root, fr1, fan, out, ledger=None):
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "-I", "-B", ASCRIPT, "--res", a.res, "--ledger", ledger or os.path.join(a.state, "ledger.json"),
                        "--round1", root, "--final-r1", fr1, "--final-analysis", fan, "--pr3-leaf-table", LT_REAL,
                        "--out", out], capture_output=True, text=True, env=env, cwd=T)
    return p


LTR = json.load(open(LT_REAL))
for scen, modes in (("A", {"X": "extra", "Z": "pass"}), ("B", {"X": "fail", "Z": "pass"}), ("C", {"X": "fail", "Z": "fail"})):
    print(f"---- scenario {scen}: {modes}")
    T = tempfile.mkdtemp(dir=SCR)
    code, a = run_driver(T, modes)
    Tl = json.load(open(os.path.join(a.res, "r10_tally.json")))
    check(code == 0 and Tl["complete"], f"{scen}: driver exit {code}, tally complete {Tl['complete']}, still failing "
          f"{[x.split('/', 2)[-1] for x in Tl['still_failing']]}")
    root, fr1, fan, truth, excl = make_round1(T)
    out = os.path.join(T, "out")
    p = run_analysis(T, a, root, fr1, fan, out)
    check(p.returncode == 0, f"{scen}: analysis exit {p.returncode} {p.stderr.strip()[-300:]}")
    o = json.load(open(os.path.join(out, "analysis", "r10_results.json")))
    md = open(os.path.join(out, "analysis", "r10_results.md")).read()
    check(o["all_checks_ok"], f"{scen}: all {len(o['checks'])} checks of the analysis pass "
          f"{[c['what'] for c in o['checks'] if not c['ok']]}")
    want_pass = [b for b in ("X", "Z") if modes[b] != "fail"]
    check(o["passing_bases"] == want_pass, f"{scen}: passing bases {o['passing_bases']} (want {want_pass})")
    for b in ("X", "Z"):
        lf = C.leaf(b, 10); row = o["leaves"][lf]
        want = {"pass": "pass (ladder)", "extra": "pass (extra rung)", "fail": "still fails: excluded and flagged"}[modes[b]]
        check(row["verdict"] == want and len(row["extra_parts_done"]) == (0 if modes[b] == "pass" else 5),
              f"{scen}: {b}/r10 row: {row['verdict']}, extra parts {row['extra_parts_done']}")
    # leaf table
    LT = json.load(open(os.path.join(out, "leaf_table", "pr3_leaf_table_with_r10.json")))
    same_other = all({k: v for k, v in LT["leaves"][lf].items() if k != "source"} == LTR["leaves"][lf]
                     for lf in LTR["leaves"] if not lf.endswith("/r10"))
    check(len(LT["leaves"]) == 6 and same_other and all(LT["leaves"][C.leaf(b, 10)]["source"].startswith("post hoc") for b in "XZ"),
          f"{scen}: combined leaf table: 6 rows, PR3's 4 rows unchanged, 2 post hoc rows")
    lmd = open(os.path.join(out, "leaf_table", "pr3_leaf_table_with_r10.md")).read()
    check(lmd.count("| post hoc |") == 2 and lmd.count("| PR3 |") == 4, f"{scen}: leaf table md has 2 post hoc and 4 PR3 rows")
    if not want_pass:
        check("gaps" not in o and "No r = 10 leaf passes" in md, f"{scen}: no gap numbers; the md says so")
        continue
    # direct computations
    t7 = d7_truth(a.res)
    for b in want_pass:
        for e in A.ENTS:
            k = int((t7[b][e] != t7[b]["obs"]).sum())
            if o["d7_counts"][b]["errors"][e] != k:
                check(False, f"{scen}: {b} {e} errors {o['d7_counts'][b]['errors'][e]} vs direct {k}")
    check(True, f"{scen}: d = 7 error counts equal direct counts ({len(want_pass)} bases x {len(A.ENTS)} decoders)")
    nser = {3: 9 * len(want_pass), 5: (3 if want_pass == ["Z"] else (4 if want_pass == ["X"] else 7)), 7: len(want_pass)}
    worst = 0.0
    for e in A.ENTS:
        G = o["gaps"][e]
        for d in (3, 5):
            tabs = [(t["obs"], t["preds"][e], t["preds"]["ML.rl"]) for rel, t in sorted(truth.items())
                    if t["d"] == d and t["basis"] in want_pass and rel not in excl]
            g = g_direct(tabs)
            worst = max(worst, abs(G[str(d)]["g"] - g))
            check(G[str(d)]["n_series"] == nser[d] == len(tabs), f"{scen}: {e} d = {d}: {G[str(d)]['n_series']} series (want {nser[d]})") \
                if e == "libra.rl" else None
        tabs7 = [(t7[b]["obs"], t7[b][e], t7[b]["ML.rl"]) for b in want_pass]
        worst = max(worst, abs(G["7"]["g"] - g_direct(tabs7)))
        gr = o["growth"][e]
        r57 = G["7"]["g"] / G["5"]["g"]
        check(abs(gr["g7/g5"]["ratio"] - r57) < 1e-12 and (gr["g7/g5"]["ci95"] is None or gr["g7/g5"]["ci95"][0] <= r57 <= gr["g7/g5"]["ci95"][1]),
              f"{scen}: {e} g7/g5 = {r57:.3f}, CI {gr['g7/g5']['ci95']}") if e == "libra.rl" else None
    check(worst < 1e-12, f"{scen}: every g equals the direct mean(eps_A - eps_ML) / mean(eps_ML) (largest diff {worst:.1e})")
    # SE against a bootstrap made here
    e = "libra.rl"
    tabs3 = [(t["obs"], t["preds"][e], t["preds"]["ML.rl"]) for rel, t in sorted(truth.items())
             if t["d"] == 3 and t["basis"] in want_pass and rel not in excl]
    sd3 = boot_g(tabs3, 300, 5); se3 = o["gaps"][e]["3"]["se"]
    check(0.8 < sd3 / se3 < 1.25, f"{scen}: Libra g_3: delta-method SE {se3:.4f}, bootstrap here {sd3:.4f}")
    tabs7 = [(t7[b]["obs"], t7[b][e], t7[b]["ML.rl"]) for b in want_pass]
    sd7 = boot_g(tabs7, 600, 6); se7 = o["gaps"][e]["7"]["se"]
    bs = o["bootstrap_d7"][e]
    check(0.8 < sd7 / se7 < 1.25 and 0.8 < bs["sd_g"] / se7 < 1.25,
          f"{scen}: Libra g_7: delta-method SE {se7:.4f}, bootstrap here {sd7:.4f}, the script's bootstrap {bs['sd_g']:.4f}")
    # eps and Lambda
    for e in ("ML.rl", "libra.rl"):
        e7 = np.mean([eps(float((t7[b][e] != t7[b]["obs"]).mean())) for b in want_pass])
        e5 = np.mean([eps(float((t["preds"][e] != t["obs"]).mean())) for rel, t in truth.items()
                      if t["d"] == 5 and t["basis"] in want_pass and not (e == "ML.rl" and rel in excl)])
        L = o["lambda_5_7_r10"][e]
        n5 = len([1 for rel, t in truth.items() if t["d"] == 5 and t["basis"] in want_pass and not (e == "ML.rl" and rel in excl)])
        check(abs(o["eps_d7_r10"][e]["mean"] - e7) < 1e-15 and abs(L["Lambda_5_7"]["Lambda"] - e5 / e7) < 1e-9
              and len(L["d5_series"]) == n5,
              f"{scen}: {e}: eps_7 {e7:.4e}, Lambda_5/7 {L['Lambda_5_7']['Lambda']:.3f} = {e5 / e7:.3f}, d = 5 leaves {len(L['d5_series'])}")
    if len(want_pass) == 1:
        check(f"Only {want_pass[0]}/r10 passes" in md, f"{scen}: the md labels the one-basis result")
    # negative tests on scenario A only
    if scen == "A":
        bad = os.path.join(T, "res_bad"); shutil.copytree(a.res, bad)
        tb = json.load(open(os.path.join(bad, "r10_tally.json")))
        tb["verdicts"][C.leaf("Z", 10)]["ladder"]["errors_tau2"] += 1
        json.dump(tb, open(os.path.join(bad, "r10_tally.json"), "w"))
        a2 = types.SimpleNamespace(**dict(vars(a), res=bad))
        p2 = run_analysis(T, a2, root, fr1, fan, os.path.join(T, "out_bad"))
        o2 = json.load(open(os.path.join(T, "out_bad", "analysis", "r10_results.json")))
        check(p2.returncode == 1 and not o2["all_checks_ok"],
              f"A, tampered tally: exit {p2.returncode}; failed: {[c['what'] for c in o2['checks'] if not c['ok']]}")
        tb["complete"] = False
        json.dump(tb, open(os.path.join(bad, "r10_tally.json"), "w"))
        p3 = run_analysis(T, a2, root, fr1, fan, os.path.join(T, "out_bad2"))
        check(p3.returncode == 1 and "not complete" in p3.stderr and not os.path.exists(os.path.join(T, "out_bad2")),
              f"A, incomplete tally: refused (exit {p3.returncode}: {p3.stderr.strip()[-80:]}), nothing written")

# fieller() against a brute-force inversion
for a_, sa, b_, sb in ((0.10, 0.02, 0.05, 0.01), (0.07, 0.03, 0.02, 0.008), (-0.01, 0.02, 0.05, 0.02)):
    f = A.fieller(a_, sa, b_, sb)
    grid = np.linspace(-20, 20, 400001)
    inside = grid[(a_ - grid * b_) ** 2 <= 1.96 ** 2 * (sa ** 2 + grid ** 2 * sb ** 2)]
    check(f["ci95"] is not None and abs(f["ci95"][0] - inside.min()) < 2e-4 and abs(f["ci95"][1] - inside.max()) < 2e-4,
          f"fieller({a_}, {sa}, {b_}, {sb}) = {f['ratio']:.3f} {[round(x, 4) for x in f['ci95']]}; brute force "
          f"[{inside.min():.4f}, {inside.max():.4f}]")
f = A.fieller(0.1, 0.02, 0.01, 0.01)
check(f["ci95"] is None and "unbounded" in f["note"], "fieller with a denominator CI containing 0: unbounded")
print(f"{sum(ok)} of {len(ok)} checks pass")
sys.exit(0 if all(ok) else 1)
