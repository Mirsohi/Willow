"""Synthetic check of pr3_runner.Runner.judge(): verdicts on the PR3 sample from fake part outputs."""
import json, os, sys, types, hashlib, tempfile
sys.path.insert(0, "/home/claude/work/decoder/pr3")
import numpy as np
import pr3_common as C
import pr3_runner as PRR

rng = np.random.default_rng(1)
T = tempfile.mkdtemp(dir="/home/claude/work/decoder/pr3/tests")
pilot = {"measured_rates": {"c1_10": 43.0, "s_10": 0.005, "c2_10": 383.0, "c1_50": 51.3, "s_50": 0.02, "c2_50": 326.0,
                            "c3_50": 1035.0, "cb_50": 1.0, "ovh_extra_s": 4.0, "ovh_ladder_s": 0.0},
         "stage1_ladder": {"peak_rss": 5e8}, "stage2_extra": {"peak_rss": 3.5e8}}
json.dump(pilot, open(os.path.join(T, "pilot.json"), "w"))
a = types.SimpleNamespace(root="/x", res=os.path.join(T, "res"), project="/nonexistent", work=os.path.join(T, "w"),
                          state=os.path.join(T, "s"), logs=os.path.join(T, "l"), pilot=os.path.join(T, "pilot.json"),
                          design="25k", guard_core_hours=116.8, max_heavy=4, min_free_gib=3, cap_gib=4, poll=1, dry_run=True)
R = PRR.Runner(a)
lf = C.leaf("X", 10); r = 10
truth = {"k1": 0, "kf": 0, "ch": 0, "k3": 0, "ch3": 0}
allidx = []
for j in range(C.PARTS):
    sl = C.part_slice("25k", j); st, sp, stp = (int(x) for x in sl.split(":"))
    idx = np.arange(st, sp, stp).astype(np.int32); n = len(idx)
    obs = rng.integers(0, 2, n).astype(np.uint8)
    llr1 = rng.normal(0, 3, n).astype(np.float32)
    red = np.abs(1 / (1 + np.exp(-llr1)) - 0.5) < 0.1
    llr = llr1.copy(); llr[red] = llr1[red] + rng.normal(0, 0.5, red.sum()).astype(np.float32)
    pred = (llr > 0).astype(np.uint8)[:, None]
    d = R.ladder_dir(lf, j); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, PRR.LADDER_DN + ".npz"), idx=idx, pred=pred, llr=llr, llr_t1=llr1, redecoded=red)
    np.savez_compressed(os.path.join(d, "shots.npz"), idx=idx, obs=obs)
    json.dump({"status": "complete", "decoders": {C.LADDER: {}}, "selection": sl}, open(os.path.join(d, "summary.json"), "w"))
    b = np.where(red)[0]
    l3 = llr[b] + rng.normal(0, 0.3, len(b)).astype(np.float32)
    ed = os.path.dirname(R.extra_json(lf, j)); os.makedirs(ed, exist_ok=True)
    np.savez_compressed(os.path.join(ed, "extra_rung.npz"), idx=idx[b], llr_t3=l3)
    zmd5 = hashlib.md5(open(os.path.join(d, PRR.LADDER_DN + ".npz"), "rb").read()).hexdigest()
    json.dump({"tau3": {}, "ladder_npz_md5": zmd5}, open(R.extra_json(lf, j), "w"))
    m = idx % 25 != 0
    d1 = (llr1 > 0).astype(np.uint8); df = pred[:, 0]
    truth["k1"] += int(((d1 != obs) & m).sum()); truth["kf"] += int(((df != obs) & m).sum()); truth["ch"] += int(((d1 != df) & m).sum())
    d3 = (l3 > 0).astype(np.uint8); mb = m[b]
    truth["ch3"] += int(((d3 != df[b]) & mb).sum())
    truth["k3"] += int(((df != obs) & m).sum()) - int(((df[b] != obs[b]) & mb).sum()) + int(((d3 != obs[b]) & mb).sum())
    allidx.append(idx)
R.refresh()
v = R.verdicts[lf]
print("sample", v["n_sample"], "decoded", v["n_decoded"], "expected sample", sum(int((i % 25 != 0).sum()) for i in allidx))
print("ladder", v["ladder"]); print("truth k1 kf ch", truth["k1"], truth["kf"], truth["ch"])
assert v["ladder"]["errors_tau1"] == truth["k1"] and v["ladder"]["errors_tau2"] == truth["kf"] and v["ladder"]["decisions_changed"] == truth["ch"]
print("extra", v.get("extra")); print("truth k3 ch3", truth["k3"], truth["ch3"])
assert v["extra"]["errors_tau3"] == truth["k3"] and v["extra"]["decisions_changed"] == truth["ch3"]
print("verdict", v["verdict"])
print("other leaves", {k.split('/')[-2] + '/' + k.split('/')[-1]: x["verdict"] for k, x in R.verdicts.items() if k != lf})
print("projection", R.projection())
print("OK")
