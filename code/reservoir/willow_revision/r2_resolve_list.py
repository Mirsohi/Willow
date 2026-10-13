"""R2's resolve list (declared 2026-10-13T15:53Z in notes/reservoir/willow_revision_declarations.md, "Resolve"): every
d = 5 shot at r = 10, 13, 30 whose class under R2's max-product ladder (MPE_tau, mode 1) differs from the stored
Tesseract class. The list is sorted by (leaf, idx) and then put in one random order (numpy default_rng(20261013)); it
is the input of `r1_certify.py resolve`, which proves the MPE of each shot by integer programming in that order.
POST HOC revision analysis; reads no outcome (only the two decoders' predicted classes).

reservoir, 2026-10-13.
Usage: python3 -I -B r2_resolve_list.py DECODE R2_WORK LIST_OUT
  DECODE    results/decoder/willow/decode (the stored Tesseract: <ARCH>/<leaf>/rl_s5t/tesseract.npz)
  R2_WORK   r2_semiring.py's work folder (<leaf with / as _>/m1.npz)
  LIST_OUT  JSON list of {"leaf", "idx", "classes": {"mpe_tau": c, "tesseract": c}}; a summary is printed
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "willow_gapmech"))
from exact_run import BASES, P5, rel_of, stored_pred  # noqa: E402

SEED = 20261013
ROUNDS = (10, 13, 30)


def main(dec, work, out):
    items, per_leaf, missing = [], {}, []
    for r in ROUNDS:
        for p in P5:
            for b in BASES:
                rel = rel_of(5, p, b, r)
                f = os.path.join(work, rel.replace("/", "_"), "m1.npz")
                if not os.path.exists(f):
                    missing.append(rel)
                    continue
                z = np.load(f, allow_pickle=False)
                idx = z["idx"].astype(np.int64)
                mpe = z["pred"].reshape(idx.size, -1)[:, 0].astype(np.uint8)
                tes = stored_pred(dec, rel, "rl_s5t", "tesseract.npz", idx)
                k = np.flatnonzero(mpe != tes)
                per_leaf[rel] = {"shots": int(idx.size), "class_differs": int(k.size)}
                items += [{"leaf": rel, "idx": int(idx[j]), "classes": {"mpe_tau": int(mpe[j]),
                                                                        "tesseract": int(tes[j])}} for j in k]
    items.sort(key=lambda x: (x["leaf"], x["idx"]))
    perm = np.random.default_rng(SEED).permutation(len(items))
    items = [items[i] for i in perm]
    with open(out, "w") as fh:
        json.dump(items, fh)
    print(json.dumps({"items": len(items), "leaves": len(per_leaf), "missing": missing,
                      "per_round": {f"r{r}": sum(v["class_differs"] for k, v in per_leaf.items()
                                                 if k.endswith(f"/r{r:02d}")) for r in ROUNDS}}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
