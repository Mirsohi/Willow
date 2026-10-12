"""Detector-error-model (DEM) utilities for ML decoding.

parse_dem(dem) -> DemMatrices with
  dets[j]   : sorted tuple of detector indices flipped by fault j
  obs[j]    : bitmask of observables flipped by fault j
  p[j]      : probability of fault j (independent faults)
Faults with identical (dets, obs) symptoms are merged: p = p1 + p2 - 2 p1 p2.
Decomposition separators ('^') are removed by XOR-ing the components, so the full symptom of each
fault is kept (exact for ML). Faults with empty symptom are dropped; faults with no detectors but a
logical flip are kept separately in `undetectable` (they act as a syndrome-independent flip).
"""
from dataclasses import dataclass, field
import numpy as np
import stim


@dataclass
class DemMatrices:
    num_detectors: int
    num_observables: int
    dets: list            # list of tuples of detector ids
    obs: np.ndarray       # int64 bitmask per fault
    p: np.ndarray         # float64 per fault
    coords: dict          # detector id -> tuple of coords
    undetectable: list = field(default_factory=list)  # list of (obsmask, p)

    @property
    def num_faults(self):
        return len(self.dets)

    def check_matrix(self):
        import scipy.sparse as sp
        rows, cols = [], []
        for j, ds in enumerate(self.dets):
            rows.extend(ds); cols.extend([j] * len(ds))
        H = sp.csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                          shape=(self.num_detectors, self.num_faults))
        Lr, Lc = [], []
        for j, o in enumerate(self.obs):
            for k in range(self.num_observables):
                if (int(o) >> k) & 1:
                    Lr.append(k); Lc.append(j)
        L = sp.csc_matrix((np.ones(len(Lr), dtype=np.uint8), (Lr, Lc)),
                          shape=(self.num_observables, self.num_faults))
        return H, L


def parse_dem(dem: stim.DetectorErrorModel, merge: bool = True) -> DemMatrices:
    dem_flat = dem.flattened()
    symptoms = {}
    order = []
    for inst in dem_flat:
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dset = set()
        omask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                v = t.val
                if v in dset:
                    dset.remove(v)
                else:
                    dset.add(v)
            elif t.is_logical_observable_id():
                omask ^= (1 << t.val)
            elif t.is_separator():
                continue
        key = (tuple(sorted(dset)), omask)
        if key == ((), 0) or prob == 0:
            continue
        if merge and key in symptoms:
            q = symptoms[key]
            symptoms[key] = q + prob - 2 * q * prob
        elif merge:
            symptoms[key] = prob
            order.append(key)
        else:
            order.append((key, prob))
    if merge:
        items = [(k, symptoms[k]) for k in order]
    else:
        items = order
    dets, obs, ps, und = [], [], [], []
    for (ds, om), pr in items:
        if len(ds) == 0:
            und.append((om, pr))
            continue
        dets.append(ds); obs.append(om); ps.append(pr)
    coords = dem.get_detector_coordinates()
    return DemMatrices(dem.num_detectors, dem.num_observables, dets,
                       np.array(obs, dtype=np.int64), np.array(ps, dtype=np.float64),
                       coords, und)


def restrict_dem(dm: DemMatrices, keep_detectors) -> DemMatrices:
    """Marginalise out detectors not in keep_detectors (delete them from every symptom, re-merge).
    Returns a DEM on the kept detectors, renumbered 0..K-1 in the given order."""
    keep = list(keep_detectors)
    remap = {d: i for i, d in enumerate(keep)}
    sym = {}
    order = []
    und = list(dm.undetectable)
    for ds, om, pr in zip(dm.dets, dm.obs, dm.p):
        nds = tuple(sorted(remap[d] for d in ds if d in remap))
        key = (nds, int(om))
        if key == ((), 0):
            continue
        if key in sym:
            q = sym[key]; sym[key] = q + pr - 2 * q * pr
        else:
            sym[key] = pr; order.append(key)
    dets, obs, ps = [], [], []
    for (ds, om) in order:
        if len(ds) == 0:
            und.append((om, sym[(ds, om)])); continue
        dets.append(ds); obs.append(om); ps.append(sym[(ds, om)])
    coords = {remap[d]: dm.coords[d] for d in keep if d in dm.coords}
    return DemMatrices(len(keep), dm.num_observables, dets, np.array(obs, dtype=np.int64),
                       np.array(ps), coords, und)


SWEEP_NAMES = ("xy", "yx", "plain", "diag1", "diag2", "diag3", "diag4")


def detector_time_space_keys(dm: DemMatrices, spatial="xy"):
    """Return a sort key per detector: (t, spatial position).
    Requires coords (x, y, t, ...): t is the third coordinate, both in stim's generated circuits (x, y, t) and in
    the Willow files, which append the coordinates of the compared measurements (x, y, t, x', y', t', ...).
    Falls back to detector index.
    'xy', 'yx': snakes for stim's coordinates; 'plain': (t, x, y); 'diag1'-'diag4': sweeps along the 45-degree
    rotated axes u = x - y and v = x + y (Willow's grid is rotated against stim's)."""
    n = dm.num_detectors
    keys = []
    for i in range(n):
        c = dm.coords.get(i, None)
        if c is None or len(c) < 3:
            keys.append((i, 0.0, 0.0))
            continue
        x, y, t = c[0], c[1], c[2]
        if spatial == "xy":
            keys.append((t, x, y if int(round(x)) % 4 < 2 else -y))
        elif spatial == "yx":
            keys.append((t, y, x if int(round(y)) % 4 < 2 else -x))
        elif spatial == "diag1":
            keys.append((t, x - y, -(x + y)))
        elif spatial == "diag2":
            keys.append((t, x - y, x + y))
        elif spatial == "diag3":
            keys.append((t, x + y, x - y))
        elif spatial == "diag4":
            keys.append((t, x + y, y - x))
        else:
            keys.append((t, x, y))
    return keys


def fault_order(dm: DemMatrices, det_rank: np.ndarray):
    """Order faults by (max detector rank, min detector rank). det_rank: rank of each detector in sweep order."""
    keys = []
    for j, ds in enumerate(dm.dets):
        r = [det_rank[d] for d in ds]
        keys.append((max(r), min(r), j))
    keys.sort()
    return [k[2] for k in keys]


def frontier_profile(dm: DemMatrices, order):
    """Simulate the open-detector frontier for a fault order. Returns (max_open, mean_open, open_counts)."""
    first, last = {}, {}
    for pos, j in enumerate(order):
        for d in dm.dets[j]:
            first.setdefault(d, pos)
            last[d] = pos
    opens = np.zeros(len(order) + 1, dtype=np.int64)
    for d in first:
        opens[first[d]] += 1
        opens[last[d] + 1] -= 1
    cnt = np.cumsum(opens)[:-1]
    return int(cnt.max()), float(cnt.mean()), cnt


def best_sweep_order(dm: DemMatrices):
    """Try a few detector sweep orders and return (order, max_open, mean_open, name): the smallest (max_open,
    mean_open), the first candidate on ties. The order changes only the cost of an exact contraction (and the
    rounding), not its result."""
    best = None
    for name in SWEEP_NAMES:
        keys = detector_time_space_keys(dm, name)
        perm = sorted(range(dm.num_detectors), key=lambda i: keys[i])
        rank = np.empty(dm.num_detectors, dtype=np.int64)
        rank[perm] = np.arange(dm.num_detectors)
        order = fault_order(dm, rank)
        mx, mean, _ = frontier_profile(dm, order)
        if best is None or (mx, mean) < (best[1], best[2]):
            best = (order, mx, mean, name)
    return best
