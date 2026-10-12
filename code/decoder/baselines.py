"""Baseline decoders on a stim DetectorErrorModel. Each returns predicted observable flips (nshots, nobs) uint8.
  mwpm        : PyMatching 2 (sparse blossom) on the decomposed DEM
  corr_mwpm   : PyMatching 2 correlated matching (enable_correlations=True)
  beliefmatch : BeliefMatching (BP on the hypergraph + MWPM with BP-updated weights)
  bposd       : ldpc BP+OSD (osd_cs, order 7 by default) on the undecomposed DEM check matrix
  tesseract   : Google's Tesseract A* search decoder
"""
import numpy as np
import stim


def _dem_decomposed(dem):
    return dem  # caller passes a decomposed DEM when needed


class MWPM:
    name = "mwpm"
    def __init__(self, dem_decomposed: stim.DetectorErrorModel):
        import pymatching
        self.m = pymatching.Matching.from_detector_error_model(dem_decomposed)
    def decode(self, det):
        return self.m.decode_batch(det).astype(np.uint8)


class CorrMWPM:
    name = "corr_mwpm"
    def __init__(self, dem_decomposed: stim.DetectorErrorModel):
        import pymatching
        self.m = pymatching.Matching.from_detector_error_model(dem_decomposed, enable_correlations=True)
    def decode(self, det):
        return self.m.decode_batch(det, enable_correlations=True).astype(np.uint8)


class BeliefMatch:
    name = "beliefmatch"
    def __init__(self, dem_decomposed: stim.DetectorErrorModel, max_bp_iters=20):
        import beliefmatching
        self.bm = beliefmatching.BeliefMatching(dem_decomposed, max_bp_iters=max_bp_iters)
    def decode(self, det):
        return self.bm.decode_batch(det).astype(np.uint8)


class BPOSD:
    name = "bposd"
    def __init__(self, dem: stim.DetectorErrorModel, osd_order=7, max_iter=100):
        from ldpc import BpOsdDecoder
        from dem_utils import parse_dem
        dm = parse_dem(dem)
        H, L = dm.check_matrix()
        self.L = L.tocsr()
        self.dec = BpOsdDecoder(H.tocsr(), error_channel=list(dm.p), max_iter=max_iter,
                                bp_method="product_sum", osd_method="osd_cs", osd_order=osd_order)
        self.und = dm.undetectable
    def decode(self, det):
        out = np.zeros((det.shape[0], self.L.shape[0]), dtype=np.uint8)
        for i in range(det.shape[0]):
            e = self.dec.decode(det[i].astype(np.uint8))
            out[i] = (self.L @ e) % 2
        return out


class Tesseract:
    name = "tesseract"
    def __init__(self, dem: stim.DetectorErrorModel, det_beam=20, beam_climbing=True, pqlimit=1_000_000,
                 num_det_orders=20, no_revisit_dets=True):
        from tesseract_decoder import tesseract
        cfg = tesseract.TesseractConfig(dem=dem, det_beam=det_beam, beam_climbing=beam_climbing,
                                        pqlimit=pqlimit, no_revisit_dets=no_revisit_dets,
                                        num_det_orders=num_det_orders)
        self.dec = cfg.compile_decoder()
    def decode(self, det):
        return np.asarray(self.dec.decode_batch(det.astype(bool))).astype(np.uint8).reshape(det.shape[0], -1)


def make(name, dem_undecomposed, dem_decomposed, **kw):
    if name == "mwpm":
        return MWPM(dem_decomposed)
    if name == "corr_mwpm":
        return CorrMWPM(dem_decomposed)
    if name == "beliefmatch":
        return BeliefMatch(dem_decomposed, **kw)
    if name == "bposd":
        return BPOSD(dem_undecomposed, **kw)
    if name == "tesseract":
        return Tesseract(dem_undecomposed, **kw)
    raise ValueError(name)
