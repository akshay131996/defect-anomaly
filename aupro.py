#!/usr/bin/env python3
"""Pixel-level localisation metrics: pixel AUROC and AU-PRO.

Deliberately depends on nothing but numpy and scipy - no torch, no timm, no dataset.
A metric that can only run behind a GPU pipeline cannot be unit-tested, and this one
produced a number 16x below a published baseline on its first outing, which is exactly
when you want to test it in isolation. See test_aupro.py.

**AU-PRO** (per-region overlap) is the metric MVTec scores on, and it is not pixel AUROC.
For each *connected component* of the ground-truth mask, measure the fraction of that
component detected at a given threshold, then average across components. A one-pixel
scratch counts the same as a defect covering a quarter of the frame - which is the point,
because pixel AUROC is dominated by whichever defects happen to be largest. The curve is
integrated over false-positive rate up to a limit and normalised by that limit.

Implementation note: the obvious approach binarises the map at every threshold, which for
5-megapixel images across hundreds of samples is billions of operations. Instead each
pixel's score is binned once. The false-positive curve is a histogram over normal pixels;
the per-region curve is a histogram per component. One pass, and the only approximation
is bin width.
"""
import numpy as np
from scipy import ndimage

NBINS = 2048
PRO_LIMITS = [0.05, 0.30]
MIN_REGION_PX = 4          # smaller components are annotation noise, not defects


def _survival(hist):
    """Fraction of mass at or above each bin, walking thresholds high to low."""
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return np.cumsum(hist[::-1])[::-1] / total


def evaluate(maps_good, maps_bad, masks, lo, hi, nbins=NBINS, pro_limits=PRO_LIMITS, valid=None, region_labels=None):
    """Pixel AUROC and AU-PRO from anomaly maps and boolean ground-truth masks.

    maps_good     : anomaly maps for images with no defect
    maps_bad      : anomaly maps for defective images
    masks         : boolean arrays, same shapes as maps_bad, True where defective
    lo, hi        : score range spanned by the shared histogram bins
    valid         : optional boolean array or list of boolean arrays, True on genuine
                    image pixels, False on letterbox/padding regions.
    region_labels : optional list of (label_map, n_components) per bad image, derived
                    at native resolution to fix the region set across geometries (E4a).
    """
    edges = np.linspace(lo, hi, nbins + 1)

    def hist(v):
        return np.histogram(v, bins=edges)[0].astype(np.float64)

    h_norm = np.zeros(nbins)     # every genuinely-normal pixel
    h_anom = np.zeros(nbins)     # every defective pixel
    region_hists = []            # one normalised histogram per connected component

    for i, m in enumerate(maps_good):
        v = valid[i] if isinstance(valid, (list, tuple)) else valid
        if v is not None:
            h_norm += hist(m[v].ravel())
        else:
            h_norm += hist(m.ravel())

    for i, (m, mask) in enumerate(zip(maps_bad, masks)):
        v = valid[i] if isinstance(valid, (list, tuple)) else valid
        if v is not None:
            assert not (mask & ~v).any(), "Mask contains defects outside valid region"
            if mask.any():
                h_norm += hist(m[v & ~mask].ravel())
                h_anom += hist(m[v & mask].ravel())
            else:
                h_norm += hist(m[v].ravel())
                continue
        else:
            if mask.any():
                h_norm += hist(m[~mask].ravel())
                h_anom += hist(m[mask].ravel())
            else:
                h_norm += hist(m.ravel())
                continue

        if region_labels is not None:
            lab_map, n_comp = region_labels[i]
            for r in range(1, n_comp + 1):
                sel = (lab_map == r)
                npx = int(sel.sum())
                if npx > 0:
                    region_hists.append(hist(m[sel].ravel()) / npx)
                else:
                    # Region was erased by downsampling -> 0 detected pixels -> 0 TPR
                    region_hists.append(np.zeros(nbins))
        else:
            labelled, n = ndimage.label(mask)
            for r in range(1, n + 1):
                sel = labelled == r
                npx = int(sel.sum())
                if npx >= MIN_REGION_PX:
                    region_hists.append(hist(m[sel].ravel()) / npx)

    fpr = _survival(h_norm)
    tpr = _survival(h_anom)

    out = {"n_regions": len(region_hists), "pixel_auroc": None}
    if h_anom.sum() > 0:
        order = np.argsort(fpr)
        out["pixel_auroc"] = float(np.trapezoid(tpr[order], fpr[order]))

    if region_hists:
        # PRO(t): mean over regions of the fraction of that region above threshold t.
        pro = np.mean([_survival(h * h.size) if False else np.cumsum(h[::-1])[::-1]
                       for h in region_hists], axis=0)
        # Integrate over the FULL [0, lim] range, not just the sampled points inside it.
        #
        # A good detector produces a near-vertical ROC: FPR jumps from 0 to ~1 across one
        # bin, so *no* sample lands strictly between 0 and lim. Integrating only over
        # sampled points then stops short and silently under-reports - a perfect detector
        # scored 0.879 instead of 1.0 in test_aupro.py. Interpolating onto a dense grid
        # over [0, lim] fixes it and matches the definition, which treats PRO as a
        # function of FPR rather than as a point set.
        o = np.argsort(fpr)
        f_sorted, p_sorted = fpr[o], pro[o]
        for lim in pro_limits:
            grid = np.linspace(0.0, lim, 512)
            p_interp = np.interp(grid, f_sorted, p_sorted,
                                 left=float(p_sorted[0]), right=float(p_sorted[-1]))
            out[f"au_pro@{lim}"] = float(np.trapezoid(p_interp, grid) / lim)
    else:
        for lim in pro_limits:
            out[f"au_pro@{lim}"] = None
    return out
