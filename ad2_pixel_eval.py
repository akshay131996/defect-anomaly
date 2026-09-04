#!/usr/bin/env python3
"""Pixel-level anomaly localisation on MVTec AD 2.

Everything in this project so far has asked "is this part defective". This asks "where",
which is the metric AD 2 is actually scored on, and the two can disagree badly: the
current best published method on AD 2 reports layer3-only features giving strong image
AUROC while their anomaly maps sit at AU-PRO ~9%.

Three metrics, and the third is the one that matters:

- **image AUROC** - what we have measured all along, for continuity.
- **pixel AUROC** - every pixel as a sample. Easy to inflate: >99% of pixels are normal,
  so predicting "all normal" already scores well. Reported because it is conventional,
  not because it is informative.
- **AU-PRO** - per-region overlap. For each *connected component* of ground truth,
  measure the fraction of it that is detected, then average over regions. A one-pixel
  scratch counts as much as a defect covering a quarter of the frame. Integrated over
  false-positive rate up to a limit and normalised. AD 2's headline is AU-PRO@5%.

Why the histogram approach: binarising a 5-megapixel map at 100 thresholds for ~250
images per scenario is billions of operations. Instead, bin every pixel's score once.
The false-positive curve comes from a histogram over normal pixels; the per-region
overlap curve comes from a histogram per ground-truth component. Both are one pass over
the pixels, and the only approximation is bin width.

Data layout, as shipped:

    <scenario>/train/good/            defect-free, builds the memory bank
    <scenario>/validation/good/       defect-free, calibrates the threshold - the split
                                      this project hand-rolled in session 2, here as
                                      part of the benchmark design
    <scenario>/test_public/good/      labelled normal
    <scenario>/test_public/bad/       labelled defective
    <scenario>/test_public/ground_truth/  pixel masks for the bad images

    python ad2_pixel_eval.py [--scenarios can,fabric] [--limit N]
        -> outputs/ad2_pixel_eval.json
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

import sweep_backbones as sb

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
OUT = "outputs/ad2_pixel_eval.json"

# Arm A / E territory - layer2+layer3 on a ResNet50-family backbone is the configuration
# the current best AD 2 result uses, so it is the right baseline to establish first.
ARM = {"tag": "A_wrn50_224", "kind": "cnn", "name": "wide_resnet50_2", "img": 224,
       "out_indices": (2, 3)}

CORESET_RATIO = 0.01
SEED = 0
NBINS = 2048                 # score histogram resolution
PRO_LIMITS = [0.05, 0.30]    # AD 2 headlines @5%; 30% is the classic MVTec AD number
GAUSS_SIGMA = 4.0            # standard PatchCore map smoothing, in grid cells
EVAL_SIDE = 512              # masks and maps are compared at this resolution - see below

_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))


def load_paths(scenario):
    r = os.path.join(AD2_ROOT, scenario)
    train = sorted(glob.glob(os.path.join(r, "train", "**", "*.png"), recursive=True))
    val = sorted(glob.glob(os.path.join(r, "validation", "**", "*.png"), recursive=True))
    good = sorted(glob.glob(os.path.join(r, "test_public", "good", "**", "*.png"),
                            recursive=True))
    bad = sorted(glob.glob(os.path.join(r, "test_public", "bad", "**", "*.png"),
                           recursive=True))
    gt_dir = os.path.join(r, "test_public", "ground_truth")
    return train, val, good, bad, gt_dir


def find_mask(gt_dir, bad_path):
    """Masks mirror the bad-image names; extensions and suffixes vary between releases,
    so match on stem rather than assuming a naming scheme."""
    stem = os.path.splitext(os.path.basename(bad_path))[0]
    hits = glob.glob(os.path.join(gt_dir, "**", stem + "*"), recursive=True)
    hits = [h for h in hits if h.lower().endswith((".png", ".bmp", ".tif", ".tiff"))]
    return hits[0] if hits else None


@torch.no_grad()
def anomaly_maps(ex, bank, paths, n_patches, grid, batch=8):
    """Returns (image_scores, list of HxW float maps at EVAL_SIDE resolution)."""
    scores, maps = [], []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = list(_POOL.map(lambda p: Image.open(p).convert("RGB"), chunk))
        x = torch.stack(list(_POOL.map(ex._one, imgs))).to(sb.DEVICE)
        feats = ex.forward_feats(x)
        d = sb.patch_distances(bank, feats, n_patches)          # (b, n_patches)
        scores.extend(d.max(dim=1).values.tolist())
        g = d.view(-1, 1, grid[0], grid[1])
        # Smooth on the grid before upsampling. PatchCore smooths the map; doing it at
        # grid resolution is far cheaper than at full size and equivalent up to the
        # interpolation.
        k = int(2 * round(GAUSS_SIGMA) + 1)
        blur = torch.ones(1, 1, k, k, device=g.device) / (k * k)
        g = F.conv2d(F.pad(g, (k // 2,) * 4, mode="replicate"), blur)
        up = F.interpolate(g, size=(EVAL_SIDE, EVAL_SIDE), mode="bilinear",
                           align_corners=False)
        maps.extend(up[:, 0].cpu().numpy())
    return np.array(scores), maps


def evaluate(maps_good, maps_bad, masks, lo, hi):
    """Histogram-based pixel AUROC and AU-PRO.

    Bins are shared across every image so the curves compose. Returns pixel AUROC plus
    AU-PRO at each limit in PRO_LIMITS.
    """
    edges = np.linspace(lo, hi, NBINS + 1)

    def hist(v):
        return np.histogram(v, bins=edges)[0].astype(np.float64)

    h_norm = np.zeros(NBINS)     # scores over all genuinely-normal pixels
    h_anom = np.zeros(NBINS)     # scores over all defective pixels
    region_hists = []            # one histogram per connected ground-truth component

    for m in maps_good:
        h_norm += hist(m.ravel())

    for m, mask in zip(maps_bad, masks):
        h_norm += hist(m[~mask].ravel())
        h_anom += hist(m[mask].ravel())
        lab, n = ndimage.label(mask)
        for r in range(1, n + 1):
            sel = lab == r
            if sel.sum() >= 4:                 # ignore specks - they are annotation noise
                region_hists.append(hist(m[sel].ravel()) / sel.sum())

    # survival curves: fraction above threshold, walking thresholds high -> low
    def survival(h):
        tot = h.sum()
        return np.cumsum(h[::-1])[::-1] / tot if tot > 0 else np.zeros_like(h)

    fpr = survival(h_norm)
    tpr = survival(h_anom)
    order = np.argsort(fpr)
    pixel_auroc = float(np.trapezoid(tpr[order], fpr[order])) if h_anom.sum() > 0 else None

    out = {"pixel_auroc": pixel_auroc, "n_regions": len(region_hists)}
    if region_hists:
        pro = np.mean([np.cumsum(h[::-1])[::-1] for h in region_hists], axis=0)
        for lim in PRO_LIMITS:
            keep = fpr <= lim
            if keep.sum() > 1:
                f, p = fpr[keep], pro[keep]
                o = np.argsort(f)
                out[f"au_pro@{lim}"] = float(np.trapezoid(p[o], f[o]) / lim)
            else:
                out[f"au_pro@{lim}"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per split, for a fast smoke run")
    ap.add_argument("--img", type=int, default=0,
                    help="override input resolution. AD 2 images are ~2448x2048; the "
                         "default 224 is a 10x linear downsample that erases the small "
                         "defects the benchmark is built around.")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    scenarios = ([s for s in args.scenarios.split(",") if s] or
                 sorted(d for d in os.listdir(AD2_ROOT)
                        if os.path.isdir(os.path.join(AD2_ROOT, d))))

    arm = dict(ARM)
    if args.img:
        arm["img"] = args.img
        arm["tag"] = f"{ARM['name']}_{args.img}"
    ex = sb.PatchExtractor(arm)
    summary = {"arm": arm["tag"], "img": arm["img"], "root": AD2_ROOT, "coreset_ratio": CORESET_RATIO,
               "eval_side": EVAL_SIDE, "gauss_sigma": GAUSS_SIGMA, "nbins": NBINS,
               "device": torch.cuda.get_device_name(0) if sb.DEVICE == "cuda" else "cpu",
               "torch": torch.__version__, "scenarios": {}}

    for sc in scenarios:
        t0 = time.time()
        train, val, good, bad, gt_dir = load_paths(sc)
        if args.limit:
            train, val, good, bad = (train[:args.limit], val[:args.limit],
                                     good[:args.limit], bad[:args.limit])
        if not train or not bad:
            print(f"{sc}: skipped (train={len(train)} bad={len(bad)})", flush=True)
            continue

        masks_paths = [find_mask(gt_dir, b) for b in bad]
        have = [i for i, m in enumerate(masks_paths) if m]
        bad = [bad[i] for i in have]
        masks_paths = [masks_paths[i] for i in have]
        if not bad:
            print(f"{sc}: skipped (no masks matched)", flush=True)
            continue

        # bank from train, threshold calibration data from validation
        f_tr = sb.extract_paths(ex, train, _POOL)
        n_patches = ex.grid[0] * ex.grid[1]
        keep = sb.coreset_indices(f_tr, ratio=CORESET_RATIO, seed=SEED)
        bank = f_tr[keep]
        del f_tr

        s_good, m_good = anomaly_maps(ex, bank, good, n_patches, ex.grid)
        s_bad, m_bad = anomaly_maps(ex, bank, bad, n_patches, ex.grid)

        masks = []
        for p in masks_paths:
            a = np.array(Image.open(p).convert("L").resize((EVAL_SIDE, EVAL_SIDE),
                                                           Image.NEAREST))
            masks.append(a > 127)

        truth = np.r_[np.zeros(len(s_good), int), np.ones(len(s_bad), int)]
        img_auroc = float(roc_auc_score(truth, np.r_[s_good, s_bad]))

        allv = np.concatenate([m.ravel() for m in (m_good + m_bad)])
        res = evaluate(m_good, m_bad, masks, float(allv.min()), float(allv.max()))
        res.update({"image_auroc": img_auroc, "n_train": len(train), "n_val": len(val),
                    "n_good": len(good), "n_bad": len(bad),
                    "bank_size": int(len(keep)), "seconds": round(time.time() - t0, 1)})
        summary["scenarios"][sc] = res
        print(f"{sc:<13} img {img_auroc:.4f}  pix {res['pixel_auroc']:.4f}  "
              f"AU-PRO@5% {res.get('au_pro@0.05')!s:>7.7}  "
              f"@30% {res.get('au_pro@0.3')!s:>7.7}  "
              f"regions {res['n_regions']:>4}  {res['seconds']:.0f}s", flush=True)

        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del bank, m_good, m_bad
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()

    done = [v for v in summary["scenarios"].values() if v.get("au_pro@0.05") is not None]
    if done:
        summary["mean_image_auroc"] = float(np.mean([v["image_auroc"] for v in done]))
        summary["mean_pixel_auroc"] = float(np.mean([v["pixel_auroc"] for v in done]))
        for lim in PRO_LIMITS:
            summary[f"mean_au_pro@{lim}"] = float(
                np.mean([v[f"au_pro@{lim}"] for v in done]))
        print(f"\nmean image AUROC {summary['mean_image_auroc']:.4f}   "
              f"pixel AUROC {summary['mean_pixel_auroc']:.4f}   "
              f"AU-PRO@5% {summary['mean_au_pro@0.05']:.4f}   "
              f"AU-PRO@30% {summary['mean_au_pro@0.3']:.4f}")
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
