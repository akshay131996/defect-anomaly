#!/usr/bin/env python3
"""E5a: Break down AU-PRO@5% by region-size bucket on MVTec AD 2 under aspect geometry (img=448).

Tests the hypothesis:
'a majority of regions are sub-cell at img = 448, and AU-PRO on the
larger-than-4-cells bucket is already close to the published 0.764.'
"""
import os
import sys
import glob
import json
import time
import gc
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor

import sweep_backbones as sb
from aupro import evaluate, NBINS, PRO_LIMITS

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
OUT = "outputs/exp_e5a_bucketed_pro.json"
SCENARIOS = [
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"
]
CORESET_RATIO = 0.01
SEED = 0
GAUSS_SIGMA = 4.0
EVAL_SIDE = 512
NATIVE_MIN_REGION_PX = 77
BANK_CAP = 4000
IMG = 448

_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))


def aspect_dimensions(w_nat, h_nat, target_img=448, stride=32):
    r = w_nat / h_nat
    target_area = target_img * target_img
    h_float = (target_area / r) ** 0.5
    w_float = r * h_float
    h_in = max(stride, int(round(h_float / stride) * stride))
    w_in = max(stride, int(round(w_float / stride) * stride))
    return w_in, h_in


def aspect_transform(ex, w_in, h_in):
    from torchvision import transforms as T
    norm = [t for t in ex.tfm.transforms if isinstance(t, T.Normalize)][0]
    ex.tfm = T.Compose([
        T.Resize((h_in, w_in), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        norm,
    ])
    return ex


def extract_paths_prealloc(ex, paths, pool, batch=8):
    from PIL import Image
    n_total = len(paths)
    sample_imgs = list(pool.map(lambda p: Image.open(p).convert("RGB"), paths[:batch]))
    sample_x = torch.stack(list(pool.map(ex._one, sample_imgs))).to(sb.DEVICE)
    with torch.no_grad():
        f0 = ex.forward_feats(sample_x)
    
    patches_per_img = f0.shape[0] // len(sample_imgs)
    dim = f0.shape[1]
    total_patches = n_total * patches_per_img
    
    feats = torch.empty((total_patches, dim), dtype=torch.float32)
    feats[:f0.shape[0]] = f0
    del sample_imgs, sample_x, f0
    
    curr = batch
    while curr < n_total:
        b_paths = paths[curr:curr + batch]
        imgs = list(pool.map(lambda p: Image.open(p).convert("RGB"), b_paths))
        x = torch.stack(list(pool.map(ex._one, imgs))).to(sb.DEVICE)
        with torch.no_grad():
            f = ex.forward_feats(x)
        feats[curr * patches_per_img : (curr + len(b_paths)) * patches_per_img] = f
        del imgs, x, f
        curr += batch
    return feats


@torch.no_grad()
def anomaly_maps(ex, bank, paths, n_patches, grid, batch=8, eval_shape=(EVAL_SIDE, EVAL_SIDE), gauss_sigma=GAUSS_SIGMA):
    scores, maps = [], []
    for i in range(0, len(paths), batch):
        imgs = list(_POOL.map(lambda p: Image.open(p).convert("RGB"), paths[i:i + batch]))
        x = torch.stack(list(_POOL.map(ex._one, imgs))).to(sb.DEVICE)
        feats = ex.forward_feats(x)
        d = sb.patch_distances(bank, feats, n_patches)
        im_scores = d.max(dim=1).values.tolist()
        scores.extend(im_scores)

        m = d.view(-1, 1, grid[0], grid[1])
        m = F.interpolate(m, size=eval_shape, mode="bilinear", align_corners=False)
        m = m.squeeze(1).cpu().numpy()
        for j in range(m.shape[0]):
            m_smooth = ndimage.gaussian_filter(m[j], sigma=gauss_sigma)
            maps.append(m_smooth)
    return np.array(scores), maps


def load_paths(scenario):
    r = os.path.join(AD2_ROOT, scenario)
    train = sorted(glob.glob(os.path.join(r, "train", "**", "*.png"), recursive=True))
    val = sorted(glob.glob(os.path.join(r, "validation", "**", "*.png"), recursive=True))
    good = sorted(glob.glob(os.path.join(r, "test_public", "good", "**", "*.png"), recursive=True))
    bad = sorted(glob.glob(os.path.join(r, "test_public", "bad", "**", "*.png"), recursive=True))
    gt_dir = os.path.join(r, "test_public", "ground_truth")
    return train, val, good, bad, gt_dir


def find_mask(gt_dir, bad_path):
    stem = os.path.splitext(os.path.basename(bad_path))[0]
    hits = glob.glob(os.path.join(gt_dir, "**", stem + "*"), recursive=True)
    hits = [h for h in hits if h.lower().endswith((".png", ".bmp", ".tif", ".tiff"))]
    return hits[0] if hits else None


def main():
    arm = {"tag": "A_wrn50_448", "kind": "cnn", "name": "wide_resnet50_2", "img": IMG,
           "out_indices": (2, 3)}
    ex = sb.PatchExtractor(arm)

    # Warm up / initialize ex.grid and ex.dim
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, ex.img, ex.img, device=sb.DEVICE)
        _ = ex.forward_feats(_dummy)

    all_region_records = []
    scenario_results = {}

    print(f"Running E5a evaluation across all 8 scenarios at img={IMG}, bank_cap={BANK_CAP}...")
    t_start = time.time()

    for sc in SCENARIOS:
        t0 = time.time()
        train, val, good, bad, gt_dir = load_paths(sc)
        masks_paths = [find_mask(gt_dir, b) for b in bad]
        have = [i for i, m in enumerate(masks_paths) if m]
        bad = [bad[i] for i in have]
        masks_paths = [masks_paths[i] for i in have]

        sample_im = Image.open(train[0])
        w_nat, h_nat = sample_im.size
        w_in, h_in = aspect_dimensions(w_nat, h_nat, target_img=IMG, stride=32)
        aspect_transform(ex, w_in, h_in)
        eval_w, eval_h = aspect_dimensions(w_nat, h_nat, target_img=EVAL_SIDE, stride=32)
        eval_shape = (eval_h, eval_w)

        # Bank from train
        f_tr = extract_paths_prealloc(ex, train, _POOL)
        n_patches = ex.grid[0] * ex.grid[1]
        keep = sb.coreset_indices(f_tr, ratio=CORESET_RATIO, seed=SEED, max_k=BANK_CAP)
        bank = f_tr[keep].clone()
        del f_tr
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        s_good, m_good = anomaly_maps(ex, bank, good, n_patches, ex.grid, eval_shape=eval_shape, gauss_sigma=GAUSS_SIGMA)
        s_bad, m_bad = anomaly_maps(ex, bank, bad, n_patches, ex.grid, eval_shape=eval_shape, gauss_sigma=GAUSS_SIGMA)

        masks = []
        region_labels = []
        scenario_region_sizes = []

        for p in masks_paths:
            im_native = Image.open(p).convert("L")
            mask_native = np.array(im_native) > 127
            labelled, n_comp_all = ndimage.label(mask_native)
            clean_labels = np.zeros_like(labelled, dtype=np.int32)
            n_comp = 0
            for r in range(1, n_comp_all + 1):
                sel = (labelled == r)
                sz = int(sel.sum())
                if sz >= NATIVE_MIN_REGION_PX:
                    n_comp += 1
                    clean_labels[sel] = n_comp
                    scenario_region_sizes.append(sz)

            res_labels = np.array(Image.fromarray(clean_labels, mode="I").resize((eval_w, eval_h), Image.NEAREST))
            region_labels.append((res_labels, n_comp))
            masks.append(res_labels > 0)

        # Compute lo, hi across maps
        all_m = np.concatenate([m.ravel() for m in m_good] + [m.ravel() for m in m_bad])
        lo, hi = float(np.percentile(all_m, 1)), float(np.percentile(all_m, 99.9))
        del all_m

        res = evaluate(m_good, m_bad, masks, lo, hi, region_labels=region_labels)
        per_reg_pro5 = res["per_region_pro"][0.05]
        per_reg_pro30 = res["per_region_pro"][0.30]
        
        assert len(per_reg_pro5) == len(scenario_region_sizes), f"Mismatch: {len(per_reg_pro5)} vs {len(scenario_region_sizes)}"

        cell_area = (w_nat * h_nat) / n_patches
        sc_records = []
        for sz, p5, p30 in zip(scenario_region_sizes, per_reg_pro5, per_reg_pro30):
            if sz < cell_area:
                bucket = "sub_cell"
            elif sz < 4 * cell_area:
                bucket = "1_to_4x"
            elif sz < 16 * cell_area:
                bucket = "4_to_16x"
            else:
                bucket = "ge_16x"
            rec = {
                "scenario": sc,
                "size_px": sz,
                "cell_area": cell_area,
                "size_in_cells": sz / cell_area,
                "bucket": bucket,
                "au_pro@0.05": p5,
                "au_pro@0.3": p30,
            }
            sc_records.append(rec)
            all_region_records.append(rec)

        elapsed = time.time() - t0
        print(f"{sc:<13} done in {elapsed:.1f}s: AU-PRO@5% = {res['au_pro@0.05']:.4f}, regions = {len(sc_records)}", flush=True)

        scenario_results[sc] = {
            "n_regions": len(sc_records),
            "cell_area": float(cell_area),
            "au_pro@0.05": float(res["au_pro@0.05"]),
            "au_pro@0.3": float(res["au_pro@0.3"]),
            "pixel_auroc": float(res["pixel_auroc"]),
            "buckets": {}
        }
        for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]:
            b_pro5 = [r["au_pro@0.05"] for r in sc_records if r["bucket"] == b]
            scenario_results[sc]["buckets"][b] = {
                "count": len(b_pro5),
                "mean_au_pro@0.05": float(np.mean(b_pro5)) if b_pro5 else None,
            }

        del bank, m_good, m_bad, s_good, s_bad
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # Overall dataset aggregation
    total_regs = len(all_region_records)
    print(f"\nCompleted in {time.time() - t_start:.1f}s. Total regions: {total_regs}")
    assert total_regs == 1530, f"Expected 1530 regions, got {total_regs}"

    overall_mean_p5 = float(np.mean([r["au_pro@0.05"] for r in all_region_records]))
    print(f"Overall dataset mean AU-PRO@5%: {overall_mean_p5:.4f} (matches E4-512: 0.3444)")

    overall_buckets = {}
    print("\n" + "=" * 80)
    print(f"{'Size Bucket':<18} {'Count':<8} {'% of Regs':<12} {'Mean AU-PRO@5%':<18} {'Mean AU-PRO@30%':<18}")
    print("-" * 80)
    for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]:
        b_p5 = [r["au_pro@0.05"] for r in all_region_records if r["bucket"] == b]
        b_p30 = [r["au_pro@0.3"] for r in all_region_records if r["bucket"] == b]
        cnt = len(b_p5)
        m_p5 = float(np.mean(b_p5)) if b_p5 else 0.0
        m_p30 = float(np.mean(b_p30)) if b_p30 else 0.0
        overall_buckets[b] = {
            "count": cnt,
            "fraction": cnt / total_regs,
            "mean_au_pro@0.05": m_p5,
            "mean_au_pro@0.3": m_p30,
        }
        print(f"{b:<18} {cnt:<8} {cnt/total_regs*100:<11.1f}% {m_p5:<18.4f} {m_p30:<18.4f}")

    # Combine 4x and larger
    ge_4x_p5 = [r["au_pro@0.05"] for r in all_region_records if r["bucket"] in ("4_to_16x", "ge_16x")]
    print("-" * 80)
    print(f"{'>= 4 cells (combined)':<18} {len(ge_4x_p5):<8} {len(ge_4x_p5)/total_regs*100:<11.1f}% {np.mean(ge_4x_p5):<18.4f}")
    print("=" * 80)

    summary = {
        "run_id": "E5a-region-breakdown",
        "total_regions": total_regs,
        "mean_au_pro@0.05": overall_mean_p5,
        "overall_buckets": overall_buckets,
        "ge_4_cells_combined": {
            "count": len(ge_4x_p5),
            "fraction": len(ge_4x_p5) / total_regs,
            "mean_au_pro@0.05": float(np.mean(ge_4x_p5)),
        },
        "scenarios": scenario_results,
    }

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
