#!/usr/bin/env python3
"""
REF-official-256: Official Amazon PatchCore Ensemble on MVTec AD 2 test_public.
Evaluated with fixed native region set via aupro.evaluate.

Configuration (from arXiv:2503.21622 and official PatchCore repo):
- Backbones: WideResNet-101 + ResNeXt-101 + DenseNet-201
- Layers: layer2 + layer3 per backbone
- Pretrain embed dimension: 1024 (adaptive avg pooling per layer)
- Target embed dimension: 384 (adaptive avg pooling across stacked layers)
- Patch size: 3, Stride: 1
- Coreset: Approximate Greedy Coreset 1% (uncapped)
- Image size: 256x256 squash (no center crop)
- Smoothing: Gaussian filter sigma=4 on 256x256 map
- Score aggregation: [0, 1] min-max normalized per model, mean ensemble
- Metric: aupro.evaluate with native 77px fixed regions pinned to 448 reference cell edges
"""

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure project root is in sys.path for aupro
sys.path.insert(0, "/workspace")
import aupro
from aupro import evaluate

from patchcore.backbones import load as load_backbone
from patchcore.patchcore import PatchCore
from patchcore.sampler import ApproximateGreedyCoresetSampler
from patchcore.common import FaissNN

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
OUT_PATH = "/workspace/outputs/runs/REF-official-256.json"
SCENARIOS = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts"
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MODELS_CONFIG = [
    ("wideresnet101", ["layer2", "layer3"]),
    ("resnext101", ["layer2", "layer3"]),
    ("densenet201", ["features.denseblock2", "features.denseblock3"]),
]


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        is_bad = 1 if "/bad/" in self.paths[idx].replace("\\", "/") else 0
        return {
            "image": im,
            "is_anomaly": torch.tensor(is_bad, dtype=torch.long),
            "mask": torch.zeros((1, 256, 256), dtype=torch.float32),
            "path": self.paths[idx]
        }


def find_mask(gt_dir, bad_path):
    stem = os.path.splitext(os.path.basename(bad_path))[0]
    hits = glob.glob(os.path.join(gt_dir, "**", stem + "*"), recursive=True)
    hits = [h for h in hits if h.lower().endswith((".png", ".bmp", ".tif", ".tiff"))]
    return hits[0] if hits else None


def aspect_dimensions(w_nat, h_nat, target_img=448, stride=32):
    aspect = w_nat / h_nat
    h_in = int(round(np.sqrt((target_img ** 2) / aspect)))
    w_in = int(round(h_in * aspect))
    h_in = max(stride, int(round(h_in / stride)) * stride)
    w_in = max(stride, int(round(w_in / stride)) * stride)
    return w_in, h_in


def run_scenario(sc):
    t0 = time.time()
    r = os.path.join(AD2_ROOT, sc)
    train_paths = sorted(glob.glob(os.path.join(r, "train", "**", "*.png"), recursive=True))
    val_paths = sorted(glob.glob(os.path.join(r, "validation", "**", "*.png"), recursive=True))
    good_paths = sorted(glob.glob(os.path.join(r, "test_public", "good", "**", "*.png"), recursive=True))
    bad_paths = sorted(glob.glob(os.path.join(r, "test_public", "bad", "**", "*.png"), recursive=True))
    gt_dir = os.path.join(r, "test_public", "ground_truth")

    masks_paths = [find_mask(gt_dir, b) for b in bad_paths]
    have = [i for i, m in enumerate(masks_paths) if m]
    bad_paths = [bad_paths[i] for i in have]
    masks_paths = [masks_paths[i] for i in have]

    train_ds = PathDataset(train_paths, transform=TRANSFORM_256)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    test_paths = good_paths + bad_paths
    test_ds = PathDataset(test_paths, transform=TRANSFORM_256)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    ensemble_scores = []
    ensemble_segs = []

    for name, layers in MODELS_CONFIG:
        print(f"  [{sc}] fitting {name}...")
        bb = load_backbone(name)
        bb.name = name
        sampler = ApproximateGreedyCoresetSampler(percentage=0.01, device=DEVICE)
        nn_method = FaissNN(on_gpu=False, num_workers=8)
        pc = PatchCore(DEVICE)
        pc.load(
            backbone=bb,
            layers_to_extract_from=layers,
            device=DEVICE,
            input_shape=(3, 256, 256),
            pretrain_embed_dimension=1024,
            target_embed_dimension=384,
            patchsize=3,
            patchstride=1,
            anomaly_score_num_nn=1,
            featuresampler=sampler,
            nn_method=nn_method,
        )
        pc.fit(train_loader)
        scores, segmentations, _, _ = pc.predict(test_loader)
        ensemble_scores.append(scores)
        ensemble_segs.append(segmentations)
        del pc, bb
        torch.cuda.empty_cache()

    # Normalize scores per model and average
    norm_scores = []
    for s in ensemble_scores:
        s = np.array(s, dtype=np.float32)
        s_min, s_max = float(s.min()), float(s.max())
        norm_scores.append((s - s_min) / (s_max - s_min + 1e-8))
    final_scores = np.mean(norm_scores, axis=0)

    # Normalize segmentations per image per model and average
    norm_segs = []
    for segs in ensemble_segs:
        segs = np.array(segs, dtype=np.float32)  # (N_test, 256, 256)
        s_min = segs.reshape(len(segs), -1).min(axis=-1).reshape(-1, 1, 1)
        s_max = segs.reshape(len(segs), -1).max(axis=-1).reshape(-1, 1, 1)
        norm_segs.append((segs - s_min) / (s_max - s_min + 1e-8))
    final_maps = np.mean(norm_segs, axis=0)  # (N_test, 256, 256)

    n_good = len(good_paths)
    s_good = final_scores[:n_good]
    s_bad = final_scores[n_good:]
    m_good = [final_maps[i] for i in range(n_good)]
    m_bad = [final_maps[i] for i in range(n_good, len(test_paths))]

    # Resize anomaly maps to 512x512 for standard comparison
    m_good_512 = [np.array(Image.fromarray(m).resize((512, 512), Image.BILINEAR)) for m in m_good]
    m_bad_512 = [np.array(Image.fromarray(m).resize((512, 512), Image.BILINEAR)) for m in m_bad]

    # Process masks and fixed native regions
    sample_im = Image.open(train_paths[0])
    w_nat, h_nat = sample_im.size

    masks = []
    region_labels = []
    scenario_region_sizes = []
    total_native_regions = 0

    for p in masks_paths:
        im_native = Image.open(p).convert("L")
        mask_native = np.array(im_native) > 127

        labelled, n_comp_all = ndimage.label(mask_native)
        clean_labels = np.zeros_like(labelled, dtype=np.int32)
        n_comp = 0
        for r_idx in range(1, n_comp_all + 1):
            sel = (labelled == r_idx)
            sz = int(sel.sum())
            if sz >= 77:
                n_comp += 1
                clean_labels[sel] = n_comp
                scenario_region_sizes.append(sz)
        total_native_regions += n_comp

        # Resize clean_labels to 512x512 with NEAREST
        res_labels = np.array(Image.fromarray(clean_labels, mode="I").resize((512, 512), Image.NEAREST))
        masks.append(res_labels > 0)
        region_labels.append((res_labels, n_comp))

    truth = np.r_[np.zeros(len(s_good), int), np.ones(len(s_bad), int)]
    img_auroc = float(roc_auc_score(truth, np.r_[s_good, s_bad]))

    lo = min(float(m.min()) for m in (m_good_512 + m_bad_512))
    hi = max(float(m.max()) for m in (m_good_512 + m_bad_512))

    res = evaluate(m_good_512, m_bad_512, masks, lo, hi, valid=None, region_labels=region_labels)
    assert res["n_regions"] == total_native_regions, f"Region count mismatch: {res['n_regions']} vs {total_native_regions}"

    # Pinned 448 reference cell edges (D-04)
    w_448, h_448 = aspect_dimensions(w_nat, h_nat, target_img=448, stride=32)
    n_patches_448 = (h_448 // 8) * (w_448 // 8)
    cell_area_448 = (w_nat * h_nat) / n_patches_448

    if "per_region_pro" in res and res["per_region_pro"].get(0.05) and len(scenario_region_sizes) == len(res["per_region_pro"][0.05]):
        p5s = res["per_region_pro"][0.05]
        p30s = res["per_region_pro"][0.3]
        buckets = {b: {"count": 0, "pro5_sum": 0.0, "pro30_sum": 0.0}
                   for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]}
        for sz, p5, p30 in zip(scenario_region_sizes, p5s, p30s):
            if sz < cell_area_448:
                b = "sub_cell"
            elif sz < 4 * cell_area_448:
                b = "1_to_4x"
            elif sz < 16 * cell_area_448:
                b = "4_to_16x"
            else:
                b = "ge_16x"
            buckets[b]["count"] += 1
            buckets[b]["pro5_sum"] += p5
            buckets[b]["pro30_sum"] += p30

        res["cell_area_448"] = float(cell_area_448)
        res["cell_area_nominal"] = float((w_nat * h_nat) / (32 * 32))
        res["buckets"] = {
            b: {
                "count": buckets[b]["count"],
                "mean_au_pro@0.05": float(buckets[b]["pro5_sum"] / buckets[b]["count"]) if buckets[b]["count"] else None,
                "mean_au_pro@0.3": float(buckets[b]["pro30_sum"] / buckets[b]["count"]) if buckets[b]["count"] else None,
            }
            for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]
        }
        del res["per_region_pro"]

    elapsed = round(time.time() - t0, 1)
    res.update({
        "image_auroc": img_auroc,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_good": len(good_paths),
        "n_bad": len(bad_paths),
        "seconds": elapsed,
        "native_regions": int(total_native_regions),
        "eval_shape": [512, 512]
    })

    print(f"{sc:<13} img {img_auroc:.4f}  pix {res['pixel_auroc']:.4f}  "
          f"AU-PRO@5% {res['au_pro@0.05']:.5f}  @30% {res['au_pro@0.3']:.5f}  "
          f"regs ({res['n_active_regions']}/{res['n_regions']} act)  {int(elapsed)}s", flush=True)

    return res


def main():
    print("Starting REF-official-256 (Amazon PatchCore Ensemble @ 256px squash)...", flush=True)
    t_start = time.time()
    results = {
        "run_id": "REF-official-256",
        "hypothesis": "Official Amazon PatchCore Ensemble reproduced on MVTec AD 2 test_public with native region evaluator",
        "command": "python run_ref_official_256.py",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "config": {
            "backbones": ["wideresnet101", "resnext101", "densenet201"],
            "layers": [
                ["layer2", "layer3"],
                ["layer2", "layer3"],
                ["features.denseblock2", "features.denseblock3"]
            ],
            "img": 256,
            "geometry": "squash",
            "pretrain_embed_dim": 1024,
            "target_embed_dim": 384,
            "coreset_ratio": 0.01,
            "bank_cap": 0,
            "smoothing": 4,
            "fixed_regions": True,
            "native_min_region_px": 77
        },
        "scenarios": {}
    }

    for sc in SCENARIOS:
        results["scenarios"][sc] = run_scenario(sc)

    total_wall = round(time.time() - t_start, 1)
    results["wall_seconds"] = total_wall

    # Aggregates
    all_sc = list(results["scenarios"].values())
    results["mean_image_auroc"] = float(np.mean([s["image_auroc"] for s in all_sc]))
    results["mean_pixel_auroc"] = float(np.mean([s["pixel_auroc"] for s in all_sc]))
    results["mean_au_pro@0.05"] = float(np.mean([s["au_pro@0.05"] for s in all_sc]))
    results["mean_au_pro@0.3"] = float(np.mean([s["au_pro@0.3"] for s in all_sc]))
    results["total_regions"] = sum(s["n_regions"] for s in all_sc)
    results["total_active_regions"] = sum(s["n_active_regions"] for s in all_sc)

    # Buckets aggregation
    agg_buckets = {b: {"count": 0, "pro5_sum": 0.0} for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]}
    for s in all_sc:
        if "buckets" in s:
            for b, data in s["buckets"].items():
                cnt = data["count"]
                if cnt and data["mean_au_pro@0.05"] is not None:
                    agg_buckets[b]["count"] += cnt
                    agg_buckets[b]["pro5_sum"] += data["mean_au_pro@0.05"] * cnt

    results["buckets"] = {
        b: {
            "count": agg_buckets[b]["count"],
            "fraction": agg_buckets[b]["count"] / results["total_regions"],
            "mean_au_pro@0.05": agg_buckets[b]["pro5_sum"] / agg_buckets[b]["count"] if agg_buckets[b]["count"] else None
        }
        for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]
    }

    print(f"\nCompleted REF-official-256 in {total_wall}s.")
    print(f"Mean Image AUROC: {results['mean_image_auroc']:.4f}")
    print(f"Mean Pixel AUROC: {results['mean_pixel_auroc']:.4f}")
    print(f"Mean AU-PRO@5%:  {results['mean_au_pro@0.05']:.4f}")
    print(f"Active Regions:   {results['total_active_regions']} / {results['total_regions']}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
