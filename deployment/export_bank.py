#!/usr/bin/env python3
"""Export fitted PatchCore coreset memory bank and metadata to Triton model directory.

Usage:
    # 1. Quick synthetic / smoke bank generation (no external dataset needed):
    python deployment/export_bank.py --synthetic --max-k 1000

    # 2. From local directory of defect-free images (e.g. MVTec AD 1 or AD 2):
    python deployment/export_bank.py --train-dir /opt/ad2/mvtec_ad_2/can/train/good \
                                    --val-dir /opt/ad2/mvtec_ad_2/can/validation/good \
                                    --backbone wide_resnet50_2 --img-size 224 --max-k 4000

    # 3. From HuggingFace MVTec AD dataset:
    python deployment/export_bank.py --hf-category bottle --max-k 4000
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import timm
except ImportError:
    timm = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))


class PatchExtractor:
    """Extracts aggregated patch features using backbone + 3x3 avg pooling."""

    def __init__(self, backbone_name: str = "wide_resnet50_2", kind: str = "cnn",
                 img_size: int = 224, out_indices: Tuple[int, ...] = (2, 3), device=DEVICE):
        if timm is None:
            raise RuntimeError("timm is required for feature extraction. Run `pip install timm`.")

        self.backbone_name = backbone_name
        self.kind = kind
        self.img_size = img_size
        self.out_indices = out_indices
        self.device = device

        if self.kind == "cnn":
            self.model = timm.create_model(
                self.backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=self.out_indices,
            ).to(self.device).eval()
        else:
            self.model = timm.create_model(
                self.backbone_name,
                pretrained=True,
                num_classes=0,
                img_size=self.img_size,
            ).to(self.device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        # ImageNet normalization config
        cfg = timm.data.resolve_data_config({}, model=self.model)
        cfg["input_size"] = (3, self.img_size, self.img_size)
        self.tfm = timm.data.create_transform(**cfg, is_training=False)
        self.mean = list(cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = list(cfg.get("std", (0.229, 0.224, 0.225)))
        self.feature_dim = None

    def _transform_image(self, img: Image.Image) -> torch.Tensor:
        return self.tfm(img.convert("RGB"))

    @torch.no_grad()
    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if self.kind == "cnn":
            fs = self.model(x)
            ref = fs[0].shape[-2:]
            fs = [
                f if f.shape[-2:] == ref else F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                for f in fs
            ]
            fmap = torch.cat(fs, dim=1)
        else:
            toks = self.model.forward_features(x)
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            toks = toks[:, n_prefix:, :]
            g = int(round(toks.shape[1] ** 0.5))
            fmap = toks.transpose(1, 2).reshape(b, -1, g, g)

        fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
        b, c, h, w = fmap.shape
        self.feature_dim = c
        self.grid = (h, w)
        return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()

    @torch.no_grad()
    def extract_from_pil(self, pil_images: List[Image.Image]) -> torch.Tensor:
        x = torch.stack(list(_POOL.map(self._transform_image, pil_images))).to(self.device)
        return self.forward_tensor(x)

    @torch.no_grad()
    def extract_paths(self, paths: List[str], batch_size: int = 16) -> torch.Tensor:
        all_feats = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            imgs = list(_POOL.map(lambda p: Image.open(p).convert("RGB"), chunk))
            feats = self.extract_from_pil(imgs)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)


def greedy_kcenter_coreset(feats: torch.Tensor, ratio: float = 0.01, max_k: int = 4000,
                           proj_dim: int = 128, seed: int = 0, chunk: int = 16384,
                           device=DEVICE) -> torch.Tensor:
    """Greedy k-center coreset reduction with random orthogonal projection.

    Reduces the total pool of normal patch vectors to a representative coreset memory bank.
    Capping at max_k prevents quadratic explosion of search cost at high resolutions.
    """
    n = feats.shape[0]
    k = min(max(int(n * ratio), 32), n)
    if max_k and max_k > 0:
        k = min(k, max_k)

    if k >= n:
        print(f"[Coreset] Retaining all {n} patch vectors without reduction.", flush=True)
        return feats

    print(f"[Coreset] Reducing {n} total patch vectors to coreset of {k} vectors (proj_dim={proj_dim})...", flush=True)
    g = torch.Generator().manual_seed(seed)
    P = (torch.randn(feats.shape[1], proj_dim, generator=g) / (proj_dim ** 0.5)).to(device)

    # Project in chunks to avoid GPU OOM
    projected = []
    for i in range(0, n, chunk):
        proj_chunk = feats[i : i + chunk].to(device) @ P
        projected.append(proj_chunk)
    X = torch.cat(projected, dim=0)

    # Greedy minimax distance selection
    start = int(torch.randint(n, (1,), generator=g))
    selected = [start]
    dist = torch.cdist(X, X[start : start + 1]).squeeze(1)

    t0 = time.time()
    for step in range(k - 1):
        idx = int(torch.argmax(dist))
        selected.append(idx)
        new_dists = torch.cdist(X, X[idx : idx + 1]).squeeze(1)
        dist = torch.minimum(dist, new_dists)
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Selected {step + 1}/{k} coreset points ({elapsed:.1f}s)...", flush=True)

    selected_indices = torch.tensor(selected, dtype=torch.long)
    coreset_bank = feats[selected_indices]
    print(f"[Coreset] Coreset selection complete: bank shape = {coreset_bank.shape}", flush=True)
    return coreset_bank


def calibrate_threshold(bank: torch.Tensor, val_feats: torch.Tensor, n_patches: int,
                        percentile: float = 99.0, device=DEVICE) -> float:
    """Calculates anomaly decision threshold from defect-free validation patch features."""
    print(f"[Calibration] Computing patch distances on validation set ({val_feats.shape[0]} patches)...", flush=True)
    bank_dev = bank.to(device)
    mins = []
    chunk = 8192
    for i in range(0, val_feats.shape[0], chunk):
        d = torch.cdist(val_feats[i : i + chunk].to(device), bank_dev)
        mins.append(d.min(dim=1).values.cpu())
    patch_scores = torch.cat(mins).view(-1, n_patches)
    img_scores = patch_scores.max(dim=1).values.numpy()
    thr = float(np.percentile(img_scores, percentile))
    print(f"[Calibration] Calibrated threshold at p{percentile:.1f} = {thr:.4f} "
          f"(val min={img_scores.min():.4f}, mean={img_scores.mean():.4f}, max={img_scores.max():.4f})", flush=True)
    return thr


def export_pipeline(output_dir: str, bank: torch.Tensor, metadata: dict,
                    backbone_model=None, export_onnx: bool = False):
    """Saves bank.pt, bank.npy, metadata.json, and optional backbone/ONNX to Triton directory."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Coreset Memory Bank
    bank_pt_path = os.path.join(output_dir, "bank.pt")
    bank_npy_path = os.path.join(output_dir, "bank.npy")
    torch.save(bank.clone().detach().cpu(), bank_pt_path)
    np.save(bank_npy_path, bank.cpu().numpy())
    print(f"[Export] Saved memory bank to {bank_pt_path} ({os.path.getsize(bank_pt_path) / 1e6:.2f} MB)", flush=True)

    # 2. Save Metadata Configuration
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Export] Saved metadata configuration to {meta_path}", flush=True)

    # 3. Optional: Save backbone weights for fully air-gapped Triton container deployment
    if backbone_model is not None:
        weights_path = os.path.join(output_dir, "backbone.pt")
        try:
            torch.save(backbone_model.state_dict(), weights_path)
            print(f"[Export] Saved offline backbone weights to {weights_path}", flush=True)
        except Exception as e:
            print(f"[Export] Note: Could not save offline backbone weights ({e}). Timm will initialize directly.", flush=True)

    # 4. Optional: Export backbone to ONNX (for ensemble pipelines)
    if export_onnx and backbone_model is not None:
        onnx_path = os.path.join(output_dir, "backbone.onnx")
        img_size = metadata.get("img_size", 224)
        dummy_in = torch.randn(1, 3, img_size, img_size, device=DEVICE)
        print(f"[Export] Exporting backbone to ONNX at {onnx_path}...", flush=True)
        try:
            torch.onnx.export(
                backbone_model,
                dummy_in,
                onnx_path,
                input_names=["IMAGE"],
                output_names=["FEAT_MAP"],
                dynamic_axes={"IMAGE": {0: "batch_size"}, "FEAT_MAP": {0: "batch_size"}},
                opset_version=14,
            )
            print(f"[Export] Successfully exported ONNX model ({os.path.getsize(onnx_path) / 1e6:.2f} MB)", flush=True)
        except Exception as e:
            print(f"[Export] ONNX export encountered error: {e}. Triton Python backend remains primary.", flush=True)


def generate_synthetic_bank(dim: int = 1536, size: int = 1000) -> torch.Tensor:
    """Generates synthetic normal patch features clustered around prototypes."""
    torch.manual_seed(42)
    # 20 prototype clusters
    protos = torch.randn(20, dim)
    cluster_ids = torch.randint(0, 20, (size,))
    noise = torch.randn(size, dim) * 0.15
    return protos[cluster_ids] + noise


def main():
    parser = argparse.ArgumentParser(description="Export PatchCore memory bank and metadata for Triton.")
    parser.add_argument("--output-dir", default="deployment/triton_models/patchcore/1",
                        help="Target Triton model version directory")
    parser.add_argument("--backbone", default="wide_resnet50_2",
                        help="Backbone architecture name (e.g. wide_resnet50_2, resnet18, vit_base_patch14_dinov2)")
    parser.add_argument("--img-size", type=int, default=224, help="Model input image resolution")
    parser.add_argument("--train-dir", default="", help="Path to directory containing normal train images")
    parser.add_argument("--val-dir", default="", help="Path to directory containing normal validation images")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic bank for fast verification")
    parser.add_argument("--max-k", type=int, default=4000, help="Maximum coreset vectors in memory bank")
    parser.add_argument("--coreset-ratio", type=float, default=0.01, help="Fraction of patches to keep")
    parser.add_argument("--percentile", type=float, default=99.0, help="Calibration threshold percentile")
    parser.add_argument("--export-onnx", action="store_true", help="Export backbone to ONNX format")
    parser.add_argument("--save-backbone-weights", action="store_true", default=True,
                        help="Save PyTorch backbone weights for air-gapped Triton instances")

    args = parser.parse_args()

    kind = "vit" if ("dino" in args.backbone.lower() or "vit" in args.backbone.lower()) else "cnn"

    print("=================================================================")
    print("           PatchCore Triton Export Utility                       ")
    print("=================================================================")
    print(f"Target Directory: {args.output_dir}")
    print(f"Backbone:         {args.backbone} (kind: {kind})")
    print(f"Image Resolution: {args.img_size}x{args.img_size}")
    print(f"Bank Cap (max_k): {args.max_k}")
    print("=================================================================")

    if args.synthetic or (not args.train_dir and not os.path.exists(args.train_dir)):
        print("[Mode] Generating synthetic coreset memory bank for testing / prototyping...")
        feature_dim = 1536 if args.backbone == "wide_resnet50_2" else 768
        bank = generate_synthetic_bank(dim=feature_dim, size=min(args.max_k, 1000))
        threshold = 0.550
        extractor = None
        if timm is not None:
            try:
                extractor = PatchExtractor(backbone_name=args.backbone, kind=kind, img_size=args.img_size)
            except Exception as e:
                print(f"[Warning] Could not initialize backbone for weight saving: {e}")
    else:
        # Load images from train-dir
        train_paths = sorted(glob.glob(os.path.join(args.train_dir, "**", "*.*"), recursive=True))
        train_paths = [p for p in train_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not train_paths:
            raise FileNotFoundError(f"No image files found in {args.train_dir}")

        print(f"[Data] Found {len(train_paths)} training images in {args.train_dir}")
        extractor = PatchExtractor(backbone_name=args.backbone, kind=kind, img_size=args.img_size)

        t0 = time.time()
        print(f"[Extract] Extracting patch features from training images on {DEVICE}...")
        train_feats = extractor.extract_paths(train_paths)
        print(f"[Extract] Done in {time.time() - t0:.1f}s. Extracted {train_feats.shape[0]} patches of dim {train_feats.shape[1]}")

        # Coreset selection
        bank = greedy_kcenter_coreset(
            train_feats, ratio=args.coreset_ratio, max_k=args.max_k, seed=0, device=DEVICE
        )

        # Threshold calibration
        threshold = 0.50
        if args.val_dir and os.path.exists(args.val_dir):
            val_paths = sorted(glob.glob(os.path.join(args.val_dir, "**", "*.*"), recursive=True))
            val_paths = [p for p in val_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
            if val_paths:
                val_feats = extractor.extract_paths(val_paths)
                n_patches = extractor.grid[0] * extractor.grid[1]
                threshold = calibrate_threshold(bank, val_feats, n_patches=n_patches, percentile=args.percentile)

    metadata = {
        "backbone_name": args.backbone,
        "backbone_kind": kind,
        "out_indices": [2, 3] if kind == "cnn" else [],
        "img_size": args.img_size,
        "feature_dim": bank.shape[1],
        "coreset_size": bank.shape[0],
        "threshold": threshold,
        "percentile": args.percentile,
        "gauss_sigma": 4.0,
        "eval_side": args.img_size,
        "mean": extractor.mean if extractor else [0.485, 0.456, 0.406],
        "std": extractor.std if extractor else [0.229, 0.224, 0.225],
        "created_at": datetime.now().isoformat(),
        "device": str(DEVICE),
    }

    backbone_model = extractor.model if (extractor and args.save_backbone_weights) else None
    export_pipeline(
        output_dir=args.output_dir,
        bank=bank,
        metadata=metadata,
        backbone_model=backbone_model,
        export_onnx=args.export_onnx,
    )

    print("\n[Complete] PatchCore Triton deployment artifacts successfully generated!")
    print(f"  Model directory: {args.output_dir}")
    print(f"  Ready for Triton inference server.\n")


if __name__ == "__main__":
    main()
