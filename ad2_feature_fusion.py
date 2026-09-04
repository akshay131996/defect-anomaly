#!/usr/bin/env python3
"""MVTec AD 2 High-Performance Feature Fusion & SuperADD Alignment.

Implements the architectural specification from docs/MVTEC_AD2_IMPLEMENTATION_SPEC.md:
1. wrn50_l123: Multi-scale Layer 1 (stride 4), Layer 2, Layer 3 fusion for micro-defects (sheet_metal).
2. dinov2_448: Self-supervised ViT patch tokens for repetitive woven textures (fabric).
3. Feature Whitening / Cosine Centering & LCN to neutralize 2.7 sigma illumination shift (can).
4. SuperADD-style Morphological Closing post-processing (scipy.ndimage.grey_closing) on anomaly heatmaps.
5. Fixed 4,000-vector bank cap (max_k=4000) for fast linear inference.
6. Adaptive mode selecting the specialized optimal representation per scenario.

Outputs results to outputs/ad2_feature_fusion.json.
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

import sweep_backbones as sb
from aupro import evaluate, NBINS, PRO_LIMITS

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
CORESET_RATIO = 0.01
SEED = 0
GAUSS_SIGMA = 3.0
EVAL_SIDE = 512

_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))


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


def local_contrast_norm(t, kernel_size=15, eps=1e-4):
    """Local Contrast Normalization on float tensor [3, H, W]."""
    x = t.unsqueeze(0)
    pad = kernel_size // 2
    blur_k = torch.ones(3, 1, kernel_size, kernel_size, device=t.device) / (kernel_size * kernel_size)
    mu = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), blur_k, groups=3)
    diff = x - mu
    var = F.conv2d(F.pad(diff ** 2, (pad, pad, pad, pad), mode="reflect"), blur_k, groups=3)
    sigma = torch.sqrt(var + eps)
    return (diff / sigma).squeeze(0)


class MultiScaleExtractor:
    """Extractor supporting multi-layer CNNs (L1+L2+L3), DINOv2 ViT, and Hybrid Fusion."""

    def __init__(self, mode: str, img_size: int = 448, use_lcn: bool = False):
        self.mode = mode
        self.img_size = img_size
        self.use_lcn = use_lcn
        self.device = sb.DEVICE

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False

        if mode in ("wrn50_l123", "wrn50_l23"):
            out_idx = (1, 2, 3) if mode == "wrn50_l123" else (2, 3)
            self.model = timm.create_model(
                "wide_resnet50_2", pretrained=True, features_only=True, out_indices=out_idx
            ).to(self.device).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            cfg["input_size"] = (3, img_size, img_size)
            self.tfm = timm.data.create_transform(**cfg, is_training=False)

        elif mode in ("dinov2_448", "dinov3_448"):
            dino_name = sb.resolve_dinov3() if mode == "dinov3_448" else sb.resolve_dinov2()
            if not dino_name:
                dino_name = sb.resolve_dinov2()
            self.model = timm.create_model(
                dino_name, pretrained=True, num_classes=0, img_size=img_size
            ).to(self.device).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            cfg["input_size"] = (3, img_size, img_size)
            self.tfm = timm.data.create_transform(**cfg, is_training=False)

        elif mode == "fusion":
            # Hybrid: WideResNet50 layers 1+2+3 combined with DINOv2
            self.cnn = timm.create_model(
                "wide_resnet50_2", pretrained=True, features_only=True, out_indices=(1, 2, 3)
            ).to(self.device).eval()
            dino_name = sb.resolve_dinov2()
            self.vit = timm.create_model(
                dino_name, pretrained=True, num_classes=0, img_size=img_size
            ).to(self.device).eval()
            cfg = timm.data.resolve_data_config({}, model=self.cnn)
            cfg["input_size"] = (3, img_size, img_size)
            self.tfm = timm.data.create_transform(**cfg, is_training=False)

        self.grid = None
        self.dim = None

    def _one(self, im):
        t = self.tfm(im.convert("RGB"))
        if self.use_lcn:
            t = local_contrast_norm(t)
        return t

    @torch.no_grad()
    def forward_feats(self, x, stride: int = 1):
        b = x.shape[0]

        if self.mode == "wrn50_l123":
            # Layer 1 is H/4 (112x112 at 448px, stride 4)
            # Layer 2 is H/8, Layer 3 is H/16
            fs = self.model(x)
            ref = fs[0].shape[-2:]  # Match Layer 1's spatial grid for micro-defect preservation
            fs_aligned = [
                f if f.shape[-2:] == ref else F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                for f in fs
            ]
            fmap = torch.cat(fs_aligned, dim=1)  # 256 + 512 + 1024 = 1792 dims
            fmap = F.normalize(fmap, p=2, dim=1)
            fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
            if stride > 1:
                fmap = fmap[:, :, ::stride, ::stride]
            b, c, h, w = fmap.shape
            self.grid, self.dim = (h, w), c
            return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()

        elif self.mode == "wrn50_l23":
            # Standard Layer 2+3 reference
            fs = self.model(x)
            ref = fs[0].shape[-2:]
            fs_aligned = [
                f if f.shape[-2:] == ref else F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                for f in fs
            ]
            fmap = torch.cat(fs_aligned, dim=1)  # 512 + 1024 = 1536 dims
            fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
            if stride > 1:
                fmap = fmap[:, :, ::stride, ::stride]
            b, c, h, w = fmap.shape
            self.grid, self.dim = (h, w), c
            return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()

        elif self.mode in ("dinov2_448", "dinov3_448"):
            toks = self.model.forward_features(x)
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            toks = toks[:, n_prefix:, :]
            g = int(round(toks.shape[1] ** 0.5))
            fmap = toks.transpose(1, 2).reshape(b, -1, g, g)
            fmap = F.normalize(fmap, p=2, dim=1)
            fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
            if stride > 1:
                fmap = fmap[:, :, ::stride, ::stride]
            b, c, h, w = fmap.shape
            self.grid, self.dim = (h, w), c
            return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()

        elif self.mode == "fusion":
            # CNN multi-scale features aligned to Layer 1 grid
            c_fs = self.cnn(x)
            ref = c_fs[0].shape[-2:]
            c_aligned = [
                f if f.shape[-2:] == ref else F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                for f in c_fs
            ]
            c_fmap = torch.cat(c_aligned, dim=1)
            c_fmap = F.avg_pool2d(c_fmap, kernel_size=3, stride=1, padding=1)

            # ViT features aligned to Layer 1 grid
            v_toks = self.vit.forward_features(x)
            n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
            v_toks = v_toks[:, n_prefix:, :]
            vg = int(round(v_toks.shape[1] ** 0.5))
            v_fmap = v_toks.transpose(1, 2).reshape(b, -1, vg, vg)
            v_fmap = F.avg_pool2d(v_fmap, kernel_size=3, stride=1, padding=1)
            v_fmap = F.interpolate(v_fmap, size=ref, mode="bilinear", align_corners=False)

            # Normalize and combine with equal balance
            c_norm = F.normalize(c_fmap, p=2, dim=1) * 0.5
            v_norm = F.normalize(v_fmap, p=2, dim=1) * 0.5
            fmap = torch.cat([c_norm, v_norm], dim=1)  # 1792 + 768 = 2560 dims
            if stride > 1:
                fmap = fmap[:, :, ::stride, ::stride]

            b, c, h, w = fmap.shape
            self.grid, self.dim = (h, w), c
            return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()


def extract_features(ex, paths, batch=8, stride=1):
    feats = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = list(_POOL.map(lambda p: Image.open(p).convert("RGB"), chunk))
        x = torch.stack(list(_POOL.map(ex._one, imgs))).to(ex.device)
        feats.append(ex.forward_feats(x, stride=stride))
    return torch.cat(feats, dim=0)


def postprocess_anomaly_map(raw_map, eval_side=EVAL_SIDE, gauss_sigma=GAUSS_SIGMA, closing_k=5):
    """SuperADD-style morphological closing post-processing for continuous anomaly heatmaps.

    Bridges disconnected patch predictions and suppresses single-pixel noise spikes to
    maximize the Per-Region Overlap (PRO) integral.
    """
    # 1. Bilinear/bicubic interpolation to target evaluation dimensions
    if raw_map.shape != (eval_side, eval_side):
        raw_t = torch.from_numpy(raw_map).unsqueeze(0).unsqueeze(0).float()
        up = F.interpolate(raw_t, size=(eval_side, eval_side), mode="bicubic", align_corners=False)
        m = up[0, 0].numpy()
    else:
        m = raw_map.copy()

    # 2. Gaussian smoothing
    if gauss_sigma > 0:
        m = ndimage.gaussian_filter(m, sigma=gauss_sigma)

    # 3. Grayscale morphological closing (dilation followed by erosion with circular structuring element)
    if closing_k > 1:
        radius = closing_k // 2
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        footprint = (x * x + y * y) <= radius ** 2
        m = ndimage.grey_closing(m, footprint=footprint)

    return m


@torch.no_grad()
def anomaly_maps(ex, bank, paths, whiten=False, mu_bank=None, postprocess=True, closing_k=5, batch=8):
    """Computes anomaly scores and heatmaps with optional feature whitening and morphological closing."""
    scores, maps = [], []
    device = ex.device
    bank_dev = bank.to(device)

    if whiten and mu_bank is not None:
        mu_dev = mu_bank.to(device)
        bank_dev = F.normalize(bank_dev - mu_dev, p=2, dim=-1)

    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = list(_POOL.map(lambda p: Image.open(p).convert("RGB"), chunk))
        x = torch.stack(list(_POOL.map(ex._one, imgs))).to(device)
        feats = ex.forward_feats(x, stride=1).to(device)
        grid = ex.grid
        n_patches = grid[0] * grid[1]

        if whiten and mu_bank is not None:
            feats = F.normalize(feats - mu_dev, p=2, dim=-1)

        # Chunked k-NN search against bank
        mins = []
        c_chunk = 8192
        for ci in range(0, feats.shape[0], c_chunk):
            d = torch.cdist(feats[ci:ci + c_chunk], bank_dev)
            mins.append(d.min(dim=1).values.cpu())
        d_all = torch.cat(mins).view(-1, n_patches)  # [B, n_patches]

        scores.extend(d_all.max(dim=1).values.tolist())
        g = d_all.view(-1, 1, grid[0], grid[1])
        up = F.interpolate(g, size=(EVAL_SIDE, EVAL_SIDE), mode="bilinear", align_corners=False)
        raw_maps = up[:, 0].cpu().numpy()

        for m in raw_maps:
            if postprocess:
                maps.append(postprocess_anomaly_map(m, eval_side=EVAL_SIDE, gauss_sigma=GAUSS_SIGMA, closing_k=closing_k))
            else:
                maps.append(m)

    return np.array(scores), maps


def get_scenario_config(scenario: str, default_arm: str, default_whiten: bool, default_lcn: bool):
    """VAND 4.0 / SuperADD Alignment Strategy: Selects optimal representation per scenario."""
    if default_arm == "adaptive":
        if scenario == "fabric":
            # Repetitive woven texture: DINOv2 self-supervised ViT patch tokens
            return "dinov2_448", False, False
        elif scenario == "sheet_metal":
            # Microscopic defects: Stride-4 Layer 1+2+3 multi-scale CNN
            return "wrn50_l123", False, False
        elif scenario == "can":
            # Illumination shift: DINOv2 with feature whitening
            return "dinov2_448", True, False
        elif scenario in ("fruit_jelly", "walnuts", "rice"):
            # Rich textures & structural variation: Hybrid Fusion
            return "fusion", False, False
        elif scenario == "vial":
            # Transparent object with fluid: Layer 2+3 CNN
            return "wrn50_l23", False, False
        else:
            return "wrn50_l123", default_whiten, default_lcn
    return default_arm, default_whiten, default_lcn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["wrn50_l123", "wrn50_l23", "dinov2_448", "dinov3_448", "fusion", "adaptive"],
                    default="adaptive", help="Model arm (use 'adaptive' for SuperADD-aligned per-scenario selection)")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bank-cap", type=int, default=4000, help="Cap memory bank to 4,000 vectors for O(N) compute")
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--whiten", action="store_true", help="Enable feature whitening / cosine centering")
    ap.add_argument("--lcn", action="store_true", help="Enable local contrast normalization on input images")
    ap.add_argument("--no-closing", action="store_true", help="Disable morphological closing post-processing")
    ap.add_argument("--closing-k", type=int, default=5, help="Kernel diameter for morphological closing")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/ad2_feature_fusion.json"
    scenarios = [s for s in args.scenarios.split(",") if s] or sorted(
        d for d in os.listdir(AD2_ROOT) if os.path.isdir(os.path.join(AD2_ROOT, d))
    )

    summary = {
        "arm": args.arm,
        "img": args.img,
        "bank_cap": args.bank_cap,
        "whiten": args.whiten,
        "lcn": args.lcn,
        "closing_k": 0 if args.no_closing else args.closing_k,
        "device": torch.cuda.get_device_name(0) if sb.DEVICE == "cuda" else "cpu",
        "scenarios": {},
    }

    print(f"=========================================================================")
    print(f"       MVTec AD 2 SuperADD / VAND 4.0 Feature Fusion Evaluation          ")
    print(f"=========================================================================")
    print(f"Arm:            {args.arm}")
    print(f"Resolution:     {args.img}px (Eval Side: {EVAL_SIDE}px)")
    print(f"Bank Cap:       {args.bank_cap} vectors")
    print(f"Post-Process:   {'Closing (k=' + str(args.closing_k) + ')' if not args.no_closing else 'Raw'}")
    print(f"Whitening/LCN:  Whiten={args.whiten}, LCN={args.lcn}")
    print(f"=========================================================================\n")

    for sc in scenarios:
        t0 = time.time()
        sc_arm, sc_whiten, sc_lcn = get_scenario_config(sc, args.arm, args.whiten, args.lcn)
        train, val, good, bad, gt_dir = load_paths(sc)
        if args.limit:
            train, val, good, bad = (
                train[:args.limit], val[:args.limit], good[:args.limit], bad[:args.limit]
            )
        if not train or not bad:
            print(f"{sc}: skipped", flush=True)
            continue

        masks_paths = [find_mask(gt_dir, b) for b in bad]
        have = [i for i, m in enumerate(masks_paths) if m]
        bad = [bad[i] for i in have]
        masks_paths = [masks_paths[i] for i in have]
        if not bad:
            print(f"{sc}: skipped (no masks matched)", flush=True)
            continue

        # Instantiate extractor for this scenario's chosen configuration
        ex = MultiScaleExtractor(mode=sc_arm, img_size=args.img, use_lcn=sc_lcn)

        stride_tr = 2 if sc_arm in ("wrn50_l123", "fusion") else 1
        f_tr = extract_features(ex, train, stride=stride_tr)
        keep = sb.coreset_indices(f_tr, ratio=CORESET_RATIO, seed=SEED, max_k=args.bank_cap or None)
        bank = f_tr[keep]
        del f_tr

        mu_bank = bank.mean(dim=0, keepdim=True) if sc_whiten else None

        s_good, m_good = anomaly_maps(
            ex, bank, good,
            whiten=sc_whiten, mu_bank=mu_bank,
            postprocess=(not args.no_closing), closing_k=args.closing_k
        )
        s_bad, m_bad = anomaly_maps(
            ex, bank, bad,
            whiten=sc_whiten, mu_bank=mu_bank,
            postprocess=(not args.no_closing), closing_k=args.closing_k
        )

        masks = []
        for p in masks_paths:
            a = np.array(Image.open(p).convert("L").resize((EVAL_SIDE, EVAL_SIDE), Image.NEAREST))
            masks.append(a > 127)

        truth = np.r_[np.zeros(len(s_good), int), np.ones(len(s_bad), int)]
        img_auroc = float(roc_auc_score(truth, np.r_[s_good, s_bad]))

        allv = np.concatenate([m.ravel() for m in (m_good + m_bad)])
        res = evaluate(m_good, m_bad, masks, float(allv.min()), float(allv.max()))
        res.update({
            "image_auroc": img_auroc,
            "scenario_arm": sc_arm,
            "whiten": sc_whiten,
            "lcn": sc_lcn,
            "n_train": len(train),
            "n_val": len(val),
            "n_good": len(good),
            "n_bad": len(bad),
            "bank_size": int(len(keep)),
            "feature_dim": ex.dim,
            "grid": list(ex.grid),
            "seconds": round(time.time() - t0, 1),
        })
        summary["scenarios"][sc] = res
        print(
            f"{sc:<13} [{sc_arm:<10}] img {img_auroc:.4f}  pix {res['pixel_auroc']:.4f}  "
            f"AU-PRO@5% {res.get('au_pro@0.05')!s:>7.7}  "
            f"@30% {res.get('au_pro@0.3')!s:>7.7}  "
            f"regions {res['n_regions']:>4}  {res['seconds']:.0f}s",
            flush=True,
        )

        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        del bank, m_good, m_bad, ex
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()

    done = [v for v in summary["scenarios"].values() if v.get("au_pro@0.05") is not None]
    if done:
        summary["mean_image_auroc"] = float(np.mean([v["image_auroc"] for v in done]))
        summary["mean_pixel_auroc"] = float(np.mean([v["pixel_auroc"] for v in done]))
        for lim in PRO_LIMITS:
            summary[f"mean_au_pro@{lim}"] = float(np.mean([v[f"au_pro@{lim}"] for v in done]))
        print(
            f"\n=========================================================================\n"
            f"SUMMARY SCOREBOARD [{args.arm.upper()}]:\n"
            f"  Mean Image AUROC:   {summary['mean_image_auroc']:.4f}\n"
            f"  Mean Pixel AUROC:   {summary['mean_pixel_auroc']:.4f}\n"
            f"  Mean AU-PRO@5%:     {summary['mean_au_pro@0.05']:.4f}  (Baseline was 0.1307)\n"
            f"  Mean AU-PRO@30%:    {summary['mean_au_pro@0.3']:.4f}  (Baseline was 0.3416)\n"
            f"========================================================================="
        )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
