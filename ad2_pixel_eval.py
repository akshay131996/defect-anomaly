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

# Pod target RTX 4000 Ada in container: disable cuDNN
torch.backends.cudnn.enabled = False

import sweep_backbones as sb
# The metric lives in its own numpy-only module so it can be unit-tested
# without a GPU - see test_aupro.py. It had an under-reporting bug that a
# local copy here would have silently reintroduced.
from aupro import evaluate, NBINS, PRO_LIMITS

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
OUT = "outputs/ad2_pixel_eval.json"

# Arm A / E territory - layer2+layer3 on a ResNet50-family backbone is the configuration
# the current best AD 2 result uses, so it is the right baseline to establish first.
ARM = {"tag": "A_wrn50_224", "kind": "cnn", "name": "wide_resnet50_2", "img": 224,
       "out_indices": (2, 3)}

CORESET_RATIO = 0.01
SEED = 0
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


def squash_transform(ex):
    """Replace timm's resize-then-centre-crop with a direct squash to (img, img).

    timm's eval transform resizes the short side to img/crop_pct and centre-crops. That
    is right for ImageNet, where the subject is centred and the frame is roughly square.
    On AD 2 it is wrong twice over.

    First, the images are strongly non-square and the aspect differs per scenario, so the
    crop discards a different slice of each: sheet_metal is 4224x1056 and keeps 21.9% of
    its width, can is 2232x1024 and keeps 40.1%. Defects outside the crop are simply not
    visible to the model.

    Second - and this is the part that matters for the metric - the ground-truth masks are
    squashed FULL-FRAME to EVAL_SIDE (see main). So the anomaly map covers a centre
    sub-rectangle stretched to a square while the mask covers the whole image stretched to
    a square. They are in different coordinate frames, and by a different amount in each
    scenario.

    That misregistration is nearly invisible to the two metrics we had been reading as
    "fine" and fatal to the one we could not explain:

      image AUROC  - a max over patches. A defect anywhere in the visible region still
                     scores high, so this barely moves (we measured 0.663 vs 0.659 published).
      pixel AUROC  - dominated by the overwhelming mass of correctly-scored normal pixels
                     (0.733 vs 0.763).
      AU-PRO       - per-region overlap. It is the only one of the three that requires the
                     map and the mask to be spatially registered, and it collapsed (0.13
                     vs 0.76).

    It is also resolution-invariant, because crop_pct is constant - which is why pushing
    the input to 768 and 1024 never recovered anything.
    """
    from torchvision import transforms as T
    norm = [t for t in ex.tfm.transforms if isinstance(t, T.Normalize)][0]
    ex.tfm = T.Compose([
        T.Resize((ex.img, ex.img), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        norm,
    ])
    return ex


class LetterboxTransform:
    """Resize longest side to target_size, pad short side to square with zero after Normalize."""

    def __init__(self, target_size, norm):
        self.target_size = target_size
        self.norm = norm

    def __call__(self, img):
        w, h = img.size
        s = self.target_size / max(w, h)
        nw, nh = int(round(w * s)), int(round(h * s))
        from torchvision import transforms as T
        resized = img.resize((nw, nh), Image.BICUBIC)
        t = T.functional.to_tensor(resized)
        if self.norm is not None:
            t = self.norm(t)
        pad_left = (self.target_size - nw) // 2
        pad_right = self.target_size - nw - pad_left
        pad_top = (self.target_size - nh) // 2
        pad_bottom = self.target_size - nh - pad_top
        return F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)


def letterbox_transform(ex):
    from torchvision import transforms as T
    norm = [t for t in ex.tfm.transforms if isinstance(t, T.Normalize)][0]
    ex.tfm = LetterboxTransform(ex.img, norm)
    return ex


def aspect_dimensions(w_nat, h_nat, target_img=448, stride=32):
    """Calculates non-square (w_in, h_in) preserving aspect ratio, rounded to stride,
    holding total patch area roughly constant against (target_img x target_img)."""
    r = w_nat / h_nat
    target_area = target_img * target_img
    h_float = (target_area / r) ** 0.5
    w_float = r * h_float
    h_in = max(stride, int(round(h_float / stride) * stride))
    w_in = max(stride, int(round(w_float / stride) * stride))
    return w_in, h_in


def aspect_transform(ex, w_in, h_in):
    """Aspect-preserving non-square resize to (h_in, w_in), divisible by backbone stride."""
    from torchvision import transforms as T
    norm = [t for t in ex.tfm.transforms if isinstance(t, T.Normalize)][0]
    ex.tfm = T.Compose([
        T.Resize((h_in, w_in), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        norm,
    ])
    return ex


def extract_paths_prealloc(ex, paths, pool, batch=8):
    """Preallocate CPU tensor for patch feature extraction to eliminate the 2x memory
    spike from torch.cat. Bit-identical to sb.extract_paths (verified in test_prealloc.py)."""
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
def anomaly_maps(ex, bank, paths, n_patches, grid, batch=8, valid_grid=None, eval_shape=(EVAL_SIDE, EVAL_SIDE), gauss_sigma=GAUSS_SIGMA):
    """Returns (image_scores, list of HxW float maps at eval_shape resolution)."""
    scores, maps = [], []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = list(_POOL.map(lambda p: Image.open(p).convert("RGB"), chunk))
        x = torch.stack(list(_POOL.map(ex._one, imgs))).to(sb.DEVICE)
        feats = ex.forward_feats(x)
        d = sb.patch_distances(bank, feats, n_patches)          # (b, n_patches)
        g = d.view(-1, 1, grid[0], grid[1])
        if valid_grid is not None:
            g_top, g_bottom, g_left, g_right = valid_grid
            d_valid = g[:, 0, g_top:g_bottom, g_left:g_right].reshape(d.shape[0], -1)
            scores.extend(d_valid.max(dim=1).values.tolist())
        else:
            scores.extend(d.max(dim=1).values.tolist())
        # UPSAMPLE FIRST, then smooth at pixel scale.
        up = F.interpolate(g, size=eval_shape, mode="bilinear",
                           align_corners=False)
        if gauss_sigma > 0:
            r = int(3 * gauss_sigma)
            xs = torch.arange(-r, r + 1, device=up.device, dtype=up.dtype)
            kern = torch.exp(-(xs ** 2) / (2 * gauss_sigma ** 2))
            kern = kern / kern.sum()
            # separable: rows then columns, so cost is linear in kernel width
            up = F.conv2d(F.pad(up, (r, r, 0, 0), mode="replicate"),
                          kern.view(1, 1, 1, -1))
            up = F.conv2d(F.pad(up, (0, 0, r, r), mode="replicate"),
                          kern.view(1, 1, -1, 1))
        maps.extend(up[:, 0].cpu().numpy())
    return np.array(scores), maps


def main():
    import argparse
    import sys
    import hashlib
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per split, for a fast smoke run")
    ap.add_argument("--bank-cap", type=int, default=0,
                    help="cap the memory bank in absolute vectors.")
    ap.add_argument("--img", type=int, default=0,
                    help="override input resolution.")
    ap.add_argument("--eval-side", type=int, default=EVAL_SIDE,
                    help=f"Nominal side length for evaluation frame (default: {EVAL_SIDE})")
    ap.add_argument("--gauss-sigma", type=float, default=GAUSS_SIGMA,
                    help=f"Gaussian smoothing sigma for anomaly maps (default: {GAUSS_SIGMA})")
    ap.add_argument("--geometry", choices=["crop", "squash", "letterbox", "aspect"], default="crop",
                    help="Coordinate frame geometry (crop, squash, letterbox, or aspect)")
    ap.add_argument("--squash", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=OUT, help="Path for output JSON record")
    ap.add_argument("--run-id", default="", help="Optional run identifier for LEDGER tracking")
    ap.add_argument("--hypothesis", default="", help="Stated hypothesis for this run")
    ap.add_argument("--native-min-region-px", type=int, default=77,
                    help="Minimum native pixel area to keep a defect component (E4a fixed regions)")
    ap.add_argument("--no-fixed-regions", action="store_true",
                    help="Disable native fixed region sets and use legacy post-resize connected components")
    ap.add_argument("--resume", action="store_true",
                    help="Resume evaluation by reusing scenarios already computed in output JSON")
    args = ap.parse_args()

    if args.squash:
        args.geometry = "squash"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scenarios = ([s for s in args.scenarios.split(",") if s] or
                 sorted(d for d in os.listdir(AD2_ROOT)
                        if os.path.isdir(os.path.join(AD2_ROOT, d))))

    arm = dict(ARM)
    if args.img:
        arm["img"] = args.img
        arm["tag"] = f"{ARM['name']}_{args.img}"
    ex = sb.PatchExtractor(arm)

    if args.geometry == "squash":
        squash_transform(ex)
    elif args.geometry == "letterbox":
        letterbox_transform(ex)

    # Warm up / initialize ex.grid and ex.dim
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, ex.img, ex.img, device=sb.DEVICE)
        _ = ex.forward_feats(_dummy)

    t_start = time.time()
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    code_hashes = {}
    for fn in ["ad2_pixel_eval.py", "sweep_backbones.py", "aupro.py"]:
        if os.path.isfile(fn):
            with open(fn, "rb") as f:
                code_hashes[fn] = hashlib.sha256(f.read()).hexdigest()[:8]

    summary = {
        "run_id": args.run_id or f"run_{args.geometry}_{arm['img']}",
        "hypothesis": args.hypothesis,
        "command": " ".join(sys.argv),
        "code_sha256": code_hashes,
        "started_utc": started_utc,
        "wall_seconds": 0,
        "env": {
            "gpu": torch.cuda.get_device_name(0) if sb.DEVICE == "cuda" else "cpu",
            "torch": torch.__version__,
        },
        "config": {
            "img": arm["img"],
            "bank_cap": args.bank_cap,
            "eval_side": args.eval_side,
            "gauss_sigma": args.gauss_sigma,
            "coreset_ratio": CORESET_RATIO,
            "geometry": args.geometry,
            "fixed_regions": not args.no_fixed_regions,
            "native_min_region_px": args.native_min_region_px if not args.no_fixed_regions else None,
        },
        "scenarios": {},
        "deviations": [],
    }

    for sc in scenarios:
        if args.resume and os.path.isfile(args.out):
            try:
                with open(args.out) as f:
                    prev = json.load(f)
                if sc in prev.get("scenarios", {}) and prev["scenarios"][sc].get("au_pro@0.05") is not None:
                    print(f"{sc:<13} already computed in {args.out}, reusing", flush=True)
                    summary["scenarios"][sc] = prev["scenarios"][sc]
                    continue
            except Exception:
                pass

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

        # Inspect scenario native image resolution for letterbox or aspect geometry
        valid = None
        valid_grid = None
        eval_shape = (args.eval_side, args.eval_side)
        if args.geometry == "letterbox":
            sample_im = Image.open(train[0])
            w, h = sample_im.size
            s_eval = args.eval_side / max(w, h)
            nw_eval, nh_eval = int(round(w * s_eval)), int(round(h * s_eval))
            pad_left_eval = (args.eval_side - nw_eval) // 2
            pad_top_eval = (args.eval_side - nh_eval) // 2

            valid = np.zeros((args.eval_side, args.eval_side), bool)
            valid[pad_top_eval:pad_top_eval + nh_eval, pad_left_eval:pad_left_eval + nw_eval] = True

            # Patch grid bounds
            # For WideResNet50 stride 8 at layer 2
            H_g, W_g = ex.grid[0], ex.grid[1]
            stride_y = arm["img"] / H_g
            stride_x = arm["img"] / W_g
            s_in = arm["img"] / max(w, h)
            nw_in, nh_in = int(round(w * s_in)), int(round(h * s_in))
            pad_left_in = (arm["img"] - nw_in) // 2
            pad_top_in = (arm["img"] - nh_in) // 2

            g_top = int(round(pad_top_in / stride_y))
            g_bottom = min(H_g, int(round((pad_top_in + nh_in) / stride_y)))
            g_left = int(round(pad_left_in / stride_x))
            g_right = min(W_g, int(round((pad_left_in + nw_in) / stride_x)))
            valid_grid = (g_top, g_bottom, g_left, g_right)
        elif args.geometry == "aspect":
            sample_im = Image.open(train[0])
            w_nat, h_nat = sample_im.size
            w_in, h_in = aspect_dimensions(w_nat, h_nat, target_img=arm["img"], stride=32)
            aspect_transform(ex, w_in, h_in)
            eval_w, eval_h = aspect_dimensions(w_nat, h_nat, target_img=args.eval_side, stride=32)
            eval_shape = (eval_h, eval_w)

        # Bank from train (preallocated to eliminate transient 2x memory doubling)
        f_tr = extract_paths_prealloc(ex, train, _POOL)
        n_patches = ex.grid[0] * ex.grid[1]
        keep = sb.coreset_indices(f_tr, ratio=CORESET_RATIO, seed=SEED,
                                  max_k=args.bank_cap or None)
        bank = f_tr[keep].clone()
        del f_tr
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        s_good, m_good = anomaly_maps(ex, bank, good, n_patches, ex.grid, valid_grid=valid_grid, eval_shape=eval_shape, gauss_sigma=args.gauss_sigma)
        s_bad, m_bad = anomaly_maps(ex, bank, bad, n_patches, ex.grid, valid_grid=valid_grid, eval_shape=eval_shape, gauss_sigma=args.gauss_sigma)

        masks = []
        region_labels = []
        scenario_region_sizes = []
        total_native_regions = 0
        use_fixed_regions = not args.no_fixed_regions

        for p in masks_paths:
            im_native = Image.open(p).convert("L")
            w_nat, h_nat = im_native.size
            mask_native = np.array(im_native) > 127

            if use_fixed_regions:
                labelled, n_comp_all = ndimage.label(mask_native)
                clean_labels = np.zeros_like(labelled, dtype=np.int32)
                n_comp = 0
                for r in range(1, n_comp_all + 1):
                    sel = (labelled == r)
                    sz = int(sel.sum())
                    if sz >= args.native_min_region_px:
                        n_comp += 1
                        clean_labels[sel] = n_comp
                        scenario_region_sizes.append(sz)
                total_native_regions += n_comp

                # Resize clean_labels into evaluation space with NEAREST
                if args.geometry == "letterbox":
                    s_m = args.eval_side / max(w_nat, h_nat)
                    nmw, nmh = int(round(w_nat * s_m)), int(round(h_nat * s_m))
                    res_labels = np.array(Image.fromarray(clean_labels, mode="I").resize((nmw, nmh), Image.NEAREST))
                    pad_l = (args.eval_side - nmw) // 2
                    pad_t = (args.eval_side - nmh) // 2
                    eval_labels = np.zeros((args.eval_side, args.eval_side), dtype=np.int32)
                    eval_labels[pad_t:pad_t + nmh, pad_l:pad_l + nmw] = res_labels
                elif args.geometry == "aspect":
                    eval_h, eval_w = eval_shape
                    eval_labels = np.array(Image.fromarray(clean_labels, mode="I").resize((eval_w, eval_h), Image.NEAREST))
                else:  # squash or crop
                    eval_labels = np.array(Image.fromarray(clean_labels, mode="I").resize((args.eval_side, args.eval_side), Image.NEAREST))

                eval_mask = eval_labels > 0
                masks.append(eval_mask)
                region_labels.append((eval_labels, n_comp))
            else:
                # Legacy behavior
                if args.geometry == "letterbox":
                    s_m = args.eval_side / max(w_nat, h_nat)
                    nmw, nmh = int(round(w_nat * s_m)), int(round(h_nat * s_m))
                    m_res = im_native.resize((nmw, nmh), Image.NEAREST)
                    pad_l = (args.eval_side - nmw) // 2
                    pad_t = (args.eval_side - nmh) // 2
                    m_eval = np.zeros((args.eval_side, args.eval_side), bool)
                    m_eval[pad_t:pad_t + nmh, pad_l:pad_l + nmw] = np.array(m_res) > 127
                    masks.append(m_eval)
                elif args.geometry == "aspect":
                    eval_h, eval_w = eval_shape
                    a = np.array(im_native.resize((eval_w, eval_h), Image.NEAREST))
                    masks.append(a > 127)
                else:
                    a = np.array(im_native.resize((args.eval_side, args.eval_side), Image.NEAREST))
                    masks.append(a > 127)

        truth = np.r_[np.zeros(len(s_good), int), np.ones(len(s_bad), int)]
        img_auroc = float(roc_auc_score(truth, np.r_[s_good, s_bad]))

        # Streaming min/max without allocating full array copies (memory guard for 2048)
        if valid is not None:
            lo = min(float(m[valid].min()) for m in (m_good + m_bad))
            hi = max(float(m[valid].max()) for m in (m_good + m_bad))
        else:
            lo = min(float(m.min()) for m in (m_good + m_bad))
            hi = max(float(m.max()) for m in (m_good + m_bad))

        rl = region_labels if use_fixed_regions else None
        res = evaluate(m_good, m_bad, masks, lo, hi, valid=valid, region_labels=rl)
        if use_fixed_regions:
            assert res["n_regions"] == total_native_regions, \
                f"Region count mismatch: {res['n_regions']} vs {total_native_regions}"

        # Fixed native pixel bucket edges pinned to 448 aspect geometry (D-04)
        # Prevents population migration across arms when input resolution changes
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
            res["cell_area_nominal"] = float((w_nat * h_nat) / n_patches)
            res["buckets"] = {
                b: {
                    "count": buckets[b]["count"],
                    "mean_au_pro@0.05": float(buckets[b]["pro5_sum"] / buckets[b]["count"]) if buckets[b]["count"] else None,
                    "mean_au_pro@0.3": float(buckets[b]["pro30_sum"] / buckets[b]["count"]) if buckets[b]["count"] else None,
                }
                for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]
            }
            del res["per_region_pro"]

        res.update({
            "image_auroc": img_auroc, "n_train": len(train), "n_val": len(val),
            "n_good": len(good), "n_bad": len(bad),
            "bank_size": int(len(keep)), "seconds": round(time.time() - t0, 1),
            "grid": list(ex.grid), "n_patches": int(n_patches),
            "native_regions": int(total_native_regions) if use_fixed_regions else res["n_regions"],
            "n_active_regions": res.get("n_active_regions", res["n_regions"]),
            "eval_shape": list(eval_shape),
        })
        summary["scenarios"][sc] = res
        pix_str = f"{res['pixel_auroc']:.4f}" if res['pixel_auroc'] is not None else "  None"
        act_str = f"({res['n_active_regions']}/{res['n_regions']} act)"
        print(f"{sc:<13} img {img_auroc:.4f}  pix {pix_str}  "
              f"AU-PRO@5% {res.get('au_pro@0.05')!s:>7.7}  "
              f"@30% {res.get('au_pro@0.3')!s:>7.7}  "
              f"regs {act_str:>14}  {res['seconds']:.0f}s", flush=True)

        try:
            import resource
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform.startswith("linux"):
                peak_rss_mb = peak_rss_mb / 1024.0
            summary["peak_rss_mb"] = round(peak_rss_mb, 1)
        except Exception:
            pass

        summary["wall_seconds"] = round(time.time() - t_start, 1)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        del bank, m_good, m_bad
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    done = [v for v in summary["scenarios"].values() if v.get("au_pro@0.05") is not None]
    if done:
        summary["mean_image_auroc"] = float(np.mean([v["image_auroc"] for v in done]))
        summary["mean_pixel_auroc"] = float(np.mean([v["pixel_auroc"] for v in done]))
        for lim in PRO_LIMITS:
            summary[f"mean_au_pro@{lim}"] = float(
                np.mean([v[f"au_pro@{lim}"] for v in done]))

        total_suite_regions = sum(s["n_regions"] for s in done)
        total_active = sum(s["n_active_regions"] for s in done)
        summary["total_regions"] = total_suite_regions
        summary["total_active_regions"] = total_active

        if not args.no_fixed_regions and len(done) == 8:
            assert total_suite_regions == 1530, (
                f"Suite total region count must be exactly 1530 across all 8 scenarios, got {total_suite_regions}"
            )

        print(f"\nmean image AUROC {summary['mean_image_auroc']:.4f}   "
              f"pixel AUROC {summary['mean_pixel_auroc']:.4f}   "
              f"AU-PRO@5% {summary['mean_au_pro@0.05']:.4f}   "
              f"AU-PRO@30% {summary['mean_au_pro@0.3']:.4f}   "
              f"active {total_active}/{total_suite_regions}")

        total_bucket_counts = {"sub_cell": 0, "1_to_4x": 0, "4_to_16x": 0, "ge_16x": 0}
        total_bucket_p5_sums = {"sub_cell": 0.0, "1_to_4x": 0.0, "4_to_16x": 0.0, "ge_16x": 0.0}
        has_buckets = False
        for sc_name, data in summary["scenarios"].items():
            if "buckets" in data:
                has_buckets = True
                for b in total_bucket_counts:
                    cnt = data["buckets"][b]["count"]
                    total_bucket_counts[b] += cnt
                    if data["buckets"][b]["mean_au_pro@0.05"] is not None:
                        total_bucket_p5_sums[b] += data["buckets"][b]["mean_au_pro@0.05"] * cnt
        if has_buckets:
            tot_b_regs = sum(total_bucket_counts.values())
            if not args.no_fixed_regions and len(done) == 8:
                assert total_bucket_counts["sub_cell"] == 756, f"sub_cell bucket count mismatch: {total_bucket_counts['sub_cell']} vs 756"
                assert total_bucket_counts["1_to_4x"] == 354, f"1_to_4x bucket count mismatch: {total_bucket_counts['1_to_4x']} vs 354"
                assert total_bucket_counts["4_to_16x"] == 173, f"4_to_16x bucket count mismatch: {total_bucket_counts['4_to_16x']} vs 173"
                assert total_bucket_counts["ge_16x"] == 247, f"ge_16x bucket count mismatch: {total_bucket_counts['ge_16x']} vs 247"
            summary["buckets"] = {
                b: {
                    "count": total_bucket_counts[b],
                    "fraction": total_bucket_counts[b] / tot_b_regs if tot_b_regs else 0.0,
                    "mean_au_pro@0.05": float(total_bucket_p5_sums[b] / total_bucket_counts[b]) if total_bucket_counts[b] else None,
                }
                for b in total_bucket_counts
            }
            ge4_cnt = total_bucket_counts["4_to_16x"] + total_bucket_counts["ge_16x"]
            ge4_sum = total_bucket_p5_sums["4_to_16x"] + total_bucket_p5_sums["ge_16x"]
            summary["buckets"]["ge_4_cells_combined"] = {
                "count": ge4_cnt,
                "fraction": ge4_cnt / tot_b_regs if tot_b_regs else 0.0,
                "mean_au_pro@0.05": float(ge4_sum / ge4_cnt) if ge4_cnt else None,
            }
            print("\n" + "=" * 70)
            print(f"{'Size Bucket':<18} {'Count':<8} {'% of Regs':<12} {'Mean AU-PRO@5%':<16}")
            print("-" * 70)
            for b in ["sub_cell", "1_to_4x", "4_to_16x", "ge_16x"]:
                c = total_bucket_counts[b]
                f = c / tot_b_regs * 100 if tot_b_regs else 0.0
                m = summary["buckets"][b]["mean_au_pro@0.05"]
                m_str = f"{m:.4f}" if m is not None else "None"
                print(f"{b:<18} {c:<8} {f:<11.1f}% {m_str:<16}")
            print("-" * 70)
            ge4_m = summary["buckets"]["ge_4_cells_combined"]["mean_au_pro@0.05"]
            ge4_m_str = f"{ge4_m:.4f}" if ge4_m is not None else "None"
            print(f"{'>= 4 cells combined':<18} {ge4_cnt:<8} {ge4_cnt/tot_b_regs*100:<11.1f}% {ge4_m_str:<16}")
            print("=" * 70 + "\n")

    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            peak_rss_mb = peak_rss_mb / 1024.0
        summary["peak_rss_mb"] = round(peak_rss_mb, 1)
    except Exception:
        pass
    summary["wall_seconds"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out} (peak RSS: {summary.get('peak_rss_mb', 'N/A')} MB)")


if __name__ == "__main__":
    main()
