#!/usr/bin/env python3
"""Backbone / resolution / descriptor-width sweep over all 15 MVTec AD categories.

This is an experiment runner, not a teaching notebook. The scoring, coreset and
calibration logic is kept textually identical to build_session2.py so the numbers stay
comparable; if that file changes, this one has to change with it.

Every pair of arms differs in exactly one variable - see ARMS below.

    python sweep_backbones.py            -> outputs/sweep_backbones.json

Results are written after every arm, so a crash mid-sweep keeps what already ran.
"""
import json
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ID = "TheoM55/mvtec_all_objects_split"
OUT = "outputs/sweep_backbones.json"

CORESET_RATIO = 0.01
CAL_FRAC, CAL_SEED = 0.20, 0
PCTL = 99.0
COST_ESCAPE, COST_FALSE_ALARM = 100.0, 1.0

# Each arm differs from A in exactly one respect. `img` drives the grid: CNNs at
# out_indices (2,3) produce img/8; ViT patch14 produces img/14.
ARMS = [
    {"tag": "A_wrn50_224",    "kind": "cnn", "name": "wide_resnet50_2", "img": 224,
     "out_indices": (2, 3)},
    {"tag": "B0_dinov2_224",  "kind": "vit", "name": None, "img": 224},
    {"tag": "B1_dinov2_392",  "kind": "vit", "name": None, "img": 392},
    {"tag": "C_resnet18_224", "kind": "cnn", "name": "resnet18", "img": 224,
     "out_indices": (2, 3)},
    {"tag": "D_wrn50_320",    "kind": "cnn", "name": "wide_resnet50_2", "img": 320,
     "out_indices": (2, 3)},
    {"tag": "E_resnet50_224", "kind": "cnn", "name": "resnet50", "img": 224,
     "out_indices": (2, 3)},
]


def resolve_dinov2():
    found = timm.list_models("*dinov2*", pretrained=True)
    for want in ("vit_base_patch14_dinov2", "vit_small_patch14_dinov2"):
        hit = next((m for m in found if m.startswith(want)), None)
        if hit:
            return hit
    raise RuntimeError(f"no dinov2 model in timm; saw {found[:10]}")


class PatchExtractor:
    """Identical to build_session2.py except that IMG is per-arm rather than global."""

    def __init__(self, spec):
        self.kind = spec["kind"]
        self.img = spec["img"]
        if self.kind == "cnn":
            self.model = timm.create_model(
                spec["name"], pretrained=True, features_only=True,
                out_indices=spec["out_indices"]).to(DEVICE).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            cfg["input_size"] = (3, self.img, self.img)
        else:
            self.model = timm.create_model(
                spec["name"], pretrained=True, num_classes=0,
                img_size=self.img).to(DEVICE).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            cfg["input_size"] = (3, self.img, self.img)
        self.tfm = timm.data.create_transform(**cfg, is_training=False)
        self.grid = None
        self.dim = None

    @torch.no_grad()
    def __call__(self, pil_batch):
        x = torch.stack([self.tfm(im.convert("RGB")) for im in pil_batch]).to(DEVICE)
        if self.kind == "cnn":
            fs = self.model(x)
            ref = fs[0].shape[-2:]
            fs = [f if f.shape[-2:] == ref else
                  F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                  for f in fs]
            fmap = torch.cat(fs, dim=1)
        else:
            toks = self.model.forward_features(x)
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            toks = toks[:, n_prefix:, :]
            g = int(round(toks.shape[1] ** 0.5))
            assert g * g == toks.shape[1], f"{toks.shape[1]} patch tokens is not square"
            fmap = toks.transpose(1, 2).reshape(toks.shape[0], -1, g, g)

        fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
        b, c, h, w = fmap.shape
        self.grid, self.dim = (h, w), c
        return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()


@torch.no_grad()
def extract(ex, sub, image_col, indices, batch=16):
    out = []
    for i in range(0, len(indices), batch):
        out.append(ex([sub[j][image_col] for j in indices[i:i + batch]]))
    return torch.cat(out)


def coreset_indices(feats, ratio=CORESET_RATIO, proj_dim=128, seed=0, chunk=16384):
    n = feats.shape[0]
    k = max(int(n * ratio), 32)
    g = torch.Generator().manual_seed(seed)
    P = (torch.randn(feats.shape[1], proj_dim, generator=g) / (proj_dim ** 0.5)).to(DEVICE)
    X = torch.cat([feats[i:i + chunk].to(DEVICE) @ P for i in range(0, n, chunk)])

    start = int(torch.randint(n, (1,), generator=g))
    sel = [start]
    d = torch.cdist(X, X[start:start + 1]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(d))
        sel.append(i)
        d = torch.minimum(d, torch.cdist(X, X[i:i + 1]).squeeze(1))
    return sel


@torch.no_grad()
def patch_distances(bank, feats, n_patches, chunk=8192):
    bank = bank.to(DEVICE)
    mins = []
    for i in range(0, feats.shape[0], chunk):
        d = torch.cdist(feats[i:i + chunk].to(DEVICE), bank)
        mins.append(d.min(dim=1).values.cpu())
    return torch.cat(mins).view(-1, n_patches)


def operating(scores, cal_scores, truth, pctl):
    thr = float(np.percentile(cal_scores, pctl))
    pred = (scores > thr).astype(int)
    fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    n_pos, n_neg = int((truth == 1).sum()), int((truth == 0).sum())
    return {
        "threshold": thr,
        "escapes": fn,
        "false_alarms": fp,
        "recall": float((n_pos - fn) / n_pos) if n_pos else None,
        "far": float(fp / n_neg) if n_neg else None,
        "cost": fn * COST_ESCAPE + fp * COST_FALSE_ALARM,
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    dino_id = resolve_dinov2()
    print(f"dinov2 resolved to {dino_id}", flush=True)

    dd = load_dataset(DATASET_ID)
    parts = []
    for split_name, dset in dd.items():
        if "split" not in dset.column_names:
            dset = dset.add_column("split", [split_name] * len(dset))
        parts.append(dset)
    full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    def pick(cands):
        return next((c for c in cands if c in full.column_names), None)

    image_col = pick(("image_path", "image", "img"))
    label_col = pick(("label", "labels", "is_anomaly"))
    object_col = pick(("object", "category", "class_name"))
    defect_col = pick(("defect", "defect_type", "anomaly_type"))
    split_col = pick(("split", "set"))
    assert all([image_col, label_col, object_col, defect_col, split_col])

    import collections
    pairs = collections.Counter(zip(full[defect_col], full[label_col]))
    good_label = next(l for (d, l) in pairs if str(d).lower() == "good")

    sub = full
    OBJ = sub[object_col]
    SPLIT = [str(s).lower() for s in sub[split_col]]
    LABEL = sub[label_col]
    categories = sorted(set(OBJ))
    print(f"{len(categories)} categories: {categories}", flush=True)

    by_cat = {c: {"train": [], "test": []} for c in categories}
    for i in range(len(sub)):
        by_cat[OBJ[i]]["train" if "train" in SPLIT[i] else "test"].append(i)

    summary = {
        "dataset": DATASET_ID,
        "categories": categories,
        "coreset_ratio": CORESET_RATIO,
        "calibration_split": {"frac": CAL_FRAC, "seed": CAL_SEED},
        "threshold_rule": f"{PCTL}th percentile of held-out calibration scores",
        "cost_ratio_escape_to_false_alarm": COST_ESCAPE,
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "arms": {},
    }

    for spec in ARMS:
        spec = dict(spec)
        if spec["kind"] == "vit":
            spec["name"] = dino_id
        tag = spec["tag"]
        print(f"\n=== {tag}  ({spec['name']} @ {spec['img']}px) ===", flush=True)
        t_arm = time.time()
        try:
            ex = PatchExtractor(spec)
        except Exception as e:
            print(f"  SKIP: extractor failed: {type(e).__name__}: {e}", flush=True)
            summary["arms"][tag] = {"error": f"{type(e).__name__}: {e}"}
            continue

        arm = {"backbone": spec["name"], "img": spec["img"], "categories": {}}
        for cat in categories:
            t0 = time.time()
            tr, te = by_cat[cat]["train"], by_cat[cat]["test"]
            rng = np.random.default_rng(CAL_SEED)
            perm = rng.permutation(len(tr))
            n_cal = max(int(round(len(tr) * CAL_FRAC)), 1)
            cal = [tr[i] for i in perm[:n_cal]]
            fit = [tr[i] for i in perm[n_cal:]]

            bank = None
            try:
                f_fit = extract(ex, sub, image_col, fit)
                n_patch = ex.grid[0] * ex.grid[1]
                keep = coreset_indices(f_fit)
                bank = f_fit[keep]
                del f_fit

                scores = patch_distances(
                    bank, extract(ex, sub, image_col, te), n_patch).max(dim=1).values.numpy()
                cal_scores = patch_distances(
                    bank, extract(ex, sub, image_col, cal), n_patch).max(dim=1).values.numpy()
                truth = np.array([0 if LABEL[i] == good_label else 1 for i in te])

                rec = {
                    "auroc": float(roc_auc_score(truth, scores)),
                    "grid": list(ex.grid),
                    "descriptor_dim": int(ex.dim),
                    "patches_per_image": int(n_patch),
                    "bank_size": int(len(keep)),
                    "n_fit_images": len(fit),
                    "n_cal_images": len(cal),
                    "operating_point": operating(scores, cal_scores, truth, PCTL),
                    "seconds": round(time.time() - t0, 1),
                }
                arm["categories"][cat] = rec
                op = rec["operating_point"]
                print(f"  {cat:<12} AUROC {rec['auroc']:.4f}  grid {ex.grid[0]}x{ex.grid[1]}"
                      f"  dim {ex.dim:5d}  bank {rec['bank_size']:6d}"
                      f"  esc {op['escapes']:3d}  fa {op['false_alarms']:3d}"
                      f"  cost {op['cost']:8.0f}  {rec['seconds']:6.1f}s", flush=True)
            except Exception as e:
                print(f"  {cat:<12} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
                arm["categories"][cat] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            finally:
                bank = None
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        ok = [r for r in arm["categories"].values() if "auroc" in r]
        arm["mean_auroc"] = float(np.mean([r["auroc"] for r in ok])) if ok else None
        arm["total_cost"] = float(sum(r["operating_point"]["cost"] for r in ok))
        arm["seconds_total"] = round(time.time() - t_arm, 1)
        summary["arms"][tag] = arm
        print(f"  -- {tag}: mean AUROC {arm['mean_auroc']:.4f}"
              f"  total cost {arm['total_cost']:.0f}"
              f"  {arm['seconds_total']:.0f}s", flush=True)

        # Write after every arm so a crash keeps what already ran.
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del ex
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
