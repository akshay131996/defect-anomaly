#!/usr/bin/env python3
"""Session 3 - how many defect-free images does a line need before this works?

The question the benchmark literature does not answer, and the one that decides whether
the method is viable for a short production run. A plant commissioning a new product wants
to know whether that is two hours of photography or two weeks.

Design, and why it is shaped this way:

- **Only the bank size varies.** The calibration set is held FIXED across every point on
  the curve. Otherwise a small-N run would be penalised twice - a worse bank *and* a worse
  threshold - and the curve would not answer either question. Calibration size gets its own
  sweep later.
- **Several seeds per N.** At N=5 the answer depends heavily on *which* five images, and a
  single draw would report noise as signal. The spread is the finding, not an error bar to
  be apologised for.
- **Test and calibration features are extracted once per category** and reused across every
  (N, seed). They do not depend on the bank, so re-extracting them would multiply runtime by
  the number of grid points for no change in result.

    python data_efficiency.py            -> outputs/data_efficiency.json
"""
import collections
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import timm
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ID = "TheoM55/mvtec_all_objects_split"
OUT = "outputs/data_efficiency.json"

# Backbone: DINOv2 at 392px, which gives a 28x28 grid matching the CNN arms. Best mean
# AUROC of the six arms swept (0.9828).
#
# Its cost advantage did NOT survive the seed audit. The sweep put it 32% ahead of
# wide_resnet50_2 @224 on single-seed totals; averaging five coreset seeds over the six
# categories that carry the cost narrows that to 3.6% (15182 against 15748), which is
# inside the noise of the nine categories still on one seed. Treat A and B1 as equivalent
# overall and choose per category.
#
# Caveat carried forward: B1 loses badly on `screw` and on every small-part object. The
# curve below is therefore the curve *for this backbone*; a line inspecting screws would
# pick wide_resnet50_2 at 320px and should re-run this with those settings.
BACKBONE, IMG, KIND = "vit_base_patch14_dinov2.lvd142m", 392, "vit"
OUT_INDICES = (2, 3)          # CNN only; ignored when KIND == "vit"

CORESET_RATIO = 0.01
CAL_FRAC, CAL_SEED = 0.20, 0
PCTL = 99.0
COST_ESCAPE, COST_FALSE_ALARM = 100.0, 1.0

N_GRID = [2, 5, 10, 20, 40, 80, 160, None]   # None = every fit image available
SEEDS = [0, 1, 2]


# Sized from the box rather than hardcoded, but capped: past ~16 the decode is no longer
# the limit and extra threads just add contention.
_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))


class PatchExtractor:
    """Same extractor as sweep_backbones.py, restricted to the one chosen arm."""

    def __init__(self):
        if KIND == "cnn":
            self.model = timm.create_model(
                BACKBONE, pretrained=True, features_only=True,
                out_indices=OUT_INDICES).to(DEVICE).eval()
        else:
            self.model = timm.create_model(
                BACKBONE, pretrained=True, num_classes=0,
                img_size=IMG).to(DEVICE).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        cfg["input_size"] = (3, IMG, IMG)
        self.tfm = timm.data.create_transform(**cfg, is_training=False)
        self.grid = None
        self.dim = None

    def _one(self, im):
        return self.tfm(im.convert("RGB"))

    @torch.no_grad()
    def forward_feats(self, x):
        if KIND == "cnn":
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
    def __call__(self, pil_batch):
        # Decode and transform in parallel. Measured on the RTX 4000 Ada: with this done
        # serially the GPU was idle in 12 of 14 samples and peaked at 764 MiB of 20 GB -
        # the bottleneck was one CPU core doing PIL decode, not the card. PIL and the
        # torchvision transforms release the GIL in their C paths, so threads are enough
        # and avoid the pickling problems that fork-based workers hit with an
        # arrow-backed HF dataset. `executor.map` preserves input order, so the batch is
        # assembled identically to the serial version - this is a speed change only.
        x = torch.stack(list(_POOL.map(self._one, pil_batch))).to(DEVICE)
        return self.forward_feats(x)


@torch.no_grad()
def extract(ex, sub, image_col, indices, batch=16):
    out = []
    for i in range(0, len(indices), batch):
        out.append(ex([sub[j][image_col] for j in indices[i:i + batch]]))
    return torch.cat(out)


def coreset_indices(feats, ratio=CORESET_RATIO, proj_dim=128, seed=0, chunk=16384):
    n = feats.shape[0]
    k = max(int(n * ratio), 32)
    k = min(k, n)                      # tiny banks: cannot select more than exist
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


def operating(scores, cal_scores, truth):
    thr = float(np.percentile(cal_scores, PCTL))
    pred = (scores > thr).astype(int)
    fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    n_pos, n_neg = int((truth == 1).sum()), int((truth == 0).sum())
    return {"escapes": fn, "false_alarms": fp,
            "recall": float((n_pos - fn) / n_pos) if n_pos else None,
            "far": float(fp / n_neg) if n_neg else None,
            "cost": fn * COST_ESCAPE + fp * COST_FALSE_ALARM}


def main():
    os.makedirs("outputs", exist_ok=True)
    dd = load_dataset(DATASET_ID)
    parts = []
    for split_name, dset in dd.items():
        if "split" not in dset.column_names:
            dset = dset.add_column("split", [split_name] * len(dset))
        parts.append(dset)
    sub = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    def pick(c):
        return next((x for x in c if x in sub.column_names), None)

    image_col = pick(("image_path", "image", "img"))
    label_col = pick(("label", "labels", "is_anomaly"))
    object_col = pick(("object", "category", "class_name"))
    defect_col = pick(("defect", "defect_type", "anomaly_type"))
    split_col = pick(("split", "set"))
    assert all([image_col, label_col, object_col, defect_col, split_col])

    pairs = collections.Counter(zip(sub[defect_col], sub[label_col]))
    good_label = next(l for (d, l) in pairs if str(d).lower() == "good")

    OBJ, LABEL = sub[object_col], sub[label_col]
    SPLIT = [str(s).lower() for s in sub[split_col]]
    categories = sorted(set(OBJ))

    by_cat = {c: {"train": [], "test": []} for c in categories}
    for i in range(len(sub)):
        by_cat[OBJ[i]]["train" if "train" in SPLIT[i] else "test"].append(i)

    ex = PatchExtractor()
    summary = {
        "dataset": DATASET_ID, "backbone": BACKBONE, "img": IMG, "kind": KIND,
        "coreset_ratio": CORESET_RATIO,
        "calibration_split": {"frac": CAL_FRAC, "seed": CAL_SEED,
                              "note": "fixed across all N so the curve isolates bank size"},
        "threshold_rule": f"{PCTL}th percentile of held-out calibration scores",
        "n_grid": N_GRID, "seeds": SEEDS,
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "categories": {},
    }

    for cat in categories:
        t_cat = time.time()
        tr, te = by_cat[cat]["train"], by_cat[cat]["test"]

        # Fixed calibration hold-out, identical for every N and seed below.
        rng = np.random.default_rng(CAL_SEED)
        perm = rng.permutation(len(tr))
        n_cal = max(int(round(len(tr) * CAL_FRAC)), 1)
        cal = [tr[i] for i in perm[:n_cal]]
        pool = [tr[i] for i in perm[n_cal:]]

        # Extracted once, reused across every (N, seed). These do not depend on the bank.
        f_te = extract(ex, sub, image_col, te)
        f_cal = extract(ex, sub, image_col, cal)
        n_patch = ex.grid[0] * ex.grid[1]
        truth = np.array([0 if LABEL[i] == good_label else 1 for i in te])

        rec = {"n_pool": len(pool), "n_cal": len(cal), "n_test": len(te),
               "grid": list(ex.grid), "points": []}
        print(f"\n=== {cat}  pool {len(pool)}  cal {len(cal)}  test {len(te)} ===",
              flush=True)

        for N in N_GRID:
            n_use = len(pool) if N is None else min(N, len(pool))
            aurocs, costs, escapes = [], [], []
            for seed in SEEDS:
                r = np.random.default_rng(1000 + seed)
                take = [pool[i] for i in r.permutation(len(pool))[:n_use]]
                f_fit = extract(ex, sub, image_col, take)
                keep = coreset_indices(f_fit, seed=seed)
                bank = f_fit[keep]
                s_te = patch_distances(bank, f_te, n_patch).max(dim=1).values.numpy()
                s_cal = patch_distances(bank, f_cal, n_patch).max(dim=1).values.numpy()
                aurocs.append(float(roc_auc_score(truth, s_te)))
                op = operating(s_te, s_cal, truth)
                costs.append(op["cost"])
                escapes.append(op["escapes"])
                del f_fit, bank
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                if N is None:
                    break            # full pool is deterministic; one seed suffices

            rec["points"].append({
                "n_requested": N, "n_used": n_use, "seeds": len(aurocs),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "auroc_min": float(np.min(aurocs)), "auroc_max": float(np.max(aurocs)),
                "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                "escapes_mean": float(np.mean(escapes)),
            })
            p = rec["points"][-1]
            print(f"  N={str(N):>4}  used {n_use:4d}  AUROC {p['auroc_mean']:.4f} "
                  f"+/- {p['auroc_std']:.4f}  (min {p['auroc_min']:.4f})  "
                  f"cost {p['cost_mean']:8.0f}", flush=True)

        # The knee: fewest images reaching 99% of the full-pool AUROC.
        full = rec["points"][-1]["auroc_mean"]
        knee = next((p["n_used"] for p in rec["points"]
                     if p["auroc_mean"] >= 0.99 * full), None)
        rec["full_pool_auroc"] = full
        rec["knee_n_for_99pct_of_full"] = knee
        rec["seconds"] = round(time.time() - t_cat, 1)
        print(f"  -> knee at N={knee} for 99% of full-pool AUROC {full:.4f}"
              f"  ({rec['seconds']:.0f}s)", flush=True)

        summary["categories"][cat] = rec
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del f_te, f_cal
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    knees = [r["knee_n_for_99pct_of_full"] for r in summary["categories"].values()
             if r["knee_n_for_99pct_of_full"] is not None]
    summary["knee_median"] = float(np.median(knees)) if knees else None
    summary["knee_max"] = max(knees) if knees else None
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nmedian knee {summary['knee_median']}  worst {summary['knee_max']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
