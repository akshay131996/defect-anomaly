#!/usr/bin/env python3
"""How much of a result is the coreset draw?

The sweep reported `screw` at 0.8795 for DINOv2 @392; session 3 got 0.8252 at full pool on
the same data and the same configuration. The only difference was the greedy k-center
starting point, which is seeded. That is a large gap for a supposedly deterministic method,
and it means every single-seed `screw` figure carries uncertainty the other numbers may not.

This re-runs each arm with several coreset seeds and reports the spread. `screw` is the
suspect; the other three categories are controls chosen to span the difficulty range, so
the result can distinguish "screw is unstable" from "everything is unstable and we never
looked".

Reuses sweep_backbones.py rather than reimplementing it, so the two cannot drift.

    python seed_variance.py            -> outputs/seed_variance.json
"""
import collections
import json
import os
import time

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import roc_auc_score

import sweep_backbones as sb

OUT = "outputs/seed_variance.json"
CATEGORIES = ["screw", "capsule", "grid", "bottle"]   # suspect + 3 controls
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs("outputs", exist_ok=True)
    dino_id = sb.resolve_dinov2()

    dd = load_dataset(sb.DATASET_ID)
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

    pairs = collections.Counter(zip(sub[defect_col], sub[label_col]))
    good_label = next(l for (d, l) in pairs if str(d).lower() == "good")

    OBJ, LABEL = sub[object_col], sub[label_col]
    SPLIT = [str(s).lower() for s in sub[split_col]]
    by_cat = {c: {"train": [], "test": []} for c in CATEGORIES}
    for i in range(len(sub)):
        if OBJ[i] in by_cat:
            by_cat[OBJ[i]]["train" if "train" in SPLIT[i] else "test"].append(i)

    summary = {"seeds": SEEDS, "categories": CATEGORIES,
               "note": "identical data and config per row; only the coreset seed varies",
               "arms": {}}

    for spec in sb.ARMS:
        spec = dict(spec)
        if spec["kind"] == "vit":
            spec["name"] = dino_id
        tag = spec["tag"]
        print(f"\n=== {tag} ===", flush=True)
        ex = sb.PatchExtractor(spec)
        arm = {}

        for cat in CATEGORIES:
            t0 = time.time()
            tr, te = by_cat[cat]["train"], by_cat[cat]["test"]
            rng = np.random.default_rng(sb.CAL_SEED)
            perm = rng.permutation(len(tr))
            n_cal = max(int(round(len(tr) * sb.CAL_FRAC)), 1)
            cal = [tr[i] for i in perm[:n_cal]]
            fit = [tr[i] for i in perm[n_cal:]]

            # Features do not depend on the seed - extract once, reseed only the coreset.
            f_fit = sb.extract(ex, sub, image_col, fit)
            f_te = sb.extract(ex, sub, image_col, te)
            f_cal = sb.extract(ex, sub, image_col, cal)
            n_patch = ex.grid[0] * ex.grid[1]
            truth = np.array([0 if LABEL[i] == good_label else 1 for i in te])

            aurocs, costs, escapes = [], [], []
            for seed in SEEDS:
                keep = sb.coreset_indices(f_fit, seed=seed)
                bank = f_fit[keep]
                s_te = sb.patch_distances(bank, f_te, n_patch).max(dim=1).values.numpy()
                s_cal = sb.patch_distances(bank, f_cal, n_patch).max(dim=1).values.numpy()
                aurocs.append(float(roc_auc_score(truth, s_te)))
                op = sb.operating(s_te, s_cal, truth, sb.PCTL)
                costs.append(op["cost"])
                escapes.append(op["escapes"])
                del bank
                if sb.DEVICE == "cuda":
                    torch.cuda.empty_cache()

            arm[cat] = {
                "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
                "auroc_min": float(np.min(aurocs)), "auroc_max": float(np.max(aurocs)),
                "auroc_all": aurocs,
                "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                "escapes_all": escapes,
                "seconds": round(time.time() - t0, 1),
            }
            a = arm[cat]
            print(f"  {cat:<10} AUROC {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}"
                  f"   range [{a['auroc_min']:.4f}, {a['auroc_max']:.4f}]"
                  f"   spread {a['auroc_max'] - a['auroc_min']:.4f}"
                  f"   escapes {escapes}   {a['seconds']:.0f}s", flush=True)
            del f_fit, f_te, f_cal
            if sb.DEVICE == "cuda":
                torch.cuda.empty_cache()

        summary["arms"][tag] = arm
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del ex
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n--- spread (max - min AUROC over seeds) ---")
    print(f"{'arm':<18}" + "".join(f"{c:>11}" for c in CATEGORIES))
    for tag, arm in summary["arms"].items():
        row = f"{tag:<18}"
        for c in CATEGORIES:
            row += f"{arm[c]['auroc_max'] - arm[c]['auroc_min']:>11.4f}"
        print(row)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
