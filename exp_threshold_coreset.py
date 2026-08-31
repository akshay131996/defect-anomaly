#!/usr/bin/env python3
"""Two knobs the project fixed by assumption and never tested.

**1. Coreset ratio.** Every run so far used 1% because the PatchCore paper does. That is a
deployment parameter - it sets bank size, so it sets search latency and memory - and it is
also the prime suspect for the seed instability the audit found: a bigger coreset should
depend less on where greedy k-center starts. If 1% is costing accuracy or stability for no
latency benefit, that is worth knowing before anything ships.

**2. Calibration size and threshold rule.** Session 3 varied the bank and held calibration
fixed at 20% of train, which isolates one question but leaves the other untouched. The
threshold is what actually makes the decision. `toothbrush` calibrates a 99th percentile
from 12 images, which cannot be sound - a 99th percentile of 12 points is barely
distinguishable from the maximum.

Design notes:

- The bank is built ONCE per (category, ratio, seed) and every training image is scored
  against it in one pass. Any calibration subset is then a subset of scores already
  computed, so the calibration sweep is nearly free.
- Calibration pool is 40% of train rather than 20%, so n_cal can reach 40+ without
  running out of images. The bank comes from the other 60%.
- Three threshold rules, because the percentile is arbitrary:
    p99      - what the project has used throughout
    max      - the most conservative rule that uses only defect-free data
    mu+3sd   - assumes approximate normality, which patch-max scores likely violate;
               included precisely to see whether that assumption costs anything.

    python exp_threshold_coreset.py      -> outputs/exp_threshold_coreset.json
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

OUT = "outputs/exp_threshold_coreset.json"

# Arm A - the published recipe, the reference every other arm is measured against.
ARM = {"tag": "A_wrn50_224", "kind": "cnn", "name": "wide_resnet50_2", "img": 224,
       "out_indices": (2, 3)}

# The four cost-carrying categories plus two controls.
CATEGORIES = ["screw", "capsule", "pill", "cable", "grid", "bottle"]

CORESET_RATIOS = [0.002, 0.01, 0.05, 0.25]   # 1.0 dropped: no coreset at all
                                             # is 4x the cost of 0.25 and the
                                             # trend is already unambiguous
SEEDS = [0, 1, 2]
CAL_POOL_FRAC = 0.40
N_CAL_GRID = [5, 10, 20, 40, None]        # None = the whole calibration pool
CAL_DRAWS = [0, 1, 2, 3, 4]               # which images land in a small calibration set
RULES = ["p99", "max", "mu3sd"]

COST_ESCAPE, COST_FALSE_ALARM = 100.0, 1.0


def threshold(cal_scores, rule):
    if rule == "p99":
        return float(np.percentile(cal_scores, 99.0))
    if rule == "max":
        return float(np.max(cal_scores))
    if rule == "mu3sd":
        return float(np.mean(cal_scores) + 3.0 * np.std(cal_scores))
    raise ValueError(rule)


def decide(scores, truth, thr):
    pred = (scores > thr).astype(int)
    fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    return fn, fp, fn * COST_ESCAPE + fp * COST_FALSE_ALARM


def main():
    os.makedirs("outputs", exist_ok=True)
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

    ex = sb.PatchExtractor(ARM)
    summary = {
        "arm": ARM["tag"], "backbone": ARM["name"], "img": ARM["img"],
        "categories": CATEGORIES, "coreset_ratios": CORESET_RATIOS, "seeds": SEEDS,
        "cal_pool_frac": CAL_POOL_FRAC, "n_cal_grid": N_CAL_GRID, "rules": RULES,
        "cost_ratio_escape_to_false_alarm": COST_ESCAPE,
        "results": {},
    }

    for cat in CATEGORIES:
        t_cat = time.time()
        tr, te = by_cat[cat]["train"], by_cat[cat]["test"]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(tr))
        n_pool = max(int(round(len(tr) * CAL_POOL_FRAC)), 2)
        cal_pool = [tr[i] for i in perm[:n_pool]]
        fit = [tr[i] for i in perm[n_pool:]]

        f_fit = sb.extract(ex, sub, image_col, fit)
        f_te = sb.extract(ex, sub, image_col, te)
        f_cal = sb.extract(ex, sub, image_col, cal_pool)
        n_patch = ex.grid[0] * ex.grid[1]
        truth = np.array([0 if LABEL[i] == good_label else 1 for i in te])
        print(f"\n=== {cat}  fit {len(fit)}  cal_pool {len(cal_pool)}  test {len(te)} "
              f"patches/img {n_patch} ===", flush=True)

        rec = {"n_fit": len(fit), "n_cal_pool": len(cal_pool), "n_test": len(te),
               "patches_per_image": n_patch, "by_ratio": {}}

        for ratio in CORESET_RATIOS:
            aurocs, bank_sizes = [], []
            # rule -> n_cal -> list of costs over (seed, draw)
            costs = {r: {str(n): [] for n in N_CAL_GRID} for r in RULES}
            escapes = {r: {str(n): [] for n in N_CAL_GRID} for r in RULES}

            for seed in SEEDS:
                keep = sb.coreset_indices(f_fit, ratio=ratio, seed=seed)
                bank = f_fit[keep]
                bank_sizes.append(len(keep))
                s_te = sb.patch_distances(bank, f_te, n_patch).max(dim=1).values.numpy()
                s_cal_all = sb.patch_distances(
                    bank, f_cal, n_patch).max(dim=1).values.numpy()
                aurocs.append(float(roc_auc_score(truth, s_te)))

                for n_cal in N_CAL_GRID:
                    use = len(s_cal_all) if n_cal is None else min(n_cal, len(s_cal_all))
                    draws = [0] if n_cal is None else CAL_DRAWS
                    for draw in draws:
                        r = np.random.default_rng(500 + draw)
                        idx = r.permutation(len(s_cal_all))[:use]
                        cs = s_cal_all[idx]
                        for rule in RULES:
                            fn, fp, c = decide(s_te, truth, threshold(cs, rule))
                            costs[rule][str(n_cal)].append(c)
                            escapes[rule][str(n_cal)].append(fn)
                del bank
                if sb.DEVICE == "cuda":
                    torch.cuda.empty_cache()

            entry = {
                "bank_size_mean": float(np.mean(bank_sizes)),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_spread": float(np.max(aurocs) - np.min(aurocs)),
                "threshold_sweep": {
                    rule: {n: {"cost_mean": float(np.mean(v)),
                               "cost_std": float(np.std(v)),
                               "escapes_mean": float(np.mean(escapes[rule][n]))}
                           for n, v in per_n.items() if v}
                    for rule, per_n in costs.items()
                },
            }
            rec["by_ratio"][str(ratio)] = entry
            full = entry["threshold_sweep"]["p99"]["None"]
            print(f"  ratio {ratio:<6} bank {entry['bank_size_mean']:>7.0f}"
                  f"  AUROC {entry['auroc_mean']:.4f} (spread {entry['auroc_spread']:.4f})"
                  f"  p99/full-cal cost {full['cost_mean']:8.0f}", flush=True)

        rec["seconds"] = round(time.time() - t_cat, 1)
        summary["results"][cat] = rec
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del f_fit, f_te, f_cal
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
