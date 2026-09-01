#!/usr/bin/env python3
"""Which percentile should the threshold use, and can it be derived without labels?

The previous experiment showed the 99th percentile is systematically too high for a 100:1
escape-to-false-alarm ratio: more calibration data raised the threshold and roughly tripled
escapes. The rule was wrong, not the data.

So sweep the percentile directly, at three cost ratios, on all 15 categories, and ask three
questions:

1. What percentile actually minimises cost at each ratio?
2. Is there ONE percentile that works across categories, or is it category-specific? A
   single number would be deployable; a per-category number would need defect labels to
   find, which a commissioning line does not have.
3. How much of the oracle gap does the best fixed percentile close? The oracle - the best
   threshold chosen with full knowledge of the test labels - is not achievable, but it
   bounds what any rule could win.

Scores are computed once per category and every percentile/ratio combination is then
arithmetic on arrays already in memory, so the sweep itself is free.

    python exp_percentile_rule.py        -> outputs/exp_percentile_rule.json
"""
import collections
import json
import os
import time

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset

import sweep_backbones as sb

OUT = "outputs/exp_percentile_rule.json"
ARM = {"tag": "A_wrn50_224", "kind": "cnn", "name": "wide_resnet50_2", "img": 224,
       "out_indices": (2, 3)}
CORESET_RATIO = 0.01
CAL_FRAC, CAL_SEED = 0.20, 0
SEEDS = [0, 1, 2]

PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 93, 95, 97, 98, 99, 99.5, 100]
COST_RATIOS = [10.0, 100.0, 1000.0]


def cost_of(scores, truth, thr, ratio):
    pred = (scores > thr).astype(int)
    fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    return fn * ratio + fp, fn, fp


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
    categories = sorted(set(OBJ))

    by_cat = {c: {"train": [], "test": []} for c in categories}
    for i in range(len(sub)):
        by_cat[OBJ[i]]["train" if "train" in SPLIT[i] else "test"].append(i)

    ex = sb.PatchExtractor(ARM)
    # Record the GPU: everything before this experiment ran on an RTX 4000 Ada and this
    # runs on an RTX A4000. Results should match - same seeds, same deterministic ops - but
    # cdist can differ in the last bits across architectures, so if a number moves this is
    # where to look first.
    summary = {"arm": ARM["tag"], "coreset_ratio": CORESET_RATIO, "seeds": SEEDS,
               "percentiles": PERCENTILES, "cost_ratios": COST_RATIOS,
               "device": torch.cuda.get_device_name(0) if sb.DEVICE == "cuda" else "cpu",
               "categories": {}}

    for cat in categories:
        t0 = time.time()
        tr, te = by_cat[cat]["train"], by_cat[cat]["test"]
        rng = np.random.default_rng(CAL_SEED)
        perm = rng.permutation(len(tr))
        n_cal = max(int(round(len(tr) * CAL_FRAC)), 1)
        cal = [tr[i] for i in perm[:n_cal]]
        fit = [tr[i] for i in perm[n_cal:]]

        f_fit = sb.extract(ex, sub, image_col, fit)
        f_te = sb.extract(ex, sub, image_col, te)
        f_cal = sb.extract(ex, sub, image_col, cal)
        n_patch = ex.grid[0] * ex.grid[1]
        truth = np.array([0 if LABEL[i] == good_label else 1 for i in te])

        # Average the cost curve over coreset seeds, so a lucky draw cannot pick the winner.
        per_seed = []
        for seed in SEEDS:
            keep = sb.coreset_indices(f_fit, ratio=CORESET_RATIO, seed=seed)
            bank = f_fit[keep]
            s_te = sb.patch_distances(bank, f_te, n_patch).max(dim=1).values.numpy()
            s_cal = sb.patch_distances(bank, f_cal, n_patch).max(dim=1).values.numpy()
            per_seed.append((s_te, s_cal))
            del bank
            if sb.DEVICE == "cuda":
                torch.cuda.empty_cache()

        rec = {"n_test": len(te), "n_cal": len(cal), "by_ratio": {}}
        for ratio in COST_RATIOS:
            curve = {}
            for p in PERCENTILES:
                cs = [cost_of(s_te, truth, float(np.percentile(s_cal, p)), ratio)[0]
                      for s_te, s_cal in per_seed]
                curve[str(p)] = float(np.mean(cs))
            # Oracle: best threshold with full knowledge of the test labels. Not
            # achievable - it bounds what any label-free rule could possibly win.
            oracle = []
            for s_te, _ in per_seed:
                grid = np.unique(np.concatenate([s_te, np.linspace(s_te.min(), s_te.max(), 400)]))
                oracle.append(min(cost_of(s_te, truth, t, ratio)[0] for t in grid))
            best_p = min(curve, key=curve.get)
            rec["by_ratio"][str(ratio)] = {
                "curve": curve,
                "best_percentile": float(best_p),
                "best_cost": curve[best_p],
                "cost_at_p99": curve["99"],
                "oracle_cost": float(np.mean(oracle)),
            }
        rec["seconds"] = round(time.time() - t0, 1)
        summary["categories"][cat] = rec
        r100 = rec["by_ratio"]["100.0"]
        print(f"{cat:<12} best p={r100['best_percentile']:<5} "
              f"cost {r100['best_cost']:8.0f}   p99 {r100['cost_at_p99']:8.0f}   "
              f"oracle {r100['oracle_cost']:8.0f}   ({rec['seconds']:.0f}s)", flush=True)

        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del f_fit, f_te, f_cal
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Is one percentile good enough for everyone?
    for ratio in COST_RATIOS:
        rs = str(ratio)
        tot = {str(p): sum(summary["categories"][c]["by_ratio"][rs]["curve"][str(p)]
                           for c in summary["categories"]) for p in PERCENTILES}
        best = min(tot, key=tot.get)
        per_cat_best = sum(summary["categories"][c]["by_ratio"][rs]["best_cost"]
                           for c in summary["categories"])
        oracle = sum(summary["categories"][c]["by_ratio"][rs]["oracle_cost"]
                     for c in summary["categories"])
        summary.setdefault("global", {})[rs] = {
            "best_single_percentile": float(best),
            "cost_at_best_single": tot[best],
            "cost_at_p99": tot["99"],
            "cost_if_percentile_chosen_per_category": per_cat_best,
            "oracle_cost": oracle,
        }
        print(f"\nratio {ratio:.0f}:1  best single percentile {best}"
              f"  cost {tot[best]:.0f}  (p99 {tot['99']:.0f},"
              f" per-category {per_cat_best:.0f}, oracle {oracle:.0f})")

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
