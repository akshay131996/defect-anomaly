#!/usr/bin/env python3
"""Do the backbone rankings survive a correctly-chosen threshold?

Every arm comparison in this repo was scored at the 99th percentile, which the previous
experiment showed is 15x worse than a single global p50 at a 100:1 cost ratio. Rankings
established at a badly wrong operating point are not obviously the rankings you get at a
right one - an arm whose score distribution has a heavy normal tail is punished at p99 in a
way it would not be at p50, and vice versa.

So re-evaluate the arms across the full percentile grid and compare them where each is
actually operated: at the single global percentile that minimises total cost.

Two changes from exp_percentile_rule.py:

- The percentile grid extends down to 20. At 1000:1 the previous run put the optimum at
  the bottom edge of its grid (p50), which means the real optimum may be lower and was
  never seen. An optimum sitting on a boundary is not an optimum, it is a truncation.
- Four arms rather than six. B0 and C were far behind on every measure and at every seed;
  spending an hour to re-rank arms that lose by 40% is not worth the pod time.

    python exp_arms_at_optimal_threshold.py   -> outputs/exp_arms_optimal_threshold.json
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

OUT = "outputs/exp_arms_optimal_threshold.json"
CORESET_RATIO = 0.01
CAL_FRAC, CAL_SEED = 0.20, 0
SEEDS = [0, 1, 2]

PERCENTILES = [20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
COST_RATIOS = [10.0, 100.0, 1000.0]

ARMS = [a for a in sb.ARMS if a["tag"] in
        ("A_wrn50_224", "B1_dinov2_392", "D_wrn50_320", "E_resnet50_224")]


def cost_of(scores, truth, thr, ratio):
    pred = (scores > thr).astype(int)
    fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    return fn * ratio + fp, fn, fp


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
    categories = sorted(set(OBJ))

    by_cat = {c: {"train": [], "test": []} for c in categories}
    for i in range(len(sub)):
        by_cat[OBJ[i]]["train" if "train" in SPLIT[i] else "test"].append(i)

    summary = {"coreset_ratio": CORESET_RATIO, "seeds": SEEDS,
               "percentiles": PERCENTILES, "cost_ratios": COST_RATIOS,
               "device": torch.cuda.get_device_name(0) if sb.DEVICE == "cuda" else "cpu",
               "arms": {}}

    if os.path.exists(OUT):
        with open(OUT) as f:
            prev = json.load(f)
        if prev.get("percentiles") == PERCENTILES and prev.get("seeds") == SEEDS:
            summary["arms"] = prev.get("arms", {})
            print(f"resuming: {list(summary['arms'])}", flush=True)

    for spec in ARMS:
        spec = dict(spec)
        if spec["kind"] == "vit":
            spec["name"] = dino_id
        tag = spec["tag"]
        if tag in summary["arms"]:
            print(f"=== {tag} (cached) ===", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        t_arm = time.time()
        ex = sb.PatchExtractor(spec)
        arm = {"backbone": spec["name"], "img": spec["img"], "categories": {}}

        for cat in categories:
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

            per_seed, aurocs = [], []
            for seed in SEEDS:
                keep = sb.coreset_indices(f_fit, ratio=CORESET_RATIO, seed=seed)
                bank = f_fit[keep]
                s_te = sb.patch_distances(bank, f_te, n_patch).max(dim=1).values.numpy()
                s_cal = sb.patch_distances(bank, f_cal, n_patch).max(dim=1).values.numpy()
                per_seed.append((s_te, s_cal))
                aurocs.append(float(roc_auc_score(truth, s_te)))
                del bank
                if sb.DEVICE == "cuda":
                    torch.cuda.empty_cache()

            rec = {"auroc_mean": float(np.mean(aurocs)), "by_ratio": {}}
            for ratio in COST_RATIOS:
                curve = {str(p): float(np.mean(
                    [cost_of(s_te, truth, float(np.percentile(s_cal, p)), ratio)[0]
                     for s_te, s_cal in per_seed])) for p in PERCENTILES}
                oracle = []
                for s_te, _ in per_seed:
                    grid = np.unique(np.concatenate(
                        [s_te, np.linspace(s_te.min(), s_te.max(), 400)]))
                    oracle.append(min(cost_of(s_te, truth, t, ratio)[0] for t in grid))
                rec["by_ratio"][str(ratio)] = {
                    "curve": curve, "oracle_cost": float(np.mean(oracle))}
            arm["categories"][cat] = rec
            del f_fit, f_te, f_cal
            if sb.DEVICE == "cuda":
                torch.cuda.empty_cache()

        # Each arm is judged at ITS OWN best single percentile - the fair comparison,
        # since a deployment would tune that one number for whatever arm it shipped.
        arm["summary"] = {}
        for ratio in COST_RATIOS:
            rs = str(ratio)
            tot = {str(p): sum(arm["categories"][c]["by_ratio"][rs]["curve"][str(p)]
                               for c in categories) for p in PERCENTILES}
            best = min(tot, key=tot.get)
            arm["summary"][rs] = {
                "best_percentile": float(best),
                "cost_at_best": tot[best],
                "cost_at_p99": tot["99"],
                "oracle": sum(arm["categories"][c]["by_ratio"][rs]["oracle_cost"]
                              for c in categories),
                "curve_total": tot,
            }
        arm["mean_auroc"] = float(np.mean(
            [arm["categories"][c]["auroc_mean"] for c in categories]))
        arm["seconds"] = round(time.time() - t_arm, 1)
        s100 = arm["summary"]["100.0"]
        print(f"  -> AUROC {arm['mean_auroc']:.4f}   best p={s100['best_percentile']}"
              f"   cost {s100['cost_at_best']:.0f}   (p99 {s100['cost_at_p99']:.0f},"
              f" oracle {s100['oracle']:.0f})   {arm['seconds']:.0f}s", flush=True)

        summary["arms"][tag] = arm
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)
        del ex
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n=== ranking at 100:1, each arm at its own best percentile ===")
    rows = [(a["summary"]["100.0"]["cost_at_best"], t, a) for t, a in summary["arms"].items()]
    for cost, tag, a in sorted(rows):
        s = a["summary"]["100.0"]
        print(f"{tag:<18} p={s['best_percentile']:<6} cost {cost:>8.0f}"
              f"   was(p99) {s['cost_at_p99']:>8.0f}   AUROC {a['mean_auroc']:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
