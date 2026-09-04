#!/usr/bin/env python3
"""Is MVTec AD 2 hard because of small defects, or because of distribution shift?

Symptom that prompted this: on `can`, image AUROC sits at ~0.5 across 224/448/768 px and
AU-PRO plateaus after 448. If small defects were the whole problem, resolution would keep
paying. It stops, so something else dominates.

Hypothesis: AD 2 deliberately varies lighting between train and test. A nearest-neighbour
-to-train method scores *any* shifted image as anomalous, defect or not, which would
compress the gap between good and bad test images toward nothing.

The test needs no labels beyond the splits themselves. Score three defect-free sets
against the same bank:

    validation/good     same conditions as train  -> the in-distribution reference
    test_public/good    defect-free but from test -> shifted, if the hypothesis holds
    test_public/bad     defective                 -> what we actually want to flag

If the hypothesis is right, test-good scores sit far above validation-good and close to
test-bad: the method is mostly reporting "this photo looks different", not "this part is
broken". If it is wrong, test-good and validation-good will overlap and the defect signal
is simply weak.

    python ad2_shift_check.py [--scenarios can,fabric] [--img 448]
"""
import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import sweep_backbones as sb

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
OUT = "outputs/ad2_shift_check.json"
ARM = {"tag": "wrn50", "kind": "cnn", "name": "wide_resnet50_2", "img": 448,
       "out_indices": (2, 3)}
_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 8)))


def png(*parts):
    return sorted(glob.glob(os.path.join(AD2_ROOT, *parts, "**", "*.png"), recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--img", type=int, default=448)
    ap.add_argument("--bank-cap", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)

    arm = dict(ARM, img=args.img)
    ex = sb.PatchExtractor(arm)
    scenarios = ([s for s in args.scenarios.split(",") if s] or
                 sorted(d for d in os.listdir(AD2_ROOT)
                        if os.path.isdir(os.path.join(AD2_ROOT, d))))
    summary = {"img": args.img, "bank_cap": args.bank_cap, "scenarios": {}}

    print(f"{'scenario':<13}{'val_good':>10}{'test_good':>11}{'test_bad':>10}"
          f"{'shift':>9}{'signal':>9}")
    for sc in scenarios:
        train, val = png(sc, "train"), png(sc, "validation")
        tg, tb = png(sc, "test_public", "good"), png(sc, "test_public", "bad")
        if not (train and val and tg and tb):
            continue
        f_tr = sb.extract_paths(ex, train, _POOL)
        n_patch = ex.grid[0] * ex.grid[1]
        keep = sb.coreset_indices(f_tr, seed=0, max_k=args.bank_cap)
        bank = f_tr[keep]
        del f_tr

        def score(paths):
            f = sb.extract_paths(ex, paths, _POOL)
            return sb.patch_distances(bank, f, n_patch).max(dim=1).values.numpy()

        s_val, s_tg, s_tb = score(val), score(tg), score(tb)
        # "shift" = how far test-good has moved from validation-good, in units of
        # validation's own spread. "signal" = the same distance for the defect gap.
        sd = float(np.std(s_val)) or 1e-9
        shift = float((s_tg.mean() - s_val.mean()) / sd)
        signal = float((s_tb.mean() - s_tg.mean()) / sd)
        summary["scenarios"][sc] = {
            "val_good_mean": float(s_val.mean()), "test_good_mean": float(s_tg.mean()),
            "test_bad_mean": float(s_tb.mean()), "val_good_std": sd,
            "shift_sigmas": shift, "defect_signal_sigmas": signal,
            "n": [len(val), len(tg), len(tb)],
        }
        print(f"{sc:<13}{s_val.mean():>10.2f}{s_tg.mean():>11.2f}{s_tb.mean():>10.2f}"
              f"{shift:>8.1f}s{signal:>8.1f}s", flush=True)
        del bank
        if sb.DEVICE == "cuda":
            torch.cuda.empty_cache()
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)

    v = list(summary["scenarios"].values())
    if v:
        ms = float(np.mean([x["shift_sigmas"] for x in v]))
        mg = float(np.mean([x["defect_signal_sigmas"] for x in v]))
        summary["mean_shift_sigmas"] = ms
        summary["mean_defect_signal_sigmas"] = mg
        summary["shift_to_signal_ratio"] = ms / mg if mg else None
        print(f"\nmean shift {ms:.1f} sigma   mean defect signal {mg:.1f} sigma"
              f"   ratio {ms/mg if mg else float('nan'):.1f}x")
        print("A ratio >> 1 means the method is mostly reporting a lighting change,"
              "\nnot a defect.")
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
