#!/usr/bin/env python3
"""E5a: Measure native defect region size distribution vs patch cell size on MVTec AD 2.

No GPU required. Runs in seconds on CPU over ground-truth masks.
Verifies the hypothesis:
'a majority of regions are sub-cell at img = 448'.
"""
import os
import glob
import json
import numpy as np
from PIL import Image
from scipy import ndimage

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
NATIVE_MIN_REGION_PX = 77
SCENARIOS = [
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"
]


def aspect_dimensions(w_nat, h_nat, target_img=448, stride=32):
    r = w_nat / h_nat
    target_area = target_img * target_img
    h_float = (target_area / r) ** 0.5
    w_float = r * h_float
    h_in = max(stride, int(round(h_float / stride) * stride))
    w_in = max(stride, int(round(w_float / stride) * stride))
    return w_in, h_in


def find_mask(gt_dir, bad_path):
    stem = os.path.splitext(os.path.basename(bad_path))[0]
    hits = glob.glob(os.path.join(gt_dir, "**", stem + "*"), recursive=True)
    hits = [h for h in hits if h.lower().endswith((".png", ".bmp", ".tif", ".tiff"))]
    return hits[0] if hits else None


def analyze_scenario(scenario):
    sc_dir = os.path.join(AD2_ROOT, scenario)
    bad_paths = sorted(glob.glob(os.path.join(sc_dir, "test_public", "bad", "**", "*.png"), recursive=True))
    gt_dir = os.path.join(sc_dir, "test_public", "ground_truth")
    
    masks_paths = [find_mask(gt_dir, b) for b in bad_paths]
    have = [i for i, m in enumerate(masks_paths) if m]
    masks_paths = [masks_paths[i] for i in have]

    sample_im = Image.open(masks_paths[0])
    w_nat, h_nat = sample_im.size
    w_in, h_in = aspect_dimensions(w_nat, h_nat, target_img=448, stride=32)
    grid_h = h_in // 8
    grid_w = w_in // 8
    n_patches = grid_h * grid_w
    cell_area = (w_nat * h_nat) / n_patches

    region_sizes = []
    for p in masks_paths:
        im = Image.open(p).convert("L")
        mask = np.array(im) > 127
        labelled, n_comp = ndimage.label(mask)
        for r in range(1, n_comp + 1):
            sz = int((labelled == r).sum())
            if sz >= NATIVE_MIN_REGION_PX:
                region_sizes.append(sz)

    region_sizes = np.array(region_sizes)
    n = len(region_sizes)
    
    q25, med, q75 = np.percentile(region_sizes, [25, 50, 75])
    
    lt_1x = np.sum(region_sizes < cell_area)
    lt_2x = np.sum(region_sizes < 2 * cell_area)
    lt_4x = np.sum(region_sizes < 4 * cell_area)
    b_1_4x = np.sum((region_sizes >= cell_area) & (region_sizes < 4 * cell_area))
    b_4_16x = np.sum((region_sizes >= 4 * cell_area) & (region_sizes < 16 * cell_area))
    ge_16x = np.sum(region_sizes >= 16 * cell_area)

    return {
        "scenario": scenario,
        "n_regions": n,
        "native_size": [w_nat, h_nat],
        "input_aspect": [w_in, h_in],
        "grid": [grid_h, grid_w],
        "n_patches": n_patches,
        "cell_area": float(cell_area),
        "median_px": float(med),
        "q25_px": float(q25),
        "q75_px": float(q75),
        "iqr_px": float(q75 - q25),
        "frac_sub_cell (<1x)": float(lt_1x / n),
        "frac_<2x": float(lt_2x / n),
        "frac_<4x": float(lt_4x / n),
        "frac_1_to_4x": float(b_1_4x / n),
        "frac_4_to_16x": float(b_4_16x / n),
        "frac_ge_16x": float(ge_16x / n),
        "counts": {
            "sub_cell": int(lt_1x),
            "1_to_4x": int(b_1_4x),
            "4_to_16x": int(b_4_16x),
            "ge_16x": int(ge_16x),
        },
        "all_sizes": region_sizes.tolist(),
    }


def main():
    results = {}
    all_regions = []
    all_sub_cell = 0
    all_1_to_4x = 0
    all_4_to_16x = 0
    all_ge_16x = 0

    print(f"{'Scenario':<14} {'Regs':<6} {'CellPx':<8} {'MedPx':<8} {'IQR':<8} {'<1x(sub)':<10} {'1-4x':<10} {'4-16x':<10} {'>=16x':<8}")
    print("-" * 86)

    for sc in SCENARIOS:
        res = analyze_scenario(sc)
        results[sc] = res
        all_regions.extend(res["all_sizes"])
        all_sub_cell += res["counts"]["sub_cell"]
        all_1_to_4x += res["counts"]["1_to_4x"]
        all_4_to_16x += res["counts"]["4_to_16x"]
        all_ge_16x += res["counts"]["ge_16x"]
        print(f"{sc:<14} {res['n_regions']:<6} {res['cell_area']:<8.1f} {res['median_px']:<8.1f} {res['iqr_px']:<8.1f} "
              f"{res['frac_sub_cell (<1x)']*100:<9.1f}% {res['frac_1_to_4x']*100:<9.1f}% {res['frac_4_to_16x']*100:<9.1f}% {res['frac_ge_16x']*100:<7.1f}%")

    total_n = len(all_regions)
    all_regions = np.array(all_regions)
    q25, med, q75 = np.percentile(all_regions, [25, 50, 75])

    print("-" * 86)
    print(f"{'TOTAL':<14} {total_n:<6} {'---':<8} {med:<8.1f} {q75-q25:<8.1f} "
          f"{all_sub_cell/total_n*100:<9.1f}% {all_1_to_4x/total_n*100:<9.1f}% {all_4_to_16x/total_n*100:<9.1f}% {all_ge_16x/total_n*100:<7.1f}%")
    print("-" * 86)
    print(f"Total regions asserted: {total_n} (matches 1530: {total_n == 1530})")

    summary = {
        "total_regions": total_n,
        "median_px": float(med),
        "q25_px": float(q25),
        "q75_px": float(q75),
        "iqr_px": float(q75 - q25),
        "counts": {
            "sub_cell": all_sub_cell,
            "1_to_4x": all_1_to_4x,
            "4_to_16x": all_4_to_16x,
            "ge_16x": all_ge_16x,
        },
        "fractions": {
            "sub_cell": float(all_sub_cell / total_n),
            "1_to_4x": float(all_1_to_4x / total_n),
            "4_to_16x": float(all_4_to_16x / total_n),
            "ge_16x": float(all_ge_16x / total_n),
        },
        "scenarios": results,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/exp_e5a_region_sizes.json", "w") as f:
        clean_summary = dict(summary)
        clean_summary["scenarios"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "all_sizes"}
            for k, v in results.items()
        }
        json.dump(clean_summary, f, indent=2)
    print("Saved outputs/exp_e5a_region_sizes.json")


if __name__ == "__main__":
    main()
