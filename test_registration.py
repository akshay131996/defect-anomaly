#!/usr/bin/env python3
"""Registration unit test for anomaly detection coordinate frames (E0).

The coordinate-frame bug survived four sessions because no unit test asserted
that an anomaly map and a ground-truth mask describe the same spatial location.
On non-square industrial images (e.g. sheet_metal at 4224x1056, 4:1 aspect ratio),
center-cropping the image while full-frame resizing the mask causes spatial
misregistration: defects near edges fall completely outside the crop, and defects
inside the crop are stretched relative to the mask.

This test constructs synthetic non-square images with defects at known normalized
locations (including off-center and near frame edges). It tests whether an
otherwise perfect detector registers with the mask under:
  1. 'crop': timm-style resize-short-side and center-crop.
     Fails the test (< 0.20 AU-PRO@5%).
  2. 'squash': direct full-frame resize to (img, img).
     Passes the test (> 0.95 AU-PRO@5%).
  3. 'squash_shifted': squashed map shifted by 20% of width.
     Collapses (< 0.20 AU-PRO@5%), proving the test is sensitive to misalignment.
  4. 'letterbox': aspect-preserving resize with symmetrical padding.
     Passes the test (> 0.95 AU-PRO@5%).
  5. 'letterbox_shifted': letterboxed map shifted by 20% of width.
     Collapses (< 0.20 AU-PRO@5%).

Hypothesis (E0):
  The old default geometry (crop) fails the first assertion (AU-PRO@5% < 0.20),
  while registered geometries (squash, letterbox) pass (> 0.95).
"""
import sys
import numpy as np
from PIL import Image
from aupro import evaluate

H0, W0 = 1056, 4224
EVAL_SIDE = 512
IMG = 448
RNG = np.random.default_rng(42)


def make_synthetic_data():
    """Builds non-square masks with defects placed near the edge and off-center."""
    mask_native = np.zeros((H0, W0), bool)

    # Defect 1: Near the right frame edge (x in [0.75, 0.85], y in [0.2, 0.4])
    # Outside the center crop window [0.375, 0.625]!
    mask_native[int(0.20 * H0):int(0.40 * H0), int(0.75 * W0):int(0.85 * W0)] = True

    # Defect 2: Off-center defect inside crop window (x in [0.52, 0.58], y in [0.6, 0.8])
    mask_native[int(0.60 * H0):int(0.80 * H0), int(0.52 * W0):int(0.58 * W0)] = True

    return mask_native


def simulate_geometry(mask_native, geometry="squash", shift_pct=0.0):
    """Simulates how the detector's anomaly map and the evaluator's mask are formed."""
    noise_bad = RNG.random((EVAL_SIDE, EVAL_SIDE)) * 0.05
    noise_good = RNG.random((EVAL_SIDE, EVAL_SIDE)) * 0.05

    if geometry == "squash":
        # Full-frame mask squashed directly to EVAL_SIDE
        mask_eval = np.array(
            Image.fromarray(mask_native).resize((EVAL_SIDE, EVAL_SIDE), Image.NEAREST)
        ) > 0

        # In squash mode, perfect detector sees full normalized coordinates
        map_bad = mask_eval.astype(float) + noise_bad

        if shift_pct != 0.0:
            shift_px = int(shift_pct * EVAL_SIDE)
            map_bad = np.roll(map_bad, shift_px, axis=1)

        return map_bad, mask_eval, noise_good

    elif geometry == "crop":
        # Ground-truth mask is squashed full-frame to EVAL_SIDE (evaluator behavior)
        mask_eval = np.array(
            Image.fromarray(mask_native).resize((EVAL_SIDE, EVAL_SIDE), Image.NEAREST)
        ) > 0

        # Model only sees the center crop:
        # Short side (1056) -> 448; Long side (4224) -> 1792
        # Crop window is x in [672, 1120], i.e. normalized [0.375, 0.625]
        crop_x_min, crop_x_max = 0.375, 0.625
        map_crop = np.zeros((EVAL_SIDE, EVAL_SIDE), float)

        # Defect 1 (0.75..0.85) is completely outside the crop window -> invisible!
        # Defect 2 (0.52..0.58) is inside crop window:
        # In crop coords: x_norm_crop = (x - crop_x_min) / (crop_x_max - crop_x_min)
        # 0.52 -> (0.52 - 0.375) / 0.25 = 0.58; 0.58 -> (0.58 - 0.375) / 0.25 = 0.82
        x1_c = int((0.52 - crop_x_min) / (crop_x_max - crop_x_min) * EVAL_SIDE)
        x2_c = int((0.58 - crop_x_min) / (crop_x_max - crop_x_min) * EVAL_SIDE)
        y1_c = int(0.60 * EVAL_SIDE)
        y2_c = int(0.80 * EVAL_SIDE)

        map_crop[y1_c:y2_c, x1_c:x2_c] = 1.0
        map_bad = map_crop + noise_bad
        return map_bad, mask_eval, noise_good

    elif geometry == "letterbox":
        # Long side 4224 -> 512, Short side 1056 -> 128
        # Symmetric vertical padding: (512 - 128) // 2 = 192 px
        new_w = EVAL_SIDE
        new_h = int(H0 * EVAL_SIDE / W0)
        pad_top = (EVAL_SIDE - new_h) // 2

        mask_pil = Image.fromarray(mask_native).resize((new_w, new_h), Image.NEAREST)
        mask_eval = np.zeros((EVAL_SIDE, EVAL_SIDE), bool)
        mask_eval[pad_top:pad_top + new_h, :] = np.array(mask_pil) > 0

        map_bad = mask_eval.astype(float) + noise_bad

        if shift_pct != 0.0:
            shift_px = int(shift_pct * EVAL_SIDE)
            map_bad = np.roll(map_bad, shift_px, axis=1)

        return map_bad, mask_eval, noise_good

    elif geometry == "aspect":
        # Non-square rectangular frame preserving native aspect ratio (e.g. 896x224)
        aspect_w, aspect_h = 896, 224
        noise_bad = RNG.random((aspect_h, aspect_w)) * 0.05
        noise_good = RNG.random((aspect_h, aspect_w)) * 0.05

        mask_eval = np.array(
            Image.fromarray(mask_native).resize((aspect_w, aspect_h), Image.NEAREST)
        ) > 0

        map_bad = mask_eval.astype(float) + noise_bad

        if shift_pct != 0.0:
            shift_px = int(shift_pct * aspect_w)
            map_bad = np.roll(map_bad, shift_px, axis=1)

        return map_bad, mask_eval, noise_good

    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def run_test():
    print("=========================================================================")
    print("               E0: Anomaly Map & Mask Registration Unit Test              ")
    print("=========================================================================")
    print(f"Synthetic Frame:  {W0}x{H0} (Aspect Ratio {W0/H0:.2f}:1, matching sheet_metal)")
    print(f"Evaluation Grid:  {EVAL_SIDE}x{EVAL_SIDE}\n")

    mask_native = make_synthetic_data()

    cases = [
        ("crop", "crop", 0.0, False, "Old default: center-crop misses edge defects & distorts"),
        ("squash", "squash", 0.0, True, "Squash geometry: full-frame spatial registration"),
        ("squash_shifted", "squash", 0.20, False, "Sensitivity check: 20% shift collapses PRO"),
        ("letterbox", "letterbox", 0.0, True, "Letterbox geometry: aspect-preserving registration"),
        ("letterbox_shifted", "letterbox", 0.20, False, "Letterbox sensitivity check: 20% shift collapses PRO"),
        ("aspect", "aspect", 0.0, True, "Aspect geometry: non-square rectangular registration"),
        ("aspect_shifted", "aspect", 0.20, False, "Aspect sensitivity check: 20% shift collapses PRO"),
    ]

    all_passed = True
    print(f"{'Geometry':<18} {'Pixel AUROC':<13} {'AU-PRO@5%':<12} {'Expected':<12} {'Result'}")
    print("-" * 68)

    for name, geom, shift, should_pass, desc in cases:
        map_bad, mask_eval, map_good = simulate_geometry(mask_native, geometry=geom, shift_pct=shift)
        lo = min(map_bad.min(), map_good.min())
        hi = max(map_bad.max(), map_good.max())

        res = evaluate([map_good], [map_bad], [mask_eval], lo, hi)
        p5 = res.get("au_pro@0.05", 0.0)
        pa = res.get("pixel_auroc", 0.0)

        if should_pass:
            ok = (p5 > 0.95) and (pa > 0.99)
            expect_str = "> 0.95"
        else:
            ok = (p5 < 0.20)
            expect_str = "< 0.20"

        status = "PASS" if ok else "** FAIL **"
        if not ok:
            all_passed = False

        print(f"{name:<18} {pa:<13.4f} {p5:<12.4f} {expect_str:<12} {status}  ({desc})")

    print("-" * 68)
    if all_passed:
        print("RESULT: ALL EXPECTATIONS SATISFIED. E0 VERDICT: SUPPORTS")
        print("Hypothesis confirmed: 'crop' fails registration, 'squash' and 'letterbox' pass.")
        return 0
    else:
        print("RESULT: UNIT TEST FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_test())
