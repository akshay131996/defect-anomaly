#!/usr/bin/env python3
"""Unit test for the AU-PRO implementation, independent of any model or dataset.

Our AD 2 run produced AU-PRO@5% of 0.55% against a published baseline of 8.87% - using a
*better* feature configuration than that baseline. A gap that large is more likely a bug
in a metric written yesterday than a real result, and validating against a published
number would confound the metric with the model.

So test the metric alone, with predictors whose correct answer is known by construction:

  perfect   anomaly map == the mask          -> AU-PRO must be ~1.0
  inverted  anomaly map == NOT the mask      -> AU-PRO must be ~0.0
  random    uniform noise                    -> PRO tracks FPR, so the integral of
                                                 FPR dFPR / limit is ~limit/2
  blurred   perfect, then softened           -> high but below perfect

If these pass, the metric is sound and AD 2 really is that punishing at 224px. If they
fail, every AD 2 number so far is noise.
"""
import numpy as np
from aupro import evaluate, PRO_LIMITS

rng = np.random.default_rng(0)
SIDE, N_IMG = 256, 12


def make_masks():
    masks = []
    for _ in range(N_IMG):
        m = np.zeros((SIDE, SIDE), bool)
        for _ in range(rng.integers(1, 4)):            # 1-3 defect blobs per image
            cy, cx = rng.integers(30, SIDE - 30, 2)
            r = rng.integers(6, 22)
            y, x = np.ogrid[:SIDE, :SIDE]
            m |= (y - cy) ** 2 + (x - cx) ** 2 < r * r
        masks.append(m)
    return masks


def run(name, maps_bad, masks, maps_good, expect):
    lo = min(m.min() for m in maps_bad + maps_good)
    hi = max(m.max() for m in maps_bad + maps_good)
    r = evaluate(maps_good, maps_bad, masks, float(lo), float(hi))
    p5, p30 = r.get("au_pro@0.05"), r.get("au_pro@0.3")
    ok = expect(p5, p30, r["pixel_auroc"])
    print(f"{name:<10} pixel_auroc {r['pixel_auroc']:.4f}   "
          f"AU-PRO@5% {p5:.4f}   @30% {p30:.4f}   regions {r['n_regions']:>3}   "
          f"{'PASS' if ok else '** FAIL **'}")
    return ok


from scipy import ndimage

masks = make_masks()
results = []

# Real anomaly maps are continuous. A two-valued map makes the FPR curve a step
# function with no resolution below the 5% limit, which makes AU-PRO@5% undefined -
# a property of the input, not of the metric. So every case below carries noise.
def noisy(base, scale=0.05):
    return base + rng.random(base.shape) * scale

# perfect: high inside the defect, low outside
results.append(run("perfect",
    [noisy(m.astype(float)) for m in masks], masks,
    [noisy(np.zeros((SIDE, SIDE))) for _ in range(N_IMG)],
    lambda p5, p30, pa: p5 > 0.95 and p30 > 0.95 and pa > 0.99))

# inverted: confidently wrong - low inside the defect, high outside
results.append(run("inverted",
    [noisy((~m).astype(float)) for m in masks], masks,
    [noisy(np.ones((SIDE, SIDE))) for _ in range(N_IMG)],
    lambda p5, p30, pa: p5 < 0.15 and pa < 0.15))

# random: PRO(t) tracks FPR(t), so the integral is lim^2/2 and the NORMALISED value is
# lim/2 - 0.025 at the 5% limit and 0.15 at 30%. (An earlier version of this test
# asserted ~0.5 here, which was simply the wrong expectation; the metric was right.)
results.append(run("random",
    [rng.random((SIDE, SIDE)) for _ in range(N_IMG)], masks,
    [rng.random((SIDE, SIDE)) for _ in range(N_IMG)],
    lambda p5, p30, pa: abs(p5 - 0.025) < 0.02 and abs(p30 - 0.15) < 0.05
                        and abs(pa - 0.5) < 0.15))

# blurred perfect: still strong, and must not beat perfect
results.append(run("blurred",
    [noisy(ndimage.gaussian_filter(m.astype(float), 4)) for m in masks], masks,
    [noisy(np.zeros((SIDE, SIDE))) for _ in range(N_IMG)],
    lambda p5, p30, pa: p5 > 0.5 and pa > 0.9))

# bit-identical check: valid = all-True must produce bit-identical results to valid = None
res_none = evaluate([noisy(np.zeros((SIDE, SIDE))) for _ in range(N_IMG)],
                    [noisy(m.astype(float)) for m in masks], masks, 0.0, 1.05, valid=None)
res_all_true = evaluate([noisy(np.zeros((SIDE, SIDE))) for _ in range(N_IMG)],
                        [noisy(m.astype(float)) for m in masks], masks, 0.0, 1.05,
                        valid=np.ones((SIDE, SIDE), bool))
bit_id = (res_none["pixel_auroc"] == res_all_true["pixel_auroc"] and
          res_none["au_pro@0.05"] == res_all_true["au_pro@0.05"] and
          res_none["au_pro@0.3"] == res_all_true["au_pro@0.3"])
results.append(bit_id)
print(f"{'valid_all':<10} pixel_auroc {res_all_true['pixel_auroc']:.4f}   "
      f"AU-PRO@5% {res_all_true['au_pro@0.05']:.4f}   @30% {res_all_true['au_pro@0.3']:.4f}   "
      f"regions {res_all_true['n_regions']:>3}   {'PASS' if bit_id else '** FAIL **'}")

# letterbox padding check: padding has high out-of-manifold anomaly scores
valid_lb = np.zeros((SIDE, SIDE), bool)
valid_lb[SIDE // 4: 3 * SIDE // 4, :] = True
masks_lb = []
for _ in range(N_IMG):
    m = np.zeros((SIDE, SIDE), bool)
    for _ in range(rng.integers(1, 4)):
        cy = rng.integers(SIDE // 4 + 16, 3 * SIDE // 4 - 16)
        cx = rng.integers(30, SIDE - 30)
        r = rng.integers(6, 16)
        y, x = np.ogrid[:SIDE, :SIDE]
        m |= (y - cy) ** 2 + (x - cx) ** 2 < r * r
    masks_lb.append(m)

maps_good_lb = [noisy(np.zeros((SIDE, SIDE))) for _ in range(N_IMG)]
maps_bad_lb = [noisy(m.astype(float)) for m in masks_lb]
for mg, mb in zip(maps_good_lb, maps_bad_lb):
    mg[~valid_lb] += 2.0
    mb[~valid_lb] += 2.0

# with valid passed: range is computed on valid pixels, padding excluded -> near-perfect (> 0.95)
allv_valid = np.concatenate([m[valid_lb].ravel() for m in (maps_good_lb + maps_bad_lb)])
lo_v, hi_v = float(allv_valid.min()), float(allv_valid.max())
res_lb_v = evaluate(maps_good_lb, maps_bad_lb, masks_lb, lo_v, hi_v, valid=valid_lb)
ok_v = res_lb_v.get("au_pro@0.05", 0.0) > 0.95
results.append(ok_v)
print(f"{'lb_valid':<10} pixel_auroc {res_lb_v['pixel_auroc']:.4f}   "
      f"AU-PRO@5% {res_lb_v['au_pro@0.05']:.4f}   @30% {res_lb_v['au_pro@0.3']:.4f}   "
      f"regions {res_lb_v['n_regions']:>3}   {'PASS' if ok_v else '** FAIL **'}")

# without valid passed: range includes padding, padding noise pollutes h_norm -> collapses (< 0.20)
allv_all = np.concatenate([m.ravel() for m in (maps_good_lb + maps_bad_lb)])
lo_all, hi_all = float(allv_all.min()), float(allv_all.max())
res_lb_nv = evaluate(maps_good_lb, maps_bad_lb, masks_lb, lo_all, hi_all, valid=None)
ok_nv = res_lb_nv.get("au_pro@0.05", 0.0) < 0.20
results.append(ok_nv)
print(f"{'lb_novalid':<10} pixel_auroc {res_lb_nv['pixel_auroc']:.4f}   "
      f"AU-PRO@5% {res_lb_nv['au_pro@0.05']:.4f}   @30% {res_lb_nv['au_pro@0.3']:.4f}   "
      f"regions {res_lb_nv['n_regions']:>3}   {'PASS' if ok_nv else '** FAIL **'}")

print()
print("ALL PASS" if all(results) else "** SOME CHECKS FAILED - the metric is suspect **")
