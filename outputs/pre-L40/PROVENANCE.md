# Results as of 2026-09-02, before the L40 rerun

Frozen snapshot of every result produced up to this point. The seven-arm sweep (six
existing arms plus DINOv3) will be rerun on an **NVIDIA L40**, overwriting the files one
level up. These copies are what the new numbers get compared against.

## Which GPU produced what

The pod was rescheduled twice mid-project, so this set is **not** device-homogeneous.
Device is recorded inside the JSON where the script captured it; where it is not, it is
inferred from the session the file was written in.

| file | GPU | recorded? |
|---|---|---|
| `session1_baseline.json` | RTX 4000 Ada | inferred |
| `session2_patchcore.json` | RTX 4000 Ada | inferred |
| `sweep_backbones.json` | RTX 4000 Ada | yes |
| `data_efficiency.json` | RTX 4000 Ada | yes |
| `seed_variance.json` | RTX 4000 Ada | inferred |
| `exp_threshold_coreset.json` | RTX 4000 Ada | inferred |
| `exp_percentile_rule.json` | **RTX A4000** | yes |
| `exp_arms_optimal_threshold.json` | **RTX A4000** | yes |

The two A4000 files are the reason `device` is now captured explicitly in every new
script — it was added when the pod first came back on different hardware.

## Whether the device split matters

Probably not, but it was never tested, so treat it as an open question rather than a
settled one.

The operations are deterministic given a seed, so the two cards should agree. The
plausible source of disagreement is reduction order in `torch.cdist`, which varies with
SM count and could differ in the last bits. That would normally be irrelevant — except
this project has repeatedly found that tiny score differences move *decisions* a lot
(the seed audit swung `screw` escapes 19 → 50 from nothing but a different coreset
start). A metric that sensitive does not get to assume float noise is harmless.

**No cross-device comparison was ever made directly**, so nothing here is known to be
contaminated. The risk is that `exp_percentile_rule` and `exp_arms_optimal_threshold`
(A4000) are compared against the sweep (Ada) without anyone noticing the difference.

The L40 rerun fixes this for the sweep by putting all seven arms on one card.

## Reference timings — RTX 4000 Ada class

Total measured GPU time to produce this snapshot: **208 minutes (3.5 h)**.

| experiment | minutes |
|---|---|
| Backbone sweep, 6 arms x 15 categories | 48.0 |
| Seed variance, 4 arms x 6 cats x 5 seeds | 45.8 |
| Arms at optimal threshold, 4 x 15 | 44.8 |
| Data efficiency, 15 x 8 sizes x 3 seeds | 34.5 |
| Coreset + threshold, 6 cats x 4 ratios | 25.9 |
| Percentile rule, 15 categories | 9.0 |

Slowest arm in every run was `D_wrn50_320` (11.3 min in the sweep, 16.1 min at optimal
threshold) — a 40x40 grid is 1,600 patches per image against 784 for the 28x28 arms.

These are the baseline for judging the L40's speedup. Expect **1.5-2x**, not the ~3x its
core count suggests: feature extraction and `cdist` scale with compute, but the greedy
coreset selection is thousands of tiny sequential kernel launches and is latency-bound.

## Headline numbers being preserved

So that a regression is obvious without opening the JSON:

- Data efficiency: **median knee 10 images**, range 2 to 256, `screw` never converges
- Sweep, seed-corrected cost: B1 15,182 · A 15,748 · D 17,762 · E 17,870 · B0 26,043 · C 26,774
- Arms at their own best percentile: **D 559** · E 654 · A 667 · B1 671, all optimal at p20
- Threshold rule: p99 is **15x** worse than a single global p50 at 100:1
- Degenerate baseline: scrapping every part costs **467**, beating the best detector's 559
