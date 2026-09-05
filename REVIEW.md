# Peer review — 2026-09-05

An adversarial peer review of the whole project. Ten review dimensions were planned; the
workflow lost its verification stage to a spend limit twice, so **three dimensions completed
(`patchcore-core`, `inference-validity`, `deployment`) and nothing was machine-verified.** The
raw reviewer output is in `REVIEW_FINDINGS.md`. This file records what was **independently
verified by hand** and what it changes.

Five further findings (P-1 to P-5) came from the planner's own pass and are listed at the end.

**Read this before quoting any AU-PRO number from HANDOFF.md.** Several are wrong.

---

## 1. The headline reversal: resolution matters far more than we concluded

**M-10's ceiling argument is invalid, and the data refuting it was already in the repo.**

M-10 argued that regions at `>= 16x` cell area are already spatially resolved, score 0.6056,
and therefore cap what resolution can buy at ~0.61 — leaving ~0.16 that only representation
(E7) could carry. That argument carried one stated condition: *it assumes higher resolution
does not also lift the `ge_16x` bucket.*

**That condition is false.** `outputs/runs/E5-inputres-224.json` was committed on 2026-09-05 and
filed in BRIDGE M-13 as a partial "non-result" to be re-run. Its numbers were never read. It
carries full bucket data under the M6 native-pinned edges, and its per-bucket counts are
**bit-identical** to the 448 arm for all six completed scenarios — so the 224 vs 448 comparison
is exactly the test D-04 specified.

Pooled over those six scenarios:

| bucket | n | 224 | 448 | delta |
|---|---|---|---|---|
| sub-cell | 618 | 0.0873 | 0.2130 | **+0.126** |
| 1-4x | 246 | 0.2059 | 0.4263 | **+0.221** |
| 4-16x | 101 | 0.3960 | 0.6439 | **+0.248** |
| **>= 16x** | 163 | 0.4672 | 0.5639 | **+0.097** |

Per scenario, one resolution doubling gained **+0.154 mean AU-PRO@5% (1.86x)**, improving
**6 of 6**:

| scenario | 224 | 448 | delta |
|---|---|---|---|
| vial | 0.4607 | 0.7191 | +0.258 |
| fabric | 0.0432 | 0.1888 | +0.146 |
| sheet_metal | 0.1151 | 0.2529 | +0.138 |
| rice | 0.0902 | 0.2263 | +0.136 |
| fruit_jelly | 0.3495 | 0.4758 | +0.126 |
| can | 0.0193 | 0.1385 | +0.119 |
| **mean** | **0.1797** | **0.3336** | **+0.154** |

Two consequences.

**The ceiling is gone.** `ge_16x` is not fixed — it rose 0.467 -> 0.564 across the same
doubling, 3.2x the `< 0.03` bound registered in D-04 prediction 2. **That prediction is
falsified.** The claim "resolution alone provably cannot close the gap" does not hold.

**But the mechanism claimed was wrong.** D-04 predicted the gain would concentrate in the
sub-cell bucket. It did not: sub-cell gained least of the three smallest buckets (+0.126 vs
+0.221 and +0.248). Every bucket moved together, which is precisely the outcome D-04 named as
*"resolution is a confound rather than the cause"*. Resolution helps a great deal; the
"sub-cell defects are unresolvable" story is not why.

**Action:** E5 becomes the highest-value experiment in the queue, not a formality. Strike the
ceiling argument from HANDOFF §6 and BRIDGE M-10. E7 remains worth running but is no longer
justified by "resolution provably cannot close ~0.16".

---

## 2. The monotone size trend is largely scenario composition

The claim that AU-PRO rises monotonically with defect size (0.227 / 0.428 / 0.550 / 0.606) is a
**pooled** statistic and does not hold within scenarios.

| scenario | sub-cell | 1-4x | 4-16x | >= 16x | monotonic? |
|---|---|---|---|---|---|
| can | 0.138 (66) | — | — | — | n/a |
| fabric | 0.157 (102) | 0.000 (6) | 0.995 (6) | 0.111 (6) | no |
| fruit_jelly | 0.366 (116) | 0.781 (40) | 0.585 (24) | 0.419 (36) | no |
| rice | 0.127 (18) | 0.154 (60) | 0.367 (24) | 0.456 (12) | **yes** |
| sheet_metal | 0.124 (288) | 0.436 (126) | 0.648 (12) | 0.769 (18) | **yes** |
| vial | 0.929 (28) | 0.679 (14) | 0.813 (35) | 0.625 (91) | **inverted** |
| wallplugs | 0.000 (24) | 0.345 (36) | 0.334 (18) | 0.536 (12) | no |
| walnuts | 0.347 (114) | 0.475 (72) | 0.445 (54) | 0.711 (72) | no |

**Only 2 of 7 are monotonic.** The pooled trend is driven by composition: the `ge_16x` bucket is
**66% supplied by the two best-performing scenarios** (vial 36.8%, walnuts 29.1%), while
sub-cell is dominated by `sheet_metal` (288 regions) and `fabric`. Bigger regions score better
partly because *easier scenarios have bigger regions*.

`sheet_metal` — the case registered in M-09 as the cleanest pure-resolution test — **is** one of
the two monotonic scenarios (0.124 -> 0.769). So the size effect is real where it was predicted
to be real and largely absent elsewhere. That is a refinement of the hypothesis, not a refutation.

---

## 3. `vial`'s headline number does not exist

**`vial`'s `ge_16x` AU-PRO@5% is 0.6247, not 0.7516.** The value 0.7516 appears in **no**
artifact — not in any JSON, not in any log. It was quoted three times as the load-bearing
evidence that "large defects already reach published parity":

- `BRIDGE.md:210` (worker M-06)
- `BRIDGE.md:253` (planner M-08 — which explicitly asserted "the reported values reproduce")
- `HANDOFF.md:556`

The planner verified bucket *counts* and *means* against the artifact and reported them as
reproducing, then repeated a per-scenario number from the worker's summary without checking it
against the same artifact — whose real value had already been printed in the planner's own
session output.

`vial` also does **not** support the claim it was used for: its largest bucket (0.625) is its
**worst**, and its sub-cell bucket (0.929) its best.

---

## 4. Aggregation frames are mixed in the ceiling comparison

The project's reported mean AU-PRO@5% (0.3444) is a **macro** mean over 8 scenarios. The bucket
value 0.6056 is a **micro** mean over 247 regions. M-10 compared 0.6056 against 0.3444 and
0.7640 — three numbers in two different frames.

- Micro-averaged current score: **0.3709**, not 0.3444.
- Macro `ge_16x` over the 7 scenarios that have any: **0.5181**.
- `can` has **zero** `ge_16x` regions, so its ceiling is undefined and cannot be 0.606.

HANDOFF.md:626 already computed the macro version (0.625), called it noisy, and instructed
"use the global 0.606" — the wrong frame for the comparison being made. The ceiling conclusion
is void anyway (§1), but the frame error would have mattered regardless.

---

## 5. The representation track was never measured in a valid frame

`ad2_feature_fusion.py` has **no geometry support at all** — no `--geometry`, no squash,
letterbox or aspect. It builds its transform with `timm.data.create_transform` (resize +
centre-crop) at lines 90/101/114 and resizes masks full-frame to a square at line 399. That is
exactly the map/mask misregistration fixed in commit `06038e8` for `ad2_pixel_eval.py`.

The fusion script was written and run **after** that fix existed and never inherited it. Its
recorded grids are square (`[112,112]`, `[56,56]`) where the aspect runs record `[36,84]`,
`[28,112]`, `[64,48]`; its mean AU-PRO@5% of 0.1130 sits in the broken-geometry regime (0.1307),
not the fixed one (0.3444).

**Everything in the representation track is therefore unmeasured**: the "four architectural
lessons", the adaptive routing table, and each claim about DINOv2 on `fabric`, whitening on
`can`, L1+L2+L3 on `sheet_metal`. E7 is currently load-bearing for closing the gap and **the
evidence base it would build on does not exist.**

Two "new project high" claims (image AUROC 0.6914, pixel AUROC 0.7700) are also wrong — plain
arm A under aspect reaches 0.7236 / 0.8495.

---

## 6. Verified code defects

**`avg_pool2d` shrinks border descriptors** — `sweep_backbones.py:143` (and
`ad2_feature_fusion.py:140,156,171,179`) uses the default `count_include_pad=True`, so grid-edge
cells divide by 9 with only 6 or 4 real contributors. Measured: interior descriptor L2 norm
17.39, top edge 12.13 (69.8%), corner 8.57 (49.3%). Border cells are 7.0-8.8% of the grid. Under
AU-PRO every region counts equally, so a border region scored a third low is a region that never
clears the FPR budget. One-argument fix — but it changes every number, so it must be a deliberate
re-baseline, not a silent patch.

**Bank density varies 3.1x across scenarios and is misrecorded.** `--bank-cap 4000` yields an
effective coreset ratio of 0.297% (walnuts) to 0.931% (sheet_metal) — every one below the
nominal 1% — while the run record writes `"coreset_ratio": 0.01`. Per-scenario AU-PRO differences
are therefore partly a bank-density artifact. The docstring justification at
`sweep_backbones.py:189` ("7% across a 125x range") does not survive recomputation: at the best
operating point per ratio, cost runs 11900 / 8061 / 6644 / 6619 — a 1.8x range — and that
experiment was image-level on AD 1 with no AU-PRO in it at all.

**`--geometry` defaults to `crop`** — the frame the project proved destroys AU-PRO.

**`--resume` has no config guard** (planner P-5) — `ad2_pixel_eval.py:352` reuses a scenario on
name and a non-null score alone, comparing no config field and no code hash, wrapped in
`except Exception: pass`. Resuming with changed flags silently merges configurations and reports
only the new one. It is already in the committed E5 768 launch command.

---

## 7. The serving path cannot produce a correct answer

Four defects, all independently verified:

- **`IS_DEFECTIVE` is constant True.** `metadata.json` sets `threshold: 0.55`; our own calibrated
  operating points for the same backbone are **36.8-47.2** (`session2_patchcore.json`) — a factor
  of **67-86x**. Every image exceeds 0.55.
- **Bank and query use different geometries.** `export_bank.py:75` fits the bank through
  `Resize(256)+CenterCrop(224)`; `model.py:296` serves a direct squash. The coordinate-frame bug
  again, now corrupting the score itself rather than the localisation.
- **float32 input in [0,255] skips both `/255` and normalisation** — the `/255` branch is
  uint8-only and the `max <= 1.05` guard then fails.
- **The random-bank fallback is real** (`model.py:262`, `torch.randn(100, dim)`), though it does
  print a warning, so "silently" overstates it.

`test_client.py` asserts only that outputs exist, so it passes through all of this. **E8 is not
"swap in a real bank" — it is rebuild the serving path and give it a test that can fail.**

---

## 8. Process findings (planner pass)

**P-1 — the silent run deaths are `nohup` without `setsid`, not memory.** `E5-inputres-224` died
at 6/8 with peak RSS of **3.5 GB** while completed runs used 18-19 GB. `HANDOFF.md:313` teaches
the broken pattern. Direct A/B in session history: E1 with `nohup` died at 5/8; the relaunch with
`setsid nohup ... < /dev/null` completed. Three runs lost this way. **Closes OQ-1; the host-OOM
hypothesis was wrong.**

**P-2 — `E4-evalside-512` is a stitched record with `deviations: []`.** Its log shows a first
execution dying after 3 scenarios and a second `--resume` run reusing them. It is the only record
of ten whose `wall_seconds` (468) disagrees with the sum of its per-scenario times (800); every
other is 1.00. That ratio is a cheap general detector for stitching and belongs in the §0 audit
list. This record is both E4's 512 arm and E5's intended 448 arm.

**P-3 — the scaling-law table overstates cost 5.2x.** `HANDOFF.md:715` claims `can` at 448 costs
"~10 min"; measured is **115 s**. It predates the bank cap and the 9.8x decode fix, and it is the
sole basis for "native resolution is therefore infeasible" — a claim that shaped the entire
resolution strategy, which §1 now shows was the most valuable direction available.

**P-4 / R-14 — the 0.764 baseline has no provenance.** It appears 16 times across HANDOFF and
BRIDGE and anchors every framing of the problem. No paper, URL, or split is cited anywhere in the
repo or its git history. Ours is `test_public`; MVTec's leaderboard is `test_private`.

**R-13 — every number and every selection is on `test_public`, violating the project's own rule
5.** `ad2_pixel_eval.py` loads the `validation` split at line 82, records `n_val`, and **never
uses it**. The geometry winner, the eval-side choice and the pending resolution choice were all
selected on `test_public`, and 0.3444 is reported against a published baseline on that basis. The
rule is enforced only against E7's fusion routing and nowhere else.

**P-6 — REFUTED by measurement.** The planner suspected opposite endpoint biases in the two
metrics, which would have made 0.344 optimistic. Measured: AU-PRO bias against an anchored
reference is **exactly 0.0000** at every detector quality; pixel AUROC matches an exact
Mann-Whitney computation to **1e-5**. `aupro.py` is correct and nothing needs re-scoring.

---

## What this changes, in order

1. **Run E5.** It is now the highest-value experiment in the queue, on direct evidence rather
   than on the ceiling argument that opposed it.
2. **Strike the ceiling argument** (HANDOFF §6, BRIDGE M-10) and D-04 prediction 2, which is
   falsified.
3. **Correct `vial` 0.7516 -> 0.6247** in three places, and stop citing vial as evidence that
   large defects reach parity — its largest bucket is its worst.
4. **Mark `outputs/ad2_feature_fusion.json` void** and strike the two "new project high" claims.
   E7 must be re-run under `aspect` before any of its conclusions mean anything.
5. **Fix the launch pattern** (`setsid`) and the `--resume` config guard before the next run.
6. **Decide on `count_include_pad`** — a genuine defect whose fix re-baselines every number.
7. **Cite the baseline's source and split**, or state that the gap is approximate.
8. **Rebuild the serving path** with a test that can fail.

Nothing here invalidates the two findings the project rests on: the coordinate-frame fix
(0.131 -> 0.301) and the fixed native region set. Both were re-verified during this review.
