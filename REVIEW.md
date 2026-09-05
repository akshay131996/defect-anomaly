# Peer review — 2026-09-05

An adversarial peer review of the whole project. Ten dimensions were planned. Spend limits killed
the machine verification stage on every attempt, so **five dimensions completed across two runs**
(`patchcore-core`, `inference-validity`, `deployment`, `design-gaps`, `documentation-integrity`)
and **nothing was machine-verified** — every finding quoted here was verified by hand instead.
Raw reviewer output is in `REVIEW_FINDINGS.md`.

`geometry-registration` and `numerical-reproduction` were covered by the planner directly (§10);
**`optimization` and `fusion-and-legacy` remain unexamined.**

Further findings from the planner's own pass are at the end. §1 was **corrected** after a second
pass caught it overstating its own result — that correction is the most important thing in this
file.

**Read this before quoting any AU-PRO number from HANDOFF.md.** Several are wrong.

---

## 0. The target was wrong — resolved 2026-09-05

**P-4 asked where 0.764 came from. It has now been traced, and the answer changes the project's
standing more than any experiment in this review.**

### Provenance

`0.764` / `76.35%` comes from **`github.com/yyqmeow/patchcore-mvtec-ad2`**, companion code for an
**IEEE ETFA 2026 submission**, "Recovering Total Recall: Multi-Scale Feature Fusion for
High-Resolution Industrial Anomaly Localization". Its claim:

> "Multi-scale feature fusion (ResNet50 layer 2 + layer 3) raises mean AU-PRO@5% from 8.87% to
> 76.35%"

**That is this project's method.** WideResNet50-2, layer2+layer3, PatchCore. We have been treating
a single unreviewed conference submission's self-reported number as the benchmark, and building a
research programme around closing a gap to it.

### What the benchmark's own authors report

The MVTec AD 2 dataset paper — Heckler-Kram, Neudeck, Scheler, König, Steger,
[arXiv:2503.21622](https://arxiv.org/abs/2503.21622) — evaluates seven methods (PatchCore,
EfficientAD, RD, MSFlow, SimpleNet and others) and states:

> **"state-of-the-art methods ... remain below 60% average AU-PRO"**

**76.35% is above what the dataset's authors report as state of the art.** An unreviewed
submission claiming to beat the published field by a wide margin, on the field's own benchmark,
is a claim to verify before adopting as a target — not a bar to assume.

The paper also confirms our metric choice: AU-PRO@30% was judged **too permissive** for AD 2
because its defects are very small, which is why the stricter **AU-PRO@5%** is the headline.

### Where we actually stand against the published PatchCore baseline

Reported AU-PRO@30% for PatchCore on AD 2 (two figures per scenario, the two splits), against ours:

| scenario | ours @30% | published PatchCore | |
|---|---|---|---|
| can | **0.343** | 0.216 / 0.181 | **beat** |
| fabric | **0.462** | 0.346 / 0.353 | **beat** |
| vial | **0.913** | 0.905 / 0.892 | **beat** |

**Our implementation beats the published PatchCore baseline on all three scenarios that can
currently be checked**, and our mean AU-PRO@30% of **0.5736** sits inside the "below 60%" band the
dataset authors describe for state of the art.

### What this means

The project is **not 2.2x behind the field.** It is at or slightly above the published PatchCore
baseline, and within the SOTA band the benchmark's authors describe. The "2.2x gap" was a gap to
an unverified number produced by our own method — which, if true, would mean our implementation is
badly broken, and if false, means we have been chasing a phantom.

Given the dataset authors put SOTA below 60% and our @30% is 57.4%, **the phantom reading is much
more likely.**

### What remains genuinely open

- **The exact PatchCore AU-PRO@5% baseline is not yet in hand.** The per-scenario figures above are
  @30%, taken from secondary sources rather than read off the paper's table. **Get the table from
  the PDF and record it**, per scenario, per split, per limit.
- **Whether "below 60%" refers to @5% or @30%** is not established. If @5%, our 0.344 is behind the
  best of seven methods but nothing like 2.2x behind. If @30%, we are essentially at SOTA.
- **We still report on `test_public` and select on it.** Whatever the target, that must be fixed
  before any external claim.
- **The ETFA submission may still be worth reading.** If its 76.35% is real, its method is ours and
  the difference is findable — that would be the single most valuable thing to diff against. If it
  is not real, that is worth knowing before another GPU-hour is spent chasing it.

### The process lesson

The number entered the project in conversation, was written into HANDOFF, and was quoted **16
times** across two documents as the thing to beat. **Nobody checked it for four sessions.** Every
prioritisation decision — E5, E7, the ceiling argument, "representation must carry 0.16" — was
anchored to it.

A target is a load-bearing input. It deserved the same audit as a result, and got none.


---

## 1. Resolution is a strong lever — but the ceiling is UNTESTED, not refuted

**This section was overstated in its first version and is corrected here.** A second review pass
caught it, and the error is the same one this review criticised in others: claiming a result the
data does not reach.

### What the data actually shows

`outputs/runs/E5-inputres-224.json` was committed and filed by BRIDGE M-13 as a partial
"non-result to be re-run". Its numbers were never read. It carries full bucket data under the M6
native-pinned edges, with bucket counts **bit-identical** to the 448 arm across all six completed
scenarios. Pooled over those six:

| bucket | n | 224 | 448 | delta |
|---|---|---|---|---|
| sub-cell | 618 | 0.0873 | 0.2130 | +0.126 |
| 1-4x | 246 | 0.2059 | 0.4263 | +0.221 |
| 4-16x | 101 | 0.3960 | 0.6439 | +0.248 |
| >= 16x | 163 | 0.4672 | 0.5639 | +0.097 |

Per scenario, one doubling gained **+0.154 mean AU-PRO@5% (1.86x), improving 6 of 6**:
vial +0.258, fabric +0.146, sheet_metal +0.138, rice +0.136, fruit_jelly +0.126, can +0.119.

**Resolution is a strong lever in the 224 -> 448 range. That much is solid and it is the most
useful result in this review.**

### What it does NOT show, and what was wrongly claimed

The first version of this section said D-04 prediction 2 was falsified and the M-10 ceiling was
void. **Both claims are withdrawn.**

D-04 prediction 2 bounded the `ge_16x` bucket's movement at `< 0.03` **over 448 -> 768**. The
measurement above is **224 -> 448** — a different step. The prediction is untested, not falsified.

Worse, the substitution is not even approximately valid, because the bucket edges are pinned to
the **448** cell area. Measured from the artifact: `fabric`'s cell at 224 is 6528 native px
against 1607 at 448, a ratio of **4.06x**. So a region at ">= 16x the 448-cell" occupies only
**3.9x the cell at 224** — moderately resolved, not "already resolved". Its +0.097 gain is
exactly what one expects of a region moving from 4x to 16x cell coverage, and says nothing about
whether a region already at 16x cell at 448 gains further at 768.

**The ceiling argument is weakened but not refuted.** Its premise — that already-resolved regions
have little left to gain — remains untested. The 768 arm is the only thing that tests it.

### A further complication the size analysis has not accounted for

`sweep_backbones.py:130-134` bilinearly upsamples layer3 2x onto the layer2 grid. layer3 supplies
**1024 of the 1536 descriptor dimensions (66.7%)** and is natively stride 16, so **two thirds of
every descriptor carries no spatial detail finer than stride 16** — four times the cell area used
throughout the bucket analysis.

Every "sub-cell" claim in this project is therefore measured against the cell of the *minority*
feature block. Regions are more sub-cell than reported, and the `ge_16x` bucket sits at only ~4x
the effective layer3 cell. This does not change the measured numbers, but it changes what the
size buckets mean, and it points at a much cheaper experiment than E5 — see §9.

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
"use the global 0.606" — the wrong frame for the comparison being made. The ceiling conclusion is
untested rather than void (§1), so this frame error still matters: whatever the 768 arm shows,
the comparison must be made in one frame.

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

**P-4 / R-14 — RESOLVED, see §0.** The 0.764 baseline was traced to an unreviewed IEEE ETFA 2026 submission whose claim exceeds what the dataset's own authors report as SOTA. Original finding: It appears 16 times across HANDOFF and
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

## 9. The cheapest experiment in the queue is not in the queue: dilated layer3

`sweep_backbones.py:130-134` upsamples layer3 bilinearly 2x onto the layer2 grid. layer3 is
**1024 of 1536 dims (66.7%)** and natively stride 16, so two thirds of every descriptor is a 2x
blur of a stride-16 map.

timm's ResNet accepts **`output_stride=8`**, which dilates layer3 to a native stride-8 map at the
**same input size**: identical grid, identical patch count, identical 4000-vector bank, identical
NN-scoring cost and identical feature-tensor memory. Only layer3's convolutions run at 4x spatial.

That is the single-variable experiment the project's own rule ("change one variable at a time")
demands and which **E5 does not provide** — E5 moves input pixels, layer2 density, layer3 density,
bank size and object-to-receptive-field ratio simultaneously, so whatever it returns it cannot say
which of the five caused it.

Costs, recomputed from recorded configs:

| arm | patches x bank | feature tensor | est. GPU-h |
|---|---|---|---|
| current 448 | 3120 x 4000 | 8.3 GB | 0.22 |
| **dilated layer3 @448** | **3120 x 4000** | **8.3 GB** | **~0.4** |
| E5 768 | 9152 x 11755 | 24.3 GB | ~1.9 |

The dilated arm gives layer3 a **finer** effective map (52x60) than the 768 arm does (44x52), at
roughly a fifth of the cost. **Run it before E5's 768 arm.**

Two things the reviewer verified and set aside, worth recording so they are not re-proposed:

- **PatchCore's nearest-neighbour reweighting cannot move AU-PRO at all.** It reweights the
  image-level max score only; `sweep_backbones.py:215-222` returns per-patch minima and the map is
  built from them unchanged. It touches image AUROC and leaves AU-PRO bit-identical.
- **Native-resolution tiling is not the fast path and should rank last.** `fabric` at native
  2448x2048 gives 78,336 patches/image — 25x the current count — and a 387-image train set needs
  **186 GB** of feature tensor before coreset selection, against a 58 GiB ceiling.

Also verified and *not* filed as a finding: arm A is the **most** seed-stable of the six arms in
`outputs/seed_variance.json` (macro-mean range 0.0011 across 5 seeds, per-category std 0.0043), so
the "within 0.01 = equivalent" decision rule is probably safe despite having been set with no
measured AU-PRO noise floor on AD 2. The evidence points the other way from the concern.

---

## 10. Documentation defects, and what the planner verified directly

The documentation reviewer **re-derived REVIEW.md §1's own arithmetic from the artifacts and it
reproduces to 3 dp**, including the bit-identical bucket counts across all 24 scenario/bucket
pairs. The `vial` 0.7516 -> 0.6247 correction is complete in all three places with no fourth
occurrence anywhere. Means 0.3444 / 0.8501 / 0.7236 recompute exactly. All ten other run records
satisfy the §0 output contract. `deployment/README.md`'s quickstart flags all exist in the
corresponding argparse blocks.

Outstanding defects:

- **`README.md` still leads with the void fusion results** and HANDOFF still lists them under
  "Done and trustworthy". The README has not been updated for any finding in this review.
- **README's headline cost table is arm E (ResNet50@224), not arm A** — every number comes from a
  different arm than the one the project actually uses.
- **`HANDOFF.md:159` still lists "7% cost range over a 125x bank-size range"** in "Done and
  trustworthy"; §6 recomputed it as a **1.8x** range, from an image-level AD 1 experiment
  containing no AU-PRO at all.
- **`docs/MVTEC_AD2_IMPLEMENTATION_SPEC.md` is anchored entirely to the void crop-frame baseline**,
  so all of its acceptance targets are meaningless. Its Step 2 command **cannot run**, and the
  obvious repair silently evaluates **zero scenarios**.
- **Two different Triton latency figures** appear in two documents, neither backed by any artifact
  — and both were measured against the synthetic bank (§7).
- **`deployment/POD_REBUILD.md:44`** points `pip install -r` at a path where the freeze file does
  not exist. The planner wrote that file during this session; the path is wrong.
- **README says "8 arms x 15 categories" in one place and "Six arms" in another.** Arms
  `F_dinov3_448` and `G_dinov2_448` exist in `outputs/sweep_backbones.json` — G at mean AUROC
  0.9813 and the **lowest total cost of all eight** — and are dropped from the README table
  without a word, because only six were re-seeded. A defensible choice, undisclosed.

Planner-verified directly, no reviewer involved:

- **Numerical layer is clean.** Every `mean_*` reproduces, AU-PRO is monotonic in its limit
  everywhere, bank caps hold, every fixed-region run is exactly 1530, `n_good`/`n_bad` stable
  across all 11 runs. Only defect: the known stitched `E4-evalside-512` (wall/sum 0.58).
- **Registration is clean.** The native region-label carry is correct for `aspect` and `letterbox`
  — labels resize into exactly the frame the map occupies, NEAREST preserves ids, zero-pixel
  regions stay in the denominator, and the assert at `ad2_pixel_eval.py:511` guards the count.
  No third registration bug.

---

## What this changes, in order

1. **Run the dilated-layer3 arm (`output_stride=8`) at 448 before E5's 768 arm** (§9). Same
   patches, same bank, same memory, ~a fifth of the cost, and it is the only single-variable test
   of whether feature-map density or input pixels drive the gain. E5 changes five things at once.
2. **Then run E5's 768 arm.** Resolution is a strong lever (+0.154 per doubling, 6/6 scenarios) and
   768 is the *only* thing that tests the ceiling — which is untested, not refuted (§1).
3. **Correct the record**: the ceiling claim and D-04 prediction 2 are withdrawn-as-untested, not
   falsified; `vial` 0.7516 -> 0.6247 (done); aggregation frames must be matched (§4).
4. **Mark `outputs/ad2_feature_fusion.json` void**, strike the two "new project high" claims, and
   re-run the fusion arms under `aspect` before E7's conclusions mean anything (§5).
5. **Fix the launch pattern (`setsid`) and the `--resume` config guard** — both done in code; the
   HANDOFF text that teaches the broken pattern still needs updating.
6. **Decide on `count_include_pad`** — a genuine defect whose fix re-baselines every number (§6).
7. **Cite the baseline's source and split**, or state that the gap is approximate (§8).
8. **Rebuild the serving path** with a test that can fail (§7).
9. **Update README and the AD2 spec** — both still lead with void numbers (§10).
10. **Run the two dimensions nobody has reviewed**: `optimization` and `fusion-and-legacy`.

Nothing here invalidates the two findings the project rests on: the coordinate-frame fix
(0.131 -> 0.301) and the fixed native region set. Both were re-verified during this review, and
the numerical and registration layers were checked directly and are clean.

**The most important lesson is about this review itself.** Its first version claimed the ceiling
was refuted, on data that tested a different step. That is the same failure mode the review
criticised in the worker's "proven" and in the `vial` number: a real measurement, stretched one
step past what it supports. The correction came from a second adversarial pass, which is an
argument for running one.
