# Peer review findings — raw reviewer output

Produced by an adversarial peer-review workflow, 2026-09-05. **These are the reviewers' own
words, unverified except where `REVIEW.md` says otherwise.** The workflow's verification stage
was lost to a spend limit twice, so treat every finding here as a claim to check, not a result.
`REVIEW.md` records which ones were independently verified and what survived.

---

### R-01 · Every representation experiment ran in the crop coordinate frame the project proved is broken

**CRITICAL · inference-validity · patchcore-core** — `ad2_feature_fusion.py:90`

ad2_feature_fusion.py has no geometry support whatsoever - `grep -n "aspect|geometry|squash|letterbox"` on the file returns zero hits. It builds its transform with `timm.data.create_transform(**cfg, is_training=False)` (lines 90, 101, 114), i.e. resize-to-img/crop_pct then centre-crop, and it resizes ground-truth masks full-frame to a square at line 399: `Image.open(p).convert("L").resize((EVAL_SIDE, EVAL_SIDE), Image.NEAREST)`. That is exactly the map/mask misregistration documented in ad2_pixel_eval.py:100-131 and fixed in commit 06038e8 ('fix AD 2 map/mask coordinate-frame bug; mean AU-PRO@5% 0.131 -> 0.301'). The fusion script was written (Sep 4 20:08) and run (JSON mtime Sep 4 20:51) after that fix existed in ad2_pixel_eval.py, but the fix was never carried over.

**Evidence.** outputs/ad2_feature_fusion.json reports mean_au_pro@0.05 = 0.1130 - the crop-geometry regime (0.1307), not the aspect regime (0.3444 in outputs/runs/E4-evalside-512.json). Every recorded `grid` in that JSON is square ([112,112], [56,56], [32,32]), confirming square centre-cropped input, while the E3R/E4 aspect runs record [36,84], [28,112], [64,48]. HANDOFF.md:165-166 nonetheless reports 'Mean Image AUROC 0.6914 (new project high)' and 'Mean Pixel AUROC 0.7700 (new project high)' against arm A's aspect run at 0.7236 / 0.8495.

**Impact.** The entire representation track is unmeasured. The 'four architectural lessons' (HANDOFF.md:677-...), the adaptive per-scenario routing table, and each individual claim - DINOv2 wins fabric, whitening fixes can, L1+L2+L3 helps sheet_metal, morphological closing helps/hurts - are all derived in a frame where AU-PRO is dominated by misregistration rather than by the representation. HANDOFF.md:630 makes E7 (representation) load-bearing for closing the 0.344 -> 0.764 gap; the evidence base E7 would build on does not currently exist. Two 'new project high' claims in HANDOFF and the README are also wrong: plain arm A under aspect beats them on both metrics.

**Fix.** Factor `squash_transform` / `letterbox_transform` / `aspect_transform` / `aspect_dimensions` and the native-resolution region-labelling block out of ad2_pixel_eval.py into a shared module, import it in ad2_feature_fusion.py, add the same `--geometry` flag (defaulting to `aspect`), and re-run all fusion arms. Until then, mark outputs/ad2_feature_fusion.json void in the LEDGER and strike the 'new project high' lines from HANDOFF.md:165-166 and the README.

### R-02 · Bank cap 4000 gives an effective coreset ratio of 0.30-0.93%, and its stated justification does not survive recomputation

**MAJOR · inference-validity · patchcore-core** — `sweep_backbones.py:189`

The `max_k` docstring (sweep_backbones.py:189-192) justifies capping the bank with: 'the coreset sweep measured total cost varying only 7% across a 125x range of bank sizes ... It is a stability knob, not an accuracy knob.' Recomputing outputs/exp_threshold_coreset.json shows that 7% is one cell of a 3-rule x 5-calibration-size grid (the p99/None cell: 21404 -> 19938, 7.4%). At the best operating point for each ratio, total cost runs 11900 / 8061 / 6644 / 6619 across ratios 0.002 / 0.01 / 0.05 / 0.25 - a 1.8x range, not 7%. Mean image AUROC moves 0.9351 -> 0.9630 and `screw` alone moves 0.8256 -> 0.9684. That experiment is also image-level cost on MVTec AD at 224px over 6 categories; it contains no pixel metric and no AU-PRO, yet it is the only cited basis for capping the bank in the AD 2 pixel-level runs.

**Evidence.** With `--bank-cap 4000` (run_e4.sh:10, run_e4a.sh:8) and the per-scenario n_train x n_patches in outputs/runs/E4-evalside-512.json, the effective ratio is: walnuts 4000/1,347,840 = 0.297%, can 0.321%, fabric 0.331%, rice 0.410%, wallplugs 0.438%, vial 0.447%, fruit_jelly 0.495%, sheet_metal 0.931%. That is a 3.1x spread across scenarios, every one of them below the nominal 1%, while the run record writes `"coreset_ratio": 0.01` into config (ad2_pixel_eval.py:342).

**Impact.** Two things. (a) AU-PRO@5% is a false-positive-budgeted metric: a sparser bank raises the nearest-neighbour distance of poorly-covered NORMAL patches, which spends the 5% FPR budget and pushes PRO down - precisely the failure mode that image-level max-over-patches AUROC is insensitive to, so the cited experiment cannot rule it out. The bank cap is an untested candidate cause of the very gap the project is chasing. (b) Bank density varies 3.1x between scenarios, so per-scenario AU-PRO differences (e.g. sheet_metal at 0.93% vs walnuts at 0.30%) are partly a bank-density artifact rather than a scenario property. The cost of removing the cap is small: `patch_distances` at bank=12,000 is roughly 30 GFLOP per scenario against a 468 s run dominated by feature extraction.

**Fix.** Run one arm with `--bank-cap 0` (true 1% ratio) and one at 3x the current cap under aspect/448 and compare mean AU-PRO@5%; this is a ~3x cost increase on a step that is not the bottleneck. Regardless of the outcome, record the realised bank size and the realised ratio per scenario in the config block instead of the nominal `coreset_ratio: 0.01`, and either equalise the ratio across scenarios or the absolute bank size - not neither, as now. Also correct the docstring at sweep_backbones.py:189-192 to state what was actually measured.

### R-03 · 3x3 avg_pool with count_include_pad=True shrinks border descriptors by up to 51%

**MAJOR · correctness · patchcore-core** — `sweep_backbones.py:143`

`fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)` uses PyTorch's default `count_include_pad=True`, so cells on the grid boundary are divided by 9 even though only 6 (edge) or 4 (corner) real neighbours contributed. The border patch descriptor is therefore a scaled-down copy of the true 3x3 neighbourhood mean, not the mean. This is the neighbourhood-aggregation step of PatchCore, and the same defect is repeated in ad2_feature_fusion.py at lines 140, 156, 171 and 179.

**Evidence.** Verified directly: `F.avg_pool2d(torch.ones(1,4,8,8), 3, 1, 1)` gives interior 1.000, edge 0.667, corner 0.444; with `count_include_pad=False` all are 1.0. On a realistic 1536-channel post-ReLU map at grid 52x60, mean descriptor L2 norm is 17.39 interior, 12.13 on the top edge (69.8%) and 8.57 at the corner (49.3%). Border cells are 7.05% of the grid for fabric/rice/wallplugs/walnuts (52x60), 7.16% for fruit_jelly/vial, 7.80% for can (36x84) and 8.80% for sheet_metal (28x112).

**Impact.** Because distance is plain Euclidean on raw magnitudes, the whole distance scale at the image border is compressed by roughly a third. A defect touching the frame edge produces ~0.67x the anomaly score of the identical defect one cell further in. Under AU-PRO every region counts equally, so a border region under-scored by a third is a region that never clears the threshold inside the 5% FPR budget - it contributes ~0 to a mean taken over 1530 regions. It also perturbs the greedy k-center selection, which operates on these same shrunken vectors.

**Fix.** Pass `count_include_pad=False` in all four places. It is a one-argument change and it is the semantics the code comment already implies ('3x3 neighbourhood aggregation'). Re-run E3R-aspect-448 with and without it and report the delta; if border regions are a material share of the 1530, this is free AU-PRO.

### R-04 · ViT arms use only the final transformer block while CNN arms get layer2+layer3

**MAJOR · design-gap · patchcore-core** — `sweep_backbones.py:136`

The CNN branch (lines 128-134) concatenates two mid-level stages via `out_indices=(2,3)`. The ViT branch (lines 136-141) calls `self.model.forward_features(x)`, which returns only the output of the last block after the final norm, and uses it alone. So arms B0/B1/F/G differ from arm A in TWO respects - backbone family and feature depth/multi-scale - not one, contradicting the file's own contract at line 8 ('Every pair of arms differs in exactly one variable'). ad2_feature_fusion.py:279-289 repeats it for `dinov2_448`/`dinov3_448`, and the `fusion` mode at line 302 also takes only the last ViT block.

**Evidence.** sweep_backbones.py:136-138 is `toks = self.model.forward_features(x); n_prefix = getattr(...); toks = toks[:, n_prefix:, :]` - a single tensor, no intermediate blocks, no concatenation. Contrast lines 130-134, which build `fmap` from a list of two stages. HANDOFF.md:729 states the published AD 2 improvement (8.87% -> 76.35% AU-PRO) 'came from multi-scale layer2+layer3' - the exact ingredient the ViT arms were never given.

**Impact.** The two conclusions recorded in HANDOFF.md:422-423 - 'Backbone choice is category-dependent; DINOv2 wins textures, CNNs win small parts' and 'DINOv3 vs DINOv2: v2 wins' - rest on a comparison that confounds architecture with feature depth. The last ViT block is the most semantically abstract and the least localised; using it alone is the ViT analogue of running the CNN on layer4 only, which PatchCore explicitly avoids. If a mid-block DINOv2 stack (the standard choice for dense tasks) is stronger, the project has ruled out its most promising representation for the wrong reason - and representation is the declared path to closing the AU-PRO gap.

**Fix.** Use timm's `get_intermediate_layers` (or forward hooks) to pull 2-4 intermediate blocks (e.g. thirds of the depth), concatenate them the way the CNN branch concatenates stages, and re-run B1/F/G. Then re-state the backbone conclusion, or withdraw it.

### R-05 · Fusion memory bank is built from a stride-2 subsampled grid but queried at full stride

**MAJOR · correctness · patchcore-core** — `ad2_feature_fusion.py:378`

Line 378 sets `stride_tr = 2 if sc_arm in ("wrn50_l123", "fusion") else 1` and line 379 builds the bank with it; `forward_feats` implements it as `fmap = fmap[:, :, ::stride, ::stride]` (line 141/158/...). Query features are always extracted with `stride=1` (line 262). The memory bank therefore contains descriptors only from even-index grid positions - one quarter of the spatial sites - while every test patch, including all odd-phase sites, is scored against it. In a nearest-neighbour detector the gallery and the query must be drawn from the same distribution; here they are not, and no arm was run with matched stride to check what the mismatch costs.

**Evidence.** The four scenarios routed to a stride-mismatched arm are the four with the worst pixel AUROC in outputs/ad2_feature_fusion.json: sheet_metal (wrn50_l123) 0.5306, rice (fusion) 0.6489, wallplugs (wrn50_l123) 0.7409, walnuts (fusion) 0.8296. The two stride-1 arms score 0.9734 (fabric, dinov2) and 0.8726 (vial, wrn50_l23). sheet_metal at 0.5306 is essentially chance, against 0.9166 for plain arm A on the same scenario.

**Impact.** HANDOFF.md promotes this to 'Architectural lesson 3: Training Stride vs. Inference Stride for Micro-Defects' and asserts that unstrided evaluation 'retains the full sub-patch spatial fidelity needed for hairline detection'. That is stated as a design principle without ever testing the consistent alternative (stride=2 on both sides), which costs the same memory and removes the asymmetry. The confound is entangled with the crop-geometry problem above, so neither the lesson nor the routing decision it drives can be trusted.

**Fix.** Add a matched-stride arm (stride=2 for both bank and query) and a stride-1-both arm where memory permits, and compare all three on the same geometry. If the mismatch is harmless, the lesson stands with evidence; if not, the four affected scenarios need re-scoring. Either way, record `stride_tr` and the query stride in the output JSON - neither is currently written.

### R-06 · --geometry defaults to "crop", the frame the project proved destroys AU-PRO

**MINOR · reproduction · patchcore-core** — `ad2_pixel_eval.py:279`

`ap.add_argument("--geometry", choices=["crop","squash","letterbox","aspect"], default="crop")`. The project has established (commit 06038e8, HANDOFF.md:365-427, and the module docstring at lines 100-131 of this very file) that `crop` puts the anomaly map and the ground-truth mask in different coordinate frames and costs mean AU-PRO@5% 0.301 -> 0.131. Running the script with no flags silently reproduces the bug, and nothing in the run record flags it - `deviations` stays empty and the config block just records `"geometry": "crop"` alongside a normal-looking set of numbers.

**Evidence.** The default output file outputs/ad2_pixel_eval.json still carries the crop-geometry results (mean_au_pro@0.05 = 0.1307, mean_image_auroc = 0.6629) at the canonical path `OUT = "outputs/ad2_pixel_eval.json"`, with no marker distinguishing it from a valid run; the corrected numbers live under outputs/runs/ instead. ad2_pixel_eval.py:279 is also the only place `crop` is still reachable - the E0 registration test (commit 4a84367) already records that crop fails registration while squash and letterbox pass.

**Impact.** The cheapest way to lose the project's largest single win is to forget one flag. The stale canonical JSON compounds it: a reader or a downstream script picking up outputs/ad2_pixel_eval.json gets the void numbers, which are the ones the README and HANDOFF's 'baseline' column still quote.

**Fix.** Change the default to `aspect`, and either drop `crop` from `choices` or gate it behind an explicit `--allow-broken-geometry` that appends an entry to `summary["deviations"]`. Separately, delete or rename outputs/ad2_pixel_eval.json so the canonical path cannot serve pre-fix numbers.

### R-07 · vial's ge_16x AU-PRO is 0.6247 in the artifact, not the 0.7516 quoted three times

**CRITICAL · reproduction · inference-validity** — `HANDOFF.md:556`

The claim "`vial`'s `>= 16x` regions reach 0.7516 against the published 0.764" cannot be reproduced from any file in the repo. In `outputs/runs/E5a-region-breakdown.json` the value is `scenarios.vial.buckets.ge_16x.mean_au_pro@0.05 = 0.6246926498746853`. The string 0.7516 appears in no JSON, no log (including `logs/E5a.log` and `logs/pod-tmp/`), and no other artifact. It is not vial's @30% (0.8704), nor vial's >=4x pooled value (0.6769), nor any other aggregation I could construct. This number is the load-bearing evidence for "large defects are already at or near published parity" and it is quoted verbatim in three places.

**Evidence.** grep -rn '7516' over outputs/ and logs/ returns nothing; the only 0.751x hits are `E1R-squash-448.json` image_auroc 0.7519. Recomputed from E5a-region-breakdown.json: vial buckets = sub_cell 0.9290 (n=28), 1_to_4x 0.6793 (n=14), 4_to_16x 0.8125 (n=35), ge_16x 0.6247 (n=91). BRIDGE.md:210 (worker M-06), BRIDGE.md:253 (planner M-08, which explicitly asserts "the reported values reproduce"), HANDOFF.md:556.

**Impact.** The single strongest stated piece of evidence that resolution alone can reach published parity is 0.14 too high. At the true 0.6247, vial's largest-defect bucket is barely above the pooled ge_16x mean of 0.6056 and 0.14 below the published 0.764 -- no scenario reaches parity in any bucket. It also means the M-08 audit's verification claim is false, so the audit trail cannot be relied on as an independent check.

**Fix.** Replace 0.7516 with 0.6247 at HANDOFF.md:556 and BRIDGE.md:210/253, and append a correction entry to BRIDGE noting that the M-08 verification did not catch it. Strike the phrase "directly reaching published parity" -- no per-scenario bucket in the artifact reaches 0.764. Add a check to the audit checklist in HANDOFF §0 that every number quoted in prose must grep-match an artifact.

### R-08 · The already-committed 224 arm falsifies D-04's registered prediction 2 and is filed as a non-result

**CRITICAL · inference-validity · inference-validity** — `E5-inputres-224.json:1`

BRIDGE M-13 (BRIDGE.md:461) files the partial 224 arm as "evidence about what happened" that "must be re-run, not completed in place" -- correct about stitching a record, but it caused a decisive result to go unread. The 224 arm ran under the M6-pinned bucket edges (hash c03ff2a7) and its per-scenario bucket counts are bit-identical to the 448 arm for all 6 completed scenarios, so the 224 vs 448 bucket comparison is valid exactly as D-04 specifies. It shows every bucket moving with resolution, with the sub-cell bucket the *least* responsive of the three smallest -- the outcome D-04 named as "resolution is a confound rather than the cause".

**Evidence.** Pooled over the 6 completed scenarios (counts verified identical between the two files), AU-PRO@5% at 224 -> 448: sub_cell (n=618) 0.0873 -> 0.2130, delta +0.126; 1_to_4x (n=246) 0.2059 -> 0.4263, delta +0.221; 4_to_16x (n=101) 0.3960 -> 0.6439, delta +0.248; ge_16x (n=163) 0.4672 -> 0.5639, delta +0.097. BRIDGE.md:83 registers "the ge_16x bucket moves less than 0.03"; it moves 0.097, 3.2x that bound. Per scenario, the largest gain is vial (+0.2584), the scenario with the largest defects (median 17.1 cells).

**Impact.** The ceiling argument in M-10 and HANDOFF's "resolution recovers the sub-cell half" framing both rest on the assumption that resolution acts specifically on sub-cell regions. The repo's own data contradicts that on the one resolution step already measured, and the project is about to spend a ~45-70 min GPU sweep to re-derive it. It also means the E5 verdict risks being read as confirming a mechanism the existing arm already argues against.

**Fix.** Analyse the committed 224 arm now as a standalone 6-scenario result before re-running E5, and record the ge_16x delta of +0.097 against prediction 2 in BRIDGE. Keep the re-run for the record contract, but state up front that the 448<->224 step already shows all buckets moving together, so the 448->768 arm must be interpreted as a test of the same pattern rather than as a first look. Note honestly that the 224 step is the downward direction and covers 6/8 scenarios.

### R-09 · The monotone size-vs-AU-PRO trend is about half between-scenario composition, not a size effect

**MAJOR · inference-validity · inference-validity** — `HANDOFF.md:546`

The headline "AU-PRO@5% is monotonic in defect size: 0.2265 / 0.4280 / 0.5498 / 0.6056" is computed by pooling regions across scenarios, and the buckets have very different scenario composition. Decomposing the pooled bucket means against what scenario mix alone predicts, roughly half the 0.379 sub-cell-to-ge_16x gap is composition. Within scenarios the effect is weaker, is not significant at n=7, and reverses in 2 of 7. HANDOFF.md:530 states the mechanism as "A defect that fits inside one cell cannot be localised at all"; vial's 28 sub-cell regions score 0.929, its highest bucket.

**Evidence.** From outputs/runs/E5a-region-breakdown.json. Bucket composition: sub_cell is 47% (354/756) can+sheet_metal, the two lowest-scoring scenarios; ge_16x is 66% (163/247) vial+walnuts, the two highest. Scenario-mix-alone prediction: sub_cell 0.3198, 1_to_4x 0.3390, 4_to_16x 0.4448, ge_16x 0.5210 -- i.e. 0.201 of the 0.379 pooled gap. Within-scenario contrast (ge_16x minus sub_cell), n=7: fabric -0.046, fruit_jelly +0.054, rice +0.328, sheet_metal +0.645, vial -0.304, wallplugs +0.536, walnuts +0.364; mean +0.225, sd 0.338, one-sample t = 1.76, p = 0.128. Only 3 of 8 scenarios are monotonic across their own buckets (can trivially, rice, sheet_metal).

**Impact.** The pooled table overstates the size effect by roughly 2x and presents as a clean law something that is not significant once scenario is controlled. This table is the sole justification for E5 being "the load-bearing experiment" and for the queue ordering in M-11. vial's reversal specifically refutes the stated mechanism that sub-cell defects are unlocalisable.

**Fix.** Report the bucket table with scenario controlled -- either the per-scenario paired contrast with its t/CI, or bucket means after centring each region's PRO on its scenario mean -- alongside the pooled numbers, and label the pooled version as confounded with scenario. Add the two reversing scenarios (vial, fabric) to the text next to the monotonicity claim, and delete or qualify "a defect that fits inside one cell cannot be localised at all" given vial sub-cell = 0.929.

### R-10 · The M-10 ceiling compares a region-pooled 0.606 against scenario-mean 0.344 and 0.764

**MAJOR · correctness · inference-validity** — `HANDOFF.md:624`

The ceiling argument sets 0.6056 (the ge_16x bucket mean, pooled over 247 regions) against a published 0.764 and a current 0.344, but 0.344 and 0.764 are means over the 8 scenarios while 0.6056 is a mean over regions. The project's own micro-averaged current score is 0.3709, not 0.3444, so the two frames are not interchangeable. In the macro frame that matches the comparison target, the ceiling is 0.518 over the 7 scenarios that have any ge_16x regions -- and can has zero such regions at all, so its ceiling is undefined and cannot be 0.606. Separately, the stated condition ("assumes higher resolution does not also lift ge_16x") is not sufficient: the argument also needs that no region can score above the ge_16x mean, and vial's sub_cell bucket (0.929) already exceeds its own ge_16x bucket (0.625).

**Evidence.** Recomputed from outputs/runs/E5a-region-breakdown.json: sum(n_i * au_pro_i)/sum(n_i) = 0.3709 vs recorded mean_au_pro@0.05 = 0.3444. Macro ge_16x over the 7 scenarios that have one = 0.5181; can has ge_16x count 0. HANDOFF.md:620-632 and BRIDGE.md:335-353 both quote 0.6056 -> "~0.61" -> "~0.16 below 0.764". HANDOFF.md:626 already notes the macro version (0.625) but dismisses it as noisy and instructs "use the global 0.606" -- the wrong frame for the comparison being made.

**Impact.** The stated headroom left for representation ("~0.16") is wrong in the frame that matters; in the macro frame it is ~0.25. The direction happens to strengthen the conclusion, but the arithmetic is presented as decisive ("this is arithmetic rather than inference") and it is not. A number this specific, used to promote E7 to a load-bearing experiment, must be in the same averaging frame as its comparison target.

**Fix.** Recompute the ceiling as a scenario mean (0.518 over 7 scenarios, and state explicitly that can contributes no ge_16x regions), or restate the whole comparison in micro terms (current 0.3709, ceiling 0.6056) and say which frame the published 0.764 uses. Add the second condition to the argument -- that the ge_16x mean upper-bounds achievable per-region scores -- and note that vial (0.929 sub_cell vs 0.625 ge_16x) already violates it.

### R-11 · r=+0.788 rests entirely on vial; no confidence interval is reported anywhere in the project

**MAJOR · inference-validity · inference-validity** — `HANDOFF.md:557`

Both correlations reported at n=8 scenarios are given as bare point estimates. r(median region/cell, AU-PRO) = 0.788 collapses to 0.247 (p = 0.59) when vial alone is removed -- vial's median is 17.1 cells against 3.3 for the next largest, so it is a single high-leverage point carrying the entire relationship. The comparison "detection quality predicts AU-PRO better than defect scale does (R^2 0.744 vs 0.605)", which is the stated reason E7 becomes a load-bearing experiment, is not distinguishable at this sample size: a bootstrap over scenarios puts the R^2 difference at 0.176 with 95% CI [-0.35, +0.80] and a 30% chance the ordering reverses.

**Evidence.** Recomputed from E5a-region-breakdown.json + exp_e5a_region_sizes.json. Leave-one-out r(med/cell, au): drop vial -> 0.247 (p=0.593); every other deletion leaves r in [0.778, 0.883]. Fisher-z 95% CIs: r(size) [0.16, 0.96], r(imgAUROC) [0.40, 0.98] -- almost fully overlapping. Incremental F-tests on the two-predictor model (R^2 0.877, adj 0.828): size over detection F(1,5)=5.41 p=0.068; detection over size F(1,5)=11.1 p=0.021. 20k-resample bootstrap of R^2(img) - R^2(size): mean 0.176, 95% CI [-0.347, +0.797], P(<0) = 0.302.

**Impact.** The correlation is quoted at HANDOFF.md:557 and BRIDGE.md:219 as if it were an established relationship, and the R^2 ranking at HANDOFF.md:580 and BRIDGE.md:149 drives the experiment queue. Both are single-point-driven or indistinguishable at n=8. r(imgAUROC, au) is in fact the stable one under leave-one-out (0.83-0.92), so the queue ordering may survive -- but not for the reason given, and the size correlation that motivates E5 does not survive at all.

**Fix.** Report all n=8 correlations with a CI and a leave-one-out sensitivity, and state that vial drives r(size). Replace the R^2 ranking with the incremental F-tests plus the bootstrap CI on the difference, and say plainly that at n=8 the two predictors cannot be ordered. Since detection-quality correlation is stable while size correlation is not, note that the evidence base for E5 is weaker than for E7 -- the opposite of the current framing.

### R-12 · The 0.01 equivalence threshold is unanchored and E3R's "definitively wins" fails a paired test

**MAJOR · inference-validity · inference-validity** — `HANDOFF.md:787`

HANDOFF.md:787 introduces a 0.01 equivalence threshold with no derivation, and LEDGER.md:19 records aspect as "definitively" winning because +0.029 > 0.01. A paired test over the 8 scenarios does not support "definitively": mean difference +0.0290, sd 0.043, 95% CI [-0.007, +0.065], paired t p = 0.098, Wilcoxon p = 0.055. Nothing in the repo measures run-to-run variance of AU-PRO on AD 2 at all -- every AD 2 run is a single greedy-coreset draw. The only variance measurement, seed_variance.json, is AD 1 image AUROC on 6 categories, where the per-seed mean already ranges up to 0.0169 for two arms -- larger than the threshold, on a strictly more stable metric.

**Evidence.** Recomputed from E1R/E2R/E3R JSONs, per-scenario aspect-minus-squash: can +0.0210, fabric +0.0317, fruit_jelly -0.0070, rice +0.0319, sheet_metal +0.1289, vial +0.0230, wallplugs -0.0010, walnuts +0.0039. sheet_metal alone is 4.4x the mean margin. Excluding it: mean +0.0148, p = 0.050. seed_variance.json per-seed mean AUROC spreads: A_wrn50_224 range 0.0011, B0_dinov2_224 0.0169, B1_dinov2_392 0.0138, E_resnet50_224 0.0163. HANDOFF.md:119 separately notes a torch 2.13->2.14 bump moved an arm by 0.010.

**Impact.** Every downstream experiment is anchored to a geometry choice whose margin's confidence interval includes zero, and the project makes repeated 0.01-margin calls (E6's "<0.03 is a weak lever", E5's monotonicity read, D-04's 0.03/0.10 predictions) against a noise floor that has never been measured on this metric or this dataset. The correct read of E3R is "aspect wins 6 of 8 with a small consistent margin", which is defensible; "definitively" is not.

**Fix.** Run the coreset seed at 3-5 values for one arm on AD 2 and report the sd of mean AU-PRO@5%; set the equivalence threshold from that measurement rather than by assertion. Restate the E3R verdict in LEDGER.md:19 with the paired CI [-0.007, +0.065] and the 6-of-8 sign count, and drop "definitively". Note that the mean margin is dominated by sheet_metal (+0.129) and that the sign consistency, not the mean, is the evidence.

### R-13 · Every reported number and every model selection is on test_public, violating the project's own rule 5

**MAJOR · design-gap · inference-validity** — `ad2_pixel_eval.py:82`

HANDOFF.md:56 states "Never select or tune on `test_public`. Anything fitted on `test_public` is not a benchmark number". But `ad2_pixel_eval.py` loads `validation` at line 82, records only its count as `n_val` at line 550, and never uses it: every metric comes from `test_public/good` and `test_public/bad`. The geometry winner (E1R vs E2R vs E3R), the eval-side choice (E4), and the pending E5 resolution choice are all selected by comparing mean AU-PRO@5% on test_public, and the resulting 0.3444 is then reported at HANDOFF.md:201-207 as the project's standing against a published baseline. The rule is enforced only against the fusion routing in E7 (HANDOFF.md:1180) and nowhere else.

**Evidence.** ad2_pixel_eval.py:82 `val = sorted(glob.glob(... 'validation' ...))`; line 550 `"n_val": len(val)` is the only downstream use -- no metric, threshold, or selection reads it. n_val across scenarios sums to 302, and HANDOFF.md:350 states the validation split is defect-free. E5a per-scenario records carry n_val 46/43/37/35/19/41/33/48 and no validation-derived metric. HANDOFF.md:787 defines the geometry winner as "the one with the highest mean AU-PRO@5% across all 8 scenarios" -- computed on test_public.

**Impact.** Three selections have now been made on the split the headline is reported on, so 0.3444 carries a selection bias of unknown size (the geometry selection alone spanned a 0.049 range across three arms). Worse, E7 as specified is not executable: validation is defect-free with no masks, so neither AU-PRO@5% nor image AUROC can be computed on it, and the hypothesis at HANDOFF.md:1186 ("validation-selected routing beats the single best backbone on mean AU-PRO@5%") has no way to be evaluated. E7 will silently fall back to test_public selection unless a proxy is defined first.

**Fix.** State explicitly in HANDOFF §2 that the geometry and eval-protocol choices were made on test_public and that 0.3444 is therefore selection-biased, or re-derive the choices on a held-out subset of test_public bad images. For E7, define the validation-split selection criterion before running it -- validation has no defects, so the only available signals are unsupervised (e.g. the normal-score distribution shift that `ad2_shift_check.py` already computes, or held-out normal-image score calibration); write that criterion into the item or the item cannot be executed as specified.

### R-14 · The published baseline being chased (0.764 / 0.659 / 0.763) has no recorded source, split, or FPR limit

**MINOR · documentation · inference-validity** — `HANDOFF.md:201`

The comparison table at HANDOFF.md:201-207 and the 2.2x gap that has driven five experiments cite "published baseline" values 0.659 / 0.763 / 0.764 with no citation, no paper or table reference, no statement of which split they were measured on, and no statement of the AU-PRO FPR limit. The only related note is HANDOFF.md:729, "the published AD 2 improvement (8.87% -> 76.35% AU-PRO)", which is also uncited. Our own numbers are on test_public; HANDOFF.md:352 states that test_private is what makes AD 2 results externally credible, so the two sides of the table may be different splits.

**Evidence.** grep for '0.764', '0.659', '0.763' across HANDOFF.md and README.md returns only usages, never a source. Our own mean AU-PRO@30% is 0.5736 against @5% 0.3444 (outputs/runs/E5a-region-breakdown.json) -- so a limit mismatch alone would move the target by 0.23 and change the gap from 2.2x to 1.3x. The project has already had two defects in its own AU-PRO definition (coordinate frame, region set), each of which moved the number by more than the current gap's remaining share.

**Impact.** The entire E5/E6/E7 queue is justified by the size of this gap. If the target is @30%, or on test_private, or uses a different minimum-region rule than the 77-native-pixel floor chosen here, the gap being chased is partly a definitional artifact -- exactly the failure class §6 documents twice already. There is currently no way for a reader or auditor to check it.

**Fix.** Add a provenance line next to the table: source (paper/table/leaderboard), split (test_public vs test_private), FPR limit, and the region-inclusion rule used by the reference implementation. If the reference uses test_private, state that our numbers are not directly comparable and that only the private-server submission can settle it. If the reference limit is 30%, report our 0.5736 alongside 0.3444 in the same table.

### R-15 · Bank is fitted through timm's centre-crop transform but served on a full-frame squash

**CRITICAL · correctness · deployment** — `export_bank.py:75`

export_bank.py builds its preprocessing with `cfg = timm.data.resolve_data_config(...)` / `self.tfm = timm.data.create_transform(**cfg, is_training=False)` (lines 73-75), i.e. Resize(short side to img/crop_pct) + CenterCrop(img) — the exact transform ad2_pixel_eval.py:100 (`squash_transform`) exists to remove, and whose docstring states it keeps only 21.9% of sheet_metal's width and 40.1% of can's. The serving path does something completely different: model.py:296-297 does `F.interpolate(t, size=(self.img_size, self.img_size))`, a full-frame squash with no crop. So memory-bank vectors describe a centre sub-rectangle of the training image while query patches describe the whole frame stretched to a square. This is not fixed by E8 — refitting a real bank with this script reproduces it exactly.

**Evidence.** export_bank.py:73-75 `cfg = timm.data.resolve_data_config({}, model=self.model)` ... `self.tfm = timm.data.create_transform(**cfg, is_training=False)` vs model.py:296-297 `if t.shape[-2] != self.img_size or t.shape[-1] != self.img_size: t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)`. ad2_pixel_eval.py:100-133 documents the same mismatch as the bug that moved mean AU-PRO@5% from 0.131 to 0.301.

**Impact.** Every served patch distance is computed between features from two different fields of view and two different scales, so normal peripheral content the bank has never seen scores as anomalous. Separately, model.py can only emit a square input, so the `aspect` geometry that produces every headline number (0.724 / 0.850 / 0.344, grids like 28x112 for sheet_metal) is unrepresentable in the deployment — metadata.json carries a single scalar `img_size`, not (w_in, h_in).

**Fix.** Replace export_bank.py's `create_transform` with the same geometry the research path uses — reuse `ad2_pixel_eval.squash_transform` / `aspect_transform` rather than a second copy — and make model.py apply the identical resize. Add `geometry` and, for aspect, `in_w`/`in_h` to metadata.json and have model.py resize to (in_h, in_w) instead of a hardcoded square.

### R-16 · Threshold 0.55 is below the bank's distance floor, so IS_DEFECTIVE is constant True

**CRITICAL · correctness · deployment** — `metadata.json:12`

The shipped threshold is 0.55 (metadata.json, from the hardcoded `threshold = 0.550` at export_bank.py:282). I recomputed the minimum achievable distance against the committed bank.npy for non-negative (post-ReLU-shaped) queries across a range of norms: the smallest min-distance is 38.28 and it only grows with query norm. Since model.py:378 does `is_defective = (image_scores > self.threshold)`, IS_DEFECTIVE is True for every possible input — the output carries zero information. This is not merely 'hardcoded': the value is two orders of magnitude below the scale of the quantity it is compared against.

**Evidence.** Recomputed on the committed artifact: query norm 5 -> min dist 38.28; norm 20 -> 41.86; norm 39.8 -> 53.15; norm 100 -> 103.12. Real calibrated thresholds from this same codebase for the same arm (outputs/sweep_backbones.json, arms/A_wrn50_224/*/operating_point) are 35.41 (leather) to 66.52 (wood) — 64x to 121x the shipped 0.55. export_bank.py:282 `threshold = 0.550`; export_bank.py:310 `threshold = 0.50` is the fallback used whenever `--val-dir` is omitted.

**Impact.** Every served frame is classified DEFECT. Worse, the same failure survives E8: fitting a real bank without passing `--val-dir` takes the export_bank.py:310 fallback of 0.50 and ships an equally constant-true detector, with nothing in the artifact signalling that the threshold was never calibrated.

**Fix.** Make `--val-dir` mandatory (or fail loudly) rather than falling back to a literal; record `threshold_calibrated: true/false` and the calibration percentile and val-set size in metadata.json; have model.py refuse to load a bank whose metadata says the threshold is uncalibrated, or at minimum assert that the threshold is within the observed distance range of the bank's own nearest-neighbour distribution.

### R-17 · Served heatmap is a 9x9 box blur on the patch grid, not a Gaussian at pixel scale

**MAJOR · inference-validity · deployment** — `model.py:161`

metadata.json carries `gauss_sigma: 4.0`, but model.py does not apply a Gaussian. Line 159-161 turns it into `k = 2*round(4)+1 = 9` and builds a uniform box kernel `torch.ones(1,1,9,9)/81`, which lines 383-384 apply to the 28x28 patch grid, and only then line 385 upsamples to eval_side. The research path does the opposite in both respects: ad2_pixel_eval.py:244-256 has an explicit `# UPSAMPLE FIRST, then smooth at pixel scale` comment, interpolates to eval_shape, then applies a separable Gaussian with sigma=4 pixels. A width-9 box has std sqrt((81-1)/12) = 2.58 grid cells; the research Gaussian at eval_side=512 over a ~56-cell grid is 4/9.1 = 0.44 cells. The served map is therefore ~6x more smoothed, in grid-cell units, than anything ever evaluated.

**Evidence.** model.py:159-161 `k = int(2 * round(self.gauss_sigma) + 1)` / `self.blur_kernel = (torch.ones(1, 1, k, k, ...) / (k * k))`; model.py:383-385 pad -> conv2d on `grid_maps` [B,1,h,w] -> `F.interpolate(smoothed, size=(eval_side, eval_side))`. ad2_pixel_eval.py:245 `up = F.interpolate(g, size=eval_shape, ...)` then 248-256 build and apply `kern = torch.exp(-(xs**2)/(2*gauss_sigma**2))` separably. E4 (HANDOFF.md:997) does NOT cover this: it swept eval_side 512/1024/2048 with *proportional* sigma 4/8/16, holding relative smoothing constant, so it never tested smoothing scale.

**Impact.** No AU-PRO number in outputs/ characterises the map the server actually emits. Given E5a's finding that 49.4% of ground-truth regions are sub-cell, a 2.58-cell box blur smears each such defect over roughly 26 cells — the deployment is degraded on exactly the metric the project is trying to close, and silently. export_bank.py:329 compounds this by writing `eval_side = img_size` (224) while the research path evaluates at 512.

**Fix.** Port ad2_pixel_eval.py's upsample-then-separable-Gaussian block verbatim into model.py, keep `gauss_sigma` meaning pixels at eval_side, and write the real `eval_side` (512) into metadata rather than reusing img_size. Then add a parity test asserting the server's map matches `anomaly_maps()` for a fixed input.

### R-18 · README claims a bit-identical parity PASS and 4,000-vector benchmarks that no test or artifact supports

**MAJOR · reproduction · deployment** — `README.md:283`

README.md:283-285 tabulates three deployment rows at 'Bank Size: 4,000 vectors' with 'PASS (bit-identical)' as the verification status. The committed bank is 500 vectors (bank.npy shape (500, 1536), metadata coreset_size 500), and grep across the repo finds no parity test between the Triton path and the research path — the repo has that pattern elsewhere (test_prealloc.py:56, scratch/test_threaded_decode.py:42) but never applied it here. Given the geometry and smoothing divergences above, the two paths cannot be bit-identical. The gRPC row is '< 12.0 ms (est)' presented inside a table of measured numbers.

**Evidence.** README.md:283 `| **Direct Native Execution** | NVIDIA RTX 4000 Ada | 1 | 4,000 vectors | **6.34 ms** | **~157.7 FPS** | PASS (bit-identical) |`. Verified bank: np.load('bank.npy').shape == (500, 1536), and it reproduces `generate_synthetic_bank(dim=1536, size=500)` exactly (torch.allclose == True with seed 42). deployment/README.md:152-153 quotes a *different* pair, 6.68 ms / ~150 FPS, uncaveated under 'Benchmarked Performance', while HANDOFF.md:1215 says the 6.34 figure 'describes nothing and must not be quoted until re-measured'. outputs/runs/ has no deployment entry; outputs/LEDGER.md:28 lists E8-triton-bank as _pending_.

**Impact.** A reader of either README takes away a verified 157 FPS production deployment with a 4,000-vector bank and proven numerical parity. All three claims are false, and the two READMEs disagree with each other by 0.34 ms, so neither is traceable to a run. This is the most externally visible artifact of the project.

**Fix.** Strike the latency table from both READMEs until re-measured, or mark it explicitly as a synthetic-bank smoke number with the bank size (500) stated. Write an actual parity test (fixed image, fixed real bank -> assert max abs diff vs `ad2_pixel_eval.anomaly_maps` and `sb.patch_distances` below tolerance) before any row says 'bit-identical'.

### R-19 · test_client.py cannot fail: zero assertions on the main path, confounded fixtures on the other

**MAJOR · design-gap · deployment** — `test_client.py:385`

`run_verification`'s direct/grpc/http branch (lines 364-385) makes no assertion of any kind — it prints the score and then prints 'Verification PASSED successfully!' unconditionally. The deepstream-mock branch asserts only that the key 'anomaly_score' exists and that the heatmap is (224, 224) (lines 358-361); it never asserts that the defective frame scores above the normal one, or that is_defective differs between them. Compounding this, `create_synthetic_frame` seeds 42 for normal and 99 for defective (line 294), so the two frames are independent noise fields rather than one base with a defect added — even if a score-ordering assertion were added, it would be confounded by the different noise realisation.

**Evidence.** test_client.py:385 `print("\n[Triton Client] Verification PASSED successfully!")` with no preceding assert in that branch; test_client.py:358-361 the only two asserts, on key presence and shape; test_client.py:294 `np.random.seed(42 if not defect else 99)`.

**Impact.** This is the mechanism by which a bank of pure Gaussian noise, a threshold below the distance floor, and a wrong smoothing kernel all reached the repo and got benchmarked. Any of the three would have been caught by one assertion. The test currently only proves the process does not crash.

**Fix.** Build the defect fixture from the normal fixture (same seed, then draw the scratch/spot on a copy) and assert score_defect > score_normal by a margin, assert is_defective differs, and assert the argmax of the anomaly map lands inside the drawn defect's bounding box. Add an assertion that the loaded bank has no negative entries (post-ReLU features cannot), which alone would have rejected the committed bank.

### R-20 · Nothing in the pipeline distinguishes a fabricated deployment from a real one

**MAJOR · design-gap · deployment** — `export_bank.py:319`

export_bank.py:319-334 builds an identical metadata dict on both the synthetic and the fitted branch — there is no `synthetic`, `source`, `n_train_images`, or bank-hash field, so metadata.json for a noise bank is indistinguishable from one for a real bank. model.py then adds two more fabrication paths: lines 258-262 substitute `torch.randn(100, dim)` when no bank file is found (a printed WARNING, then the model reports ready), and lines 205-235 substitute a randomly-initialised 3-conv `_FallbackWideResNet` when timm is absent (a printed note, then the model reports ready). Line 169-174 prints 'Initialized successfully' in every one of these cases.

**Evidence.** export_bank.py:319-334 metadata dict, reached identically from the synthetic branch (line 279-288) and the fitted branch; model.py:261 `print(f"[PatchCore Triton] WARNING: No bank.pt found ... Initializing dummy bank (100, {dim})")` followed by normal startup; model.py:207 the timm-absent fallback. The only trace of the fabrication in the shipped metadata.json is `"device": "cpu"` — and note backbone.pt is absent from the model directory despite `--save-backbone-weights` defaulting True, so the export ran with no working extractor at all.

**Impact.** E8 replaces the bank but leaves every one of these paths intact, so the same class of artifact can be regenerated and shipped again with no signal. A Triton instance can come up fully healthy, answer requests, and serve a random backbone against a random bank.

**Fix.** Write provenance into metadata.json (synthetic flag, train image count, sha256 of the bank, backbone weight source) and have model.py refuse to initialize on a missing bank or a missing timm rather than fabricating one — a hard failure at load is the correct behaviour for a serving path. Fold the same fields into a startup log line so the deployment states what it is serving.

### R-21 · FP32 pixel values in [0,255] are silently fed to the backbone unnormalised

**MAJOR · correctness · deployment** — `model.py:301`

model.py:301 gates ImageNet normalisation on a range heuristic: `if t.min() >= 0.0 and t.max() <= 1.05`. config.pbtxt:19 declares IMAGE as TYPE_FP32 only, so the uint8 branch at model.py:272 is unreachable over Triton — despite config.pbtxt:14-15 promising 'the Python backend also gracefully handles uint8 [0, 255] and automatically normalizes'. A client that sends float32 in [0,255] (the natural result of `cv2.imread(...).astype(np.float32)`) fails the `<= 1.05` test, is never divided by 255 and never normalised, and the backbone receives values ~100x too large. No error is raised. test_client.py itself does this: the 4-D input path (lines 103-104 direct, 144 gRPC, 181 HTTP) does `image_np.astype(np.float32)` with no scaling and no range check.

**Evidence.** model.py:301 `if t.min() >= 0.0 and t.max() <= 1.05:` ; config.pbtxt:19 `data_type: TYPE_FP32` with no uint8 input declared; test_client.py:104 `batched = image_np.astype(np.float32)` in the `elif image_np.ndim == 4:` branch, versus the 3-D branch at line 100-101 which does check `if chw.max() > 1.0: chw /= 255.0`.

**Impact.** A silent wrong-answer path in the serving contract, guarded only by a magic constant. Batched clients — the ones dynamic_batching exists for — hit it by default. The scores are garbage but plausible-looking, and nothing downstream can tell.

**Fix.** Make the input contract explicit rather than inferred: add a `pixel_scale` / `already_normalized` field to metadata.json (or a Triton input parameter) and normalise deterministically; raise a TritonError on out-of-contract ranges instead of silently skipping normalisation. Fix the 4-D branches in test_client.py to apply the same scaling as the 3-D branch.

### R-22 · dynamic_batching is inert — execute() runs one backbone forward per request

**MINOR · optimization · deployment** — `model.py:353`

config.pbtxt:54-56 enables `dynamic_batching { max_queue_delay_microseconds: 5000 }` with `max_batch_size: 16`, and the comment at lines 52-53 claims 'e.g. 200 FPS throughput'. But Triton's Python backend hands grouped requests to `execute()` as a list; the model must concatenate them itself. model.py:353 instead loops `for request in requests:` and runs a separate `_extract_features` (a full WideResNet50-2 forward) and a separate `cdist` per request. So batching buys nothing, and the 5 ms queue delay is added latency with no compensating throughput.

**Evidence.** model.py:353 `for request in requests:` with the full preprocess/extract/cdist/response body inside the loop (lines 354-398); config.pbtxt:54-56 `dynamic_batching { max_queue_delay_microseconds: 5000 }`.

**Impact.** Up to 16 sequential WRN50 forwards where one batch-16 forward would do, plus up to 5 ms of pure queueing latency per request. The advertised throughput figure is unreachable by this implementation, which also means any re-measurement done for E8 will understate what the deployment could do.

**Fix.** Concatenate the preprocessed tensors across all requests in the list into one [sum(B), 3, H, W] batch, run a single forward and a single chunked cdist, then split the outputs back per request by their original batch sizes. Only then does the 5 ms queue delay earn its cost.
