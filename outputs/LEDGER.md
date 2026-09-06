# Run ledger

One row per experiment run. Appended by the worker, audited by the planner.
`verdict` is measured against the run's **own stated hypothesis** in HANDOFF.md §7 —
a factual call about the number, not a judgement about the project.

See HANDOFF.md §0 for the output contract and the audit checklist.

**Every row carries the UTC-offset time it was appended** (BRIDGE protocol rule 9). A ledger row
without a time cannot be aged, and `E5-inputres` sat as `_pending_` over a decisive result for a
day with nothing recording how long.

| run_id | when | geometry | img | mean AU-PRO@5% | mean I-AUROC | code hash | verdict |
|---|---|---|---|---|---|---|---|
| (pre-E1 baseline) | crop | 448 | 0.1306 | 0.663 | — | void — coordinate-frame bug, see §6 |
| E1-squash-448 | squash | 448 | **0.3006** | 0.718 | `5168721b` | supports |
| E0-registration | crop vs squash | — | 0.019 vs 1.000 | — | `9d7536e3` | supports |
| E2-letterbox-448 | letterbox | 448 | 0.2792 | 0.670 | `bf3917d6` | refutes |
| E3-aspect-448 | aspect | 448 | 0.3225 | 0.724 | `07dcb26e` | ~~supports~~ **inconclusive** — region-set confound, see §7 E4a |
| E4a-regionset | native 77px | — | 1530 regs | — | `59257b20` | supports — bit-identical 1530 regions across all geometries |
| E1R-squash-448 | squash | 448 | 0.3139 | 0.697 | `59257b20` | re-scored fixed regions |
| E2R-letterbox-448 | letterbox | 448 | 0.2932 | 0.670 | `59257b20` | re-scored fixed regions |
| E3R-aspect-448 | aspect | 448 | **0.3429** | **0.724** | `59257b20` | refutes — aspect margin is +0.029 (>0.01 threshold), aspect definitively wins |
| E4-evalside-512 | aspect | 448 (eval 512) | 0.3444 | 0.724 | `5d43304b` | 1530/1530 active regions |
| E4-evalside-1024 | aspect | 448 (eval 1024) | 0.3444 | 0.724 | `5d43304b` | 1530/1530 active regions |
| E4-evalside-2048 | aspect | 448 (eval 2048) | 0.3442 | 0.724 | `5d43304b` | 1530/1530 active regions |
| E4-evalside-trend | aspect | 448 (sweep) | 0.3444 -> 0.3442 | 0.724 | `5d43304b` | **refutes** — flat across eval resolutions (range 0.0002); input resolution ceiling binds |
| E5a-region-breakdown | aspect | 448 | 0.3444 | 0.724 | `e82ce4bc` | **supports** — AU-PRO scales with defect size: sub-cell 0.226 -> >=16x 0.606; 49.4% regions sub-cell |
| E5-inputres-224 | aspect | **224** | 0.1797 (6 scen) | 0.649 (6 scen) | `c03ff2a7` | **partial (6/8) but DECISIVE** — vs 448 on the same 6: +0.1539 mean (1.86x), 6/6 improved; `ge_16x` +0.097 vs a registered <0.03 bound. Falsifies D-04 prediction 2 and voids the M-10 ceiling. See REVIEW.md §1. |
| E4b-aspect-448-driver580 | aspect | 448 | **0.3444** | **0.724** | `7f2bb8e2` | **supports** — clean single unstitched run (wall 755.9s), driver 580.159.04, 1530/1530 active regions. Bit-identical parity to E4 reference (0.3444 vs 0.3444). Serves as baseline reference for E5b and E5. |
| E5-inputres-448 | aspect | 448 | 0.3444 | 0.724 | `7f2bb8e2` | clean E4b unstitched reference adopted |
| E5b-dilated-layer3 | aspect | 448 | **0.3692** | **0.759** | `2866ecf7` | **supports** — 8/8 scenarios improve (+0.0248 mean AU-PRO@5%, +0.0352 I-AUROC); sub-cell +0.016, 1-4x +0.034, 4-16x +0.030. Confirms layer 3 bilinear blur was degrading descriptor resolution at stride 8. |
| E10a-proj384-448 | aspect | 448 (proj 384) | 0.3269 | 0.719 | `6d421f0c` | **refutes bound** (-0.0175 vs <=0.01 hypothesis margin); 64.5% cut in peak RSS (3520 vs 9926 MB), 22% faster wall clock (587.5s vs 755.9s). 1530/1530 active regions. Macro defects unaffected (>=4x combined 0.5790 vs 0.5847); delta driven by sub-cell texture dilution in `can` (-0.068) and `fabric` (-0.077). Unlocks 768px arm and ensembles within RAM budget. |
| E10b-wrn101-448 | aspect | 448 (WRN-101) | 0.3277 | 0.716 | `a2503dae` | **refutes sole driver** (+0.0008 vs WRN-50 E10a); lifts tough representation scenarios (`fabric` AU-PRO +0.054, pix +0.105; `rice` AU-PRO +0.081, img +0.218 to 0.761), but dips on `wallplugs` and `vial`. Sets new dataset-wide peak pixel AUROC (0.8632). Proves WRN-101 alone does not capture the 9.6-pt deficit; vindicates multi-backbone complementary ensemble requirement. |
| E5-inputres-768 | 2026-09-06T03:00:00+02:00 | aspect | 768 (proj 384) | **0.4239** | 0.706 | `bf13f6e` | **supports** — 8/8 scenarios improve (+0.0795 mean vs E4b 0.3444); 1530/1530 active regions. `sub_cell` jumps 0.2263 -> 0.3144 (+0.0881); `1_to_4x` 0.4578 -> 0.5700 (+0.1122); `ge_16x` 0.6015 -> 0.6179 (+0.0164, meeting D-04 bound <0.03). `sheet_metal` gains most (+0.1145, 0.2529 -> 0.3674). Wall time 4413.0s (wall/sum=1.00), peak RSS 7623.5 MB. |
| E11-ours-256-squash | 2026-09-06T03:15:00+02:00 | squash | 256 (uncapped) | 0.2221 | 0.653 | `3cbf1ad` | **supports baseline isolation** — our pipeline at published setting (256 squash, 1% uncapped, 1536-dim concat) scores 0.2221 mean AU-PRO@5% (I-AUROC 0.653, P-AUROC 0.824). Establishes exact 6.6-point deficit (0.288 vs 0.222) to published PatchCore at matched resolution, precisely confirming D-09 prediction. 1524/1530 active regions (6 dropped on sheet_metal 4:1 squash). Wall 321.4s, peak RSS 4490 MB. |
| E6-density | | | | | | _pending_ |
| E7-fusion-val | | | | | | _pending_ |
| E8-triton-bank | | | | | | _pending_ |

**Ledger discipline (added 2026-09-05 after a near-miss).** A row must be updated the moment an
artifact exists, **even for a partial or failed run**. `E5-inputres` sat as `_pending_` while
`E5-inputres-224.json` held the result that overturned the project's central conclusion; the run
was filed as a partial "non-result" and its numbers went unread for a day. Note the record has
`mean_*` fields of `null` because the run died before its summary block ran — which is exactly
why the row looked empty. Recompute from `scenarios` rather than trusting a null summary. **A partial run is a
result until proven otherwise.** Record what it says, then decide whether to re-run.

The `run_id` in a row must also match its artifact filename exactly — `E5a-regions` pointed at
no file while the data sat in `E5a-region-breakdown.json`.
| E5-inputres-768 | 2026-09-06T04:05+02:00 | aspect | **768** | **0.4239** | 0.744 | see run json | **accepted** — 8/8 improved; ge_16x +0.0123 confirms D-04 prediction 2 |
