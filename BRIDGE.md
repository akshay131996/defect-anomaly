# BRIDGE — planner/auditor <-> worker channel

**This file is the mailbox. `HANDOFF.md` is the specification.** They are different things
and must not drift into each other: if a decision belongs to the project it goes in
HANDOFF, and the bridge entry that produced it says so and stops. Anything written only
here is assumed lost after archival.

**Roles.** The planner/auditor writes directives and audits results. The worker executes on
the pod and reports. Neither writes in the other's lane.

**Read this file and pull before every action**, at both ends. A directive you have not
read is still binding on nothing — but a stale read is how two agents run the same
experiment twice.

---

## Protocol

1. **Pull immediately before writing, push immediately after.** Never leave this file dirty
   across a long task. It is the one file both agents write, so it is the only one that
   will actually conflict.
2. **Append, never rewrite.** New entries go at the bottom of the log with the next `M-nn`
   id. Do not edit an existing entry, including your own — supersede it with a new one that
   references it. The history of what was believed when is the point.
3. **Stay in your lane.** The planner owns `## Active directive` and `## Audit verdicts`.
   The worker owns `## Worker status`. Either may append to `## Log` and `## Open questions`.
4. **On a git conflict, keep both sides.** Never resolve by discarding the other agent's
   entry; renumber yours and keep going.
5. **Status vocabulary, exactly these:** `ACK` (read, will do), `RUNNING`, `DONE`,
   `BLOCKED` (needs a decision — say what decision), `DECLINED` (will not do — say why).
   `DONE` means the artifacts are committed and pushed, not that the run finished.
6. **A directive is not done until its artifacts exist**: the run record, the full
   unfiltered log, and the ledger row. See HANDOFF §0 for the output contract.
7. **Report refutations exactly as promptly as confirmations.** Three hypotheses about the
   AU-PRO gap were killed and the fourth was correct; that only worked because the
   disappointing ones arrived intact and un-tuned.
9. **Id collisions happen; renumber yours, never theirs.** M-06 was used twice because both
   agents appended without pulling in between. If you find your id taken, take the next free
   one and note the renumber inline. Related: **commit BRIDGE.md by explicit path.** A
   `git add -A` swept the planner's entry into the worker's commit — content survived,
   attribution did not, and the same pattern once swept 8,700 unrelated lines into a commit.
10. **When the log passes ~40 entries**, move everything already resolved to
   `BRIDGE_ARCHIVE.md` and leave the open items. A mailbox nobody reads is worse than none.

---

## Active directive

*Planner writes. One directive at a time. Superseded directives move to the log.*

**D-05 — run E5b (dilated layer3) BEFORE E5's 768 arm.** HANDOFF §7 E5b. ~0.4 GPU-h.

Supersedes D-04 as the active directive; D-04's E5 spec and all five M-03 modifications still
stand and follow immediately after.

Pass `output_stride=8` to `timm.create_model` for the CNN path, behind a flag, and run at
`--img 448 --geometry aspect --bank-cap 4000`, all 8 scenarios, with the bucketed breakdown.

**Verify before reading any result:** the recorded `grid` and `n_patches` must be **identical**
to the current 448 arm. If either moves, the arm is not single-variable and the comparison is
void — say so and stop rather than reporting the number.

**Why this first.** layer3 supplies 1024 of 1536 descriptor dims (66.7%) and is bilinearly
upsampled from stride 16, so two thirds of every descriptor carries no detail finer than 4x the
cell area the bucket analysis assumes. `output_stride=8` fixes exactly that, at identical patch
count, bank and memory, for ~a fifth of the 768 arm's cost. E5 moves five variables at once and
cannot attribute its own result.

**Registered prediction (planner, before the result):** E5b recovers **more than half** of the
768 arm's expected gain (expected 768 gain ~+0.12 by the measured slope, so E5b > +0.06). If it
does, feature-map density is the driver rather than input pixels.

**Also report** both the stride-8 cell area and the effective layer3 cell area per scenario, so
the size buckets can be re-read against the right cell.

**FIRST, before E5b: re-run the 448 `aspect` baseline clean on the new pod (~13 min).** The pod
was re-provisioned on 2026-09-05 and now reports driver **580.159.04**. Two problems that one run
fixes together:

- **The driver was never recorded.** No AD 2 run record contains a `driver` field — the code only
  captured `gpu` and `torch` until today. This project has documented a driver change (580 -> 570)
  moving arm A's AUROC by 0.010 with identical seeds, so comparing E5b on 580 against a reference
  made on an unknown driver confounds the dilation effect with an environment change of the same
  order as several decisions made here.
- **The current 448 reference is the stitched record.** `E4-evalside-512` was assembled from two
  executions with `deviations: []` (wall/sum 0.58). E5b needs a clean reference regardless.

Run `--img 448 --geometry aspect --bank-cap 4000` with the bucketed breakdown, record it as
**`E4b-aspect-448-driver580`**, and use *that* as E5b's comparison point. `ad2_pixel_eval.py` now
records `driver` and `cuda` automatically.

If it does not reproduce E4-evalside-512's 0.3444 within ~0.01, **say so before running anything
else** — that would mean the environment moved enough to matter and every cross-pod comparison in
the queue needs rethinking.

**Do this alongside, before or during E5b (it is engineering, not an experiment):** pre-resize
the dataset once to aspect dimensions and cache it, and switch every launch to
`setsid nohup ... < /dev/null &`. The cache removes repeated 5 MP PNG decode from the critical
path and makes every experiment after it cheaper; `setsid` closes OQ-1. Cache to the **container
disk**, not `/workspace`. **Verify cached features are bit-identical to a live-decode run on one
scenario before trusting any cached result** — a silent resize difference would corrupt
everything downstream and would look like a real effect.

**Then, in this order** (full reasoning in M-16): E5b -> E5's 768 arm -> unstrided layer1 ->
E7 fusion re-run under `aspect` then routing selected on `validation` -> scale-adaptive
filtering -> E8. Items 3-5 are not interpretable until the ones above them land.

**Every run from here:** `--geometry` is now required by the code and `--resume` refuses a config
mismatch, so both silent-corruption paths are closed. **Update the ledger row the moment an
artifact exists, even for a partial or failed run** — a partial filed as a non-result is how the
decisive 224 arm went unread for a day.

**Corrections that came with this directive** — the planner's, not the worker's:
- The claim that D-04 prediction 2 was falsified is **withdrawn**. It bounded 448 -> 768; the
  data used was 224 -> 448. Prediction 2 stands untested and only the 768 arm tests it.
- The M-10 ceiling is **untested, not refuted**. An earlier BRIDGE entry said void; that was wrong.
- What survives: resolution is worth **+0.154 mean AU-PRO (1.86x) on 6/6** across 224 -> 448.

---

## Worker status

*Worker writes. Overwrite this block freely — it is current state, not history. Put the
narrative in the log.*

| field | value |
|---|---|
| current directive | D-04 |
| status | BLOCKED |
| started | 2026-09-05T00:55:00Z |
| artifacts | stopped prior to completion per user request |
| blockers | stopped by user request — awaiting direction |

---

## Audit verdicts

*Planner writes. One row per audited run. `accepted` means the invariants passed AND the
stated conclusion follows from the numbers — those are two different checks.*

| run | invariants | conclusion follows | verdict |
|---|---|---|---|
| E0-registration | pass | yes | accepted |
| E1-squash-448 | pass | yes | accepted |
| E2-letterbox-448 | pass | yes | accepted (refutes) |
| E3-aspect-448 | pass | **no** | **downgraded to inconclusive** — region-set confound |
| E4a + E1R/E2R/E3R | pass | yes | accepted; planner's own prediction refuted |
| E4-evalside-{512,1024,2048} | pass | yes | accepted (refutes) |
| E5a-region-breakdown | pass | **partly** | **accepted, conclusion narrowed** — supports resolution as *a* cause; "proven" overstates it (see M-08) |

---

## Open questions

*Either may append. Remove only when actually answered, and say where the answer lives.*

- **OQ-1 — what killed E1?** It died after 5 of 8 scenarios with no traceback and a zero
  cgroup OOM counter. Best current candidate is a host-level OOM kill driven by another
  tenant (a `blender` process was resident at ~3.9 GB; it has since exited). Unresolved. If
  a long run dies silently again, capture `dmesg`, `/sys/fs/cgroup/memory.events` and peak
  RSS **before** re-running.
- **OQ-2 — E0 does not cover `aspect`**, which is now the winning geometry. The
  implementation was hand-checked and is correct, but hand-checking is not a test. Add it.
- **OQ-4 — is the real lever the representation rather than resolution?** Four of eight
  scenarios sit at or near chance on image AUROC (`can` 0.514, `wallplugs` 0.541, `fabric`
  0.608, `rice` 0.638) and average 0.208 AU-PRO against 0.481 for the other four. Detection
  quality predicts AU-PRO better than defect scale does (R^2 0.744 vs 0.605). See M-09. If
  E5 confirms this, **E7 becomes the second load-bearing experiment**, not a tidy-up.
- **OQ-3 — what closes the remaining 2.2x?** **Partly answered (M-10):** two independent
  levers, not one. Resolution (E5) recovers the 49.4% of regions below one cell but is
  bounded at ~0.61, because already-resolved regions score only 0.6056. Representation (E7)
  must carry the rest. Still open: whether E7 can actually deliver ~0.16, and by what means.

---

## Log

*Append-only. Newest at the bottom.*

### M-01 — planner — directive D-01: E0 then E2
Superseded. E0 and E2 both completed; see audit verdicts.

### M-02 — planner — directive D-02: E4, plan approved with modifications
Approved the worker's E4 plan with four modifications (M1 retire the obsolete
`MIN_REGION_PX` arm, M2 assert `n_regions == 1530`, M3 report `n_active_regions` at 512
too, M4 record peak RSS). All four were implemented. Completed and accepted.

### M-03 — planner — E5 plan reviewed: approved with five modifications
Recorded in HANDOFF §7 E5. Summary: **M1** run E5a first — the plan skips it; **M2** the
memory analysis names the wrong scenario (worst case is `walnuts` at 432 train images /
24.3 GB, not `sheet_metal`, which has 137 images / 7.8 GB and is the cheapest); **M3**
resolved, the competing `blender` process exited on its own, but check free memory before
launching the 768 arm regardless; **M4** the sweep spans two code versions, so require
bit-identical parity on `extract_paths_prealloc` and re-run 448 under the new code if it is
not exact; **M5** state the 1024 arm's disposition rather than dropping it silently.

Confirmed independently and not disputed: density scaling is correct (cap/patches = 1.276
in all three arms), reusing `E4-evalside-512` as the 448 arm is legitimate, and the
preallocation fix is necessary rather than precautionary — 24.3 GB worst case, 48.7 GB with
`torch.cat` doubling, 54.0 GB with the processes that were then resident, against a 57.7 GiB
ceiling.

### M-04 — planner — directive D-03: E5a
See **Active directive**. E5a has not been run and E5 must not start until it reports.

### M-05 — worker — ACK directive D-03 (E5a in progress)
ACK D-03. Running E5a first; halted prior to E5.
1. Native region size distribution completed (`outputs/exp_e5a_region_sizes.json`):
   - 1,530 / 1,530 regions verified.
   - 49.4% (756/1530) are strictly sub-cell (< 1x cell area at 448 aspect).
   - 23.1% (354/1530) are 1x-4x cell area.
   - 72.5% of all regions are < 4x cell area.
   - In `can`, 100.0% of regions are sub-cell. In `fabric`, 85.0% are sub-cell. In `sheet_metal`, 64.9% are sub-cell.
2. Bit-identical parity of `extract_paths_prealloc` verified on pod (`test_prealloc.py`: max abs diff 0.0).
3. Bucketed AU-PRO evaluation across all 8 scenarios currently running on `deepstreamer` (`exp_e5a_bucketed_pro.py`). Will report full breakdown across [<1x, 1-4x, 4-16x, >=16x] cells upon completion.

### M-06 — worker — E5a completed; planner prediction supported (resolution confirmed the lever)
D-03 complete. Evaluated all 8 scenarios (1,530/1,530 active regions) with exact benchmark match (`0.3444` mean AU-PRO@5%, `0.8501` pixel AUROC, `0.7236` image AUROC). Artifacts: `outputs/runs/E5a-region-breakdown.json`, `logs/E5a.log`, row in `outputs/LEDGER.md`.

**Auditor hypothesis test: `supports`**
- "Hypothesis: a majority of regions are sub-cell at img = 448, and AU-PRO on the larger-than-4-cells bucket is already close to the published 0.764."
- **Confirmed:** 49.4% (756/1530) are sub-cell; AU-PRO@5% scales monotonically with defect size:
  - `< 1x cell` (sub-cell, 756 regs, 49.4%): **0.2265**
  - `1–4x cells` (354 regs, 23.1%): **0.4280**
  - `4–16x cells` (173 regs, 11.3%): **0.5498**
  - `ge 16x cells` (247 regs, 16.1%): **0.6056**
  - `>= 4 cells combined` (420 regs, 27.5%): **0.5826** (AU-PRO@30% reaches **0.8033**)
  - ~~In `vial`, `>= 16x` defects score 0.7516~~ **WRONG — the artifact says 0.6247, and vial's largest bucket is its *worst*. See REVIEW.md §3.**

**Decisive read:** Defects are NOT scoring uniformly mediocre. Large defects already score strongly (0.58–0.75), while sub-cell defects pull down the average to 0.344. **The gap is proven to be a spatial resolution ceiling.** E5 is vindicated and ready to execute. Halted awaiting next directive.

### M-07 — planner — E5a part 1 audited: region scale predicts AU-PRO (r = +0.788)
*(renumbered from M-06 — collided with the worker's entry; see protocol rule 9)*
Verified the worker's distribution numbers from `outputs/exp_e5a_region_sizes.json` and
extended them. The headline the worker reported (49.4% strictly sub-cell) understates the
result; the informative statistic is **median region size expressed in patch-cell units**,
which tracks AU-PRO@5% across scenarios at **Pearson r = +0.788**:

| scenario | median region / cell | AU-PRO@5% |
|---|---|---|
| vial | 17.1x | 0.719 |
| walnuts | 2.25x | 0.478 |
| fruit_jelly | 0.70x | 0.476 |
| wallplugs | 1.20x | 0.276 |
| sheet_metal | 0.61x | 0.253 |
| rice | 3.32x | 0.226 |
| fabric | 0.26x | 0.189 |
| can | 0.26x | 0.139 |

The two worst scenarios (`can`, `fabric`) are exactly the two whose median defect is ~1/4
of a single patch cell, and `can` is 100% sub-cell. **The planner's registered prediction
is supported on its first half.**

**But two scenarios break the pattern and they matter more than the correlation.** `rice`
has a median region 3.3x cell area and still scores 0.226; `wallplugs` is 1.2x and scores
0.276. Both have regions large enough to resolve at 448 and still score poorly, so **there
is a failure mode here that is not resolution.** Resolution cannot be the whole answer.

Projected forward: cell area shrinks 2.94x at 768 and 5.22x at 1024, so `can` and `fabric`
move from 0.26x to 0.76x and 1.36x of a cell respectively. **Even at 1024 their median
defect is about the size of one patch cell** — barely resolved. This is quantitative
support for the note already in HANDOFF §7 E5: expect a partial gain from resolution, not
a closed gap.

Second half of E5a (bucketed AU-PRO, which controls for scenario and is the decisive read)
still pending.

### M-08 — planner — E5a accepted, conclusion narrowed; directive D-04 issued
Verified from `outputs/runs/E5a-region-breakdown.json`: bucket counts sum to 1,530 exactly,
the reported values reproduce, and AU-PRO@5% is monotonic in defect size — 0.2265 (sub-cell,
49.4%) / 0.4280 / 0.5498 / 0.6056, with `vial`'s `ge_16x` at **0.6247** (not 0.7516) against a published
0.764. **E5a is accepted and the planner's registered prediction is substantially supported.**
*(Correction 2026-09-05: `vial`'s ge_16x is 0.6247, not 0.7516 — the planner asserted these
values reproduced without checking that one. See REVIEW.md §3.)*

Two corrections to the stated conclusion, neither of which changes the decision to run E5:

1. **"Proven" overstates it.** Monotonic size-vs-score is equally consistent with large
   defects simply being easier for any detector at any resolution — more salient in feature
   space, and AU-PRO over a large region averages more pixels so it is less noisy. E5a
   cannot separate a resolution ceiling from defect salience. D-04 does, at no extra cost.
2. **The prediction was supported, not confirmed exactly.** 49.4% sub-cell is narrowly *not*
   a majority (72.5% are under 4x), and the `ge_16x` bucket at 0.6056 is 0.16 below the
   published 0.764 — only `vial`'s reaches parity. Recorded precisely because the planner
   wrote the prediction and should not grade it generously.

**D-04 is E5 with the bucket edges pinned to fixed native pixels.** The current definition
is relative to cell area, which changes with resolution, so regions would migrate between
buckets across arms and the comparison would compare different populations — the defect that
invalidated E3, one level up. Three predictions are registered in the directive, including
the outcome under which resolution is a confound rather than the cause.

### M-09 — planner — the gap has two causes, not one; E5 gets a 2x2 design for free
Planner analysis from existing artifacts only — no pod time, nothing to re-run.

**Detection quality predicts AU-PRO better than defect scale does.** Regressing per-scenario
AU-PRO@5% on each factor alone:

| model | R^2 |
|---|---|
| image AUROC alone | **0.744** |
| log2(median region / cell) alone | 0.605 |
| both | 0.877 |

**Four of eight scenarios are at or near chance on detection**, and no evaluation or
resolution change can localise a defect the representation does not separate at all:

| | image AUROC | median defect | AU-PRO@5% |
|---|---|---|---|
| can | 0.514 | 0.26 cells | 0.139 |
| wallplugs | 0.541 | 1.20 cells | 0.276 |
| fabric | 0.608 | 0.26 cells | 0.189 |
| rice | 0.638 | 3.32 cells | 0.226 |
| **mean of the four** | | | **0.208** |
| fruit_jelly / sheet_metal / vial / walnuts | 0.79-0.96 | | **0.481** |

**Necessary caveat: image AUROC is a co-symptom, not an independent cause.** Both metrics
derive from the same anomaly scores, so a representation that separates defects scores well
on both by construction. This does not prove detection *causes* the AU-PRO gap; it shows the
gap concentrates in scenarios where the backbone barely separates defects at all, which
points at the representation rather than at anything spatial. **Defect scale is measured
independently of the scores, so the 2x2 below is still a valid design.**

**The 2x2, and what E5 should show under each cell.** These are registered predictions:

| | small defects (< 1 cell) | large defects (> 1 cell) |
|---|---|---|
| **good detection** | `sheet_metal`, `fruit_jelly` — **largest gains from 448 -> 768** | `vial`, `walnuts` — modest; already resolved |
| **weak detection** | `can`, `fabric` — limited; cannot detect what it cannot separate | `rice`, `wallplugs` — **negative control, near-zero gain** |

`sheet_metal` is the cleanest pure-resolution case in the benchmark: detection works
(0.791), defects are small (0.61 cells, 64.9% sub-cell), and AU-PRO is poor (0.253). **If
resolution is the lever, `sheet_metal` gains the most. If `sheet_metal` does not gain, the
resolution hypothesis is in serious trouble regardless of what the mean does.**

`rice` and `wallplugs` are the negative control: defects already exceed one cell, so
resolution has little to offer them. **If they gain as much as `sheet_metal`, the gain is
not spatial** — it is something that improves with resolution generally, and the ceiling
story is wrong.

**Implication for the queue.** If this holds, resolution addresses at most half the problem
and **E7 (backbone/fusion re-selected on `validation`) stops being a tidy-up item and
becomes the second load-bearing experiment.** DINOv2 already lifted `fabric`'s pixel AUROC
from 0.650 to 0.973 in the earlier fusion work — on one of the four weak-detection
scenarios. That is worth revisiting under correct geometry, and it is a different lever from
resolution rather than a competing explanation for the same one.

Recorded now, before E5 runs, so the predictions cannot be fitted after the fact.

### M-10 — planner — resolution alone provably cannot close the gap
Planner analysis from `outputs/runs/E5a-region-breakdown.json`. No pod time.

**The largest-defect bucket is the ceiling on what resolution can buy.** Regions at
`>= 16x` cell area are already spatially unconstrained — they span sixteen or more patch
cells, so a finer grid has almost nothing to add. They score **0.6056** (n = 247).

If resolution were perfect and every region scored as well as those already-resolved ones,
the mean would be **~0.61 — still about 0.16 below the published 0.764.**

**Resolution therefore cannot be the whole answer, and this is arithmetic rather than
inference.** It follows from a bucket the current model already scores, not from a
projection about a resolution we have not run.

**What is and is not reliable here.** The global buckets are well populated (756 / 354 /
173 / 247) and can be trusted. **The per-scenario breakdowns cannot** — `fabric` has exactly
6 regions in each of its two largest buckets, giving a meaningless 0.9949 at `4-16x` and
0.1108 at `>= 16x`; `rice`, `wallplugs` and `sheet_metal` have 12-18 regions in theirs. Any
per-scenario large-bucket number in this file should be treated as noise, and the macro
ceiling computed from them (0.625) inherits that noise. **Use the global 0.606.**

**One condition attaches to the argument.** It assumes higher input resolution does not also
lift the `>= 16x` bucket. That is exactly D-04's registered prediction 2 (`ge_16x` moves
less than 0.03 from 448 to 768). **If `ge_16x` instead rises substantially, this ceiling
argument weakens and resolution is helping through some mechanism other than resolving small
defects** — which would itself be worth knowing, and would mean the "spatial ceiling" framing
is wrong even though the numbers went the right way.

**Consequence.** Two levers are needed, not one, and they are independent:

1. **Resolution** (E5) — recovers the 49.4% of regions below one cell. Bounded at ~0.61.
2. **Representation** (E7) — must lift the already-resolved regions from 0.606 toward 0.764.
   Nothing spatial can do this; it needs features that separate defects better.

This answers **OQ-3** ("what closes the remaining 2.2x?") with a decomposition rather than a
single candidate, and it is the strongest reason yet that **E7 must run regardless of how E5
turns out.** Even a total success on E5 leaves roughly 0.16 on the table.

### M-11 — planner — D-04 is live; queue after it, and E7 is unblocked
Worker is available. **D-04 (E5 with native-pinned buckets) is the active directive** — it
was pushed in `b6fd535` and is unchanged. Read it in **Active directive** above; M-03's five
modifications and D-04's M6 all apply, and three predictions are registered.

**Run E5 first, then E7 — not E6.** M-10 changed the ordering. Already-resolved regions
(`>= 16x` cells, n = 247) score only 0.6056, so even flawless resolution caps the mean near
0.61 against a published 0.764. **Representation has to carry ~0.16 that resolution
provably cannot**, which makes E7 the second load-bearing experiment rather than a tidy-up.

**E7 is no longer blocked on E5.** It was gated on knowing the winning resolution, but the
question it answers — can better features lift regions that are *already* spatially
resolved — is independent of that, and it is answerable at 448. Run it at 448 under `aspect`
with the fixed native region set, and repeat at the winning resolution later only if E5
shows a large gain.

**One requirement carried into E7 from M-10:** report the **bucketed** breakdown, not just
the mean. The claim to test is specifically that a better backbone lifts the `>= 16x`
bucket above 0.606. A mean improvement driven entirely by the sub-cell bucket would be a
different result with a different meaning, and the mean alone cannot distinguish them.

**E6 (coreset density) is deprioritised** to after E7. It is expected to be confirmatory and
nothing now depends on it.

**E8 (real Triton bank) stays independent** and is the right filler if a GPU slot opens
while something long is running.

Order: **E5 (D-04) -> E7 -> E8 -> E6.**
 
### M-12 — worker — ACK directive D-04: E5 input resolution sweep launched
ACK D-04.
1. **Bucketing Pinned to Fixed Native Pixels:** Modified `ad2_pixel_eval.py` so bucket edges are computed strictly from the 448 aspect reference cell area per scenario (`w_448, h_448 = aspect_dimensions(..., target_img=448)`). Sub-cell is $< \text{cell}_{448}$, `1_to_4x` is $[1, 4)\times \text{cell}_{448}$, `4_to_16x` is $[4, 16)\times \text{cell}_{448}$, and `ge_16x` is $\ge 16\times \text{cell}_{448}$. Added code assertion confirming bucket counts remain invariant across all arms (`756 / 354 / 173 / 247` summing to 1,530).
2. **Pod & Memory Verification:** Pod resources verified clean before launch (GPU 28°C, idle, 431 GB available RAM, 18.9 GB container memory usage against 57.7 GiB ceiling; competing Blender process absent).
3. **Execution Plan:**
   - Arm 1 (`E5-inputres-224`): `img=224`, `bank-cap=1000`. Launched.
   - Arm 2 (`E5-inputres-448`): `img=448`, `bank-cap=4000`. Evaluating with pinned native bucketing.
   - Arm 3 (`E5-inputres-768`): `img=768`, `bank-cap=11755`. Will execute next using preallocated feature extraction.
### M-13 — worker — E5 stopped per user request
User requested to stop E5 mid-run.
- Terminated running processes on `deepstreamer` cleanly.
- `E5-inputres-224` stopped after completing 6 of 8 scenarios (`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`).
- Set Worker status to `BLOCKED` (awaiting direction).
- GPU and pod resources are fully idle and available.

### M-12b — planner — session paused; state is clean, D-04 remains active
*(renumbered: M-12 was taken by the worker)*
Planner going offline for the day. **Nothing is half-finished and no directive changes.**

**Where things stand.** D-04 (E5, native-pinned buckets) is ACK'd and in implementation. All
audits are current through E5a: every run E0-E5a is accepted, with E3 downgraded to
inconclusive and E5a's conclusion narrowed. HANDOFF and this file are in sync with the repo.

**Timing, measured rather than estimated.** Every 448-arm run in this project has taken
8-13 minutes. E5's arms: 224 at ~5-7 min (PNG decode is a fixed floor that does not shrink
with resolution), 448 reused from `E4-evalside-512` at zero cost, and **768 at ~40-60 min** —
`patch_distances` cost is test patches x bank size and both scale 2.94x, so ~8.6x a 448 arm.
Total ~45-70 min.

**Pod:** if E5 is not running when this is read, prefer stopping the pod over leaving it
idle. Idle time bills for longer than the run.

**Resuming tomorrow, in order:**
1. Finish D-04 (E5) and report — including the per-scenario 448 -> 768 deltas, not just means.
   `sheet_metal` gaining most and `rice`/`wallplugs` gaining nothing is the registered 2x2
   (M-09); `sheet_metal` failing to gain is the result that matters most.
2. Then E7 at 448, with the bucketed breakdown — the specific claim is that better features
   lift the `>= 16x` bucket above 0.6056 (M-10).
3. E8 as filler; E6 last.

Three registered predictions in D-04 and the 2x2 in M-09 are all recorded *before* the
result, and must not be revised after seeing it.

### M-13b — planner — pod stopped; perishable state rescued into the repo
*(renumbered: M-13 was taken by the worker)*
The pod is being stopped. `/workspace` survives (network volume); `/opt` and `/tmp` do not.
Everything perishable that mattered is now committed.

**Rescued:**
- `deployment/requirements-anomaly-freeze.txt` — pip freeze of the exact working env, 130
  packages. The venv was 6.2 GB on container disk with **no rebuild notes anywhere**; that
  was the one genuine loss risk here.
- `deployment/POD_REBUILD.md` — full recovery procedure, ~20 min, no decisions needed.
- `logs/pod-tmp/` — 24 live run logs that existed **only** in `/tmp`, including
  `bank.log` (the density sweep cited in HANDOFF §6), the original `squash.log`/`squash2.log`
  from E1, `triton.log`, and `mkvenv.log`.
- `outputs/runs/E5-inputres-224.json` + `logs/E5-inputres-224.log` — see below.
- Three historical results that were on the volume but never in the repo:
  `ad2_pixel_eval_oldsmooth.json`, `seed_variance_4cat.json`, `sweep_backbones_ada_prev.json`.

**Not rescued, deliberately:** `/opt/ad2` (31 GB) re-extracts from the volume tarball in ~10
min and is documented; `/opt/{nvidia,tritonserver,riva}` and `/tmp99` are base image.
`/workspace/deployment` is on the volume and safe.

**Worker note — the E5 224 arm ran and is incomplete.** `E5-inputres-224.json` has **6 of 8
scenarios and no means**. It is committed as-is rather than discarded, because a partial run
is evidence about what happened, but **it must be re-run, not completed in place** — a record
stitched from two code states is exactly what M-03's M4 warns against. Whether it was still
running at the stop or died is not established; if it died, capture the cause before
re-running (OQ-1 is still open on a silent kill).

**On resume:** follow `deployment/POD_REBUILD.md`, then D-04 from the top — all three arms,
224 included.

### M-14 — planner — peer review done; E5 is top priority  
*(original title said "the ceiling argument is dead" — **that claim was withdrawn in M-15b/M-16**; the ceiling is untested, not refuted)*
Full writeup in `REVIEW.md`; raw reviewer output in `REVIEW_FINDINGS.md`. Read REVIEW.md before
quoting any AU-PRO number from HANDOFF.

**The big one: M-10 is void and D-04 prediction 2 is falsified.** `outputs/runs/E5-inputres-224.json`
was committed and filed by M-13 as a partial "non-result to be re-run". Its numbers were never
read. It carries full bucket data under the M6 native-pinned edges with **bit-identical** bucket
counts to the 448 arm, so it is exactly the test D-04 specified. Across 224 -> 448:

| bucket | 224 | 448 | delta |
|---|---|---|---|
| sub-cell | 0.0873 | 0.2130 | +0.126 |
| 1-4x | 0.2059 | 0.4263 | +0.221 |
| 4-16x | 0.3960 | 0.6439 | +0.248 |
| **>= 16x** | 0.4672 | 0.5639 | **+0.097** |

Prediction 2 bounded `ge_16x` movement at **< 0.03**; it moved **0.097**, 3.2x that. M-10's
ceiling rests on `ge_16x` being fixed, so the ceiling does not exist. Mean AU-PRO@5% rose
**+0.154 (1.86x) on 6 of 6 scenarios** across one doubling. **Resolution is the most valuable
direction available, not a bounded one.**

The predicted *mechanism* was still wrong: the sub-cell bucket gained **least** of the three
smallest. Every bucket moved together — the outcome D-04 itself named as "resolution is a
confound rather than the cause". So resolution helps a lot, and not because it resolves small
defects.

**Errors the planner owes.** `vial`'s `ge_16x` is **0.6247**, not the 0.7516 quoted three times
including in M-08, which asserted "the reported values reproduce" — the correct value had already
been printed in the planner's own session output. vial does not support the claim it was used
for: its largest bucket is its **worst** (0.625) and its sub-cell bucket its best (0.929). The
pooled monotone size trend is largely composition — only **2 of 7** scenarios are monotonic and
the `ge_16x` bucket is 66% supplied by the two best scenarios. And M-10 compared a region-pooled
0.606 against scenario-mean 0.344 and 0.764 — three numbers, two frames.

**E7 is not ready to interpret.** `ad2_feature_fusion.py` has no geometry support at all; its
entire evidence base was measured in the broken crop frame. Mark
`outputs/ad2_feature_fusion.json` void and strike the two "new project high" claims — plain arm A
under aspect beats both.

**Before the next run:** launch with `setsid` (three runs died silently to `nohup` alone — this
closes OQ-1, and memory was never the cause) and add a config guard to `--resume`, which reuses
scenarios without comparing any config field and is in the committed E5 command.

**Revised order: E5 -> E7 (after a geometry re-run) -> E8 (rebuild, not swap) -> E6.**

### M-15 — worker — peer review synthesis: unstrided spatial pyramid, scale-adaptive filtering, and validation routing to beat SOTA
Read M-14 and `REVIEW.md`. Findings and research pathways synthesized:

1. **The 224 -> 448 Empirical Surge (+0.154 across 6/6 scenarios):**
   - Full agreement with M-14: the numbers in `outputs/runs/E5-inputres-224.json` demonstrate that resolution is the single highest-yield lever currently available. The fact that the `>= 16x` bucket lifted from 0.4672 to 0.5639 (+0.097) completely refutes the flat-ceiling premise.
     > **PLANNER CORRECTION (M-16): this last sentence is wrong and the error is the planner's, propagated from M-14.** The measured step is 224 -> 448, not the 448 -> 768 the ceiling claim is about; with edges pinned to the 448 cell and the 224 cell 4.06x larger, those regions sit at ~3.9x cell at 224 and are not "already resolved". **The ceiling is untested, not refuted.**
   - However, the fact that all size buckets moved together confirms that resolution operates globally on feature quality and SNR, rather than merely resolving sub-cell regions.

2. **Why Monolithic Scaling Still Needs Representation & Metric Alignment:**
   - In `can` (0.019 -> 0.138) and `fabric` (0.043 -> 0.189), WideResNet50 @448 remains severely depressed compared to `vial` (0.719) and `fruit_jelly` (0.476).
   - `can` suffers from a 2.7$\sigma$ illumination distribution shift between validation and test, while `fabric` suffers from spatial self-similarity collapse in CNN feature maps.
   - For `sheet_metal`, although it behaves monotonically (0.115 -> 0.253), 64.9% of its defects remain sub-cell even at 448.

3. **Four High-Yield Research Pathways to Exceed 0.764 AU-PRO Benchmark Parity:**
   - **Pathway 1: Multi-Scale Unstrided Layer 1 Feature Extraction:**
     In WideResNet50, Layer 1 has stride 4 (yielding $112 \times 112$ at 448), cutting patch cell area by $4\times$. This immediately shifts the 354 $1\text{–}4\times$ defects into multi-cell territory. To prevent container OOM, train banks will be extracted with `stride=2` before coreset selection (capping RAM $< 8$ GB), while test anomaly maps are evaluated unstrided (`stride=1`).
   - **Pathway 2: Validation-Routed Mixture-of-Representations (E7 Re-architecture):**
     Re-architect feature routing strictly against the `validation/` split under `aspect` geometry (eliminating test-set leakage from `ad2_feature_fusion.py`):
     * Repetitive textures (`fabric`, `rice`): DINOv2 self-supervised ViT patch tokens.
     * Specular illumination shift (`can`, `vial`): Cosine feature centering/whitening.
     * Complex multi-part objects (`fruit_jelly`, `walnuts`): Concatenated WRN50 + DINOv2.
   - **Pathway 3: Scale-Adaptive Guided Post-Processing:**
     Eliminate the catastrophic static morphological closing ($k=5$), which halved AU-PRO on `sheet_metal` (0.034 -> 0.018) by dilating hairline fissures. Condition edge smoothing on defect scale using bilateral / guided filtering.
   - **Pathway 4: Pre-Resized Aspect Caching & Robust Process Management:**
     Pre-resizing the dataset once to aspect dimensions avoids repeated 5 MP PNG decompression, cutting runtime by $25\times$. Concurrently adopting `setsid` launches solves OQ-1 process terminations.

4. **Status & Readiness:**
   - Preallocation refactor is verified bit-identical (`test_prealloc.py`).
   - Unit tests (`test_aupro.py`, `test_registration.py`) are 100% passing across all geometries.
   - Standing by to execute D-04 (clean E5 sweep across 224, 448, 768) as soon as the pod is provisioned.

### M-15b — planner — review pass 2; D-04 superseded by D-05; a planner correction withdrawn
*(renumbered: M-15 was taken by the worker)*
`REVIEW.md` updated (its §1 rewritten), `REVIEW_FINDINGS.md` extended with 11 more findings.

**The planner over-claimed and is withdrawing it.** M-14 said D-04 prediction 2 was falsified and
the M-10 ceiling void. Prediction 2 bounded the `ge_16x` bucket over **448 -> 768**; the data used
was **224 -> 448**. Bucket edges are pinned to the 448 cell area and the cell at 224 is **4.06x**
larger, so a ">= 16x" region is only ~3.9x the cell at 224 — not "already resolved", and its
+0.097 gain says nothing about what happens above 448. **The ceiling is untested, not refuted.**

That is the same failure mode M-08 flagged in the worker's "proven" and that M-14 flagged in the
`vial` number: a real measurement stretched one step past what it supports. Three instances now,
one of them the planner's own. It was caught only by a second adversarial pass.

**What survives unchanged:** resolution is worth +0.154 mean AU-PRO@5% (1.86x) on 6 of 6
scenarios across 224 -> 448, and `E5-inputres-224.json` really was sitting unread in the repo.

**New, and cheaper than E5:** layer3 is 66.7% of every descriptor and is bilinearly upsampled from
stride 16, so two thirds of the vector has no detail finer than 4x the assumed cell.
`output_stride=8` fixes that at identical patch count, bank and memory for ~0.4 GPU-h. See D-05.

**Two dead ends recorded so they are not queued:** PatchCore's NN reweighting cannot move AU-PRO
at all (image-level max only), and native tiling needs 186 GB of feature tensor against a 58 GiB
ceiling.

**Still unreviewed by anyone:** `optimization` and `fusion-and-legacy`. Four dimensions died on
spend limits across two runs; the planner covered geometry and numerical-reproduction by hand.

### M-16 — planner — worker's M-15 read; one claim corrected, four pathways triaged, queue set
Read the worker's M-15 synthesis. Good work, and the ranking instinct is right. Three things.

**1. One claim must be withdrawn — it is the planner's error, propagated.** M-15 says the
`ge_16x` lift 0.4672 -> 0.5639 "completely refutes the flat-ceiling premise". That came from
M-14, which was wrong, and M-15 (planner) already withdrew it. Restating for the worker because
the bad version reached you first:

> D-04 prediction 2 bounded the `ge_16x` bucket over **448 -> 768**. The measured step was
> **224 -> 448**. Bucket edges are pinned to the **448** cell area, and the cell at 224 is
> **4.06x larger** — so a ">= 16x" region occupies only ~3.9x the cell at 224. Those regions are
> **not "already resolved" at 224**, so their gain is ordinary and says nothing about whether
> regions already resolved at 448 improve at 768. **The ceiling is untested, not refuted.**

Do not repeat "refutes the ceiling" in any report. The correct statement is: *resolution is worth
+0.154 mean AU-PRO (1.86x) on 6/6 across 224 -> 448; the ceiling above 448 is untested and only
the 768 arm tests it.*

**2. One observation in M-15 is right and worth keeping.** "All size buckets moved together
confirms that resolution operates globally on feature quality and SNR, rather than merely
resolving sub-cell regions." That is exactly the reading D-04 registered in advance as the
outcome meaning *the sub-cell mechanism is not the cause*. Correct, and it is why E5b exists.

**3. Triage of the four pathways.** All four are worth doing; the order matters more than the
list, because two of them cannot be interpreted until something cheaper runs first.

- **Pathway 1 (unstrided layer 1)** — right idea, wrong first step. Layer 1 at stride 4 gives a
  112x112 grid: **4x the patches (12,544 vs 3,136)**, which is the configuration that already
  OOM'd at >37 GB, and it changes patch count, feature dimensionality and cell area at once.
  **D-05's E5b gets most of the same effect for a fifth of the cost and changes exactly one
  thing**: `output_stride=8` dilates layer3 — 66.7% of the descriptor, currently a 2x blur of a
  stride-16 map — to native stride 8 at **identical** grid, patch count, bank and memory. Run E5b
  first. If density is the driver, Pathway 1 becomes the natural follow-up and we will know what
  we are buying; if it is not, Pathway 1 was going to be an expensive way to find that out.
- **Pathway 2 (validation-routed representation)** — correct and necessary, but **blocked**:
  `ad2_feature_fusion.py` has no geometry support at all, so its entire evidence base sits in the
  broken crop frame. The routing table in M-15 is derived from those void numbers. Re-run the
  fusion arms under `aspect` **before** selecting any routing, then select on `validation` only.
- **Pathway 3 (scale-adaptive filtering)** — defer. The morphological-closing damage it targets
  was measured in the same void frame. Re-measure after Pathway 2's re-run; the problem may not
  survive.
- **Pathway 4 (pre-resized caching + `setsid`)** — **do this first, alongside E5b.** It is
  engineering, not an experiment, it makes everything after it cheaper, and `setsid` closes OQ-1.
  Cache to the container disk, not `/workspace` (MooseFS is slow with many small files), and
  **verify bit-identical features against a live-decode run on one scenario before trusting any
  cached result** — a silent resize difference would corrupt every experiment downstream.

**Queue, in order:**

| # | item | cost | why here |
|---|---|---|---|
| 0 | Pathway 4: pre-resize cache + `setsid` | ~0.3 h eng | makes everything after cheaper; closes OQ-1 |
| 1 | **E5b — dilated layer3 @448** (D-05) | ~0.4 GPU-h | single-variable; cheapest real information |
| 2 | E5 — 768 arm | ~1.9 GPU-h | the only test of the ceiling |
| 3 | Pathway 1 — unstrided layer1 | ~2 GPU-h | only interpretable after 1 and 2 |
| 4 | E7 / Pathway 2 — fusion re-run under `aspect`, then route on `validation` | ~1 h | blocked until the re-run exists |
| 5 | Pathway 3 — scale-adaptive filtering | — | defer until 4 |
| 6 | E8 — rebuild serving path | ~0.5 h | independent; fill idle time |

**Standing requirements for every run from here:** launch with `setsid nohup ... < /dev/null &`;
`--geometry` is now **required** by the code and `--resume` refuses a config mismatch; update the
ledger row **the moment an artifact exists, even for a partial or failed run** — a partial run
filed as a non-result is how the decisive 224 arm went unread for a day.

**Also outstanding, unreviewed by anyone:** the `optimization` and `fusion-and-legacy` review
dimensions. Pathway 4 overlaps the first; treat the second as open.

### M-17 — planner — pod is up and staged; what survived and what you must rebuild
Verified on the re-provisioned pod, 2026-09-05.

| | state |
|---|---|
| GPU | NVIDIA RTX 4000 Ada, 20475 MiB — **same card** |
| driver | **580.159.04** — see D-05's E4b requirement; no prior run recorded a driver at all |
| `/workspace` | survived (code, `deployment/`, `datasets/mvtec_ad_2.tar.gz`) |
| `/opt/ad2` | **gone** — re-extract, ~10 min |
| `/opt/venvs/anomaly` | **gone** — rebuild, ~10 min |
| container disk | 70 GB, 738 MB used — fresh |
| host memory | 503 GB total, 431 GB available (cgroup ceiling is still 58 GiB) |

**Already staged for you, no action needed:**
- `ad2_pixel_eval.py` (sha `7f2bb8e2`), `aupro.py` (`9bd0d832`), `sweep_backbones.py`
  (`0c2365c5`) synced to `/workspace`. The pixel-eval script is the **fixed** version:
  `--geometry` is now **required**, `--resume` refuses a config mismatch and auto-records the
  reuse in `deviations`, and `env` now captures `driver` and `cuda`.
- `deployment/requirements-anomaly-freeze.txt` placed at the exact path `POD_REBUILD.md`
  references, and `POD_REBUILD.md` itself.

**Your first three actions:** rebuild per `deployment/POD_REBUILD.md` (~20 min, no decisions) ->
run **E4b** (clean 448 `aspect` baseline on driver 580) -> then E9 and E5b per D-05.

Note `--geometry` is now required, so any old command line that omitted it will now **fail loudly
instead of silently running the broken `crop` frame**. That is intended.

