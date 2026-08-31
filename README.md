# Industrial Anomaly Detection

Finding defective parts when you have **no examples of defects** to train on.

A production line makes overwhelmingly good parts — that is the point of a good line. So the
training set is defect-free by construction, the model learns what *normal* looks like, and
anything far enough from normal gets flagged. No defect labels anywhere in the pipeline.

**📖 [Read the write-up for non-specialists](https://claude.ai/code/artifact/1c912ba0-6954-4021-963a-c1992f54e9a3)** —
the same material pitched at an engineering manager, with an interactive threshold figure.

---

## Status

Measured on an RTX 4000 Ada, all 15 MVTec AD categories. Every number below is from a run
in `outputs/`, not from the literature.

- [x] Session 1 — pooled k-NN baseline, executed
- [x] Session 2 — PatchCore plus the backbone ablation, executed
- [x] Backbone / resolution / width sweep — 6 arms x 15 categories (`sweep_backbones.py`)
- [x] **Session 3 — data efficiency**, the question this project exists to answer
- [x] All 15 categories, reported per category
- [x] Seed-variance audit (`seed_variance.py`) — all 6 arms, 5 seeds, and it found a real problem, below
- [ ] Re-seed the 9 low-cost categories (they carry <25% of escapes; unlikely to move totals)
- [ ] Pixel-level evaluation (the dataset ships masks; localisation is currently qualitative)
- [ ] MVTec AD 2 — the successor benchmark, where SOTA is still below 60% AU-PRO
- [ ] Demo + write-up

> **The sweep's cost totals were single-seed, and that mattered.** Re-running the six
> categories that carry the cost with five coreset seeds each found seed 0 sitting at an
> extreme of its own range in most cells. Arm A's `capsule` was recorded at 56 escapes
> against a 5-seed mean of 31; arm B1's `screw` at 74 against a mean of 92. Every one of
> A's committed values was pessimistic and every one of B1's was optimistic.
>
> | arm | committed | corrected | mean AUROC |
> |---|---|---|---|
> | B1 DINOv2 @392 | 13,223 | **15,182** | 0.9828 |
> | A WideResNet50-2 @224 | 19,527 | **15,748** | 0.9785 |
> | D WideResNet50-2 @320 | 18,721 | 17,762 | 0.9826 |
> | E ResNet50 @224 | 17,032 | 17,870 | 0.9717 |
> | B0 DINOv2 @224 | 24,427 | 26,043 | 0.9560 |
> | C ResNet18 @224 | 26,739 | 26,774 | 0.9523 |
>
> **B1's lead over A is 3.6%, not the 32% first reported.** Worse, the single-seed
> ordering was wrong: it read B1 → E → D → A, and the truth is B1 → A → D → E. Arm A was
> the most misreported in the sweep, sitting fourth when it is second.
>
> The defensible conclusion is not "B1 is the best arm" but **"A and B1 are equivalent;
> choose per category"** — which is what the per-class table said all along. The top two
> *are* clearly separated from the rest (15.2k/15.7k against 17.8k and up), so that split
> is real.
>
> Nine categories remain single-seed, but they cannot move these totals much: the six
> re-seeded ones carry 78% of arm A's escapes and 95% of B1's.
>
> The qualitative findings are unaffected — the resolution knee, quality-over-width, and
> category-dependent backbone choice are gaps far larger than this noise.

### Headline

**A line needs a median of 10 defect-free photographs** to reach 99% of its full-data
AUROC. The range across categories is 2 to 256 — four categories exceed 0.999 from *two*
images, while `screw` never converges at all. At a few seconds per photo that is about a
minute of production for the median part, and under an hour for the worst well-behaved one.

The spread is the finding. A single number would have been the wrong answer.

| knee (images) | categories |
|---|---|
| 2 | bottle, carpet, leather, tile |
| 5 | grid, hazelnut, toothbrush |
| 10 | wood |
| 20 | metal_nut, transistor |
| 40 | pill |
| 80 | cable, capsule, zipper |
| 256 (never converges) | screw |

---

## The question this project is actually answering

Not "can I beat state of the art" — MVTec AD is saturated at ~99% published image AUROC, and
the remaining headroom is noise.

The question is **how many defect-free photographs a line needs before this works.** A plant
commissioning a new product wants to know whether that is two hours of photography or two
weeks. The benchmark literature does not report it, and it decides whether the approach is
viable for short production runs.

Everything else here — the baseline, the operating point, the backbone ablation — exists to
make that answer trustworthy.

---

## What's here

| File | What it is |
|---|---|
| `build_session1.py` → `run_session1.ipynb` | Baseline: frozen features, global pooling, k-NN. Built to be beaten |
| `build_session2.py` → `run_session2.ipynb` | PatchCore, then the same code with a DINOv2 backbone |
| `sweep_backbones.py` | 6 arms x 15 categories. Every pair differs in exactly one variable |
| `data_efficiency.py` | Session 3. Bank size varied, calibration split held fixed |
| `brief.html` | The write-up for a non-specialist decision-maker |
| `outputs/` | Result JSON — each session reads the previous one's file |

The two sweeps are plain scripts rather than notebooks on purpose: they are experiment
runners producing JSON, not explanations. The narrative belongs in the notebooks.

---

## What the sweep found

Six arms, each differing from the reference in exactly one respect.

Costs below are the **seed-corrected** ones, not the single-seed values the sweep first
produced — see the status block above for why that distinction matters.

| arm | backbone | input | grid | dims | mean AUROC | cost (corrected) |
|---|---|---|---|---|---|---|
| **B1** | **DINOv2 ViT-B/14** | **392** | **28x28** | **768** | **0.9828** | **15,182** |
| **A** | **WideResNet50-2** | **224** | **28x28** | **1536** | **0.9785** | **15,748** |
| D | WideResNet50-2 | 320 | 40x40 | 1536 | 0.9826 | 17,762 |
| E | ResNet50 | 224 | 28x28 | 1536 | 0.9717 | 17,870 |
| B0 | DINOv2 ViT-B/14 | 224 | 16x16 | 768 | 0.9560 | 26,043 |
| C | ResNet18 | 224 | 28x28 | 384 | 0.9523 | 26,774 |

**Resolution has a knee near 28x28.** Below it, resolution is the dominant variable:
16x16 → 28x28 on the same DINOv2 backbone buys +0.027 AUROC and cuts cost 42%. Above it,
the return goes negative — 28x28 → 40x40 on the same WideResNet50-2 buys +0.004 AUROC but
costs 13% *more* at the operating point, for 33% more runtime. Past the knee you are
paying for patches that no longer separate anything.

**Descriptor width is not the driver.** B1 beats A with half the dimensions; C loses badly
with a quarter. The relationship is not monotonic, so what matters is descriptor *quality*.

**The best backbone is category-dependent.** DINOv2 wins on textures (grid 1.000 vs 0.951,
carpet 0.9996, tile 1.000) and loses on every small-part object, worst on `screw` — 0.8795
against WideResNet50-2's 0.9549. Reporting only the mean would have hidden this entirely,
which is why this repo reports per class.

**Session 2's original conclusion was wrong, and the sweep is why.** It compared a CNN at
28x28 against a ViT at 16x16 and attributed the gap to the backbone. Two variables had
moved. At matched resolution the result reverses.

**Resolution also buys reproducibility.** Arm D looked like a poor trade on accuracy alone
— +0.004 AUROC for 33% more runtime. But across the six re-seeded categories it has the
lowest mean AUROC spread of any arm (0.0105, against A's 0.0121 and C's 0.0234). With
1,600 patches instead of 784, greedy k-center is far less sensitive to where it starts.
For a plant that has to certify an inspection system, halving run-to-run variance in the
decision may be worth more than the accuracy that variance was hiding.

No arm wins on all three axes. D ties B1 on mean AUROC (0.9826 vs 0.9828) and is the most
reproducible, but costs 17% more at the operating point. Ranking quality, decision quality
and stability are separate properties and this project has now caught them disagreeing
three times.

`bottle` shows exactly 0.0000 spread on every arm and every seed, so this is confined to
the hard categories rather than being noise everywhere.

**Why builders rather than notebooks in git.** The `.ipynb` files are generated from the
`build_*.py` scripts, which validate every code cell with `ast.parse` before writing. Editing
notebook JSON in place corrupted a multi-line string in a sibling project and cost a
debugging cycle; generating from plain Python literals removes that failure mode entirely.
Run the builder, get the notebook.

---

## Method

Three runs, changing **one thing at a time**, because two simultaneous changes produce a
result nobody can attribute.

| run | method | backbone | isolates |
|---|---|---|---|
| baseline | pooled k-NN | WideResNet50-2 | — |
| A | PatchCore | WideResNet50-2 | what spatial locality is worth |
| B | PatchCore | DINOv2 ViT | what feature quality is worth |

Run A keeps the published PatchCore recipe so its numbers are checkable against the
literature. Run B changes only the feature extractor.

### The baseline is deliberately weak

Global average pooling collapses a 7×7 feature map into one vector, so a three-pixel scratch
contributes about 1/49th of one channel average. It should fail on small defects and it
should do *fine* on textures, where the anomaly is distributed across the whole image rather
than localised.

Being weak in a **predictable** way is the point. When PatchCore beats it, the gain is
attributable to spatial locality rather than to "the sophisticated method was better somehow."

### What PatchCore adds

1. **Keep the grid.** Tap `layer2` + `layer3` feature maps instead of the pooled output —
   784 patch descriptors per image rather than one. The deepest layer is skipped
   deliberately: most ImageNet-committed, least spatially precise.
2. **Neighbourhood aggregation.** 3×3 average at stride 1 — widens the receptive field
   without shrinking the grid.
3. **Memory bank + coreset.** Store every training patch, then keep ~1% via greedy k-centre
   selection. It walks the *outside* of the distribution, where the decision boundary lives.
   Random sampling would keep the dense middle and lose the rare-but-normal patches that
   prevent false alarms.

Image score is the **maximum** over patches — one bad region condemns a part, which is the
correct inspection semantics and the exact opposite of averaging.

---

## Design decisions worth not re-litigating

**The threshold is set from defect-free data only.** At commissioning a line has good parts
and nothing else. Tuning a threshold against a test set containing defects reports numbers
the deployed system cannot reproduce. Every run here uses the 99th percentile of *training*
scores.

**Per category, never the mean alone.** MVTec categories differ enormously in difficulty.
`screw` has few-pixel defects on arbitrarily rotated parts; `bottle` defects are large. A
15-category average conceals exactly the failure that matters.

**AUROC is reported but decides nothing.** It measures ranking, averaged over all possible
thresholds. Two systems with identical AUROC can differ substantially in escapes at the
threshold actually deployed. The comparison table that matters is escapes / false alarms /
cost at a fixed operating point.

**Costs are asymmetric.** An escape ships a defective part; a false alarm costs an inspector
a few minutes. The runs assume 100:1 and report the cost curve, because the ratio is a
business input rather than a modelling one.

**Session 2 hard-fails without session 1's output file.** It reads the category list,
threshold rule and cost ratio from `outputs/session1_baseline.json` rather than restating
them — config drift between sessions would silently invalidate every comparison.

---

## Results

Pending — see Status above. Tables land here as each session executes.

---

## Reproducing

Needs a CUDA GPU. Developed against an RTX 4000 Ada; the first run downloads ~5.3 GB.

```bash
python build_session1.py && python build_session2.py
jupyter lab run_session1.ipynb      # then run_session2.ipynb
```

Each notebook installs its own dependencies in the first cell. If the Hugging Face Hub
rate-limits (HTTP 429), authenticate first — anonymous requests get a much tighter limit:

```bash
huggingface-cli login
```

No credentials belong in the notebooks; they read `huggingface_hub.get_token()`.

---

## Data

[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) — 15 product
categories, ~5,350 images, training split defect-free by construction.

The official download endpoint 404s intermittently as of early 2026, which also breaks
`anomalib`'s auto-download. This uses the `TheoM55/mvtec_all_objects_split` Hub mirror,
whose schema was verified against the datasets-server API before being written into the
notebook — it preserves `object`, `split`, `defect`, `label` and the ground-truth masks. A
second mirror, `Voxel51/mvtec-ad`, flattens everything to `{image}` and is unusable here.

**Licence: CC BY-NC-SA 4.0 — non-commercial research use only.** Fine for a portfolio and a
write-up. Not licensed for commercial deployment.
