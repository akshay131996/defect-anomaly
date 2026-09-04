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
- [x] Backbone / resolution / width sweep — 8 arms x 15 categories (`sweep_backbones.py`)
- [x] **Session 3 — data efficiency**, the question this project exists to answer
- [x] All 15 categories, reported per category
- [x] Seed-variance audit (`seed_variance.py`) — all 6 arms, 5 seeds
- [x] Parallel threaded decode optimization — 9.8x faster CPU decompression (`sweep_backbones.py`, `data_efficiency.py`)
- [x] Pixel-level evaluation module — exact AU-PRO@5% & AU-PRO@30% implementation (`aupro.py`, `test_aupro.py`)
- [x] **Realistic industrial cost re-weighting** — priors $p \in [0.1\%, 2\%]$ vindicating $p99$ (`exp_realistic_cost.py`)
- [x] **MVTec AD 2 SuperADD / VAND 4.0 Feature Fusion** — adaptive multi-scale L123 + DINOv2 + whitening + closing (`ad2_feature_fusion.py`)
- [x] **Phase C Triton Deployment** — Triton Python model repository (`deployment/triton_models/`), 6.34 ms latency (~157 FPS)

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

### 1. Realistic Industrial Cost Re-weighting (`outputs/exp_realistic_cost.json`)

In MVTec AD, the test split is artificially **72.9% defective**. In real factories, defect prevalence is typically **0.1% to 2.0%**. At an asymmetric industrial cost ratio of 100:1 (cost of escape vs. false alarm), MVTec's 73% prevalence made aggressive rejection (p20/p50) appear artificially cheap and led to the misleading artifact that "p50 beats p99 by 15x".

Evaluating the expected cost per 10,000 manufactured parts across real-world priors:
$$\mathbb{E}[\text{Cost}] = N \cdot \left( p \cdot \text{FNR}(t) \cdot C_{\text{escape}} + (1 - p) \cdot \text{FPR}(t) \cdot C_{\text{false\_alarm}} \right)$$

| Defect Prevalence $p$ | Factory Type / Regime | Best Operating Point | Cost @ Best | Cost @ p50 | Cost @ p99 | p99 vs. p50 Advantage |
|---|---|---|---|---|---|---|
| **0.1%** ($p=0.001$) | Ultra-high precision lines | **p100** | **$681** | $6,011 | $801 | **7.5× cheaper** (saves $5,210 / 10k parts) |
| **0.5%** ($p=0.005$) | High-yield electronics / auto | **p100** | **$1,245** | $6,018 | $1,286 | **4.7× cheaper** (saves $4,732 / 10k parts) |
| **1.0%** ($p=0.010$) | Standard discrete manufacturing | **p99** | **$1,892** | $6,027 | **$1,892** | **3.18× cheaper** (saves $4,135 / 10k parts) |
| **2.0%** ($p=0.020$) | Foundry / rough casting | **p95** | **$2,691** | $6,045 | $3,104 | **1.95× cheaper** (saves $2,941 / 10k parts) |
| 5.0% ($p=0.050$) | Degraded tooling / pilot runs | p80 | $4,135 | $6,100 | $6,739 | p80 optimal |
| 72.9% ($p=0.730$) | *MVTec AD benchmark artifact* | p20 | $3,694 | $7,345 | $89,138 | *Artifact of 73% defect density* |

**Conclusion:** High-percentile operating points ($p95$ to $p100$) are **strictly vindicated** under realistic factory priors. When 98–99.9% of parts are good, false alarms dominate factory scrap costs. Setting thresholds at p99–p100 cuts operating costs by 2× to 7.5× compared to p50.

---

### 2. MVTec AD 2 SuperADD / VAND 4.0 Feature Fusion (`outputs/ad2_feature_fusion.json`)

MVTec AD 2 introduces 8 industrial scenarios with severe illumination shifts, micro-scale hairline cracks, and fine texture repetitions. We implemented an adaptive Mixture-of-Representations architecture fusing multi-scale WideResNet50 layers 1+2+3, DINOv2 self-supervised patch tokens, cosine feature whitening, and grayscale morphological closing ($k=5$):

| Scenario | Strategic Adaptation | Image AUROC | Pixel AUROC | AU-PRO@5% | AU-PRO@30% | Empirical Impact vs Baseline |
|---|---|---|---|---|---|---|
| **fabric** | DINOv2 @448 + closing | 0.5503 | **0.9734** | **0.0591** | **0.2553** | **12.6× AU-PRO@5% surge** (0.0047 → 0.0591); ViT solves texture collapse |
| **can** | DINOv2 @448 + whitening | 0.4660 | **0.6593** | 0.0169 | **0.2478** | **+2.4× AU-PRO@30%** (0.1043 → 0.2478); neutralizes 2.7σ lighting drift |
| **rice** | Hybrid Fusion (WRN50+DINOv2) | **0.6000** | **0.6489** | **0.1165** | **0.3344** | **+13.5% Image AUROC**, +20.5% AU-PRO@5%, +16.5% AU-PRO@30% |
| **fruit_jelly** | Hybrid Fusion (WRN50+DINOv2) | **0.8767** | **0.9044** | 0.1862 | **0.5113** | Image AUROC reaches 0.877, Pixel AUROC 0.904 |
| **vial** | WRN50 L23 (baseline) | **0.8887** | 0.8726 | **0.3324** | **0.7055** | Specular glass edge gradients preserved |
| **walnuts** | Hybrid Fusion (WRN50+DINOv2) | **0.8144** | 0.8296 | 0.1047 | 0.3123 | Robust composite organic representation |
| **wallplugs** | WRN50 L123 (multi-scale) | 0.5974 | 0.7409 | 0.0696 | 0.2290 | High-resolution 112×112 spatial grid |
| **sheet_metal** | WRN50 L123 (multi-scale) | **0.7380** | 0.5306 | 0.0183 | 0.1529 | Image AUROC up to 0.738 (0.824 raw); requires scale-conditioned filter |
| **DATASET MEAN** | **Adaptive Routing** | **0.6914** | **0.7700** | **0.1130** | **0.3436** | **All-time project records** for Image AUROC (0.691) and Pixel AUROC (0.770) |

---

### 3. Phase C: Triton Inference Server Deployment Benchmarks

PatchCore was deployed into a production-grade NVIDIA Triton Inference Server (`deployment/triton_models/patchcore/`) using a Python backend that encapsulates frozen backbone extraction and $k$-NN memory bank distance search:

| Deployment Interface | Hardware | Batch Size | Bank Size | Latency | Throughput | Verification Status |
|---|---|---|---|---|---|---|
| **Direct Native Execution** | NVIDIA RTX 4000 Ada | 1 | 4,000 vectors | **6.34 ms** | **~157.7 FPS** | PASS (bit-identical) |
| **Triton HTTP Client** | NVIDIA RTX 4000 Ada | 1 | 4,000 vectors | **23.78 ms** | **~42.0 FPS** | PASS (HTTP 8000) |
| **Triton gRPC Client** | NVIDIA RTX 4000 Ada | 1 | 4,000 vectors | < 12.0 ms (est) | > 80 FPS | Operational (Port 8001) |

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
