# Industrial Anomaly Detection

Finding defective parts when you have **no examples of defects** to train on.

A production line makes overwhelmingly good parts — that is the point of a good line. So the
training set is defect-free by construction, the model learns what *normal* looks like, and
anything far enough from normal gets flagged. No defect labels anywhere in the pipeline.

**📖 [Read the write-up for non-specialists](https://claude.ai/code/artifact/1c912ba0-6954-4021-963a-c1992f54e9a3)** —
the same material pitched at an engineering manager, with an interactive threshold figure.

---

## Status

Measurements are **not in yet**. The notebooks are written and validated; nothing has been
run end to end. No performance figures are claimed anywhere in this repo until they are.

- [x] Session 1 — baseline written: pooled k-NN on frozen features, honest threshold rule
- [x] Session 2 — written: PatchCore, plus a single-variable backbone ablation
- [ ] **Session 1 executed** — the first real numbers
- [ ] Session 2 executed
- [ ] Session 3 — data efficiency: how many good images before performance plateaus
- [ ] All 15 categories, reported per category
- [ ] Pixel-level evaluation (the dataset ships masks; localisation is currently qualitative)
- [ ] Demo + write-up

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
| `brief.html` | The write-up for a non-specialist decision-maker |
| `outputs/` | Result JSON — each session reads the previous one's file |

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
