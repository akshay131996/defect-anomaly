# MVTec AD 2 High-Performance Implementation Specification (VAND 4.0 / SuperADD Alignment)

**Target Audience:** Autonomous Coding Agent / Systems Engineer  
**Date:** September 2026  
**Repository:** `github.com/akshay131996/defect-anomaly`  
**Primary Objective:** Lift local `mean AU-PRO@5%` on MVTec AD 2 from **0.131 (13.1%)** to **> 0.60 (60%+)** by adopting the architectural discoveries from the IJCV 2026 benchmark paper and the CVPR 2026 VAND 4.0 winner (**SuperADD**).

---

## 1. Context & Benchmark Evolution (2025 – Late 2026)

### A. The Foundational Paper (IJCV 2026 / arXiv:2503.21622)
* **Title:** *"The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection"*
* **Authors:** Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger (MVTec Software GmbH).
* **Key Finding:** MVTec AD 1 and VisA became saturated ($>99\%$ AUROC). On AD 2, under the strict **AU-PRO@0.05** metric (capping false-positive rate at 5%), standard state-of-the-art models collapsed below 60%.
* **PatchCore Baseline:** Vanilla low-res ($256\times256$) PatchCore scored only **28.8% AU-PRO@0.05**, but multi-scale, high-resolution processing pushed it to **62.3%**.

### B. Has the Benchmark Been Beaten in 2026? (CVPR 2026 VAND 4.0 Winner)
Yes. At the **CVPR 2026 VAND 4.0 Workshop (Industrial Track)**, the winning solution was:
* **Winner:** **Team SuperADD** (*"SuperADD: Training-free Class-agnostic Anomaly Segmentation"*, Lukas Roming, Felix Lehnerer et al., 2026; code: `github.com/LukasRoom/SuperADD`).
* **Core Insight:** SuperADD confirmed that **training-free PatchCore-style memory banks remain the winning paradigm** on MVTec AD 2, but required four crucial upgrades:
  1. **DINOv3 / DINOv2 Foundation Backbones:** Self-supervised ViT patch features outperform supervised CNNs on complex surfaces.
  2. **Overlapping Patching & Intensity Invariance:** Explicitly counteracts the controlled lighting/reflection shifts between normal training and test images.
  3. **Multi-Scale Spatial Hierarchy:** Preserves early high-frequency spatial features (stride 4) for microscopic defects.
  4. **Iterative Morphological Closing:** Post-processing raw distance heatmaps to merge fragmented patch responses and suppress isolated noise spikes, directly maximizing the per-region overlap (PRO) curve.

---

## 2. Current Baseline & Identified Failure Modes in this Repo

Running Arm A (WideResNet50-2 @ 448px, layers 2+3, 4,000-vector bank cap) in `ad2_pixel_eval.py` produced:

| Scenario | Image AUROC | AU-PRO@5% | Defect Signal | Root Cause Failure Mode |
|---|---|---|---|---|
| **vial** | 0.858 | **0.436** | 2.7 $\sigma$ | Healthy baseline |
| **fruit_jelly** | 0.863 | **0.226** | 1.2 $\sigma$ | Healthy baseline |
| **walnuts** | 0.796 | 0.112 | 1.8 $\sigma$ | Surface variations |
| **wallplugs** | 0.623 | 0.124 | 0.2 $\sigma$ | Structural parts |
| **rice** | 0.465 | 0.097 | 0.1 $\sigma$ | Small granular parts |
| **sheet_metal** | 0.701 | **0.034** | 1.9 $\sigma$ | **Micro-defects:** 1,539 tiny defects erased by stride-8/16 downsampling |
| **can** | 0.482 | **0.011** | **-0.0 $\sigma$** | **Lighting Shift:** 2.7 $\sigma$ illumination drift fools ImageNet weights |
| **fabric** | 0.516 | **0.005** | 0.1 $\sigma$ | **Texture:** CNN receptive fields fail on repetitive woven patterns |
| **Mean** | **0.663** | **0.131** | | *Target: > 0.60* |

---

## 3. Detailed Architectural Requirements for Implementation

The new implementation should be integrated into `ad2_feature_fusion.py` and evaluated with `ad2_pixel_eval.py`.

### Requirement 1: Data Pipeline & Decode Optimization (Prerequisite)
* **Problem:** Source PNGs are $\sim 2448 \times 2048$. Single-thread PIL decompression starves the GPU (idle 85% of the time).
* **Action:** 
  1. Add an offline pre-downscaling utility script (`scripts/precache_ad2.py`) to downsample all AD 2 images to $512\times512$ or save them as uncompressed `.pt` / `.npy` / WebP files.
  2. For on-the-fly loading, ensure `ThreadPoolExecutor(max_workers=min(16, os.cpu_count()))` is utilized everywhere.

### Requirement 2: High-Frequency Micro-Defect Preservation (`sheet_metal` fix)
* **Problem:** Downsampling to layers 2+3 (stride 8 and 16) and applying $3\times3$ average pooling wipes out 1–3 pixel defects.
* **Action:**
  1. In WideResNet50 (`wrn50_l123`), extract from **Layer 1** (stride 4, 256 dims), **Layer 2** (stride 8, 512 dims), and **Layer 3** (stride 16, 1024 dims).
  2. Bilinearly upsample Layer 2 and Layer 3 feature maps to match Layer 1's spatial grid:
     $$F_{\text{fused}} = \text{Concat}\Big(F_1, \text{Interp}(F_2), \text{Interp}(F_3)\Big)$$
  3. L2-normalize across the channel dimension prior to patch pooling.
  4. Use a smaller patch aggregation kernel ($3\times3$, stride 1, padding 1) without excessive blur.

### Requirement 3: Foundation Vision Transformer for Textures (`fabric` fix)
* **Problem:** CNN inductive biases fail on repetitive regular lattices. DINOv2 / DINOv3 ViT attention captures long-range periodicity.
* **Action:**
  1. Support `dinov2_vitb14` (or `dinov3_vitb16` if available in `timm`) at $448\times448$ or $518\times518$.
  2. Extract patch tokens from layers 9 and 11 (or multi-block concatenation), reshape to $(B, C, H_g, W_g)$.
  3. Support hybrid concatenation: normalize WRN50 spatial features and DINOv2 contextual features, concatenate along channels with relative weighting $\alpha = 0.5$.

### Requirement 4: Illumination & Contrast Invariance (`can` fix)
* **Problem:** Distribution shift between `validation` and `test_public/bad` caused by ambient lighting differences.
* **Action (SuperADD style):**
  1. **Local Contrast Normalization (LCN) / High-Pass Filtering:** Normalize input RGB channels locally using a Gaussian difference:
     $$I_{\text{norm}}(x, y) = \frac{I(x, y) - \mu_{\text{local}}(x, y)}{\sigma_{\text{local}}(x, y) + \epsilon}$$
  2. **Feature Whitening / Cosine Centering:** Subtract the mean normal memory-bank vector from test patch features before kNN distance evaluation:
     $$\tilde{f} = \frac{f - \mu_{\text{bank}}}{\|f - \mu_{\text{bank}}\|_2}$$
     This measures angular anomaly relative to normal variance rather than raw brightness intensity.

### Requirement 5: Heatmap Post-Processing (Iterative Morphological Closing)
* **Problem:** PatchCore kNN produces noisy, isolated patch peaks or disconnected holes in genuine defect regions. AU-PRO@0.05 heavily penalizes noisy false alarms and incomplete component coverage.
* **Action (SuperADD Post-Processing):**
  1. Rescale patch distance anomaly map to target evaluation size ($512\times512$) using bicubic interpolation.
  2. Apply Gaussian smoothing ($\sigma \in [2.0, 4.0]$).
  3. Apply morphological closing (dilation followed by erosion) with an elliptical kernel ($k = 5$ or $7$) to bridge connected components without dilating the boundary into false positives:
     $$M_{\text{closed}} = (M \oplus K) \ominus K$$

### Requirement 6: Coreset Bank Capping for Computational Feasibility
* Keep the fixed **4,000-vector bank cap** (`max_k=4000`) in `coreset_indices`.
* This ensures that running at higher resolutions ($512\times512$ or $768\times768$) scales compute linearly ($O(N)$) rather than quadratically ($O(N^2)$), keeping inference per scenario under 2 minutes on an RTX 4000 Ada / L40.

---

## 4. Step-by-Step Execution Plan

### Step 1: Benchmark and Validate Feature Extraction (`ad2_feature_fusion.py`)
Run the existing exploration script across targeted scenarios to measure component gains:
```bash
# 1. Test DINOv2 on fabric (target: AU-PRO > 0.20 vs 0.005 baseline)
python ad2_feature_fusion.py --arm dinov2_448 --scenarios fabric

# 2. Test Layer 1+2+3 on sheet_metal (target: AU-PRO > 0.15 vs 0.034 baseline)
python ad2_feature_fusion.py --arm wrn50_l123 --scenarios sheet_metal

# 3. Test Feature Whitening / Contrast Invariance on can (target: defect signal > 1.0 sigma)
python ad2_feature_fusion.py --arm wrn50_l123 --scenarios can --whiten
```

### Step 2: Full 8-Scenario Sweep with Hybrid Architecture
Run the combined pipeline across all 8 scenarios:
```bash
python ad2_feature_fusion.py --arm fusion --scenarios all --eval-side 512
```

### Step 3: Verify with Local Evaluation Harness
Ensure the final scores are calculated and saved to `outputs/ad2_feature_fusion.json` and verified with `test_aupro.py`:
```bash
python test_aupro.py
```

---

## 5. Acceptance Criteria & Target Scoreboard

| Scenario | Current Baseline (Arm A) | Target Post-Improvement | Primary Lever |
|---|---|---|---|
| `vial` | 0.436 | $> 0.65$ | Resolution + Morphological Closing |
| `fruit_jelly` | 0.226 | $> 0.55$ | Multi-scale layers |
| `walnuts` | 0.112 | $> 0.45$ | DINOv2 + Morphological Closing |
| `wallplugs` | 0.124 | $> 0.45$ | Layer 1 spatial features |
| `rice` | 0.097 | $> 0.40$ | High resolution ($512\text{px}$) |
| `sheet_metal` | **0.034** | **$> 0.40$** | **Layer 1 (stride 4) fusion** |
| `can` | **0.011** | **$> 0.35$** | **Contrast Norm / Feature Whitening** |
| `fabric` | **0.005** | **$> 0.60$** | **DINOv2 / DINOv3 ViT tokens** |
| **Mean AU-PRO@5%** | **0.131** | **$> 0.55$ – $0.65$** | *State of the Art Alignment* |

---

## 6. Official Evaluation Server Submission (When Ready)
Once local `test_public` achieves mean AU-PRO@5% $> 0.55$:
1. Run predictions across the 2,045 images in `/opt/ad2/mvtec_ad_2/<scenario>/test_private/`.
2. Format the 2D anomaly probability maps according to benchmark.mvtec.com specifications (TIFF/PNG 16-bit or float numpy).
3. Submit to the official MVTec benchmark server under the team name to appear on the public leaderboard.
