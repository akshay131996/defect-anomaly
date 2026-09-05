# Published MVTec AD 2 Benchmark Baselines

Data extracted directly from the dataset publication:
**The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection**
Authors: L. Heckler-Kram, P. Neudeck, M. Scheler, R. König, C. Steger (arXiv:2503.21622, March 2025).

---

## 1. Primary Benchmark: AU-PRO@0.05 at 256x256 (Table VII)
Format: TESTpriv / TESTpriv,mix (percentage)

| Object | PatchCore | EfficientAD | RD++ | RD | MSFlow | SimpleNet | DSR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Can** | 4.7 / 4.6 | 9.6 / 1.3 | 7.7 / 7.0 | 7.0 / 7.5 | 6.7 / 0.8 | 8.4 / 1.9 | 13.9 / 3.5 |
| **Fabric** | 11.0 / 12.0 | 22.2 / 13.0 | 4.8 / 5.3 | 3.5 / 3.6 | 14.1 / 14.3 | 6.6 / 7.3 | 6.8 / 5.3 |
| **Fruit Jelly** | 46.7 / 46.7 | 50.5 / 47.6 | 54.4 / 54.4 | 48.2 / 48.2 | 49.4 / 38.3 | 39.8 / 38.5 | 36.0 / 34.2 |
| **Rice** | 25.6 / 18.5 | 27.6 / 4.3 | 12.2 / 10.2 | 11.2 / 11.4 | 21.5 / 12.2 | 8.7 / 4.2 | 7.8 / 8.3 |
| **Sheet Metal** | 15.2 / 13.0 | 11.8 / 5.3 | 9.3 / 8.9 | 9.5 / 8.8 | 11.6 / 7.7 | 12.0 / 9.1 | 18.0 / 16.1 |
| **Vial** | 62.2 / 59.1 | 55.6 / 47.7 | 63.0 / 57.0 | 62.1 / 60.3 | 38.0 / 4.0 | 47.8 / 23.3 | 50.0 / 48.1 |
| **Wall Plugs** | 12.8 / 9.9 | 20.3 / 1.2 | 15.4 / 10.1 | 19.7 / 12.2 | 12.6 / 0.2 | 5.7 / 1.9 | 3.9 / 6.5 |
| **Walnuts** | 51.8 / 44.5 | 48.8 / 33.0 | 49.7 / 48.9 | 50.2 / 48.3 | 40.4 / 18.1 | 40.0 / 23.4 | 25.7 / 17.1 |
| **Mean** | **28.8 / 26.0** | **30.8 / 19.2** | **27.1 / 25.2** | **26.4 / 25.0** | **24.3 / 11.9** | **21.1 / 13.7** | **20.3 / 17.4** |

*Note on published PatchCore configuration (Section IV-A):*
- Ensembling of 3 backbones: **WideResNet-101**, **ResNeXt-101**, **DenseNet-201**.
- Aggregated embedding reduced to **384** dimensions.
- Greedy coreset ratio: 0.01% (or 0.01 fraction).
- Center cropping: **disabled** (same as our aspect-preserving coordinates).

---

## 2. Standard Benchmark: AU-PRO@0.30 at 256x256 (Table IX)
Format: TESTpriv / TESTpriv,mix (percentage)

| Object | PatchCore | EfficientAD | RD++ | RD | MSFlow | SimpleNet | DSR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Can** | 21.6 / 18.1 | 38.1 / 24.4 | 44.7 / 29.4 | 42.9 / 30.1 | 33.4 / 15.8 | 36.3 / 20.7 | 50.0 / 26.3 |
| **Fabric** | 34.6 / 35.3 | 46.8 / 41.3 | 26.2 / 29.5 | 22.3 / 25.6 | 38.2 / 38.4 | 25.7 / 26.0 | 23.7 / 25.6 |
| **Fruit Jelly** | 74.0 / 74.0 | 79.5 / 78.0 | 80.4 / 80.3 | 78.9 / 79.0 | 78.8 / 73.5 | 71.8 / 70.3 | 70.1 / 69.8 |
| **Rice** | 50.9 / 43.1 | 52.1 / 19.1 | 34.2 / 32.9 | 30.2 / 31.3 | 48.5 / 38.8 | 25.5 / 24.3 | 28.4 / 28.0 |
| **Sheet Metal** | 39.8 / 34.2 | 36.1 / 24.5 | 27.6 / 25.1 | 26.6 / 25.1 | 35.8 / 29.0 | 32.3 / 22.9 | 46.9 / 45.2 |
| **Vial** | 90.5 / 89.2 | 88.7 / 85.2 | 91.5 / 87.3 | 91.3 / 90.0 | 79.1 / 39.7 | 82.5 / 67.5 | 88.1 / 85.2 |
| **Wall Plugs** | 37.4 / 34.8 | 51.5 / 17.9 | 51.0 / 40.5 | 51.0 / 41.9 | 35.5 / 9.7 | 27.4 / 12.9 | 23.6 / 26.8 |
| **Walnuts** | 81.7 / 77.4 | 76.5 / 62.3 | 78.9 / 78.0 | 81.2 / 79.6 | 72.1 / 51.8 | 69.4 / 56.1 | 60.8 / 54.5 |
| **Mean** | **53.8 / 50.8** | **58.7 / 44.1** | **54.3 / 50.4** | **53.0 / 50.3** | **52.7 / 37.1** | **46.4 / 37.6** | **49.0 / 45.2** |

---

## 3. Resolution Scaling: AU-PRO@0.05 at Larger Dimensions (Tables X & XI)

### Table X: Intermediate Resolution Scaling
Format: TESTpriv / TESTpriv,mix (percentage)

| Object | PatchCore | EfficientAD | RD++ | RD | MSFlow | SimpleNet | DSR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Can** | 8.4 / 7.0 | 14.2 / 2.4 | 13.1 / 12.6 | 10.5 / 10.7 | 24.2 / 1.3 | 19.0 / 2.3 | 19.8 / 3.3 |
| **Fabric** | 23.7 / 23.5 | 44.5 / 28.3 | 41.2 / 43.0 | 36.8 / 39.6 | 31.7 / 28.2 | 22.5 / 23.6 | 30.3 / 28.7 |
| **Fruit Jelly** | 64.9 / 64.8 | 55.4 / 53.5 | 58.4 / 58.4 | 50.6 / 50.4 | 63.7 / 52.3 | 54.9 / 51.7 | 45.8 / 42.9 |
| **Rice** | 33.6 / 15.5 | 29.9 / 3.1 | 16.0 / 14.4 | 13.1 / 12.0 | 27.2 / 19.0 | 10.1 / 6.1 | 20.5 / 15.9 |
| **Sheet Metal** | 27.3 / 24.1 | 21.7 / 12.4 | 24.3 / 22.0 | 20.0 / 20.1 | 23.4 / 9.5 | 19.0 / 14.1 | 10.1 / 8.4 |
| **Vial** | 72.6 / 68.4 | 60.4 / 55.3 | 71.9 / 69.1 | 67.6 / 64.7 | 53.2 / 4.6 | 56.7 / 32.5 | 23.2 / 14.1 |
| **Wall Plugs** | 34.0 / 21.3 | 33.2 / 2.3 | 35.5 / 19.2 | 43.9 / 21.3 | 31.4 / 0.3 | 24.8 / 6.9 | 6.3 / 7.3 |
| **Walnuts** | 70.5 / 61.4 | 60.8 / 45.4 | 55.5 / 52.4 | 55.8 / 53.5 | 59.0 / 35.7 | 49.3 / 32.6 | 44.1 / 30.2 |
| **Mean** | **41.9 / 35.8** | **40.0 / 25.3** | **39.5 / 36.4** | **37.3 / 34.0** | **39.2 / 18.9** | **32.0 / 21.2** | **25.0 / 18.8** |

### Table XI: Half-Native Resolution Scaling (~1000-2000px)
Format: TESTpriv / TESTpriv,mix (percentage)

| Object | PatchCore | EfficientAD | RD++ | RD | MSFlow | SimpleNet |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Can** | 12.8 / 10.2 | 20.8 / 3.3 | 18.2 / 15.8 | 15.0 / 13.7 | 26.6 / 2.5 | 21.9 / 3.2 |
| **Fabric** | 69.0 / 59.1 | 72.5 / 45.4 | 82.1 / 77.8 | 81.1 / 77.9 | 82.9 / 60.6 | 66.0 / 55.6 |
| **Fruit Jelly** | 71.5 / 70.8 | 54.5 / 52.9 | 63.3 / 62.8 | 54.9 / 54.7 | 72.8 / 67.6 | 64.9 / 63.0 |
| **Rice** | 47.8 / 28.9 | 37.7 / 5.4 | 31.2 / 28.5 | 27.4 / 26.2 | 47.1 / 32.7 | 21.9 / 12.3 |
| **Sheet Metal** | 72.4 / 54.0 | 52.0 / 37.2 | 57.9 / 46.8 | 54.2 / 51.4 | 42.2 / 11.9 | 46.8 / 34.1 |
| **Vial** | 75.8 / 72.2 | 61.4 / 56.4 | 73.2 / 58.8 | 69.9 / 67.7 | 56.1 / 1.7 | 56.1 / 38.8 |
| **Wall Plugs** | 68.4 / 53.7 | 42.2 / 12.4 | 52.8 / 24.7 | 55.1 / 39.5 | 65.1 / 1.1 | 25.6 / 10.5 |
| **Walnuts** | 80.4 / 71.6 | 65.0 / 54.9 | 64.1 / 59.1 | 71.7 / 64.6 | 72.6 / 53.8 | 62.7 / 55.7 |
| **Mean** | **62.3 / 52.6** | **50.8 / 33.5** | **55.3 / 46.8** | **53.7 / 49.5** | **58.2 / 29.0** | **45.7 / 34.1** |

---

## 4. Comparison with Our Pipeline (	est_public)

| Metric | Published PatchCore (	est_priv @ 256) | Published PatchCore (	est_priv @ Half-Native) | Our Baseline E4b (	est_public @ 448 Aspect) | Our E5b Dilated L3 (	est_public @ 448 Aspect) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean AU-PRO@0.05** | 28.8% | 62.3% | **34.4%** | **36.9%** |
| **Mean AU-PRO@0.30** | 53.8% | — | **57.4%** | **60.4%** |
| **Active Regions** | — | — | 1,530 / 1,530 | 1,530 / 1,530 |
| **Embedding Dims** | 384 (Projected) | 384 (Projected) | 1,536 (Full) | 1,536 (Full) |
| **Backbones** | Ensemble (3 models) | Ensemble (3 models) | Single WideResNet50-2 | Single WideResNet50-2 |
