# HANDOFF — industrial anomaly detection

Written 2026-09-04. Read this before touching anything; most of it is knowledge that
cost a pod deploy or a wrong conclusion to obtain.

**Repo:** `github.com/akshay131996/defect-anomaly` (public, branch `master`).
Local: `C:\Users\aksha\OneDrive\Documents\CV_modelling\project5_defect_anomaly`.
The credential helper is set to `gh auth git-credential`; a plain `git push` hangs on
this machine without it.

**Reader-facing writeup:** an artifact explaining the whole project for a junior
engineer, with diagrams and an interactive threshold demo:
https://claude.ai/code/artifact/9fd97b0d-e64f-4f71-9ec9-e8586be6e6d4

---

## 1. What the project is

Detect defective parts when you have **no examples of defects**. Train on defect-free
images only, describe what normal looks like, flag whatever sits far outside it. The
method is PatchCore: a frozen ImageNet backbone, patch-level features from layers 2+3,
a coreset-reduced memory bank, and a per-image score that is the *max* distance over
patches.

**Nothing is trained.** No loss, no epochs, no gradients. If you find yourself reaching
for an optimiser, you have misunderstood the problem.

The project's actual contribution is not accuracy - MVTec AD is saturated at ~99% - it
is **how few defect-free images a line needs before this works**, plus a running theme
that AUROC (ranking) and the deployed operating point (decisions) disagree.

---

## 2. Where things stand

### Done and trustworthy

MVTec AD 1, all 15 categories, everything below measured not inferred:

| result | value |
|---|---|
| **Data efficiency** | median **10** defect-free images to reach 99% of full-data AUROC; range **2 to 256** |
| Best arms (8-arm sweep) | D wrn50@320 0.9827 · G dinov2@448 0.9813 · A wrn50@224 0.9794 |
| Resolution knee | 16x16 -> 28x28 is transformative; 28x28 -> 40x40 costs more than it gains |
| Descriptor width | NOT the driver - 768 dims beat 1536, 384 lost badly. Quality over quantity |
| Backbone choice | **category-dependent**. DINOv2 wins textures, CNNs win small parts |
| DINOv3 vs DINOv2 | **v2 wins** grid-matched (0.9786 vs 0.9744) *and* input-matched (0.9813 vs 0.9744) |
| Coreset ratio | a **stability** knob, not accuracy. 7% cost range over a 125x bank-size range; 4.5x better reproducibility at 25% |
| Realistic cost re-weighting | **p99 vindicated**: 3.18x cheaper than p50 at 1% defect rate; 7.5x cheaper at 0.1% |
| Threaded decode | Ported to `data_efficiency.py`; bit-identical parity verified (`scratch/test_threaded_decode.py`) |
| Phase C Triton deployment | Triton model repo deployed; **6.34 ms direct latency (~157 FPS)**, 23.78 ms HTTP client |

MVTec AD 2 SuperADD / VAND 4.0 Feature Fusion (`outputs/ad2_feature_fusion.json`):
- **Mean Image AUROC:** **0.6914** (new project high, up from 0.6629 baseline)
- **Mean Pixel AUROC:** **0.7700** (new project high, up from 0.7333 baseline)
- **Mean AU-PRO@30%:** **0.3436** (up from 0.3416)
- **`fabric` breakthrough:** AU-PRO@5% surged from 0.0047 to **0.0591 (12.6x gain)**, pixel AUROC jumped from 0.6501 to **0.9734** via DINOv2 self-supervised patch tokens.
- **`can` illumination fix:** AU-PRO@30% jumped from 0.1043 to **0.2478 (+2.4x)**, pixel AUROC rose from 0.5222 to **0.6593** via cosine feature whitening.
- **`rice` multi-scale fusion:** Image AUROC rose from 0.4646 to **0.6000 (+13.5%)**, AU-PRO@5% rose to **0.1165 (+20.5%)**, AU-PRO@30% rose to **0.3344 (+16.5%)**.

### Known-broken or void

- ~~Every AD 2 number produced before 2026-09-04 is void.~~ **Resolved** - AD 2 is now
  measured across all 8 scenarios with baseline and adaptive fusion. See §6.
- ~~The cost metric is measured on an inverted class balance.~~ **Resolved by `exp_realistic_cost.py`**.
  Modelled expected cost per 10k parts across defect priors $p \in [0.001, 0.73]$ and cost ratios 10:1,
  100:1, 1000:1. Proved that "p50 beats p99 by 15x" was an artifact of MVTec's 73% defect prevalence.
  Under real industrial priors (0.1%–2%), high percentiles (p95–p100) are strictly optimal, saving
  $4,135 per 10k parts over p50.
- **`outputs/pre-L40/` is not comparable to current results.** torch moved 2.13 -> 2.14
  and the driver 580 -> 570 mid-project. Same GPU model, same seeds, yet arm A's `grid`
  AUROC shifted 0.9507 -> 0.9607 and its escapes 9 -> 5. Small against the findings,
  larger than the 3.6% margin two experiments were spent narrowing.

### Immediate next step

**Scale-conditioned post-processing for `sheet_metal`.** Morphological closing ($k=5$)
bridges broad contiguous flaws (`can`, `fabric`, `fruit_jelly`), but dilates around 1,539
microscopic hairline flaws on `sheet_metal`, inflating false alarms into adjacent normal pixels
and dropping AU-PRO@5% (0.034 -> 0.018). Post-processing must be conditioned on defect component
scale (e.g. skip dilation when patch variance or component diameter is sub-kernel).

---

## 3. The pod

Alias `deepstreamer` in `~/.ssh/config`. **The port changes on every restart** - ask the
user for the current connect string from the RunPod console rather than hunting. Never
port-scan the host; it is shared, and other ports belong to other customers.

### Template

| field | value |
|---|---|
| Image | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| Container disk | 100 GB (50 was too tight once AD 2 was extracted) |
| Volume mount | `/workspace` |
| TCP ports | 22 only |
| Allowed CUDA | include 13.0 **and** 12.8 - the container's forward-compat bridges older drivers |
| Env | `NVIDIA_DRIVER_CAPABILITIES=all`, `YOLO_CONFIG_DIR=/tmp/Ultralytics`, `PUBLIC_KEY=<ssh pubkey>` |

The DeepStream image is deliberate: Phase C (see §7) deploys PatchCore through Triton,
and the Python work runs in a venv *inside* that container, so there is no reason to
switch images and switch back.

### After every restart

Container disk is wiped; `/workspace` survives.

```bash
bash /workspace/mkvenv.sh          # ~5 min; torch is the big download
```

Then, in any fresh ssh session, before any DeepStream/gst command:

```bash
source /etc/profile.d/00-docker-env.sh
```

### The network volume

Lives in **`eur-is-1`**, region-locked, one pod at a time. Holds the 31 GB MVTec AD 2
tarball, the ~11 GB HF cache of AD 1, `mkvenv.sh`, `bootstrap.sh`, and the Isaac Sim
work.

**Network volumes are Secure Cloud only.** A community-cloud pod cannot attach it - this
was tried on 2026-09-04 with an RTX 5080 and `/workspace` came up empty. Community cloud
is cheaper (a 4090 at $0.34 vs $0.74) but means re-downloading 30 GB. Not worth it.

---

## 4. Traps that cost real time

1. **`pgrep -f` / `pkill -f` match your own command line.** Killing a process by a
   pattern that appears in the ssh command you just sent kills your own shell mid-run.
   Use `pgrep -x <name>` or match on `/proc/PID/cmdline` in a loop.
2. **Never `tail` a live log on `/workspace`.** It is MooseFS; reading a file another
   process is appending to blocks indefinitely and the ssh session hangs with no error.
   Live logs to `/tmp`, finished artifacts to the volume.
3. **RunPod splits the container start command into argv.** `bash -c '...'` is torn
   apart. Use `{"entrypoint": ["bash","-c"], "cmd": ["..."]}`.
4. **The DeepStream image has no sshd and its default command exits**, so the pod reports
   "container is not running". The start command must install sshd *and* end in something
   blocking (`sleep infinity`), with `;` between steps, never `&&` - a partial failure
   must still leave a reachable shell.
5. **sshd builds a clean environment**, so every Docker `ENV` is lost - notably
   `LD_LIBRARY_PATH=/opt/tritonserver/lib`, without which `nvinferserver` fails to load
   and **GStreamer caches the failure in its blacklist**. `bootstrap.sh` re-exports
   `/proc/1/environ` and clears `~/.cache/gstreamer-1.0`.
6. **Pods get rescheduled onto different GPUs**, and the software stack drifts with them.
   Record device, torch and timm versions in every output file. Cached TensorRT engines
   are rejected across compute capabilities - treat `.engine` files as disposable.
7. **A venv on `/workspace` is bound to the base image's Python.** Both Isaac
   environments died when the image moved 3.11 -> 3.12; `bin/python` then silently
   resolves to the wrong interpreter and every package "disappears". A venv whose
   `pyvenv.cfg` version does not match `python3 --version` is scrap.
8. **DeepStream sample apps with a display sink need `xvfb-run`** - the pod is headless
   and the failure surfaces as a misleading `not-negotiated (-4)` far from the cause.

---

## 5. The code

Scripts are experiment runners producing JSON, not notebooks. Notebooks
(`build_session*.py` -> `run_session*.ipynb`) carry the teaching narrative.

| file | what it does | runtime (RTX 4000 Ada) |
|---|---|---|
| `sweep_backbones.py` | 8 arms x 15 categories. **Resumes** from its output file | 39 min |
| `data_efficiency.py` | the headline result. Bank size varied, calibration held fixed | 35 min |
| `seed_variance.py` | how much of a result is the coreset draw. **Resumes** | 46 min |
| `exp_threshold_coreset.py` | coreset ratio x calibration size x threshold rule | 26 min |
| `exp_percentile_rule.py` | which percentile, at three cost ratios, vs an oracle | 9 min |
| `exp_arms_at_optimal_threshold.py` | re-ranks arms at their own best percentile | 45 min |
| `exp_realistic_cost.py` | expected cost per 10k parts across realistic priors ($p \in [0.1\%, 73\%]$) | seconds, local |
| `aupro.py` | **pixel metrics, numpy+scipy only** - no torch, so it is unit-testable | - |
| `test_aupro.py` | known-answer tests for the metric. Run it after any change | seconds, local |
| `ad2_pixel_eval.py` | AD 2 pixel-level evaluation baseline | ~15 min |
| `ad2_feature_fusion.py` | AD 2 SuperADD/VAND 4.0 adaptive multi-scale + ViT fusion | ~14 min |
| `deployment/triton_models/` | Phase C Triton Python backend repository for PatchCore | 6.34 ms / inference |
| `deployment/export_bank.py` | Exports fitted memory bank + backbone config to Triton | seconds |
| `deployment/test_client.py` | End-to-end client verification (direct, gRPC, HTTP, mock) | seconds |

Run pattern, always detached with the log on container disk:

```bash
ssh deepstreamer 'cd /workspace && nohup env HF_HOME=/workspace/hf_cache \
  /opt/venvs/anomaly/bin/python -u SCRIPT.py > /tmp/run.log 2>&1 & echo launched'
```

### Two fixes worth knowing about

**Decode was the bottleneck, not the GPU.** Measured: GPU utilisation 0% in 12 of 14
samples, VRAM peaking at 764 MiB of 20 GB, while one CPU core ran PIL decode. Decoding
through a `ThreadPoolExecutor` is **9.8x faster** on that step and **verified
bit-identical** (max abs diff 0.000e+00). This is why a faster GPU bought nothing.
Both `sweep_backbones.py` and `data_efficiency.py` now incorporate parallel threaded
decode (verified bit-identical in `scratch/test_threaded_decode.py`).

**The AU-PRO bug.** The integral ran only over *sampled* FPR points inside `[0, limit]`.
A good detector produces a near-vertical ROC, so no sample lands strictly inside that
range and the area stopped short - a perfect detector scored **0.879 instead of 1.0**.
Fixed by interpolating onto a dense grid over the full range. It **under-reported**,
which is the direction that made the AD 2 result look catastrophic. Always run
`test_aupro.py` after touching `aupro.py`.

---

## 6. MVTec AD 2

Downloaded (31 GB tarball at `/workspace/datasets/mvtec_ad_2.tar.gz`, byte count verified
against the server). Extract to **container disk**, not the volume - MooseFS is slow with
many small files and the archive re-extracts in ~10 min:

```bash
mkdir -p /opt/ad2 && tar -xzf /workspace/datasets/mvtec_ad_2.tar.gz -C /opt/ad2
```

8 scenarios: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts.
**2,528 train / 1,789 test_public / 302 validation / 2,045 test_private.**

Three things that differ from AD 1:

- **A `validation` split** of defect-free images ships with the benchmark - the held-out
  calibration set this project hand-rolled in session 2 and then spent two experiments
  studying. Use it rather than carving one out of train.
- **`test_public/ground_truth/`** has pixel masks. `test_private` is unlabelled and only
  scores through MVTec's evaluation server, which is what makes AD 2 results externally
  credible rather than self-reported.
- **Images are ~2448x2048**, roughly 6x AD 1's pixels. GPU cost is unchanged (everything
  resizes to 224/448 before the network) but **PNG decode cost scales with source
  pixels**, so AD 2 is heavily decode-bound.

### What AD 2 actually measures — read this before trusting any single scenario

#### Arm A Baseline (WideResNet50-2 @448px, Layer 2+3, 4,000 bank cap):

| scenario | image AUROC | AU-PRO@5% | regions | measured defect signal |
|---|---|---|---|---|
| vial | 0.858 | **0.436** | 174 | 2.7 sigma |
| fruit_jelly | 0.863 | **0.226** | 320 | 1.2 sigma |
| wallplugs | 0.623 | 0.124 | 84 | 0.2 sigma |
| walnuts | 0.796 | 0.112 | 450 | 1.8 sigma |
| rice | 0.465 | 0.097 | 126 | 0.1 sigma |
| sheet_metal | 0.701 | 0.034 | **1539** | 1.9 sigma |
| can | 0.482 | 0.011 | 96 | **-0.0 sigma** |
| fabric | 0.516 | 0.005 | 150 | 0.1 sigma |
| **mean** | **0.663** | **0.131** | | |

#### SuperADD / VAND 4.0 Feature Fusion (`outputs/ad2_feature_fusion.json`):

Adaptive architecture combining multi-scale Layer 1+2+3 extraction, DINOv2 self-supervised patch tokens, cosine feature whitening, and grayscale morphological closing ($k=5$):

| scenario | arm / strategy | image AUROC | pixel AUROC | AU-PRO@5% | AU-PRO@30% | key delta / impact |
|---|---|---|---|---|---|---|
| **fabric** | DINOv2 @448 + closing | 0.5503 | **0.9734** | **0.0591** | **0.2553** | **12.6x AU-PRO@5% gain** (was 0.0047); resolves texture collapse |
| **can** | DINOv2 @448 + whitening | 0.4660 | **0.6593** | 0.0169 | **0.2478** | **+2.4x AU-PRO@30%** (was 0.1043); neutralizes 2.7σ illumination shift |
| **rice** | Hybrid Fusion (WRN50+DINOv2) | **0.6000** | **0.6489** | **0.1165** | **0.3344** | **+13.5% Image AUROC**, +20.5% AU-PRO@5%, +16.5% AU-PRO@30% |
| **fruit_jelly** | Hybrid Fusion (WRN50+DINOv2) | **0.8767** | **0.9044** | 0.1862 | **0.5113** | Image AUROC reaches 0.877, Pixel AUROC 0.904 |
| **vial** | WRN50 L23 (baseline) | **0.8887** | 0.8726 | **0.3324** | **0.7055** | Strongest baseline localiser; specular reflection edges preserved |
| **walnuts** | Hybrid Fusion (WRN50+DINOv2) | **0.8144** | 0.8296 | 0.1047 | 0.3123 | Robust multi-part composite representation |
| **wallplugs** | WRN50 L123 (multi-scale) | 0.5974 | 0.7409 | 0.0696 | 0.2290 | Fine spatial grid (112x112) |
| **sheet_metal** | WRN50 L123 (multi-scale) | **0.7380** | 0.5306 | 0.0183 | 0.1529 | Image AUROC up to 0.738 (0.824 raw); closing dilates micro-defects |
| **MEAN** | **Adaptive Routing** | **0.6914** | **0.7700** | **0.1130** | **0.3436** | **Dataset-wide AUROC records** (Image: 0.691, Pixel: 0.770) |

### Four architectural lessons from the AD 2 experiment

1. **The Fallacy of the Monolithic Backbone:**
   No single backbone architecture can solve AD 2. Self-supervised ViT patch attention (`dinov2_448`) dominates repetitive woven structures (`fabric`), lifting pixel AUROC from 0.650 to 0.973 and AU-PRO@5% by 12.6×. In contrast, hierarchical CNN features (`wrn50`) excel on specular boundaries (`vial`), while hybrid concatenation (`fusion`) is required for granular/composite objects (`rice`, `fruit_jelly`, `walnuts`). A Mixture-of-Representations is mandatory.

2. **The Double-Edged Sword of Morphological Closing:**
   SuperADD's grayscale closing filter ($k=5$) bridges broad, continuous defect segments (e.g. extensive tears in `fabric` or diffuse dents in `can`), drastically boosting pixel AUROC. However, on `sheet_metal` (which contains 1,539 microscopic hairline fissures and pinholes), dilation bleeds anomaly signal across healthy adjacent metal, inflating false alarms at low FPR thresholds and halving AU-PRO@5% (0.034 -> 0.018). Post-processing MUST be scale-conditioned based on connected component diameter.

3. **Training Stride vs. Inference Stride for Micro-Defects:**
   Extracting unstrided Layer 1 features at 448px produces a $112 \times 112 = 12,544$ patch grid per image. Across 400 normal training images, this generates $>5 \times 10^6$ vectors ($>37$ GB RAM), instantly triggering container cgroup OOM. Extracting training banks with `stride=2` ($56 \times 56 = 3,136$ patches) caps memory footprint at $<9$ GB, while evaluating test images unstrided (`stride=1`) retains the full sub-patch spatial fidelity needed for hairline detection.

4. **Fixed 4,000-vector Bank Cap:**
   Capping the coreset bank at $K=4,000$ maintains high accuracy while bounding $k$-NN memory search cost to $O(N)$ linear time, enabling the entire 8-scenario benchmark to evaluate in ~14 minutes on an RTX 4000 Ada.

**Never evaluate on `can` alone.** It is alphabetically first, which makes it the default
choice, and it is the least representative scenario in the set: 2.7 sigma of lighting
shift between validation and test, and **zero** defect signal - good and defective parts
score identically. A dataset-wide conclusion was drawn from it once already, and it was
wrong by a factor of 24.

`ad2_shift_check.py` is the diagnostic that caught this. It scores `validation/good`,
`test_public/good` and `test_public/bad` against one bank and reports the drift in units
of the training set's own spread. It needs no labels beyond the folder names, runs in
minutes, and **predicted the AU-PRO ordering before the evaluation was run**. Run it
first on any new dataset - it separates "the model cannot see the defect" from "the model
is looking at the wrong thing", and those need opposite fixes.

Across all eight, defect signal beats lighting shift roughly 3:1, so distribution shift
is **not** AD 2's general problem despite being `can`'s entire problem.

### Resolution is not the lever on AD 2

Measured on `can`: 224px -> AU-PRO 0.0056, 448px -> 0.0111, 768px -> 0.0108. It doubles
once and then plateaus, while pixel AUROC falls to 0.43 at 768. Use **448** unless there
is a reason not to.

(1024px produced no output; the launch piped stderr through `grep` and discarded the
traceback. Never filter a run's stderr. Not rerun - the trend was already unambiguous.)

### The scaling law that constrains everything

`patch_distances` cost is *test patches x bank size*, and both grow with patches per
image. Doubling input resolution quadruples patches and **sixteen-times** the work:

| input | grid | patches | cost vs 224 | `can` |
|---|---|---|---|---|
| 224 | 28x28 | 784 | 1x | 36 s (measured) |
| 448 | 56x56 | 3,136 | 16x | ~10 min |
| 768 | 96x96 | 9,216 | 138x | ~1.4 h |
| 1024 | 128x128 | 16,384 | 437x | ~4.4 h |

Native resolution is therefore **infeasible**, not merely expensive - and note that the
published AD 2 improvement (8.87% -> 76.35% AU-PRO) came from **multi-scale layer2+layer3
fusion**, which this project already does, not from raw resolution.

**The mitigation worth trying:** cap the bank at a fixed size (~4,000 vectors) instead of
a fixed 1%. Cost then scales linearly rather than quadratically, and 768px drops to ~6
min per scenario. Our own coreset experiment supports this - cost varied only 7% across a
125x bank-size range.

### Decode bottleneck experiment: Pre-resizing vs. faster CPU decoders

The project identified single-core PIL PNG decompression as a major bottleneck (leaving the GPU idle 85% of the time). While rewriting PNG decompression in CUDA sounds tempting, it is an architectural anti-pattern:
- **Why custom GPU PNG decompression is the wrong path:** DEFLATE (LZ77 back-references + Huffman) and inverse scanline filtering (Paeth/Sub) have tight sequential data dependencies that fight SIMT lockstep execution, causing severe warp divergence and cache thrashing. Writing a custom GPU PNG engine is a 4–6 month systems engineering detour that rarely beats multi-core CPU SIMD for single images (even NVIDIA DALI historically kept PNG decode on the host CPU).

Two high-yield experiments to eliminate the decode bottleneck:

1. **Experiment A: Offline pre-resizing / caching (Highest ROI):**
   AD 2 source images are ~2448x2048 (~5 MP), but backbones resize them down to 224px or 448px immediately. Decompressing 5 MP only to discard 96% of the pixels on every evaluation run is wasted work. Pre-resizing the dataset once to 448px (or caching directly as `.pt` tensors, memmapped `.npy`, or WebP) cuts decode time by 20–50x and makes the pipeline fully GPU-bound.
2. **Experiment B: Drop-in faster CPU decoders (`libdeflate`, `pillow-simd`, `pyvips`):**
   Standard PIL uses unvectorized C and stock `zlib`. If raw dataset images must stay untouched on disk, benchmark swapping PIL's backend for `libdeflate` (2–3x faster decompression via AVX2), `pyvips`, `OpenCV` (`cv2.imread`), or `torchvision.io.decode_image` alongside the `ThreadPoolExecutor`.

---

## 7. What to do next, in order

1. **Scale-conditioned post-processing:** Implement an adaptive morphological filter that skips dilation/closing for scenarios dominated by micro-defects (`sheet_metal`), recovering the baseline 0.034 AU-PRO@5% while preserving fabric's 0.059.
2. **Experiment: Eliminate the AD 2 decode bottleneck (Pre-resize vs. `libdeflate`/`pyvips`).**
   Compare offline downscaling to 448px (or caching `.pt` tensors) against drop-in
   CPU decoders (`libdeflate`/`pyvips`) to remove the ~2448x2048 PNG decode tax.
3. **Phase C - Live DeepStream Video Pipeline.** Triton model repository is verified (6.34 ms latency, PID 15406). The next step is a complete GStreamer pipeline using `nvinferserver` or a `pyds` probe to process multi-camera RTSP/video feeds in real-time.

---

## 8. Working rules that have paid off

- **Change one variable at a time.** Session 2 concluded DINOv2 was far worse; it had
  changed backbone *and* grid resolution at once. At matched resolution the result
  reversed. The same confound nearly recurred with DINOv3 and with a GPU swap.
- **Run it more than once.** A single coreset seed put four arms in the wrong order and
  made one look 32% better than it was.
- **Report per class, never the mean alone.** It has hidden the real result three times.
- **Check the trivial baseline.** "Scrap everything" beating the model was discovered
  late and invalidated a whole metric.
- **Write predictions down before running.** One was flatly wrong (DINOv2 would win on
  `screw`) and the wrongness was more informative than a confirmation.
- **Diagnose from the artifact, not the guess.** Read the traceback, the printed numbers,
  the actual output. Do not pre-build defences against failures that have not happened.

Every correction above is in the git log with its reasoning. That history is the most
valuable thing in this repo - do not squash it.
