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
| Threshold percentile | p99 is **15x** worse than a single global p50 at 100:1 cost |

### Known-broken or void

- ~~Every AD 2 number produced before 2026-09-04 is void.~~ **Resolved** - AD 2 is now
  measured across all 8 scenarios: **mean AU-PRO@5% = 0.131**, above the published
  baseline of 0.0887. See §6.
- **The cost metric is measured on an inverted class balance.** MVTec's test set is
  **73% defective**; a real line is under 2%. At 100:1 escape:false-alarm, "scrap every
  part unexamined" costs **467** and beats our best detector's **559**. So absolute cost
  numbers are not meaningful, and the "p50 beats p99" result is partly an artefact of
  the same inversion. AUROC, data-efficiency curves, the resolution knee and the
  reproducibility findings are all unaffected - they are threshold-free or
  balance-free.
- **`outputs/pre-L40/` is not comparable to current results.** torch moved 2.13 -> 2.14
  and the driver 580 -> 570 mid-project. Same GPU model, same seeds, yet arm A's `grid`
  AUROC shifted 0.9507 -> 0.9607 and its escapes 9 -> 5. Small against the findings,
  larger than the 3.6% margin two experiments were spent narrowing.

### Immediate next step

**Diagnose `sheet_metal`.** It has strong image-level signal (1.9 sigma, AUROC 0.70) and
almost no localisation (AU-PRO 0.034), with **1,539 defect regions** - far more than any
other scenario and mostly tiny. The model knows the part is bad and cannot say where.
That is the clearest open lead in the project, and it is the same
ranking-is-not-deciding theme arriving between image and pixel level.

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
| `aupro.py` | **pixel metrics, numpy+scipy only** - no torch, so it is unit-testable | - |
| `test_aupro.py` | known-answer tests for the metric. Run it after any change | seconds, local |
| `ad2_pixel_eval.py` | AD 2 pixel-level evaluation | see §6 |

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
`sweep_backbones.py` has the fix; **`data_efficiency.py` still has its own un-patched
copy of `PatchExtractor`** and will be ~1.6x slower until someone ports it.

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

Full run, arm A at 448px with a 4,000-vector bank cap:

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

---

## 7. What to do next, in order

1. **Diagnose `sheet_metal`** - detectable but not localisable, 1539 tiny regions. Look
   at its anomaly maps against ground truth before theorising.
2. **A second arm on AD 2.** Everything so far is arm A. DINOv2 @448 (arm G) won on AD 1;
   does that survive on a harder dataset?
3. **Port the threaded decode into `data_efficiency.py`** - it still has its own
   un-patched `PatchExtractor` and runs ~1.6x slower than it needs to.
6. **Re-weight cost by a realistic defect rate** (~1%) rather than MVTec's 73%. This is
   the open item that would make every cost number in the repo meaningful. At 1% the
   weighting roughly inverts and the optimal threshold should swing back toward a high
   percentile - which would partly vindicate the original p99.
7. **Phase C - DeepStream deployment.** PatchCore is not a single ONNX: the feature
   extractor and pooling export cleanly, but the memory-bank kNN is not a standard op.
   Either host the whole thing in a Triton Python backend behind `nvinferserver`, or run
   `nvinfer` for features with the bank lookup in a `pyds` probe. That "my model does not
   fit the happy path" problem is what makes it a worthwhile portfolio piece.

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
