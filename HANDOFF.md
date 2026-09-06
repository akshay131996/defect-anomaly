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

## 0. Working agreement — planner/auditor and worker

**`THINKING_PROCESS.md` explains how the planner reasons** — the judgement calls between the rules
below, each tied to the episode in this project that produced it. Read it once; it is what lets you
predict what will be accepted, and argue with it on the same terms.

**Day-to-day coordination happens in `BRIDGE.md`, not here.** That file is the mailbox:
the current directive, worker status, audit verdicts, open questions, and an append-only
log. **This file is the specification** — what the experiments are and why. Read BRIDGE
first for what to do now; read HANDOFF for what the thing you are doing actually means.

Keep them separate. A decision that belongs to the project goes in HANDOFF; the bridge
entry that produced it points at HANDOFF and stops. Anything recorded only in BRIDGE is
assumed lost once the log is archived.

Two agents share this project. **The planner/auditor writes §7's experiment queue and
validates what comes back. The worker executes the queue on the pod and reports.** The
split exists because the expensive failures in this project have not been coding errors —
they have been *wrong conclusions drawn from real numbers*. Section 4 lists eight of them.
Separating who runs the experiment from who interprets it is the cheapest guard available.

### What the worker does

Take the next unblocked item from §7. Run exactly what it specifies. Report. Stop.

**Do not interpret, and do not repair a disappointing result.** If a run contradicts its
own stated hypothesis, that is the deliverable — write it down and move to the next item.
Three hypotheses about the AU-PRO gap were killed that way and the fourth was correct;
none of that would have surfaced if a disappointing number had been quietly tuned away.

### Rules that are not negotiable

Each of these is here because breaking it has already cost this project time.

1. **One variable per run.** Session 2 concluded DINOv2 was far worse; it had changed
   backbone *and* resolution at once. At matched resolution the result reversed.
2. **Never filter a run's stderr.** A 1024px run produced no output at all because its
   launch piped through `grep`, discarding the traceback. Pipe to a file, then read it.
3. **Never write to a fixed output path.** `ad2_pixel_eval.py` hardcodes
   `outputs/ad2_pixel_eval.json` and silently destroyed a baseline mid-investigation on
   2026-09-04. Use `--out outputs/runs/<run_id>.json`; add the flag if it is missing.
4. **Report all eight scenarios, never the mean alone.** The mean has hidden the real
   result three times. A mean without its per-scenario table will be rejected unread.
5. **Never select or tune on `test_public`.** AD 2 ships a `validation` split for this.
   Anything fitted on `test_public` is not a benchmark number and cannot be published.
6. **Never `pkill -f` / `pgrep -f`.** The pattern matches the ssh command carrying it and
   kills your own shell — this happened twice. Use `pgrep -x`, or match `/proc/PID/cmdline`.
7. **Write live logs to `/tmp`, not `/workspace`.** MooseFS blocks on a live-appended file
   and hangs the session with no error.
8. **Report deviations explicitly.** If you changed anything the item did not ask for —
   a flag, a default, a file — say so in the report. An unreported deviation makes every
   number in the run unusable, because the audit cannot tell what produced them.

### Engineering standards

The bar any organisation would expect, stated as checks rather than aspirations. Each one is here
because it is either already violated or was nearly violated in this repo.

**Every committed result must be reproducible from tracked code.** Verified 2026-09-06 and
currently **true** — every `outputs/runs/*.json` names its generating script in `command`, and all
of those (`ad2_pixel_eval.py`, `sweep_backbones.py`, `aupro.py`, `exp_e5a_region_sizes.py`) are
tracked with their `code_sha256` recorded. Do not let this lapse: a result whose generator is not
in the repo is not a result.

**Tests pass before you commit.** `test_aupro.py`, `test_registration.py` and `test_prealloc.py`
are cheap, run without a GPU, and cover the two metric bugs and one refactor that have already
cost this project real time. Run them; both currently pass.

**No code in limbo.** Anything in the repo root is either tracked or deleted — not left untracked
indefinitely. `exp_e5a_bucketed_pro.py` sat untracked in the root while a superseded copy of its
output sat beside it; that is how a reader ends up unable to tell which script produced which
number.

**Load-bearing code does not live in `scratch/`.** `scratch/build_aspect_cache.py` built the cache
that E10a and E10b depend on. A directory named `scratch` tells the next reader its contents are
disposable, and results now depend on it. **Move it to the root when no run is using it** — not
mid-run, which is why this is a note rather than a completed change.

**Docs match artifacts, or say they do not.** `README.md` still leads with the fusion results that
REVIEW.md §5 shows were measured in the broken crop frame, and still carries an arm-E cost table
while the project runs arm A. A README that contradicts the repo's own results is worse than no
README, because it is the first thing an outside reader trusts.

**Commit messages say what changed and why, with the evidence.** Not "update eval" — the reason
and the number. The git log is the only place the reasoning behind a change survives after the
bridge is archived.

### The output contract

Per run, three artifacts, committed together:

- `outputs/runs/<run_id>.json` — the record below. It already carries `started_utc`; **every
  BRIDGE entry, worker-status update, active directive and LEDGER row now carries a timestamp
  too** (BRIDGE protocol rule 9). Ids have collided nine times between the two agents — the id is
  a label, the timestamp is the ordering key.
- `logs/<run_id>.log` — complete unfiltered stdout+stderr.
- one row appended to `outputs/LEDGER.md`.

```json
{
  "run_id": "E3-aspect-448",
  "hypothesis": "one sentence, copied from the queue item",
  "command": "verbatim command line, all flags",
  "code_sha256": {"ad2_pixel_eval.py": "5168721b...", "sweep_backbones.py": "..."},
  "started_utc": "2026-09-04T19:10:00Z",
  "wall_seconds": 1180,
  "env": {"gpu": "NVIDIA RTX 4000 Ada", "torch": "2.14.0+cu130", "driver": "580.159.04", "cuda": "13.0"},
  "config": {"img": 448, "bank_cap": 4000, "eval_side": 512,
             "gauss_sigma": 4.0, "coreset_ratio": 0.01, "geometry": "aspect"},
  "scenarios": {
    "can": {"image_auroc": 0.0, "pixel_auroc": 0.0, "au_pro@0.05": 0.0,
            "au_pro@0.3": 0.0, "n_regions": 0, "n_good": 0, "n_bad": 0,
            "bank_size": 0, "seconds": 0}
  },
  "mean_image_auroc": 0.0, "mean_pixel_auroc": 0.0,
  "mean_au_pro@0.05": 0.0, "mean_au_pro@0.3": 0.0,
  "deviations": []
}
```

**Provenance is a hash, not a commit sha.** `/workspace` on the pod is *not* a git
checkout — files get there by `scp`, so `git rev-parse` fails and a `commit` field would
be silently empty (it was, on E1, until the audit caught it). Record
`sha256sum` prefixes of every script the run actually executed. That is stronger than a
sha anyway: it pins the bytes that ran, not the bytes that were committed, and those have
already diverged on this project.

Ledger row: `| run_id | geometry | img | mean AU-PRO@5% | mean I-AUROC | code hash | verdict |`
where verdict is `supports` / `refutes` / `inconclusive` **against the item's own stated
hypothesis** — that is a factual call about the number, not an interpretation of what it
means for the project.

Commit message: `run <run_id>: <one line of what happened>`. Never `git add -A` — it
swept 8,700 lines of another agent's work into an unrelated commit once already.

### What the auditor checks

Stated here so the worker knows what will be verified, and can pre-empt a rejection:

- `n_good` / `n_bad` per scenario match previous runs on the same split. A silent change
  means the data selection moved and nothing is comparable.
- `config` echoes the flags actually present in `command`.
- `env` matches the run being compared against — **including `driver`**. torch 2.13 -> 2.14
  alone moved an AD 1 arm by 0.010 AUROC, and a driver change of 580 -> 570 moved it again,
  both larger than margins two experiments were spent narrowing. **`driver` was not captured
  at all until 2026-09-05**, so every AD 2 record before that is missing the field that would
  say whether it survives a pod re-provision. Treat `driver: null` as "unknown, assume not
  comparable across a re-provision", not as "fine".
- `au_pro@0.05` <= `au_pro@0.3` for every scenario. The metric is monotonic in its limit;
  a violation means the integration is broken, as it once was.
- Means recompute from the per-scenario values.
- `bank_size` <= the cap.
- `code_sha256` is present and non-empty for every script named in `command`.

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
| Coreset ratio | ~~a **stability** knob, not accuracy. 7% cost range over a 125x bank-size range~~ **RECOMPUTED AS 1.8x, NOT 7%** — and from an image-level AD 1 experiment containing no AU-PRO, so it says nothing about the pixel metric. See REVIEW.md §6. Reproducibility claim (4.5x at 25%) not re-checked. |
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

*updated 2026-09-06*

**Where we are.** On the six scenarios the 768 arm has finished, mean AU-PRO@5% is **0.406**,
against the published PatchCore's **0.276** and the best published method (EfficientAD) at
**0.308** on `test_private`. Resolution is still paying: 224 -> 448 gave +0.154, 448 -> 768 gives
**+0.072** so far — diminishing, but the largest lever we have. `E5b` (dilated layer3) adds
**+0.025** at 448 and is orthogonal to resolution.

Best results to date, all on `test_public`, driver 580:

| run | mean AU-PRO@5% |
|---|---|
| E4b (448 baseline) | 0.3444 |
| E5b (448 + dilated layer3) | **0.3692** |
| E5-768 (6 of 8 scenarios) | **0.406** |

**Order of work:**

1. **Let `E5-inputres-768` finish, then audit it.** Report the **bucketed table**, because it
   finally tests D-04 prediction 2 — the `ge_16x` bucket bounded at **< 0.03 over 448 -> 768**.
   That prediction has never been testable until now; the earlier attempt to settle it used
   224 -> 448 and was withdrawn. **Report the bucket deltas before the headline mean.**
2. **E5b + 768 together.** E5b is orthogonal to resolution and each is worth +0.025 and +0.072
   alone. **Do not assume they stack** — measure it. If they do, we are near 0.43. Watch memory:
   dilating layer3 at 768 raises layer3 compute fourfold at an already expensive resolution, so
   check headroom before launching and record peak RSS.
3. **Move model selection to the `validation` split.** `ad2_pixel_eval.py` loads it at line 82 and
   never uses it. Everything so far is selected *and* reported on `test_public`, which violates §0
   rule 5 and **blocks any external claim**. This is now the gating item, not a tidy-up.
4. **Prepare the `test_private` submission.** We have a result worth validating externally and the
   server is the only route to a comparable number. See `docs/MVTEC_AD2_IMPLEMENTATION_SPEC.md` §6.
5. **`rice` and `walnuts`** — still the only scenarios where we trail the published PatchCore, and
   nobody has looked at why. `ad2_shift_check.py` on those two is minutes.

**Open question worth your own read:** 768 still pays +0.072. Is 1024 worth its cost, or is the
curve flattening enough that E5b-style architectural arms are the better spend? You have the
timings; make the call and say why.

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

### The coordinate-frame bug — read before trusting any AU-PRO number above

**Found 2026-09-04. Every AD 2 pixel-level number in this section predates it and is
measured through a broken metric.** Image-level AD 1 results are unaffected.

The symptom was a 5.8x gap against the published AD 2 baseline that survived three
hypotheses. Our detector matched the paper on two of three metrics and collapsed on the
third:

| metric | ours | published | |
|---|---|---|---|
| image AUROC | 0.663 | 0.659 | matches |
| pixel AUROC | 0.733 | 0.763 | close |
| **AU-PRO@5%** | **0.131** | **0.764** | **5.8x gap** |

The cause is that the anomaly map and the ground-truth mask were in **different
coordinate frames**. timm's eval transform resizes the short side to `img/crop_pct` and
centre-crops — correct for ImageNet, where the subject is centred and the frame is roughly
square. AD 2 images are neither, and their aspect ratio differs per scenario:

| scenario | native | area the model actually saw |
|---|---|---|
| sheet_metal | 4224x1056 | **19.1%** |
| can | 2232x1024 | 35.1% |
| fruit_jelly | 2100x1520 | 55.4% |
| vial | 1400x1900 | 56.4% |
| fabric, rice, wallplugs, walnuts | 2448x2048 | 64.1% |

Meanwhile the masks were squashed **full-frame** to `EVAL_SIDE`. So the map covered a
centre sub-rectangle stretched to a square, the mask covered the whole image stretched to
a square, and the offset differed per scenario.

**Why it hid for so long** — it is nearly invisible to the two metrics that looked healthy
and fatal to the one that did not:

- *image AUROC* is a max over patches. A defect anywhere in the visible region still
  scores high, so misregistration costs almost nothing.
- *pixel AUROC* is dominated by the overwhelming mass of correctly-scored normal pixels.
- *AU-PRO* scores per-region overlap. It is the **only one of the three that requires the
  map and mask to be spatially registered**, and it absorbed the entire error.

It is also resolution-invariant, because `crop_pct` is constant. That is why 768px and
1024px never recovered anything, and it is what made the earlier "resolution is not the
lever" conclusion look solid — **that conclusion is now void and must be re-measured.**

**Fixed, and measured across all 8** (`outputs/runs/E1-squash-448.json`, geometry
`--squash`: resize straight to `(img,img)` so map and mask share one frame; everything
else identical to the baseline):

| scenario | baseline AU-PRO@5% | squash | |
|---|---|---|---|
| vial | 0.4364 | **0.7386** | 1.7x |
| fruit_jelly | 0.2258 | **0.4583** | 2.0x |
| walnuts | 0.1120 | **0.3737** | 3.3x |
| wallplugs | 0.1241 | **0.2951** | 2.4x |
| rice | 0.0967 | **0.1863** | 1.9x |
| sheet_metal | 0.0345 | **0.1346** | 3.9x |
| fabric | 0.0047 | **0.1334** | 28x |
| can | 0.0111 | **0.0846** | 7.6x |
| **mean** | **0.1306** | **0.3006** | **2.3x** |

Improved 8 of 8. Against the published AD 2 baseline:

| metric | before | after | published | |
|---|---|---|---|---|
| image AUROC | 0.663 | **0.718** | 0.659 | now above |
| pixel AUROC | 0.733 | **0.846** | 0.763 | now above |
| AU-PRO@5% | 0.131 | **0.301** | 0.764 | 2.3x closer, still 2.5x short |

Two things follow. **The 5.8x gap was mostly, but not entirely, this bug** — a real
2.5x remains and needs its own explanation; do not treat AD 2 as solved. And **the
"all-time records" in the fusion table below (image 0.691, pixel 0.770) are superseded by
plain arm A with correct geometry** (0.718, 0.846), so that claim needs correcting
wherever it appears, including the README.

`vial` at 0.7386 is on its own within 3% of the published *mean*, which is the strongest
single piece of evidence that the remaining gap is not a modelling deficiency.

**Caveat:** squashing distorts aspect ratio, visible in the two most extreme frames —
`can` (2.18:1) fell to 0.467 image AUROC and `fabric` to 0.640, the only two scenarios
where image AUROC got *worse*. Squash buys registration and pays in distortion, so two
alternatives were tested.

### Geometry comparison — settled: aspect-preserving wins

The first comparison was invalid. `evaluate` derived regions by labelling the mask *after*
resizing it into each geometry's own evaluation frame, then dropped components under
`MIN_REGION_PX = 4`. Each geometry crushed a different subset below the threshold and
scored a **different population of ground-truth regions** — `sheet_metal` alone came out at
1539 / 426 / 1101 across the three. Dropping small regions makes the test *easier*, because
AU-PRO weights every region equally and the small ones are the hard ones.

**E4a fixed it** by labelling regions once at native mask resolution with a 77-native-pixel
floor, carrying the label map into each evaluation frame, and scoring a region that is
erased by downsampling as PRO = 0 rather than dropping it. Region counts are now
**bit-identical across geometries** (1,530 total: 66 / 120 / 216 / 114 / 444 / 168 / 90 /
312). Only then are the geometries comparable:

| run | geometry | mean AU-PRO@5% | mean I-AUROC |
|---|---|---|---|
| (baseline) | crop | 0.1306 | 0.663 |
| E2R | letterbox | 0.2932 | 0.670 |
| E1R | squash | 0.3139 | 0.697 |
| **E3R** | **aspect-preserving** | **0.3429** | **0.724** |

**Aspect wins on both axes and the margin is real.** +0.0290 over squash, above the 0.01
equivalence threshold. `sheet_metal` accounts for 55% of it and aspect wins 5 of the other
7 scenarios, so it is not a single-scenario effect.

**Letterbox lost on both axes.** It was the intuitive fix — registration without distortion
— and it came last. The likely mechanism is in §7 E2: padding is perfectly uniform, sits
far off the normal manifold, and `avg_pool2d(k=3)` bleeds it into genuine border patches.
Excluding padded pixels from the metric does not undo that.

**Worth recording, because the prediction was wrong.** The planner predicted that fixing
the region set would shrink aspect's margin below 0.01 and leave it tied with squash. The
opposite happened: the margin *grew* from +0.0219 to +0.0290, `sheet_metal`'s share fell
from 81% to 55%, and aspect went from losing 3 of 7 non-`sheet_metal` scenarios to losing
2. **The confound was masking a real effect, not manufacturing one.** A confound is a reason
to distrust a number in either direction, not a reason to assume the number is inflated.

**The guard that would have caught the original coordinate-frame bug** did not exist:
nothing asserted that a synthetic map at a known location scores high against a mask at the
same location. E0 now does, parameterised over geometry, with shifted cases so it cannot
pass trivially — `crop` scores 0.019 and `squash` 1.000. It does not yet cover `aspect`.

**The general lesson, now the same one three times.** The coordinate-frame bug, the region
set confound, and (pending) the evaluation-frame question are all *evaluation-protocol*
defects. Every one left image AUROC nearly untouched while moving AU-PRO substantially.
When a localisation number moves on this project, check what the metric is computed over
before concluding anything about the model.


### Evaluation resolution is not the lever — and why that matters

E4 swept the evaluation frame at {512, 1024, 2048} nominal under `aspect` geometry, with
Gaussian sigma scaled {4, 8, 16} to hold native-space blur constant. **Flat:**

| eval frame | mean AU-PRO@5% | mean I-AUROC | active regions |
|---|---|---|---|
| 512 | 0.3444 | 0.724 | 1530 / 1530 |
| 1024 | 0.3444 | 0.724 | 1530 / 1530 |
| 2048 | 0.3442 | 0.724 | 1530 / 1530 |

Total range 0.0002 across a 16x change in evaluation pixels. The 2048 arm puts
`sheet_metal` at 1024x4096, essentially its native 1056x4224, so this reaches the ceiling
of what the metric could ever see.

**The mechanism is confirmed, not inferred: 1,530 of 1,530 regions were already active at
512.** Nothing was being erased by downsampling, so there was nothing for a finer frame to
recover. The prediction that this would be so was recorded before the run.

**This closes the evaluation-protocol line.** Three protocol defects were proposed for the
AU-PRO gap: the coordinate frame (real, worth 2.3x), the region set (real, worth +0.03 and
it *raised* aspect's margin), and evaluation resolution (**refuted**). The remaining gap is
not an artifact of how we score.

### Where the remaining gap lives: the patch grid is coarser than the defects

At `img = 448` the grid is about 52x60 per image, so **one grid cell covers 756-1,607
native pixels**, while the smallest scored region is 77:

| scenario | grid (h,w) | native | px per cell | cell area | vs 77px floor |
|---|---|---|---|---|---|
| can | 36x84 | 2232x1024 | 27x28 | 756 | 10x |
| vial | 64x48 | 1400x1900 | 29x30 | 866 | 11x |
| fruit_jelly | 48x64 | 2100x1520 | 33x32 | 1039 | 13x |
| sheet_metal | 28x112 | 4224x1056 | 38x38 | 1422 | 18x |
| fabric, rice, wallplugs, walnuts | 52x60 | 2448x2048 | 41x39 | 1607 | 21x |

**The smallest scored regions are 10-20x smaller than a single patch cell.** A defect that
fits inside one cell cannot be localised at all: the anomaly map has no spatial detail below
cell size no matter how finely it is upsampled. That is exactly why E4 was flat, and it is
the most plausible remaining explanation for the 2.2x gap — the published baseline runs at
1024x1024 input, where cells fall to roughly 330 native pixels, about 5x closer to defect
scale.

**This is a hypothesis, not a conclusion.** It rests on the 77-pixel floor, which is the
*minimum* region size; whether it explains the gap depends on the distribution of region
sizes, which has never been measured. §7 E5a measures it in minutes with no GPU and decides
whether E5's hours are worth spending. Do not skip it — if regions of every size score
uniformly mediocre, resolution is not the answer and E5 would burn hours confirming that.


### Defect scale, measured — and what it does and does not establish

E5a measured the native area of all 1,530 regions and broke AU-PRO@5% down by size, at
`img = 448` under `aspect`:

| bucket | regions | share | AU-PRO@5% |
|---|---|---|---|
| sub-cell (< 1x) | 756 | 49.4% | **0.2265** |
| 1-4x cells | 354 | 23.1% | 0.4280 |
| 4-16x cells | 173 | 11.3% | 0.5498 |
| >= 16x cells | 247 | 16.1% | **0.6056** |

Monotonic **only when pooled across scenarios** -- within scenarios just 2 of 7 hold, and
`vial` is inverted (sub-cell 0.929 > `>= 16x` 0.625). The figure 0.7516 quoted here previously
for `vial` appears in no artifact; the real value is **0.6247**. See REVIEW.md §2-§3. Across scenarios, median region size in cell units tracks AU-PRO@5% at
**r = +0.788**: `vial` at 17.1x cell scores 0.719, while `can` and `fabric` — whose median
defect is about **a quarter of one patch cell** — score 0.139 and 0.189.

**What this establishes:** the low mean is not a detector that fails uniformly. Large
defects are already at or near published parity; the mean is dragged down by the half of
all regions that are smaller than a single patch cell.

**What it does not establish:** that resolution is the *cause*. Monotonic size-vs-score is
equally consistent with large defects simply being easier for any detector at any
resolution — more salient in feature space, and AU-PRO over a large region averages more
pixels so it is less noisy. E5a cannot separate the two. E5 does, by holding the bucket
edges fixed in native pixels across resolutions: a genuine ceiling lifts the sub-cell bucket
and leaves the largest one flat; salience moves them together.

Two scenarios already warn against the simple story. **`rice` has a median region 3.3x cell
area and still scores 0.226; `wallplugs` is 1.2x and scores 0.276.** Both have defects large
enough to resolve at 448 and score poorly anyway, so at least one failure mode here is not
resolution.

### The gap has two causes, and resolution is the smaller one

Regressing per-scenario AU-PRO@5% on each candidate factor: **image AUROC alone gives
R^2 0.744, defect scale alone 0.605, both 0.877.** Detection quality is the stronger
predictor.

**Four of eight scenarios are at or near chance on detection.** No evaluation protocol or
input resolution can localise a defect the representation does not separate at all:

| | image AUROC | median defect | AU-PRO@5% |
|---|---|---|---|
| can | 0.514 | 0.26 cells | 0.139 |
| wallplugs | 0.541 | 1.20 cells | 0.276 |
| fabric | 0.608 | 0.26 cells | 0.189 |
| rice | 0.638 | 3.32 cells | 0.226 |
| **mean of these four** | | | **0.208** |
| **the other four** | 0.79-0.96 | | **0.481** |

**Caveat that must travel with this:** image AUROC and AU-PRO derive from the same anomaly
scores, so image AUROC is a **co-symptom, not an independent cause**. It localises where the
gap lives — scenarios where the backbone barely separates defects — rather than proving what
creates it. Defect scale *is* measured independently of the scores, which is why the 2x2
below is still a valid design.

**The 2x2 E5 tests for free** (registered in BRIDGE M-09 before E5 ran):

| | small defects (< 1 cell) | large defects (> 1 cell) |
|---|---|---|
| **good detection** | `sheet_metal`, `fruit_jelly` — should gain most | `vial`, `walnuts` — modest, already resolved |
| **weak detection** | `can`, `fabric` — limited | `rice`, `wallplugs` — negative control, near-zero |

`sheet_metal` is the cleanest pure-resolution case in the benchmark: detection works
(0.791), defects are small (0.61 cells, 64.9% sub-cell), AU-PRO is poor (0.253). **If it
does not gain, the resolution hypothesis is in trouble regardless of what the mean does.**

**Consequence for the queue: E7 is now a second load-bearing experiment, not a tidy-up.**
Half the benchmark is limited by representation rather than geometry or scale, and DINOv2
previously lifted `fabric`'s pixel AUROC from 0.650 to 0.973 — on one of these four. That
work was done under broken geometry and selected on `test_public`, so it must be redone, but
it is aimed at the right half of the problem.

### ~~The ceiling on resolution~~ — UNTESTED, see REVIEW.md §1

> **Do not rely on this section, and do not treat it as refuted either.** Its stated
> condition — that higher resolution does not also lift the `>= 16x` bucket — has never
> been tested. Only the 768 arm can test it.
>
> An earlier correction claimed this was refuted by the 224 -> 448 data. **That claim was
> itself wrong** and is withdrawn: the bucket edges are pinned to the *448* cell area, and
> the cell at 224 is 4.06x larger, so a ">= 16x" region occupies only ~3.9x the cell at 224
> — it is not "already resolved" there, and its gain says nothing about what happens above
> 448.
>
> What IS established: resolution is a strong lever in the 224 -> 448 range, worth
> **+0.154 mean AU-PRO@5% (1.86x) on 6 of 6 scenarios**. That data
> (`outputs/runs/E5-inputres-224.json`) sat unread in the repo, filed as a partial run.


Regions at `>= 16x` cell area are already spatially unconstrained — sixteen or more patch
cells across, so a finer grid has little to add. **They score 0.6056 (n = 247).**

If resolution were perfect and every region scored like those already-resolved ones, the
mean would reach **~0.61 — still ~0.16 below the published 0.764.** That is arithmetic on a
bucket the current model already scores, not a projection about a resolution we have not run.

**So two independent levers are needed, and E7 must run regardless of how E5 turns out:**

1. **Resolution (E5)** — recovers the 49.4% of regions below one cell. Bounded at ~0.61.
2. **Representation (E7)** — must lift already-resolved regions from 0.606 toward 0.764.
   Nothing spatial can do this.

**Trust the global buckets, not the per-scenario ones.** Global counts are 756 / 354 / 173 /
247 and are sound. Per-scenario large buckets are not: `fabric` has exactly **6 regions** in
each of its top two buckets (yielding a meaningless 0.9949 and 0.1108), and `rice`,
`wallplugs` and `sheet_metal` have 12-18. A macro ceiling computed from those (0.625)
inherits the noise; use 0.606.

**The argument has one condition:** it assumes higher resolution does not also lift the
`>= 16x` bucket. That is D-04's registered prediction 2. If `ge_16x` rises substantially
instead, resolution is helping by some mechanism other than resolving small defects, the
ceiling argument weakens, and the "spatial ceiling" framing is wrong even though the numbers
moved the right way.

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
min per scenario. ~~Our own coreset experiment supports this - cost varied only 7% across a
125x bank-size range.~~ **That 7% is one cell of a 3x5 grid; at the best operating point per
ratio the range is 1.8x (11900 / 8061 / 6644 / 6619), and the experiment is image-level AD 1
with no AU-PRO in it. See REVIEW.md §6.** Note also that the whole table above overstates
measured cost by 5.2x (REVIEW.md §8, P-3).

### Decode bottleneck experiment: Pre-resizing vs. faster CPU decoders

The project identified single-core PIL PNG decompression as a major bottleneck (leaving the GPU idle 85% of the time). While rewriting PNG decompression in CUDA sounds tempting, it is an architectural anti-pattern:
- **Why custom GPU PNG decompression is the wrong path:** DEFLATE (LZ77 back-references + Huffman) and inverse scanline filtering (Paeth/Sub) have tight sequential data dependencies that fight SIMT lockstep execution, causing severe warp divergence and cache thrashing. Writing a custom GPU PNG engine is a 4–6 month systems engineering detour that rarely beats multi-core CPU SIMD for single images (even NVIDIA DALI historically kept PNG decode on the host CPU).

Two high-yield experiments to eliminate the decode bottleneck:

1. **Experiment A: Offline pre-resizing / caching (Highest ROI):**
   AD 2 source images are ~2448x2048 (~5 MP), but backbones resize them down to 224px or 448px immediately. Decompressing 5 MP only to discard 96% of the pixels on every evaluation run is wasted work. Pre-resizing the dataset once to 448px (or caching directly as `.pt` tensors, memmapped `.npy`, or WebP) cuts decode time by 20–50x and makes the pipeline fully GPU-bound.
2. **Experiment B: Drop-in faster CPU decoders (`libdeflate`, `pillow-simd`, `pyvips`):**
   Standard PIL uses unvectorized C and stock `zlib`. If raw dataset images must stay untouched on disk, benchmark swapping PIL's backend for `libdeflate` (2–3x faster decompression via AVX2), `pyvips`, `OpenCV` (`cv2.imread`), or `torchvision.io.decode_image` alongside the `ThreadPoolExecutor`.

---

## 7. Experiment queue

Worker: take the next unblocked item, follow §0's output contract, report, stop. Items are
ordered by what unblocks the most downstream work, not by expected payoff.

Every item states a hypothesis in a form that can be **refuted by the number it produces**.
A refuted hypothesis is a completed item, not a failed one — three of the four proposed
about the AU-PRO gap were refuted, and that is how the fourth was found.

| id | what | blocked on | GPU | cost | order |
|---|---|---|---|---|---|
| ~~E0-E5a~~ | geometry, region set, eval protocol, region scale | — | — | **done** — see §6 | — |
| **rebuild** | `deployment/POD_REBUILD.md` | — | no | ~20 min | **1** |
| **E4b** | clean 448 `aspect` baseline on driver 580 | rebuild | yes | ~13 min | **2** |
| **E9** | pre-resize cache + `setsid` launches | — | no | ~0.3 h eng | **3** |
| **E5b** | dilated layer3 `output_stride=8` @448 | E4b | yes | ~0.4 GPU-h | **4** |
| **E8** | rebuild the serving path (not a bank swap) | — | yes | ~0.5 h | **5 / filler** |
| **table** | commit the paper's Table VII per scenario/split/limit | — | no | minutes | **any time** |
| E10a | reduce embedding 1536 -> 384 — **enabler** | E4b | yes | ~0.4 GPU-h | 6 |
| E10b | backbone ensemble, **one at a time** | E10a | yes | ~1.5 GPU-h | 7 |
| E5 | input resolution 768 arm | E10a (for cost) | yes | ~0.5-1.9 GPU-h | 8 |
| E7 | fusion/routing re-run under `aspect`, select on `validation` | E4b | yes | ~1-2 h | 9 |
| E5c | unstrided layer1 pyramid (stride 4) | E5, E5b | yes | ~2 GPU-h | 10 |
| E6 | coreset density, re-checked | — | yes | ~1 h | last |
| E10c | coreset ratio — **probably a units artifact, read the table first** | table | — | — | skip |

**Items 1-5 are D-06, the active directive**, chosen to be independent of E10 so nothing in them
gets invalidated when E10 lands. **E10a is the enabler**: the 1536 -> 384 reduction is a 4x cut in
bank memory and distance cost, which is what makes E5's 768 arm affordable, so it comes before E5
rather than after.

**E5a is the one to start with** — no GPU, minutes, and it decides whether E5 is worth hours. E8 is independent of everything and can run in
parallel or first if the pod is otherwise occupied.

The chain E2 -> E3 -> E4 exists because each fixes the configuration the next one varies
against. Running them out of order does not just weaken the result, it makes the
comparisons meaningless — E4's sweep is only interpretable once one geometry has won.

**What "the winning geometry" means:** the one with the highest **mean AU-PRO@5%** across
all 8 scenarios, with mean image AUROC as the tiebreaker if two are within 0.01. Decide it
**from E1R/E2R/E3R** — the E4a re-scores, not the original E1/E2/E3, which were each scored
against a different region population and cannot be compared. State the choice in E4's
record so every later run is anchored to something written down rather than a recollection.

If the re-scored geometries land within 0.01 of each other, **say they are equivalent and
take the cheapest**, rather than picking a nominal winner and building four experiments on
a difference that is noise.

---

### E0 — registration unit test  **DONE** — supports (crop 0.019 / squash 1.000)

**Why first:** the coordinate-frame bug survived four sessions because no test asserted
that a map and a mask describe the same place. Until this exists, every item below can
regress silently in the same way.

Add `test_registration.py` alongside `test_aupro.py`, numpy-only, no GPU:

- Build a synthetic mask with one filled rectangle at a known normalised location, in a
  deliberately non-square frame (use 4224x1056, `sheet_metal`'s shape, where the old crop
  kept only 19% of the width).
- Build a "perfect" anomaly map that is high exactly on that rectangle, pushed through
  the **same geometry the evaluator uses**.
- Assert `au_pro@0.05 > 0.95`.
- Repeat with the map offset by 20% of the width; assert it collapses (`< 0.2`). Without
  this second case the test passes trivially for a detector that flags everything.

**Parameterise it over geometry**, not just over the current default. Write it so a new
geometry is one line to add, and it becomes the standing regression guard for E2 and E3
rather than a one-off. `crop` is expected to fail; `squash` is expected to pass. Do not
delete the failing case once E2 lands — a test that only ever passes proves nothing.

Also place the rectangle **near the frame edge** in at least one case. A centred defect
survives a centre-crop and would have hidden this bug just as thoroughly as no test at all.

**Hypothesis:** the old default geometry (`resize+centre-crop` image, full-frame mask)
fails the first assertion. **Verdict `supports` = it fails on `crop` and passes on
`squash`.** Commit the failing-then-passing pair so the diff shows the bug being caught.

**Deliverable:** the test file, plus `logs/E0.log` showing both outcomes.

---

### E1 — squash geometry, all 8 scenarios ~~*(pending)*~~ **DONE 2026-09-04**

Result in §6: mean AU-PRO@5% 0.1306 -> **0.3006**, improved 8/8. Record at
`outputs/runs/E1-squash-448.json`, log at `logs/E1-squash-448.log`.
**Hypothesis (>2x) supported.** Mean image AUROC also rose, 0.663 -> 0.718, though `can`
and `fabric` individually fell — the aspect-distortion cost E2/E3 exist to remove.

*Deviation on record:* the first process was killed after 5 scenarios with no traceback
and no cgroup OOM; cause not established. The remaining 3 ran separately with identical
flags and identical script hashes. If a long AD 2 run dies silently again, capture it — this is currently
an unexplained failure, not a known one.

---

### E2 — letterbox geometry, all 8  **DONE** — **refutes** (0.2792 vs squash 0.3006; I-AUROC 0.670 vs 0.718)

**The idea.** Resize the **longest** side to `img` preserving aspect, pad the short side
to square, and apply *identical* letterboxing to the masks. Registration is preserved (as
in E1) but nothing is stretched, so the aspect distortion that cost `can` and `fabric`
their image AUROC in E1 goes away.

**Hypothesis:** letterbox beats squash on mean image AUROC, and matches it within 0.02 on
mean AU-PRO@5% — registration is preserved either way, only distortion differs.

This one needs code, not just a flag. Three changes, all specified here so you do not have
to invent an API:

**(a) `--geometry {crop,squash,letterbox}` in `ad2_pixel_eval.py`**, replacing the boolean
`--squash`. Keep `--squash` as a hidden alias for `--geometry squash` so E1's recorded
command still reproduces — do not silently change what an existing run record means.

**(b) Pad with the normalisation mean** (i.e. zero *after* `Normalize`), which is the
conventional letterbox. Two things to watch and report on:

- Padded regions are perfectly uniform, so they sit far off the normal manifold and will
  likely score as strongly anomalous. Excluding them from the metric (below) handles the
  score, but **not** the `avg_pool2d(k=3)` neighbourhood aggregation in
  `PatchExtractor.forward_feats`, which will bleed padding into genuine border patches.
- If E2 underperforms, that bleed is the first suspect. The diagnostic is to report
  AU-PRO with a 1-patch border of real image also excluded; if the number jumps, it is
  contamination and not geometry.

**(c) `evaluate(..., valid=None)` in `aupro.py`.** Padded pixels are not image, and
counting them inflates the normal-pixel mass that AU-PRO's false-positive axis is
normalised by — which would flatter E2 for a spurious reason. `valid` is a boolean array
broadcastable to a map, True on real-image pixels. Apply it at all three accumulation
sites:

```python
for m in maps_good:
    h_norm += hist(m[valid].ravel())          # was m.ravel()

for m, mask in zip(maps_bad, masks):
    if mask.any():
        h_norm += hist(m[valid & ~mask].ravel())
        h_anom += hist(m[valid & mask].ravel())
    else:
        h_norm += hist(m[valid].ravel())
```

Regions need no change — the masks are letterboxed identically, so padding is never
labelled defective. **Assert that** rather than assuming it: `assert not (mask & ~valid).any()`.
Also compute the histogram range (`allv` in `ad2_pixel_eval.main`) over valid pixels only,
or the padding's extreme scores will stretch the bins and cost you resolution everywhere else.

Default `valid=None` must behave exactly as today, so E1 and the existing tests are
unaffected. **Extend `test_aupro.py` to cover it**: a letterboxed perfect detector with
padding excluded should still score ~1.0, and should *not* if `valid` is ignored.

### E3 — aspect-preserving rectangular input, all 8  **DONE** — verdict **downgraded to inconclusive**, see E4a

**The idea.** The cleanest geometry available: feed a **non-square** input preserving the
native aspect, rounded to a multiple of the backbone stride (32 for WideResNet50-2).
`sheet_metal` 4224x1056 -> 896x224; `vial` 1400x1900 -> 352x480. CNNs are fully
convolutional, so this needs no architectural change. Resize masks to the same rectangle.

No distortion, no padding, full frame visible, map and mask registered — it is the only
option that has none of the three defects.

**Hypothesis:** E3 >= max(E1, E2) on mean AU-PRO@5% **and** on mean image AUROC.

Implementation notes:

- Adds `--geometry aspect` to E2's flag.
- `PatchExtractor.grid` is inferred per forward pass and already handles non-square
  feature maps — but `ad2_pixel_eval.anomaly_maps` takes `grid` as a tuple and reshapes
  with `d.view(-1, 1, grid[0], grid[1])`. Verify that ordering is right for a non-square
  grid; a transposed reshape here would silently produce a rotated map, which is exactly
  the class of bug §6 is about. **E0 must pass on `aspect` before you trust any number
  from this run.**
- **Hold total patches roughly constant against E1** when picking each rectangle, or the
  comparison confounds geometry with resolution — §6's scaling law applies. E1 at 448
  square is 3,136 patches; target that, so `sheet_metal` at 896x224 (784 patches) is *not*
  matched and should be scaled up to roughly 1792x448.

**ViT arms cannot do this** without interpolating position embeddings. Restrict E3 to the
CNN arm and note the limitation rather than working around it.

**Blocked on:** E2 (it is a direct comparison).

---

### E4a — fix the region set before comparing geometries  **DONE** — supports; region sets now bit-identical (1530), and it *raised* aspect's margin

**Audit finding, 2026-09-04. E3's "supports" verdict does not hold up and the geometry
winner cannot be declared until this is fixed.**

E1, E2 and E3 were each scored against a **different set of ground-truth regions**, so they
are not comparable. `evaluate` derives regions with `ndimage.label(mask)` on the mask
*after* it has been resized into that geometry's evaluation frame, and then drops anything
below `MIN_REGION_PX = 4`. Each geometry therefore resizes the masks differently, crushes a
different subset below the threshold, and scores a different population:

| scenario | E1 regions | E2 | E3 | E3-E1 AU-PRO |
|---|---|---|---|---|
| sheet_metal | 1539 | 426 | 1101 | **+0.1421** |
| fruit_jelly | 320 | 244 | 252 | -0.0380 |
| walnuts | 450 | 432 | 432 | +0.0294 |
| fabric | 150 | 120 | 114 | +0.0240 |
| vial | 174 | 175 | 174 | -0.0386 |

`sheet_metal` is the extreme case: E2 scored **28% of the regions E1 did**, and E3 dropped
438 of E1's 1,539. Dropping small regions makes the test *easier*, because AU-PRO weights
every region equally and the small ones are the hard ones.

Decomposing E3's +0.0219 margin over E1:

- **sheet_metal alone contributes +0.0178 — 81% of the entire margin**, and it is also the
  scenario whose region count moved most.
- Over the other seven scenarios E3 beats E1 by **+0.0047** and **loses on three of seven**.

That is not a geometry result. It is consistent with E3 winning because it scored an easier
region population, and the current evidence cannot separate the two.

**The fix: define the region set once, at native mask resolution, independent of geometry.**

- Run `ndimage.label` on the **native** mask, before any resize.
- Apply `MIN_REGION_PX` there, in native pixels, so the same regions survive for every
  geometry. Pick the native threshold that reproduces today's intent (~77 native px, i.e.
  4 px at 512 on a 2448x2048 frame) and state the number chosen in the record.
- Carry each surviving component's label map into the evaluation frame with NEAREST
  resize, so a region keeps its identity even if it shrinks.
- **Assert `n_regions` is now identical across geometries for a given scenario.** That
  assertion is the deliverable — it is what makes E1/E2/E3 comparable, and it is cheap to
  check.

Then **re-score E1, E2 and E3** under the fixed region set (`--geometry` is the only
variable) and record them as `E1R`, `E2R`, `E3R`. The re-scored numbers decide the winner.

**Hypothesis:** under a fixed region set, E3's margin over E1 falls below 0.01 mean
AU-PRO@5% and the two are within noise.

If that is what happens, **say so** — it means aspect-preserving geometry is not clearly
better than squash, and the honest conclusion is that E1 and E3 are equivalent and the
cheaper one wins. Do not go looking for a configuration that restores E3's lead.

**Also fold in two smaller audit items while you are here:**

1. **E0 does not cover `aspect`.** E3's spec required it to, precisely because a
   transposed reshape on a non-square grid would silently rotate the map. I checked the
   implementation by hand and it is correct — `ex.grid` is `(h, w)` from `fmap.shape`,
   `aspect_transform` is applied before `extract_paths` so the grid is re-derived, and
   `aspect_dimensions` does hold patch area constant at 448^2. But hand-checking is not a
   test. Add `aspect` to `test_registration.py`.
2. **Record `grid` and `n_patches` per scenario** in every run record from now on. The
   constant-patch-count requirement in E3 could not be verified from its record; it had to
   be re-derived from source, which defeats the point of the record.
3. `logs/E0.log` is UTF-16 (a PowerShell redirect artifact). Write logs as UTF-8.

---

### E4 — evaluation protocol: `EVAL_SIDE` and smoothing scale  **DONE** — verdict **refutes** (AU-PRO@5% flat at 0.3444 -> 0.3442; 1530/1530 active regions; input resolution ceiling binds)

**Sweep results across EVAL_SIDE in {512, 1024, 2048} with proportional $\sigma \in \{4.0, 8.0, 16.0\}$:**

| Arm | Mean AU-PRO@5% | Mean AU-PRO@30% | Pixel AUROC | Image AUROC | Active Regions | Peak RSS |
|---|---|---|---|---|---|---|
| `E4-evalside-512` | 0.3444 | 0.5736 | 0.8501 | 0.7236 | 1530 / 1530 | 17,715 MB |
| `E4-evalside-1024` | 0.3444 | 0.5723 | 0.8505 | 0.7236 | 1530 / 1530 | 18,098 MB |
| `E4-evalside-2048` | 0.3442 | 0.5723 | 0.8508 | 0.7236 | 1530 / 1530 | 19,480 MB |

**Key findings & auditor checks satisfied:**
- **M1/M2:** Region set held strictly constant at 1,530 native regions across all 3 arms.
- **M3:** 1,530 / 1,530 regions (100.0%) survive downsampling with $n_{px} > 0$ at `eval_side=512`. Zero regions are erased by downsampling, exactly confirming the auditor's prior.
- **M4:** Peak RSS remained below 19.5 GB throughout, safely within the 58 GiB cgroup limit.
- **Hypothesis verdict: `refutes`.** The metric is invariant to evaluation resolution ($\Delta \le 0.0002$).
- **Structural conclusion:** The anomaly map is upsampled from a ~56x56 patch grid fixed by `img=448`. The evaluation frame cannot hallucinate high-frequency boundary information absent in the backbone feature map. **The input resolution ceiling is binding** — this directly cues E5.

---

### E5a — region size distribution vs patch cell size  **<- NEXT (no GPU, minutes)**

**Run this before E5.** It costs no GPU and it predicts whether E5 can work at all. This is
the same move as `ad2_shift_check.py`, which correctly predicted the AU-PRO ordering before
any evaluation was run.

E4 established that the remaining gap is not an evaluation artifact. The next suspect is
the patch grid, and there is a concrete reason to suspect it. At `img = 448` the grid is
about 52x60 per image, so **one grid cell covers 756-1607 native pixels**:

| scenario | grid (h,w) | native | native px per cell | cell area | vs the 77px floor |
|---|---|---|---|---|---|
| can | 36x84 | 2232x1024 | 27x28 | 756 | 10x |
| vial | 64x48 | 1400x1900 | 29x30 | 866 | 11x |
| fruit_jelly | 48x64 | 2100x1520 | 33x32 | 1039 | 13x |
| sheet_metal | 28x112 | 4224x1056 | 38x38 | 1422 | 18x |
| fabric, rice, wallplugs, walnuts | 52x60 | 2448x2048 | 41x39 | 1607 | 21x |

The smallest scored regions are **10-20x smaller than a single patch cell**. A defect that
fits inside one cell cannot be localised — the anomaly map has no spatial detail below cell
size, regardless of how finely it is upsampled, which is precisely why E4 was flat.

**But the 77px floor is the minimum, not the typical region.** Whether this actually
explains the gap depends on the *distribution* of region sizes, which has never been
measured. Measure it:

- For each scenario, the native pixel area of all 1,530 ground-truth regions: median, IQR,
  and the fraction below 1x, 2x and 4x the local cell area.
- **Break E3R's AU-PRO@5% down by region-size bucket** (sub-cell, 1-4 cells, 4-16 cells,
  larger). The per-region PRO values already exist inside `evaluate`; expose them rather
  than recomputing.

**Hypothesis:** a majority of regions are sub-cell at `img = 448`, and AU-PRO on the
larger-than-4-cells bucket is already close to the published 0.764.

**This is the decisive read.** If large regions already score well and small ones score
near zero, the gap is a resolution ceiling and E5 will close much of it. **If regions of
every size score uniformly mediocre, resolution is not the answer** and E5 would burn hours
confirming that — in which case say so and stop, rather than running it because it is next
in the list.

---

### E9 — pre-resize cache and `setsid` launches  **(do first; engineering, not an experiment)**

Two changes that make everything after them cheaper and safer.

**Pre-resize the dataset once** to each scenario's aspect dimensions and cache it. AD 2 source
images are ~5 MP and every run decompresses them only to discard ~96% of the pixels; PNG decode
was measured as the historical bottleneck (GPU idle in 12 of 14 samples). Cache to the
**container disk**, not `/workspace` — MooseFS is slow with many small files.

**Verify before trusting it:** run one scenario from cache and one live-decode and require the
extracted features to be **bit-identical**. A silent resize difference would corrupt every
downstream experiment and would present as a real effect. This is the same class of check that
`test_prealloc.py` applies to the preallocation refactor.

**Switch every launch to `setsid nohup <cmd> > /tmp/<run>.log 2>&1 < /dev/null &`.** Three runs
(E1 at 5/8, E4-evalside-512 at 3/8, E5-inputres-224 at 6/8) died silently to `nohup` alone;
memory was never the cause — E5-224 died at 3.5 GB while completed runs used 18-19 GB. This
closes OQ-1. **The instructions at §3 still teach the broken pattern — fix them.**

---

### E5c — unstrided layer1 pyramid  **(after E5b and E5, not before)**

Layer 1 of WideResNet50-2 is stride 4, giving a 112x112 grid at 448 and cutting cell area 4x.
That would move the 354 regions currently in the 1-4x bucket into multi-cell territory.

**Why it is not first.** It quadruples patches (12,544 vs 3,136) — the configuration that already
triggered a container OOM above 37 GB — and it changes patch count, feature dimensionality and
cell area simultaneously. E5b buys most of the same spatial benefit for a fifth of the cost while
moving exactly one variable, and E5 establishes what input resolution alone is worth. Run those
two first and E5c becomes interpretable; run it now and it is another five-variable arm.

The memory mitigation proposed by the worker is sound and should be kept: extract training banks
with `stride=2` (3,136 patches, <9 GB) and evaluate test maps unstrided at `stride=1` to retain
sub-patch fidelity. **Note this makes the train and test grids differ**, which needs its own
verification that bank and query remain comparable — that mismatch is exactly the defect found in
the Triton serving path (§7 of REVIEW.md).

---

### E10 — close the 9.6-point low-resolution deficit  **<- highest expected value in the queue**

At matched resolution the paper's PatchCore beats ours by **9.6 points** — larger than our current
**+5.8** lead. Closing it while keeping 448 roughly doubles the margin. Their configuration is
published, so this is a configuration diff, not a search.

| | theirs (paper §benchmark) | ours (arm A) |
|---|---|---|
| backbone | **ensemble: WideResNet-101 + ResNeXt-101 + DenseNet-201** | single WideResNet50-2 |
| embedding dim | **384** (reduced) | **1536** (full, no reduction) |
| coreset ratio | "0.01%" — see note | 1% nominal, 0.3-0.9% effective |
| centre crop | **disabled** | disabled (`aspect`) — same |
| input | 256x256 | 448 |

Two things this settles immediately.

**They never had our coordinate-frame bug.** The paper says they "disable the center cropping to
enable the detection of defects occurring at the image borders" — the same conclusion this project
reached the hard way. So the geometry fix that took us 0.131 -> 0.301 bought us **parity, not
advantage**, which is exactly why geometry does not correlate with our per-scenario margin.

**Our only real edge is resolution**, and the deficit underneath it is architectural.

---

#### E10a — reduce the embedding to 384 (do this first; it is an enabler, not just an arm)

PatchCore reduces the aggregated embedding to **384** by random projection. We store and score in
the full **1536**. Ours uses a projection *only* inside `coreset_indices` (`proj_dim=128`, at
`sweep_backbones.py:206`) for greedy selection — the bank and every distance are full-dimension.

This is a **4x cut in bank memory and in `patch_distances` cost**, which is why it comes first:
it directly buys the 768 arm we have been calling expensive. The `walnuts` feature tensor at 768
drops from **24.3 GB to ~6 GB**, well clear of the 58 GiB ceiling, and E5's ~1.9 GPU-h falls
towards ~0.5.

**Hypothesis:** 384-dim scoring is within 0.01 mean AU-PRO@5% of 1536-dim at 448, at ~4x lower
cost. PatchCore's own design says the discarded dimensions are redundant; if that holds here, every
later arm gets cheaper for free.

**If it is *better*, not merely equal**, that is a genuine finding — it would mean our full-dim
bank is diluting the nearest-neighbour metric, which is the mechanism PatchCore's projection
exists to prevent.

---

#### E10b — backbone ensemble

The largest architectural gap and the most likely bulk of the 9.6 points. **Do not implement all
three at once** — that is a three-variable arm and this project has been burned by exactly that.

Order: **WideResNet-101 alone** (one variable against arm A) -> add **ResNeXt-101** -> add
**DenseNet-201**. Report each step. If WRN-101 alone captures most of the gain, the ensemble is not
worth its 3x inference cost and we stop there.

**Hypothesis:** WRN-101 alone recovers more than half of the 9.6-point deficit at matched
resolution.

**Cost note:** an ensemble triples feature extraction and triples embedding width before
reduction. **E10a first makes this affordable**; without it, three backbones at 448 is close to
the memory ceiling.

---

#### E10c — coreset ratio *(low priority, and check the number first)*

The paper reports "0.01%". PatchCore's own paper uses **1%** as its headline (`PatchCore-1%`), and
`0.01` written as a *fraction* is exactly 1%. **This is very likely a units artifact, not a real
difference from ours.** Read the paper's own table before spending a run on it — if it is 1%, we
already match and there is nothing here.

---

**Sequencing.** E10a -> E10b -> then re-run E5's 768 arm on top of whatever wins. E10a is the
prerequisite that makes the rest affordable, and E5b (dilated layer3) stays worth running because
it is orthogonal to all of this.

**One caution on framing.** Everything here narrows a gap to *their configuration at 256*. Our
lead over the leaderboard exists because they did not run at 448. If we adopt their architecture
and keep our resolution we should be clearly ahead — but that is still a configuration advantage,
and an honest write-up says so.

---

### E5b — dilated layer3 (`output_stride=8`) at 448  **(orthogonal to E10; still worth running)**

**The single-variable test E5 does not provide.** `sweep_backbones.py:130-134` upsamples layer3
bilinearly 2x onto the layer2 grid. layer3 is **1024 of 1536 dims (66.7%)** and natively stride
16, so two thirds of every descriptor carries no spatial detail finer than stride 16 — **4x the
cell area the whole bucket analysis assumes**.

timm's ResNet accepts `output_stride=8`, which dilates layer3 to a native stride-8 map at the
same input size. Identical grid, identical patch count, identical 4000-vector bank, identical
NN-scoring cost, identical feature-tensor memory. **Only layer3's convolutions run at 4x spatial.**

| arm | patches x bank | feature tensor | est. GPU-h |
|---|---|---|---|
| current 448 | 3120 x 4000 | 8.3 GB | 0.22 |
| **E5b dilated @448** | **3120 x 4000** | **8.3 GB** | **~0.4** |
| E5 768 | 9152 x 11755 | 24.3 GB | ~1.9 |

E5b gives layer3 a **finer** effective map (52x60) than the 768 arm does (44x52), at ~a fifth of
the cost.

**Hypothesis:** E5b recovers a majority of the 768 arm's expected gain. If it does, the driver is
feature-map density rather than input pixels, and the resolution ladder can stop.

**Why this ordering.** E5 changes input pixels, layer2 density, layer3 density, bank size and
object-to-receptive-field ratio **simultaneously**, so whatever it returns it cannot say which of
the five caused it — a direct violation of §8's "change one variable at a time", the rule that has
already caught three wrong conclusions in this project. E5b holds everything fixed but layer3's
effective stride.

**Implementation:** pass `output_stride=8` to `timm.create_model` in `PatchExtractor.__init__` for
the CNN path, behind a flag. Verify the recorded `grid` is unchanged and `n_patches` identical to
the current 448 arm — if either moves, the arm is not single-variable and the comparison is void.

**Also re-derive the bucket cell area.** `ad2_pixel_eval.py:561` and
`exp_e5a_bucketed_pro.py:207` compute cell area from the stride-8 grid alone, understating the
effective cell of 66.7% of the descriptor by 4x. Report both the stride-8 and the
effective-layer3 cell so the size buckets mean something.

**Blocked on:** nothing.

---

### E5 — input resolution, 768 arm

**(plan approved with modifications — but run E5b first)**

§6 once concluded "resolution is not the lever" from 224/448/768 on `can`. That was measured
through the coordinate-frame bug, which is resolution-invariant — exactly what flattens a
real trend into an apparent plateau. **Void; this is a fresh measurement.**

Sweep `img` in {224, 448, 768} under `aspect` geometry, all 8 scenarios, evaluation held at
`eval_side 512` / `sigma 4.0` / fixed 1,530-region native set. **Bank cap scaled with patch
count** so coreset density is constant: 1000 / 4000 / 11755 for 784 / 3136 / 9216 nominal
patches (ratio 1.276 in all three — verified).

**The worker's plan is approved.** The modifications below are the whole delta.

**M1 — run E5a first. It is not optional.** The plan skips straight to E5. E5a costs
minutes and no GPU, and it is what tells us *which* regions are failing — information E5
does not produce at any resolution. It also lets a prediction be registered before the
result exists, which on this project has repeatedly been worth more than the result:
three hypotheses about this gap were killed that way and the fourth was correct.

If E5a shows regions of every size scoring uniformly mediocre, **resolution is not the
answer and E5 should not run as specified**. Spending hours to discover that, when minutes
would have shown it, is the whole reason E5a exists.

**M2 — the memory analysis names the wrong scenario.** The plan attributes the worst case
to `sheet_metal` at ~3.99M vectors / 24.5 GB. `sheet_metal` has only **137 training
images** and extracts 1.26M vectors (7.8 GB) — it is the *cheapest* scenario, not the worst.
The real worst case is **`walnuts`: 432 train images, 3.96M vectors, 24.3 GB**, with `can`
(22.5 GB) and `fabric` (21.8 GB) close behind. The magnitude is right and the fix is
global, so the conclusion stands — but validate the guard against `walnuts`, or it will be
tested on a scenario that was never going to fail.

The prealloc fix is **necessary, not precautionary** — verified independently:

| | peak |
|---|---|
| `walnuts` feature tensor at 768 | 24.3 GB |
| with `torch.cat` transient doubling | 48.7 GB |
| plus concurrent Blender + Triton | **54.0 GB vs a 57.7 GiB ceiling** |

**M3 — resolved, but keep the guard.** A `blender` process was resident at ~3.9 GB while
this plan was being reviewed; the user confirmed it could be stopped, and it exited on its
own before the 768 arm was launched. Only Triton remains (~1.4 GB), which is small enough
to ignore.

Two things survive its departure. **Check free memory immediately before launching the 768
arm rather than assuming the pod is empty** — this is a shared box and another tenant
appearing mid-run is exactly the failure mode that is hardest to diagnose afterwards. And
note it as a live candidate for **E1's unexplained silent kill**: that run died after 5
scenarios with no traceback and a zero cgroup OOM counter, which is consistent with a
host-level OOM kill driven by another process rather than by our own cgroup accounting.
The user has said experiments take priority, so a competing workload can be stopped if one
reappears.

**M4 — arm 2 reuse is legitimate, but the code hash will differ.** Reusing
`E4-evalside-512` for the 448 arm is correct and saves an hour; its config matches exactly.
But arms 1 and 3 will run *new* code (`extract_paths_prealloc`) under a different hash, so
the sweep spans two code versions. The plan's parity test covers this — **require the parity
to be bit-identical**, not merely close, and if it is not, re-run 448 under the new code
rather than comparing across versions. §0 rule: torch 2.13 -> 2.14 alone moved an arm by
0.010, which is larger than margins this project has spent experiments narrowing.

**M5 — state the 1024 arm's disposition explicitly.** The published baseline is at 1024,
so dropping it silently leaves the headline comparison unmade. Deferring it on cost is
fine; make it conditional and say so: **if 768 shows a clear monotonic gain, 1024 becomes
the next item; if 768 is flat, 1024 is pointless and the hypothesis is refuted.**

Worth knowing before reading the result: cell area falls from 1,607 native px at 448 to
~547 at 768 and ~308 at 1024, against a 77 px floor. **Even 1024 does not reach the
smallest defects.** A partial gain at 768 is therefore the expected outcome, not a
disappointment — and it would not by itself close a 2.2x gap.

**Hypothesis:** mean AU-PRO@5% increases monotonically with input resolution, and the gain
concentrates in the scenarios and size buckets E5a identifies as sub-cell.

One sweep-level verdict, not three.

**M6 — pin the bucket edges to fixed native pixels.** `exp_e5a_bucketed_pro.py:207` sets
`cell_area = (w_nat * h_nat) / n_patches`, so buckets are relative to cell size, which
changes with input resolution. At 768 every region jumps 2.94x in cell units and migrates
between buckets, so each bucket would hold a different population per arm — the defect that
invalidated E3, one level up. Compute the edges once from the **448** geometry (1x/4x/16x of
that scenario's 448 cell area) and reuse them verbatim at 224 and 768. **Assert per-bucket
counts are identical across all three arms.**

This is also what makes E5 decisive rather than merely confirmatory. E5a established that
AU-PRO rises with defect size, but that is equally consistent with large defects being
easier for any detector at any resolution. With native-pinned buckets the two separate: a
resolution ceiling lifts the sub-cell bucket sharply from 448 to 768 while leaving `ge_16x`
flat, whereas salience moves every bucket together. See BRIDGE D-04 for the registered
predictions.

**Blocked on:** nothing — E5a passed the gate.

---

### E6 — coreset density, re-checked under fixed geometry

A bank-density sweep on `vial` at 448 gave 4000 -> 0.3596, 2000 -> 0.3351, 1000 -> 0.3437,
500 -> 0.3180: a weak, non-monotonic effect. Measured through the broken metric, on one
scenario, so it establishes less than it appears to.

Re-run at the winning geometry across all 8 with caps {1000, 4000, 16000, uncapped 1%}.

**Hypothesis:** density remains a weak lever (<0.03 mean AU-PRO@5% across the full range),
confirming it is a cost/stability knob rather than an accuracy one.

**Deprioritised to last** (M-11). Expected to be confirmatory and nothing depends on it.

Expected to be *confirmatory*. Run it anyway — it is the cheapest way to close off a
variable that would otherwise keep resurfacing as an explanation, and it has already
resurfaced twice.

**Blocked on:** E4.

---

### E7 — re-select the fusion routing on `validation`

`ad2_feature_fusion.py` routes each scenario to a backbone via a hardcoded if-chain. Two
problems: the routing was chosen against `test_public` (§0 rule 5 — that makes it not a
benchmark number), and it was chosen against pre-bug AU-PRO, so **the evidence it was
fitted to no longer exists.**

Re-select on the `validation` split under the winning geometry, then report the held-out
`test_public` score **once**, with no further adjustment. If the routing is re-tuned after
seeing that number, it is no longer held out and the result is void.

**Hypothesis:** validation-selected routing beats the single best backbone on mean
AU-PRO@5%, **and specifically lifts the `>= 16x` bucket above 0.6056.**

**Report the bucketed breakdown, not just the mean.** The claim being tested is that better
features lift regions that are *already spatially resolved* — the ones resolution cannot
help. A mean improvement driven entirely by the sub-cell bucket is a different result with a
different meaning, and the mean alone cannot tell them apart.

**No longer blocked on E5** (M-11). It was gated on knowing the winning resolution, but the
question it answers is independent of that and is answerable at 448. Run it at 448 under
`aspect` with the fixed native region set; repeat at the winning resolution later only if E5
shows a large gain.

**Report honestly regardless:** the current fusion numbers are a **13.5% regression** on
mean AU-PRO@5% (0.1306 -> 0.1130; improved 3/8, regressed 5/8) even though the README
calls them "all-time project records". That was true for image and pixel AUROC only, and
is now false for those too — plain arm A with correct geometry reaches 0.718/0.846 against
fusion's 0.691/0.770. **Correcting the README is part of this item.**

**Blocked on:** E4.

---

### E8 — replace the synthetic Triton bank

`deployment/triton_models/patchcore/1/bank.npy` is **not a real memory bank**: 49.7% of
its values are negative, which post-ReLU features cannot be, and its row norms sit at
39.78 ~ sqrt(1536) — the signature of Gaussian noise, not features. The threshold is
hardcoded at `export_bank.py:282` and `coreset_size` is 500 on CPU. The README's
"6.34 ms / 157 FPS" was measured against this, so **that figure describes nothing** and
must not be quoted until re-measured.

Fit and export a real bank, re-measure latency on GPU, correct the README.

**Hypothesis:** real-bank latency is materially worse than 6.34 ms once the bank is a
realistic size. Report the honest number *and* the bank size it corresponds to — latency
without bank size is meaningless, since §6's scaling law makes it a free parameter.

**Not blocked.** Independent of all geometry work; safe to run in parallel, or first if
the pod is otherwise busy.

---

### When the queue is done

Report to the planner; do not write conclusions into §6 yourself. The audit exists because
the expensive failures on this project have been wrong conclusions drawn from real numbers,
and an agent auditing its own results reproduces exactly the failure mode §0 prevents.

**Save every run, including refuted ones.** E4, E5 and E6 are all partly attempts to kill
hypotheses, and a refuted hypothesis that is not recorded gets re-proposed — the bank
density question has already come back twice.

If a run dies without a traceback, **capture the evidence before re-running**: `dmesg`,
the cgroup counters (`/sys/fs/cgroup/memory.events`), and peak RSS. E1's kill is still
unexplained, and a second unexplained kill with nothing collected is a debt, not a data
point.

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
