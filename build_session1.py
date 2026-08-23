#!/usr/bin/env python3
# Builds session-1 notebook for the industrial anomaly-detection project.
#
# Generated rather than hand-edited: patching .ipynb JSON in place corrupted a
# multi-line string in the wildlife project, so cell sources here come from clean
# triple-quoted literals and the notebook is written once, whole.
#
#   python build_session1.py            -> writes run_session1.ipynb next to this file

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------- 0. framing
md(r'''
# Industrial anomaly detection — the baseline, and the number a factory actually needs

Train on defect-free parts only. Flag anomalies without ever having seen a labelled defect.
That is the real industrial constraint: a line produces thousands of good parts and a
handful of bad ones, and nobody can wait to collect a balanced dataset.

**Runs on the RTX 4000 Ada pod.** First run downloads ~5.3 GB and caches it; later sessions reuse it.

---

### Why this is not "run PatchCore, report 99%"

MVTec AD is close to saturated — published methods report ~99% image AUROC. Reproducing that
is a tutorial, not a portfolio piece. Three questions the leaderboard does not answer:

1. **What happens at the threshold you would actually ship?** AUROC is threshold-free: it
   measures *ranking*, and a factory needs a *decision*. Every AUROC in the literature is
   compatible with a wide range of real behaviour.
2. **Does the complexity earn itself?** k-NN on frozen ImageNet features is a famously
   strong baseline. PatchCore has to beat *that*, not chance.
3. **How few good images do you actually need?** The question a plant engineer asks first.

This session answers 1 and 2. Session 3 answers 3.

### The rules carried in from the earlier projects

- **Report per class, never the mean alone.** MVTec categories are not equally hard —
  `screw` and `transistor` are notoriously difficult, `bottle` is nearly solved. A mean
  across 15 categories hides exactly the failure that matters.
- **Measure the baseline before improving it.** In [soccer-analytics](https://akshay131996.github.io/soccer-analytics/validation-trap.html)
  a fine-tune scored 0.958 AP on its own validation set and made the pipeline 2.5x *worse*
  than the off-the-shelf model nobody had measured.

### Licence

MVTec AD is **CC BY-NC-SA 4.0 — non-commercial research use only**. Fine for a portfolio and
a blog post; attribute it, and do not reuse it commercially.
''')

# ---------------------------------------------------------------- 1. env
code(r'''
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "datasets", "timm", "scikit-learn", "matplotlib", "pillow"])

import torch, timm
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("timm ", timm.__version__)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print(f"vram  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("\nNo GPU visible — feature extraction will be slow but will still finish.")

# Credentials: huggingface_hub picks up either HF_TOKEN from the environment or the token
# written by `huggingface-cli login`. Never put a token in this notebook — the repo is
# public, and Jupyter autosave/checkpoints bake in whatever is in a cell.
from huggingface_hub import get_token
print("\nHF credential found:", bool(get_token()))
if not get_token():
    print("  Anonymous access is rate-limited. If downloads 429, run this in the pod shell:")
    print("    huggingface-cli login --token <token from huggingface.co/settings/tokens>")
''')

# ---------------------------------------------------------------- 2. data md
md(r'''
## 1. Get the data, and verify what actually arrived

The official MVTec endpoint 404s intermittently as of early 2026, which also breaks
`anomalib`'s auto-download. This uses a Hub mirror whose schema was checked *before* being
written into this notebook.

Two mirrors exist and only one is usable:

| mirror | verdict |
|---|---|
| `Voxel51/mvtec-ad` | schema is `{image}` only — category and defect labels lost |
| `TheoM55/mvtec_all_objects_split` | `object`, `split`, `defect`, `label`, `mask_path` all present |

Per-category counts on the good mirror match the official dataset (bottle 209/83,
screw 320/160), so it is a faithful copy. Columns are still resolved **by name** below,
because a mirror can be re-uploaded with a different layout at any time.
''')

code(r'''
from datasets import load_dataset, concatenate_datasets

DATASET_ID = "TheoM55/mvtec_all_objects_split"

# Non-streaming: ~5.3 GB downloaded once, then cached for sessions 2 and 3. Image columns
# are lazy (arrow-backed), so this does not decode 5,000 images into RAM.
dd = load_dataset(DATASET_ID)
print(dd)

# The mirror may expose train/test as HF splits, or as a `split` column inside one split.
# Handle both rather than assuming, and say which one was found.
parts = []
for split_name, dset in dd.items():
    if "split" not in dset.column_names:
        dset = dset.add_column("split", [split_name] * len(dset))
    parts.append(dset)
full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

print(f"\ncolumns: {full.column_names}")
print(f"rows   : {len(full)}")

# Resolve every column by NAME and fail loudly. The image column here is called
# `image_path` but holds decoded images, which is exactly the kind of naming that
# breaks positional assumptions.
def pick(cands, where):
    hit = next((c for c in cands if c in where), None)
    return hit

image_col  = pick(("image_path", "image", "img"), full.column_names)
label_col  = pick(("label", "labels", "is_anomaly"), full.column_names)
object_col = pick(("object", "category", "class_name"), full.column_names)
defect_col = pick(("defect", "defect_type", "anomaly_type"), full.column_names)
split_col  = pick(("split", "set"), full.column_names)

missing = [n for n, v in [("image", image_col), ("label", label_col),
                          ("object", object_col), ("defect", defect_col),
                          ("split", split_col)] if v is None]
assert not missing, f"could not resolve columns {missing} in {full.column_names}"
print(f"resolved -> image={image_col} label={label_col} object={object_col} "
      f"defect={defect_col} split={split_col}")
''')

# ---------------------------------------------------------------- 3. verify md
md(r'''
## 2. Verify the training set is defect-free

Every method here rests on one structural assumption: **the training split contains only
good parts.** If the mirror got that wrong, or if `label` does not mean what it looks like,
then the memory bank is contaminated with defects, anomaly scores collapse toward zero, and
the result is a plausible-looking number that measures nothing.

So check it rather than trust it. Two things worth confirming:

- `label` and `defect` agree — every row labelled good says `good`, and vice versa. That
  pins down the integer encoding without guessing whether 0 or 1 means anomalous.
- the train split has exactly one class in it.

This is the same discipline as resolving detector classes by name: an assumption that is
cheap to check now, and expensive to discover four steps later.
''')

code(r'''
import collections

lab = full[label_col]
dfc = full[defect_col]
spl = full[split_col]

# Which integer corresponds to "good"? Read it off the defect column instead of assuming.
pairs = collections.Counter(zip(dfc, lab))
good_labels = {l for (d, l) in pairs if str(d).lower() == "good"}
bad_labels  = {l for (d, l) in pairs if str(d).lower() != "good"}

print("defect -> label pairings (top 10):")
for (d, l), n in collections.Counter(pairs).most_common(10):
    print(f"  {str(d):<20} label={l}   {n:5d} rows")

assert len(good_labels) == 1, f"'good' maps to multiple labels {good_labels} — encoding is not clean"
assert not (good_labels & bad_labels), (
    f"label {good_labels & bad_labels} is used for BOTH good and defective rows — "
    "this column cannot be used as ground truth")
GOOD_LABEL = good_labels.pop()
print(f"\ngood == label {GOOD_LABEL}; anomalous == everything else")

# Now the structural check.
by_split = collections.defaultdict(collections.Counter)
for s, l in zip(spl, lab):
    by_split[s][l] += 1

print("\nlabel distribution per split:")
for s, c in by_split.items():
    total = sum(c.values())
    good = c.get(GOOD_LABEL, 0)
    print(f"  {s:<8} {total:5d} rows   good {good:5d} ({100*good/total:5.1f}%)   "
          f"anomalous {total-good:5d}")

train_splits = [s for s in by_split if "train" in str(s).lower()]
assert train_splits, f"no split name contains 'train' — found {list(by_split)}"
for s in train_splits:
    contamination = sum(n for l, n in by_split[s].items() if l != GOOD_LABEL)
    assert contamination == 0, (
        f"train split '{s}' contains {contamination} defective images. The one-class "
        "assumption is violated and every score below would be meaningless.")
print("\nOK — training data is defect-free by construction, as the method requires.")
''')

# ---------------------------------------------------------------- 4. look md
md(r'''
## 3. Choose categories that span the difficulty range

Fifteen categories is more compute than one session needs, and running only the easy ones is
self-deception. Three, chosen to span the space:

| category | type | why |
|---|---|---|
| `bottle` | object | close to solved — the sanity check |
| `screw` | object | consistently the hardest in the benchmark: small defects, rotated parts |
| `carpet` | texture | textures fail differently from objects, so one belongs in the set |

Look at the images before modelling. Some of these defects are a few pixels across, and
knowing that in advance changes how you read a low score later.
''')

code(r'''
import matplotlib.pyplot as plt

CATEGORIES = ["bottle", "screw", "carpet"]

available = sorted(set(full[object_col]))
print(f"{len(available)} categories in the mirror: {available}\n")
missing = [c for c in CATEGORIES if c not in available]
assert not missing, f"requested categories not in dataset: {missing}"

sub = full.filter(lambda r: r[object_col] in CATEGORIES)
print(f"working subset: {len(sub)} rows\n")

for cat in CATEGORIES:
    rows = [i for i, o in enumerate(sub[object_col]) if o == cat]
    defects = collections.Counter(sub[defect_col][i] for i in rows)
    n_train = sum(1 for i in rows if "train" in str(sub[split_col][i]).lower())
    print(f"{cat:<10} train {n_train:4d}   test {len(rows)-n_train:4d}   "
          f"defect types: {dict(defects)}")

# One good / one defective example per category, so the scale of the defects is visible.
fig, axes = plt.subplots(len(CATEGORIES), 2, figsize=(7, 3.4 * len(CATEGORIES)))
for r, cat in enumerate(CATEGORIES):
    idx_good = next(i for i in range(len(sub))
                    if sub[object_col][i] == cat and sub[label_col][i] == GOOD_LABEL)
    idx_bad = next(i for i in range(len(sub))
                   if sub[object_col][i] == cat and sub[label_col][i] != GOOD_LABEL)
    for c, (idx, title) in enumerate([(idx_good, "good"), (idx_bad, sub[defect_col][idx_bad])]):
        axes[r, c].imshow(sub[idx][image_col])
        axes[r, c].set_title(f"{cat} — {title}", fontsize=10)
        axes[r, c].axis("off")
plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- 5. baseline0 md
md(r'''
## 4. Baseline 0 — is this task trivial?

Before reaching for pretrained features, establish the floor. If a defective part can be
spotted by raw pixel statistics alone, the problem is not interesting and no deep model
deserves credit for solving it.

The crudest possible detector: average all training images into one "reference part", then
score each test image by its distance from that average. This catches gross changes in
brightness or layout and nothing subtler.

Expect it to be near chance (AUROC 0.5) on most categories. **That is the point** — it is
the equivalent of the majority-class baseline, and it is what makes the later numbers mean
something.
''')

code(r'''
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

RES = 64  # deliberately crude


def to_small_gray(img):
    return np.asarray(img.convert("L").resize((RES, RES)), dtype=np.float32) / 255.0


def split_indices(cat):
    tr, te = [], []
    for i in range(len(sub)):
        if sub[object_col][i] != cat:
            continue
        (tr if "train" in str(sub[split_col][i]).lower() else te).append(i)
    return tr, te


baseline0 = {}
for cat in CATEGORIES:
    tr, te = split_indices(cat)
    ref = np.mean([to_small_gray(sub[i][image_col]) for i in tr], axis=0)
    scores = np.array([np.linalg.norm(to_small_gray(sub[i][image_col]) - ref) for i in te])
    truth = np.array([0 if sub[label_col][i] == GOOD_LABEL else 1 for i in te])
    auroc = roc_auc_score(truth, scores)
    baseline0[cat] = float(auroc)
    print(f"{cat:<10} AUROC {auroc:.3f}   ({truth.sum()} defective / {len(truth)} test images)")

print(f"\nmean {np.mean(list(baseline0.values())):.3f}   (chance = 0.500)")
print("Anything close to 0.5 means raw pixels carry no usable signal — which is the")
print("result that makes the pretrained-feature numbers below worth reporting.")
''')

# ---------------------------------------------------------------- 6. baseline1 md
md(r'''
## 5. Baseline 1 — k-NN on frozen ImageNet features

This is the number that matters, because it is the one PatchCore has to beat.

No training, no fine-tuning, no coreset subsampling, no patch-level locality. Take a frozen
ImageNet backbone, embed every training image into one pooled vector, and score a test image
by its distance to the *k* nearest training vectors. Roughly twenty lines.

The reason it works: ImageNet features already encode "what normal objects look like", and a
scratch on a metal nut pushes the embedding away from every good example. The reason it has
limits: pooling over the whole image dilutes a small local defect into a global average —
which is precisely the gap patch-based methods exist to close, and precisely why `screw`
should be expected to suffer most here.

Predicting *which* category the baseline will fail on, before running it, is the difference
between a benchmark and an experiment.
''')

code(r'''
import torch, timm
from sklearn.neighbors import NearestNeighbors

BACKBONE = "wide_resnet50_2"   # PatchCore's backbone, so session 2 compares like with like
K = 3
BATCH = 32

model = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(DEVICE).eval()
cfg = timm.data.resolve_data_config({}, model=model)
tfm = timm.data.create_transform(**cfg, is_training=False)
print(f"{BACKBONE}: input {cfg['input_size']}, feature dim {model.num_features}")


@torch.no_grad()
def embed(indices):
    out = []
    for i in range(0, len(indices), BATCH):
        batch = torch.stack([tfm(sub[j][image_col].convert("RGB"))
                             for j in indices[i:i + BATCH]]).to(DEVICE)
        f = model(batch)
        out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


baseline1, per_cat = {}, {}
for cat in CATEGORIES:
    tr, te = split_indices(cat)
    f_tr, f_te = embed(tr), embed(te)

    nn = NearestNeighbors(n_neighbors=K).fit(f_tr)
    dist, _ = nn.kneighbors(f_te)
    scores = dist.mean(axis=1)                       # higher = more anomalous
    truth = np.array([0 if sub[label_col][i] == GOOD_LABEL else 1 for i in te])

    # Scores on the TRAINING data too: the threshold later has to be set from these alone,
    # because defect-free parts are all a factory has at calibration time.
    d_tr, _ = nn.kneighbors(f_tr, n_neighbors=K + 1)
    train_scores = d_tr[:, 1:].mean(axis=1)          # drop self-match

    auroc = roc_auc_score(truth, scores)
    baseline1[cat] = float(auroc)
    per_cat[cat] = {"scores": scores, "truth": truth, "train_scores": train_scores}
    print(f"{cat:<10} AUROC {auroc:.3f}")

print(f"\nmean {np.mean(list(baseline1.values())):.3f}")
''')

# ---------------------------------------------------------------- 7. percat md
md(r'''
## 6. Per category — where the mean was hiding things
''')

code(r'''
print(f"{'category':<12}{'baseline 0':>13}{'baseline 1':>13}{'gain':>9}")
print("-" * 47)
for cat in CATEGORIES:
    g = baseline1[cat] - baseline0[cat]
    print(f"{cat:<12}{baseline0[cat]:>13.3f}{baseline1[cat]:>13.3f}{g:>+9.3f}")
print("-" * 47)
print(f"{'MEAN':<12}{np.mean(list(baseline0.values())):>13.3f}"
      f"{np.mean(list(baseline1.values())):>13.3f}")

hardest = min(baseline1, key=baseline1.get)
easiest = max(baseline1, key=baseline1.get)
print(f"\neasiest '{easiest}' {baseline1[easiest]:.3f}   hardest '{hardest}' {baseline1[hardest]:.3f}")
print(f"spread {baseline1[easiest] - baseline1[hardest]:.3f} AUROC across three categories of")
print("the same benchmark, with the same model. A single headline number would have")
print("reported none of that, and a 15-category mean would have buried it further.")
''')

# ---------------------------------------------------------------- 8. operating md
md(r'''
## 7. The operating point — from AUROC to something shippable

AUROC is threshold-free. It says how well the scores *rank* good against defective, and a
factory cannot ship a ranking: it needs a line that says pass or fail.

Two things follow, and the second is the one tutorials get wrong.

**The threshold must be set on defect-free data alone.** At calibration time a plant has
good parts and nothing else — that is the entire premise of one-class detection. Choosing a
threshold by looking at test-set defects is leakage, and it silently reports a number the
deployed system can never reproduce. Here the threshold is the 99th percentile of the
*training* scores: reject roughly 1% of known-good parts, and accept whatever detection rate
that buys.

**The reported number changes completely.** Same model, same scores, same AUROC.
''')

code(r'''
PCTL = 99.0

print(f"threshold = {PCTL:.0f}th percentile of TRAIN scores (defect-free data only)\n")
print(f"{'category':<10}{'AUROC':>8}{'thresh':>9}{'recall':>9}{'escapes':>10}"
      f"{'false alarm':>13}")
print("-" * 59)

operating = {}
for cat in CATEGORIES:
    d = per_cat[cat]
    thr = float(np.percentile(d["train_scores"], PCTL))
    pred = (d["scores"] > thr).astype(int)
    truth = d["truth"]

    tp = int(((pred == 1) & (truth == 1)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())

    recall = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    operating[cat] = {"threshold": thr, "recall": recall, "escapes": fn,
                      "false_alarm_rate": far, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    print(f"{cat:<10}{baseline1[cat]:>8.3f}{thr:>9.3f}{recall:>9.1%}{fn:>10d}{far:>13.1%}")

print("-" * 59)
print("\n'escapes' is defective parts that passed inspection and shipped to a customer.")
print("That column is the product requirement. AUROC does not contain it, and two models")
print("with identical AUROC can differ substantially in it.")
''')

# ---------------------------------------------------------------- 9. cost md
md(r'''
## 8. The costs are not symmetric, so the threshold should not be either

A missed defect and a false alarm are both errors and they are not the same error.

An escape ships a bad part: warranty claims, recalls, a customer audit, in automotive
potentially a safety report. A false alarm pulls a good part off the line for human review:
a few minutes of an inspector's time, and at worst a brief line stop.

Put a ratio on it — say an escape costs 100x a false alarm — and the optimal threshold moves,
often a long way from anything a symmetric metric would choose. Sweeping it is the actual
engineering decision, and it is what separates "the model scores 0.97" from "here is where
to run it and what that costs".

The ratio itself is a business input, not a modelling one. What an engineer owes the
business is this curve, so the ratio can be argued about with numbers attached.
''')

code(r'''
COST_ESCAPE, COST_FALSE_ALARM = 100.0, 1.0

fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(5 * len(CATEGORIES), 4))
if len(CATEGORIES) == 1:
    axes = [axes]

cost_summary = {}
for ax, cat in zip(axes, CATEGORIES):
    d = per_cat[cat]
    grid = np.percentile(d["train_scores"], np.linspace(50, 100, 120))
    costs, escapes_at = [], []
    for thr in grid:
        pred = (d["scores"] > thr).astype(int)
        fn = int(((pred == 0) & (d["truth"] == 1)).sum())
        fp = int(((pred == 1) & (d["truth"] == 0)).sum())
        costs.append(fn * COST_ESCAPE + fp * COST_FALSE_ALARM)
        escapes_at.append(fn)
    costs = np.array(costs)

    best = int(np.argmin(costs))
    default_thr = float(np.percentile(d["train_scores"], PCTL))
    cost_summary[cat] = {
        "best_threshold": float(grid[best]),
        "best_cost": float(costs[best]),
        "escapes_at_best": int(escapes_at[best]),
        "cost_at_99th_pctl": float(costs[np.argmin(np.abs(grid - default_thr))]),
    }

    ax.plot(grid, costs, lw=2)
    ax.axvline(grid[best], ls="--", c="tab:green", label=f"cost-optimal ({grid[best]:.3f})")
    ax.axvline(default_thr, ls="--", c="tab:red", label=f"99th pctl ({default_thr:.3f})")
    ax.set_title(f"{cat} — escape:false-alarm = {COST_ESCAPE:.0f}:{COST_FALSE_ALARM:.0f}")
    ax.set_xlabel("threshold"); ax.set_ylabel("total cost"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

for cat, c in cost_summary.items():
    delta = c["cost_at_99th_pctl"] - c["best_cost"]
    print(f"{cat:<10} cost-optimal {c['best_threshold']:.3f} (cost {c['best_cost']:.0f}, "
          f"{c['escapes_at_best']} escapes) vs 99th-pctl cost {c['cost_at_99th_pctl']:.0f}"
          f"  -> {delta:+.0f}")
print("\nWhere that difference is large, the percentile heuristic everyone reaches for")
print("first is quietly the wrong choice for the business it is running in.")
''')

# ---------------------------------------------------------------- 10. summary md
md(r'''
## 9. What this establishes

- **A floor.** Raw-pixel distance shows how much of the result is the method and how much
  was already sitting in the data.
- **A baseline with no training in it.** k-NN on frozen features is what any more complex
  method has to beat. Session 2 compares PatchCore and PaDiM against *this*, at the *same
  operating point* — not against chance, and not on AUROC alone.
- **Per-category numbers**, because the spread across three categories of one benchmark is
  wide enough that a mean is close to meaningless.
- **An operating point set the way a factory has to set it** — from defect-free data — and
  a cost curve that says what choosing it differently would be worth.

### Next

1. **Session 2:** PatchCore + PaDiM via `anomalib`, same categories, same threshold rule.
   The question is not whether they win but by how much, and whether it survives at the
   operating point rather than only in AUROC.
2. **Session 3:** the data-efficiency ablation — 10 / 25 / 50 / 100 / all good images. The
   question a plant engineer asks first, and the one the leaderboard never answers.
3. All 15 categories once the pipeline is settled, reported per category.
''')

code(r'''
import json

summary = {
    "dataset": DATASET_ID,
    "categories": CATEGORIES,
    "backbone": BACKBONE,
    "k": K,
    "threshold_rule": f"{PCTL}th percentile of train-only scores",
    "cost_ratio_escape_to_false_alarm": COST_ESCAPE / COST_FALSE_ALARM,
    "baseline0_raw_pixel_auroc": {c: round(v, 4) for c, v in baseline0.items()},
    "baseline1_knn_frozen_features_auroc": {c: round(v, 4) for c, v in baseline1.items()},
    "operating_point": {c: {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in d.items()} for c, d in operating.items()},
    "cost_sweep": {c: {k: round(v, 4) for k, v in d.items()} for c, d in cost_summary.items()},
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/session1_baseline.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2)[:1800])
print("\nwrote outputs/session1_baseline.json")
print("This file is session 2's comparison target — PatchCore has to beat these numbers")
print("at the same operating point, not just on AUROC.")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_session1.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

# Fail here rather than in Jupyter if any cell is syntactically broken.
import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse(c["source"])
print(f"wrote {out}  ({len(nb['cells'])} cells, all code cells parse)")
