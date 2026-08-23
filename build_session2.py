#!/usr/bin/env python3
# Builds session-2 notebook: PatchCore, and the backbone ablation.
#
# Generated rather than hand-edited, same as session 1 — see build_session1.py.
#
#   python build_session2.py      -> writes run_session2.ipynb next to this file

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------- 0
md(r'''
# Session 2 — does the complexity pay, and does the backbone?

Session 1 established a floor: a frozen backbone, global-average-pooled to one vector per
image, scored by distance to its nearest training neighbours. It was built to be weak in a
*predictable* way — pooling 7x7 down to 1x1 destroys any defect small enough to matter.

This session changes two things, **one at a time**, and measures each separately.

| run | method | backbone | what it isolates |
|---|---|---|---|
| baseline | pooled k-NN | WideResNet50-2 | *(session 1)* |
| **A** | PatchCore | WideResNet50-2 | what spatial locality is worth |
| **B** | PatchCore | DINOv2 ViT | what feature quality is worth |

Run A keeps the published PatchCore recipe so the numbers are checkable against the
literature. Run B changes only the feature extractor. If both had moved at once, a gain
would be unattributable — which is exactly the trap that cost a week in
[soccer-analytics](https://akshay131996.github.io/soccer-analytics/validation-trap.html),
where a fine-tune and a label space changed together.

**Every comparison below is at the same operating point as session 1**, using the same
threshold rule read from that session's output file. AUROC is reported, but it is not the
number the comparison turns on.
''')

# ---------------------------------------------------------------- 1
code(r'''
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "datasets", "timm", "scikit-learn", "matplotlib", "pillow"])

import os, json, torch, timm
print("torch", torch.__version__, "| timm", timm.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("device:", DEVICE)

from huggingface_hub import get_token
print("HF credential found:", bool(get_token()))

# Session 2 exists to be compared against session 1. Read that run's configuration rather
# than restating it here — a threshold rule or category list that drifts between sessions
# silently invalidates every comparison below.
BASE_PATH = "outputs/session1_baseline.json"
if not os.path.exists(BASE_PATH):
    raise FileNotFoundError(
        f"{BASE_PATH} not found. Run run_session1.ipynb to completion first — this "
        "notebook compares against its numbers and reuses its categories, threshold rule "
        "and cost ratio so the two runs stay commensurable.")

with open(BASE_PATH) as f:
    BASE = json.load(f)

CATEGORIES  = BASE["categories"]
PCTL        = float(BASE["threshold_rule"].split("th")[0])
COST_ESCAPE = float(BASE["cost_ratio_escape_to_false_alarm"])
DATASET_ID  = BASE["dataset"]

print(f"\ncomparing against session 1:")
print(f"  categories   {CATEGORIES}")
print(f"  threshold    {PCTL}th percentile of train-only scores")
print(f"  cost ratio   {COST_ESCAPE:.0f}:1 escape to false alarm")
for c, v in BASE["baseline1_knn_frozen_features_auroc"].items():
    print(f"  baseline {c:<10} AUROC {v:.3f}")
''')

# ---------------------------------------------------------------- 2
md(r'''
## 1. Same data, same split logic

Identical to session 1 — the mirror whose schema was verified before use, columns resolved
by name.
''')

code(r'''
import collections
from datasets import load_dataset, concatenate_datasets

dd = load_dataset(DATASET_ID)
parts = []
for split_name, dset in dd.items():
    if "split" not in dset.column_names:
        dset = dset.add_column("split", [split_name] * len(dset))
    parts.append(dset)
full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

def pick(cands, where):
    return next((c for c in cands if c in where), None)

image_col  = pick(("image_path", "image", "img"), full.column_names)
label_col  = pick(("label", "labels", "is_anomaly"), full.column_names)
object_col = pick(("object", "category", "class_name"), full.column_names)
defect_col = pick(("defect", "defect_type", "anomaly_type"), full.column_names)
split_col  = pick(("split", "set"), full.column_names)
assert all([image_col, label_col, object_col, defect_col, split_col]), full.column_names

pairs = collections.Counter(zip(full[defect_col], full[label_col]))
GOOD_LABEL = next(l for (d, l) in pairs if str(d).lower() == "good")

sub = full.filter(lambda r: r[object_col] in CATEGORIES)

def split_indices(cat):
    tr, te = [], []
    for i in range(len(sub)):
        if sub[object_col][i] != cat:
            continue
        (tr if "train" in str(sub[split_col][i]).lower() else te).append(i)
    return tr, te

for cat in CATEGORIES:
    tr, te = split_indices(cat)
    print(f"{cat:<10} train {len(tr):4d}   test {len(te):4d}")
''')

# ---------------------------------------------------------------- 3
md(r'''
## 2. What PatchCore actually changes

Three mechanisms, and it is worth being able to name which one is doing the work.

**1. Tap intermediate layers, keep the grid.** Rather than the pooled output of the final
stage, take the feature *maps* from `layer2` and `layer3`, upsample the deeper one to match
the shallower, and concatenate. Each spatial position becomes its own descriptor. A 224 px
input yields a 28x28 grid — 784 patch vectors per image instead of one.

The deepest layer is deliberately skipped. Its features are the most ImageNet-committed —
tuned to separate dog breeds — and the most spatially coarse. Neither property helps here.

**2. Local neighbourhood aggregation.** Average each patch descriptor with its immediate
neighbours (a 3x3 window, stride 1). This widens what each descriptor sees *without*
reducing the grid — receptive field and resolution are usually traded against each other,
and this buys a little of the first without paying in the second.

**3. A memory bank, then a coreset.** Store every patch descriptor from every training
image. That is hundreds of thousands of vectors per category, and searching it per part is
too slow for a line. So keep a **coreset**: a greedy selection that repeatedly adds the
point furthest from everything already chosen. It preserves the *shape* of the set —
including its edges, which is where the decision boundary lives — at around 1% of the size.
Random sampling would preserve the dense middle and lose exactly the rare-but-normal
patches that prevent false alarms.

Scoring a part: for each of its patches, distance to the nearest bank entry. The image score
is the **maximum** over patches — one bad region is enough to condemn a part, which is the
correct semantics for inspection and the direct opposite of averaging.
''')

# ---------------------------------------------------------------- 4
code(r'''
import numpy as np
import torch.nn.functional as F

IMG = 224

# Backbone identifiers are resolved against what timm actually ships rather than hardcoded.
# A model id that has been renamed fails here, loudly, instead of four cells later.
def resolve_dinov2():
    found = timm.list_models("*dinov2*", pretrained=True)
    if not found:
        raise RuntimeError(
            "no pretrained dinov2 models in this timm build. Upgrade timm, or set "
            "RUNS to run A only.")
    for want in ("vit_base_patch14_dinov2", "vit_small_patch14_dinov2"):
        hit = next((m for m in found if m.startswith(want)), None)
        if hit:
            return hit
    return found[0]

DINO_ID = resolve_dinov2()
print("dinov2 checkpoint resolved to:", DINO_ID)

RUNS = [
    {"tag": "A · patchcore + wrn50",  "kind": "cnn", "name": "wide_resnet50_2",
     "out_indices": (2, 3)},
    {"tag": "B · patchcore + dinov2", "kind": "vit", "name": DINO_ID},
]


class PatchExtractor:
    """Turns a batch of images into a grid of patch descriptors. The two branches differ
    only in how the grid is obtained; everything downstream is shared."""

    def __init__(self, spec):
        self.kind = spec["kind"]
        if self.kind == "cnn":
            self.model = timm.create_model(
                spec["name"], pretrained=True, features_only=True,
                out_indices=spec["out_indices"]).to(DEVICE).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
        else:
            self.model = timm.create_model(
                spec["name"], pretrained=True, num_classes=0,
                img_size=IMG).to(DEVICE).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            cfg["input_size"] = (3, IMG, IMG)
        self.tfm = timm.data.create_transform(**cfg, is_training=False)
        self.grid = None
        self.dim = None

    @torch.no_grad()
    def __call__(self, pil_batch):
        x = torch.stack([self.tfm(im.convert("RGB")) for im in pil_batch]).to(DEVICE)

        if self.kind == "cnn":
            fs = self.model(x)
            ref = fs[0].shape[-2:]
            fs = [f if f.shape[-2:] == ref else
                  F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                  for f in fs]
            fmap = torch.cat(fs, dim=1)
        else:
            toks = self.model.forward_features(x)
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            toks = toks[:, n_prefix:, :]
            g = int(round(toks.shape[1] ** 0.5))
            assert g * g == toks.shape[1], f"{toks.shape[1]} patch tokens is not square"
            fmap = toks.transpose(1, 2).reshape(toks.shape[0], -1, g, g)

        # Mechanism 2: widen the receptive field without shrinking the grid.
        fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)

        b, c, h, w = fmap.shape
        self.grid, self.dim = (h, w), c
        return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()


@torch.no_grad()
def extract(ex, indices, batch=16):
    out = []
    for i in range(0, len(indices), batch):
        out.append(ex([sub[j][image_col] for j in indices[i:i + batch]]))
    return torch.cat(out)
''')

# ---------------------------------------------------------------- 5
md(r'''
## 3. Coreset and scoring

The greedy selection below is the one part worth reading closely. It starts from a random
patch and repeatedly adds whichever remaining patch is *furthest* from everything selected
so far — so it walks the outside of the distribution rather than its centre.

The distances are computed on a random low-dimensional projection. That sounds like a
corner cut and is not: the Johnson–Lindenstrauss lemma says random projection preserves
pairwise distances within a small distortion, and selection only needs the *ordering* of
distances to be roughly right. The bank itself keeps full-dimensional vectors.
''')

code(r'''
@torch.no_grad()
def coreset_indices(feats, ratio=0.01, proj_dim=128, seed=0):
    n = feats.shape[0]
    k = max(int(n * ratio), 32)
    g = torch.Generator().manual_seed(seed)

    P = torch.randn(feats.shape[1], proj_dim, generator=g) / (proj_dim ** 0.5)
    X = (feats @ P).to(DEVICE)

    start = int(torch.randint(n, (1,), generator=g))
    sel = [start]
    d = torch.cdist(X, X[start:start + 1]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(d))
        sel.append(i)
        d = torch.minimum(d, torch.cdist(X, X[i:i + 1]).squeeze(1))
    return sel


@torch.no_grad()
def patch_distances(bank, feats, n_patches, chunk=8192):
    bank = bank.to(DEVICE)
    mins = []
    for i in range(0, feats.shape[0], chunk):
        d = torch.cdist(feats[i:i + chunk].to(DEVICE), bank)
        mins.append(d.min(dim=1).values.cpu())
    return torch.cat(mins).view(-1, n_patches)
''')

# ---------------------------------------------------------------- 6
md(r'''
## 4. Run both backbones through identical code

The only thing that differs between run A and run B is the object constructed at the top of
the loop. Everything after it — aggregation, coreset ratio, scoring rule, threshold rule —
is shared, which is what makes the difference in the results attributable to the backbone.
''')

code(r'''
import time
from sklearn.metrics import roc_auc_score

CORESET_RATIO = 0.01
results = {}

for spec in RUNS:
    ex = PatchExtractor(spec)
    results[spec["tag"]] = {}
    print(f"\n=== {spec['tag']} ===")

    for cat in CATEGORIES:
        t0 = time.time()
        tr, te = split_indices(cat)

        f_tr = extract(ex, tr)
        n_patch = ex.grid[0] * ex.grid[1]

        keep = coreset_indices(f_tr, CORESET_RATIO)
        bank = f_tr[keep]

        d_te = patch_distances(bank, extract(ex, te), n_patch)
        scores = d_te.max(dim=1).values.numpy()

        # Train scores calibrate the threshold, exactly as in session 1 — defect-free data
        # is all a line has at commissioning.
        d_tr = patch_distances(bank, f_tr, n_patch)
        train_scores = d_tr.max(dim=1).values.numpy()

        truth = np.array([0 if sub[label_col][i] == GOOD_LABEL else 1 for i in te])
        auroc = roc_auc_score(truth, scores)

        results[spec["tag"]][cat] = {
            "auroc": float(auroc), "scores": scores, "truth": truth,
            "train_scores": train_scores, "grid": ex.grid, "dim": ex.dim,
            "bank_size": len(keep), "patches_per_image": n_patch,
            "bank": bank,          # kept for the anomaly maps in section 8
            "seconds": time.time() - t0,
        }
        print(f"  {cat:<10} AUROC {auroc:.3f}   grid {ex.grid[0]}x{ex.grid[1]}"
              f"   dim {ex.dim:5d}   bank {len(keep):6d}   {time.time()-t0:5.1f}s")

    del ex
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
''')

# ---------------------------------------------------------------- 7
md(r'''
## 5. Ranking quality — the number that is easy to quote
''')

code(r'''
base = BASE["baseline1_knn_frozen_features_auroc"]
tags = [r["tag"] for r in RUNS]

hdr = f"{'category':<10}{'baseline':>11}" + "".join(f"{t.split(' · ')[0]:>11}" for t in tags)
print(hdr); print("-" * len(hdr))
for cat in CATEGORIES:
    row = f"{cat:<10}{base[cat]:>11.3f}"
    for t in tags:
        row += f"{results[t][cat]['auroc']:>11.3f}"
    print(row)
print("-" * len(hdr))
row = f"{'MEAN':<10}{np.mean([base[c] for c in CATEGORIES]):>11.3f}"
for t in tags:
    row += f"{np.mean([results[t][c]['auroc'] for c in CATEGORIES]):>11.3f}"
print(row)

print("\nPer-category gain over the session-1 baseline:")
for t in tags:
    gains = {c: results[t][c]["auroc"] - base[c] for c in CATEGORIES}
    best, worst = max(gains, key=gains.get), min(gains, key=gains.get)
    print(f"  {t:<26} biggest {best} {gains[best]:+.3f}   smallest {worst} {gains[worst]:+.3f}")
print("\nIf the biggest gain landed on the category with the smallest defects, that is the")
print("locality mechanism showing up where it was predicted to. If it landed elsewhere,")
print("the explanation is wrong and worth chasing before trusting the mean.")
''')

# ---------------------------------------------------------------- 8
md(r'''
## 6. The comparison that actually decides anything

AUROC above, operating point here. Same threshold rule as session 1 — the percentile of
*training* scores — so all three systems are being asked the same question: at a false-alarm
budget you can staff for, how many defective parts ship?
''')

code(r'''
def operating(scores, train_scores, truth, pctl):
    thr = float(np.percentile(train_scores, pctl))
    pred = (scores > thr).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum()); fn = int(((pred == 0) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum()); tn = int(((pred == 0) & (truth == 0)).sum())
    return {"threshold": thr, "escapes": fn, "false_alarms": fp,
            "recall": tp / max(tp + fn, 1), "far": fp / max(fp + tn, 1),
            "cost": fn * COST_ESCAPE + fp}

print(f"threshold = {PCTL:.0f}th percentile of train scores   cost = {COST_ESCAPE:.0f}:1\n")
print(f"{'category':<10}{'run':<26}{'recall':>9}{'escapes':>9}{'false al.':>11}{'cost':>9}")
print("-" * 74)

op = {}
for cat in CATEGORIES:
    b = BASE["operating_point"][cat]
    print(f"{cat:<10}{'baseline (session 1)':<26}{b['recall']:>9.1%}"
          f"{b['escapes']:>9d}{b['fp']:>11d}"
          f"{b['escapes']*COST_ESCAPE + b['fp']:>9.0f}")
    for t in tags:
        d = results[t][cat]
        o = operating(d["scores"], d["train_scores"], d["truth"], PCTL)
        op.setdefault(t, {})[cat] = o
        print(f"{'':<10}{t:<26}{o['recall']:>9.1%}{o['escapes']:>9d}"
              f"{o['false_alarms']:>11d}{o['cost']:>9.0f}")
    print("-" * 74)

print("\nRead the escapes column first. A method that improves AUROC while leaving escapes")
print("unchanged has improved the ranking of parts you were already catching, which is")
print("worth nothing on a line.")
''')

# ---------------------------------------------------------------- 9
md(r'''
## 7. What it costs to run

Accuracy is only half a deployment decision. A line has a takt time, and a memory bank that
must be searched per part is the slowest thing in this pipeline. The descriptor
dimensionality drives both the bank size and the search cost, and the two backbones differ
substantially there.
''')

code(r'''
print(f"{'run':<26}{'grid':>9}{'dim':>7}{'patches/img':>13}{'bank':>9}{'total s':>9}")
print("-" * 73)
for t in tags:
    d0 = results[t][CATEGORIES[0]]
    total = sum(results[t][c]["seconds"] for c in CATEGORIES)
    print(f"{t:<26}{d0['grid'][0]}x{d0['grid'][1]:<6}{d0['dim']:>7}"
          f"{d0['patches_per_image']:>13}{d0['bank_size']:>9}{total:>9.1f}")

print("\nThese timings include feature extraction, coreset selection and scoring for all")
print("three categories, on a warm GPU with data already cached. They are a relative")
print("comparison between backbones, not a latency figure for a production line — that")
print("needs a fixed batch size, an export format, and the target hardware.")
''')

# ---------------------------------------------------------------- 10
md(r'''
## 8. Where the model says the defect is

The pooled baseline could only ever emit one number per image. PatchCore scores every
patch, so the same forward pass yields a map — and a map is inspectable in a way a score
never is. An operator can see whether the system flagged the actual scratch or a reflection
on the fixture, which matters as much for trust as the metric does.

Worth looking at the false positives here rather than the wins.
''')

code(r'''
import matplotlib.pyplot as plt

# The banks were kept from the run above, so this only needs one forward pass per image —
# rebuilding the extractor per category would repeat the whole session to draw three
# pictures.
fig, axes = plt.subplots(len(CATEGORIES), len(tags) + 1,
                         figsize=(4.2 * (len(tags) + 1), 4 * len(CATEGORIES)))
if len(CATEGORIES) == 1:
    axes = axes[None, :]

extractors = {s["tag"]: PatchExtractor(s) for s in RUNS}

for r, cat in enumerate(CATEGORIES):
    tr, te = split_indices(cat)
    d = results[tags[0]][cat]

    # The highest-scoring true defect: what the system is most confident about.
    worst = int(np.argmax(np.where(d["truth"] == 1, d["scores"], -np.inf)))
    img = sub[te[worst]][image_col].convert("RGB")
    W, H = img.size

    axes[r, 0].imshow(img); axes[r, 0].axis("off")
    axes[r, 0].set_title(f"{cat} — {sub[te[worst]][defect_col]}", fontsize=10)

    for c, t in enumerate(tags):
        dd = results[t][cat]
        gh, gw = dd["grid"]
        feats = extractors[t]([img])
        amap = patch_distances(dd["bank"], feats, gh * gw).view(gh, gw).numpy()

        axes[r, c + 1].imshow(img)
        axes[r, c + 1].imshow(amap, extent=[0, W, H, 0], alpha=0.55,
                              cmap="inferno", interpolation="bilinear")
        axes[r, c + 1].axis("off")
        axes[r, c + 1].set_title(t, fontsize=10)

for ex in extractors.values():
    del ex
torch.cuda.empty_cache() if DEVICE == "cuda" else None

plt.tight_layout(); plt.show()
print("Bright = far from anything in the memory bank. Check whether the bright region sits")
print("on the actual defect or on a fixture edge — a right answer for the wrong reason is")
print("the failure this visualisation exists to catch.")
''')

# ---------------------------------------------------------------- 11
md(r'''
## 9. What this establishes

Read the three rows against each other in this order.

- **Baseline to run A** is the value of *spatial locality* — the same features, kept on a
  grid instead of averaged into one vector. If the gain concentrates on the small-defect
  category, the mechanism is behaving as predicted rather than as hoped.
- **Run A to run B** is the value of *feature quality* — the same algorithm, a backbone
  trained without labels and without a classification objective pushing it to discard
  texture. This is the only comparison in the project where a single variable moved.
- **Escapes, not AUROC**, decides which one you would actually ship, and the runtime table
  decides whether you could.

### Next

1. **Pixel-level evaluation.** The mirror ships ground-truth masks; the maps above are
   currently qualitative. Pixel AUROC and PRO would make localisation a measurement.
2. **All 15 categories**, reported per category. Three was enough to develop against and is
   not enough to conclude from.
3. **Session 3 — data efficiency.** How many good images before performance plateaus. It is
   the question a plant asks first and the literature answers least.
''')

code(r'''
summary = {
    "compared_against": BASE_PATH,
    "dataset": DATASET_ID,
    "categories": CATEGORIES,
    "coreset_ratio": CORESET_RATIO,
    "threshold_rule": f"{PCTL}th percentile of train-only scores",
    "cost_ratio_escape_to_false_alarm": COST_ESCAPE,
    "runs": {},
}
for t in tags:
    summary["runs"][t] = {
        "backbone": next(s["name"] for s in RUNS if s["tag"] == t),
        "auroc": {c: round(results[t][c]["auroc"], 4) for c in CATEGORIES},
        "grid": list(results[t][CATEGORIES[0]]["grid"]),
        "descriptor_dim": results[t][CATEGORIES[0]]["dim"],
        "operating_point": {c: {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in op[t][c].items()} for c in CATEGORIES},
        "seconds_total": round(sum(results[t][c]["seconds"] for c in CATEGORIES), 1),
    }
summary["baseline_session1_auroc"] = BASE["baseline1_knn_frozen_features_auroc"]

os.makedirs("outputs", exist_ok=True)
with open("outputs/session2_patchcore.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary["runs"], indent=2)[:2000])
print("\nwrote outputs/session2_patchcore.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_session2.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

import ast
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse(c["source"])
print(f"wrote {out}  ({len(nb['cells'])} cells, all code cells parse)")
