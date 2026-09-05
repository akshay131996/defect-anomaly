# Rebuilding the pod after a stop

The pod was stopped on 2026-09-05. **`/workspace` is a network volume and survives;
everything else is container overlay and does not.** This file is what you need to get back
to a working state.

## What was lost, and what it costs

| path | size | survives? | recovery |
|---|---|---|---|
| `/workspace` (code, outputs, logs, `deployment/`, datasets) | — | **yes** | nothing to do |
| `/opt/ad2` — extracted MVTec AD 2 | 31 GB | no | re-extract, ~10 min |
| `/opt/venvs/anomaly` — the Python env | 6.2 GB | no | reinstall, ~10 min |
| `/tmp/*.log` — live run logs | small | no | **already copied to `logs/pod-tmp/`** |
| `/opt/{nvidia,tritonserver,riva,...}`, `/tmp99` | — | no | base image, reprovisioned automatically |

Total recovery is roughly 20 minutes and needs no decisions.

## 1. Re-extract the dataset

The 32.7 GB tarball is on the volume and was byte-verified against the server. **Extract to
the container disk, not the volume** — MooseFS is slow with many small files:

```bash
mkdir -p /opt/ad2 && tar -xzf /workspace/datasets/mvtec_ad_2.tar.gz -C /opt/ad2
```

Check the container disk has room first: the overlay is 70 GB and was at 73% used with both
the dataset and the venv present.

## 2. Rebuild the venv

Python 3.12.3, torch 2.14.0+cu130, CUDA 13.0. `requirements-anomaly-freeze.txt` in this
directory is a `pip freeze` of the exact working environment (130 packages), captured before
the stop.

```bash
python3 -m venv /opt/venvs/anomaly
/opt/venvs/anomaly/bin/pip install --upgrade pip
# torch first and from the CUDA 13.0 index - the freeze pins 2.14.0, but a plain
# `pip install torch==2.14.0` resolves to a different CUDA build
/opt/venvs/anomaly/bin/pip install torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu130
/opt/venvs/anomaly/bin/pip install -r /workspace/requirements-anomaly-freeze.txt
```

Verify before running anything, because a silently-CPU torch wastes a whole run:

```bash
/opt/venvs/anomaly/bin/python -c "import torch,timm;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), timm.__version__)"
```

Expect `2.14.0+cu130 True NVIDIA RTX 4000 Ada Generation 1.0.29`.

**The torch version matters more than it looks.** Moving 2.13 -> 2.14 mid-project shifted an
AD 1 arm's AUROC by 0.010 with identical seeds — larger than margins two experiments were
spent narrowing. If the rebuilt env lands on a different torch, **say so in the run record**;
results from before and after are not directly comparable.

## 3. Watch the container disk

The overlay is 70 GB. Dataset (31 GB) plus venv (6.2 GB) plus the base image left it at 73%
full. There is room, but not much — do not extract a second dataset copy onto it.

## Notes

- `/workspace` is **not** a git checkout. Files get there by `scp`, which is why run records
  carry `code_sha256` rather than a commit sha. See HANDOFF §0.
- Write live logs to `/tmp`, never `/workspace` — MooseFS blocks on a live-appended file and
  hangs the session with no error. That is why `/tmp/*.log` existed at all, and why they were
  copied into the repo before the stop.
- The pod's SSH port changes on every restart. Get the current connect string from the RunPod
  console; never port-scan the host, it is shared.
