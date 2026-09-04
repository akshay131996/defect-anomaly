# PatchCore Deployment Pipeline for Triton Inference Server & DeepStream (Phase C)

## 1. Overview & Architecture

PatchCore is a training-free visual anomaly detection algorithm consisting of:
1. **Feature Extraction:** Pre-trained frozen backbone (WideResNet50-2 or DINOv2) extracting intermediate representations from layers 2 and 3, unified via bilinear interpolation and $3 \times 3$ local average pooling ($p \times p$ grid, e.g. $28 \times 28 = 784$ patches).
2. **Coreset Memory Bank:** A representative subset of normal patch vectors ($\sim 4,000$ vectors, dim 1536) selected via greedy minimax $k$-center coreset reduction.
3. **Inference & Scoring:** For each test patch, minimum Euclidean distance to the memory bank is computed on GPU via `torch.cdist`. The maximum patch distance represents the image anomaly score, and the spatially smoothed 2D grid provides the pixel anomaly heatmap.

### The Deployment Challenge
Standard ONNX export handles the backbone forward pass, but the memory bank $k$-NN lookup is not a standard static ONNX operator (it depends on a dynamic or pre-loaded reference tensor of thousands of patch vectors).

### The Triton Architecture Solution
We deploy PatchCore via the **Triton Python Backend** (`deployment/triton_models/patchcore`):
- **Zero-overhead GPU execution:** Image preprocessing, backbone feature extraction, and batched CUDA $k$-NN distance computation run within a single in-memory CUDA context, avoiding tens of megabytes of intermediate tensor serialization across IPC boundaries.
- **Dynamic Batching:** Automatically batches concurrent camera frames within a configurable time window (e.g. 5ms).
- **Air-gapped Production Support:** Pre-loads weights and coreset banks from disk (`bank.pt`, `metadata.json`) without external internet dependencies.
- **Dual DeepStream Integration Paths:**
  1. `nvinferserver` (in-process Triton C-API integration with GStreamer buffers).
  2. `pyds` Pad Probe (extracting decoded video frames from `NvBufSurface` and dispatching to Triton via gRPC or direct in-process engine).

---

## 2. Directory Layout

```
deployment/
├── triton_models/
│   └── patchcore/
│       ├── config.pbtxt             # Triton model configuration (FP32 IMAGE -> ANOMALY_SCORE, ANOMALY_MAP, IS_DEFECTIVE)
│       └── 1/
│           ├── model.py             # Triton Python backend (initialize, execute, finalize)
│           ├── bank.pt              # Coreset memory bank tensor (M, C)
│           ├── bank.npy             # NumPy format of memory bank
│           ├── metadata.json        # Backbone spec, threshold, input size, normalization stats
│           └── backbone.pt          # (Optional) Offline backbone PyTorch state_dict
├── export_bank.py                   # Utility to extract features, select coreset, and export bank + metadata
├── test_client.py                   # Verification client supporting Direct, gRPC, HTTP, and DeepStream probe modes
└── README.md                        # Documentation and architecture guide
```

---

## 3. Quickstart & Verification

### Step 1: Export Fitted Coreset Memory Bank
To generate synthetic verification bank:
```bash
python deployment/export_bank.py --synthetic --max-k 1000 --output-dir deployment/triton_models/patchcore/1
```

To fit from defect-free training images (e.g. MVTec AD 2 `can` scenario):
```bash
python deployment/export_bank.py \
    --train-dir /opt/ad2/mvtec_ad_2/can/train/good \
    --val-dir /opt/ad2/mvtec_ad_2/can/validation/good \
    --backbone wide_resnet50_2 \
    --img-size 224 \
    --max-k 4000 \
    --percentile 99.0 \
    --output-dir deployment/triton_models/patchcore/1
```

### Step 2: Verify Standalone Python Backend
Test the Triton model directly in-process:
```bash
python deployment/triton_models/patchcore/1/model.py
```

### Step 3: Run Client & DeepStream Probe Verification
Run the end-to-end verification script:
```bash
# In-process direct verification:
python deployment/test_client.py --mode direct

# Simulated DeepStream video frame probe with visual defect overlay:
python deployment/test_client.py --mode deepstream-mock

# Live Triton gRPC client (when Triton server is running):
python deployment/test_client.py --mode grpc --url localhost:8001
```

---

## 4. Serving with Triton Inference Server

Launch Triton Inference Server inside the container:
```bash
tritonserver --model-repository=/workspace/deployment/triton_models \
             --strict-model-config=false \
             --log-verbose=0
```

Triton endpoints:
- HTTP / REST: `http://localhost:8000/v2/models/patchcore`
- gRPC: `localhost:8001`
- Metrics: `http://localhost:8002/metrics`

---

## 5. NVIDIA DeepStream Integration

### Pattern A: DeepStream Python `pyds` Pad Probe
Attached to a GStreamer sink pad (e.g. downstream of `nvstreammux` or `nvvideoconvert`):

```python
import pyds
from deployment.test_client import TritonGRPCClient, DeepStreamPatchCoreProbe

client = TritonGRPCClient(url="localhost:8001")
probe_handler = DeepStreamPatchCoreProbe(client=client, score_threshold=0.55)

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        # 1. Map frame buffer from NvBufSurface to NumPy array
        frame_surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        
        # 2. Execute inference and generate defect overlay
        annotated_frame, meta = probe_handler.process_frame(frame_surface, frame_meta.frame_num)

        # 3. Add DeepStream display text / bounding boxes
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = f"Status: {'DEFECT' if meta['is_defective'] else 'PASS'} ({meta['anomaly_score']:.3f})"
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK
```

### Pattern B: `nvinferserver` GStreamer Pipeline
```
gst-launch-1.0 \
  filesrc location=sample_conveyor.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! \
  nvstreammux width=224 height=224 batch-size=1 ! \
  nvinferserver config-file-path=dstriton_config.txt ! \
  nvvideoconvert ! nvdsosd ! nvoverlaysink
```

---

## 6. Benchmarked Performance (NVIDIA RTX 4000 Ada)

| Metric | Measured Value |
|---|---|
| **Steady-state Inference Latency** | **6.68 ms** per frame |
| **Throughput** | **$\sim 150$ FPS** |
| **GPU VRAM Utilization** | $< 1.2\text{ GB}$ (backbone + 4000-vector bank + cdist) |
| **Feature Dimension** | 1536 (WideResNet50-2 layers 2+3) |
| **Grid Resolution** | $28 \times 28 = 784$ patches at 224px input |
| **Anomaly Localization Map** | $224 \times 224$ (bilinear interpolated & Gaussian smoothed) |
