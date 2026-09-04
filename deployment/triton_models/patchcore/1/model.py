#!/usr/bin/env python3
"""Triton Python Backend for PatchCore Anomaly Detection.

This backend serves the complete PatchCore pipeline on NVIDIA Triton Inference Server:
1. Batched image input ingestion (supports float32 or uint8, NCHW or NHWC).
2. Deep feature extraction from backbone stages (layers 2+3 of CNN or ViT patch tokens)
   followed by 3x3 local average pooling.
3. GPU-accelerated k-NN minimum Euclidean distance computation against a pre-fitted
   coreset memory bank (~4,000 vectors).
4. Generation of:
   - ANOMALY_SCORE: Image-level anomaly score (scalar float32 per image)
   - ANOMALY_MAP: Pixel-level anomaly heatmap (1 x H x W float32 per image, smoothed)
   - IS_DEFECTIVE: Binary decision flag (boolean per image) vs calibrated threshold.

Compatible with NVIDIA DeepStream nvinferserver and Python gRPC/HTTP clients.
"""

import json
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# Attempt import of Triton Python backend utils; provide mock fallback for standalone
# offline testing and linting outside Triton container.
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    class _MockTritonUtils:
        class InferenceResponse:
            def __init__(self, output_tensors=None, error=None):
                self.output_tensors = output_tensors or []
                self._error = error

            def has_error(self):
                return self._error is not None

            def error(self):
                return self._error

        class InferenceRequest:
            def __init__(self, inputs=None):
                self._inputs = {inp.name(): inp for inp in (inputs or [])}

            def inputs(self):
                return list(self._inputs.values())

        class Tensor:
            def __init__(self, name, nparray):
                self._name = name
                self._nparray = nparray

            def name(self):
                return self._name

            def as_numpy(self):
                return self._nparray

        class TritonError(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                self._msg = msg

            def message(self):
                return self._msg

        class TritonModelException(Exception):
            pass

        @staticmethod
        def get_input_tensor_by_name(request, name):
            for inp in request.inputs():
                if inp.name() == name:
                    return inp
            return None

    pb_utils = _MockTritonUtils()

# Optional import of timm for feature extractor backbone
try:
    import timm
except ImportError:
    timm = None


class TritonPythonModel:
    """Triton Python Model implementation for PatchCore Anomaly Detection."""

    def initialize(self, args: dict):
        """Called once when the model is loaded into Triton.

        Loads metadata.json, pre-trained backbone model, and coreset memory bank.
        """
        # Determine model version directory
        model_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = args.get("model_repository", "")
        version = args.get("model_version", "1")
        if repo_dir and os.path.isdir(os.path.join(repo_dir, version)):
            model_dir = os.path.join(repo_dir, version)

        # Parse Triton model config if provided
        self.model_config = {}
        if "model_config" in args:
            try:
                self.model_config = json.loads(args["model_config"])
            except Exception:
                pass

        # Select computation device
        instance_device_id = args.get("model_instance_device_id", "0")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{instance_device_id}")
        else:
            self.device = torch.device("cpu")

        # Load metadata configuration
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            parent_meta = os.path.join(os.path.dirname(model_dir), "metadata.json")
            if os.path.exists(parent_meta):
                metadata_path = parent_meta

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            # Fallback sensible defaults if metadata is not yet exported
            self.metadata = {
                "backbone_name": "wide_resnet50_2",
                "backbone_kind": "cnn",
                "out_indices": [2, 3],
                "img_size": 224,
                "feature_dim": 1536,
                "threshold": 0.50,
                "gauss_sigma": 4.0,
                "eval_side": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }

        self.img_size = int(self.metadata.get("img_size", 224))
        self.backbone_name = self.metadata.get("backbone_name", "wide_resnet50_2")
        self.backbone_kind = self.metadata.get("backbone_kind", "cnn")
        self.out_indices = tuple(self.metadata.get("out_indices", [2, 3]))
        self.threshold = float(self.metadata.get("threshold", 0.50))
        self.gauss_sigma = float(self.metadata.get("gauss_sigma", 4.0))
        self.eval_side = int(self.metadata.get("eval_side", 224))

        # Setup normalization tensors
        mean = self.metadata.get("mean", [0.485, 0.456, 0.406])
        std = self.metadata.get("std", [0.229, 0.224, 0.225])
        self.mean_tensor = torch.tensor(mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

        # Pre-build 2D blur kernel for grid anomaly map smoothing
        k = int(2 * round(self.gauss_sigma) + 1)
        self.blur_k = k
        self.blur_kernel = (torch.ones(1, 1, k, k, device=self.device) / (k * k)).float()

        # Initialize backbone
        self._init_backbone(model_dir)

        # Load fitted coreset memory bank
        self._load_memory_bank(model_dir)

        print(
            f"[PatchCore Triton] Initialized successfully on {self.device}: "
            f"Backbone={self.backbone_name}, BankShape={self.bank.shape if self.bank is not None else None}, "
            f"Threshold={self.threshold:.4f}, EvalSide={self.eval_side}",
            flush=True,
        )

    def _init_backbone(self, model_dir: str):
        """Loads or instantiates the frozen backbone feature extractor."""
        if torch.cuda.is_available():
            # In CUDA 13 / CUDNN 9 containers where dynamic engine sublibraries may not be present,
            # disabling cuDNN allows PyTorch to use native high-performance CUDA convolution kernels.
            torch.backends.cudnn.enabled = False
        weights_file = os.path.join(model_dir, "backbone.pt")

        if timm is not None:
            if self.backbone_kind == "cnn":
                model = timm.create_model(
                    self.backbone_name,
                    pretrained=(not os.path.exists(weights_file)),
                    features_only=True,
                    out_indices=self.out_indices,
                )
                if os.path.exists(weights_file):
                    state = torch.load(weights_file, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)
            else:
                model = timm.create_model(
                    self.backbone_name,
                    pretrained=(not os.path.exists(weights_file)),
                    num_classes=0,
                    img_size=self.img_size,
                )
                if os.path.exists(weights_file):
                    state = torch.load(weights_file, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)
        else:
            # Standalone fallback when timm is not installed locally
            print("[PatchCore Triton] timm not found; initializing standard WideResNet50 feature emulation fallback.", flush=True)
            dim = int(self.metadata.get("feature_dim", 1536))
            dim2 = dim // 3  # 512
            dim3 = dim - dim2  # 1024

            class _FallbackWideResNet(torch.nn.Module):
                def __init__(self, d2, d3):
                    super().__init__()
                    self.stem = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    )
                    self.stage2 = torch.nn.Sequential(
                        torch.nn.Conv2d(64, d2, kernel_size=3, stride=2, padding=1),
                        torch.nn.ReLU(inplace=True),
                    )
                    self.stage3 = torch.nn.Sequential(
                        torch.nn.Conv2d(d2, d3, kernel_size=3, stride=2, padding=1),
                        torch.nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    x = self.stem(x)
                    f2 = self.stage2(x)
                    f3 = self.stage3(f2)
                    return [f2, f3]

            model = _FallbackWideResNet(dim2, dim3)

        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.backbone = model

    def _load_memory_bank(self, model_dir: str):
        """Loads the pre-fitted coreset memory bank vectors."""
        bank_pt = os.path.join(model_dir, "bank.pt")
        bank_npy = os.path.join(model_dir, "bank.npy")

        if os.path.exists(bank_pt):
            loaded = torch.load(bank_pt, map_location=self.device, weights_only=False)
            if isinstance(loaded, dict) and "bank" in loaded:
                self.bank = loaded["bank"].to(self.device).float()
            elif isinstance(loaded, torch.Tensor):
                self.bank = loaded.to(self.device).float()
            else:
                self.bank = torch.tensor(loaded, device=self.device, dtype=torch.float32)
        elif os.path.exists(bank_npy):
            arr = np.load(bank_npy)
            self.bank = torch.from_numpy(arr).to(self.device).float()
        else:
            # Standalone placeholder for initial startup if export_bank has not yet run
            dim = int(self.metadata.get("feature_dim", 1536))
            print(f"[PatchCore Triton] WARNING: No bank.pt found at {bank_pt}. Initializing dummy bank (100, {dim})", flush=True)
            self.bank = torch.randn(100, dim, device=self.device, dtype=torch.float32)

        # Ensure contiguous memory layout for optimal CUDA cdist
        self.bank = self.bank.contiguous()

    def _preprocess_input(self, np_img: np.ndarray) -> torch.Tensor:
        """Converts arbitrary client image arrays into normalized [B, 3, H, W] tensor."""
        t = torch.from_numpy(np_img)

        # Handle uint8 -> float32
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        elif t.dtype in (torch.float16, torch.float64):
            t = t.float()

        # Handle channel position: NHWC vs NCHW or HWC vs CHW
        if t.ndim == 3:
            # Could be [H, W, 3] or [3, H, W]
            if t.shape[-1] == 3:  # HWC
                t = t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            elif t.shape[0] == 3:  # CHW
                t = t.unsqueeze(0)  # [1, 3, H, W]
            else:
                raise ValueError(f"Unrecognized 3D image shape: {t.shape}")
        elif t.ndim == 4:
            # Could be [B, H, W, 3] or [B, 3, H, W]
            if t.shape[-1] == 3:  # NHWC
                t = t.permute(0, 3, 1, 2)  # [B, 3, H, W]
        else:
            raise ValueError(f"Expected 3D or 4D image tensor, got shape {t.shape}")

        t = t.to(self.device)

        # Resize to model input size if required
        if t.shape[-2] != self.img_size or t.shape[-1] != self.img_size:
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # Check if already normalized (e.g. negative values or values > 1.0)
        # If values are in [0, 1], apply ImageNet mean and std
        if t.min() >= 0.0 and t.max() <= 1.05:
            t = (t - self.mean_tensor) / self.std_tensor

        return t

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor):
        """Runs backbone feature extraction and 3x3 average pooling."""
        b = x.shape[0]
        if self.backbone_kind == "cnn":
            fs = self.backbone(x)
            ref = fs[0].shape[-2:]
            fs = [
                f if f.shape[-2:] == ref else F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                for f in fs
            ]
            fmap = torch.cat(fs, dim=1)
        else:
            toks = self.backbone.forward_features(x)
            n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            toks = toks[:, n_prefix:, :]
            g = int(round(toks.shape[1] ** 0.5))
            fmap = toks.transpose(1, 2).reshape(b, -1, g, g)

        # Local patch neighborhood aggregation: 3x3 avg pool with stride 1
        fmap = F.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
        b, c, h, w = fmap.shape
        # Permute to (B, H, W, C) and flatten to (B * H * W, C)
        patch_feats = fmap.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return patch_feats, b, h, w

    @torch.no_grad()
    def _compute_patch_distances(self, patch_feats: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Computes minimum Euclidean distance from each patch to the memory bank."""
        total_patches = patch_feats.shape[0]
        min_dists = []
        for i in range(0, total_patches, chunk_size):
            chunk = patch_feats[i : i + chunk_size]
            # dist shape: (chunk_size, memory_bank_size)
            dist = torch.cdist(chunk, self.bank)
            min_val = dist.min(dim=1).values
            min_dists.append(min_val)
        return torch.cat(min_dists)

    def execute(self, requests: List) -> List:
        """Triton batch execution callback.

        Processes incoming inference requests, computes anomaly metrics, and
        returns output tensors.
        """
        responses = []

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            if in_tensor is None:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Input tensor 'IMAGE' was not found in request")
                    )
                )
                continue

            try:
                np_img = in_tensor.as_numpy()
                x = self._preprocess_input(np_img)

                # Feature extraction + avg pooling
                patch_feats, b, h, w = self._extract_features(x)

                # Distance computation against coreset memory bank
                min_distances = self._compute_patch_distances(patch_feats)
                grid_scores = min_distances.view(b, h, w)

                # 1. Image-level score: maximum patch anomaly distance
                image_scores = grid_scores.view(b, -1).max(dim=1).values  # [B]

                # 2. Defect classification decision against threshold
                is_defective = (image_scores > self.threshold).cpu().numpy().astype(bool).reshape(b, 1)

                # 3. Anomaly localization heatmap: smoothed grid map interpolated to eval resolution
                grid_maps = grid_scores.unsqueeze(1)  # [B, 1, h, w]
                pad_w = self.blur_k // 2
                padded = F.pad(grid_maps, (pad_w, pad_w, pad_w, pad_w), mode="replicate")
                smoothed = F.conv2d(padded, self.blur_kernel)
                anomaly_maps = F.interpolate(
                    smoothed, size=(self.eval_side, self.eval_side), mode="bilinear", align_corners=False
                )  # [B, 1, eval_side, eval_side]

                # Prepare numpy outputs
                scores_np = image_scores.unsqueeze(-1).cpu().numpy().astype(np.float32)  # [B, 1]
                maps_np = anomaly_maps.cpu().numpy().astype(np.float32)  # [B, 1, eval_side, eval_side]

                out_score = pb_utils.Tensor("ANOMALY_SCORE", scores_np)
                out_map = pb_utils.Tensor("ANOMALY_MAP", maps_np)
                out_defect = pb_utils.Tensor("IS_DEFECTIVE", is_defective)

                response = pb_utils.InferenceResponse(output_tensors=[out_score, out_map, out_defect])
                responses.append(response)

            except Exception as e:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(f"PatchCore inference failed: {str(e)}")))

        return responses

    def finalize(self):
        """Called when Triton unloads the model."""
        print("[PatchCore Triton] Finalizing model and freeing GPU memory...", flush=True)
        if hasattr(self, "backbone"):
            del self.backbone
        if hasattr(self, "bank"):
            del self.bank
        if hasattr(self, "blur_kernel"):
            del self.blur_kernel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Module-level self-test runner
if __name__ == "__main__":
    print("Executing standalone PatchCore Triton Model self-test...")
    model = TritonPythonModel()
    mock_args = {
        "model_repository": os.path.dirname(os.path.abspath(__file__)),
        "model_version": "1",
        "model_instance_device_id": "0",
    }
    model.initialize(mock_args)

    # Test with synthetic input tensor [1, 3, 224, 224]
    dummy_input = np.random.uniform(0.0, 1.0, (1, 3, 224, 224)).astype(np.float32)
    req = pb_utils.InferenceRequest(inputs=[pb_utils.Tensor("IMAGE", dummy_input)])
    resps = model.execute([req])

    assert len(resps) == 1, "Expected 1 response"
    assert not resps[0].has_error(), f"Inference failed with error: {resps[0].error().message()}"

    tensors = {t.name(): t.as_numpy() for t in resps[0].output_tensors}
    print("Self-test passed! Generated output tensors:")
    for k, v in tensors.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")
    model.finalize()
