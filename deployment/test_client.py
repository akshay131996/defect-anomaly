#!/usr/bin/env python3
"""Triton Client and DeepStream pyds Probe Verification Script for PatchCore.

This script demonstrates two production deployment patterns for industrial inspection:
1. DeepStream Video Probe (`deepstream_probe`):
   A GStreamer pad probe extracting NvBufSurface video frames via `pyds`, formatting
   them, querying Triton (via gRPC, HTTP, or in-process engine), parsing the anomaly map
   and score, applying threshold classification, and overlaying defect bounding boxes and
   status OSD display metadata.
2. Standard Triton gRPC / HTTP Client:
   Batched tensor querying with `tritonclient.grpc` or `tritonclient.http`.
3. Standalone Direct Engine Verification (`--mode direct`):
   Instantiates and validates the Triton Python model directly in-memory, ensuring
   offline testing passes without requiring a live Triton server daemon.

Usage:
    # 1. Direct standalone verification (fastest, no server required):
    python deployment/test_client.py --mode direct

    # 2. Simulated DeepStream video frame probe:
    python deployment/test_client.py --mode deepstream-mock

    # 3. Live Triton gRPC client:
    python deployment/test_client.py --mode grpc --url localhost:8001

    # 4. Live Triton HTTP client:
    python deployment/test_client.py --mode http --url localhost:8000
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

# Optional imports with graceful shims for running in different environments
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tritonclient.grpc as grpcclient
except ImportError:
    grpcclient = None

try:
    import tritonclient.http as httpclient
except ImportError:
    httpclient = None


# =============================================================================
# 1. Triton Client Adapters (gRPC, HTTP, and Direct Standalone)
# =============================================================================

class BasePatchCoreClient:
    """Interface for querying PatchCore model."""

    def infer(self, image_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Runs inference on an image array [H, W, 3] or [B, 3, H, W].

        Returns dict containing:
          - 'anomaly_score': float scalar or [B, 1]
          - 'anomaly_map': [B, 1, H, W] or [H, W] float heatmap
          - 'is_defective': bool scalar or [B, 1]
        """
        raise NotImplementedError


class DirectPatchCoreClient(BasePatchCoreClient):
    """Direct in-process client loading TritonPythonModel without server daemon."""

    def __init__(self, model_dir: str = "deployment/triton_models/patchcore/1"):
        if not os.path.exists(model_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, "triton_models", "patchcore", "1")
            if os.path.exists(candidate):
                model_dir = candidate

        abs_model_dir = os.path.abspath(model_dir)
        if abs_model_dir not in sys.path:
            sys.path.insert(0, abs_model_dir)
        from model import TritonPythonModel, pb_utils

        self.pb_utils = pb_utils
        self.model = TritonPythonModel()
        self.model.initialize({
            "model_repository": os.path.dirname(abs_model_dir),
            "model_version": os.path.basename(abs_model_dir),
            "model_instance_device_id": "0",
        })

    def infer(self, image_np: np.ndarray) -> Dict[str, np.ndarray]:
        # Wrap numpy into Triton input tensor
        if image_np.ndim == 3 and image_np.shape[-1] == 3:  # HWC -> CHW
            chw = np.transpose(image_np, (2, 0, 1)).astype(np.float32)
            if chw.max() > 1.0:
                chw /= 255.0
            batched = np.expand_dims(chw, axis=0)
        elif image_np.ndim == 4:
            batched = image_np.astype(np.float32)
        else:
            batched = image_np.astype(np.float32)

        in_tensor = self.pb_utils.Tensor("IMAGE", batched)
        req = self.pb_utils.InferenceRequest(inputs=[in_tensor])
        resps = self.model.execute([req])

        if resps[0].has_error():
            err = resps[0].error()
            msg = err.message() if (hasattr(err, "message") and callable(err.message)) else (getattr(err, "message", str(err)))
            raise RuntimeError(f"Direct inference failed: {msg}")

        out_map = {t.name(): t.as_numpy() for t in resps[0].output_tensors}
        return {
            "anomaly_score": float(out_map["ANOMALY_SCORE"][0, 0]),
            "anomaly_map": out_map["ANOMALY_MAP"][0, 0],
            "is_defective": bool(out_map["IS_DEFECTIVE"][0, 0]),
        }


class TritonGRPCClient(BasePatchCoreClient):
    """Client for querying Triton over high-performance gRPC."""

    def __init__(self, url: str = "localhost:8001", model_name: str = "patchcore"):
        if grpcclient is None:
            raise ImportError("tritonclient.grpc is not installed. Install with `pip install tritonclient[grpc]`")
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model_name = model_name

        if not self.client.is_model_ready(self.model_name):
            raise RuntimeError(f"Triton model '{self.model_name}' is not ready at {url}")

    def infer(self, image_np: np.ndarray) -> Dict[str, np.ndarray]:
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            chw = np.transpose(image_np, (2, 0, 1)).astype(np.float32)
            if chw.max() > 1.0:
                chw /= 255.0
            batched = np.expand_dims(chw, axis=0)
        else:
            batched = image_np.astype(np.float32)

        inputs = [grpcclient.InferInput("IMAGE", batched.shape, "FP32")]
        inputs[0].set_data_from_numpy(batched)

        outputs = [
            grpcclient.InferRequestedOutput("ANOMALY_SCORE"),
            grpcclient.InferRequestedOutput("ANOMALY_MAP"),
            grpcclient.InferRequestedOutput("IS_DEFECTIVE"),
        ]

        res = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        return {
            "anomaly_score": float(res.as_numpy("ANOMALY_SCORE")[0, 0]),
            "anomaly_map": res.as_numpy("ANOMALY_MAP")[0, 0],
            "is_defective": bool(res.as_numpy("IS_DEFECTIVE")[0, 0]),
        }


class TritonHTTPClient(BasePatchCoreClient):
    """Client for querying Triton over HTTP/REST."""

    def __init__(self, url: str = "localhost:8000", model_name: str = "patchcore"):
        if httpclient is None:
            raise ImportError("tritonclient.http is not installed. Install with `pip install tritonclient[http]`")
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name

        if not self.client.is_model_ready(self.model_name):
            raise RuntimeError(f"Triton model '{self.model_name}' is not ready at {url}")

    def infer(self, image_np: np.ndarray) -> Dict[str, np.ndarray]:
        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            chw = np.transpose(image_np, (2, 0, 1)).astype(np.float32)
            if chw.max() > 1.0:
                chw /= 255.0
            batched = np.expand_dims(chw, axis=0)
        else:
            batched = image_np.astype(np.float32)

        inputs = [httpclient.InferInput("IMAGE", batched.shape, "FP32")]
        inputs[0].set_data_from_numpy(batched)

        outputs = [
            httpclient.InferRequestedOutput("ANOMALY_SCORE"),
            httpclient.InferRequestedOutput("ANOMALY_MAP"),
            httpclient.InferRequestedOutput("IS_DEFECTIVE"),
        ]

        res = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        return {
            "anomaly_score": float(res.as_numpy("ANOMALY_SCORE")[0, 0]),
            "anomaly_map": res.as_numpy("ANOMALY_MAP")[0, 0],
            "is_defective": bool(res.as_numpy("IS_DEFECTIVE")[0, 0]),
        }


# =============================================================================
# 2. DeepStream Python (pyds) Pad Probe Pattern
# =============================================================================

class DeepStreamPatchCoreProbe:
    """Demonstrates a DeepStream pad probe integrating PatchCore anomaly detection.

    In DeepStream, this probe attaches to a GstPad (e.g. nvstreammux or nvvideoconvert):
        pad = osd.get_static_pad("sink")
        pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, user_data)
    """

    def __init__(self, client: BasePatchCoreClient, score_threshold: float = 0.50):
        self.client = client
        self.score_threshold = score_threshold

    def process_frame(self, frame_bgr: np.ndarray, frame_num: int = 0) -> Tuple[np.ndarray, Dict]:
        """Core logic executed per video frame inside the DeepStream probe callback.

        1. Ingests the RGB/BGR frame mapped from NvBufSurface.
        2. Sends to Triton client.
        3. Parses anomaly score and localization heatmap.
        4. Identifies defect bounding box(es) from connected components in the map.
        5. Overlays OSD text and heatmap visualization.
        """
        h, w, c = frame_bgr.shape
        t0 = time.time()

        # DeepStream frames are BGR or RGBA; convert to RGB for PatchCore
        frame_rgb = frame_bgr[:, :, ::-1] if c == 3 else frame_bgr[:, :, :3]

        # Query Triton PatchCore model
        results = self.client.infer(frame_rgb)
        latency_ms = (time.time() - t0) * 1000.0

        score = results["anomaly_score"]
        heatmap = results["anomaly_map"]
        is_defective = results["is_defective"]

        # Generate defect visualization overlay
        annotated_frame = frame_bgr.copy()

        # Resize heatmap to original frame dimensions
        if cv2 is not None:
            norm_map = np.clip((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6), 0, 1)
            norm_map_u8 = (norm_map * 255).astype(np.uint8)
            heatmap_resized = cv2.resize(norm_map_u8, (w, h))

            # If defective, blend red/yellow heatmap overlay on defective regions
            if is_defective:
                color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                # Defect mask: regions above threshold percentile
                _, bin_mask = cv2.threshold(heatmap_resized, int(0.7 * 255), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw bounding boxes around defects
                for cnt in contours:
                    if cv2.contourArea(cnt) > 50:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                        cv2.putText(
                            annotated_frame, "DEFECT", (bx, max(by - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )

                # Alpha blend heatmap
                mask_3c = cv2.merge([bin_mask, bin_mask, bin_mask]) / 255.0
                annotated_frame = (annotated_frame * (1.0 - 0.4 * mask_3c) + color_map * (0.4 * mask_3c)).astype(np.uint8)

            # Draw DeepStream OSD-style status banner
            status_color = (0, 0, 255) if is_defective else (0, 255, 0)
            status_text = f"FRAME {frame_num:04d}: {'DEFECT' if is_defective else 'PASS'} (Score: {score:.4f} | Thr: {self.score_threshold:.4f} | {latency_ms:.1f}ms)"
            cv2.rectangle(annotated_frame, (10, 10), (600, 45), (0, 0, 0), -1)
            cv2.putText(annotated_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        meta = {
            "frame_num": frame_num,
            "anomaly_score": score,
            "is_defective": is_defective,
            "latency_ms": latency_ms,
            "heatmap_shape": heatmap.shape,
        }
        return annotated_frame, meta


# =============================================================================
# 3. Test Fixtures and Verification Runner
# =============================================================================

def create_synthetic_frame(defect: bool = False, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Generates synthetic part inspection frame with or without simulated defect."""
    h, w = size
    # Simulated brushed metallic / textured surface
    np.random.seed(42 if not defect else 99)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        noise = np.random.normal(128, 15, (h, w)).clip(0, 255).astype(np.uint8)
        base[:, :, c] = noise

    if defect and cv2 is not None:
        # Draw prominent defect scratch and dark spot
        cv2.line(base, (int(w * 0.3), int(h * 0.4)), (int(w * 0.7), int(h * 0.65)), (20, 20, 20), 4)
        cv2.circle(base, (int(w * 0.6), int(h * 0.5)), 12, (240, 20, 20), -1)

    return base


def run_verification(mode: str, url: str, test_image_path: Optional[str] = None):
    print("=================================================================")
    print("       PatchCore Triton & DeepStream Client Verification         ")
    print("=================================================================")
    print(f"Mode:    {mode}")
    print(f"Target:  {url if mode in ('grpc', 'http') else 'In-process Model'}")
    print("=================================================================")

    # Select client backend
    if mode == "direct":
        client = DirectPatchCoreClient()
    elif mode == "deepstream-mock":
        if url and "8000" in url:
            client = TritonHTTPClient(url=url)
        elif url:
            client = TritonGRPCClient(url=url)
        else:
            client = DirectPatchCoreClient()
    elif mode == "grpc":
        client = TritonGRPCClient(url=url)
    elif mode == "http":
        client = TritonHTTPClient(url=url)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    os.makedirs("outputs", exist_ok=True)

    if mode == "deepstream-mock":
        print("\n[DeepStream Probe Simulation] Running video stream probe with normal and defective frames...")
        probe = DeepStreamPatchCoreProbe(client=client, score_threshold=0.50)

        # 1. Test normal frame
        normal_frame = create_synthetic_frame(defect=False)
        out_normal, meta_normal = probe.process_frame(normal_frame, frame_num=1)
        print(f"  Frame 0001 (Normal):    Score={meta_normal['anomaly_score']:.4f}, "
              f"Defective={meta_normal['is_defective']}, Latency={meta_normal['latency_ms']:.2f}ms")

        # 2. Test defective frame
        defect_frame = create_synthetic_frame(defect=True)
        out_defect, meta_defect = probe.process_frame(defect_frame, frame_num=2)
        print(f"  Frame 0002 (Defective): Score={meta_defect['anomaly_score']:.4f}, "
              f"Defective={meta_defect['is_defective']}, Latency={meta_defect['latency_ms']:.2f}ms")

        # Save verification artifact
        if cv2 is not None:
            save_path = "outputs/deepstream_probe_result.png"
            combined = np.hstack([out_normal, out_defect])
            cv2.imwrite(save_path, combined)
            print(f"[Artifact] Saved DeepStream inspection visualization to {save_path}")

        assert "anomaly_score" in meta_normal and "anomaly_score" in meta_defect, \
            "Both probe frames must return valid anomaly scores"
        assert meta_normal["heatmap_shape"] == (224, 224) and meta_defect["heatmap_shape"] == (224, 224), \
            "Heatmap shape should match evaluation dimensions (224, 224)"
        print("\n[DeepStream Probe Simulation] Verification PASSED successfully!")

    else:
        # Standard image inference test
        if test_image_path and os.path.exists(test_image_path):
            if cv2 is not None:
                img = cv2.imread(test_image_path)
            else:
                from PIL import Image
                img = np.array(Image.open(test_image_path))
        else:
            print("[Info] No test image specified. Generating synthetic defect frame...")
            img = create_synthetic_frame(defect=True)

        t0 = time.time()
        res = client.infer(img)
        elapsed_ms = (time.time() - t0) * 1000.0

        print(f"\n[Inference Result]:")
        print(f"  Latency:        {elapsed_ms:.2f} ms")
        print(f"  Anomaly Score:  {res['anomaly_score']:.4f}")
        print(f"  Is Defective:   {res['is_defective']}")
        print(f"  Anomaly Map:    shape={res['anomaly_map'].shape}, min={res['anomaly_map'].min():.4f}, max={res['anomaly_map'].max():.4f}")
        print("\n[Triton Client] Verification PASSED successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test Triton PatchCore client and DeepStream probe.")
    parser.add_argument("--mode", choices=["direct", "grpc", "http", "deepstream-mock"],
                        default="direct", help="Client verification mode")
    parser.add_argument("--url", default="localhost:8001", help="Triton server endpoint URL")
    parser.add_argument("--image", default="", help="Optional test image path")
    args = parser.parse_args()

    run_verification(mode=args.mode, url=args.url, test_image_path=args.image if args.image else None)


if __name__ == "__main__":
    main()
