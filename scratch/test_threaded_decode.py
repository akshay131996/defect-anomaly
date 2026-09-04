#!/usr/bin/env python3
"""Verification test script for parallel threaded decode in data_efficiency.py."""
import os
import sys
import numpy as np
import torch
from PIL import Image

# Add current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_efficiency as de


def main():
    print("Testing parallel threaded decode vs sequential decode...")
    rng = np.random.default_rng(12345)

    # 1. Create a batch of synthetic PIL images with varied dimensions
    batch_size = 8
    images = []
    for i in range(batch_size):
        w = rng.integers(256, 512)
        h = rng.integers(256, 512)
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))

    # 2. Instantiate PatchExtractor from data_efficiency
    extractor = de.PatchExtractor()

    # 3. Test transform: Sequential vs ThreadPoolExecutor (_POOL.map)
    print("Running sequential transforms...")
    x_seq = torch.stack([extractor.tfm(im.convert("RGB")) for im in images]).to(de.DEVICE)

    print("Running threaded transforms using _POOL.map...")
    x_thread = torch.stack(list(de._POOL.map(extractor._one, images))).to(de.DEVICE)

    # 4. Check numerical equivalence
    tfm_diff = (x_seq - x_thread).abs().max().item()
    print(f"Transform tensor max abs diff: {tfm_diff}")
    assert torch.allclose(x_seq, x_thread), f"Transforms not allclose, max diff: {tfm_diff}"
    assert tfm_diff == 0.0, f"Expected bit-identical transforms (0.0 diff), got {tfm_diff}"
    print("Transform numerical equivalence: PASSED (bit-identical, diff = 0.0)")

    # 5. Test full PatchExtractor call: forward_feats(x_seq) vs extractor(images)
    print("Running full forward_feats and PatchExtractor.__call__...")
    with torch.no_grad():
        out_seq = extractor.forward_feats(x_seq)
        out_thread = extractor(images)

    full_diff = (out_seq - out_thread).abs().max().item()
    print(f"Full extractor output max abs diff: {full_diff}")
    assert torch.allclose(out_seq, out_thread), f"Full outputs not allclose, max diff: {full_diff}"
    assert full_diff == 0.0, f"Expected bit-identical outputs (0.0 diff), got {full_diff}"
    print("Full PatchExtractor equivalence: PASSED (bit-identical, diff = 0.0)")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
