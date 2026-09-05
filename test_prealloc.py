#!/usr/bin/env python3
"""Verify that preallocated extract_paths produces bit-identical results to torch.cat."""
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import sweep_backbones as sb

def extract_paths_prealloc(ex, paths, pool, batch=4):
    from PIL import Image
    n_total = len(paths)
    sample_imgs = list(pool.map(lambda p: Image.open(p).convert("RGB"), paths[:batch]))
    sample_x = torch.stack(list(pool.map(ex._one, sample_imgs))).to(sb.DEVICE)
    with torch.no_grad():
        f0 = ex.forward_feats(sample_x)
    
    patches_per_img = f0.shape[0] // len(sample_imgs)
    dim = f0.shape[1]
    total_patches = n_total * patches_per_img
    
    feats = torch.empty((total_patches, dim), dtype=torch.float32)
    feats[:f0.shape[0]] = f0
    del sample_imgs, sample_x, f0
    
    curr = batch
    while curr < n_total:
        b_paths = paths[curr:curr + batch]
        imgs = list(pool.map(lambda p: Image.open(p).convert("RGB"), b_paths))
        x = torch.stack(list(pool.map(ex._one, imgs))).to(sb.DEVICE)
        with torch.no_grad():
            f = ex.forward_feats(x)
        feats[curr * patches_per_img : (curr + len(b_paths)) * patches_per_img] = f
        del imgs, x, f
        curr += batch
    return feats

def main():
    arm = {"tag": "test_arm", "kind": "cnn", "name": "wide_resnet50_2", "img": 224, "out_indices": (2, 3)}
    ex = sb.PatchExtractor(arm)
    pool = ThreadPoolExecutor(max_workers=2)
    
    # Create 7 synthetic test images on disk
    os.makedirs("scratch/test_imgs", exist_ok=True)
    paths = []
    for i in range(7):
        p = f"scratch/test_imgs/img_{i}.png"
        img = Image.new("RGB", (224, 224), color=(i * 30, (i * 45) % 255, (i * 60) % 255))
        img.save(p)
        paths.append(p)
        
    f_orig = sb.extract_paths(ex, paths, pool, batch=3)
    f_pre = extract_paths_prealloc(ex, paths, pool, batch=3)
    
    diff = (f_orig - f_pre).abs().max().item()
    print(f"Max abs diff: {diff}")
    assert diff == 0.0, f"Expected bit-identical 0.0 diff, got {diff}"
    print("Prealloc bit-identical parity: PASS")

if __name__ == "__main__":
    main()
