import os
import sys
import glob
import torch
from PIL import Image

sys.path.insert(0, "/workspace")
import sweep_backbones as sb
import ad2_pixel_eval as ape

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
CACHE_ROOT = "/opt/ad2/cache_aspect448"

def main():
    print("Testing bit-identical feature extraction between live-decode and /opt/ad2/cache_aspect448...")
    scenarios = ["can", "fabric", "fruit_jelly", "sheet_metal", "vial", "walnuts"]
    
    spec = {"name": "wide_resnet50_2", "kind": "cnn", "out_indices": (2, 3), "img": 448}
    ex = sb.PatchExtractor(spec)
    
    max_feat_diff = 0.0
    total_tested = 0
    
    for sc in scenarios:
        train, val, good, bad, gt_dir = ape.load_paths(sc)
        sample_paths = (train[:3] + good[:2] + bad[:2])
        
        with Image.open(sample_paths[0]) as sample:
            w_nat, h_nat = sample.size
        w_in, h_in = ape.aspect_dimensions(w_nat, h_nat, target_img=448, stride=32)
        ex = ape.aspect_transform(ex, w_in, h_in)
        
        for p in sample_paths:
            rel = os.path.relpath(p, AD2_ROOT)
            cached_p = os.path.join(CACHE_ROOT, rel)
            assert os.path.isfile(cached_p), f"Missing cache file: {cached_p}"
            
            # Live decode & transform
            im_live = Image.open(p).convert("RGB")
            t_live = ex._one(im_live).unsqueeze(0).to(sb.DEVICE)
            with torch.no_grad():
                f_live = ex.forward_feats(t_live)
                
            # Cached decode & transform
            im_cached = Image.open(cached_p).convert("RGB")
            t_cached = ex._one(im_cached).unsqueeze(0).to(sb.DEVICE)
            with torch.no_grad():
                f_cached = ex.forward_feats(t_cached)
                
            diff = (f_live - f_cached).abs().max().item()
            if diff > max_feat_diff:
                max_feat_diff = diff
            total_tested += 1
            
        print(f"Scenario {sc:<12}: {len(sample_paths)} images tested, local max diff = {diff}")
        
    print(f"\nTotal images verified across scenarios: {total_tested}")
    print(f"Dataset-wide max feature difference: {max_feat_diff}")
    
    if max_feat_diff == 0.0:
        print("SUCCESS: 100% BIT-IDENTICAL PARITY CONFIRMED (0.0 diff)!")
    else:
        print(f"FAILURE: Features differed by {max_feat_diff}")
        sys.exit(1)

if __name__ == "__main__":
    main()
