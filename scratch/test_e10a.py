import os
import sys
import torch
from PIL import Image

sys.path.insert(0, "/workspace")
import sweep_backbones as sb
import ad2_pixel_eval as ape

def main():
    print("Testing E10a: 384-dim random projection on scenario 'can'...")
    scen = "can"
    train, val, good, bad, gt_dir = ape.load_paths(scen)
    
    with Image.open(train[0]) as sample:
        w_nat, h_nat = sample.size
    w_in, h_in = ape.aspect_dimensions(w_nat, h_nat, target_img=448, stride=32)
    
    # 1. Arm without projection (1536 dim)
    spec_full = {"name": "wide_resnet50_2", "kind": "cnn", "out_indices": (2, 3), "img": 448}
    ex_full = sb.PatchExtractor(spec_full)
    ex_full = ape.aspect_transform(ex_full, w_in, h_in)
    
    # 2. Arm with 384-dim projection
    spec_proj = dict(spec_full)
    spec_proj["proj_dim"] = 384
    ex_proj = sb.PatchExtractor(spec_proj)
    ex_proj = ape.aspect_transform(ex_proj, w_in, h_in)
    
    # Initialize projection matrix on CPU with fixed seed then move to DEVICE
    g = torch.Generator().manual_seed(0)
    P = (torch.randn(1536, 384, generator=g) / (384 ** 0.5)).to(sb.DEVICE)
    ex_proj.proj = P
    ex_proj.proj_dim = 384
    
    # Monkey patch forward_feats for ex_proj
    orig_forward = ex_proj.forward_feats
    def forward_with_proj(x):
        # Call model up to avg_pool2d
        fs = ex_proj.model(x)
        ref = fs[0].shape[-2:]
        fs = [f if f.shape[-2:] == ref else
              torch.nn.functional.interpolate(f, size=ref, mode="bilinear", align_corners=False)
              for f in fs]
        fmap = torch.cat(fs, dim=1)
        fmap = torch.nn.functional.avg_pool2d(fmap, kernel_size=3, stride=1, padding=1)
        b, c, h, w = fmap.shape
        ex_proj.grid = (h, w)
        flat = fmap.permute(0, 2, 3, 1).reshape(b * h * w, c)
        flat_proj = flat @ ex_proj.proj
        ex_proj.dim = 384
        return flat_proj.cpu()
    ex_proj.forward_feats = forward_with_proj
    
    # Test on a batch
    sample_imgs = [Image.open(p).convert("RGB") for p in train[:4]]
    x = torch.stack([ex_full._one(im) for im in sample_imgs]).to(sb.DEVICE)
    
    with torch.no_grad():
        f_full = ex_full.forward_feats(x)
        f_proj = ex_proj.forward_feats(x)
        
    print(f"Full feature shape: {f_full.shape} (dim = {f_full.shape[1]})")
    print(f"Projected feature shape: {f_proj.shape} (dim = {f_proj.shape[1]})")
    
    expected_proj = (f_full.to(sb.DEVICE) @ P).cpu()
    diff = (f_proj - expected_proj).abs().max().item()
    print(f"Max abs diff between forward_feats projection and explicit f @ P: {diff}")
    assert diff < 1e-5, "Projection mismatch!"
    print("SUCCESS: 384-dim projection verified!")

if __name__ == "__main__":
    main()
