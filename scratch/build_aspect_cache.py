import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

sys.path.insert(0, "/workspace")
import ad2_pixel_eval as ape

SCENARIOS = [
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"
]

AD2_ROOT = "/opt/ad2/mvtec_ad_2"
CACHE_ROOT = "/opt/ad2/cache_aspect448"

def resize_one(args):
    src_path, dst_path, target_size = args
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return dst_path
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with Image.open(src_path) as im:
        rgb = im.convert("RGB")
        resized = rgb.resize(target_size, resample=Image.BICUBIC)
        resized.save(dst_path, format="PNG")
    return dst_path

def main():
    print(f"Building pre-resized aspect cache at {CACHE_ROOT}...")
    tasks = []
    
    for scen in SCENARIOS:
        train, val, good, bad, gt_dir = ape.load_paths(scen)
        all_imgs = train + val + good + bad
        if not all_imgs:
            print(f"Warning: no images found for {scen}")
            continue
            
        with Image.open(all_imgs[0]) as sample:
            w_nat, h_nat = sample.size
        w_in, h_in = ape.aspect_dimensions(w_nat, h_nat, target_img=448, stride=32)
        print(f"Scenario {scen:12s}: {len(all_imgs):4d} images, native ({w_nat:4d}, {h_nat:4d}) -> target ({w_in:4d}, {h_in:4d})")
        
        for src in all_imgs:
            rel = os.path.relpath(src, AD2_ROOT)
            dst = os.path.join(CACHE_ROOT, rel)
            tasks.append((src, dst, (w_in, h_in)))
            
    print(f"Total images to cache: {len(tasks)}")
    
    import time
    t0 = time.time()
    n_workers = min(16, (os.cpu_count() or 8))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(resize_one, tasks))
    elapsed = time.time() - t0
    
    print(f"Done! Cached {len(results)} images in {elapsed:.1f}s ({len(results)/elapsed:.1f} img/s)")
    
    # Check total size of cache
    total_bytes = sum(os.path.getsize(p) for p in results)
    print(f"Cache disk footprint: {total_bytes / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
