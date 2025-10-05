#!/usr/bin/env python
"""
Detect craniotomy region by comparing two textures (original vs outpaint/edited) mapped on the brain surface.

Outputs (in --outdir, default data/):
  - craniotomy_region_texture.npz
      indices        : brain vertex indices inside detected opening
      threshold      : numeric threshold applied on diff map
      coverage       : ratio (#indices / total_vertices)
      method         : 'texture_diff'
      stats          : dict of diff statistics
  - craniotomy_region_texture_meta.json (same info + file references)

Workflow:
  1. Load texture_ref (e.g. texture.png) and texture_alt (e.g. texture_outpaintr.png)
  2. Compute diff map (default mean absolute RGB or optional CIE Lab ΔE if --use-lab and skimage available).
  3. Smooth (Gaussian, sigma) and threshold (percentile or Otsu) to get binary pixel mask.
  4. Keep largest connected component (optional) and optionally morphologically clean.
  5. Load brain surface OBJ, parse UVs; assign each vertex a UV (first occurrence) and sample mask.
  6. Export vertex indices whose UV hit the mask.

Usage:
  python tools/detect_craniotomy_from_textures.py \
      --brain-surface data/surface_full_decimated.obj \
      --texture-ref data/texture.png \
      --texture-alt data/texture_outpaint.png \
      --outdir data

Options:
  --percentile 90        Threshold percentile (ignored if --otsu)
  --gauss-sigma 2.0      Gaussian sigma for diff smoothing
  --use-lab              Use Lab ΔE (requires skimage)
  --otsu                 Use Otsu threshold instead of percentile (requires skimage or fallback simple)
  --keep-multiple        Do not reduce to largest component (keep all)
  --save-diff            Save diff_npy and mask PNG

Limitations: UV seams assign first seen UV only; if you need full seam aggregation, extend mapping logic.
"""
from __future__ import annotations
import argparse, os, json, sys
import numpy as np
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# ---------------- OBJ loader (positions + UVs) -----------------

def load_obj_with_uv(path: str):
    verts: List[List[float]] = []
    uvs: List[List[float]] = []
    faces: List[List[Tuple[int,int|None]]] = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): continue
            sp = line.strip().split()
            if not sp: continue
            if sp[0] == 'v' and len(sp) >= 4:
                try:
                    verts.append([float(sp[1]), float(sp[2]), float(sp[3])])
                except ValueError: pass
            elif sp[0] == 'vt' and len(sp) >= 3:
                try:
                    uvs.append([float(sp[1]), float(sp[2])])
                except ValueError: pass
            elif sp[0] == 'f' and len(sp) >= 4:
                toks = sp[1:]
                tri = []
                for tk in toks:
                    seg = tk.split('/')
                    try:
                        vi = int(seg[0]);  vi = vi-1 if vi>0 else len(verts)+vi
                    except: continue
                    ti = None
                    if len(seg) > 1 and seg[1] != '':
                        try:
                            ti = int(seg[1]); ti = ti-1 if ti>0 else len(uvs)+ti
                        except: ti = None
                    tri.append((vi, ti))
                if len(tri) >= 3:
                    for i in range(1, len(tri)-1):
                        faces.append([tri[0], tri[i], tri[i+1]])
    V = np.array(verts, np.float32)
    UV = np.array(uvs, np.float32) if uvs else np.zeros((0,2), np.float32)
    # Assign first seen UV per vertex
    v2uv = np.zeros((V.shape[0],2), np.float32)
    seen = np.zeros(V.shape[0], dtype=bool)
    for face in faces:
        for (vi, ti) in face:
            if ti is not None and not seen[vi] and ti < UV.shape[0]:
                v2uv[vi] = UV[ti]; seen[vi] = True
    return V, v2uv

# ---------------- Diff computation -----------------

def load_texture(path: str):
    if Image is None:
        raise RuntimeError('Pillow not installed')
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0

def compute_diff(texA: np.ndarray, texB: np.ndarray, use_lab: bool):
    if texA.shape != texB.shape:
        raise ValueError('Texture size mismatch')
    if use_lab:
        try:
            from skimage.color import rgb2lab  # type: ignore
            labA = rgb2lab(texA)
            labB = rgb2lab(texB)
            diff = np.linalg.norm(labA - labB, axis=2)
        except Exception:
            diff = np.mean(np.abs(texA - texB), axis=2)
    else:
        diff = np.mean(np.abs(texA - texB), axis=2)
    return diff

# ---------------- Thresholding -----------------

def smooth_gaussian(arr: np.ndarray, sigma: float):
    if sigma <= 0: return arr
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(arr, sigma)
    except Exception:
        return arr

def otsu_threshold(arr: np.ndarray):
    try:
        from skimage.filters import threshold_otsu  # type: ignore
        return float(threshold_otsu(arr))
    except Exception:
        # Simple fallback: midpoint between med and 90th percentile
        med = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        return (med + p90) * 0.5


def largest_component(mask: np.ndarray):
    try:
        from scipy.ndimage import label  # type: ignore
        lab, n = label(mask)
        if n <= 1: return mask
        sizes = [(lab == i).sum() for i in range(1, n+1)]
        keep = 1 + int(np.argmax(sizes))
        return (lab == keep)
    except Exception:
        return mask

# ---------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description='Detect craniotomy via texture differences.')
    ap.add_argument('--brain-surface', default='data/surface_full_decimated.obj')
    ap.add_argument('--texture-ref', default='data/texture.png')
    ap.add_argument('--texture-alt', default='data/texture_outpaint.png')
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--percentile', type=float, default=90.0, help='High percentile for mode=changed')
    ap.add_argument('--low-percentile', type=float, default=10.0, help='Low percentile for mode=common')
    ap.add_argument('--gauss-sigma', type=float, default=2.0)
    ap.add_argument('--use-lab', action='store_true')
    ap.add_argument('--otsu', action='store_true')
    ap.add_argument('--keep-multiple', action='store_true')
    ap.add_argument('--save-diff', action='store_true')
    ap.add_argument('--mode', choices=['changed','common'], default='changed', help='changed: detect modified region (high diff). common: detect similar region (low diff).')
    args = ap.parse_args()

    texA = load_texture(args.texture_ref)
    texB = load_texture(args.texture_alt)
    diff = compute_diff(texA, texB, args.use_lab)
    diff_s = smooth_gaussian(diff, args.gauss_sigma)

    mode = args.mode
    if mode == 'changed':
        if args.otsu:
            thresh = otsu_threshold(diff_s)
        else:
            thresh = float(np.percentile(diff_s, args.percentile))
        mask = diff_s >= thresh
        thresh_low = None
    else:  # common
        if args.otsu:
            # With Otsu, choose lower side of threshold
            thresh_low = otsu_threshold(diff_s)
        else:
            thresh_low = float(np.percentile(diff_s, args.low_percentile))
        mask = diff_s <= thresh_low
        thresh = None
    if not args.keep_multiple:
        mask = largest_component(mask)

    V, UV = load_obj_with_uv(args.brain_surface)
    H, W = diff.shape
    u = np.clip(UV[:,0], 0, 1)
    v = np.clip(UV[:,1], 0, 1)
    px = (u * (W - 1)).astype(int)
    py = ((1.0 - v) * (H - 1)).astype(int)
    vmask = mask[py, px]
    indices = np.nonzero(vmask)[0].astype(np.int32)

    os.makedirs(args.outdir, exist_ok=True)
    # Output filenames reflect mode
    basename = 'craniotomy_region_texture_common' if mode == 'common' else 'craniotomy_region_texture'
    out_npz = os.path.join(args.outdir, f'{basename}.npz')
    coverage = float(indices.size / max(1, V.shape[0]))
    stats = {
        'diff_min': float(diff.min()),
        'diff_max': float(diff.max()),
        'diff_mean': float(diff.mean()),
        'diff_median': float(np.median(diff)),
    }
    save_kwargs = dict(indices=indices,
                       coverage=coverage,
                       method=f'texture_diff_{mode}',
                       stats=stats)
    if thresh is not None:
        save_kwargs['threshold_high'] = thresh
    if 'thresh_low' in locals() and thresh_low is not None:
        save_kwargs['threshold_low'] = thresh_low
    np.savez_compressed(out_npz, **save_kwargs)
    meta = {
        'method': f'texture_diff_{mode}',
        'brain_surface': args.brain_surface,
        'texture_ref': args.texture_ref,
        'texture_alt': args.texture_alt,
        'threshold_high': thresh if thresh is not None else None,
        'threshold_low': thresh_low if 'thresh_low' in locals() else None,
        'coverage_ratio': coverage,
        'vertex_count': int(V.shape[0]),
        'mask_count': int(indices.size),
        'use_lab': bool(args.use_lab),
        'otsu': bool(args.otsu),
        'gauss_sigma': args.gauss_sigma,
        'percentile': args.percentile,
        'low_percentile': args.low_percentile,
        'keep_multiple_components': bool(args.keep_multiple),
        'npz': out_npz,
        'stats': stats
    }
    meta_name = 'craniotomy_region_texture_common_meta.json' if mode == 'common' else 'craniotomy_region_texture_meta.json'
    with open(os.path.join(args.outdir, meta_name), 'w') as f:
        json.dump(meta, f, indent=2)

    if args.save_diff and Image is not None:
        try:
            from PIL import Image as PILImage
            # Normalize diff for visualization
            dvis = (diff - diff.min()) / max(1e-9, diff.max() - diff.min())
            PILImage.fromarray((dvis * 255).astype(np.uint8)).save(os.path.join(args.outdir, 'craniotomy_diff.png'))
            PILImage.fromarray((mask.astype(np.uint8) * 255)).save(os.path.join(args.outdir, 'craniotomy_mask.png'))
        except Exception:
            pass

    if mode == 'changed':
        print(f"Texture-based craniotomy detection (changed) complete. Coverage={coverage:.4f}, high_thresh={thresh:.4f}, indices={indices.size}")
    else:
        print(f"Texture-based craniotomy detection (common) complete. Coverage={coverage:.4f}, low_thresh={thresh_low:.4f}, indices={indices.size}")

if __name__ == '__main__':
    main()
