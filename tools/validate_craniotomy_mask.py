#!/usr/bin/env python
"""
Validate the craniotomy_region.npz mask produced by detect_craniotomy.py.
Checks & Outputs:
  - Computes nearest distances brain->skull and compares masked vs unmasked stats.
  - Reports Z-distribution comparison (masked should bias toward higher Z if opening on top).
  - Coverage ratio (masked vertex count / total brain vertices).
  - Writes data/craniotomy_validation_stats.json with all metrics.
  - Exports colored OBJ (brain_mask_colored.obj) with masked vertices tinted (requires simple grouping).
    * Masked vertices duplicated as a separate object for easy inspection.

Usage:
  python tools/validate_craniotomy_mask.py \
      --skull data/surface_skull.obj \
      --brain data/surface_full_decimated.obj \
      --mask  data/craniotomy_region.npz \
      --outdir data

Optional:
  --save-distances : also saves full distance arrays (brain_distance.npy and masked_distance.npy)

Dependencies: none mandatory. Uses SciPy KDTree if available for speed, else brute force.
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import List, Tuple
import numpy as np

# Minimal OBJ loader (triangulates)

def load_obj_minimal(path: str):
    verts = []
    faces = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            sp = line.strip().split()
            if not sp:
                continue
            if sp[0] == 'v' and len(sp) >= 4:
                try:
                    verts.append([float(sp[1]), float(sp[2]), float(sp[3])])
                except ValueError:
                    continue
            elif sp[0] == 'f' and len(sp) >= 4:
                idxs = []
                for tok in sp[1:]:
                    ts = tok.split('/')
                    try:
                        vid = int(ts[0])
                        if vid < 0:  # negative index
                            vid = len(verts) + 1 + vid
                        idxs.append(vid - 1)
                    except Exception:
                        pass
                if len(idxs) < 3:
                    continue
                for i in range(1, len(idxs)-1):
                    faces.append([idxs[0], idxs[i], idxs[i+1]])
    V = np.asarray(verts, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)
    return V, F


def nearest_dist(brain_pts: np.ndarray, skull_pts: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(skull_pts)
        d, _ = tree.query(brain_pts, k=1, n_jobs=-1 if hasattr(tree, 'query') else 1)
        return d.astype(np.float32)
    except Exception:
        # Brute force chunked
        B = brain_pts
        S = skull_pts
        dmin = np.full(B.shape[0], np.inf, dtype=np.float64)
        chunk = 4096
        for i in range(0, S.shape[0], chunk):
            Sblk = S[i:i+chunk]
            diff = B[:, None, :] - Sblk[None, :, :]
            dsq = np.einsum('ijk,ijk->ij', diff, diff)
            dmin = np.minimum(dmin, dsq.min(axis=1))
        return np.sqrt(dmin).astype(np.float32)


def write_colored_obj(path: str, V: np.ndarray, F: np.ndarray, mask: np.ndarray):
    """Export two groups: full brain (g brain) and masked subset duplicated (g craniotomy_mask)."""
    with open(path, 'w') as f:
        f.write('# Brain with craniotomy mask\n')
        f.write('g brain\n')
        for v in V:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for tri in F:
            a,b,c = tri + 1
            f.write(f'f {a} {b} {c}\n')
        # Duplicate masked verts (offset indices)
        f.write('g craniotomy_mask\n')
        base = V.shape[0]
        mv = V[mask]
        for v in mv:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        # Simple point markers (as degenerate lines) to be visible in some viewers
        for i in range(mv.shape[0]):
            idx = base + i + 1
            f.write(f'l {idx} {idx}\n')


def main():
    ap = argparse.ArgumentParser(description='Validate craniotomy mask file.')
    ap.add_argument('--skull', required=True)
    ap.add_argument('--brain', required=True)
    ap.add_argument('--mask', required=True, help='craniotomy_region.npz path')
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--save-distances', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.mask):
        print('[ERROR] mask file not found')
        sys.exit(1)

    V_skull, F_skull = load_obj_minimal(args.skull)
    V_brain, F_brain = load_obj_minimal(args.brain)
    if V_skull.size == 0 or V_brain.size == 0:
        print('[ERROR] empty mesh(s)')
        sys.exit(1)

    data = np.load(args.mask, allow_pickle=True)
    indices = data.get('indices')
    if indices is None or indices.size == 0:
        print('[ERROR] mask has no indices')
        sys.exit(1)

    mask = np.zeros(V_brain.shape[0], dtype=bool)
    mask[indices.astype(np.int64)] = True

    # Distances
    d = nearest_dist(V_brain, V_skull)
    d_mask = d[mask]
    d_other = d[~mask]

    def stats(arr: np.ndarray):
        return {
            'count': int(arr.size),
            'min': float(arr.min()) if arr.size else 0.0,
            'max': float(arr.max()) if arr.size else 0.0,
            'mean': float(arr.mean()) if arr.size else 0.0,
            'median': float(np.median(arr)) if arr.size else 0.0,
            'p90': float(np.percentile(arr, 90)) if arr.size else 0.0,
            'p95': float(np.percentile(arr, 95)) if arr.size else 0.0,
        }

    # Z distribution
    z = V_brain[:, 2]
    z_mask = z[mask]
    z_other = z[~mask]

    metrics = {
        'coverage_ratio': float(mask.sum() / max(1, V_brain.shape[0])),
        'distance_mask': stats(d_mask),
        'distance_other': stats(d_other),
        'z_mask': stats(z_mask),
        'z_other': stats(z_other),
        'distance_contrast_factor': (float(d_mask.mean() / max(1e-9, d_other.mean())) if d_other.size else 0.0),
        'z_mean_diff': float(z_mask.mean() - z_other.mean()) if z_other.size else 0.0,
        'method': str(data.get('method') if 'method' in data else 'loop'),
    }

    os.makedirs(args.outdir, exist_ok=True)
    stats_path = os.path.join(args.outdir, 'craniotomy_validation_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[INFO] Wrote stats: {stats_path}')

    obj_path = os.path.join(args.outdir, 'brain_mask_colored.obj')
    write_colored_obj(obj_path, V_brain, F_brain, mask)
    print(f'[INFO] Wrote colored OBJ: {obj_path}')

    if args.save_distances:
        np.save(os.path.join(args.outdir, 'brain_distance.npy'), d)
        np.save(os.path.join(args.outdir, 'brain_distance_masked.npy'), d_mask)
        print('[INFO] Saved distance arrays.')

    # Quick heuristic verdict
    verdict = []
    if metrics['coverage_ratio'] < 0.01:
        verdict.append('Mask too small (<1%).')
    if metrics['coverage_ratio'] > 0.6:
        verdict.append('Mask too large (>60%).')
    if metrics['distance_contrast_factor'] < 1.2:
        verdict.append('Distance contrast low (masked not much farther from skull).')
    if metrics['z_mean_diff'] < 0.0:
        verdict.append('Masked region not higher in Z than rest.')
    if not verdict:
        verdict.append('Mask plausibly represents opening region.')
    print('[VERDICT]', ' '.join(verdict))

if __name__ == '__main__':
    main()
