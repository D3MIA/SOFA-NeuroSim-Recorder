#!/usr/bin/env python3
"""
Rasterize 3D surface forces from projected NPZ into 2D per-pixel heatmaps.

Inputs (from npz_projection output):
  - projected_pixels (F,V,2), visibility_masks (F,V), depth_values (F,V)
  - frames (F,V,3) [not required for this method]
  - surface_forces (F,V,3) [optional]
  - surface_external_forces (F,V,3) [optional]
  - metadata JSON (same basename + _metadata.json) for camera orientation

Outputs (.npz):
  - force_map_xy (F,H,W,2) in N (camera-space X/Y components)
  - force_map_mag (F,H,W) in N (||camera-space XY||)
  - If external forces available: external_force_map_xy, external_force_map_mag
  - plus a small metadata json next to it describing units and params

Notes:
  - Uses z-buffer per pixel (front-most vertex wins). Optionally sums contributions.
  - Camera-space vector: v_cam = R^T * v_world, with R from camera quaternion [qx,qy,qz,qw].
"""

import os, json, argparse
import numpy as np


def quaternion_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    return R


def load_camera_orientation(meta_json_path):
    try:
        with open(meta_json_path, 'r') as f:
            meta = json.load(f)
        cam = meta.get('camera_params', {})
        ori = cam.get('orientation', None)
        if ori is None:
            # fallback to zero-rotation
            return np.eye(3, dtype=np.float64)
        R = quaternion_to_rotation_matrix(ori)
        # world -> view rotation for vectors is R^T
        return R.T
    except Exception:
        return np.eye(3, dtype=np.float64)


def rasterize_force_maps(px, vis, depth, F_cam, width, height, mode='zbuffer'):
    """
    px: (V,2) float32 pixel coords; vis: (V,) bool; depth: (V,) float32; F_cam: (V,3) float32
    returns: fx_map (H,W), fy_map (H,W), mag_map (H,W)
    """
    H, W = height, width
    fx = np.zeros((H, W), dtype=np.float32)
    fy = np.zeros((H, W), dtype=np.float32)
    mag = np.zeros((H, W), dtype=np.float32)

    if mode == 'sum':
        # simple splat sum
        xi = np.clip(np.round(px[:, 0]).astype(np.int32), 0, W-1)
        yi = np.clip(np.round(px[:, 1]).astype(np.int32), 0, H-1)
        mask = vis
        for i in np.nonzero(mask)[0]:
            x, y = xi[i], yi[i]
            fx[y, x] += F_cam[i, 0]
            fy[y, x] += F_cam[i, 1]
            mag[y, x] += np.linalg.norm(F_cam[i, :2])
        return fx, fy, mag

    # default: z-buffer front-most
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    xi = np.clip(np.round(px[:, 0]).astype(np.int32), 0, W-1)
    yi = np.clip(np.round(px[:, 1]).astype(np.int32), 0, H-1)
    mask = vis
    idx = np.nonzero(mask)[0]
    # process front-most first: sort by depth ascending
    order = idx[np.argsort(depth[idx])]
    for i in order:
        x, y = xi[i], yi[i]
        if depth[i] < zbuf[y, x]:
            zbuf[y, x] = depth[i]
            fx[y, x] = F_cam[i, 0]
            fy[y, x] = F_cam[i, 1]
            mag[y, x] = np.linalg.norm(F_cam[i, :2])
    return fx, fy, mag


def process_projected_npz(npz_path, out_width=None, out_height=None, mode='zbuffer'):
    base = os.path.splitext(npz_path)[0]
    meta_json = base + '_metadata.json'
    Rv = load_camera_orientation(meta_json)

    data = np.load(npz_path)
    px = data['projected_pixels']  # (F,V,2)
    vis = data['visibility_masks']  # (F,V)
    depth = data['depth_values']  # (F,V)

    F_total = data['surface_forces'] if 'surface_forces' in data.files else None
    F_ext = data['surface_external_forces'] if 'surface_external_forces' in data.files else None

    F, V, _ = px.shape
    # Default size from metadata viewport, fallback to px range
    if out_width is None or out_height is None:
        w_meta = h_meta = None
        try:
            with open(meta_json, 'r') as f:
                meta = json.load(f)
            vp = meta.get('camera_params', {}).get('viewport', None)
            if isinstance(vp, (list, tuple)) and len(vp) == 2:
                w_meta, h_meta = int(vp[0]), int(vp[1])
        except Exception:
            pass
        if w_meta and h_meta:
            out_width = out_width or w_meta
            out_height = out_height or h_meta
        else:
            # infer from px
            w = int(np.nanmax(px[..., 0]) + 1)
            h = int(np.nanmax(px[..., 1]) + 1)
            out_width = out_width or max(1, w)
            out_height = out_height or max(1, h)

    # If original images are larger/smaller, px already matches that space.
    # We assume px is in the intended pixel grid.

    # Output buffers
    maps = {
        'force_map_xy': np.zeros((F, out_height, out_width, 2), dtype=np.float32),
        'force_map_mag': np.zeros((F, out_height, out_width), dtype=np.float32),
    }
    if F_ext is not None:
        maps['external_force_map_xy'] = np.zeros((F, out_height, out_width, 2), dtype=np.float32)
        maps['external_force_map_mag'] = np.zeros((F, out_height, out_width), dtype=np.float32)

    for t in range(F):
        vis_t = vis[t]
        px_t = px[t]
        dp_t = depth[t]

        if F_total is not None:
            Ft = F_total[t].astype(np.float64)
            Fcam = (Ft @ Rv.T).astype(np.float32)  # (V,3)
            fx, fy, mag = rasterize_force_maps(px_t, vis_t, dp_t, Fcam, out_width, out_height, mode=mode)
            maps['force_map_xy'][t, :, :, 0] = fx
            maps['force_map_xy'][t, :, :, 1] = fy
            maps['force_map_mag'][t] = mag

        if F_ext is not None:
            Fe = F_ext[t].astype(np.float64)
            Fcam_e = (Fe @ Rv.T).astype(np.float32)
            fx, fy, mag = rasterize_force_maps(px_t, vis_t, dp_t, Fcam_e, out_width, out_height, mode=mode)
            maps['external_force_map_xy'][t, :, :, 0] = fx
            maps['external_force_map_xy'][t, :, :, 1] = fy
            maps['external_force_map_mag'][t] = mag

    # Save alongside the input npz
    out_path = base + '_forcemaps.npz'
    np.savez_compressed(out_path, **maps)

    # Write small metadata
    meta_out = base + '_forcemaps_meta.json'
    meta = {
        'source': os.path.basename(npz_path),
        'width': int(out_width),
        'height': int(out_height),
        'frames': int(F),
        'mode': mode,
        'units': {
            'force_map_xy': 'N (camera-space X,Y)',
            'force_map_mag': 'N (||camera-space XY||)',
        },
        'includes_external': bool(F_ext is not None)
    }
    with open(meta_out, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {os.path.basename(out_path)} and {os.path.basename(meta_out)}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Rasterize projected NPZ forces into 2D force heatmaps.')
    ap.add_argument('npz', help='Path to a projected NPZ produced by npz_projection.py')
    ap.add_argument('--width', type=int, default=None, help='Output width (defaults to projection width)')
    ap.add_argument('--height', type=int, default=None, help='Output height (defaults to projection height)')
    ap.add_argument('--mode', choices=['zbuffer', 'sum'], default='zbuffer', help='Rasterization mode')
    args = ap.parse_args()

    process_projected_npz(args.npz, out_width=args.width, out_height=args.height, mode=args.mode)


if __name__ == '__main__':
    main()
