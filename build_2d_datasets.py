#!/usr/bin/env python3
"""Build 2D Displacement Datasets (Adapted for ForceEstimation Project)

Takes projected NPZ files from run_seed_* directories with keys including:
  - displacements (T,N,3) + surface_forces (T,N,3) + projected_pixels (T,N,2)
  - or frames (T,N,3) + rest (N,3) + surface_forces + projected_pixels

Produces compressed NPZ with:
  disp2d: (T,N,2) - 2D displacements (X,Y only)
  force_mag: (T,N) - force magnitudes 
  projected_pixels: (T,N,2) - pixel coordinates
  visibility_masks: (T,N) - visibility boolean masks
  times: (T,) - timestamps
  meta: JSON string with metadata

Usage:
  python build_2d_datasets.py --auto --root . --out-root datasets_2d
  python build_2d_datasets.py --inputs run_seed_5555/*.npz --out-root datasets_2d
"""
import argparse, os, json, glob, numpy as np, hashlib


def safe_relpath(path, root):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root))

def hash_path(path):
    return hashlib.sha256(path.encode()).hexdigest()[:10]

def derive_disp2d(data):
    """Extract 2D displacements (X,Y only) from NPZ data"""
    if 'displacements' in data.files:
        return data['displacements'][..., :2].astype(np.float32)
    if 'frames' in data.files and 'rest' in data.files:
        frames = data['frames'][..., :2]
        rest = data['rest'][..., :2]
        return (frames - rest[None,:,:]).astype(np.float32)
    raise ValueError('Missing displacements or frames+rest')

def process_file(path, out_root, root, overwrite=False, verbose=True):
    """Process a single projected NPZ file from run_seed_* directory"""
    if verbose:
        print(f'Processing: {path}')
    
    data = np.load(path, mmap_mode='r')
    
    # Check required keys for projected data
    required_keys = ['surface_external_forces']
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    
    # Extract 2D displacements
    disp2d = derive_disp2d(data)
    
    # Extract force magnitudes
    force_mag = np.linalg.norm(data['surface_external_forces'], axis=2).astype(np.float32)
    
    # Determine output path structure: run_seed_*/filename_2d.npz
    rel = safe_relpath(path, root)
    parts = rel.split(os.sep)  # Use os.sep for cross-platform compatibility
    
    if len(parts) >= 2 and parts[0].startswith('run_'):
        # Structure: run_E*_nu*_seed*/filename.npz -> datasets_2d/run_E*_nu*_seed*/filename_2d.npz
        subdir = os.path.join(out_root, parts[0])
    else:
        # Fallback: put in root output directory
        subdir = out_root
        
    os.makedirs(subdir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(parts[-1])[0]
    out_filename = f"{base_name}_2d.npz"
    out_path = os.path.join(subdir, out_filename)
    
    if os.path.exists(out_path) and not overwrite:
        if verbose: 
            print(f'[SKIP] {out_path} (already exists)')
        return out_path
    
    # Prepare data to save
    save_data = {
        'disp2d': disp2d,
        'force_mag': force_mag,
        'meta': json.dumps({
            'source': path,
            'disp2d_shape': list(disp2d.shape),
            'force_mag_shape': list(force_mag.shape),
            'has_times': 'times' in data.files,
            'has_projected_pixels': 'projected_pixels' in data.files,
            'has_visibility_masks': 'visibility_masks' in data.files,
            'has_depth_values': 'depth_values' in data.files,
            'all_keys': list(data.files),
            'hash_path': hash_path(path)
        })
    }
    
    # Add optional projected data
    if 'times' in data.files:
        save_data['times'] = data['times'].astype(np.float32)
    if 'projected_pixels' in data.files:
        save_data['projected_pixels'] = data['projected_pixels'].astype(np.float32)
    if 'visibility_masks' in data.files:
        save_data['visibility_masks'] = data['visibility_masks'].astype(np.bool_)
    if 'depth_values' in data.files:
        save_data['depth_values'] = data['depth_values'].astype(np.float32)
    if 'image_frame_indices' in data.files:
        save_data['image_frame_indices'] = data['image_frame_indices'].astype(np.int32)
    
    # Save compressed NPZ
    np.savez_compressed(out_path, **save_data)
    
    if verbose:
        keys_info = f"disp2d{disp2d.shape}, force_mag{force_mag.shape}"
        if 'projected_pixels' in save_data:
            keys_info += f", projected_pixels{save_data['projected_pixels'].shape}"
        if 'visibility_masks' in save_data:
            keys_info += f", visibility{save_data['visibility_masks'].shape}"
        print(f'[CREATED] {out_path} | {keys_info}')
    
    return out_path

def discover(root):
    """Find all projected NPZ files in run_* directories"""
    # Pattern for projected NPZ files: run_*/brain_surface_*_projected_*.npz
    pattern1 = os.path.join(root, 'run_*', '*projected*.npz')
    pattern2 = os.path.join(root, 'run_*', 'brain_surface_*.npz')
    
    files = []
    files.extend(glob.glob(pattern1))
    files.extend(glob.glob(pattern2))
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    
    return files

def main():
    """Build 2D datasets from projected NPZ files in run_seed_* directories"""
    parser = argparse.ArgumentParser(description='Convert projected 3D simulation data to 2D displacement datasets')
    parser.add_argument('--root', '-r', default='projected_npz', help='Root directory containing run_* folders (default: projected_npz)')
    parser.add_argument('--out', '-o', default='datasets_2d', help='Output directory for 2D datasets')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out)
    
    print(f'Root directory: {root}')
    print(f'Output directory: {out_root}')
    
    # Discover projected NPZ files
    files = discover(root)
    print(f'Found {len(files)} projected NPZ files')
    
    if not files:
        print('No projected NPZ files found in run_seed_* directories')
        print('Expected pattern: run_seed_*/brain_surface_*_projected_*.npz')
        return
    
    # Process each file
    for i, npz_path in enumerate(files):
        print(f'\n[{i+1}/{len(files)}] Processing: {os.path.relpath(npz_path, root)}')
        try:
            out_path = process_file(npz_path, out_root, root, 
                                  overwrite=args.overwrite, 
                                  verbose=args.verbose)
            if args.verbose:
                print(f'  -> {os.path.relpath(out_path, out_root)}')
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
    
    print(f'\nCompleted processing {len(files)} files')
    print(f'Output datasets saved to: {out_root}')

if __name__ == '__main__':
    main()