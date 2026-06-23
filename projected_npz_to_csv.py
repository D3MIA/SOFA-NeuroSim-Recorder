#!/usr/bin/env python3
# projected_npz_to_csv.py - Convert projected NPZ files to CSV

import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime

def estimate_projected_csv_size(npz_file):
    """Estimate CSV file size for a projected NPZ"""

    print("PROJECTED CSV SIZE ESTIMATE")
    print("=" * 40)

    data = np.load(npz_file)

    total_rows = 0
    total_columns = 0

    # Data analysis
    has_frames = False
    has_projected = False
    n_frames = 0
    n_vertices = 0

    for key in data.files:
        array = data[key]
        print(f"{key}: {array.shape} ({array.dtype})")

        if key == 'frames' and len(array.shape) == 3:
            # frames: (n_frames, n_vertices, 3)
            n_frames, n_vertices, coords = array.shape
            has_frames = True
            print(f"   → 3D positions: {n_frames} frames × {n_vertices} vertices")

        elif key == 'projected_pixels' and len(array.shape) == 3:
            # projected_pixels: (n_frames, n_vertices, 2)
            proj_frames, proj_vertices, coords = array.shape
            has_projected = True
            print(f"   → Pixel projections: {proj_frames} frames × {proj_vertices} vertices")

        elif key == 'visibility_masks' and len(array.shape) == 2:
            # visibility_masks: (n_frames, n_vertices)
            vis_frames, vis_vertices = array.shape
            print(f"   → Visibility masks: {vis_frames} frames × {vis_vertices} vertices")

        elif key == 'depth_values' and len(array.shape) == 2:
            # depth_values: (n_frames, n_vertices)
            depth_frames, depth_vertices = array.shape
            print(f"   → Depth values: {depth_frames} frames × {depth_vertices} vertices")

        elif key == 'displacements' and len(array.shape) == 3:
            # displacements: (n_frames, n_vertices, 3)
            disp_frames, disp_vertices, coords = array.shape
            print(f"   → Displacements: {disp_frames} frames × {disp_vertices} vertices")

        elif key == 'rest' and len(array.shape) == 2:
            # rest: (n_vertices, 3)
            rest_vertices, coords = array.shape
            print(f"   → Rest position: {rest_vertices} vertices")

        elif key == 'times' and len(array.shape) == 1:
            print(f"   → Timestamps: {len(array)} frames")

    # Compute total row and column count
    if has_frames and has_projected:
        rows_for_data = n_frames * n_vertices
        total_rows += rows_for_data

        # Columns: frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible
        base_columns = 10  # frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible

        # Add optional columns
        if 'displacements' in data.files:
            base_columns += 4  # disp_x, disp_y, disp_z, displacement_magnitude
        if 'rest' in data.files:
            base_columns += 3  # rest_x, rest_y, rest_z

        total_columns = base_columns

        # Display with recommended chunk information
        avg_rows_per_frame = n_vertices
        recommended_frames_per_chunk = max(1, 1000000 // avg_rows_per_frame)  # ~1M rows per chunk

        print(f"   → {rows_for_data:,} rows with {base_columns} columns")
        print(f"   → Recommendation: {recommended_frames_per_chunk} frames per chunk for ~1M rows")

    # Size estimate
    if total_rows > 0:
        # Average ~12 characters per cell (numbers + separators)
        estimated_chars = total_rows * total_columns * 12
        estimated_mb = estimated_chars / 1024 / 1024
        estimated_gb = estimated_mb / 1024

        print(f"\nESTIMATE:")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Columns: {total_columns}")
        print(f"   Estimated size: {estimated_mb:.1f} MB ({estimated_gb:.2f} GB)")

        # Warnings
        if estimated_gb > 2:
            print(f"WARNING: Very large file ({estimated_gb:.1f} GB)")
            print(f"   - Write time: ~{estimated_gb*10:.0f} minutes")
            print(f"   - RAM required: ~{estimated_gb*1.5:.1f} GB")
            print(f"   - Recommendation: Use chunked conversion")

        return estimated_gb

    data.close()
    return 0

def convert_projected_npz_to_csv_chunked(npz_file, frames_per_chunk=50, out_root=None):
    """Projected NPZ → CSV conversion by chunks"""

    print("\nPROJECTED NPZ → CSV CHUNKED CONVERSION")
    print("=" * 50)

    data = np.load(npz_file)

    # Check required keys
    required_keys = ['frames', 'projected_pixels', 'visibility_masks', 'depth_values']
    missing_keys = [key for key in required_keys if key not in data.files]

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
        data.close()
        return None

    # Load data
    frames = data['frames']  # Shape: (n_frames, n_vertices, 3)
    projected_pixels = data['projected_pixels']  # Shape: (n_frames, n_vertices, 2)
    visibility_masks = data['visibility_masks']  # Shape: (n_frames, n_vertices)
    depth_values = data['depth_values']  # Shape: (n_frames, n_vertices)
    
    # Optional data
    displacements = data.get('displacements', None)  # Shape: (n_frames, n_vertices, 3) or None
    rest_positions = data.get('rest', None)  # Shape: (n_vertices, 3) or None
    times = data.get('times', np.arange(len(frames)) * 0.01)  # Fallback times

    n_frames, n_vertices, _ = frames.shape
    total_rows = n_frames * n_vertices

    print("Detected data:")
    print(f"   Frames: {n_frames}")
    print(f"   Vertices per frame: {n_vertices}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Pixel projections: Available")
    print(f"   Visibility masks: Available") 
    print(f"   Depth values: Available")
    print(f"   Displacements: {'Available' if displacements is not None else 'Not available'}")
    print(f"   Rest position: {'Available' if rest_positions is not None else 'Not available'}")

    # Prepare output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"brain_projected_{timestamp}"

    # Rows per chunk based on frames
    rows_per_chunk = frames_per_chunk * n_vertices

    if frames_per_chunk >= n_frames:
        # Simple conversion (single file)
        print(f"Simple conversion ({frames_per_chunk} frames ≥ {n_frames} total frames)")
        return _convert_single_projected_csv(frames, projected_pixels, visibility_masks, depth_values, 
                                           times, base_filename, displacements, rest_positions, out_root=out_root)
    else:
        # Chunked conversion
        n_chunks = (n_frames // frames_per_chunk) + (1 if n_frames % frames_per_chunk > 0 else 0)
        print(f"Chunked conversion ({n_chunks} files of {frames_per_chunk} frames each)")
        return _convert_chunked_projected_csv(frames, projected_pixels, visibility_masks, depth_values,
                                            times, base_filename, frames_per_chunk, displacements, rest_positions, out_root=out_root)

def _convert_single_projected_csv(frames, projected_pixels, visibility_masks, depth_values, times, 
                                base_filename, displacements=None, rest_positions=None, out_root=None):
    """Convert to a single CSV file with all projected data"""

    if out_root is None:
        out_root = "projected_npz"
    os.makedirs(out_root, exist_ok=True)
    output_file = os.path.join(out_root, f"{base_filename}.csv")

    print(f"Writing: {os.path.basename(output_file)}")

    # Define columns
    columns = ['frame', 'vertex_id', 'time', 'x', 'y', 'z', 'pixel_x', 'pixel_y', 'depth_ndc', 'is_visible']

    if displacements is not None:
        columns.extend(['disp_x', 'disp_y', 'disp_z', 'displacement_magnitude'])

    if rest_positions is not None:
        columns.extend(['rest_x', 'rest_y', 'rest_z'])

    print(f"CSV columns: {columns}")

    # Prepare data
    rows = []
    n_frames, n_vertices, _ = frames.shape

    for frame_idx in range(len(frames)):
        frame_data = frames[frame_idx]
        frame_pixels = projected_pixels[frame_idx]
        frame_visibility = visibility_masks[frame_idx]
        frame_depths = depth_values[frame_idx]
        time_val = times[frame_idx] if frame_idx < len(times) else frame_idx * 0.01

        # Displacement data for this frame (if available)
        frame_displacements = displacements[frame_idx] if displacements is not None else None

        for vertex_idx in range(n_vertices):
            position = frame_data[vertex_idx]
            pixel_pos = frame_pixels[vertex_idx]
            is_visible = int(frame_visibility[vertex_idx])
            depth = frame_depths[vertex_idx]

            # Base: frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible
            row = [
                frame_idx,
                vertex_idx,
                time_val,
                position[0],
                position[1],
                position[2],
                pixel_pos[0],
                pixel_pos[1],
                depth,
                is_visible
            ]

            # Add displacements if available
            if frame_displacements is not None:
                displacement_vec = frame_displacements[vertex_idx]
                displacement_mag = np.linalg.norm(displacement_vec)
                row.extend([
                    displacement_vec[0],
                    displacement_vec[1],
                    displacement_vec[2],
                    displacement_mag
                ])

            # Add rest position if available
            if rest_positions is not None and vertex_idx < len(rest_positions):
                rest_pos = rest_positions[vertex_idx]
                row.extend([
                    rest_pos[0],
                    rest_pos[1],
                    rest_pos[2]
                ])

            rows.append(row)

        # Progress
        if (frame_idx + 1) % 10 == 0:
            print(f"   Frame {frame_idx + 1}/{len(frames)} processed...")

    # Create DataFrame and save
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False, float_format='%.6f')

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"File created: {size_mb:.1f} MB")
    print(f"Statistics: {len(rows):,} rows × {len(columns)} columns")

    return output_file

def _convert_chunked_projected_csv(frames, projected_pixels, visibility_masks, depth_values, times,
                                 base_filename, frames_per_chunk, displacements=None, rest_positions=None, out_root=None):
    """Convert to multiple CSV files (chunks) with projected data"""

    n_frames, n_vertices, _ = frames.shape

    print(f"Chunk configuration:")
    print(f"   Frames per chunk: {frames_per_chunk}")
    print(f"   Rows per chunk: {frames_per_chunk * n_vertices:,}")

    # Define columns
    columns = ['frame', 'vertex_id', 'time', 'x', 'y', 'z', 'pixel_x', 'pixel_y', 'depth_ndc', 'is_visible']

    if displacements is not None:
        columns.extend(['disp_x', 'disp_y', 'disp_z', 'displacement_magnitude'])

    if rest_positions is not None:
        columns.extend(['rest_x', 'rest_y', 'rest_z'])

    print(f"CSV columns: {columns}")

    output_files = []
    chunk_idx = 0

    if out_root is None:
        out_root = "projected_npz"
    os.makedirs(out_root, exist_ok=True)

    for start_frame in range(0, n_frames, frames_per_chunk):
        end_frame = min(start_frame + frames_per_chunk, n_frames)

        # Chunk filename
        chunk_filename = os.path.join(out_root, f"{base_filename}_chunk_{chunk_idx:03d}.csv")

        chunk_frame_count = end_frame - start_frame
        chunk_row_count = chunk_frame_count * n_vertices
        print(f"Chunk {chunk_idx}: frames {start_frame}-{end_frame-1} ({chunk_frame_count} frames, {chunk_row_count:,} rows)")

        # Data for this chunk
        chunk_frames = frames[start_frame:end_frame]
        chunk_pixels = projected_pixels[start_frame:end_frame]
        chunk_visibility = visibility_masks[start_frame:end_frame]
        chunk_depths = depth_values[start_frame:end_frame]
        chunk_times = times[start_frame:end_frame] if start_frame < len(times) else [i * 0.01 for i in range(start_frame, end_frame)]
        chunk_displacements = displacements[start_frame:end_frame] if displacements is not None else None

        # Conversion
        rows = []
        for rel_frame_idx in range(chunk_frame_count):
            abs_frame_idx = start_frame + rel_frame_idx
            frame_data = chunk_frames[rel_frame_idx]
            frame_pixels = chunk_pixels[rel_frame_idx]
            frame_visibility = chunk_visibility[rel_frame_idx]
            frame_depths = chunk_depths[rel_frame_idx]
            time_val = chunk_times[rel_frame_idx] if rel_frame_idx < len(chunk_times) else abs_frame_idx * 0.01

            # Displacement data for this frame (if available)
            frame_displacements = chunk_displacements[rel_frame_idx] if chunk_displacements is not None else None

            for vertex_idx in range(n_vertices):
                position = frame_data[vertex_idx]
                pixel_pos = frame_pixels[vertex_idx]
                is_visible = int(frame_visibility[vertex_idx])
                depth = frame_depths[vertex_idx]

                # Base: frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible
                row = [
                    abs_frame_idx,  # Absolute frame
                    vertex_idx,
                    time_val,
                    position[0],
                    position[1],
                    position[2],
                    pixel_pos[0],
                    pixel_pos[1],
                    depth,
                    is_visible
                ]

                # Add displacements if available
                if frame_displacements is not None:
                    displacement_vec = frame_displacements[vertex_idx]
                    displacement_mag = np.linalg.norm(displacement_vec)
                    row.extend([
                        displacement_vec[0],
                        displacement_vec[1],
                        displacement_vec[2],
                        displacement_mag
                    ])

                # Add rest position if available
                if rest_positions is not None and vertex_idx < len(rest_positions):
                    rest_pos = rest_positions[vertex_idx]
                    row.extend([
                        rest_pos[0],
                        rest_pos[1],
                        rest_pos[2]
                    ])

                rows.append(row)

        # Save chunk
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(chunk_filename, index=False, float_format='%.6f')

        size_mb = os.path.getsize(chunk_filename) / 1024 / 1024
        print(f"   {os.path.basename(chunk_filename)}: {size_mb:.1f} MB ({len(rows):,} rows)")

        output_files.append(chunk_filename)
        chunk_idx += 1

        # Memory cleanup
        del rows, df
        gc.collect()

    # Create index file
    index_file = os.path.join(out_root, f"{base_filename}_INDEX.txt")
    with open(index_file, 'w') as f:
        f.write(f"Projected NPZ → CSV chunked conversion\n")
        f.write(f"=======================================\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total frames: {n_frames}\n")
        f.write(f"Total vertices: {n_vertices}\n")
        f.write(f"Columns: {', '.join(columns)}\n")
        f.write(f"Projections included: Yes\n")
        f.write(f"Displacements included: {'Yes' if displacements is not None else 'No'}\n")
        f.write(f"Rest position included: {'Yes' if rest_positions is not None else 'No'}\n")
        f.write(f"Frames per chunk: {frames_per_chunk}\n")
        f.write(f"Number of chunks: {len(output_files)}\n\n")
        f.write("Created files:\n")
        for i, filepath in enumerate(output_files):
            f.write(f"  {i+1}. {os.path.basename(filepath)}\n")

    print(f"Index created: {os.path.basename(index_file)}")

    return output_files

def smart_projected_conversion_menu():
    """Smart menu for projected NPZ conversion"""

    print("SMART PROJECTED NPZ → CSV CONVERSION")
    print("=" * 60)

    # Search projected NPZ files (recursive)
    export_root = "projected_npz"
    if not os.path.exists(export_root):
        print(f"Folder {export_root} not found")
        return

    npz_files = []
    for root, _dirs, files in os.walk(export_root):
        for fn in files:
            if fn.endswith('.npz') and 'projected' in fn:
                npz_files.append(os.path.join(root, fn))

    if not npz_files:
        print("No projected NPZ files found")
        print("Also searching in simulation_output folder...")
        
        # Fallback: search in simulation_output
        fallback_dir = "simulation_output"
        if os.path.exists(fallback_dir):
            fallback_paths = []
            for f in os.listdir(fallback_dir):
                if f.endswith('.npz') and 'projected' in f:
                    fallback_paths.append(os.path.join(fallback_dir, f))
            if fallback_paths:
                print(f"Files found in {fallback_dir}:")
                for i, filepath in enumerate(fallback_paths, 1):
                    size_mb = os.path.getsize(filepath) / 1024 / 1024
                    print(f"   {i}. {os.path.basename(filepath)} ({size_mb:.1f} MB)")
                
                # Use fallback_dir as working folder
                npz_files = fallback_paths
            else:
                print("No projected NPZ files found anywhere")
                return
        else:
            return

    print("Available projected NPZ files:")
    for i, filepath in enumerate(npz_files, 1):
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        rel = os.path.relpath(filepath, export_root)
        print(f"   {i}. {rel} ({size_mb:.1f} MB)")

    # File selection
    try:
        choice = int(input(f"\nChoose file (1-{len(npz_files)}): ")) - 1
        if 0 <= choice < len(npz_files):
            selected_file = npz_files[choice]
        else:
            print("Invalid choice")
            return
    except ValueError:
        print("Invalid choice")
        return

    # Size estimate
    estimated_gb = estimate_projected_csv_size(selected_file)

    # Determine mirror output subfolder
    run_subdir = os.path.relpath(os.path.dirname(selected_file), export_root)
    out_root = os.path.join(export_root, run_subdir) if run_subdir not in ('.', '') else export_root

    if estimated_gb > 2:
        print(f"\nLARGE FILE DETECTED ({estimated_gb:.1f} GB)")
        print("Recommended options:")
        print("  1. Chunked conversion (recommended)")
        print("  2. Simple conversion (crash risk)")
        print("  3. Cancel")

        try:
            option = int(input("Your choice (1-3): "))
            if option == 1:
                # Compute recommendation based on data
                data_temp = np.load(selected_file)
                if 'frames' in data_temp.files:
                    frames_shape = data_temp['frames'].shape
                    n_frames_total = frames_shape[0]
                    n_vertices = frames_shape[1]
                    # Recommendation: ~50 frames per chunk to avoid large files
                    recommended_frames = min(50, max(1, 1000000 // n_vertices))

                    print(f"\nInformation:")
                    print(f"   Total frames: {n_frames_total}")
                    print(f"   Vertices per frame: {n_vertices}")
                    print(f"   Recommendation: {recommended_frames} frames per chunk")

                    frames_per_chunk = int(input(f"Frames per chunk (default {recommended_frames}): ") or str(recommended_frames))
                    convert_projected_npz_to_csv_chunked(selected_file, frames_per_chunk, out_root=out_root)
                else:
                    frames_per_chunk = int(input("Frames per chunk (default 50): ") or "50")
                    convert_projected_npz_to_csv_chunked(selected_file, frames_per_chunk, out_root=out_root)
                data_temp.close()
            elif option == 2:
                convert_projected_npz_to_csv_chunked(selected_file, float('inf'), out_root=out_root)  # All frames in one chunk
            else:
                print("Cancelled")
        except ValueError:
            print("Invalid choice")
    else:
        # Direct conversion for small files
        convert_projected_npz_to_csv_chunked(selected_file, out_root=out_root)

if __name__ == "__main__":
    smart_projected_conversion_menu()
