#!/usr/bin/env python3
"""
MODIFY EXISTING FRAMES WITH GEOMETRIC NOISE
==========================================
Modifies existing frames in datasets with geometric noise instead of adding new ones
"""

import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetFrameModifier:
    """Modifies existing frames in datasets with geometric noise"""
    
    def __init__(self, noise_percentage=0.05, displacement_noise=0.05):
        self.noise_percentage = noise_percentage
        self.displacement_noise = displacement_noise
        
        logger.info(f"🎲 Frame Modifier configured:")
        logger.info(f"   - {noise_percentage*100:.1f}% of points noised per frame")
        logger.info(f"   - ±{displacement_noise*100:.1f}% of displacement range")
    
    def identify_impact_region_simple(self, forces, coordinates):
        """Identifies the main impact region"""
        threshold = np.percentile(forces, 90)
        high_force_mask = forces > threshold
        
        if not np.any(high_force_mask):
            max_idx = np.argmax(forces)
            high_force_mask[max_idx] = True
        
        impact_points = coordinates[high_force_mask]
        impact_center = np.mean(impact_points, axis=0)
        
        distances_to_center = np.linalg.norm(impact_points - impact_center, axis=1)
        impact_radius = np.max(distances_to_center) * 1.5
        
        coord_range = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        min_radius = np.linalg.norm(coord_range) * 0.05
        impact_radius = max(impact_radius, min_radius)
        
        return impact_center, impact_radius
    
    def find_noise_candidates(self, coordinates, impact_center, impact_radius, forces):
        """Finds candidates outside impact region + low forces"""
        distances_to_impact = np.linalg.norm(coordinates - impact_center, axis=1)
        outside_impact = distances_to_impact > impact_radius
        
        force_threshold = np.percentile(forces, 30)
        low_forces = forces < force_threshold
        
        candidates = outside_impact & low_forces
        return candidates
    
    def calculate_frame_noise(self, displacement, n_noise_points):
        """Calculates noise based on frame range (±%)"""
        dx_values = displacement[:, 0]
        dy_values = displacement[:, 1]
        
        dx_range = np.max(dx_values) - np.min(dx_values)
        dy_range = np.max(dy_values) - np.min(dy_values)
        
        dx_noise_amplitude = dx_range * self.displacement_noise
        dy_noise_amplitude = dy_range * self.displacement_noise
        
        dx_noises = np.random.uniform(-dx_noise_amplitude, dx_noise_amplitude, n_noise_points)
        dy_noises = np.random.uniform(-dy_noise_amplitude, dy_noise_amplitude, n_noise_points)
        
        noise_vectors = np.column_stack([dx_noises, dy_noises])
        
        return noise_vectors, dx_range, dy_range
    
    def modify_single_timestep(self, displacement, forces, coordinates):
        """Modifies a single timestep with noise"""
        noisy_displacement = displacement.copy()
        noisy_forces = forces.copy()
        
        impact_center, impact_radius = self.identify_impact_region_simple(forces, coordinates)
        candidates_mask = self.find_noise_candidates(coordinates, impact_center, impact_radius, forces)
        
        if not np.any(candidates_mask):
            return noisy_displacement, noisy_forces, 0, 0, 0
        
        candidate_indices = np.where(candidates_mask)[0]
        n_noise_points = int(len(coordinates) * self.noise_percentage)
        n_noise_points = min(n_noise_points, len(candidate_indices))
        
        if n_noise_points == 0:
            return noisy_displacement, noisy_forces, 0, 0, 0
        
        noise_indices = np.random.choice(candidate_indices, size=n_noise_points, replace=False)
        noise_vectors, dx_range, dy_range = self.calculate_frame_noise(displacement, n_noise_points)
        
        for i, idx in enumerate(noise_indices):
            noisy_displacement[idx] += noise_vectors[i]
            noisy_forces[idx] = 0.0  # Force = 0
        
        return noisy_displacement, noisy_forces, n_noise_points, dx_range, dy_range
    
    def modify_dataset_file(self, input_npz_path, output_npz_path, modify_ratio=0.3):
        """Modifies existing frames in a dataset file with noise"""
        
        logger.info(f"📦 Modifying: {input_npz_path.name}")
        
        # Load original data
        data = np.load(input_npz_path)
        original_displacement = data['disp2d'].copy()  # Make a copy to modify
        original_forces = data['force_mag'].copy()     # Make a copy to modify
        
        # Use coordinates from first timestep
        coordinates = data['projected_pixels'][0]
        
        n_total = len(original_displacement)
        n_modify = int(n_total * modify_ratio)
        
        logger.info(f"   Total frames: {n_total}")
        logger.info(f"   Frames to modify: {n_modify} ({modify_ratio*100:.1f}%)")
        
        # Select frames to modify (avoid beginning/end)
        safe_start = min(50, n_total // 10)
        safe_end = max(n_total - 50, n_total - n_total // 10)
        
        if safe_end <= safe_start:
            frame_candidates = list(range(n_total))
        else:
            frame_candidates = list(range(safe_start, safe_end))
        
        if len(frame_candidates) < n_modify:
            frame_candidates = list(range(n_total))
        
        # Select frames to modify
        modify_frames = np.random.choice(
            frame_candidates, 
            size=min(n_modify, len(frame_candidates)), 
            replace=False
        )
        
        # Stats for monitoring
        total_noise_points = 0
        stats_dx_ranges = []
        stats_dy_ranges = []
        
        # Progress bar
        pbar = tqdm(modify_frames, desc="🎲 Modifying frames", leave=False)
        
        for t_idx in pbar:
            # Modify this frame IN PLACE
            noisy_disp, noisy_force, n_noise, dx_range, dy_range = self.modify_single_timestep(
                original_displacement[t_idx],
                original_forces[t_idx], 
                coordinates
            )
            
            # Replace the original frame with the noisy version
            original_displacement[t_idx] = noisy_disp
            original_forces[t_idx] = noisy_force
            
            # Stats
            total_noise_points += n_noise
            if dx_range > 0:
                stats_dx_ranges.append(dx_range)
                stats_dy_ranges.append(dy_range)
            
            pbar.set_postfix({
                'noise_pts': n_noise,
                'dx_range': f'{dx_range:.4f}' if dx_range > 0 else '0'
            })
        
        # Prepare save data - use modified arrays
        save_data = {
            'disp2d': original_displacement,        # Now contains modified frames
            'force_mag': original_forces,           # Now contains modified forces
            'projected_pixels': data['projected_pixels']  # Keep original coordinates
        }
        
        # Copy other fields if they exist
        for key in ['times', 'visibility_masks', 'depth_values', 'meta', 'image_frame_indices']:
            if key in data.files:
                save_data[key] = data[key]
        
        # Add modification metadata
        if 'meta' in data.files:
            try:
                existing_meta = json.loads(str(data['meta']))
            except:
                existing_meta = {}
        else:
            existing_meta = {}
        
        existing_meta.update({
            'modified_frames': len(modify_frames),
            'total_frames': n_total,
            'modify_ratio': modify_ratio,
            'noise_percentage': self.noise_percentage,
            'displacement_noise': self.displacement_noise,
            'modified_frame_indices': modify_frames.tolist()
        })
        
        save_data['meta'] = json.dumps(existing_meta)
        
        # Save
        output_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz_path, **save_data)
        
        # Final stats
        avg_dx_range = np.mean(stats_dx_ranges) if stats_dx_ranges else 0
        avg_dy_range = np.mean(stats_dy_ranges) if stats_dy_ranges else 0
        
        logger.info(f"   ✅ Saved: {output_npz_path}")
        logger.info(f"   📊 Modified: {len(modify_frames)} frames out of {n_total}")
        logger.info(f"   🎲 Noisy points: {total_noise_points}")
        logger.info(f"   📏 Avg range: dx={avg_dx_range:.5f}, dy={avg_dy_range:.5f}")
        
        return output_npz_path
    
    def modify_all_datasets(self, input_dir, output_dir, exclude_patterns=None, modify_ratio=0.3):
        """Modifies frames in all datasets from datasets_2d directory"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        npz_files = list(input_path.glob('run_*/*_2d.npz'))
        
        if exclude_patterns:
            for pattern in exclude_patterns:
                npz_files = [f for f in npz_files if pattern not in str(f)]
        
        logger.info(f"🎯 Frame modification of {len(npz_files)} datasets")
        logger.info(f"📁 Input: {input_path}")
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"📊 {modify_ratio*100:.1f}% of frames will be modified per dataset")
        
        modified_files = []
        
        main_pbar = tqdm(npz_files, desc="📦 Datasets")
        
        for npz_file in main_pbar:
            main_pbar.set_description(f"📦 {npz_file.name[:30]}")
            
            relative_path = npz_file.relative_to(input_path)
            output_file = output_path / relative_path
            
            try:
                modified_file = self.modify_dataset_file(
                    npz_file, 
                    output_file,
                    modify_ratio=modify_ratio
                )
                modified_files.append(modified_file)
                
            except Exception as e:
                logger.error(f"❌ Error {npz_file.name}: {e}")
                continue
        
        logger.info(f"✅ {len(modified_files)} datasets modified successfully")
        
        return modified_files


def main():
    parser = argparse.ArgumentParser(description='Modify existing frames with geometric noise')
    parser.add_argument('--input_dir', default='datasets_2d', help='Input datasets directory')
    parser.add_argument('--output_dir', default='datasets_2d_modified', help='Output datasets directory')
    parser.add_argument('--noise_percentage', type=float, default=0.05, help='%% of points to noise')
    parser.add_argument('--displacement_noise', type=float, default=0.05, help='%% of displacement range')
    parser.add_argument('--modify_ratio', type=float, default=0.3, help='Ratio of frames to modify (0-1)')
    parser.add_argument('--exclude_patterns', nargs='*', help='Patterns to exclude from modification')
    
    args = parser.parse_args()
    
    # Create modifier
    modifier = DatasetFrameModifier(
        noise_percentage=args.noise_percentage,
        displacement_noise=args.displacement_noise
    )
    
    # Launch frame modification
    modified_files = modifier.modify_all_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        exclude_patterns=args.exclude_patterns,
        modify_ratio=args.modify_ratio
    )
    
    logger.info(f"🎉 Frame modification completed!")
    logger.info(f"📊 {len(modified_files)} datasets with modified frames created")
    logger.info(f"📁 Directory: {args.output_dir}")


if __name__ == '__main__':
    main()