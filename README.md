# Brain Simulation Data Processing Pipeline

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Complete Pipeline](#complete-pipeline)
- [Data Flow](#data-flow)

---

## Overview

This pipeline processes biomechanical brain simulation data from the SOFA Framework to create datasets for advanced computational analysis.

### Key Capabilities

- Surface deformation capture with displacement vectors
- 3D→Pixel projection using SOFA camera matrices
- CSV chunk processing for memory optimization
- Vertex overlay visualization on original images

---

## Architecture

```
SOFA Simulation → NPZ Export → 3D Projection → CSV Export (Optional) → Overlay On images (Debugging)
      ↓              ↓                ↓                     ↓                            ↓
   brain.py     AnimationRecorder  npz_projection   projected_npz_to_csv.py     overlay_vertices_on_images.py
                Surface Capture    .py                  (optional)                     (optional)

```

### Data Processing Stages

1. **Simulation Stage** - SOFA biomechanical simulation
2. **Capture Stage** - Surface deformation recording with AnimationRecorder
3. **Export Stage** - NPZ format with displacement vectors to simulation_output/
4. **Projection Stage** - Direct NPZ to projected NPZ with pixel coordinates
5. **CSV Export Stage** - Optional conversion of projected NPZ to CSV format
6. **Visualization Stage** - Vertex overlays on images

---

## File Structure

```
brain/               
├──  Core Simulation Files
│   ├── brain.py                           # Main SOFA simulation with AnimationRecorder
│   └── data/                              # Brain mesh and texture files
├──  Data Processing Pipeline
│   ├── npz_projection.py           # Direct NPZ to projected NPZ conversion
│   ├── projected_npz_to_csv.py            # Optional: Projected NPZ→CSV conversion
│   └── overlay_vertices_on_images.py      # Vertex visualization on images
├──  Output Directories
    ├── simulation_output/                  # Main output directory
    │   ├── images/                        # Original simulation frames
    │   ├── brain_surface_*.npz            # Surface capture data
    │   ├── brain_surface_*_summary.csv    # Summary statistics
    │   ├── brain_surface_*_meta.json      # Metadata files
    │   └── camera_params_*.json           # SOFA camera parameters
    ├── projected_npz/                     # Projection results directory
    │   ├── brain_surface_*_projected.npz  # Projected data with pixel coordinates
    │   └── brain_surface_*_projected.csv  # Projected CSV data (optional)
    └── overlayed_frames/                  # Visualization outputs
```

---

## Complete Pipeline

### Stage 1: Simulation & Capture

```bash
# Run SOFA simulation with surface capture
python brain.py
```

**What happens:**

- Loads brain mesh (241,011 vertices)
- Applies biomechanical forces
- Captures surface deformations with AnimationRecorder
- Exports NPZ with 4 keys: `rest`, `frames`, `displacements`, `times`
- Saves camera parameters in JSON format

**Output:** `simulation_output/brain_surface_[ID].npz`

### Stage 2: 3D→Pixel Projection

```bash
# Project 3D coordinates directly to NPZ with pixel data
python npz_projection.py
```

**What happens:**

- Loads original NPZ file and camera parameters
- Processes all frames with 3D→pixel transformation
- Adds pixel coordinates, visibility masks, and depth values
- Saves incrementally every 50 frames (backup protection)
- Creates single projected NPZ file

**Output:** `projected_npz/brain_surface_[ID]_projected.npz`

### Stage 3: CSV Export (Optional)

```bash
# Convert projected NPZ to CSV format for analysis
python projected_npz_to_csv.py
```

**What happens:**

- Loads projected NPZ file
- Processes data in memory-efficient chunks
- Exports CSV with all projection data
- Includes pixel coordinates, visibility, and depth

**Output:** `projected_npz/brain_surface_[ID]_projected.csv`

### Stage 4: Visualization & Validation

```bash
# Create vertex overlays on original images
python overlay_vertices_on_images.py
```

**What happens:**

- Loads original simulation images
- Loads projected NPZ data directly
- Overlays vertices on images (colored by depth)
- Creates comparison grids and statistics
- Validates projection accuracy

**Output:** `overlayed_frames/overlay_frame_*.png` files

---

## Data Flow

### Data Formats Throughout Pipeline

#### 1. **NPZ Format** (simulation_output/)

```python
# Structure: 4 keys
{
    'rest': (241011, 3),        # Rest positions XYZ
    'frames': (300, 241011, 3), # Deformed positions per frame
    'displacements': (300, 241011, 3), # Displacement vectors
    'times': (300,)             # Time stamps
}
```

#### 2. **Projected NPZ Format** (projected_npz/)

```python
# Structure: 7 keys (original + projected data)
{
    'rest': (241011, 3),           # Rest positions XYZ
    'frames': (300, 241011, 3),    # Deformed positions per frame
    'displacements': (300, 241011, 3), # Displacement vectors
    'times': (300,),               # Time stamps
    'projected_pixels': (300, 241011, 2), # Pixel coordinates XY
    'visibility_masks': (300, 241011),    # Visibility boolean
    'depth_values': (300, 241011)         # Depth in NDC
}
```

#### 3. **CSV Format** (projected_npz/ - Optional)

```csv
# Columns: 17 total (all projection data included)
frame,vertex_id,time,x,y,z,pixel_x,pixel_y,depth_ndc,is_visible,disp_x,disp_y,disp_z,displacement_magnitude,rest_x,rest_y,rest_z

0,0,0.0,-45.1,12.3,5.9,856.2,423.7,0.654,1,0.1,0.2,0.1,0.245,-45.2,12.1,5.8
```

#### 4. **Camera Parameters JSON**

```json
{
  "essential_params": {
    "position": [-86.338, -17.669, 126.0],
    "orientation": [0.049, -0.297, 0.051, 0.952],
    "viewport_width": 1920,
    "viewport_height": 1080,
    "projectionMatrix": [...],
    "modelViewMatrix": [...]
  }
}
```

---

## Projection Results

Below are examples of vertex projections onto images:

#### Frame 000

![Frame 000](overlayed_frames/overlay_frame_0000.png)

#### Frame 050

![Frame 050](overlayed_frames/overlay_frame_0050.png)

_Created: August 2025 | Version: 1.0 | Brain Simulation Data Processing Pipeline_
