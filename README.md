# Brain Force Estimation Pipeline

This repository generates supervised data for brain-surface force estimation from SOFA simulations.

The current pipeline is:

1. Simulate brain deformation in SOFA with a sliding surgical tool.
2. Record synchronized surface geometry, external-force labels, camera parameters, and images.
3. Project 3D vertices into image space.
4. Convert projected data into compact 2D ML datasets.
5. Optionally add geometric noise for robustness.
6. Render overlays for visual inspection.

The target downstream model is: image-space motion -> force magnitude at each visible brain surface point.

## Current Workspace Status

This workspace already contains a full parameter sweep:

- 18 simulation runs in `simulation_output/`
- 18 projected runs in `projected_npz/`
- 18 2D datasets in `datasets_2d/`
- 18 augmented datasets in `datasets_2d_modified/`

Representative run metadata confirms the current export settings:

- `2000` frames per run
- `60.0 s` simulated duration at `dt = 0.03`
- `20326` saved surface vertices with `record_stride = 2`
- `667` JPEG images per run with `image_every = 3`

## Requirements

### SOFA

- Tested with SOFA `v25.06.00`
- `runSofa` must be available on your PATH, or you must call it with a full path

### Python packages

```bash
pip install numpy Pillow tqdm matplotlib opencv-python
```

Optional packages used by some utilities:

```bash
pip install scipy scikit-image
```

## Repository Layout

```text
Sofa-Simulation-Recorder/
|-- brain.py
|-- run_sweep.ps1
|-- npz_projection.py
|-- build_2d_datasets.py
|-- modify_frames_with_noise.py
|-- overlay_vertices_on_images.py
|-- projected_npz_to_csv.py
|-- verify_img.py
|-- simlib/
|   |-- camera.py
|   |-- deformers.py
|   |-- forces.py
|   |-- recorders.py
|   `-- visual.py
|-- tools/
|   |-- decimate_mesh.py
|   |-- detect_craniotomy_from_textures.py
|   |-- npz_to_force_heatmaps.py
|   `-- validate_craniotomy_mask.py
|-- data/
|   |-- surface_full.obj
|   |-- surface_full_decimated.obj
|   |-- surface_full_decimated_10k.obj
|   |-- surface_full_decimated_30k.obj
|   |-- surface_skull.obj
|   |-- volume_simplified.obj
|   |-- texture.png
|   |-- texture_outpaint.png
|   |-- texture_outpaint3.png
|   |-- craniotomy_region_texture_common.npz
|   `-- craniotomy_region_texture_common_meta.json
|-- simulation_output/
|   `-- run_E{E}_nu{nu}_seed{seed}/
|       |-- brain_surface_*_auto.npz
|       |-- brain_surface_*_auto_meta.json
|       |-- brain_surface_*_auto_summary.csv
|       |-- brain_surface_*_auto_external_force_quality.csv
|       |-- camera_params_COMPLETE_*.json
|       `-- images/frame_*.jpg
|-- projected_npz/
|   `-- run_E{E}_nu{nu}_seed{seed}/
|       |-- brain_surface_*_projected_*.npz
|       `-- brain_surface_*_projected_*_metadata.json
|-- datasets_2d/
|   `-- run_E{E}_nu{nu}_seed{seed}/
|       `-- brain_surface_*_2d.npz
|-- datasets_2d_modified/
|   `-- run_E{E}_nu{nu}_seed{seed}/
|       `-- brain_surface_*_2d.npz
`-- overlayed_frames/
    `-- run_E{E}_nu{nu}_seed{seed}/
        |-- overlay_frame_*.png
        |-- comparison_grid_*.png
        |-- statistics_summary_*.png
        `-- overlay_index_*.json
```

## Architecture

```text
tools/detect_craniotomy_from_textures.py
        |
        v
data/craniotomy_region_texture_common.npz
        |
        v
brain.py
  QuadSlideDeformer
        +
  ExternalForceAggregator
        +
  AnimationRecorder
        |
        v
simulation_output/run_E{E}_nu{nu}_seed{seed}/
        |
        v
npz_projection.py
        |
        v
projected_npz/run_E{E}_nu{nu}_seed{seed}/
        |
        v
build_2d_datasets.py
        |
        v
datasets_2d/run_E{E}_nu{nu}_seed{seed}/
        |
        v
modify_frames_with_noise.py
        |
        v
datasets_2d_modified/run_E{E}_nu{nu}_seed{seed}/
        |
        +--> overlay_vertices_on_images.py -> overlayed_frames/run_E.../
        |
        `--> training pipeline in Force-Estimation-Models
```

## Core Components

### `brain.py`

`brain.py` is a SOFA scene file, not a plain Python CLI script.

It configures:

- a sparse-grid FEM brain volume
- a visual brain surface mesh
- a fixed interactive camera
- one active `QuadSlideDeformer`
- one `ExternalForceAggregator`
- one `AnimationRecorder`

The active run directory is encoded directly from the material parameters and seed:

```text
simulation_output/run_E{YOUNG:.2f}_nu{POISSON:.3f}_seed{SEED}/
```

### `simlib.deformers.QuadSlideDeformer`

This is the active surgical tool generator used by `brain.py`.

It produces slide gestures with an optional push after each sequence. Force is distributed over FEM nodes with Gaussian weights normalized to conserve total applied force:

```text
d_i = ||x_i - center||
w_i = exp(-0.5 * (d_i / sigma)^2)
w_i = w_i / sum(w_i)
F_i = w_i * F_total * scale(t) * direction
```

Important current settings from `brain.py`:

- `radius = 20.0 mm`
- `slide_displacement = 10.0 mm`
- `slide_force_range = (200, 1000) mN`
- `push_force_range = (150, 800) mN`
- `push_probability = 0.2`
- `inward_bias = 0.25`
- `ramp_in = 8`, `hold_min = 10`, `hold_max = 20`, `ramp_out = 8`
- `release_frames = 5`, `cooldown_between_points = 10`

`DeepPressPusher` exists in `simlib/deformers.py`, but it is currently commented out in `brain.py` and is not part of the active pipeline.

### `simlib.forces.ExternalForceAggregator`

`ExternalForceAggregator` sums each deformer's `_frame_force` and writes the combined result into `MechanicalObject.externalForce`.

It also re-asserts that force at `onAnimateEndEvent`, because SOFA can zero `externalForce` during the solve step. Without this controller, the recorder can read zeros even when the deformer applied a force during the frame.

### `simlib.recorders.AnimationRecorder`

The recorder captures:

- rest positions
- deformed positions
- displacements
- timestamps
- surface external-force labels
- synchronized images

Important current settings from `brain.py`:

- `auto_export_frames = 2000`
- `capture_images = True`
- `image_every = 3`
- `image_format = 'jpg'`
- `image_quality = 92`
- `image_resolution = [1920, 1080]`
- `force_sampling_k = 8`
- `record_stride = 2`
- `force_label_mode = 'intensity'`
- all force-noise knobs are `0.0` except `force_noise_outlier_scale = 10.0`

#### Intensity force labels

The recorder first interpolates FEM external forces onto the surface mesh. In `intensity` mode, it then rescales the surface field so that the peak surface vertex equals the true total applied tool force:

```text
F_total = ||sum(frame_force over all FEM nodes)||
peak = max(||surface_external_forces||)
surface_external_forces = surface_external_forces * (F_total / peak)
```

This makes the label usable for direct displacement-to-force learning: the contact center reads the real tool intensity instead of a small distributed fraction.

## Simulation Parameters

### Material sweep

`run_sweep.ps1` currently evaluates:

| Parameter | Values |
|---|---|
| Young's modulus (`BRAIN_YOUNG`) | `2.5`, `3.0`, `5.0` kPa |
| Poisson ratio (`BRAIN_POISSON`) | `0.45`, `0.49` |
| Seed (`BRAIN_SEED`) | `1111`, `2222`, `3333` |

That gives `3 x 2 x 3 = 18` runs.

Interpretation used in the sweep script:

| Young's modulus | Interpretation |
|---|---|
| `2.5 kPa` | soft parenchyma / post-op edema |
| `3.0 kPa` | standard parenchyma |
| `5.0 kPa` | firm tissue / white matter |

If you launch `brain.py` manually and do not set environment variables, the defaults are:

- `BRAIN_YOUNG = 3.0`
- `BRAIN_POISSON = 0.45`
- `BRAIN_SEED = 1111`

### Current scene settings

| Setting | Value |
|---|---|
| Time step | `0.03 s` |
| Gravity | `[0, 0, 0]` |
| Total mass | `1.25 kg` |
| Velocity damping | `1.5` |
| Craniotomy boundary spring | `500 mN/mm` |
| Surface mesh | `data/surface_full_decimated.obj` if present, otherwise `data/surface_full.obj` |
| Volume topology | `data/volume_simplified.obj` with `16 x 16 x 16` sparse grid |
| Camera resolution | `1920 x 1080` |
| Camera field of view | `45 deg` |

## End-to-End Pipeline

### Stage 0: Optional craniotomy-mask regeneration

This repository already includes `data/craniotomy_region_texture_common.npz`, so you usually do not need to rerun mask generation.

If you want to regenerate it:

```bash
python tools/detect_craniotomy_from_textures.py \
  --brain-surface data/surface_full_decimated.obj \
  --texture-ref data/texture.png \
  --texture-alt data/texture_outpaint.png \
  --outdir data \
  --mode common \
  --gauss-sigma 2.0
```

Notes:

- `--mode common` is what matches the currently used `craniotomy_region_texture_common.npz`
- the repo currently contains `texture_outpaint.png` and `texture_outpaint3.png`
- optional dependencies such as `scipy` and `scikit-image` improve smoothing and thresholding behavior

### Stage 1: Run SOFA simulation

For one run:

```powershell
$env:BRAIN_YOUNG = "3.0"
$env:BRAIN_POISSON = "0.45"
$env:BRAIN_SEED = "1111"
runSofa .\brain.py
```

For the full sweep:

```powershell
.\run_sweep.ps1
```
Important:

- `brain.py` does not implement `--seed`, `--frames`, or `--output-dir` command-line flags
- the sweep script expects `runSofa` to be callable directly
- each run exits automatically after `2000` frames because the recorder auto-exports and closes SOFA

### Stage 2: Project 3D vertices into image space

```bash
python npz_projection.py
```

`npz_projection.py` is interactive. The typical flow is:

1. Press Enter to keep the default backup interval of `50` frames, or enter `2000` to save only once near the end for a 2000-frame run.
2. Choose option `1` to process all simulation NPZ files.

The projector:

- scans `simulation_output/` recursively for `brain_surface_*.npz`
- loads the matching camera JSON for each run
- uses the actual image resolution if it differs from the camera viewport
- saves projected files under the mirrored `projected_npz/run_E.../` subdirectory
- adds `projected_pixels`, `visibility_masks`, `depth_values`, and `image_frame_indices`

### Stage 3: Build ML-ready 2D datasets

```bash
python build_2d_datasets.py --root projected_npz --out datasets_2d --verbose
```

This script:

- scans `projected_npz/run_*/`
- extracts XY displacement as `disp2d`
- converts `surface_external_forces` to scalar `force_mag`
- preserves `projected_pixels`, `visibility_masks`, `depth_values`, `times`, and `image_frame_indices` when present

### Stage 4: Add geometric noise augmentation

```bash
python modify_frames_with_noise.py \
  --input_dir datasets_2d \
  --output_dir datasets_2d_modified \
  --noise_percentage 0.05 \
  --displacement_noise 0.05 \
  --modify_ratio 0.3
```

This modifies existing frames in place within copied datasets:

- selects `30%` of frames by default
- identifies the impact region from the top `10%` force values
- perturbs `5%` of candidate vertices outside that region
- sets the noised vertices' force magnitude to zero

### Stage 5: Visualize projected force labels on images

```bash
python overlay_vertices_on_images.py simulation_output/run_E3.00_nu0.450_seed1111
```

If no run directory is passed, the script auto-detects the latest run.

`overlay_vertices_on_images.py` is also interactive. Typical usage:

1. Launch the script.
2. Choose option `1` for the default batch overlay generation.
3. Inspect results in `overlayed_frames/run_E.../`.

It uses the projected NPZ, the run images, and the camera metadata to render force-colored overlays with a fixed global color scale.

## Data Products

### Raw simulation exports

Files written to `simulation_output/run_E.../`:

- `brain_surface_*_auto.npz`
- `brain_surface_*_auto_meta.json`
- `brain_surface_*_auto_summary.csv`
- `brain_surface_*_auto_external_force_quality.csv`
- `camera_params_COMPLETE_*.json`
- `images/frame_*.jpg`

Representative shapes from the current workspace:

```text
rest:                     (20326, 3)
frames:                   (2000, 20326, 3)
displacements:            (2000, 20326, 3)
times:                    (2000,)
surface_external_forces:  (2000, 20326, 3)
```

Units in the exported NPZ and metadata are:

- length: `mm`
- time: `s`
- mass: `kg`
- force labels: `N`

The conversion constant is defined in `simlib/forces.py` as:

```python
FORCE_TO_NEWTON = 1e-3
```

So deformer forces specified in `mN` are converted to Newtons before they are saved.

### Projected NPZ files

Files written to `projected_npz/run_E.../`:

- `brain_surface_*_projected_*.npz`
- `brain_surface_*_projected_*_metadata.json`

Projected NPZ content:

```text
frames:                   (2000, 20326, 3)
rest:                     (20326, 3)
displacements:            (2000, 20326, 3)
times:                    (2000,)
surface_external_forces:  (2000, 20326, 3)
projected_pixels:         (2000, 20326, 2)
visibility_masks:         (2000, 20326)
depth_values:             (2000, 20326)
image_frame_indices:      (667,)
```

`image_frame_indices` identifies which simulation frames have a matching saved image on disk.

### 2D dataset NPZ files

Files written to `datasets_2d/run_E.../`:

- `brain_surface_*_2d.npz`

2D dataset content:

```text
disp2d:              (2000, 20326, 2)
force_mag:           (2000, 20326)
projected_pixels:    (2000, 20326, 2)
visibility_masks:    (2000, 20326)
depth_values:        (2000, 20326)
times:               (2000,)
image_frame_indices: (667,)
meta:                JSON string
```

`force_mag` is computed as:

```text
force_mag = ||surface_external_forces||
```

### Augmented dataset NPZ files

Files written to `datasets_2d_modified/run_E.../` keep the same structure as the 2D dataset, but their `meta` JSON is extended with augmentation details such as:

- `modified_frames`
- `total_frames`
- `modify_ratio`
- `noise_percentage`
- `displacement_noise`
- `modified_frame_indices`

## Camera and Rendering Notes

The active camera configured in `brain.py` is fixed to:

```json
{
  "position": [-86.337846, -17.669077, 126.000266],
  "orientation": [0.0491668, -0.296558, 0.0513037, 0.952368],
  "fieldOfView": 45,
  "widthViewport": 1920,
  "heightViewport": 1080,
  "zNear": 0.1,
  "zFar": 1000.0
}
```

The projector and overlay tool both prefer the actual saved image resolution when available, which keeps pixel coordinates aligned even if the camera viewport and exported image size ever differ.

## Practical Notes

- `brain.py` currently uses `texture_outpaint.png` for rendering.
- The active craniotomy restriction file is `data/craniotomy_region_texture_common.npz`.
- Downstream scripts now consume `surface_external_forces`. Do not document or depend on `surface_forces` for the current pipeline.
- Run folders are named `run_E..._nu..._seed...`, not `run_seed_*`.
- `npz_projection.py` and `overlay_vertices_on_images.py` are menu-driven scripts, not pure flag-only CLIs.

## Related Utility Scripts

- `projected_npz_to_csv.py`: optional conversion of NPZ outputs to CSV
- `tools/npz_to_force_heatmaps.py`: force-map visualization helper
- `tools/validate_craniotomy_mask.py`: validate the generated craniotomy mask
- `tools/decimate_mesh.py`: create lighter surface meshes

## Summary

The repository is currently configured for a reproducible 18-run sweep over realistic Young's modulus, Poisson ratio, and seed combinations. The active force-label representation is `intensity` mode, the exported images are JPEGs every third frame, and the downstream learning datasets are built from `surface_external_forces` projected into image space.