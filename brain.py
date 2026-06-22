import os, uuid, numpy as np
import Sofa, Sofa.Core
from simlib import (
    SimpleCameraExtractor, CameraAutoFramer, VisualWatchdog,
    AnimationRecorder, BatchRecorder, DeformationPrinter, QuadSlideDeformer,
    RandomDeformer, SurgicalToolDeformer, TemporaryForwardPusher,
    ExternalForceAggregator, FORCE_TO_NEWTON, DeepPressPusher,
)


def createScene(root):
   
    # ── Simulation parameters (overridable via environment variables) ──────────
    YOUNG   = float(os.environ.get("BRAIN_YOUNG",   "3.0"))   # kPa  (internal unit = kPa in mm/kg/s)
    POISSON = float(os.environ.get("BRAIN_POISSON", "0.45"))  # dimensionless
    SEED    = int(os.environ.get("BRAIN_SEED",     "1111"))   # RNG seed
    print(f"[brain.py] E={YOUNG} kPa  nu={POISSON}  seed={SEED}")
    # ────────────────────────────────────────────────────────────────────────────
    plugins = [
        "MultiThreading",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.GL.Component.Shader",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Controller",
        "Sofa.GUI.Component",
        "Sofa.Component.Setting",
    ]

    for i, pl in enumerate(plugins):
        root.addObject("RequiredPlugin", name=f"Plugin_{i:02d}", pluginName=pl)

    try:
        root.addObject(
            "DisplayFlagsDataFields",
            showVisual=True,
            showBehavior=False,
            showMapping=False,
            showForceFields=False,
            showNormals=False,
            showWireFrame=False,
        )
    except Exception:
        pass

    root.gravity = [0, 0, 0]   # Brain in CSF → near-neutral buoyancy → net gravity ≈ 0 (physically correct)
    root.dt = 0.03
    root.addObject("DefaultAnimationLoop")

    root.addObject("BackgroundSetting", color=[0.05, 0.05, 0.05, 1])
    root.addObject("LightManager")
    root.addObject("DirectionalLight", name="MainLight", direction=[0, 0, -1], color="3 3 3")
    root.addObject("DirectionalLight", name="FillLight", direction=[0.3, 0.3, -0.8], color="1.5 1.5 1.8")
    root.addObject("CollisionPipeline")
    root.addObject("ParallelBruteForceBroadPhase")
    root.addObject("ParallelBVHNarrowPhase")
    root.addObject("MinProximityIntersection", alarmDistance=1.0, contactDistance=1.0)
    root.addObject(
        "CollisionResponse",
        response="PenaltyContactForceField",
        responseParams="k=1e6",
    )

    br = root.addChild("Brain")
    br.addObject("EulerImplicitSolver")
    br.addObject("CGLinearSolver", iterations=200, threshold=1e-9, tolerance=1e-9)
    br.addObject(
        "SparseGridTopology",
        fileTopology="data/volume_simplified.obj",
        n=[16, 16, 16],
    )
    br.addObject("MechanicalObject", name="dofs")
    br.addObject("UniformMass", totalMass=1.25)  # ~1.04 g/cm³ × 1200 cm³ ≈ 1.25 kg
    br.addObject("ParallelTetrahedronFEMForceField", youngModulus=YOUNG, poissonRatio=POISSON)
    br.addObject("DiagonalVelocityDampingForceField", dampingCoefficient=1.5)  # scaled with stiffness: ~1.5 N·s/m equivalent

    vis = br.addChild("Visual")
    # Prefer decimated mesh if available; fallback to full-resolution mesh
    # ~30k faces / ~15k vertices: good balance of speed and surface detail
    decimated_path = os.path.join("data", "surface_full_decimated.obj")
    full_path = os.path.join("data", "surface_full.obj")
    mesh_file = decimated_path if os.path.exists(decimated_path) else full_path
    mesh_file = os.path.abspath(mesh_file)
    loader = vis.addObject("MeshOBJLoader", name="surf", filename=mesh_file)
    surface_model = vis.addObject(
        "OglModel",
        name="surfaceModel",
        src="@surf",
        texturename="data/texture_outpaint.png",
        color=[1.0, 1.0, 1.0, 1.0],
    )
    vis.addObject("TriangleCollisionModel")
    vis.addObject("BarycentricMapping", input="@../dofs")

    pins = [
        [-70, -50, -60, 50, 50, -20],
        [20, -50, -50, 70, 50, 50],
        [-70, 30, -60, 70, 50, 50],
        [-70, -40, -60, 70, -60, 50],
    ]
    for i, box in enumerate(pins):
        br.addObject("BoxROI", name=f"pinROI{i}", box=box, drawBoxes=False)
        br.addObject(
            "FixedProjectiveConstraint",
            name=f"pin{i}",
            indices=f"@pinROI{i}.indices",
        )

    br.addObject(
        "SphereROI",
        name="ring",
        centers=[0, 25, 45],
        radii=[12],
        drawSphere=False,
    )
    br.addObject(
        "RestShapeSpringsForceField",
        name="holeSprings",
        points="@ring.indices",
        stiffness=500,  # 500 mN/mm = 0.5 N/mm – soft craniotomy boundary
        angularStiffness=0,
    )

    br.addObject(DeformationPrinter(mo=br.dofs))

    interactive_camera = root.addObject(
        "InteractiveCamera",
        name="recordingCamera",
        position=[-86.337846, -17.669077, 126.000266],
        orientation=[0.0491668, -0.296558, 0.0513037, 0.952368],
        distance=10.0,
        fieldOfView=45,
        widthViewport=1920,
        heightViewport=1080,
        zNear=0.1,
        zFar=1000.0,
        zoomSpeed=250,
        panSpeed=0.1,
        pivot=2,
        activated=True,
    )

    print("InteractiveCamera created with fixed parameters")
    try:
        interactive_camera.position.value = [-86.337846, -17.669077, 126.000266]
        interactive_camera.orientation.value = [0.0491668, -0.296558, 0.0513037, 0.952368]
        if hasattr(interactive_camera, "computeZ") and hasattr(interactive_camera.computeZ, "value"):
            interactive_camera.computeZ.value = False
        print("SOFA camera parameters configured")
    except Exception as e:
        print(f"Unable to force parameters: {e}")

    # Auto-framing disabled to keep camera parameters stable during deformation

    # Guard: if editing loader params clears the surface, auto-reload
    try:
        vis_guard = VisualWatchdog(loader=loader, ogl_model=surface_model, check_every=10)
        vis.addObject(vis_guard)
    except Exception:
        pass

    # Use a single seed value for both the tool and recorder run naming
    tool_seed = SEED
    # Run folder encodes all parameters for easy identification
    run_subdir = f"run_E{YOUNG:.2f}_nu{POISSON:.3f}_seed{tool_seed}"
    run_dir = os.path.join("simulation_output", run_subdir)

    # Save camera params and an initial screenshot into the same run directory
    camera_extractor = SimpleCameraExtractor(camera_component=interactive_camera, outDir=run_dir)
    root.addObject(camera_extractor)

    # Configure measurement noise (set to 0.0 to disable)
    force_noise_std_N = 0.0       # epsilon in Newtons (absolute). e.g., 0.05 for ±0.05N std
    force_noise_rel = 0.0         # relative std (fraction of |F|). e.g., 0.02 for 2%
    # Optional extras
    # Set a small constant bias so the epsilon is not zero-mean; use scalar or a vector [bx,by,bz]
    force_noise_bias = 0.0            # 0.0 = no bias; any nonzero adds a uniform floor to ALL vertices (destroys spatial localization)
    force_noise_outlier_prob = 0.0    # probability of outlier per vertex
    force_noise_outlier_scale = 10.0  # outlier std multiplier

    recorder = AnimationRecorder(
        surface_ogl_model=surface_model,
        every=1,
        force_every_frame=True,
        auto_export_frames=1000,
        capture_images=True,
        image_every=3,              # capture image every 3 frames → 3× less GPU stall
        image_format='jpg',         # JPEG: ~5× faster to write than PNG
        image_quality=92,           # 92% quality, visually lossless
        force_debug_images=False,
        image_resolution=[1920, 1080],
        outDir="simulation_output",
        volume_mo=br.dofs,
        force_sampling_k=8,
        record_stride=2,            # save every 2nd vertex: 40k → 20k (2× less data, same quality)
        camera_component=interactive_camera,  # filter to camera-visible surface vertices only
        run_name=run_subdir,
        # Noise knobs
        force_noise_std=force_noise_std_N,
        force_noise_rel=force_noise_rel,
        force_noise_bias=force_noise_bias,
        force_noise_outlier_prob=force_noise_outlier_prob,
        force_noise_outlier_scale=force_noise_outlier_scale,
        force_noise_seed=tool_seed,
        # Force label representation for ML training
        force_label_mode='intensity',   # 'intensity': peak vertex = true applied force (best for direct displacement→force mapping)
        deformers=[],                   # will be set after tool is created below
    )

    region_npz = (
        os.path.exists(os.path.join("data", "craniotomy_region_texture_common.npz"))
        and os.path.join("data", "craniotomy_region_texture_common.npz")
    ) or (
        os.path.exists(os.path.join("data", "craniotomy_region_texture.npz"))
        and os.path.join("data", "craniotomy_region_texture.npz")
    ) or None
    # region_npz est déjà préparé plus haut dans brain.py
    
    tool = QuadSlideDeformer(
        mo=br.dofs,
        surface_model=surface_model,
        seed=tool_seed,
        region_surface_npz=region_npz,
        restrict_to_region=True,

        # Sliding gesture timing  (dt=0.03s → 1 frame = 30 ms)
        slide_displacement=10.0,
        slide_force_range=(200, 1000),  # 0.20–1.00 N, drawn fresh each direction
        ramp_in=8,                  # 0.24 s smooth ramp up
        hold_frames=15,             # nominal hold (overridden by hold_min/max)
        hold_min=10, hold_max=20,   # 0.30–0.60 s varied sustained contact
        ramp_out=8,                 # 0.24 s smooth release (was 1 → shock)
        release_frames=5,           # 0.15 s rest between directions (was 0 → jerky)
        cooldown_between_points=10, # 0.30 s rest between tool positions

        inward_dir=(0.0, 0.0, -1.0),
        inward_bias=0.25,           # 25% inward component → realistic indentation

        # Occasional push for gesture variety
        push_probability=0.2,       # 20% chance of a push after each full sequence
        push_force_range=(150, 800),  # 0.15–0.80 N total, normalized across nodes
        push_radius=20.0,           # mm – matches slide radius for consistent tool size
        push_frames=15,             # 0.45 s sustained push

        apply_to_mo=True,
        name="quadSlide",
    )
    br.addObject(tool)

    # Aggregate all deformer forces and re-assert externalForce at onAnimateEndEvent
    # so the recorder can read the correct value (SOFA resets externalForce during solve)
    aggregator = ExternalForceAggregator(mo=br.dofs, deformers=[tool], name="forceAggregator")
    br.addObject(aggregator)
    recorder._deformers = [tool]   # wire deformer into recorder after tool is instantiated

    # Pour tester un appui profond, décommentez ce bloc:
    # press = DeepPressPusher(
    #     mo=br.dofs,
    #     surface_model=surface_model,
    #     seed=tool_seed,
    #     region_surface_npz=region_npz,
    #     restrict_to_region=True,
    #
    #     force_value=400,          # 400 mN = 0.4 N (firm but safe probe press)
    #     radius=9.0,
    #     ramp_in=10, hold_frames=20, ramp_out=10,
    #     cooldown_frames=8,
    #
    #     inward_dir=(0.0, 0.0, -1.0),
    #     use_surface_normal=True,
    #     normal_neighbors=24,
    #
    #     apply_to_mo=True,
    #     name="deepPress",
    # )
    # br.addObject(press)

    # Add the recorder after aggregator to ensure end-of-frame re-assert is visible to it
    br.addObject(recorder)

    root.addObject("MouseInteractor", template="Vec3d", listening=1)

    return root
