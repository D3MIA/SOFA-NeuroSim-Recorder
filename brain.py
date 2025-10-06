import os, uuid, numpy as np
import Sofa, Sofa.Core
from simlib import (
    SimpleCameraExtractor, CameraAutoFramer, VisualWatchdog,
    AnimationRecorder, BatchRecorder, DeformationPrinter, QuadSlideDeformer,
    RandomDeformer, SurgicalToolDeformer, TemporaryForwardPusher,
    ExternalForceAggregator, FORCE_TO_NEWTON, DeepPressPusher,
)


def createScene(root):
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

    root.gravity = [0, -9.81, 0]
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
    br.addObject("UniformMass", totalMass=1.00)
    br.addObject("ParallelTetrahedronFEMForceField", youngModulus=500, poissonRatio=0.4)
    br.addObject("DiagonalVelocityDampingForceField", dampingCoefficient=2.0)

    vis = br.addChild("Visual")
    # Prefer decimated mesh if available; fallback to full-resolution mesh
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
        stiffness=8e4,
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
    tool_seed = 1111
    run_subdir = f"run_seed_{tool_seed}"
    run_dir = os.path.join("simulation_output", run_subdir)

    # Save camera params and an initial screenshot into the same run directory
    camera_extractor = SimpleCameraExtractor(camera_component=interactive_camera, outDir=run_dir)
    root.addObject(camera_extractor)

    # Configure measurement noise (set to 0.0 to disable)
    force_noise_std_N = 0.0       # epsilon in Newtons (absolute). e.g., 0.05 for ±0.05N std
    force_noise_rel = 0.0         # relative std (fraction of |F|). e.g., 0.02 for 2%
    # Optional extras
    # Set a small constant bias so the epsilon is not zero-mean; use scalar or a vector [bx,by,bz]
    force_noise_bias = 0.02       # e.g., 0.02 N or [0.0, 0.0, -0.02]
    force_noise_outlier_prob = 0.0    # probability of outlier per vertex
    force_noise_outlier_scale = 10.0  # outlier std multiplier

    recorder = AnimationRecorder(
        surface_ogl_model=surface_model,
        every=1,
        force_every_frame=True,
        auto_export_frames=2000,
        capture_images=True,
        force_debug_images=False,
        image_resolution=[1920, 1080],
        outDir="simulation_output",
        volume_mo=br.dofs,
        force_sampling_k=8,
        run_name=run_subdir,
        # Noise knobs
        force_noise_std=force_noise_std_N,
        force_noise_rel=force_noise_rel,
        force_noise_bias=force_noise_bias,
        force_noise_outlier_prob=force_noise_outlier_prob,
        force_noise_outlier_scale=force_noise_outlier_scale,
        force_noise_seed=tool_seed,
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

        # SLIDE plus nerveux (relâche + rapide)
        slide_displacement=10.0,
        slide_force=3.6e4,          # un peu plus fort
        ramp_in=8, hold_frames=12, ramp_out=1,   # quasi continu: relâche minimale
        release_frames=0,                         # pas de pause entre directions
        cooldown_between_points=0, 

        inward_dir=(0.0, 0.0, -1.0),
        inward_bias=0.18,

        # PUSH plus présent et plus ferme
        push_probability=0.0,       # pas de push aléatoire pour continuité
        push_force=4.2e4,           # un cran au-dessus
        push_radius=8.0,
        push_frames=12,             # un peu plus long

        apply_to_mo=True,
        name="quadSlide",
    )
    br.addObject(tool)

    # Pour tester un appui profond, décommentez ce bloc:
    # press = DeepPressPusher(
    #     mo=br.dofs,
    #     surface_model=surface_model,
    #     seed=tool_seed,
    #     region_surface_npz=region_npz,
    #     restrict_to_region=True,
    #
    #     force_value=5.0e4,
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
