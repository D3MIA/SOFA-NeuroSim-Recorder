import os, uuid, numpy as np
import Sofa, Sofa.Core

class SimpleCameraExtractor(Sofa.Core.Controller):
    
    def __init__(self, camera_component, outDir="simulation_output", **kw):
        super().__init__(**kw)
        self.camera = camera_component
        self.outDir = outDir
        self.extracted = False
        os.makedirs(outDir, exist_ok=True)
        
    def onAnimateBeginEvent(self, *_):
        if not self.extracted:
            try:
                pos = list(self.camera.position.value)
                orientation = list(self.camera.orientation.value) if hasattr(self.camera, 'orientation') else [0, 0, 0, 1]
                lookAt = list(self.camera.lookAt.value) if hasattr(self.camera, 'lookAt') else [0, 0, 0]
                fov = float(self.camera.fieldOfView.value)
                width = int(self.camera.widthViewport.value)
                height = int(self.camera.heightViewport.value)
                
                fx = fy = (width / 2.0) / np.tan(np.radians(fov / 2.0))
                cx = width / 2.0
                cy = height / 2.0
                
                all_camera_params = {}
                
                camera_attributes = [
                    'position', 'orientation', 'lookAt', 'fieldOfView', 
                    'widthViewport', 'heightViewport', 'distance', 
                    'zNear', 'zFar', 'zoomSpeed', 'panSpeed', 'pivot',
                    'activated', 'projectionType', 'projectionMatrix', 
                    'modelViewMatrix', 'computeZ', 'minBBox', 'maxBBox'
                ]
                
                for attr in camera_attributes:
                    if hasattr(self.camera, attr):
                        attr_obj = getattr(self.camera, attr)
                        if hasattr(attr_obj, 'value'):
                            value = attr_obj.value
                            if isinstance(value, (list, tuple)):
                                all_camera_params[attr] = list(value)
                            elif hasattr(value, '__iter__') and not isinstance(value, str):
                                try:
                                    all_camera_params[attr] = list(value)
                                except:
                                    all_camera_params[attr] = str(value)
                            else:
                                all_camera_params[attr] = value
                        else:
                            all_camera_params[attr] = str(attr_obj)
                    else:
                        all_camera_params[attr] = None
                
                camera_params = {
                    "camera_type": "InteractiveCamera",
                    "sofa_version": "25.06",
                    "extracted_on_animate": True,
                    "timestamp": str(uuid.uuid4()),
                    "method": "complete_parameter_extraction",
                    
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "intrinsic_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                }
                
                camera_params.update(all_camera_params)
                
                import json
                session_id = uuid.uuid4().hex[:8]
                json_file = os.path.join(self.outDir, f"camera_params_COMPLETE_{session_id}.json")
                with open(json_file, 'w') as f:
                    json.dump(camera_params, f, indent=2)
                
                screenshot_file = os.path.join(self.outDir, f"frame_0000.png")
                self._take_fullscreen_screenshot(screenshot_file)
                
                self.extracted = True
                
            except Exception as e:
                pass

    def _take_fullscreen_screenshot(self, image_path):
        try:
            import Sofa.Gui as G
            from PIL import Image
            
            mgr = getattr(G, 'GUIManager', None)
            if mgr:
                inst = mgr.getInstance()
                if hasattr(inst, 'takeScreenshot'):
                    temp_path = image_path.replace('.png', '_temp.png')
                    inst.takeScreenshot(temp_path)
                    
                    try:
                        img = Image.open(temp_path)
                        target_size = (1920, 1080)
                        img_resized = img.resize(target_size, Image.LANCZOS)
                        img_resized.save(image_path)
                        
                        import os
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        return True
                    except Exception:
                        pass
                        
            try:
                import OpenGL.GL as gl
                
                vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
                w, h = vp[2], vp[3]
                
                gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
                pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                
                img = Image.frombytes('RGB', (w, h), pixels)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                target_size = (1920, 1080)
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(image_path)
                return True
            except Exception:
                pass
        except Exception:
            pass
        return False


class AnimationRecorder(Sofa.Core.Controller):
    def __init__(self, surface_ogl_model, name="AnimationRecorder", 
                 outDir="simulation_output", every=1, auto_export_frames=100,
                 force_every_frame=True, capture_images=True, image_resolution=[1920, 1080], 
                 force_debug_images=False, **kw):
        super().__init__(name=name, **kw)
        self.surface_model = surface_ogl_model
        self.every = every
        self.force_every_frame = force_every_frame
        self.auto_export_frames = auto_export_frames
        self.capture_images = capture_images
        self.force_debug_images = force_debug_images
        self.image_resolution = image_resolution
        self.step = 0

        self.animation_started = False

        self.images_saved = 0
        self.last_image_method = None

        self.surface_positions = []
        self.surface_displacements = []
        self.timestamps = []
        self.rest_surface = None
        self.total_frames_recorded = 0
        self.last_export_frame = 0
        self.current_npz_frame_index = 0

        os.makedirs(outDir, exist_ok=True)
        self.outDir = outDir
        if self.capture_images:
            self.images_dir = os.path.join(outDir, "images")
            os.makedirs(self.images_dir, exist_ok=True)

        self.session_id = uuid.uuid4().hex[:8]
        print(f"AnimationRecorder configured for OGL surface (session: {self.session_id})")

    def _capture_rest_positions(self):
        if self.rest_surface is None and hasattr(self.surface_model, 'position'):
            if len(self.surface_model.position.value) > 0:
                self.rest_surface = np.array(self.surface_model.position.value, np.float32)
                print(f"Surface REST captured: {len(self.rest_surface)} vertices")

    def _should_capture_image(self, max_deformation):
        return self.capture_images

    def _capture_image(self, npz_frame_index, timestamp):
        fn = f"frame_{npz_frame_index:04d}.png"
        path = os.path.join(self.images_dir, fn)
        
        if self._capture_opengl_brain_only(path, npz_frame_index):
            self.last_image_method = 'opengl_clean'
            return True
            
        if self._capture_brain_only_image(path, npz_frame_index):
            self.last_image_method = 'brain_only'
            return True
            
        try:
            import Sofa.Gui as G
            hidden = []
            root = self.getContext().getRoot()
            self._temporarily_hide_debug_elements(root, hidden)
            mgr = getattr(G, 'GUIManager', None)
            if mgr:
                inst = mgr.getInstance()
                if hasattr(inst, 'takeScreenshot'):
                    inst.takeScreenshot(path)
                    self._restore_debug_elements(hidden)
                    self.last_image_method = 'sofa_gui'
                    return True
        except Exception:
            pass
            
        if hasattr(self, '_create_clean_debug_image'):
            self._create_clean_debug_image(path, npz_frame_index)
            self.last_image_method = 'debug_fallback'
            return True
            
        return False

    def _capture_opengl_brain_only(self, image_path, npz_frame_index):
        try:
            import OpenGL.GL as gl
            from PIL import Image
            
            vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
            w, h = vp[2], vp[3]
            
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            
            img = Image.frombytes('RGB', (w, h), pixels)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            target_size = tuple(self.image_resolution)
            img = img.resize(target_size, Image.LANCZOS)
            img.save(image_path)
            
            return True
        except Exception as e:
            pass
            return False

    def _capture_brain_only_image(self, image_path, npz_frame_index):
        try:
            ctx = self.getContext()
            root = ctx.getRootContext() if hasattr(ctx, 'getRootContext') else ctx.getRoot()
            hidden = []
            
            self._temporarily_hide_debug_elements(root, hidden)
            
            try:
                import Sofa.Gui as G
                mgr = G.GUIManager.getInstance()
                if hasattr(mgr, 'takeScreenshot'):
                    mgr.takeScreenshot(image_path)
                    self._restore_debug_elements(hidden)
                    return True
            except:
                pass
                
            comps = []
            self._find_viewer_components(root, comps)
            for v in comps:
                if hasattr(v, 'saveScreenshot'):
                    v.saveScreenshot(image_path)
                    self._restore_debug_elements(hidden)
                    return True
                    
            self._restore_debug_elements(hidden)
        except:
            pass
        return False

    def _temporarily_hide_debug_elements(self, node, hidden_objects):
        if not hasattr(node, 'getChildren'):
            return
        for child in node.getChildren():
            self._temporarily_hide_debug_elements(child, hidden_objects)
            if hasattr(child, 'getObjects'):
                for obj in child.getObjects():
                    name = type(obj).__name__.lower()
                    if any(p in name for p in ('visualmodel', 'forcefield', 'constraint', 'arrow', 'vector', 'debug', 'grid', 'frame')):
                        if hasattr(obj, 'showVisual') and hasattr(obj.showVisual, 'value') and obj.showVisual.value:
                            hidden_objects.append((obj, 'showVisual', True))
                            obj.showVisual.value = False
                        elif hasattr(obj, 'drawMode') and hasattr(obj.drawMode, 'value') and obj.drawMode.value != 'None':
                            hidden_objects.append((obj, 'drawMode', obj.drawMode.value))
                            obj.drawMode.value = 'None'

    def _restore_debug_elements(self, hidden_objects):
        for obj, attr, val in hidden_objects:
            if hasattr(obj, attr):
                a = getattr(obj, attr)
                if hasattr(a, 'value'):
                    a.value = val

    def _find_viewer_components(self, node, components):
        if hasattr(node, 'getObjects'):
            for obj in node.getObjects():
                nm = obj.getName().lower() if hasattr(obj, 'getName') else ''
                if any(k in nm for k in ('viewer', 'camera', 'visual')):
                    components.append(obj)
        if hasattr(node, 'getChildren'):
            for c in node.getChildren():
                self._find_viewer_components(c, components)

    def onAnimateEndEvent(self, *_):
        if not self.animation_started:
            self.animation_started = True
            print("Animation detected - starting surface recording")

        if not self.force_every_frame and (self.step % self.every != 0):
            self.step += 1
            return
        
        self._capture_rest_positions()
        
        if hasattr(self.surface_model, 'position') and len(self.surface_model.position.value) > 0:
            pos = np.array(self.surface_model.position.value, np.float32)
        else:
            print("Surface model positions not available")
            self.step += 1
            return
            
        ts = float(self.getContext().time.value)
        
        self.surface_positions.append(pos)
        
        if self.rest_surface is not None:
            displacement = pos - self.rest_surface
            self.surface_displacements.append(displacement)
        else:
            displacement = np.zeros_like(pos)
            self.surface_displacements.append(displacement)
        
        self.timestamps.append(ts)
        self.total_frames_recorded += 1
        
        max_def = 0.0
        if self.rest_surface is not None:
            current_displacement = self.surface_displacements[-1]
            max_def = np.max(np.linalg.norm(current_displacement, axis=1))
        
        current_npz_index = len(self.surface_positions) - 1
        
        if self._should_capture_image(max_def):
            if self._capture_image(current_npz_index, ts):
                self.images_saved += 1
                print(f"Image {current_npz_index:04d} captured (deformation: {max_def:.3f}mm)")
        
        if len(self.surface_positions) % self.auto_export_frames == 0:
            print(f"Auto-export at {len(self.surface_positions)} frames")
            self._export_simulation_data(auto=True)
            self.last_export_frame = self.total_frames_recorded
        
        self.step += 1

    def onEndAnimation(self, *_):
        print("Animation finished - final export")
        self._export_simulation_data(auto=False)
        
    def _export_simulation_data(self, auto=False):
        print(f"{'Automatic' if auto else 'Final'} export in progress...")
        
        if not self.surface_positions:
            print("No surface data to export")
            return
            
        suf = '_auto' if auto else '_final'
        main = os.path.join(self.outDir, f"brain_surface_{self.session_id}{suf}.npz")
        
        np.savez_compressed(main,
                            rest=self.rest_surface,
                            frames=np.stack(self.surface_positions),
                            displacements=np.stack(self.surface_displacements),
                            times=np.array(self.timestamps, np.float32))
        print(f"NPZ surface exported with displacements: {len(self.surface_positions)} frames -> {main}")

        csv_file = os.path.join(self.outDir, f"brain_surface_{self.session_id}{suf}_summary.csv")
        with open(csv_file, 'w') as f:
            f.write("frame,time,max_displacement\n")
            for i, t in enumerate(self.timestamps):
                if i < len(self.surface_displacements):
                    disp = self.surface_displacements[i]
                    max_d = np.max(np.linalg.norm(disp, axis=1))
                else:
                    max_d = 0.0
                f.write(f"{i},{t:.6f},{max_d:.6f}\n")
        print(f"CSV summary exported: {csv_file}")

        meta = os.path.join(self.outDir, f"brain_surface_{self.session_id}{suf}_meta.json")
        import json
        m = {
            'session_id': self.session_id,
            'frame_count': len(self.timestamps),
            'duration': float(self.timestamps[-1]) if self.timestamps else 0.0,
            'surface_vertex_count': len(self.rest_surface) if self.rest_surface is not None else 0,
            'data_type': 'surface_mesh_positions_and_displacements',
            'npz_keys': ['rest', 'frames', 'displacements', 'times']
        }
        with open(meta, 'w') as f:
            json.dump(m, f, indent=2)
        print(f"Metadata exported: {meta}")


class BatchRecorder(Sofa.Core.Controller):
    def __init__(self, mo, name="batchRecorder", outDir="results",
                 every=1, batchFrames=3000, **kw):
        super().__init__(name=name, **kw)
        self.mo, self.every, self.batch = mo, every, batchFrames
        self.pos, self.vel, self.time = [], [], []
        self.step, self.bid = 0, 0
        os.makedirs(outDir, exist_ok=True)
        self.outDir = outDir

    def _dump(self):
        if not self.pos:
            return
        tag = f"{uuid.uuid4().hex[:8]}_b{self.bid:04d}"
        path = os.path.join(self.outDir, f"run_{tag}.npz")
        np.savez_compressed(path,
                            pos=np.stack(self.pos),
                            vel=np.stack(self.vel),
                            time=np.array(self.time, np.float32))
        self.pos = []; self.vel = []; self.time = []
        self.bid += 1

    def onAnimateEndEvent(self, *_):
        if self.step % self.every == 0:
            self.pos.append(np.array(self.mo.position.value, np.float32))
            self.vel.append(np.array(self.mo.velocity.value, np.float32))
            self.time.append(float(self.getContext().time.value))
        self.step += 1
        if self.step >= self.batch:
            self._dump()
            self.step = 0

    def onEndAnimation(self, *_):
        self._dump()


class DeformationPrinter(Sofa.Core.Controller):
    def __init__(self, mo, name="printer",
                 disp_start=0.01, disp_stop=0.003,
                 quiet_frames=5, every=1, top=5, **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.s2, self.e2 = disp_start**2, disp_stop**2
        self.q, self.qcnt, self.active = quiet_frames, 0, False
        self.every, self.top = every, top
        self.f, self.rest = 0, None

    def _ready(self):
        if self.rest is None and len(self.mo.position.value) > 0:
            self.rest = np.array(self.mo.rest_position.value, np.float32)
        return self.rest is not None

    def onAnimateEndEvent(self, *_):
        if not self._ready():
            self.f += 1
            return

        disp = np.asarray(self.mo.position.value, np.float32) - self.rest
        d2 = np.sum(disp * disp, axis=1)
        dmax = d2.max()

        if self.active:
            if dmax < self.e2:
                self.qcnt += 1
                if self.qcnt >= self.q:
                    self.active = False
        else:
            if dmax > self.s2:
                self.active = True
                self.qcnt = 0

        if self.active and (self.f % self.every == 0):
            idx = np.argsort(-d2)[:self.top]
            for i in idx:
                pass

        self.f += 1


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
        root.addObject("DisplayFlagsDataFields", 
                       showVisual=True,
                       showBehavior=False,
                       showMapping=False,
                       showForceFields=False,
                       showNormals=False,
                       showWireFrame=False)
    except:
        pass

    root.gravity = [0, 0, 0]
    root.dt = 0.005
    root.addObject("DefaultAnimationLoop")
    
    root.addObject("BackgroundSetting", color=[0.05, 0.05, 0.05, 1])
    
    root.addObject("LightManager")
    
    root.addObject("DirectionalLight", name="MainLight", direction=[0, 0, -1], color="3 3 3")
    root.addObject("DirectionalLight", name="FillLight", direction=[0.3, 0.3, -0.8], color="1.5 1.5 1.8")
    root.addObject("CollisionPipeline")
    root.addObject("ParallelBruteForceBroadPhase")
    root.addObject("ParallelBVHNarrowPhase")
    root.addObject("MinProximityIntersection",
                   alarmDistance=1.0, contactDistance=1.0)
    root.addObject("CollisionResponse",
                   response="PenaltyContactForceField",
                   responseParams="k=1e6")

    br = root.addChild("Brain")
    br.addObject("EulerImplicitSolver")
    br.addObject("CGLinearSolver", iterations=200, threshold=1e-9, tolerance=1e-9)
    br.addObject("SparseGridTopology",
                 fileTopology="data/volume_simplified.obj",
                 n=[16, 16, 16])
    br.addObject("MechanicalObject", name="dofs")
    br.addObject("UniformMass", totalMass=0.01)
    br.addObject("ParallelTetrahedronFEMForceField",
                 youngModulus=500, poissonRatio=0.4)
    br.addObject("DiagonalVelocityDampingForceField",
                 dampingCoefficient=2.0)
    vis = br.addChild("Visual")
    vis.addObject("MeshOBJLoader", name="surf",
                  filename="data/surface_full.obj")
    surface_model = vis.addObject("OglModel", name="surfaceModel", src="@surf",
                  texturename="data/texture_outpaint.png")
    vis.addObject("TriangleCollisionModel")
    vis.addObject("BarycentricMapping", input="@../dofs")

    pins = [
        [-70, -50, -60, 50, 50, -20],
        [20, -50, -50, 70, 50, 50],
        [-70, 30, -60, 70, 50, 50],
        [-70, -40, -60, 70, -60, 50],
    ]
    for i, box in enumerate(pins):
        roi = br.addObject("BoxROI", name=f"pinROI{i}",
                           box=box, drawBoxes=False)
        br.addObject("FixedProjectiveConstraint",
                     name=f"pin{i}",
                     indices=f"@pinROI{i}.indices")

    ring = br.addObject("SphereROI", name="ring",
                        centers=[0, 25, 45], radii=[12],
                        drawSphere=False)
    br.addObject("RestShapeSpringsForceField",
                 name="holeSprings",
                 points="@ring.indices",
                 stiffness=8e4, angularStiffness=0)

    br.addObject(DeformationPrinter(mo=br.dofs))

    interactive_camera = root.addObject("InteractiveCamera", 
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
                                       activated=True)
    
    print("InteractiveCamera created with fixed parameters")
    
    try:
        interactive_camera.position.value = [-86.337846, -17.669077, 126.000266]
        interactive_camera.orientation.value = [0.0491668, -0.296558, 0.0513037, 0.952368]
        print("SOFA camera parameters configured")
    except Exception as e:
        print(f"Unable to force parameters: {e}")
    
    camera_extractor = SimpleCameraExtractor(camera_component=interactive_camera)
    root.addObject(camera_extractor)
    
    recorder = AnimationRecorder(
        surface_ogl_model=surface_model,
        every=1,
        force_every_frame=True,
        auto_export_frames=100,
        capture_images=True,
        force_debug_images=False,
        image_resolution=[1920, 1080],
        outDir="simulation_output"
    )
    
    br.addObject(recorder)

    root.addObject("MouseInteractor", template="Vec3d", listening=1)

    return root
