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
                from PIL import Image
                
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


class CameraAutoFramer(Sofa.Core.Controller):
    def __init__(self, camera_component, target_model, pad=1.2, set_once=True, **kw):
        super().__init__(**kw)
        self.camera = camera_component
        self.target = target_model
        self.pad = float(pad)
        self.set_once = set_once
        self._done = False

    def onAnimateBeginEvent(self, *_):
        if self._done and self.set_once:
            return
        try:
            if not hasattr(self.target, 'position'):
                return
            pos = list(self.target.position.value)
            if not pos:
                return
            P = np.asarray(pos, dtype=np.float32)
            bb_min = P.min(axis=0)
            bb_max = P.max(axis=0)
            center = (bb_min + bb_max) * 0.5
            radius = float(np.linalg.norm(bb_max - bb_min)) * 0.5
            if radius <= 1e-6:
                radius = 10.0

            fov_deg = float(self.camera.fieldOfView.value) if hasattr(self.camera, 'fieldOfView') else 45.0
            fov = np.radians(max(10.0, min(120.0, fov_deg)))
            dist = (radius * self.pad) / max(1e-6, np.tan(0.5 * fov))

            new_pos = [float(center[0]), float(center[1]), float(center[2] + dist)]
            if hasattr(self.camera, 'position'):
                self.camera.position.value = new_pos
            if hasattr(self.camera, 'lookAt'):
                self.camera.lookAt.value = [float(center[0]), float(center[1]), float(center[2])]
            if hasattr(self.camera, 'orientation'):
                self.camera.orientation.value = [0.0, 0.0, 0.0, 1.0]

            self._done = True
        except Exception:
            pass
