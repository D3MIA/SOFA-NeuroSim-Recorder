import os, uuid, numpy as np
import threading, queue
import Sofa, Sofa.Core
from .forces import FORCE_TO_NEWTON


class AnimationRecorder(Sofa.Core.Controller):
    def __init__(self, surface_ogl_model, name="AnimationRecorder",
                 outDir="simulation_output", every=1, auto_export_frames=100,
                 force_every_frame=True, capture_images=True, image_resolution=[1920, 1080],
                 image_every=3,           # capture image every N frames (1=every frame, 3=every 3rd)
                 image_format='jpg',      # 'jpg' (fast) or 'png' (lossless)
                 image_quality=92,        # JPEG quality (ignored for PNG)
                 force_debug_images=False,
                 volume_mo=None, force_sampling_k=8,
                 record_stride=1,          # save every Nth surface vertex (1=all, 2=half, 3=third...)
                 run_name=None, images_subdir="images",
                 # Camera component for visibility-based vertex filtering
                 camera_component=None,   # SOFA InteractiveCamera; if set, only camera-visible surface vertices are saved
                 # Force label representation
                 force_label_mode: str = 'distributed',  # 'distributed' | 'intensity'
                 deformers=None,           # list of deformer objects (needed for intensity mode)
                 # Noise configuration (forces are in Newtons at recording time)
                 force_noise_std: float = 0.0,            # absolute Gaussian std in N
                 force_noise_rel: float = 0.0,            # relative std (e.g., 0.02 for 2% of |F|)
                 force_noise_bias=None,                   # optional constant bias (scalar or 3-list) in N
                 force_noise_outlier_prob: float = 0.0,   # probability per-vertex of a noisy outlier
                 force_noise_outlier_scale: float = 10.0, # multiplies std on outliers
                 force_noise_seed=None,
                 **kw):
        super().__init__(name=name, **kw)
        self.surface_model = surface_ogl_model
        self.every = every
        self.force_every_frame = force_every_frame
        self.auto_export_frames = auto_export_frames
        self.capture_images = capture_images
        self.force_debug_images = force_debug_images
        self.image_resolution = image_resolution
        self.image_every = max(1, int(image_every))
        self.image_format = image_format.lower().strip('.')
        self.image_quality = int(image_quality)
        self.step = 0
        # Async image save queue (background thread to avoid blocking simulation)
        self._img_queue = queue.Queue(maxsize=32)
        self._img_thread = threading.Thread(target=self._image_save_worker, daemon=True)
        self._img_thread.start()

        self.animation_started = False

        self.images_saved = 0
        self.last_image_method = None

        self.surface_positions = []
        self.surface_displacements = []
        self.surface_forces = []
        self.surface_external_forces = []
        self.timestamps = []
        self.rest_surface = None
        self.rest_volume = None
        self.total_frames_recorded = 0
        self.last_export_frame = 0
        self.current_npz_frame_index = 0

        self.volume_mo = volume_mo
        self.force_sampling_k = int(max(1, force_sampling_k))
        self.record_stride = max(1, int(record_stride))
        self.force_label_mode = str(force_label_mode).lower().strip()  # 'distributed' or 'intensity'
        self._deformers = list(deformers) if deformers is not None else []
        self._record_idx = None   # set after rest_surface is first captured
        self._camera_component = camera_component  # optional: filter to camera-visible vertices only
        self._force_weights_ready = False
        self._force_neighbor_idx = None
        self._force_neighbor_w = None

        base_out = outDir
        if run_name and run_name.strip():
            base_out = os.path.join(outDir, run_name.strip())
        os.makedirs(base_out, exist_ok=True)
        self.outDir = base_out

        if self.capture_images:
            self.images_dir = os.path.join(self.outDir, images_subdir)
            os.makedirs(self.images_dir, exist_ok=True)

        self.session_id = uuid.uuid4().hex[:8]
        print(f"AnimationRecorder configured (session: {self.session_id})")
        print(f"  outDir={self.outDir}")
        if self.capture_images:
            print(f"  images_dir={self.images_dir}")

        self.session_id = uuid.uuid4().hex[:8]
        print(f"AnimationRecorder configured for OGL surface (session: {self.session_id})")

        # Noise state
        self.force_noise_std = float(force_noise_std or 0.0)
        self.force_noise_rel = float(force_noise_rel or 0.0)
        self.force_noise_bias = None
        if force_noise_bias is not None:
            try:
                b = np.array(force_noise_bias, dtype=np.float32).reshape(-1)
                if b.size == 1:
                    self.force_noise_bias = float(b[0])
                elif b.size == 3:
                    self.force_noise_bias = b.astype(np.float32)
            except Exception:
                self.force_noise_bias = None
        self.force_noise_outlier_prob = float(force_noise_outlier_prob or 0.0)
        self.force_noise_outlier_scale = float(force_noise_outlier_scale or 10.0)
        try:
            self._rng = np.random.default_rng(int(force_noise_seed)) if force_noise_seed is not None else np.random.default_rng()
        except Exception:
            self._rng = np.random.default_rng()

    def _apply_force_noise(self, F: np.ndarray) -> np.ndarray:
        """Apply configurable noise to a (N,3) force array in Newtons.
        - Gaussian zero-mean noise with std = force_noise_std + force_noise_rel*|F|
        - Optional per-vertex outliers that scale std
        - Optional constant bias (scalar or 3D)
        """
        if F is None:
            return None
        if (self.force_noise_std == 0.0 and self.force_noise_rel == 0.0 and self.force_noise_bias is None and self.force_noise_outlier_prob == 0.0):
            return F
        try:
            mag = np.linalg.norm(F, axis=1, keepdims=True).astype(np.float32)
            sigma = (self.force_noise_std + self.force_noise_rel * mag).astype(np.float32)
            sigma = np.clip(sigma, 0.0, np.finfo(np.float32).max)

            if self.force_noise_outlier_prob > 0.0:
                mask = self._rng.random((F.shape[0], 1)) < self.force_noise_outlier_prob
                sigma = np.where(mask, sigma * self.force_noise_outlier_scale, sigma)

            noise = self._rng.normal(loc=0.0, scale=1.0, size=F.shape).astype(np.float32) * sigma
            Fnoisy = F + noise
            if self.force_noise_bias is not None:
                if isinstance(self.force_noise_bias, float) or isinstance(self.force_noise_bias, np.floating):
                    Fnoisy = Fnoisy + float(self.force_noise_bias)
                else:
                    # vector bias
                    Fnoisy = Fnoisy + self.force_noise_bias[None, :]
            return np.nan_to_num(Fnoisy, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return F

    def _quat_to_rotation_matrix(self, q):
        """Quaternion [qx, qy, qz, qw] → 3×3 rotation matrix."""
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n > 1e-12:
            q = q / n
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)

    def _compute_camera_visible_indices(self, vertices):
        """Return indices (into `vertices`) of surface vertices visible from the camera.

        Uses pinhole backprojection with the camera's intrinsic and extrinsic
        parameters (position + quaternion orientation).  Visibility is determined
        at rest pose and kept fixed for the entire recording session — a valid
        approximation for the small deformations produced by SOFA brain surgery
        simulations.

        Returns None if the camera component is unavailable or projection fails,
        so the caller can fall back to saving all stride-subsampled vertices.
        """
        if self._camera_component is None:
            return None
        try:
            pos = np.array(self._camera_component.position.value, dtype=np.float64)
            ori = np.array(self._camera_component.orientation.value, dtype=np.float64)
            fov = float(self._camera_component.fieldOfView.value)
            W = int(self._camera_component.widthViewport.value)
            H = int(self._camera_component.heightViewport.value)
            znear = 0.1
            if hasattr(self._camera_component, 'zNear'):
                try:
                    znear = float(self._camera_component.zNear.value)
                except Exception:
                    pass

            # Camera-space axes from quaternion
            R = self._quat_to_rotation_matrix(ori)
            forward = -R[:, 2]   # camera looks along -Z in OpenGL
            right   =  R[:, 0]
            up      =  R[:, 1]

            # Build 4×4 view matrix
            view = np.array([
                [right[0],    right[1],    right[2],   -np.dot(right,   pos)],
                [up[0],       up[1],       up[2],      -np.dot(up,      pos)],
                [-forward[0], -forward[1], -forward[2], np.dot(forward, pos)],
                [0, 0, 0, 1],
            ], dtype=np.float64)

            V = np.asarray(vertices, dtype=np.float64)
            N = len(V)
            Vh = np.concatenate([V, np.ones((N, 1), dtype=np.float64)], axis=1)
            Vview = Vh @ view.T  # (N, 4)

            # Intrinsics derived from horizontal FOV (matches camera.py convention)
            fx = fy = (W / 2.0) / np.tan(np.radians(fov / 2.0))
            cx, cy = W / 2.0, H / 2.0

            # Visibility: vertex must be in front of the near plane
            in_front = Vview[:, 2] < -znear  # OpenGL: camera looks -Z

            # Perspective projection to pixel coordinates
            pos_depth = np.where(in_front, -Vview[:, 2], 1.0)   # positive depth
            px = np.where(in_front,  fx * Vview[:, 0] / pos_depth + cx, -1.0)
            # Y-axis: OpenGL y-up → image y-down flip
            py = np.where(in_front, -fy * Vview[:, 1] / pos_depth + cy, -1.0)

            in_image = in_front & (px >= 0.0) & (px < W) & (py >= 0.0) & (py < H)
            visible_idx = np.where(in_image)[0]
            return visible_idx

        except Exception as e:
            print(f"[AnimationRecorder] Camera visibility filter failed: {e}")
            return None

    def _capture_rest_positions(self):
        if self.rest_surface is None and hasattr(self.surface_model, 'position'):
            if len(self.surface_model.position.value) > 0:
                full = np.array(self.surface_model.position.value, np.float32)
                stride_idx = np.arange(0, len(full), self.record_stride)

                # Apply camera-visibility filter on top of stride subsampling
                vis_idx = self._compute_camera_visible_indices(full[stride_idx])
                if vis_idx is not None and len(vis_idx) > 0:
                    self._record_idx = stride_idx[vis_idx]
                    print(f"Surface REST captured: {len(full)} vertices → {len(stride_idx)} after stride={self.record_stride} → {len(self._record_idx)} camera-visible")
                else:
                    # Fallback: keep all stride-subsampled vertices
                    self._record_idx = stride_idx
                    if vis_idx is not None and len(vis_idx) == 0:
                        print(f"[AnimationRecorder] WARNING: camera visibility returned 0 vertices; falling back to stride-only ({len(stride_idx)} vertices)")
                    else:
                        print(f"Surface REST captured: {len(full)} vertices → saving {len(self._record_idx)} (stride={self.record_stride}, no camera filter)")

                self.rest_surface = full[self._record_idx]

    def _should_capture_image(self, max_deformation):
        return self.capture_images and (self.step % self.image_every == 0)

    def _image_save_worker(self):
        """Background thread: receives (img, path, quality, fmt) and saves to disk."""
        while True:
            item = self._img_queue.get()
            if item is None:  # sentinel to stop
                break
            try:
                img, path, quality, fmt = item
                if fmt == 'jpg':
                    img.save(path, format='JPEG', quality=quality, optimize=False)
                else:
                    img.save(path, format='PNG')
            except Exception as e:
                print(f"[recorder] async save failed: {e}")
            finally:
                self._img_queue.task_done()

    def _capture_rest_volume(self):
        if self.volume_mo is None:
            return False
        try:
            if hasattr(self.volume_mo, 'rest_position') and len(self.volume_mo.rest_position.value) > 0:
                self.rest_volume = np.array(self.volume_mo.rest_position.value, np.float32)
            elif hasattr(self.volume_mo, 'position') and len(self.volume_mo.position.value) > 0:
                self.rest_volume = np.array(self.volume_mo.position.value, np.float32)
            return self.rest_volume is not None
        except Exception:
            return False

    def _precompute_force_weights(self):
        if self._force_weights_ready or self.volume_mo is None or self.rest_surface is None:
            return
        if self.rest_volume is None:
            if not self._capture_rest_volume():
                return

        V = self.rest_volume
        S = self.rest_surface
        Nv = V.shape[0]
        Ns = S.shape[0]
        K = max(8, self.force_sampling_k)

        def tolerant_unique(a, decimals=6):
            ar = np.round(a.astype(np.float64), decimals)
            uniq = np.unique(ar)
            return uniq, ar

        Xs, Xr = tolerant_unique(V[:, 0])
        Ys, Yr = tolerant_unique(V[:, 1])
        Zs, Zr = tolerant_unique(V[:, 2])

        idx_x = {val: i for i, val in enumerate(Xs)}
        idx_y = {val: i for i, val in enumerate(Ys)}
        idx_z = {val: i for i, val in enumerate(Zs)}

        ix = np.array([idx_x.get(val, 0) for val in Xr], dtype=np.int32)
        iy = np.array([idx_y.get(val, 0) for val in Yr], dtype=np.int32)
        iz = np.array([idx_z.get(val, 0) for val in Zr], dtype=np.int32)

        grid_to_node = {}
        for n in range(Nv):
            key = (int(ix[n]), int(iy[n]), int(iz[n]))
            if key not in grid_to_node:
                grid_to_node[key] = n

        def axis_lookup(axis_vals, p):
            if len(axis_vals) < 2:
                return 0, 0, 0.0
            i1 = int(np.searchsorted(axis_vals, p, side='right'))
            i0 = max(0, min(i1 - 1, len(axis_vals) - 2))
            i1 = i0 + 1
            v0, v1 = axis_vals[i0], axis_vals[i1]
            denom = float(v1 - v0)
            if abs(denom) < 1e-12:
                t = 0.0
            else:
                t = float((p - v0) / denom)
            if t < 0.0: t = 0.0
            if t > 1.0: t = 1.0
            return i0, i1, t

        neigh_idx = np.empty((Ns, 8), dtype=np.int32)
        neigh_w = np.empty((Ns, 8), dtype=np.float32)
        fallback_used = 0

        def knn_weights(p, Kfallback):
            d2 = np.sum((V - p[None, :])**2, axis=1)
            k = min(Kfallback, Nv)
            part = np.argpartition(d2, kth=k-1)[:k]
            part = part[np.argsort(d2[part])]
            dd = np.maximum(1e-12, d2[part])
            w = (1.0 / dd).astype(np.float32)
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            s = float(np.sum(w))
            if s <= 1e-12:
                w[:] = 0.0
                w[0] = 1.0
            else:
                w /= s
            return part, w

        for si in range(Ns):
            p = S[si]
            i0x, i1x, tx = axis_lookup(Xs, p[0])
            i0y, i1y, ty = axis_lookup(Ys, p[1])
            i0z, i1z, tz = axis_lookup(Zs, p[2])

            corners = [
                (i0x, i0y, i0z, (1-tx)*(1-ty)*(1-tz)),
                (i1x, i0y, i0z, (tx)*(1-ty)*(1-tz)),
                (i0x, i1y, i0z, (1-tx)*(ty)*(1-tz)),
                (i1x, i1y, i0z, (tx)*(ty)*(1-tz)),
                (i0x, i0y, i1z, (1-tx)*(1-ty)*(tz)),
                (i1x, i0y, i1z, (tx)*(1-ty)*(tz)),
                (i0x, i1y, i1z, (1-tx)*(ty)*(tz)),
                (i1x, i1y, i1z, (tx)*(ty)*(tz)),
            ]

            have_all = True
            idx8 = np.empty(8, dtype=np.int32)
            w8 = np.empty(8, dtype=np.float32)
            for c, (cx, cy, cz, w) in enumerate(corners):
                key = (cx, cy, cz)
                nid = grid_to_node.get(key, -1)
                if nid < 0:
                    have_all = False
                    break
                idx8[c] = nid
                w8[c] = float(w)

            if have_all:
                wsum = float(np.sum(w8))
                if wsum <= 1e-12:
                    part, w = knn_weights(p, K)
                    if part.shape[0] < 8:
                        pad = 8 - part.shape[0]
                        part = np.pad(part, (0, pad), constant_values=part[0])
                        w = np.pad(w, (0, pad), constant_values=0.0)
                    neigh_idx[si] = part[:8]
                    neigh_w[si] = w[:8]
                    fallback_used += 1
                else:
                    w8 = np.nan_to_num(w8 / wsum, nan=0.0, posinf=0.0, neginf=0.0)
                    neigh_idx[si] = idx8
                    neigh_w[si] = w8
            else:
                part, w = knn_weights(p, K)
                if part.shape[0] < 8:
                    pad = 8 - part.shape[0]
                    part = np.pad(part, (0, pad), constant_values=part[0])
                    w = np.pad(w, (0, pad), constant_values=0.0)
                neigh_idx[si] = part[:8]
                neigh_w[si] = w[:8]
                fallback_used += 1

        self._force_neighbor_idx = neigh_idx
        self._force_neighbor_w = neigh_w
        self._force_weights_ready = True
        row_sums = np.sum(neigh_w, axis=1)
        rs_min = float(np.min(row_sums)) if row_sums.size else 0.0
        rs_max = float(np.max(row_sums)) if row_sums.size else 0.0
        self._force_mapping_info = {
            'Ns': int(Ns),
            'Nv': int(Nv),
            'fallback_rows': int(fallback_used),
            'row_sum_min': rs_min,
            'row_sum_max': rs_max,
        }
        print(f"[AnimationRecorder] Force weights ready: Ns={Ns}, Nv={Nv}, fallback_rows={fallback_used}, row_sum[min,max]=[{rs_min:.6f},{rs_max:.6f}]")

    def _compute_surface_forces(self):
        if self.volume_mo is None or not self._force_weights_ready:
            return None
        try:
            fvol = np.array(self.volume_mo.force.value, dtype=np.float32)
            fvol = np.nan_to_num(fvol, nan=0.0, posinf=0.0, neginf=0.0)
            idx = self._force_neighbor_idx
            w = self._force_neighbor_w
            fv = fvol[idx]
            fs = np.sum(fv * w[..., None], axis=1)
            fs = np.nan_to_num(fs, nan=0.0, posinf=0.0, neginf=0.0)
            return fs
        except Exception:
            return None

    def _compute_surface_external_forces(self):
        if self.volume_mo is None or not self._force_weights_ready:
            return None
        try:
            ext = None
            for name in ('externalForce', 'external_forces', 'external_force', 'externalForces'):
                if hasattr(self.volume_mo, name):
                    data = getattr(self.volume_mo, name)
                    if hasattr(data, 'value'):
                        ext = np.array(data.value, dtype=np.float32)
                        break
            if ext is None:
                return None
            ext = np.nan_to_num(ext, nan=0.0, posinf=0.0, neginf=0.0)
            idx = self._force_neighbor_idx
            w = self._force_neighbor_w
            ev = ext[idx]
            es = np.sum(ev * w[..., None], axis=1)
            es = np.nan_to_num(es, nan=0.0, posinf=0.0, neginf=0.0)

            # Intensity mode: scale so the peak surface vertex reads the true total applied force
            # instead of a tiny KNN-distributed fraction.
            if self.force_label_mode == 'intensity' and self._deformers:
                # Sum all deformer frame-forces to get the total applied force vector (FEM-level)
                F_total_vec = np.zeros(3, dtype=np.float64)
                for d in self._deformers:
                    ff = getattr(d, '_frame_force', None)
                    if ff is not None:
                        if isinstance(ff, list):
                            ff = np.array(ff, dtype=np.float32)
                        if ff.ndim == 2 and ff.shape[1] == 3:
                            F_total_vec += np.sum(ff, axis=0).astype(np.float64)
                F_total = float(np.linalg.norm(F_total_vec))
                if F_total > 1e-9:
                    es_mag = np.linalg.norm(es, axis=1)  # (Ns,)
                    peak = float(es_mag.max())
                    if peak > 1e-9:
                        es = es * (F_total / peak)

            return es
        except Exception:
            return None

    def _capture_image(self, npz_frame_index, timestamp):
        fn = f"frame_{npz_frame_index:04d}.png"
        path = os.path.join(self.images_dir, fn)

        if self._capture_opengl_brain_only(path, npz_frame_index):
            self.last_image_method = 'opengl_clean'
            print(f"[recorder] capture={self.last_image_method} → {os.path.relpath(path)}")
            return True

        if self._capture_brain_only_image(path, npz_frame_index):
            self.last_image_method = 'brain_only'
            print(f"[recorder] capture={self.last_image_method} → {os.path.relpath(path)}")
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
                    print(f"[recorder] capture={self.last_image_method} → {os.path.relpath(path)}")
                    return True
        except Exception:
            pass

        if hasattr(self, '_create_clean_debug_image'):
            self._create_clean_debug_image(path, npz_frame_index)
            self.last_image_method = 'debug_fallback'
            print(f"[recorder] capture={self.last_image_method} → {os.path.relpath(path)}")
            return True

        return False
    def _create_clean_debug_image(self, image_path, npz_frame_index):
        # Fallback: simple image noire avec le numéro de frame
        try:
            from PIL import Image, ImageDraw, ImageFont
            w, h = int(self.image_resolution[0]), int(self.image_resolution[1])
            img = Image.new('RGB', (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            text = f"DEBUG FALLBACK\nframe {npz_frame_index:04d}"
            # Police par défaut
            draw.text((20, 20), text, fill=(220, 220, 220))
            img.save(image_path)
            return True
        except Exception:
            return False

    def _capture_opengl_brain_only(self, image_path, npz_frame_index):
        try:
            import OpenGL.GL as gl
            from PIL import Image

            vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
            w, h = int(vp[2]), int(vp[3])

            if w <= 0 or h <= 0:
                print(f"[capture:opengl] FAIL: viewport is {w}x{h} (window minimized or no GL context)")
                return False

            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            img = Image.frombytes('RGB', (w, h), pixels)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

            target_size = tuple(self.image_resolution)
            if (w, h) != target_size:
                img = img.resize(target_size, Image.BILINEAR)  # BILINEAR >> LANCZOS for speed

            # Queue save to background thread — don't block simulation
            ext = 'jpg' if self.image_format == 'jpg' else 'png'
            final_path = os.path.splitext(image_path)[0] + '.' + ext
            try:
                self._img_queue.put_nowait((img, final_path, self.image_quality, self.image_format))
            except queue.Full:
                img.save(final_path)  # fallback: save synchronously if queue full
            return True
        except Exception as e:
            print(f"[capture:opengl] FAIL: {type(e).__name__}: {e}")
            return False

    def _capture_brain_only_image(self, image_path, npz_frame_index):
        try:
            ctx = self.getContext()
            root = ctx.getRootContext() if hasattr(ctx, 'getRootContext') else ctx.getRoot()
            hidden = []

            self._temporarily_hide_debug_elements(root, hidden)

            try:
                import Sofa.Gui as G
                mgr = None
                # Try different API names across SOFA versions
                for method in ('getInstance', 'GetGUI', 'getGUI', 'get'):
                    fn = getattr(G.GUIManager, method, None)
                    if callable(fn):
                        try:
                            mgr = fn()
                            break
                        except Exception:
                            pass
                # Some versions expose the GUI directly as an attribute
                if mgr is None:
                    for attr in ('gui', 'GUI', '_gui'):
                        mgr = getattr(G.GUIManager, attr, None)
                        if mgr is not None:
                            break
                print(f"[capture:sofa_gui] mgr={mgr}, has takeScreenshot={hasattr(mgr, 'takeScreenshot') if mgr else False}")
                if mgr and hasattr(mgr, 'takeScreenshot'):
                    mgr.takeScreenshot(image_path)
                    self._restore_debug_elements(hidden)
                    return True
                # Last resort: SendMessage / saveScreenshot on root viewer
                if mgr and hasattr(mgr, 'saveScreenshot'):
                    mgr.saveScreenshot(image_path)
                    self._restore_debug_elements(hidden)
                    return True
            except Exception as e:
                print(f"[capture:sofa_gui] FAIL: {type(e).__name__}: {e}")

            comps = []
            self._find_viewer_components(root, comps)
            print(f"[capture:viewer] found {len(comps)} viewer components")
            for v in comps:
                if hasattr(v, 'saveScreenshot'):
                    v.saveScreenshot(image_path)
                    self._restore_debug_elements(hidden)
                    return True

            self._restore_debug_elements(hidden)
        except Exception as e:
            print(f"[capture:brain_only] FAIL: {type(e).__name__}: {e}")
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
        if self.rest_volume is None and self.volume_mo is not None:
            self._capture_rest_volume()
        if (not self._force_weights_ready) and (self.rest_surface is not None) and (self.rest_volume is not None):
            self._precompute_force_weights()

        if hasattr(self.surface_model, 'position') and len(self.surface_model.position.value) > 0:
            pos = np.array(self.surface_model.position.value, np.float32)
        else:
            print("Surface model positions not available")
            self.step += 1
            return

        ts = float(self.getContext().time.value)

        # Apply vertex subsampling (record_stride > 1 reduces saved vertex count)
        if self._record_idx is not None and self.record_stride > 1:
            pos = pos[self._record_idx]

        self.surface_positions.append(pos)

        if self.rest_surface is not None:
            displacement = pos - self.rest_surface
            self.surface_displacements.append(displacement)
        else:
            displacement = np.zeros_like(pos)
            self.surface_displacements.append(displacement)

        self.timestamps.append(ts)
        surf_ext_forces = self._compute_surface_external_forces()
        if surf_ext_forces is None:
            self.surface_external_forces.append(None)
        else:
            efN = (surf_ext_forces.astype(np.float32) * FORCE_TO_NEWTON)
            efN = self._apply_force_noise(efN)
            self.surface_external_forces.append(efN)
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
            # Auto-quit after the first full export so batch sweep can continue
            if len(self.surface_positions) >= self.auto_export_frames:
                print(f"[recorder] {self.auto_export_frames} frames recorded – closing SOFA.")
                self._img_queue.join()  # wait for all images to finish saving
                import sys
                sys.exit(0)

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
        ext_forces_list = []
        has_ext_forces = False
        for i, f in enumerate(self.surface_external_forces):
            if f is None:
                if self.rest_surface is not None:
                    ext_forces_list.append(np.zeros_like(self.rest_surface, dtype=np.float32))
                else:
                    ext_forces_list.append(np.zeros((0, 3), dtype=np.float32))
            else:
                has_ext_forces = True
                ext_forces_list.append(f.astype(np.float32))
        save_kwargs = dict(
            rest=self.rest_surface,
            frames=np.stack(self.surface_positions),
            displacements=np.stack(self.surface_displacements),
            times=np.array(self.timestamps, np.float32)
        )

        ext_force_quality_summary = None
        if has_ext_forces:
            ext_arr = np.stack(ext_forces_list)
            save_kwargs['surface_external_forces'] = ext_arr
            try:
                norms = np.linalg.norm(ext_arr, axis=2)
                frame_max = norms.max(axis=1)
                frame_mean = norms.mean(axis=1)
                frame_p95 = np.percentile(norms, 95, axis=1)
                nonfinite = (~np.isfinite(ext_arr)).sum(axis=(1, 2))
                zeros = (norms <= 1e-12).sum(axis=1)
                if frame_max.shape[0] > 1:
                    prev = np.maximum(frame_max[:-1], 1e-9)
                    spikes = int(np.sum((frame_max[1:] > 10.0 * prev) & (frame_max[1:] > 1e-6)))
                else:
                    spikes = 0
                zero_frames = int(np.sum(frame_max <= 1e-12))
                numeric_ok = (int(np.sum(nonfinite)) == 0) and (zero_frames < len(self.timestamps))
                temporal_ok = (spikes <= max(1, len(self.timestamps) // 10))

                ext_force_quality_summary = {
                    'overall_max': float(np.max(frame_max)) if frame_max.size else 0.0,
                    'overall_mean_of_means': float(np.mean(frame_mean)) if frame_mean.size else 0.0,
                    'overall_mean_p95': float(np.mean(frame_p95)) if frame_p95.size else 0.0,
                    'total_nonfinite_elements': int(np.sum(nonfinite)),
                    'zero_frames': zero_frames,
                    'spike_frames': spikes,
                    'frames': len(self.timestamps),
                    'numeric_ok': bool(numeric_ok),
                    'temporal_ok': bool(temporal_ok),
                }

                qcsv_e = os.path.join(self.outDir, f"brain_surface_{self.session_id}{suf}_external_force_quality.csv")
                with open(qcsv_e, 'w') as qf:
                    qf.write('frame,max_norm_N,mean_norm_N,p95_norm_N,zero_count,nonfinite_count\n')
                    for i in range(norms.shape[0]):
                        qf.write(f"{i},{float(frame_max[i]):.6f},{float(frame_mean[i]):.6f},{float(frame_p95[i]):.6f},{int(zeros[i])},{int(nonfinite[i])}\n")
                print(f"External force quality CSV exported: {qcsv_e}")
            except Exception:
                pass

        np.savez_compressed(main, **save_kwargs)
        print(f"NPZ exported{(' with external_forces' if has_ext_forces else ' (no external forces)')}: {len(self.surface_positions)} frames → {main}")

        csv_file = os.path.join(self.outDir, f"brain_surface_{self.session_id}{suf}_summary.csv")
        with open(csv_file, 'w') as f:
            f.write("frame,time,max_displacement_mm,max_external_force_N\n")
            for i, t in enumerate(self.timestamps):
                if i < len(self.surface_displacements):
                    disp = self.surface_displacements[i]
                    max_d = float(np.max(np.linalg.norm(disp, axis=1)))
                else:
                    max_d = 0.0
                if i < len(self.surface_external_forces) and self.surface_external_forces[i] is not None:
                    efcur = self.surface_external_forces[i]
                    max_ef = float(np.max(np.linalg.norm(efcur, axis=1)))
                else:
                    max_ef = 0.0
                f.write(f"{i},{t:.6f},{max_d:.6f},{max_ef:.6f}\n")
        print(f"CSV summary exported: {csv_file}")

        meta = os.path.join(self.outDir, f"brain_surface_{self.session_id}{suf}_meta.json")
        import json
        m = {
            'session_id': self.session_id,
            'frame_count': len(self.timestamps),
            'duration': float(self.timestamps[-1]) if self.timestamps else 0.0,
            'surface_vertex_count': len(self.rest_surface) if self.rest_surface is not None else 0,
            'image_every': self.image_every,
            'image_format': self.image_format,
            'record_stride': self.record_stride,
            'camera_visibility_filter': self._camera_component is not None,
            'data_type': 'surface_mesh_positions_displacements_forces',
            'npz_keys': ['rest', 'frames', 'displacements', 'times'] +
                        (['surface_external_forces'] if ('surface_external_forces' in save_kwargs) else [])
        }
        if hasattr(self, '_force_mapping_info') and isinstance(getattr(self, '_force_mapping_info'), dict):
            m['force_mapping'] = self._force_mapping_info
            m['force_mapping']['note'] = 'row_sum near 1.0 indicates normalized gather weights; fallback_rows > 0 means KNN was used for some vertices.'
        if 'surface_external_forces' in save_kwargs and ext_force_quality_summary is not None:
            m['external_force_quality'] = ext_force_quality_summary
        m['units'] = {
            "length": "mm",
            "time": "s",
            "mass": "kg",
            "displacement": "mm",
            "surface_external_forces": "N",
            "force_to_newton": FORCE_TO_NEWTON
        }
        # Record noise configuration for reproducibility
        m['force_noise'] = {
            'std_abs_N': self.force_noise_std,
            'std_rel': self.force_noise_rel,
            'bias': (float(self.force_noise_bias) if isinstance(self.force_noise_bias, (int, float, np.floating)) else (self.force_noise_bias.tolist() if isinstance(self.force_noise_bias, np.ndarray) else None)),
            'outlier_prob': self.force_noise_outlier_prob,
            'outlier_scale': self.force_noise_outlier_scale,
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
