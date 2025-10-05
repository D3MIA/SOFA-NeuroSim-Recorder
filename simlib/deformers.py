import os, numpy as np
import Sofa, Sofa.Core


class RandomDeformer(Sofa.Core.Controller):
    """
    Apply randomized, localized external forces over the exposed (top) region
    to generate automatic, varied deformations across the craniotomy.

    - Picks a random center within the allowed region every `cooldown_frames`.
    - Applies a Gaussian falloff force over `burst_frames` frames.
    - Writes into MechanicalObject.externalForce so it is recorded/exported.
    """
    def __init__(self, mo, surface_model=None, name="randomDeformer",
                 seed=0,
                 radius_range=(6.0, 14.0),
                 force_range=(6e3, 2.2e4),
                 cooldown_frames=18,
                 inward_dir=(0.0, 0.0, -1.0),
                 jitter_angle_deg=8.0,
                 region_surface_npz=None,
                 region_center_jitter=1.5,
                 restrict_to_region=True,
                 ramp_in=5, hold_frames=10, ramp_out=6,
                 drift_radius=1.2,  # lateral micro-drift (mm)
                 **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.surface_model = surface_model
        self.rng = np.random.default_rng(int(seed))
        self.radius_range = (float(radius_range[0]), float(radius_range[1]))
        self.force_range = (float(force_range[0]), float(force_range[1]))
        self.cooldown_frames = int(cooldown_frames)
        self.inward_dir = np.array(inward_dir, dtype=np.float32)
        self.inward_dir /= max(1e-9, np.linalg.norm(self.inward_dir))
        self.jitter_angle = float(jitter_angle_deg)
        if not hasattr(self, '_region_loaded'):
            self.region_surface_npz = region_surface_npz
            self.region_center_jitter = float(region_center_jitter)
            self._region_surface_indices = None
            self._region_loaded = False

            self._candidates = None
            self._burst_left = 0
            self._cooldown_left = 0
            self._center_pos = None
            self._radius = None
            self._amp = None
            self._dir = None
            self.restrict_to_region = bool(restrict_to_region)
            self.ramp_in = int(max(0, ramp_in))
            self.hold_frames = int(max(0, hold_frames))
            self.ramp_out = int(max(0, ramp_out))
            self.drift_radius = float(max(0.0, drift_radius))
            total_phase = self.ramp_in + self.hold_frames + self.ramp_out
            if total_phase == 0:
                self.ramp_in, self.hold_frames, self.ramp_out = 1, 4, 1
                total_phase = 6
            self.phase_total = total_phase
            self._phase_frame = 0
            self._volume_region_mask = None
            self._region_volume_indices = None

    def _ensure_region_loaded(self):
        if not self.region_surface_npz or self._region_loaded:
            return
        try:
            if os.path.exists(self.region_surface_npz):
                data = np.load(self.region_surface_npz)
                if 'indices' in data:
                    inds = np.array(data['indices']).astype(np.int32)
                    if self.surface_model is not None and hasattr(self.surface_model, 'position') and len(self.surface_model.position.value) > 0:
                        vmax = len(self.surface_model.position.value)
                        inds = inds[(inds >= 0) & (inds < vmax)]
                    if inds.size > 0:
                        self._region_surface_indices = inds
                        print(f"[RandomDeformer] Region loaded: {inds.size} surface verts")
                    else:
                        print("[RandomDeformer] Region NPZ contained no valid indices")
            else:
                print(f"[RandomDeformer] Region NPZ not found: {self.region_surface_npz}")
        except Exception as e:
            print(f"[RandomDeformer] Failed loading region NPZ: {e}")
        self._region_loaded = True

    def _build_volume_region_mask(self):
        if self._volume_region_mask is not None:
            return
        self._ensure_region_loaded()
        if self._region_surface_indices is None or self.surface_model is None or not hasattr(self.surface_model, 'position'):
            return
        try:
            surf_pos = np.array(self.surface_model.position.value, dtype=np.float32)
            if surf_pos.size == 0:
                return
            region_set = set(int(i) for i in self._region_surface_indices.tolist())
            Pvol = np.array(self.mo.position.value, dtype=np.float32)
            if Pvol.size == 0:
                return
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(surf_pos)
                _, nearest = tree.query(Pvol, k=1)
            except Exception:
                d2 = np.sum((Pvol[:, None, :] - surf_pos[None, :, :])**2, axis=2)
                nearest = np.argmin(d2, axis=1)
            mask = np.array([1 if int(i) in region_set else 0 for i in nearest], dtype=bool)
            self._volume_region_mask = mask
            self._region_volume_indices = np.nonzero(mask)[0].astype(np.int32)
            print(f"[RandomDeformer] Volume region mask: {mask.sum()}/{mask.size} nodes inside region")
        except Exception as e:
            print(f"[RandomDeformer] Failed building volume mask: {e}")

    def _pick_new_burst(self):
        # Prefer surface region vertices as selection centers
        self._build_volume_region_mask()
        P = np.array(self.mo.position.value, dtype=np.float32)
        surf_pos = None
        if self.surface_model is not None and hasattr(self.surface_model, 'position'):
            try:
                surf_pos = np.array(self.surface_model.position.value, dtype=np.float32)
            except Exception:
                surf_pos = None
        if self.restrict_to_region and self._region_surface_indices is not None and surf_pos is not None and surf_pos.size > 0:
            s_idx = int(self.rng.choice(self._region_surface_indices))
            center = surf_pos[s_idx]
        elif surf_pos is not None and surf_pos.size > 0:
            s_idx = int(self.rng.integers(0, surf_pos.shape[0]))
            center = surf_pos[s_idx]
        else:
            # Fallback to volume DOFs if surface unavailable
            v_idx = int(self.rng.integers(0, P.shape[0])) if P.shape[0] else 0
            center = P[v_idx]

        rad = float(self.rng.uniform(self.radius_range[0], self.radius_range[1]))
        amp = float(self.rng.uniform(self.force_range[0], self.force_range[1]))

        ang = np.radians(self.jitter_angle)
        jitter = self.rng.normal(0.0, np.tan(ang) * 0.3, size=3)
        jitter[2] = 0.0
        direction = self.inward_dir + jitter.astype(np.float32)
        nrm = np.linalg.norm(direction)
        if nrm > 1e-9:
            direction /= nrm

        self._center_pos = center
        self._radius = rad
        self._amp = amp
        self._dir = direction
        self._burst_left = self.phase_total
        self._phase_frame = 0
        return True

    def _apply_burst(self):
        P = np.array(self.mo.position.value, dtype=np.float32)
        if P.size == 0 or self._center_pos is None:
            return
        if self.drift_radius > 0 and self._phase_frame > 0:
            drift = self.rng.normal(0.0, self.drift_radius * 0.15, size=3).astype(np.float32)
            drift[2] *= 0.05
            self._center_pos = self._center_pos + drift
        d = np.linalg.norm(P - self._center_pos[None, :], axis=1)
        sigma = max(1e-3, 0.5 * self._radius)
        w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        if self.restrict_to_region and self._volume_region_mask is not None:
            w *= self._volume_region_mask.astype(np.float32)
        pf = self._phase_frame
        if pf < self.ramp_in:
            scale = (pf + 1) / max(1, self.ramp_in)
        elif pf < self.ramp_in + self.hold_frames:
            scale = 1.0
        else:
            down = pf - (self.ramp_in + self.hold_frames)
            scale = max(0.0, 1.0 - (down + 1) / max(1, self.ramp_out))
        F = (w[:, None] * (self._amp * scale * self._dir[None, :]).astype(np.float32))
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if hasattr(self.mo, 'externalForce') and hasattr(self.mo.externalForce, 'value'):
                self.mo.externalForce.value = F.tolist()
            else:
                if hasattr(self.mo, 'external_force') and hasattr(self.mo.external_force, 'value'):
                    self.mo.external_force.value = F.tolist()
        except Exception:
            pass

    def _clear_forces(self):
        try:
            N = len(self.mo.position.value)
            zeros = [[0.0, 0.0, 0.0]] * N
            if hasattr(self.mo, 'externalForce') and hasattr(self.mo.externalForce, 'value'):
                self.mo.externalForce.value = zeros
            elif hasattr(self.mo, 'external_force') and hasattr(self.mo.external_force, 'value'):
                self.mo.external_force.value = zeros
        except Exception:
            pass

    def onAnimateBeginEvent(self, *_):
        if self._burst_left > 0:
            self._apply_burst()
            self._burst_left -= 1
            self._phase_frame += 1
            if self._burst_left == 0:
                self._clear_forces()
                self._cooldown_left = self.cooldown_frames
            return

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._clear_forces()
            return

        if not self._pick_new_burst():
            self._clear_forces()
            return
        self._apply_burst()

    def onEndAnimation(self, *_):
        self._clear_forces()


class SurgicalToolDeformer(Sofa.Core.Controller):
    def __init__(self, mo, surface_model=None, name="surgicalTool",
                 seed=0,
                 region_surface_npz=None,
                 force_range=(5e3, 1.8e4),
                 radius_range=(5.0, 12.0),
                 ramp_in=6, hold_frames=12, ramp_out=6,
                 cooldown_frames=(12, 28),
                 drift_radius=1.0,
                 sweep_displacement=6.0,
                 circle_radius=5.0,
                 circle_speed=0.35,
                 shear_factor=0.4,
                 inward_dir=(0.0, 0.0, -1.0),
                 modes=("press","sweep","press","circle"),
                 mode_weights=None,
                 restrict_to_region=True,
                 **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.surface_model = surface_model
        self.region_surface_npz = region_surface_npz
        self.restrict = bool(restrict_to_region)
        self.force_range = force_range
        self.radius_range = radius_range
        self.rng = np.random.default_rng(int(seed))
        self.ramp_in = ramp_in
        self.hold_frames = hold_frames
        self.ramp_out = ramp_out
        self.cooldown_frames_range = cooldown_frames if isinstance(cooldown_frames, (tuple,list)) else (cooldown_frames, cooldown_frames)
        self.drift_radius = drift_radius
        self.sweep_disp = sweep_displacement
        self.circle_radius = circle_radius
        self.circle_speed = circle_speed
        self.shear_factor = shear_factor
        self.inward_dir = np.array(inward_dir, np.float32)
        nrm = np.linalg.norm(self.inward_dir)
        if nrm > 1e-9:
            self.inward_dir /= nrm
        self.modes = list(modes)
        self.mode_weights = np.array(mode_weights, dtype=np.float32) if mode_weights is not None else None
        if self.mode_weights is not None:
            self.mode_weights = self.mode_weights / max(1e-9, self.mode_weights.sum())
        self._region_loaded = False
        self._region_surface_indices = None
        self._volume_region_mask = None
        self._region_volume_indices = None
        self._phase_total = self.ramp_in + self.hold_frames + self.ramp_out
        if self._phase_total <= 0: self._phase_total = 1
        self._phase_frame = 0
        self._burst_left = 0
        self._cooldown_left = 0
        self._center = None
        self._radius = None
        self._amp = None
        self._dir = self.inward_dir.copy()
        self._mode = 'press'
        self._sweep_vec = None
        self._circle_angle = 0.0

    def _load_region(self):
        if self._region_loaded: return
        if self.region_surface_npz and os.path.exists(self.region_surface_npz):
            try:
                data = np.load(self.region_surface_npz)
                if 'indices' in data:
                    self._region_surface_indices = np.array(data['indices']).astype(np.int32)
                    print(f"[SurgicalToolDeformer] Loaded region surface indices: {self._region_surface_indices.size}")
            except Exception as e:
                print(f"[SurgicalToolDeformer] Failed loading region: {e}")
        self._region_loaded = True

    def _build_volume_mask(self):
        if self._volume_region_mask is not None: return
        self._load_region()
        if self._region_surface_indices is None or self.surface_model is None or not hasattr(self.surface_model,'position'):
            return
        try:
            surf_pos = np.array(self.surface_model.position.value, np.float32)
            Pvol = np.array(self.mo.position.value, np.float32)
            if surf_pos.size == 0 or Pvol.size == 0: return
            region_set = set(int(i) for i in self._region_surface_indices.tolist())
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(surf_pos)
                _, nearest = tree.query(Pvol, k=1)
            except Exception:
                d2 = np.sum((Pvol[:,None,:]-surf_pos[None,:,:])**2, axis=2)
                nearest = np.argmin(d2, axis=1)
            mask = np.array([int(i) in region_set for i in nearest], dtype=bool)
            self._volume_region_mask = mask
            self._region_volume_indices = np.nonzero(mask)[0].astype(np.int32)
            print(f"[SurgicalToolDeformer] Volume region nodes: {mask.sum()}/{mask.size}")
        except Exception as e:
            print(f"[SurgicalToolDeformer] Failed building volume mask: {e}")

    def _pick_mode(self):
        if not self.modes:
            return 'press'
        if self.mode_weights is None:
            return self.rng.choice(self.modes)
        return self.rng.choice(self.modes, p=self.mode_weights)

    def _start_gesture(self):
        self._build_volume_mask()
        P = np.array(self.mo.position.value, np.float32)
        # Prefer surface region vertices for centers
        surf_pos = None
        if self.surface_model is not None and hasattr(self.surface_model, 'position'):
            try:
                surf_pos = np.array(self.surface_model.position.value, np.float32)
            except Exception:
                surf_pos = None
        if self.restrict and self._region_surface_indices is not None and surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.choice(self._region_surface_indices))
            base = surf_pos[sidx]
        elif surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.integers(0, surf_pos.shape[0]))
            base = surf_pos[sidx]
        else:
            vidx = int(self.rng.integers(0, P.shape[0])) if P.shape[0] else 0
            base = P[vidx] if P.size else np.zeros(3, np.float32)
        jitter = self.rng.normal(0.0, 1.0, size=3).astype(np.float32)*0.5
        jitter[2] *= 0.2
        self._center = base + jitter
        self._radius = float(self.rng.uniform(*self.radius_range))
        self._amp = float(self.rng.uniform(*self.force_range))
        self._mode = self._pick_mode()
        if self._mode == 'sweep':
            vec = self.rng.normal(0.0,1.0,size=3).astype(np.float32)
            vec[2] = 0.0
            n = np.linalg.norm(vec)
            if n>1e-9: vec/=n
            self._sweep_vec = vec * self.sweep_disp
        else:
            self._sweep_vec = None
        if self._mode == 'circle':
            self._circle_angle = 0.0
        self._phase_frame = 0
        self._burst_left = self._phase_total
        ang = np.radians(6.0)
        jitter_d = self.rng.normal(0.0, np.tan(ang)*0.25, size=3).astype(np.float32)
        jitter_d[2] *= 0.1
        base_dir = self.inward_dir + jitter_d
        nrm = np.linalg.norm(base_dir)
        if nrm>1e-9: base_dir/=nrm
        self._dir = base_dir

    def _apply(self):
        if self._center is None:
            return
        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            return
        if self._mode == 'sweep' and self._sweep_vec is not None:
            t = (self._phase_frame / max(1, self._phase_total - 1))
            tri = 2.0 * (0.5 - abs(t - 0.5))
            target = self._center + (tri - 0.5) * 2.0 * self._sweep_vec
            self._active_center = target
        elif self._mode == 'circle':
            self._circle_angle += self.circle_speed
            offset = np.array([np.cos(self._circle_angle), np.sin(self._circle_angle), 0.0], np.float32) * self.circle_radius
            self._active_center = self._center + offset
        else:
            if self.drift_radius > 0 and self._phase_frame > 0:
                drift = self.rng.normal(0.0, self.drift_radius * 0.2, size=3).astype(np.float32)
                drift[2] *= 0.05
                self._center += drift
            self._active_center = self._center
        center = self._active_center
        d = np.linalg.norm(P - center[None, :], axis=1)
        sigma = max(1e-3, 0.5 * self._radius)
        w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        if self.restrict and self._volume_region_mask is not None:
            w *= self._volume_region_mask.astype(np.float32)
        pf = self._phase_frame
        if pf < self.ramp_in:
            scale = (pf + 1) / max(1, self.ramp_in)
        elif pf < self.ramp_in + self.hold_frames:
            scale = 1.0
        else:
            down = pf - (self.ramp_in + self.hold_frames)
            scale = max(0.0, 1.0 - (down + 1) / max(1, self.ramp_out))

        dir_vec = self._dir.copy()
        if self._mode in ('sweep', 'circle') and self.shear_factor > 0:
            if self._phase_frame > 0:
                tang = (center - self._last_center) if hasattr(self, '_last_center') else np.zeros(3, np.float32)
                tang[2] = 0.0
                nt = np.linalg.norm(tang)
                if nt > 1e-6:
                    tang /= nt
                    dir_vec = (1.0 - self.shear_factor) * dir_vec + self.shear_factor * tang
                    nd = np.linalg.norm(dir_vec)
                    if nd > 1e-9:
                        dir_vec /= nd
        self._last_center = center.copy()

        F = (w[:, None] * (self._amp * scale) * dir_vec[None, :]).astype(np.float32)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        self._frame_force = F

    def _clear(self):
        self._frame_force = None

    def onAnimateBeginEvent(self,*_):
        if self._burst_left>0:
            self._apply()
            self._burst_left -= 1
            self._phase_frame += 1
            if self._burst_left==0:
                self._clear()
                self._cooldown_left = int(self.rng.integers(self.cooldown_frames_range[0], self.cooldown_frames_range[1]+1))
            return
        if self._cooldown_left>0:
            self._cooldown_left -= 1
            self._clear()
            return
        self._start_gesture()
        self._apply()

    def onEndAnimation(self,*_):
        self._clear()


class TemporaryForwardPusher(Sofa.Core.Controller):
    def __init__(self, mo, surface_model=None, name="tempPusher",
                 seed=777, region_surface_npz=None, restrict_to_region=True,
                 force_value=1.6e4, radius=8.0,
                 burst_frames=12, cooldown_frames=10,
                 direction=(0.0, 0.0, -1.0),
                 **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.surface_model = surface_model
        self.rng = np.random.default_rng(int(seed))
        self.region_surface_npz = region_surface_npz
        self.restrict = bool(restrict_to_region)
        self.force_value = float(force_value)
        self.radius = float(radius)
        self.burst_frames = int(max(1, burst_frames))
        self.cooldown_frames = int(max(0, cooldown_frames))
        self.dir = np.array(direction, dtype=np.float32)
        n = np.linalg.norm(self.dir)
        if n > 1e-9:
            self.dir /= n
        self._region_loaded = False
        self._region_surface_indices = None
        self._volume_region_mask = None
        self._region_volume_indices = None
        self._center = None
        self._left = 0
        self._cooldown = 0
        self._frame_force = None

    def _load_region(self):
        if self._region_loaded:
            return
        if self.region_surface_npz and os.path.exists(self.region_surface_npz):
            try:
                data = np.load(self.region_surface_npz)
                if 'indices' in data:
                    self._region_surface_indices = np.array(data['indices']).astype(np.int32)
                    print(f"[TemporaryForwardPusher] Region loaded: {self._region_surface_indices.size} surface verts")
            except Exception as e:
                print(f"[TemporaryForwardPusher] Region load failed: {e}")
        self._region_loaded = True

    def _build_volume_mask(self):
        if self._volume_region_mask is not None:
            return
        self._load_region()
        if self._region_surface_indices is None or self.surface_model is None or not hasattr(self.surface_model, 'position'):
            return
        try:
            surf_pos = np.array(self.surface_model.position.value, np.float32)
            Pvol = np.array(self.mo.position.value, np.float32)
            if surf_pos.size == 0 or Pvol.size == 0:
                return
            region_set = set(int(i) for i in self._region_surface_indices.tolist())
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(surf_pos)
                _, nearest = tree.query(Pvol, k=1)
            except Exception:
                d2 = np.sum((Pvol[:, None, :] - surf_pos[None, :, :])**2, axis=2)
                nearest = np.argmin(d2, axis=1)
            mask = np.array([int(i) in region_set for i in nearest], dtype=bool)
            self._volume_region_mask = mask
            self._region_volume_indices = np.nonzero(mask)[0].astype(np.int32)
            print(f"[TemporaryForwardPusher] Volume region nodes: {mask.sum()}/{mask.size}")
        except Exception as e:
            print(f"[TemporaryForwardPusher] Volume mask failed: {e}")

    def _pick_center(self):
        self._build_volume_mask()
        P = np.array(self.mo.position.value, dtype=np.float32)
        if P.size == 0:
            self._center = None
            return
        # Prefer surface region vertices
        surf_pos = None
        if self.surface_model is not None and hasattr(self.surface_model, 'position'):
            try:
                surf_pos = np.array(self.surface_model.position.value, dtype=np.float32)
            except Exception:
                surf_pos = None
        if self.restrict and self._region_surface_indices is not None and surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.choice(self._region_surface_indices))
            base = surf_pos[sidx]
        elif surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.integers(0, surf_pos.shape[0]))
            base = surf_pos[sidx]
        else:
            vidx = int(self.rng.integers(0, P.shape[0]))
            base = P[vidx]
        jitter = self.rng.normal(0.0, 0.8, size=3).astype(np.float32)
        jitter[2] *= 0.2
        self._center = base + jitter
        self._left = self.burst_frames

    def _apply(self):
        if self._center is None:
            self._frame_force = None
            return
        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            self._frame_force = None
            return
        d = np.linalg.norm(P - self._center[None, :], axis=1)
        sigma = max(1e-3, 0.5 * self.radius)
        w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        if self.restrict and self._volume_region_mask is not None:
            w *= self._volume_region_mask.astype(np.float32)
        F = (w[:, None] * (self.force_value * self.dir[None, :])).astype(np.float32)
        self._frame_force = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _clear(self):
        self._frame_force = None

    def onAnimateBeginEvent(self, *_):
        if self._left > 0:
            self._apply()
            self._left -= 1
            if self._left == 0:
                self._clear()
                self._cooldown = self.cooldown_frames
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            self._clear()
            return
        self._pick_center()
        self._apply()


class QuadSlideDeformer(Sofa.Core.Controller):
    """
    Automatic sequence on a craniotomy point:
      8 directions (cardinal + diagonal) in random order per point,
      with a release period (zero force) between each direction.
    The force center remains fixed for the entire phase (no back-and-forth).
    Gaussian weights are cached per phase to avoid oscillations.
    """

    def __init__(self, mo, surface_model=None, name="quadSlide",
                 seed=123,
                 region_surface_npz=None,
                 restrict_to_region=True,
                 # sliding
                 radius=7.0,
                 slide_displacement=10.0,
                 slide_force=3.2e4,
                 ramp_in=8, hold_frames=12, hold_min=5, hold_max=10, ramp_out=8,
                 release_frames=5,
                 cooldown_between_points=10,
                 # indentation
                 inward_dir=(0.0, 0.0, -1.0),
                 inward_bias=0.18,
                 # occasional pushes
                 push_probability=0.35,
                 push_force=3.6e4,
                 push_radius=8.0,
                 push_frames=10,
                 # direct write to externalForce
                 apply_to_mo=True,
                 **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.surface_model = surface_model
        self.rng = np.random.default_rng(int(seed))

        # Region / constraint
        self.region_surface_npz = region_surface_npz
        self.restrict = bool(restrict_to_region)

        # Sliding/pushing parameters
        self.radius = float(radius)
        self.slide_disp = float(slide_displacement)
        self.slide_force = float(slide_force)
        self.ramp_in = int(max(1, ramp_in))
        self.hold_frames = int(max(0, hold_frames))
        self.hold_min = int(max(1, hold_min))
        self.hold_max = int(max(self.hold_min, hold_max))
        self.ramp_out = int(max(1, ramp_out))
        self.release_frames = int(max(0, release_frames))
        self.cooldown_between_points = int(max(0, cooldown_between_points))

        self.inward_dir = np.asarray(inward_dir, np.float32)
        n = np.linalg.norm(self.inward_dir)
        if n > 1e-9:
            self.inward_dir /= n
        self.inward_bias = float(max(0.0, min(1.0, inward_bias)))

        self.push_probability = float(max(0.0, min(1.0, push_probability)))
        self.push_force = float(push_force)
        self.push_radius = float(push_radius)
        self.push_frames = int(max(1, push_frames))

        self.apply_to_mo = bool(apply_to_mo)

        # Region state
        self._region_loaded = False
        self._region_surface_indices = None
        self._volume_region_mask = None
        self._region_volume_indices = None

        # Unit directions (cardinal)
        self._dir_up = np.array([0.0, 1.0, 0.0], np.float32)
        self._dir_down = np.array([0.0, -1.0, 0.0], np.float32)
        self._dir_right = np.array([1.0, 0.0, 0.0], np.float32)
        self._dir_left = np.array([-1.0, 0.0, 0.0], np.float32)

        def _unit(v):
            vv = np.asarray(v, np.float32)
            nn = np.linalg.norm(vv)
            return (vv / nn) if nn > 1e-9 else vv

        # Diagonals
        self._dir_up_right = _unit(self._dir_up + self._dir_right)
        self._dir_down_right = _unit(self._dir_down + self._dir_right)
        self._dir_down_left = _unit(self._dir_down + self._dir_left)
        self._dir_up_left = _unit(self._dir_up + self._dir_left)

        # Sequence state
        self._base_center = None
        self._active_center = None
        self._phase = 0
        self._num_phases = 8
        self._in_release = True
        self._release_left = self.release_frames
        self._phase_frame = 0
        self._hold_frames_cur = self.hold_frames  # will be drawn for each phase
        self._dir_seq = None  # random order per point
        self._frame_force = None
        self._phase_weights = None
        self._doing_push = False
        self._push_left = 0
        self._cooldown_left = 0

    # ---------------- Region: surface->volume mask ----------------
    def _load_region(self):
        if self._region_loaded:
            return
        if self.region_surface_npz and os.path.exists(self.region_surface_npz):
            try:
                data = np.load(self.region_surface_npz)
                if 'indices' in data:
                    self._region_surface_indices = np.array(data['indices']).astype(np.int32)
            except Exception:
                pass
        self._region_loaded = True

    def _build_volume_mask(self):
        if self._volume_region_mask is not None:
            return
        self._load_region()
        if self._region_surface_indices is None or self.surface_model is None or not hasattr(self.surface_model, 'position'):
            return
        try:
            surf_pos = np.array(self.surface_model.position.value, np.float32)
            Pvol = np.array(self.mo.position.value, np.float32)
            if surf_pos.size == 0 or Pvol.size == 0:
                return
            region_set = set(int(i) for i in self._region_surface_indices.tolist())
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(surf_pos)
                _, nearest = tree.query(Pvol, k=1)
            except Exception:
                d2 = np.sum((Pvol[:, None, :] - surf_pos[None, :, :])**2, axis=2)
                nearest = np.argmin(d2, axis=1)
            mask = np.array([int(i) in region_set for i in nearest], dtype=bool)
            self._volume_region_mask = mask
            self._region_volume_indices = np.nonzero(mask)[0].astype(np.int32)
        except Exception:
            pass

    # ---------------- Base point selection ------------------
    def _pick_new_base_center(self):
        self._build_volume_mask()
        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            self._base_center = None
            return
        # Prefer surface region indices for choosing centers on the surface
        surf_pos = None
        if self.surface_model is not None and hasattr(self.surface_model, 'position'):
            try:
                surf_pos = np.array(self.surface_model.position.value, np.float32)
            except Exception:
                surf_pos = None
        if self.restrict and self._region_surface_indices is not None and surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.choice(self._region_surface_indices))
            self._base_center = surf_pos[sidx]
        elif surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.integers(0, surf_pos.shape[0]))
            self._base_center = surf_pos[sidx]
        else:
            vidx = int(self.rng.integers(0, P.shape[0]))
            self._base_center = P[vidx]
        self._active_center = self._base_center.copy()
        self._phase = 0
        self._in_release = True
        self._release_left = self.release_frames
        self._phase_frame = 0
        self._phase_weights = None
        # random directions for this point
        dirs = [
            self._dir_up,
            self._dir_right,
            self._dir_left,
            self._dir_down,
            self._dir_up_right,
            self._dir_down_right,
            self._dir_down_left,
            self._dir_up_left,
        ]
        self.rng.shuffle(dirs)
        self._dir_seq = dirs
        self._num_phases = len(dirs)

    # ---------------- Current direction --------------------------
    def _current_slide_dir(self):
        if self._dir_seq is not None and 0 <= self._phase < len(self._dir_seq):
            return self._dir_seq[self._phase]
        # Fallback (should not happen)
        mapping = [
            self._dir_up, self._dir_right, self._dir_left, self._dir_down,
            self._dir_up_right, self._dir_down_right, self._dir_down_left, self._dir_up_left
        ]
        return mapping[min(self._phase, 7)]

    # ---------------- Force generation (slide) -----------------
    def _apply_slide(self):
        if self._base_center is None:
            return
        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            return

        # The center remains fixed at the base point during the entire phase
        self._active_center = self._base_center

        # Calculate weights only once per phase
        if self._phase_weights is None:
            d = np.linalg.norm(P - self._active_center[None, :], axis=1)
            sigma = max(1e-3, 0.5 * self.radius)
            w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
            if self.restrict and self._volume_region_mask is not None:
                w *= self._volume_region_mask.astype(np.float32)
            self._phase_weights = w
        
        w = self._phase_weights

        # temporal scale (ramp-in -> hold -> ramp-out) with random hold per phase
        if self._phase_frame < self.ramp_in:
            scale = (self._phase_frame + 1) / max(1, self.ramp_in)
        elif self._phase_frame < self.ramp_in + self._hold_frames_cur:
            scale = 1.0
        else:
            down = self._phase_frame - (self.ramp_in + self._hold_frames_cur)
            scale = max(0.0, 1.0 - (down + 1) / max(1, self.ramp_out))

        # force direction = lateral + small indentation bias
        dir_vec = (1.0 - self.inward_bias) * self._current_slide_dir() + self.inward_bias * self.inward_dir
        nrm = np.linalg.norm(dir_vec)
        if nrm > 1e-9:
            dir_vec = dir_vec / nrm

        F = (w[:, None] * (self.slide_force * scale) * dir_vec[None, :]).astype(np.float32)
        self._frame_force = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[DEBUG] Deformer {self.name}: Phase={self._phase}, "
              f"PhaseFrame={self._phase_frame}, Scale={scale:.4f}, "
              f"Direction={np.round(dir_vec, 2)}, ForceNorm={np.linalg.norm(self._frame_force):.4f}")

    # ---------------- Force generation (push) ------------------
    def _apply_push(self):
        # local push around the same base point (or choose one)
        if self._base_center is None:
            self._pick_new_base_center()
            if self._base_center is None:
                self._frame_force = None
                return

        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            self._frame_force = None
            return

        d = np.linalg.norm(P - self._base_center[None, :], axis=1)
        sigma = max(1e-3, 0.5 * self.push_radius)
        w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        if self.restrict and self._volume_region_mask is not None:
            w *= self._volume_region_mask.astype(np.float32)

        F = (w[:, None] * (self.push_force * self.inward_dir[None, :])).astype(np.float32)
        self._frame_force = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- Utilities ------------------
    def _write_external_force(self):
        if not self.apply_to_mo:
            return
        try:
            if self._frame_force is not None:
                self.mo.externalForce.value = self._frame_force.tolist()
            else:
                N = len(self.mo.position.value)
                self.mo.externalForce.value = [[0.0, 0.0, 0.0]] * N
        except Exception:
            pass

    def _clear(self):
        self._frame_force = None
        self._write_external_force()

    # ---------------- Per-frame loop ----------------
    def onAnimateBeginEvent(self, *_):
        # push in progress?
        if self._doing_push:
            self._apply_push()
            self._write_external_force()
            self._push_left -= 1
            if self._push_left <= 0:
                self._doing_push = False
                self._clear()
            return

        # cooldown between points
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._clear()
            return

        # choose base point if necessary
        if self._base_center is None:
            self._pick_new_base_center()
            if self._base_center is None:
                self._clear()
                return

        # release period (zero force between directions)
        if self._in_release:
            self._release_left -= 1
            self._clear()
            if self._release_left <= 0:
                self._in_release = False
                self._phase_frame = 0
                self._phase_weights = None  # Réinit poids pour la nouvelle phase
                # Draw the hold for this phase
                self._hold_frames_cur = int(self.rng.integers(self.hold_min, self.hold_max + 1))
            return

        # apply a slide step
        self._apply_slide()
        self._write_external_force()

        # advance the phase time
        self._phase_frame += 1
        total = self.ramp_in + self._hold_frames_cur + self.ramp_out
        if self._phase_frame >= total:
            # end of direction -> release
            self._in_release = True
            self._release_left = self.release_frames
            self._phase_frame = 0

            # end of sequence?
            if self._phase >= self._num_phases - 1:
                # potentially a push
                if self.push_probability > 0.0 and self.rng.random() < self.push_probability:
                    self._doing_push = True
                    self._push_left = self.push_frames
                    return
                # otherwise, new point
                self._cooldown_left = self.cooldown_between_points
                self._base_center = None
                self._active_center = None
                self._dir_seq = None
                self._phase = 0
            else:
                # next direction
                self._phase += 1

    def onEndAnimation(self, *_):
        self._clear()


class DeepPressPusher(Sofa.Core.Controller):
    """
    Strongly pushes the brain inward from a point.
    - Chooses a center (preference for craniotomy region if provided)
    - Calculates an indentation direction following the local surface normal
      (PCA on surface neighbors), aligned with `inward_dir`.
    - Applies a Gaussian force over `radius` with ramp-in / hold / ramp-out.

    Writes force per frame in `self._frame_force` and, if `apply_to_mo=True`,
    pushes it into `MechanicalObject.externalForce`.
    """

    def __init__(self, mo, surface_model=None, name="deepPress",
                 seed=2025,
                 region_surface_npz=None,
                 restrict_to_region=True,
                 force_value=4.5e4,
                 radius=9.0,
                 ramp_in=8, hold_frames=16, ramp_out=8,
                 cooldown_frames=10,
                 inward_dir=(0.0, 0.0, -1.0),
                 use_surface_normal=True,
                 normal_neighbors=24,
                 apply_to_mo=True,
                 **kw):
        super().__init__(name=name, **kw)
        self.mo = mo
        self.surface_model = surface_model
        self.rng = np.random.default_rng(int(seed))
        self.region_surface_npz = region_surface_npz
        self.restrict = bool(restrict_to_region)

        self.force_value = float(force_value)
        self.radius = float(radius)
        self.ramp_in = int(max(1, ramp_in))
        self.hold_frames = int(max(0, hold_frames))
        self.ramp_out = int(max(1, ramp_out))
        self.cooldown_frames = int(max(0, cooldown_frames))

        self.inward_dir = np.asarray(inward_dir, np.float32)
        n = np.linalg.norm(self.inward_dir)
        if n > 1e-9:
            self.inward_dir /= n

        self.use_surface_normal = bool(use_surface_normal)
        self.normal_neighbors = int(max(3, normal_neighbors))
        self.apply_to_mo = bool(apply_to_mo)

        # Region
        self._region_loaded = False
        self._region_surface_indices = None
        self._volume_region_mask = None
        self._region_volume_indices = None

        # State
        self._center = None
        self._dir = self.inward_dir.copy()
        self._phase_frame = 0
        self._left = 0
        self._cooldown = 0
        self._frame_force = None
        self._phase_weights = None

    # --- Region: surface -> volume mask ---
    def _load_region(self):
        if self._region_loaded:
            return
        if self.region_surface_npz and os.path.exists(self.region_surface_npz):
            try:
                data = np.load(self.region_surface_npz)
                if 'indices' in data:
                    self._region_surface_indices = np.array(data['indices']).astype(np.int32)
            except Exception:
                pass
        self._region_loaded = True

    def _build_volume_mask(self):
        if self._volume_region_mask is not None:
            return
        self._load_region()
        if self._region_surface_indices is None or self.surface_model is None or not hasattr(self.surface_model, 'position'):
            return
        try:
            surf_pos = np.array(self.surface_model.position.value, np.float32)
            Pvol = np.array(self.mo.position.value, np.float32)
            if surf_pos.size == 0 or Pvol.size == 0:
                return
            region_set = set(int(i) for i in self._region_surface_indices.tolist())
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(surf_pos)
                _, nearest = tree.query(Pvol, k=1)
            except Exception:
                d2 = np.sum((Pvol[:, None, :] - surf_pos[None, :, :])**2, axis=2)
                nearest = np.argmin(d2, axis=1)
            mask = np.array([int(i) in region_set for i in nearest], dtype=bool)
            self._volume_region_mask = mask
            self._region_volume_indices = np.nonzero(mask)[0].astype(np.int32)
        except Exception:
            pass

    # --- Point & direction selection ---
    def _pick_center_and_dir(self):
        self._build_volume_mask()
        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            self._center = None
            self._dir = self.inward_dir.copy()
            return

        # Center choice: prefer surface (region if available)
        surf_pos = None
        if self.surface_model is not None and hasattr(self.surface_model, 'position'):
            try:
                surf_pos = np.array(self.surface_model.position.value, np.float32)
            except Exception:
                surf_pos = None

        if self.restrict and self._region_surface_indices is not None and surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.choice(self._region_surface_indices))
            base = surf_pos[sidx]
        elif surf_pos is not None and surf_pos.size > 0:
            sidx = int(self.rng.integers(0, surf_pos.shape[0]))
            base = surf_pos[sidx]
        else:
            vidx = int(self.rng.integers(0, P.shape[0]))
            base = P[vidx]

        jitter = self.rng.normal(0.0, 0.8, size=3).astype(np.float32)
        jitter[2] *= 0.2
        self._center = base + jitter

        # Direction: local surface normal (PCA on k nearest)
        dir_vec = self.inward_dir.copy()
        if self.use_surface_normal and surf_pos is not None and surf_pos.size > 0:
            try:
                try:
                    from scipy.spatial import cKDTree  # type: ignore
                    tree = cKDTree(surf_pos)
                    dists, idxs = tree.query(self._center, k=min(self.normal_neighbors, surf_pos.shape[0]))
                    neigh = surf_pos[idxs if np.ndim(idxs) else [idxs]]
                except Exception:
                    # Fallback: brute-force KNN
                    d2 = np.sum((surf_pos - self._center[None, :])**2, axis=1)
                    idxs = np.argsort(d2)[:min(self.normal_neighbors, surf_pos.shape[0])]
                    neigh = surf_pos[idxs]
                if neigh.shape[0] >= 3:
                    C = np.cov((neigh - neigh.mean(axis=0)).T)
                    evals, evecs = np.linalg.eigh(C)
                    nrm = evecs[:, np.argmin(evals)].astype(np.float32)
                    # Align inward
                    if np.dot(nrm, self.inward_dir) < 0.0:
                        nrm = -nrm
                    dn = np.linalg.norm(nrm)
                    if dn > 1e-9:
                        dir_vec = nrm / dn
            except Exception:
                pass
        self._dir = dir_vec
        self._phase_frame = 0
        self._left = self.ramp_in + self.hold_frames + self.ramp_out
        self._phase_weights = None

    # --- Per-frame calculation ---
    def _apply(self):
        if self._center is None:
            self._frame_force = None
            return
        P = np.array(self.mo.position.value, np.float32)
        if P.size == 0:
            self._frame_force = None
            return

        # Gaussian weights (cache per "press")
        if self._phase_weights is None:
            d = np.linalg.norm(P - self._center[None, :], axis=1)
            sigma = max(1e-3, 0.5 * self.radius)
            w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
            if self.restrict and self._volume_region_mask is not None:
                w *= self._volume_region_mask.astype(np.float32)
            self._phase_weights = w
        w = self._phase_weights

        # Ramp/hold
        if self._phase_frame < self.ramp_in:
            scale = (self._phase_frame + 1) / max(1, self.ramp_in)
        elif self._phase_frame < self.ramp_in + self.hold_frames:
            scale = 1.0
        else:
            down = self._phase_frame - (self.ramp_in + self.hold_frames)
            scale = max(0.0, 1.0 - (down + 1) / max(1, self.ramp_out))

        F = (w[:, None] * (self.force_value * scale) * self._dir[None, :]).astype(np.float32)
        self._frame_force = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _write_external_force(self):
        if not self.apply_to_mo:
            return
        try:
            if self._frame_force is not None:
                self.mo.externalForce.value = self._frame_force.tolist()
            else:
                N = len(self.mo.position.value)
                self.mo.externalForce.value = [[0.0, 0.0, 0.0]] * N
        except Exception:
            pass

    def _clear(self):
        self._frame_force = None
        self._write_external_force()

    # --- SOFA loop ---
    def onAnimateBeginEvent(self, *_):
        if self._left > 0:
            self._apply()
            self._write_external_force()
            self._left -= 1
            self._phase_frame += 1
            if self._left == 0:
                self._cooldown = self.cooldown_frames
                self._clear()
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            self._clear()
            return
        self._pick_center_and_dir()
        self._apply()

    def onEndAnimation(self, *_):
        self._clear()
