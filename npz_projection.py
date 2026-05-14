#!/usr/bin/env python3
# npz_projection.py — Projection directe sur NPZ avec sauvegarde incrémentale (version patchée)

import os
import json
import uuid
import numpy as np
from datetime import datetime
import glob
import fnmatch

class NPZDirectProjector:
    """Projecteur 3D→Pixel direct sur fichiers NPZ avec sauvegarde incrémentale"""

    def __init__(self, camera_params_file=None, backup_interval=50):
        self.session_id = str(uuid.uuid4())[:8]
        self.backup_interval = int(max(1, backup_interval))

        # Chargement paramètres caméra SOFA réels
        self.camera_params = self._load_real_camera_params(camera_params_file)

        # Dimensions initiales (par défaut) depuis la caméra.
        # Elles seront recalées par run dans process_npz_file().
        self.screen_width = int(self.camera_params['viewport_width'])
        self.screen_height = int(self.camera_params['viewport_height'])
        self.resolution_source = 'camera'

        print(f"NPZDirectProjector {self.session_id} - PROJECTION DIRECTE NPZ")
        print(f"   Sauvegarde incrémentale: toutes les {self.backup_interval} frames sur le même fichier")
        print(f"NPZDirectProjector {self.session_id} - PROJECTION DIRECTE NPZ")
        print(f"   Camera SOFA InteractiveCamera - MATRICES PRE-CALCULEES")
        p = self.camera_params['position']; q = self.camera_params['orientation']
        print(f"   Position: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]")
        print(f"   Quaternion: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
        print(f"   Viewport REEL: {self.screen_width}x{self.screen_height}")
        print(f"   FOV: {self.camera_params['field_of_view']:.1f}°")
        print(f"   Sauvegarde incrémentale: toutes les {self.backup_interval} frames sur le même fichier")

        # Dossier de sortie
        self.output_dir = "projected_npz"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"   Dossier sortie: {self.output_dir}/")

    # -------------------- Chargement & détection --------------------

    def _detect_dataset_image_resolution(self, run_dir=None):
        """Détecte la résolution des images exportées (priorité au dossier du run)."""
        try:
            img_dir = None
            if run_dir is not None:
                cand = os.path.join(run_dir, 'images')
                if os.path.isdir(cand):
                    img_dir = cand
            if img_dir is None:
                img_dir = os.path.join('simulation_output', 'images')
            candidates = []
            f0_png = os.path.join(img_dir, 'frame_0000.png')
            f0_jpg = os.path.join(img_dir, 'frame_0000.jpg')
            if os.path.exists(f0_png):
                candidates.append(f0_png)
            elif os.path.exists(f0_jpg):
                candidates.append(f0_jpg)
            else:
                if os.path.isdir(img_dir):
                    for fn in os.listdir(img_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            candidates.append(os.path.join(img_dir, fn))
                            break
            if not candidates:
                return None
            try:
                from PIL import Image
                with Image.open(candidates[0]) as im:
                    w, h = im.size
                    print(f"   Résolution images détectée ({os.path.relpath(img_dir)}): {w}x{h}")
                    return (w, h)
            except Exception:
                return None
        except Exception:
            return None

    def _load_real_camera_params(self, camera_params_file):
        """Charge les paramètres RÉELS de la caméra SOFA InteractiveCamera"""
        print("Chargement paramètres caméra SOFA RÉELS...")

        if camera_params_file is None:
            camera_params_file = self._find_latest_camera_params()

        if camera_params_file and os.path.exists(camera_params_file):
            try:
                with open(camera_params_file, 'r') as f:
                    data = json.load(f)

                print(f"   Fichier chargé: {os.path.basename(camera_params_file)}")

                # Extraction des paramètres essentiels (supporte ancien et nouveau format)
                params = data.get("essential_params", data)

                cam = {
                    'position': np.array(params['position'], dtype=np.float64),
                    'orientation': np.array(params['orientation'], dtype=np.float64),
                    'lookat': np.array(params.get('lookAt', params.get('lookat', [0, 0, 0])), dtype=np.float64),
                    'field_of_view': float(params.get('fieldOfView', params.get('field_of_view_degrees', 45.0))),
                    'viewport_width': int(params.get('widthViewport', params.get('viewport', {}).get('width', 1920))),
                    'viewport_height': int(params.get('heightViewport', params.get('viewport', {}).get('height', 1080))),
                    'intrinsic_matrix': None,
                    'focal_length': None,
                    'principal_point': None,
                    'znear': float(params.get('zNear', params.get('znear', 0.1))),
                    'zfar': float(params.get('zFar', params.get('zfar', 1000.0))),
                    'distance': float(params.get('distance', 10.0)),
                    'projection_matrix': None,
                    'modelview_matrix': None
                }

                if 'intrinsic_matrix' in params:
                    cam['intrinsic_matrix'] = np.array(params['intrinsic_matrix'], dtype=np.float64)
                elif 'intrinsics' in params and 'intrinsic_matrix' in params['intrinsics']:
                    cam['intrinsic_matrix'] = np.array(params['intrinsics']['intrinsic_matrix'], dtype=np.float64)

                if 'fx' in params:
                    cam['focal_length'] = {'fx': float(params.get('fx')), 'fy': float(params.get('fy'))}
                elif 'intrinsics' in params and 'focal_length' in params['intrinsics']:
                    cam['focal_length'] = params['intrinsics']['focal_length']

                if 'cx' in params:
                    cam['principal_point'] = {'cx': float(params.get('cx')), 'cy': float(params.get('cy'))}
                elif 'intrinsics' in params and 'principal_point' in params['intrinsics']:
                    cam['principal_point'] = params['intrinsics']['principal_point']

                # Matrices SOFA si présentes
                if "all_camera_attributes" in data:
                    attrs = data["all_camera_attributes"]
                    if 'projectionMatrix' in attrs:
                        cam['projection_matrix'] = np.array(attrs['projectionMatrix'], dtype=np.float64).reshape(4, 4)
                    if 'modelViewMatrix' in attrs:
                        cam['modelview_matrix'] = np.array(attrs['modelViewMatrix'], dtype=np.float64).reshape(4, 4)
                else:
                    if 'projectionMatrix' in params:
                        cam['projection_matrix'] = np.array(params['projectionMatrix'], dtype=np.float64).reshape(4, 4)
                    if 'modelViewMatrix' in params:
                        cam['modelview_matrix'] = np.array(params['modelViewMatrix'], dtype=np.float64).reshape(4, 4)

                pos, ori = cam['position'], cam['orientation']
                print(f"   Position extraite: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"   Quaternion extrait: [{ori[0]:.6f}, {ori[1]:.6f}, {ori[2]:.6f}, {ori[3]:.6f}]")
                print(f"   FOV: {cam['field_of_view']:.1f}°")
                print(f"   Viewport REEL: {cam['viewport_width']}x{cam['viewport_height']}")
                if cam['projection_matrix'] is not None:
                    print("   Matrice projection SOFA: disponible (4x4) - UTILISÉE DIRECTEMENT")
                if cam['modelview_matrix'] is not None:
                    print("   Matrice ModelView SOFA: disponible (4x4) - UTILISÉE DIRECTEMENT")

                return cam

            except Exception as e:
                print(f"Erreur lecture {camera_params_file}: {e}")

        # Fallback par défaut
        print("   Utilisation paramètres par défaut SOFA InteractiveCamera")
        return {
            'position': np.array([-81.2953, -21.7954, 112.397], dtype=np.float64),
            'orientation': np.array([0.0778814, -0.314528, 0.0648681, 0.943821], dtype=np.float64),
            'lookat': np.array([0, 0, 0], dtype=np.float64),
            'field_of_view': 45.0,
            'viewport_width': 1920,
            'viewport_height': 1080,
            'intrinsic_matrix': None,
            'focal_length': None,
            'principal_point': None,
            'znear': 0.1,
            'zfar': 1000.0,
            'distance': 10.0,
            'projection_matrix': None,
            'modelview_matrix': None
        }

    def _find_latest_camera_params(self, base_dir="simulation_output"):
        """Trouve le fichier de paramètres caméra le plus récent (recherche récursive)."""
        if base_dir is None or not os.path.exists(base_dir):
            return None
        camera_files = []
        for root, _dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.startswith('camera_params_') and filename.endswith('.json'):
                    camera_files.append(os.path.join(root, filename))
        if not camera_files:
            return None
        latest_file = max(camera_files, key=lambda f: os.path.getmtime(f))
        print(f"   Fichier auto-détecté: {os.path.basename(latest_file)}")
        return latest_file

    def _prepare_run_context(self, run_dir):
        """Ajuste la caméra et le viewport pour le dossier de run donné."""
        cam_file = self._find_latest_camera_params(run_dir) or self._find_latest_camera_params("simulation_output")
        if cam_file is not None:
            self.camera_params = self._load_real_camera_params(cam_file)
        # Réinitialise le viewport depuis la caméra
        self.screen_width = int(self.camera_params['viewport_width'])
        self.screen_height = int(self.camera_params['viewport_height'])
        self.resolution_source = 'camera'
        # Override éventuel depuis les images du run
        img_res = self._detect_dataset_image_resolution(run_dir)
        if img_res is not None:
            iw, ih = img_res
            if iw > 0 and ih > 0 and (iw != self.screen_width or ih != self.screen_height):
                self.screen_width, self.screen_height = int(iw), int(ih)
                self.resolution_source = 'image'
        # Log
        p = self.camera_params['position']; q = self.camera_params['orientation']
        print(f"   Contexte run: {os.path.relpath(run_dir)}")
        print(f"     Viewport: {self.screen_width}x{self.screen_height} (source: {self.resolution_source})")
    # -------------------- Math caméra --------------------

    def _quaternion_to_rotation_matrix(self, q):
        """Convertit quaternion en matrice de rotation (normalisé)"""
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float64)

    def _create_view_matrix_from_lookat(self, eye, target, up=None):
        """Crée une matrice view avec lookAt"""
        if up is None:
            up = np.array([0, 0, 1], dtype=np.float64)
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - eye
        n = np.linalg.norm(forward)
        if n == 0:
            forward = np.array([0, 0, -1], dtype=np.float64)
        else:
            forward = forward / n

        right = np.cross(forward, up)
        right = right / max(1e-12, np.linalg.norm(right))
        upc = np.cross(right, forward)

        R = np.array([
            [right[0], right[1], right[2]],
            [upc[0],   upc[1],   upc[2]],
            [-forward[0], -forward[1], -forward[2]]
        ], dtype=np.float64)
        t = -R @ eye

        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = R
        view[:3, 3] = t
        return view

    def _create_view_matrix_from_quaternion(self, position, orientation):
        """Crée matrice View depuis quaternion"""
        R = self._quaternion_to_rotation_matrix(orientation)
        forward = -R[:, 2]
        up = R[:, 1]
        right = R[:, 0]

        view = np.array([
            [right[0], right[1], right[2], -np.dot(right, position)],
            [up[0],    up[1],    up[2],    -np.dot(up, position)],
            [-forward[0], -forward[1], -forward[2],  np.dot(forward, position)],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        return view

    def _intrinsic_to_projection_matrix(self, K, near, far):
        """Convertit matrice intrinsèque en matrice de projection OpenGL"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        width, height = self.screen_width, self.screen_height

        proj = np.array([
            [2*fx/width, 0,            (width - 2*cx)/width, 0],
            [0,          2*fy/height,  (2*cy - height)/height, 0],
            [0,          0,            -(far+near)/(far-near), -2*far*near/(far-near)],
            [0,          0,            -1,                     0]
        ], dtype=np.float64)
        return proj

    def _create_projection_matrices(self):
        """Construit view/proj — utilise SOFA si dispo, sinon fallback."""
        pos = self.camera_params['position']
        fov = self.camera_params['field_of_view']
        near = self.camera_params['znear']
        far = self.camera_params['zfar']

        if (self.camera_params['projection_matrix'] is not None and
            self.camera_params['modelview_matrix'] is not None):
            proj_matrix = self.camera_params['projection_matrix'].copy()
            view_matrix = self.camera_params['modelview_matrix'].copy()
            print("   Utilisation MATRICES SOFA PRÉ-CALCULÉES (projection + modelview).")
        else:
            print("   Fallback: Calcul matrices manuellement...")
            if self.camera_params['intrinsic_matrix'] is not None:
                K = self.camera_params['intrinsic_matrix'].copy()
                ow, oh = self.camera_params['viewport_width'], self.camera_params['viewport_height']
                sx, sy = self.screen_width / ow, self.screen_height / oh
                K[0, 0] *= sx; K[1, 1] *= sy
                K[0, 2] *= sx; K[1, 2] *= sy
                proj_matrix = self._intrinsic_to_projection_matrix(K, near, far)
                print(f"      Intrinsèque mise à l’échelle: {ow}x{oh} → {self.screen_width}x{self.screen_height}")
            else:
                aspect = self.screen_width / self.screen_height
                f = 1.0 / np.tan(np.radians(fov) / 2.0)
                proj_matrix = np.array([
                    [f/aspect, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
                    [0, 0, -1, 0]
                ], dtype=np.float64)
                print(f"      Projection depuis FOV: {fov:.1f}°")

            if self.camera_params.get('lookat') is not None:
                target = self.camera_params['lookat']
                view_matrix = self._create_view_matrix_from_lookat(pos, target)
                print(f"      View (lookAt): eye={pos} → target={target}")
            else:
                ori = self.camera_params['orientation']
                view_matrix = self._create_view_matrix_from_quaternion(pos, ori)
                print("      View (quaternion)")

        return view_matrix, proj_matrix

    # -------------------- Projection vectorisée --------------------

    def project_vertices_to_pixels(self, vertices_3d, view_matrix=None, proj_matrix=None):
        """Projection 3D→Pixel (vectorisée)"""
        if view_matrix is None or proj_matrix is None:
            view_matrix, proj_matrix = self._create_projection_matrices()

        V = np.asarray(vertices_3d, dtype=np.float64)
        N = V.shape[0]
        Vh = np.concatenate([V, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N,4]

        Vview = Vh @ view_matrix.T
        Vclip = Vview @ proj_matrix.T

        w = Vclip[:, 3:4]
        safe = np.abs(w) > 1e-12

        ndc = np.full((N, 3), np.nan, dtype=np.float64)
        ndc[safe[:, 0]] = Vclip[safe[:, 0], :3] / w[safe[:, 0]]

        x = (ndc[:, 0] + 1.0) * 0.5 * self.screen_width
        y = (1.0 - ndc[:, 1]) * 0.5 * self.screen_height
        z = ndc[:, 2]

        # Bornes strictes (évite le "pixel de bord" fantôme)
        in_screen = (x >= 0.0) & (x < self.screen_width) & (y >= 0.0) & (y < self.screen_height)
        in_front = Vview[:, 2] < -0.1  # caméra OpenGL regardant -Z

        pixel_coords = np.column_stack([
            np.nan_to_num(x, nan=0.0), np.nan_to_num(y, nan=0.0)
        ]).astype(np.float32)
        depths = np.nan_to_num(z, nan=1.0).astype(np.float32)
        visibility_mask = (safe[:, 0] & in_screen & in_front)

        return pixel_coords, visibility_mask.astype(bool), depths

    # -------------------- I/O NPZ --------------------

    def find_npz_files(self, search_pattern="brain_surface_*.npz"):
        """Trouve tous les fichiers NPZ (parcours récursif de simulation_output, ignore déjà projetés)."""
        export_root = "simulation_output"
        if not os.path.exists(export_root):
            print(f"Dossier {export_root} introuvable")
            return []
        files = []
        for root, _dirs, filenames in os.walk(export_root):
            for fn in filenames:
                if fnmatch.fnmatch(fn, search_pattern) and ("_projected_" not in fn):
                    files.append(os.path.join(root, fn))
        if not files:
            print(f"Aucun fichier NPZ trouvé avec pattern: {search_pattern}")
            return []
        files.sort()
        print("Fichiers NPZ détectés:")
        for i, npz_file in enumerate(files):
            size_mb = os.path.getsize(npz_file) / 1024 / 1024
            relp = os.path.relpath(npz_file, start=export_root)
            print(f"   {i+1}. {relp} ({size_mb:.1f} MB)")
        return files

    def _save_incremental_backup(self, output_file, data_dict, frame_idx):
        """Sauvegarde incrémentale sur le même fichier"""
        np.savez_compressed(output_file, **data_dict)
        print(f"      Sauvegarde incrémentale frame {frame_idx}: {os.path.basename(output_file)}")
        return output_file

    # -------------------- Pixel-space rasterization --------------------

    def _rasterize_frame(self, pixel_coords, visibility_mask, depth_values, forces_3d, out_H, out_W):
        """Z-buffer rasterize one frame's visible vertices onto an (out_H, out_W) pixel grid.

        For each pixel the nearest vertex (smallest NDC depth) wins.
        Pixels with no vertex remain zero.

        Returns:
            force_map : (out_H, out_W, 3) float32  — 3-D force vector per pixel
            covered   : (out_H, out_W) bool         — True where a vertex was rasterized
        """
        force_map = np.zeros((out_H, out_W, 3), dtype=np.float32)
        covered   = np.zeros((out_H, out_W), dtype=bool)

        if forces_3d is None or not np.any(visibility_mask):
            return force_map, covered

        scale_x = out_W / self.screen_width
        scale_y = out_H / self.screen_height

        vis = visibility_mask.astype(bool)
        px = pixel_coords[vis, 0] * scale_x
        py = pixel_coords[vis, 1] * scale_y
        d  = depth_values[vis]               # NDC depth: smaller = closer to camera
        F  = forces_3d[vis].astype(np.float32)

        pxi = np.floor(px).astype(np.int32)
        pyi = np.floor(py).astype(np.int32)

        in_bounds = (pxi >= 0) & (pxi < out_W) & (pyi >= 0) & (pyi < out_H)
        pxi, pyi, d, F = pxi[in_bounds], pyi[in_bounds], d[in_bounds], F[in_bounds]

        if len(pxi) == 0:
            return force_map, covered

        lin   = pyi * out_W + pxi             # flat pixel index
        order = np.argsort(d)                 # ascending: nearest vertex first
        lin_s, F_s = lin[order], F[order]

        # np.unique returns first occurrence per unique value → nearest vertex per pixel
        _, first = np.unique(lin_s, return_index=True)

        flat_f = force_map.reshape(-1, 3)
        flat_c = covered.reshape(-1)
        flat_f[lin_s[first]] = F_s[first]
        flat_c[lin_s[first]] = True

        return force_map, covered

    def _fill_holes(self, force_map, covered, max_dist=10):
        """Nearest-neighbor hole filling inside the object silhouette.

        Each uncovered pixel within max_dist pixels of a covered pixel is
        assigned the force of its nearest covered neighbour.  Pixels farther
        than max_dist from any surface vertex are treated as background and
        remain zero.
        """
        if not np.any(covered) or np.all(covered):
            return force_map
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            print("[rasterize] scipy unavailable — hole filling skipped")
            return force_map

        dist, nearest = distance_transform_edt(
            ~covered, return_distances=True, return_indices=True
        )
        fill = ~covered & (dist <= max_dist)
        if not np.any(fill):
            return force_map
        fy, fx = nearest[0][fill], nearest[1][fill]
        force_map[fill] = force_map[fy, fx]
        return force_map

    # -------------------- Traitement principal --------------------

    def process_npz_file(self, npz_file):
        """Traite un fichier NPZ avec projection et sauvegarde incrémentale"""
        print(f"\nTRAITEMENT NPZ: {os.path.basename(npz_file)}")
        print("=" * 60)
        try:
            # Préparation du contexte par run (caméra + viewport + résolution images)
            run_dir = os.path.dirname(npz_file)
            self._prepare_run_context(run_dir)

            data = np.load(npz_file)
            print(f"   Fichier NPZ chargé: {os.path.basename(npz_file)}")

            required_keys = ['frames']
            missing = [k for k in required_keys if k not in data.files]
            if missing:
                print(f"   Clés manquantes: {missing}")
                return None

            frames = data['frames']
            rest_positions = data.get('rest', None)
            displacements = data.get('displacements', None)
            times = data.get('times', None)
            surface_external_forces = data.get('surface_external_forces', None)

            n_frames, n_vertices, _ = frames.shape

            # Rasterized pixel-force maps at half the camera viewport resolution
            out_H = self.screen_height // 2   # 540 for a 1080p camera
            out_W = self.screen_width  // 2   # 960 for a 1920p camera
            raster_force_map = np.zeros((n_frames, out_H, out_W, 3), dtype=np.float32)
            raster_force_mag = np.zeros((n_frames, out_H, out_W),    dtype=np.float32)
            print(f"   Pixel-force maps: {n_frames}×{out_H}×{out_W} "
                  f"({raster_force_map.nbytes / 1e9:.1f} GB pre-allocated)")

            # Detect image_every from meta.json in the same folder
            image_every = 1
            meta_candidates = [
                f.replace('.npz', '_meta.json') for f in [npz_file]
            ] + glob.glob(os.path.join(run_dir, '*_meta.json'))
            for mc in meta_candidates:
                if os.path.exists(mc):
                    try:
                        with open(mc) as mf:
                            meta_data = json.load(mf)
                        image_every = int(meta_data.get('image_every', 1))
                        if image_every > 1:
                            print(f"   image_every={image_every} lu depuis {os.path.basename(mc)}")
                        break
                    except Exception:
                        pass
            # Fallback: infer image_every from actual filename gaps
            if image_every == 1:
                img_dir = os.path.join(run_dir, 'images')
                if os.path.isdir(img_dir):
                    import re
                    nums = sorted([
                        int(m.group(1))
                        for fn in os.listdir(img_dir)
                        if (m := re.match(r'frame_(\d+)\.(jpg|jpeg|png)$', fn, re.IGNORECASE))
                        and int(m.group(1)) < n_frames   # only frames within this NPZ's range
                    ])
                    if len(nums) >= 2:
                        gaps = [nums[i+1] - nums[i] for i in range(min(10, len(nums)-1))]
                        image_every = int(round(sum(gaps) / len(gaps)))
                        print(f"   image_every={image_every} déduit depuis noms de fichiers ({len(nums)} images dans plage NPZ)")

            image_frame_indices = np.arange(0, n_frames, image_every, dtype=np.int32)

            print("   Données extraites:")
            print(f"     Frames: {n_frames}")
            print(f"     Vertices par frame: {n_vertices}")
            print(f"     Rest positions: {'Disponible' if rest_positions is not None else 'Non disponible'}")
            print(f"     Displacements: {'Disponible' if displacements is not None else 'Non disponible'}")
            print(f"     Times: {'Disponible' if times is not None else 'Non disponible'}")
            print(f"     Surface external forces: {'Disponible' if surface_external_forces is not None else 'Non disponible'}")
            print(f"     Images attendues: {len(image_frame_indices)} (every {image_every} frames)")

            # Buffers de sortie
            projected_pixels = np.zeros((n_frames, n_vertices, 2), dtype=np.float32)
            visibility_masks = np.zeros((n_frames, n_vertices), dtype=bool)
            depth_values = np.zeros((n_frames, n_vertices), dtype=np.float32)

            # Fichier de sortie (miroir du sous-dossier de run sous projected_npz)
            output_filename = os.path.basename(npz_file).replace('.npz', f'_projected_{self.session_id}.npz')
            export_root_abs = os.path.abspath("simulation_output")
            npz_dir_abs = os.path.abspath(run_dir)
            output_base = self.output_dir
            try:
                if os.path.commonpath([export_root_abs, npz_dir_abs]) == export_root_abs:
                    rel_subdir = os.path.relpath(npz_dir_abs, export_root_abs)
                    output_base = os.path.join(self.output_dir, rel_subdir)
            except Exception:
                pass
            os.makedirs(output_base, exist_ok=True)
            output_file = os.path.join(output_base, output_filename)

            print(f"   Pré-calcul matrices view/proj...")
            view_matrix, proj_matrix = self._create_projection_matrices()

            print(f"   Début projection de {n_frames} frames...")
            for frame_idx in range(n_frames):
                frame_vertices = frames[frame_idx]
                pixel_coords, visibility_mask, depths = self.project_vertices_to_pixels(
                    frame_vertices, view_matrix=view_matrix, proj_matrix=proj_matrix
                )

                projected_pixels[frame_idx] = pixel_coords
                visibility_masks[frame_idx] = visibility_mask
                depth_values[frame_idx] = depths

                # Rasterize forces onto the pixel grid for this frame
                forces_frame = (surface_external_forces[frame_idx]
                                if surface_external_forces is not None else None)
                fmap, cov = self._rasterize_frame(
                    pixel_coords, visibility_mask, depths, forces_frame, out_H, out_W
                )
                fmap = self._fill_holes(fmap, cov, max_dist=10)
                raster_force_map[frame_idx] = fmap
                raster_force_mag[frame_idx] = np.linalg.norm(fmap, axis=-1).astype(np.float32)

                if (frame_idx + 1) % 10 == 0:
                    vis_count  = int(np.sum(visibility_mask))
                    vis_rate   = (vis_count / len(visibility_mask) * 100.0) if len(visibility_mask) else 0.0
                    n_covered  = int(np.sum(cov))
                    total_pix  = out_H * out_W
                    print(f"      Frame {frame_idx+1}/{n_frames}: {vis_count}/{len(visibility_mask)} vertices visible ({vis_rate:.1f}%), "
                          f"{n_covered}/{total_pix} pixels covered ({n_covered/total_pix*100:.1f}%)")

                if (frame_idx + 1) % self.backup_interval == 0:
                    print(f"   Sauvegarde incrémentale à la frame {frame_idx+1}...")
                    current_data = {
                        'frames': frames[:frame_idx+1],
                        'projected_pixels': projected_pixels[:frame_idx+1],
                        'visibility_masks': visibility_masks[:frame_idx+1],
                        'depth_values': depth_values[:frame_idx+1]
                    }
                    if rest_positions is not None:
                        current_data['rest'] = rest_positions
                    if displacements is not None:
                        current_data['displacements'] = displacements[:frame_idx+1]
                    if times is not None:
                        current_data['times'] = times[:frame_idx+1]
                    if surface_external_forces is not None:
                        current_data['surface_external_forces'] = surface_external_forces[:frame_idx+1]
                    current_data['image_frame_indices'] = image_frame_indices[image_frame_indices <= frame_idx]
                    self._save_incremental_backup(output_file, current_data, frame_idx+1)

            print("   Sauvegarde finale...")
            final_data = {
                'frames': frames,
                'projected_pixels': projected_pixels,
                'visibility_masks': visibility_masks,
                'depth_values': depth_values,
                # Dense pixel-space force maps (H×W per frame, background = 0)
                'pixel_force_map':       raster_force_map,   # (T, H, W, 3) float32
                'pixel_force_magnitude': raster_force_mag,   # (T, H, W)    float32
            }
            if rest_positions is not None:
                final_data['rest'] = rest_positions
            if displacements is not None:
                final_data['displacements'] = displacements
            if times is not None:
                final_data['times'] = times
            if surface_external_forces is not None:
                final_data['surface_external_forces'] = surface_external_forces
            final_data['image_frame_indices'] = image_frame_indices  # frames that have a matching image file

            np.savez_compressed(output_file, **final_data)

            total_visible = int(np.sum(visibility_masks))
            total_vertices = int(n_frames * n_vertices)
            overall_visibility = (total_visible / total_vertices * 100.0) if total_vertices else 0.0
            size_mb = os.path.getsize(output_file) / 1024 / 1024

            print("   Projection terminée:")
            print(f"     Fichier final: {output_filename}")
            print(f"     Taille: {size_mb:.1f} MB")
            print(f"     Total vertices: {total_vertices:,}")
            print(f"     Vertices visibles: {total_visible:,} ({overall_visibility:.1f}%)")
            print(f"     Nouvelles clés NPZ: projected_pixels, visibility_masks, depth_values")

            metadata = {
                'session_id': self.session_id,
                'processing_date': datetime.now().isoformat(),
                'input_file': os.path.basename(npz_file),
                'output_file': output_filename,
                'camera_params': {
                    'position': self.camera_params['position'].tolist(),
                    'orientation': self.camera_params['orientation'].tolist(),
                    'lookat': self.camera_params['lookat'].tolist(),
                    'fov': float(self.camera_params['field_of_view']),
                    'viewport': [int(self.screen_width), int(self.screen_height)],
                    'resolution_source': self.resolution_source
                },
                'projection_stats': {
                    'total_frames': int(n_frames),
                    'vertices_per_frame': int(n_vertices),
                    'total_vertices': total_vertices,
                    'total_visible': total_visible,
                    'visibility_rate': float(overall_visibility),
                    'backup_interval': int(self.backup_interval),
                },
                'pixel_force_map_info': {
                    'shape': [int(n_frames), int(out_H), int(out_W)],
                    'resolution': f'{out_W}x{out_H}',
                    'hole_fill_max_dist_px': 10,
                    'background_value': 0.0,
                    'keys': ['pixel_force_map', 'pixel_force_magnitude'],
                    'note': (
                        'pixel_force_map (T,H,W,3): 3-D force vector per pixel. '
                        'pixel_force_magnitude (T,H,W): ||F|| per pixel. '
                        'Background and non-surface pixels are zero. '
                        'Holes within 10 px of a surface vertex are filled by nearest-neighbor.'
                    ),
                },
                'npz_keys': {
                    'original': list(data.files),
                    'added': ['projected_pixels', 'visibility_masks', 'depth_values',
                              'image_frame_indices', 'pixel_force_map', 'pixel_force_magnitude'],
                    'carried_over': [k for k in ['rest', 'displacements', 'times', 'surface_external_forces'] if k in data.files]
                },
                'data_units': {
                    'length': 'mm',
                    'time': 's',
                    'mass': 'kg',
                    'forces': 'N',
                    'note': 'surface_external_forces are stored in Newtons',
                    'force_to_newton': 1e-3
                }
            }
            meta_file = output_file.replace('.npz', '_metadata.json')
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   Métadonnées sauvées: {os.path.basename(meta_file)}")

            data.close()
            return {
                'input_file': npz_file,
                'output_file': output_file,
                'metadata_file': meta_file,
                'total_frames': n_frames,
                'vertices_per_frame': n_vertices,
                'total_visible': total_visible,
                'visibility_rate': overall_visibility,
                'output_size_mb': size_mb
            }

        except Exception as e:
            print(f"   Erreur traitement NPZ: {e}")
            return None

# -------------------- CLI --------------------

def main():
    """Menu principal pour projection directe NPZ"""
    print("NPZ DIRECT PROJECTOR")
    print("=" * 60)
    print("Projection 3D→Pixel directe sur fichiers NPZ avec sauvegarde incrémentale\n")

    # Configuration backup interval
    backup_interval = 50
    try:
        user_interval = input(f"Interval de sauvegarde (défaut {backup_interval} frames): ").strip()
        if user_interval:
            backup_interval = int(user_interval)
    except ValueError:
        print("Utilisation valeur par défaut")

    projector = NPZDirectProjector(backup_interval=backup_interval)

    print("Options disponibles:")
    print("  1. Traiter tous les fichiers NPZ automatiquement")
    print("  2. Traiter avec pattern personnalisé")
    print("  3. Traiter un seul fichier NPZ spécifique")
    print("  4. Quitter")

    try:
        choice = int(input("\nVotre choix (1-4): "))
        if choice == 1:
            npz_files = projector.find_npz_files()
            for npz_file in npz_files:
                projector.process_npz_file(npz_file)

        elif choice == 2:
            pattern = input("Pattern de recherche (ex: brain_surface_*.npz): ").strip()
            if pattern:
                npz_files = projector.find_npz_files(pattern)
                for npz_file in npz_files:
                    projector.process_npz_file(npz_file)
            else:
                print("Pattern vide")

        elif choice == 3:
            npz_files = projector.find_npz_files()
            if npz_files:
                print("\nFichiers NPZ disponibles:")
                for i, npz_file in enumerate(npz_files):
                    print(f"  {i+1}. {os.path.basename(npz_file)}")
                idx = int(input(f"\nChoisir fichier (1-{len(npz_files)}): ")) - 1
                if 0 <= idx < len(npz_files):
                    projector.process_npz_file(npz_files[idx])
                else:
                    print("Choix invalide")
            else:
                print("Aucun fichier NPZ trouvé")

        elif choice == 4:
            print("Au revoir!")

        else:
            print("Choix invalide")

    except ValueError:
        print("Choix invalide")
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur")

if __name__ == "__main__":
    main()
