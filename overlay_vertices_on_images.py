#!/usr/bin/env python3
# overlay_vertices_on_images.py - Superposition des sommets sur les images

import os
import json
import uuid
import numpy as np
from datetime import datetime
import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

class VertexImageOverlay:
    """Superposition des sommets sur les images originales"""
    
    def __init__(self, run_dir=None, camera_params_file=None):
        self.session_id = str(uuid.uuid4())[:8]
        
        # Auto-detect run_dir if not provided
        if run_dir is None:
            run_dir = self._find_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError("Aucun run_dir trouvé dans simulation_output/. "
                                    "Passez run_dir='simulation_output/run_E...' en argument.")
        self.run_dir = os.path.normpath(run_dir)
        run_subdir = os.path.basename(self.run_dir)  # e.g. run_E2.50_nu0.450_seed1111
        
        # Dossiers
        self.images_dir   = os.path.join(self.run_dir, "images")
        self.projected_dir = os.path.join("projected_npz", run_subdir)
        self.output_dir   = os.path.join("overlayed_frames", run_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache NPZ
        self._npz_data = None
        self._npz_path = None
        
        # Chargement paramètres caméra SOFA réels
        self.camera_params = self._load_real_camera_params(camera_params_file)

        # Dimensions: use actual image resolution from disk (projection was computed in that space)
        # Camera viewport may differ (e.g. 2560×1600) from saved images (e.g. 1920×1080)
        img_res = self._detect_image_resolution()
        if img_res is not None:
            self.screen_width, self.screen_height = img_res
            print(f"   Résolution détectée depuis images: {self.screen_width}x{self.screen_height}")
        else:
            self.screen_width  = self.camera_params['viewport_width']
            self.screen_height = self.camera_params['viewport_height']
            print(f"   Résolution fallback (camera params): {self.screen_width}x{self.screen_height}")
        
        print(f"VertexImageOverlay {self.session_id} - SUPERPOSITION VERTICES SUR IMAGES")
        print(f"   Run dir        : {self.run_dir}")
        print(f"   Position caméra: [{self.camera_params['position'][0]:.1f}, {self.camera_params['position'][1]:.1f}, {self.camera_params['position'][2]:.1f}]")
        print(f"   Viewport: {self.screen_width}x{self.screen_height}")
        print(f"   Images originales: {self.images_dir}/")
        print(f"   Données projetées: {self.projected_dir}/")
        print(f"   Sortie overlays  : {self.output_dir}/")
    
    def _load_real_camera_params(self, camera_params_file):
        """Charge les paramètres caméra (version simplifiée)"""
        print(f"Chargement paramètres caméra...")
        
        # Recherche automatique si pas spécifié
        if camera_params_file is None:
            camera_params_file = self._find_latest_camera_params()
        
        if camera_params_file and os.path.exists(camera_params_file):
            try:
                with open(camera_params_file, 'r') as f:
                    data = json.load(f)
                
                print(f"   Fichier chargé: {os.path.basename(camera_params_file)}")
                
                params = data.get("essential_params", data)
                
                camera_params = {
                    'position': np.array(params['position']),
                    'orientation': np.array(params['orientation']),
                    'lookat': np.array(params.get('lookAt', params.get('lookat', [0, 0, 0]))),
                    'field_of_view': params.get('fieldOfView', 45.0),
                    'viewport_width': params.get('widthViewport', 1920),
                    'viewport_height': params.get('heightViewport', 1080),
                    'znear': params.get('zNear', 0.1),
                    'zfar': params.get('zFar', 1000.0),
                    'projection_matrix': None,
                    'modelview_matrix': None
                }
                
                # Matrices SOFA si disponibles
                if "all_camera_attributes" in data:
                    attrs = data["all_camera_attributes"]
                    if 'projectionMatrix' in attrs:
                        camera_params['projection_matrix'] = np.array(attrs['projectionMatrix']).reshape(4, 4)
                    if 'modelViewMatrix' in attrs:
                        camera_params['modelview_matrix'] = np.array(attrs['modelViewMatrix']).reshape(4, 4)
                
                return camera_params
                
            except Exception as e:
                print(f"Erreur lecture paramètres: {e}")
        
        # Paramètres par défaut
        print(f"   Utilisation paramètres par défaut")
        return {
            'position': np.array([-86.338, -17.669, 126.000]),
            'orientation': np.array([0.049167, -0.296558, 0.051304, 0.952367]),
            'lookat': np.array([0, 0, 0]),
            'field_of_view': 45.0,
            'viewport_width': 1920,
            'viewport_height': 1080,
            'znear': 0.1,
            'zfar': 1000.0,
            'projection_matrix': None,
            'modelview_matrix': None
        }
    
    def _detect_image_resolution(self):
        """Détecte la résolution réelle des images sauvegardées (source de vérité pour pixel coords)."""
        if not os.path.exists(self.images_dir):
            return None
        for fname in sorted(os.listdir(self.images_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(self.images_dir, fname))
                    w, h = img.size
                    img.close()
                    return (w, h)
                except Exception:
                    continue
        return None

    def _find_latest_camera_params(self):
        """Trouve le fichier de paramètres caméra le plus récent dans run_dir ou simulation_output/"""
        # Search in run_dir first, then parent simulation_output/
        search_dirs = []
        if hasattr(self, 'run_dir') and self.run_dir:
            search_dirs.append(self.run_dir)
        search_dirs.append("simulation_output")

        camera_files = []
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for filename in os.listdir(search_dir):
                if filename.startswith('camera_params_') and filename.endswith('.json'):
                    camera_files.append(os.path.join(search_dir, filename))

        if camera_files:
            return max(camera_files, key=lambda f: os.path.getmtime(f))
        return None

    def _find_latest_run_dir(self):
        """Trouve le run_dir le plus récent dans simulation_output/"""
        sim_dir = "simulation_output"
        if not os.path.exists(sim_dir):
            return None
        run_dirs = [
            os.path.join(sim_dir, d)
            for d in os.listdir(sim_dir)
            if os.path.isdir(os.path.join(sim_dir, d)) and d.startswith("run_")
        ]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda d: os.path.getmtime(d))

    def _load_projected_npz(self):
        """Charge (et met en cache) le fichier NPZ projeté pour ce run"""
        if self._npz_data is not None:
            return self._npz_data
        if not os.path.exists(self.projected_dir):
            print(f"❌ Dossier projeté introuvable: {self.projected_dir}")
            return None
        npz_files = sorted(glob.glob(os.path.join(self.projected_dir, "*_projected*.npz")))
        if not npz_files:
            print(f"❌ Aucun fichier *_projected*.npz dans {self.projected_dir}")
            return None
        self._npz_path = npz_files[0]
        print(f"   Chargement NPZ: {os.path.basename(self._npz_path)} ...")
        self._npz_data = np.load(self._npz_path, allow_pickle=False)
        n_frames, n_verts, _ = self._npz_data['frames'].shape
        print(f"   NPZ chargé: {n_frames} frames × {n_verts} vertices")
        return self._npz_data
    
    def _make_frame_image_path(self, sim_frame_idx):
        """Retourne le chemin de l'image correspondant au frame de simulation donné."""
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join(self.images_dir, f"frame_{sim_frame_idx:04d}{ext}")
            if os.path.exists(p):
                return p
        return None

    def _compute_global_force_vmax(self):
        """Scans all frames in the NPZ and returns the global max force magnitude.
        Cached in self._global_force_vmax so it is only computed once."""
        if hasattr(self, '_global_force_vmax') and self._global_force_vmax is not None:
            return self._global_force_vmax
        npz = self._load_projected_npz()
        if npz is None or 'surface_external_forces' not in npz:
            self._global_force_vmax = None
            return None
        print("   Calcul vmax force global sur toutes les frames...")
        forces = npz['surface_external_forces']          # (N_frames, N_verts, 3)
        mags   = np.linalg.norm(forces, axis=2)          # (N_frames, N_verts)
        self._global_force_vmax = float(mags.max())
        print(f"   vmax force global: {self._global_force_vmax*1e3:.3f} mN")
        return self._global_force_vmax

    def _build_frame_data_entry(self, npz, sim_frame_idx):
        """
        Construit l'entrée frame_data pour un index de frame simulation.
        projected_data shape: (N, 4) → [pixel_x, pixel_y, is_visible, depth]
        force_magnitudes shape: (N,)
        """
        pixels  = npz['projected_pixels'][sim_frame_idx]          # (N, 2)
        visible = npz['visibility_masks'][sim_frame_idx].astype(np.float32)  # (N,)
        depths  = npz['depth_values'][sim_frame_idx]               # (N,)
        forces  = npz['surface_external_forces'][sim_frame_idx]    # (N, 3)
        force_mag = np.linalg.norm(forces, axis=1)                 # (N,)
        # Peak applied force = max per-vertex magnitude
        peak_force_N = float(force_mag.max())

        projected_data = np.column_stack([
            pixels[:, 0],   # pixel_x
            pixels[:, 1],   # pixel_y
            visible,        # is_visible
            depths          # depth
        ])

        image_file = self._make_frame_image_path(sim_frame_idx)

        return {
            'frame_index'     : sim_frame_idx,
            'image_file'      : image_file,
            'projected_data'  : projected_data,   # (N, 4)
            'force_magnitudes': force_mag,         # (N,)
            'peak_force_N'    : peak_force_N,      # max vertex force this frame (N)
        }

    def find_single_frame_data(self, frame_idx):
        """Trouve les données pour une seule frame spécifique (index simulation)."""

        print(f"\nRECHERCHE DONNÉES FRAME {frame_idx}:")

        npz = self._load_projected_npz()
        if npz is None:
            return None

        n_frames = npz['frames'].shape[0]
        if not (0 <= frame_idx < n_frames):
            print(f"❌ frame_idx {frame_idx} hors bornes (0–{n_frames-1})")
            return None

        entry = self._build_frame_data_entry(npz, frame_idx)

        if entry['image_file'] is None:
            print(f"⚠️  Image frame {frame_idx} introuvable (données projetées disponibles)")
        else:
            print(f"   Image trouvée: {os.path.basename(entry['image_file'])}")

        visible_count = int(np.sum(entry['projected_data'][:, 2] == 1))
        print(f"   Vertices: {len(entry['projected_data']):,} (visibles: {visible_count:,})")

        return entry

    def find_images_and_data(self, max_frames=200):
        """Trouve les images sur disque et leurs données projetées depuis le NPZ."""

        print(f"\nRECHERCHE IMAGES ET DONNÉES:")

        npz = self._load_projected_npz()
        if npz is None:
            return []

        n_frames = npz['frames'].shape[0]

        # --- Scan actual images on disk (ground truth) ---
        if not os.path.exists(self.images_dir):
            print(f"❌ Dossier images introuvable: {self.images_dir}")
            return []

        image_map = {}   # sim_frame_idx → image_path
        for fname in os.listdir(self.images_dir):
            root, ext = os.path.splitext(fname)
            if ext.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            # expect: frame_NNNN.ext
            parts = root.split('_')
            if len(parts) >= 2 and parts[0] == 'frame':
                try:
                    sim_idx = int(parts[1])
                    if 0 <= sim_idx < n_frames:
                        image_map[sim_idx] = os.path.join(self.images_dir, fname)
                except ValueError:
                    pass

        sorted_indices = sorted(image_map.keys())
        print(f"   Images sur disque (≤ n_frames): {len(sorted_indices)}")

        limit = min(max_frames, len(sorted_indices))
        frame_data = []

        for i, sim_idx in enumerate(sorted_indices[:limit]):
            entry = self._build_frame_data_entry(npz, sim_idx)
            entry['image_file'] = image_map[sim_idx]   # use confirmed on-disk path
            frame_data.append(entry)

            if i < 5:
                visible_count = int(np.sum(entry['projected_data'][:, 2] == 1))
                print(f"   Frame sim {sim_idx:04d}: {os.path.basename(image_map[sim_idx])}"
                      f" + {visible_count:,} vertices visibles")

        print(f"   Correspondances trouvées: {len(frame_data)} frames")
        return frame_data

    
    def create_single_overlay(self, frame_data, vertex_size=2, show_invisible=False,
                              color_by='force', force_vmax=None):
        """Crée un overlay pour une seule frame.

        color_by  : 'force' (défaut) → magnitude force en N
                    'depth'          → profondeur NDC
        force_vmax: échelle fixe pour le colormap force (en N).
                    Si None, utilise le max de la frame courante.
        """
        frame_idx     = frame_data['frame_index']
        image_file    = frame_data['image_file']
        projected_data = frame_data['projected_data']
        force_magnitudes = frame_data.get('force_magnitudes', None)

        if color_by == 'force' and force_magnitudes is None:
            color_by = 'depth'  # fallback

        print(f"DÉBUT création overlay frame {frame_idx:03d}: {len(projected_data)} vertices")

        # Chargement image originale (ou fond noir si image manquante)
        if image_file and os.path.exists(image_file):
            try:
                print(f"   Chargement image: {os.path.basename(image_file)}")
                original_image = Image.open(image_file)
                img_width, img_height = original_image.size
                # Always use actual image dimensions as the coordinate space
                # (projected_pixels were computed at this resolution)
                self.screen_width  = img_width
                self.screen_height = img_height
                print(f"   Dimensions image: {img_width}x{img_height}")
            except Exception as e:
                print(f"   Erreur chargement image: {e}")
                original_image = Image.new('RGB', (self.screen_width, self.screen_height), 'black')
        else:
            print(f"   ⚠️  Image introuvable — fond noir utilisé")
            original_image = Image.new('RGB', (self.screen_width, self.screen_height), 'black')

        print(f"   Création figure matplotlib...")
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(original_image, extent=[0, self.screen_width, self.screen_height, 0])

        # Séparation vertices visibles/invisibles
        visible_mask     = projected_data[:, 2] == 1
        visible_vertices = projected_data[visible_mask]
        invisible_vertices = projected_data[~visible_mask]

        visible_count   = len(visible_vertices)
        invisible_count = len(invisible_vertices)
        print(f"   Vertices: {visible_count:,} visibles, {invisible_count:,} invisibles")

        # Overlay vertices visibles
        if len(visible_vertices) > 0:
            pixel_x = visible_vertices[:, 0]
            pixel_y = visible_vertices[:, 1]

            if color_by == 'force' and force_magnitudes is not None:
                color_values   = force_magnitudes[visible_mask]
                cmap_name      = 'hot'
                colorbar_label = 'Force magnitude (N)'
                vmin_val = 0.0
                vmax_val = force_vmax if force_vmax is not None else float(color_values.max()) if len(color_values) else 1.0
            else:
                color_values   = visible_vertices[:, 3]   # depth_ndc
                cmap_name      = 'plasma'
                colorbar_label = 'Profondeur NDC'
                vmin_val, vmax_val = None, None

            scatter = ax.scatter(pixel_x, pixel_y, c=color_values, cmap=cmap_name,
                                 vmin=vmin_val, vmax=vmax_val,
                                 s=vertex_size**2, alpha=0.8, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(colorbar_label, rotation=270, labelpad=15)
            if vmax_val is not None:
                cbar.ax.axhline(y=vmax_val, color='cyan', linewidth=1, alpha=0.6)

        # Overlay vertices invisibles (optionnel)
        if show_invisible and len(invisible_vertices) > 0:
            ax.scatter(invisible_vertices[:, 0], invisible_vertices[:, 1],
                       c='red', s=vertex_size**2/4, alpha=0.3, marker='x',
                       label='Hors écran')

        # Configuration axes
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        color_label = 'force' if color_by == 'force' else 'profondeur'
        ax.set_title(f'Brain Vertices Overlay - Frame {frame_idx:04d}  '
                     f'(couleur={color_label})\n'
                     f'Visibles: {visible_count:,} | Invisibles: {invisible_count:,}',
                     fontsize=14, fontweight='bold')

        # --- Statistiques textuelles ---
        npz_times = self._npz_data['times'] if self._npz_data is not None and 'times' in self._npz_data else None
        time_s = float(npz_times[frame_idx]) if npz_times is not None else frame_idx * 0.03
        peak_force_N = frame_data.get('peak_force_N', None)

        info_lines = [
            f"Frame: {frame_idx:04d}",
            f"Time : {time_s:.2f} s",
            f"Visibles  : {visible_count:,}",
            f"Invisibles: {invisible_count:,}",
        ]

        if color_by == 'force' and force_magnitudes is not None:
            fv = force_magnitudes[visible_mask]
            if len(fv):
                f_peak_mN  = (peak_force_N or fv.max()) * 1e3
                f_mean_mN  = float(fv.mean()) * 1e3
                f_scale_mN = (force_vmax or fv.max()) * 1e3
                info_lines += [
                    "",
                    f"Force pic   : {f_peak_mN:.2f} mN",
                    f"Force moy   : {f_mean_mN:.2f} mN",
                    f"Scale (vmax): {f_scale_mN:.2f} mN",
                ]

        stats_text = "\n".join(info_lines)
        ax.text(10, 15, stats_text,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.75),
                fontsize=11, color='white', fontweight='bold',
                verticalalignment='top', transform=ax.transData)

        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2, alpha=0.8)

        if show_invisible and invisible_count > 0:
            ax.legend(loc='upper right')

        plt.tight_layout()

        output_filename = f"overlay_frame_{frame_idx:04d}.png"
        output_path = os.path.join(self.output_dir, output_filename)

        print(f"   SAUVEGARDE en cours: {output_filename}")
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            import gc; gc.collect()
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024
                print(f"   FICHIER SAUVÉ! {output_filename} ({file_size:.1f} KB)")
                return output_path
            else:
                print(f"   ERREUR: Fichier non créé après plt.savefig!")
                return None
        except Exception as e:
            print(f"   ERREUR sauvegarde: {e}")
            plt.close()
            return None

    
    def create_all_overlays(self, max_frames=200, vertex_size=2, show_invisible=False):
        """Crée tous les overlays"""
        
        print(f"CRÉATION OVERLAYS COMPLETS")
        print("=" * 60)

        # Compute global force scale FIRST so n_loaded uses the correct threshold
        global_vmax = self._compute_global_force_vmax()
        
        # Recherche données
        frame_data_list = self.find_images_and_data(max_frames)
        
        if not frame_data_list:
            print(f"❌ Aucune donnée trouvée")
            return []
        
        print(f"\nPlan de traitement:")
        print(f"   Frames à traiter: {len(frame_data_list)}")
        print(f"   Taille vertices: {vertex_size}px")
        print(f"   Afficher invisibles: {'Oui' if show_invisible else 'Non'}")
        print(f"   Dossier sortie: {self.output_dir}/")
        
        # Vérification accès dossier de sortie
        try:
            # Test création fichier temporaire
            test_file = os.path.join(self.output_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"   Accès écriture confirmé dans: {os.path.abspath(self.output_dir)}")
        except Exception as e:
            print(f"   ERREUR accès dossier: {e}")
            return []
        
        # global_vmax already computed above; log it here
        if global_vmax is not None:
            print(f"   Colormap fixe: 0 – {global_vmax*1e3:.2f} mN (identique pour toutes les frames)")

        # Traitement de tous les overlays
        output_files = []
        
        print(f"\nDÉBUT TRAITEMENT - SAUVEGARDE EN TEMPS RÉEL")
        print(f"📁 Dossier de sortie: {os.path.abspath(self.output_dir)}")
        
        for i, frame_data in enumerate(frame_data_list):
            frame_idx = frame_data['frame_index']
            print(f"\n{'='*15} FRAME {i+1}/{len(frame_data_list)} (Frame #{frame_idx:04d}) {'='*15}")

            # Traitement et sauvegarde immédiate
            output_file = self.create_single_overlay(
                frame_data, vertex_size, show_invisible,
                color_by='force', force_vmax=global_vmax
            )
            
            if output_file:
                output_files.append(output_file)
                # Vérification que le fichier existe vraiment
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / 1024  # KB
                    print(f"   SUCCÈS! Frame {frame_idx:03d} sauvée: {os.path.basename(output_file)} ({file_size:.1f} KB)")
                    print(f"   Chemin complet: {output_file}")
                else:
                    print(f"   ERREUR: Fichier non trouvé après sauvegarde!")
            else:
                print(f"   ÉCHEC traitement frame {frame_idx:03d}")
            
            # Progress avec indication de sauvegarde
            progress = (i + 1) / len(frame_data_list) * 100
            print(f"   Progression globale: {progress:.1f}% ({i+1}/{len(frame_data_list)}) - Status: {'✅ SAUVÉ' if output_file else '❌ ÉCHOUÉ'}")
            
            # Pause courte pour permettre l'écriture sur disque
            import time
            time.sleep(0.1)
        
        # Statistiques finales
        print(f"\n{'='*20} RÉSUMÉ FINAL {'='*20}")
        print(f"✅ Overlays créés: {len(output_files)}/{len(frame_data_list)}")
        
        # Création index et métadonnées
        self._create_overlay_index(frame_data_list, output_files)
        
        return output_files
    
    def _create_overlay_index(self, frame_data_list, output_files):
        """Crée un index des overlays créés"""
        
        index_data = {
            'session_id': self.session_id,
            'creation_date': datetime.now().isoformat(),
            'camera_params': {
                'position': self.camera_params['position'].tolist(),
                'orientation': self.camera_params['orientation'].tolist(),
                'lookat': self.camera_params['lookat'].tolist(),
                'fov': self.camera_params['field_of_view'],
                'viewport': [self.screen_width, self.screen_height]
            },
            'processing_summary': {
                'total_frames': len(frame_data_list),
                'successful_overlays': len(output_files),
                'total_vertices_processed': sum(len(fd['projected_data']) for fd in frame_data_list)
            },
            'frame_details': []
        }
        
        # Détails par frame
        for i, frame_data in enumerate(frame_data_list):
            projected_data = frame_data['projected_data']
            visible_count = np.sum(projected_data[:, 2] == 1)
            
            frame_info = {
                'frame_index': frame_data['frame_index'],
                'image_file': os.path.basename(frame_data['image_file']),
                'total_vertices': len(projected_data),
                'visible_vertices': int(visible_count),
                'visibility_rate': float(visible_count / len(projected_data) * 100),
                'output_file': os.path.basename(output_files[i]) if i < len(output_files) else None
            }
            
            index_data['frame_details'].append(frame_info)
        
        # Sauvegarde index
        index_file = os.path.join(self.output_dir, f"overlay_index_{self.session_id}.json")
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Index créé: {os.path.basename(index_file)}")
        
        return index_file
    
    def create_comparison_grid(self, output_files, grid_size=(4, 4)):
        """Crée une grille de comparaison de plusieurs overlays"""
        
        if not output_files:
            return None
        
        print(f"\nCRÉATION GRILLE COMPARAISON:")
        
        rows, cols = grid_size
        max_images = rows * cols
        
        # Sélection images équidistantes
        if len(output_files) > max_images:
            step = len(output_files) // max_images
            selected_files = [output_files[i * step] for i in range(max_images)]
        else:
            selected_files = output_files[:max_images]
        
        print(f"   Images sélectionnées: {len(selected_files)} sur {len(output_files)}")
        
        # Création grille
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        fig.suptitle(f'Brain Vertex Overlays - Comparison Grid\nSession: {self.session_id}', 
                     fontsize=16, fontweight='bold')
        
        for i, image_file in enumerate(selected_files):
            row = i // cols
            col = i % cols
            
            if row >= rows:
                break
            
            try:
                img = Image.open(image_file)
                axes[row][col].imshow(img)
                axes[row][col].set_title(f'Frame {os.path.basename(image_file)[13:17]}', fontsize=10)
                axes[row][col].axis('off')
            except:
                axes[row][col].text(0.5, 0.5, 'Erreur\nchargement', 
                                   ha='center', va='center', transform=axes[row][col].transAxes)
                axes[row][col].axis('off')
        
        # Masquer axes vides
        for i in range(len(selected_files), rows * cols):
            row = i // cols
            col = i % cols
            if row < rows:
                axes[row][col].axis('off')
        
        plt.tight_layout()
        
        # Sauvegarde
        grid_filename = os.path.join(self.output_dir, f"comparison_grid_{self.session_id}.png")
        plt.savefig(grid_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Grille sauvée: {os.path.basename(grid_filename)}")
        
        return grid_filename
    
    def create_statistics_summary(self, frame_data_list):
        """Crée un résumé statistique des overlays"""
        
        if not frame_data_list:
            return None
        
        print(f"\nCRÉATION RÉSUMÉ STATISTIQUE:")
        
        # Collecte statistiques
        frame_indices = []
        visibility_rates = []
        total_vertices = []
        visible_vertices = []
        
        for frame_data in frame_data_list:
            projected_data = frame_data['projected_data']
            visible_count = np.sum(projected_data[:, 2] == 1)
            
            frame_indices.append(frame_data['frame_index'])
            visibility_rates.append(visible_count / len(projected_data) * 100)
            total_vertices.append(len(projected_data))
            visible_vertices.append(int(visible_count))
        
        # Création graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Vertex Overlay Statistics - Session {self.session_id}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Taux de visibilité par frame
        axes[0, 0].plot(frame_indices, visibility_rates, 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].fill_between(frame_indices, visibility_rates, alpha=0.3)
        axes[0, 0].set_xlabel('Frame Index')
        axes[0, 0].set_ylabel('Visibilité (%)')
        axes[0, 0].set_title('Taux de Visibilité par Frame')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # 2. Nombre de vertices visibles
        axes[0, 1].bar(frame_indices, visible_vertices, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Vertices Visibles')
        axes[0, 1].set_title('Vertices Visibles par Frame')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution taux de visibilité
        axes[1, 0].hist(visibility_rates, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Taux de Visibilité (%)')
        axes[1, 0].set_ylabel('Nombre de Frames')
        axes[1, 0].set_title('Distribution Taux de Visibilité')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistiques textuelles
        axes[1, 1].axis('off')
        
        stats_text = f"""
STATISTIQUES GÉNÉRALES

Frames traitées: {len(frame_data_list)}
Vertices total moyen: {np.mean(total_vertices):.0f}
Vertices visibles moyen: {np.mean(visible_vertices):.0f}
Visibilité moyenne: {np.mean(visibility_rates):.1f}%
Visibilité médiane: {np.median(visibility_rates):.1f}%
Visibilité min: {np.min(visibility_rates):.1f}%
Visibilité max: {np.max(visibility_rates):.1f}%

Paramètres Caméra:
Position: [{self.camera_params['position'][0]:.1f}, {self.camera_params['position'][1]:.1f}, {self.camera_params['position'][2]:.1f}]
FOV: {self.camera_params['field_of_view']:.1f}°
Résolution: {self.screen_width}×{self.screen_height}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarde
        stats_filename = os.path.join(self.output_dir, f"statistics_summary_{self.session_id}.png")
        plt.savefig(stats_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Résumé statistique sauvé: {os.path.basename(stats_filename)}")
        
        return stats_filename

def overlay_vertices(image_path, vertices, output_path):
    """Superpose les sommets sur une image et sauvegarde le résultat"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    for vertex in vertices:
        x, y = int(vertex[0]), int(vertex[1])
        cv2.circle(image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(output_path, image)
    print(f"Image sauvegardée avec sommets: {output_path}")

def load_vertices_from_json(json_path):
    """Charge les sommets à partir d'un fichier JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['vertices'])

def process_images(image_dir, vertices_json, output_dir):
    """Traite les images en superposant les sommets"""
    os.makedirs(output_dir, exist_ok=True)

    vertices = load_vertices_from_json(vertices_json)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, f"overlay_{image_name}")
        overlay_vertices(image_path, vertices, output_path)

def main():
    """Menu principal pour création overlays"""
    import sys

    print("VERTEX IMAGE OVERLAY")
    print("=" * 60)
    print("Superposition des vertices projetés sur les images originales")
    print()

    # Optional: run_dir as first CLI argument
    # e.g.:  python overlay_vertices_on_images.py simulation_output/run_E2.50_nu0.450_seed1111
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Initialisation
    try:
        overlay_creator = VertexImageOverlay(run_dir=run_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Usage: python overlay_vertices_on_images.py [run_dir]")
        return
    
    # Menu options
    print("\nOptions disponibles:")
    print("  1. Créer tous les overlays (200 frames max)")
    print("  2. Créer overlays personnalisés")
    print("  3. Test sur 10 premières frames avec image")
    print("  4. Test sur une frame de simulation spécifique")
    print("  5. Quitter")
    
    try:
        choice = int(input("\nVotre choix (1-5): "))
        
        if choice == 1:
            print("\nCréation overlays complets...")
            output_files = overlay_creator.create_all_overlays(max_frames=200, vertex_size=2)
            
            if output_files:
                overlay_creator.create_comparison_grid(output_files, grid_size=(4, 4))
                frame_data_list = overlay_creator.find_images_and_data(200)
                overlay_creator.create_statistics_summary(frame_data_list)
                print(f"\nTraitement terminé! {len(output_files)} overlays créés")
                print(f"Résultats dans: {overlay_creator.output_dir}/")
            
        elif choice == 2:
            max_frames  = int(input("Nombre max de frames (défaut 200): ") or "200")
            vertex_size = int(input("Taille vertices en pixels (défaut 2): ") or "2")
            show_invisible = input("Afficher vertices invisibles? (y/N): ").lower() == 'y'
            color_by    = input("Colorier par force ou profondeur? (force/depth) [force]: ").strip() or "force"
            
            output_files = overlay_creator.create_all_overlays(
                max_frames=max_frames,
                vertex_size=vertex_size,
                show_invisible=show_invisible
            )
            
            if output_files:
                overlay_creator.create_comparison_grid(output_files)
                frame_data_list = overlay_creator.find_images_and_data(max_frames)
                overlay_creator.create_statistics_summary(frame_data_list)
                print(f"\nTraitement terminé! {len(output_files)} overlays créés")
            
        elif choice == 3:
            print("\nTest sur 10 premières frames avec image...")
            output_files = overlay_creator.create_all_overlays(max_frames=10, vertex_size=3)
            
            if output_files:
                overlay_creator.create_comparison_grid(output_files, grid_size=(2, 5))
                print(f"\nTest terminé! {len(output_files)} overlays créés")
            
        elif choice == 4:
            frame_number = int(input("Numéro de frame simulation (0-1999): "))
            if 0 <= frame_number <= 1999:
                print(f"\nTest frame spécifique: {frame_number}")
                
                target_frame = overlay_creator.find_single_frame_data(frame_number)
                
                if target_frame:
                    vertex_size    = int(input("Taille vertices en pixels (défaut 3): ") or "3")
                    show_invisible = input("Afficher vertices invisibles? (y/N): ").lower() == 'y'
                    color_by       = input("Colorier par force ou profondeur? (force/depth) [force]: ").strip() or "force"
                    
                    print(f"\nCréation overlay frame {frame_number}...")
                    global_vmax = overlay_creator._compute_global_force_vmax()
                    output_file = overlay_creator.create_single_overlay(
                        target_frame,
                        vertex_size=vertex_size,
                        show_invisible=show_invisible,
                        color_by=color_by,
                        force_vmax=global_vmax
                    )
                    
                    if output_file:
                        projected_data   = target_frame['projected_data']
                        force_magnitudes = target_frame.get('force_magnitudes')
                        visible_mask     = projected_data[:, 2] == 1
                        visible_count    = int(visible_mask.sum())
                        total_vertices   = len(projected_data)
                        visibility_rate  = visible_count / total_vertices * 100

                        print(f"\nStatistiques frame {frame_number}:")
                        print(f"   Total vertices     : {total_vertices:,}")
                        print(f"   Vertices visibles  : {visible_count:,}")
                        print(f"   Taux visibilité    : {visibility_rate:.1f}%")
                        if force_magnitudes is not None:
                            fv = force_magnitudes[visible_mask]
                            if len(fv):
                                print(f"   Force max visible  : {fv.max()*1e3:.3f} mN")
                                print(f"   Force moy visible  : {fv.mean()*1e3:.3f} mN")
                        depths = projected_data[visible_mask, 3]
                        if len(depths):
                            print(f"   Profondeur min/max : {depths.min():.3f} / {depths.max():.3f}")
                        print(f"   Fichier sauvé      : {output_file}")
                    else:
                        print(f"Erreur création overlay frame {frame_number}")
                else:
                    print(f"Frame {frame_number} introuvable dans les données")
            else:
                print("Numéro de frame invalide (doit être entre 0 et 1999)")
            
        elif choice == 5:
            print("Au revoir!")
            
        else:
            print("Choix invalide")
            
    except ValueError:
        print("Entrée invalide - veuillez entrer un nombre")
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur")

if __name__ == "__main__":
    main()
