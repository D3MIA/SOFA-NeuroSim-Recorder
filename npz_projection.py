#!/usr/bin/env python3
# npz_projection_direct.py - Projection directe sur fichiers NPZ avec sauvegarde incremental

import os
import json
import uuid
import numpy as np
from datetime import datetime
import glob

class NPZDirectProjector:
    """Projecteur 3D→Pixel direct sur fichiers NPZ avec sauvegarde incremental"""
    
    def __init__(self, camera_params_file=None, backup_interval=50):
        self.session_id = str(uuid.uuid4())[:8]
        self.backup_interval = backup_interval
        
        # Chargement paramètres caméra SOFA réels
        self.camera_params = self._load_real_camera_params(camera_params_file)
        
        # Utilisation des dimensions RÉELLES du viewport SOFA
        self.screen_width = self.camera_params['viewport_width']
        self.screen_height = self.camera_params['viewport_height']
        
        print(f"NPZDirectProjector {self.session_id} - PROJECTION DIRECTE NPZ")
        print(f"   Camera SOFA InteractiveCamera - MATRICES PRE-CALCULEES")
        print(f"   Position: [{self.camera_params['position'][0]:.1f}, {self.camera_params['position'][1]:.1f}, {self.camera_params['position'][2]:.1f}]")
        print(f"   Quaternion: [{self.camera_params['orientation'][0]:.3f}, {self.camera_params['orientation'][1]:.3f}, {self.camera_params['orientation'][2]:.3f}, {self.camera_params['orientation'][3]:.3f}]")
        print(f"   Viewport REEL: {self.screen_width}x{self.screen_height}")
        print(f"   FOV: {self.camera_params['field_of_view']:.1f}°")
        print(f"   Sauvegarde incremental: toutes les {self.backup_interval} frames sur le même fichier")
        
        # Création dossier de sortie
        self.output_dir = "projected_npz"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"   Dossier sortie: {self.output_dir}/")
    
    def _load_real_camera_params(self, camera_params_file):
        """Charge les paramètres RÉELS de la caméra SOFA InteractiveCamera"""
        
        print("Chargement paramètres caméra SOFA RÉELS...")
        
        # Recherche automatique du fichier de paramètres le plus récent
        if camera_params_file is None:
            camera_params_file = self._find_latest_camera_params()
        
        if camera_params_file and os.path.exists(camera_params_file):
            try:
                with open(camera_params_file, 'r') as f:
                    data = json.load(f)
                
                print(f"   Fichier chargé: {os.path.basename(camera_params_file)}")
                
                # Extraction des paramètres essentiels
                if "essential_params" in data:
                    params = data["essential_params"]
                else:
                    # Fallback vers l'ancien format
                    params = data
                
                camera_params = {
                    'position': np.array(params['position']),
                    'orientation': np.array(params['orientation']),
                    'lookat': np.array(params.get('lookAt', params.get('lookat', [0, 0, 0]))),
                    'field_of_view': params.get('fieldOfView', params.get('field_of_view_degrees', 45.0)),
                    'viewport_width': params.get('widthViewport', params['viewport']['width'] if 'viewport' in params else 1920),
                    'viewport_height': params.get('heightViewport', params['viewport']['height'] if 'viewport' in params else 1080),
                    'intrinsic_matrix': np.array(params['intrinsic_matrix']) if 'intrinsic_matrix' in params else (np.array(params['intrinsics']['intrinsic_matrix']) if 'intrinsics' in params else None),
                    'focal_length': {'fx': params.get('fx'), 'fy': params.get('fy')} if 'fx' in params else (params['intrinsics']['focal_length'] if 'intrinsics' in params else None),
                    'principal_point': {'cx': params.get('cx'), 'cy': params.get('cy')} if 'cx' in params else (params['intrinsics']['principal_point'] if 'intrinsics' in params else None),
                    'znear': params.get('zNear', params.get('znear', 0.1)),
                    'zfar': params.get('zFar', params.get('zfar', 1000.0)),
                    'distance': params.get('distance', 10.0),
                    'projection_matrix': None,
                    'modelview_matrix': None
                }
                
                # Si on a accès aux matrices SOFA
                if "all_camera_attributes" in data:
                    attrs = data["all_camera_attributes"]
                    camera_params.update({
                        'projection_matrix': np.array(attrs['projectionMatrix']).reshape(4, 4) if 'projectionMatrix' in attrs else None,
                        'modelview_matrix': np.array(attrs['modelViewMatrix']).reshape(4, 4) if 'modelViewMatrix' in attrs else None,
                    })
                else:
                    # Fallback to direct access
                    if 'projectionMatrix' in params:
                        camera_params['projection_matrix'] = np.array(params['projectionMatrix']).reshape(4, 4)
                    if 'modelViewMatrix' in params:
                        camera_params['modelview_matrix'] = np.array(params['modelViewMatrix']).reshape(4, 4)
                
                print(f"   Position extraite: [{camera_params['position'][0]:.3f}, {camera_params['position'][1]:.3f}, {camera_params['position'][2]:.3f}]")
                print(f"   Quaternion extrait: [{camera_params['orientation'][0]:.6f}, {camera_params['orientation'][1]:.6f}, {camera_params['orientation'][2]:.6f}, {camera_params['orientation'][3]:.6f}]")
                print(f"   FOV: {camera_params['field_of_view']:.1f}°")
                print(f"   Viewport REEL: {camera_params['viewport_width']}x{camera_params['viewport_height']}")
                
                if camera_params['projection_matrix'] is not None:
                    print(f"   Matrice projection SOFA: disponible (4x4) - SERA UTILISEE DIRECTEMENT")
                    
                if camera_params['modelview_matrix'] is not None:
                    print(f"   Matrice ModelView SOFA: disponible (4x4) - SERA UTILISEE DIRECTEMENT")
                
                return camera_params
                
            except Exception as e:
                print(f"Erreur lecture {camera_params_file}: {e}")
        
        # Fallback vers paramètres par défaut
        print("   Utilisation paramètres par défaut SOFA InteractiveCamera")
        return {
            'position': np.array([-81.2953, -21.7954, 112.397]),
            'orientation': np.array([0.0778814, -0.314528, 0.0648681, 0.943821]),
            'lookat': np.array([0, 0, 0]),
            'field_of_view': 45.0,
            'viewport_width': 1920,
            'viewport_height': 1080,
            'intrinsic_matrix': None,
            'focal_length': None,
            'principal_point': None,
            'znear': 4.34275,
            'zfar': 236.208,
            'distance': 10.0,
            'projection_matrix': None,
            'modelview_matrix': None
        }
    
    def _find_latest_camera_params(self):
        """Trouve le fichier de paramètres caméra le plus récent"""
        export_dir = "simulation_output"
        if not os.path.exists(export_dir):
            return None
        
        # Cherche les fichiers de paramètres caméra
        camera_files = []
        for filename in os.listdir(export_dir):
            if filename.startswith('camera_params_') and filename.endswith('.json'):
                camera_files.append(os.path.join(export_dir, filename))
        
        if not camera_files:
            return None
        
        # Prend le plus récent
        latest_file = max(camera_files, key=lambda f: os.path.getmtime(f))
        print(f"   Fichier auto-détecté: {os.path.basename(latest_file)}")
        return latest_file
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convertit quaternion en matrice de rotation"""
        qx, qy, qz, qw = q
        
        rotation_matrix = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return rotation_matrix
    
    def _create_view_matrix_from_lookat(self, eye, target, up=None):
        """Crée une matrice view avec lookAt"""
        if up is None:
            up = np.array([0, 0, 1])  # Up vector par défaut
        
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)
        
        # Direction de vision (normalisée)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        # Vecteur right
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Vecteur up corrigé
        up_corrected = np.cross(right, forward)
        
        # Matrice de rotation (world → camera)
        R = np.array([
            [right[0], right[1], right[2]],
            [up_corrected[0], up_corrected[1], up_corrected[2]],
            [-forward[0], -forward[1], -forward[2]]
        ])
        
        # Translation
        t = -R @ eye
        
        # Matrice view complète
        view_matrix = np.zeros((4, 4))
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = t
        view_matrix[3, 3] = 1.0
        
        return view_matrix
    
    def _create_projection_matrices(self):
        """Utilisation des matrices SOFA pré-calculées"""
        
        # Paramètres SOFA réels pour info
        position = self.camera_params['position']
        fov = self.camera_params['field_of_view']
        near = self.camera_params['znear']
        far = self.camera_params['zfar']
        
        # PRIORITÉ 1: Utilise les matrices SOFA pré-calculées si disponibles
        if (self.camera_params['projection_matrix'] is not None and 
            self.camera_params['modelview_matrix'] is not None):
            
            proj_matrix = self.camera_params['projection_matrix'].copy()
            view_matrix = self.camera_params['modelview_matrix'].copy()
            
            print(f"   Utilisation MATRICES SOFA PRE-CALCULEES:")
            print(f"      Projection Matrix SOFA (4x4): DIRECTE")
            print(f"      ModelView Matrix SOFA (4x4): DIRECTE")
            print(f"      Précision MAXIMALE (matrices exactes SOFA)")
            
        # FALLBACK: Si pas de matrices SOFA, calcul manuel
        else:
            print(f"   Fallback: Calcul matrices manuellement...")
            
            # Utilise la matrice intrinsèque si disponible
            if self.camera_params['intrinsic_matrix'] is not None:
                # Adaptation de la matrice intrinsèque à notre résolution
                K = self.camera_params['intrinsic_matrix'].copy()
                
                # Mise à l'échelle pour notre résolution cible
                original_width = self.camera_params['viewport_width']
                original_height = self.camera_params['viewport_height']
                
                scale_x = self.screen_width / original_width
                scale_y = self.screen_height / original_height
                
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] *= scale_x  # cx
                K[1, 2] *= scale_y  # cy
                
                print(f"      Utilisation matrice intrinsèque mise à l'échelle:")
                print(f"      Résolution: {original_width}x{original_height} → {self.screen_width}x{self.screen_height}")
                
                # Conversion en matrice de projection OpenGL
                proj_matrix = self._intrinsic_to_projection_matrix(K, near, far)
            else:
                # Calcul classique depuis FOV
                aspect = self.screen_width / self.screen_height
                fov_radians = np.radians(fov)
                
                f = 1.0 / np.tan(fov_radians / 2.0)
                proj_matrix = np.array([
                    [f/aspect, 0, 0, 0],
                    [0, f, 0, 0],
                    [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
                    [0, 0, -1, 0]
                ])
                
                print(f"      Calcul projection depuis FOV: {fov:.1f}°")
            
            # Matrice View : utilise lookAt si disponible
            if 'lookat' in self.camera_params and self.camera_params['lookat'] is not None:
                target = self.camera_params['lookat']
                view_matrix = self._create_view_matrix_from_lookat(position, target)
                print(f"      Matrice View (lookAt): eye={position} → target={target}")
            else:
                # Méthode quaternion
                orientation = self.camera_params['orientation']
                view_matrix = self._create_view_matrix_from_quaternion(position, orientation)
                print(f"      Matrice View (quaternion)")
        
        return view_matrix, proj_matrix
    
    def _intrinsic_to_projection_matrix(self, K, near, far):
        """Convertit matrice intrinsèque en matrice de projection OpenGL"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Normalisation par rapport à la résolution
        width, height = self.screen_width, self.screen_height
        
        # Matrice de projection OpenGL
        proj_matrix = np.array([
            [2*fx/width, 0, (width - 2*cx)/width, 0],
            [0, 2*fy/height, (2*cy - height)/height, 0],
            [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
            [0, 0, -1, 0]
        ])
        
        return proj_matrix
    
    def _create_view_matrix_from_quaternion(self, position, orientation):
        """Crée matrice View depuis quaternion"""
        # Matrice rotation depuis quaternion
        rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
        
        # Vecteurs directionnels
        forward = -rotation_matrix[:, 2]
        up = rotation_matrix[:, 1]
        right = rotation_matrix[:, 0]
        
        # Matrice View (EXTRINSÈQUE)
        view_matrix = np.array([
            [right[0], right[1], right[2], -np.dot(right, position)],
            [up[0], up[1], up[2], -np.dot(up, position)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, position)],
            [0, 0, 0, 1]
        ])
        
        return view_matrix
    
    def project_vertices_to_pixels(self, vertices_3d):
        """Projection 3D→Pixel"""
        
        view_matrix, proj_matrix = self._create_projection_matrices()
        
        n_vertices = len(vertices_3d)
        pixel_coords = np.zeros((n_vertices, 2))
        visibility_mask = np.zeros(n_vertices, dtype=bool)
        depths = np.zeros(n_vertices)
        
        # Transformation de tous les vertices
        vertices_homogeneous = np.column_stack([vertices_3d, np.ones(n_vertices)])
        
        # View + Projection
        vertices_view = (view_matrix @ vertices_homogeneous.T).T
        vertices_clip = (proj_matrix @ vertices_view.T).T
        
        # Conversion pixels
        for i in range(n_vertices):
            if vertices_clip[i, 3] != 0:
                # NDC
                ndc_x = vertices_clip[i, 0] / vertices_clip[i, 3]
                ndc_y = vertices_clip[i, 1] / vertices_clip[i, 3]
                ndc_z = vertices_clip[i, 2] / vertices_clip[i, 3]
                
                # Pixels
                pixel_x = (ndc_x + 1.0) * 0.5 * self.screen_width
                pixel_y = (1.0 - ndc_y) * 0.5 * self.screen_height
                
                pixel_coords[i] = [pixel_x, pixel_y]
                depths[i] = ndc_z
                
                # Visibilité
                in_screen = (0 <= pixel_x <= self.screen_width and 
                           0 <= pixel_y <= self.screen_height)
                in_front = (vertices_view[i, 2] < -0.1)
                
                visibility_mask[i] = in_screen and in_front
            else:
                pixel_coords[i] = [0, 0]
                depths[i] = 1.0
                visibility_mask[i] = False
        
        return pixel_coords, visibility_mask, depths
    
    def find_npz_files(self, search_pattern="brain_surface_*.npz"):
        """Trouve tous les fichiers NPZ"""
        export_dir = "simulation_output"
        
        if not os.path.exists(export_dir):
            print(f"Dossier {export_dir} introuvable")
            return []
        
        # Recherche des fichiers NPZ
        search_path = os.path.join(export_dir, search_pattern)
        npz_files = glob.glob(search_path)
        
        if not npz_files:
            print(f"Aucun fichier NPZ trouvé avec pattern: {search_pattern}")
            return []
        
        # Tri par nom pour ordre cohérent
        npz_files.sort()
        
        print(f"Fichiers NPZ détectés:")
        for i, npz_file in enumerate(npz_files):
            size_mb = os.path.getsize(npz_file) / 1024 / 1024
            print(f"   {i+1}. {os.path.basename(npz_file)} ({size_mb:.1f} MB)")
        
        return npz_files
    
    def _save_incremental_backup(self, output_file, data_dict, frame_idx):
        """Sauvegarde incremental sur le même fichier"""
        np.savez_compressed(output_file, **data_dict)
        print(f"      Sauvegarde incremental frame {frame_idx}: {os.path.basename(output_file)}")
        return output_file
    
    def process_npz_file(self, npz_file):
        """Traite un fichier NPZ avec projection et sauvegarde incremental"""
        
        print(f"\nTRAITEMENT NPZ: {os.path.basename(npz_file)}")
        print("=" * 60)
        
        try:
            # Chargement du fichier NPZ
            data = np.load(npz_file)
            print(f"   Fichier NPZ chargé: {os.path.basename(npz_file)}")
            
            # Vérification des clés requises
            required_keys = ['frames']
            missing_keys = [key for key in required_keys if key not in data.files]
            
            if missing_keys:
                print(f"   Clés manquantes: {missing_keys}")
                return None
            
            # Extraction des données
            frames = data['frames']
            rest_positions = data.get('rest', None)
            displacements = data.get('displacements', None)
            times = data.get('times', None)
            
            n_frames, n_vertices, _ = frames.shape
            print(f"   Données extraites:")
            print(f"     Frames: {n_frames}")
            print(f"     Vertices par frame: {n_vertices}")
            print(f"     Rest positions: {'Disponible' if rest_positions is not None else 'Non disponible'}")
            print(f"     Displacements: {'Disponible' if displacements is not None else 'Non disponible'}")
            print(f"     Times: {'Disponible' if times is not None else 'Non disponible'}")
            
            # Préparation des données de sortie
            projected_pixels = np.zeros((n_frames, n_vertices, 2), dtype=np.float32)
            visibility_masks = np.zeros((n_frames, n_vertices), dtype=bool)
            depth_values = np.zeros((n_frames, n_vertices), dtype=np.float32)
            
            # Fichier de sortie
            output_filename = os.path.basename(npz_file).replace('.npz', f'_projected_{self.session_id}.npz')
            output_file = os.path.join(self.output_dir, output_filename)
            
            print(f"   Début projection de {n_frames} frames...")
            
            # Traitement frame par frame avec sauvegarde incremental
            for frame_idx in range(n_frames):
                frame_vertices = frames[frame_idx]
                
                # Projection 3D→Pixel
                pixel_coords, visibility_mask, depths = self.project_vertices_to_pixels(frame_vertices)
                
                # Stockage
                projected_pixels[frame_idx] = pixel_coords
                visibility_masks[frame_idx] = visibility_mask
                depth_values[frame_idx] = depths
                
                # Affichage progression
                if (frame_idx + 1) % 10 == 0:
                    visible_count = np.sum(visibility_mask)
                    visibility_rate = visible_count / len(visibility_mask) * 100
                    print(f"      Frame {frame_idx+1}/{n_frames}: {visible_count}/{len(visibility_mask)} visibles ({visibility_rate:.1f}%)")
                
                # Sauvegarde incremental
                if (frame_idx + 1) % self.backup_interval == 0:
                    print(f"   Sauvegarde incremental à la frame {frame_idx+1}...")
                    
                    # Données à sauvegarder jusqu'à maintenant
                    current_data = {
                        'frames': frames[:frame_idx+1],
                        'projected_pixels': projected_pixels[:frame_idx+1],
                        'visibility_masks': visibility_masks[:frame_idx+1],
                        'depth_values': depth_values[:frame_idx+1]
                    }
                    
                    # Ajout des données optionnelles si disponibles
                    if rest_positions is not None:
                        current_data['rest'] = rest_positions
                    if displacements is not None:
                        current_data['displacements'] = displacements[:frame_idx+1]
                    if times is not None:
                        current_data['times'] = times[:frame_idx+1]
                    
                    # Sauvegarde incremental sur le même fichier
                    self._save_incremental_backup(output_file, current_data, frame_idx+1)
            
            # Sauvegarde finale
            print(f"   Sauvegarde finale...")
            
            # Données complètes
            final_data = {
                'frames': frames,
                'projected_pixels': projected_pixels,
                'visibility_masks': visibility_masks,
                'depth_values': depth_values
            }
            
            # Ajout des données optionnelles
            if rest_positions is not None:
                final_data['rest'] = rest_positions
            if displacements is not None:
                final_data['displacements'] = displacements
            if times is not None:
                final_data['times'] = times
            
            # Sauvegarde fichier final
            np.savez_compressed(output_file, **final_data)
            
            # Statistiques finales
            total_visible = np.sum(visibility_masks)
            total_vertices = n_frames * n_vertices
            overall_visibility = total_visible / total_vertices * 100
            
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            
            print(f"   Projection terminée:")
            print(f"     Fichier final: {output_filename}")
            print(f"     Taille: {size_mb:.1f} MB")
            print(f"     Total vertices: {total_vertices:,}")
            print(f"     Vertices visibles: {total_visible:,} ({overall_visibility:.1f}%)")
            print(f"     Nouvelles clés NPZ: projected_pixels, visibility_masks, depth_values")
            
            # Création métadonnées
            metadata = {
                'session_id': self.session_id,
                'processing_date': datetime.now().isoformat(),
                'input_file': os.path.basename(npz_file),
                'output_file': output_filename,
                'camera_params': {
                    'position': self.camera_params['position'].tolist(),
                    'orientation': self.camera_params['orientation'].tolist(),
                    'lookat': self.camera_params['lookat'].tolist(),
                    'fov': self.camera_params['field_of_view'],
                    'viewport': [self.screen_width, self.screen_height]
                },
                'projection_stats': {
                    'total_frames': int(n_frames),
                    'vertices_per_frame': int(n_vertices),
                    'total_vertices': int(total_vertices),
                    'total_visible': int(total_visible),
                    'visibility_rate': float(overall_visibility),
                    'backup_interval': self.backup_interval
                },
                'npz_keys': {
                    'original': list(data.files),
                    'added': ['projected_pixels', 'visibility_masks', 'depth_values']
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
                'total_visible': int(total_visible),
                'visibility_rate': overall_visibility,
                'output_size_mb': size_mb
            }
            
        except Exception as e:
            print(f"   Erreur traitement NPZ: {e}")
            return None

def main():
    """Menu principal pour projection directe NPZ"""
    
    print("NPZ DIRECT PROJECTOR")
    print("=" * 60)
    print("Projection 3D→Pixel directe sur fichiers NPZ avec sauvegarde incremental")
    print()
    
    # Configuration backup interval
    backup_interval = 50
    try:
        user_interval = input(f"Interval de sauvegarde (défaut {backup_interval} frames): ").strip()
        if user_interval:
            backup_interval = int(user_interval)
    except ValueError:
        print("Utilisation valeur par défaut")
    
    # Initialisation projecteur
    projector = NPZDirectProjector(backup_interval=backup_interval)
    
    # Menu options
    print("Options disponibles:")
    print("  1. Traiter tous les fichiers NPZ automatiquement")
    print("  2. Traiter avec pattern personnalisé")
    print("  3. Traiter un seul fichier NPZ spécifique")
    print("  4. Quitter")
    
    try:
        choice = int(input("\nVotre choix (1-4): "))
        
        if choice == 1:
            # Traitement automatique
            npz_files = projector.find_npz_files()
            if npz_files:
                for npz_file in npz_files:
                    result = projector.process_npz_file(npz_file)
                    if result:
                        print(f"Fichier traité avec succès: {result['output_file']}")
                        
        elif choice == 2:
            # Pattern personnalisé
            pattern = input("Pattern de recherche (ex: brain_surface_*.npz): ")
            if pattern.strip():
                npz_files = projector.find_npz_files(pattern.strip())
                if npz_files:
                    for npz_file in npz_files:
                        result = projector.process_npz_file(npz_file)
                        if result:
                            print(f"Fichier traité avec succès: {result['output_file']}")
            else:
                print("Pattern vide")
                
        elif choice == 3:
            # Fichier spécifique
            npz_files = projector.find_npz_files()
            if npz_files:
                print("\nFichiers NPZ disponibles:")
                for i, npz_file in enumerate(npz_files):
                    print(f"  {i+1}. {os.path.basename(npz_file)}")
                
                file_choice = int(input(f"\nChoisir fichier (1-{len(npz_files)}): ")) - 1
                if 0 <= file_choice < len(npz_files):
                    result = projector.process_npz_file(npz_files[file_choice])
                    if result:
                        print(f"Fichier traité avec succès!")
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
