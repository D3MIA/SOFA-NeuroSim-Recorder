#!/usr/bin/env python3
# overlay_vertices_on_images.py - Superposition des sommets sur les images

import os
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

class VertexImageOverlay:
    """Superposition des sommets sur les images originales"""
    
    def __init__(self, camera_params_file=None):
        self.session_id = str(uuid.uuid4())[:8]
        
        # Chargement paramètres caméra SOFA réels
        self.camera_params = self._load_real_camera_params(camera_params_file)
        
        # Dimensions viewport
        self.screen_width = self.camera_params['viewport_width']
        self.screen_height = self.camera_params['viewport_height']
        
        print(f"VertexImageOverlay {self.session_id} - SUPERPOSITION VERTICES SUR IMAGES")
        print(f"   Position caméra: [{self.camera_params['position'][0]:.1f}, {self.camera_params['position'][1]:.1f}, {self.camera_params['position'][2]:.1f}]")
        print(f"   Viewport: {self.screen_width}x{self.screen_height}")
        
        # Dossiers
        self.images_dir = "simulation_output/images"
        self.projected_dir = "projected_chunks"
        self.output_dir = "overlayed_frames"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"   Images originales: {self.images_dir}/")
        print(f"   Données projetées: {self.projected_dir}/")
        print(f"   Sortie overlays: {self.output_dir}/")
    
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
    
    def _find_latest_camera_params(self):
        """Trouve le fichier de paramètres caméra le plus récent"""
        export_dir = "simulation_output"
        if not os.path.exists(export_dir):
            return None
        
        camera_files = []
        for filename in os.listdir(export_dir):
            if filename.startswith('camera_params_') and filename.endswith('.json'):
                camera_files.append(os.path.join(export_dir, filename))
        
        if camera_files:
            return max(camera_files, key=lambda f: os.path.getmtime(f))
        return None
    
    def find_single_frame_data(self, frame_idx):
        """Trouve les données pour une seule frame spécifique"""
        
        print(f"\nRECHERCHE DONNÉES FRAME {frame_idx}:")
        
        # 1. Vérification image correspondante
        if not os.path.exists(self.images_dir):
            print(f"❌ Dossier images introuvable: {self.images_dir}")
            return None
        
        # Recherche image pour cette frame
        expected_image = os.path.join(self.images_dir, f"frame_{frame_idx:04d}.png")
        if not os.path.exists(expected_image):
            # Essai avec d'autres formats
            for ext in ['.jpg', '.jpeg']:
                alt_image = os.path.join(self.images_dir, f"frame_{frame_idx:04d}{ext}")
                if os.path.exists(alt_image):
                    expected_image = alt_image
                    break
            else:
                print(f"❌ Image frame {frame_idx} introuvable")
                return None
        
        print(f"   Image trouvée: {os.path.basename(expected_image)}")
        
        # 2. Recherche données projetées pour cette frame
        if not os.path.exists(self.projected_dir):
            print(f"❌ Dossier données projetées introuvable: {self.projected_dir}")
            return None
        
        projected_files = glob.glob(os.path.join(self.projected_dir, "*_projected.csv"))
        if not projected_files:
            print(f"❌ Aucune donnée projetée trouvée")
            return None
        
        # Trouve les vertices pour cette frame spécifique
        projected_data = self._find_projected_data_for_frame(frame_idx, projected_files)
        
        if projected_data is None:
            print(f"❌ Aucune donnée trouvée pour frame {frame_idx}")
            return None
        
        print(f"   Vertices trouvés: {len(projected_data)}")
        
        return {
            'frame_index': frame_idx,
            'image_file': expected_image,
            'projected_data': projected_data
        }

    def find_images_and_data(self, max_frames=200):
        """Trouve les images et données projetées correspondantes"""
        
        print(f"\nRECHERCHE IMAGES ET DONNÉES:")
        
        # 1. Recherche images originales
        if not os.path.exists(self.images_dir):
            print(f"❌ Dossier images introuvable: {self.images_dir}")
            return []
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Tri par nom pour ordre cohérent
        image_files.sort()
        
        if not image_files:
            print(f"❌ Aucune image trouvée dans {self.images_dir}")
            return []
        
        print(f"   Images trouvées: {len(image_files)}")
        
        # 2. Recherche données projetées
        if not os.path.exists(self.projected_dir):
            print(f"❌ Dossier données projetées introuvable: {self.projected_dir}")
            return []
        
        projected_files = glob.glob(os.path.join(self.projected_dir, "*_projected.csv"))
        projected_files.sort()
        
        if not projected_files:
            print(f"❌ Aucune donnée projetée trouvée dans {self.projected_dir}")
            return []
        
        print(f"   Fichiers projetés trouvées: {len(projected_files)}")
        
        # 3. Correspondance frame par frame
        frame_data = []
        
        for frame_idx in range(min(max_frames, len(image_files))):
            if frame_idx >= len(image_files):
                break
                
            image_file = image_files[frame_idx]
            
            # Trouve les données projetées pour cette frame
            projected_data = self._find_projected_data_for_frame(frame_idx, projected_files)
            
            if projected_data is not None:
                frame_data.append({
                    'frame_index': frame_idx,
                    'image_file': image_file,
                    'projected_data': projected_data
                })
                
                if frame_idx < 10:  # Log des premières frames
                    print(f"   Frame {frame_idx:03d}: {os.path.basename(image_file)} + {len(projected_data)} vertices")
        
        print(f"   Correspondances trouvées: {len(frame_data)} frames")
        
        return frame_data
    
    def _find_projected_data_for_frame(self, frame_idx, projected_files):
        """Trouve les données projetées pour une frame spécifique"""
        
        all_vertices = []
        
        # Parcourt tous les fichiers chunks projetés
        for projected_file in projected_files:
            try:
                df = pd.read_csv(projected_file)
                
                # Filtre pour la frame spécifique
                frame_data = df[df['frame'] == frame_idx]
                
                if len(frame_data) > 0:
                    # Extrait les données nécessaires
                    vertices_data = frame_data[['pixel_x', 'pixel_y', 'is_visible', 'depth_ndc']].values
                    all_vertices.extend(vertices_data)
                    
            except Exception as e:
                continue  # Ignore les erreurs de lecture
        
        if len(all_vertices) > 0:
            return np.array(all_vertices)
        
        return None
    
    def create_single_overlay(self, frame_data, vertex_size=2, show_invisible=False):
        """Crée un overlay pour une seule frame"""
        
        frame_idx = frame_data['frame_index']
        image_file = frame_data['image_file']
        projected_data = frame_data['projected_data']
        
        print(f"DÉBUT création overlay frame {frame_idx:03d}: {len(projected_data)} vertices")
        
        # Chargement image originale
        try:
            print(f"   Chargement image: {os.path.basename(image_file)}")
            original_image = Image.open(image_file)
            img_width, img_height = original_image.size
            
            # Redimensionnement si nécessaire
            if img_width != self.screen_width or img_height != self.screen_height:
                original_image = original_image.resize((self.screen_width, self.screen_height), Image.LANCZOS)
                print(f"   Image redimensionnée: {img_width}x{img_height} → {self.screen_width}x{self.screen_height}")
            
        except Exception as e:
            print(f"   Erreur chargement image: {e}")
            return None
        
        print(f"   Création figure matplotlib...")
        # Création overlay avec matplotlib pour meilleur contrôle
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        
        # Affichage image de fond
        ax.imshow(original_image, extent=[0, self.screen_width, self.screen_height, 0])
        
        # Séparation vertices visibles/invisibles
        visible_mask = projected_data[:, 2] == 1  # is_visible
        
        visible_vertices = projected_data[visible_mask]
        invisible_vertices = projected_data[~visible_mask]
        
        visible_count = len(visible_vertices)
        invisible_count = len(invisible_vertices)
        
        print(f"   Vertices: {visible_count:,} visibles, {invisible_count:,} invisibles")
        
        # Overlay vertices visibles (points colorés par profondeur)
        if len(visible_vertices) > 0:
            pixel_x = visible_vertices[:, 0]
            pixel_y = visible_vertices[:, 1]
            depths = visible_vertices[:, 3]
            
            # Points colorés par profondeur
            # Colormaps disponibles:
            # 'plasma' - Violet(proche) → Rouge(loin) [DÉFAUT]
            # 'viridis' - Violet(proche) → Jaune(loin) 
            # 'coolwarm' - Bleu(proche) → Rouge(loin)
            # 'jet' - Bleu(proche) → Rouge(loin) [classique]
            # 'hot' - Noir(proche) → Blanc(loin)
            scatter = ax.scatter(pixel_x, pixel_y, c=depths, cmap='plasma', 
                               s=vertex_size**2, alpha=0.8, edgecolors='white', linewidth=0.3)
            
            # Colorbar pour profondeur
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Profondeur NDC', rotation=270, labelpad=15)
        
        # Overlay vertices invisibles (optionnel)
        if show_invisible and len(invisible_vertices) > 0:
            pixel_x_inv = invisible_vertices[:, 0]
            pixel_y_inv = invisible_vertices[:, 1]
            
            ax.scatter(pixel_x_inv, pixel_y_inv, c='red', s=vertex_size**2/4, 
                      alpha=0.3, marker='x', label='Hors écran')
        
        # Configuration axes
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)  # Inversion Y pour image
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        ax.set_title(f'Brain Vertices Overlay - Frame {frame_idx:03d}\n'
                    f'Visibles: {visible_count:,} | Invisibles: {invisible_count:,}', 
                    fontsize=14, fontweight='bold')
        
        # Ajout statistiques
        stats_text = f'Frame: {frame_idx:03d}\nVisibles: {visible_count:,}\nInvisibles: {invisible_count:,}\nTotal: {len(projected_data):,}'
        ax.text(10, 40, stats_text, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
               fontsize=10, color='white', fontweight='bold')
        
        # Marqueur centre écran
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2, alpha=0.8)
        
        # Légende
        if show_invisible and invisible_count > 0:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Sauvegarde avec feedback détaillé
        output_filename = f"overlay_frame_{frame_idx:04d}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        
        print(f"   SAUVEGARDE en cours: {output_filename}")
        print(f"   Chemin: {output_path}")
        
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            
            # Force flush to ensure immediate save
            import gc
            gc.collect()  # Force garbage collection to free memory immediately
            
            # Vérification que le fichier a bien été créé
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024  # KB
                print(f"   FICHIER SAUVÉ! {output_filename} ({file_size:.1f} KB)")
                return output_path
            else:
                print(f"   ERREUR: Fichier non créé après plt.savefig!")
                return None
                
        except Exception as e:
            print(f"   ERREUR sauvegarde: {e}")
            plt.close()  # Ferme quand même la figure
            return None
    
    def create_all_overlays(self, max_frames=200, vertex_size=2, show_invisible=False):
        """Crée tous les overlays"""
        
        print(f"CRÉATION OVERLAYS COMPLETS")
        print("=" * 60)
        
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
        
        # Traitement de tous les overlays
        output_files = []
        
        print(f"\nDÉBUT TRAITEMENT - SAUVEGARDE EN TEMPS RÉEL")
        print(f"📁 Dossier de sortie: {os.path.abspath(self.output_dir)}")
        
        for i, frame_data in enumerate(frame_data_list):
            frame_idx = frame_data['frame_index']
            print(f"\n{'='*15} FRAME {i+1}/{len(frame_data_list)} (Frame #{frame_idx:03d}) {'='*15}")
            print(f"🎯 DÉBUT traitement frame {frame_idx:03d}...")
            
            # Traitement et sauvegarde immédiate
            output_file = self.create_single_overlay(frame_data, vertex_size, show_invisible)
            
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
    
    print("VERTEX IMAGE OVERLAY")
    print("=" * 60)
    print("Superposition des vertices projetés sur les images originales")
    print()
    
    # Initialisation
    overlay_creator = VertexImageOverlay()
    
    # Menu options
    print("Options disponibles:")
    print("  1. Créer tous les overlays (200 frames max)")
    print("  2. Créer overlays personnalisés")
    print("  3. Test sur quelques frames")
    print("  4. Test sur une frame spécifique")
    print("  5. Quitter")
    
    try:
        choice = int(input("\nVotre choix (1-5): "))
        
        if choice == 1:
            # Traitement complet
            print("\nCréation overlays complets...")
            output_files = overlay_creator.create_all_overlays(max_frames=200, vertex_size=2)
            
            if output_files:
                # Grille de comparaison
                overlay_creator.create_comparison_grid(output_files, grid_size=(4, 4))
                
                # Statistiques
                frame_data_list = overlay_creator.find_images_and_data(200)
                overlay_creator.create_statistics_summary(frame_data_list)
                
                print(f"\nTraitement terminé! {len(output_files)} overlays créés")
                print(f"Résultats dans: {overlay_creator.output_dir}/")
            
        elif choice == 2:
            # Paramètres personnalisés
            max_frames = int(input("Nombre max de frames (défaut 200): ") or "200")
            vertex_size = int(input("Taille vertices en pixels (défaut 2): ") or "2")
            show_invisible = input("Afficher vertices invisibles? (y/N): ").lower() == 'y'
            
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
            # Test sur quelques frames
            print("\nTest sur 10 premières frames...")
            output_files = overlay_creator.create_all_overlays(max_frames=10, vertex_size=3)
            
            if output_files:
                overlay_creator.create_comparison_grid(output_files, grid_size=(2, 5))
                print(f"\nTest terminé! {len(output_files)} overlays créés")
            
        elif choice == 4:
            # Test frame spécifique
            frame_number = int(input("Numéro de frame à tester (0-199): "))
            if 0 <= frame_number <= 199:
                print(f"\nTest frame spécifique: {frame_number}")
                
                # Trouve les données pour cette frame uniquement
                target_frame = overlay_creator.find_single_frame_data(frame_number)
                
                if target_frame:
                    # Paramètres pour frame spécifique
                    vertex_size = int(input("Taille vertices en pixels (défaut 3): ") or "3")
                    show_invisible = input("Afficher vertices invisibles? (y/N): ").lower() == 'y'
                    
                    print(f"\nCréation overlay frame {frame_number}...")
                    output_file = overlay_creator.create_single_overlay(
                        target_frame, 
                        vertex_size=vertex_size, 
                        show_invisible=show_invisible
                    )
                    
                    if output_file:
                        print(f"Overlay frame {frame_number} créé!")
                        print(f"Fichier: {output_file}")
                        
                        # Statistiques détaillées pour cette frame
                        projected_data = target_frame['projected_data']
                        visible_count = np.sum(projected_data[:, 2] == 1)
                        total_vertices = len(projected_data)
                        visibility_rate = visible_count / total_vertices * 100
                        
                        print(f"\nStatistiques frame {frame_number}:")
                        print(f"   Total vertices: {total_vertices:,}")
                        print(f"   Vertices visibles: {visible_count:,}")
                        print(f"   Taux visibilité: {visibility_rate:.1f}%")
                        
                        # Analyse profondeur
                        depths = projected_data[:, 3]  # depth_ndc
                        visible_depths = depths[projected_data[:, 2] == 1]
                        if len(visible_depths) > 0:
                            print(f"   Profondeur min: {visible_depths.min():.3f}")
                            print(f"   Profondeur max: {visible_depths.max():.3f}")
                            print(f"   Profondeur moyenne: {visible_depths.mean():.3f}")
                    else:
                        print(f"Erreur création overlay frame {frame_number}")
                else:
                    print(f"Frame {frame_number} introuvable dans les données")
            else:
                print("Numéro de frame invalide (doit être entre 0 et 199)")
            
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
