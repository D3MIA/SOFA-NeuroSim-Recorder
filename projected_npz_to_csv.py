#!/usr/bin/env python3
# projected_npz_to_csv.py - Conversion des fichiers NPZ projetés vers CSV

import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime

def estimate_projected_csv_size(npz_file):
    """Estime la taille du fichier CSV pour un NPZ projeté"""

    print("ESTIMATION TAILLE CSV PROJETE")
    print("=" * 40)

    data = np.load(npz_file)

    total_rows = 0
    total_columns = 0

    # Analyse des données
    has_frames = False
    has_projected = False
    n_frames = 0
    n_vertices = 0

    for key in data.files:
        array = data[key]
        print(f"{key}: {array.shape} ({array.dtype})")

        if key == 'frames' and len(array.shape) == 3:
            # frames: (n_frames, n_vertices, 3)
            n_frames, n_vertices, coords = array.shape
            has_frames = True
            print(f"   → Positions 3D: {n_frames} frames × {n_vertices} vertices")

        elif key == 'projected_pixels' and len(array.shape) == 3:
            # projected_pixels: (n_frames, n_vertices, 2)
            proj_frames, proj_vertices, coords = array.shape
            has_projected = True
            print(f"   → Projections pixel: {proj_frames} frames × {proj_vertices} vertices")

        elif key == 'visibility_masks' and len(array.shape) == 2:
            # visibility_masks: (n_frames, n_vertices)
            vis_frames, vis_vertices = array.shape
            print(f"   → Masques visibilité: {vis_frames} frames × {vis_vertices} vertices")

        elif key == 'depth_values' and len(array.shape) == 2:
            # depth_values: (n_frames, n_vertices)
            depth_frames, depth_vertices = array.shape
            print(f"   → Valeurs profondeur: {depth_frames} frames × {depth_vertices} vertices")

        elif key == 'displacements' and len(array.shape) == 3:
            # displacements: (n_frames, n_vertices, 3)
            disp_frames, disp_vertices, coords = array.shape
            print(f"   → Déplacements: {disp_frames} frames × {disp_vertices} vertices")

        elif key == 'rest' and len(array.shape) == 2:
            # rest: (n_vertices, 3)
            rest_vertices, coords = array.shape
            print(f"   → Position repos: {rest_vertices} vertices")

        elif key == 'times' and len(array.shape) == 1:
            print(f"   → Timestamps: {len(array)} frames")

    # Calcul du nombre total de lignes et colonnes
    if has_frames and has_projected:
        rows_for_data = n_frames * n_vertices
        total_rows += rows_for_data

        # Colonnes: frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible
        base_columns = 10  # frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible

        # Ajout colonnes optionnelles
        if 'displacements' in data.files:
            base_columns += 4  # disp_x, disp_y, disp_z, displacement_magnitude
        if 'rest' in data.files:
            base_columns += 3  # rest_x, rest_y, rest_z

        total_columns = base_columns

        # Affichage avec information sur les chunks recommandés
        avg_rows_per_frame = n_vertices
        recommended_frames_per_chunk = max(1, 1000000 // avg_rows_per_frame)  # ~1M lignes par chunk

        print(f"   → {rows_for_data:,} lignes avec {base_columns} colonnes")
        print(f"   → Recommandation: {recommended_frames_per_chunk} frames par chunk pour ~1M lignes")

    # Estimation taille
    if total_rows > 0:
        # Moyenne ~12 caractères par cellule (nombres + séparateurs)
        estimated_chars = total_rows * total_columns * 12
        estimated_mb = estimated_chars / 1024 / 1024
        estimated_gb = estimated_mb / 1024

        print(f"\nESTIMATION:")
        print(f"   Lignes totales: {total_rows:,}")
        print(f"   Colonnes: {total_columns}")
        print(f"   Taille estimée: {estimated_mb:.1f} MB ({estimated_gb:.2f} GB)")

        # Warnings
        if estimated_gb > 2:
            print(f"ATTENTION: Fichier très volumineux ({estimated_gb:.1f} GB)")
            print(f"   - Temps d'écriture: ~{estimated_gb*10:.0f} minutes")
            print(f"   - RAM nécessaire: ~{estimated_gb*1.5:.1f} GB")
            print(f"   - Recommandation: Utiliser conversion par chunks")

        return estimated_gb

    data.close()
    return 0

def convert_projected_npz_to_csv_chunked(npz_file, frames_per_chunk=50):
    """Conversion NPZ projeté → CSV par chunks"""

    print("\nCONVERSION NPZ PROJETE → CSV PAR CHUNKS")
    print("=" * 50)

    data = np.load(npz_file)

    # Vérification des clés requises
    required_keys = ['frames', 'projected_pixels', 'visibility_masks', 'depth_values']
    missing_keys = [key for key in required_keys if key not in data.files]

    if missing_keys:
        print(f"Clés manquantes: {missing_keys}")
        data.close()
        return None

    # Chargement des données
    frames = data['frames']  # Shape: (n_frames, n_vertices, 3)
    projected_pixels = data['projected_pixels']  # Shape: (n_frames, n_vertices, 2)
    visibility_masks = data['visibility_masks']  # Shape: (n_frames, n_vertices)
    depth_values = data['depth_values']  # Shape: (n_frames, n_vertices)
    
    # Données optionnelles
    displacements = data.get('displacements', None)  # Shape: (n_frames, n_vertices, 3) ou None
    rest_positions = data.get('rest', None)  # Shape: (n_vertices, 3) ou None
    times = data.get('times', np.arange(len(frames)) * 0.01)  # Fallback times

    n_frames, n_vertices, _ = frames.shape
    total_rows = n_frames * n_vertices

    print("Données détectées:")
    print(f"   Frames: {n_frames}")
    print(f"   Vertices par frame: {n_vertices}")
    print(f"   Total lignes: {total_rows:,}")
    print(f"   Projections pixel: Disponibles")
    print(f"   Masques visibilité: Disponibles") 
    print(f"   Valeurs profondeur: Disponibles")
    print(f"   Déplacements: {'Disponibles' if displacements is not None else 'Non disponibles'}")
    print(f"   Position repos: {'Disponible' if rest_positions is not None else 'Non disponible'}")

    # Préparation fichier de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"brain_projected_{timestamp}"

    # Calcul des lignes par chunk basé sur les frames
    rows_per_chunk = frames_per_chunk * n_vertices

    if frames_per_chunk >= n_frames:
        # Conversion simple (un seul fichier)
        print(f"Conversion simple ({frames_per_chunk} frames ≥ {n_frames} frames totales)")
        return _convert_single_projected_csv(frames, projected_pixels, visibility_masks, depth_values, 
                                           times, base_filename, displacements, rest_positions)
    else:
        # Conversion par chunks
        n_chunks = (n_frames // frames_per_chunk) + (1 if n_frames % frames_per_chunk > 0 else 0)
        print(f"Conversion par chunks ({n_chunks} fichiers de {frames_per_chunk} frames chacun)")
        return _convert_chunked_projected_csv(frames, projected_pixels, visibility_masks, depth_values,
                                            times, base_filename, frames_per_chunk, displacements, rest_positions)

def _convert_single_projected_csv(frames, projected_pixels, visibility_masks, depth_values, times, 
                                base_filename, displacements=None, rest_positions=None):
    """Conversion en un seul fichier CSV avec toutes les données projetées"""

    output_file = f"projected_npz/{base_filename}.csv"
    os.makedirs("projected_npz", exist_ok=True)

    print(f"Écriture: {os.path.basename(output_file)}")

    # Définition des colonnes
    columns = ['frame', 'vertex_id', 'time', 'x', 'y', 'z', 'pixel_x', 'pixel_y', 'depth_ndc', 'is_visible']

    if displacements is not None:
        columns.extend(['disp_x', 'disp_y', 'disp_z', 'displacement_magnitude'])

    if rest_positions is not None:
        columns.extend(['rest_x', 'rest_y', 'rest_z'])

    print(f"Colonnes CSV: {columns}")

    # Préparation des données
    rows = []
    n_frames, n_vertices, _ = frames.shape

    for frame_idx in range(len(frames)):
        frame_data = frames[frame_idx]
        frame_pixels = projected_pixels[frame_idx]
        frame_visibility = visibility_masks[frame_idx]
        frame_depths = depth_values[frame_idx]
        time_val = times[frame_idx] if frame_idx < len(times) else frame_idx * 0.01

        # Données de déplacement pour cette frame (si disponibles)
        frame_displacements = displacements[frame_idx] if displacements is not None else None

        for vertex_idx in range(n_vertices):
            position = frame_data[vertex_idx]
            pixel_pos = frame_pixels[vertex_idx]
            is_visible = int(frame_visibility[vertex_idx])
            depth = frame_depths[vertex_idx]

            # Base: frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible
            row = [
                frame_idx,
                vertex_idx,
                time_val,
                position[0],
                position[1],
                position[2],
                pixel_pos[0],
                pixel_pos[1],
                depth,
                is_visible
            ]

            # Ajout des déplacements si disponibles
            if frame_displacements is not None:
                displacement_vec = frame_displacements[vertex_idx]
                displacement_mag = np.linalg.norm(displacement_vec)
                row.extend([
                    displacement_vec[0],
                    displacement_vec[1],
                    displacement_vec[2],
                    displacement_mag
                ])

            # Ajout position de repos si disponible
            if rest_positions is not None and vertex_idx < len(rest_positions):
                rest_pos = rest_positions[vertex_idx]
                row.extend([
                    rest_pos[0],
                    rest_pos[1],
                    rest_pos[2]
                ])

            rows.append(row)

        # Progress
        if (frame_idx + 1) % 10 == 0:
            print(f"   Frame {frame_idx + 1}/{len(frames)} traité...")

    # Création DataFrame et sauvegarde
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False, float_format='%.6f')

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"Fichier créé: {size_mb:.1f} MB")
    print(f"Statistiques: {len(rows):,} lignes × {len(columns)} colonnes")

    return output_file

def _convert_chunked_projected_csv(frames, projected_pixels, visibility_masks, depth_values, times,
                                 base_filename, frames_per_chunk, displacements=None, rest_positions=None):
    """Conversion en plusieurs fichiers CSV (chunks) avec données projetées"""

    n_frames, n_vertices, _ = frames.shape

    print(f"Configuration chunks:")
    print(f"   Frames par chunk: {frames_per_chunk}")
    print(f"   Lignes par chunk: {frames_per_chunk * n_vertices:,}")

    # Définition des colonnes
    columns = ['frame', 'vertex_id', 'time', 'x', 'y', 'z', 'pixel_x', 'pixel_y', 'depth_ndc', 'is_visible']

    if displacements is not None:
        columns.extend(['disp_x', 'disp_y', 'disp_z', 'displacement_magnitude'])

    if rest_positions is not None:
        columns.extend(['rest_x', 'rest_y', 'rest_z'])

    print(f"Colonnes CSV: {columns}")

    output_files = []
    chunk_idx = 0

    os.makedirs("projected_npz", exist_ok=True)

    for start_frame in range(0, n_frames, frames_per_chunk):
        end_frame = min(start_frame + frames_per_chunk, n_frames)

        # Nom du chunk
        chunk_filename = f"projected_npz/{base_filename}_chunk_{chunk_idx:03d}.csv"

        chunk_frame_count = end_frame - start_frame
        chunk_row_count = chunk_frame_count * n_vertices
        print(f"Chunk {chunk_idx}: frames {start_frame}-{end_frame-1} ({chunk_frame_count} frames, {chunk_row_count:,} lignes)")

        # Données pour ce chunk
        chunk_frames = frames[start_frame:end_frame]
        chunk_pixels = projected_pixels[start_frame:end_frame]
        chunk_visibility = visibility_masks[start_frame:end_frame]
        chunk_depths = depth_values[start_frame:end_frame]
        chunk_times = times[start_frame:end_frame] if start_frame < len(times) else [i * 0.01 for i in range(start_frame, end_frame)]
        chunk_displacements = displacements[start_frame:end_frame] if displacements is not None else None

        # Conversion
        rows = []
        for rel_frame_idx in range(chunk_frame_count):
            abs_frame_idx = start_frame + rel_frame_idx
            frame_data = chunk_frames[rel_frame_idx]
            frame_pixels = chunk_pixels[rel_frame_idx]
            frame_visibility = chunk_visibility[rel_frame_idx]
            frame_depths = chunk_depths[rel_frame_idx]
            time_val = chunk_times[rel_frame_idx] if rel_frame_idx < len(chunk_times) else abs_frame_idx * 0.01

            # Données de déplacement pour cette frame (si disponibles)
            frame_displacements = chunk_displacements[rel_frame_idx] if chunk_displacements is not None else None

            for vertex_idx in range(n_vertices):
                position = frame_data[vertex_idx]
                pixel_pos = frame_pixels[vertex_idx]
                is_visible = int(frame_visibility[vertex_idx])
                depth = frame_depths[vertex_idx]

                # Base: frame, vertex_id, time, x, y, z, pixel_x, pixel_y, depth_ndc, is_visible
                row = [
                    abs_frame_idx,  # Frame absolu
                    vertex_idx,
                    time_val,
                    position[0],
                    position[1],
                    position[2],
                    pixel_pos[0],
                    pixel_pos[1],
                    depth,
                    is_visible
                ]

                # Ajout des déplacements si disponibles
                if frame_displacements is not None:
                    displacement_vec = frame_displacements[vertex_idx]
                    displacement_mag = np.linalg.norm(displacement_vec)
                    row.extend([
                        displacement_vec[0],
                        displacement_vec[1],
                        displacement_vec[2],
                        displacement_mag
                    ])

                # Ajout position de repos si disponible
                if rest_positions is not None and vertex_idx < len(rest_positions):
                    rest_pos = rest_positions[vertex_idx]
                    row.extend([
                        rest_pos[0],
                        rest_pos[1],
                        rest_pos[2]
                    ])

                rows.append(row)

        # Sauvegarde chunk
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(chunk_filename, index=False, float_format='%.6f')

        size_mb = os.path.getsize(chunk_filename) / 1024 / 1024
        print(f"   {os.path.basename(chunk_filename)}: {size_mb:.1f} MB ({len(rows):,} lignes)")

        output_files.append(chunk_filename)
        chunk_idx += 1

        # Nettoyage mémoire
        del rows, df
        gc.collect()

    # Création d'un fichier index
    index_file = f"projected_npz/{base_filename}_INDEX.txt"
    with open(index_file, 'w') as f:
        f.write(f"Conversion NPZ projeté → CSV par chunks\n")
        f.write(f"=======================================\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total frames: {n_frames}\n")
        f.write(f"Total vertices: {n_vertices}\n")
        f.write(f"Colonnes: {', '.join(columns)}\n")
        f.write(f"Projections incluses: Oui\n")
        f.write(f"Déplacements inclus: {'Oui' if displacements is not None else 'Non'}\n")
        f.write(f"Position repos incluse: {'Oui' if rest_positions is not None else 'Non'}\n")
        f.write(f"Frames par chunk: {frames_per_chunk}\n")
        f.write(f"Nombre de chunks: {len(output_files)}\n\n")
        f.write("Fichiers créés:\n")
        for i, filepath in enumerate(output_files):
            f.write(f"  {i+1}. {os.path.basename(filepath)}\n")

    print(f"Index créé: {os.path.basename(index_file)}")

    return output_files

def smart_projected_conversion_menu():
    """Menu intelligent pour conversion des NPZ projetés"""

    print("CONVERSION INTELLIGENTE NPZ PROJETE → CSV")
    print("=" * 60)

    # Recherche fichiers NPZ projetés
    export_dir = "projected_npz"
    if not os.path.exists(export_dir):
        print(f"Dossier {export_dir} introuvable")
        return

    npz_files = [f for f in os.listdir(export_dir) if f.endswith('.npz') and 'projected' in f]

    if not npz_files:
        print("Aucun fichier NPZ projeté trouvé")
        print("Cherchez également dans le dossier simulation_output...")
        
        # Fallback: chercher dans simulation_output
        fallback_dir = "simulation_output"
        if os.path.exists(fallback_dir):
            fallback_files = [f for f in os.listdir(fallback_dir) if f.endswith('.npz') and 'projected' in f]
            if fallback_files:
                print(f"Fichiers trouvés dans {fallback_dir}:")
                for i, filename in enumerate(fallback_files, 1):
                    filepath = os.path.join(fallback_dir, filename)
                    size_mb = os.path.getsize(filepath) / 1024 / 1024
                    print(f"   {i}. {filename} ({size_mb:.1f} MB)")
                
                # Utiliser fallback_dir comme dossier de travail
                export_dir = fallback_dir
                npz_files = fallback_files
            else:
                print("Aucun fichier NPZ projeté trouvé nulle part")
                return
        else:
            return

    print("Fichiers NPZ projetés disponibles:")
    for i, filename in enumerate(npz_files, 1):
        filepath = os.path.join(export_dir, filename)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"   {i}. {filename} ({size_mb:.1f} MB)")

    # Sélection fichier
    try:
        choice = int(input(f"\nChoisir fichier (1-{len(npz_files)}): ")) - 1
        if 0 <= choice < len(npz_files):
            selected_file = os.path.join(export_dir, npz_files[choice])
        else:
            print("Choix invalide")
            return
    except ValueError:
        print("Choix invalide")
        return

    # Estimation taille
    estimated_gb = estimate_projected_csv_size(selected_file)

    if estimated_gb > 2:
        print(f"\nFICHIER VOLUMINEUX DÉTECTÉ ({estimated_gb:.1f} GB)")
        print("Options recommandées:")
        print("  1. Conversion par chunks (recommandé)")
        print("  2. Conversion simple (risque de crash)")
        print("  3. Annuler")

        try:
            option = int(input("Votre choix (1-3): "))
            if option == 1:
                # Calculer la recommandation basée sur les données
                data_temp = np.load(selected_file)
                if 'frames' in data_temp.files:
                    frames_shape = data_temp['frames'].shape
                    n_frames_total = frames_shape[0]
                    n_vertices = frames_shape[1]
                    # Recommandation: ~50 frames par chunk pour éviter les gros fichiers
                    recommended_frames = min(50, max(1, 1000000 // n_vertices))

                    print(f"\nInformations:")
                    print(f"   Total frames: {n_frames_total}")
                    print(f"   Vertices par frame: {n_vertices}")
                    print(f"   Recommandation: {recommended_frames} frames par chunk")

                    frames_per_chunk = int(input(f"Frames par chunk (défaut {recommended_frames}): ") or str(recommended_frames))
                    convert_projected_npz_to_csv_chunked(selected_file, frames_per_chunk)
                else:
                    frames_per_chunk = int(input("Frames par chunk (défaut 50): ") or "50")
                    convert_projected_npz_to_csv_chunked(selected_file, frames_per_chunk)
                data_temp.close()
            elif option == 2:
                convert_projected_npz_to_csv_chunked(selected_file, float('inf'))  # Toutes les frames en un seul chunk
            else:
                print("Annulé")
        except ValueError:
            print("Choix invalide")
    else:
        # Conversion directe pour petits fichiers
        convert_projected_npz_to_csv_chunked(selected_file)

if __name__ == "__main__":
    smart_projected_conversion_menu()
