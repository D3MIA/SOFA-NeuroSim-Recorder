import argparse
import os

def main():
    ap = argparse.ArgumentParser(description="Decimate an OBJ mesh while preserving texture coordinates (UV) using PyMeshLab.")
    ap.add_argument("--input", required=True, help="Input OBJ path")
    ap.add_argument("--output", required=True, help="Output OBJ path")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--target-faces", type=int, default=None, help="Target number of faces (triangles)")
    g.add_argument("--ratio", type=float, default=None, help="Target face ratio (0-1). Example: 0.25 keeps ~25% faces")
    ap.add_argument("--preserve-boundary", action="store_true", help="Try to preserve boundary/seams")
    ap.add_argument("--no-topology-preserve", action="store_true", help="Do not preserve topology (allows more aggressive decimation)")
    ap.add_argument("--quality-thr", type=float, default=0.3, help="Quality threshold (MeshLab default ~0.3)")
    ap.add_argument("--planar", action="store_true", help="Enable planar quadric optimization")
    args = ap.parse_args()

    try:
        import pymeshlab as ml
    except Exception as e:
        raise SystemExit("PyMeshLab is required. Install with: pip install pymeshlab")

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    ms = ml.MeshSet()
    ms.load_new_mesh(args.input)

    # Determine target faces
    cur = ms.current_mesh()
    cur_faces = cur.face_number()
    if args.target_faces is None and args.ratio is None:
        # default: 25% of current
        target_faces = max(1000, int(cur_faces * 0.25))
    elif args.target_faces is not None:
        target_faces = max(1000, int(args.target_faces))
    else:
        r = max(0.01, min(1.0, float(args.ratio)))
        target_faces = max(1000, int(cur_faces * r))

    # Apply Quadric Edge Collapse Decimation
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=bool(args.preserve_boundary),
        preservenormal=True,
        preservetopology=not bool(args.no_topology_preserve),
        optimalplacement=True,
        planarquadric=bool(args.planar),
        qualitythr=float(args.quality_thr),
        autoclean=True,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save OBJ with UVs (wedge texcoords). PyMeshLab writes an MTL if materials exist.
    ms.save_current_mesh(
        args.output,
        save_face_color=False,
        save_wedge_texcoord=True,
    )
    print(f"Saved decimated mesh to {args.output}")

if __name__ == "__main__":
    main()
