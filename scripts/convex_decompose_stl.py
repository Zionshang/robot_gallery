#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose an STL mesh into convex parts with trimesh and CoACD."
    )
    parser.add_argument("input_stl", help="Path to the input STL file.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=50,
        help="CoACD sampling resolution.",
    )
    parser.add_argument(
        "--max-convex-hull",
        type=int,
        default=1,
        help="Maximum number of convex hulls.",
    )
    return parser.parse_args()


def load_single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            tuple(geom for geom in loaded.geometry.values())
        )
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Input mesh is empty: {path}")
    return loaded


def run_coacd(mesh: trimesh.Trimesh, args: argparse.Namespace) -> list[trimesh.Trimesh]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    coacd_mesh = coacd.Mesh(vertices, faces)

    parts = coacd.run_coacd(
        coacd_mesh,
        resolution=args.resolution,
        max_convex_hull=args.max_convex_hull,
    )

    convex_parts = []
    for part_vertices, part_faces in parts:
        convex_parts.append(
            trimesh.Trimesh(
                vertices=np.asarray(part_vertices),
                faces=np.asarray(part_faces),
                process=False,
            )
        )
    return convex_parts


def export_parts(parts: list[trimesh.Trimesh], input_stl: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, part in enumerate(parts):
        output_path = output_dir / f"{input_stl.stem}_convex_{index:03d}.stl"
        part.export(output_path)
        print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()

    global coacd, np, trimesh
    import coacd
    import numpy as np
    import trimesh

    input_stl = Path(args.input_stl).expanduser().resolve()
    if not input_stl.is_file():
        raise FileNotFoundError(f"STL not found: {input_stl}")

    output_dir = input_stl.with_name(f"{input_stl.stem}_convex")

    mesh = load_single_mesh(input_stl)
    parts = run_coacd(mesh, args)
    if not parts:
        raise RuntimeError("CoACD returned no convex parts.")

    export_parts(parts, input_stl, output_dir)
    print(f"Generated {len(parts)} convex part(s).")


if __name__ == "__main__":
    main()
