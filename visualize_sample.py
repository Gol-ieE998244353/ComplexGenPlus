import argparse
import math
import pickle
from pathlib import Path

import numpy as np

from load_ply import write_ply
from visualize_worst import (
    create_default_mtl,
    curve_type_list,
    extract_corners_from_curves,
    gen_cylinder_from_two_points,
    gen_sphere_from_point,
    patch_type_list,
    write_obj_grouped,
)


CURVE_TYPE_TO_ID = {
    "Circle": 0,
    "BSpline": 1,
    "Line": 2,
    "Ellipse": 3,
}

PATCH_TYPE_TO_ID = {
    "Cylinder": 0,
    "Torus": 1,
    "BSpline": 2,
    "Extrusion": 2,
    "Revolution": 2,
    "Plane": 3,
    "Cone": 4,
    "Sphere": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize raw samples stored in a pickle file")
    parser.add_argument("--input", required=True, help="Path to a single-sample or packed pickle file")
    parser.add_argument("--output", default="visualized_samples", help="Output directory")
    parser.add_argument("--sample-index", type=int, default=None, help="0-based sample index inside the pickle stream")
    parser.add_argument("--sample-name", default=None, help="Match sample_id or filename to export a single sample")
    parser.add_argument("--corner-threshold", type=float, default=0.02, help="Corner clustering threshold")
    parser.add_argument("--curve-radius", type=float, default=0.008, help="Curve cylinder radius")
    parser.add_argument("--corner-radius", type=float, default=0.02, help="Corner sphere radius")
    return parser.parse_args()


def ensure_numpy(data, min_dim=None):
    array = np.asarray(data)
    if min_dim is not None and array.ndim < min_dim:
        return None
    return array


def sanitize_name(name):
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def sample_name_from_data(sample, fallback_idx):
    if isinstance(sample, dict):
        for key in ["sample_id", "filename", "name"]:
            value = sample.get(key)
            if value:
                return sanitize_name(Path(str(value)).stem)
    return f"sample_{fallback_idx:06d}"


def read_pickle_stream(pkl_path):
    samples = []
    with open(pkl_path, "rb") as handle:
        while True:
            try:
                obj = pickle.load(handle)
            except EOFError:
                break
            # Support several container formats:
            # - legacy: a sequence of pickled objects (list/tuple/obj)
            # - grouped: single dict {"samples": [...]}
            if isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], list):
                samples.extend(obj["samples"])
            elif isinstance(obj, list):
                samples.extend(obj)
            elif isinstance(obj, tuple):
                samples.extend(list(obj))
            else:
                samples.append(obj)
    return samples


def detect_sample_format(sample):
    if not isinstance(sample, dict):
        return "unknown"
    if "surface_points" in sample or isinstance(sample.get("curves"), list) or isinstance(sample.get("patches"), list):
        return "raw"
    return "unknown"


def normalize_curve_label(label):
    if isinstance(label, str):
        return CURVE_TYPE_TO_ID.get(label, 1)
    return int(label)


def normalize_patch_label(label):
    if isinstance(label, str):
        return PATCH_TYPE_TO_ID.get(label, 2)
    return int(label)


def infer_grid(points):
    points = ensure_numpy(points, min_dim=2)
    if points is None or points.shape[1] < 3:
        return None, None, None

    xyz = points[:, :3].astype(np.float32)
    grid_size = int(round(math.sqrt(len(xyz))))
    if grid_size * grid_size != len(xyz):
        return xyz, None, None
    return xyz, grid_size, grid_size


def gen_patch_quads(rows, cols, offset, wrap_rows=False, wrap_cols=False):
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = offset + i * cols + j
            v2 = v1 + 1
            v3 = v1 + cols
            v4 = v3 + 1
            faces.append([v1, v2, v4, v3])

    if wrap_cols:
        for i in range(rows - 1):
            v1 = offset + i * cols + (cols - 1)
            v2 = offset + i * cols
            v3 = offset + (i + 1) * cols
            v4 = offset + (i + 1) * cols + (cols - 1)
            faces.append([v1, v2, v3, v4])

    if wrap_rows:
        for j in range(cols - 1):
            v1 = offset + (rows - 1) * cols + j
            v2 = offset + (rows - 1) * cols + j + 1
            v3 = offset + j + 1
            v4 = offset + j
            faces.append([v1, v2, v3, v4])

    if wrap_rows and wrap_cols:
        faces.append([
            offset + rows * cols - 1,
            offset + (rows - 1) * cols,
            offset,
            offset + cols - 1,
        ])

    return faces


def export_geometry_obj(curves, patches, corners, output_path, corner_threshold, curve_radius, corner_radius):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_path.parent / "default.mtl"
    if not mtl_path.exists():
        create_default_mtl(mtl_path)

    all_verts = []
    all_faces = []
    all_mtls = []
    all_names = []
    vertex_offset = 0

    for patch_idx, patch in enumerate(patches):
        patch_points, rows, cols = infer_grid(patch.get("points"))
        if patch_points is None or rows is None or cols is None:
            continue

        grid = patch_points.reshape(rows, cols, 3)
        grid = np.transpose(grid, (1, 0, 2))
        rows_t, cols_t = grid.shape[:2]

        patch_label = normalize_patch_label(patch.get("label", 2))
        all_names.append(f"patch_{patch_idx}_{patch_type_list[patch_label]}")
        all_mtls.append(f"m{patch_idx % 15}")
        all_verts.append(grid.reshape(-1, 3))
        all_faces.append(
            gen_patch_quads(
                rows_t,
                cols_t,
                vertex_offset,
                wrap_rows=bool(patch.get("v_closed", False)),
                wrap_cols=bool(patch.get("u_closed", False)),
            )
        )
        vertex_offset += rows_t * cols_t

    for curve_idx, curve in enumerate(curves):
        curve_points = ensure_numpy(curve.get("points"), min_dim=2)
        if curve_points is None or curve_points.shape[1] < 3 or len(curve_points) < 2:
            continue

        points = curve_points[:, :3].astype(np.float32)
        curve_label = normalize_curve_label(curve.get("label", 1))
        curve_verts = []
        curve_faces = []

        for point_idx in range(len(points) - 1):
            vertices, faces = gen_cylinder_from_two_points(
                points[point_idx],
                points[point_idx + 1],
                vertex_offset,
                radius=curve_radius,
            )
            if len(faces) == 0:
                continue
            curve_verts.append(vertices)
            curve_faces.extend(faces)
            vertex_offset += len(vertices)

        if curve.get("is_closed", False):
            vertices, faces = gen_cylinder_from_two_points(
                points[-1],
                points[0],
                vertex_offset,
                radius=curve_radius,
            )
            if len(faces) > 0:
                curve_verts.append(vertices)
                curve_faces.extend(faces)
                vertex_offset += len(vertices)

        if curve_verts:
            all_names.append(f"curve_{curve_idx}_{curve_type_list[curve_label]}")
            all_mtls.append("cylinder")
            all_verts.append(np.concatenate(curve_verts, axis=0))
            all_faces.append(curve_faces)

    if corners is None:
        curve_points = [curve.get("points") for curve in curves if curve.get("points") is not None]
        curve_points = [ensure_numpy(points, min_dim=2)[:, :3] for points in curve_points if ensure_numpy(points, min_dim=2) is not None]
        closed_flags = np.array([bool(curve.get("is_closed", False)) for curve in curves], dtype=np.float32)
        if curve_points:
            corners, _ = extract_corners_from_curves(
                curve_points,
                closed_flags,
                np.ones(len(curve_points), dtype=np.float32),
                threshold=corner_threshold,
            )

    if corners is not None:
        corners = ensure_numpy(corners, min_dim=2)
        if corners is not None and corners.shape[1] >= 3:
            for corner_idx, corner in enumerate(corners[:, :3]):
                vertices, faces = gen_sphere_from_point(corner, resolution=8, radius=corner_radius)
                all_names.append(f"corner_{corner_idx}")
                all_mtls.append("sphere")
                all_verts.append(vertices)
                all_faces.append(faces)
                vertex_offset += len(vertices)

    write_obj_grouped(output_path, all_verts, all_faces, all_mtls, all_names)


def build_raw_geometry(sample):
    curves = []
    for curve in sample.get("curves", []):
        points = ensure_numpy(curve.get("points"), min_dim=2)
        if points is None or points.shape[1] < 3:
            continue
        curves.append(
            {
                "points": points[:, :3].astype(np.float32),
                "is_closed": bool(curve.get("is_closed", False)),
                "label": normalize_curve_label(curve.get("type", 1)),
            }
        )

    patches = []
    for patch in sample.get("patches", []):
        patch_points = patch.get("grid_normal", patch.get("patch_points"))
        points = ensure_numpy(patch_points, min_dim=2)
        if points is None:
            continue
        if points.ndim == 3:
            points = points.reshape(-1, points.shape[-1])
        if points.shape[1] < 3:
            continue
        patches.append(
            {
                "points": points[:, :3].astype(np.float32),
                "u_closed": bool(patch.get("u_closed", False)),
                "v_closed": bool(patch.get("v_closed", False)),
                "label": normalize_patch_label(patch.get("type", 2)),
            }
        )

    corners = sample.get("corners")
    corners = ensure_numpy(corners, min_dim=2) if corners is not None else None
    return curves, patches, corners


def export_raw_sample(sample, sample_dir, args):
    curves, patches, corners = build_raw_geometry(sample)
    export_geometry_obj(
        curves,
        patches,
        corners,
        sample_dir / f"{sample_dir.name}.obj",
        args.corner_threshold,
        args.curve_radius,
        args.corner_radius,
    )

    surface_points = sample.get("surface_points")
    if surface_points is not None:
        points = ensure_numpy(surface_points, min_dim=2)
        if points is not None and points.shape[1] >= 3:
            export_points = points[:, :6] if points.shape[1] >= 6 else points[:, :3]
            write_ply(sample_dir / f"{sample_dir.name}_input.ply", export_points)


def select_samples(samples, args):
    indexed_samples = [(idx, sample_name_from_data(sample, idx), sample) for idx, sample in enumerate(samples)]

    if args.sample_index is not None:
        selected = [item for item in indexed_samples if item[0] == args.sample_index]
        if not selected:
            raise IndexError(f"sample-index {args.sample_index} is out of range, total samples: {len(samples)}")
        return selected

    if args.sample_name is not None:
        selected = [
            item
            for item in indexed_samples
            if item[1] == sanitize_name(args.sample_name)
            or (isinstance(item[2], dict) and str(item[2].get("sample_id", "")) == args.sample_name)
            or (isinstance(item[2], dict) and str(item[2].get("filename", "")) == args.sample_name)
        ]
        if not selected:
            raise KeyError(f"sample-name {args.sample_name} not found in pickle file")
        return selected

    return indexed_samples


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_pickle_stream(args.input)
    if not samples:
        raise RuntimeError(f"No samples found in {args.input}")

    selected_samples = select_samples(samples, args)
    print(f"Loaded {len(samples)} sample(s) from {args.input}")
    print(f"Exporting {len(selected_samples)} sample(s) to {output_dir}")

    for index, sample_name, sample in selected_samples:
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_format = detect_sample_format(sample)
        print(f"[{index}] {sample_name}: format={sample_format}")

        if sample_format == "raw":
            export_raw_sample(sample, sample_dir, args)
        else:
            raise ValueError(
                f"Unsupported sample format for {sample_name}. visualize_sample.py only supports raw dataset samples; keys={list(sample.keys()) if isinstance(sample, dict) else type(sample)}"
            )


if __name__ == "__main__":
    main()