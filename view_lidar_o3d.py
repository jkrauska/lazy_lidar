#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import laspy
import numpy as np
import open3d as o3d


def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def read_header_info(path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    with laspy.open(path) as laz:
        point_count = laz.header.point_count
        # mins/maxs are scaled values
        mins = np.asarray(laz.header.mins, dtype=np.float64)
        maxs = np.asarray(laz.header.maxs, dtype=np.float64)
    return int(point_count), mins, maxs


def estimate_memory_bytes(
    num_points: int,
    include_rgb: bool,
    extra_overhead_per_point: int = 8,
) -> int:
    xyz_bytes = 12  # float32 x,y,z
    color_bytes = 12 if include_rgb else 0  # float32 r,g,b in Open3D
    per_point = xyz_bytes + color_bytes + extra_overhead_per_point
    return int(num_points) * per_point


def compute_auto_voxel_size_from_bounds(
    mins: np.ndarray, maxs: np.ndarray, num_points: int, target_points: int
) -> float:
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    extent = np.maximum(maxs - mins, 1e-6)
    volume = float(extent[0] * extent[1] * extent[2])
    # One point per voxel assumption -> number of voxels ~ volume / voxel_size^3
    # Set volume / s^3 ~= target_points  -> s ~= (volume / target_points)^(1/3)
    safe_target = max(1, min(int(target_points), int(num_points)))
    voxel_size = (volume / float(safe_target)) ** (1.0 / 3.0)
    # Clamp to reasonable range
    return float(np.clip(voxel_size, 1e-3, max(extent.max() * 0.1, 1e-3)))


def color_by_elevation(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        norm = np.zeros_like(z, dtype=np.float64)
    else:
        norm = (z - z_min) / (z_max - z_min)
    # Simple perceptual-ish gradient: blue -> cyan -> yellow -> red
    # Construct with piecewise linear ramps
    r = np.clip(1.5 * norm - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * norm - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - norm) - 0.25, 0.0, 1.0)
    colors = np.stack([r, g, b], axis=1).astype(np.float32)
    return colors


def build_point_cloud(
    xyz: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return pcd


def has_rgb(las: laspy.LasData) -> bool:
    return all(hasattr(las, attr) for attr in ("red", "green", "blue"))


def to_rgb_colors_from_las(las: laspy.LasData) -> Optional[np.ndarray]:
    if not has_rgb(las):
        return None
    # LAS RGB is typically 16-bit; normalize to [0,1]
    r = np.asarray(las.red, dtype=np.float32)
    g = np.asarray(las.green, dtype=np.float32)
    b = np.asarray(las.blue, dtype=np.float32)
    max_val = 65535.0 if np.max([r.max(initial=0), g.max(initial=0), b.max(initial=0)]) > 255 else 255.0
    colors = np.stack([r, g, b], axis=1) / max_val
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    return colors


def load_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)
    return xyz, las


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open3D LIDAR Viewer with safety checks and optional downsampling."
    )
    parser.add_argument("path", type=str, nargs="?", help="Path to .laz/.las file. If omitted, pick one from CWD.")
    parser.add_argument(
        "--voxel-size",
        type=str,
        default="auto",
        help='Voxel size for downsampling in same units as data. Use "auto" for automatic selection.',
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=3_000_000,
        help="When voxel-size=auto, target number of points after downsampling.",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["elevation", "rgb", "none"],
        default="elevation",
        help="Coloring strategy for the point cloud.",
    )
    parser.add_argument(
        "--warn-points",
        type=int,
        default=20_000_000,
        help="Warn when file has more than this many points.",
    )
    parser.add_argument(
        "--block-points",
        type=int,
        default=50_000_000,
        help="Block (unless --force) when file has more than this many points.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override safety block on large files.",
    )
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        # Auto-pick from current directory: prefer .laz, then .las
        candidates = sorted(Path(".").glob("*.laz")) + sorted(Path(".").glob("*.las"))
        if not candidates:
            print("Error: No .laz or .las files found in current directory.", file=sys.stderr)
            sys.exit(2)
        path = candidates[0]
        print(f"No path provided. Using: {path}")
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(2)

    num_points, mins, maxs = read_header_info(path)
    include_rgb_estimate = False  # elevation color by default
    est_bytes = estimate_memory_bytes(num_points, include_rgb=include_rgb_estimate)

    print(f"File: {path}")
    print(f"Points: {num_points:,}")
    print(f"Bounds (min): {mins}")
    print(f"Bounds (max): {maxs}")
    print(f"Estimated memory (viewer): {human_bytes(est_bytes)}")

    if num_points >= args.block_points and not args.force:
        print(
            f"Safety block: {num_points:,} points >= {args.block_points:,}. "
            f"Use --force or downsample with --voxel-size/auto.",
            file=sys.stderr,
        )
        sys.exit(1)
    if num_points >= args.warn_points and not args.force:
        print(
            f"Warning: Large file ({num_points:,} points). Consider downsampling.",
            file=sys.stderr,
        )

    # Load points
    print("Loading points...")
    xyz, las = load_xyz(path)

    # Determine voxel size
    voxel_size_value: Optional[float]
    if args.voxel_size.lower() == "auto":
        voxel_size_value = compute_auto_voxel_size_from_bounds(
            mins, maxs, num_points, args.target_points
        )
        print(f'Auto voxel size: {voxel_size_value:.6f} (target ~{args.target_points:,} points)')
    else:
        try:
            voxel_size_value = float(args.voxel_size)
        except ValueError:
            print(f'Invalid --voxel-size "{args.voxel_size}". Use a float or "auto".', file=sys.stderr)
            sys.exit(2)
        if voxel_size_value <= 0:
            voxel_size_value = None

    # Colors
    colors: Optional[np.ndarray] = None
    color_mode: Literal["elevation", "rgb", "none"] = args.color  # type: ignore
    if color_mode == "rgb":
        colors = to_rgb_colors_from_las(las)
        if colors is None:
            print("RGB not present; falling back to elevation color.")
            color_mode = "elevation"
    if color_mode == "elevation":
        colors = color_by_elevation(xyz[:, 2])
    elif color_mode == "none":
        colors = None

    # Build point cloud
    pcd = build_point_cloud(xyz, colors)

    # Downsample if requested
    if voxel_size_value and voxel_size_value > 0:
        print(f"Downsampling with voxel size {voxel_size_value:.6f} ...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_value)
        print(f"Points after downsampling: {np.asarray(pcd.points).shape[0]:,}")

    # Visualize
    print("Opening viewer...")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()


