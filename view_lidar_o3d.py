#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import laspy
import numpy as np
import open3d as o3d
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
try:
    import rasterio
    import rasterio.transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


CLASSIFICATION_NAMES = {
    0: "Created, never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    8: "Model Key-point",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Overlap Points",
    13: "Wire – Guard",
    14: "Wire – Conductor",
    15: "Transmission Tower",
    16: "Wire – Connector/Insulator",
    17: "Bridge Deck",
    18: "High Noise",
}


def classification_label(code: int) -> str:
    name = CLASSIFICATION_NAMES.get(int(code), "Unknown")
    return f"{int(code)} ({name})"


def classification_list_label(codes) -> str:
    return ", ".join(classification_label(int(c)) for c in codes)


def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"




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
    try:
        dims = set(getattr(las.point_format, "dimension_names", []))
    except Exception:
        return False
    return {"red", "green", "blue"}.issubset(dims)


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


def load_xyz(path: Path) -> Tuple[np.ndarray, Optional[laspy.LasData]]:
    """Load point cloud from .laz/.las file."""
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)
    return xyz, las


def load_geotiff(path: Path, downsample_factor: int = 1) -> Tuple[np.ndarray, None]:
    """Load GeoTIFF and convert to point cloud.
    
    Args:
        path: Path to GeoTIFF file
        downsample_factor: Factor to downsample the raster (1 = no downsampling, 2 = every 2nd pixel, etc.)
    
    Returns:
        Tuple of (xyz points array, None) - second element is None since GeoTIFF doesn't have LAS metadata
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required to load GeoTIFF files. Install with: pip install rasterio")
    
    with rasterio.open(path) as src:
        # Read the raster data
        data = src.read(1)  # Read first band (typically elevation/DEM)
        
        # Get geotransform to convert pixel coordinates to world coordinates
        transform = src.transform
        
        # Downsample if requested
        if downsample_factor > 1:
            data = data[::downsample_factor, ::downsample_factor]
            # Adjust transform for downsampling
            transform = rasterio.Affine(
                transform.a * downsample_factor, transform.b, transform.c,
                transform.d, transform.e * downsample_factor, transform.f
            )
        
        # Get valid pixels (non-NaN, non-NoData)
        valid_mask = ~np.isnan(data)
        if src.nodata is not None:
            valid_mask = valid_mask & (data != src.nodata)
        
        # Get pixel coordinates for valid pixels
        rows, cols = np.where(valid_mask)
        
        # Convert pixel coordinates to world coordinates (vectorized)
        # Use transform to convert row/col to x/y coordinates
        x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)
        z_coords = data[rows, cols]
        
        # Stack into xyz array
        xyz = np.stack([
            np.array(x_coords, dtype=np.float32),
            np.array(y_coords, dtype=np.float32),
            np.array(z_coords, dtype=np.float32)
        ], axis=1)
        
        return xyz, None


def read_header_info(path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    """Read header information from LAS/LAZ or GeoTIFF file."""
    suffix = path.suffix.lower()
    
    if suffix in ['.laz', '.las']:
        with laspy.open(path) as laz:
            point_count = laz.header.point_count
            # mins/maxs are scaled values
            mins = np.asarray(laz.header.mins, dtype=np.float64)
            maxs = np.asarray(laz.header.maxs, dtype=np.float64)
        return int(point_count), mins, maxs
    elif suffix in ['.tif', '.tiff']:
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required to read GeoTIFF files. Install with: pip install rasterio")
        
        with rasterio.open(path) as src:
            data = src.read(1)
            valid_mask = ~np.isnan(data)
            if src.nodata is not None:
                valid_mask = valid_mask & (data != src.nodata)
            
            point_count = int(np.sum(valid_mask))
            
            # Get bounds from geotransform
            bounds = src.bounds
            valid_data = data[valid_mask]
            mins = np.array([bounds.left, bounds.bottom, float(np.nanmin(valid_data))], dtype=np.float64)
            maxs = np.array([bounds.right, bounds.top, float(np.nanmax(valid_data))], dtype=np.float64)
        
        return point_count, mins, maxs
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .laz, .las, .tif, .tiff")


def visualize_with_controller(
    pcd: o3d.geometry.PointCloud,
    start_pos: Optional[Tuple[float, float, float]] = None,
    start_forward: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Visualize point cloud with PS5 controller support - left stick for first-person look.
    
    Args:
        pcd: Point cloud to visualize
        start_pos: Starting camera position (x, y, z). Default: (53160.87, 27147.06, 33.72)
        start_forward: Starting forward direction vector (x, y, z). Default: (0.00, 1.00, 0.0105)
    """
    # Initialize pygame for joystick support
    pygame.init()
    pygame.joystick.init()
    
    # Try to find a connected controller
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Controller detected: {joystick.get_name()}")
        print("Controls:")
        print("  Left stick: Move forward/back/strafe (FPS-style)")
    else:
        print("No controller detected. Using keyboard/mouse controls.")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("LiDAR Viewer (PS5 Controller)")
    vis.add_geometry(pcd)
    
    # Get render options and camera
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    ctr = vis.get_view_control()
    
    # Set initial camera position and orientation
    if start_pos is None:
        start_pos = (53160.87, 27147.06, 33.72)  # Default position
    if start_forward is None:
        start_forward = (0.00, 1.00, 0.0105)  # Default forward direction
    
    # Normalize forward vector
    start_forward = np.array(start_forward, dtype=np.float64)
    start_forward = start_forward / np.linalg.norm(start_forward)
    
    # Calculate right and up vectors from forward
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(start_forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # If forward is parallel to world_up, use a different right vector
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, start_forward)
    up = up / np.linalg.norm(up)
    
    # Construct rotation matrix
    # In Open3D camera space: +X is right, +Y is up, -Z is forward
    initial_rotation = np.column_stack([right, up, -start_forward])
    
    # Construct extrinsic matrix (world-to-camera)
    initial_translation = -initial_rotation @ np.array(start_pos)
    
    initial_extrinsic = np.eye(4)
    initial_extrinsic[:3, :3] = initial_rotation
    initial_extrinsic[:3, 3] = initial_translation
    
    # Set initial camera parameters
    param = ctr.convert_to_pinhole_camera_parameters()
    param.extrinsic = initial_extrinsic
    ctr.convert_from_pinhole_camera_parameters(param)
    
    # Verify initial forward vector
    verify_param = ctr.convert_to_pinhole_camera_parameters()
    verify_rotation = verify_param.extrinsic[:3, :3]
    verify_forward = -verify_rotation[:, 2]
    verify_forward_xy = np.array([verify_forward[0], verify_forward[1]])
    verify_forward_xy = verify_forward_xy / np.linalg.norm(verify_forward_xy)
    verify_heading = np.degrees(np.arctan2(verify_forward_xy[0], verify_forward_xy[1]))
    if verify_heading < 0:
        verify_heading += 360.0
    print(f"DEBUG: Initial forward vector: {verify_forward}")
    print(f"DEBUG: Initial heading: {verify_heading:.1f}°")
    
    # Camera control parameters
    move_speed = 3.0  # Movement speed for FPS-style movement (reduced for smoother movement)
    
    # Dead zone for analog stick (to prevent drift)
    dead_zone = 0.15
    
    print(f"Viewer ready. Starting at {start_pos} looking {start_forward}")
    print("Use left stick to move. Close window to exit.")
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    # Track camera position for display
    last_camera_pos = None
    
    while running:
        # Poll pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update controller state
        if joystick:
            pygame.event.pump()  # Update joystick state
            
            # Get current camera parameters
            param = ctr.convert_to_pinhole_camera_parameters()
            extrinsic = param.extrinsic
            rotation_matrix = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            camera_pos = -rotation_matrix.T @ translation
            
            # Track if we need to update camera
            camera_updated = False
            
            # Left stick (axes 0, 1) - FPS-style movement
            if joystick.get_numaxes() > 1:
                left_x = joystick.get_axis(0)  # Left stick X (strafe left/right)
                left_y = joystick.get_axis(1)  # Left stick Y (move forward/back)
                
                if abs(left_x) > dead_zone or abs(left_y) > dead_zone:
                    # Get camera's forward and right vectors from current rotation
                    # In Open3D camera space: +X is right, +Y is up, -Z is forward
                    forward = -rotation_matrix[:, 2]  # Negative Z in camera space = forward (where camera is looking)
                    right = rotation_matrix[:, 0]     # Positive X in camera space = right
                    
                    # Calculate movement vector relative to camera's current orientation
                    # left_y: forward (positive) / back (negative) - relative to where camera is looking
                    # left_x: strafe right (positive) / strafe left (negative) - relative to camera's right
                    move_vector = (forward * left_y + right * left_x) * move_speed
                    
                    # Update camera position
                    camera_pos = camera_pos + move_vector
                    camera_updated = True
            
            # Apply camera updates if any
            if camera_updated:
                # Reconstruct extrinsic matrix (world-to-camera)
                # New translation = -R * camera_pos
                new_translation = -rotation_matrix @ camera_pos
                
                new_extrinsic = np.eye(4)
                new_extrinsic[:3, :3] = rotation_matrix
                new_extrinsic[:3, 3] = new_translation
                param.extrinsic = new_extrinsic
                
                # Apply updated camera parameters
                ctr.convert_from_pinhole_camera_parameters(param)
                
                # Store camera position for display
                last_camera_pos = camera_pos
        
        # Display camera position and orientation (update every frame)
        # Always get current camera state for display
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        display_rotation_matrix = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        display_camera_pos = -display_rotation_matrix.T @ translation
        
        # Calculate forward, right, and up vectors for orientation display
        # In Open3D, the rotation matrix R in the extrinsic matrix transforms world to camera coordinates
        # The extrinsic matrix is [R | t] where R is 3x3 rotation, t is 3x1 translation
        # The columns of R are the camera axes expressed in world coordinates:
        # Column 0: camera X axis (right) in world coords
        # Column 1: camera Y axis (up) in world coords  
        # Column 2: camera Z axis in world coords
        # In Open3D camera space: +X is right, +Y is up, -Z is forward (looking into screen)
        # So forward direction in world coordinates = -R[:, 2]
        # But wait - R transforms world->camera, so R^T transforms camera->world
        # Camera forward in camera space is (0, 0, -1)
        # Forward in world = R^T @ (0, 0, -1) = -R^T[:, 2] = -R[:, 2] (since R^T[:, 2] = R[2, :]^T, but we want column)
        # Actually, let me verify: if p_camera = R @ p_world, then p_world = R^T @ p_camera
        # Camera forward vector in camera space: v_camera = (0, 0, -1)
        # Forward in world: v_world = R^T @ v_camera = R^T @ (0, 0, -1) = -R^T[:, 2]
        # But R^T[:, 2] is the third row of R, not the third column
        # Actually, R^T[:, 2] means the 2nd column of R^T, which is the 2nd row of R
        # We want the 2nd column of R, which is R[:, 2]
        # So forward = -R[:, 2] should be correct
        
        right = display_rotation_matrix[:, 0]     # Camera X axis (right) in world coords
        up = display_rotation_matrix[:, 1]        # Camera Y axis (up) in world coords
        camera_z = display_rotation_matrix[:, 2]  # Camera Z axis in world coords
        forward = -camera_z  # Forward is negative Z in camera space
        
        # Calculate compass heading in X/Y plane (ignoring Z)
        # Standard coordinate system: X = East, Y = North, Z = Up
        # Heading: 0° = North (+Y), 90° = East (+X), 180° = South (-Y), 270° = West (-X)
        # Project forward onto X/Y plane and calculate angle
        forward_xy = np.array([forward[0], forward[1]])
        forward_xy_norm = np.linalg.norm(forward_xy)
        if forward_xy_norm > 1e-6:
            forward_xy = forward_xy / forward_xy_norm
            # atan2(X, Y) gives angle from positive Y axis (North)
            # atan2(0, 1) = 0° (North)
            # atan2(1, 0) = 90° (East)  
            # atan2(0, -1) = 180° (South)
            # atan2(-1, 0) = -90° = 270° (West)
            heading_rad = np.arctan2(forward_xy[0], forward_xy[1])
            heading_deg = np.degrees(heading_rad)
            if heading_deg < 0:
                heading_deg += 360.0
        else:
            # Forward is vertical (parallel to Z), heading undefined
            heading_deg = float('nan')
        
        # Print camera position and orientation to console
        if np.isnan(heading_deg):
            heading_str = "N/A"
        else:
            heading_str = f"{heading_deg:.1f}°"
        # Show forward vector with more precision to debug
        print(f"Pos: X={display_camera_pos[0]:.2f}, Y={display_camera_pos[1]:.2f}, Z={display_camera_pos[2]:.2f} | "
              f"Forward: X={forward[0]:.4f}, Y={forward[1]:.4f}, Z={forward[2]:.4f} | "
              f"Heading: {heading_str}", end='\r')
        
        # Update visualizer
        if not vis.poll_events():
            running = False
        vis.update_renderer()
        
        # Cap frame rate
        clock.tick(60)
    
    # Cleanup
    vis.destroy_window()
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open3D LIDAR Viewer with safety checks and optional downsampling."
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        help="Path to .laz/.las/.tif/.tiff file. If omitted, pick one from CWD.",
    )
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
        "--class-filter",
        type=int,
        nargs="+",
        metavar="CLASS",
        help="Keep only points whose classification code is in this list (e.g. --class-filter 2 5).",
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
    parser.add_argument(
        "--start-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[53160.87, 27147.06, 33.72],
        help="Starting camera position (x, y, z). Default: 53160.87 27147.06 33.72",
    )
    parser.add_argument(
        "--start-forward",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.00, 1.00, 0.0105],
        help="Starting camera forward direction vector (x, y, z). Default: 0.00 1.00 0.0105",
    )
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        # Auto-pick from current directory: prefer .laz, then .las, then .tif/.tiff
        candidates = (sorted(Path(".").glob("*.laz")) + 
                     sorted(Path(".").glob("*.las")) +
                     sorted(Path(".").glob("*.tif")) +
                     sorted(Path(".").glob("*.tiff")))
        if not candidates:
            print("Error: No .laz, .las, .tif, or .tiff files found in current directory.", file=sys.stderr)
            sys.exit(2)
        path = candidates[0]
        print(f"No path provided. Using: {path}")
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    
    # Determine file type
    suffix = path.suffix.lower()
    is_geotiff = suffix in ['.tif', '.tiff']

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
    if is_geotiff:
        # For GeoTIFF, we might want to downsample large rasters
        # Calculate a reasonable downsample factor if the raster is very large
        downsample_factor = 1
        if num_points > args.target_points * 10:
            # If we have way more points than target, downsample
            downsample_factor = max(1, int(np.sqrt(num_points / args.target_points)))
            print(f"Large GeoTIFF detected. Downsampling by factor {downsample_factor}...")
        xyz, las = load_geotiff(path, downsample_factor=downsample_factor)
    else:
        xyz, las = load_xyz(path)

    # Optional classification filtering (only for LAS/LAZ)
    class_mask: Optional[np.ndarray] = None
    if args.class_filter is not None:
        if las is None:
            print("Warning: --class-filter only works with LAS/LAZ files, ignoring.", file=sys.stderr)
        else:
            try:
                cls = np.asarray(las.classification)
                class_mask = np.isin(cls, np.asarray(args.class_filter, dtype=cls.dtype))
                if not np.any(class_mask):
                    print(
                        f"No points found with classification codes: {classification_list_label(args.class_filter)}.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                before = xyz.shape[0]
                xyz = xyz[class_mask]
                print(
                    f"Applied classification filter {classification_list_label(args.class_filter)}: "
                    f"{xyz.shape[0]:,} / {before:,} points kept."
                )
            except AttributeError:
                print(
                    "Classification filter requested, but this file has no "
                    "'classification' dimension. Showing all points.",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"Error applying classification filter {args.class_filter}: {e}",
                    file=sys.stderr,
                )

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
        if las is None:
            # GeoTIFF doesn't have RGB, use elevation
            print("Warning: RGB coloring only available for LAS/LAZ files, using elevation instead.", file=sys.stderr)
            color_mode = "elevation"
        else:
            base_colors = to_rgb_colors_from_las(las)
            if base_colors is None:
                print("RGB not present; falling back to elevation color.")
                color_mode = "elevation"
            else:
                if class_mask is not None:
                    base_colors = base_colors[class_mask]
                colors = base_colors
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
    if PYGAME_AVAILABLE:
        start_pos = tuple(args.start_pos)
        start_forward = tuple(args.start_forward)
        visualize_with_controller(pcd, start_pos=start_pos, start_forward=start_forward)
    else:
        print("Note: pygame not available, using standard viewer. Install pygame for PS5 controller support.")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()


