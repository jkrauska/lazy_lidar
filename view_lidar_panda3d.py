#!/usr/bin/env python3
"""Panda3D-based LiDAR viewer with PS5 controller support."""
import argparse
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import laspy
import numpy as np
try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        Point3, Vec3, NodePath, Geom, GeomNode, GeomVertexData, GeomVertexFormat,
        GeomVertexWriter, GeomPoints, TransparencyAttrib, ColorBlendAttrib,
        AmbientLight, DirectionalLight, TextNode
    )
    from direct.task import Task
    PAND3D_AVAILABLE = True
except ImportError:
    PAND3D_AVAILABLE = False

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


def load_xyz(path: Path) -> Tuple[np.ndarray, Optional[laspy.LasData]]:
    """Load point cloud from .laz/.las file."""
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)
    return xyz, las


def load_geotiff(path: Path, downsample_factor: int = 1) -> Tuple[np.ndarray, None]:
    """Load GeoTIFF and convert to point cloud."""
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required to load GeoTIFF files. Install with: pip install rasterio")
    
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        
        if downsample_factor > 1:
            data = data[::downsample_factor, ::downsample_factor]
            transform = rasterio.Affine(
                transform.a * downsample_factor, transform.b, transform.c,
                transform.d, transform.e * downsample_factor, transform.f
            )
        
        valid_mask = ~np.isnan(data)
        if src.nodata is not None:
            valid_mask = valid_mask & (data != src.nodata)
        
        rows, cols = np.where(valid_mask)
        x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)
        z_coords = data[rows, cols]
        
        xyz = np.stack([
            np.array(x_coords, dtype=np.float32),
            np.array(y_coords, dtype=np.float32),
            np.array(z_coords, dtype=np.float32)
        ], axis=1)
        
        return xyz, None


def color_by_elevation(z: np.ndarray) -> np.ndarray:
    """Generate colors based on elevation."""
    z = np.asarray(z, dtype=np.float64)
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        norm = np.zeros_like(z, dtype=np.float64)
    else:
        norm = (z - z_min) / (z_max - z_min)
    
    r = np.clip(1.5 * norm - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * norm - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - norm) - 0.25, 0.0, 1.0)
    colors = np.stack([r, g, b], axis=1).astype(np.float32)
    return colors


def to_rgb_colors_from_las(las: laspy.LasData) -> Optional[np.ndarray]:
    """Extract RGB colors from LAS file if available."""
    if not (hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")):
        return None
    
    r = np.asarray(las.red, dtype=np.float32)
    g = np.asarray(las.green, dtype=np.float32)
    b = np.asarray(las.blue, dtype=np.float32)
    max_val = 65535.0 if np.max([r.max(initial=0), g.max(initial=0), b.max(initial=0)]) > 255 else 255.0
    colors = np.stack([r, g, b], axis=1) / max_val
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    return colors


class LidarViewer(ShowBase):
    """Panda3D-based LiDAR viewer with controller support."""
    
    def __init__(
        self,
        xyz: np.ndarray,
        colors: Optional[np.ndarray] = None,
        start_pos: Optional[Tuple[float, float, float]] = None,
        start_forward: Optional[Tuple[float, float, float]] = None,
    ):
        ShowBase.__init__(self)
        
        # Panda3D coordinate system: X=right, Y=forward, Z=up
        # Convert from standard (X=East, Y=North, Z=Up) to Panda3D (X=East, Y=North, Z=Up)
        # Actually, we can keep the same coordinate system
        
        # Initialize controller
        self.joystick = None
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Controller detected: {self.joystick.get_name()}")
                print("Controls:")
                print("  Left stick: Move forward/back/strafe (FPS-style)")
        
        # Camera parameters
        self.move_speed = 3.0
        self.dead_zone = 0.15
        
        # Set initial camera position
        if start_pos is None:
            start_pos = (53160.87, 27147.06, 33.72)
        if start_forward is None:
            start_forward = (0.00, 1.00, 0.0105)
        
        # Normalize forward vector
        start_forward = np.array(start_forward, dtype=np.float64)
        start_forward = start_forward / np.linalg.norm(start_forward)
        
        # Set camera position and look at
        camera_pos = Point3(start_pos[0], start_pos[1], start_pos[2])
        look_at = Point3(
            start_pos[0] + start_forward[0] * 100,
            start_pos[1] + start_forward[1] * 100,
            start_pos[2] + start_forward[2] * 100
        )
        self.camera.setPos(camera_pos)
        self.camera.lookAt(look_at)
        
        # Create point cloud geometry
        self.create_point_cloud(xyz, colors)
        
        # Setup lighting
        self.setup_lighting()
        
        # Setup controls
        self.setup_controls()
        
        # Display info
        self.setup_info_display()
        
        print(f"Viewer ready. Starting at {start_pos} looking {start_forward}")
        print("Use left stick to move. Close window to exit.")
    
    def create_point_cloud(self, xyz: np.ndarray, colors: Optional[np.ndarray] = None):
        """Create Panda3D point cloud geometry."""
        num_points = len(xyz)
        
        # Create vertex format
        vformat = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('pointcloud', vformat, Geom.UHStatic)
        vdata.setNumRows(num_points)
        
        # Write vertices
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        
        if colors is None:
            colors = color_by_elevation(xyz[:, 2])
        
        for i in range(num_points):
            # Panda3D: X=right, Y=forward, Z=up
            # Our data: X=East, Y=North, Z=Up
            # They match! So we can use directly
            vertex.addData3f(xyz[i, 0], xyz[i, 1], xyz[i, 2])
            color.addData4f(colors[i, 0], colors[i, 1], colors[i, 2], 1.0)
        
        # Create points primitive
        prim = GeomPoints(Geom.UHStatic)
        prim.addConsecutiveVertices(0, num_points)
        prim.closePrimitive()
        
        # Create geometry
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        # Create node
        node = GeomNode('pointcloud')
        node.addGeom(geom)
        
        # Add to scene
        pointcloud_np = self.render.attachNewNode(node)
        pointcloud_np.setTwoSided(True)
        pointcloud_np.setTransparency(TransparencyAttrib.MNone)
        pointcloud_np.setRenderModeWireframe(False)
        pointcloud_np.setPointSize(1.0)
        
        self.pointcloud_np = pointcloud_np
    
    def setup_lighting(self):
        """Setup basic lighting."""
        ambient_light = AmbientLight('ambient')
        ambient_light.setColor((0.3, 0.3, 0.3, 1))
        ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_light_np)
        
        directional_light = DirectionalLight('directional')
        directional_light.setDirection(Vec3(-1, -1, -1))
        directional_light.setColor((0.7, 0.7, 0.7, 1))
        directional_light_np = self.render.attachNewNode(directional_light)
        self.render.setLight(directional_light_np)
    
    def setup_controls(self):
        """Setup keyboard and controller controls."""
        self.accept('escape', sys.exit)
        self.accept('q', sys.exit)
        
        # Enable mouse look for camera rotation
        # Mouse controls are handled by Panda3D's default camera controls
        
        # Task for controller updates
        if self.joystick:
            self.taskMgr.add(self.update_controller, 'update_controller')
    
    def update_controller(self, task):
        """Update controller state and move camera."""
        if not self.joystick:
            return task.cont
        
        pygame.event.pump()
        
        # Left stick movement
        if self.joystick.get_numaxes() > 1:
            left_x = self.joystick.get_axis(0)  # Strafe left/right
            left_y = self.joystick.get_axis(1)  # Move forward/back
            
            if abs(left_x) > self.dead_zone or abs(left_y) > self.dead_zone:
                # Get camera's forward and right vectors
                # Panda3D camera: getQuat() gives orientation
                quat = self.camera.getQuat()
                
                # Forward vector in Panda3D: Y axis (0, 1, 0) rotated by quaternion
                forward = quat.getForward()
                right = quat.getRight()
                
                # Calculate movement
                move_vector = (forward * left_y + right * left_x) * self.move_speed
                
                # Update camera position
                current_pos = self.camera.getPos()
                new_pos = current_pos + move_vector
                self.camera.setPos(new_pos)
        
        return task.cont
    
    def setup_info_display(self):
        """Setup on-screen info display."""
        # This would display camera position, heading, etc.
        # For now, just print to console
        self.taskMgr.add(self.update_info_display, 'update_info_display')
    
    def update_info_display(self, task):
        """Update info display."""
        # Get camera position and forward vector
        pos = self.camera.getPos()
        quat = self.camera.getQuat()
        forward = quat.getForward()
        
        # Calculate heading (in X/Y plane)
        forward_xy = np.array([forward.x, forward.y])
        forward_xy_norm = np.linalg.norm(forward_xy)
        if forward_xy_norm > 1e-6:
            forward_xy = forward_xy / forward_xy_norm
            heading_rad = np.arctan2(forward_xy[0], forward_xy[1])
            heading_deg = np.degrees(heading_rad)
            if heading_deg < 0:
                heading_deg += 360.0
            heading_str = f"{heading_deg:.1f}°"
        else:
            heading_str = "N/A"
        
        # Print to console (could be displayed on screen)
        print(f"Pos: X={pos.x:.2f}, Y={pos.y:.2f}, Z={pos.z:.2f} | "
              f"Forward: X={forward.x:.4f}, Y={forward.y:.4f}, Z={forward.z:.4f} | "
              f"Heading: {heading_str}", end='\r')
        
        return task.cont


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panda3D LiDAR Viewer with PS5 controller support."
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        help="Path to .laz/.las/.tif/.tiff file. If omitted, pick one from CWD.",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["elevation", "rgb", "none"],
        default="elevation",
        help="Coloring strategy for the point cloud.",
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
    
    if not PAND3D_AVAILABLE:
        print("Error: panda3d is required. Install with: pip install panda3d", file=sys.stderr)
        sys.exit(1)
    
    # Find file
    if args.path:
        path = Path(args.path)
    else:
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
    
    # Determine file type and load
    suffix = path.suffix.lower()
    is_geotiff = suffix in ['.tif', '.tiff']
    
    print(f"Loading {path}...")
    if is_geotiff:
        xyz, las = load_geotiff(path)
    else:
        xyz, las = load_xyz(path)
    
    print(f"Loaded {len(xyz):,} points")
    
    # Load colors
    colors: Optional[np.ndarray] = None
    if args.color == "rgb":
        if las is None:
            print("Warning: RGB coloring only available for LAS/LAZ files, using elevation instead.", file=sys.stderr)
            colors = color_by_elevation(xyz[:, 2])
        else:
            colors = to_rgb_colors_from_las(las)
            if colors is None:
                print("Warning: No RGB data in file, using elevation instead.", file=sys.stderr)
                colors = color_by_elevation(xyz[:, 2])
    elif args.color == "elevation":
        colors = color_by_elevation(xyz[:, 2])
    # else: colors = None
    
    # Create and run viewer
    start_pos = tuple(args.start_pos)
    start_forward = tuple(args.start_forward)
    app = LidarViewer(xyz, colors, start_pos=start_pos, start_forward=start_forward)
    app.run()


if __name__ == "__main__":
    main()

