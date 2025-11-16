#!/usr/bin/env python3
"""
Load a LAZ file and display its metadata.
"""

import argparse
import sys
from pathlib import Path

import laspy
from pyproj import Transformer


def display_metadata(laz_file: Path) -> None:
    """
    Load a LAZ file and display its metadata.
    
    Args:
        laz_file: Path to the LAZ file
    """
    if not laz_file.exists():
        print(f"Error: File not found: {laz_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading LAZ file: {laz_file}")
    print("=" * 80)
    
    try:
        las = laspy.read(laz_file)
        
        # Header information
        print("\n📋 HEADER INFORMATION")
        print("-" * 80)
        print(f"File Source ID: {las.header.file_source_id}")
        print(f"Version: {las.header.version}")
        print(f"Major Version: {las.header.major_version}")
        print(f"Minor Version: {las.header.minor_version}")
        print(f"System Identifier: {las.header.system_identifier}")
        print(f"Generating Software: {las.header.generating_software}")
        if hasattr(las.header, 'creation_date') and las.header.creation_date:
            print(f"Creation Date: {las.header.creation_date}")
        
        # Point count and format
        print(f"\n📊 POINT DATA")
        print("-" * 80)
        print(f"Number of Points: {las.header.point_count:,}")
        print(f"Point Format: {las.header.point_format}")
        print(f"Point Format ID: {las.header.point_format.id}")
        if hasattr(las.header.point_format, 'description'):
            print(f"Point Format Description: {las.header.point_format.description}")
        if hasattr(las.header.point_format, 'size'):
            print(f"Point Size: {las.header.point_format.size} bytes")
        
        # Bounding box and center (projected)
        print(f"\n🗺️  SPATIAL EXTENT")
        print("-" * 80)
        print(f"X Min: {las.header.x_min:.3f}")
        print(f"X Max: {las.header.x_max:.3f}")
        print(f"Y Min: {las.header.y_min:.3f}")
        print(f"Y Max: {las.header.y_max:.3f}")
        print(f"Z Min: {las.header.z_min:.3f}")
        print(f"Z Max: {las.header.z_max:.3f}")
        print(f"X Scale: {las.header.x_scale}")
        print(f"Y Scale: {las.header.y_scale}")
        print(f"Z Scale: {las.header.z_scale}")
        print(f"X Offset: {las.header.x_offset}")
        print(f"Y Offset: {las.header.y_offset}")
        print(f"Z Offset: {las.header.z_offset}")

        x_center = (las.header.x_min + las.header.x_max) / 2.0
        y_center = (las.header.y_min + las.header.y_max) / 2.0
        print(f"\nTile center (projected): X={x_center:.3f}, Y={y_center:.3f}")

        # Center in lat/long, if CRS is available
        try:
            crs = las.header.parse_crs()
        except Exception:
            crs = None

        if crs is not None:
            # Transform center to WGS84 (EPSG:4326) lat/long
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x_center, y_center)
            print(f"Tile center (geographic, WGS84): lat={lat:.8f}, lon={lon:.8f}")
            maps_url = f"https://www.google.com/maps?q={lat:.8f},{lon:.8f}"
            print(f"Google Maps link: {maps_url}")
        else:
            print("Tile center (geographic): CRS not found in header, cannot compute lat/long")
        
        # Available dimensions
        print(f"\n📐 AVAILABLE DIMENSIONS")
        print("-" * 80)
        if hasattr(las.point_format, 'dimension_names'):
            for dim_name in sorted(las.point_format.dimension_names):
                try:
                    dim = las.point_format.dimension_by_name(dim_name)
                    dim_type = getattr(dim, 'type', 'unknown')
                    dim_size = getattr(dim, 'size', 'unknown')
                    print(f"  {dim_name}: {dim_type} (size: {dim_size} bytes)")
                except Exception:
                    print(f"  {dim_name}: (available)")
        
        # Sample point data
        print(f"\n🔍 SAMPLE POINT DATA (first 5 points)")
        print("-" * 80)
        for i in range(min(5, len(las.points))):
            print(f"\nPoint {i}:")
            print(f"  X: {las.x[i]:.3f}")
            print(f"  Y: {las.y[i]:.3f}")
            print(f"  Z: {las.z[i]:.3f}")
            if hasattr(las, 'intensity'):
                print(f"  Intensity: {las.intensity[i]}")
            if hasattr(las, 'return_number'):
                print(f"  Return Number: {las.return_number[i]}")
            if hasattr(las, 'number_of_returns'):
                print(f"  Number of Returns: {las.number_of_returns[i]}")
            if hasattr(las, 'classification'):
                print(f"  Classification: {las.classification[i]}")
            if hasattr(las, 'gps_time'):
                print(f"  GPS Time: {las.gps_time[i]}")
        
        # VLRs (Variable Length Records)
        if las.header.vlrs:
            print(f"\n📝 VARIABLE LENGTH RECORDS (VLRs)")
            print("-" * 80)
            print(f"Number of VLRs: {len(las.header.vlrs)}")
            for i, vlr in enumerate(las.header.vlrs[:10]):  # Show first 10
                print(f"\n  VLR {i}:")
                print(f"    Record ID: {vlr.record_id}")
                print(f"    User ID: {vlr.user_id}")
                print(f"    Description: {vlr.description}")
                data_length = getattr(vlr, 'string', getattr(vlr, 'record_length_after_header', 'unknown'))
                print(f"    Data Length: {data_length} bytes")
            if len(las.header.vlrs) > 10:
                print(f"\n  ... and {len(las.header.vlrs) - 10} more VLRs")
        
        # EVLRs (Extended Variable Length Records)
        if hasattr(las.header, 'evlrs') and las.header.evlrs:
            print(f"\n📝 EXTENDED VARIABLE LENGTH RECORDS (EVLRs)")
            print("-" * 80)
            print(f"Number of EVLRs: {len(las.header.evlrs)}")
            for i, evlr in enumerate(las.header.evlrs[:10]):  # Show first 10
                print(f"\n  EVLR {i}:")
                print(f"    Record ID: {evlr.record_id}")
                print(f"    User ID: {evlr.user_id}")
                print(f"    Description: {evlr.description}")
                data_length = getattr(evlr, 'string', getattr(evlr, 'record_length_after_header', 'unknown'))
                print(f"    Data Length: {data_length} bytes")
            if len(las.header.evlrs) > 10:
                print(f"\n  ... and {len(las.header.evlrs) - 10} more EVLRs")
        
        print("\n" + "=" * 80)
        print("✅ Metadata display complete!")
        
    except Exception as e:
        print(f"Error loading LAZ file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Load a LAZ file and display its metadata"
    )
    parser.add_argument(
        'file',
        type=Path,
        nargs='?',
        help='Path to the LAZ file (default: first .laz file in current directory)'
    )
    
    args = parser.parse_args()
    # If no file provided, pick the first .laz file in the current directory
    if args.file is None:
        laz_files = sorted(Path.cwd().glob('*.laz'))
        if not laz_files:
            print("Error: No .laz files found in the current directory.", file=sys.stderr)
            sys.exit(1)
        args.file = laz_files[0]
    display_metadata(args.file)


if __name__ == '__main__':
    main()

