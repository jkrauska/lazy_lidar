#!/usr/bin/env python3
"""
Lidar downloader that reads a USGS directory listing and downloads a single LAZ file.
"""

import argparse
import random
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Hardcoded USGS LPC directory URL
USGS_URL = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/CA_SanFrancisco_B23/CA_SanFrancisco_1_B23/LAZ/"


def get_laz_files(url: str) -> list[str]:
    """
    Parse the directory listing page and extract all LAZ file URLs.
    
    Args:
        url: The URL of the directory listing page
        
    Returns:
        List of LAZ file URLs (relative or absolute)
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching page: {e}", file=sys.stderr)
        sys.exit(1)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    laz_files = []
    
    # Find all links in the page
    for link in soup.find_all('a'):
        href = link.get('href', '')
        # Check if it's a LAZ file (ends with .laz)
        if href.endswith('.laz'):
            # Decode URL-encoded filenames
            decoded_href = href.replace('%5F', '_').replace('%20', ' ')
            laz_files.append(decoded_href)
    
    return laz_files


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL to the specified path.
    
    Args:
        url: The URL of the file to download
        output_path: Path where the file should be saved
        chunk_size: Size of chunks to read/write
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print()  # New line after progress
        print(f"Downloaded: {output_path}")
        
    except requests.RequestException as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        if output_path.exists():
            output_path.unlink()  # Remove partial file
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download a single LAZ file from a USGS directory listing"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory (default: current directory)',
        default=Path.cwd()
    )
    parser.add_argument(
        '-f', '--file',
        help='Specific LAZ file to download (by name or index). If not specified, downloads a random file.'
    )
    parser.add_argument(
        '-i', '--index',
        type=int,
        help='Download file by index (0-based)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available LAZ files and exit'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Get list of LAZ files
    print(f"Fetching directory listing from {USGS_URL}...")
    laz_files = get_laz_files(USGS_URL)
    
    if not laz_files:
        print("No LAZ files found on the page.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(laz_files)} LAZ file(s)")
    
    # List files if requested
    if args.list:
        print("\nAvailable LAZ files:")
        for i, filename in enumerate(laz_files):
            print(f"  [{i}] {filename}")
        return
    
    # Determine which file to download
    if args.file:
        # Try to find by name
        matching_files = [f for f in laz_files if args.file in f]
        if not matching_files:
            print(f"Error: No file matching '{args.file}' found.", file=sys.stderr)
            sys.exit(1)
        elif len(matching_files) > 1:
            print(f"Warning: Multiple files match '{args.file}'. Using first match.")
        selected_file = matching_files[0]
    elif args.index is not None:
        if args.index < 0 or args.index >= len(laz_files):
            print(f"Error: Index {args.index} is out of range (0-{len(laz_files)-1})", file=sys.stderr)
            sys.exit(1)
        selected_file = laz_files[args.index]
    else:
        # Download random file by default
        selected_file = random.choice(laz_files)
    
    # Build full URL
    base_url = USGS_URL.rstrip('/') + '/'
    file_url = urljoin(base_url, selected_file)
    
    # Determine output filename
    output_filename = Path(selected_file).name
    output_path = args.output / output_filename
    
    # Check if file already exists
    if output_path.exists():
        print(f"File already exists: {output_path}")
        print(f"Skipping download.")
        return
    
    print(f"Downloading: {selected_file}")
    print(f"URL: {file_url}")
    print(f"Output: {output_path}")
    
    # Download the file
    download_file(file_url, output_path)


if __name__ == '__main__':
    main()

