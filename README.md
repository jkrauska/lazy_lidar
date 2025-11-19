# lazy_lidar
Lidar Data Walkthrough

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To get started:

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

This will create a virtual environment and install all project dependencies. You can then activate the environment with `uv sync` or run commands using `uv run <command>`.

## Getting LIDAR Data

### USGS LPC Data Downloader

The project includes a script to download LAZ files from the USGS San Francisco LPC directory (URL is hardcoded). By default, it randomly selects a file and only downloads if it doesn't already exist:

```bash
# Download a random LAZ file (skips if already exists)
uv run download_lidar.py

# List all available files
uv run download_lidar.py --list

# Download a specific file by index
uv run download_lidar.py --index 5

# Download a specific file by name pattern
uv run download_lidar.py --file "04200250"

# Specify output directory
uv run download_lidar.py -o ./data
```

### Load and Inspect LAZ Files

After downloading, you can quickly inspect a LAZ file's header, point stats, spatial extent, available dimensions, sample points, and any VLR/EVLR records.

```bash
# Load the first .laz file in the current directory
uv run load_lidar.py

# Or specify a path explicitly
uv run load_lidar.py ./data/USGS_LPC_CA_SanFrancisco_B23_05100185.laz
```

What you'll see:
- Header info (version, system identifier, generating software, creation date)
- Point data summary (count, format, size)
- Spatial extent and scale/offset; tile center in projected units
- Geographic center (lat/lon, with a Google Maps link) when CRS is present
- Available dimensions and the first few points as a sample
- VLRs and EVLRs (truncated if there are many)

Note:
- The tile `USGS_LPC_CA_SanFrancisco_B23_05300270.laz` corresponds to the San Francisco baseball park area.

### Other Data Sources

For the purposes of this walk through, I'm also using data from noaa.gov:

https://www.fisheries.noaa.gov/inport/item/73386


## 3D Viewer (Open3D)

Visualize `.laz`/`.las` point clouds with safety checks and optional downsampling.

```bash
# Basic usage (auto downsampling to ~3M points and color by elevation)
uv run view_lidar_o3d.py USGS_LPC_CA_SanFrancisco_B23_05100185.laz

# Explicit voxel size (in the same units as the data)
uv run view_lidar_o3d.py ./data/file.laz --voxel-size 0.5

# Choose coloring: elevation | rgb | none
uv run view_lidar_o3d.py ./data/file.laz --color rgb

# Override safety block thresholds
uv run view_lidar_o3d.py ./data/file.laz --force
```

Safety notes:
- The script reads only the header first to estimate feasibility.
- It warns above 20M points and blocks above 50M unless `--force`.
- With `--voxel-size auto`, it computes a voxel size to target ~3M points.

## Web Viewer (Streamlit + Plotly)

Run an in-browser viewer with interactive controls and WebGL rendering.

```bash
# Start the Streamlit app
uv run streamlit run app_streamlit.py
```

Features:
- Choose `.laz`/`.las` from the `data/` folder
- Color by elevation or embedded RGB (if present)
- Random downsample to a target number of points for performance
- Responsive Plotly 3D scatter rendering

Tips:
- Rendering millions of points in the browser can be slow; use the max points slider to tune performance.

### Deck.GL (pydeck) renderer
- Switch the "Renderer" toggle to "Deck.GL (pydeck)" for generally smoother interaction at higher point counts.
- Works in Cartesian coordinates with an orbit view; no base map needed.

## Potree Web Viewer (experimental)

You can also view (converted) point clouds with **Potree**, a WebGL point cloud renderer ([Potree on GitHub](https://github.com/potree/potree)).

### 1. Convert a LAZ to Potree format

Potree expects point clouds in its own octree format (a folder containing a `cloud.js` and `data/` files). Use **PotreeConverter** (from the Potree project) on one of your `.laz` files, for example:

```bash
./PotreeConverter ./data/USGS_LPC_CA_SanFrancisco_B23_05300270.laz \
  -o ./potree_viewer/pointclouds/sample
```

After this, you should have something like:

- `potree_viewer/pointclouds/sample/cloud.js`
- `potree_viewer/pointclouds/sample/data/...`

### 2. Run the Potree viewer locally

From the project root:

```bash
cd potree_viewer
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser. The bundled `index.html` loads `pointclouds/sample/cloud.js` via Potree’s viewer.

If you convert additional tiles, put each in its own subfolder under `potree_viewer/pointclouds/` and either:

- Update `POINTCLOUD_PATH` in `potree_viewer/index.html`, or
- Duplicate `index.html` per dataset and change the path.
