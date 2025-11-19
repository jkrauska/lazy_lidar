#!/usr/bin/env python3
"""
Streamlit + Plotly LiDAR Viewer
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import laspy
import pandas as pd
import pydeck as pdk


DATA_DIR = Path(__file__).parent / "data"


def human_bytes(num_bytes: float) -> str:
	size_units = ["B", "KB", "MB", "GB", "TB"]
	size = float(num_bytes)
	for unit in size_units:
		if size < 1024.0:
			return f"{size:.2f} {unit}"
		size /= 1024.0
	return f"{size:.2f} PB"


def has_rgb(las: laspy.LasData) -> bool:
	try:
		dims = set(getattr(las.point_format, "dimension_names", []))
	except Exception:
		return False
	return {"red", "green", "blue"}.issubset(dims)


def to_rgb_0_255_from_las(las: laspy.LasData) -> Optional[np.ndarray]:
	if not has_rgb(las):
		return None
	r = np.asarray(las.red)
	g = np.asarray(las.green)
	b = np.asarray(las.blue)
	max_val = 65535.0 if max(r.max(initial=0), g.max(initial=0), b.max(initial=0)) > 255 else 255.0
	arr = np.stack([r, g, b], axis=1).astype(np.float32) / float(max_val)
	arr = np.clip(arr, 0.0, 1.0)
	return (arr * 255.0).astype(np.uint8)


def color_by_elevation(z: np.ndarray) -> np.ndarray:
	# Returns normalized [0,1] for use with Plotly colorscale
	z = np.asarray(z, dtype=np.float64)
	z_min = float(np.min(z))
	z_max = float(np.max(z))
	if z_max <= z_min:
		return np.zeros_like(z, dtype=np.float32)
	return ((z - z_min) / (z_max - z_min)).astype(np.float32)


@st.cache_data(show_spinner=False)
def list_laz_files(data_dir: Path) -> list[Path]:
	if not data_dir.exists():
		return []
	return sorted(list(data_dir.glob("*.laz")) + list(data_dir.glob("*.las")))


@st.cache_data(show_spinner=True)
def read_las_header_info(path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
	with laspy.open(path) as laz:
		point_count = int(laz.header.point_count)
		mins = np.asarray(laz.header.mins, dtype=np.float64)
		maxs = np.asarray(laz.header.maxs, dtype=np.float64)
	return point_count, mins, maxs


@st.cache_data(show_spinner=True)
def load_points(path: Path) -> np.ndarray:
	las = laspy.read(path)
	x = np.asarray(las.x, dtype=np.float32)
	y = np.asarray(las.y, dtype=np.float32)
	z = np.asarray(las.z, dtype=np.float32)
	xyz = np.stack([x, y, z], axis=1)
	return xyz


@st.cache_data(show_spinner=True)
def load_rgb_colors(path: Path) -> Optional[np.ndarray]:
	# Load only RGB as uint8 if present; return None if not available
	try:
		with laspy.open(path) as laz:
			dims = set(getattr(laz.header.point_format, "dimension_names", []))
		if not {"red", "green", "blue"}.issubset(dims):
			return None
		las = laspy.read(path)
		r = np.asarray(las.red)
		g = np.asarray(las.green)
		b = np.asarray(las.blue)
		max_val = 65535.0 if max(r.max(initial=0), g.max(initial=0), b.max(initial=0)) > 255 else 255.0
		arr = np.stack([r, g, b], axis=1).astype(np.float32) / float(max_val)
		arr = np.clip(arr, 0.0, 1.0)
		return (arr * 255.0).astype(np.uint8)
	except Exception:
		return None


def compute_auto_voxel_size_from_bounds(mins: np.ndarray, maxs: np.ndarray, target_points: int) -> float:
	mins = np.asarray(mins, dtype=np.float64)
	maxs = np.asarray(maxs, dtype=np.float64)
	extent = np.maximum(maxs - mins, 1e-6)
	volume = float(extent[0] * extent[1] * extent[2])
	safe_target = max(1, int(target_points))
	voxel_size = (volume / float(safe_target)) ** (1.0 / 3.0)
	return float(np.clip(voxel_size, 1e-3, max(extent.max() * 0.1, 1e-3)))


def random_downsample(xyz: np.ndarray, colors_opt: Optional[np.ndarray], max_points: int, seed: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	n = xyz.shape[0]
	if n <= max_points:
		return xyz, colors_opt
	rng = np.random.default_rng(seed)
	indices = rng.choice(n, size=int(max_points), replace=False)
	xyz_ds = xyz[indices]
	if colors_opt is None:
		return xyz_ds, None
	return xyz_ds, colors_opt[indices]


def voxel_grid_downsample(
	xyz: np.ndarray,
	colors_opt: Optional[np.ndarray],
	voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	if voxel_size <= 0:
		return xyz, colors_opt
	q = np.floor(xyz / float(voxel_size)).astype(np.int64)
	# Hash quantized coords to unique voxels
	# Use a mixed radix hash to reduce collisions
	hashes = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
	# Keep first index per voxel
	_, first_idx = np.unique(hashes, return_index=True)
	first_idx.sort()
	xyz_ds = xyz[first_idx]
	if colors_opt is None:
		return xyz_ds, None
	return xyz_ds, colors_opt[first_idx]


def build_plotly_figure(
	xyz: np.ndarray,
	color_mode: Literal["elevation", "rgb", "none"],
	colors_opt: Optional[np.ndarray],
	point_size: int,
) -> go.Figure:
	if color_mode == "rgb" and colors_opt is not None:
		# Convert to 'rgb(r,g,b)' strings for Plotly
		color_strings = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors_opt]
		marker_kwargs = dict(color=color_strings)
	elif color_mode == "elevation":
		marker_kwargs = dict(
			color=color_by_elevation(xyz[:, 2]),
			colorscale="Viridis",
			colorbar=dict(title="Elevation (norm)"),
			cmin=0.0,
			cmax=1.0,
		)
	else:
		marker_kwargs = dict(color="lightgray")

	scatter = go.Scatter3d(
		x=xyz[:, 0],
		y=xyz[:, 1],
		z=xyz[:, 2],
		mode="markers",
		marker=dict(size=point_size, opacity=0.9, **marker_kwargs),
	)
	fig = go.Figure(data=[scatter])
	fig.update_traces(hoverinfo="skip")
	fig.update_layout(
		scene=dict(
			xaxis_title="X",
			yaxis_title="Y",
			zaxis_title="Z",
			aspectmode="data",
		),
		margin=dict(l=0, r=0, t=30, b=0),
	)
	return fig


def build_deck_gl_chart(
	xyz: np.ndarray,
	color_mode: Literal["elevation", "rgb", "none"],
	colors_opt: Optional[np.ndarray],
	point_size: int,
):
	# Prepare colors as uint8 [r,g,b]
	if color_mode == "rgb" and colors_opt is not None:
		rgb_u8 = colors_opt.astype(np.uint8)
	elif color_mode == "elevation":
		norm = color_by_elevation(xyz[:, 2])
		# Map to Viridis-like simple gradient with numpy (approx)
		# We'll create a simple gradient: blue->cyan->yellow->red via piecewise
		r = np.clip(1.5 * norm - 0.25, 0.0, 1.0)
		g = np.clip(1.5 - np.abs(2.0 * norm - 1.0) * 1.5, 0.0, 1.0)
		b = np.clip(1.5 * (1.0 - norm) - 0.25, 0.0, 1.0)
		rgb_u8 = (np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)
	else:
		rgb_u8 = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)

	df = pd.DataFrame(
		{
			"x": xyz[:, 0],
			"y": xyz[:, 1],
			"z": xyz[:, 2],
			"r": rgb_u8[:, 0],
			"g": rgb_u8[:, 1],
			"b": rgb_u8[:, 2],
		}
	)
	# Drop any non-finite rows to avoid deck.gl assertions
	df = df.replace([np.inf, -np.inf], np.nan).dropna()
	if len(df) == 0:
		# Return an empty deck to avoid crash; caller can warn
		return pdk.Deck(layers=[], initial_view_state=pdk.ViewState(target=[0, 0, 0], zoom=0.1), map_provider=None, map_style=None)

	# Recentre coordinates around origin to avoid precision issues with large world coords
	center = [float(np.mean(df["x"])), float(np.mean(df["y"])), float(np.mean(df["z"]))]
	df["x_rel"] = df["x"] - center[0]
	df["y_rel"] = df["y"] - center[1]
	df["z_rel"] = df["z"] - center[2]
	extent = [
		float((df["x"].max() - df["x"].min())),
		float((df["y"].max() - df["y"].min())),
		float((df["z"].max() - df["z"].min())),
	]
	max_extent = max(extent) if max(extent) > 0 else 1.0
	# Conservative default zoom; OrbitView zoom is unitless, keep moderate
	zoom = 0.0

	layer = pdk.Layer(
		"PointCloudLayer",
		data=df,
		get_position="[x_rel, y_rel, z_rel]",
		get_color="[r, g, b]",
		point_size=int(point_size),  # in pixels
		get_normal=[0, 0, 1],
		coordinate_system=0,  # CARTESIAN for OrbitView
		pickable=False,
	)

	view = pdk.View(type="OrbitView", controller=False)
	# Minimal, robust view state
	view_state = pdk.ViewState(target=[0, 0, 0], zoom=1.0)

	deck = pdk.Deck(layers=[layer], views=[view], initial_view_state=view_state, map_provider=None, map_style=None)
	return deck


def main() -> None:
	st.set_page_config(page_title="Lazy LiDAR (Streamlit + Plotly)", layout="wide")
	st.title("Lazy LiDAR – Web Viewer (Streamlit + Plotly)")
	st.caption("Interactive LiDAR point cloud viewer in your browser.")

	files = list_laz_files(DATA_DIR)
	col_left, col_right = st.columns([1, 2])

	with col_left:
		if not files:
			st.error(f"No .laz/.las files found in `{DATA_DIR}`")
			st.stop()

		choices = {f.name: f for f in files}
		selected_name = st.selectbox("Choose a file", list(choices.keys()), index=0)
		selected_path = choices[selected_name]

		point_count, mins, maxs = read_las_header_info(selected_path)
		st.write(f"Points: {point_count:,}")
		st.write(f"Bounds min: {mins}")
		st.write(f"Bounds max: {maxs}")

		color_mode: Literal["elevation", "rgb", "none"] = st.selectbox(
			"Color mode",
			options=["elevation", "rgb", "none"],
			index=0,
			help="Use elevation color or embedded RGB (if available).",
		)

		downsample_mode: Literal["Random", "Voxel grid"] = st.selectbox(
			"Downsample method",
			options=["Random", "Voxel grid"],
			index=0,
			help="Random picks points uniformly; voxel grid preserves spatial structure.",
		)

		max_points = st.slider(
			"Max points to display (downsamples randomly if exceeded)",
			min_value=50_000,
			max_value=3_000_000,
			value=min(50_000, int(point_count)),
			step=50_000,
		)
		point_size = st.slider("Point size", min_value=1, max_value=5, value=2)

		if point_count > 5_000_000:
			st.warning(
				"This is a large file. Rendering all points in the browser can be slow. "
				"Consider lowering the max points."
			)

	with st.spinner("Loading points..."):
		xyz = load_points(selected_path)
		colors_opt: Optional[np.ndarray] = None
		if color_mode == "rgb":
			colors_opt = load_rgb_colors(selected_path)
			if colors_opt is None:
				st.info("No RGB in this file; falling back to elevation.")
				color_mode = "elevation"  # type: ignore

	if downsample_mode == "Voxel grid":
		# Estimate voxel size to roughly target max_points
		voxel_size = compute_auto_voxel_size_from_bounds(mins, maxs, target_points=max_points)
		xyz_vis, colors_vis = voxel_grid_downsample(xyz, colors_opt, voxel_size=voxel_size)
		# If still too many, cap randomly to max_points as a fallback
		if xyz_vis.shape[0] > max_points:
			xyz_vis, colors_vis = random_downsample(xyz_vis, colors_vis, max_points=max_points)
	else:
		xyz_vis, colors_vis = random_downsample(xyz, colors_opt, max_points=max_points)

	fig = build_plotly_figure(
		xyz=xyz_vis,
		color_mode=color_mode,
		colors_opt=colors_vis,
		point_size=point_size,
	)
	with col_right:
		renderer = st.radio(
			"Renderer",
			options=["Plotly 3D", "Deck.GL (pydeck)"],
			horizontal=True,
			index=1,
			help="Deck.GL can handle larger point counts more smoothly in many cases.",
		)
		if renderer == "Deck.GL (pydeck)":
			deck = build_deck_gl_chart(
				xyz=xyz_vis,
				color_mode=color_mode,
				colors_opt=colors_vis,
				point_size=point_size,
			)
			st.pydeck_chart(deck, width="stretch", height=700)
		else:
			st.plotly_chart(fig, width="stretch", config={"responsive": True})

	# Memory estimate
	bytes_estimate = xyz_vis.nbytes + (0 if colors_vis is None else colors_vis.nbytes)
	st.caption(f"Approx render memory (client): {human_bytes(bytes_estimate)}")


if __name__ == "__main__":
	main()


