"""Dependency-light point-cloud rendering and result figure composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_cloud(points: np.ndarray, size: int = 320, title: str = "") -> Image.Image:
    """Render an isometric point-cloud view with a simple depth cue."""
    points = np.asarray(points)
    # A steep elevation makes the vertical stool structure legible while the
    # azimuth still separates its four legs in projection.
    azimuth, elevation = np.deg2rad(38), np.deg2rad(62)
    rz = np.array([[np.cos(azimuth), -np.sin(azimuth), 0],
                   [np.sin(azimuth), np.cos(azimuth), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(elevation), -np.sin(elevation)],
                   [0, np.sin(elevation), np.cos(elevation)]])
    rotated = points @ rz.T @ rx.T
    xy = rotated[:, :2]
    span = max(float(np.ptp(xy, axis=0).max()), 1e-6)
    xy = (xy / span * (size * 0.72)) + np.array([size / 2, size * 0.53])
    canvas = Image.new("RGB", (size, size), "#f7f4ed")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((4, 4, size - 5, size - 5), radius=14, outline="#d8d1c4", width=2)
    depth = rotated[:, 2]
    depth = (depth - depth.min()) / (np.ptp(depth) + 1e-8)
    for index in np.argsort(rotated[:, 1]):
        shade = int(90 + 120 * depth[index])
        color = (32, shade, 150 + int(70 * depth[index]))
        x, y = xy[index]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    if title:
        draw.text((14, 12), title, fill="#252525", font=ImageFont.load_default())
    return canvas


def contact_sheet(clouds: np.ndarray, path: str | Path, titles: list[str] | None = None,
                  columns: int = 4, cell_size: int = 260) -> Path:
    rows = int(np.ceil(len(clouds) / columns))
    sheet = Image.new("RGB", (columns * cell_size, rows * cell_size), "white")
    for index, cloud in enumerate(clouds):
        title = titles[index] if titles else f"Shape {index + 1}"
        sheet.paste(render_cloud(cloud, cell_size, title),
                    ((index % columns) * cell_size, (index // columns) * cell_size))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


def loss_plot(history: list[dict], path: str | Path, size: tuple[int, int] = (800, 480)) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    margin = 65
    draw.line((margin, 25, margin, size[1] - margin), fill="#333333", width=2)
    draw.line((margin, size[1] - margin, size[0] - 25, size[1] - margin), fill="#333333", width=2)
    series = [("train", "#147d92"), ("val", "#d05a45")]
    values = [record[group]["loss"] for record in history for group, _ in series]
    low, high = min(values), max(values)
    for group, color in series:
        coordinates = []
        for i, record in enumerate(history):
            x = margin + i * (size[0] - margin - 30) / max(len(history) - 1, 1)
            y = size[1] - margin - (record[group]["loss"] - low) / max(high - low, 1e-8) * (size[1] - 2 * margin)
            coordinates.append((x, y))
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=4)
    draw.text((margin, 6), "Training and validation objective", fill="#222222")
    draw.text((size[0] - 165, 10), "train", fill=series[0][1])
    draw.text((size[0] - 90, 10), "validation", fill=series[1][1])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
