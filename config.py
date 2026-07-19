from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

PASSPORT_CONFIG: Dict[str, float] = {
    "width_mm": 35,
    "height_mm": 45,
    "dpi": 300,
    # Chin-to-crown occupies 70-80% in many standards, but FaceMesh only measures hairline-to-chin.
    # We must use ~0.50 to leave room for the actual hair.
    "head_ratio": 0.55,
    "top_margin_ratio": 0.16,
    "top_margin_min_mm": 8.0,
    "top_margin_max_mm": 15.0,
    # Keep a healthy bottom area under chin for neck/shoulder visibility.
    "chin_bottom_ratio": 0.28,
    "chin_bottom_min_mm": 10.0,
    # Eye line lands at ~36% from top (international passport standard).
    "eye_line_ratio": 0.38,
    # FaceNet/MTCNN-driven crop profile (primary path).
    # Tighter crop to minimize empty background space.
    "facenet_head_ratio": 0.50,
    "facenet_top_margin_ratio": 0.20,
    "facenet_chin_bottom_ratio": 0.30,
    "facenet_chin_bottom_min_mm": 10.0,
    "facenet_width_expand_ratio": 0.0,
    "replace_background": 1.0,
    "prefer_u2net": 1.0,
    "background_color": (255, 255, 255),
    # Enhancement defaults (overridable from UI sliders)
    "skin_retouch_strength": 0.52,
    "lighting_clip_limit": 2.8,
}

SHEET_CONFIG: Dict[str, int] = {
    "width_in": 4,
    "height_in": 6,
    "dpi": 300,
    "background_color": (255, 255, 255),
}

SUPPORTED_COPIES = (4, 6, 8)
OUTPUT_DIR = Path("output")

FRAMING_PRESETS: Dict[str, Dict[str, float]] = {
    "More Shoulders": {
        "head_ratio": 0.45,
        "top_margin_ratio": 0.25,
        "top_margin_max_mm": 16.0,
        "chin_bottom_ratio": 0.30,
        "chin_bottom_min_mm": 13.0,
        "eye_line_ratio": 0.32,
        "facenet_head_ratio": 0.40,
        "facenet_top_margin_ratio": 0.28,
        "facenet_chin_bottom_ratio": 0.32,
        "facenet_chin_bottom_min_mm": 12.0,
        "facenet_width_expand_ratio": 0.0,
        "replace_background": 1.0,
        "prefer_u2net": 1.0,
    },
    "Balanced": {
        "head_ratio": 0.55,
        "top_margin_ratio": 0.16,
        "top_margin_max_mm": 13.0,
        "chin_bottom_ratio": 0.28,
        "chin_bottom_min_mm": 10.0,
        "eye_line_ratio": 0.38,
        "facenet_head_ratio": 0.50,
        "facenet_top_margin_ratio": 0.20,
        "facenet_chin_bottom_ratio": 0.30,
        "facenet_chin_bottom_min_mm": 10.0,
        "facenet_width_expand_ratio": 0.0,
        "replace_background": 1.0,
        "prefer_u2net": 1.0,
    },
    "Face Bigger": {
        "head_ratio": 0.65,
        "top_margin_ratio": 0.10,
        "top_margin_max_mm": 10.0,
        "chin_bottom_ratio": 0.25,
        "chin_bottom_min_mm": 8.0,
        "eye_line_ratio": 0.39,
        "facenet_head_ratio": 0.60,
        "facenet_top_margin_ratio": 0.12,
        "facenet_chin_bottom_ratio": 0.28,
        "facenet_chin_bottom_min_mm": 8.0,
        "facenet_width_expand_ratio": 0.0,
        "replace_background": 1.0,
        "prefer_u2net": 1.0,
    },
}


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round((mm / 25.4) * dpi))


def passport_size_px(config: Dict[str, float] | None = None) -> Tuple[int, int]:
    cfg = config or PASSPORT_CONFIG
    dpi = int(cfg["dpi"])
    width_px = mm_to_px(float(cfg["width_mm"]), dpi)
    height_px = mm_to_px(float(cfg["height_mm"]), dpi)
    return width_px, height_px


def sheet_size_px(config: Dict[str, int] | None = None) -> Tuple[int, int]:
    cfg = config or SHEET_CONFIG
    dpi = int(cfg["dpi"])
    return int(cfg["width_in"] * dpi), int(cfg["height_in"] * dpi)
