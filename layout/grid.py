from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import PASSPORT_CONFIG, SHEET_CONFIG, SUPPORTED_COPIES, passport_size_px, sheet_size_px

GRID_PRESETS = {
    4: {"cols": 2, "rows": 2, "rotate": True},
    6: {"cols": 2, "rows": 3, "rotate": True},
    8: {"cols": 2, "rows": 4, "rotate": True},
}


def _bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def generate_4x6_sheet(
    passport_image_bgr: np.ndarray,
    copies: int,
    passport_config: Dict[str, float] | None = None,
    sheet_config: Dict[str, int] | None = None,
) -> Image.Image:
    if copies not in SUPPORTED_COPIES:
        raise ValueError(f"Unsupported copy count: {copies}. Use one of {SUPPORTED_COPIES}.")

    p_cfg = passport_config or PASSPORT_CONFIG
    s_cfg = sheet_config or SHEET_CONFIG
    sheet_w, sheet_h = sheet_size_px(s_cfg)
    photo_w, photo_h = passport_size_px(p_cfg)
    background = tuple(s_cfg.get("background_color", (255, 255, 255)))

    photo = _bgr_to_pil(passport_image_bgr).resize((photo_w, photo_h), Image.Resampling.LANCZOS)
    
    # Add a thin 2px border for easy cutting
    photo = ImageOps.expand(photo, border=2, fill="black")
    
    preset = GRID_PRESETS[copies]
    if preset["rotate"]:
        photo = photo.rotate(90, expand=True)

    tile_w, tile_h = photo.size
    cols = int(preset["cols"])
    rows = int(preset["rows"])

    gap_x = (sheet_w - (cols * tile_w)) / (cols + 1)
    
    # Force the vertical gap to pack tightly at the top (simulate max rows)
    max_rows = sheet_h // tile_h
    gap_y = (sheet_h - (max_rows * tile_h)) / (max_rows + 1)
    if gap_x < 0 or gap_y < 0:
        raise ValueError("Selected layout does not fit 4x6 at actual passport size.")

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=background)
    for row in range(rows):
        y = int(round(gap_y + row * (tile_h + gap_y)))
        for col in range(cols):
            x = int(round(gap_x + col * (tile_w + gap_x)))
            sheet.paste(photo, (x, y))

    return sheet
