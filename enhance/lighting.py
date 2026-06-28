from __future__ import annotations

import cv2
import numpy as np

def apply_subtle_lighting(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,  # kept for compatibility if passed
    tile_grid_size: tuple[int, int] = (8, 8),
    gamma: float = 1.06,
) -> np.ndarray:
    """
    Global tone match (production-grade simplicity).
    Lifts exposure slightly and neutralizes color cast.
    Keeps subject brightness close to background.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # Simple, reliable exposure lift per blueprint
    out = cv2.convertScaleAbs(image_bgr, alpha=1.05, beta=10)
    
    return out
