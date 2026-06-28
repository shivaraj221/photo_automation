from __future__ import annotations

import cv2
import numpy as np


def subtle_skin_smooth(
    image_bgr: np.ndarray,
    diameter: int = 7,
    sigma_color: int = 55,
    sigma_space: int = 55,
) -> np.ndarray:
    return cv2.bilateralFilter(
        image_bgr,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
