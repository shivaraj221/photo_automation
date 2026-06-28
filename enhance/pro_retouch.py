from __future__ import annotations

import cv2
import numpy as np

def _soft_subject_mask(
    subject_alpha: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if subject_alpha is None:
        return np.ones(shape, dtype=np.float32)
    mask = cv2.resize(
        subject_alpha.astype(np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(mask, 0.0, 1.0)

def _skin_mask(
    image_bgr: np.ndarray,
    subject_alpha: np.ndarray | None = None,
) -> np.ndarray:
    """
    Skin mask in YCrCb + HSV to exclude eyes, lips, hair.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    skin_ycrcb = (
        (cr >= 130) & (cr <= 180)
        & (cb >= 75) & (cb <= 135)
    )
    skin_hsv = (
        ((h <= 25) | (h >= 160))
        & (s >= 20) & (s <= 220)
        & (v >= 40)
    )
    skin = (skin_ycrcb | skin_hsv).astype(np.uint8) * 255

    e7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, e7, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, e7, iterations=2)

    skin_f = cv2.GaussianBlur(skin.astype(np.float32) / 255.0, (11, 11), 0)

    subject = _soft_subject_mask(subject_alpha, image_bgr.shape[:2])
    return np.clip(skin_f * subject, 0.0, 1.0)

def apply_pro_retouch(
    image_bgr: np.ndarray,
    subject_alpha: np.ndarray | None = None,
    skin_strength: float = 0.5,
) -> np.ndarray:
    """
    Adobe-style masked and subtle skin retouch.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    strength = float(np.clip(skin_strength, 0.0, 1.0))
    
    if strength == 0.0:
        return image_bgr

    # Create skin mask
    skin = _skin_mask(image_bgr, subject_alpha)

    # Light bilateral filter for smoothing
    smooth = cv2.bilateralFilter(image_bgr, 7, 50, 50)
    
    # Blend with low strength based on skin mask
    blend = (skin * strength)[:, :, None]
    final_face = image_bgr.astype(np.float32) * (1.0 - blend) + smooth.astype(np.float32) * blend

    return np.clip(final_face, 0, 255).astype(np.uint8)
