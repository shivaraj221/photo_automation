from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

class _MaskEngines:
    _mp_segmenter = None
    _rembg_session = None

    @classmethod
    def mediapipe_mask(cls, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            import mediapipe as mp
        except Exception:
            return None

        if cls._mp_segmenter is None:
            if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "selfie_segmentation"):
                return None
            cls._mp_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = cls._mp_segmenter.process(rgb)
        if result.segmentation_mask is None:
            return None
        return np.clip(result.segmentation_mask.astype(np.float32), 0.0, 1.0)

    @classmethod
    def rembg_mask(cls, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            from rembg import new_session, remove  # type: ignore
        except Exception:
            return None

        if cls._rembg_session is None:
            # 1. Try u2net_human_seg (Best for portraits, preserves clothing)
            try:
                cls._rembg_session = new_session("u2net_human_seg")
            except Exception:
                # 2. Try u2net (High quality)
                try:
                    cls._rembg_session = new_session("u2net")
                except Exception:
                    # 3. Fallback to u2netp (Tiny, can cause aggressive cropping)
                    try:
                        cls._rembg_session = new_session("u2netp")
                    except Exception:
                        cls._rembg_session = None
                        return None

        try:
            # Send the ORIGINAL image to rembg — no gamma tricks.
            # Gamma correction was causing light-colored clothing (white shirts)
            # to be mistakenly identified as background and removed.
            ok, buf = cv2.imencode(".png", image_bgr)
            if not ok:
                return None
            
            # First try with post_process_mask for cleaner edges
            try:
                out = remove(
                    buf.tobytes(), 
                    session=cls._rembg_session, 
                    only_mask=True, 
                    post_process_mask=True,
                )
            except TypeError:
                # Fallback for older rembg versions
                out = remove(
                    buf.tobytes(), 
                    session=cls._rembg_session, 
                    only_mask=True, 
                )
        except Exception:
            return None

        if isinstance(out, (bytes, bytearray)):
            arr = np.frombuffer(out, dtype=np.uint8)
            mask_u8 = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                return None
            return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

        if isinstance(out, np.ndarray):
            if out.ndim == 2:
                mask = out.astype(np.float32)
            elif out.ndim == 3 and out.shape[2] == 4:
                mask = out[:, :, 3].astype(np.float32)
            else:
                return None
            if mask.max() > 1.0:
                mask = mask / 255.0
            return np.clip(mask, 0.0, 1.0)

        return None


def replace_with_clean_background(
    image_bgr: np.ndarray,
    background_bgr: Tuple[int, int, int] = (255, 255, 255),
    prefer_u2net: bool = True,
    return_alpha: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Replace background using high-quality matting and simple edge refinement."""
    raw_mask = _MaskEngines.rembg_mask(image_bgr) if prefer_u2net else None
    if raw_mask is None:
        raw_mask = _MaskEngines.mediapipe_mask(image_bgr)
    if raw_mask is None:
        if return_alpha:
            return image_bgr, np.ones(image_bgr.shape[:2], dtype=np.float32)
        return image_bgr
    
    # Ensure mask is a clean alpha
    alpha = np.clip(raw_mask.astype(np.float32), 0.0, 1.0)

    # Edge refinement: close small holes, then smooth edges for natural look
    kernel = np.ones((5, 5), np.uint8)
    # Dilate slightly first to capture fine hair strands, then erode back
    alpha_u8 = (alpha * 255).astype(np.uint8)
    alpha_u8 = cv2.dilate(alpha_u8, kernel, iterations=1)
    alpha_u8 = cv2.erode(alpha_u8, kernel, iterations=1)
    # Close remaining small holes
    alpha_u8 = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
    alpha = alpha_u8.astype(np.float32) / 255.0
    # Smooth feather for natural composite edge
    alpha = cv2.GaussianBlur(alpha, (11, 11), sigmaX=2.5)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Alpha blend
    bg = np.full_like(image_bgr, background_bgr, dtype=np.uint8)
    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)
    composited = image_bgr.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1.0 - alpha_3)
    result = np.clip(composited, 0, 255).astype(np.uint8)

    if return_alpha:
        return result, alpha.astype(np.float32)
    return result
