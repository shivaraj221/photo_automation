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
            try:
                cls._rembg_session = new_session("u2netp") # Tiny model for low RAM
            except Exception:
                try:
                    cls._rembg_session = new_session("u2net")
                except Exception:
                    cls._rembg_session = None
                    return None

        try:
            # Trick for black backgrounds: Temporarily brighten shadows (Gamma Correction)
            # so the AI can distinguish dark hair/clothing from the dark background
            img_f = image_bgr.astype(np.float32) / 255.0
            brightened = np.clip(np.power(img_f, 0.45) * 255.0, 0, 255).astype(np.uint8)
            
            ok, buf = cv2.imencode(".png", brightened)
            if not ok:
                return None
            
            # Using alpha_matting also drastically improves edges on hard contrasts
            out = remove(
                buf.tobytes(), 
                session=cls._rembg_session, 
                only_mask=True, 
                post_process_mask=True,
                alpha_matting=True
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

    # Edge refinement (feather slightly, close small holes)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Alpha blend
    bg = np.full_like(image_bgr, background_bgr, dtype=np.uint8)
    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)
    composited = image_bgr.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1.0 - alpha_3)
    result = np.clip(composited, 0, 255).astype(np.uint8)

    if return_alpha:
        return result, alpha.astype(np.float32)
    return result
