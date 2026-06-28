from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


def maybe_restore_with_gfpgan(
    image_bgr: np.ndarray,
    enabled: bool = False,
    model_path: Optional[str] = None,
) -> np.ndarray:
    if not enabled:
        return image_bgr

    try:
        from gfpgan import GFPGANer  # type: ignore
    except Exception:
        return image_bgr

    resolved_model = model_path or os.environ.get("GFPGAN_MODEL_PATH")
    if not resolved_model or not Path(resolved_model).exists():
        return image_bgr

    try:
        restorer = GFPGANer(
            model_path=resolved_model,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _, _, restored = restorer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
    except Exception:
        return image_bgr

    return restored if restored is not None else image_bgr
