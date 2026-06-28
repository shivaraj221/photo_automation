import os
import sys
import cv2
import numpy as np
from PIL import Image
from contextlib import contextmanager

# Global variables for caching the model
_ai_model = None
_model_loaded = False

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def _get_realesrgan_model():
    """Lazily loads the RealESRGAN model and caches it."""
    global _ai_model, _model_loaded
    if _model_loaded:
        return _ai_model
    
    try:
        import torch
        import torchvision.transforms.functional as F
        # Monkeypatch for RealESRGAN on modern PyTorch where functional_tensor was removed
        sys.modules['torchvision.transforms.functional_tensor'] = F
        
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the official x4 model architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=400, # Use tiling to prevent VRAM exhaustion
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=device
        )
        
        _ai_model = upsampler
    except Exception as e:
        # print(f"Failed to load RealESRGAN: {e}")
        _ai_model = None
    
    _model_loaded = True
    return _ai_model

def _should_use_ai(image_bgr: np.ndarray, threshold: float = 400.0) -> bool:
    """Detects noise/softness using Laplacian variance."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def ai_denoise(image_bgr: np.ndarray, force_ai: bool = False) -> np.ndarray:
    """
    Adaptive AI Denoise/Enhancement.
    Runs RealESRGAN if the image is low quality, otherwise uses fast OpenCV denoising.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # Decide if we need heavy AI
    use_ai = force_ai or _should_use_ai(image_bgr)

    if use_ai:
        upsampler = _get_realesrgan_model()
        if upsampler is not None:
            try:
                # Suppress the library's tiling prints
                with suppress_stdout():
                    output_bgr, _ = upsampler.enhance(image_bgr, outscale=4)
                
                # Resize back to original dimensions
                h, w = image_bgr.shape[:2]
                output_bgr = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_AREA)
                
                return output_bgr
            except Exception:
                pass
    
    # Fast OpenCV Fallback (Non-Local Means)
    denoised = cv2.fastNlMeansDenoisingColored(image_bgr, None, 3, 3, 7, 21)
    return denoised

