from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageOps

from background.replace import replace_with_clean_background
from config import OUTPUT_DIR, PASSPORT_CONFIG, SHEET_CONFIG
from enhance.gfpgan import maybe_restore_with_gfpgan
from enhance.lighting import apply_subtle_lighting
from enhance.pro_retouch import apply_pro_retouch
from face.align import align_by_eye_points, align_face_by_eyes, transform_points
from face.crop import crop_to_passport_spec
from face.detect import FaceMeshDetector, FaceNetMTCNNDetector
from layout.grid import generate_4x6_sheet


def _load_image(path: str | Path) -> np.ndarray:
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return image


def _save_bgr_with_dpi(path: Path, image_bgr: np.ndarray, dpi: int) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, format="JPEG", quality=95, dpi=(dpi, dpi))


def _resize_for_speed(image_bgr: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_bgr
    scale = max_dim / float(longest)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _detect_with_rotation_fallback(image_bgr: np.ndarray, detector):
    rotations = [
        ("0", None),
        ("90cw", cv2.ROTATE_90_CLOCKWISE),
        ("180", cv2.ROTATE_180),
        ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    last_error = None
    for _, code in rotations:
        candidate = cv2.rotate(image_bgr, code) if code is not None else image_bgr
        try:
            detection = detector.detect(candidate)
            return candidate, detection
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(
        "No face detected after trying orientation fixes. Use a front-facing photo with visible eyes."
    ) from last_error


def process_passport_photo(
    input_path: str | Path,
    copies: int = 4,
    output_dir: str | Path = OUTPUT_DIR,
    passport_config: Dict[str, float] | None = None,
    sheet_config: Dict[str, int] | None = None,
    use_gfpgan: bool = False,
    force_ai: bool = False,
    gfpgan_model_path: str | None = None,
) -> Dict[str, str]:
    p_cfg = passport_config or PASSPORT_CONFIG
    s_cfg = sheet_config or SHEET_CONFIG

    image = _load_image(input_path)
    image = _resize_for_speed(image, max_dim=2600)

    try:
        facenet_detector = FaceNetMTCNNDetector(min_confidence=0.72)
        image, facenet_detection = _detect_with_rotation_fallback(image, facenet_detector)

        aligned, rotation_matrix, _ = align_by_eye_points(
            image_bgr=image,
            left_eye=facenet_detection.left_eye,
            right_eye=facenet_detection.right_eye,
        )

        bbox = facenet_detection.bbox_xyxy.astype(np.float32)
        bbox_points = np.array(
            [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
            dtype=np.float32,
        )
        bbox_points_rot = transform_points(bbox_points, rotation_matrix)
        x_min = float(np.min(bbox_points_rot[:, 0]))
        y_min = float(np.min(bbox_points_rot[:, 1]))
        x_max = float(np.max(bbox_points_rot[:, 0]))
        y_max = float(np.max(bbox_points_rot[:, 1]))
        aligned_bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # Rotate the eye keypoints so they stay in sync with the aligned image
        eye_points = np.array(
            [facenet_detection.left_eye, facenet_detection.right_eye], dtype=np.float32
        )
        eye_points_rot = transform_points(eye_points, rotation_matrix)
        aligned_left_eye  = eye_points_rot[0]
        aligned_right_eye = eye_points_rot[1]

        replace_bg = bool(int(p_cfg.get("replace_background", 1.0)))
        prefer_u2net = bool(int(p_cfg.get("prefer_u2net", 1.0)))
        background_color = tuple(int(v) for v in p_cfg.get("background_color", (255, 255, 255)))
        subject_alpha = None

        if replace_bg:
            aligned, aligned_alpha = replace_with_clean_background(
                aligned,
                background_bgr=background_color,
                prefer_u2net=prefer_u2net,
                return_alpha=True,
            )
            # Make alpha 3-channel so we can crop it with the same function
            alpha_3 = np.repeat(aligned_alpha[:, :, None], 3, axis=2)
            alpha_cfg = p_cfg.copy()
            alpha_cfg["background_color"] = (0, 0, 0) # pad with black (0.0 alpha)
            cropped_alpha = crop_to_passport_spec(
                (alpha_3 * 255).astype(np.uint8),
                config=alpha_cfg,
                face_bbox_xyxy=aligned_bbox,
                left_eye=aligned_left_eye,
                right_eye=aligned_right_eye,
            )
            subject_alpha = (cropped_alpha[:, :, 0].astype(np.float32) / 255.0)

        cropped = crop_to_passport_spec(
            aligned,
            config=p_cfg,
            face_bbox_xyxy=aligned_bbox,
            left_eye=aligned_left_eye,
            right_eye=aligned_right_eye,
        )
    except Exception:
        detector = FaceMeshDetector()
        image, mesh_detection = _detect_with_rotation_fallback(image, detector)
        aligned, aligned_landmarks, _ = align_face_by_eyes(image, mesh_detection.landmarks)

        replace_bg = bool(int(p_cfg.get("replace_background", 1.0)))
        prefer_u2net = bool(int(p_cfg.get("prefer_u2net", 1.0)))
        background_color = tuple(int(v) for v in p_cfg.get("background_color", (255, 255, 255)))
        subject_alpha = None

        if replace_bg:
            aligned, aligned_alpha = replace_with_clean_background(
                aligned,
                background_bgr=background_color,
                prefer_u2net=prefer_u2net,
                return_alpha=True,
            )
            alpha_3 = np.repeat(aligned_alpha[:, :, None], 3, axis=2)
            alpha_cfg = p_cfg.copy()
            alpha_cfg["background_color"] = (0, 0, 0)
            cropped_alpha = crop_to_passport_spec(
                (alpha_3 * 255).astype(np.uint8),
                config=alpha_cfg,
                landmarks=aligned_landmarks,
            )
            subject_alpha = (cropped_alpha[:, :, 0].astype(np.float32) / 255.0)

        print("✂️ Step 4: Cropping to passport specifications...")
        cropped = crop_to_passport_spec(aligned, aligned_landmarks, p_cfg)

    print("✨ Step 5: Running AI Enhancement & Denoising...")
    from enhance.denoise import ai_denoise
    cropped = ai_denoise(cropped, force_ai=force_ai)

    print("💡 Step 6: Balancing lighting and shadows...")
    enhanced = apply_subtle_lighting(cropped)
    
    print("🎨 Step 7: Applying professional skin retouching...")
    enhanced = apply_pro_retouch(
        enhanced,
        subject_alpha=subject_alpha,
        skin_strength=float(p_cfg.get("skin_retouch_strength", 0.35)),
    )
    enhanced = maybe_restore_with_gfpgan(
        enhanced,
        enabled=use_gfpgan,
        model_path=gfpgan_model_path,
    )

    print("📝 Step 8: Generating print layout...")
    sheet = generate_4x6_sheet(enhanced, copies=copies, passport_config=p_cfg, sheet_config=s_cfg)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    passport_path = out_dir / f"passport_{timestamp}.jpg"
    sheet_path = out_dir / f"sheet_{copies}up_{timestamp}.jpg"

    dpi = int(p_cfg["dpi"])
    _save_bgr_with_dpi(passport_path, enhanced, dpi=dpi)
    sheet.save(sheet_path, format="JPEG", quality=95, dpi=(dpi, dpi))

    print(f"✅ Processing Complete! Files saved in: {output_dir}")
    print("-------------------------------------------\n")

    return {
        "passport_path": str(passport_path),
        "sheet_path": str(sheet_path),
    }
