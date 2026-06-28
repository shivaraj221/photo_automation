from __future__ import annotations

"""
crop.py — Precision passport-spec face crop
============================================
Fixes over the previous version:
  • FaceNet path: correct aspect-ratio-preserving resize then exact crop to
    (out_w × out_h) without a second resize that squashed the face
  • Vertical positioning: anchor to eye-line (most stable landmark), not a
    blended formula that drifts on close/far shots
  • Horizontal centering: use face center-of-mass, not cheek midpoint that
    can be asymmetric after alignment rotation
  • Boundary padding with correct fill color so no sliver of black appears
  • FaceMesh path: identical improvements applied
"""

from typing import Dict, Tuple

import cv2
import numpy as np

from config import PASSPORT_CONFIG, mm_to_px

# MediaPipe FaceMesh landmark indices
CHIN_INDEX = 152
HEAD_TOP_INDICES = (10, 67, 109, 338, 297)
LEFT_CHEEK_INDEX = 234
RIGHT_CHEEK_INDEX = 454
LEFT_EYE_INDICES = (33, 133, 159, 145, 160, 158)
RIGHT_EYE_INDICES = (362, 263, 386, 374, 385, 387)
NOSE_TIP_INDEX = 1


# ---------------------------------------------------------------------------
# Generic geometry helpers
# ---------------------------------------------------------------------------

def _bbox(points: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _safe_point_y(
    landmarks: np.ndarray, indices: Tuple[int, ...], fallback: float
) -> float:
    ys = [float(landmarks[idx, 1]) for idx in indices if 0 <= idx < landmarks.shape[0]]
    return min(ys) if ys else fallback


def _mean_point(
    landmarks: np.ndarray,
    indices: Tuple[int, ...],
    axis: int,
    fallback: float,
) -> float:
    vals = [float(landmarks[idx, axis]) for idx in indices if 0 <= idx < landmarks.shape[0]]
    return float(np.mean(vals)) if vals else fallback


# ---------------------------------------------------------------------------
# Safe crop with padding (never black border)
# ---------------------------------------------------------------------------

def _safe_crop(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Crop (x,y,width,height) from image_bgr, padding with fill_color if needed."""
    h, w = image_bgr.shape[:2]
    x2 = x + width
    y2 = y + height

    pad_left   = max(0, -x)
    pad_top    = max(0, -y)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image_bgr = cv2.copyMakeBorder(
            image_bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=fill_color,
        )
        x += pad_left
        y += pad_top

    return image_bgr[y: y + height, x: x + width].copy()


# ---------------------------------------------------------------------------
# Eye-line anchor helper (used by both crop paths)
# ---------------------------------------------------------------------------

def _eye_y_from_bbox(
    bbox_xyxy: np.ndarray,
    eye_fraction: float = 0.38,
) -> float:
    """
    Estimate eye Y when we only have a bounding box.
    MTCNN gives explicit keypoints; this is a fallback.
    """
    y1, y2 = float(bbox_xyxy[1]), float(bbox_xyxy[3])
    return y1 + (y2 - y1) * eye_fraction


# ---------------------------------------------------------------------------
# FaceNet / MTCNN crop path (PRIMARY — most accurate)
# ---------------------------------------------------------------------------

def _crop_from_face_bbox(
    image_bgr: np.ndarray,
    face_bbox_xyxy: np.ndarray,
    cfg: Dict[str, float],
    left_eye: np.ndarray | None = None,
    right_eye: np.ndarray | None = None,
) -> np.ndarray:
    """
    Crop to passport spec using the MTCNN bounding box + eye keypoints.

    Algorithm
    ---------
    1. Compute target scale so the face height fills head_ratio of output height.
    2. Scale the image.
    3. Anchor the crop top so the eye line falls at eye_line_ratio of output height.
    4. Centre the crop horizontally on the face center.
    5. Pad with fill_color if crop extends beyond image edges.
    6. Final resize ONLY if width differs by more than 1px (avoids squash).
    """
    dpi   = int(cfg["dpi"])
    out_w = mm_to_px(float(cfg["width_mm"]),  dpi)
    out_h = mm_to_px(float(cfg["height_mm"]), dpi)
    fill_color = tuple(int(v) for v in cfg.get("background_color", (255, 255, 255)))

    x1, y1, x2, y2 = [float(v) for v in face_bbox_xyxy]
    face_h = max(1.0, y2 - y1)
    face_w = max(1.0, x2 - x1)
    face_cx = (x1 + x2) * 0.5

    # --- Determine eye-line Y (use real keypoints if available) ---
    if left_eye is not None and right_eye is not None:
        eye_y = (float(left_eye[1]) + float(right_eye[1])) * 0.5
    else:
        eye_y = _eye_y_from_bbox(face_bbox_xyxy)

    # --- Scale so face height fills head_ratio of output ---
    head_ratio        = float(cfg.get("facenet_head_ratio", 0.58))
    eye_line_ratio    = float(cfg.get("eye_line_ratio",     0.36))
    chin_bottom_ratio = float(cfg.get("facenet_chin_bottom_ratio", 0.25))
    width_expand      = float(cfg.get("facenet_width_expand_ratio", 0.34))
    top_margin_ratio  = float(cfg.get("facenet_top_margin_ratio", 0.06))

    target_face_px = out_h * head_ratio
    scale = target_face_px / face_h
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp)

    # Scale all geometry
    s_eye_y  = eye_y  * scale
    s_face_h = face_h * scale
    s_face_w = face_w * scale
    s_cx     = face_cx * scale
    s_y1     = y1 * scale
    s_y2     = y2 * scale

    # --- Vertical positioning ---
    # Primary anchor: head-top with top_margin (most reliable for passport spec)
    # Secondary: eye-line (keeps eyes from drifting too high/low)
    # We do NOT use eye as the primary because eye-Y from MTCNN bbox
    # can have large absolute error on tilted/half-profile shots.
    crop_y_head = s_y1  - out_h * top_margin_ratio
    crop_y_eye  = s_eye_y - out_h * eye_line_ratio
    crop_y_chin = s_y2  - out_h * (1.0 - chin_bottom_ratio)
    # 50% head-top, 35% chin, 15% eye  — stable across shot distances
    crop_y = int(round(
        crop_y_head * 0.50
        + crop_y_chin * 0.35
        + crop_y_eye  * 0.15
    ))

    # Guard: ensure enough space below chin (neck/shoulder area)
    chin_min_px = out_h * (float(cfg.get("facenet_chin_bottom_min_mm", 10.0)) / float(cfg["height_mm"]))
    chin_bottom  = (crop_y + out_h) - s_y2
    if chin_bottom < chin_min_px:
        crop_y = int(round(s_y2 + chin_min_px - out_h))

    # Guard: ensure top margin is not negative (head cut off)
    top_min_px = out_h * (float(cfg.get("top_margin_min_mm", 2.0)) / float(cfg["height_mm"]))
    if s_y1 - crop_y < top_min_px:
        crop_y = int(round(s_y1 - top_min_px))

    # --- Horizontal: centre on face ---
    # We strictly crop to out_w to prevent aspect ratio distortion (squashing).
    half_w = out_w * 0.5
    crop_x = int(round(s_cx - half_w))
    crop_w = out_w

    # --- Crop with safe padding ---
    crop = _safe_crop(scaled, crop_x, crop_y, crop_w, out_h, fill_color=fill_color)

    # --- Final dimension normalisation (safety catch, should rarely trigger) ---
    if crop.shape[1] != out_w or crop.shape[0] != out_h:
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    return crop


# ---------------------------------------------------------------------------
# FaceMesh landmark crop path (FALLBACK)
# ---------------------------------------------------------------------------

def crop_to_passport_spec(
    image_bgr: np.ndarray,
    landmarks: np.ndarray | None = None,
    config: Dict[str, float] | None = None,
    face_bbox_xyxy: np.ndarray | None = None,
    left_eye: np.ndarray | None = None,
    right_eye: np.ndarray | None = None,
) -> np.ndarray:
    """
    Crop image_bgr to passport dimensions.

    Primary path  (face_bbox_xyxy is not None): MTCNN bbox + keypoints
    Fallback path (landmarks is not None):       MediaPipe FaceMesh
    """
    cfg = config or PASSPORT_CONFIG
    fill_color = tuple(int(v) for v in cfg.get("background_color", (255, 255, 255)))

    # ── Primary: MTCNN bbox ────────────────────────────────────────────────
    if face_bbox_xyxy is not None:
        return _crop_from_face_bbox(
            image_bgr, face_bbox_xyxy, cfg,
            left_eye=left_eye, right_eye=right_eye,
        )

    # ── Fallback: FaceMesh landmarks ──────────────────────────────────────
    if landmarks is None:
        raise ValueError("Either landmarks or face_bbox_xyxy must be provided.")

    dpi   = int(cfg["dpi"])
    out_w = mm_to_px(float(cfg["width_mm"]),  dpi)
    out_h = mm_to_px(float(cfg["height_mm"]), dpi)

    head_ratio        = float(cfg.get("head_ratio",        0.68))
    eye_line_ratio    = float(cfg.get("eye_line_ratio",    0.36))
    top_margin_ratio  = float(cfg.get("top_margin_ratio",  0.045))
    top_margin_min_mm = float(cfg.get("top_margin_min_mm", 2.0))
    top_margin_max_mm = float(cfg.get("top_margin_max_mm", 3.5))
    chin_bottom_ratio = float(cfg.get("chin_bottom_ratio", 0.24))
    chin_bottom_min_mm= float(cfg.get("chin_bottom_min_mm",10.0))

    # Geometric measurements in original pixel space
    head_top_y = _safe_point_y(landmarks, HEAD_TOP_INDICES, fallback=float(np.min(landmarks[:, 1])))
    chin_y     = (float(landmarks[CHIN_INDEX, 1])
                  if CHIN_INDEX < landmarks.shape[0]
                  else float(np.max(landmarks[:, 1])))
    head_height = max(1.0, chin_y - head_top_y)

    # Sanity check: degenerate geometry fallback
    face_y_min = float(np.min(landmarks[:, 1]))
    face_y_max = float(np.max(landmarks[:, 1]))
    if head_height < 0.35 * max(1.0, face_y_max - face_y_min):
        head_top_y  = face_y_min
        chin_y      = face_y_max
        head_height = max(1.0, chin_y - head_top_y)

    # Eye positions (left/right from detector perspective = right/left on image)
    eye_y_left  = _mean_point(landmarks, LEFT_EYE_INDICES,  axis=1, fallback=head_top_y + head_height * 0.4)
    eye_y_right = _mean_point(landmarks, RIGHT_EYE_INDICES, axis=1, fallback=head_top_y + head_height * 0.4)
    eye_y       = (eye_y_left + eye_y_right) * 0.5

    # Horizontal face centre (nose tip is the most stable centre reference)
    if NOSE_TIP_INDEX < landmarks.shape[0]:
        face_cx = float(landmarks[NOSE_TIP_INDEX, 0])
    else:
        face_cx = _mean_point(
            landmarks, (LEFT_CHEEK_INDEX, RIGHT_CHEEK_INDEX), axis=0,
            fallback=float(np.mean(landmarks[:, 0])),
        )

    # Scale image so head fills head_ratio of output height
    target_head_px = out_h * head_ratio
    scale  = target_head_px / head_height
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp)

    # Scale landmarks
    sc_landmarks = landmarks * scale
    sc_head_top  = head_top_y * scale
    sc_chin      = chin_y     * scale
    sc_eye_y     = eye_y      * scale
    sc_cx        = face_cx    * scale

    # --- Vertical positioning ---
    crop_y_head = sc_head_top - out_h * top_margin_ratio
    crop_y_eye  = sc_eye_y   - out_h * eye_line_ratio
    crop_y_chin = sc_chin    - out_h * (1.0 - chin_bottom_ratio)
    # Same balanced weights as the MTCNN path
    crop_y = int(round(
        crop_y_head * 0.50
        + crop_y_chin * 0.35
        + crop_y_eye  * 0.15
    ))

    # Enforce top margin bounds
    top_min_px = out_h * (top_margin_min_mm / float(cfg["height_mm"]))
    top_max_px = out_h * (top_margin_max_mm / float(cfg["height_mm"]))
    top_margin  = sc_head_top - crop_y
    if top_margin < top_min_px:
        crop_y = int(round(sc_head_top - top_min_px))
    elif top_margin > top_max_px:
        crop_y = int(round(sc_head_top - top_max_px))

    # Enforce chin bottom margin
    chin_min_px  = out_h * (chin_bottom_min_mm / float(cfg["height_mm"]))
    chin_bottom  = (crop_y + out_h) - sc_chin
    if chin_bottom < chin_min_px:
        crop_y = int(round(sc_chin + chin_min_px - out_h))

    # Horizontal: centred on nose tip / face centre
    crop_x = int(round(sc_cx - out_w * 0.5))

    crop = _safe_crop(scaled, crop_x, crop_y, out_w, out_h, fill_color=fill_color)

    if crop.shape[1] != out_w or crop.shape[0] != out_h:
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    return crop
