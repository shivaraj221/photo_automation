from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

LEFT_EYE_INDICES = (33, 133, 159, 145, 160, 158)
RIGHT_EYE_INDICES = (362, 263, 386, 374, 385, 387)


def _eye_center(landmarks: np.ndarray, indices: Tuple[int, ...]) -> np.ndarray:
    return np.mean(landmarks[list(indices)], axis=0)


def align_by_eye_points(
    image_bgr: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray, float]:
    dy = float(right_eye[1] - left_eye[1])
    dx = float(right_eye[0] - left_eye[0])
    if dx == 0.0:
        dx = 1e-6

    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
    applied_rotation = -angle
    rotation_matrix = cv2.getRotationMatrix2D(center, applied_rotation, 1.0)

    aligned = cv2.warpAffine(
        image_bgr,
        rotation_matrix,
        (image_bgr.shape[1], image_bgr.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color,
    )
    return aligned, rotation_matrix.astype(np.float32), applied_rotation


def transform_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    flat_points = points.reshape(-1, 2).astype(np.float32)
    points_h = np.hstack([flat_points, np.ones((flat_points.shape[0], 1), dtype=np.float32)])
    rotated = (rotation_matrix @ points_h.T).T
    return rotated.reshape(points.shape).astype(np.float32)


def align_face_by_eyes(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray, float]:
    left_eye = _eye_center(landmarks, LEFT_EYE_INDICES)
    right_eye = _eye_center(landmarks, RIGHT_EYE_INDICES)
    aligned, rotation_matrix, applied_rotation = align_by_eye_points(
        image_bgr=image_bgr,
        left_eye=left_eye,
        right_eye=right_eye,
        fill_color=fill_color,
    )
    rotated_points = transform_points(landmarks, rotation_matrix)
    return aligned, rotated_points.astype(np.float32), applied_rotation
