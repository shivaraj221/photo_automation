from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
from PIL import Image


@dataclass
class FaceNetDetectionResult:
    bbox_xyxy: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    confidence: float
    image_shape: Tuple[int, int, int]


@dataclass
class FaceDetectionResult:
    landmarks: np.ndarray
    image_shape: Tuple[int, int, int]


class FaceMeshDetector:
    _shared_face_mesh = None

    @staticmethod
    def _get_face_mesh_class():
        # MediaPipe packaging differs across builds. Support both public and internal paths.
        if mp is None:
            raise RuntimeError("MediaPipe is not available in this environment.")
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return mp.solutions.face_mesh.FaceMesh
        try:
            from mediapipe.python.solutions import face_mesh  # type: ignore

            return face_mesh.FaceMesh
        except Exception as exc:  # pragma: no cover - environment dependent
            package_path = getattr(mp, "__file__", "unknown")
            raise RuntimeError(
                "MediaPipe face mesh API not found in this environment. "
                "Reinstall the official package: pip uninstall -y mediapipe && pip install mediapipe. "
                f"Loaded module path: {package_path}"
            ) from exc

    def __init__(
        self,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.50,
    ) -> None:
        if FaceMeshDetector._shared_face_mesh is None:
            face_mesh_class = self._get_face_mesh_class()
            FaceMeshDetector._shared_face_mesh = face_mesh_class(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        self._face_mesh = FaceMeshDetector._shared_face_mesh

    def detect(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty.")

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected. Use a clearer front-facing image.")

        face_landmarks = results.multi_face_landmarks[0].landmark
        points = np.zeros((len(face_landmarks), 2), dtype=np.float32)
        for idx, lm in enumerate(face_landmarks):
            x = np.clip(lm.x * w, 0, w - 1)
            y = np.clip(lm.y * h, 0, h - 1)
            points[idx] = (x, y)

        return FaceDetectionResult(landmarks=points, image_shape=image_bgr.shape)


class FaceNetMTCNNDetector:
    _shared_mtcnn = None

    def __init__(self, min_confidence: float = 0.80) -> None:
        self.min_confidence = float(min_confidence)
        if FaceNetMTCNNDetector._shared_mtcnn is None:
            try:
                import torch
                from facenet_pytorch import MTCNN
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "facenet-pytorch is not installed. Run: pip install facenet-pytorch"
                ) from exc

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            FaceNetMTCNNDetector._shared_mtcnn = MTCNN(
                keep_all=True,
                device=device,
                post_process=False,
            )
        self._mtcnn = FaceNetMTCNNDetector._shared_mtcnn

    def detect(self, image_bgr: np.ndarray) -> FaceNetDetectionResult:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty.")

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)

        boxes, probs, landmarks = self._mtcnn.detect(pil, landmarks=True)
        if boxes is None or probs is None:
            raise ValueError("No face detected by FaceNet MTCNN.")

        valid_indices = [idx for idx, p in enumerate(probs) if p is not None and float(p) >= self.min_confidence]
        if not valid_indices:
            valid_indices = list(range(len(probs)))
        best_idx = max(valid_indices, key=lambda idx: float(probs[idx]))

        bbox = boxes[best_idx].astype(np.float32)
        bbox[0] = np.clip(bbox[0], 0, w - 1)
        bbox[1] = np.clip(bbox[1], 0, h - 1)
        bbox[2] = np.clip(bbox[2], 0, w - 1)
        bbox[3] = np.clip(bbox[3], 0, h - 1)

        if landmarks is None:
            raise ValueError("Face detected but no landmarks returned by FaceNet MTCNN.")
        keypoints = landmarks[best_idx].astype(np.float32)
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye

        return FaceNetDetectionResult(
            bbox_xyxy=bbox,
            left_eye=left_eye,
            right_eye=right_eye,
            confidence=float(probs[best_idx]),
            image_shape=image_bgr.shape,
        )
