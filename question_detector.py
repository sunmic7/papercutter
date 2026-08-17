"""CPU runtime and conservative fusion for the trainable question detector.

The module is deliberately independent from PaddleOCR.  A missing or failed
model is never an error for the cutter: callers receive an empty result and
keep the established OCR-rule pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "runtime" / "models" / "question_detector" / "question_detector.onnx"


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0


def box_iou(left: list[int] | tuple[int, int, int, int], right: list[int] | tuple[int, int, int, int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


class QuestionDetector:
    """ONNX Runtime wrapper for a one-class YOLO export with embedded NMS."""

    def __init__(self, model_path: str | Path | None = None, *, confidence: float = 0.72, imgsz: int = 960):
        self.model_path = Path(model_path or os.environ.get("PAPERCUTTER_QUESTION_DETECTOR_MODEL", DEFAULT_MODEL_PATH))
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self._session: Any | None = None
        self._error: str = ""

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    @property
    def error(self) -> str:
        return self._error

    def _load(self) -> bool:
        if self._session is not None:
            return True
        if not self.available:
            return False
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"],
            )
            # Deployment exports are normally fixed-size ONNX graphs.  Read the
            # graph shape instead of assuming the constructor default so a 640
            # export can be used safely by the main application.
            input_shape = self._session.get_inputs()[0].shape
            if len(input_shape) == 4:
                graph_height, graph_width = input_shape[2], input_shape[3]
                if (
                    isinstance(graph_height, int)
                    and isinstance(graph_width, int)
                    and graph_height > 0
                    and graph_height == graph_width
                ):
                    self.imgsz = graph_height
            return True
        except Exception as exc:  # Optional runtime; preserve cutter availability.
            self._error = str(exc)
            return False

    @staticmethod
    def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(size / width, size / height)
        resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (size - resized_width) // 2, (size - resized_height) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _nms(detections: list[Detection], threshold: float = 0.50) -> list[Detection]:
        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(box_iou(detection.box, chosen.box) < threshold for chosen in selected):
                selected.append(detection)
        return selected

    def _decode(self, output: np.ndarray, scale: float, pad_x: int, pad_y: int, width: int, height: int) -> list[Detection]:
        values = np.asarray(output)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        # Raw one-class YOLO: (5, anchors).  Export with NMS: (detections, 6).
        if values.ndim == 2 and values.shape[0] in {5, 6} and values.shape[1] > values.shape[0]:
            values = values.T
        if values.ndim != 2 or values.shape[1] < 5:
            return []

        detections: list[Detection] = []
        embedded_nms = False
        for row in values:
            if row.shape[0] >= 6 and row[4] <= 1.0 and row[5] <= 1.0:
                # nms=True export: xyxy, confidence, class.
                x0, y0, x1, y1, confidence = map(float, row[:5])
                class_id = int(round(float(row[5])))
                embedded_nms = True
            else:
                # Raw one-class export: cx, cy, width, height, class confidence.
                cx, cy, box_width, box_height, confidence = map(float, row[:5])
                x0, y0 = cx - box_width / 2, cy - box_height / 2
                x1, y1 = cx + box_width / 2, cy + box_height / 2
                class_id = 0
            if class_id != 0 or confidence < self.confidence:
                continue
            x0 = int(round((x0 - pad_x) / scale))
            y0 = int(round((y0 - pad_y) / scale))
            x1 = int(round((x1 - pad_x) / scale))
            y1 = int(round((y1 - pad_y) / scale))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 - x0 >= 12 and y1 - y0 >= 12:
                detections.append(Detection((x0, y0, x1, y1), confidence, class_id))
        # The formal deployment contract preserves the current two-stage
        # behaviour: the ONNX graph performs its own NMS, then this stricter
        # 0.50 deduplication is applied as a runtime safety net.  Raw YOLO
        # outputs use this same runtime NMS as their required fallback.
        return self._nms(detections) if detections or not embedded_nms else detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0 or not self._load():
            return []
        prepared, scale, pad_x, pad_y = self._letterbox(image, self.imgsz)
        tensor = prepared[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        try:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: tensor})
        except Exception as exc:
            self._error = str(exc)
            return []
        height, width = image.shape[:2]
        return self._decode(outputs[0], scale, pad_x, pad_y, width, height)


_detector: QuestionDetector | None = None


def get_question_detector() -> QuestionDetector:
    global _detector
    if _detector is None:
        _detector = QuestionDetector()
    return _detector


def refine_legacy_cuts(
    cuts: list[dict], image: np.ndarray, detections: list[Detection], *, min_iou: float = 0.55,
) -> tuple[list[dict], dict[str, int]]:
    """Refine only one-to-one high-confidence legacy matches.

    This deliberately does not add unmatched detector boxes.  It protects the
    current book/section logic while a detector trained only on English data is
    being validated on Chinese teaching materials.
    """
    stats = {"matched": 0, "refined": 0, "unmatched_detections": len(detections)}
    if image is None or not cuts or not detections:
        return cuts, stats
    claimed: set[int] = set()
    for cut in cuts:
        source_box = cut.get("_source_box")
        if not isinstance(source_box, (list, tuple)) or len(source_box) != 4:
            continue
        candidates = [
            (index, detection, box_iou(source_box, detection.box))
            for index, detection in enumerate(detections)
            if index not in claimed
        ]
        if not candidates:
            continue
        index, detection, overlap = max(candidates, key=lambda item: item[2])
        if overlap < min_iou:
            continue
        stats["matched"] += 1
        claimed.add(index)
        x0, y0, x1, y1 = detection.box
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        cut["crop_img"] = crop
        cut["_source_box"] = [x0, y0, x1, y1]
        cut["_source_segments"] = []
        cut["_boundary_source"] = "detector_assisted"
        cut["_detector_confidence"] = float(detection.confidence)
        stats["refined"] += 1
    stats["unmatched_detections"] = len(detections) - len(claimed)
    return cuts, stats


def _horizontal_overlap_ratio(left: list[int], right: tuple[int, int, int, int]) -> float:
    """Return overlap relative to the narrower box, for same-column checks."""
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    narrower = min(max(1, left[2] - left[0]), max(1, right[2] - right[0]))
    return overlap / narrower


def recover_sequence_gap_cuts(
    cuts: list[dict], image: np.ndarray, detections: list[Detection], *,
    min_confidence: float = 0.86,
) -> tuple[list[dict], dict[str, int]]:
    """Conservatively recover exactly one missed question inside a numbered gap.

    The usual OCR failure is a merged ``6 + 7`` crop followed by a correctly
    recognised ``8``.  A regular detector-assisted match cannot help because
    the model's box for question 7 is intentionally unmatched.  This recovery
    is deliberately narrower than general model-first cutting: it runs only for
    a one-number gap, only inside the preceding legacy crop, and only with a
    high-confidence box in the same column.  The recovered number is marked as
    inferred so downstream answer pairing and review remain cautious.
    """
    stats = {"recovered": 0, "unresolved_gaps": 0}
    if image is None or image.size == 0 or len(cuts) < 2 or not detections:
        return cuts, stats

    ordered = sorted(
        enumerate(cuts),
        key=lambda entry: (
            entry[1].get("_source_box", [0, 0, 0, 0])[1],
            entry[1].get("_source_box", [0, 0, 0, 0])[0],
        ),
    )
    recovered: list[dict] = []
    for position in range(len(ordered) - 1):
        _, previous = ordered[position]
        _, following = ordered[position + 1]
        previous_num = previous.get("num")
        following_num = following.get("num")
        previous_box = previous.get("_source_box")
        following_box = following.get("_source_box")
        if not (
            isinstance(previous_num, int) and isinstance(following_num, int)
            and following_num == previous_num + 2
            and isinstance(previous_box, (list, tuple)) and len(previous_box) == 4
            and isinstance(following_box, (list, tuple)) and len(following_box) == 4
        ):
            continue

        # Keep the original question anchor, if available: a detector box for
        # the preceding question must not be mistaken for the missing one.
        major_box = previous.get("_major_box", [0, 0, 0, 0])
        anchor_y = int(previous_box[1])
        if isinstance(major_box, (list, tuple)) and len(major_box) == 4:
            anchor_y = max(anchor_y, int(major_box[1]) + int(previous.get("_anchor_y", 0)))
        candidates = []
        for detection in detections:
            if detection.confidence < min_confidence:
                continue
            box = detection.box
            # The recovery box must begin after the previous number, fit inside
            # the merged crop, and share its column.  Also reject model boxes
            # that already substantially match an existing crop.
            if not (
                box[1] >= anchor_y + 24
                and box[1] < int(following_box[1]) - 18
                and box[2] > int(previous_box[0])
                and box[0] < int(previous_box[2])
                and _horizontal_overlap_ratio(list(previous_box), box) >= 0.62
                and box_iou(previous_box, box) < 0.50
                and all(
                    box_iou(existing.get("_source_box", [0, 0, 0, 0]), box) < 0.50
                    for existing in cuts
                    if isinstance(existing.get("_source_box"), (list, tuple))
                    and len(existing.get("_source_box")) == 4
                )
            ):
                continue
            candidates.append(detection)

        if not candidates:
            stats["unresolved_gaps"] += 1
            following["_sequence_gap_before"] = [previous_num + 1]
            continue
        # Prefer the first valid block in reading order; confidence only breaks
        # ties on nearly identical starts.
        detection = min(candidates, key=lambda item: (item.box[1], -item.confidence))
        split_y = max(int(previous_box[1]) + 1, int(detection.box[1]))
        x0, y0, x1, y1 = (int(value) for value in previous_box)
        preceding_crop = image[y0:split_y, x0:x1]
        recovered_crop = image[detection.box[1]:detection.box[3], detection.box[0]:detection.box[2]]
        if preceding_crop.size == 0 or recovered_crop.size == 0:
            stats["unresolved_gaps"] += 1
            following["_sequence_gap_before"] = [previous_num + 1]
            continue

        previous["crop_img"] = preceding_crop
        previous["_source_box"] = [x0, y0, x1, split_y]
        recovered.append({
            "num": previous_num + 1,
            "crop_img": recovered_crop,
            "number_detected": False,
            "_source_box": list(detection.box),
            "_source_segments": [],
            "_major_box": list(previous.get("_major_box", [])),
            "_source_path": previous.get("_source_path", ""),
            "_source_size": list(previous.get("_source_size", [])),
            "_page_deskew_angle": float(previous.get("_page_deskew_angle", 0.0)),
            "_boundary_source": "detector_sequence_recovery",
            "_detector_confidence": float(detection.confidence),
            "_number_confidence": 0.0,
            "_inferred_number": True,
            "_sequence_gap_recovered": True,
        })
        stats["recovered"] += 1

    if not recovered:
        return cuts, stats
    return sorted(
        [*cuts, *recovered],
        key=lambda cut: (
            cut.get("_source_box", [0, 0, 0, 0])[1],
            cut.get("_source_box", [0, 0, 0, 0])[0],
        ),
    ), stats


def model_manifest(model_path: str | Path) -> dict:
    path = Path(model_path)
    manifest = path.with_suffix(".json")
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
