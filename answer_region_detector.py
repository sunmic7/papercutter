"""CPU inference and conservative fusion for HiLEx answer-region boxes.

HiLEx ``Answer_Block`` means a region inside a question page (for example an
option, blank or answer area).  It is deliberately separate from
``answer_detector``: it must never be used as an answer-key entry or as the
authority for question--answer association.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from question_detector import Detection, QuestionDetector


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "runtime" / "models" / "answer_region_detector" / "answer_region_detector.onnx"


class AnswerRegionDetector(QuestionDetector):
    """One-class detector for question-page internal answer/option regions."""

    def __init__(self, model_path: str | Path | None = None, *, confidence: float = 0.78, imgsz: int = 640):
        selected = model_path or os.environ.get("PAPERCUTTER_ANSWER_REGION_DETECTOR_MODEL", DEFAULT_MODEL_PATH)
        super().__init__(selected, confidence=confidence, imgsz=imgsz)


_detector: AnswerRegionDetector | None = None


def get_answer_region_detector() -> AnswerRegionDetector:
    global _detector
    if _detector is None:
        _detector = AnswerRegionDetector()
    return _detector


def _horizontal_overlap_ratio(left: list[int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    narrower = min(max(1, left[2] - left[0]), max(1, right[2] - right[0]))
    return overlap / narrower


def protect_question_cuts(
    cuts: list[dict], image: np.ndarray, detections: list[Detection], *,
    min_confidence: float = 0.78, padding: int = 8,
) -> tuple[list[dict], dict[str, int]]:
    """Extend a legacy crop only when a detected inner region proves it is cut.

    Existing OCR-derived top/left/right boundaries remain authoritative.  The
    extension is capped before the next question's start, so this helper never
    transfers content from the next question into the current one.
    """
    stats = {"candidates": 0, "protected": 0, "skipped_cross_next": 0}
    if image is None or image.size == 0 or not cuts or not detections:
        return cuts, stats
    valid = [
        cut for cut in cuts
        if isinstance(cut.get("_source_box"), (list, tuple)) and len(cut["_source_box"]) == 4
    ]
    valid.sort(key=lambda cut: (cut["_source_box"][1], cut["_source_box"][0]))
    height = int(image.shape[0])
    for index, cut in enumerate(valid):
        x0, y0, x1, y1 = (int(value) for value in cut["_source_box"])
        next_top = int(valid[index + 1]["_source_box"][1]) if index + 1 < len(valid) else height
        ceiling = min(height, next_top - max(1, padding))
        if ceiling <= y1:
            continue
        candidates: list[Detection] = []
        for detection in detections:
            dx0, dy0, dx1, dy1 = detection.box
            if not (
                detection.confidence >= min_confidence
                and dy0 >= y0 - 10
                and dy0 <= y1 + padding
                and dy1 > y1 + 6
                and _horizontal_overlap_ratio([x0, y0, x1, y1], detection.box) >= 0.60
            ):
                continue
            stats["candidates"] += 1
            if dy1 > ceiling:
                stats["skipped_cross_next"] += 1
                continue
            candidates.append(detection)
        if not candidates:
            continue
        chosen = max(candidates, key=lambda item: (item.box[3], item.confidence))
        new_y1 = min(ceiling, int(chosen.box[3]) + padding)
        crop = image[y0:new_y1, x0:x1]
        if crop.size == 0 or new_y1 <= y1:
            continue
        cut["crop_img"] = crop
        cut["_source_box"] = [x0, y0, x1, new_y1]
        cut["_answer_region_assisted"] = True
        cut["_answer_region_confidence"] = float(chosen.confidence)
        cut["_answer_region_box"] = [int(value) for value in chosen.box]
        stats["protected"] += 1
    return cuts, stats
