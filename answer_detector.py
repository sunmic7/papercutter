"""Optional CPU detector for individual entries on separate answer pages.

The detector is trained only from reviewed Chinese question--answer pairs.  It
never replaces OCR numbering or pairing: it can only refine an OCR-derived
answer crop when both boxes agree, preventing a visual false positive from
shifting later answers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from question_detector import Detection, QuestionDetector, refine_legacy_cuts


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "runtime" / "models" / "answer_detector" / "answer_detector.onnx"


class AnswerDetector(QuestionDetector):
    """One-class YOLO ONNX wrapper for reviewed answer-entry boxes."""

    def __init__(self, model_path: str | Path | None = None, *, confidence: float = 0.70, imgsz: int = 640):
        super().__init__(
            model_path or os.environ.get("PAPERCUTTER_ANSWER_DETECTOR_MODEL", DEFAULT_MODEL_PATH),
            confidence=confidence,
            imgsz=imgsz,
        )


_detector: AnswerDetector | None = None


def get_answer_detector() -> AnswerDetector:
    global _detector
    if _detector is None:
        _detector = AnswerDetector()
    return _detector


def refine_answer_records(
    records: list[dict], image: np.ndarray, detections: list[Detection], *, min_iou: float = 0.55,
) -> tuple[list[dict], dict[str, int]]:
    """Apply one-to-one detector refinement to OCR-confirmed answer records."""
    cuts = [
        {"crop_img": record.get("crop_img"), "_source_box": record.get("source_box")}
        for record in records
    ]
    refined, stats = refine_legacy_cuts(cuts, image, detections, min_iou=min_iou)
    for record, cut in zip(records, refined):
        if cut.get("_boundary_source") != "detector_assisted":
            continue
        record["crop_img"] = cut["crop_img"]
        record["source_box"] = cut["_source_box"]
        record["detector_assisted"] = True
        record["detector_confidence"] = cut.get("_detector_confidence")
    return records, stats
