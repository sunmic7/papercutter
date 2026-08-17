#!/usr/bin/env python3
"""Run a delivered question detector against a multi-page acceptance set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from question_detector import QuestionDetector, box_iou


def evaluate_record(predictions, record: dict) -> dict:
    expected_boxes = record.get("expected_boxes", [])
    expected_count = int(record.get("expected_count", len(expected_boxes)))
    allow_extra = int(record.get("allow_extra_detections", 0))
    minimum_iou = float(record.get("minimum_iou", 0.85))
    claimed: set[int] = set()
    matched = 0
    ious: list[float] = []
    for expected in expected_boxes:
        candidates = [
            (index, box_iou(expected, prediction.box))
            for index, prediction in enumerate(predictions) if index not in claimed
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= minimum_iou:
                claimed.add(index)
                matched += 1
                ious.append(overlap)
    count_ok = expected_count <= len(predictions) <= expected_count + allow_extra
    boxes_ok = not expected_boxes or matched == len(expected_boxes)
    return {
        "image": record["image"], "expected_count": expected_count, "predicted_count": len(predictions),
        "allow_extra_detections": allow_extra, "minimum_iou": minimum_iou,
        "matched_boxes": matched, "mean_iou": sum(ious) / len(ious) if ious else None,
        "ok": count_ok and boxes_ok,
    }


def evaluate(model: str | Path, acceptance_file: str | Path) -> dict:
    acceptance_path = Path(acceptance_file).resolve()
    document = json.loads(acceptance_path.read_text(encoding="utf-8"))
    records = document if isinstance(document, list) else document.get("samples", [])
    if not records:
        raise ValueError("验收清单必须包含非空 samples")
    confidence = float(document.get("confidence_threshold", 0.72)) if isinstance(document, dict) else 0.72
    detector = QuestionDetector(model_path=model, confidence=confidence)
    results = []
    for record in records:
        image_path = (acceptance_path.parent / record["image"]).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            results.append({"image": record["image"], "ok": False, "error": "无法读取图片"})
            continue
        results.append(evaluate_record(detector.detect(image), record))
    return {"model": str(Path(model).resolve()), "acceptance_file": str(acceptance_path), "passed": all(item["ok"] for item in results), "samples": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行题目框模型多场景验收集")
    parser.add_argument("model")
    parser.add_argument("acceptance_file", help="expected.json；图片路径相对此文件")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = evaluate(args.model, args.acceptance_file)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
