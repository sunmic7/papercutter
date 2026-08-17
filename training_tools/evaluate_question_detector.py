#!/usr/bin/env python3
"""Evaluate a trained question detector against a held-out YOLO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(weights: str | Path, data: str | Path, imgsz: int = 960) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Evaluation requires the optional Ultralytics package") from exc
    weights_path, data_path = Path(weights).resolve(), Path(data).resolve()
    if not weights_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Both model weights and data.yaml must exist")
    result = YOLO(str(weights_path)).val(
        data=str(data_path), split="test", imgsz=imgsz, batch=1, device="cpu", workers=4,
    )
    summary = {
        "schema_version": 1,
        "weights": str(weights_path),
        "data": str(data_path),
        "split": "test",
        "imgsz": imgsz,
        "metrics": {key: float(value) for key, value in result.results_dict.items()},
    }
    report_path = weights_path.parent / "question_detector_evaluation.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(report_path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a question detector on the HiLEx test split")
    parser.add_argument("weights", help="best.pt or exported model")
    parser.add_argument("--data", default="runtime/training_data/hilex_question/data.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.weights, args.data, args.imgsz), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

