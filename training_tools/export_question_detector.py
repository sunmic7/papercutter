#!/usr/bin/env python3
"""Export a trained detector to a CPU deployment artifact and write metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from write_question_detector_manifest import write_manifest


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = APP_DIR / "runtime" / "models" / "question_detector"


def export_detector(weights: str | Path, output: str | Path = DEFAULT_OUTPUT, imgsz: int = 640) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Export requires the optional Ultralytics package") from exc
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported = Path(model.export(
        format="onnx", imgsz=imgsz, batch=1, dynamic=False, simplify=True,
        half=False, nms=True, conf=0.0, iou=0.7, max_det=300, device="cpu",
    ))
    target = output_dir / "question_detector.onnx"
    shutil.copy2(exported, target)
    manifest_path = write_manifest(target, str(weights_path), imgsz=imgsz)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"model": str(target), "manifest": str(manifest_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the question detector for CPU inference")
    parser.add_argument("weights", help="best.pt from a training run")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(export_detector(args.weights, args.output, args.imgsz), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
