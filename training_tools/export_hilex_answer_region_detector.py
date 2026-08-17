#!/usr/bin/env python3
"""Export the HiLEx question-page answer-region detector for CPU runtime."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = APP_DIR / "runtime" / "models" / "answer_region_detector"


def export_detector(weights: str | Path, output: str | Path = DEFAULT_OUTPUT, imgsz: int = 640) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Export requires the optional Ultralytics package") from exc
    weights_path, output_dir = Path(weights).resolve(), Path(output).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(str(weights_path)).export(
        format="onnx", imgsz=imgsz, dynamic=False, nms=True, device="cpu", simplify=True,
    ))
    target = output_dir / "answer_region_detector.onnx"
    shutil.copy2(exported, target)
    manifest = {
        "schema_version": 1,
        "name": "hilex_answer_region_detector",
        "format": "onnx",
        "task": "single-class question-page internal answer/option-region detection",
        "classes": ["answer_region"],
        "input_size": imgsz,
        "nms_embedded": True,
        "source_weights": str(weights_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    target.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": str(target), "manifest": str(target.with_suffix('.json')), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HiLEx answer-region detector for CPU inference")
    parser.add_argument("weights")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(export_detector(args.weights, args.output, args.imgsz), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
