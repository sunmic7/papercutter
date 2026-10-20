#!/usr/bin/env python3
"""Create a versioned, hashed manifest for a delivered question-detector ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except (ImportError, AttributeError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    model: str | Path,
    source_weights: str = "",
    output: str | Path | None = None,
    *,
    imgsz: int = 640,
) -> Path:
    model_path = Path(model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("请安装 onnxruntime 后再生成清单") from exc
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_info, output_info = session.get_inputs()[0], session.get_outputs()[0]
    manifest = {
        "schema_version": 2,
        "model_name": "question_detector",
        "model_version": "1.0.0",
        "architecture": "yolo11n",
        "task": "detect",
        "classes": {"0": "question"},
        "input": {
            "shape": input_info.shape,
            "dtype": "float32",
            "color": "RGB",
            "resize": "ultralytics_letterbox",
            "normalize": "divide_by_255",
        },
        "output": {
            "format": "ultralytics_nms",
            "shape": output_info.shape,
            "box_format": "xyxy",
            "fields": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
            "nms_in_model": True,
        },
        "export_postprocess": {
            "confidence_threshold": 0.0,
            "iou_threshold": 0.7,
            "max_detections": 300,
        },
        "runtime_postprocess": {
            "confidence_threshold": 0.72,
            "apply_nms_to_embedded_nms_output": True,
            "iou_threshold_when_applied": 0.50,
        },
        "raw_output_fallback": {
            "confidence_threshold": 0.72,
            "iou_threshold": 0.50,
            "apply_nms": True,
        },
        "runtime": {"backend": "onnxruntime-cpu"},
        "export": {
            "dynamic": False,
            "batch": 1,
            "ultralytics_version": _version("ultralytics"),
            "torch_version": _version("torch"),
            "onnx_version": _version("onnx"),
            "onnxruntime_version": _version("onnxruntime"),
            "command": (
                f'model.export(format="onnx", imgsz={int(imgsz)}, batch=1, dynamic=False, '
                'simplify=True, half=False, nms=True, conf=0.0, iou=0.7, max_det=300, device="cpu")'
            ),
        },
        "artifact": {"filename": model_path.name, "sha256": _sha256(model_path)},
        "source_weights": source_weights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output).resolve() if output else model_path.with_suffix(".json")
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成题目框 ONNX 的版本化模型清单")
    parser.add_argument("model")
    parser.add_argument("--source-weights", default="")
    parser.add_argument("--output")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(write_manifest(args.model, args.source_weights, args.output, imgsz=args.imgsz))


if __name__ == "__main__":
    main()
