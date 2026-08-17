#!/usr/bin/env python3
"""Read-only compatibility check for a delivered question-detector ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_CLASS = "question"


def _normalise_shape(shape: Any) -> list[Any]:
    return [int(value) if isinstance(value, int) else value for value in shape]


def validate_contract_metadata(manifest: dict[str, Any], input_shape: list[Any], input_type: str, output_shape: list[Any], output_type: str) -> list[str]:
    """Return human-readable violations; kept separate for unit testing."""
    errors: list[str] = []
    if input_shape != EXPECTED_INPUT_SHAPE:
        errors.append(f"输入形状必须是 {EXPECTED_INPUT_SHAPE}，实际为 {input_shape}")
    if input_type != "tensor(float)":
        errors.append(f"输入类型必须是 tensor(float)，实际为 {input_type}")
    if output_type != "tensor(float)":
        errors.append(f"输出类型必须是 tensor(float)，实际为 {output_type}")
    if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[2] != 6:
        errors.append(f"内嵌 NMS 输出必须是 [1, N, 6]，实际为 {output_shape}")
    if manifest.get("schema_version") != 2:
        errors.append("清单 schema_version 必须为 2")
    if manifest.get("model_name") != "question_detector":
        errors.append("清单 model_name 必须为 question_detector")
    if manifest.get("classes") != {"0": EXPECTED_CLASS}:
        errors.append("清单 classes 必须严格为 {\"0\": \"question\"}")
    if manifest.get("input", {}).get("shape") != EXPECTED_INPUT_SHAPE:
        errors.append("清单 input.shape 必须为 [1, 3, 640, 640]")
    output = manifest.get("output", {})
    if output.get("nms_in_model") is not True or output.get("format") != "ultralytics_nms":
        errors.append("清单 output 必须声明 ultralytics_nms 与 nms_in_model=true")
    if output.get("shape") != output_shape:
        errors.append("清单 output.shape 必须与 ONNX 实际输出一致")
    artifact = manifest.get("artifact", {})
    if artifact.get("filename") != "question_detector.onnx" or not artifact.get("sha256"):
        errors.append("清单 artifact 必须包含 question_detector.onnx 与 sha256")
    return errors


def validate_model(model: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    model_path = Path(model).expanduser().resolve()
    selected_manifest = Path(manifest_path).expanduser().resolve() if manifest_path else model_path.with_suffix(".json")
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到 ONNX 模型：{model_path}")
    if not selected_manifest.is_file():
        raise FileNotFoundError(f"未找到模型清单：{selected_manifest}")
    try:
        manifest = json.loads(selected_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型清单不是有效 JSON：{selected_manifest}") from exc
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("请安装 onnxruntime 后再校验模型") from exc
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"要求单输入单输出，实际为输入 {len(inputs)} 个、输出 {len(outputs)} 个")
    input_info, output_info = inputs[0], outputs[0]
    input_shape = _normalise_shape(input_info.shape)
    output_shape = _normalise_shape(output_info.shape)
    errors = validate_contract_metadata(manifest, input_shape, input_info.type, output_shape, output_info.type)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if manifest.get("artifact", {}).get("sha256") != digest:
        errors.append("ONNX 的 SHA-256 与清单不匹配")
    return {
        "ok": not errors,
        "model": str(model_path),
        "manifest": str(selected_manifest),
        "input": {"name": input_info.name, "shape": input_shape, "type": input_info.type},
        "output": {"name": output_info.name, "shape": output_shape, "type": output_info.type},
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="校验题目框 ONNX 是否符合项目模型接入契约 v1")
    parser.add_argument("model", help="待交付的 question_detector.onnx")
    parser.add_argument("--manifest", help="模型清单路径；默认使用模型同名 .json")
    args = parser.parse_args()
    try:
        report = validate_model(args.model, args.manifest)
    except (OSError, ValueError, RuntimeError) as exc:
        report = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
