#!/usr/bin/env python3
"""Run a reproducible VLM benchmark for complete-question boxes.

This tool deliberately stays outside the production cutter.  It reads the
existing original-page annotation manifest, asks a vision model for semantic
question boxes, stores all non-secret evidence, and compares the result with
the manually labelled boxes.  It is intended to answer whether a VLM can
understand a *complete question* before the main cutting pipeline is changed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from review_tools.evaluate_cutter import compare_page_boxes


SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"
NORMALIZED_MAX = 1000.0

SYSTEM_PROMPT = """你是教材版面切题系统。识别页面中的所有“完整题目”。
一个题目必须尽量包含：题干、选项、属于该题的公式、插图和所有子问；不得带入下一道独立题。
不要根据题号或行距机械切分，而要按语义判断归属。
若一段材料或一张图服务多个题，把它放进 shared_regions，并在相关题目的 shared_region_ids 中引用它。
坐标必须相对整页归一化到 0~1000，bbox 格式为 [x1,y1,x2,y2]。只返回符合给定 JSON Schema 的结果，不要解释。"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return value


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _schema() -> dict[str, Any]:
    box = {
        "type": "array", "minItems": 4, "maxItems": 4,
        "items": {"type": "number", "minimum": 0, "maximum": 1000},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["coordinate_system", "questions", "shared_regions"],
        "properties": {
            "coordinate_system": {"type": "string", "enum": ["normalized_0_1000"]},
            "questions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["question_id", "bbox", "shared_region_ids"],
                    "properties": {
                        "question_id": {"type": "string"},
                        "bbox": box,
                        "shared_region_ids": {"type": "array", "items": {"type": "string"}},
                        "contains_figure": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "shared_regions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "type", "bbox"],
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "bbox": box,
                    },
                },
            },
        },
    }


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("模型响应不含 JSON 文本")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型响应未找到 JSON 对象")
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"模型响应 JSON 无效: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError("模型响应根节点必须是对象")
    return parsed


def _number_box(box: Any) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox 必须是四个数字")
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError) as error:
        raise ValueError("bbox 必须是四个数字") from error
    if not all(math.isfinite(value) and 0 <= value <= NORMALIZED_MAX for value in values):
        raise ValueError("归一化 bbox 必须位于 0~1000")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("bbox 宽高必须为正")
    return values


def normalized_box_to_pixels(box: Any, width: int, height: int) -> list[int]:
    """Map a validated 0~1000 xyxy box back to the original page."""
    x0, y0, x1, y1 = _number_box(box)
    result = [
        max(0, min(width - 1, int(math.floor(x0 / NORMALIZED_MAX * width)))),
        max(0, min(height - 1, int(math.floor(y0 / NORMALIZED_MAX * height)))),
        max(1, min(width, int(math.ceil(x1 / NORMALIZED_MAX * width)))),
        max(1, min(height, int(math.ceil(y1 / NORMALIZED_MAX * height)))),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("映射后的 bbox 宽高必须为正")
    return result


def normalize_prediction(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    if payload.get("coordinate_system") != "normalized_0_1000":
        raise ValueError("coordinate_system 必须为 normalized_0_1000")
    raw_questions = payload.get("questions")
    raw_shared = payload.get("shared_regions")
    if not isinstance(raw_questions, list) or not isinstance(raw_shared, list):
        raise ValueError("questions 与 shared_regions 必须是数组")

    shared_regions = []
    shared_ids: set[str] = set()
    for item in raw_shared:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ValueError("shared_regions 的每项必须具有非空 id")
        region_id = item["id"].strip()
        if region_id in shared_ids:
            raise ValueError(f"shared_regions 存在重复 id: {region_id}")
        shared_ids.add(region_id)
        shared_regions.append({
            "id": region_id,
            "type": str(item.get("type", "shared_material")),
            "bbox_normalized": _number_box(item.get("bbox")),
            "bbox": normalized_box_to_pixels(item.get("bbox"), width, height),
        })

    questions = []
    question_ids: set[str] = set()
    for position, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            raise ValueError("questions 的每项必须是对象")
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            raise ValueError(f"第 {position} 个 question 缺少 question_id")
        if question_id in question_ids:
            raise ValueError(f"questions 存在重复 question_id: {question_id}")
        question_ids.add(question_id)
        linked = item.get("shared_region_ids", [])
        if not isinstance(linked, list) or not all(isinstance(value, str) for value in linked):
            raise ValueError("shared_region_ids 必须是字符串数组")
        unknown = sorted(set(linked) - shared_ids)
        if unknown:
            raise ValueError(f"question {question_id} 引用了不存在的 shared_region: {unknown}")
        question = {
            "question_id": question_id,
            "bbox_normalized": _number_box(item.get("bbox")),
            "bbox": normalized_box_to_pixels(item.get("bbox"), width, height),
            "shared_region_ids": linked,
            "contains_figure": bool(item.get("contains_figure", False)),
        }
        if isinstance(item.get("confidence"), (int, float)):
            question["model_confidence"] = float(item["confidence"])
        questions.append(question)
    return {"questions": questions, "shared_regions": shared_regions}


def prepare_image(path: Path, max_long_side: int) -> tuple[str, dict[str, int]]:
    # cv2.imread is unreliable for Chinese Windows paths.  Keep this aligned
    # with the main cutter's Unicode-safe image loading behaviour.
    try:
        encoded_source = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded_source, cv2.IMREAD_COLOR)
    except OSError as error:
        raise ValueError("无法读取输入图像") from error
    if image is None or image.size == 0:
        raise ValueError("无法读取输入图像")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_long_side) / max(width, height))
    sent_width, sent_height = max(1, round(width * scale)), max(1, round(height * scale))
    if (sent_width, sent_height) != (width, height):
        image = cv2.resize(image, (sent_width, sent_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("无法编码测试图像")
    data_uri = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return data_uri, {
        "original_width": width, "original_height": height,
        "sent_width": sent_width, "sent_height": sent_height,
    }


def _read_image_unicode(path: Path) -> Any:
    try:
        source = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(source, cv2.IMREAD_COLOR)
    except OSError:
        return None


def _write_image_unicode(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("无法写入预览图")
    encoded.tofile(str(path))


def write_visual_artifacts(image_path: Path, image_name: str, prediction: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """Write an original-size overlay and plain question crops for visual review."""
    image = _read_image_unicode(image_path)
    if image is None or image.size == 0:
        raise ValueError("无法读取原图以生成预览")
    stem = Path(image_name).stem
    digest = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:12]
    key = f"{stem}-{digest}"
    overlay = image.copy()
    for region in prediction.get("shared_regions", []):
        x0, y0, x1, y1 = region["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 215, 255), 5)
    crop_dir = run_dir / "crops" / key
    for index, question in enumerate(prediction.get("questions", []), start=1):
        x0, y0, x1, y1 = question["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 5)
        cv2.putText(overlay, str(index), (x0 + 6, max(28, y0 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        crop = image[y0:y1, x0:x1]
        if crop.size:
            _write_image_unicode(crop_dir / f"{index:03d}.jpg", crop)
    overlay_path = run_dir / "previews" / f"{key}.jpg"
    _write_image_unicode(overlay_path, overlay)
    return {
        "overlay": str(overlay_path.relative_to(run_dir)).replace("\\", "/"),
        "crops": str(crop_dir.relative_to(run_dir)).replace("\\", "/"),
    }


def build_request(model: str, image_data_uri: str, structured_mode: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        }],
    }
    if structured_mode == "json_schema":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "complete_question_boxes", "strict": True, "schema": _schema()},
        }
    elif structured_mode == "json_object":
        request["response_format"] = {"type": "json_object"}
    return request


def call_openai_compatible(base_url: str, api_key: str, request_data: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    encoded = json.dumps(request_data, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=encoded,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"API 请求失败: {error.reason}") from error
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("API 返回的外层 JSON 无效") from error
    if not isinstance(payload, dict):
        raise RuntimeError("API 返回的外层 JSON 必须是对象")
    return payload, elapsed


def response_content(response: dict[str, Any]) -> Any:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("API 响应缺少 choices[0].message.content") from error


def _usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    def integer(*names: str) -> int:
        for name in names:
            try:
                return max(0, int(usage.get(name, 0)))
            except (TypeError, ValueError):
                continue
        return 0
    prompt = integer("prompt_tokens", "input_tokens")
    completion = integer("completion_tokens", "output_tokens")
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": integer("total_tokens") or prompt + completion}


def _record_path(run_dir: Path, image_name: str, attempt: int) -> Path:
    stem = Path(image_name).stem
    digest = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:12]
    return run_dir / "responses" / f"{stem}-{digest}" / f"attempt-{attempt}.json"


def _page_metrics(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = compare_page_boxes(predictions, truth)
    strict_count = sum(
        not match["under_crop"] and not match["over_crop"]
        for match in comparison["matches"]
    )
    comparison["strict_complete_count"] = strict_count
    comparison["strict_complete_rate"] = strict_count / comparison["truth_count"] if comparison["truth_count"] else 1.0
    comparison["perfect_page"] = (
        comparison["extra_prediction_count"] == 0
        and comparison["missed_truth_count"] == 0
        and strict_count == comparison["truth_count"]
    )
    return comparison


def _stability(primary: list[dict[str, Any]], repeated: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = compare_page_boxes(primary, repeated, association_threshold=0.80)
    return {
        "same_count": comparison["predicted_count"] == comparison["truth_count"],
        "matched_count": comparison["matched_count"],
        "first_count": comparison["predicted_count"],
        "repeat_count": comparison["truth_count"],
        "mean_iou": comparison["mean_iou"],
        "exact_match": comparison["exact_count_page"] and comparison["matched_count"] == comparison["predicted_count"],
    }


def _load_manifest(path: Path, include_unlabeled: bool) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    # Windows PowerShell commonly writes JSON as UTF-8 with a BOM.  Accept
    # both that form and the regular UTF-8 files written by this project.
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("pages"), list):
        raise ValueError("标注清单必须是 schema_version=1 且包含 pages 数组")
    root = Path(manifest.get("image_root") or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    pages = [page for page in manifest["pages"] if include_unlabeled or page.get("status") == "labeled"]
    return manifest, pages, root


def _cost(usage: dict[str, int], input_price: float, output_price: float) -> float:
    return usage["input_tokens"] / 1_000_000 * input_price + usage["output_tokens"] / 1_000_000 * output_price


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.annotations).resolve()
    _, pages, image_root = _load_manifest(manifest_path, args.include_unlabeled)
    # A manifest may have been written by a legacy Windows shell that could not
    # preserve non-ASCII paths.  The explicit override also makes benchmark
    # runs portable without editing the source annotation manifest.
    image_root_override = getattr(args, "image_root", None)
    if image_root_override:
        image_root = Path(image_root_override).resolve()
    if args.max_pages is not None:
        pages = pages[:args.max_pages]
    requested_images = getattr(args, "images", None)
    if requested_images:
        requested = {name.strip() for name in requested_images.split(",") if name.strip()}
        pages = [page for page in pages if str(page.get("image", "")) in requested]
    if not pages:
        raise ValueError("没有可执行页面：默认只运行 status=labeled 的页面")

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        raise ValueError(f"未设置环境变量 {args.api_key_env}；不会在命令行参数中读取或保存密钥")
    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vlm_complete_question_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotations": str(manifest_path),
        "image_root": str(image_root),
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "structured_mode": args.structured_mode,
        "max_long_side": args.max_long_side,
        "repeats": args.repeats,
        "dry_run": args.dry_run,
        "pricing_cny_per_million_tokens": {"input": args.input_price, "output": args.output_price},
    }
    write_json_atomic(run_dir / "config.json", config)

    results = []
    for index, page in enumerate(pages, start=1):
        image_name = str(page.get("image", ""))
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        result: dict[str, Any] = {"image": image_name, "status": "failed", "attempts": []}
        if not image_path.is_file():
            result["error"] = "image_missing"
            results.append(result)
            continue
        try:
            data_uri, dimensions = prepare_image(image_path, args.max_long_side)
            result["dimensions"] = dimensions
            for attempt in range(1, args.repeats + 1):
                request_data = build_request(args.model, data_uri, args.structured_mode)
                record: dict[str, Any] = {"attempt": attempt, "request": {k: v for k, v in request_data.items() if k != "messages"}}
                if args.dry_run:
                    record.update({"status": "dry_run", "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
                else:
                    response, elapsed = call_openai_compatible(args.base_url, api_key, request_data, args.timeout)
                    parsed = _extract_json(response_content(response))
                    normalized = normalize_prediction(parsed, dimensions["original_width"], dimensions["original_height"])
                    usage = _usage(response)
                    record.update({
                        "status": "ok", "elapsed_seconds": round(elapsed, 3), "usage": usage,
                        "estimated_cost_cny": round(_cost(usage, args.input_price, args.output_price), 8),
                        "raw_response": response, "parsed_response": parsed, "prediction": normalized,
                    })
                write_json_atomic(_record_path(run_dir, image_name, attempt), record)
                result["attempts"].append(record)

            if args.dry_run:
                result["status"] = "dry_run"
            else:
                primary = result["attempts"][0]["prediction"]
                truth = page.get("questions", [])
                result["status"] = "ok"
                result["prediction"] = primary
                if args.write_previews:
                    result["visual_artifacts"] = write_visual_artifacts(image_path, image_name, primary, run_dir)
                result["metrics"] = _page_metrics(primary["questions"], truth) if page.get("status") == "labeled" else None
                if len(result["attempts"]) > 1:
                    result["stability"] = [_stability(primary["questions"], attempt["prediction"]["questions"])
                                           for attempt in result["attempts"][1:]]
        except (ValueError, RuntimeError, OSError, urllib.error.URLError) as error:
            result["error"] = str(error)
        results.append(result)
        print(f"[{index}/{len(pages)}] {image_name}: {result['status']}", flush=True)

    valid = [item for item in results if item.get("status") == "ok" and item.get("metrics")]
    totals = {key: sum(item["metrics"][key] for item in valid) for key in (
        "predicted_count", "truth_count", "matched_count", "extra_prediction_count", "missed_truth_count",
        "under_crop_count", "over_crop_count", "merged_prediction_count", "split_truth_count", "strict_complete_count",
    )}
    matched, truth_count = totals["matched_count"], totals["truth_count"]
    total_cost = sum(attempt.get("estimated_cost_cny", 0.0) for item in results for attempt in item["attempts"])
    total_seconds = sum(attempt.get("elapsed_seconds", 0.0) for item in results for attempt in item["attempts"])
    summary = {
        "page_count": len(valid), "failed_page_count": sum(item.get("status") == "failed" for item in results),
        "dry_run_page_count": sum(item.get("status") == "dry_run" for item in results), **totals,
        "precision": matched / totals["predicted_count"] if totals["predicted_count"] else 0.0,
        "question_recall": matched / truth_count if truth_count else 0.0,
        "strict_complete_crop_rate": totals["strict_complete_count"] / truth_count if truth_count else 0.0,
        "perfect_page_rate": sum(item["metrics"]["perfect_page"] for item in valid) / len(valid) if valid else 0.0,
        "estimated_cost_cny": round(total_cost, 8), "model_elapsed_seconds": round(total_seconds, 3),
    }
    stability = [metric for item in valid for metric in item.get("stability", [])]
    if stability:
        summary["repeat_stability"] = {
            "page_pairs": len(stability),
            "same_count_rate": sum(item["same_count"] for item in stability) / len(stability),
            "exact_match_rate": sum(item["exact_match"] for item in stability) / len(stability),
            "mean_iou": sum(item["mean_iou"] for item in stability) / len(stability),
        }
    report = {"schema_version": SCHEMA_VERSION, "config": config, "summary": summary, "pages": results}
    write_json_atomic(run_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="基准测试 VLM 对完整教材题目的语义坐标能力")
    parser.add_argument("--annotations", required=True, help="原始页面真值标注 JSON；默认只运行 status=labeled 页面")
    parser.add_argument("--image-root", help="覆盖标注清单中的图片根目录（Windows 中文路径兼容用）")
    parser.add_argument("--output", required=True, help="基准运行目录；保存配置、逐页原始响应与 report.json")
    parser.add_argument("--provider", default="qwen_openai_compatible")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="仅从此环境变量读取 API 密钥")
    parser.add_argument("--structured-mode", choices=("json_schema", "json_object", "none"), default="json_schema")
    parser.add_argument("--max-long-side", type=int, default=2200)
    parser.add_argument("--repeats", type=int, default=1, help="同页重复调用次数；大于 1 时报告坐标稳定性")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--input-price", type=float, default=1.0, help="每百万输入 token 的人民币规划价")
    parser.add_argument("--output-price", type=float, default=10.0, help="每百万输出 token 的人民币规划价")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--images", help="逗号分隔的图片文件名；只运行这些页面，用于失败重试")
    parser.add_argument("--include-unlabeled", action="store_true", help="允许只验证 API 连通性的未标注页面")
    parser.add_argument("--dry-run", action="store_true", help="只校验清单和图像预处理，不发送网络请求")
    parser.add_argument("--no-previews", dest="write_previews", action="store_false", help="不写叠框预览图和题目裁图")
    parser.set_defaults(write_previews=True)
    args = parser.parse_args()
    if args.max_long_side < 256 or args.repeats < 1 or args.timeout < 1:
        parser.error("--max-long-side 至少 256，--repeats 和 --timeout 必须为正数")
    try:
        report = run_benchmark(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    summary = report["summary"]
    print(
        "[VLM 基准] pages={page_count} failed={failed_page_count} recall={question_recall:.3f} "
        "strict={strict_complete_crop_rate:.3f} perfect_page={perfect_page_rate:.3f} cost≈¥{estimated_cost_cny:.4f}".format(**summary),
        flush=True,
    )


if __name__ == "__main__":
    main()
