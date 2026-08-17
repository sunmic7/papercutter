#!/usr/bin/env python3
"""One-unit direct VLM cutter used to validate model-first question crops.

It deliberately does not use the legacy OCR question splitter.  Two independent
Qwen thinking passes judge each source page; their numbered item lists are
merged, cropped, and recorded for visual audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import vlm_hybrid


def _iou(left, right):
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union else 0.0


def _merge_passes(passes):
    merged = {}
    for pass_index, result in enumerate(passes, 1):
        for item in result.get("items", []):
            key = (item["major_number"], item["question_number"])
            candidate = dict(item)
            candidate["passes"] = [pass_index]
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue
            existing["passes"].append(pass_index)
            # Prefer the tighter high-confidence box when the two passes agree;
            # otherwise keep the larger box so shared stems are not clipped.
            if _iou(existing["bbox"], candidate["bbox"]) >= 0.35:
                if candidate["confidence"] > existing["confidence"]:
                    candidate["passes"] = existing["passes"]
                    merged[key] = candidate
            else:
                # A union is unsafe: it often pulls the preceding question
                # into a tight model crop.  Retain the tighter proposal and
                # flag it for visual audit instead.
                old_area = (existing["bbox"][2] - existing["bbox"][0]) * (existing["bbox"][3] - existing["bbox"][1])
                new_area = (candidate["bbox"][2] - candidate["bbox"][0]) * (candidate["bbox"][3] - candidate["bbox"][1])
                if new_area < old_area:
                    candidate["passes"] = existing["passes"]
                    candidate["disagreement"] = True
                    merged[key] = candidate
                else:
                    existing["disagreement"] = True
    return sorted(merged.values(), key=lambda item: (item["major_number"], item["question_number"]))


def _write_jpeg(path: Path, image) -> bool:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False


def cut_page(path: Path, destination: Path, sequence_start: int):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    passes = [vlm_hybrid.analyze_direct_cuts(image) or {"items": []} for _ in range(2)]
    items = _merge_passes(passes)
    height, width = image.shape[:2]
    records = []
    for offset, item in enumerate(items):
        x0, y0, x1, y1 = item["bbox"]
        margin = 12
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(width, x1 + margin), min(height, y1 + margin)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        sequence = sequence_start + offset
        filename = f"{sequence:02d}_大题{item['major_number']}_小题{item['question_number']}.jpg"
        if not _write_jpeg(destination / filename, crop):
            raise RuntimeError(f"cannot write {filename}")
        records.append({**item, "source_image": str(path), "source_box": [x0, y0, x1, y1], "file": filename})
    return records, passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--pages", required=True, help="comma-separated image file names")
    args = parser.parse_args()
    if not vlm_hybrid.enabled() or not vlm_hybrid.thinking_enabled():
        raise RuntimeError("set PAPERCUTTER_VLM_MODE=assist and PAPERCUTTER_VLM_THINKING=on")
    source, output = Path(args.source), Path(args.output)
    questions = output / "第一单元_模型主切_题目"
    questions.mkdir(parents=True, exist_ok=True)
    all_records, audits, sequence = [], [], 1
    for name in args.pages.split(","):
        records, passes = cut_page(source / name.strip(), questions, sequence)
        sequence += len(records)
        all_records.extend(records)
        audits.append({"page": name.strip(), "pass_item_counts": [len(value.get("items", [])) for value in passes]})
    (output / "第一单元_模型主切_审计.json").write_text(
        json.dumps({"question_count": len(all_records), "items": all_records, "passes": audits}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"question_count": len(all_records), "audit": audits, "output": str(questions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
