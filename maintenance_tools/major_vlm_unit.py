#!/usr/bin/env python3
"""Model-first cutter for one paper unit at printed-major-question granularity.

This validation path intentionally keeps every small question, reading material,
diagram and answer area under its printed Chinese major heading.  A major
question which continues over two source pages is stitched into one image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import vlm_hybrid


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _choose_major(passes):
    candidates = {}
    for pass_no, result in enumerate(passes, 1):
        for raw in result.get("items", []):
            major = raw["major_number"]
            if major <= 0:
                continue
            item = dict(raw)
            item["passes"] = [pass_no]
            previous = candidates.get(major)
            if previous is None:
                candidates[major] = item
                continue
            previous["passes"].append(pass_no)
            if _iou(previous["bbox"], item["bbox"]) < 0.45:
                previous["disagreement"] = True
                # A tiny heading-only box can be returned with an unjustified
                # confidence of 1.0.  At major-question granularity, retain
                # the larger proposal on a real disagreement; heading snapping
                # later removes harmless leading material, but missing small
                # questions cannot be reconstructed.
                old_area = (previous["bbox"][2] - previous["bbox"][0]) * (previous["bbox"][3] - previous["bbox"][1])
                new_area = (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])
                if new_area > old_area:
                    item["passes"] = previous["passes"]
                    item["disagreement"] = True
                    candidates[major] = item
            elif item["confidence"] > previous["confidence"]:
                item["passes"] = previous["passes"]
                candidates[major] = item
    return candidates


def _write_jpeg(path: Path, image) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"cannot encode {path.name}")
    encoded.tofile(str(path))


def _crop(image, box):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    # Major boundaries must be kept complete; a small outer margin prevents
    # glyphs on a boundary from being shaved off without reaching the next item.
    margin = 16
    # A vertical margin routinely leaked the next red major heading (or the
    # preceding line) into an otherwise correct model boundary.  Keep only a
    # horizontal anti-clipping margin; vertical boundaries stay model-exact.
    x0, x1 = max(0, x0 - margin), min(width, x1 + margin)
    return image[y0:y1, x0:x1], [x0, y0, x1, y1]


def _snap_to_red_heading(image, box):
    """Trim a near-miss top boundary to the first red printed major heading."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    # Red heading ink is a strong visual anchor on these worksheets.  Search
    # only shortly below the model proposal, so red material inside a question
    # cannot move a boundary across a whole question group.
    search_end = min(y1, y0 + 420)
    region = image[y0:search_end, max(0, x0):min(width, x1)]
    if region.size == 0:
        return box
    blue, green, red = cv2.split(region)
    mask = ((red.astype(np.int16) > 120) &
            (red.astype(np.int16) > green.astype(np.int16) * 1.35) &
            (red.astype(np.int16) > blue.astype(np.int16) * 1.35))
    row_counts = mask.sum(axis=1)
    rows = np.flatnonzero(row_counts >= 8)
    if not len(rows):
        return box
    return [x0, y0 + int(rows[0]), x1, y1]


def _keep_heading_half(image, box):
    """For a spread-crossing proposal, keep the physical page of its heading."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    split = width // 2
    if not (x0 < split < x1):
        return box
    region = image[y0:min(y1, y0 + 420), x0:x1]
    if region.size == 0:
        return box
    blue, green, red = cv2.split(region)
    mask = ((red.astype(np.int16) > 120) &
            (red.astype(np.int16) > green.astype(np.int16) * 1.35) &
            (red.astype(np.int16) > blue.astype(np.int16) * 1.35))
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return box
    # A heading concentrated on the right page means a model accidentally
    # absorbed lower-left material belonging to the preceding major question.
    global_x = xs + x0
    if float(np.mean(global_x >= split)) >= 0.85:
        return [split, y0, x1, y1]
    return box


def _stitch(parts):
    if len(parts) == 1:
        return parts[0]
    width = max(part.shape[1] for part in parts)
    gap = 28
    total_height = sum(part.shape[0] for part in parts) + gap * (len(parts) - 1)
    canvas = np.full((total_height, width, 3), 255, dtype=np.uint8)
    cursor = 0
    for part in parts:
        canvas[cursor:cursor + part.shape[0], :part.shape[1]] = part
        cursor += part.shape[0] + gap
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--pages", required=True, help="source images in paper reading order")
    args = parser.parse_args()
    if not vlm_hybrid.enabled() or not vlm_hybrid.thinking_enabled():
        raise RuntimeError("set PAPERCUTTER_VLM_MODE=assist and PAPERCUTTER_VLM_THINKING=on")

    source, output = Path(args.source), Path(args.output)
    target = output / "\u7b2c\u4e00\u5355\u5143_\u5927\u9898\u6a21\u578b\u4e3b\u5207_\u9898\u76ee"
    target.mkdir(parents=True, exist_ok=True)
    major_parts, audit, page_layouts = {}, [], []

    for page_order, raw_name in enumerate(args.pages.split(","), 1):
        name = raw_name.strip()
        path = source / name
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read {path}")
        passes = [vlm_hybrid.analyze_major_cuts(image) or {"items": []} for _ in range(2)]
        selected = _choose_major(passes)
        for item in selected.values():
            item["bbox"] = _keep_heading_half(image, item["bbox"])
        audit.append({"page": name, "pass_item_counts": [len(p.get("items", [])) for p in passes],
                      "selected_majors": sorted(selected)})
        for major, item in selected.items():
            snapped_box = _snap_to_red_heading(image, item["bbox"])
            crop, source_box = _crop(image, snapped_box)
            if crop.size:
                major_parts.setdefault(major, []).append({"page_order": page_order, "page": name,
                                                          "crop": crop, "source_box": source_box,
                                                          "model": {**item, "heading_snapped": snapped_box != item["bbox"]}})
        page_layouts.append({"order": page_order, "name": name, "image": image, "selected": selected})

    # A major group can be L-shaped in a two-page spread: its left page keeps
    # flowing below the point where the following major begins on the right
    # page.  Represent it as two non-overlapping page regions in reading order
    # rather than either losing its tail or including the next major question.
    for layout in page_layouts:
        image, selected = layout["image"], layout["selected"]
        height, width = image.shape[:2]
        split = width // 2
        for major, item in selected.items():
            if not (item["bbox"][0] < split < item["bbox"][2] and item["bbox"][3] < height * 0.90):
                continue
            following_right = [other for other_major, other in selected.items()
                               if other_major > major and other["bbox"][0] >= split]
            if not following_right:
                continue
            cutoff = min(other["bbox"][1] for other in following_right)
            if abs(item["bbox"][3] - cutoff) > 90:
                continue
            # Remove the rectangular candidate for this page and replace it.
            kept = [part for part in major_parts.get(major, [])
                    if not (part["page_order"] == layout["order"] and part["page"] == layout["name"])]
            left_box = _snap_to_red_heading(image, [item["bbox"][0], item["bbox"][1], split, height])
            left_crop, left_source_box = _crop(image, left_box)
            right_crop = image[:cutoff, split:item["bbox"][2]]
            kept.extend([
                {"page_order": layout["order"], "page": layout["name"], "crop": left_crop,
                 "source_box": left_source_box,
                 "model": {**item, "layout_repaired": "left_page_tail"}},
                {"page_order": layout["order"], "page": layout["name"], "crop": right_crop,
                 "source_box": [split, 0, item["bbox"][2], cutoff],
                 "model": {**item, "layout_repaired": "right_page_head"}},
            ])
            major_parts[major] = kept

    # When a spread is scanned as two facing pages, a major question can end at
    # the bottom of the right page and continue at the top of the next image's
    # left page.  The model normally returns that continuation itself.  If it
    # does not, preserve the only structurally possible unclaimed band instead
    # of silently dropping it.  The audit marks this conservative repair.
    for previous, current in zip(page_layouts, page_layouts[1:]):
        prior_image, next_image = previous["image"], current["image"]
        prior_height, prior_width = prior_image.shape[:2]
        first_next_y = min((item["bbox"][1] for item in current["selected"].values()), default=0)
        touching_bottom = [
            (major, item) for major, item in previous["selected"].items()
            if item["bbox"][3] >= prior_height * 0.90 and item["bbox"][0] >= prior_width * 0.45
        ]
        if not touching_bottom or first_next_y < 40:
            continue
        major, item = max(touching_bottom, key=lambda pair: pair[0])
        # The previous question occupied the right-facing page, so its next
        # physical page is the left half of the following spread.
        continuation = next_image[:first_next_y, :next_image.shape[1] // 2]
        if continuation.size:
            major_parts.setdefault(major, []).append({
                "page_order": current["order"], "page": current["name"], "crop": continuation,
                "source_box": [0, 0, next_image.shape[1] // 2, first_next_y],
                "model": {"id": "inferred_continuation", "major_number": major,
                          "bbox": [0, 0, next_image.shape[1] // 2, first_next_y],
                          "confidence": None, "continuation_inferred": True},
            })

    records = []
    for sequence, major in enumerate(sorted(major_parts), 1):
        parts = sorted(major_parts[major], key=lambda part: part["page_order"])
        image = _stitch([part["crop"] for part in parts])
        filename = f"{sequence:02d}_\u5927\u9898{major}.jpg"
        _write_jpeg(target / filename, image)
        records.append({"major_number": major, "file": filename,
                        "parts": [{key: value for key, value in part.items() if key != "crop"} for part in parts]})

    report = {"major_question_count": len(records), "items": records, "passes": audit}
    (output / "\u7b2c\u4e00\u5355\u5143_\u5927\u9898\u6a21\u578b\u4e3b\u5207_\u5ba1\u8ba1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"major_question_count": len(records), "audit": audit, "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
