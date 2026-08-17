#!/usr/bin/env python3
"""Export accepted manual question/answer pairs as local training samples."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from review_tools import manual_annotation_data as data


def _crop(path: str, box: list[int]):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    x0, y0, x1, y1 = (int(value) for value in box)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"无效框：{box}")
    return image[y0:y1, x0:x1]


def _write_image(path: Path, image) -> None:
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("图片编码失败")
    payload.tofile(str(path))


def _stack_segments(top, bottom, gap: int = 0):
    """Compose adjacent source segments edge-to-edge without an artificial seam."""
    width = max(top.shape[1], bottom.shape[1])

    def pad(image):
        if image.shape[1] == width:
            return image
        canvas = np.full((image.shape[0], width, 3), 255, dtype=np.uint8)
        canvas[:, :image.shape[1]] = image
        return canvas

    pieces = [pad(top)]
    if gap > 0:
        pieces.append(np.full((gap, width, 3), 255, dtype=np.uint8))
    pieces.append(pad(bottom))
    return np.vstack(pieces)


def _page_index(project: dict, page: dict) -> int:
    for index, candidate in enumerate(project.get("pages", [])):
        if candidate is page or candidate.get("image") == page.get("image"):
            return index
    return -1


def _question_crop(project: dict, page: dict, question: dict):
    path = data.source_path(project, page)
    own = _crop(path, question["bbox"])
    own_page_index = _page_index(project, page)
    pieces = []
    context_segments = question.get("shared_context_segments", []) if question.get("use_shared_context") else []
    for segment in context_segments:
        if not isinstance(segment, dict):
            continue
        page_index = int(segment.get("page_index", -1))
        box = segment.get("bbox", [])
        if page_index < 0 or page_index >= len(project.get("pages", [])) or len(box) != 4:
            continue
        if page_index == own_page_index:
            qx0, qy0, qx1, qy1 = (int(value) for value in question["bbox"])
            sx0, sy0, sx1, sy1 = (int(value) for value in box)
            if qx0 <= sx0 and qy0 <= sy0 and qx1 >= sx1 and qy1 >= sy1:
                continue
        segment_page = project["pages"][page_index]
        pieces.append(_crop(data.source_path(project, segment_page), box))
    shared = question.get("shared_context_bbox", [])
    if not pieces and question.get("use_shared_context") and isinstance(shared, list) and len(shared) == 4:
        qx0, qy0, qx1, qy1 = (int(value) for value in question["bbox"])
        sx0, sy0, sx1, sy1 = (int(value) for value in shared)
        if not (qx0 <= sx0 and qy0 <= sy0 and qx1 >= sx1 and qy1 >= sy1):
            pieces.append(_crop(path, shared))
    pieces.append(own)
    continuations = [
        (continuation_page, continuation)
        for _index, continuation_page, continuation in data.all_questions(project)
        if continuation.get("region_role") == "continuation"
        and continuation.get("continuation_of") == question.get("id")
    ]
    for continuation_page, continuation in continuations:
        pieces.append(_crop(data.source_path(project, continuation_page), continuation["bbox"]))
    composed = pieces[0]
    for piece in pieces[1:]:
        composed = _stack_segments(composed, piece)
    return composed


def export_project(project_path: str, output_dir: str) -> dict:
    project = data.load_project(project_path)
    questions = {
        record["id"]: (page, record) for _, page, record in data.all_questions(project)
        if record.get("region_role", "question") == "question"
    }
    answers = {
        record["id"]: (page, record) for _, page, record in data.all_answers(project)
        if record.get("region_role", "question") == "question"
    }
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    exported, skipped, index = 0, 0, []
    for pair in project.get("pairs", []):
        if pair.get("status") != "accepted" or not pair.get("answer_ids"):
            continue
        question_info = questions.get(pair.get("question_id"))
        answer_infos = [answers.get(answer_id) for answer_id in pair.get("answer_ids", [])]
        if not question_info or any(info is None for info in answer_infos):
            skipped += 1; continue
        sample_id = f"{exported + 1:06d}_{re.sub(r'[^A-Za-z0-9_-]', '', pair['id'])[-12:]}"
        sample = root / sample_id; sample.mkdir(exist_ok=False)
        q_page, question = question_info
        try:
            _write_image(sample / "question.jpg", _question_crop(project, q_page, question))
            answer_files = []
            for pos, (a_page, answer) in enumerate(answer_infos, 1):
                name = f"answer_{pos}.jpg"; answer_files.append(name)
                _write_image(sample / name, _crop(data.source_path(project, a_page, True), answer["bbox"]))
        except (OSError, ValueError) as exc:
            skipped += 1
            (sample / "export_error.txt").write_text(str(exc), encoding="utf-8")
            continue
        metadata = {"pair": pair, "question": question, "answers": [info[1] for info in answer_infos], "answer_files": answer_files}
        (sample / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({
            "id": sample_id,
            "path": sample_id,
            "question_id": question["id"],
            "answer_ids": pair["answer_ids"],
            "display_number": data.display_number(question),
            "major_title": question.get("major_title", question.get("major", "")),
        })
        exported += 1
    report = {"schema_version": 1, "source_project": str(Path(project_path).resolve()), "exported": exported, "skipped": skipped, "samples": index}
    (root / "manual_annotation_index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="导出人工标注的题目—答案训练样本")
    parser.add_argument("project", help="人工标注项目 JSON")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()
    report = export_project(args.project, args.output)
    print(f"已导出 {report['exported']} 对，跳过 {report['skipped']} 对：{Path(args.output) / 'manual_annotation_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
