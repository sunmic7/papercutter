#!/usr/bin/env python3
"""Create database-ready OCR JSON and same-ID verification crops.

Only manually accepted question/answer pairs are exported.  Public delivery IDs are
persistent, ASCII-alphanumeric identifiers stored in the annotation project, so a
question keeps the same filename and database key across repeated exports.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from review_tools import manual_annotation_data as data
from review_tools.export_manual_annotations import _crop, _question_crop, _write_image


OCRRunner = Callable[[str], dict]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_ocr(path: str) -> dict:
    """Run the existing local high-accuracy PaddleOCR engine and retain traceable lines."""
    import cut_questions_v4 as cutter

    items = cutter.ocr_image(path)
    lines = []
    for item in items:
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        box = item.get("bbox", [])
        normalized_box = []
        if isinstance(box, list) and box:
            try:
                normalized_box = [
                    int(round(min(point[0] for point in box))),
                    int(round(min(point[1] for point in box))),
                    int(round(max(point[0] for point in box))),
                    int(round(max(point[1] for point in box))),
                ]
            except (TypeError, ValueError, IndexError):
                normalized_box = []
        lines.append({
            "text": text,
            "confidence": round(float(item.get("conf", 0.0) or 0.0), 6),
            "bbox": normalized_box,
        })
    confidences = [line["confidence"] for line in lines if line["confidence"] > 0]
    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
        "engine": "PaddleOCRLocal",
    }


def _ocr_with_cache(record: dict, image_path: Path, runner: OCRRunner) -> dict:
    image_hash = _sha256(image_path)
    cached = record.get("ocr", {})
    if (
        isinstance(cached, dict)
        and cached.get("image_sha256") == image_hash
        and cached.get("status") == "success"
        and isinstance(cached.get("text"), str)
    ):
        return dict(cached, cached=True)
    try:
        result = runner(str(image_path))
        ocr = {
            "status": "success",
            "engine": str(result.get("engine", "PaddleOCRLocal")),
            "text": str(result.get("text", "") or ""),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "lines": result.get("lines", []) if isinstance(result.get("lines", []), list) else [],
            "image_sha256": image_hash,
            "cached": False,
        }
    except Exception as error:  # One bad crop must not discard the rest of a book.
        ocr = {
            "status": "failed", "engine": "PaddleOCRLocal", "text": "",
            "confidence": 0.0, "lines": [], "image_sha256": image_hash,
            "cached": False, "error": f"{type(error).__name__}: {error}",
        }
    record["ocr"] = {key: value for key, value in ocr.items() if key != "cached"}
    return ocr


def _source_info(page_index: int, page: dict, record: dict) -> dict:
    return {
        "page_index": page_index,
        "page_image": str(page.get("image", "")),
        "annotation_id": str(record.get("id", "")),
        "bbox": list(record.get("bbox", [])),
    }


def export_ocr_delivery(
    project_path: str, output_dir: str, *, ocr_runner: OCRRunner | None = None,
) -> dict:
    project_file = Path(project_path).resolve()
    project = data.load_project(project_file)
    data.ensure_delivery_ids(project)
    runner = ocr_runner or _local_ocr

    root = Path(output_dir).resolve()
    question_image_dir = root / "images" / "questions"
    answer_image_dir = root / "images" / "answers"
    question_json_dir = root / "json" / "questions"
    answer_json_dir = root / "json" / "answers"
    for directory in (question_image_dir, answer_image_dir, question_json_dir, answer_json_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for page_number in range(1, len(project.get("pages", [])) + 1):
        (question_image_dir / f"P{page_number:04d}").mkdir(parents=True, exist_ok=True)
        (question_json_dir / f"P{page_number:04d}").mkdir(parents=True, exist_ok=True)
    for page_number in range(1, len(project.get("answer_pages", [])) + 1):
        (answer_image_dir / f"P{page_number:04d}").mkdir(parents=True, exist_ok=True)
        (answer_json_dir / f"P{page_number:04d}").mkdir(parents=True, exist_ok=True)

    questions = {
        record["id"]: (page_index, page, record)
        for page_index, page, record in data.all_questions(project)
        if record.get("region_role", "question") == "question"
    }
    answers = {
        record["id"]: (page_index, page, record)
        for page_index, page, record in data.all_answers(project)
        if record.get("region_role", "question") == "question"
    }
    accepted_pairs = [
        pair for pair in project.get("pairs", [])
        if pair.get("status") == "accepted" and pair.get("answer_ids")
    ]
    answer_to_questions: dict[str, list[str]] = defaultdict(list)
    valid_pairs = []
    skipped = 0
    for pair in accepted_pairs:
        question_info = questions.get(pair.get("question_id"))
        answer_infos = [answers.get(answer_id) for answer_id in pair.get("answer_ids", [])]
        if not question_info or not answer_infos or any(info is None for info in answer_infos):
            skipped += 1
            continue
        valid_pairs.append((pair, question_info, answer_infos))
        question_delivery_id = question_info[2]["delivery_id"]
        for _page_index, _page, answer in answer_infos:
            answer_to_questions[answer["delivery_id"]].append(question_delivery_id)

    answer_documents: dict[str, dict] = {}
    failed_ocr = 0
    for _pair, _question_info, answer_infos in valid_pairs:
        for page_index, page, answer in answer_infos:
            answer_id = answer["delivery_id"]
            if answer_id in answer_documents:
                continue
            page_folder = f"P{page_index + 1:04d}"
            image_path = answer_image_dir / page_folder / f"{answer_id}.jpg"
            _write_image(image_path, _crop(data.source_path(project, page, True), answer["bbox"]))
            ocr = _ocr_with_cache(answer, image_path, runner)
            if ocr["status"] != "success":
                failed_ocr += 1
            document = {
                "schema_version": 1,
                "id": answer_id,
                "kind": "answer",
                "book_id": project["delivery_book_id"],
                "question_ids": list(dict.fromkeys(answer_to_questions[answer_id])),
                "text": ocr.get("text", ""),
                "ocr": ocr,
                "image_file": f"images/answers/{page_folder}/{answer_id}.jpg",
                "source": _source_info(page_index, page, answer),
            }
            answer_documents[answer_id] = document
            (answer_json_dir / page_folder / f"{answer_id}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8",
            )

    question_documents = []
    for pair, (page_index, page, question), answer_infos in valid_pairs:
        question_id = question["delivery_id"]
        page_folder = f"P{page_index + 1:04d}"
        image_path = question_image_dir / page_folder / f"{question_id}.jpg"
        _write_image(image_path, _question_crop(project, page, question))
        ocr = _ocr_with_cache(question, image_path, runner)
        if ocr["status"] != "success":
            failed_ocr += 1
        answer_ids = [answer["delivery_id"] for _index, _page, answer in answer_infos]
        document = {
            "schema_version": 1,
            "id": question_id,
            "kind": "question",
            "book_id": project["delivery_book_id"],
            "relation_id": pair["delivery_id"],
            "number_path": data.number_path(question),
            "display_number": data.display_number(question),
            "text": ocr.get("text", ""),
            "ocr": ocr,
            "answer_ids": answer_ids,
            "answers": [
                {"id": answer_id, "text": answer_documents[answer_id]["text"]}
                for answer_id in answer_ids
            ],
            "image_file": f"images/questions/{page_folder}/{question_id}.jpg",
            "answer_image_files": [answer_documents[answer_id]["image_file"] for answer_id in answer_ids],
            "source": _source_info(page_index, page, question),
        }
        question_documents.append(document)
        (question_json_dir / page_folder / f"{question_id}.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    database_payload = {
        "schema_version": 1,
        "kind": "papercutter_ocr_delivery",
        "book": {
            "id": project["delivery_book_id"],
            "source_root": str(project.get("book_root", project.get("image_root", ""))),
            "isbn": str(project.get("isbn", "")),
            "subject": str(project.get("subject", "")),
            "grade": str(project.get("grade", "")),
        },
        "questions": question_documents,
        "answers": list(answer_documents.values()),
    }
    (root / "database_import.json").write_text(
        json.dumps(database_payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    index = {
        "schema_version": 1,
        "book_id": project["delivery_book_id"],
        "question_count": len(question_documents),
        "answer_count": len(answer_documents),
        "skipped_pair_count": skipped,
        "failed_ocr_count": failed_ocr,
        "database_file": "database_import.json",
        "items": [
            {
                "question_id": item["id"], "answer_ids": item["answer_ids"],
                "question_json": f"json/questions/P{item['source']['page_index'] + 1:04d}/{item['id']}.json",
                "question_image": item["image_file"],
            }
            for item in question_documents
        ],
    }
    (root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    data.save_project(project, project_file, increment_revision=False)
    return index
