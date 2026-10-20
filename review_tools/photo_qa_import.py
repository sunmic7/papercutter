#!/usr/bin/env python3
"""Import already-separated photographed question/answer images as QA data.

This workflow deliberately does not cut images.  A filename such as
``5217答案2.png`` is only a *candidate* for question ``5217.png``: a visual
reviewer must still decide whether it is a continuation, a supporting diagram,
or content which belongs to another question.  Only high-confidence, complete
records enter ``dataset.jsonl``; everything else remains in ``review_queue``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANSWER_NAME = re.compile(r"^(?P<source_id>\d+)答案(?P<part>\d*)$", re.IGNORECASE)
QUESTION_NAME = re.compile(r"^(?P<source_id>\d+)$")
SEMANTIC_ROLES = {"answer", "continuation", "answer_figure", "unrelated", "unknown"}
REQUIRED_QA_FIELDS = ("question_text", "answer", "cot")

SemanticReviewer = Callable[[Path, list[Path]], dict[str, Any] | None]
OCRRunner = Callable[[Path], str]


def _atomic_json(path: Path, value: Any, *, lines: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if lines:
        content = "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in value)
        temporary.write_text(content + ("\n" if content else ""), encoding="utf-8")
    else:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _image_order(path: Path) -> tuple[int, int, str]:
    answer = ANSWER_NAME.fullmatch(path.stem.strip())
    if answer:
        part = answer.group("part")
        return (int(answer.group("source_id")), int(part) if part else 0, path.name.lower())
    question = QUESTION_NAME.fullmatch(path.stem.strip())
    if question:
        return (int(question.group("source_id")), -1, path.name.lower())
    return (10**12, 10**12, path.name.lower())


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # Required by the customer's target schema.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path):
    """Decode Windows paths containing Chinese characters reliably."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    except OSError:
        return None


def image_info(path: Path) -> dict[str, Any]:
    image = _read_image(path)
    if image is None or image.size == 0:
        return {
            "path": str(path), "name": path.name, "readable": False,
            "MD5": _md5(path), "image_size": {"height": 0, "width": 0},
            "quality_flags": ["image_unreadable"],
        }
    height, width = image.shape[:2]
    flags = []
    if min(width, height) < 720:
        flags.append("short_edge_below_720")
    # This is intentionally a review marker rather than a pass/fail blur
    # threshold: formula photos have legitimate high-frequency line drawings.
    sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharpness < 18.0:
        flags.append("possible_blur")
    return {
        "path": str(path), "name": path.name, "readable": True,
        "MD5": _md5(path), "image_size": {"height": int(height), "width": int(width)},
        "quality_flags": flags, "sharpness": round(sharpness, 3),
    }


def parse_named_images(root: str | Path) -> dict[str, Any]:
    """Return filename-based candidate groups without treating them as matches."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"导入目录不存在：{base}")
    questions: dict[str, Path] = {}
    candidates: dict[str, list[Path]] = defaultdict(list)
    unrecognised: list[Path] = []
    for path in sorted((item for item in base.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS), key=_image_order):
        stem = path.stem.strip()
        answer = ANSWER_NAME.fullmatch(stem)
        question = QUESTION_NAME.fullmatch(stem)
        if answer:
            candidates[answer.group("source_id")].append(path)
        elif question:
            source_id = question.group("source_id")
            if source_id in questions:
                unrecognised.append(path)
            else:
                questions[source_id] = path
        else:
            unrecognised.append(path)
    return {
        "root": base, "questions": questions,
        "candidates": {source_id: sorted(paths, key=_image_order) for source_id, paths in candidates.items()},
        "unrecognised": sorted(unrecognised, key=lambda item: item.name.lower()),
    }


def local_ocr(path: Path) -> str:
    """Read a full already-separated image using the current local OCR engine."""
    import cut_questions_v4 as cutter

    rows = cutter.ocr_image(str(path))
    return "\n".join(str(row.get("text", "") or "").strip() for row in rows if str(row.get("text", "") or "").strip())


def _clean(value: Any) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(value or "").strip())


def _normalise_review(raw: dict[str, Any] | None, candidates: list[Path]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    by_name = {path.name: path for path in candidates}
    candidate_rows = raw.get("candidates", [])
    roles: dict[str, dict[str, Any]] = {}
    if isinstance(candidate_rows, list):
        for item in candidate_rows:
            if not isinstance(item, dict):
                continue
            name = str(item.get("filename", "")).strip()
            if name not in by_name:
                continue
            role = str(item.get("role", "unknown")).strip().lower()
            if role not in SEMANTIC_ROLES:
                role = "unknown"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            roles[name] = {"role": role, "confidence": round(confidence, 4)}
    for path in candidates:
        roles.setdefault(path.name, {"role": "unknown", "confidence": 0.0})
    try:
        confidence = min(1.0, max(0.0, float(raw.get("overall_confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "question_text": _clean(raw.get("question_text")),
        "answer": _clean(raw.get("answer")),
        "cot": _clean(raw.get("cot")),
        "image_domain": _clean(raw.get("image_domain")),
        "QA_diff": _clean(raw.get("QA_diff")),
        "overall_confidence": round(confidence, 4),
        "candidates": roles,
    }


def _fallback_review(question_ocr: str, answer_ocr: dict[str, str], candidates: list[Path]) -> dict[str, Any]:
    return {
        "question_text": question_ocr,
        "answer": "",
        "cot": "\n".join(answer_ocr.get(path.name, "") for path in candidates if answer_ocr.get(path.name, "")),
        "image_domain": "物理",
        "QA_diff": "大学",
        "overall_confidence": 0.0,
        "candidates": {path.name: {"role": "unknown", "confidence": 0.0} for path in candidates},
    }


def build_batch(
    root: str | Path,
    *,
    semantic_reviewer: SemanticReviewer | None = None,
    ocr_runner: OCRRunner | None = None,
    image_domain: str = "物理",
    qa_diff: str = "大学",
    confidence_threshold: float = 0.88,
) -> dict[str, Any]:
    """Build one immutable candidate manifest and a conservative final dataset."""
    parsed = parse_named_images(root)
    run_ocr = ocr_runner or local_ocr
    items, review_queue, final_records = [], [], []
    for source_id, question_path in sorted(parsed["questions"].items(), key=lambda item: int(item[0])):
        candidate_paths = parsed["candidates"].get(source_id, [])
        question_meta = image_info(question_path)
        candidate_meta = {path.name: image_info(path) for path in candidate_paths}
        try:
            question_ocr = _clean(run_ocr(question_path))
        except Exception as error:
            question_ocr = ""
            question_meta["ocr_error"] = f"{type(error).__name__}: {error}"
        candidate_ocr: dict[str, str] = {}
        for path in candidate_paths:
            try:
                candidate_ocr[path.name] = _clean(run_ocr(path))
            except Exception as error:
                candidate_ocr[path.name] = ""
                candidate_meta[path.name]["ocr_error"] = f"{type(error).__name__}: {error}"

        raw_review = semantic_reviewer(question_path, candidate_paths) if semantic_reviewer else None
        review = _normalise_review(raw_review, candidate_paths) or _fallback_review(question_ocr, candidate_ocr, candidate_paths)
        review["image_domain"] = review["image_domain"] or image_domain
        review["QA_diff"] = review["QA_diff"] or qa_diff
        accepted_answer_paths = [
            path for path in candidate_paths
            if review["candidates"][path.name]["role"] in {"answer", "continuation"}
            and review["candidates"][path.name]["confidence"] >= confidence_threshold
        ]
        support_paths = [path for path in candidate_paths if review["candidates"][path.name]["role"] == "answer_figure"]
        flags = list(question_meta["quality_flags"])
        if not candidate_paths:
            flags.append("no_named_answer_candidate")
        if not accepted_answer_paths:
            flags.append("no_confirmed_answer_page")
        if any(review["candidates"][path.name]["role"] in {"unrelated", "unknown"} for path in candidate_paths):
            flags.append("candidate_assignment_unresolved")
        if review["overall_confidence"] < confidence_threshold:
            flags.append("semantic_confidence_below_threshold")
        if any(not review[key] for key in REQUIRED_QA_FIELDS):
            flags.append("qa_content_incomplete")

        record = {
            "batch": parsed["root"].name,
            "source_id": source_id,
            "question_image": question_meta,
            "answer_candidates": [
                {**candidate_meta[path.name], "ocr_text": candidate_ocr.get(path.name, ""), **review["candidates"][path.name]}
                for path in candidate_paths
            ],
            "accepted_answer_images": [path.name for path in accepted_answer_paths],
            "support_images": [path.name for path in support_paths],
            "question_ocr": question_ocr,
            "semantic": {key: review[key] for key in ("question_text", "answer", "cot", "image_domain", "QA_diff", "overall_confidence")},
            "quality_flags": sorted(set(flags)),
        }
        eligible = not record["quality_flags"]
        record["eligible"] = eligible
        items.append(record)
        if eligible:
            final_records.append({
                "MD5": question_meta["MD5"],
                "image_domain": review["image_domain"],
                "image_size": question_meta["image_size"],
                "QA_diff": review["QA_diff"],
                "question": review["question_text"],
                "opinion": "",
                "answer": review["answer"],
                "COT": review["cot"],
            })
        else:
            review_queue.append(record)

    for source_id, paths in sorted(parsed["candidates"].items(), key=lambda item: int(item[0])):
        if source_id not in parsed["questions"]:
            review_queue.append({
                "batch": parsed["root"].name, "source_id": source_id,
                "reason": "answer_named_file_without_question_image",
                "files": [path.name for path in paths], "eligible": False,
            })
    for path in parsed["unrecognised"]:
        review_queue.append({
            "batch": parsed["root"].name, "reason": "unrecognised_filename",
            "files": [path.name], "eligible": False,
        })
    return {
        "schema_version": 1, "batch": parsed["root"].name,
        "source_root": str(parsed["root"]), "items": items,
        "dataset": final_records, "review_queue": review_queue,
    }


def write_batch(batch: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve() / str(batch["batch"])
    _atomic_json(root / "candidates.jsonl", batch["items"], lines=True)
    _atomic_json(root / "review_queue.json", batch["review_queue"])
    _atomic_json(root / "dataset.jsonl", batch["dataset"], lines=True)
    report = {
        "schema_version": 1, "batch": batch["batch"], "source_root": batch["source_root"],
        "question_count": len(batch["items"]), "eligible_count": len(batch["dataset"]),
        "review_count": len(batch["review_queue"]), "dataset_file": str(root / "dataset.jsonl"),
        "review_file": str(root / "review_queue.json"),
    }
    _atomic_json(root / "report.json", report)
    return report


def _qwen_reviewer(question_path: Path, candidate_paths: list[Path]) -> dict[str, Any] | None:
    from review_tools.photo_qa_vlm import review_group

    question = _read_image(question_path)
    candidates = [(path.name, _read_image(path)) for path in candidate_paths]
    if question is None or any(image is None for _name, image in candidates):
        return None
    return review_group(question, candidates)


def run_batches(
    roots: Iterable[str | Path], output_root: str | Path, *, semantic_review: bool = False,
    ocr_runner: OCRRunner | None = None,
) -> list[dict[str, Any]]:
    reviewer = _qwen_reviewer if semantic_review else None
    return [write_batch(build_batch(root, semantic_reviewer=reviewer, ocr_runner=ocr_runner), output_root) for root in roots]


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperFlow 已命名拍照 QA 导入器（不切题）")
    parser.add_argument("roots", nargs="+", help="每套题目的根目录")
    parser.add_argument("--output", required=True, help="输出根目录；不会修改源图片")
    parser.add_argument("--semantic-review", action="store_true", help="使用已配置的 Qwen 视觉模型做题答归属与文字归纳")
    args = parser.parse_args()
    reports = run_batches(args.roots, args.output, semantic_review=args.semantic_review)
    print(json.dumps({"ok": True, "batches": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
