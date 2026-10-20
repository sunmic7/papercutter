#!/usr/bin/env python3
"""Prepare the Exam Question Detector data as one complete-question class.

The source set labels two visual variants of an exam question (``questions``
and ``questions_o``).  Both boxes cover a complete, individually cuttable
question, so the cutter should learn one conservative ``question`` class.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SOURCE_QUESTION_CLASS_IDS = frozenset({0, 1})
SPLITS = ("train", "valid", "test")


def _read_question_lines(label_path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        if class_id not in SOURCE_QUESTION_CLASS_IDS:
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            continue
        lines.append("0 " + " ".join(fields[1:]))
    return lines


def _ensure_empty_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_exam_questions(source: str | Path, output: str | Path, *, overwrite: bool = False) -> dict:
    """Copy the source set and normalize both question variants to class 0."""
    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Exam Question Detector directory was not found: {source_root}")

    _ensure_empty_directory(output_root, overwrite)
    summary: dict[str, object] = {
        "schema_version": 1,
        "source": str(source_root),
        "target_class": "question",
        "source_classes": {"0": "questions", "1": "questions_o"},
        "splits": {},
    }
    for split in SPLITS:
        source_images = source_root / split / "images"
        source_labels = source_root / split / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Invalid Exam Question Detector split: {source_root / split}")
        target_images = output_root / split / "images"
        target_labels = output_root / split / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        image_count = box_count = empty_count = 0
        for image_path in sorted(path for path in source_images.iterdir() if path.is_file()):
            label_path = source_labels / f"{image_path.stem}.txt"
            lines = _read_question_lines(label_path) if label_path.is_file() else []
            shutil.copy2(image_path, target_images / image_path.name)
            (target_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            image_count += 1
            box_count += len(lines)
            empty_count += int(not lines)
        summary["splits"][split] = {
            "images": image_count,
            "question_boxes": box_count,
            "empty_images": empty_count,
        }

    (output_root / "data.yaml").write_text(
        f"path: {output_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: question\n",
        encoding="utf-8",
    )
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Exam Question Detector to one-class question data"
    )
    parser.add_argument("source", help="Source YOLO directory")
    parser.add_argument("--output", required=True, help="New dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    args = parser.parse_args()
    print(json.dumps(
        prepare_exam_questions(args.source, args.output, overwrite=args.overwrite),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
