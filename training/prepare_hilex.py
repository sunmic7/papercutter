#!/usr/bin/env python3
"""Prepare the HiLEx dataset as a single-class complete-question dataset.

HiLEx contains nested boxes.  For question cutting we retain only
``Question_Answer_Block`` (class 3 in the supplied YOLO data), because it is
the only annotation representing a complete question item.  Keeping the
question-text box as the same class would train duplicate boxes for one item.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


HILEX_COMPLETE_QUESTION_CLASS = 3
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
        if class_id != HILEX_COMPLETE_QUESTION_CLASS:
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


def prepare_hilex(source: str | Path, output: str | Path, *, overwrite: bool = False) -> dict:
    """Copy HiLEx images and project labels to one ``question`` class."""
    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"HiLEx YOLO directory was not found: {source_root}")

    _ensure_empty_directory(output_root, overwrite)
    summary: dict[str, object] = {
        "schema_version": 1,
        "source": str(source_root),
        "target_class": "question",
        "source_class": "Question_Answer_Block",
        "source_class_id": HILEX_COMPLETE_QUESTION_CLASS,
        "splits": {},
    }

    for split in SPLITS:
        source_images = source_root / split / "images"
        source_labels = source_root / split / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Invalid HiLEx split structure: {source_root / split}")
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

    data_yaml = (
        f"path: {output_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: question\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HiLEx to one-class complete-question YOLO data")
    parser.add_argument("source", help="HiLex_Yolo_Format directory")
    parser.add_argument("--output", required=True, help="New dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    args = parser.parse_args()
    summary = prepare_hilex(args.source, args.output, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

