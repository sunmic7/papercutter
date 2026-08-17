#!/usr/bin/env python3
"""Project HiLEx ``Answer_Block`` labels to a one-class YOLO dataset.

These boxes represent answer choices or designated answer regions inside a
question-paper page.  They are useful visual pretraining for answer-region
layout, but are deliberately kept separate from reviewed answer-key pages.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ANSWER_BLOCK_CLASS = 0
SPLITS = ("train", "valid", "test")


def _prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _answer_lines(label_path: Path) -> list[str]:
    retained = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        values = raw.split()
        if len(values) != 5:
            continue
        try:
            class_id, coordinates = int(values[0]), [float(value) for value in values[1:]]
        except ValueError:
            continue
        if class_id == ANSWER_BLOCK_CLASS and all(0 <= value <= 1 for value in coordinates):
            retained.append("0 " + " ".join(values[1:]))
    return retained


def prepare_hilex_answer_blocks(source: str | Path, output: str | Path, *, overwrite: bool = False) -> dict:
    source_root, output_root = Path(source).resolve(), Path(output).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"HiLEx YOLO directory was not found: {source_root}")
    _prepare_directory(output_root, overwrite)
    summary = {
        "schema_version": 1,
        "source": str(source_root),
        "target_class": "answer_region",
        "source_class": "Answer_Block",
        "source_class_id": ANSWER_BLOCK_CLASS,
        "splits": {},
    }
    for split in SPLITS:
        input_images, input_labels = source_root / split / "images", source_root / split / "labels"
        if not input_images.is_dir() or not input_labels.is_dir():
            raise FileNotFoundError(f"Invalid HiLEx split: {source_root / split}")
        output_images, output_labels = output_root / split / "images", output_root / split / "labels"
        output_images.mkdir(parents=True), output_labels.mkdir(parents=True)
        images = boxes = empty = 0
        for image_path in sorted(path for path in input_images.iterdir() if path.is_file()):
            label_path = input_labels / f"{image_path.stem}.txt"
            labels = _answer_lines(label_path) if label_path.is_file() else []
            shutil.copy2(image_path, output_images / image_path.name)
            (output_labels / f"{image_path.stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            images += 1
            boxes += len(labels)
            empty += int(not labels)
        summary["splits"][split] = {"images": images, "answer_boxes": boxes, "empty_images": empty}
    (output_root / "data.yaml").write_text(
        f"path: {output_root.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnames:\n  0: answer_region\n",
        encoding="utf-8",
    )
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HiLEx Answer_Block labels to one-class YOLO data")
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_hilex_answer_blocks(args.source, args.output, overwrite=args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
