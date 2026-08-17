#!/usr/bin/env python3
"""Convert reviewed question--answer pairs into a one-class YOLO answer set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


SPLIT_NAMES = {"train": "train", "validation": "valid", "valid": "valid", "test": "test"}


def _empty_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _source_path(value: object, manifest_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else manifest_root / path


def _normalised_box(box: object, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
    y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height)


def prepare_reviewed_answers(manifest: str | Path, output: str | Path, *, overwrite: bool = False) -> dict:
    """Create labels only from reviewer-accepted, valid answer source boxes."""
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest was not found: {manifest_path}")
    output_root = Path(output).expanduser().resolve()
    _empty_output(output_root, overwrite)

    grouped: dict[tuple[str, Path], list[list[float]]] = defaultdict(list)
    skipped = defaultdict(int)
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            skipped["invalid_json"] += 1
            continue
        if not record.get("pair_valid") or not record.get("answer_valid"):
            skipped["not_accepted"] += 1
            continue
        answer = record.get("answer") or {}
        source = _source_path(answer.get("source_image"), manifest_path.parent)
        box = answer.get("target_box") or answer.get("predicted_box")
        split = SPLIT_NAMES.get(str(record.get("dataset_split", "")))
        if source is None or split is None or not source.is_file():
            skipped["missing_source_or_split"] += 1
            continue
        image = cv2.imread(str(source))
        if image is None:
            skipped["unreadable_source"] += 1
            continue
        height, width = image.shape[:2]
        normalised = _normalised_box(box, width, height)
        if normalised is None:
            skipped["invalid_box"] += 1
            continue
        grouped[(split, source)].append(list(normalised))

    split_summary = {}
    for split in ("train", "valid", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
        entries = [(source, boxes) for (entry_split, source), boxes in grouped.items() if entry_split == split]
        box_count = 0
        for source, boxes in entries:
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:24]
            destination_name = f"{digest}{source.suffix.lower() or '.jpg'}"
            shutil.copy2(source, output_root / split / "images" / destination_name)
            labels = ["0 " + " ".join(f"{value:.7f}" for value in box) for box in boxes]
            (output_root / split / "labels" / f"{Path(destination_name).stem}.txt").write_text(
                "\n".join(labels) + "\n", encoding="utf-8"
            )
            box_count += len(labels)
        split_summary[split] = {"images": len(entries), "answer_boxes": box_count}

    summary = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "target_class": "answer_entry",
        "only_reviewer_accepted_pairs": True,
        "splits": split_summary,
        "skipped": dict(skipped),
    }
    (output_root / "data.yaml").write_text(
        f"path: {output_root.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnames:\n  0: answer_entry\n",
        encoding="utf-8",
    )
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build answer-entry YOLO data from reviewed pair manifests")
    parser.add_argument("manifest", help="training_manifest.jsonl made by export_training_data.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_reviewed_answers(args.manifest, args.output, overwrite=args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
