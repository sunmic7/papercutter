#!/usr/bin/env python3
"""CPU training entry for HiLEx answer-region visual pretraining."""

from __future__ import annotations

import argparse
import json

try:  # Package import for tests/tools; direct script import for the dashboard.
    from .train_question_detector import APP_DIR, DEFAULT_PROJECT, train
except ImportError:
    from train_question_detector import APP_DIR, DEFAULT_PROJECT, train


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only HiLEx answer-region detector training")
    parser.add_argument("--data", default=str(APP_DIR / "runtime" / "training_data" / "hilex_answer_blocks" / "data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--close-mosaic", type=int, default=8)
    parser.add_argument("--cache", choices=("disk", "False"), default="disk")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", help="Path to an interrupted run's weights/last.pt")
    args = parser.parse_args()
    if args.cache == "False":
        args.cache = False
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
