#!/usr/bin/env python3
"""Train the single-class question detector on CPU using Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = APP_DIR / "runtime" / "training_data" / "hilex_question" / "data.yaml"
DEFAULT_PROJECT = APP_DIR / "runtime" / "training_runs"


def train(args: argparse.Namespace) -> dict:
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    data_path = Path(args.data).resolve()
    if resume_path is None and not data_path.is_file():
        raise FileNotFoundError(
            f"Training data does not exist: {data_path}. Run training/prepare_hilex.py first."
        )
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("CPU training requires the optional Ultralytics/PyTorch packages") from exc

    project = Path(args.project).resolve()
    run_name = (
        resume_path.parent.parent.name if resume_path is not None
        else args.name or f"hilex_yolo11n_cpu_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_config = {
        "schema_version": 1,
        "dataset": str(data_path),
        "model": str(resume_path) if resume_path is not None else args.model,
        "resume": str(resume_path) if resume_path is not None else "",
        "device": "cpu",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "cache": args.cache,
        "seed": args.seed,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    project.mkdir(parents=True, exist_ok=True)
    (project / f"{run_name}.config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Ultralytics restores the original data, optimizer, learning-rate and
    # early-stopping state from ``last.pt`` when resume=True.  Do not pass new
    # hyperparameters in that path: they could silently invalidate a true
    # interruption recovery.
    model = YOLO(str(resume_path) if resume_path is not None else args.model)
    if resume_path is not None:
        results = model.train(resume=True)
        return {"run_dir": str(resume_path.parent.parent), "results": str(results), **run_config}
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        cache=args.cache,
        project=str(project),
        name=run_name,
        exist_ok=False,
        patience=args.patience,
        seed=args.seed,
        pretrained=True,
        optimizer="auto",
        close_mosaic=args.close_mosaic,
        degrees=2.0,
        translate=0.03,
        scale=0.10,
        fliplr=0.0,
        mosaic=0.15,
        plots=True,
    )
    return {"run_dir": str(project / run_name), "results": str(results), **run_config}


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only HiLEx question-detector training")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="One-class YOLO data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO checkpoint or local checkpoint")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Directory for run artifacts")
    parser.add_argument("--name", help="Run name; default includes the current time")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cache", choices=("disk", "False"), default="disk")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", help="Path to an interrupted run's weights/last.pt")
    args = parser.parse_args()
    if args.cache == "False":
        args.cache = False
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
