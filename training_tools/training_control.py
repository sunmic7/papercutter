#!/usr/bin/env python3
"""Start, observe, attach and stop CPU training runs from one stable interface."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runtime" / "training_runs"
STATE_PATH = RUNS_DIR / "active_training.json"


def _write_state(state: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_PATH)


def _read_state() -> dict:
    if not STATE_PATH.is_file():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def process_running(pid: int | None) -> bool:
    if not pid:
        return False
    # ``tasklist`` can return access denied for a Python process that owns
    # worker children.  Get-Process is reliable on the Windows installations
    # supported by this application.
    command = (
        f"$p=Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
        "if($p){$p.Id}else{exit 1}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    return result.returncode == 0 and str(int(pid)) in result.stdout


def _latest_metrics(run_dir: Path) -> dict:
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        return rows[-1] if rows else {}
    except (OSError, csv.Error):
        return {}


def status() -> dict:
    state = _read_state()
    if not state:
        return {"active": False, "message": "No managed training run"}
    run_dir = Path(state.get("run_dir", ""))
    active = process_running(state.get("pid"))
    return {
        **state,
        "active": active,
        "metrics": _latest_metrics(run_dir) if run_dir.is_dir() else {},
        "results_path": str(run_dir / "results.csv") if run_dir else "",
    }


def attach(pid: int, run_dir: str | Path, *, log_path: str | Path | None = None) -> dict:
    folder = Path(run_dir).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {folder}")
    state = {
        "schema_version": 1,
        "pid": int(pid),
        "run_dir": str(folder),
        "log_path": str(Path(log_path).resolve()) if log_path else "",
        "attached_at": datetime.now(timezone.utc).isoformat(),
        "managed": False,
    }
    _write_state(state)
    return status()


TRAINING_SCRIPTS = {
    "question": "train_question_detector.py",
    "answer_hilex": "train_hilex_answer_detector.py",
    "answer": "train_answer_detector.py",
}


def start(
    *, epochs: int, imgsz: int, batch: int, workers: int, name: str | None = None,
    kind: str = "question", resume: str | Path | None = None,
) -> dict:
    current = status()
    if current.get("active"):
        raise RuntimeError(f"Training is already running (PID {current['pid']})")
    if kind not in TRAINING_SCRIPTS:
        raise ValueError(f"Unsupported training kind: {kind}")
    resume_path = Path(resume).resolve() if resume else None
    if resume_path is not None:
        if not resume_path.is_file() or resume_path.name != "last.pt" or resume_path.parent.name != "weights":
            raise FileNotFoundError("断点续训需要选择完整训练目录中的 weights\\last.pt")
        run_dir = resume_path.parent.parent
        run_name = run_dir.name
    else:
        run_name = name or f"{kind}_cpu_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = RUNS_DIR / run_name
    log_dir = RUNS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    command = [sys.executable, "-u", str(TOOLS_DIR / TRAINING_SCRIPTS[kind])]
    if resume_path is not None:
        command.extend(["--resume", str(resume_path)])
    else:
        command.extend([
            "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch),
            "--workers", str(workers), "--name", run_name,
        ])
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(APP_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    state = {
        "schema_version": 1,
        "pid": process.pid,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "managed": True,
        "kind": kind,
        "resume_checkpoint": str(resume_path) if resume_path is not None else "",
    }
    _write_state(state)
    return status()


def stop() -> dict:
    current = status()
    if not current.get("active"):
        return {**current, "stopped": False}
    subprocess.run(["taskkill", "/PID", str(current["pid"]), "/T", "/F"], check=False)
    return {**status(), "stopped": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage CPU question-detector training")
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start", help="Start a visible/logged background training run")
    start_parser.add_argument("--epochs", type=int, default=5)
    start_parser.add_argument("--imgsz", type=int, default=640)
    start_parser.add_argument("--batch", type=int, default=2)
    start_parser.add_argument("--workers", type=int, default=4)
    start_parser.add_argument("--name")
    start_parser.add_argument("--kind", choices=sorted(TRAINING_SCRIPTS), default="question")
    start_parser.add_argument("--resume", help="Interrupted run's weights/last.pt")
    attach_parser = commands.add_parser("attach", help="Attach an existing process to the dashboard")
    attach_parser.add_argument("--pid", type=int, required=True)
    attach_parser.add_argument("--run-dir", required=True)
    attach_parser.add_argument("--log-path")
    commands.add_parser("status", help="Show current status and latest metrics")
    commands.add_parser("stop", help="Stop the managed training process")
    args = parser.parse_args()
    if args.command == "start":
        result = start(
            epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
            name=args.name, kind=args.kind, resume=args.resume,
        )
    elif args.command == "attach":
        result = attach(args.pid, args.run_dir, log_path=args.log_path)
    elif args.command == "stop":
        result = stop()
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
