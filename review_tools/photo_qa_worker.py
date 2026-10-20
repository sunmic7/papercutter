#!/usr/bin/env python3
"""Long-running, resumable-status worker for the standalone photographed-QA tool."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_tools.photo_qa_import import build_batch, write_batch


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_env_file(path: str) -> None:
    if not path:
        return
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"环境配置文件不存在：{source}")
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run(roots: list[str], output: str, *, semantic_review: bool, env_file: str = "") -> int:
    output_root = Path(output).expanduser().resolve()
    status_path = output_root / "job_status.json"
    status = {
        "schema_version": 1, "started_at": datetime.now().isoformat(timespec="seconds"),
        "state": "running", "phase": "initializing", "roots": roots,
        "semantic_review": semantic_review, "reports": [], "message": "正在初始化后台任务…",
    }
    _atomic_json(status_path, status)
    try:
        _load_env_file(env_file)
        if semantic_review and not os.environ.get("DASHSCOPE_API_KEY", "").strip():
            status["warning"] = "未配置 DASHSCOPE_API_KEY：将只生成审核清单，不生成正式 JSON。"
        from review_tools.photo_qa_import import _qwen_reviewer

        reviewer = _qwen_reviewer if semantic_review else None
        for index, root in enumerate(roots, start=1):
            status.update(phase="processing", current_batch=index, total_batches=len(roots),
                          message=f"正在处理第 {index}/{len(roots)} 套图片：{Path(root).name}")
            _atomic_json(status_path, status)
            batch = build_batch(root, semantic_reviewer=reviewer)
            report = write_batch(batch, output_root / "batches")
            status["reports"].append(report)
            _atomic_json(status_path, status)
        status.update(state="completed", phase="completed", finished_at=datetime.now().isoformat(timespec="seconds"),
                      message="后台导入完成。")
        _atomic_json(status_path, status)
        return 0
    except Exception as error:
        status.update(state="failed", phase="failed", finished_at=datetime.now().isoformat(timespec="seconds"),
                      message=f"导入失败：{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=12))
        _atomic_json(status_path, status)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperFlow 拍照 QA 后台导入任务")
    parser.add_argument("--output", required=True, help="本次任务独立输出目录")
    parser.add_argument("--semantic-review", action="store_true", help="允许 Qwen 读取题图并判断题答归属")
    parser.add_argument("--env-file", default="", help="可选 .env 文件，仅在后台进程中读取")
    parser.add_argument("roots", nargs="+", help="题目根目录，可一次传入三套")
    args = parser.parse_args()
    raise SystemExit(run(args.roots, args.output, semantic_review=args.semantic_review, env_file=args.env_file))


if __name__ == "__main__":
    main()
