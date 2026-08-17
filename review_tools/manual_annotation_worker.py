#!/usr/bin/env python3
"""Separate low-priority OCR process used by the annotation UI."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

RESULT_PREFIX = "@@MANUAL_ANNOTATION_RESULT@@"


def _lower_priority() -> None:
    """Keep pen/zoom interaction responsive while OCR consumes CPU."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in {"question", "answer"}:
        print(RESULT_PREFIX + json.dumps({"ok": False, "error": "用法错误"}, ensure_ascii=False))
        return 2
    _lower_priority()
    kind, image_path, profile = sys.argv[1:]
    try:
        from review_tools.manual_annotation_workspace import _answer_suggestions, _question_suggestions
        records = (_answer_suggestions if kind == "answer" else _question_suggestions)(image_path, profile)
        print(RESULT_PREFIX + json.dumps({"ok": True, "records": records}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(RESULT_PREFIX + json.dumps({"ok": False, "error": f"{exc}\n{traceback.format_exc(limit=2)}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
