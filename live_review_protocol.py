"""Local, resumable page-review mailbox used by the cutter and desktop UI.

The cutting worker writes a request before finalising a page.  In preview mode
the request is informational; in gate mode it waits for the desktop UI to
atomically write the paired decision document.  Keeping this on disk makes a
review session inspectable and recoverable without a local web service.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
VALID_MODES = {"off", "preview", "gate"}


def mode() -> str:
    value = os.environ.get("PAPERCUTTER_LIVE_REVIEW_MODE", "off").strip().lower()
    return value if value in VALID_MODES else "off"


def enabled() -> bool:
    return mode() != "off"


def review_root() -> Path:
    configured = os.environ.get("PAPERCUTTER_LIVE_REVIEW_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "runtime" / "live_review"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _valid_box(box: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(value))) for value in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def create_request(
    *, image_path: str, image_size: tuple[int, int], candidates: list[dict[str, Any]],
    book_name: str = "", unit_name: str = "", major_title: str = "", page_label: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Persist a page proposal and return its path and JSON-safe payload."""
    width, height = (max(1, int(image_size[0])), max(1, int(image_size[1])))
    request_id = f"page-{uuid.uuid4().hex}"
    root = review_root()
    request_path = root / "requests" / f"{request_id}.json"
    decision_path = root / "decisions" / f"{request_id}.json"
    safe_candidates = []
    for index, candidate in enumerate(candidates):
        box = _valid_box(candidate.get("box"), width, height)
        if box is None:
            continue
        safe_candidates.append({
            "id": str(candidate.get("id") or f"cut-{index + 1}"),
            "source_id": str(candidate.get("source_id") or candidate.get("id") or f"cut-{index + 1}"),
            "box": box,
            "number": candidate.get("number"),
            "number_detected": bool(candidate.get("number_detected", False)),
            "risk_flags": [str(value) for value in candidate.get("risk_flags", [])],
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "papercutter_page_review_request",
        "request_id": request_id,
        "mode": mode(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": os.path.abspath(image_path),
        "image_size": [width, height],
        "book_name": str(book_name),
        "unit_name": str(unit_name),
        "major_title": str(major_title),
        "page_label": str(page_label),
        "decision_path": str(decision_path),
        "candidates": safe_candidates,
    }
    _write_json_atomic(request_path, payload)
    return request_path, payload


def load_request(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != "papercutter_page_review_request":
        raise ValueError("不是受支持的实时审核请求")
    return payload


def normalise_decision(request: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Validate user edits and retain only legal boxes within the source page."""
    width, height = request["image_size"]
    source_ids = {item["id"] for item in request.get("candidates", [])}
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("candidates", [])):
        if not isinstance(item, dict):
            continue
        box = _valid_box(item.get("box"), width, height)
        if box is None:
            continue
        source_id = str(item.get("source_id") or "")
        candidates.append({
            "id": str(item.get("id") or f"review-{index + 1}"),
            "source_id": source_id if source_id in source_ids else "",
            "box": box,
            "number": item.get("number"),
            "number_detected": bool(item.get("number_detected", False)),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "papercutter_page_review_decision",
        "request_id": request["request_id"],
        "status": "accepted",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": str(raw.get("reviewer") or "本地审核"),
        "notes": str(raw.get("notes") or ""),
        "candidates": candidates,
    }


def write_decision(request: dict[str, Any], raw: dict[str, Any]) -> Path:
    decision = normalise_decision(request, raw)
    path = Path(request["decision_path"])
    _write_json_atomic(path, decision)
    return path


def wait_for_decision(request: dict[str, Any], poll_seconds: float = 0.2) -> dict[str, Any]:
    """Wait until the UI writes a valid matching decision document."""
    path = Path(request["decision_path"])
    while True:
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("request_id") == request.get("request_id") and raw.get("status") == "accepted":
                    return normalise_decision(request, raw)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        time.sleep(max(0.05, poll_seconds))
