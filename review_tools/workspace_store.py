"""SQLite read model for the staff annotation workbench.

The JSON project remains the recovery-friendly source of truth.  This module
mirrors it transactionally into SQLite so page progress, region state and
question-answer relations can be queried without repeatedly scanning a large
project document.  It uses only Python's standard library for easy deployment.
"""

from __future__ import annotations

import hashlib
import os
import json
import sqlite3
from contextlib import closing
from pathlib import Path


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS project (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    project_path TEXT NOT NULL,
    book_root TEXT NOT NULL DEFAULT '',
    annotation_version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS page (
    kind TEXT NOT NULL,
    page_index INTEGER NOT NULL,
    image TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'unlabeled',
    record_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, page_index)
);
CREATE TABLE IF NOT EXISTS region (
    region_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    page_index INTEGER NOT NULL,
    display_number TEXT NOT NULL DEFAULT '',
    region_role TEXT NOT NULL DEFAULT 'question',
    bbox_json TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    delivery_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    FOREIGN KEY (kind, page_index) REFERENCES page(kind, page_index) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_region_page ON region(kind, page_index);
CREATE TABLE IF NOT EXISTS relation (
    question_id TEXT PRIMARY KEY,
    answer_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft'
);
"""


def database_path(project_path: str | Path, runtime_root: str | Path) -> Path:
    resolved_project = Path(project_path).resolve()
    canonical = os.path.normcase(str(resolved_project)).encode("utf-8")
    token = hashlib.sha256(canonical).hexdigest()[:16]
    target = Path(runtime_root) / "workspace_db" / f"{resolved_project.stem}-{token}.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def sync_project(project_path: str | Path, project: dict, runtime_root: str | Path) -> Path:
    """Replace the read model inside one transaction and return its path."""
    db_path = database_path(project_path, runtime_root)
    with closing(sqlite3.connect(db_path, timeout=10)) as connection:
        connection.executescript(SCHEMA)
        connection.execute("DELETE FROM relation")
        connection.execute("DELETE FROM region")
        connection.execute("DELETE FROM page")
        connection.execute("DELETE FROM project")
        connection.execute(
            "INSERT INTO project(id, project_path, book_root, annotation_version, updated_at) VALUES(1, ?, ?, ?, ?)",
            (str(Path(project_path).resolve()), str(project.get("book_root", project.get("image_root", ""))),
             int(project.get("annotation_version", 1)), str(project.get("updated_at", ""))),
        )
        for kind, pages_key, records_key in (
            ("question", "pages", "questions"), ("answer", "answer_pages", "answers")
        ):
            for page_index, page in enumerate(project.get(pages_key, [])):
                records = [item for item in page.get(records_key, []) if isinstance(item, dict)]
                connection.execute(
                    "INSERT INTO page(kind,page_index,image,source_path,status,record_count) VALUES(?,?,?,?,?,?)",
                    (kind, page_index, str(page.get("image", "")), str(page.get("source_path", "")),
                     str(page.get("status", "unlabeled")), len(records)),
                )
                for order, record in enumerate(records):
                    region_id = str(record.get("id", "")) or f"{kind}-{page_index}-{order}"
                    display_number = str(record.get("display_number", ""))
                    connection.execute(
                        "INSERT INTO region(region_id,kind,page_index,display_number,region_role,bbox_json,text,delivery_id,payload_json) VALUES(?,?,?,?,?,?,?,?,?)",
                        (region_id, kind, page_index, display_number, str(record.get("region_role", "question")),
                         json.dumps(record.get("bbox", []), ensure_ascii=False), str(record.get("text", "")),
                         str(record.get("delivery_id", "")), json.dumps(record, ensure_ascii=False)),
                    )
        for pair in project.get("pairs", []):
            if not isinstance(pair, dict) or not pair.get("question_id"):
                continue
            connection.execute(
                "INSERT INTO relation(question_id,answer_ids_json,status) VALUES(?,?,?)",
                (str(pair["question_id"]), json.dumps(pair.get("answer_ids", []), ensure_ascii=False),
                 str(pair.get("status", "draft"))),
            )
        connection.commit()
    return db_path


def page_snapshot(project_path: str | Path, project: dict, runtime_root: str | Path,
                  kind: str, page_index: int) -> dict:
    """Return one page's read model including the enrich/level fields.

    补全字段（快照 ID、分级、AI 结果、词表码…）直接从 ``payload_json`` 里解析：
    ``region`` 表本来就存了整条记录 JSON，所以 UI 读得到它们，**不需要迁移、
    不需要加列**。这也是"JSON 是权威源、SQLite 只是读模型"的直接好处。
    """
    db_path = sync_project(project_path, project, runtime_root)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        page = connection.execute(
            "SELECT * FROM page WHERE kind=? AND page_index=?", (kind, page_index)
        ).fetchone()
        regions = connection.execute(
            "SELECT region_id,display_number,region_role,bbox_json,text,delivery_id,payload_json "
            "FROM region WHERE kind=? AND page_index=? ORDER BY rowid",
            (kind, page_index),
        ).fetchall()
    page_payload = None
    pages_key = "answer_pages" if str(kind) == "answer" else "pages"
    for index, item in enumerate((project or {}).get(pages_key) or []):
        if index == int(page_index) and isinstance(item, dict):
            page_payload = item
            break
    page_view = dict(page) if page else None
    if page_view is not None and isinstance(page_payload, dict):
        for key in ENRICH_PAGE_KEYS:
            if key in page_payload:
                page_view[key] = _json_value(page_payload[key])
    return {
        "database_path": str(db_path),
        "page": page_view,
        "regions": [_region_view(row) for row in regions],
    }


# 记录级补全字段：UI 读它们渲染徽标/风险原因/候选虚线框
ENRICH_REGION_KEYS = (
    "snapshot_id", "review_level", "review_reasons", "enrich_status", "enrich_error",
    "field_source", "field_confidence", "kp_codes", "kp_rejected", "analysis_check",
    "pair", "review_trace", "illustration_candidates",
)
# 页级补全字段
ENRICH_PAGE_KEYS = (
    "review_summary", "illustration_ready", "layout_revision",
    "page_consistency", "question_density",
)


def _json_value(value):
    """返回一个可安全外传的副本：读模型不能把自己的引用漏给调用方。"""
    if isinstance(value, (dict, list)):
        return json.loads(json.dumps(value, ensure_ascii=False))
    return value


def _region_view(row) -> dict:
    """把一行 ``region`` 转成 UI 用的字典，并从 ``payload_json`` 解析补全字段。"""
    view = {
        "region_id": row["region_id"],
        "display_number": row["display_number"],
        "region_role": row["region_role"],
        "text": row["text"],
        "delivery_id": row["delivery_id"],
        "bbox": json.loads(row["bbox_json"]),
    }
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    if isinstance(payload, dict):
        for key in ENRICH_REGION_KEYS:
            if key in payload:
                view[key] = _json_value(payload[key])
    return view

