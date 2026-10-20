"""Local, versioned data model for manual question/answer annotation.

The question part deliberately remains compatible with ``annotate_cuts.py``:
``pages[*].questions[*].bbox`` still uses the original xyxy pixel format.
The additional answer pages and pair records are ignored safely by the legacy
evaluation tools, while preserving enough information for later training.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from review_tools import review_levels


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
REGION_ROLES = {"question", "group", "continuation", "shared_material"}
ANSWER_MODES = {"textbook", "generate_later", "mixed"}
DELIVERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
GENERATED_DELIVERY_MARKERS = {"manifest.json", "database_import.json", "quality_report.json"}
# 记录级"后端独占"字段：渲染层从不写它们，整工程 autosave 时必须原样接住。
# 这是 R2 的唯一防线 —— 少了它，员工用旧版 Electron 打开工程点一次保存，
# 整本的 AI 补全结果（快照/分级/知识点）就会被前端快照覆盖抹掉。
ENRICH_PERSISTENT_KEYS = (
    "snapshot", "snapshot_id", "review_level", "review_reasons",
    "field_source", "field_confidence", "enrich_status", "enrich_error",
    "illustration_candidates", "kp_codes", "kp_rejected", "analysis_check",
    "pair", "review_trace",
)
# 页级同类字段（§2.3）：候选就绪标记与分级计数同样只由后端写。
ENRICH_PERSISTENT_PAGE_KEYS = (
    "review_summary", "illustration_ready", "layout_revision",
    "page_consistency", "question_density",
)
# 工程级：整本补全状态与书级口径，渲染层只读。
ENRICH_PERSISTENT_PROJECT_KEYS = (
    "enrichment", "book_meta_rev", "book_meta_locked", "difficulty_scheme",
)
# 新字段的默认值：老工程读出来字段齐全，调用方不必到处 .get 兜底。
ENRICH_DEFAULT_VALUES = {
    "snapshot": None,
    "snapshot_id": "",
    "review_level": "L2",          # 没算过就按需人看（§2.4 第 3 条）
    "review_reasons": [],
    "field_source": {},
    "field_confidence": {},
    "enrich_status": "none",
    "enrich_error": "",
    "illustration_candidates": [],
    "kp_codes": [],
    "kp_rejected": [],
    "analysis_check": {},
    "pair": {},
    "review_trace": {},
}
# normalize_record 只补惰性字段（含 review_reasons 但不含 review_level）：
# 记录在"后端算出级别"之前不带任何分级外观。
ENRICH_NORMALIZE_DEFAULTS = {
    key: value for key, value in ENRICH_DEFAULT_VALUES.items() if key != "review_level"
}


def enrich_fields_enabled() -> bool:
    """补全流水线总开关（与桥服务同一个环境变量、同一套取值）。

    ``PAPERFLOW_ENRICH=0`` 时数据层也不补这 14 个字段，读盘结果与接入前逐字节
    一致 —— "关闭开关 = 回到旧行为"必须连字段都不出现，否则 UI/交付层仍会
    因为读到默认 ``L2`` 而改变行为。
    """
    value = str(os.environ.get("PAPERFLOW_ENRICH", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}
_CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                   "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def chinese_number_to_int(value: object) -> int | None:
    """Convert the short Chinese ordinals normally used by exercise headings."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _CIRCLED_NUMBERS:
        return _CIRCLED_NUMBERS.index(text) + 1
    if any(char not in _CHINESE_DIGITS and char not in _CHINESE_UNITS for char in text):
        return None
    if not any(char in _CHINESE_UNITS for char in text):
        try:
            return int("".join(str(_CHINESE_DIGITS[char]) for char in text))
        except (KeyError, ValueError):
            return None
    total = current = 0
    for char in text:
        if char in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[char]
        else:
            total += (current or 1) * _CHINESE_UNITS[char]
            current = 0
    return total + current


def int_to_chinese(value: object) -> str:
    number = chinese_number_to_int(value)
    if number is None or number <= 0:
        return str(value or "").strip()
    digits = "零一二三四五六七八九"
    if number < 10:
        return digits[number]
    if number < 20:
        return "十" + (digits[number % 10] if number % 10 else "")
    if number < 100:
        return digits[number // 10] + "十" + (digits[number % 10] if number % 10 else "")
    return str(number)


def extract_major_number(title: object, fallback: object = "") -> str:
    """Return the printed major ordinal from headings such as '五、综合实践'."""
    text = re.sub(r"\s+", "", str(title or ""))
    patterns = (
        r"^[（(]?([一二三四五六七八九十百千〇零两]+)[）)]?[、,，.．]",
        r"^第([一二三四五六七八九十百千〇零两]+|\d{1,3})(?:大题|题|部分|单元|章|节)",
        r"^[（(]?(\d{1,3})[）)]?[、,，.．]",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return int_to_chinese(match.group(1))
    if text and text[:1] in _CIRCLED_NUMBERS:
        return int_to_chinese(_CIRCLED_NUMBERS.index(text[0]) + 1)
    return int_to_chinese(fallback) if str(fallback or "").strip() else ""


def extract_subsection_number(title: object) -> str:
    """Return ordinals from group headings such as '（二）课外阅读'."""
    text = re.sub(r"\s+", "", str(title or ""))
    match = re.match(r"^[（(]([一二三四五六七八九十百千〇零两]+|\d{1,3})[）)]", text)
    return int_to_chinese(match.group(1)) if match else ""


def _canonical_number(value: object) -> str:
    text = re.sub(r"\s+", "", str(value or "")).translate(
        str.maketrans("０１２３４５６７８９", "0123456789")
    ).replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    text = re.sub(r"^第", "", text)
    text = re.sub(r"(?:大题|小题|题|单元|章|节)$", "", text)
    wrappers = (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"))
    while len(text) >= 2 and any(text.startswith(left) and text.endswith(right)
                                 for left, right in wrappers):
        text = text[1:-1].strip()
    converted = chinese_number_to_int(text)
    if converted is not None:
        return str(converted)
    match = re.fullmatch(r"0*(\d+)(.*)", text)
    return f"{int(match.group(1))}{match.group(2)}" if match else text.lower()


def _normalize_number_path(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()][:8]


def number_path(record: dict) -> list[str]:
    """Return a v5 arbitrary-depth number path, migrating legacy 2/3 levels."""
    explicit = _normalize_number_path(record.get("number_path", []))
    if explicit:
        return explicit
    major = str(record.get("major_number", "") or "").strip()
    subsection = str(record.get("subsection_number", "") or "").strip()
    minor = str(record.get("minor_number", record.get("number", "")) or "").strip()
    return [part for part in (major, subsection, minor) if part]


def display_number(record: dict) -> str:
    parts = number_path(record)
    return "-".join(parts)


def canonical_number_path(record: dict) -> tuple[str, ...]:
    """Return the language-independent number key used by scoped matching."""
    return tuple(
        part for part in (_canonical_number(value) for value in number_path(record)) if part
    )


def canonical_display_number(record: dict) -> str:
    return "-".join(canonical_number_path(record))


def hierarchy_warnings(records: list[dict]) -> list[str]:
    """Return actionable numbering warnings without blocking manual saves.

    A minor (sub-question) number is optional: many pages contain a single
    large question with no printed sub-question number.  Such a record is
    identified by its major number (and optional middle levels) and is valid
    for page confirmation and delivery.
    """
    seen: dict[str, int] = {}
    missing = 0
    dangling_continuations = 0
    for record in records:
        normalize_record(record)
        role = record.get("region_role", "question")
        if role in {"group", "shared_material"}:
            continue
        if role == "continuation":
            if not record.get("continuation_of"):
                dangling_continuations += 1
            continue
        # A question without a major number cannot be reliably placed in the
        # book hierarchy.  Do not require a minor number here.
        major = str(record.get("major_number", "") or "").strip()
        if not major:
            missing += 1
        else:
            key = display_number(record)
            seen[key] = seen.get(key, 0) + 1
    duplicates = [key for key, count in seen.items() if count > 1]
    warnings = []
    if missing:
        warnings.append(f"有 {missing} 个框尚未填写大题号")
    if duplicates:
        warnings.append("同页存在重复层级题号：" + "、".join(duplicates[:8]))
    if dangling_continuations:
        warnings.append(f"有 {dangling_continuations} 个续接片段尚未关联上一题")
    return warnings


def _normalize_context_segments(value: object) -> list[dict]:
    result = []
    for segment in value if isinstance(value, list) else []:
        if not isinstance(segment, dict):
            continue
        box = segment.get("bbox", [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            normalized_box = [int(round(float(item))) for item in box]
        except (TypeError, ValueError):
            continue
        if normalized_box[2] <= normalized_box[0] or normalized_box[3] <= normalized_box[1]:
            continue
        result.append({
            "page_index": int(segment.get("page_index", -1)),
            "bbox": normalized_box,
        })
    return result


def normalize_record(record: dict) -> dict:
    """Add v4 hierarchy/region fields while retaining v1-v3 compatibility keys."""
    minor = str(record.get("minor_number", record.get("number", "")) or "").strip()
    major_title = str(record.get("major_title", record.get("major", "")) or "").strip()
    major_number = str(record.get("major_number", "") or "").strip()
    if not major_number:
        major_number = extract_major_number(major_title, record.get("major_idx", ""))
    subsection_title = str(record.get("subsection_title", "") or "").strip()
    subsection_number = str(record.get("subsection_number", "") or "").strip()
    if not subsection_number:
        subsection_number = extract_subsection_number(subsection_title)
    region_role = str(record.get("region_role", "question") or "question").strip()
    if region_role not in REGION_ROLES:
        region_role = "question"
    if region_role == "shared_material":
        region_role = "group"
    path = _normalize_number_path(record.get("number_path", []))
    if not path:
        path = [part for part in (major_number, subsection_number, minor) if part]
    if path:
        major_number = path[0]
        minor = path[-1] if region_role == "question" and len(path) >= 2 else ""
        subsection_number = path[1] if (
            len(path) >= 3 or (region_role == "group" and len(path) >= 2)
        ) else ""
    record["number"] = minor
    record["minor_number"] = minor
    record["major"] = major_title
    record["major_title"] = major_title
    record["major_number"] = major_number
    record["subsection_title"] = subsection_title
    record["subsection_number"] = subsection_number
    record["number_path"] = path
    record["region_role"] = region_role
    record["parent_id"] = str(record.get("parent_id", "") or "").strip()
    record["continuation_of"] = str(record.get("continuation_of", "") or "").strip()
    record["match_scope_id"] = str(record.get("match_scope_id", "") or "").strip()
    record["shared_context_segments"] = _normalize_context_segments(
        record.get("shared_context_segments", [])
    )
    record["display_number"] = display_number(record)
    record["minor_number_inferred"] = bool(record.get("minor_number_inferred", False))
    # 补全流水线的记录级字段：只补缺失，不覆盖已有值。分级字段（review_level）
    # 刻意不在这里补 —— 新建记录在"后端算出级别"之前不该带任何徽标，
    # 否则关闭 PAPERFLOW_ENRICH 时老工程会突然显示成一片 L2。
    if enrich_fields_enabled():
        for key, default in ENRICH_NORMALIZE_DEFAULTS.items():
            if key not in record:
                record[key] = _clone(default)
    return record


def apply_enrich_defaults(record: dict) -> dict:
    """读盘时把 14 个补全字段补齐（老工程读出来字段齐全）。

    与 :func:`normalize_record` 的区别只有一条：这里连 ``review_level`` 也补成
    ``L2``（§2.4 的默认值"没算过就按需人看"）。只在 ``load_project`` 里用 ——
    也就是说：**磁盘上读出来的记录**是齐的，**前端新建的记录**不带分级。

    默认级别的两个取值都来自 :mod:`review_tools.review_levels`（``DEFAULT_LEVEL``
    与 ``RULE_VERSION``），**不写字面量**：否则导出守门与这里会在某次规则升版后
    各说各话（QA Round 8 缺口 3）。补出来的这一条在 ``review_trace`` 里明确标
    ``level_source='default'``、``computed_at=''``，交付门禁据此区分
    "算过是 L2" 和 "从没算过"。
    """
    if not isinstance(record, dict):
        return record
    for key, default in ENRICH_DEFAULT_VALUES.items():
        if key not in record:
            record[key] = _clone(default)
        elif record[key] is None and default is not None:
            record[key] = _clone(default)
    if "review_level" not in record or not str(record.get("review_level") or "").strip():
        record["review_level"] = review_levels.DEFAULT_LEVEL
        trace = record.get("review_trace")
        trace = dict(trace) if isinstance(trace, dict) else {}
        if not str(trace.get("computed_at") or ""):
            trace.setdefault("level_source", "default")
            trace["rule_version"] = review_levels.RULE_VERSION
            record["review_trace"] = trace
    return record


def _clone(value):
    """深拷贝一个 JSON 值；用于把默认值/权威字段接进另一条记录。"""
    if isinstance(value, (dict, list)):
        return json.loads(json.dumps(value, ensure_ascii=False))
    return value


def _image_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem
    return (0, int(stem), path.name.lower()) if stem.isdigit() else (1, stem.lower(), path.name.lower())


def _images(root: str | os.PathLike[str]) -> list[Path]:
    base = Path(root).resolve()
    result = []
    generated_names = {"k12_dataset", "delivery_packages", "manual_annotation_exports"}
    for current, directories, files in os.walk(base):
        current_path = Path(current)
        # A manifest/database marker identifies a generated delivery subtree,
        # regardless of its random book-id folder name.
        if current_path != base and any((current_path / marker).is_file()
                                        for marker in GENERATED_DELIVERY_MARKERS):
            directories[:] = []
            continue
        directories[:] = [name for name in directories
                           if not name.startswith("_") and name.lower() not in generated_names]
        for name in files:
            path = current_path / name
            if path.suffix.lower() in IMAGE_EXTS:
                result.append(path)
    return sorted(result, key=lambda path: _image_key(path.relative_to(base)))


def remove_cached_delivery_pages(project: dict, book_root: str | os.PathLike[str]) -> int:
    """Remove legacy cached pages that point into a generated delivery package.

    Older projects may already contain cropped delivery images in ``pages``.
    Merely improving the directory scanner does not remove those stale cache
    entries, so they otherwise remain visible forever.  This check is targeted:
    it never drops a missing source page and never scans outside the selected
    textbook directory.
    """
    root = Path(book_root).resolve()

    def is_generated(page: dict) -> bool:
        relative = str(page.get("image", "") or "")
        try:
            candidate = (root / relative).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            return False
        current = candidate.parent
        while current != root:
            if any((current / marker).is_file() for marker in GENERATED_DELIVERY_MARKERS):
                return True
            parent = current.parent
            if parent == current:
                break
            current = parent
        return False

    removed_question_ids: set[str] = set()
    removed_answer_ids: set[str] = set()
    removed = 0
    for pages_key, records_key, removed_ids in (
        ("pages", "questions", removed_question_ids),
        ("answer_pages", "answers", removed_answer_ids),
        ("ignored_pages", "questions", removed_question_ids),
    ):
        kept = []
        for page in project.get(pages_key, []):
            if not is_generated(page):
                kept.append(page)
                continue
            removed += 1
            for record in page.get(records_key, []):
                if isinstance(record, dict) and record.get("id"):
                    removed_ids.add(str(record["id"]))
            if pages_key == "ignored_pages":
                for record in page.get("answers", []):
                    if isinstance(record, dict) and record.get("id"):
                        removed_answer_ids.add(str(record["id"]))
        project[pages_key] = kept

    if removed:
        project["pairs"] = [
            pair for pair in project.get("pairs", [])
            if str(pair.get("question_id", "")) not in removed_question_ids
            and not any(str(answer_id) in removed_answer_ids for answer_id in pair.get("answer_ids", []))
        ]
    return removed


def reconcile_book_pages(project: dict, book_root: str) -> bool:
    """Drop generated-output pages and add real source pages without losing annotations."""
    fresh = create_book_project(book_root, str(project.get("profile", "general") or "general"))
    changed = False
    available = {
        str(page.get("image", "")).replace("\\", "/"): page
        for page in fresh.get("pages", [])
    }
    ignored_keys = {
        str(page.get("image", "")).replace("\\", "/")
        for page in project.get("ignored_pages", [])
    }
    existing_questions = {
        str(page.get("image", "")).replace("\\", "/"): page
        for page in project.get("pages", [])
    }
    existing_answers = {
        str(page.get("image", "")).replace("\\", "/"): page
        for page in project.get("answer_pages", [])
    }
    answer_keys = set(existing_answers)
    reconciled_questions = [
        existing_questions.get(key, fresh_page)
        for key, fresh_page in available.items()
        if key not in answer_keys and key not in ignored_keys
    ]
    reconciled_answers = [
        page for key, page in existing_answers.items()
        if key in available and key not in ignored_keys
    ]
    if [page.get("image") for page in reconciled_questions] != [page.get("image") for page in project.get("pages", [])]:
        project["pages"] = reconciled_questions
        changed = True
    if [page.get("image") for page in reconciled_answers] != [page.get("image") for page in project.get("answer_pages", [])]:
        project["answer_pages"] = reconciled_answers
        changed = True
    project["book_root"] = str(Path(book_root).resolve())
    project["image_root"] = project["book_root"]
    project["answer_image_root"] = project["book_root"] if project.get("answer_pages") else ""
    return changed


def _page_records(root: Path, field: str, profile: str) -> list[dict]:
    return [{
        "image": path.relative_to(root).as_posix(), "profile": profile,
        "status": "unlabeled", field: [],
    } for path in _images(root)]


def _records_from_paths(root: Path, paths: list[Path], field: str, profile: str) -> list[dict]:
    return [{
        "image": path.relative_to(root).as_posix(), "profile": profile,
        "status": "unlabeled", field: [],
    } for path in paths]


def create_project(question_root: str, answer_root: str = "", profile: str = "general") -> dict:
    """Create a project without OCR. Directory traversal is the only up-front work."""
    q_root = str(Path(question_root).resolve())
    a_root = str(Path(answer_root).resolve()) if answer_root else ""
    return {
        # Keep this at 1 so existing cut-evaluation readers can open it.
        "schema_version": 1,
        "kind": "papercutter_manual_annotation",
        "annotation_version": 5,
        "created_at": _now(),
        "updated_at": _now(),
        "coordinate_format": "xyxy_pixels",
        "image_root": q_root,
        "pages": _page_records(Path(q_root), "questions", profile),
        "answer_image_root": a_root,
        "answer_pages": _page_records(Path(a_root), "answers", profile) if a_root else [],
        "ignored_pages": [],
        "match_scopes": [],
        "answer_mode": "textbook",
        "ocr_settings": {"auto_page_ocr_on_confirm": True},
        "pairs": [],
    }


def create_book_project(book_root: str, profile: str = "general") -> dict:
    """Create an annotation project from one chosen textbook directory.

    Every source image starts in the browsable textbook-page list.  Answer pages
    are deliberately not guessed from folder names or OCR: staff find them while
    browsing and explicitly move the relevant pages to ``answer_pages``.  This
    avoids a failed heuristic making the answer-cutting workflow inaccessible.
    """
    root = Path(book_root).resolve()
    all_paths = _images(root)
    project = {
        "schema_version": 1,
        "kind": "papercutter_manual_annotation",
        "annotation_version": 5,
        "created_at": _now(),
        "updated_at": _now(),
        "coordinate_format": "xyxy_pixels",
        "book_root": str(root),
        "image_root": str(root),
        "pages": _records_from_paths(root, all_paths, "questions", profile),
        "answer_image_root": "",
        "answer_pages": [],
        "ignored_pages": [],
        "match_scopes": [],
        "answer_mode": "textbook",
        "ocr_settings": {"auto_page_ocr_on_confirm": True},
        "pairs": [],
    }
    return project


def sync_page_profiles(project: dict, profile: str, previous: str = "") -> bool:
    """让页面级 profile 跟随教材级：只覆盖空值、general 和上一版自动值。

    页面级 profile 是 Qt 工作台、切题评估等入口直接读取的字段，只改教材级
    会让这些入口仍按旧版式切题；而人工单独调过某页时不应当被覆盖。
    """
    if not profile:
        return False
    changed = False
    for pages_key in ("pages", "answer_pages"):
        for page in project.get(pages_key, []):
            if not isinstance(page, dict):
                continue
            current = str(page.get("profile", "") or "").strip()
            if current not in {"", "general"} and current != previous:
                continue
            if current != profile:
                page["profile"] = profile
                changed = True
    return changed


def default_book_project_path(book_root: str, runtime_root: str | os.PathLike[str]) -> Path:
    """Keep the complete annotation project beside its textbook."""
    return Path(book_root).resolve() / "PaperFlow标注项目.json"


def legacy_book_project_path(book_root: str, runtime_root: str | os.PathLike[str]) -> Path:
    """Return the pre-v6 central-runtime path for one-time migration."""
    canonical = os.path.normcase(os.path.abspath(book_root)).encode("utf-8")
    token = hashlib.sha1(canonical).hexdigest()[:16]
    return Path(runtime_root) / "manual_annotation_projects" / f"{token}.json"


def _delivery_id(prefix: str) -> str:
    """Return a database/file-system safe identifier containing only ASCII letters and digits."""
    return prefix + uuid.uuid4().hex.upper()


def _valid_delivery_id(value: object, prefix: str) -> bool:
    text = str(value or "")
    return bool(text.startswith(prefix) and DELIVERY_ID_PATTERN.fullmatch(text))


def ensure_delivery_ids(project: dict) -> None:
    """Assign persistent delivery IDs without changing internal annotation references.

    Annotation IDs remain untouched because pair/parent relations use them.  Delivery IDs
    are the stable public identifiers used by cropped images, JSON files and databases.
    """
    book_id = str(project.get("delivery_book_id", "") or "")
    if not _valid_delivery_id(book_id, "B"):
        project["delivery_book_id"] = _delivery_id("B")

    used: set[str] = {project["delivery_book_id"]}
    for _page_index, _page, record in all_questions(project):
        value = str(record.get("delivery_id", "") or "")
        if not _valid_delivery_id(value, "Q") or value in used:
            value = _delivery_id("Q")
            record["delivery_id"] = value
        used.add(value)
    for _page_index, _page, record in all_answers(project):
        value = str(record.get("delivery_id", "") or "")
        if not _valid_delivery_id(value, "A") or value in used:
            value = _delivery_id("A")
            record["delivery_id"] = value
        used.add(value)
    for pair in project.get("pairs", []):
        value = str(pair.get("delivery_id", "") or "")
        if not _valid_delivery_id(value, "R") or value in used:
            value = _delivery_id("R")
            pair["delivery_id"] = value
        used.add(value)


def merge_persistent_delivery_state(incoming: dict, existing: dict) -> None:
    """Keep backend-owned public identities when a renderer snapshot is saved.

    The renderer intentionally edits the full project in memory, but a record
    can receive its public delivery ID in the backend after that snapshot was
    created.  Matching by the internal annotation ID prevents a later whole-
    project autosave (or the Electron emergency exit save) from rotating those
    public IDs or silently dropping a relation that the backend just accepted.
    """
    existing_book_id = str(existing.get("delivery_book_id", "") or "")
    if _valid_delivery_id(existing_book_id, "B"):
        incoming["delivery_book_id"] = existing_book_id

    # These fields are changed through dedicated backend endpoints and are not
    # editable by the whole-project cache form.  The layout profile belongs here
    # as well: it can be resolved by a background detection job after the
    # renderer took its snapshot, and a stale snapshot must not undo that.
    for key in (
        "book_root", "image_root", "answer_image_root",
        "delivery_output_root", "language", "stage", "grade", "subject",
        "profile", "profile_source", "profile_reason", "profile_locked",
        *ENRICH_PERSISTENT_PROJECT_KEYS,
    ):
        if key in existing:
            incoming[key] = _clone(existing[key])
        elif key in {"book_root", "image_root", "answer_image_root"}:
            incoming.pop(key, None)

    def records(project: dict, pages_key: str, records_key: str):
        for page in project.get(pages_key, []):
            if not isinstance(page, dict):
                continue
            for record in page.get(records_key, []):
                if isinstance(record, dict) and record.get("id"):
                    yield record

    for pages_key, records_key, prefix in (
        ("pages", "questions", "Q"),
        ("answer_pages", "answers", "A"),
        ("ignored_pages", "questions", "Q"),
        ("ignored_pages", "answers", "A"),
    ):
        authoritative = {
            str(record["id"]): record
            for record in records(existing, pages_key, records_key)
        }
        for record in records(incoming, pages_key, records_key):
            source = authoritative.get(str(record["id"]), {})
            public_id = str(source.get("delivery_id", "") or "")
            if _valid_delivery_id(public_id, prefix):
                record["delivery_id"] = public_id
            # R2 的唯一防线：这些字段只有后端会写，渲染层的整工程快照里可能
            # 根本没有（旧版 UI）或是过期值 —— 一律以磁盘上的后端状态为准。
            for key in ENRICH_PERSISTENT_KEYS:
                if key in source:
                    record[key] = _clone(source[key])

    # Page rotation is changed only through the dedicated backend endpoint.
    # Preserve it by immutable source-image identity during whole-cache saves.
    # 页级的候选就绪标记与分级计数同理：它们是后端写的，旧版渲染层不认识。
    for pages_key in ("pages", "answer_pages", "ignored_pages"):
        authoritative_pages = {
            str(page.get("image", "")): page
            for page in existing.get(pages_key, []) if isinstance(page, dict)
        }
        for page in incoming.get(pages_key, []):
            if not isinstance(page, dict):
                continue
            authoritative = authoritative_pages.get(str(page.get("image", "")), {})
            for key in ("rotation", "rotated_image"):
                if key in authoritative:
                    page[key] = authoritative[key]
                else:
                    page.pop(key, None)
            for key in ENRICH_PERSISTENT_PAGE_KEYS:
                if key in authoritative:
                    page[key] = _clone(authoritative[key])

    incoming_question_ids = {
        str(record["id"])
        for record in records(incoming, "pages", "questions")
    }
    incoming_answer_ids = {
        str(record["id"])
        for record in records(incoming, "answer_pages", "answers")
    }
    existing_pairs = {
        str(pair.get("question_id", "")): pair
        for pair in existing.get("pairs", [])
        if isinstance(pair, dict) and pair.get("question_id")
    }
    merged_pairs = []
    seen_questions: set[str] = set()
    for pair in incoming.get("pairs", []):
        if not isinstance(pair, dict) or not pair.get("question_id"):
            continue
        question_id = str(pair["question_id"])
        if question_id not in incoming_question_ids or question_id in seen_questions:
            continue
        authoritative = existing_pairs.get(question_id, {})
        # Relations are edited exclusively by the dedicated set-pair endpoint,
        # so the on-disk backend copy is authoritative for the entire object.
        candidate = dict(authoritative) if authoritative else dict(pair)
        candidate["answer_ids"] = [
            str(value) for value in candidate.get("answer_ids", [])
            if str(value) in incoming_answer_ids
        ]
        if candidate.get("status") == "accepted" and not candidate["answer_ids"]:
            continue
        merged_pairs.append(candidate)
        seen_questions.add(question_id)
    # Whole-project caching has no "delete relation" operation.  Preserve a
    # relation that exists only in the newer backend state as long as its
    # question still exists in the incoming project.
    for question_id, pair in existing_pairs.items():
        if question_id in incoming_question_ids and question_id not in seen_questions:
            candidate = dict(pair)
            candidate["answer_ids"] = [
                str(value) for value in candidate.get("answer_ids", [])
                if str(value) in incoming_answer_ids
            ]
            if candidate.get("status") == "accepted" and not candidate["answer_ids"]:
                continue
            merged_pairs.append(candidate)
    incoming["pairs"] = merged_pairs


def save_project(project: dict, path: str | os.PathLike[str], *, increment_revision: bool = True) -> None:
    # Internal marker used only while opening an older batch-preannotation
    # project.  It must never leak into the durable project schema.
    project.pop("_paperflow_scope_migration_pending", None)
    project["annotation_version"] = 5
    try:
        revision = max(0, int(project.get("project_revision", 0)))
    except (TypeError, ValueError):
        revision = 0
    project["project_revision"] = revision + 1 if increment_revision else revision
    ensure_ids(project)
    ensure_delivery_ids(project)
    project["updated_at"] = _now()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    _mirror_project_to_shared_delivery(project)


def is_company_public_delivery_root(value: object) -> bool:
    """Return whether a legacy PaperFlow public-share path was selected.

    Final delivery now travels through the authenticated review service.  Keep
    this narrow allowlist of retired paths so local folders and unrelated
    company shares are not unexpectedly changed.
    """
    normalized = str(value or "").strip().replace("/", "\\").rstrip("\\").casefold()
    return normalized == r"\\192.168.1.204\paperflowtest\projects" or normalized == r"f:\paperflowdata\public\projects"


def _mirror_project_to_shared_delivery(project: dict) -> None:
    """Publish a read-only progress snapshot beside a shared delivery package.

    The local annotation project remains authoritative.  A transient SMB failure
    must never prevent a staff member from saving local work, so this best-effort
    mirror is intentionally non-fatal and will be refreshed by the next save.
    """
    configured_root = str(project.get("delivery_output_root", "") or "").strip()
    book_id = str(project.get("delivery_book_id", "") or "").strip()
    if not configured_root or not book_id or is_company_public_delivery_root(configured_root):
        return
    root = Path(configured_root).expanduser()
    if not root.is_absolute():
        return
    target = root.resolve() / book_id / "PaperFlow标注项目.json"
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _migrate_batch_page_scopes_to_frame_regions(project: dict) -> bool:
    """Convert legacy cutter page-wide scopes into the actual emitted frames.

    Only the old batch-preannotation format is migrated.  A staff member's
    deliberate page-wide scope must retain its meaning, so normal/manual
    projects are never changed here.
    """
    if str(project.get("preannotation_mode", "") or "") != "paperflow":
        return False
    changed = False
    for pages_key, records_key in (("pages", "questions"), ("answer_pages", "answers")):
        for page in project.get(pages_key, []):
            page_scope = str(page.get("match_scope_id", "") or "").strip()
            regions = page.get("match_scope_regions", [])
            if not page_scope or not isinstance(regions, list) or regions:
                continue
            migrated_regions = []
            seen = set()
            for record in page.get(records_key, []):
                record_scope = str(record.get("match_scope_id", "") or "").strip()
                # Old batch output used the page as a fallback for ordinary
                # records.  Explicitly unassigned fallback answer frames must
                # stay out of any group.
                if not record_scope and record.get("origin") != "batch_unmatched_answer":
                    record_scope = page_scope
                    record["match_scope_id"] = page_scope
                if record_scope != page_scope:
                    continue
                box = record.get("bbox", [])
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    normalized_box = [int(round(float(value))) for value in box]
                except (TypeError, ValueError, OverflowError):
                    continue
                if normalized_box[2] <= normalized_box[0] or normalized_box[3] <= normalized_box[1]:
                    continue
                key = (record_scope, tuple(normalized_box))
                if key in seen:
                    continue
                seen.add(key)
                migrated_regions.append({
                    "id": "region-prelabel-" + str(record.get("id", "") or len(migrated_regions) + 1),
                    "scope_id": record_scope,
                    "bbox": normalized_box,
                })
            if migrated_regions:
                page["match_scope_id"] = ""
                page["match_scope_regions"] = migrated_regions
                changed = True
    return changed


def _migrate_region_scope_overrides(project: dict) -> bool:
    """Resolve the old "whole page + later region" assignment ambiguity.

    A region selection is an explicit, later user action and therefore replaces
    a previous page-wide default. Earlier UI versions retained both fields,
    which made the coverage panel report "whole page" and kept unselected
    frames in the old group. Materialise the region memberships on the records,
    then remove the stale page default.
    """
    changed = False
    for pages_key, records_key in (("pages", "questions"), ("answer_pages", "answers")):
        for page in project.get(pages_key, []):
            page_scope = str(page.get("match_scope_id", "") or "").strip()
            regions = page.get("match_scope_regions", [])
            if not page_scope or not isinstance(regions, list) or not regions:
                continue
            valid_regions = []
            for region in regions:
                if not isinstance(region, dict):
                    continue
                scope_id = str(region.get("scope_id", "") or "").strip()
                box = region.get("bbox", [])
                if not scope_id or not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x0, y0, x1, y1 = (float(value) for value in box)
                except (TypeError, ValueError, OverflowError):
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue
                valid_regions.append((scope_id, x0, y0, x1, y1))
            if not valid_regions:
                continue
            for record in page.get(records_key, []):
                if str(record.get("match_scope_id", "") or "").strip() == page_scope:
                    record["match_scope_id"] = ""
                    changed = True
            for scope_id, x0, y0, x1, y1 in valid_regions:
                for record in page.get(records_key, []):
                    box = record.get("bbox", [])
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    try:
                        center_x = (float(box[0]) + float(box[2])) / 2
                        center_y = (float(box[1]) + float(box[3])) / 2
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                        if str(record.get("match_scope_id", "") or "") != scope_id:
                            record["match_scope_id"] = scope_id
                            changed = True
            page["match_scope_id"] = ""
            changed = True
    return changed


def load_project(path: str | os.PathLike[str]) -> dict:
    project = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(project.get("pages"), list):
        raise ValueError("不是可用的题目页标注文件")
    project.setdefault("kind", "papercutter_manual_annotation")
    project["annotation_version"] = 5
    try:
        project["project_revision"] = max(0, int(project.get("project_revision", 0)))
    except (TypeError, ValueError):
        project["project_revision"] = 0
    project.setdefault("answer_image_root", "")
    project.setdefault("answer_pages", [])
    project.setdefault("ignored_pages", [])
    project.setdefault("match_scopes", [])
    answer_mode = str(project.get("answer_mode", "textbook") or "textbook").strip()
    project["answer_mode"] = answer_mode if answer_mode in ANSWER_MODES else "textbook"
    project.setdefault("pairs", [])
    for page in project["pages"]:
        page.setdefault("questions", [])
        page.setdefault("match_scope_id", "")
        page.setdefault("match_scope_regions", [])
    for page in project["answer_pages"]:
        page.setdefault("answers", [])
        page.setdefault("match_scope_id", "")
        page.setdefault("match_scope_regions", [])
    for page in project["ignored_pages"]:
        page.setdefault("questions", [])
        page.setdefault("answers", [])
        page.setdefault("original_kind", "question")
        page.setdefault("match_scope_id", "")
        page.setdefault("match_scope_regions", [])
    if _migrate_batch_page_scopes_to_frame_regions(project):
        project["_paperflow_scope_migration_pending"] = True
    if _migrate_region_scope_overrides(project):
        project["_paperflow_scope_migration_pending"] = True
    ensure_ids(project)
    ensure_delivery_ids(project)
    # 老工程读出来补齐 14 个补全字段：UI/交付/分级三处都直接读，不再各自兜底。
    # 开关关闭时整段跳过（读盘结果与接入前一致）。
    if enrich_fields_enabled():
        for pages_key, records_key in (("pages", "questions"), ("answer_pages", "answers"),
                                      ("ignored_pages", "questions"), ("ignored_pages", "answers")):
            for page in project.get(pages_key) or []:
                if not isinstance(page, dict):
                    continue
                for record in page.get(records_key) or []:
                    apply_enrich_defaults(record)
    return project


def ensure_ids(project: dict) -> None:
    for page in project.get("pages", []):
        for record in page.get("questions", []):
            record.setdefault("id", "q-" + uuid.uuid4().hex)
            normalize_record(record)
        groups = [
            record for record in page.get("questions", [])
            if record.get("region_role") == "group"
        ]
        for record in page.get("questions", []):
            if record.get("parent_id") or record.get("region_role") != "question":
                continue
            path = number_path(record)
            candidates = [
                group for group in groups
                if number_path(group) and path[:len(number_path(group))] == number_path(group)
                and len(path) > len(number_path(group))
            ]
            if candidates:
                parent = max(candidates, key=lambda item: len(number_path(item)))
                record["parent_id"] = parent["id"]
    for page in project.get("answer_pages", []):
        for record in page.get("answers", []):
            record.setdefault("id", "a-" + uuid.uuid4().hex)
            normalize_record(record)


def source_path(project: dict, page: dict, answer: bool = False) -> str:
    root = project.get("answer_image_root" if answer else "image_root", "")
    resolved_root = Path(str(root)).resolve()
    rotation = int(page.get("rotation", 0) or 0) % 360
    relative_value = page.get("rotated_image", "") if rotation else page.get("image", "")
    relative = Path(str(relative_value or ""))
    if relative.is_absolute():
        raise ValueError("教材页面路径必须使用教材目录内的相对路径。")
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("教材页面路径超出当前教材目录。") from error
    return str(candidate)


def original_source_path(project: dict, page: dict, answer: bool = False) -> str:
    """Resolve the immutable textbook image, ignoring a page rotation view."""
    root = project.get("answer_image_root" if answer else "image_root", "")
    resolved_root = Path(str(root)).resolve()
    relative = Path(str(page.get("image", "") or ""))
    if relative.is_absolute():
        raise ValueError("教材页面路径必须使用教材目录内的相对路径。")
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("教材页面路径超出当前教材目录。") from error
    return str(candidate)


def reclassify_pages(project: dict, source_kind: str, page_indices: list[int],
                     target_kind: str) -> dict:
    """Move source pages between question, answer and ignored collections.

    Ignoring a page preserves its records and original kind, so restoring it is
    lossless. Moving an annotated page directly between question and answer
    converts record identities and removes stale pair references.
    """
    old_question_pages = list(project.get("pages", []))
    collection = {"question": "pages", "answer": "answer_pages", "ignored": "ignored_pages"}
    records_key = {"question": "questions", "answer": "answers"}
    if source_kind not in collection or target_kind not in collection:
        raise ValueError("页面分类无效。")
    if source_kind == target_kind:
        return {"moved": 0, "removed_pairs": 0}
    source_pages = project.setdefault(collection[source_kind], [])
    target_pages = project.setdefault(collection[target_kind], [])
    indices = sorted({int(value) for value in page_indices}, reverse=True)
    if not indices or any(index < 0 or index >= len(source_pages) for index in indices):
        raise ValueError("页面序号无效。")

    removed_question_ids: set[str] = set()
    removed_answer_ids: set[str] = set()
    moved_pages = []
    for index in indices:
        page = source_pages.pop(index)
        actual_source = source_kind
        if source_kind == "ignored":
            actual_source = str(page.pop("original_kind", "question"))
            if actual_source not in {"question", "answer"}:
                actual_source = "question"
        if target_kind == "ignored":
            hidden_records = page.get(records_key[actual_source], [])
            hidden_ids = {str(record.get("id", "")) for record in hidden_records}
            if actual_source == "question":
                removed_question_ids.update(hidden_ids)
            else:
                removed_answer_ids.update(hidden_ids)
            page["original_kind"] = actual_source
            page.setdefault("questions", [])
            page.setdefault("answers", [])
            moved_pages.append(page)
            continue

        if actual_source != target_kind:
            old_key = records_key[actual_source]
            new_key = records_key[target_kind]
            records = list(page.pop(old_key, []))
            for record in records:
                old_id = str(record.get("id", ""))
                if actual_source == "question":
                    removed_question_ids.add(old_id)
                else:
                    removed_answer_ids.add(old_id)
                record["id"] = ("a-" if target_kind == "answer" else "q-") + uuid.uuid4().hex
                record.pop("delivery_id", None)
                record["parent_id"] = ""
                record["continuation_of"] = ""
            page[new_key] = records
            page["status"] = "draft" if records else "unlabeled"
        else:
            page.setdefault(records_key[target_kind], [])
        moved_pages.append(page)

    target_pages.extend(reversed(moved_pages))
    target_pages.sort(key=lambda page: _image_key(Path(str(page.get("image", "")))))

    old_pairs = list(project.get("pairs", []))
    cleaned_pairs = []
    for pair in old_pairs:
        if str(pair.get("question_id", "")) in removed_question_ids:
            continue
        answer_ids = [
            str(value) for value in pair.get("answer_ids", [])
            if str(value) not in removed_answer_ids
        ]
        if pair.get("answer_ids") and not answer_ids:
            continue
        pair["answer_ids"] = answer_ids
        cleaned_pairs.append(pair)
    project["pairs"] = cleaned_pairs
    project["answer_image_root"] = project.get("book_root", project.get("image_root", "")) \
        if project.get("answer_pages") else ""

    # Cross-page shared material used to store a volatile array index.  Remap
    # it through the page image identity after any pop/sort so it can never
    # silently point at a different textbook page.  If the referenced page was
    # moved out of the question collection, dropping the segment is safer than
    # cropping unrelated content.
    # Page objects retain identity while this in-memory move runs.  Mapping by
    # object identity is unambiguous even for a malformed historical cache that
    # contains the same image path more than once.
    new_question_indexes = {
        id(page): index for index, page in enumerate(project.get("pages", []))
    }
    removed_context_segments = 0
    for pages_key in ("pages", "ignored_pages"):
        for page in project.get(pages_key, []):
            for record in page.get("questions", []):
                remapped = []
                for segment in record.get("shared_context_segments", []):
                    try:
                        old_index = int(segment.get("page_index", -1))
                    except (AttributeError, TypeError, ValueError):
                        removed_context_segments += 1
                        continue
                    if old_index < 0 or old_index >= len(old_question_pages):
                        removed_context_segments += 1
                        continue
                    new_index = new_question_indexes.get(id(old_question_pages[old_index]))
                    if new_index is None:
                        removed_context_segments += 1
                        continue
                    remapped.append({**segment, "page_index": new_index})
                record["shared_context_segments"] = remapped
    ensure_ids(project)
    ensure_delivery_ids(project)
    return {
        "moved": len(indices),
        "removed_pairs": len(old_pairs) - len(cleaned_pairs),
        "removed_context_segments": removed_context_segments,
    }


def new_record(box: list[int], answer: bool = False, **values) -> dict:
    record = {
        "id": ("a-" if answer else "q-") + uuid.uuid4().hex,
        "bbox": [int(round(value)) for value in box],
        "origin": "manual",
    }
    record.update(values)
    return normalize_record(record)


def all_questions(project: dict) -> list[tuple[int, dict, dict]]:
    return [(index, page, record) for index, page in enumerate(project.get("pages", []))
            for record in page.get("questions", [])]


def all_answers(project: dict) -> list[tuple[int, dict, dict]]:
    return [(index, page, record) for index, page in enumerate(project.get("answer_pages", []))
            for record in page.get("answers", [])]


def paired_answer_ids(project: dict, question_id: str) -> list[str]:
    for pair in project.get("pairs", []):
        if pair.get("question_id") == question_id:
            return list(pair.get("answer_ids", []))
    return []


def number_path_matches(left: dict, right: dict) -> bool:
    """Return true only when both records have the same complete number hierarchy."""
    left_path = canonical_number_path(left)
    right_path = canonical_number_path(right)
    return bool(left_path and right_path and left_path == right_path)


def candidate_reason(question: dict, answer: dict) -> str:
    """Give staff a short, non-technical explanation for a suggested answer."""
    if number_path_matches(question, answer):
        return "完整题号层级一致"
    reasons = []
    if (_canonical_number(question.get("major_number", ""))
            and _canonical_number(question.get("major_number", ""))
            == _canonical_number(answer.get("major_number", ""))):
        reasons.append("大题号一致")
    if (_canonical_number(question.get("minor_number", question.get("number", "")))
            and _canonical_number(question.get("minor_number", question.get("number", "")))
            == _canonical_number(answer.get("minor_number", answer.get("number", "")))):
        reasons.append("小题号一致")
    return "、".join(reasons) or "文字与顺序参考"


def set_pair(project: dict, question_id: str, answer_ids: list[str], status: str = "accepted",
             source: str = "manual") -> None:
    previous = next(
        (pair for pair in project.get("pairs", []) if pair.get("question_id") == question_id),
        None,
    )
    project["pairs"] = [pair for pair in project.get("pairs", []) if pair.get("question_id") != question_id]
    pair = {
        "id": str((previous or {}).get("id", "")) or "pair-" + uuid.uuid4().hex,
        "question_id": question_id,
        "answer_ids": list(dict.fromkeys(answer_ids)),
        "status": status,
        "source": str(source or "manual"),
        "updated_at": _now(),
    }
    if previous and previous.get("delivery_id"):
        pair["delivery_id"] = previous["delivery_id"]
    project["pairs"].append(pair)


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]{1,6}|[A-Za-z]{2,}|\d+", str(value or "").lower()))


def is_match_ready(page: dict, record: dict) -> bool:
    """Whether one confirmed record is currently available to answer matching.

    An answer page can be confirmed as a page while the reviewer deliberately
    exposes only selected match-scope rectangles to matching.  Older projects
    have no selection field and retain the historical whole-page behaviour.
    """
    if str(page.get("status", "")) != "accepted":
        return False
    # New confirmations persist the exact answer-frame IDs rather than only
    # the selected scope rectangles.  This makes "select all on this page"
    # include ungrouped frames too, and keeps the result stable after reload.
    if "confirmed_match_record_ids" in page:
        return str(record.get("id", "")).strip() in {
            str(value).strip()
            for value in page.get("confirmed_match_record_ids", [])
            if str(value).strip()
        }
    regions = page.get("match_scope_regions", [])
    if not isinstance(regions, list) or not regions:
        return True
    if "confirmed_match_scope_region_ids" not in page:
        return True
    selected_ids = {
        str(value).strip()
        for value in page.get("confirmed_match_scope_region_ids", [])
        if str(value).strip()
    }
    if not selected_ids:
        return False
    box = record.get("bbox", []) if isinstance(record, dict) else []
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        center_x = (float(box[0]) + float(box[2])) / 2
        center_y = (float(box[1]) + float(box[3])) / 2
    except (TypeError, ValueError, OverflowError):
        return False
    for region in regions:
        if not isinstance(region, dict) or str(region.get("id", "")).strip() not in selected_ids:
            continue
        region_box = region.get("bbox", [])
        if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in region_box)
        except (TypeError, ValueError, OverflowError):
            continue
        if x0 <= center_x <= x1 and y0 <= center_y <= y1:
            return True
    return False


def candidates(project: dict, question: dict, limit: int = 30,
               accepted_only: bool = False) -> list[tuple[float, int, dict, dict]]:
    """Rank local candidates; it is assistance, never an automatic pairing."""
    normalize_record(question)
    if question.get("region_role") != "question":
        return []
    minor_number = _canonical_number(question.get("minor_number", ""))
    major_number = _canonical_number(question.get("major_number", ""))
    subsection_number = _canonical_number(question.get("subsection_number", ""))
    question_path = [_canonical_number(value) for value in number_path(question)]
    context = _tokens(" ".join(str(question.get(key, "")) for key in ("major_title", "context", "text")))
    ranked = []
    for page_index, page, answer in all_answers(project):
        if accepted_only and not is_match_ready(page, answer):
            continue
        normalize_record(answer)
        if answer.get("region_role") != "question":
            continue
        score = 0.0
        answer_minor = _canonical_number(answer.get("minor_number", ""))
        answer_major = _canonical_number(answer.get("major_number", ""))
        answer_subsection = _canonical_number(answer.get("subsection_number", ""))
        answer_path = [_canonical_number(value) for value in number_path(answer)]
        minor_match = bool(minor_number and answer_minor and minor_number == answer_minor)
        major_match = bool(major_number and answer_major and major_number == answer_major)
        subsection_match = bool(
            subsection_number and answer_subsection and subsection_number == answer_subsection
        )
        if major_match and minor_match:
            score += 180.0
        else:
            if minor_match:
                score += 70.0 if not major_number or not answer_major else 35.0
            elif minor_number and answer_minor:
                score -= 35.0
            if major_match:
                score += 45.0
            elif major_number and answer_major:
                score -= 80.0
        if subsection_match:
            score += 90.0
        elif subsection_number and answer_subsection:
            score -= 120.0
        # Arbitrary-depth paths supersede the legacy single subsection field.
        # Reward each matching level and strongly demote a conflicting branch.
        if question_path and answer_path:
            common_depth = 0
            for left, right in zip(question_path, answer_path):
                if left != right:
                    break
                common_depth += 1
            score += common_depth * 45.0
            if common_depth < min(len(question_path), len(answer_path)):
                score -= 100.0
            if question_path == answer_path:
                score += 140.0
        common = context & _tokens(" ".join(str(answer.get(key, "")) for key in ("major_title", "context", "text")))
        score += min(60.0, float(len(common)) * 6.0)
        # Earlier answer pages generally correspond to earlier question groups.
        score -= page_index * 0.02
        ranked.append((score, page_index, page, answer))
    return sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]
