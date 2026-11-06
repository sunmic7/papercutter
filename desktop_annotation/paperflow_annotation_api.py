"""Small, local-only bridge between the Electron annotation desk and PaperFlow.

The Electron application owns the user interface and workspace files.  This
server deliberately exposes only a single page-at-a-time pre-annotation API;
it never uploads an image or starts a cloud model.  Keeping the bridge in the
standard library avoids adding a Flask dependency to every staff computer.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import Request, urlopen

class _LazyModule:
    """Keep native libraries off the authentication server's cold-start path."""
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attribute):
        return getattr(importlib.import_module(self._name), attribute)


def _lazy_function(module, name):
    def call(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)
    call.__name__ = name
    return call


np = _LazyModule('numpy')
_core = _LazyModule('cut_questions_v4')


def __getattr__(name):
    # Preserve callers that import these public constants from the bridge.
    if name in {'BOOK_PROFILE_CHOICES', 'BOOK_PROFILE_LABELS'}:
        return getattr(_core, name)
    raise AttributeError(name)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_answer_suggestions = _lazy_function('review_tools.manual_annotation_workspace', '_answer_suggestions')
_question_suggestions = _lazy_function('review_tools.manual_annotation_workspace', '_question_suggestions')
from review_tools import book_meta
from review_tools import enrich_snapshot
from review_tools import manual_annotation_data as annotation_data
from review_tools import review_levels
from review_tools.review_session import ReviewSession, REVIEWERS
from review_tools import online_review_sync as review_sync
export_project = _lazy_function('review_tools.export_manual_annotations', 'export_project')
delivery_root = _lazy_function('review_tools.delivery_workspace', 'delivery_root')
sync_delivery_workspace = _lazy_function('review_tools.delivery_workspace', 'sync_delivery_workspace')
analyze_delivery = _lazy_function('review_tools.delivery_workspace', 'analyze_delivery')
export_scope = _lazy_function('review_tools.group_delivery', 'export_scope')
preflight_scope = _lazy_function('review_tools.group_delivery', 'preflight_scope')
from review_tools.workspace_store import database_path, page_snapshot, sync_project
from review_tools.online_review_sync import (
    DECISIONS_FILENAME,
    ReviewConflictError,
    fetch_project_source_images,
    list_remote_projects,
    pull_project_snapshot,
    pull_review_decisions,
    push_project_snapshot,
    submit_review_snapshot,
    upload_project_source_images,
)
auto_book_profile = _lazy_function('cut_questions_v4', 'auto_book_profile')
detect_book_profile = _lazy_function('cut_questions_v4', 'detect_book_profile')
infer_book_profile = _lazy_function('cut_questions_v4', 'infer_book_profile')
ocr_image = _lazy_function('cut_questions_v4', 'ocr_image')
read_book_profile_cache = _lazy_function('cut_questions_v4', 'read_book_profile_cache')
from layout_strategy import describe_profile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
PROJECT_WRITE_LOCK = threading.RLock()
# PaddleOCR/PaddleX predictors are process-wide singletons.  The HTTP bridge is
# threaded, but a predictor invocation must not overlap another invocation.
PRELABEL_INFERENCE_LOCK = threading.Lock()
PRELABEL_INFLIGHT_LOCK = threading.Lock()
PRELABEL_INFLIGHT: dict[tuple, "_PrelabelFlight"] = {}
ANSWER_PRELABEL_REVIEW_THRESHOLD = 120
PAPERFLOW_BRIDGE_REVISION = "column-prefix-20261029"
REVIEW_SESSION = ReviewSession()
_configured_client_id = os.environ.get('PAPERFLOW_CLIENT_ID', '')
LIBRARY_CLIENT_ID = _configured_client_id if re.fullmatch(r'[a-f0-9]{32}', _configured_client_id) else uuid.uuid4().hex
MODEL_WARMUP_READY = threading.Event()


def _review_login(payload):
    result = REVIEW_SESSION.login(payload)
    MODEL_WARMUP_READY.set()
    return result


def _library_request(payload: dict) -> dict:
    recovery_target = None
    if payload.get("operation") == "/api/project/save-cache":
        status = REVIEW_SESSION.status()
        username = str(status.get("user", {}).get("username", "unknown"))
        recovery = ROOT / "runtime" / "server_recovery_drafts" / hashlib.sha256(username.encode()).hexdigest()[:16]
        recovery.mkdir(parents=True, exist_ok=True)
        book_id = str(payload.get("book_id") or "")
        if re.fullmatch(r"B[A-F0-9]{32}", book_id):
            target = recovery / (book_id + ".json")
            recovery_target = target
            temporary = target.with_suffix(".tmp")
            # No credentials or full textbook images are stored in recovery drafts.
            temporary.write_text(json.dumps({"book_id": book_id, "base_revision": payload.get("base_revision"),
                "project": (payload.get("payload") or {}).get("project")}, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, target)
    credentials = REVIEW_SESSION.credentials()
    base, token = credentials["server_url"], credentials["_review_token"]
    action = payload.get("action", "list")
    if action == "list":
        suffix = '?view=catalog' if payload.get('view') == 'catalog' else ''
        return review_sync._request_json(base + "/api/library" + suffix, token=token)
    if action == "assign":
        return review_sync._request_json(base + "/api/library/assign", "POST", {
            'book_ids': payload.get('book_ids') or [payload.get('book_id')],
            'annotator_id': payload.get('annotator_id'),
        }, token)
    if action == "scan":
        return review_sync._request_json(base + "/api/library/scan", "POST", {}, token)
    if action == "annotators":
        return review_sync._request_json(base + "/api/library/annotators", token=token)
    if action == "open":
        result = review_sync._request_json(base + "/api/library/open", "POST", {
            "book_id": payload["book_id"], "annotator_id": payload.get("annotator_id"),
            "client_id": LIBRARY_CLIENT_ID,
        }, token, timeout=300)
        username = credentials["username"]
        recovery = ROOT / "runtime" / "server_recovery_drafts" / hashlib.sha256(username.encode()).hexdigest()[:16]
        target = recovery / (str(payload["book_id"]) + ".json")
        if target.is_file():
            draft = json.loads(target.read_text(encoding="utf-8"))
            if not draft.get("synced"):
                if int(draft.get("base_revision", -1)) == int(result["project"].get("project_revision", 0)):
                    result["recovery_project"] = draft.get("project")
                else:
                    result["recovery_conflict"] = True
        return result
    result = review_sync._request_json(base + "/api/library/operate", "POST", {
        "book_id": payload["book_id"], "operation": payload["operation"],
        "payload": payload.get("payload") or {}, "client_id": LIBRARY_CLIENT_ID,
        "base_revision": payload.get("base_revision", -1),
    }, token, timeout=1800)
    if recovery_target and recovery_target.is_file():
        draft = json.loads(recovery_target.read_text(encoding="utf-8"))
        draft["synced"] = True
        temporary = recovery_target.with_suffix(".tmp")
        temporary.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, recovery_target)
    return result
# 总开关：置 0/false/no/off 时 _save_page 不冻结不分级、三条 enrich 路由直接返回
# {"ok": false, "disabled": true}，主流程行为与接入前完全一致。
ENRICH_DISABLED_VALUES = {"0", "false", "no", "off"}
# 整页 OCR 的"文本确实变了"阈值：整页重识别一次，逐字抖动不该把 20 题全打 stale。
OCR_TEXT_SIMILARITY_THRESHOLD = 0.98
MUTATING_ROUTES = {
    "/api/project/open-book", "/api/project/save-page", "/api/project/set-pair",
    "/api/project/sync-delivery", "/api/project/export", "/api/project/set-delivery-output",
    "/api/project/set-book-metadata",
    "/api/project/set-ocr-settings",
    "/api/project/reclassify-pages",
    "/api/project/rotate-page",
    "/api/project/accept-confident-matches",
    "/api/project/upsert-match-scope",
    "/api/project/assign-match-scope",
    "/api/project/remove-match-scope-pages",
    "/api/project/auto-match-scopes",
    "/api/project/save-cache",
    "/api/project/export-match-scope",
    "/api/project/enrich-prepare",
    "/api/review/pull-decisions",
    "/api/review/import-project",
    "/api/review/push-project",
    "/api/review/upload-source-images",
}


class _PrelabelFlight:
    """Result shared by simultaneous requests for the same source page."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: dict | None = None
        self.error: tuple[type[BaseException], str] | None = None


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _run_prelabel_once(key: tuple, operation) -> dict:
    """Coalesce identical in-flight work while serialising all model calls."""
    with PRELABEL_INFLIGHT_LOCK:
        flight = PRELABEL_INFLIGHT.get(key)
        owner = flight is None
        if owner:
            flight = _PrelabelFlight()
            PRELABEL_INFLIGHT[key] = flight

    assert flight is not None
    if owner:
        try:
            with PRELABEL_INFERENCE_LOCK:
                flight.result = _json_copy(operation())
        except Exception as exc:  # publish failures so waiters never hang
            flight.error = (type(exc), str(exc))
        finally:
            flight.event.set()
            with PRELABEL_INFLIGHT_LOCK:
                if PRELABEL_INFLIGHT.get(key) is flight:
                    PRELABEL_INFLIGHT.pop(key, None)
    else:
        flight.event.wait()

    if flight.error:
        error_type, message = flight.error
        if issubclass(error_type, ValueError):
            raise ValueError(message)
        raise RuntimeError(message)
    if flight.result is None:
        raise RuntimeError("预标注任务未返回结果。")
    return _json_copy(flight.result)


def _prelabel_diagnostics(records: list[dict], kind: str) -> dict:
    """Flag risky answer pages without deleting or silently merging candidates."""
    contexts: set[str] = set()
    numbered: dict[tuple[str, tuple[str, ...]], int] = {}
    missing_number_count = 0
    for record in records:
        context = str(record.get("context", "") or "").strip()
        if context:
            contexts.add(context)
        path = tuple(annotation_data.number_path(record))
        if not path:
            missing_number_count += 1
            continue
        key = (context, path)
        numbered[key] = numbered.get(key, 0) + 1
    duplicate_number_path_count = sum(max(0, count - 1) for count in numbered.values())

    warning_codes: list[str] = []
    warnings: list[str] = []
    if kind == "answer" and len(records) >= ANSWER_PRELABEL_REVIEW_THRESHOLD:
        warning_codes.append("dense_answer_candidates")
        warnings.append(
            f"本页识别到 {len(records)} 个答案候选，属于高密度答案页，请分段核对后再确认。"
        )
    if kind == "answer" and missing_number_count:
        warning_codes.append("missing_answer_number")
        warnings.append(f"有 {missing_number_count} 个答案候选缺少可配对题号。")
    if kind == "answer" and duplicate_number_path_count:
        warning_codes.append("duplicate_answer_number_path")
        warnings.append(f"有 {duplicate_number_path_count} 个答案候选与同上下文题号重复。")
    return {
        "record_count": len(records),
        "context_count": len(contexts),
        "missing_number_count": missing_number_count,
        "duplicate_number_path_count": duplicate_number_path_count,
        "requires_review": bool(warning_codes),
        "bulk_accept_safe": not warning_codes,
        "warning_codes": warning_codes,
        "warnings": warnings,
    }


def _remove_parent_question_boxes(project: dict) -> bool:
    """Flatten obsolete group/shared-material boxes without losing real questions."""
    changed = False
    for pages_key, records_key in (("pages", "questions"), ("answer_pages", "answers")):
        for page in project.get(pages_key, []):
            records = [record for record in page.get(records_key, []) if isinstance(record, dict)]
            removed_ids = {
                str(record.get("id", "")) for record in records
                if str(record.get("region_role", "question")) in {"group", "shared_material"}
            }
            kept = []
            for record in records:
                if str(record.get("region_role", "question")) in {"group", "shared_material"}:
                    changed = True
                    continue
                if str(record.get("parent_id", "")) in removed_ids:
                    record["parent_id"] = ""
                    changed = True
                kept.append(record)
            if len(kept) != len(records):
                page[records_key] = kept
    return changed


def _sync_page_profiles(project: dict, profile: str, previous: str = "") -> bool:
    """转发到数据层，保证各个入口对“页面级版式跟随”的口径一致。"""
    return annotation_data.sync_page_profiles(project, profile, previous)


def _store_book_profile(project: dict, profile: str, source: str, reason: str,
                        locked: bool | None = None) -> bool:
    previous = str(project.get("profile", "") or "").strip()
    previous_metadata = (project.get('profile_source'),project.get('profile_reason'))
    project["profile"] = profile
    project["profile_source"] = source
    project["profile_reason"] = reason
    if locked is True:
        project["profile_locked"] = True
    elif locked is False:
        project.pop("profile_locked", None)
    changed = profile != previous or previous_metadata != (source,reason)
    return _sync_page_profiles(project, profile, previous) or changed


def _auto_book_profile(book_root: Path) -> tuple[str, str, str]:
    """书名优先，其次复用已落盘的抽样结论。返回 (profile, source, reason)。"""
    return auto_book_profile(str(book_root), str(ROOT / "runtime"))


def _apply_book_profile(project: dict, requested: str, book_root: Path,
                        respect_lock: bool = True) -> bool:
    """把一次版式设置写进项目；``requested`` 允许 ``auto``。

    人工指定会置 ``profile_locked``，之后自动识别不再覆盖它。``respect_lock``
    为 False 时表示员工明确选择了“自动识别”，此时要主动解除人工锁定。
    """
    requested = str(requested or "").strip()
    if requested == "auto":
        if (
            respect_lock and project.get("profile_locked")
            and str(project.get("profile") or "") in _core.BOOK_PROFILE_CHOICES
        ):
            return False
        profile, source, reason = _auto_book_profile(book_root)
        return _store_book_profile(project, profile, source, reason, locked=False)
    if requested not in _core.BOOK_PROFILE_CHOICES:
        raise ValueError("教材版式填写无效。")
    return _store_book_profile(project, requested, "manual", "人工指定", locked=True)


def _resolve_book_profile(project: dict, image_path: Path) -> str:
    """预标注实际使用的版式：项目落盘值优先，其次磁盘缓存，最后按书名兜底。"""
    stored = str(project.get("profile", "") or "").strip()
    if project.get("profile_locked") and stored in _core.BOOK_PROFILE_CHOICES:
        return stored
    if stored in _core.BOOK_PROFILE_CHOICES and stored != "general":
        return stored
    book_root = Path(str(project.get("book_root") or project.get("image_root") or image_path.parent))
    cached = read_book_profile_cache(str(book_root), str(ROOT / "runtime"))
    if cached and cached.get("profile") in _core.BOOK_PROFILE_CHOICES and cached["profile"] != "general":
        return cached["profile"]
    if stored in _core.BOOK_PROFILE_CHOICES:
        return stored
    return infer_book_profile(str(book_root))


PROFILE_DETECT_JOBS: dict[str, threading.Thread] = {}
PROFILE_DETECT_LOCK = threading.Lock()
# 打开教材后先让界面把首页画出来，再占 CPU 抽样；否则首屏渲染和后台 OCR
# 抢同一批核。等待期间用户如果已经点“自动框题”，前台请求会先进推理锁。
PROFILE_DETECT_DELAY_SECONDS = 6.0


def _profile_detect_job(book_root: Path) -> None:
    """后台用抽样正文补判版式。

    只写版式字段并复用 ``increment_revision=False``：递增修订号会让前端下一次
    整本缓存保存因 base_revision 不匹配而失败，而这只是一次后台推断。
    """
    try:
        time.sleep(max(0.0, PROFILE_DETECT_DELAY_SECONDS))
        result = detect_book_profile(
            str(book_root), runtime_root=str(ROOT / "runtime"),
            ocr_guard=PRELABEL_INFERENCE_LOCK,
        )
    except Exception as error:  # noqa: BLE001 - 后台推断失败必须静默降级
        print(f"[版式] 自动识别失败（不影响切题）：{error}", flush=True)
        return
    if result.get("source") != "text" or result.get("profile") == "general":
        return
    project_path = annotation_data.default_book_project_path(str(book_root), str(ROOT / "runtime"))
    if not project_path.is_file():
        return
    try:
        with PROJECT_WRITE_LOCK:
            project = annotation_data.load_project(project_path)
            if project.get("profile_locked"):
                return
            if (str(project.get("profile") or "") != "general"
                    and not (project.get('profile_source') in {'auto_name','default'}
                             and _core.book_profile_needs_confirmation(str(book_root)))):
                return
            _store_book_profile(
                project, result["profile"], "auto_text", str(result.get("reason", "") or ""), locked=None,
            )
            annotation_data.save_project(project, project_path, increment_revision=False)
        print(f"[版式] 抽样正文判定为 {result['profile']}：{result.get('reason', '')}", flush=True)
    except Exception as error:  # noqa: BLE001 - 结果落盘失败不影响预标注
        print(f"[版式] 判定结果落盘失败（不影响切题）：{error}", flush=True)


def _profile_detection_enabled() -> bool:
    flag = os.environ.get("PAPERFLOW_DISABLE_PROFILE_DETECT", "").strip().lower()
    return flag not in {"1", "true", "yes", "on"}


def _start_book_profile_detection(book_root: Path) -> None:
    if not _profile_detection_enabled():
        return
    key = str(Path(book_root).resolve()).casefold()
    with PROFILE_DETECT_LOCK:
        running = PROFILE_DETECT_JOBS.get(key)
        if running is not None and running.is_alive():
            return
        thread = threading.Thread(
            target=_profile_detect_job, args=(Path(book_root),),
            name="paperflow-profile-detect", daemon=True,
        )
        PROFILE_DETECT_JOBS[key] = thread
    thread.start()


def _prelabel(payload: dict) -> dict:
    image_path = Path(str(payload.get("image_path", ""))).expanduser()
    if not image_path.is_absolute() or not image_path.is_file():
        raise ValueError("当前页图片不存在，无法生成预标注框。")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("当前页不是支持的图片格式。")

    profile = str(payload.get("profile", "") or "").strip()
    kind = str(payload.get("kind", "question") or "question")
    force_refresh = bool(payload.get("force_refresh", False))
    refresh_token = str(payload.get("refresh_token", "") or "")
    page_index = int(payload.get("page_index", -1))
    project = None
    if payload.get("project_path"):
        _project_path, project = _load_project(payload)
        pages_key = "answer_pages" if kind == "answer" else "pages"
        pages = project.get(pages_key, [])
        if page_index < 0 or page_index >= len(pages):
            raise ValueError("预标注页面序号无效。")
        expected = Path(annotation_data.source_path(project, pages[page_index], kind == "answer")).resolve()
        if image_path.resolve() != expected:
            raise ValueError("预标注图片与当前教材页面不匹配。")
        requested_auto_profile = not profile or profile == "auto"
        if requested_auto_profile:
            page_profile = str(pages[page_index].get("profile", "") or "").strip()
            # Answer parsing is unusually sensitive to this legacy setting;
            # preserve a page's recorded profile unless it is empty.  Question
            # pages may safely infer a more specific scene from a general
            # placeholder (for example a book titled "大试卷").
            profile = page_profile if (kind == "answer" or page_profile != "general") else ""
        if not profile:
            profile = _resolve_book_profile(project, image_path)
    if not profile or profile == "auto":
        profile = infer_book_profile(str(image_path.parent))
    if kind not in {"question", "answer"}:
        raise ValueError("未知的预标注类型。")

    previous_question = None
    if kind == "question" and project is not None and page_index >= 0:
        pages = project.get("pages", [])
        for previous_index in range(min(page_index, len(pages)) - 1, max(-1, page_index - 2), -1):
            candidates = [
                record for record in pages[previous_index].get("questions", [])
                if annotation_data.normalize_record(record).get("region_role") == "question"
            ]
            if candidates:
                previous_question = candidates[-1]
                break

    source_stat = image_path.stat()
    previous_signature = tuple(
        str(previous_question.get(field, "")) if previous_question else ""
        for field in ("id", "major_number", "subsection_number", "minor_number", "bbox")
    )
    request_key = (
        str(image_path.resolve()).casefold(), source_stat.st_size, source_stat.st_mtime_ns,
        kind, profile, page_index, previous_signature,
        # A manual regenerate must never join an earlier in-flight request.
        "force-refresh" if force_refresh else "cached-ocr-ok",
        refresh_token if force_refresh else "",
    )

    def infer() -> dict:
        if kind == "answer":
            records = _answer_suggestions(
                str(image_path), profile, force_refresh=force_refresh,
            )
        else:
            records = _question_suggestions(
                str(image_path), profile, page_index=page_index,
                previous_question=previous_question, force_refresh=force_refresh,
            )
            # Suggested question numbers use the same canonical digit form that
            # the front end persists on save: 中文/圆括号/全角 digits collapse to
            # digits, so a freshly generated box and a re-opened box display the
            # identical label instead of switching 四-10 -> 4-10 after page hop.
            records = [
                annotation_data.normalize_record(
                    {**record, "number_path": list(annotation_data.canonical_number_path(record))}
                )
                for record in records
            ]
        return {
            "ok": True,
            "records": records,
            "engine": "paperflow-local",
            "bridge_revision": PAPERFLOW_BRIDGE_REVISION,
            "force_refresh": force_refresh,
            "profile": profile,
            "diagnostics": _prelabel_diagnostics(records, kind),
        }

    return _run_prelabel_once(request_key, infer)


def _project_response(project_path: Path, project: dict) -> dict:
    snapshot = json.loads(json.dumps(project, ensure_ascii=False))
    for pages_key, root_key in (("pages", "image_root"), ("answer_pages", "answer_image_root")):
        pages = snapshot.get(pages_key, [])
        if pages:
            resolved_root = Path(str(project.get(root_key, ""))).resolve()
            for page in pages:
                page["source_path"] = annotation_data._source_path_in_root(resolved_root, page)
    for page in snapshot.get("ignored_pages", []):
        page["source_path"] = str(Path(str(project.get("book_root") or project.get("image_root", ""))) / page.get("image", ""))
    return {
        "ok": True,
        "project_path": str(project_path),
        "project": snapshot,
        # 版式中文名只在这里维护一份，界面按它渲染下拉框，避免前端另抄一套。
        "book_profiles": [
            {"value": value, "label": label} for value, label in _core.BOOK_PROFILE_LABELS.items()
        ],
        # 每个版式实际启用了哪些特殊处理，界面在选版式时直接显示，
        # 让人知道“改成试卷”到底会改变什么。
        "book_profile_notes": {
            value: describe_profile(value)["notes"] for value in _core.BOOK_PROFILE_LABELS
        },
    }


def _annotation_count(project: dict) -> int:
    return sum(len(page.get(field, [])) for pages, field in (
        (project.get("pages", []), "questions"),
        (project.get("answer_pages", []), "answers"),
    ) for page in pages)


def _book_content_signature(project: dict) -> tuple:
    """Identify a copied/moved textbook without relying on its absolute path."""
    result = []
    for pages_key, answer in (("pages", False), ("answer_pages", True)):
        pages = project.get(pages_key, [])
        entries = []
        for page in pages:
            relative = str(page.get("image", "")).replace("\\", "/")
            source = Path(annotation_data.source_path(project, page, answer))
            try:
                size = source.stat().st_size
            except OSError:
                size = -1
            entries.append((relative, size))
        result.append(tuple(entries))
    return tuple(result)


def _ordered_subsequence(needle: tuple, haystack: tuple) -> bool:
    if not needle:
        return not haystack
    position = 0
    for item in haystack:
        if item == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False


def _same_book_content(left: tuple, right: tuple) -> bool:
    return all(a == b or _ordered_subsequence(a, b) or _ordered_subsequence(b, a)
               for a, b in zip(left, right))


def _recover_moved_book_project(book_root: Path, current: dict) -> dict | None:
    """Recover annotations when the same textbook folder was copied or renamed."""
    if _annotation_count(current):
        return None
    signature = _book_content_signature(current)
    candidates = []
    central_root = ROOT / "runtime" / "manual_annotation_projects"
    for candidate_path in central_root.glob("*.json"):
        try:
            candidate = annotation_data.load_project(candidate_path)
            count = _annotation_count(candidate)
            if count and _same_book_content(_book_content_signature(candidate), signature):
                candidates.append((count, candidate_path.stat().st_mtime_ns, candidate))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    recovered = max(candidates, key=lambda item: (item[0], item[1]))[2]
    recovered["book_root"] = str(book_root)
    recovered["image_root"] = str(book_root)
    if recovered.get("answer_pages"):
        recovered["answer_image_root"] = str(book_root)
    return recovered


def _open_book(payload: dict) -> dict:
    book_root = Path(str(payload.get("book_root", ""))).expanduser().resolve()
    if not book_root.is_dir():
        raise ValueError("教材目录不存在。")
    project_path = annotation_data.default_book_project_path(book_root, ROOT / "runtime")
    if project_path.is_file():
        project = annotation_data.load_project(project_path)
        # Persist the one-time legacy batch-scope migration as soon as the
        # textbook is opened, so the matching dialog always reflects frames
        # rather than a misleading full answer page.
        changed = bool(project.pop("_paperflow_scope_migration_pending", False))
        ocr_settings = project.get("ocr_settings")
        if not isinstance(ocr_settings, dict):
            # 确认页面只处理尚未识别或已改框的区域，未改过的已有文字不会
            # 再调用云端模型，因此可以安全地作为标注流程的一步。
            project["ocr_settings"] = {"auto_page_ocr_on_confirm": True}
            changed = True
        elif ocr_settings.get("auto_page_ocr_on_confirm") is not True:
            # 迁移旧工程：确认页面现在只识别待处理框。
            ocr_settings["auto_page_ocr_on_confirm"] = True
            changed = True
        # Old clients stored the company SMB share as a delivery root.  It is
        # publicly mounted for legacy reads, so clear only that exact retired
        # root on open; newly generated packages then stay local until the
        # employee explicitly uploads them through the authenticated service.
        if annotation_data.is_company_public_delivery_root(project.get("delivery_output_root")):
            project.pop("delivery_output_root", None)
            changed = True
        # 教材文件夹被整体复制/搬移时，随行的项目文件里 book_root / image_root
        # 仍指向旧位置；后续所有走 _load_project 的接口都会报"标注项目路径与
        # 教材目录不匹配"。项目文件就放在本次打开的教材目录旁，它属于这本教材，
        # 因此以实际打开目录为准把根字段改回来（自愈），而不是让用户手工改 JSON。
        for stale_root_key in ("book_root", "image_root", "answer_image_root"):
            stale_root = project.get(stale_root_key)
            if stale_root and Path(str(stale_root)).resolve() != book_root:
                project[stale_root_key] = str(book_root)
                changed = True
        changed = bool(annotation_data.remove_cached_delivery_pages(project, book_root)) or changed
        changed = _remove_parent_question_boxes(project) or changed
        # 版式画像落盘：以前每次 /api/prelabel 都按目录名现猜，结果不落盘、
        # 界面不可见、判错改不了。打开教材时统一确定一次，后续预标注直接复用。
        changed = _apply_book_profile(project, "auto", book_root) or changed
        if changed:
            annotation_data.save_project(project, project_path)
    else:
        project = annotation_data.create_book_project(str(book_root))
        _apply_book_profile(project, "auto", book_root)
        annotation_data.save_project(project, project_path)
    if (
        (str(project.get("profile") or "general") == "general"
         or _core.book_profile_needs_confirmation(str(book_root)))
        and not project.get("profile_locked")
    ):
        # 书名判不出具体版式时后台抽样正文补判；判出结果只落缓存与版式字段。
        _start_book_profile_detection(book_root)
    # Opening should be proportional to the source pages, not to every field in
    # every historical record. The SQLite read model is refreshed lazily only
    # when its workspace endpoint is requested.
    return _project_response(project_path, project)


def _load_project(payload: dict) -> tuple[Path, dict]:
    project_path = Path(str(payload.get("project_path", ""))).expanduser().resolve()
    if not project_path.is_file():
        raise ValueError("标注项目不存在。")
    project = annotation_data.load_project(project_path)
    book_root = Path(str(project.get("book_root") or project.get("image_root") or "")).resolve()
    central_root = (ROOT / "runtime" / "manual_annotation_projects").resolve()
    is_central = project_path.parent == central_root
    is_beside_book = project_path.name == "PaperFlow标注项目.json" and project_path.parent == book_root
    server_root = os.environ.get("PAPERFLOW_CENTRAL_PROJECT_ROOT", "")
    is_server = bool(server_root and Path(server_root).resolve() in project_path.parents)
    if not is_central and not is_beside_book and not is_server:
        raise ValueError("标注项目路径与教材目录不匹配。")
    return project_path, project


def _set_ocr_settings(payload: dict) -> dict:
    """Persist per-textbook OCR behaviour instead of relying on one browser profile."""
    project_path, project = _load_project(payload)
    enabled = payload.get("auto_page_ocr_on_confirm")
    if not isinstance(enabled, bool):
        raise ValueError("自动整页识别设置无效。")
    project["ocr_settings"] = {"auto_page_ocr_on_confirm": enabled}
    annotation_data.save_project(project, project_path)
    return {
        "ok": True,
        "ocr_settings": dict(project["ocr_settings"]),
        "project_revision": project["project_revision"],
    }


def _enrich_enabled() -> bool:
    """补全流水线总开关。每次现读环境变量，便于单测与灰度逐本切换。"""
    value = os.environ.get("PAPERFLOW_ENRICH", "1")
    return str(value).strip().lower() not in ENRICH_DISABLED_VALUES


def _enrich_disabled_response() -> dict:
    return {"ok": False, "disabled": True,
            "error": "补全流水线已按 PAPERFLOW_ENRICH 关闭。"}


def _save_page_enrich(project: dict, kind: str, page_index: int, *,
                      accepted: bool) -> dict:
    """页保存后的冻结/失效 + 单页分级 + ``review_summary`` 回写。

    只在``PAPERFLOW_ENRICH`` 开启时生效：
    - 题目页确认 → 逐条 ``freeze_page``（record 级失效粒度，§3.6）
    - 题目页**未**确认 → 只重算级别、不冻快照（§3.4 只允许 accepted 冻结）。
      这一段是必要的：``PAPERFLOW_ENRICH=0`` 期间存过的记录没有快照，
      ``load_project`` 会按默认值补 ``review_level='L2'``；若不重算，这些记录
      会永久挂着一个"算出来的 L2"，账永远对不上（QA Round 8 第 3 条）。
    - 答案页确认 → ``refresh_answer_segment`` 回写"答案后到"（§3.4）并回刷级别
    """
    if not _enrich_enabled():
        return {"enabled": False}
    pages_key = "answer_pages" if kind == "answer" else "pages"
    pages = project.get(pages_key) or []
    if page_index < 0 or page_index >= len(pages):
        return {"enabled": True, "skipped": "page_index_out_of_range"}
    page = pages[page_index]
    if not isinstance(page, dict):
        return {"enabled": True, "skipped": "page_not_dict"}
    if kind == "answer":
        if not accepted:
            return {"enabled": True, "skipped": "page_not_accepted"}
        answer_ids = [
            str(record.get("id", ""))
            for record in page.get("answers") or []
            if isinstance(record, dict) and record.get("id")
        ]
        return {"enabled": True, "answer_refresh": _guard_enrich(
            project, kind, page_index, True,
            lambda: enrich_snapshot.refresh_answer_segment(project, answer_ids),
        )}
    if not accepted:
        levels = enrich_snapshot.refresh_page_levels(project, page_index, kind="question")
        return {
            "enabled": True,
            "skipped": "page_not_accepted",
            "review_summary": levels["review_summary"],
            "levels_refreshed": levels["updated"],
        }
    return {"enabled": True, **_guard_enrich(
        project, kind, page_index, False,
        lambda: enrich_snapshot.freeze_page(project, page_index, kind="question"),
    )}


def _guard_enrich(project: dict, kind: str, page_index: int, is_answer: bool, run):
    """跑一次补全/分级动作；任何异常都降级成返回值，绝不向上冒泡。

    冒泡的代价是**整页人工标注作废**（调用方 ``_save_page`` 在这之后才写盘）。
    这一层与 ``_save_page`` 的兜底是双保险：前者保证本函数自洽可单测，后者保证
    将来往调用点加代码时也不会漏。
    """
    try:
        return run() or {}
    except Exception as error:  # noqa: BLE001 - 降级优先于一切
        result = _degrade_enrich_failure(project, kind, page_index, error)
        if is_answer:
            result = {"ok": False, "updated": 0, **result}
        return result


ENRICH_ERROR_LIMIT = 300          # §2.4：enrich_error 最长 300 字


def _enrich_error_text(error: object) -> str:
    """异常文本（可传异常对象或字符串）规范化 + 截断。

    空消息退化成异常类名 / ``"unknown"``，避免 UI 上出现空白原因。
    """
    text = str(error or "").strip()
    if not text:
        text = error.__class__.__name__ if isinstance(error, BaseException) else "unknown"
    return text[:ENRICH_ERROR_LIMIT]


def _degrade_enrich_failure(project: dict, kind: str, page_index: int,
                            error: BaseException) -> dict:
    """冻结/回填失败后的降级：标记失败 + 尽力重算级别，**绝不阻断保存**。

    级别必须重算：``enrich_status='failed'`` 会命中 ``review_levels`` 的
    ``enrich_failed`` 规则 → 强制 L2。不重算的话，这条题会带着上一次的
    （很可能是 L0 的）级别继续往下走 —— 那正是"补全坏了却看不出来"。

    答案页保存失败时，题目侧才是真正需要升 L2 的一方（回填没发生），
    所以额外把引用这些答案的题目页也重算一次。
    """
    text = _enrich_error_text(error)
    pages_key = "answer_pages" if kind == "answer" else "pages"
    records_key = "answers" if kind == "answer" else "questions"
    pages = (project or {}).get(pages_key) or []
    page = pages[page_index] if isinstance(page_index, int) and 0 <= page_index < len(pages) else None
    touched = 0
    failed_ids: set[str] = set()
    for record in (page or {}).get(records_key) or []:
        if not isinstance(record, dict):
            continue
        record["enrich_status"] = "failed"
        record["enrich_error"] = text
        record_id = str(record.get("id") or "")
        if record_id:
            failed_ids.add(record_id)
        touched += 1

    question_pages: list[int] = []
    if kind == "answer" and failed_ids:
        for index, candidate in enumerate((project or {}).get("pages") or []):
            if not isinstance(candidate, dict):
                continue
            for record in candidate.get("questions") or []:
                if not isinstance(record, dict):
                    continue
                if failed_ids & set(enrich_snapshot.answer_ids_of(
                        project, str(record.get("id") or ""))):
                    question_pages.append(index)
                    break

    # 重算目标：本页（按 kind）+ 引用了失败答案的题目页（题目侧才是真正要升 L2 的一方）
    targets: list[tuple[int, str]] = []
    if isinstance(page_index, int) and 0 <= page_index < len(pages):
        targets.append((page_index, kind))
    targets.extend((index, "question") for index in question_pages)

    refreshed = 0
    for index, level_kind in targets:
        try:
            refreshed += int(
                enrich_snapshot.refresh_page_levels(
                    project, index, kind=level_kind,
                ).get("updated", 0)
            )
        except Exception:  # noqa: BLE001 - 级别重算失败同样不能阻断保存
            continue

    return {"enabled": True, "degraded": True, "error": text,
            "records": touched, "levels_refreshed": refreshed}


def _pair_metrics(project: dict, question: dict, answer_ids) -> dict:
    """按"人工选中的答案 vs 最强竞争者"重算 ``pair`` 字典（D1）。

    - ``score``/``margin`` 一律用候选原始分，**不读** ``display_confidence``；
    - ``ok`` 由 :func:`review_tools.review_levels.pair_ok` 算（三条件合取），
      阈值只有 T01 那一份；
    - ``display_confidence`` 只供匹配面板排序与可视化。
    """
    chosen = [str(value) for value in answer_ids or [] if str(value or "").strip()]
    if not chosen:
        return {"score": 0.0, "margin": 0.0, "path_match": False, "ok": False,
                "display_confidence": 0.0}
    chosen_set = set(chosen)
    ranked = annotation_data.candidates(
        project, question, limit=max(4, len(chosen) + 2), accepted_only=True,
    )
    chosen_scores = [
        float(score) for score, _page_index, _page, answer in ranked
        if str(answer.get("id", "")) in chosen_set
    ]
    other_scores = [
        float(score) for score, _page_index, _page, answer in ranked
        if str(answer.get("id", "")) not in chosen_set
    ]
    score = max(chosen_scores) if chosen_scores else 0.0
    second = max(other_scores) if other_scores else 0.0
    index = enrich_snapshot.record_index(project)
    path_match = True
    for answer_id in chosen:
        answer = index.get(answer_id)
        if not isinstance(answer, dict) or not annotation_data.number_path_matches(question, answer):
            path_match = False
            break
    pair = {
        "score": round(float(score), 2),
        "margin": round(float(score) - float(second), 2),
        "path_match": bool(path_match),
        "ok": False,
        "display_confidence": review_levels.pair_confidence(score),
    }
    pair["ok"] = bool(review_levels.pair_ok({"pair": pair}))
    return pair


def _write_pair_metrics(project: dict, question_id: str, answer_ids) -> dict:
    """把 ``pair`` 字典写到**题记录**上（分级读记录，不读关系表）。"""
    index = enrich_snapshot.record_index(project)
    question = index.get(str(question_id))
    if not isinstance(question, dict):
        return {}
    pair = _pair_metrics(project, question, answer_ids)
    question["pair"] = pair
    # D1：单值 float 的旧口径已废止，读到就清掉，避免两套阈值并存
    question.pop("pair_confidence", None)
    return pair


def _save_page(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    kind = str(payload.get("kind", "question"))
    pages_key, records_key = ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
    page_index = int(payload.get("page_index", -1))
    pages = project.get(pages_key, [])
    if page_index < 0 or page_index >= len(pages):
        raise ValueError("页面序号无效。")
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("标注框数据无效。")
    drawing_tool_active = payload.get("drawing_tool_active", False)
    if not isinstance(drawing_tool_active, bool):
        raise ValueError("绘制状态无效。")
    existing_records = {
        str(record.get("id", "")): record
        for record in pages[page_index].get(records_key, [])
        if isinstance(record, dict) and record.get("id")
    }
    page_scope_id = str(pages[page_index].get("match_scope_id", "") or "").strip()
    scope_regions = pages[page_index].get("match_scope_regions", [])
    confirmed_region_ids = payload.get("confirmed_match_scope_region_ids", None)
    confirm_all_answer_records = payload.get("confirm_all_answer_records", None)
    if confirm_all_answer_records is not None and not isinstance(confirm_all_answer_records, bool):
        raise ValueError("全选答案框状态无效。")
    if confirmed_region_ids is not None:
        if kind != "answer":
            raise ValueError("只有答案页可以按框选分组确认。")
        if not isinstance(confirmed_region_ids, list):
            raise ValueError("确认的分组区域无效。")
        valid_region_ids = {
            str(region.get("id", "")).strip()
            for region in scope_regions if isinstance(region, dict)
            if str(region.get("id", "")).strip()
        }
        normalized_confirmed_region_ids = []
        for value in confirmed_region_ids:
            region_id = str(value or "").strip()
            if not region_id or region_id not in valid_region_ids:
                raise ValueError("确认的分组区域不存在或已变更。")
            if region_id not in normalized_confirmed_region_ids:
                normalized_confirmed_region_ids.append(region_id)
        if (payload.get("accepted") and valid_region_ids and not normalized_confirmed_region_ids
                and not confirm_all_answer_records):
            raise ValueError("请至少勾选一个要纳入答案匹配的分组区域。")

    def inherited_scope(item: dict, box: list[int]) -> str:
        explicit = str(item.get("match_scope_id", "") or "").strip()
        if explicit:
            return explicit
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        for region in reversed(scope_regions if isinstance(scope_regions, list) else []):
            region_box = region.get("bbox", []) if isinstance(region, dict) else []
            if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
                continue
            try:
                rx0, ry0, rx1, ry1 = (float(value) for value in region_box)
            except (TypeError, ValueError):
                continue
            if rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1:
                return str(region.get("scope_id", "") or "").strip()
        return page_scope_id
    cleaned = []
    for record_index, item in enumerate(records, 1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {record_index} 个标注框数据无效，页面未保存。")
        if str(item.get("region_role", "question")) in {"group", "shared_material"}:
            continue
        box = item.get("bbox", [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"第 {record_index} 个标注框坐标无效，页面未保存。")
        try:
            x0, y0, x1, y1 = (int(round(float(value))) for value in box)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"第 {record_index} 个标注框坐标无效，页面未保存。") from error
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"第 {record_index} 个标注框范围无效，页面未保存。")
        minor_number = str(item.get("minor_number", item.get("number", "")))
        major_title = str(item.get("major_title", item.get("major", "")))
        shared_context_box = item.get("shared_context_bbox", [])
        if not isinstance(shared_context_box, list) or len(shared_context_box) != 4:
            shared_context_box = []
        shared_context_segments = item.get("shared_context_segments", [])
        record = annotation_data.new_record(
            [x0, y0, x1, y1], answer=kind == "answer",
            number=minor_number,
            minor_number=minor_number,
            major=major_title,
            major_title=major_title,
            major_number=str(item.get("major_number", "")),
            subsection_title=str(item.get("subsection_title", "")),
            subsection_number=str(item.get("subsection_number", "")),
            number_path=item.get("number_path", []),
            parent_id=str(item.get("parent_id", "")),
            region_role=str(item.get("region_role", "question")),
            continuation_of=str(item.get("continuation_of", "")),
            minor_number_inferred=bool(item.get("minor_number_inferred", False)),
            shared_context_bbox=[int(round(float(value))) for value in shared_context_box],
            shared_context_segments=shared_context_segments,
            use_shared_context=bool(item.get("use_shared_context", False)),
            context=str(item.get("context", "")),
            text=str(item.get("text", "")),
            question_type=str(item.get("question_type", "")),
            difficulty=str(item.get("difficulty", "")),
            knowledge_points=item.get("knowledge_points", []),
            options=item.get("options", []),
            analysis=str(item.get("analysis", "")),
            illustrations=item.get("illustrations", []),
            origin=str(item.get("origin", "manual")),
            answer_split_status=str(item.get("answer_split_status", "")),
            boundary_source=str(item.get("boundary_source", "")),
            detector_assisted=bool(item.get("detector_assisted", False)),
            detector_confidence=float(item.get("detector_confidence", 0.0) or 0.0),
            match_scope_id=inherited_scope(item, [x0, y0, x1, y1]),
        )
        record["answer"] = str(item.get("answer", ""))
        if item.get("id"):
            record["id"] = str(item["id"])
        # OCR and public delivery identity belong to the logical record, not
        # the editable form. Preserve them while bbox/number fields are saved.
        previous = existing_records.get(record.get("id", ""), {})
        for persistent_key in ("delivery_id",):
            if persistent_key in previous:
                record[persistent_key] = previous[persistent_key]
        # 补全字段（快照/分级/AI 结果/词表码…）同样属于逻辑记录：页面保存是
        # 重建记录的白名单路径，不在这里接住就会把整条 AI 结果抹掉。
        for enrich_key in annotation_data.ENRICH_PERSISTENT_KEYS:
            if enrich_key in previous:
                record[enrich_key] = _json_copy(previous[enrich_key])
        if isinstance(item.get("ocr"), dict):
            record["ocr"] = dict(item["ocr"])
        elif "ocr" in previous:
            record["ocr"] = previous["ocr"]
        cleaned.append(record)
    cleaned_ids = [str(record.get("id", "")) for record in cleaned]
    if any(not value for value in cleaned_ids) or len(cleaned_ids) != len(set(cleaned_ids)):
        raise ValueError("标注框内部 ID 为空或重复，页面未保存。")
    other_ids: set[str] = set()
    for other_pages_key, other_records_key in (
        ("pages", "questions"), ("answer_pages", "answers"),
        ("ignored_pages", "questions"), ("ignored_pages", "answers"),
    ):
        for other_page_index, other_page in enumerate(project.get(other_pages_key, [])):
            is_current_page = (
                other_pages_key == pages_key and other_page_index == page_index
            )
            if is_current_page:
                continue
            for other_record in other_page.get(other_records_key, []):
                if isinstance(other_record, dict) and other_record.get("id"):
                    other_ids.add(str(other_record["id"]))
    duplicate_ids = sorted(set(cleaned_ids).intersection(other_ids))
    if duplicate_ids:
        raise ValueError("标注框内部 ID 与教材其他页面重复，页面未保存。")
    previous_ids = set(existing_records)
    current_ids = {str(record.get("id", "")) for record in cleaned if record.get("id")}
    removed_ids = previous_ids - current_ids
    if payload.get("accepted") and kind == "question" and not cleaned:
        raise ValueError("题目页没有题框，不能确认；请先绘制题框或重新归类页面。")
    old_pairs = list(project.get("pairs", []))
    if removed_ids:
        if kind == "question":
            project["pairs"] = [
                pair for pair in old_pairs
                if str(pair.get("question_id", "")) not in removed_ids
            ]
        else:
            # A multi-part answer relation can remain valid after one answer
            # box is removed.  Remove only the missing IDs and drop the
            # relation if it no longer points to any answer.
            retained_pairs = []
            for pair in old_pairs:
                answer_ids = [
                    str(value) for value in pair.get("answer_ids", [])
                    if str(value) not in removed_ids
                ]
                if not answer_ids and pair.get("answer_ids"):
                    continue
                pair["answer_ids"] = answer_ids
                retained_pairs.append(pair)
            project["pairs"] = retained_pairs
    pages[page_index][records_key] = cleaned
    pages[page_index]["status"] = "accepted" if payload.get("accepted") else "draft"
    if confirmed_region_ids is not None:
        pages[page_index]["confirmed_match_scope_region_ids"] = normalized_confirmed_region_ids
        selected_region_ids = set(normalized_confirmed_region_ids)
        selected_records = []
        for record in cleaned:
            record_id = str(record.get("id", "")).strip()
            if not record_id:
                continue
            if confirm_all_answer_records:
                selected_records.append(record_id)
                continue
            box = record.get("bbox", [])
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            center_x = (float(box[0]) + float(box[2])) / 2
            center_y = (float(box[1]) + float(box[3])) / 2
            for region in scope_regions:
                if not isinstance(region, dict) or str(region.get("id", "")).strip() not in selected_region_ids:
                    continue
                region_box = region.get("bbox", [])
                if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
                    continue
                try:
                    x0, y0, x1, y1 = (float(value) for value in region_box)
                except (TypeError, ValueError, OverflowError):
                    continue
                if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                    selected_records.append(record_id)
                    break
        pages[page_index]["confirmed_match_record_ids"] = selected_records
    # 快照冻结/失效与单页分级必须紧跟记录落位（此时 project 里已是新记录），
    # 且在任何 save_project 之前完成 —— 一次保存只写一次盘。
    warnings = annotation_data.hierarchy_warnings(cleaned)
    # 补全是"增强"，绝不能阻断主流程：这里必须兜住所有异常。否则冻结一抛异常，
    # 下面这行 save_project 根本不会执行，审核员刚画的框与刚填的答案会整页丢失，
    # 而前端只看到"请求失败"。宁可这一页没有分级，也不能丢人工成果。
    try:
        enrich_result = _save_page_enrich(
            project, kind, page_index, accepted=bool(payload.get("accepted")),
        )
    except Exception as error:  # noqa: BLE001 - 降级优先于一切
        enrich_result = _degrade_enrich_failure(project, kind, page_index, error)
    if isinstance(enrich_result, dict) and enrich_result.get("degraded"):
        # 无论异常是被内层 `_guard_enrich` 吞掉还是被这层兜住，都必须让审核员看见
        warnings.append(
            "本页已保存，但补全与分级未完成（已降级，不影响标注结果）："
            + _enrich_error_text(enrich_result.get("error", ""))[:120]
        )
    annotation_data.save_project(project, project_path)
    if payload.get("accepted") and kind == "answer":
        missing_answer_count = sum(
            not str(record.get("analysis") or record.get("text") or "").strip()
            if str(record.get("region_role", "question")) == "continuation"
            else not str(record.get("answer", "")).strip()
            for record in cleaned
            if annotation_data.is_match_ready(pages[page_index], record)
        )
        if missing_answer_count:
            warnings.append(
                f"本页已确认，但有 {missing_answer_count} 个答案框尚无答案文字；请在后续标注中补充，并在交付前复核。"
            )
    response = {
        "ok": True,
        "saved": len(cleaned),
        "status": pages[page_index]["status"],
        "project_revision": project["project_revision"],
        "warnings": warnings,
        "confirmed_match_scope_region_ids": list(
            pages[page_index].get("confirmed_match_scope_region_ids", [])
        ),
        "confirmed_match_record_ids": list(pages[page_index].get("confirmed_match_record_ids", [])),
        "removed_pairs": len(old_pairs) - len(project.get("pairs", [])),
        # 分级与快照结果回给前端做页头徽标；PAPERFLOW_ENRICH=0 时只有 enabled=False。
        "enrich": enrich_result,
        # The draw button is renderer state only.  A save can contain it for
        # diagnostics, but the project always persists completed records only.
        "transient_drawing_tool_active": drawing_tool_active,
    }
    return response


def _match_candidates(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    question_id = str(payload.get("question_id", ""))
    question = next((record for _, _, record in annotation_data.all_questions(project)
                     if record.get("id") == question_id), None)
    if not question:
        raise ValueError("题目不存在。")
    paired_ids = annotation_data.paired_answer_ids(project, question_id)
    answer_count = sum(
        1 for _page_index, page, answer in annotation_data.all_answers(project)
        if annotation_data.is_match_ready(page, answer)
        and str(answer.get("region_role", "question")) == "question"
    )
    all_ranked = annotation_data.candidates(
        project, question, limit=max(100, answer_count), accepted_only=True,
    )
    ranked = [
        item for index, item in enumerate(all_ranked)
        if index < 100 or str(item[3].get("id", "")) in paired_ids
    ]
    candidates = []
    for index, (score, page_index, page, answer) in enumerate(ranked):
        second_score = ranked[index + 1][0] if index + 1 < len(ranked) else float("-inf")
        candidates.append({
            "score": score,
            "page_index": page_index,
            "source_path": annotation_data.source_path(project, page, True),
            "answer": answer,
            "match_reason": annotation_data.candidate_reason(question, answer),
            "recommended": bool(
                index == 0 and annotation_data.number_path_matches(question, answer)
                and score >= 180 and score - second_score >= 80
            ),
        })
    return {
        "ok": True,
        "question": question,
        "paired_answer_ids": paired_ids,
        "candidates": candidates,
    }


def _reclassify_pages(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    source_kind = str(payload.get("source_kind", "question"))
    target_kind = str(payload.get("target_kind", "answer"))
    page_indices = payload.get("page_indices", [])
    if not isinstance(page_indices, list):
        raise ValueError("页面序号格式无效。")
    result = annotation_data.reclassify_pages(
        project, source_kind, [int(value) for value in page_indices], target_kind,
    )
    annotation_data.save_project(project, project_path)
    return {"ok": True, **result, **_project_response(project_path, project)}


def _rotate_page(payload: dict) -> dict:
    """Rotate one project page clockwise without modifying its source image."""
    import cv2

    project_path, project = _load_project(payload)
    kind = "answer" if str(payload.get("kind", "question")) == "answer" else "question"
    pages_key, records_key = (
        ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
    )
    page_index = int(payload.get("page_index", -1))
    pages = project.get(pages_key, [])
    if page_index < 0 or page_index >= len(pages):
        raise ValueError("页面序号无效。")
    page = pages[page_index]
    original_path = Path(annotation_data.original_source_path(
        project, page, answer=(kind == "answer"),
    ))
    source = cv2.imdecode(np.fromfile(str(original_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("无法读取当前教材页。")

    old_rotation = int(page.get("rotation", 0) or 0) % 360
    if page.get(records_key):
        raise ValueError("当前页已有标注框，请先清空当前页再旋转。")
    if kind == "question":
        referenced = any(
            int(segment.get("page_index", -1)) == page_index
            for question_page in project.get("pages", [])
            for record in question_page.get("questions", [])
            for segment in record.get("shared_context_segments", [])
            if isinstance(segment, dict)
        )
        if referenced:
            raise ValueError("当前页已被其他题目作为共享材料引用，不能旋转。")
    new_rotation = (old_rotation + 90) % 360
    rotated = source
    for _ in range(new_rotation // 90):
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)

    book_root = Path(str(project.get("book_root") or project.get("image_root", ""))).resolve()
    token = hashlib.sha1(str(page.get("image", "")).encode("utf-8")).hexdigest()[:16]
    old_relative_cache = Path(str(page.get("rotated_image", "") or ""))
    relative_cache = Path("_paperflow_page_views") / f"{token}-r{new_rotation}.png"
    cache_path = (book_root / relative_cache).resolve()
    cache_path.relative_to(book_root)
    if new_rotation:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", rotated)
        if not ok:
            raise ValueError("旋转页面编码失败。")
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            encoded.tofile(str(temporary))
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        page["rotation"] = new_rotation
        page["rotated_image"] = relative_cache.as_posix()
    else:
        page.pop("rotation", None)
        page.pop("rotated_image", None)

    annotation_data.save_project(project, project_path)
    if old_relative_cache.parts and old_relative_cache != relative_cache:
        old_cache_path = (book_root / old_relative_cache).resolve()
        try:
            old_cache_path.relative_to((book_root / "_paperflow_page_views").resolve())
        except ValueError:
            pass
        else:
            old_cache_path.unlink(missing_ok=True)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True, "rotation": new_rotation, "delivery": None,
        "warnings": [],
        **_project_response(project_path, project),
    }


def _set_pair(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    question_id = str(payload.get("question_id", ""))
    answer_ids = [str(value) for value in payload.get("answer_ids", [])]
    accepted_question_ids = {
        str(question.get("id", ""))
        for _page_index, page, question in annotation_data.all_questions(project)
        if annotation_data.is_match_ready(page, question)
        and str(question.get("region_role", "question")) == "question"
    }
    if question_id not in accepted_question_ids:
        raise ValueError("只能为已经确认的题目页建立答案匹配。")
    accepted_answer_ids = {
        str(answer.get("id", ""))
        for _page_index, page, answer in annotation_data.all_answers(project)
        if annotation_data.is_match_ready(page, answer)
        and str(answer.get("region_role", "question")) == "question"
    }
    invalid_answer_ids = [value for value in answer_ids if value not in accepted_answer_ids]
    if invalid_answer_ids:
        raise ValueError("只能匹配已经确认的答案页，请先完成答案页审核。")
    status = "accepted" if answer_ids else "no_answer"
    annotation_data.set_pair(project, question_id, answer_ids, status)
    # D1：判定读 pair{score,margin,path_match,ok}；display_confidence 只展示。
    pair_metrics = _write_pair_metrics(project, question_id, answer_ids)
    if _enrich_enabled():
        # §3.4 答案后到：快照里本来就缺答案，配对成功后立刻补上并重算哈希。
        enrich_snapshot.refresh_answer_segment(project, answer_ids)
    annotation_data.save_project(project, project_path)
    sync_project(project_path, project, ROOT / "runtime")
    saved_pair = next(
        pair for pair in project.get("pairs", [])
        if str(pair.get("question_id", "")) == question_id
    )
    return {
        "ok": True, "status": status, "answer_ids": answer_ids,
        "pair": saved_pair, "pair_metrics": pair_metrics,
        "project_revision": project["project_revision"],
        "delivery": None,
    }


def _confident_match_proposals(project: dict) -> list[tuple[str, str, float]]:
    """唯一推荐自动配对的候选：题号层级完全一致、得分与分差足够、答案未被占用。

    供两处复用：`_accept_confident_matches`（人工触发批量确认）与
    `_auto_match_scopes`（自动匹配末尾对剩余题目直接应用唯一推荐）。
    """
    existing = {str(pair.get("question_id", "")): pair for pair in project.get("pairs", [])}
    used_answers = {
        str(answer_id)
        for pair in project.get("pairs", []) if pair.get("status") == "accepted"
        for answer_id in pair.get("answer_ids", [])
    }
    accepted_questions = [
        question for _page_index, page, question in annotation_data.all_questions(project)
        if str(page.get("status", "")) == "accepted"
        and str(question.get("region_role", "question")) == "question"
    ]
    proposals = []
    reserved_answers: set[str] = set()
    for question in accepted_questions:
        question_id = str(question.get("id", ""))
        if question_id in existing:
            continue
        ranked = annotation_data.candidates(project, question, limit=2, accepted_only=True)
        if not ranked:
            continue
        best_score, _page_index, _page, best_answer = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
        answer_id = str(best_answer.get("id", ""))
        # Exact hierarchy matches normally score well above 180. The margin
        # prevents repeated printed numbers in different units being accepted.
        if (not annotation_data.number_path_matches(question, best_answer)
                or best_score < 180 or best_score - second_score < 80
                or answer_id in used_answers or answer_id in reserved_answers):
            continue
        proposals.append((question_id, answer_id, float(best_score)))
        reserved_answers.add(answer_id)
    return proposals


def _accept_confident_matches(payload: dict) -> dict:
    """Conservatively accept only unambiguous, confirmed-answer suggestions."""
    project_path, project = _load_project(payload)
    proposals = _confident_match_proposals(project)
    saved_pairs: dict[str, dict] = {}
    saved_metrics: dict[str, dict] = {}
    for question_id, answer_id, _score in proposals:
        annotation_data.set_pair(project, question_id, [answer_id], "accepted")
        saved_metrics[question_id] = _write_pair_metrics(project, question_id, [answer_id])
        saved_pairs[question_id] = next(
            pair for pair in project.get("pairs", [])
            if str(pair.get("question_id", "")) == question_id
        )
    if proposals and _enrich_enabled():
        enrich_snapshot.refresh_answer_segment(
            project, [answer_id for _question_id, answer_id, _score in proposals],
        )
    if proposals:
        annotation_data.save_project(project, project_path)
        sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "accepted_count": len(proposals),
        "matches": [
            {"question_id": question_id, "answer_ids": [answer_id], "score": score,
             "pair": saved_pairs[question_id], "pair_metrics": saved_metrics[question_id]}
            for question_id, answer_id, score in proposals
        ],
        "project_revision": project.get("project_revision", 0),
        "delivery": None,
    }


def _upsert_match_scope(payload: dict) -> dict:
    """Create or rename a book-wide matching scope."""
    project_path, project = _load_project(payload)
    name = str(payload.get("name", "") or "").strip()
    if not name:
        raise ValueError("匹配分组名称不能为空。")
    if len(name) > 80:
        raise ValueError("匹配分组名称不能超过 80 个字符。")
    scope_id = str(payload.get("scope_id", "") or "").strip()
    scopes = project.setdefault("match_scopes", [])
    duplicate = next(
        (item for item in scopes if str(item.get("name", "")).casefold() == name.casefold()
         and str(item.get("id", "")) != scope_id),
        None,
    )
    if duplicate:
        raise ValueError("已经存在同名匹配分组。")
    scope = next((item for item in scopes if str(item.get("id", "")) == scope_id), None)
    if scope is None:
        scope = {"id": "scope-" + uuid.uuid4().hex, "name": name}
        scopes.append(scope)
    else:
        scope["name"] = name
    annotation_data.save_project(project, project_path)
    return {"ok": True, "scope": scope,
            "project": _project_response(project_path, project)["project"]}


def _assign_match_scope(payload: dict) -> dict:
    """Assign a shared scope to a rectangle, current page, or page selection."""
    project_path, project = _load_project(payload)
    kind = "answer" if str(payload.get("kind", "question")) == "answer" else "question"
    pages_key, records_key = ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
    pages = project.get(pages_key, [])
    scope_id = str(payload.get("scope_id", "") or "").strip()
    valid_scope_ids = {str(item.get("id", "")) for item in project.get("match_scopes", [])}
    if scope_id not in valid_scope_ids:
        raise ValueError("请选择有效的匹配分组。")
    raw_indices = payload.get("page_indices", [])
    if not isinstance(raw_indices, list):
        raise ValueError("页面范围无效。")
    try:
        page_indices = sorted(set(int(value) for value in raw_indices))
    except (TypeError, ValueError) as error:
        raise ValueError("页面范围无效。") from error
    if not page_indices or any(index < 0 or index >= len(pages) for index in page_indices):
        raise ValueError("请选择有效页面。")
    record_ids = {str(value) for value in payload.get("record_ids", []) if str(value)}
    set_page_default = bool(payload.get("set_page_default", False))
    region_box = payload.get("region_bbox")
    normalized_region = None
    if region_box is not None:
        if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
            raise ValueError("框选区域坐标无效。")
        try:
            x0, y0, x1, y1 = (int(round(float(value))) for value in region_box)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("框选区域坐标无效。") from error
        if x1 <= x0 or y1 <= y0:
            raise ValueError("框选区域范围无效。")
        if len(page_indices) != 1:
            raise ValueError("框选区域只能应用到当前一页。")
        normalized_region = [x0, y0, x1, y1]

    assigned_ids: list[str] = []
    for page_index in page_indices:
        page = pages[page_index]
        if set_page_default:
            page["match_scope_id"] = scope_id
        if normalized_region is not None:
            # A later rectangle selection replaces an earlier whole-page
            # assignment. Leaving that default in place would keep every
            # unselected answer frame in the group.
            previous_page_scope = str(page.get("match_scope_id", "") or "").strip()
            if previous_page_scope:
                page["match_scope_id"] = ""
                for record in page.get(records_key, []):
                    if str(record.get("match_scope_id", "") or "").strip() == previous_page_scope:
                        record["match_scope_id"] = ""
            page.setdefault("match_scope_regions", []).append({
                "id": "region-" + uuid.uuid4().hex,
                "scope_id": scope_id,
                "bbox": normalized_region,
            })
        for record in page.get(records_key, []):
            record_id = str(record.get("id", ""))
            selected = set_page_default or record_id in record_ids
            if normalized_region is not None:
                box = record.get("bbox", [])
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    center_x = (float(box[0]) + float(box[2])) / 2
                    center_y = (float(box[1]) + float(box[3])) / 2
                    selected = (normalized_region[0] <= center_x <= normalized_region[2]
                                and normalized_region[1] <= center_y <= normalized_region[3])
            if selected:
                record["match_scope_id"] = scope_id
                if record_id:
                    assigned_ids.append(record_id)
    annotation_data.save_project(project, project_path)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "assigned_count": len(set(assigned_ids)),
        "page_count": len(page_indices),
        "project": _project_response(project_path, project)["project"],
    }


def _remove_match_scope_pages(payload: dict) -> dict:
    """Remove selected pages from one scope without deleting pages or records."""
    project_path, project = _load_project(payload)
    scope_id = str(payload.get("scope_id", "") or "").strip()
    valid_scope_ids = {str(item.get("id", "")) for item in project.get("match_scopes", [])}
    if scope_id not in valid_scope_ids:
        raise ValueError("请选择有效的匹配分组。")
    selections = payload.get("pages", [])
    if not isinstance(selections, list) or not selections:
        raise ValueError("请选择要移出分组的页面。")
    normalized = set()
    for selection in selections:
        if not isinstance(selection, dict):
            raise ValueError("页面选择无效。")
        kind = "answer" if str(selection.get("kind", "question")) == "answer" else "question"
        try:
            page_index = int(selection.get("page_index", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("页面选择无效。") from error
        pages = project.get("answer_pages" if kind == "answer" else "pages", [])
        if page_index < 0 or page_index >= len(pages):
            raise ValueError("页面选择无效。")
        normalized.add((kind, page_index))

    removed_regions = 0
    cleared_records = 0
    for kind, page_index in sorted(normalized):
        pages_key, records_key = ("answer_pages", "answers") if kind == "answer" else ("pages", "questions")
        page = project[pages_key][page_index]
        if str(page.get("match_scope_id", "")) == scope_id:
            page["match_scope_id"] = ""
        regions = page.get("match_scope_regions", [])
        kept_regions = [
            region for region in regions
            if str(region.get("scope_id", "")) != scope_id
        ]
        removed_regions += len(regions) - len(kept_regions)
        page["match_scope_regions"] = kept_regions
        if "confirmed_match_scope_region_ids" in page:
            kept_region_ids = {str(region.get("id", "")) for region in kept_regions}
            page["confirmed_match_scope_region_ids"] = [
                str(region_id) for region_id in page.get("confirmed_match_scope_region_ids", [])
                if str(region_id) in kept_region_ids
            ]
        for record in page.get(records_key, []):
            if str(record.get("match_scope_id", "")) == scope_id:
                record["match_scope_id"] = ""
                cleared_records += 1
    annotation_data.save_project(project, project_path)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "removed_page_count": len(normalized),
        "removed_region_count": removed_regions,
        "cleared_record_count": cleared_records,
        "project": _project_response(project_path, project)["project"],
    }


def _auto_match_scopes(payload: dict) -> dict:
    """Match confirmed Q/A records sharing scope and canonical number.

    按用户规则放宽：同分组且完整题号相同即自动配对，不再要求两侧唯一。
    同号的多道题（常见于跨页续题被切成多个框）会共用同一批答案；随后对
    剩余未匹配的题应用「唯一推荐」自动配对（题号层级完全一致、分差足够），
    有误由人工后续修改。没有对应答案的题号保持待处理，交人工核对。
    """
    project_path, project = _load_project(payload)
    existing_questions = {
        str(pair.get("question_id", "")) for pair in project.get("pairs", [])
    }
    used_answers = {
        str(answer_id) for pair in project.get("pairs", []) if pair.get("status") == "accepted"
        for answer_id in pair.get("answer_ids", [])
    }
    question_map: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    answer_map: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for _index, page, record in annotation_data.all_questions(project):
        if not annotation_data.is_match_ready(page, record) or record.get("region_role", "question") != "question":
            continue
        scope_id = str(record.get("match_scope_id") or page.get("match_scope_id") or "").strip()
        path = annotation_data.canonical_number_path(record)
        if scope_id and path:
            question_map.setdefault((scope_id, path), []).append(record)
    for _index, page, record in annotation_data.all_answers(project):
        if not annotation_data.is_match_ready(page, record) or record.get("region_role", "question") != "question":
            continue
        scope_id = str(record.get("match_scope_id") or page.get("match_scope_id") or "").strip()
        path = annotation_data.canonical_number_path(record)
        if scope_id and path:
            answer_map.setdefault((scope_id, path), []).append(record)

    proposals: list[tuple[str, list[str], str]] = []
    conflicts = []
    all_keys = sorted(set(question_map).union(answer_map), key=lambda item: (item[0], item[1]))
    for scope_id, path in all_keys:
        questions = question_map.get((scope_id, path), [])
        answers = answer_map.get((scope_id, path), [])
        available_questions = [item for item in questions if str(item.get("id", "")) not in existing_questions]
        available_answers = [item for item in answers if str(item.get("id", "")) not in used_answers]
        if available_questions and available_answers:
            answer_ids = [str(item["id"]) for item in available_answers]
            for question in available_questions:
                proposals.append((str(question["id"]), answer_ids, scope_id))
        elif questions or answers:
            conflicts.append({
                "scope_id": scope_id,
                "number": "-".join(path),
                "question_count": len(questions),
                "answer_count": len(answers),
                "reason": "no_available_counterpart",
            })
    saved_pairs = []
    for question_id, answer_ids, scope_id in proposals:
        annotation_data.set_pair(
            project, question_id, answer_ids, "accepted", source="scope_number_auto"
        )
        # 同分组自动配对也必须落 pair 字典，否则分级会把它误判成 pair_fail（L2）。
        _write_pair_metrics(project, question_id, answer_ids)
        saved_pairs.append(next(
            pair for pair in project.get("pairs", [])
            if str(pair.get("question_id", "")) == question_id
        ))
    # 唯一推荐直接自动匹配（用户规则 2026-10-20）：同号配对后仍未匹配的题，
    # 若推荐答案唯一且与第二候选分差足够，先匹配上，有误由人工后续修改。
    confident_proposals = _confident_match_proposals(project)
    for question_id, answer_id, _score in confident_proposals:
        annotation_data.set_pair(
            project, question_id, [answer_id], "accepted", source="confident_auto"
        )
        _write_pair_metrics(project, question_id, [answer_id])
        saved_pairs.append(next(
            pair for pair in project.get("pairs", [])
            if str(pair.get("question_id", "")) == question_id
        ))
    touched_answer_ids = sorted({
        answer_id
        for _question_id, answer_ids, _scope_id in proposals
        for answer_id in answer_ids
    } | {answer_id for _question_id, answer_id, _score in confident_proposals})
    if touched_answer_ids and _enrich_enabled():
        enrich_snapshot.refresh_answer_segment(project, touched_answer_ids)
    if touched_answer_ids:
        annotation_data.save_project(project, project_path)
        sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "accepted_count": len(proposals),
        "confident_count": len(confident_proposals),
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:100],
        "pairs": saved_pairs,
        "project_revision": project.get("project_revision", 0),
        "delivery": None,
    }


def _set_delivery_output(payload: dict) -> dict:
    """Set a user-selected delivery root without eagerly materialising pages."""
    project_path, project = _load_project(payload)
    selected_root = Path(str(payload.get("output_root", ""))).expanduser()
    if not selected_root.is_absolute():
        raise ValueError("交付输出位置必须是本机文件夹。")
    if annotation_data.is_company_public_delivery_root(selected_root):
        raise ValueError("公司公共共享盘不再用于最终交付。请选本机文件夹，生成后使用“上传最终交付”。")
    selected_root = selected_root.resolve()
    if annotation_data.is_company_public_delivery_root(selected_root):
        raise ValueError("公司公共共享盘不再用于最终交付。请选本机文件夹，生成后使用“上传最终交付”。")
    if not selected_root.is_dir():
        raise ValueError("选择的交付输出文件夹不存在。")
    # A network share can be visible but still be read-only for the current
    # Windows user.  Verify write access before changing the project setting;
    # otherwise staff only discover the permission problem after a long export.
    probe = selected_root / f".paperflow-write-probe-{uuid.uuid4().hex}.tmp"
    try:
        probe.write_text("PaperFlow delivery output write check\n", encoding="utf-8")
    except OSError as error:
        raise ValueError(
            "选择的交付输出文件夹无法写入。请先连接公司数据服务器共享，"
            "并确认当前 Windows 账号具有写入权限。"
        ) from error
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            # A failed cleanup is harmless for delivery, and should not hide a
            # successful write-access result behind a secondary error.
            pass
    project["delivery_output_root"] = str(selected_root)
    annotation_data.save_project(project, project_path)
    output_path = delivery_root(project, ROOT / "runtime")
    return {
        "ok": True,
        "output_root": str(selected_root),
        "output_path": str(output_path),
        "project": _project_response(project_path, project)["project"],
        "delivery": {"book_id": project["delivery_book_id"], "output_path": str(output_path)},
    }


def _set_book_metadata(payload: dict) -> dict:
    """Persist fields shared by every question in one textbook."""
    project_path, project = _load_project(payload)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("教材信息格式无效。")
    allowed = {
        "language": {"zh", "en"},
        "stage": {"小学", "初中", "高中"},
        "answer_mode": annotation_data.ANSWER_MODES,
    }
    # D6：书级字段是全书共用的一个值，改错一个污染整本 350 题。锁定后要改必须
    # 显式带 unlock_reason —— 不是为了防错，是为了让"整本重跑"的代价可见。
    current_meta = {
        key: str(project.get(key, "") or "").strip()
        for key in ("language", "stage", "grade", "subject", "difficulty_scheme")
    }
    next_meta = dict(current_meta)
    for key in ("language", "stage", "grade", "subject"):
        next_meta[key] = str(metadata.get(key, "") or "").strip()
    next_meta["difficulty_scheme"] = str(
        metadata.get("difficulty_scheme", "") or current_meta.get("difficulty_scheme") or "1-5"
    ).strip()
    unlock_reason = str(payload.get("unlock_reason", "") or "").strip()
    if project.get("book_meta_locked") and not unlock_reason \
            and book_meta.book_meta_rev(next_meta) != book_meta.book_meta_rev(current_meta):
        raise ValueError(
            "教材信息已锁定，改动需要填写 unlock_reason（将导致全书重新补全，"
            "请说明为什么必须改）。"
        )
    for key in ("language", "stage", "grade", "subject"):
        value = str(metadata.get(key, "") or "").strip()
        if key in allowed and value not in allowed[key]:
            raise ValueError(f"教材{key}填写无效。")
        if not value:
            raise ValueError("语言、学段、年级、学科均不能为空。")
        project[key] = value
    answer_mode = str(metadata.get("answer_mode", project.get("answer_mode", "textbook")) or "").strip()
    if answer_mode not in annotation_data.ANSWER_MODES:
        raise ValueError("教材答案模式填写无效。")
    project["answer_mode"] = answer_mode
    requested_scheme = str(metadata.get("difficulty_scheme", "") or "").strip()
    if requested_scheme:
        if requested_scheme not in book_meta.DIFFICULTY_SCHEMES:
            raise ValueError("难度口径填写无效。")
        project["difficulty_scheme"] = requested_scheme
    else:
        project.setdefault("difficulty_scheme", "1-5")
    requested_profile = str(metadata.get("profile", "") or "").strip()
    if requested_profile:
        # 版式是整本教材共用的切题策略；空串表示本次不改，避免老客户端把
        # 已经落盘的版式覆盖成默认值。
        _apply_book_profile(
            project, requested_profile,
            Path(str(project.get("book_root") or project.get("image_root") or "")),
            respect_lock=False,
        )
    # 书级信息写入后立刻重算 book_meta_rev：快照失效判定（§3.5 第 5 条）与
    # 交付门禁都读它，晚一步就会出现"改成新学科、交付包还是旧学科"。
    project["book_meta_rev"] = book_meta.book_meta_rev(project)
    project["book_meta_locked"] = True
    if unlock_reason:
        project["book_meta_unlock_reason"] = unlock_reason
        project["book_meta_unlocked_at"] = datetime.now().isoformat(timespec="seconds")
    annotation_data.save_project(project, project_path)
    return {"ok": True, "project": _project_response(project_path, project)["project"],
            "book_meta_rev": project["book_meta_rev"],
            "book_meta_locked": True,
            "delivery": None}


def _save_project_cache(payload: dict) -> dict:
    """Atomically persist the renderer's complete in-memory textbook cache."""
    project_path, existing = _load_project(payload)
    incoming = payload.get("project")
    if not isinstance(incoming, dict):
        raise ValueError("教材缓存格式无效。")
    try:
        base_revision = int(payload.get("base_revision", incoming.get("project_revision", 0)))
        current_revision = int(existing.get("project_revision", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("教材缓存修订号无效。") from error
    if base_revision != current_revision:
        raise ValueError("教材缓存已在其他操作中更新，请重新打开教材后再保存。")
    expected_root = Path(str(existing.get("book_root") or existing.get("image_root") or "")).resolve()
    incoming_root = Path(str(incoming.get("book_root") or incoming.get("image_root") or "")).resolve()
    if incoming_root != expected_root:
        raise ValueError("教材缓存路径与当前教材不匹配。")
    for page in incoming.get("pages", []):
        page.pop("source_path", None)
    for page in incoming.get("answer_pages", []):
        page.pop("source_path", None)
    for page in incoming.get("ignored_pages", []):
        page.pop("source_path", None)
    annotation_data.merge_persistent_delivery_state(incoming, existing)
    annotation_data.save_project(incoming, project_path)
    return {"ok": True, "project_path": str(project_path),
            "project_revision": incoming["project_revision"],
            "question_pages": len(incoming.get("pages", [])),
            "answer_pages": len(incoming.get("answer_pages", []))}


def _delivery_location(payload: dict) -> dict:
    _project_path, project = _load_project(payload)
    return {"ok": True, "output_path": str(delivery_root(project, ROOT / "runtime"))}


def _workspace_page(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    kind = "answer" if str(payload.get("kind", "question")) == "answer" else "question"
    page_index = int(payload.get("page_index", 0))
    snapshot = page_snapshot(project_path, project, ROOT / "runtime", kind, page_index)
    return {"ok": True, **snapshot}


def _ensure_review_delivery_clear(project_path: Path) -> None:
    """Block local final delivery only after a pulled review has unresolved pages.

    No online file is needed here: the decision sidecar is written beside the
    local project by ``pull_review_decisions``.  Projects never submitted for
    review remain exportable, preserving the existing local-only workflow.
    """
    decision_path = project_path.resolve().with_name(DECISIONS_FILENAME)
    if not decision_path.is_file():
        return
    try:
        document = json.loads(decision_path.read_text(encoding="utf-8"))
        decisions = document.get("decisions", [])
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Local review decision file is invalid; pull decisions again before delivery.") from error
    unresolved = [item for item in decisions if item.get("review_status") != "approved"]
    # Newer review documents describe the entire submission.  A submission is
    # eligible for delivery only when the reviewer has closed it and every
    # source page is approved.  This deliberately blocks pending pages too.
    if "submission_status" in document:
        expected = int(document.get("page_count", 0) or 0)
        complete = document.get("submission_status") == "reviewed" and expected > 0 and len(decisions) == expected
        if not complete or unresolved:
            statuses = ", ".join(sorted({str(item.get("review_status", "pending")) for item in unresolved})) or "pending"
            raise ValueError(
                f"Final crops and JSON are blocked: this review is not fully approved "
                f"(status={document.get('submission_status')}, unresolved={len(unresolved)}, states={statuses}). "
                "Complete the fixes, resubmit, and pull the latest approved conclusions first."
            )
    elif unresolved:
        # Backward compatibility for a sidecar pulled from an older review
        # server, which contained only completed page decisions.
        statuses = ", ".join(sorted({str(item.get("review_status", "")) for item in unresolved}))
        raise ValueError(f"Final crops and JSON are blocked: {len(unresolved)} reviewed page(s) still require action ({statuses}). Fix, resubmit and pull the approved conclusions first.")


def _export_project(payload: dict) -> dict:
    project_path, _project = _load_project(payload)
    _ensure_review_delivery_clear(project_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = ROOT / "runtime" / "manual_annotation_exports" / project_path.stem / stamp
    report = export_project(str(project_path), str(target))
    return {"ok": True, "output_path": str(target), "report": report}


def _sync_delivery(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    # 与 export 路径同一道闸：审核未全放行的教材不允许把最终裁图/JSON 落到交付工作区。
    _ensure_review_delivery_clear(project_path)
    result = sync_delivery_workspace(project_path, ROOT / "runtime", materialize_all=True)
    return {"ok": True, "project_revision": project.get("project_revision", 0), **result}


def _group_export_preflight(payload: dict) -> dict:
    _project_path, project = _load_project(payload)
    scope_id = str(payload.get("scope_id", "") or "").strip()
    if not scope_id:
        raise ValueError("请选择要生成的匹配分组。")
    return {"ok": True, **preflight_scope(project, scope_id)}


def _export_match_scope(payload: dict) -> dict:
    project_path, _project = _load_project(payload)
    _ensure_review_delivery_clear(project_path)
    scope_id = str(payload.get("scope_id", "") or "").strip()
    if not scope_id:
        raise ValueError("请选择要生成的匹配分组。")
    result = export_scope(project_path, ROOT / "runtime", scope_id)
    # The scope package remains useful for internal review, while the customer
    # contract is the K12 JSONL. Generate both from the same saved project so
    # the visible “生成裁剪图片和 JSON” action never leaves staff with only an
    # internal schema that the customer cannot import.
    formal = sync_delivery_workspace(project_path, ROOT / "runtime", materialize_all=True)
    result["delivery_output_path"] = str(Path(formal["output_path"]))
    result["k12_output_path"] = str(Path(formal["output_path"]) / "k12_dataset")
    result["k12_eligible_count"] = int(formal.get("k12", {}).get("eligible_count", 0))
    return {"ok": True, **result}


def _review_token_kwargs(payload: dict) -> dict:
    return {"token": payload["_review_token"]} if payload.get("_review_token") else {}


def _review_status(payload: dict) -> dict:
    project_path, project = _load_project(payload)
    base = str(payload["server_url"])
    token = payload["_review_token"]
    rows = review_sync._request_json(base + "/api/submissions", token=token, timeout=30)["submissions"]
    latest = next((row for row in rows if row["book_id"] == project.get("delivery_book_id")), None)
    if not latest:
        return {"ok": True, "status": "not_submitted", "can_deliver": False, "decisions": []}
    document = review_sync._request_json(base + f"/api/submissions/{latest['id']}/decisions", token=token, timeout=30)["document"]
    try:
        current = review_sync.build_review_snapshot(project_path, include_previews=False)
        matches = current["snapshot_fingerprint"] == document["snapshot_fingerprint"]
    except ValueError:
        matches = False
    result = {"ok": True, "status": document.get("submission_status", latest["status"]),
              "submission_id": latest["id"], "matches_local": matches,
              "decisions": document.get("decisions", []), "can_deliver": False}
    result["can_deliver"] = bool(matches and result["status"] == "reviewed" and result["decisions"]
        and all(item["review_status"] == "approved" for item in result["decisions"]))
    if matches:
        target = project_path.resolve().with_name(DECISIONS_FILENAME)
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    return result


def _authenticated_review_route(handler, roles=None):
    def run(payload):
        # Session credentials override all caller-supplied server/account fields.
        credentials = REVIEW_SESSION.credentials(roles)
        return handler({**payload, **credentials})
    return run


def _review_submit(payload: dict) -> dict:
    """Submit a review-only snapshot; the delivery exporter is never invoked.

    The page scans referenced by the project are uploaded alongside the snapshot
    so a reviewer can import the book into their own workstation and redraw
    frames without asking the annotator for a copy of the textbook folder.
    """
    project_path, _project = _load_project(payload)
    result = submit_review_snapshot(
        str(payload.get("server_url", "")),
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        project_path,
        int(payload.get("max_preview_side", 1600)),
        **_review_token_kwargs(payload),
    )
    # Keep the remote CAS base separate from local autosave revisions.
    _project["online_review_base_revision"] = int(result["project"]["project_revision"])
    annotation_data.save_project(_project, project_path)
    result["project_revision"] = _project["project_revision"]
    source: dict = {"total": 0, "uploaded": 0, "skipped": 0, "bytes": 0}
    notice = "已按审核中心账号同步个人工程并提交审核快照；最终裁图和最终 JSON 由服务器在逐页审核通过后统一生成，无需在本机导出版。"
    if payload.get("upload_source_images", True):
        try:
            source = upload_project_source_images(
                str(payload.get("server_url", "")),
                str(payload.get("username", "")),
                str(payload.get("password", "")),
                project_path,
                **_review_token_kwargs(payload),
            )
            if source.get("total"):
                notice += f"原书图片已同步 {source['uploaded']} 张（{source['skipped']} 张此前已上传），审核员导入时不必再找你要教材目录。"
        except Exception as error:  # a failed image sync must not hide a good submission
            source = {"error": str(error)}
            notice += f"注意：原书图片同步未完成（{error}）。审核员导入时需要你提供教材目录。"
    return {"ok": True, **result, "source_images": source, "notice": notice}


def _review_upload_source_images(payload: dict) -> dict:
    """Upload only the page scans, e.g. after an earlier submit skipped them."""
    project_path, _project = _load_project(payload)
    result = upload_project_source_images(
        str(payload.get("server_url", "")),
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        project_path,
        force=bool(payload.get("force", False)),
        **_review_token_kwargs(payload),
    )
    return {
        "ok": True,
        **result,
        "notice": f"原书图片同步完成：共 {result['total']} 张，本次上传 {result['uploaded']} 张，"
                  f"此前已存在 {result['skipped']} 张。",
    }


def _review_pull_decisions(payload: dict) -> dict:
    """Save returned page decisions as a local sidecar without changing delivery files."""
    project_path, _project = _load_project(payload)
    submission_id = str(payload.get("submission_id", "")).strip()
    if not submission_id:
        raise ValueError("缺少线上审核提交编号。")
    result = pull_review_decisions(
        str(payload.get("server_url", "")),
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        submission_id,
        project_path,
        **_review_token_kwargs(payload),
    )
    return {
        "ok": True,
        **result,
        "notice": "审核结论已保存到教材本地目录，未生成、修改或上传任何最终裁图和最终 JSON。",
    }


def _review_list_projects(payload: dict) -> dict:
    """List textbooks visible to this review-center account (import dialog)."""
    projects = list_remote_projects(
        str(payload.get("server_url", "")),
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        **_review_token_kwargs(payload),
    )
    return {"ok": True, "projects": projects}


def _review_import_project(payload: dict) -> dict:
    """Download a server snapshot into a local textbook dir and open it.

    Reviewer-side counterpart of submit: the snapshot carries every frame the
    annotator made, but no local paths.  We re-anchor book/image roots to the
    chosen local textbook directory so the exact same open-book flow works.
    """
    raw_root = str(payload.get("book_root", "")).strip()
    if not raw_root:
        raise ValueError("请先选择本地教材目录（存放扫描页图的文件夹，可以选一个空文件夹）。")
    book_root = Path(raw_root).expanduser().resolve()
    if book_root.is_file():
        raise ValueError("选择的路径是文件，请选择一个文件夹。")
    book_root.mkdir(parents=True, exist_ok=True)
    book_id = str(payload.get("book_id", "")).strip()
    if not book_id or "/" in book_id or "\\" in book_id:
        raise ValueError("教材编号无效。")
    # Pull the page scans first: the snapshot re-anchors image paths to this
    # folder, so the files must exist before the project is written.
    source = fetch_project_source_images(
        str(payload.get("server_url", "")),
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        book_id,
        book_root,
        **_review_token_kwargs(payload),
    )
    snapshot = pull_project_snapshot(
        str(payload.get("server_url", "")),
        str(payload.get("username", "")),
        str(payload.get("password", "")),
        book_id,
        **_review_token_kwargs(payload),
    )
    project = snapshot["project"]
    for root_key in ("book_root", "image_root", "answer_image_root"):
        project[root_key] = str(book_root)
    for pages_key in ("pages", "answer_pages", "ignored_pages"):
        for page in project.get(pages_key) or []:
            if isinstance(page, dict):
                page.pop("source_path", None)
    project["delivery_book_id"] = book_id
    project["online_review_base_revision"] = int(snapshot.get("project_revision", project.get("project_revision", 0)))
    project_path = annotation_data.default_book_project_path(str(book_root), ROOT / "runtime")
    backup_path = None
    if project_path.is_file():
        backup_path = project_path.with_name(project_path.stem + ".导入前备份.json")
        shutil.copy2(project_path, backup_path)
    temporary = project_path.with_name(project_path.name + ".importing.tmp")
    temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, project_path)
    return {
        "ok": True,
        "book_id": book_id,
        "book_root": str(book_root),
        "project_path": str(project_path),
        "project_revision": snapshot.get("project_revision", project.get("project_revision", 0)),
        "backup_path": str(backup_path) if backup_path else "",
        "book_name": snapshot.get("book_name", ""),
        "source_images": source,
        "notice": (
            f"服务器工程已导入本地；原书图片已就位 {source.get('downloaded', 0)} 张"
            f"（{source.get('skipped', 0)} 张本机已有），可以在工作台上直接查看和修改。"
            if source.get("total")
            else "服务器工程已导入本地，但服务器上还没有这本教材的原书图片——"
                 "请让标注员重新点一次“提交审核”（会自动上传原书图片），或手工把教材目录放进这个文件夹。"
        ),
    }


def _review_push_project(payload: dict) -> dict:
    """Push the locally edited project back as the next server revision.

    The review snapshot travels with the push so the server can create the
    replacement review task (and freeze its project) in the same transaction.
    The old standalone refresh call is kept only as a fallback for older
    servers; it is idempotent there, so a retry cannot duplicate the task.
    """
    project_path, project = _load_project(payload)
    review_snapshot = None
    if payload.get("_review_token"):
        try:
            review_snapshot = review_sync.build_review_snapshot(project_path)
        except Exception:
            review_snapshot = None
    try:
        result = push_project_snapshot(
            str(payload.get("server_url", "")),
            str(payload.get("username", "")),
            str(payload.get("password", "")),
            project_path,
            review_snapshot=review_snapshot,
            **_review_token_kwargs(payload),
        )
    except ReviewConflictError as error:
        raise ValueError(f"版本冲突，未回传：{error}") from error
    # Reconcile the local revision with the server's stored revision so the
    # next push compares against the revision this push created, not a stale one.
    remote_revision = int(result["project_revision"])
    project["online_review_base_revision"] = remote_revision
    annotation_data.save_project(project, project_path)
    notice = f"修改已回传服务器（第 {remote_revision} 版），旧版审核已作废。"
    if payload.get("_review_token"):
        if result.get("submission"):
            notice += "网站审核预览已更新，请逐页放行新版本。"
        else:
            try:
                snapshot = dict(review_snapshot or review_sync.build_review_snapshot(project_path))
                snapshot["project_revision"] = remote_revision
                frozen = dict(project)
                frozen["project_revision"] = remote_revision
                snapshot["project_document"] = frozen
                review_sync._request_json(str(payload["server_url"]) + "/api/review-refresh", "POST", snapshot,
                                          payload["_review_token"], timeout=300)
                notice += "网站审核预览已更新，请逐页放行新版本。"
            except Exception as error:
                notice += f"审核预览更新失败，请重试「回传修改」：{error}"
    return {
        "ok": True,
        **result,
        "server_revision": remote_revision,
        "project_revision": project["project_revision"],
        "notice": notice,
    }


def _normalize_ocr_text(text: str) -> str:
    """对本地 OCR 输出的题面文字做数学符号归一化。

    保留 heavy PaddleOCR 已经正确识别的符号（±≤≥√π∞×÷），
    修复常见的 `|` 与 `1` 混淆、全角/半角不一致、多余空格。
    """
    if not text:
        return text

    # 1. 全角字母数字 → 半角
    result = text.translate(str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz",
    ))

    # 2. 全角括号/逗号/句号/等号/分号 → 半角（保留中文句号）
    result = result.replace("（", "(").replace("）", ")")
    result = result.replace("，", ",").replace("；", ";").replace("、", ",")
    result = result.replace("＝", "=").replace("＋", "+").replace("－", "-")
    result = result.replace("＜", "<").replace("＞", ">")

    # 3. 修复绝对值竖线被误读为 1 的常见模式
    #    模式：`|a1` → `|a|` 或 `|a1` → `|a|`（结尾 1 应为 |）
    #    仅当 1 紧跟在数学变量/符号后且后面是逗号/空格/句号/括号/行尾时
    import re as _re
    result = _re.sub(r"\|([a-zA-Z])(\d+)", r"|\1|", result)  # |a12 → |a|
    result = _re.sub(r"(\d)([a-zA-Z])\|", r"\1\2|", result)  # 3a| → 3a| (no-op but safe keep)
    #    绝对值的结尾 1 误读：|a1 → |a|（OCR 常见 |a1 结尾竖线被读成 1）
    result = _re.sub(r"\|([a-zA-Z0-9+\-×÷^_]+?)(\d)(?=[,;.\s\)\]）\n]|$)", lambda m: f"|{m.group(1)}|" if m.group(2) == "1" and not m.group(1).endswith("1") else m.group(0), result)

    # 4. 清理中文与数字之间多余空格（中文后空格→去掉，数字前空格→去掉）
    result = _re.sub(r"([一-鿿])\s+(\d)", r"\1\2", result)
    result = _re.sub(r"(\d)\s+([一-鿿])", r"\1\2", result)

    # 5. 叠印分数 → 行内形式（如 ½ → 1/2）
    FRACTIONS = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
                 "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
                 "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
    for frac_char, frac_text in FRACTIONS.items():
        result = result.replace(frac_char, frac_text)

    # 6. 两端空白清理
    result = result.strip()
    result = _re.sub(r"\s+", " ", result)

    return result


def _ocr_text_changed(previous: str, current: str) -> bool:
    """整页 OCR 的新旧文本是否"实质不同"。

    先去掉全部空白再比较：``_normalize_ocr_text`` 会改中文与数字之间的空格，
    只按原文比较会把"归一化差异"误判成"人改了题面"，一次识别就把整页打 stale。
    去掉空白后完全相同 → 没变；相似度 ≥ ``OCR_TEXT_SIMILARITY_THRESHOLD`` → 也
    视为同一文本（OCR 逐字抖动同理）。只有真实变化才回传新文本。
    """
    old = re.sub(r"\s+", "", str(previous or ""))
    new = re.sub(r"\s+", "", str(current or ""))
    if old == new:
        return False
    if not old or not new:
        # 一边为空：从无到有 / 从有到无都算实质变化
        return True
    ratio = difflib.SequenceMatcher(None, old, new).ratio()
    return ratio < float(OCR_TEXT_SIMILARITY_THRESHOLD)


def _page_ocr(payload: dict) -> dict:
    """整页批量 OCR：用高精度 PaddleOCR 识别整页，按框返回题面文字。

    **只覆盖文本确实变化的记录**（架构 §3.6）：整页 OCR 一定会把该页每个框的
    文字重算一遍，若把结果无条件回传，一次识别就会让一页 20 题的快照全部失效、
    人工量翻倍。所以这里按"归一化后是否实质相同"过滤：
    没变的记录原样回传**旧文本**（前端应用后等于没改），并在 ``unchanged`` 里列出。
    """
    project_path, project = _load_project(payload)
    kind = str(payload.get("kind", "question") or "question")
    page_index = int(payload.get("page_index", -1))
    pages_key = "answer_pages" if kind == "answer" else "pages"
    pages = project.get(pages_key, [])
    if page_index < 0 or page_index >= len(pages):
        raise ValueError("页面序号无效。")
    page = pages[page_index]
    if str(page.get("status", "")) != "accepted":
        raise ValueError("只有已确认的页面才能执行整页识别。")
    image_path = annotation_data.source_path(project, page, kind == "answer")

    records_key = "answers" if kind == "answer" else "questions"
    records = [r for r in page.get(records_key, []) if isinstance(r, dict) and r.get("region_role", "question") == "question"]

    if not records:
        return {"ok": True, "results": {}, "unchanged": []}

    # 整页高精度 OCR
    ocr_items = ocr_image(image_path)
    if not ocr_items:
        # 识别不出东西时不能把已有文字清空：保住旧文本，只报"没识别到"
        empty_results = {}
        for record in records:
            rid = str(record.get("id", ""))
            empty_results[rid] = {
                "text": str(record.get("text", "") or ""),
                "confidence": 0.0,
                "changed": False,
            }
        return {"ok": True, "results": empty_results,
                "unchanged": [str(record.get("id", "")) for record in records]}

    results = {}
    unchanged: list[str] = []
    for record in records:
        box = record.get("bbox", [])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        bx0, by0, bx1, by1 = (int(v) for v in box)
        if bx1 <= bx0 or by1 <= by0:
            continue
        # 收集中心点落在框内的 OCR items，按 y 排序
        inside = []
        for item in ocr_items:
            raw_box = item.get("bbox", [])
            if not isinstance(raw_box, (list, tuple)):
                continue
            if len(raw_box) == 4 and all(isinstance(p, (int, float)) for p in raw_box):
                ix0, iy0, ix1, iy1 = (int(v) for v in raw_box)
            elif len(raw_box) == 4 and isinstance(raw_box[0], (list, tuple)):
                pts = [int(round(v)) for p in raw_box for v in (p if isinstance(p, (list, tuple)) else [p])]
                if len(pts) != 8:
                    continue
                ix0, iy0, ix1, iy1 = min(pts[0::2]), min(pts[1::2]), max(pts[0::2]), max(pts[1::2])
            else:
                continue
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            cx = (ix0 + ix1) / 2.0
            cy = (iy0 + iy1) / 2.0
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                inside.append((iy0, str(item.get("text", "") or "").strip(), float(item.get("conf", 0.0) or 0.0)))
        # 按 y 排序后连接
        inside.sort(key=lambda t: t[0])
        text_parts = [t[1] for t in inside if t[1]]
        text = "".join(text_parts)
        conf = sum(t[2] for t in inside) / max(len(inside), 1) if inside else 0.0
        text = _normalize_ocr_text(text)
        rid = str(record.get("id", ""))
        previous_text = str(record.get("text", "") or "")
        if previous_text and not _ocr_text_changed(previous_text, text):
            # 文字实质没变：回传旧值，前端应用后等于没改 → 不会触发快照失效
            unchanged.append(rid)
            results[rid] = {
                "text": previous_text,
                "confidence": round(conf, 4),
                "engine": "paddleocr_local",
                "changed": False,
            }
            continue
        results[rid] = {
            "text": text,
            "confidence": round(conf, 4),
            "engine": "paddleocr_local",
            "changed": True,
        }

    return {"ok": True, "results": results, "unchanged": unchanged}


def _enrich_prepare(payload: dict) -> dict:
    """下发题图候选 + 触发快照冻结（G2 → G4 的入口）。

    产物只写 ``record['illustration_candidates']`` 与 ``page['illustration_ready']``：
    候选永远不是正式题图（``illustrations`` 只能由人点"采用"产生）。
    同时在本次调用里把已确认页的题目快照冻结/失效一遍，让前端拿到的分级是新的。
    """
    if not _enrich_enabled():
        return _enrich_disabled_response()
    import cv2

    from review_tools import illustration_candidates

    project_path, project = _load_project(payload)
    # 开关关闭期间保存的页面没有快照（0→1 的活锁）：prepare 是补建它们的自然入口，
    # 否则那些记录会永远停在默认 L2，人工白看。
    rebuilt = enrich_snapshot.rebuild_missing(project)
    pages = project.get("pages") or []
    if not pages:
        return {"ok": True, "pages": 0, "candidates": 0, "frozen": 0, "stale": 0,
                "results": []}

    raw_indices = payload.get("page_indices")
    if isinstance(raw_indices, list) and raw_indices:
        try:
            page_indices = sorted({int(value) for value in raw_indices})
        except (TypeError, ValueError) as error:
            raise ValueError("页面序号格式无效。") from error
    elif payload.get("page_index") is not None:
        try:
            page_indices = [int(payload.get("page_index"))]
        except (TypeError, ValueError) as error:
            raise ValueError("页面序号格式无效。") from error
    else:
        page_indices = [
            index for index, page in enumerate(pages)
            if isinstance(page, dict) and str(page.get("status", "")) == "accepted"
        ]
    for index in page_indices:
        if index < 0 or index >= len(pages):
            raise ValueError("页面序号无效。")

    results = []
    total_candidates = 0
    total_frozen = 0
    total_stale = 0
    for index in page_indices:
        page = pages[index] if isinstance(pages[index], dict) else {}
        if not page:
            continue
        page_result = {"page_index": index, "candidates": 0, "skipped": ""}
        if str(page.get("status", "")) != "accepted":
            page_result["skipped"] = "page_not_accepted"
            results.append(page_result)
            continue
        records = [
            record for record in page.get("questions") or []
            if isinstance(record, dict)
            and str(record.get("region_role", "question") or "question") == "question"
        ]
        image = None
        try:
            image_path = annotation_data.source_path(project, page, False)
        except ValueError:
            image_path = ""
        if image_path:
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001 - 读不到图就当没候选，不拦住冻结
                image = None
        layout = None
        if image is not None:
            try:
                import layout_arbiter

                if layout_arbiter.is_enabled():
                    layout = layout_arbiter.detect(image, with_regions=True)
            except Exception:  # noqa: BLE001 - 版面裁判是可选依赖
                layout = None
        # layout 传空 dict 而不是 None：None 会让 collect() 再跑一次版面裁判。
        candidates = illustration_candidates.collect(
            image, index, records, layout=layout if isinstance(layout, dict) else {},
        )
        by_owner: dict[str, list] = {}
        for item in candidates:
            owner_id = str(item.get("owner_id") or "")
            if owner_id:
                by_owner.setdefault(owner_id, []).append(item)
        for record in records:
            record["illustration_candidates"] = by_owner.get(str(record.get("id") or ""), [])
        # "没算" ≠ "没图"：版面裁判不可用时 illustration_ready 保持 False，
        # 否则 no_image_candidate 规则会把整页题目误判 L2（§8.2）。
        page["illustration_ready"] = isinstance(layout, dict)
        if isinstance(layout, dict):
            page["layout_revision"] = str(layout.get("revision") or "")
        # 候选落位后必须在**同一事务**里重算本页级别：`illustration_ready` 由假变真
        # 会让 no_image_candidate / image_owner_* 的结论翻转，不重算就留着假风险。
        frozen = enrich_snapshot.freeze_page(
            project, index, kind="question",
            columns=(layout.get("columns") if isinstance(layout, dict) else None),
        )
        total_frozen += int(frozen.get("frozen", 0))
        total_stale += int(frozen.get("stale", 0))
        total_candidates += len(candidates)
        page_result.update({
            "candidates": len(candidates),
            "frozen": frozen.get("frozen", 0),
            "stale": frozen.get("stale", 0),
            "illustration_ready": page["illustration_ready"],
            "review_summary": frozen.get("review_summary", {}),
        })
        results.append(page_result)

    annotation_data.save_project(project, project_path, increment_revision=False)
    sync_project(project_path, project, ROOT / "runtime")
    return {
        "ok": True,
        "pages": len([item for item in results if not item.get("skipped")]),
        "candidates": total_candidates,
        "frozen": total_frozen,
        "stale": total_stale,
        "rebuilt": rebuilt,
        "results": results,
        "project_revision": project.get("project_revision", 0),
    }


def _snapshot_source_health(records: list[dict]) -> dict:
    """C1 可见性：源图指纹核不出来的记录数（只读诊断，不参与门禁与分级）。

    为什么单独数它：``snapshot['page_sha1_status']`` 是 §3.2 的非权威诊断位 ——
    它区分"图没变"与"读不到图"，但**不进哈希**，因此没有任何门禁会因为它报警。
    T04/T05 暂缓期间出包侧没有这道门，`unavailable` 会安静地变成"看起来一切正常"：
    换机/拔盘/教材目录改名之后，整本快照的指纹都核不了，而界面毫无提示。

    这里只做计数，不改 ``is_valid`` 的语义（那会让离机审查整本失效，
    见 ``freeze_or_invalidate`` 里"读不到源图 ≠ 换了源图"的旧指纹保留分支）。
    """
    total = ok = unavailable = no_snapshot = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        total += 1
        snapshot = record.get("snapshot")
        if not isinstance(snapshot, dict) or not snapshot:
            no_snapshot += 1
            continue
        if str(snapshot.get("page_sha1_status") or "") == "ok":
            ok += 1
        else:
            unavailable += 1
    return {
        "total": total,
        "page_sha1_ok": ok,
        "page_sha1_unavailable": unavailable,
        "no_snapshot": no_snapshot,
    }


def _enrich_status(payload: dict) -> dict:
    """分级 / 待办统计（只读，不落盘）：给 UI 的"本页风险 N 条"与批量面板用。"""
    if not _enrich_enabled():
        return _enrich_disabled_response()
    _project_path, project = _load_project(payload)
    enrichment = project.get("enrichment")
    enrichment = enrichment if isinstance(enrichment, dict) else {}

    records: list[dict] = []
    page_summaries: list[dict] = []
    positions: dict[str, int] = {}
    for page_index, page in enumerate(project.get("pages") or []):
        if not isinstance(page, dict):
            continue
        page_records = [
            record for record in page.get("questions") or []
            if isinstance(record, dict)
            and str(record.get("region_role", "question") or "question") == "question"
        ]
        for record in page_records:
            record_id = str(record.get("id") or "")
            if record_id:
                positions[record_id] = page_index
            records.append(record)
        page_summaries.append({
            "page_index": page_index,
            "image": str(page.get("image", "")),
            "status": str(page.get("status", "")),
            "illustration_ready": bool(page.get("illustration_ready", False)),
            "record_count": len(page_records),
            "review_summary": (
                page.get("review_summary")
                if isinstance(page.get("review_summary"), dict)
                else enrich_snapshot.page_review_summary(page)
            ),
        })

    status_counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("enrich_status") or "none")
        status_counts[key] = status_counts.get(key, 0) + 1

    # "待办"与"不可计算/不可冻结"互斥（QA Round 8 缺口 1）：
    # 没有可用快照的记录既算不出可信级别、也没法交付补全，单独归 blocked，
    # 不能同时挂在 L2 风险与待办里 —— 否则同一道题会被数两次、人工以为有两条活。
    pending_status = ("pending", "stale", "kp_stale", "analysis_stale", "failed", "rejected")
    todo = []
    blocked = []
    for record in records:
        record_id = str(record.get("id") or "")
        level = str(record.get("review_level") or review_levels.DEFAULT_LEVEL)
        status = str(record.get("enrich_status") or "none")
        has_snapshot = enrich_snapshot.is_valid_snapshot(record.get("snapshot"))
        item = {
            "question_id": record_id,
            "page_index": positions.get(record_id, -1),
            "display_number": str(record.get("display_number", "")),
            "review_level": level,
            "review_reasons": list(record.get("review_reasons") or []),
            "enrich_status": status,
            "snapshot_id": str(record.get("snapshot_id") or ""),
        }
        if not has_snapshot:
            blocked.append({**item, "blocked": "no_snapshot"})
            continue
        if level == "L0" and status not in pending_status:
            continue
        todo.append(item)

    level_stats = review_levels.batch_stats(records)
    return {
        "ok": True,
        "enabled": True,
        "bridge_revision": PAPERFLOW_BRIDGE_REVISION,
        "rule_version": review_levels.RULE_VERSION,
        "schema_version": int(enrichment.get("schema_version", 1) or 1),
        "taxonomy_version": str(enrichment.get("taxonomy_version") or ""),
        "prompt_version": str(enrichment.get("prompt_version") or ""),
        "model": str(enrichment.get("model") or ""),
        "batch_id": str(enrichment.get("batch_id") or ""),
        "degraded": enrichment.get("degraded") if isinstance(enrichment.get("degraded"), dict) else {},
        "book_meta_rev": str(project.get("book_meta_rev") or ""),
        "book_meta_locked": bool(project.get("book_meta_locked", False)),
        "difficulty_scheme": str(project.get("difficulty_scheme") or "1-5"),
        "level_stats": level_stats,
        "review_summary": {
            "L0": level_stats.get("L0", 0),
            "L1": level_stats.get("L1", 0),
            "L2": level_stats.get("L2", 0),
        },
        "enrich_summary": status_counts,
        # C1：源图指纹核不出来的记录数（只读诊断）。见 _snapshot_source_health。
        "snapshot_health": _snapshot_source_health(records),
        "todo_count": len(todo),
        "todo": todo[:200],
        "blocked_count": len(blocked),
        "blocked": blocked[:200],
        "pages": page_summaries,
    }



def _book_meta_suggest(payload: dict) -> dict:
    """G0b：书名解析候选。**只出候选，绝不写库**（写库必须走 set-book-metadata）。"""
    if not _enrich_enabled():
        return _enrich_disabled_response()
    _project_path, project = _load_project(payload)
    title = str(payload.get("title", "") or "").strip()
    if not title:
        root_value = str(project.get("book_root") or project.get("image_root") or "")
        title = Path(root_value).name if root_value else ""
    scheme = str(project.get("difficulty_scheme") or "1-5")
    candidates = book_meta.guess_book_metadata(title, difficulty_scheme=scheme)
    current = {
        key: str(project.get(key, "") or "")
        for key in ("language", "stage", "grade", "subject", "difficulty_scheme")
    }
    proposed = dict(current)
    for key in ("language", "stage", "grade", "subject"):
        value = str(candidates.get(key) or "").strip()
        if value:
            proposed[key] = value
    proposed["difficulty_scheme"] = scheme
    validation = book_meta.validate_book_metadata_revision(
        proposed,
        str(project.get("book_meta_rev") or ""),
        locked=bool(project.get("book_meta_locked", False)),
    )
    return {
        "ok": True,
        "title": title,
        "candidates": candidates,
        "current": current,
        "proposed": proposed,
        "book_meta_rev": str(project.get("book_meta_rev") or ""),
        "book_meta_locked": bool(project.get("book_meta_locked", False)),
        "validation": validation,
        "note": "候选仅供参考，必须人工确认后由 set-book-metadata 落库；书级字段一错污染整本。",
    }


ROUTES = {
    "/api/prelabel": _prelabel,
    "/api/project/open-book": _open_book,
    "/api/project/save-page": _save_page,
    "/api/project/match-candidates": _match_candidates,
    "/api/project/reclassify-pages": _reclassify_pages,
    "/api/project/rotate-page": _rotate_page,
    "/api/project/set-pair": _set_pair,
    "/api/project/accept-confident-matches": _accept_confident_matches,
    "/api/project/upsert-match-scope": _upsert_match_scope,
    "/api/project/assign-match-scope": _assign_match_scope,
    "/api/project/remove-match-scope-pages": _remove_match_scope_pages,
    "/api/project/auto-match-scopes": _auto_match_scopes,
    "/api/project/set-delivery-output": _set_delivery_output,
    "/api/project/set-book-metadata": _set_book_metadata,
    "/api/project/set-ocr-settings": _set_ocr_settings,
    "/api/project/save-cache": _save_project_cache,
    "/api/project/delivery-location": _delivery_location,
    "/api/project/workspace-page": _workspace_page,
    "/api/project/export": _export_project,
    "/api/project/sync-delivery": _sync_delivery,
    "/api/project/group-export-preflight": _group_export_preflight,
    "/api/project/export-match-scope": _export_match_scope,
    "/api/review/submit": _review_submit,
    "/api/review/pull-decisions": _review_pull_decisions,
    "/api/review/list-projects": _review_list_projects,
    "/api/review/import-project": _review_import_project,
    "/api/review/push-project": _review_push_project,
    "/api/review/upload-source-images": _review_upload_source_images,
    "/api/project/enrich-prepare": _enrich_prepare,
    "/api/project/enrich-status": _enrich_status,
    "/api/project/book-meta-suggest": _book_meta_suggest,
    "/api/project/page-ocr": _page_ocr,
}

# Existing implementations remain callable by batch tools; the HTTP desk always
# uses a validated, process-local company session instead of per-action passwords.
for _path in tuple(ROUTES):
    if _path.startswith("/api/review/"):
        _roles = ({"annotator", "admin"} if _path.endswith("/submit") else
                  REVIEWERS if _path.endswith("/push-project") or _path.endswith("/upload-delivery")
                  or _path.endswith("/import-project") else None)
        ROUTES[_path] = _authenticated_review_route(ROUTES[_path], _roles)
ROUTES.update({
    "/api/library": _library_request,
    "/api/review/login": _review_login,
    "/api/review/session": lambda payload: REVIEW_SESSION.validated_status(),
    "/api/review/logout": lambda payload: REVIEW_SESSION.logout(),
    "/api/review/forget-on-exit": lambda payload: REVIEW_SESSION.forget_on_exit(),
    "/api/review/status": _authenticated_review_route(_review_status),
})
MUTATING_ROUTES.update({"/api/review/submit"})


class PaperFlowHandler(BaseHTTPRequestHandler):
    server_version = "PaperFlowLocal/1.0"

    def log_message(self, _format: str, *_args) -> None:
        # Electron owns the visible log.  Avoid writing a line for each browser
        # request into the console of a staff-facing application.
        return

    def _respond(self, status: HTTPStatus, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "null")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib request handler API
        self._respond(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802 - stdlib request handler API
        if urlparse(self.path).path == "/api/library/image":
            try:
                credentials = REVIEW_SESSION.credentials()
                query = parse_qs(urlparse(self.path).query)
                values = {key: query.get(key, [""])[0] for key in ("book_id", "kind", "page_index")}
                request = Request(credentials["server_url"] + "/api/library/image?" + urlencode(values),
                                  headers={"Authorization": "Bearer " + credentials["_review_token"]})
                with urlopen(request, timeout=60) as response:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "image/jpeg"))
                    if response.headers.get("Content-Length") is not None:
                        self.send_header("Content-Length", response.headers["Content-Length"])
                    self.send_header("Access-Control-Allow-Origin", "null")
                    self.end_headers()
                    shutil.copyfileobj(response, self.wfile, length=256 * 1024)
            except Exception as error:
                self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        if urlparse(self.path).path == "/api/health":
            self._respond(HTTPStatus.OK, {
                "ok": True,
                "service": "paperflow-local",
                "bridge_revision": PAPERFLOW_BRIDGE_REVISION,
                # 让排查和前端提示都能知道模型是否已经热好：
                # pending / loading / ready / failed / disabled。
                "prewarm": prewarm_state(),
            })
            return
        self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib request handler API
        route_path = urlparse(self.path).path
        route = ROUTES.get(route_path)
        if route is None:
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BODY_BYTES:
                raise ValueError("请求数据无效。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求数据格式错误。")
            if route_path not in {"/api/review/login", "/api/review/session", "/api/review/logout", "/api/review/forget-on-exit"}:
                REVIEW_SESSION.require_login()
            project_ref = str(payload.get("project_path") or "")
            if project_ref.startswith("server://"):
                result = _library_request({"action": "operate", "book_id": project_ref[9:],
                    "operation": route_path, "payload": payload,
                    "base_revision": payload.get("_server_revision", -1)})
                self._respond(HTTPStatus.OK, result)
                return
            if route_path == "/api/project/open-book" and str(payload.get("book_root", "")).startswith("server://"):
                result = _library_request({"action": "open", "book_id": payload["book_root"][9:],
                                           "annotator_id": payload.get("annotator_id")})
                self._respond(HTTPStatus.OK, result)
                return
            if route_path in MUTATING_ROUTES:
                with PROJECT_WRITE_LOCK:
                    result = route(payload)
            else:
                result = route(payload)
        except PermissionError as error:
            self._respond(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": str(error), "auth_required": True})
            return
        except ValueError as error:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        except Exception as error:  # Keep a local worker failure actionable in UI.
            traceback.print_exc()
            self._respond(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "ok": False,
                "error": f"本地预标注失败：{error}",
            })
            return
        # 成功响应必须放在 try 之外。以前它在 try 里，前端超时断开后写入
        # 会抛 ConnectionAbortedError，被 except 当成业务失败再写一次响应，
        # 日志里因此刷出两段自相矛盾的 traceback。
        self._respond(HTTPStatus.OK, result)


# ── 启动预热 ────────────────────────────────────────────────────────────
# 首次「自动框题」原本要在请求线程里加载 PP-OCRv5 轻量模型和中等模型，
# 冷启动实测 100~150 秒，远超标注员的耐心，日志里看起来就像程序卡死。
# 这里在服务启动后空闲时先把模型加载好，用户点击时只剩推理时间。
# 排查问题时可用环境变量 PAPERFLOW_DISABLE_PREWARM=1 关闭预热。
PREWARM_DELAY_SECONDS = 6.0
PREWARM_STATE_LOCK = threading.Lock()
PREWARM_STATE: dict[str, object] = {"status": "pending", "seconds": 0.0, "detail": ""}


def _set_prewarm_state(status: str, seconds: float = 0.0, detail: str = "") -> None:
    with PREWARM_STATE_LOCK:
        PREWARM_STATE.update({
            "status": status,
            "seconds": round(seconds, 1),
            "detail": detail,
        })


def prewarm_state() -> dict:
    """Return a copy so callers (health check, logs) never mutate shared state."""
    with PREWARM_STATE_LOCK:
        return dict(PREWARM_STATE)


def _prewarm_layout_arbiter() -> str:
    """预热版面裁判模型，返回一句可直接拼到日志里的说明。

    裁判只在"现有规则分出 ≥3 栏"时才介入（实测约 7% 的页），所以它的加载
    被放在 OCR 之后：绝大多数点击用不到它，但真用到时不该再等 5~15 秒。
    任何异常都只降级为"不预热"，绝不影响 OCR 预热的成功判定。
    """
    try:
        import layout_arbiter

        if not layout_arbiter.is_enabled():
            return "；版面裁判已按开关关闭"
        if not layout_arbiter.available():
            return "；版面裁判模型缺失，将按原规则运行"
        started = time.perf_counter()
        with PRELABEL_INFERENCE_LOCK:
            layout_arbiter._get_pipe()  # noqa: SLF001 预热，与 detect 复用同一单例
        elapsed = time.perf_counter() - started
        return f"；版面裁判已就绪（{elapsed:.1f} 秒）"
    except Exception as error:  # noqa: BLE001 - 预热失败必须静默降级
        print(f"[预热] 版面裁判未加载（不影响使用）：{error}", flush=True)
        return "；版面裁判未加载，将按原规则运行"


def _prewarm_ocr_models() -> None:
    """Load both local OCR predictors before an annotator asks for a page.

    Runs on a daemon thread and takes ``PRELABEL_INFERENCE_LOCK`` exactly like a
    real pre-label call: a click that arrives mid-warm-up waits for the same
    model load it would otherwise have triggered itself, instead of two threads
    driving one Paddle predictor.
    """
    started = time.perf_counter()
    _set_prewarm_state("loading")
    try:
        import cut_questions_v4 as cutter

        print("[预热] 后台加载本地 OCR 模型（此时仍可正常打开教材、翻页、画框）...", flush=True)
        with PRELABEL_INFERENCE_LOCK:
            fast_engine = cutter.get_fast_ocr_engine()
            print(
                f"[预热] 轻量结构扫描模型{'已就绪' if fast_engine is not None else '不可用，将回退高精度'}",
                flush=True,
            )
            cutter.get_ocr_engine()
            print("[预热] 高精度识别模型已就绪", flush=True)
            # 一次极小空白图推理，让 Paddle 完成首次算子编译与内存分配。
            # 个别版本在空白图上会报错，这不影响真实页面识别，忽略即可。
            canvas = np.full((240, 180, 3), 255, dtype=np.uint8)
            for label, attribute in (("轻量", "_run_fast_ocr"), ("高精度", "_run_ocr")):
                runner = getattr(cutter, attribute, None)
                if runner is None:
                    continue
                try:
                    runner(canvas)
                except Exception as error:  # noqa: BLE001 - 试跑失败必须降级
                    print(f"[预热] {label}模型试跑跳过：{error}", flush=True)
        elapsed = time.perf_counter() - started
        # 版面裁判单独预热：它只有 7% 的页面会真正用到，但一旦用到就要额外
        # 5~15 秒加载模型。放在 OCR 之后、用独立 try，失败不牵连 OCR 预热结论。
        arbiter_note = _prewarm_layout_arbiter()
        _set_prewarm_state("ready", elapsed, arbiter_note)
        print(
            f"[预热] 本地 OCR 已就绪，耗时 {elapsed:.1f} 秒；首次自动框题无需再等模型加载。"
            + (f"{arbiter_note}" if arbiter_note else ""),
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - 预热失败绝不能影响服务可用性
        elapsed = time.perf_counter() - started
        traceback.print_exc()
        _set_prewarm_state("failed", elapsed, str(error))
        print(
            f"[预热] 本地 OCR 预热失败（不影响使用，点击时仍会自行加载）：{error}",
            flush=True,
        )


def _start_prewarm_thread() -> None:
    disabled = os.environ.get("PAPERFLOW_DISABLE_PREWARM", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        _set_prewarm_state("disabled")
        print("[预热] 已按 PAPERFLOW_DISABLE_PREWARM 关闭预热。", flush=True)
        return

    def runner() -> None:
        MODEL_WARMUP_READY.wait()
        # 先让 Electron 的健康检查完成握手，再占用 CPU 加载模型，
        # 否则启动瞬间的模型加载可能让健康检查超时。
        time.sleep(PREWARM_DELAY_SECONDS)
        _prewarm_ocr_models()

    threading.Thread(target=runner, name="paperflow-ocr-prewarm", daemon=True).start()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8601), PaperFlowHandler)
    server.daemon_threads = True
    print("PaperFlow local annotation bridge listening on 127.0.0.1:8601", flush=True)
    _start_prewarm_thread()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
