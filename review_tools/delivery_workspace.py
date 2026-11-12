"""Incrementally maintain database-ready question/answer delivery files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from review_tools import book_meta
from review_tools import manual_annotation_data as data
from review_tools.export_manual_annotations import _crop, _question_crop


K12_QUESTION_TYPES = {
    "选择题": "选择", "填空题": "填空", "判断题": "判断",
    "计算题": "计算", "解答题": "计算", "应用题": "综合应用",
    "证明题": "证明", "实验题": "实验探究", "实验探究题": "实验探究",
    "阅读理解": "阅读理解", "材料分析": "材料分析", "综合应用": "综合应用",
}

# --- 交付词表（数据需求明细-供应商.xlsx / 需求1 第 3 行 N3 备注）-------------
# N3 逐字给出 language/stage/grade/subject/question_type/difficulty 的取值，
# 交付时必须逐项落表；映射后的值落在表外即拒收，不再原样透传。
# 书级字段复用 ``book_meta``，那是 UI 下拉与交付校验的唯一同源定义。
K12_LANGUAGES = book_meta.LANGUAGES                          # zh/en
K12_STAGES = book_meta.STAGES                                # 小学/初中/高中
K12_GRADES = tuple(book_meta.GRADE_NAMES.values())           # 一年级…高三
K12_SUBJECTS = book_meta.SUBJECTS
K12_QUESTION_TYPE_VALUES = (
    "选择", "填空", "判断", "计算", "证明", "实验探究",
    "阅读理解", "材料分析", "综合应用",
)
K12_DIFFICULTIES = ("基础", "中等", "困难")

# H3⑬：手机号/邮箱/身份证/社交账号统一匿名化为小写字母 x。
# 先长后短：邮箱必须先于手机号处理，否则邮箱里的数字段会被提前打散。
_PII_PATTERNS = (
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "x"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "x"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "x"),
    (re.compile(r"(?:QQ|qq|微信|WeChat|wechat)\s*[:：]?\s*\d{5,12}"), "x"),
)

# H3③：整页试卷截图不得作为题图。判据与 illustration_candidates 的
# ``page_like.page_area_ratio_min`` 保持一致（0.9），避免两套口径打架。
FULL_PAGE_AREA_RATIO = 0.9


# H3⑪：广告、导航、免责声明、HTML 字符、表情包等噪声一律不进交付文本。
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#\d{2,6});", re.IGNORECASE)
_NOISE_SENTENCE_RE = re.compile(
    r"(?:版权所有|侵权必究|免责声明|未经(?:许可|授权)[^。\n]{0,20}不得转载"
    r"|扫码(?:关注|下载)|关注(?:微信)?公众号)"
)
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s，。；、）)]+", re.IGNORECASE)
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
# H3⑥：这些字符出现在交付文本里就是 OCR/编码失败的硬信号。
GARBLED_MARKERS = ("\ufffd", "\u0000")


def _strip_noise(text: str) -> str:
    """Remove markup, ads, URLs and emoji, then normalise whitespace (H3⑥⑪)."""
    result = _HTML_TAG_RE.sub("", text)
    result = _HTML_ENTITY_RE.sub(" ", result)
    result = _NOISE_SENTENCE_RE.sub("", result)
    result = _URL_RE.sub("", result)
    result = _EMOJI_RE.sub("", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _mask_pii(text: object) -> str:
    """Replace identifiable personal data with a lowercase ``x`` (H3⑬)."""
    result = str(text or "")
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _k12_text(value: object) -> str:
    """Public pipeline for every delivered string: formulas → noise → PII."""
    return _mask_pii(_strip_noise(_unicode_math_text(value)))


def vocabulary_issues(meta: dict) -> list[dict]:
    """N3 词表闸（结构化）：值非空但不在规范枚举内即拒收。"""
    checks = (
        ("language", K12_LANGUAGES, "语言"),
        ("stage", K12_STAGES, "学段"),
        ("grade", K12_GRADES, "年级"),
        ("subject", K12_SUBJECTS, "学科"),
        ("question_type", K12_QUESTION_TYPE_VALUES, "题型"),
        ("difficulty", K12_DIFFICULTIES, "难度"),
    )
    issues = []
    for key, allowed, label in checks:
        value = str(meta.get(key, "")).strip()
        if value and value not in allowed:
            issues.append({"code": "invalid_vocabulary", "field": key,
                           "message": f"{label}不在规范词表内：{value}"})
    return issues


def _vocabulary_errors(meta: dict) -> list[str]:
    """生成闸用的字符串版本；空值由必填项检查单独报，这里只看填错。"""
    return [item["message"] for item in vocabulary_issues(meta)]


# --- 员工自检：OCR / LaTeX 乱码判据 ---------------------------------------
# 只认确定的语法破损，不搞命令白名单——OCR 常输出合法但少见的 TeX 命令，
# 拿白名单判会把正常公式误报成乱码，员工很快就会忽略提示。
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CJK_AFTER_BACKSLASH_RE = re.compile(r"\\[\u3400-\u9fff]")
_COMMAND_WITHOUT_ARG_RE = re.compile(r"\\(?:frac|sqrt|overline|text|mathrm)\b\s*(?=$|[}\]$])", re.I)
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]{2,})")
# OCR 最容易打歪、也最常出现在 K12 教材里的命令。只拿它做「编辑距离 1」的
# 邻近判断，不把它当白名单——合法但少见的 TeX 命令不会被误报成乱码。
# 具体集合在 _LATEX_SYMBOLS 定义之后构造，避免模块加载期的前向引用。


def _edit_distance_one(left: str, right: str) -> bool:
    """OCR 近似：替换/增/删一个字符，或相邻两字符换位。

    ``times → timse`` 这类换位在 OCR 里很常见，纯编辑距离是 2，漏掉就等于
    放过了最典型的一批拼错命令。
    """
    if len(left) == len(right):
        if sum(a != b for a, b in zip(left, right)) == 1:
            return True
        return any(left[:i] + left[i + 1] + left[i] + left[i + 2:] == right
                   for i in range(len(left) - 1))
    if abs(len(left) - len(right)) > 1:
        return False
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    return any(longer[:i] + longer[i + 1:] == shorter
               for i in range(len(longer)))


def _first_marker_offset(text: str, markers: tuple[str, ...]) -> int:
    """第一个乱码标记在原文里的下标；没有则 -1。"""
    positions = [text.find(marker) for marker in markers if marker in text]
    return min(positions) if positions else -1


def latex_syntax_details(value: object) -> list[dict]:
    """OCR 转写的 LaTeX 乱码/破损硬信号，并给出问题在原文里的位置。

    交给交付 JSON 的文本会经 ``_unicode_math_text`` 把 ``\\frac{1}{2}`` 转成
    ``1⁄2``、丢掉 ``$``。未知命令则**原样留下命令名**，所以破损公式必须在这里
    提前拦住，否则乱码会安静地进包。检查对象是原始 OCR 文本，不是清洗后的。

    返回 ``[{"message": ..., "offset": ...}]``；``offset`` 是原文中的字符下标，
    -1 表示整段性问题（如 ``$`` 未闭合）。``latex_syntax_errors`` 只是它的
    字符串版本——**判据只有这一份**，前端「乱码明细」的高亮位置也来自这里。
    """
    text = str(value or "")
    if not text or ("$" not in text and "\\" not in text):
        return []
    details: list[dict] = []

    def add(message: str, offset: int) -> None:
        details.append({"message": message, "offset": offset})

    marker_offset = _first_marker_offset(text, GARBLED_MARKERS)
    if marker_offset >= 0:
        add("含乱码字符", marker_offset)
    control = _CONTROL_CHAR_RE.search(text)
    if control:
        add("含控制字符", control.start())
    dollars = [match.start() for match in re.finditer(r"(?<!\\)\$", text)]
    if len(dollars) % 2:
        add("公式定界符 $ 未闭合", dollars[-1] if dollars else -1)
    depth = 0
    bad_brace = -1
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                bad_brace = index
                break
    if bad_brace >= 0 or depth:
        add("公式花括号不配对", bad_brace if bad_brace >= 0 else len(text) - 1)
    cjk = _CJK_AFTER_BACKSLASH_RE.search(text)
    if cjk:
        add("公式命令后混入中文字符", cjk.start())
    no_arg = _COMMAND_WITHOUT_ARG_RE.search(text)
    if no_arg:
        add("公式命令缺少参数", no_arg.start())
    for match in _LATEX_COMMAND_RE.finditer(text):
        command = match.group(1)
        if command in _KNOWN_LATEX_COMMANDS:
            continue
        if any(_edit_distance_one(command, known)
               for known in _KNOWN_LATEX_COMMANDS):
            add(f"疑似 OCR 拼错的公式命令 \\{command}", match.start())
            break
    return details


def latex_syntax_errors(value: object) -> list[str]:
    """生成闸用的字符串版本；判据与 ``latex_syntax_details`` 完全同源。"""
    return [item["message"] for item in latex_syntax_details(value)]


def _garbled_issue(field: str, message: str, raw_text: str,
                   offset: int, *, option_index: int | None = None) -> dict:
    """给自检面板附上一小段原文，方便员工直接看到「哪里乱码」。

    ``raw_excerpt`` 是原文切片，``excerpt_offset`` 是问题字符在切片内的下标；
    ``raw_offset`` / ``raw_length`` 是它在整段原文里的绝对位置。生成闸只读
    ``message``，这些附加字段不影响拒收结论。
    """
    item = {"code": "garbled", "field": field, "message": message}
    if option_index is not None:
        item["option_index"] = option_index
    if raw_text:
        start = max(0, (offset if offset >= 0 else 0) - 24)
        end = min(len(raw_text), start + 96)
        item["raw_excerpt"] = raw_text[start:end]
        item["raw_start"] = start
        item["raw_offset"] = offset
        item["excerpt_offset"] = offset - start if 0 <= offset < end else -1
        item["raw_length"] = len(raw_text)
    return item


def k12_field_issues(meta: dict, body: dict, *,
                     raw_texts: dict | None = None,
                     skip_fields: set[str] | None = None) -> list[dict]:
    """纯文本字段闸：自检与生成共用，唯一规则来源。

    不读图片、不碰身份库、不看审核状态，只判断「缺项 / 词表 / 乱码」。
    返回结构化 ``[{code, field, message}]``；``code`` 为
    ``missing_field`` / ``invalid_vocabulary`` / ``garbled``。

    ``raw_texts`` 是清洗前的原始 OCR 文本（question/answer/options），
    LaTeX 语法必须在原文上判——清洗会抹掉 ``$`` 和反斜杠。
    ``skip_fields`` 让自检按定位去重（书级字段只报一次、未匹配答案不判缺项）。
    """
    skip = set(skip_fields or ())
    issues: list[dict] = []
    required_meta = {
        "language": "语言", "stage": "学段", "grade": "年级", "subject": "学科",
        "question_type": "题型", "difficulty": "难度",
    }
    for key, label in required_meta.items():
        if key in skip:
            continue
        if not str(meta.get(key, "")).strip():
            issues.append({"code": "missing_field", "field": key,
                           "message": f"缺少{label}"})
    for issue in vocabulary_issues(meta):
        if issue["field"] not in skip:
            issues.append(issue)
    if "knowledge_points" not in skip and not meta.get("knowledge_points"):
        issues.append({"code": "missing_field", "field": "knowledge_points",
                       "message": "缺少知识点"})

    def add(field: str, label: str, value: object, raw: object) -> None:
        if field in skip:
            return
        text = str(value or "")
        raw_text = str(raw if raw is not None else text or "")
        if not text.strip():
            issues.append({"code": "missing_field", "field": field,
                           "message": f"缺少{label}"})
            return
        marker_offset = _first_marker_offset(text, GARBLED_MARKERS)
        reported_marker = marker_offset >= 0
        if reported_marker:
            issues.append(_garbled_issue(
                field, f"{label}含乱码字符", raw_text,
                _first_marker_offset(raw_text, GARBLED_MARKERS)))
        for detail in latex_syntax_details(raw_text):
            if detail["message"] == "含乱码字符":
                # 清洗后的文本可能把替换符抹掉，这时仍要按原文报一次；
                # 两边都报过时只保留一条，避免同一个字符重复计数。
                if reported_marker:
                    continue
                issues.append(_garbled_issue(field, f"{label}含乱码字符",
                                             raw_text, detail["offset"]))
                reported_marker = True
                continue
            issues.append(_garbled_issue(field, f"{label}{detail['message']}",
                                         raw_text, detail["offset"]))

    raw_texts = raw_texts or {}
    add("question", "题目文本", body.get("question"), raw_texts.get("question"))
    add("answer", "标准答案", body.get("answer"), raw_texts.get("answer"))
    raw_options = raw_texts.get("options") or []
    for index, option in enumerate(body.get("options", []) or []):
        raw_option = str(raw_options[index] if index < len(raw_options) else option or "")
        text = str(option or "")
        reported_marker = _first_marker_offset(text, GARBLED_MARKERS) >= 0
        if reported_marker:
            issues.append(_garbled_issue(
                "options", f"选项{index + 1}含乱码字符", raw_option,
                _first_marker_offset(raw_option, GARBLED_MARKERS),
                option_index=index + 1))
        for detail in latex_syntax_details(raw_option):
            if detail["message"] == "含乱码字符":
                if reported_marker:
                    continue
                issues.append(_garbled_issue(
                    "options", f"选项{index + 1}含乱码字符", raw_option,
                    detail["offset"], option_index=index + 1))
                reported_marker = True
                continue
            issues.append(_garbled_issue(
                "options", f"选项{index + 1}{detail['message']}", raw_option,
                detail["offset"], option_index=index + 1))
    return issues


def k12_field_errors(meta: dict, body: dict, *,
                     raw_texts: dict | None = None) -> list[str]:
    """给生成闸用的字符串列表版本，规则与 ``k12_field_issues`` 完全同源。"""
    return [item["message"] for item in k12_field_issues(
        meta, body, raw_texts=raw_texts)]


def _content_signature(sample: dict) -> str:
    """Stable fingerprint of one delivered sample, used for dedup (H3⑩ / 备注2)."""
    body = sample["data"]
    payload = "\u0000".join([
        re.sub(r"\s+", "", str(body.get("question", ""))),
        "\u0001".join(re.sub(r"\s+", "", str(value)) for value in body.get("options", [])),
        re.sub(r"\s+", "", str(body.get("answer", ""))),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


class _DeliveryIdentity:
    """跨批次样本身份库（sqlite）：题目身份 → 样本 id，外加内容指纹去重。

    两张表回答两个不同的问题，混用会出错：

    * ``questions``（``<教材>#<题目持久 id>``）回答"这是不是同一道题"。同一道题
      重出——哪怕答案改过——都复用同一个 sample_id，新的 JSON 直接覆盖旧的，
      旧内容不会作为另一条样本留在包里。
    * ``samples``（内容指纹）回答"这段内容有没有在别的书里交付过"，对应交付方
      备注第 2 条与 H3⑩。

    两者分开的直接后果：改答案不再等于"又一道新题"。以前按内容指纹发号，
    改一个字就是新 id，旧 id 那条错数据还留在身份库里。

    存 sqlite 而不是 JSON：1000 万条指纹的 JSON 全量读写在每次出包时都不可行。
    """

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None
        self.connection = None
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            "CREATE TABLE IF NOT EXISTS samples ("
            " content_hash TEXT PRIMARY KEY, book_id TEXT NOT NULL,"
            " sample_id TEXT NOT NULL, created_at TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS questions ("
            " question_key TEXT PRIMARY KEY, book_id TEXT NOT NULL,"
            " sample_id TEXT NOT NULL, content_hash TEXT NOT NULL,"
            " updated_at TEXT NOT NULL);"
            "CREATE INDEX IF NOT EXISTS idx_questions_hash ON questions(content_hash);"
            "CREATE TABLE IF NOT EXISTS counter ("
            " name TEXT PRIMARY KEY, value INTEGER NOT NULL);"
            "INSERT OR IGNORE INTO counter(name, value) VALUES ('sample_id', 1);"
        )
        self.connection.commit()

    def lookup(self, content_hash: str) -> dict | None:
        """Content-level owner (read-only). Used by the dry-run diagnosis."""
        if self.connection is None:
            return None
        row = self.connection.execute(
            "SELECT book_id, sample_id FROM samples WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return None if row is None else {"book_id": row[0], "id": row[1]}

    def claim(self, question_key: str, book_id: str, content_hash: str) -> dict:
        """Resolve one question to its stable sample id, inside a single transaction.

        Returns ``{"status": ..., "id": ...}`` where status is:

        ``kept``      同一道题、内容没变，沿用原 id；
        ``updated``   同一道题、内容改过，沿用原 id 并顶掉旧内容指纹（旧 JSON 作废）；
        ``reused``    同书同内容的历史样本（升级前的库没有题目身份），复用其 id；
        ``new``       新题，发新号；
        ``conflict``  同内容已在别的书交付过，按数据要求第 2 条拒收。
        """
        if self.connection is None:
            return {"status": "new", "id": "0000001"}
        cursor = self.connection.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            row = cursor.execute(
                "SELECT sample_id, content_hash FROM questions WHERE question_key = ?",
                (question_key,)).fetchone()
            if row is not None:
                sample_id, previous = str(row[0]), str(row[1])
                if previous == content_hash:
                    self.connection.commit()
                    return {"status": "kept", "id": sample_id}
                # 同一道题改了内容：旧指纹如果是别的题目共用的就留着，
                # 否则删掉——被顶掉的那条 JSON 不再算作已交付内容。
                cursor.execute(
                    "DELETE FROM samples WHERE content_hash = ?"
                    " AND NOT EXISTS (SELECT 1 FROM questions"
                    "                 WHERE content_hash = ? AND question_key <> ?)",
                    (previous, previous, question_key))
                cursor.execute(
                    "INSERT OR REPLACE INTO samples(content_hash, book_id, sample_id, created_at)"
                    " VALUES (?, ?, ?, ?)", (content_hash, book_id, sample_id, _stamp()))
                cursor.execute(
                    "UPDATE questions SET content_hash = ?, updated_at = ? WHERE question_key = ?",
                    (content_hash, _stamp(), question_key))
                self.connection.commit()
                return {"status": "updated", "id": sample_id}
            owner = cursor.execute(
                "SELECT book_id, sample_id FROM samples WHERE content_hash = ?",
                (content_hash,)).fetchone()
            if owner is not None and str(owner[0]) != book_id:
                self.connection.commit()
                return {"status": "conflict", "id": "", "book_id": str(owner[0])}
            if owner is not None:
                sample_id, status = str(owner[1]), "reused"
            else:
                value = cursor.execute(
                    "SELECT value FROM counter WHERE name = 'sample_id'").fetchone()[0]
                cursor.execute("UPDATE counter SET value = value + 1 WHERE name = 'sample_id'")
                sample_id = f"{int(value):07d}"
                cursor.execute(
                    "INSERT OR REPLACE INTO samples(content_hash, book_id, sample_id, created_at)"
                    " VALUES (?, ?, ?, ?)", (content_hash, book_id, sample_id, _stamp()))
                status = "new"
            cursor.execute(
                "INSERT OR REPLACE INTO questions"
                "(question_key, book_id, sample_id, content_hash, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (question_key, book_id, sample_id, content_hash, _stamp()))
            self.connection.commit()
            return {"status": status, "id": sample_id}
        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
            self.connection = None


def _page_pixel_size(project: dict, page: dict, cache: dict[int, tuple[int, int]],
                     page_index: int) -> tuple[int, int]:
    """Page width/height for the whole-page guard, decoded at 1/8 scale and cached."""
    if page_index in cache:
        return cache[page_index]
    size = (0, 0)
    try:
        import cv2
        import numpy as np

        # cv2.imread cannot open non-ASCII paths on Windows, and every textbook
        # folder here is Chinese-named. Decode from bytes instead.
        raw = np.fromfile(data.source_path(project, page), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_REDUCED_COLOR_4) if raw.size else None
        if image is not None:
            height, width = image.shape[:2]
            size = (width * 4, height * 4)
    except (OSError, ValueError, ImportError):
        size = (0, 0)
    cache[page_index] = size
    return size


def _wide_path(path: Path) -> str:
    """Windows delivery trees are deep; opt into the extended-length path form.

    A server generation writes under
    ``<root>/exports/<version>/staging/<book_id>/images/questions/P0001/...``.
    On the data machine that can exceed the classic 260-character MAX_PATH,
    where directory creation still succeeds but opening the final file fails
    with a bare FileNotFoundError.
    """
    text = str(path.resolve())
    if os.name == "nt" and len(text) >= 240 and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(_wide_path(temporary), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(_wide_path(temporary), _wide_path(path))


def _atomic_image(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("交付图片编码失败")
    temporary = path.with_suffix(path.suffix + ".tmp")
    # ndarray.tofile() goes through the C runtime's fopen, which cannot use
    # long paths and mishandles non-ASCII folders.  Python IO honours the
    # wide-path prefix, matching how PaperFlow reads scans (fromfile/imdecode).
    with open(_wide_path(temporary), "wb") as handle:
        handle.write(encoded.tobytes())
    os.replace(_wide_path(temporary), _wide_path(path))


def _delivery_artifact_path(root: Path, value: object) -> Path | None:
    """Resolve a manifest path only when it names a managed delivery artifact."""
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute():
        return None
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        managed_relative = candidate.relative_to(resolved_root)
    except ValueError:
        return None
    if (not managed_relative.parts
            or managed_relative.parts[0].lower() not in {"images", "json"}
            or candidate.suffix.lower() not in {".jpg", ".json"}):
        return None
    return candidate


def _reuse_artifact(reuse_root: Path | None, value: object) -> Path | None:
    """Resolve ``manifest.json``'s relative artifact path inside an older version."""
    text = str(value or "").strip()
    if reuse_root is None or not text:
        return None
    relative = Path(text)
    if relative.is_absolute():
        return None
    resolved_root = Path(reuse_root).resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _link_reused_file(source: Path | None, target: Path) -> bool:
    """Reuse a crop from the previous version without decoding or re-encoding it.

    Hard links keep disk usage flat (each version is immutable and is deleted
    independently); ``os.replace`` writes never mutate a shared inode, so a
    later version can safely replace its own copy.  Fall back to a plain copy
    when the two versions live on different volumes or the filesystem refuses
    links — a copy is still far cheaper than a full-page decode + JPEG encode.
    """
    if source is None or not source.is_file() or target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return True
    except OSError:
        pass
    try:
        shutil.copyfile(source, target)
        return True
    except OSError:
        return False


def _load_reuse_items(reuse_root: Path | None) -> dict[str, dict]:
    if reuse_root is None:
        return {}
    try:
        manifest = json.loads((Path(reuse_root) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return {item.get("id"): item for item in manifest.get("items", [])
            if isinstance(item, dict) and item.get("id")}


def _illustration_signature(source: str | Path, box: object) -> str:
    """K12 crop identity: page bytes + box. Illustration edits must never reuse a stale crop."""
    path = Path(source).resolve()
    stat = path.stat()
    payload = {"source": os.path.normcase(str(path)), "size": stat.st_size,
               "mtime_ns": stat.st_mtime_ns, "bbox": list(box or []),
               "crop_version": "illustration_v1"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _number_sort_key(record: dict) -> tuple:
    parts = []
    for value in data.number_path(record):
        number = data.chinese_number_to_int(value)
        normalized = re.sub(r"\s+", "", str(value)).lower()
        parts.append((0, number, "") if number is not None else (1, 0, normalized))
    return tuple(parts)


def _record_position(row: tuple[int, dict, dict]) -> tuple:
    """Stable human order: page, printed hierarchy, box position, persistent id."""
    page_index, _page, record = row
    box = record.get("bbox", [0, 0, 0, 0])
    hierarchy = _number_sort_key(record)
    return (page_index, 0 if hierarchy else 1, hierarchy,
            int(box[1]), int(box[0]), str(record.get("delivery_id", "")))


def _clean_text(value: object) -> str:
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip()
             for line in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [_clean_text(item) for item in re.split(r"[\n,，;；]+", str(value or "")) if _clean_text(item)]


_CHOICE_MARKER = re.compile(r"(^|[\n\s])([A-DＡ-Ｄ])\s*[\.．、:：]\s*")


def _split_ocr_choices(value: object) -> tuple[str, list[str]]:
    """Extract A-D choices only when OCR text contains at least two markers."""
    source = _clean_text(value)
    matches = list(_CHOICE_MARKER.finditer(source))
    if len(matches) < 2:
        return source, []
    first = matches[0]
    marker_offset = next((index for index, char in enumerate(first.group(0))
                          if char in "ABCDＡＢＣＤ"), 0)
    question = source[:first.start() + marker_offset].strip()
    options: list[str] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[match.end():body_end].strip()
        label = match.group(2)
        if "Ａ" <= label <= "Ｄ":
            label = chr(ord(label) - 0xFEE0)
        if body:
            options.append(f"{label}. {body}")
    return (question, options) if len(options) >= 2 else (source, [])


_LATEX_SYMBOLS = {
    "times": "×", "div": "÷", "cdot": "·", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "neq": "≠", "pm": "±", "mp": "∓",
    "infty": "∞", "approx": "≈", "sim": "∼", "angle": "∠",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ",
    "lambda": "λ", "mu": "μ", "pi": "π", "sigma": "σ", "omega": "ω",
    "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Omega": "Ω",
}
_KNOWN_LATEX_COMMANDS = frozenset({
    "frac", "sqrt", "text", "mathrm", "overline", "underline", "binom",
    "left", "right", "begin", "end", "displaystyle", "quad", "qquad",
    "sum", "int", "lim", "log", "sin", "cos", "tan", "vec", "hat", "bar",
    "dot", "perp", "parallel", "circ", "degree", "because", "therefore",
    "triangle", "equiv", "cong", "geq", "leq",
} | set(_LATEX_SYMBOLS))
_SUPERSCRIPTS = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPTS = str.maketrans("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

def _latex_group(source: str, start: int) -> tuple[str, int]:
    """Read one braced LaTeX group; malformed input is kept visible."""
    if start >= len(source) or source[start] != "{":
        return "", start
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index], index + 1
    return source[start + 1:], len(source)


def _unicode_math_text(value: object) -> str:
    """Turn OCR LaTeX into customer-facing text without leaking source syntax.

    The UI keeps its richer, vertically typeset DOM preview. JSON is text-only,
    so fractions use U+2044 and unsupported scripts stay readable in brackets.
    """
    source = _clean_text(value).replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")

    def convert(text: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == "$":
                index += 1
                continue
            if char == "\\":
                match = re.match(r"\\([A-Za-z]+|.)", text[index:])
                if not match:
                    index += 1
                    continue
                command = match.group(1)
                index += len(match.group(0))
                if command == "frac":
                    numerator, index = _latex_group(text, index)
                    denominator, index = _latex_group(text, index)
                    output.append(f"{convert(numerator)}⁄{convert(denominator)}")
                elif command == "sqrt":
                    if index < len(text) and text[index] == "[":
                        close = text.find("]", index + 1)
                        index = len(text) if close < 0 else close + 1
                    radicand, index = _latex_group(text, index)
                    output.append(f"√({convert(radicand)})")
                elif command in {"left", "right", "displaystyle", "mathrm", "text", "quad", "qquad", ",", "!", " "}:
                    if command in {"mathrm", "text"}:
                        group, index = _latex_group(text, index)
                        output.append(convert(group))
                else:
                    output.append(_LATEX_SYMBOLS.get(command, command))
                continue
            if char in {"^", "_"}:
                script, next_index = _latex_group(text, index + 1)
                if not script and index + 1 < len(text):
                    script, next_index = text[index + 1], index + 2
                converted = convert(script)
                table = _SUPERSCRIPTS if char == "^" else _SUBSCRIPTS
                styled = converted.translate(table)
                output.append(styled if styled != converted else f"{'⁽' if char == '^' else '₍'}{converted}{'⁾' if char == '^' else '₎'}")
                index = next_index
                continue
            output.append(char)
            index += 1
        return "".join(output)

    return _clean_text(convert(source)).replace("{", "(").replace("}", ")")


def _strip_printed_minor_prefix(text: str, minor: object) -> str:
    """剥掉答案文本开头的印刷小题号（答案框常把"8."一起框进切片）。

    只在开头数字与该答案的小题号一致时才剥；剥完只剩纯数字时不动——
    那更可能是答案本身的小数（小题 8 的答案 "8.5"），而不是题号。
    "7.6.0288" 这类出现第二个分隔点的，说明 "7." 一定是题号，剥成 "6.0288"。
    前面带标点的变体（"，、1. 12.56 …"，OCR 把答案行的序号点读歪了）同样剥。
    点号后面有空格时（印刷答案行 "3. 15立方分米" 的标准排版）也剥——真小数
    "3.14" 的点后不会有空格。
    """
    raw = str(text or "")
    try:
        minor_value = int(str(minor).strip())
    except (TypeError, ValueError):
        return raw
    match = re.match(r"^[、，,.\s]*(\d{1,3})\s*[.、．](\s*)(\S.*)$", raw, re.S)
    if not match or int(match.group(1)) != minor_value:
        return raw
    rest = match.group(3)
    if (rest[:1].isdigit() and not match.group(2)
            and not re.search(r"[.、．]", rest)):
        return raw
    return rest


def _analysis_is_numeric_fragment(value: str, body: str) -> bool:
    """判断一段解析是否只是答案数字的碎片（旧 OCR 残留的典型形态）。

    答案框重新 OCR 后 ``text`` 会刷新，但早前"答案/解析拆分"写进
    ``analysis`` 的旧值会残留（如 "12.\\n56\\n68\\n37\\n62.\\n1."）。
    特征：整段没有任何文字（去掉数字/标点/空白后为空），且每个数字串
    都能被答案正文里的某个数字串覆盖 —— 这种"解析"没有信息量，丢弃；
    真正的解析总有文字叙述，不会命中。
    """
    text = str(value or "")
    if not text.strip():
        return False
    if re.search(r"[\u4e00-\u9fffA-Za-z]", text):
        return False
    digits = re.findall(r"\d+", text)
    if not digits:
        return True  # 纯标点/空白，没有信息量
    body_digits = re.findall(r"\d+", body)
    return all(any(token in candidate for candidate in body_digits) for token in digits)


def _answer_body_text(record: dict, prefer_text: bool = False) -> str:
    """答案记录的正文：人工填的标准答案优先，其次答案框的 OCR 原文。

    快照层（``enrich_snapshot._answer_text``）会退到 ``analysis``，但交付的
    "标准答案"字段不能拿解析过程顶替，所以顺序是 answer → text → analysis。
    答案框只做了 OCR、没人工填 answer 时，这里能取到 ``text``，不该判缺答案。

    ``prefer_text=True`` 用于**有续框的答案**：框被拆分（答案换行处切出续框）
    之后，``answer`` 里还是拆分前的旧合并文本（含续框那半的内容，常见乱序
    碎片），而 ``text`` 是当前框自己的重新 OCR——此时以 text 为准，续框内容
    由续框记录自己供给，避免"旧合并文本 + 续框"拼出重复错乱的答案。
    """
    if not isinstance(record, dict):
        return ""
    keys = ("text", "answer", "analysis") if prefer_text else ("answer", "text", "analysis")
    removal_source = str(record.get("number_prefix_removal_source", "") or "")
    ocr_meta = record.get("ocr", {}) if isinstance(record.get("ocr"), dict) else {}
    number_already_excluded = bool(
        record.get("number_prefix_removed")
        or removal_source in {"answer_anchor_geometry", "geometry_mask"}
        or ocr_meta.get("number_mask_applied")
    )
    for key in keys:
        value = _clean_text(record.get(key))
        if value:
            # Geometry-based masking has already removed only the printed
            # number. Running the text heuristic again could destroy a valid
            # answer whose own first token is e.g. "3. 15".
            if number_already_excluded:
                return value
            return _strip_printed_minor_prefix(
                value, record.get("minor_number", record.get("number")))
    return ""


def _merged_answer_text(answers: list[str], analyses: list[str]) -> str:
    """答案和解析合成一段：原教材给了解析就接在答案后面，不再单列一份。

    交付 JSON 的 ``answer`` 能直接给到答案与讲解两部分；``analysis`` 字段保留
    但不再要求有值，缺解析也不拦。空答案、只有解析的题目仍然出得来。
    """
    body = "\n".join(value for value in answers if value).strip()
    extras: list[str] = []
    for value in analyses:
        text = value.strip()
        if not text:
            continue
        # 同一条记录里 answer 和 analysis 常常是同一段话，重复贴两遍没有意义。
        if text in body or text in extras:
            continue
        # 答案框重新 OCR 后旧拆分残留在 analysis 里的纯数字碎片，同样没有
        # 信息量（工作台侧不重刷旧字段，交付出口兜底）。
        if _analysis_is_numeric_fragment(text, body):
            continue
        extras.append(text)
    if not extras:
        return body
    return "\n".join([body, *extras]) if body else "\n".join(extras)


def _k12_validation(sample: dict, image_sizes: list[tuple[int, int]], *,
                     raw_texts: dict | None = None) -> tuple[list[str], list[str]]:
    """Return ``(errors, hints)``: errors block delivery, hints only inform.

    题图不是必填项 —— 交付规范要求 ``question_images`` 字段存在，空数组是
    合法值；一本练习册里大部分题本来就没有插图。原来把"没有题图"当拒收理由，
    会让整本书一条都出不来。

    解析同样不再必填：答案框里有什么就交付什么，原教材给了讲解就并进
    ``answer``，没给也不拦 —— 否则一整本没有解析的教辅一道题都出不了。

    字段/乱码部分交给 ``k12_field_errors``，员工自检调用的是同一个函数，
    界面提示与生成拒收永远同口径。
    """
    meta, body = sample["metainfo"], sample["data"]
    errors = k12_field_errors(meta, body, raw_texts=raw_texts)
    hints: list[str] = []
    if not body.get("question_images") and not any(body.get("option_images", [])):
        hints.append("本题没有题图")
    if any(min(width, height) < 128 for width, height in image_sizes):
        errors.append("图片最短边低于128像素")
    if meta.get("question_type") == "选择" and len(body.get("options", [])) < 2:
        errors.append("选择题选项不完整")
    return errors, hints


def _write_k12_delivery(root: Path, project: dict, question_rows: list, answer_rows: list,
                        question_relations: dict[str, dict], *, changed_page_index: int | None,
                        materialize_all: bool, identity_path: Path | None = None,
                        allowed_pages: set[tuple[str, int]] | None = None,
                        rejected_pages: set[tuple[str, int]] | None = None,
                        only_reviewed: bool = False,
                        reuse_root: Path | None = None) -> dict:
    """Build the customer-facing UTF-8 JSONL and meaningful illustration crops.

    Internal Q/A crops remain available for audit, but answer screenshots and whole-question
    screenshots are deliberately not referenced by this formal dataset.

    ``allowed_pages`` / ``rejected_pages`` 都是 ``{("question"|"answer", index)}``
    （两个列表各自从 0 编号，所以必须带 kind）：

    * ``rejected_pages`` —— 审核员点过"退回/需精修"的页，**永远不收录**。
    * ``only_reviewed=True`` —— 只收 ``allowed_pages`` 里已放行的页（严格版流程）。
    * 默认 ``only_reviewed=False`` —— 不看放行状态，题目字段齐全就收录；
      没审完不影响出包，适合抽检流程。
    """
    k12_root = root / "k12_dataset"
    pages_root = k12_root / "pages"
    images_root = k12_root / "images"
    pages_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    answers_by_internal = {record["id"]: (index, page, record)
                           for index, page, record in answer_rows}
    continuations_by_parent: dict[str, list[tuple]] = {}
    for row in answer_rows:
        record = row[2]
        if record.get("region_role") == "continuation" and record.get("continuation_of"):
            continuations_by_parent.setdefault(str(record["continuation_of"]), []).append(row)

    def answer_chain(row: tuple) -> list[tuple]:
        chain, queue, seen = [], [row], set()
        while queue:
            current = queue.pop(0)
            internal_id = str(current[2].get("id", ""))
            if internal_id in seen:
                continue
            seen.add(internal_id)
            chain.append(current)
            queue.extend(sorted(continuations_by_parent.get(internal_id, []), key=_record_position))
        return chain
    ordered_questions = sorted(question_rows, key=_record_position)
    lines, report_items, active_images, page_lines = [], [], set(), {}
    page_orders = {}
    page_sizes: dict[int, tuple[int, int]] = {}
    # 上一版为每张插图记录「源图 + bbox」签名，签名不变就直接复用文件，
    # 不再为了同一个框重新解码整张试卷页。
    reuse_signatures: dict[str, str] = {}
    emitted_signatures: dict[str, str] = {}
    if reuse_root is not None:
        try:
            cached = json.loads((Path(reuse_root) / "k12_dataset" / "image_signatures.json")
                                .read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                reuse_signatures = {str(key): str(value) for key, value in cached.items()}
        except (OSError, ValueError, TypeError):
            reuse_signatures = {}
    identity = _DeliveryIdentity(identity_path)
    book_id = str(project["delivery_book_id"])
    claimed_signatures: dict[str, str] = {}
    identity_counts: dict[str, int] = {}
    skipped_unreviewed_pages: set[int] = set()
    skipped_unreviewed_questions = 0
    blocked_by_review_pages: set[int] = set()
    blocked_by_review_questions = 0
    for global_order, (page_index, page, question) in enumerate(ordered_questions, 1):
        if rejected_pages and ("question", page_index) in rejected_pages:
            # 审核员明确说"退回/需精修"：字段再齐也不进包。
            blocked_by_review_pages.add(page_index)
            blocked_by_review_questions += 1
            continue
        if only_reviewed and allowed_pages is not None and ("question", page_index) not in allowed_pages:
            # 严格模式下才按"已放行"筛；默认模式不看审核进度。
            skipped_unreviewed_pages.add(page_index)
            skipped_unreviewed_questions += 1
            continue
        page_number = page_index + 1
        page_folder = f"P{page_number:04d}"
        page_orders[page_index] = page_orders.get(page_index, 0) + 1
        page_order = page_orders[page_index]
        image_root = images_root
        materialize_page = materialize_all or page_index == changed_page_index
        relation = question_relations.get(str(question.get("id", "")), {})
        primary_answer_infos = [answers_by_internal.get(str(value)) for value in relation.get("answer_ids", [])]
        primary_answer_infos = sorted((item for item in primary_answer_infos if item), key=_record_position)
        answer_infos = [part for item in primary_answer_infos for part in answer_chain(item)]

        def _answer_body(item: tuple) -> str:
            record = item[2]
            # 有续框的答案：answer 字段是拆分前的旧合并文本，改用框自身 OCR。
            prefer_text = bool(continuations_by_parent.get(str(record.get("id", ""))))
            return _answer_body_text(record, prefer_text)

        answers = [_answer_body(item) for item in answer_infos]
        analyses = [_clean_text(item[2].get("analysis")) for item in answer_infos]
        illustrations = sorted(
            (item for item in question.get("illustrations", []) if isinstance(item, dict)),
            key=lambda item: (int(item.get("bbox", [0, 0, 0, 0])[1]), int(item.get("bbox", [0, 0, 0, 0])[0])),
        )
        # Pass 1 — decide which crops may be delivered at all (no id needed yet).
        slots, image_sizes, full_page_skipped = [], [], 0
        for illustration in illustrations:
            kind = str(illustration.get("image_kind", "question_image"))
            if kind == "ignore":
                continue
            box = illustration.get("bbox", [])
            width = max(0, int(box[2]) - int(box[0])) if len(box) == 4 else 0
            height = max(0, int(box[3]) - int(box[1])) if len(box) == 4 else 0
            # H3③：整页试卷截图不得作为题图。候选阶段已判过 page_like，但那时
            # 只是"灰色不可采用"；人工把整页框标成 image-area 后仍会进 illustrations，
            # 所以交付闸必须按页面像素面积二次复查。
            page_width, page_height = _page_pixel_size(project, page, page_sizes, page_index)
            if page_width and page_height:
                area_ratio = (width * height) / float(page_width * page_height)
                if area_ratio >= FULL_PAGE_AREA_RATIO:
                    full_page_skipped += 1
                    continue
            image_sizes.append((width, height))
            label = ""
            if kind == "option_image":
                label = re.sub(r"[^A-Za-z0-9]", "", str(illustration.get("option_label", ""))).upper() or "X"
            slots.append((kind, box, label))

        ocr_question, inferred_options = _split_ocr_choices(question.get("text", ""))
        explicit_options = _string_list(question.get("options", []))
        options = explicit_options or inferred_options
        raw_type = _clean_text(question.get("question_type", ""))
        effective_type = "选择题" if inferred_options else raw_type
        final_type = K12_QUESTION_TYPES.get(effective_type, effective_type)

        def build_sample(sample_id: str) -> tuple[dict, list[tuple[Path, list]]]:
            question_images, option_images, targets = [], [], []
            q_index = option_index = 0
            for slot_kind, slot_box, slot_label in slots:
                if slot_kind == "option_image":
                    option_index += 1
                    filename = f"{sample_id}_option_{slot_label}_{option_index:02d}.jpg"
                    option_images.append(f"images/{filename}")
                else:
                    q_index += 1
                    filename = f"{sample_id}_img_{q_index:02d}.jpg"
                    question_images.append(f"images/{filename}")
                targets.append(((image_root / filename).resolve(), slot_box))
            return ({
                "id": sample_id,
                "metainfo": {
                    "language": _clean_text(project.get("language")),
                    "stage": _clean_text(project.get("stage")),
                    "grade": _clean_text(project.get("grade")),
                    "subject": _clean_text(project.get("subject")),
                    "question_type": final_type,
                    "difficulty": _clean_text(question.get("difficulty", "")),
                    "knowledge_points": _string_list(question.get("knowledge_points", [])),
                },
                "data": {
                    "question_images": question_images,
                    "option_images": option_images,
                    "question": _k12_text(ocr_question),
                    "options": [_k12_text(value) for value in options] if final_type == "选择" else [],
                    "answer": _k12_text(_merged_answer_text(answers, analyses)),
                    # 讲解已经并进 answer，不再单列一份；字段保留，缺失不算错。
                    "analysis": "",
                },
                "source": {
                    "book_id": book_id, "question_page": page_number,
                    "page_order": page_order,
                    "question_number": data.display_number(question),
                    "internal_question_id": question.get("delivery_id", ""),
                    "internal_answer_ids": [item[2].get("delivery_id", "") for item in answer_infos],
                },
            }, targets)

        sample, image_targets = build_sample("0000000")   # probe: id is irrelevant to validation
        errors, hints = _k12_validation(sample, image_sizes, raw_texts={
            "question": ocr_question,
            "answer": _merged_answer_text(answers, analyses),
            "options": options,
        })
        if full_page_skipped:
            errors.append(f"已剔除{full_page_skipped}张整页截图（H3③）")
        signature = _content_signature(sample)
        if signature in claimed_signatures:
            errors.append(f"与第{claimed_signatures[signature]}条重复（H3⑩）")
        if str(page.get("status", "")) != "accepted":
            errors.append("题目页尚未确认")
        if relation.get("status") != "accepted" or not answer_infos:
            errors.append("题目尚未匹配答案")
        elif any(str(item[1].get("status", "")) != "accepted" for item in answer_infos):
            errors.append("答案页尚未确认")
        # 题目页能收（上面的 continue 已经排除了被退回的页），但答案页被退回
        # 或者严格模式没放行时，答案不能跟着发出去。
        if rejected_pages and any(("answer", int(item[0])) in rejected_pages for item in answer_infos):
            errors.append("答案页被审核退回")
        elif only_reviewed and allowed_pages is not None and any(
                ("answer", int(item[0])) not in allowed_pages for item in answer_infos):
            errors.append("答案页尚未审核放行")
        eligible = (str(page.get("status", "")) == "accepted"
                    and relation.get("status") == "accepted"
                    and bool(answer_infos) and all(str(item[1].get("status", "")) == "accepted" for item in answer_infos)
                    and not errors)
        sample_id = f"{global_order:07d}"
        identity_status = ""
        if eligible:
            # 同一道题只保留一条样本：题目身份（教材 + 题目持久 id）决定 sample_id，
            # 重出时不论谁点的生成，都沿用同一个 id 并顶掉旧内容——旧 JSON 作废。
            # 内容指纹只用于跨书查重（数据要求第 2 条 / H3⑩）。
            question_key = f"{book_id}#{question.get('delivery_id') or question.get('id') or global_order}"
            claim = identity.claim(question_key, book_id, signature)
            if claim["status"] == "conflict":
                errors.append(f"与教材 {claim.get('book_id')} 已交付样本内容重复（备注2）")
                eligible = False
            else:
                sample_id = str(claim.get("id") or sample_id)
                identity_status = claim["status"]
                claimed_signatures[signature] = sample_id
                identity_counts[identity_status] = identity_counts.get(identity_status, 0) + 1
        report_items.append({"id": sample_id, "page": page_index + 1,
                             "question_number": data.display_number(question),
                             "eligible": eligible, "errors": errors, "hints": hints,
                             "identity": identity_status})
        if eligible:
            sample, image_targets = build_sample(sample_id)
            source_path = data.source_path(project, page)
            for target, box in image_targets:
                relative = f"images/{target.name}"
                signature = _illustration_signature(source_path, box)
                emitted_signatures[relative] = signature
                if not (materialize_page or not target.is_file()):
                    continue
                reusable = reuse_signatures.get(relative) == signature
                if reusable and target.is_file():
                    continue  # 内容已是正确版本
                if reusable and _link_reused_file(
                        _reuse_artifact(reuse_root, f"k12_dataset/{relative}"), target):
                    continue
                crop = _crop(source_path, box)
                _atomic_image(target, crop)
            encoded = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
            lines.append(encoded)
            page_lines.setdefault(page_index, []).append(encoded)
            active_images.update(target for target, _box in image_targets)
        else:
            for target, _box in image_targets:
                if target.is_file():
                    target.unlink()
    identity.close()
    for stale in images_root.glob("*.jpg"):
        if stale.resolve() not in active_images:
            stale.unlink()
    # Remove images emitted by the pre-contract page-folder layout as well.
    for stale in pages_root.glob("P*/images/*.jpg"):
        if stale.resolve() not in active_images:
            stale.unlink()
    # One folder per source page, including pages without accepted samples, so
    # operators can audit the book page-by-page without gaps.
    active_page_indexes = ({index for index in range(len(project.get("pages", [])))}
                           if materialize_all else ({changed_page_index} if changed_page_index is not None else set()))
    if only_reviewed and allowed_pages is not None:
        active_page_indexes &= {index for kind, index in allowed_pages if kind == "question"}
    active_page_indexes.update(
        int(path.name[1:]) - 1 for path in pages_root.glob("P[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    active_page_indexes.update(page_lines)
    for page_index in active_page_indexes:
        page_dir = pages_root / f"P{page_index + 1:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_jsonl = page_dir / "dataset.jsonl"
        payload = "\n".join(page_lines.get(page_index, []))
        temporary_page = page_jsonl.with_suffix(".jsonl.tmp")
        temporary_page.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        os.replace(temporary_page, page_jsonl)
    dataset_path = k12_root / "dataset.jsonl"
    temporary = dataset_path.with_suffix(".jsonl.tmp")
    temporary.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    os.replace(temporary, dataset_path)
    report = {"schema_version": 2,
              # total_questions / rejected_count 只算"本次生成范围"（已放行页面），
              # 未放行的页单独记在 scope 里：它们是"还没审"，不是"不合格"。
              "total_questions": len(report_items),
              "book_question_count": len(ordered_questions),
              "eligible_count": len(lines),
              "rejected_count": len(report_items) - len(lines),
              "scope": {"reviewed_only": bool(only_reviewed and allowed_pages is not None),
                        "skipped_unreviewed_pages": len(skipped_unreviewed_pages),
                        "skipped_unreviewed_questions": skipped_unreviewed_questions,
                        "blocked_by_review_pages": len(blocked_by_review_pages),
                        "blocked_by_review_questions": blocked_by_review_questions},
              "reason_summary": _reason_summary(report_items),
              "hint_summary": _reason_summary(report_items, key="hints", limit=12),
              "identity_summary": identity_counts,
              "items": report_items}
    _atomic_json(k12_root / "quality_report.json", report)
    _atomic_json(k12_root / "coverage_report.json", _coverage_summary(lines))
    _atomic_json(k12_root / "image_signatures.json", emitted_signatures)
    return {"dataset_file": str(dataset_path), "eligible_count": len(lines),
            "rejected_count": report["rejected_count"],
            "identity_summary": identity_counts}


def _reason_summary(items: list[dict], key: str = "errors", limit: int = 40) -> list[dict]:
    """Aggregate per-question blockers (``key="errors"``) or hints into buckets."""
    buckets: dict[str, dict] = {}
    for item in items:
        for reason in item.get(key, []) or []:
            bucket = buckets.setdefault(
                reason, {"reason": reason, "count": 0, "questions": []})
            bucket["count"] += 1
            number = str(item.get("question_number") or "").strip()
            page = item.get("page")
            # 示例只给题号：页码是扫描页序（含封面、前置页），和书上的印刷页码
            # 对不上，混在一起反而让人以为写错了。只有题号缺失时才用页号兜底定位。
            label = number or (f"第{page}页" if page else "")
            if label and len(bucket["questions"]) < 5 and label not in bucket["questions"]:
                bucket["questions"].append(label)
    return sorted(buckets.values(), key=lambda row: (-row["count"], row["reason"]))[:limit]


def _relation_maps(project: dict, questions_by_internal: dict,
                   answers_by_internal: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    """Accepted question↔answer pairs, keyed by internal record id.

    Shared by the writer and the read-only analyzer so both agree on which
    pairs count as "matched".
    """
    question_relations: dict[str, dict] = {}
    answer_relations: dict[str, dict] = {}
    for pair in project.get("pairs", []):
        if str(pair.get("status", "")) != "accepted" or not pair.get("answer_ids"):
            continue
        question = questions_by_internal.get(str(pair.get("question_id", "")))
        answers = [answers_by_internal.get(str(value)) for value in pair.get("answer_ids", [])]
        if (
            not question
            or str(question[1].get("status", "")) != "accepted"
            or not answers
            or any(row is None or str(row[1].get("status", "")) != "accepted" for row in answers)
        ):
            continue
        public_answers = [row[2]["delivery_id"] for row in answers if row]
        question_relations[question[2]["id"]] = {**pair, "public_answer_ids": public_answers}
        for answer in (row for row in answers if row):
            relation = answer_relations.setdefault(answer[2]["id"], {"public_question_ids": []})
            relation["public_question_ids"].append(question[2]["delivery_id"])
    return question_relations, answer_relations


BOOK_FIELDS = (("language", "语言"), ("stage", "学段"), ("grade", "年级"), ("subject", "学科"))


def analyze_delivery(project_path: str | Path, runtime_root: str | Path,
                     *, identity_path: str | Path | None = None,
                     project_document: dict | None = None,
                     allowed_pages: set[tuple[str, int]] | None = None,
                     rejected_pages: set[tuple[str, int]] | None = None,
                     only_reviewed: bool = False) -> dict:
    """Read-only readiness report: what blocks this book from generating, and why.

    Runs the same per-question checks as ``_write_k12_delivery`` but crops no
    image, writes no file and allocates no sample id — operators get the blocker
    list BEFORE a generation job is queued, instead of a generic failure after.

    口径与生成完全一致：默认按字段判定（``only_reviewed=False``），审核没做完
    也照样算；``only_reviewed=True`` 才只统计已放行页面。``rejected_pages``
    （审核退回/需精修）永远排除。
    """
    project_file = Path(project_path).resolve()
    project = dict(project_document) if project_document is not None else data.load_project(project_file)
    data.ensure_delivery_ids(project)
    root = delivery_root(project, runtime_root)
    question_rows = sorted(
        [row for row in data.all_questions(project)
         if row[2].get("region_role", "question") == "question"],
        key=_record_position,
    )
    answer_rows = sorted(
        [row for row in data.all_answers(project)
         if row[2].get("region_role", "question") in {"question", "continuation"}],
        key=_record_position,
    )
    questions_by_internal = {record["id"]: (index, page, record)
                             for index, page, record in question_rows}
    answers_by_internal = {record["id"]: (index, page, record)
                           for index, page, record in answer_rows}
    question_relations, _answer_relations = _relation_maps(
        project, questions_by_internal, answers_by_internal)
    continuations_by_parent: dict[str, list[tuple]] = {}
    for row in answer_rows:
        record = row[2]
        if record.get("region_role") == "continuation" and record.get("continuation_of"):
            continuations_by_parent.setdefault(str(record["continuation_of"]), []).append(row)

    def answer_chain(row: tuple) -> list[tuple]:
        chain, queue, seen = [], [row], set()
        while queue:
            current = queue.pop(0)
            internal_id = str(current[2].get("id", ""))
            if internal_id in seen:
                continue
            seen.add(internal_id)
            chain.append(current)
            queue.extend(sorted(continuations_by_parent.get(internal_id, []), key=_record_position))
        return chain

    book_id = str(project["delivery_book_id"])
    book = {"book_id": book_id, "title": _clean_text(project.get("title", "")),
            "fields": {key: _clean_text(project.get(key)) for key, _label in BOOK_FIELDS},
            "errors": []}
    for key, label in BOOK_FIELDS:
        if not _clean_text(project.get(key)):
            book["errors"].append(f"缺少{label}")
        elif key == "language" and book["fields"][key] not in K12_LANGUAGES:
            book["errors"].append(f"语言不在规范词表内：{book['fields'][key]}")
        elif key == "stage" and book["fields"][key] not in K12_STAGES:
            book["errors"].append(f"学段不在规范词表内：{book['fields'][key]}")
        elif key == "grade" and book["fields"][key] not in K12_GRADES:
            book["errors"].append(f"年级不在规范词表内：{book['fields'][key]}")
        elif key == "subject" and book["fields"][key] not in K12_SUBJECTS:
            book["errors"].append(f"学科不在规范词表内：{book['fields'][key]}")

    identity_file = (Path(identity_path) if identity_path is not None
                     else root.parent / "delivery_identity.sqlite3")
    # Look-only: opening a missing identity file would create it, and a pure
    # diagnosis must not leave artifacts behind.
    identity = _DeliveryIdentity(identity_file if identity_file.is_file() else None)
    ordered_questions = question_rows
    items: list[dict] = []
    page_orders: dict[int, int] = {}
    page_sizes: dict[int, tuple[int, int]] = {}
    claimed_signatures: dict[str, str] = {}
    preview: list[dict] = []
    skipped_unreviewed_pages: set[int] = set()
    skipped_unreviewed_questions = 0
    blocked_by_review_pages: set[int] = set()
    blocked_by_review_questions = 0
    try:
        for global_order, (page_index, page, question) in enumerate(ordered_questions, 1):
            if rejected_pages and ("question", page_index) in rejected_pages:
                blocked_by_review_pages.add(page_index)
                blocked_by_review_questions += 1
                continue
            if only_reviewed and allowed_pages is not None and ("question", page_index) not in allowed_pages:
                skipped_unreviewed_pages.add(page_index)
                skipped_unreviewed_questions += 1
                continue
            page_number = page_index + 1
            page_orders[page_index] = page_orders.get(page_index, 0) + 1
            page_order = page_orders[page_index]
            relation = question_relations.get(str(question.get("id", "")), {})
            primary = [answers_by_internal.get(str(value)) for value in relation.get("answer_ids", [])]
            primary = sorted((item for item in primary if item), key=_record_position)
            answer_infos = [part for item in primary for part in answer_chain(item)]

            def _answer_body(item: tuple) -> str:
                record = item[2]
                # 有续框的答案：answer 字段是拆分前的旧合并文本，改用框自身 OCR。
                prefer_text = bool(continuations_by_parent.get(str(record.get("id", ""))))
                return _answer_body_text(record, prefer_text)

            answers = [_answer_body(item) for item in answer_infos]
            analyses = [_clean_text(item[2].get("analysis")) for item in answer_infos]
            illustrations = sorted(
                (item for item in question.get("illustrations", []) if isinstance(item, dict)),
                key=lambda item: (int(item.get("bbox", [0, 0, 0, 0])[1]),
                                  int(item.get("bbox", [0, 0, 0, 0])[0])),
            )
            slots, image_sizes, full_page_skipped = [], [], 0
            for illustration in illustrations:
                kind = str(illustration.get("image_kind", "question_image"))
                if kind == "ignore":
                    continue
                box = illustration.get("bbox", [])
                width = max(0, int(box[2]) - int(box[0])) if len(box) == 4 else 0
                height = max(0, int(box[3]) - int(box[1])) if len(box) == 4 else 0
                page_width, page_height = _page_pixel_size(project, page, page_sizes, page_index)
                if page_width and page_height and (
                        (width * height) / float(page_width * page_height) >= FULL_PAGE_AREA_RATIO):
                    full_page_skipped += 1
                    continue
                image_sizes.append((width, height))
                label = ""
                if kind == "option_image":
                    label = re.sub(r"[^A-Za-z0-9]", "",
                                   str(illustration.get("option_label", ""))).upper() or "X"
                slots.append((kind, box, label))
            ocr_question, inferred_options = _split_ocr_choices(question.get("text", ""))
            options = _string_list(question.get("options", [])) or inferred_options
            raw_type = _clean_text(question.get("question_type", ""))
            effective_type = "选择题" if inferred_options else raw_type
            final_type = K12_QUESTION_TYPES.get(effective_type, effective_type)

            def build_sample(sample_id: str) -> dict:
                question_images, option_images = [], []
                q_index = option_index = 0
                for slot_kind, _slot_box, slot_label in slots:
                    if slot_kind == "option_image":
                        option_index += 1
                        option_images.append(
                            f"images/{sample_id}_option_{slot_label}_{option_index:02d}.jpg")
                    else:
                        q_index += 1
                        question_images.append(f"images/{sample_id}_img_{q_index:02d}.jpg")
                return {
                    "id": sample_id,
                    "metainfo": {
                        "language": _clean_text(project.get("language")),
                        "stage": _clean_text(project.get("stage")),
                        "grade": _clean_text(project.get("grade")),
                        "subject": _clean_text(project.get("subject")),
                        "question_type": final_type,
                        "difficulty": _clean_text(question.get("difficulty", "")),
                        "knowledge_points": _string_list(question.get("knowledge_points", [])),
                    },
                    "data": {
                        "question_images": question_images,
                        "option_images": option_images,
                        "question": _k12_text(ocr_question),
                        "options": ([_k12_text(value) for value in options]
                                    if final_type == "选择" else []),
                        "answer": _k12_text(_merged_answer_text(answers, analyses)),
                        # 讲解已经并进 answer，不再单列一份；字段保留，缺失不算错。
                        "analysis": "",
                    },
                    "source": {
                        "book_id": book_id, "question_page": page_number,
                        "page_order": page_order,
                        "question_number": data.display_number(question),
                        "internal_question_id": question.get("delivery_id", ""),
                        "internal_answer_ids": [item[2].get("delivery_id", "")
                                                for item in answer_infos],
                    },
                }

            sample = build_sample("0000000")
            errors, hints = _k12_validation(sample, image_sizes, raw_texts={
                "question": ocr_question,
                "answer": _merged_answer_text(answers, analyses),
                "options": options,
            })
            if full_page_skipped:
                errors.append(f"已剔除{full_page_skipped}张整页截图（H3③）")
            signature = _content_signature(sample)
            if signature in claimed_signatures:
                errors.append(f"与第{claimed_signatures[signature]}条重复（H3⑩）")
            if str(page.get("status", "")) != "accepted":
                errors.append("题目页尚未确认")
            if relation.get("status") != "accepted" or not answer_infos:
                errors.append("题目尚未匹配答案")
            elif any(str(item[1].get("status", "")) != "accepted" for item in answer_infos):
                errors.append("答案页尚未确认")
            if rejected_pages and any(("answer", int(item[0])) in rejected_pages for item in answer_infos):
                errors.append("答案页被审核退回")
            elif only_reviewed and allowed_pages is not None and any(
                    ("answer", int(item[0])) not in allowed_pages for item in answer_infos):
                errors.append("答案页尚未审核放行")
            eligible = (str(page.get("status", "")) == "accepted"
                        and relation.get("status") == "accepted"
                        and bool(answer_infos)
                        and all(str(item[1].get("status", "")) == "accepted"
                                for item in answer_infos)
                        and not errors)
            if eligible:
                owner = identity.lookup(signature)
                if owner and str(owner.get("book_id", "")) != book_id:
                    errors.append(f"与教材 {owner.get('book_id')} 已交付样本内容重复（备注2）")
                    eligible = False
                else:
                    claimed_signatures[signature] = str(
                        (owner or {}).get("id") or f"{global_order:07d}")
                    if len(preview) < 3:
                        preview.append(build_sample(claimed_signatures[signature]))
            items.append({"page": page_number,
                          "question_number": data.display_number(question),
                          "eligible": eligible, "errors": errors, "hints": hints})
    finally:
        identity.close()

    total = len(items)
    eligible_count = sum(1 for item in items if item["eligible"])
    return {
        "schema_version": 2,
        "book": book,
        "total_questions": total,
        "book_question_count": len(ordered_questions),
        "eligible_count": eligible_count,
        "rejected_count": total - eligible_count,
        "scope": {"reviewed_only": bool(only_reviewed and allowed_pages is not None),
                  "skipped_unreviewed_pages": len(skipped_unreviewed_pages),
                  "skipped_unreviewed_questions": skipped_unreviewed_questions,
                  "blocked_by_review_pages": len(blocked_by_review_pages),
                  "blocked_by_review_questions": blocked_by_review_questions},
        "deliverable": eligible_count > 0,
        "reason_summary": _reason_summary(items),
        "hint_summary": _reason_summary(items, key="hints", limit=12),
        "items": items,
        "sample_preview": preview,
        "dataset_path": str(root / "k12_dataset" / "dataset.jsonl"),
    }


def _coverage_summary(lines: list[str]) -> dict:
    """H3⑨ 均衡覆盖统计：按学段/学科/语言/题型/难度计数。

    单个批次必然不均衡（一本书只覆盖一个年级），所以这里只出报表、不做交付闸；
    均衡是投放层按报表挑书，不是让生成器拒收。
    """
    counters = {key: {} for key in ("stage", "subject", "language", "question_type", "difficulty")}
    for line in lines:
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        meta = sample.get("metainfo", {})
        for key, bucket in counters.items():
            value = str(meta.get(key, "") or "未填")
            bucket[value] = bucket.get(value, 0) + 1
    return {"schema_version": 1, "samples": len(lines), "distribution": counters}


def delivery_root(project: dict, runtime_root: str | Path) -> Path:
    data.ensure_delivery_ids(project)
    configured_root = str(project.get("delivery_output_root", "") or "").strip()
    if configured_root:
        candidate = Path(configured_root).expanduser()
        if candidate.is_absolute():
            # Keep each book self-contained even when staff choose one shared
            # output folder for multiple textbooks.
            return candidate.resolve() / project["delivery_book_id"]
    # Batch preannotation projects live beside their original textbook pages.
    # Keep the default generated images/JSON there as well, while still using
    # a dedicated book-ID folder so source scans never confuse delivery crops
    # with new textbook pages.
    book_root = Path(str(project.get("book_root") or project.get("image_root") or "")).expanduser()
    if book_root.is_absolute() and book_root.is_dir():
        return book_root.resolve() / project["delivery_book_id"]
    return Path(runtime_root).resolve() / "delivery_packages" / project["delivery_book_id"]


def _source_signature(project: dict, page: dict, record: dict, answer: bool) -> str:
    source = Path(data.source_path(project, page, answer)).resolve()
    stat = source.stat()
    payload = {
        "source": os.path.normcase(str(source)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "bbox": record.get("bbox", []),
        "segments": record.get("shared_context_segments", []),
        "shared": record.get("shared_context_bbox", []),
        "use_shared": bool(record.get("use_shared_context")),
        "continuation": record.get("continuation_of", ""),
    }
    if not answer:
        # Bump whenever the composed question-image pixel layout changes so
        # an incremental sync rewrites already-materialized crops.
        payload["composition_version"] = "edge_to_edge_v1"
        continuation_sources = []
        for _index, continuation_page, continuation in data.all_questions(project):
            if (continuation.get("region_role") != "continuation"
                    or continuation.get("continuation_of") != record.get("id")):
                continue
            continuation_path = Path(data.source_path(project, continuation_page)).resolve()
            continuation_stat = continuation_path.stat()
            continuation_sources.append({
                "source": os.path.normcase(str(continuation_path)),
                "size": continuation_stat.st_size,
                "mtime_ns": continuation_stat.st_mtime_ns,
                "bbox": continuation.get("bbox", []),
            })
        payload["continuation_sources"] = continuation_sources
    else:
        # 跨版本复用靠签名比对；答案裁图算法日后若变，必须升这个版本号，
        # 否则会一直复用旧版图片。首次启用会让历史版本重裁一次，之后稳定。
        payload["composition_version"] = "answer_crop_v1"
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _source(page_index: int, page: dict, record: dict) -> dict:
    return {
        "page_index": page_index,
        "page_image": str(page.get("image", "")),
        "annotation_id": str(record.get("id", "")),
        "bbox": list(record.get("bbox", [])),
    }


def _ocr(record: dict) -> dict:
    value = record.get("ocr", {})
    if not isinstance(value, dict):
        value = {}
    return {
        "status": str(value.get("status", "not_run")),
        "provider": str(value.get("provider", "")),
        "engine": str(value.get("engine", "")),
        "model": str(value.get("model", "")),
        "confidence": float(value.get("confidence", 0.0) or 0.0),
        "updated_at": str(value.get("updated_at", "")),
    }


def _record_document(project: dict, page_index: int, page: dict, record: dict,
                     kind: str, relations: dict[str, dict]) -> dict:
    public_id = record["delivery_id"]
    page_folder = f"P{page_index + 1:04d}"
    relation = relations.get(str(record.get("id", "")), {})
    document = {
        "schema_version": 2,
        "id": public_id,
        "kind": kind,
        "book_id": project["delivery_book_id"],
        "review_status": str(page.get("status", "unlabeled")),
        "number_path": data.number_path(record),
        "display_number": data.display_number(record),
        # The editable OCR field is the authoritative delivery text. Keep the
        # local project source (including LaTeX where useful for re-rendering),
        # but never leak that source representation into a generated JSON.
        "text": _unicode_math_text(record.get("text", "")),
        "question_type": str(record.get("question_type", "")),
        "ocr": _ocr(record),
        "image_file": f"images/{kind}s/{page_folder}/{public_id}.jpg",
        "source": _source(page_index, page, record),
    }
    if kind == "question":
        matched = (
            str(relation.get("status", "")) == "accepted"
            and bool(relation.get("public_answer_ids", []))
        )
        document.update({
            "relation_id": str(relation.get("delivery_id", "")) if matched else "",
            "relation_status": "accepted" if matched else "pending_generation",
            "answer_status": "matched" if matched else "pending_generation",
            "answer_ids": list(relation.get("public_answer_ids", [])) if matched else [],
        })
    else:
        document["question_ids"] = list(relation.get("public_question_ids", []))
    return document


def sync_delivery_workspace(project_path: str | Path, runtime_root: str | Path,
                            *, changed_kind: str | None = None,
                            changed_page_index: int | None = None,
                            materialize_all: bool = False,
                            changed_record_ids: list[str] | None = None,
                            identity_path: str | Path | None = None,
                            output_root: str | Path | None = None,
                            allowed_pages: set[tuple[str, int]] | None = None,
                            rejected_pages: set[tuple[str, int]] | None = None,
                            only_reviewed: bool = False,
                            reuse_root: str | Path | None = None) -> dict:
    """Sync edited delivery artifacts and rebuild the small authoritative indexes.

    Crops are rewritten only when their source signature changes. JSON is cheap and is
    refreshed for the changed page plus records touched by relations.

    ``only_reviewed=False``（默认）按字段判定：题目字段齐全就收录，审核没做完
    不影响出包（抽检流程）。``True`` 时只收 ``allowed_pages`` 里已放行的页。
    ``rejected_pages``（审核退回/需精修）两种模式都排除。

    ``output_root`` overrides where the package is written.  The incremental
    "ready samples" view of a book keeps its own root so that un-reviewed work
    can never overwrite a published delivery version.

    ``allowed_pages`` limits the package to pages already approved in review, so a
    book can be generated before the whole book has been signed off.

    ``reuse_root`` points at the previously published immutable version.  A brand
    new version directory has no ``manifest.json`` of its own, so without it every
    crop would be re-decoded and re-encoded on each submission.  Matching source
    signatures are reused via hard links; only genuinely changed boxes are cropped.
    """
    project_file = Path(project_path).resolve()
    project = data.load_project(project_file)
    data.ensure_delivery_ids(project)
    root = (Path(output_root).resolve() if output_root is not None
            else delivery_root(project, runtime_root))
    root.mkdir(parents=True, exist_ok=True)
    old_manifest_path = root / "manifest.json"
    try:
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        old_manifest = {}
    old_items = {item.get("id"): item for item in old_manifest.get("items", []) if isinstance(item, dict)}
    reuse_path = Path(reuse_root).resolve() if reuse_root is not None else None
    # 只读上一版的清单做签名基线，不把它并进 old_items —— 那些路径指向旧目录，
    # 混进删除逻辑会误删新版本刚复用过来的文件。
    reuse_items = _load_reuse_items(reuse_path)
    targeted_internal_ids = {str(value) for value in (changed_record_ids or []) if str(value)}

    question_rows = sorted(
        [row for row in data.all_questions(project) if row[2].get("region_role", "question") == "question"],
        key=_record_position,
    )
    answer_rows = sorted(
        [row for row in data.all_answers(project)
         if row[2].get("region_role", "question") in {"question", "continuation"}],
        key=_record_position,
    )
    questions_by_internal = {record["id"]: (index, page, record) for index, page, record in question_rows}
    answers_by_internal = {record["id"]: (index, page, record) for index, page, record in answer_rows}

    question_relations: dict[str, dict] = {}
    answer_relations: dict[str, dict] = {}
    for pair in project.get("pairs", []):
        if str(pair.get("status", "")) != "accepted" or not pair.get("answer_ids"):
            continue
        question = questions_by_internal.get(str(pair.get("question_id", "")))
        answers = [answers_by_internal.get(str(value)) for value in pair.get("answer_ids", [])]
        if (
            not question
            or str(question[1].get("status", "")) != "accepted"
            or not answers
            or any(row is None or str(row[1].get("status", "")) != "accepted" for row in answers)
        ):
            continue
        public_answers = [row[2]["delivery_id"] for row in answers if row]
        question_relations[question[2]["id"]] = {**pair, "public_answer_ids": public_answers}
        for answer in (row for row in answers if row):
            relation = answer_relations.setdefault(answer[2]["id"], {"public_question_ids": []})
            relation["public_question_ids"].append(question[2]["delivery_id"])

    selected_kind = "answer" if changed_kind == "answer" else "question"
    changed_ids: set[str] = set()
    if changed_page_index is not None:
        rows = answer_rows if selected_kind == "answer" else question_rows
        changed_ids = {
            record["delivery_id"] for index, _page, record in rows
            if index == int(changed_page_index)
        }

    documents: dict[str, dict] = {}
    manifest_items = []
    crop_updates = 0
    crop_reused = 0
    for kind, rows, relations in (
        ("question", question_rows, question_relations),
        ("answer", answer_rows, answer_relations),
    ):
        if changed_page_index is not None and kind == selected_kind:
            changed_folder = f"P{int(changed_page_index) + 1:04d}"
            (root / "images" / f"{kind}s" / changed_folder).mkdir(parents=True, exist_ok=True)
            (root / "json" / f"{kind}s" / changed_folder).mkdir(parents=True, exist_ok=True)
        for page_index, page, record in rows:
            public_id = record["delivery_id"]
            page_folder = f"P{page_index + 1:04d}"
            signature = _source_signature(project, page, record, kind == "answer")
            image_path = root / "images" / f"{kind}s" / page_folder / f"{public_id}.jpg"
            json_path = root / "json" / f"{kind}s" / page_folder / f"{public_id}.json"
            old = old_items.get(public_id, {})
            current_image_file = f"images/{kind}s/{page_folder}/{public_id}.jpg"
            current_json_file = f"json/{kind}s/{page_folder}/{public_id}.json"
            for old_key, current_file in (("image_file", current_image_file), ("json_file", current_json_file)):
                old_file = str(old.get(old_key, ""))
                if old_file and old_file != current_file:
                    old_target = _delivery_artifact_path(root, old_file)
                    if old_target is not None and old_target.is_file():
                        old_target.unlink()
            on_changed_page = (
                changed_page_index is not None
                and kind == selected_kind and page_index == int(changed_page_index)
            )
            explicitly_changed = str(record.get("id", "")) in targeted_internal_ids
            should_materialize = (
                (materialize_all and str(page.get("status", "")) == "accepted")
                or on_changed_page
                or explicitly_changed
                or image_path.is_file()
            )
            reused = reuse_items.get(public_id, {})
            baseline = old if old else reused
            # 签名一致说明像素完全一致：把上一版的文件硬链接进新版本目录，
            # 后续的 is_file() 判断就会跳过整页解码 + JPEG 编码。
            if (should_materialize and not image_path.is_file()
                    and reused.get("source_signature") == signature):
                if _link_reused_file(
                        _reuse_artifact(reuse_path, reused.get("image_file")), image_path):
                    crop_reused += 1
            if should_materialize and (baseline.get("source_signature") != signature or not image_path.is_file()):
                crop = (_crop(data.source_path(project, page, True), record["bbox"])
                        if kind == "answer" else _question_crop(project, page, record))
                _atomic_image(image_path, crop)
                crop_updates += 1
            document = _record_document(project, page_index, page, record, kind, relations)
            documents[public_id] = document
            should_write_json = (
                materialize_all or on_changed_page or explicitly_changed
                or public_id in changed_ids
            )
            if should_write_json:
                _atomic_json(json_path, document)
            manifest_items.append({
                "id": public_id, "kind": kind, "page_index": page_index,
                "review_status": document["review_status"],
                "json_file": current_json_file,
                "image_file": document["image_file"], "source_signature": signature,
                "materialized": bool(image_path.is_file() and json_path.is_file()),
            })

    current_ids = set(documents)
    removed = 0
    for stale_id, item in old_items.items():
        if stale_id in current_ids:
            continue
        for key in ("json_file", "image_file"):
            target = _delivery_artifact_path(root, item.get(key, ""))
            if target is not None and target.is_file():
                target.unlink()
        removed += 1

    ready_questions = []
    ready_answer_ids: set[str] = set()
    for _page_index, page, question in question_rows:
        question_document = documents[question["delivery_id"]]
        if question_document["review_status"] != "accepted":
            continue
        ready_questions.append(question_document)
        if question_document["answer_status"] != "matched":
            continue
        relation = question_relations.get(question["id"], {})
        answer_rows_for_pair = [answers_by_internal.get(str(value)) for value in relation.get("answer_ids", [])]
        if any(row is None or str(row[1].get("status", "")) != "accepted" for row in answer_rows_for_pair):
            question_document.update({
                "relation_id": "", "relation_status": "pending_generation",
                "answer_status": "pending_generation", "answer_ids": [],
            })
            continue
        ready_answer_ids.update(question_document["answer_ids"])
    ready_answers = [documents[value] for value in sorted(ready_answer_ids) if value in documents]
    database = {
        "schema_version": 2,
        "kind": "papercutter_database_delivery",
        "book": {
            "id": project["delivery_book_id"],
            "source_root": str(project.get("book_root", project.get("image_root", ""))),
            "isbn": str(project.get("isbn", "")),
            "subject": str(project.get("subject", "")),
            "grade": str(project.get("grade", "")),
        },
        "questions": ready_questions,
        "answers": ready_answers,
    }
    _atomic_json(root / "database_import.json", database)
    manifest = {
        "schema_version": 2,
        "book_id": project["delivery_book_id"],
        "project_path": str(project_file),
        "question_count": len(question_rows),
        "answer_count": len(answer_rows),
        "ready_question_count": len(ready_questions),
        "items": manifest_items,
        "database_file": "database_import.json",
    }
    _atomic_json(old_manifest_path, manifest)
    k12 = _write_k12_delivery(
        root, project, question_rows, answer_rows, question_relations,
        changed_page_index=(changed_page_index if selected_kind == "question" else None),
        materialize_all=materialize_all,
        # 身份库跟随"交付包根目录"共享：服务器上所有书都落在同一个交付根，
        # 跨批次去重因此覆盖全站；本机每个交付根各自独立，不会串书。
        # 服务器每个版本写在自己的 staging 目录下，所以必须由调用方显式传入
        # 交付根级路径，否则会变成"每个版本一份空库"、跨批次去重形同虚设。
        identity_path=Path(identity_path) if identity_path is not None
        else root.parent / "delivery_identity.sqlite3",
        allowed_pages=(set(allowed_pages) if allowed_pages is not None else None),
        rejected_pages=(set(rejected_pages) if rejected_pages is not None else None),
        only_reviewed=bool(only_reviewed),
        reuse_root=reuse_path,
    )
    # Delivery generation may assign missing public IDs, but it is part of the
    # caller's logical mutation and must not advance the renderer CAS revision
    # a second time.
    data.save_project(project, project_file, increment_revision=False)
    return {
        "output_path": str(root), "book_id": project["delivery_book_id"],
        "question_count": len(question_rows), "answer_count": len(answer_rows),
        "ready_question_count": len(ready_questions), "crop_updates": crop_updates,
        "crop_reused": crop_reused, "removed_count": removed, "k12": k12,
    }
