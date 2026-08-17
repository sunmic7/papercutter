"""Conservative VLM assistance for the production question cutter.

The vision model is deliberately *not* a second cutter.  OCR plus the
existing numbering and answer-pairing rules remain authoritative.  This
module supplies only three guarded signals:

* page type (to reject answer cards/answer keys and non-question pages),
* primary-question groups (for traceability, never for creating cuts), and
* answer-card regions (to suppress a rule cut that is almost entirely inside
  such a region).

It is opt-in through ``PAPERCUTTER_VLM_MODE=assist`` (or the main CLI flag)
and fails open: an unavailable API never changes the local pipeline.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"
MAX_COORDINATE = 1000.0
SKIP_PAGE_TYPES = {"answer_key", "answer_sheet", "cover", "catalog", "instruction"}
EXCLUSION_TYPES = {"answer_key", "answer_sheet", "answer_card"}

ANSWER_PROMPT = """You locate entries on Chinese reference-answer pages. Return JSON only.
If this is not an answer key, return {\"page_type\":\"not_answer\",\"entries\":[]}.
Otherwise return page_type=answer_key and one entry for each independently numbered answer.
Each entry has id, major_number (Chinese-section number as an integer, or 0), question_number, and bbox.
The bbox must include the number and its complete answer, including continuation text, but must not include headings or unrelated entries.
CRITICAL: one visual text row can contain several entries, such as "1. ...  2. ...".
Return a separate tight bbox for EACH printed number in left-to-right order; a bbox must never cross into the next printed number.
The question_number must be visibly printed inside that bbox, not inferred from the sequence.
Do not return page headings, answer-card blanks, a lone number, or explanatory paragraphs as entries.
bbox is normalized [x1,y1,x2,y2] in 0..1000."""

ANSWER_SECTION_PROMPT = """You locate MAJOR answer sections on a Chinese reference-answer page.
Return JSON only. If this is not an answer key, return
{\"page_type\":\"not_answer\",\"sections\":[]}. Otherwise return
{\"page_type\":\"answer_key\",\"sections\":[...]}.
Each section has id, major_number, and bbox [x0,y0,x1,y1] in 0..1000 normalized
coordinates. A section starts at a printed Chinese major label, for example
\"一、\", \"二、\", or \"（一）\", and covers all answer lines through the next major
label. One visual row can contain several minor answers: keep them together in
their major section rather than making minor boxes. Do not return page titles,
page numbers, answer-card blanks, explanatory paragraphs, or unclear sections."""

DIRECT_CUT_PROMPT = """You are the primary visual judge for cutting a Chinese exercise page.
Think silently and inspect the complete page before answering. Return JSON only:
{"page_type":"question|not_question","items":[...]}. Each item has
major_number (the printed Chinese major-section number, or 0), question_number
(the printed minor question number, or 0), bbox [x0,y0,x1,y1] normalized to
0..1000, and confidence 0..1. An item must be one answerable question: include
its shared stem, material, diagram and all dependent subparts, but never emit a
standalone paragraph, option, blank, serial number, answer-card area, or an
overlapping duplicate. If several numbered prompts share one stem but require
separate answers, make one item per prompt and repeat the shared stem in every
bbox. Before finalizing, check that every returned bbox contains its own printed
number and that no two boxes overlap substantially."""

MAJOR_CUT_PROMPT = """You are the primary visual judge for cutting a Chinese exercise page into its PRINTED MAJOR QUESTIONS.
Think silently and inspect the complete page before answering. Return JSON only:
{"page_type":"question|not_question","items":[...]}.

Each item has major_number (the printed Chinese major-section number as an
integer), question_number (repeat the same positive integer as major_number;
it is an internal validation field, not a minor question), bbox [x0,y0,x1,y1] normalized to 0..1000,
and confidence 0..1. A major-question box begins at its red/printed major
heading such as "一、" or "七、", and ends immediately before the next major
heading. It MUST keep the complete question group together: all small numbered
subquestions, shared material/reading passage, diagrams, options, answer lines,
and writing space belonging to that major question. Do NOT split it into its
small questions or paragraphs. Do NOT return a title, score table, page number,
answer card, standalone paragraph, or duplicate.

For a major question continued from the previous page, return its visible
continuation as one item with that same major_number even if this page portion
has no repeated heading. For a major question continued onto the next page,
return its visible portion through the page edge. The caller will stitch the
same-number portions together. Check that every printed major heading has one
box, except a continuation which may have no heading."""

DIRECT_REVIEW_PROMPT = """You are reviewing a proposed atomic-question cut list for this Chinese exercise page.
Think silently, re-read the entire image, and check every major question and minor number.
The proposal is supplied below. Return JSON only as {"page_type":"question|not_question","items":[...]} with the COMPLETE corrected list, not only changes. Each item must have major_number, question_number, bbox [x0,y0,x1,y1] normalized 0..1000, and confidence. Add every omitted answerable question; remove standalone paragraphs/options, duplicates, and boxes that cross into another question. Preserve a proposed box only when it is visibly correct. A numbered major section with no returned item is a likely omission and must be checked carefully."""

PROMPT = """你是教材切题系统中的保守语义审核器，不负责直接切出小题。

先判断整页类型：question（有待练习题）、answer_key（参考答案/解析）、answer_sheet（答题卡、填涂区、作答区）、cover、catalog、instruction、unknown。

只有 page_type=question 时才可返回 groups。每个 group 必须是一个可进入原有 OCR 题号规则和题目—答案配对流程的“一级题组”：
1. 同一题干、材料、图表及其连续子问只能是一个 group；绝不可把题号、选项、填空横线、段落、单个子问单独作为 group。
2. group 的 bbox 要包住完整题组；不能确定边界时宁可不返回该 group。
3. 答题卡、答案页、参考答案、目录、封面、版权页都不是题组。即使其中有大量数字，也不能返回 groups。
4. 如果题目页中夹有答题卡/作答区，在 ignore_regions 中标出该区域；不要把它当题目。

另外返回 paper_title：只有当本页是一本独立试卷/检测卷的首页时，填写图片中可见的完整卷名（如“第一单元检测卷”）；普通续页、课文页、答题卡、答案页返回空字符串。不要根据页内普通章节标题猜测卷名。

坐标相对整页归一化为 0~1000，bbox 为 [x1,y1,x2,y2]。只返回 JSON，不要解释。"""


def enabled() -> bool:
    return os.environ.get("PAPERCUTTER_VLM_MODE", "off").strip().lower() == "assist"


def thinking_enabled() -> bool:
    return os.environ.get("PAPERCUTTER_VLM_THINKING", "off").strip().lower() == "on"


def _apply_thinking_options(data: dict[str, Any]) -> None:
    """Enable Qwen3-VL reasoning mode without asking it to expose chain of thought.

    DashScope's compatible API does not allow JSON response_format together
    with thinking mode, so callers retain the JSON-only prompt and parse the
    final message content defensively instead.
    """
    if not thinking_enabled():
        return
    data.pop("response_format", None)
    data["enable_thinking"] = True
    budget = os.environ.get("PAPERCUTTER_VLM_THINKING_BUDGET", "")
    if budget.isdigit() and int(budget) > 0:
        data["thinking_budget"] = int(budget)


def _box(raw: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= MAX_COORDINATE for value in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    result = [
        max(0, min(width - 1, int(math.floor(x0 / MAX_COORDINATE * width)))),
        max(0, min(height - 1, int(math.floor(y0 / MAX_COORDINATE * height)))),
        max(1, min(width, int(math.ceil(x1 / MAX_COORDINATE * width)))),
        max(1, min(height, int(math.ceil(y1 / MAX_COORDINATE * height)))),
    ]
    return result if result[2] > result[0] and result[3] > result[1] else None


def _json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("VLM response has no JSON object")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response has no JSON object")
    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("VLM response root is not an object")
    return parsed


def _content(response: dict[str, Any]) -> Any:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("VLM response has no choices")
    message = choices[0].get("message", {})
    return message.get("content")


def _prepare_image(image: np.ndarray, max_long_side: int) -> tuple[str, tuple[int, int]]:
    if image is None or image.size == 0:
        raise ValueError("empty page image")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_long_side) / max(width, height))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("cannot encode page image")
    uri = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return uri, (width, height)


def _response_schema() -> dict[str, Any]:
    box = {
        "type": "array", "minItems": 4, "maxItems": 4,
        "items": {"type": "number", "minimum": 0, "maximum": 1000},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["page_type", "page_confidence", "groups", "ignore_regions"],
        "properties": {
            "page_type": {
                "type": "string",
                "enum": ["question", "answer_key", "answer_sheet", "cover", "catalog", "instruction", "unknown"],
            },
            "page_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "groups": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "bbox", "primary_numbers"],
                    "properties": {
                        "id": {"type": "string"}, "bbox": box,
                        "primary_numbers": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 999}},
                    },
                },
            },
            "ignore_regions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "type", "bbox"],
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string", "enum": ["answer_key", "answer_sheet", "answer_card"]},
                        "bbox": box,
                    },
                },
            },
        },
    }


def _request(base_url: str, api_key: str, data: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"VLM HTTP {error.code}: {body}") from error


def normalize_payload(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    page_type = str(payload.get("page_type", "unknown")).strip().lower()
    allowed_types = SKIP_PAGE_TYPES | {"question", "unknown"}
    if page_type not in allowed_types:
        page_type = "unknown"
    confidence_raw = payload.get("page_confidence")
    try:
        confidence = min(1.0, max(0.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = None

    groups = []
    for position, item in enumerate(payload.get("groups", []) if isinstance(payload.get("groups"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        box = _box(item.get("bbox"), width, height)
        if not box:
            continue
        raw_numbers = item.get("primary_numbers", [])
        numbers = [int(number) for number in raw_numbers if isinstance(number, int) and 0 < number < 1000]
        groups.append({
            "id": str(item.get("id") or f"group_{position}"),
            "bbox": box,
            "primary_numbers": numbers,
        })

    ignored = []
    for position, item in enumerate(payload.get("ignore_regions", []) if isinstance(payload.get("ignore_regions"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).strip().lower()
        box = _box(item.get("bbox"), width, height)
        if kind in EXCLUSION_TYPES and box:
            ignored.append({"id": str(item.get("id") or f"ignore_{position}"), "type": kind, "bbox": box})
    # Qwen's json_object mode occasionally omits the optional confidence.  A
    # non-question label with no returned question group is still a much
    # stronger signal than a dense answer key's incidental numbering.  All
    # other missing confidences remain deliberately below the skip threshold.
    if confidence is None:
        confidence = 0.95 if page_type in SKIP_PAGE_TYPES and not groups else 0.50
    paper_title = str(payload.get("paper_title", "")).strip()
    if len(paper_title) > 100:
        paper_title = ""
    return {
        "page_type": page_type,
        "page_confidence": confidence,
        "groups": groups,
        "ignore_regions": ignored,
        "paper_title": paper_title,
    }


def normalize_answer_payload(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Keep only numbered, geometrically valid VLM answer entries."""
    if str(payload.get("page_type", "not_answer")).strip().lower() != "answer_key":
        return {"page_type": "not_answer", "entries": []}
    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raw_entries = []
    entries = []
    seen: set[tuple[int, int, tuple[int, int, int, int]]] = set()
    for position, item in enumerate(raw_entries, start=1):
        if not isinstance(item, dict):
            continue
        box = _box(item.get("bbox"), width, height)
        try:
            major_number = int(item.get("major_number", 0))
            question_number = int(item.get("question_number"))
        except (TypeError, ValueError):
            continue
        if not box or not 0 <= major_number < 100 or not 0 < question_number < 1000:
            continue
        key = (major_number, question_number, tuple(box))
        if key in seen:
            continue
        seen.add(key)
        entries.append({
            "id": str(item.get("id") or f"answer_{position}"),
            "major_number": major_number,
            "question_number": question_number,
            "bbox": box,
        })
    return {"page_type": "answer_key", "entries": entries}


def normalize_answer_section_payload(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Keep only valid coarse major-answer regions; OCR still cuts minor answers."""
    if str(payload.get("page_type", "not_answer")).strip().lower() != "answer_key":
        return {"page_type": "not_answer", "sections": []}
    raw_sections = payload.get("sections", [])
    if not isinstance(raw_sections, list):
        raw_sections = []
    sections = []
    seen: set[tuple[int, tuple[int, int, int, int]]] = set()
    for position, item in enumerate(raw_sections, start=1):
        if not isinstance(item, dict):
            continue
        box = _box(item.get("bbox"), width, height)
        try:
            major_number = int(item.get("major_number"))
        except (TypeError, ValueError):
            continue
        if not box or not 1 <= major_number < 100:
            continue
        key = (major_number, tuple(box))
        if key in seen:
            continue
        seen.add(key)
        sections.append({
            "id": str(item.get("id") or f"section_{position}"),
            "major_number": major_number,
            "bbox": box,
        })
    return {"page_type": "answer_key", "sections": sections}


def normalize_direct_cut_payload(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    if str(payload.get("page_type", "not_question")).lower() != "question":
        return {"page_type": "not_question", "items": []}
    items = []
    seen = set()
    for index, item in enumerate(payload.get("items", []) if isinstance(payload.get("items"), list) else [], 1):
        if not isinstance(item, dict):
            continue
        box = _box(item.get("bbox"), width, height)
        try:
            major, number = int(item.get("major_number", 0)), int(item.get("question_number", 0))
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if not box or not 0 <= major < 100 or not 0 < number < 1000 or not 0 <= confidence <= 1:
            continue
        key = (major, number, tuple(box))
        if key in seen:
            continue
        seen.add(key)
        items.append({"id": str(item.get("id") or f"item_{index}"), "major_number": major,
                      "question_number": number, "bbox": box, "confidence": confidence})
    return {"page_type": "question", "items": items}


def analyze_page(image: np.ndarray) -> dict[str, Any] | None:
    """Ask the VLM once for conservative page-level assistance.

    Returns ``None`` for disabled/misconfigured/failed calls.  No exception is
    allowed to stop the local OCR and answer matching pipeline.
    """
    if not enabled():
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        image_uri, (width, height) = _prepare_image(
            image, int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200")),
        )
        data = {
            "model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL),
            "temperature": 0,
            # json_schema made this model over-segment answer keys in live
            # testing.  json_object plus the conservative gates below is both
            # faster and safer; malformed fields are ignored individually.
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]}],
        }
        _apply_thinking_options(data)
        started = time.perf_counter()
        response = _request(
            os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL), api_key, data,
            max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120"))),
        )
        result = normalize_payload(_json_object(_content(response)), width, height)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return None


def analyze_direct_cuts(image: np.ndarray) -> dict[str, Any] | None:
    """Return model-judged atomic question boxes for the direct-cut experiment."""
    if not enabled():
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        image_uri, (width, height) = _prepare_image(image, int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200")))
        data = {"model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL), "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": DIRECT_CUT_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ]}]}
        _apply_thinking_options(data)
        started = time.perf_counter()
        response = _request(os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL), api_key, data,
                            max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120"))))
        result = normalize_direct_cut_payload(_json_object(_content(response)), width, height)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return None


def analyze_major_cuts(image: np.ndarray) -> dict[str, Any] | None:
    """Return model-judged printed major-question boxes for a page."""
    if not enabled():
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        image_uri, (width, height) = _prepare_image(image, int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200")))
        data = {"model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL), "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": MAJOR_CUT_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ]}]}
        _apply_thinking_options(data)
        started = time.perf_counter()
        response = _request(os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL), api_key, data,
                            max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120"))))
        result = normalize_direct_cut_payload(_json_object(_content(response)), width, height)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return None


def review_direct_cuts(image: np.ndarray, proposed_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Second thinking pass: return a complete corrected direct-cut list."""
    if not enabled():
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        image_uri, (width, height) = _prepare_image(image, int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200")))
        compact = []
        for item in proposed_items or []:
            box = item.get("bbox", [])
            if not isinstance(box, list) or len(box) != 4:
                continue
            compact.append({
                "major_number": item.get("major_number", 0),
                "question_number": item.get("question_number", 0),
                "bbox": [round(box[0] / width * 1000), round(box[1] / height * 1000),
                         round(box[2] / width * 1000), round(box[3] / height * 1000)],
            })
        prompt = DIRECT_REVIEW_PROMPT + "\nPROPOSED_ITEMS=" + json.dumps(compact, ensure_ascii=False)
        data = {"model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL), "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ]}]}
        _apply_thinking_options(data)
        started = time.perf_counter()
        response = _request(os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL), api_key, data,
                            max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120"))))
        result = normalize_direct_cut_payload(_json_object(_content(response)), width, height)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return None


def analyze_answer_page(image: np.ndarray) -> dict[str, Any] | None:
    """Locate answer entries for OCR confirmation; never create answer text."""
    if not enabled():
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        image_uri, (width, height) = _prepare_image(
            image, int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200")),
        )
        data = {
            "model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": ANSWER_PROMPT},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]}],
        }
        _apply_thinking_options(data)
        started = time.perf_counter()
        response = _request(
            os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL), api_key, data,
            max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120"))),
        )
        result = normalize_answer_payload(_json_object(_content(response)), width, height)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return None


def analyze_answer_sections(image: np.ndarray) -> dict[str, Any] | None:
    """Locate coarse major-answer regions; never use a VLM box as a final crop."""
    if not enabled():
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        image_uri, (width, height) = _prepare_image(
            image, int(os.environ.get("PAPERCUTTER_VLM_MAX_SIDE", "2200")),
        )
        data = {
            "model": os.environ.get("PAPERCUTTER_VLM_MODEL", DEFAULT_MODEL),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": ANSWER_SECTION_PROMPT},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]}],
        }
        _apply_thinking_options(data)
        started = time.perf_counter()
        response = _request(
            os.environ.get("PAPERCUTTER_VLM_BASE_URL", DEFAULT_BASE_URL), api_key, data,
            max(1, int(os.environ.get("PAPERCUTTER_VLM_TIMEOUT", "120"))),
        )
        result = normalize_answer_section_payload(_json_object(_content(response)), width, height)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError):
        return None


def should_skip_page(analysis: dict[str, Any] | None, local_page_kind: str) -> bool:
    """Only let VLM reject a page when it is strongly and unambiguously non-question."""
    if not analysis or analysis.get("page_type") not in SKIP_PAGE_TYPES:
        return False
    if analysis.get("groups"):
        return False
    confidence = float(analysis.get("page_confidence", 0.0))
    # Existing OCR classification wins unless the VLM is highly certain.  This
    # makes an accidental label unable to silently delete an exercise page.
    return local_page_kind != "question" or confidence >= 0.93


def _intersection_area(left: list[int], right: list[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))


def apply_to_rule_cuts(minor_cuts: list[dict[str, Any]], analysis: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Annotate rule cuts and suppress only cuts proven to sit in an answer card."""
    stats = {"group_confirmed": 0, "outside_groups": 0, "answer_card_suppressed": 0}
    if not analysis:
        return minor_cuts, stats
    groups = analysis.get("groups", [])
    ignored = analysis.get("ignore_regions", [])
    retained = []
    for cut in minor_cuts:
        box = cut.get("_source_box")
        if not isinstance(box, list) or len(box) != 4:
            retained.append(cut)
            continue
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        blocked = any(_intersection_area(box, region["bbox"]) / area >= 0.88 for region in ignored)
        if blocked:
            stats["answer_card_suppressed"] += 1
            continue
        overlaps = [group for group in groups if _intersection_area(box, group["bbox"]) / area >= 0.20]
        if overlaps:
            cut["_vlm_group_id"] = overlaps[0]["id"]
            stats["group_confirmed"] += 1
        elif groups:
            cut["_vlm_outside_group"] = True
            stats["outside_groups"] += 1
        retained.append(cut)
    return retained, stats
