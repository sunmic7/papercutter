#!/usr/bin/env python3
"""On-demand local annotation and answer-matching workstation.

Unlike production cutting this program never pre-scans a whole book.  A page
is analysed only after the reviewer explicitly requests its pre-annotation,
and every proposed box remains editable before it is used for matching or
exported for training.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QProcess, QRectF, QTimer, Signal, Qt
from PySide6.QtGui import QAction, QColor, QCursor, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QLineEdit, QSplitter, QStatusBar, QTabWidget, QToolBar,
    QVBoxLayout, QWidget,
)

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from cut_questions_gui_qt import APP_STYLE, configure_cjk_font  # noqa: E402
from review_tools import manual_annotation_data as data  # noqa: E402
from review_tools.export_manual_annotations import export_project  # noqa: E402
import cut_questions_v4 as core  # noqa: E402


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

STATE_FILE = APP_DIR / "runtime" / "annotation_workspace_state.json"


def _load_workspace_state() -> dict:
    try:
        if STATE_FILE.is_file():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _save_workspace_state(**values) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = _load_workspace_state()
        state.update(values)
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    except OSError:
        pass


class ReviewBox(QGraphicsRectItem):
    """A movable detection box in original image coordinates."""
    def __init__(self, record: dict, color: QColor) -> None:
        # Stored boxes use xyxy: [left, top, right, bottom].  QRectF expects
        # xywh, so passing the four values through made every newly drawn box
        # expand by its own right/bottom coordinates after mouse release.
        x0, y0, x1, y1 = (float(value) for value in record["box"])
        super().__init__(QRectF(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)))
        self.record = record
        self.setPen(QPen(color, 3))
        self.setBrush(QColor(color.red(), color.green(), color.blue(), 24))
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setZValue(3)
        self.label = QGraphicsSimpleTextItem(str(record.get("number") or "?"), self)
        self.label.setBrush(QColor("#8f3a00"))
        # The label is a child item: use coordinates relative to this box.
        # Keeping absolute coordinates doubled the transform work on large scans.
        self.label.setPos(5, 3)
        self.label.setZValue(4)

    def current_box(self) -> list[int]:
        # sceneBoundingRect includes the 3px pen.  Persist the geometry of the
        # actual rectangle instead, otherwise each save/reopen grows a box.
        rect = self.mapRectToScene(self.rect())
        return [round(rect.left()), round(rect.top()), round(rect.right()), round(rect.bottom())]


class ReviewCanvas(QGraphicsView):
    box_drawn = Signal(object)
    boxes_delete_requested = Signal(object)
    selection_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene_ref = QGraphicsScene(self)
        self.setScene(self.scene_ref)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor("#eef2f6"))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.draw_enabled = False
        self.delete_enabled = False
        self.draw_start = None
        self.draft: QGraphicsRectItem | None = None
        self.delete_start = None
        self.delete_draft: QGraphicsRectItem | None = None
        self.pan_last = None
        self.pen_cursor = self._create_pen_cursor()
        self.delete_cursor = self._create_delete_cursor()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.scene_ref.selectionChanged.connect(self._emit_selection)

    @staticmethod
    def _create_pen_cursor() -> QCursor:
        """A compact pen cursor whose hotspot is the nib, not its centre."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # The nib is deliberately at upper-left: it is the coordinate used
        # when the reviewer starts a new rectangle.
        painter.setPen(QPen(QColor("#19304d"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(24, 25, 8, 9)
        painter.setPen(QPen(QColor("#00a99d"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(23, 24, 10, 11)
        painter.setPen(QPen(QColor("#1f2933"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(9, 10, 4, 5)
        painter.end()
        return QCursor(pixmap, 4, 5)

    @staticmethod
    def _create_delete_cursor() -> QCursor:
        pixmap = QPixmap(28, 28)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#c92a2a"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(6, 6, 22, 22)
        painter.drawLine(22, 6, 6, 22)
        painter.end()
        return QCursor(pixmap, 14, 14)

    def set_draw_enabled(self, enabled: bool) -> None:
        self.draw_enabled = enabled
        if enabled:
            self.delete_enabled = False
        self.setCursor(self.pen_cursor if enabled else (self.delete_cursor if self.delete_enabled else Qt.CursorShape.ArrowCursor))

    def set_delete_enabled(self, enabled: bool) -> None:
        self.delete_enabled = enabled
        if enabled:
            self.draw_enabled = False
        self.setCursor(self.delete_cursor if enabled else (self.pen_cursor if self.draw_enabled else Qt.CursorShape.ArrowCursor))

    @staticmethod
    def _review_box_at(item) -> ReviewBox | None:
        while item is not None:
            if isinstance(item, ReviewBox):
                return item
            item = item.parentItem()
        return None

    def load_page(self, image_path: str) -> bool:
        reader = QImageReader(image_path)
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return False
        pixmap = QPixmap.fromImage(image)
        self.scene_ref.clear()
        self.pixmap_item = self.scene_ref.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene_ref.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self.fitInView(self.scene_ref.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return True

    def add_box(self, record: dict, color: QColor | None = None) -> ReviewBox:
        item = ReviewBox(record, color or QColor("#ff7a00"))
        self.scene_ref.addItem(item)
        return item

    def boxes(self) -> list[ReviewBox]:
        return [item for item in self.scene_ref.items() if isinstance(item, ReviewBox)]

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self.pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            event.accept()
            return
        if self.delete_enabled and event.button() == Qt.MouseButton.LeftButton:
            existing = self._review_box_at(self.itemAt(event.position().toPoint()))
            if existing is not None:
                self.boxes_delete_requested.emit([existing])
                event.accept()
                return
            self.delete_start = self.mapToScene(event.position().toPoint())
            self.delete_draft = QGraphicsRectItem(QRectF(self.delete_start, self.delete_start))
            self.delete_draft.setPen(QPen(QColor("#d63939"), 2, Qt.PenStyle.DashLine))
            self.delete_draft.setBrush(QColor(214, 57, 57, 35))
            self.delete_draft.setZValue(7)
            self.scene_ref.addItem(self.delete_draft)
            event.accept()
            return
        if self.draw_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.draw_start = self.mapToScene(event.position().toPoint())
            self.draft = QGraphicsRectItem(QRectF(self.draw_start, self.draw_start))
            self.draft.setPen(QColor("#00a99d"))
            self.draft.setBrush(QColor(0, 169, 157, 28))
            self.draft.setZValue(5)
            self.scene_ref.addItem(self.draft)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self.pan_last is not None:
            point = event.position().toPoint()
            delta = point - self.pan_last
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_last = point
            event.accept()
            return
        if self.delete_draft is not None and self.delete_start is not None:
            current = self.mapToScene(event.position().toPoint())
            self.delete_draft.setRect(QRectF(self.delete_start, current).normalized())
            event.accept()
            return
        if self.draft is not None and self.draw_start is not None:
            current = self.mapToScene(event.position().toPoint())
            self.draft.setRect(QRectF(self.draw_start, current).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton and self.pan_last is not None:
            self.pan_last = None
            self.setCursor(self.pen_cursor if self.draw_enabled else (self.delete_cursor if self.delete_enabled else Qt.CursorShape.ArrowCursor))
            event.accept()
            return
        if self.delete_draft is not None and self.delete_start is not None:
            rect = self.delete_draft.rect().normalized()
            self.scene_ref.removeItem(self.delete_draft)
            self.delete_draft = None
            self.delete_start = None
            if rect.width() >= 4 and rect.height() >= 4:
                # Do this from the actual review boxes instead of relying on
                # scene item-shape containment (labels are child items, and
                # that query can miss their parent box after a zoom).
                selected = [
                    box for box in self.boxes()
                    if rect.intersects(box.mapRectToScene(box.rect()))
                ]
                if selected:
                    self.boxes_delete_requested.emit(selected)
            event.accept()
            return
        if self.draft is not None and self.draw_start is not None:
            rect = self.draft.rect().normalized()
            self.scene_ref.removeItem(self.draft)
            self.draft = None
            self.draw_start = None
            if rect.width() >= 8 and rect.height() >= 8:
                self.box_drawn.emit([round(rect.left()), round(rect.top()), round(rect.right()), round(rect.bottom())])
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _emit_selection(self) -> None:
        selected = [item for item in self.scene_ref.selectedItems() if isinstance(item, ReviewBox)]
        self.selection_changed.emit(selected[0] if selected else None)


def _manual_prelabel_detector_enabled() -> bool:
    """Learned-boundary fusion is opt-in until Chinese-book validation passes."""
    return os.environ.get("PAPERCUTTER_MANUAL_PRELABEL_DETECTOR", "").strip().lower() in _TRUE_ENV_VALUES


def _optional_question_detections(cutter, image) -> list:
    if not _manual_prelabel_detector_enabled():
        return []
    try:
        return cutter.detect_question_blocks_with_model(image)
    except Exception as exc:
        print(f"[模型] 手工预标注检测器不可用，继续使用 OCR/规则切题：{exc}")
        return []


def _refine_manual_prelabel_minors(cutter, minors: list[dict], image, detections: list) -> list[dict]:
    """Fuse only one-to-one overlaps; unmatched detector boxes never create cuts."""
    if not detections or not minors:
        return minors
    try:
        refined, _stats = cutter.question_detector.refine_legacy_cuts(
            minors, image, detections, min_iou=cutter.QUESTION_DETECTOR_ASSIST_IOU,
        )
        return refined
    except Exception as exc:
        print(f"[模型] 手工预标注边界融合失败，保留 OCR/规则框：{exc}")
        return minors


def _shared_instruction_box(block: dict, minors: list[dict]) -> list[int]:
    """Return a reusable heading/preamble segment without enlarging every UI box."""
    candidates = [minor.get("_shared_context_box") for minor in minors]
    existing = next((box for box in candidates if isinstance(box, (list, tuple)) and len(box) == 4), None)
    if existing:
        return [int(value) for value in existing]
    if len(minors) < 2 or block.get("title") == "整页":
        return []
    # 横排小问的 _source_box 顶边相同，必须再按 x 排，否则"第一题"是随机的。
    ordered = sorted(minors, key=lambda item: (
        item.get("_source_box", [0, 0, 0, 0])[1],
        item.get("_source_box", [0, 0, 0, 0])[0],
    ))
    first = ordered[0]
    major_box = first.get("_major_box", [])
    anchor_y = int(first.get("_anchor_y", 0) or 0)
    if len(major_box) != 4 or not first.get("number_detected") or anchor_y < 12:
        return []
    x0, y0, x1, y1 = (int(value) for value in major_box)
    context_y1 = min(y1, y0 + max(8, anchor_y - 4))
    return [x0, y0, x1, context_y1] if context_y1 > y0 else []


_EXPLANATORY_PREAMBLE_KEYWORDS = (
    "知识点", "概念", "定义", "性质", "公式", "法则", "运算律",
    "交换律", "结合律", "分配律", "例题", "解法", "方法", "规律",
)


def _trim_explanatory_preamble_from_first_minor(minors: list[dict], ocr_items: list[dict], block_img) -> bool:
    """Exclude a textbook explanation above the first real exercise number.

    A lesson page can place a full-column knowledge card before a normal
    numbered exercise.  That card is reference material, not part of question
    1.  Only trim it when it is both unusually tall and contains multiple
    explanatory signals; reading passages and short shared instructions remain
    attached to their first question as before.
    """
    if block_img is None or getattr(block_img, "size", 0) == 0 or not minors:
        return False
    height = int(block_img.shape[0])
    numbered = sorted(
        (minor for minor in minors if minor.get("number_detected", False)),
        key=lambda minor: int(minor.get("_y_start", 0)),
    )
    if not numbered:
        return False
    first = numbered[0]
    old_y0 = max(0, int(first.get("_y_start", 0)))
    anchor_y = max(0, int(first.get("_anchor_y", 0)))
    prefix_height = anchor_y - old_y0
    if old_y0 > 8 or prefix_height < max(220, int(height * 0.32)):
        return False

    texts = []
    for item in ocr_items or []:
        bbox = item.get("bbox") or []
        if not bbox:
            continue
        center_y = (min(point[1] for point in bbox) + max(point[1] for point in bbox)) / 2
        if center_y < anchor_y - 4:
            texts.append(re.sub(r"\s+", "", str(item.get("text", ""))))
    preamble_text = "\n".join(texts)
    signals = {keyword for keyword in _EXPLANATORY_PREAMBLE_KEYWORDS if keyword in preamble_text}
    if len(signals) < 2:
        return False

    new_y0 = min(max(old_y0, anchor_y - 8), max(old_y0, height - 1))
    old_y1 = min(height, int(first.get("_y_end", height)))
    if old_y1 <= new_y0:
        return False
    first["_y_start"] = new_y0
    first["crop_img"] = block_img[new_y0:old_y1].copy()
    first["_explanatory_preamble_removed"] = True
    first["_boundary_source"] = "ocr_number_after_explanation"
    return True


_SUBSECTION_HEADING = re.compile(
    r"^[（(]([一二三四五六七八九十百千〇零两0-9—–－-]+)([）)]?)(.{1,40})$"
)
_KNOWLEDGE_POINT_HEADING = re.compile(
    r"^知识点([一二三四五六七八九十百千〇零两0-9]+)[、:：．.]?(.*)$"
)

_SUBSECTION_KEYWORDS = (
    "阅读", "练习", "训练", "积累", "运用", "基础", "拓展", "习作",
    "填空", "选择", "计算", "解答", "实验", "综合", "写作",
)


def _normalise_subsection_number(token: str, trailing_text: str) -> str:
    """Repair conservative OCR confusions in printed group-number headings.

    A short horizontal stroke inside parentheses is very often the Chinese
    numeral ``一``.  It is only repaired when the remaining text looks like a
    real exercise-group heading, so answer subparts and punctuation are not
    promoted into new groups.
    """
    compact = re.sub(r"\s+", "", token)
    heading_like = any(keyword in trailing_text for keyword in _SUBSECTION_KEYWORDS)
    # ``(-5)表示`` and similar formula fragments are common in mathematics
    # OCR.  They are never printed group headings, even when OCR retains both
    # parentheses, and must not become a negative major number.
    if compact[:1] in "—–－-" and any(character.isdigit() for character in compact[1:]):
        return ""
    if compact and all(char in "—–－-" for char in compact):
        return "一" if heading_like else ""
    if compact.isdigit():
        return data.int_to_chinese(compact) if heading_like else ""
    return data.int_to_chinese(compact)


def _ocr_item_xy(item: dict) -> tuple[int, int]:
    box = item.get("bbox", [])
    if not box:
        return 0, 0
    return int(min(point[0] for point in box)), int(min(point[1] for point in box))


def _subsection_boundaries(ocr_items: list[dict], crop_width: int) -> list[dict]:
    """Find printed group headings, including ``知识点一/二`` textbook sections."""
    result = []
    for item in ocr_items:
        text = re.sub(r"\s+", "", str(item.get("text", "")))
        match = _SUBSECTION_HEADING.match(text)
        knowledge_match = _KNOWLEDGE_POINT_HEADING.match(text) if not match else None
        starts_segment = False
        if match:
            trailing_text = match.group(3)
            number = _normalise_subsection_number(match.group(1), trailing_text)
            if not number:
                continue
            # Missing closing parentheses are accepted only for recognisable group
            # headings.  This handles damaged scans without treating (1) subparts
            # as peer exercise groups.
            if not match.group(2) and not any(keyword in trailing_text for keyword in _SUBSECTION_KEYWORDS):
                continue
        elif knowledge_match:
            number = data.int_to_chinese(knowledge_match.group(1))
            trailing_text = knowledge_match.group(2)
            starts_segment = True
        else:
            continue
        x, y = _ocr_item_xy(item)
        if x > max(120, int(crop_width * 0.35)):
            continue
        result.append({
            "y": y,
            "number": number,
            "title": text,
            "number_repaired": bool(match and number != match.group(1)),
            "starts_segment": starts_segment,
        })
    result.sort(key=lambda value: value["y"])
    return result


def _subsection_segments(
    cutter, ocr_items: list[dict], crop_width: int, crop_height: int,
    *, active_title: str = "", active_number: str = "",
) -> list[dict]:
    """Build group ranges and recover a missed leading ``(一)`` when safe.

    Some scans lose the first group heading completely but retain ``(二)``.
    When the prefix is substantial and already contains a primary question 1,
    it cannot belong to the later group.  Preserve it as an inferred first
    group instead of swallowing it into the final question before ``(二)``.
    """
    marks = _subsection_boundaries(ocr_items, crop_width)
    if not marks:
        return [{
            "y0": 0, "y1": crop_height,
            "subsection_title": active_title,
            "subsection_number": active_number,
            "subsection_number_inferred": False,
        }]

    segments: list[dict] = []
    first = marks[0]
    first_value = data.chinese_number_to_int(first.get("number", ""))
    prefix_is_large = int(first["y"]) >= max(120, int(crop_height * 0.18))
    prefix_items = _ocr_items_in_y_range(ocr_items, 0, int(first["y"]))
    prefix_anchors = (
        _primary_number_boundaries(cutter, prefix_items, crop_width, int(first["y"]))
        if prefix_is_large else []
    )
    recover_first_group = bool(
        first_value == 2 and prefix_anchors and int(prefix_anchors[0].get("num", 0)) == 1
    )
    # A later knowledge-point heading begins a new group.  Preserve a genuine
    # numbered prefix as the prior active group, but do not turn unnumbered
    # lesson explanation above the first ``知识点一`` into a question.
    if first.get("starts_segment") and int(first["y"]) > 0 and active_number and not recover_first_group:
        segments.append({
            "y0": 0, "y1": int(first["y"]),
            "subsection_title": active_title,
            "subsection_number": active_number,
            "subsection_number_inferred": False,
        })
    if recover_first_group:
        segments.append({
            "y0": 0, "y1": int(first["y"]),
            "subsection_title": "(一)分组标题未完整识别",
            "subsection_number": "一",
            "subsection_number_inferred": True,
        })

    for index, mark in enumerate(marks):
        segments.append({
            "y0": int(mark["y"]) if (
                index > 0 or recover_first_group or mark.get("starts_segment")
            ) else 0,
            "y1": int(marks[index + 1]["y"]) if index + 1 < len(marks) else crop_height,
            "subsection_title": mark["title"],
            "subsection_number": mark["number"],
            "subsection_number_inferred": False,
        })
    return segments


def _ocr_items_in_y_range(ocr_items: list[dict], y0: int, y1: int) -> list[dict]:
    """Copy block-local OCR items into a subsection-local coordinate system."""
    result = []
    for item in ocr_items:
        _x, top = _ocr_item_xy(item)
        if top < y0 or top >= y1:
            continue
        copied = dict(item)
        copied["bbox"] = [[point[0], point[1] - y0] for point in item.get("bbox", [])]
        result.append(copied)
    return result


def _primary_number_boundaries(cutter, ocr_items: list[dict], width: int, height: int) -> list[dict]:
    """Keep a lone printed 2/3 at a continuation boundary; the production cutter rejects it."""
    trusted = cutter.find_minor_boundaries(ocr_items, width, height)
    if trusted:
        return trusted
    candidates = []
    left_limit = max(90, int(width * 0.28))
    for item in ocr_items:
        number, style = cutter.extract_minor_anchor(str(item.get("text", "")).strip())
        if number is None or style != "primary" or number > 150:
            continue
        x, y = _ocr_item_xy(item)
        if x < 0 or x > left_limit or y < 0 or y >= height:
            continue
        candidates.append({
            "num": number,
            "style": style,
            "text": str(item.get("text", "")),
            "y": y,
            "x": x,
            "confidence": float(item.get("conf", 1.0)),
            "detected": True,
        })
    return sorted(candidates, key=lambda value: (value["y"], value["x"]))


def _record_from_minor(
    minor: dict, *, title: str, major_number: str, subsection_title: str = "",
    subsection_number: str = "", page_index: int = -1,
    shared_segments: list[dict] | None = None, role: str = "question",
    continuation_of: str = "",
    parent_id: str = "",
) -> dict:
    detected = bool(minor.get("number_detected", False))
    minor_number = str(minor.get("num", "")) if (
        role == "continuation" or (detected and role == "question")
    ) else ""
    shared_segments = list(shared_segments or [])
    return data.new_record(
        minor.get("_source_box", []),
        number=minor_number,
        minor_number=minor_number,
        major=title,
        major_title=title,
        major_number=major_number,
        subsection_title=subsection_title,
        subsection_number=subsection_number,
        region_role=role,
        continuation_of=continuation_of,
        parent_id=parent_id,
        minor_number_inferred=False,
        shared_context_bbox=(shared_segments[-1]["bbox"] if shared_segments else []),
        shared_context_segments=shared_segments,
        use_shared_context=bool(shared_segments and role == "question"),
        text="",
        origin=("suggested_continuation" if role == "continuation" else "suggested_rule"),
        boundary_source=str(minor.get("_boundary_source", "") or ""),
        detector_assisted=minor.get("_boundary_source") == "detector_assisted",
        detector_confidence=float(minor.get("_detector_confidence", 0.0) or 0.0),
    )


def _knowledge_point_group(title: str) -> dict | None:
    match = _KNOWLEDGE_POINT_HEADING.match(re.sub(r"\s+", "", str(title or "")))
    if not match:
        return None
    return {
        "title": str(title or "").strip(),
        "number": data.int_to_chinese(match.group(1)),
    }


def _previous_question_context(previous_question: dict | None) -> dict:
    if not isinstance(previous_question, dict):
        return {}
    data.normalize_record(previous_question)
    if previous_question.get("region_role") != "question":
        return {}
    return {
        "id": str(previous_question.get("id", "")),
        "major_title": str(previous_question.get("major_title", "")),
        "major_number": str(previous_question.get("major_number", "")),
        "subsection_title": str(previous_question.get("subsection_title", "")),
        "subsection_number": str(previous_question.get("subsection_number", "")),
        "minor_number": str(previous_question.get("minor_number", "")),
    }


def _question_suggestions(
    image_path: str, profile: str, *, page_index: int = -1,
    previous_question: dict | None = None, force_refresh: bool = False,
) -> list[dict]:
    """Reuse the existing rule cutter for one page only (no book-level scan)."""
    import cut_questions_v4 as cutter

    image = cutter.cv2_imread(image_path)
    if image is None:
        raise ValueError("无法读取图片")
    detector_boxes = _optional_question_detections(cutter, image)
    ocr_items = cutter.ocr_page_structure(
        image_path, image, profile=profile, force_refresh=force_refresh,
    )
    blocks = cutter.split_page_into_majors(image, ocr_items, profile=profile)
    sidebar_exclusions = cutter.detect_page_sidebar_exclusions(
        image, blocks, ocr_items=ocr_items,
    )
    records: list[dict] = []
    previous = _previous_question_context(previous_question)
    active = dict(previous)
    pending_shared_segments: list[dict] = []
    pending_parent_id = ""
    for block_position, block in enumerate(blocks):
        if block.get("title") != "整页" and cutter.is_non_question_section(block.get("title", "")):
            continue
        local_items = cutter.ocr_items_for_block(ocr_items, block)
        title = block.get("title", "")
        block_index = block.get("idx", "")
        printed_major = data.extract_major_number(
            title, block_index if isinstance(block_index, int) and 0 < block_index < 100 else "",
        ) if title != "整页" else ""
        # ``知识点一/二`` is an explicit fresh grouping printed on this page.
        # It must win over the previous-page state: otherwise an untitled
        # first column silently keeps the prior major (for example ``14``),
        # producing misleading paths such as ``14-1-1``.  It is also never a
        # continuation even if OCR fails to recover the first question number.
        # Once this page has emitted a question, a later knowledge-point mark
        # belongs *inside* the same reading flow: retain the prefix above it
        # (for example questions 16/17 in the right column) and let the mark
        # start the next group below.
        starts_new_knowledge_point = bool(
            title == "整页" and not printed_major and not records and any(
                mark.get("starts_segment")
                for mark in _subsection_boundaries(
                    local_items, block["crop_img"].shape[1],
                )
            )
        )
        if starts_new_knowledge_point:
            active = {
                "major_title": "", "major_number": "",
                "subsection_title": "", "subsection_number": "", "minor_number": "",
            }
            pending_shared_segments = []
            pending_parent_id = ""

        # A leading untitled block belongs to the previous page until the first
        # strong printed question number.  Preserve the prefix as a linked
        # continuation and emit the numbered suffix as a new question.
        leading_anchors = []
        if title == "整页" and not printed_major and block.get("_col") != "right":
            height, width = block["crop_img"].shape[:2]
            leading_anchors = _primary_number_boundaries(cutter, local_items, width, height)
        if (
            title == "整页" and not printed_major and block.get("_col") != "right" and previous
            and not starts_new_knowledge_point
            # A page may be split into several visual columns.  Only its first
            # emitted block can possibly continue the previous page; treating
            # column two as a new page was what turned ``1-5`` into ``14-5``.
            and not records
            and (not leading_anchors or leading_anchors[0]["num"] > 1)
        ):
            height, width = block["crop_img"].shape[:2]
            anchors = leading_anchors
            first_new = anchors[0] if anchors else None
            split_y = max(0, int(first_new["y"]) - 4) if first_new else height
            if split_y > 12:
                continuation = {
                    "num": previous.get("minor_number", ""), "number_detected": False,
                    "crop_img": block["crop_img"][:split_y], "_y_start": 0, "_y_end": split_y,
                }
                cutter.attach_question_source_metadata(
                    [continuation], block, image_path, (image.shape[1], image.shape[0]),
                )
                cutter.trim_cuts_to_page_sidebar_exclusions(
                    [continuation], image, sidebar_exclusions,
                )
                cutter.trim_cuts_to_detached_page_footer([continuation], image)
                records.append(_record_from_minor(
                    continuation, title=previous["major_title"], major_number=previous["major_number"],
                    subsection_title=previous["subsection_title"],
                    subsection_number=previous["subsection_number"], page_index=page_index,
                    role="continuation", continuation_of=previous["id"],
                ))
            if first_new:
                for index, anchor in enumerate(anchors):
                    y0 = split_y if index == 0 else max(split_y, int(anchor["y"]) - 4)
                    y1 = max(y0 + 1, int(anchors[index + 1]["y"]) - 4) if index + 1 < len(anchors) else height
                    minor = {
                        "num": anchor["num"], "number_detected": True,
                        "crop_img": block["crop_img"][y0:y1], "_y_start": y0, "_y_end": y1,
                        "_anchor_y": int(anchor["y"]),
                    }
                    cutter.attach_question_source_metadata(
                        [minor], block, image_path, (image.shape[1], image.shape[0]),
                    )
                    cutter.trim_cuts_to_page_sidebar_exclusions(
                        [minor], image, sidebar_exclusions,
                    )
                    cutter.trim_cuts_to_detached_page_footer([minor], image)
                    records.append(_record_from_minor(
                        minor, title=previous["major_title"], major_number=previous["major_number"],
                        subsection_title=previous["subsection_title"],
                        subsection_number=previous["subsection_number"], page_index=page_index,
                    ))
                    active = dict(previous, minor_number=str(anchor["num"]))
            continue

        if printed_major:
            active = {
                "major_title": title, "major_number": printed_major,
                "subsection_title": "", "subsection_number": "", "minor_number": "",
            }
            pending_shared_segments = []
            pending_parent_id = ""
        elif title == "整页" and active:
            # Keep the current page's heading/group state across its verified
            # columns.  A right-hand column is often the final questions of
            # ``知识点一`` immediately followed by a ``知识点二`` heading.
            title = active.get("major_title", "")
            printed_major = active.get("major_number", "")
        elif title == "整页":
            active = {
                "major_title": "", "major_number": "", "subsection_title": "",
                "subsection_number": "", "minor_number": "",
            }

        segments = _subsection_segments(
            cutter, local_items, block["crop_img"].shape[1], block["crop_img"].shape[0],
            active_title=active.get("subsection_title", ""),
            active_number=active.get("subsection_number", ""),
        )

        for segment_position, segment in enumerate(segments):
            y0, y1 = int(segment["y0"]), int(segment["y1"])
            subsection_title = segment["subsection_title"]
            subsection_number = segment["subsection_number"]
            if subsection_number:
                knowledge_group = _knowledge_point_group(subsection_title)
                if knowledge_group:
                    active["major_title"] = knowledge_group["title"]
                    active["major_number"] = knowledge_group["number"]
                    active["subsection_title"] = ""
                    active["subsection_number"] = ""
                else:
                    active["subsection_title"] = subsection_title
                    active["subsection_number"] = subsection_number
                active["minor_number"] = ""
            segment_block = dict(block)
            segment_block["title"] = title
            segment_block["crop_img"] = block["crop_img"][y0:y1]
            segment_block["_y_start"] = int(block.get("_y_start", 0)) + y0
            segment_items = _ocr_items_in_y_range(local_items, y0, y1)
            minors = cutter.split_major_into_minors(
                segment_block["crop_img"], segment_items or None, profile=profile,
            )
            _trim_explanatory_preamble_from_first_minor(
                minors, segment_items, segment_block["crop_img"],
            )
            cutter.attach_question_source_metadata(
                minors, segment_block, image_path, (image.shape[1], image.shape[0]),
            )
            cutter.trim_cuts_to_page_sidebar_exclusions(minors, image, sidebar_exclusions)
            cutter.trim_cuts_to_detached_page_footer(minors, image)
            minors = _refine_manual_prelabel_minors(cutter, minors, image, detector_boxes)
            # Model-assisted refinement is optional, but when enabled it must
            # never enlarge a question over a verified three/four-column gutter.
            cutter.trim_cuts_to_column_bounds(
                minors, image, segment_block.get("_column_bounds"),
            )

            has_later_right_continuation = bool(
                block.get("_col") == "left" and segment_position == len(segments) - 1 and
                any(later.get("_col") == "right" and later.get("title") == "整页"
                    for later in blocks[block_position + 1:])
            )
            is_shared_material = bool(
                subsection_number and has_later_right_continuation and len(minors) == 1 and
                not minors[0].get("number_detected", False)
            )
            if is_shared_material:
                material = minors[0]
                box = material.get("_source_box", [])
                context_segment = {"page_index": page_index, "bbox": list(box)}
                pending_shared_segments = [context_segment]
                # Shared reading material remains internal crop context.  It is
                # deliberately not emitted as a visible "parent question" box.
                pending_parent_id = ""
                continue

            shared_context_box = _shared_instruction_box(segment_block, minors)
            if shared_context_box and minors:
                question_boxes = [
                    minor.get("_source_box", []) for minor in minors
                    if len(minor.get("_source_box", [])) == 4
                ]
                if question_boxes:
                    shared_context_box[0] = max(
                        int(shared_context_box[0]), min(int(box[0]) for box in question_boxes)
                    )
                    shared_context_box[2] = min(
                        int(shared_context_box[2]), max(int(box[2]) for box in question_boxes)
                    )
            inherited_segments = list(pending_shared_segments)
            if shared_context_box:
                inherited_segments.append({"page_index": page_index, "bbox": shared_context_box})
            for minor in minors:
                box = minor.get("_source_box", [])
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                records.append(_record_from_minor(
                    minor, title=title, major_number=printed_major,
                    subsection_title=subsection_title, subsection_number=subsection_number,
                    page_index=page_index, shared_segments=inherited_segments,
                    parent_id=pending_parent_id,
                ))
                if minor.get("number_detected", False):
                    active["minor_number"] = str(minor.get("num", ""))
            if title == block.get("title") or block.get("_col") == "right":
                pending_shared_segments = []
                pending_parent_id = ""
    if not records:  # A safe editable fallback, rather than silently skipping a page.
        height, width = image.shape[:2]
        records.append(data.new_record(
            [0, 0, width, height], number="", minor_number="", region_role="question",
            origin="suggested_page",
        ))
    return records


def _answer_suggestions(image_path: str, profile: str, *, force_refresh: bool = False) -> list[dict]:
    import cut_questions_v4 as cutter

    records = cutter.process_answer_pages(
        [image_path], profile=profile, force_refresh=force_refresh,
    )
    answers = []
    for record in records:
        box = record.get("source_box", [])
        if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
            major_index = record.get("major_idx", record.get("major", ""))
            answers.append(data.new_record(
                box, answer=True, number=record.get("num", ""),
                minor_number=record.get("num", ""),
                major=record.get("major", ""),
                major_title=record.get("major", ""),
                major_number=data.extract_major_number(record.get("major", ""), major_index),
                subsection_title=record.get("subsection_title", ""),
                subsection_number=data.extract_subsection_number(record.get("subsection_title", "")),
                region_role="question",
                context=record.get("context", ""),
                text=record.get("text", ""), origin="suggested_rule",
            ))
    return answers


class AnnotationPage(QWidget):
    changed = Signal()

    def __init__(self, owner: "ManualAnnotationWindow", answer: bool) -> None:
        super().__init__()
        self.owner, self.answer = owner, answer
        self.page_index = -1
        self.item_by_id: dict[str, ReviewBox] = {}
        self.selected_record: dict | None = None
        # One detached process per requested page.  Leaving a page must never
        # stop its pre-annotation, and another page may be requested while it
        # is still running.
        self.suggestion_workers: dict[int, QProcess] = {}
        self.pages = QListWidget()
        self.pages.currentRowChanged.connect(self.open_page)
        self.canvas = ReviewCanvas()
        self.canvas.box_drawn.connect(self.add_box)
        self.canvas.boxes_delete_requested.connect(self.delete_boxes)
        self.canvas.selection_changed.connect(self.selected_box)
        self.draw_button = QPushButton("绘制新框")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self.toggle_draw)
        self.delete_button = QPushButton("删除选中")
        self.delete_button.setObjectName("danger")
        self.delete_button.setCheckable(True)
        self.delete_button.toggled.connect(self.toggle_delete)
        self.accept_button = QPushButton("确认本页标注")
        self.accept_button.setObjectName("primary")
        self.accept_button.clicked.connect(self.accept_page)
        self.retry_button = QPushButton("生成预标注框")
        self.retry_button.clicked.connect(self.run_suggestions)
        self.number_edit = QLineEdit()
        self.number_edit.setPlaceholderText("如：8、8（1）")
        self.number_edit.editingFinished.connect(self.update_number)
        self.meta = QLabel("选择页面后才会识别该页；不会扫描整本资料。")
        self.meta.setWordWrap(True)
        self.gesture_tip = QLabel("操作提示\n绘制时左键：用左上笔尖划出新框\n右键按住拖动：移动教材位置\n删除模式：点框即删，或拖红框批量删除\n滚轮：放大 / 缩小\n关闭绘制后左键拖现有框：调整位置")
        self.gesture_tip.setWordWrap(True)
        self.gesture_tip.setStyleSheet("background:#edf7f6;color:#175c58;border:1px solid #b9e0dc;border-radius:6px;padding:7px;")
        controls = QVBoxLayout()
        controls.addWidget(QLabel("框选工具"))
        controls.addWidget(self.draw_button)
        controls.addWidget(self.delete_button)
        controls.addWidget(self.accept_button)
        controls.addWidget(self.retry_button)
        controls.addSpacing(10)
        controls.addWidget(QLabel("题号 / 答案号"))
        controls.addWidget(self.number_edit)
        controls.addWidget(self.gesture_tip)
        controls.addWidget(self.meta)
        controls.addStretch(1)
        middle = QWidget(); middle_layout = QHBoxLayout(middle); middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.addWidget(self.canvas, 1); control_widget = QWidget(); control_widget.setLayout(controls); control_widget.setFixedWidth(170); middle_layout.addWidget(control_widget)
        split = QSplitter(); split.addWidget(self.pages); split.addWidget(middle); split.setSizes([230, 980])
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 12, 12, 12); layout.addWidget(split)

    @property
    def field(self) -> str:
        return "answers" if self.answer else "questions"

    @property
    def kind(self) -> str:
        return "answer" if self.answer else "question"

    def refresh_pages(self) -> None:
        self.pages.blockSignals(True); self.pages.clear()
        for index, page in enumerate(self.owner.page_list(self.answer)):
            count = len(page.get(self.field, []))
            status = "✓" if page.get("status") == "accepted" else ("●" if count else "○")
            self.pages.addItem(QListWidgetItem(f"{status} {index + 1:>3}  {page.get('image', '')}  ({count})"))
        self.pages.blockSignals(False)
        if self.pages.count() and self.page_index < 0:
            self.pages.setCurrentRow(0)

    def open_page(self, index: int) -> None:
        if index < 0 or not self.owner.project:
            return
        self.commit_boxes()
        self.page_index = index
        page = self.owner.page_list(self.answer)[index]
        image_path = data.source_path(self.owner.project, page, self.answer)
        if not self.canvas.load_page(image_path):
            self.meta.setText("图片无法打开：" + image_path); return
        self.populate_boxes(page.get(self.field, []))
        if page_index in self.suggestion_workers:
            self.meta.setText("本页正在后台生成预标注框；可切换到其他页，完成后会自动写入本页。")
        elif page.get(self.field):
            self.meta.setText("已载入本页。可拖动题框、绘制新框，完成后点击“确认本页标注”。")
        else:
            self.meta.setText("本页尚未生成预标注框。点击右侧“生成预标注框”后，程序只分析这一页；也可直接手动绘制。")

    def populate_boxes(self, records: list[dict]) -> None:
        self.item_by_id.clear()
        for record in records:
            if "bbox" not in record:
                continue
            record["box"] = list(record["bbox"])
            item = self.canvas.add_box(record)
            self.item_by_id[record["id"]] = item

    def run_suggestions(self) -> None:
        if self.page_index < 0 or not self.owner.project:
            return
        page_index = self.page_index
        page = self.owner.page_list(self.answer)[page_index]
        # Clicking the button again explicitly means regenerate this page.
        # Only a job for the same page is restarted; jobs for other pages keep
        # running in the background.
        previous = self.suggestion_workers.pop(page_index, None)
        if previous is not None:
            previous.finished.disconnect()
            if previous.state() != QProcess.ProcessState.NotRunning:
                previous.kill()
            previous.deleteLater()
        self.meta.setText("正在后台生成本页预标注框。现在可切换到其他页；任务不会中断。")
        worker_path = APP_DIR / "review_tools" / "manual_annotation_worker.py"
        worker = QProcess(self)
        self.suggestion_workers[page_index] = worker
        worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        worker.finished.connect(
            lambda exit_code, exit_status, process=worker, target_page=page_index:
            self.suggestion_finished(process, target_page, exit_code, exit_status)
        )
        worker.start(sys.executable, [
            str(worker_path), self.kind, data.source_path(self.owner.project, page, self.answer),
            page.get("profile", "general"),
        ])

    def suggestion_finished(self, worker: QProcess, page_index: int, _exit_code: int, _exit_status) -> None:
        # A replaced/cancelled process may still emit finished.  Its output is
        # deliberately ignored so it cannot overwrite the newer result.
        if self.suggestion_workers.get(page_index) is not worker:
            return
        self.suggestion_workers.pop(page_index, None)
        output = bytes(worker.readAllStandardOutput()).decode("utf-8", errors="replace")
        payload = None
        for line in reversed(output.splitlines()):
            if line.startswith("@@MANUAL_ANNOTATION_RESULT@@"):
                try:
                    payload = json.loads(line.removeprefix("@@MANUAL_ANNOTATION_RESULT@@"))
                except json.JSONDecodeError:
                    pass
                break
        worker.deleteLater()
        if not payload or not payload.get("ok"):
            self.suggestion_failed(page_index, (payload or {}).get("error") or output or "预标注进程未返回结果")
            return
        page = self.owner.page_list(self.answer)[page_index]
        page[self.field] = payload.get("records", [])
        page["status"] = "suggested"
        if page_index == self.page_index:
            self.open_page(page_index)
            self.meta.setText(f"已生成 {len(page[self.field])} 个建议框。请核对、拖动或补画后确认本页。")
        self.changed.emit()

    def suggestion_failed(self, page_index: int, message: str) -> None:
        if page_index == self.page_index:
            self.meta.setText("本页预标注失败，可直接绘制题框。原因：" + message.splitlines()[0])

    def toggle_draw(self, active: bool) -> None:
        if active and self.delete_button.isChecked():
            self.delete_button.setChecked(False)
        self.canvas.set_draw_enabled(active)
        self.draw_button.setText("结束绘制" if active else "绘制新框")

    def toggle_delete(self, active: bool) -> None:
        if active and self.draw_button.isChecked():
            self.draw_button.setChecked(False)
        self.canvas.set_delete_enabled(active)
        self.delete_button.setText("退出删除模式" if active else "删除选中")

    def add_box(self, box: list[int]) -> None:
        page = self.owner.page_list(self.answer)[self.page_index]
        record = data.new_record(box, answer=self.answer, number="", origin="manual")
        page.setdefault(self.field, []).append(record)
        record["box"] = box
        self.item_by_id[record["id"]] = self.canvas.add_box(record)
        self.changed.emit()

    def selected_box(self, item: object) -> None:
        if not item:
            self.selected_record = None
            self.number_edit.clear()
            return
        self.selected_record = item.record
        self.number_edit.setText(str(item.record.get("number", "")))
        self.meta.setText("已选中：可拖动位置；若需要改变大小，请删除后重画该框。")

    def update_number(self) -> None:
        if self.selected_record is not None:
            self.selected_record["number"] = self.number_edit.text().strip()
            self.selected_record["minor_number"] = self.selected_record["number"]
            self.changed.emit()

    def delete_selected(self) -> None:
        selected = [item for item in self.canvas.scene_ref.selectedItems() if isinstance(item, ReviewBox)]
        self.delete_boxes(selected)

    def delete_boxes(self, selected: list[ReviewBox]) -> None:
        if not selected or self.page_index < 0:
            return
        ids = {item.record["id"] for item in selected}
        page = self.owner.page_list(self.answer)[self.page_index]
        page[self.field] = [record for record in page.get(self.field, []) if record.get("id") not in ids]
        for item in selected:
            self.canvas.scene_ref.removeItem(item)
        self.changed.emit()

    def commit_boxes(self) -> None:
        if self.page_index < 0 or not self.owner.project:
            return
        for item in self.canvas.boxes():
            item.record["bbox"] = item.current_box()
            item.record.pop("box", None)

    def accept_page(self) -> None:
        self.commit_boxes()
        if self.page_index >= 0:
            self.owner.page_list(self.answer)[self.page_index]["status"] = "accepted"
            self.refresh_pages(); self.pages.setCurrentRow(self.page_index)
            self.owner.save()
            self.meta.setText("本页已确认并保存。现在可去“答案匹配”页完成关联。")


class MatchPage(QWidget):
    def __init__(self, owner: "ManualAnnotationWindow") -> None:
        super().__init__(); self.owner = owner; self.current_question: dict | None = None
        self.questions, self.answers = QListWidget(), QListWidget()
        self.questions.currentRowChanged.connect(self.select_question)
        self.answers.itemSelectionChanged.connect(self.update_answer_preview)
        self.question_preview, self.answer_preview = QLabel("选择题目"), QLabel("选择候选答案")
        for label in (self.question_preview, self.answer_preview):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter); label.setMinimumHeight(180); label.setStyleSheet("background:#eef3f7;border:1px solid #dce5ef;")
        self.link = QPushButton("确认关联选中答案"); self.link.setObjectName("primary"); self.link.clicked.connect(self.link_selected)
        self.no_answer = QPushButton("标记暂缺答案"); self.no_answer.clicked.connect(self.mark_missing)
        left = QWidget(); ll=QVBoxLayout(left); ll.addWidget(QLabel("已确认的题目")); ll.addWidget(self.questions); ll.addWidget(self.question_preview)
        right = QWidget(); rl=QVBoxLayout(right); rl.addWidget(QLabel("候选答案（按题号、标题、内容排序）")); rl.addWidget(self.answers); rl.addWidget(self.answer_preview); buttons=QHBoxLayout(); buttons.addWidget(self.link); buttons.addWidget(self.no_answer); rl.addLayout(buttons)
        split=QSplitter(); split.addWidget(left); split.addWidget(right); split.setSizes([500,700]); layout=QVBoxLayout(self); layout.setContentsMargins(12,12,12,12); layout.addWidget(split)

    def refresh(self) -> None:
        selected_id = self.current_question.get("id") if self.current_question else ""
        self.questions.blockSignals(True); self.questions.clear()
        for page_index, page, question in data.all_questions(self.owner.project or {}):
            if page.get("status") != "accepted":
                continue
            state = "已配" if data.paired_answer_ids(self.owner.project, question["id"]) else "待配"
            item = QListWidgetItem(f"[{state}] 第{page_index+1}页  {data.display_number(question) or '未编号'}  {question.get('major_title') or question.get('major') or ''}")
            item.setData(Qt.ItemDataRole.UserRole, (page_index, question)); self.questions.addItem(item)
            if question["id"] == selected_id: self.questions.setCurrentItem(item)
        self.questions.blockSignals(False)
        if self.questions.count() and self.questions.currentRow() < 0: self.questions.setCurrentRow(0)

    def select_question(self, _row: int) -> None:
        item = self.questions.currentItem()
        if not item: return
        page_index, self.current_question = item.data(Qt.ItemDataRole.UserRole)
        page = self.owner.project["pages"][page_index]
        self.show_crop(self.question_preview, data.source_path(self.owner.project, page), self.current_question["bbox"])
        self.answers.clear()
        for score, a_page_index, a_page, answer in data.candidates(self.owner.project, self.current_question):
            item = QListWidgetItem(f"{score:5.1f}  第{a_page_index+1}页  {data.display_number(answer) or '未编号'}  {(answer.get('text') or '')[:55]}")
            item.setData(Qt.ItemDataRole.UserRole, (a_page, answer)); item.setSelected(answer["id"] in data.paired_answer_ids(self.owner.project, self.current_question["id"])); self.answers.addItem(item)
        if self.answers.count(): self.answers.setCurrentRow(0)

    def show_crop(self, label: QLabel, image_path: str, box: list[int]) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull(): label.setText("图片无法加载"); return
        x0, y0, x1, y1 = (int(value) for value in box)
        label.setPixmap(pixmap.copy(x0, y0, max(1, x1 - x0), max(1, y1 - y0)).scaled(
            460, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        ))

    def update_answer_preview(self) -> None:
        item = self.answers.currentItem()
        if item and self.owner.project:
            page, answer = item.data(Qt.ItemDataRole.UserRole)
            self.show_crop(self.answer_preview, data.source_path(self.owner.project, page, True), answer["bbox"])

    def link_selected(self) -> None:
        if not self.current_question: return
        answer_ids = [item.data(Qt.ItemDataRole.UserRole)[1]["id"] for item in self.answers.selectedItems()]
        if not answer_ids: QMessageBox.information(self, "请选择", "请先在右侧选择一个或多个答案条目。"); return
        data.set_pair(self.owner.project, self.current_question["id"], answer_ids); self.owner.save(); self.refresh()

    def mark_missing(self) -> None:
        if self.current_question:
            data.set_pair(self.owner.project, self.current_question["id"], [], "no_answer"); self.owner.save(); self.refresh()


class ManualAnnotationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.project: dict | None = None; self.project_path = ""; self.last_export_path = ""
        self.setWindowTitle("PaperFlow · 人工标注"); self.resize(1500, 940)
        self.question_page = AnnotationPage(self, False); self.answer_page = AnnotationPage(self, True); self.match_page = MatchPage(self)
        self.question_page.changed.connect(self.project_changed); self.answer_page.changed.connect(self.project_changed)
        tabs=QTabWidget(); tabs.addTab(self.question_page, "1 题目页标注"); tabs.addTab(self.answer_page, "2 答案页标注"); tabs.addTab(self.match_page, "3 答案匹配")
        tabs.currentChanged.connect(lambda index: self.match_page.refresh() if index == 2 else None)
        central = QWidget(); central_layout = QVBoxLayout(central); central_layout.setContentsMargins(12, 8, 12, 12); central_layout.setSpacing(8)
        self.path_label = QLabel("尚未打开教材。点击“打开教材”后，程序会自动保存标注项目。")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("background:#eaf3f8;color:#19304d;border:1px solid #cddfea;border-radius:7px;padding:8px 12px;")
        central_layout.addWidget(self.path_label); central_layout.addWidget(tabs, 1); self.setCentralWidget(central)
        bar=QToolBar("项目"); self.addToolBar(bar)
        open_book=QAction("打开教材", self); open_book.triggered.connect(self.open_book); bar.addAction(open_book)
        open_action=QAction("打开已有标注项目", self); open_action.triggered.connect(self.open_project); bar.addAction(open_action)
        answer_action=QAction("设置答案目录", self); answer_action.triggered.connect(self.set_answer_root); bar.addAction(answer_action)
        save_action=QAction("保存", self); save_action.triggered.connect(self.save); bar.addAction(save_action)
        export_action=QAction("导出已匹配训练样本", self); export_action.triggered.connect(self.export_matched_samples); bar.addAction(export_action)
        self.status=QStatusBar(); self.setStatusBar(self.status); self.status.showMessage("新建项目仅扫描文件名；点击“生成预标注框”才分析当前页。")

    def page_list(self, answer: bool) -> list[dict]: return self.project.get("answer_pages" if answer else "pages", []) if self.project else []
    def project_changed(self) -> None: self.status.showMessage("本页标注已变更，点击“确认本页标注”或“保存”写入项目文件。")
    def _seed_book_profile(self, book_root: str) -> bool:
        """给还没有版式的项目补一个初值，和界面（桥服务）保持同一口径。

        只读书名和已落盘的抽样结论，不触发 OCR；工作台第一次打开一本书时
        就把版式写进去，后续切题不再每页重猜。
        """
        if str(self.project.get("profile") or "") in core.BOOK_PROFILE_CHOICES:
            return False
        profile, source, reason = core.auto_book_profile(book_root)
        self.project["profile"] = profile
        self.project["profile_source"] = source
        self.project["profile_reason"] = reason
        data.sync_page_profiles(self.project, profile)
        return True
    def open_book(self, book_root: str | None = None) -> None:
        if not book_root:
            book_root = QFileDialog.getExistingDirectory(self, "选择一本教材的根目录")
        if not book_root: return
        project_path = data.default_book_project_path(book_root, APP_DIR / "runtime")
        try:
            self.project = data.load_project(project_path) if project_path.is_file() else data.create_book_project(book_root)
            self.project_path = str(project_path)
            if self._seed_book_profile(book_root) or not project_path.is_file(): self.save()
            _save_workspace_state(last_book_root=str(book_root))
            self.refresh_all()
        except Exception as exc:
            QMessageBox.critical(self, "无法打开教材", str(exc))
    def auto_restore_last_book(self) -> None:
        """Reopen the previously used book after a restart, without dialogs."""
        book_root = str(_load_workspace_state().get("last_book_root", "") or "")
        if not book_root or not Path(book_root).is_dir():
            return
        project_path = data.default_book_project_path(book_root, APP_DIR / "runtime")
        try:
            self.project = data.load_project(project_path) if project_path.is_file() else data.create_book_project(book_root)
            self.project_path = str(project_path)
            if self._seed_book_profile(book_root) or not project_path.is_file(): self.save()
            self.refresh_all()
            self.status.showMessage(f"已自动打开上次教材：{book_root}")
        except Exception:
            pass
    def new_project(self) -> None:
        question_root = QFileDialog.getExistingDirectory(self, "选择题目图片目录")
        if not question_root: return
        answer_root = QFileDialog.getExistingDirectory(self, "选择答案图片目录（可取消，之后再补）")
        target, _ = QFileDialog.getSaveFileName(self, "保存人工标注项目", str(Path(question_root) / "人工标注项目.json"), "JSON 文件 (*.json)")
        if not target: return
        self.project=data.create_project(question_root, answer_root); self.project_path=target; self.save(); self.refresh_all()
    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开人工标注项目", "", "JSON 文件 (*.json)")
        if not path: return
        try: self.project=data.load_project(path); self.project_path=path; self.refresh_all()
        except Exception as exc: QMessageBox.critical(self, "无法打开", str(exc))
    def set_answer_root(self) -> None:
        if not self.project:
            QMessageBox.information(self, "请先建立项目", "请先新建或打开一个题目标注项目。"); return
        root = QFileDialog.getExistingDirectory(self, "选择答案图片目录")
        if not root: return
        if self.project.get("answer_pages"):
            choice = QMessageBox.question(self, "替换答案目录", "当前答案页标注将被替换，是否继续？")
            if choice != QMessageBox.StandardButton.Yes: return
        self.project["answer_image_root"] = str(Path(root).resolve())
        self.project["answer_pages"] = data._page_records(Path(root).resolve(), "answers", "general")
        self.project["pairs"] = []
        self.answer_page.page_index = -1; self.answer_page.refresh_pages(); self.save()
    def refresh_all(self) -> None:
        self.question_page.page_index=-1; self.answer_page.page_index=-1; self.question_page.refresh_pages(); self.answer_page.refresh_pages(); self.match_page.refresh(); self.update_paths(); self.status.showMessage("项目已载入：点击“生成预标注框”才会分析当前页。")
    def update_paths(self) -> None:
        if not self.project:
            self.path_label.setText("尚未打开教材。点击“打开教材”后，程序会自动保存标注项目。")
            return
        book = self.project.get("book_root") or self.project.get("image_root", "")
        question_root = self.project.get("image_root", "")
        answer_root = self.project.get("answer_image_root") or "未设置（可点“设置答案目录”补充）"
        self.path_label.setText(
            f"当前教材：{book}\n题目来源：{question_root}\n答案来源：{answer_root}\n标注项目：{self.project_path}\n"
            f"已确认匹配：{self.accepted_pair_count()} 对；最近导出：{self.last_export_path or '尚未导出'}"
        )
    def save(self) -> None:
        if not self.project or not self.project_path: return
        self.question_page.commit_boxes(); self.answer_page.commit_boxes(); data.save_project(self.project, self.project_path); self.update_paths(); self.status.showMessage("已自动保存：" + self.project_path)

    def accepted_pair_count(self) -> int:
        return sum(
            1 for pair in (self.project or {}).get("pairs", [])
            if pair.get("status") == "accepted" and pair.get("answer_ids")
        )

    def export_matched_samples(self) -> None:
        """Write reviewed question-answer crops to a new local training folder."""
        if not self.project or not self.project_path:
            QMessageBox.information(self, "请先打开教材", "请先打开一本教材或已有标注项目。"); return
        self.save()
        count = self.accepted_pair_count()
        if not count:
            QMessageBox.information(self, "暂无可导出样本", "请先在“答案匹配”页确认至少一组题目与答案。"); return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = APP_DIR / "runtime" / "manual_annotation_exports" / Path(self.project_path).stem / stamp
        try:
            report = export_project(self.project_path, str(target))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc)); return
        self.last_export_path = str(target)
        self.update_paths()
        self.status.showMessage(f"已导出 {report['exported']} 对训练样本：{target}")
        QMessageBox.information(
            self, "导出完成",
            f"已导出 {report['exported']} 对，跳过 {report['skipped']} 对。\n\n位置：{target}",
        )


def main() -> int:
    app = QApplication(sys.argv); configure_cjk_font(app); app.setStyleSheet(APP_STYLE)
    window = ManualAnnotationWindow(); window.show()
    QTimer.singleShot(0, window.auto_restore_last_book)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
