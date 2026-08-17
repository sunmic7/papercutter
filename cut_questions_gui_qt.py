"""Corporate desktop shell and live page-review workspace for PaperFlow.

This is intentionally a local Qt client: it starts the existing Python cutter
as a child process and reads its structured progress lines.  No images, review
decisions or OCR data leave the workstation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QFontDatabase, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QFileDialog, QFormLayout,
    QFrame, QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMenuBar, QMessageBox, QPushButton, QProgressBar, QScrollArea, QSplitter,
    QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem, QTextEdit, QToolBar,
    QVBoxLayout, QWidget,
)

import live_review_protocol


APP_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = APP_DIR / "cut_questions_v4.py"
SETTINGS_PATH = APP_DIR / "runtime" / "gui_settings_v2.json"
PROGRESS_PREFIX = "@@PROGRESS@@"
MAX_LOG_BLOCKS = 1600


APP_STYLE = """
* { font-family: 'Microsoft YaHei', 'Noto Sans SC', 'DengXian', 'Segoe UI'; font-size: 13px; }
QMainWindow { background: #f4f7fb; }
QMenuBar { background: #ffffff; color: #19304d; border-bottom: 1px solid #dce5ef; padding: 4px 12px; }
QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #dce5ef; spacing: 8px; padding: 8px 14px; }
QFrame#brandBar { background: #102a43; border: 0; }
QLabel#brand { color: #ffffff; font-size: 20px; font-weight: 700; letter-spacing: 1px; }
QLabel#brandSub { color: #b9cce1; font-size: 12px; }
QLabel#kicker { color: #5b6f82; font-size: 12px; font-weight: 700; letter-spacing: 1px; }
QLabel#pageTitle { color: #102a43; font-size: 22px; font-weight: 700; }
QGroupBox { background: #ffffff; color: #19304d; font-weight: 700; border: 1px solid #dce5ef; border-radius: 10px; margin-top: 12px; padding: 16px 12px 12px 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 6px; }
QPushButton { background: #ffffff; color: #19304d; border: 1px solid #cbd8e6; border-radius: 6px; padding: 7px 13px; font-weight: 600; }
QPushButton:hover { background: #eef5fb; border-color: #7da6cb; }
QPushButton:disabled { color: #9aa8b7; background: #f5f7f9; border-color: #e2e8ef; }
QPushButton#primary { background: #007f78; color: #ffffff; border-color: #007f78; }
QPushButton#primary:hover { background: #006c67; }
QPushButton#annotation { background: #0b7285; color: #ffffff; border-color: #0b7285; padding: 10px 18px; font-size: 14px; }
QPushButton#annotation:hover { background: #075c6c; }
QPushButton#danger { color: #bd2c2c; border-color: #efbbbb; }
QLineEdit, QComboBox, QSpinBox { background: #ffffff; border: 1px solid #cbd8e6; border-radius: 5px; padding: 6px 8px; min-height: 19px; }
QTableWidget, QListWidget, QTextEdit { background: #ffffff; border: 1px solid #dce5ef; border-radius: 6px; }
QHeaderView::section { background: #edf3f8; color: #3d5267; border: 0; border-bottom: 1px solid #dce5ef; padding: 7px; font-weight: 700; }
QProgressBar { background: #e8eef4; border: 0; border-radius: 5px; height: 10px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #00a99d; border-radius: 5px; }
QStatusBar { background: #ffffff; color: #53677a; border-top: 1px solid #dce5ef; }
QLabel#metricValue { color: #102a43; font-size: 19px; font-weight: 700; }
QLabel#metricLabel { color: #67788a; font-size: 11px; }
"""


def card_metric(label: str) -> tuple[QFrame, QLabel]:
    frame = QFrame()
    frame.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #dce5ef; border-radius: 9px; }")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(13, 10, 13, 10)
    value = QLabel("—")
    value.setObjectName("metricValue")
    caption = QLabel(label)
    caption.setObjectName("metricLabel")
    layout.addWidget(value)
    layout.addWidget(caption)
    return frame, value


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


class LiveReviewWindow(QMainWindow):
    """Non-blocking preview and gate-mode page editor."""
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PaperFlow · 实时切题审核")
        self.resize(1510, 940)
        self.setMinimumSize(1120, 700)
        self.request: dict | None = None
        self.items_by_id: dict[str, ReviewBox] = {}
        self.selected: ReviewBox | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 14)
        header = QHBoxLayout()
        text = QVBoxLayout()
        kicker = QLabel("LIVE REVIEW WORKSPACE")
        kicker.setObjectName("kicker")
        self.title_label = QLabel("等待切题页面…")
        self.title_label.setObjectName("pageTitle")
        self.subtitle = QLabel("预览模式不会阻塞切题；逐页审核模式需要确认后才会写入题图。")
        self.subtitle.setStyleSheet("color: #637789;")
        text.addWidget(kicker)
        text.addWidget(self.title_label)
        text.addWidget(self.subtitle)
        header.addLayout(text)
        header.addStretch()
        self.mode_badge = QLabel("未连接")
        self.mode_badge.setStyleSheet("background:#e8eef4;color:#466078;border-radius:12px;padding:6px 12px;font-weight:700;")
        header.addWidget(self.mode_badge)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = ReviewCanvas()
        self.canvas.box_drawn.connect(self.add_manual_box)
        self.canvas.selection_changed.connect(self.select_box)
        splitter.addWidget(self.canvas)

        side = QWidget()
        side.setMinimumWidth(325)
        side_layout = QVBoxLayout(side)
        tool_row = QHBoxLayout()
        self.draw_button = QPushButton("绘制漏题框")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self.set_draw_mode)
        delete_button = QPushButton("删除选中")
        delete_button.clicked.connect(self.delete_selected)
        merge_button = QPushButton("合并选中")
        merge_button.clicked.connect(self.merge_selected)
        tool_row.addWidget(self.draw_button)
        tool_row.addWidget(delete_button)
        tool_row.addWidget(merge_button)
        side_layout.addLayout(tool_row)
        side_layout.addWidget(QLabel("本页题目框"))
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.select_from_list)
        side_layout.addWidget(self.box_list, 1)
        geometry = QGroupBox("选中题框坐标（原图像素）")
        form = QGridLayout(geometry)
        self.coords = []
        for index, label in enumerate(("左", "上", "右", "下")):
            form.addWidget(QLabel(label), index // 2, (index % 2) * 2)
            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.valueChanged.connect(self.apply_geometry)
            form.addWidget(spin, index // 2, (index % 2) * 2 + 1)
            self.coords.append(spin)
        side_layout.addWidget(geometry)
        self.note = QLineEdit()
        self.note.setPlaceholderText("审核备注（可选）")
        side_layout.addWidget(self.note)
        self.approve_button = QPushButton("确认本页并继续")
        self.approve_button.setObjectName("primary")
        self.approve_button.clicked.connect(self.approve)
        side_layout.addWidget(self.approve_button)
        self.help_label = QLabel("操作：框可直接拖动；需要改变大小时，在坐标区修改，或删除后重画。选中多个框可合并。")
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color:#64788b; padding:5px;")
        side_layout.addWidget(self.help_label)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.show_scanning_status()

    def show_scanning_status(self, current: int = 0, total: int = 0, message: str = "") -> None:
        """Explain why no boxes exist while the cutter builds book structure."""
        if self.request and self.request.get("mode") == "gate":
            return
        self.request = None
        self.canvas.scene_ref.clear()
        self.items_by_id.clear()
        self.box_list.clear()
        self.title_label.setText("题框尚未生成")
        progress = f"预扫描 {current} / {total} 页" if total else "正在预扫描教材"
        self.subtitle.setText(
            f"{progress}：系统先识别题目页、答案页和单元结构；完成预扫描后，"
            "进入实际切题阶段才会显示原页题框。"
        )
        self.mode_badge.setText("等待切题阶段")
        self.mode_badge.setStyleSheet("background:#e8eef4;color:#466078;border-radius:12px;padding:6px 12px;font-weight:700;")
        self.approve_button.setVisible(False)
        self.note.setVisible(False)
        self.statusBar().showMessage(message or "预扫描期间没有题框是正常现象，请等待进入实际切题阶段。")

    def present_request_path(self, path: str) -> None:
        try:
            request = live_review_protocol.load_request(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.statusBar().showMessage(f"无法读取实时审核请求：{error}")
            return
        self.request = request
        if not self.canvas.load_page(request["image"]):
            self.statusBar().showMessage("原页读取失败，无法审核")
            return
        self.items_by_id.clear()
        for index, record in enumerate(request.get("candidates", [])):
            item = self.canvas.add_box(dict(record), QColor("#ff7a00"))
            self.items_by_id[record["id"]] = item
        self.rebuild_list()
        mode = request.get("mode", "preview")
        self.mode_badge.setText("逐页审核 · 等待确认" if mode == "gate" else "实时预览 · 不阻塞")
        self.mode_badge.setStyleSheet(
            "background:#e4f7f4;color:#00756d;border-radius:12px;padding:6px 12px;font-weight:700;"
            if mode == "gate" else "background:#e8eef4;color:#466078;border-radius:12px;padding:6px 12px;font-weight:700;"
        )
        self.approve_button.setVisible(mode == "gate")
        self.note.setVisible(mode == "gate")
        self.title_label.setText(f"{request.get('book_name') or '当前教材'} · {request.get('page_label') or Path(request['image']).name}")
        self.subtitle.setText(f"{request.get('unit_name') or '未分类单元'}  ·  {request.get('major_title') or '整页'}  ·  {len(self.items_by_id)} 个预测题框")
        self.statusBar().showMessage("请检查题框；逐页审核模式下，确认后后台会继续。")
        self.show()
        self.raise_()
        self.activateWindow()

    def current_item_records(self) -> list[dict]:
        result = []
        for index, item in enumerate(sorted(self.canvas.boxes(), key=lambda box: (box.current_box()[1], box.current_box()[0]))):
            record = dict(item.record)
            record["id"] = record.get("id") or f"review-{index + 1}"
            record["box"] = item.current_box()
            result.append(record)
        return result

    def rebuild_list(self) -> None:
        current_id = self.selected.record.get("id") if self.selected else ""
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for index, item in enumerate(sorted(self.canvas.boxes(), key=lambda box: (box.current_box()[1], box.current_box()[0]))):
            box = item.current_box()
            number = item.record.get("number") or "手工"
            list_item = QListWidgetItem(f"{index + 1:02d}  第 {number} 题   [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
            list_item.setData(Qt.ItemDataRole.UserRole, item.record.get("id"))
            self.box_list.addItem(list_item)
            if item.record.get("id") == current_id:
                self.box_list.setCurrentItem(list_item)
        self.box_list.blockSignals(False)

    def select_box(self, item) -> None:
        self.selected = item
        if item is None:
            return
        box = item.current_box()
        for spin, value in zip(self.coords, box):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        for row in range(self.box_list.count()):
            if self.box_list.item(row).data(Qt.ItemDataRole.UserRole) == item.record.get("id"):
                self.box_list.blockSignals(True)
                self.box_list.setCurrentRow(row)
                self.box_list.blockSignals(False)
                break

    def select_from_list(self, row: int) -> None:
        if row < 0:
            return
        target = self.box_list.item(row).data(Qt.ItemDataRole.UserRole)
        for item in self.canvas.boxes():
            if item.record.get("id") == target:
                item.setSelected(True)
                self.select_box(item)
                return

    def set_draw_mode(self, enabled: bool) -> None:
        self.canvas.draw_enabled = enabled
        self.canvas.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        self.draw_button.setText("正在画框：点击拖动" if enabled else "绘制漏题框")

    def add_manual_box(self, box: list[int]) -> None:
        record = {
            "id": f"manual-{uuid.uuid4().hex[:10]}", "source_id": "", "box": box,
            "number": None, "number_detected": False, "risk_flags": ["人工补框"],
        }
        item = self.canvas.add_box(record, QColor("#00a99d"))
        self.items_by_id[record["id"]] = item
        item.setSelected(True)
        self.selected = item
        self.rebuild_list()

    def apply_geometry(self) -> None:
        if self.selected is None:
            return
        x0, y0, x1, y1 = (spin.value() for spin in self.coords)
        if x1 <= x0 or y1 <= y0:
            return
        self.selected.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        self.rebuild_list()

    def delete_selected(self) -> None:
        selected = [item for item in self.canvas.scene_ref.selectedItems() if isinstance(item, ReviewBox)]
        for item in selected:
            self.items_by_id.pop(item.record.get("id"), None)
            self.canvas.scene_ref.removeItem(item)
        self.selected = None
        self.rebuild_list()

    def merge_selected(self) -> None:
        selected = [item for item in self.canvas.scene_ref.selectedItems() if isinstance(item, ReviewBox)]
        if len(selected) < 2:
            self.statusBar().showMessage("请先选中至少两个要合并的题框。")
            return
        boxes = [item.current_box() for item in selected]
        merged = [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]
        self.delete_selected()
        self.add_manual_box(merged)

    def approve(self) -> None:
        if not self.request:
            return
        raw = {"reviewer": "本地逐页审核", "notes": self.note.text().strip(), "candidates": self.current_item_records()}
        try:
            live_review_protocol.write_decision(self.request, raw)
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "无法保存审核结果", str(error))
            return
        self.approve_button.setEnabled(False)
        self.statusBar().showMessage("本页审核已保存，正在继续切题…")


class CutterMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PaperFlow · 智能切题与审核平台")
        self.resize(1380, 900)
        self.setMinimumSize(1080, 700)
        self.process: QProcess | None = None
        self.sources: list[str] = []
        self.source_index = 0
        self.review_window = LiveReviewWindow()
        self.started_at = 0.0
        self.current_session = ""
        self._stdout_buffer = ""
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        self.setStyleSheet(APP_STYLE)
        menu = QMenuBar(self)
        tools = menu.addMenu("工具")
        action = QAction("打开精度标注", self)
        action.triggered.connect(lambda: self.launch_script("review_tools/annotate_cuts.py"))
        tools.addAction(action)
        action = QAction("人工标注", self)
        action.triggered.connect(lambda: self.launch_script("review_tools/manual_annotation_workspace.py"))
        tools.addAction(action)
        action = QAction("打开结果审核", self)
        action.triggered.connect(lambda: self.launch_script("review_tools/review_pairs.py"))
        tools.addAction(action)
        action = QAction("训练控制台", self)
        action.triggered.connect(lambda: self.launch_script("training_tools/training_dashboard.py"))
        tools.addAction(action)
        self.setMenuBar(menu)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        brand = QFrame()
        brand.setObjectName("brandBar")
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(28, 13, 28, 13)
        brand_name = QLabel("PAPERFLOW")
        brand_name.setObjectName("brand")
        brand_sub = QLabel("智能切题与审核平台  /  本地离线部署")
        brand_sub.setObjectName("brandSub")
        brand_layout.addWidget(brand_name)
        brand_layout.addSpacing(18)
        brand_layout.addWidget(brand_sub)
        brand_layout.addStretch()
        security = QLabel("●  数据仅保留在本机")
        security.setStyleSheet("color:#a9f1e8; font-weight:700;")
        brand_layout.addWidget(security)
        layout.addWidget(brand)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(26, 20, 26, 18)
        top = QHBoxLayout()
        title_layout = QVBoxLayout()
        kicker = QLabel("WORKSPACE")
        kicker.setObjectName("kicker")
        title = QLabel("教材切题任务中心")
        title.setObjectName("pageTitle")
        title_layout.addWidget(kicker)
        title_layout.addWidget(title)
        top.addLayout(title_layout)
        top.addStretch()
        annotation_button = QPushButton("人工标注\n逐页标注与答案匹配")
        annotation_button.setObjectName("annotation")
        annotation_button.setMinimumHeight(52)
        annotation_button.clicked.connect(lambda: self.launch_script("review_tools/manual_annotation_workspace.py"))
        top.addWidget(annotation_button)
        top.addSpacing(12)
        self.status_chip = QLabel("准备就绪")
        self.status_chip.setStyleSheet("background:#e4f7f4;color:#00756d;border-radius:14px;padding:7px 13px;font-weight:700;")
        top.addWidget(self.status_chip)
        content_layout.addLayout(top)

        metrics = QHBoxLayout()
        self.metric_queue = self._add_metric(metrics, "任务队列")
        self.metric_pages = self._add_metric(metrics, "已识别页面")
        self.metric_review = self._add_metric(metrics, "实时审核")
        self.metric_time = self._add_metric(metrics, "累计用时")
        content_layout.addLayout(metrics)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._build_task_card())
        left_layout.addWidget(self._build_run_card())
        split.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_progress_card())
        right_layout.addWidget(self._build_log_card(), 1)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        content_layout.addWidget(split, 1)
        layout.addWidget(content, 1)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("选择教材目录后即可开始本地切题。")

    def _add_metric(self, layout: QHBoxLayout, label: str) -> QLabel:
        frame, value = card_metric(label)
        layout.addWidget(frame)
        return value

    def _build_task_card(self) -> QGroupBox:
        box = QGroupBox("任务配置")
        layout = QVBoxLayout(box)
        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("单本教材", "book")
        self.mode_combo.addItem("批量年份目录", "year")
        self.review_combo = QComboBox()
        self.review_combo.addItem("标准模式 · 完成后审核", "off")
        self.review_combo.addItem("实时预览 · 不暂停切题", "preview")
        self.review_combo.addItem("逐页审核 · 确认后写入", "gate")
        self.review_combo.currentIndexChanged.connect(self.review_mode_changed)
        form.addRow("处理范围", self.mode_combo)
        form.addRow("人员审核", self.review_combo)
        self.review_hint = QLabel()
        self.review_hint.setWordWrap(True)
        self.review_hint.setStyleSheet("color:#617487; padding:3px 2px 0 2px;")
        form.addRow("审核说明", self.review_hint)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        output_button = QPushButton("选择输出位置")
        output_button.clicked.connect(self.choose_output)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(output_button)
        form.addRow("输出根目录", output_row)
        layout.addLayout(form)
        row = QHBoxLayout()
        add_button = QPushButton("＋ 添加教材目录")
        add_button.clicked.connect(self.add_source)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self.remove_source)
        row.addWidget(add_button)
        row.addWidget(remove_button)
        row.addStretch()
        layout.addLayout(row)
        self.source_table = QTableWidget(0, 2)
        self.source_table.setHorizontalHeaderLabels(["教材目录", "状态"])
        self.source_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.source_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.source_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.source_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.source_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.source_table.setMinimumHeight(205)
        layout.addWidget(self.source_table)
        return box

    def _build_run_card(self) -> QGroupBox:
        box = QGroupBox("运行控制")
        layout = QVBoxLayout(box)
        hint = QLabel("逐页审核时，人工调整的题框会直接用于本次裁切、metadata 和后续训练数据；批量并行会自动降为顺序处理。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#617487;")
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.start_button = QPushButton("开始切题")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("停止任务")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        preview_button = QPushButton("打开实时审核台")
        preview_button.clicked.connect(self.review_window.show)
        open_button = QPushButton("打开结果目录")
        open_button.clicked.connect(self.open_output)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(preview_button)
        row.addStretch()
        row.addWidget(open_button)
        layout.addLayout(row)
        return box

    def _build_progress_card(self) -> QGroupBox:
        box = QGroupBox("处理进度")
        layout = QVBoxLayout(box)
        self.book_label = QLabel("当前教材：等待开始")
        self.book_label.setStyleSheet("font-weight:700;color:#19304d;")
        self.phase_label = QLabel("准备就绪")
        self.phase_label.setStyleSheet("color:#607488;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.book_label)
        layout.addWidget(self.phase_label)
        layout.addWidget(self.progress)
        return box

    def _build_log_card(self) -> QGroupBox:
        box = QGroupBox("运行日志")
        layout = QVBoxLayout(box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log.setStyleSheet("font-family: Consolas; font-size: 12px; color: #334e68;")
        layout.addWidget(self.log)
        return box

    def add_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择包含教材图片的目录")
        if not path:
            return
        path = os.path.abspath(path)
        if path not in self.sources:
            self.sources.append(path)
        self.refresh_sources()

    def remove_source(self) -> None:
        row = self.source_table.currentRow()
        if row >= 0:
            self.sources.pop(row)
            self.refresh_sources()

    def refresh_sources(self, statuses: dict[str, str] | None = None) -> None:
        statuses = statuses or {}
        self.source_table.setRowCount(len(self.sources))
        for row, path in enumerate(self.sources):
            self.source_table.setItem(row, 0, QTableWidgetItem(path))
            status = QTableWidgetItem(statuses.get(path, "待处理"))
            status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.source_table.setItem(row, 1, status)
        self.metric_queue.setText(str(len(self.sources)))

    def choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择切题结果输出根目录")
        if path:
            self.output_edit.setText(os.path.abspath(path))

    def review_mode_changed(self) -> None:
        selected = self.review_combo.currentData()
        is_gate = selected == "gate"
        if is_gate:
            self.statusBar().showMessage("逐页审核会等待人员确认；已自动切换为单进程顺序处理。")
            self.review_hint.setText(
                "开始后会先预扫描整本教材，预扫描阶段没有题框；首次完成实际切题后，"
                "审核台会暂停等待人工确认。"
            )
        elif selected == "preview":
            self.review_hint.setText(
                "开始后会先预扫描整本教材，预扫描阶段没有题框；进入实际切题阶段后，"
                "审核台会自动显示原页与预测题框，但不会暂停后台任务。"
            )
        else:
            self.review_hint.setText("标准模式不显示实时题框；切题完成后可在“结果审核”中复核。")

    def start(self) -> None:
        if self.process is not None:
            return
        if not CORE_SCRIPT.is_file():
            QMessageBox.critical(self, "核心脚本缺失", str(CORE_SCRIPT))
            return
        if not self.sources:
            QMessageBox.warning(self, "尚未选择教材", "请至少添加一个教材目录。")
            return
        output = self.output_edit.text().strip()
        if not output:
            QMessageBox.warning(self, "尚未选择输出位置", "请选择输出根目录。")
            return
        if Path(output).exists() and not Path(output).is_dir():
            QMessageBox.warning(self, "输出目录无效", "输出根目录不能是文件。")
            return
        self.source_index = 0
        self.metric_pages.setText("0")
        self.metric_review.setText("关闭" if self.review_combo.currentData() == "off" else "待预览")
        self.started_at = time.monotonic()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.current_session = f"session-{uuid.uuid4().hex[:12]}"
        self._save_settings()
        self.start_next_source()

    def start_next_source(self) -> None:
        if self.source_index >= len(self.sources):
            self.finish_all()
            return
        source = self.sources[self.source_index]
        output = self.output_edit.text().strip()
        review_mode = self.review_combo.currentData()
        self.process = QProcess(self)
        self._stdout_buffer = ""
        self.process.setWorkingDirectory(str(APP_DIR))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PAPERCUTTER_LIVE_REVIEW_MODE", str(review_mode))
        environment.insert("PAPERCUTTER_LIVE_REVIEW_DIR", str(Path(output) / "_运行中" / "live_review" / self.current_session))
        if review_mode == "gate":
            environment.insert("PAPERCUTTER_BOOK_WORKERS", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished_source)
        args = ["-B", "-u", str(CORE_SCRIPT), source, "--out", output, "--mode", self.mode_combo.currentData(), "--live-review", str(review_mode)]
        self.append_log("\n" + "=" * 72)
        self.append_log("开始：" + source)
        self.book_label.setText("当前教材：" + Path(source).name)
        self.phase_label.setText("正在启动本地 OCR 与切题流程…")
        self.status_chip.setText("任务运行中")
        self.status_chip.setStyleSheet("background:#fff3d6;color:#9b6200;border-radius:14px;padding:7px 13px;font-weight:700;")
        self.refresh_sources({source: "处理中"})
        self.process.start(sys.executable, args)

    def read_output(self) -> None:
        if self.process is None:
            return
        self._stdout_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self.consume_line(line.rstrip("\r"))

    def read_error(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.append_log("[stderr] " + line)

    def consume_line(self, line: str) -> None:
        if not line.startswith(PROGRESS_PREFIX):
            self.append_log(line)
            return
        try:
            payload = json.loads(line[len(PROGRESS_PREFIX):])
        except (ValueError, TypeError):
            return
        phase = str(payload.get("phase") or "")
        message = str(payload.get("message") or "正在处理")
        current = int(payload.get("current") or 0)
        total = int(payload.get("total") or 0)
        book = str(payload.get("book") or "")
        if book:
            self.book_label.setText("当前教材：" + book)
        if phase == "page_preview":
            request_path = str(payload.get("review_request") or "")
            if request_path:
                self.metric_review.setText("待确认" if payload.get("review_mode") == "gate" else "预览中")
                self.review_window.present_request_path(request_path)
            return
        self.phase_label.setText(message)
        if total > 0:
            self.progress.setValue(round(100 * min(1.0, current / total)))
        if phase == "page_scan":
            self.metric_pages.setText(str(current))
            if self.review_combo.currentData() != "off":
                self.metric_review.setText(f"预扫描 {current}/{total}" if total else "预扫描中")
                self.review_window.show_scanning_status(current, total, message)
        self.metric_time.setText(self.format_time(time.monotonic() - self.started_at))

    def finished_source(self, exit_code: int, _status) -> None:
        if self._stdout_buffer:
            self.consume_line(self._stdout_buffer.rstrip("\r\n"))
            self._stdout_buffer = ""
        source = self.sources[self.source_index]
        self.append_log(f"结束：{Path(source).name}（退出码 {exit_code}）")
        self.process = None
        self.source_index += 1
        statuses = {path: ("已完成" if index < self.source_index and exit_code == 0 else "待处理") for index, path in enumerate(self.sources)}
        self.refresh_sources(statuses)
        if exit_code != 0:
            answer = QMessageBox.question(self, "当前教材处理失败", f"{Path(source).name} 退出码为 {exit_code}。是否继续下一个教材？")
            if answer != QMessageBox.StandardButton.Yes:
                self.finish_all(failed=True)
                return
        self.start_next_source()

    def stop(self) -> None:
        if self.process is None:
            return
        if QMessageBox.question(self, "停止任务", "确定停止当前切题任务吗？已生成的阶段性结果会保留。") != QMessageBox.StandardButton.Yes:
            return
        self.process.kill()
        self.append_log("已请求停止任务。")

    def finish_all(self, failed: bool = False) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setValue(100 if not failed else self.progress.value())
        self.metric_review.setText("已结束")
        self.metric_time.setText(self.format_time(time.monotonic() - self.started_at) if self.started_at else "—")
        self.status_chip.setText("处理完成" if not failed else "任务已停止")
        self.status_chip.setStyleSheet("background:#e4f7f4;color:#00756d;border-radius:14px;padding:7px 13px;font-weight:700;")
        self.statusBar().showMessage("全部任务已完成。" if not failed else "任务已停止。")

    def append_log(self, text: str) -> None:
        self.log.append(text)
        while self.log.document().blockCount() > MAX_LOG_BLOCKS:
            cursor = self.log.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    @staticmethod
    def format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def open_output(self) -> None:
        path = self.output_edit.text().strip()
        if path and Path(path).is_dir():
            os.startfile(path)
        else:
            QMessageBox.information(self, "结果目录", "结果目录尚不存在，请先选择或完成一次任务。")

    def launch_script(self, relative: str) -> None:
        script = APP_DIR / relative
        if not script.is_file():
            QMessageBox.warning(self, "工具不可用", f"缺少文件：\n{script}")
            return
        QProcess.startDetached(sys.executable, [str(script)], str(APP_DIR))

    def _load_settings(self) -> None:
        try:
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            self.sources = [str(value) for value in payload.get("input_dirs", []) if Path(str(value)).is_dir()]
            self.output_edit.setText(str(payload.get("output_dir") or ""))
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(payload.get("mode", "book"))))
            self.review_combo.setCurrentIndex(max(0, self.review_combo.findData(payload.get("review_mode", "off"))))
        except (OSError, ValueError, TypeError):
            pass
        self.refresh_sources()
        self.review_mode_changed()
        self.metric_pages.setText("0")
        self.metric_review.setText("关闭")
        self.metric_time.setText("00:00:00")

    def _save_settings(self) -> None:
        payload = {
            "input_dirs": self.sources,
            "output_dir": self.output_edit.text().strip(),
            "mode": self.mode_combo.currentData(),
            "review_mode": self.review_combo.currentData(),
        }
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = SETTINGS_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, SETTINGS_PATH)
        except OSError:
            pass

    def closeEvent(self, event):  # noqa: N802
        if self.process is not None:
            answer = QMessageBox.question(self, "任务仍在运行", "确定停止后台切题任务并退出吗？")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
        self._save_settings()
        event.accept()


def configure_cjk_font(app: QApplication) -> None:
    # Qt's offscreen/minimal plugins do not always discover Windows CJK fonts.
    # Register one explicitly so the corporate interface remains readable both
    # in packaged deployments and on restricted office desktops.
    font_candidates = (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "NotoSansSC-VF.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "Deng.ttf",
    )
    for candidate in font_candidates:
        if candidate.is_file():
            font_id = QFontDatabase.addApplicationFont(str(candidate))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QFont(families[0], 10))
                break


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PaperFlow")
    configure_cjk_font(app)
    window = CutterMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
