"""Corporate desktop shell for PaperFlow.

This is intentionally a local Qt client: it starts the existing Python cutter
as a child process and reads its structured progress lines.  No images or OCR
data leave the workstation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QAction, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMenuBar, QMessageBox, QPushButton, QProgressBar, QSplitter, QStatusBar,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)


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


class CutterMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PaperFlow · 智能切题平台")
        self.resize(1380, 900)
        self.setMinimumSize(1080, 700)
        self.process: QProcess | None = None
        self.sources: list[str] = []
        self.source_index = 0
        self.started_at = 0.0
        self._stdout_buffer = ""
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        self.setStyleSheet(APP_STYLE)
        menu = QMenuBar(self)
        tools = menu.addMenu("工具")
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
        brand_sub = QLabel("智能切题平台  /  本地离线部署")
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
        top.addSpacing(12)
        self.status_chip = QLabel("准备就绪")
        self.status_chip.setStyleSheet("background:#e4f7f4;color:#00756d;border-radius:14px;padding:7px 13px;font-weight:700;")
        top.addWidget(self.status_chip)
        content_layout.addLayout(top)

        metrics = QHBoxLayout()
        self.metric_queue = self._add_metric(metrics, "任务队列")
        self.metric_pages = self._add_metric(metrics, "已识别页面")
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
        form.addRow("处理范围", self.mode_combo)
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
        hint = QLabel("切题结果会按页保存题框坐标与训练元数据，可在输出目录中核对。")
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
        open_button = QPushButton("打开结果目录")
        open_button.clicked.connect(self.open_output)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
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
        self.started_at = time.monotonic()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._save_settings()
        self.start_next_source()

    def start_next_source(self) -> None:
        if self.source_index >= len(self.sources):
            self.finish_all()
            return
        source = self.sources[self.source_index]
        output = self.output_edit.text().strip()
        self.process = QProcess(self)
        self._stdout_buffer = ""
        self.process.setWorkingDirectory(str(APP_DIR))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished_source)
        args = ["-B", "-u", str(CORE_SCRIPT), source, "--out", output, "--mode", self.mode_combo.currentData()]
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
        self.phase_label.setText(message)
        if total > 0:
            self.progress.setValue(round(100 * min(1.0, current / total)))
        if phase == "page_scan":
            self.metric_pages.setText(str(current))
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
        except (OSError, ValueError, TypeError):
            pass
        self.refresh_sources()
        self.metric_pages.setText("0")
        self.metric_time.setText("00:00:00")

    def _save_settings(self) -> None:
        payload = {
            "input_dirs": self.sources,
            "output_dir": self.output_edit.text().strip(),
            "mode": self.mode_combo.currentData(),
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
