#!/usr/bin/env python3
"""切题工具图形界面：本地启动 cut_questions_v4.py，不依赖网络或云端服务。"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = APP_DIR / "cut_questions_v4.py"
TRAINING_DASHBOARD_SCRIPT = APP_DIR / "training_tools" / "training_dashboard.py"
SETTINGS_FILE = APP_DIR / "runtime" / "gui_settings.json"
LEGACY_SETTINGS_FILE = APP_DIR / ".cut_questions_gui.json"
WINDOW_TITLE = "教材切题工具"
PARALLEL_OPTIONS = ["自动（按硬件）", "1 本", "2 本", "3 本", "4 本"]
MAX_LOG_LINES = 5000
MAX_BOOK_NAME_LEN = 24


class CutterGui:
    """把长时间运行的切题脚本放入后台子进程，避免冻结界面。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("940x820")
        self.root.minsize(780, 680)
        self.root.configure(background="#ffffff")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="book")
        self.parallel_var = tk.StringVar(value=PARALLEL_OPTIONS[0])
        self.vlm_assist_var = tk.BooleanVar(value=False)
        self.result_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择输入和输出目录。")
        self.progress_text_var = tk.StringVar(value="当前教材：等待开始")
        self.elapsed_var = tk.StringVar(value="累计用时：00:00:00")
        self.stage_labels = {
            "answer_scan": "答案页发现",
            "page_scan": "页面识别",
            "unit": "单元切题",
        }
        self.stage_units = {
            "answer_scan": "页",
            "page_scan": "页",
            "unit": "单元",
        }
        self.stage_progress_vars = {
            key: tk.DoubleVar(value=0) for key in self.stage_labels
        }
        self.stage_text_vars = {
            key: tk.StringVar(value="等待开始") for key in self.stage_labels
        }
        self.stage_started_at: dict[str, float | None] = {
            key: None for key in self.stage_labels
        }
        self.stage_finished_at: dict[str, float | None] = {
            key: None for key in self.stage_labels
        }
        self.stage_current = {key: 0 for key in self.stage_labels}
        self.stage_total = {key: 0 for key in self.stage_labels}
        self.stage_messages = {key: "" for key in self.stage_labels}
        self.input_dirs: list[str] = []
        self.task_sources: list[Path] = []
        self.task_index = 0
        self.failed_sources: list[tuple[Path, int]] = []
        self.stop_requested = False
        self.input_var.trace_add("write", self._update_result_preview)
        self.output_var.trace_add("write", self._update_result_preview)
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.run_started_at: float | None = None
        self.progress_phase = ""
        self.progress_total = 0
        self.progress_current = 0
        # 多本并行时按书名分别记录进度；active_book 是主进度条当前跟随的书。
        self.book_progress: dict[str, dict] = {}
        self.active_book = ""
        self.book_name_var = tk.StringVar(value="")
        self.paused = False
        self.completed_book_count = 0
        self.total_minor_sum = 0
        self.total_matched_sum = 0

        self._configure_style()
        self._build_ui()
        self._load_settings()
        self._update_mode_hint()
        self.root.after(80, self._drain_events)

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        white = "#ffffff"
        style.configure("TFrame", background=white)
        style.configure("TLabel", background=white, foreground="#202020")
        style.configure("TLabelframe", background=white)
        style.configure("TLabelframe.Label", background=white, foreground="#202020")
        style.configure("TRadiobutton", background=white)
        style.map("TRadiobutton", background=[("active", white)])
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        style.configure("Status.TLabel", foreground="#1f5e93")

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(6, weight=1)

        ttk.Label(container, text=WINDOW_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            container,
            text="完全在本机运行：选择目录后启动本地 OCR 切题脚本，不会上传教材图片。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 14))

        paths = ttk.LabelFrame(container, text="任务设置", padding=12)
        paths.grid(row=2, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="处理范围：").grid(row=0, column=0, sticky="w", padx=(0, 8))
        mode_box = ttk.Frame(paths)
        mode_box.grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(mode_box, text="单本教材", variable=self.mode_var, value="book", command=self._update_mode_hint).pack(side="left")
        ttk.Radiobutton(mode_box, text="批量年份目录", variable=self.mode_var, value="year", command=self._update_mode_hint).pack(side="left", padx=(18, 0))
        ttk.Label(mode_box, text="并行切题：").pack(side="left", padx=(24, 4))
        self.parallel_combo = ttk.Combobox(
            mode_box, textvariable=self.parallel_var, state="readonly", width=13,
            values=PARALLEL_OPTIONS,
        )
        self.parallel_combo.pack(side="left")

        ttk.Label(paths, text="输入目录：").grid(row=1, column=0, sticky="w", pady=(12, 0), padx=(0, 8))
        self.input_entry = ttk.Entry(paths, textvariable=self.input_var)
        self.input_entry.grid(row=1, column=1, sticky="ew", pady=(12, 0))
        ttk.Button(paths, text="批量添加目录…", command=self.choose_input).grid(
            row=1, column=2, columnspan=2, padx=(8, 0), pady=(12, 0)
        )
        # 也支持直接粘贴路径后按回车；常用操作只需一次框选多个目录。
        self.input_entry.bind("<Return>", self._add_input_dir_from_enter)

        ttk.Label(paths, text="输入队列：").grid(row=2, column=0, sticky="nw", pady=(10, 0), padx=(0, 8))
        queue_box = ttk.Frame(paths)
        queue_box.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        queue_box.columnconfigure(0, weight=1)
        self.input_list = tk.Listbox(
            queue_box, height=4, selectmode="extended",
            background="#ffffff", foreground="#202020", selectbackground="#d8eaff",
            activestyle="none",
        )
        self.input_list.grid(row=0, column=0, sticky="ew")
        queue_scrollbar = ttk.Scrollbar(queue_box, orient="vertical", command=self.input_list.yview)
        self.input_list.configure(yscrollcommand=queue_scrollbar.set)
        queue_scrollbar.grid(row=0, column=1, sticky="ns")
        queue_actions = ttk.Frame(queue_box)
        queue_actions.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Button(queue_actions, text="移除选中", command=self.remove_selected_input_dirs).pack(side="left")
        ttk.Button(queue_actions, text="清空队列", command=self.clear_input_dirs).pack(side="left", padx=(8, 0))

        ttk.Label(paths, text="输出根目录：").grid(row=3, column=0, sticky="w", pady=(10, 0), padx=(0, 8))
        ttk.Entry(paths, textvariable=self.output_var).grid(row=3, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(paths, text="浏览…", command=self.choose_output).grid(row=3, column=3, padx=(8, 0), pady=(10, 0))

        ttk.Checkbutton(
            paths, text="启用 Qwen 语义辅助（测试：排除答题卡/非题页，不替代答案配对）",
            variable=self.vlm_assist_var,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self.mode_hint = ttk.Label(paths, style="Hint.TLabel", wraplength=760, justify="left")
        self.mode_hint.grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Label(paths, textvariable=self.result_var, style="Status.TLabel", wraplength=760, justify="left").grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(5, 0)
        )

        controls = ttk.Frame(container)
        controls.grid(row=3, column=0, sticky="ew", pady=12)
        self.run_button = ttk.Button(controls, text="开始切题", command=self.start)
        self.run_button.pack(side="left")
        self.pause_button = ttk.Button(controls, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(controls, text="停止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.open_button = ttk.Button(controls, text="打开结果目录", command=self.open_result)
        self.open_button.pack(side="left", padx=(8, 0))
        ttk.Label(controls, textvariable=self.status_var, style="Status.TLabel").pack(side="right")

        training_box = ttk.LabelFrame(container, text="可训练切题模型（CPU）", padding=(12, 8))
        training_box.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        training_box.columnconfigure(0, weight=1)
        ttk.Label(
            training_box,
            text=(
                "训练模型学习整题区域，可辅助优化题框边界、选项区域和单双栏版面；"
                "它与 OCR/题号规则协同，冲突时自动保留原切题逻辑。"
            ),
            style="Hint.TLabel", wraplength=660, justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(training_box, text="训练与进度", command=self.open_training_dashboard).grid(
            row=0, column=1, padx=(12, 6), sticky="e"
        )
        ttk.Button(training_box, text="训练说明", command=self.show_training_info).grid(
            row=0, column=2, sticky="e"
        )

        progress_box = ttk.Frame(container)
        progress_box.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        progress_box.columnconfigure(1, weight=1)
        # 并行时显示当前书名，解决“进度条看不出是哪本书”的问题。
        self.book_name_label = ttk.Label(
            progress_box, textvariable=self.book_name_var, style="Status.TLabel",
        )
        self.book_name_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        bar_start_row = 1
        for row, key in enumerate(("answer_scan", "page_scan", "unit")):
            r = row + bar_start_row
            ttk.Label(progress_box, text=self.stage_labels[key] + "：").grid(
                row=r, column=0, sticky="w", padx=(0, 8), pady=(0 if row == 0 else 6, 0),
            )
            ttk.Progressbar(
                progress_box, orient="horizontal", mode="determinate",
                variable=self.stage_progress_vars[key], maximum=100,
            ).grid(row=r, column=1, sticky="ew", pady=(0 if row == 0 else 6, 0))
            ttk.Label(progress_box, textvariable=self.stage_text_vars[key], style="Hint.TLabel").grid(
                row=r, column=2, sticky="e", padx=(12, 0), pady=(0 if row == 0 else 6, 0),
            )
        ttk.Label(progress_box, textvariable=self.progress_text_var, style="Hint.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(progress_box, textvariable=self.elapsed_var, style="Hint.TLabel").grid(
            row=4, column=2, sticky="e", padx=(16, 0), pady=(8, 0)
        )

        log_box = ttk.LabelFrame(container, text="运行日志", padding=8)
        log_box.grid(row=6, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_box, wrap="word", state="disabled", font=("Consolas", 10),
            background="#ffffff", foreground="#202020", insertbackground="#202020",
        )
        scrollbar = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._append_log("界面已就绪。请先选择目录。")

    def open_training_dashboard(self) -> None:
        """Open the train/monitor window from the main cutter interface."""
        if not TRAINING_DASHBOARD_SCRIPT.is_file():
            messagebox.showerror("训练功能不可用", f"未找到训练控制台：\n{TRAINING_DASHBOARD_SCRIPT}")
            return
        try:
            subprocess.Popen(
                [sys.executable, str(TRAINING_DASHBOARD_SCRIPT)],
                cwd=str(APP_DIR),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            self._append_log("已打开“切题模型训练控制台”，可查看训练指标、日志并控制训练。")
        except OSError as exc:
            messagebox.showerror("无法打开训练控制台", str(exc))

    def show_training_info(self) -> None:
        """Explain capabilities and realistic expectations before users train."""
        window = tk.Toplevel(self.root)
        window.title("训练切题模型说明")
        window.geometry("720x500")
        window.minsize(580, 380)
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)
        text = tk.Text(
            frame, wrap="word", font=("Microsoft YaHei UI", 10),
            background="#ffffff", foreground="#202020", padx=10, pady=8,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text.insert("1.0", """这个功能能做什么？

1. 学习“完整的一题”在页面上的视觉边界，辅助识别题干、选项、图形和单双栏排版。
2. 训练完成后，模型会先给出高置信度题框，再由现有 OCR 识别题号、章节和答案；遇到低置信度、跨页或题数冲突时，自动回退到原有规则，因此不会因为模型缺失而停止切题。
3. 控制台可查看每轮 Precision、Recall、mAP、训练日志，并可启动或停止训练。

当前能达到什么效果？

HiLEx 训练用于学习通用试卷版面结构，通常能改善题框贴合度、选项完整性和单双栏页面的稳定性。当前正在训练的版本只使用英文试卷 HiLEx，属于“版面预训练模型”：它的验证指标只代表 HiLEx 测试集，不等于中文教辅的最终准确率。

要获得中文教辅上的最终效果，需要用带标注的中文原始页做进一步微调，并按书本划分训练/验证/测试集。届时应以中文独立测试集的召回率、误切率、题框 IoU、漏内容率作为验收标准，而不是只看训练集指标。

使用建议：

• 先运行 5 轮、640 分辨率的探测训练，确认耗时和指标；
• 再运行 80 轮、960 分辨率的完整训练；
• 完整训练后先导出模型并在少量中文书籍上抽检，再逐步启用。
""")
        text.configure(state="disabled")
        ttk.Button(window, text="关闭", command=window.destroy).pack(pady=(0, 12))

    def _update_mode_hint(self) -> None:
        is_year = self.mode_var.get() == "year"
        if not is_year:
            self.mode_hint.configure(text="单本教材：输入选择一整本教材目录；输出会自动保存到“输出根目录\\输入教材目录名”。")
        else:
            self.mode_hint.configure(
                text="批量年份目录：输入选择包含多本教材的目录；每本教材会分别输出到所选输出根目录下。"
                "默认按当前电脑硬件自动决定并行切几本；配置较低或内存紧张时可在上方选“1 本”顺序处理。"
            )
        # 并行切题只对批量年份模式有意义；单本教材始终只处理一本。
        self.parallel_combo.configure(state="readonly" if is_year else "disabled")
        self._update_result_preview()

    def _update_result_preview(self, *_: object) -> None:
        output = self.output_var.get().strip()
        if not output:
            self.result_var.set("")
            return
        count = len(self.input_dirs)
        if self.mode_var.get() == "book" and count == 1:
            self.result_var.set(f"预计结果目录：{Path(output) / Path(self.input_dirs[0]).name}")
        elif count:
            mode_text = "本教材" if self.mode_var.get() == "book" else "年份目录"
            self.result_var.set(f"已添加 {count} 个{mode_text}，将依次输出到：{output}")
        else:
            self.result_var.set("点击“批量添加目录…”并框选至少一个输入目录。")

    def choose_input(self) -> None:
        """通过 Qt 的资源管理器式文件框一次选择多个教材目录。"""
        selected = self._choose_multiple_directories_qt()
        if not selected:
            return
        for path in selected:
            self.input_var.set(path)
            self.add_input_dir()

    def _choose_multiple_directories_qt(self) -> list[str]:
        """使用 DataSort 同类 Qt 文件框；显示所有盘符并支持多选目录。"""
        try:
            from PySide6.QtCore import QDir
            from PySide6.QtWidgets import (
                QApplication, QAbstractItemView, QDialog, QFileDialog, QListView, QTreeView,
            )
        except ImportError:
            messagebox.showerror(
                WINDOW_TITLE,
                "缺少 PySide6，无法打开批量目录选择窗口。\n"
                "请在 papercutter 文件夹执行：\npython -m pip install -r requirements.txt",
            )
            return []

        app = QApplication.instance()
        if app is None:
            # 不启动全局 exec；QFileDialog.exec() 自己管理这一次模态选择。
            app = QApplication([WINDOW_TITLE])
            app.setQuitOnLastWindowClosed(False)
            self._qt_application = app

        current_text = self.input_var.get().strip()
        if not current_text and self.input_dirs:
            current_text = self.input_dirs[-1]
        initial_dir = current_text if current_text and Path(current_text).is_dir() else str(Path.home())

        class MultiDirectoryDialog(QFileDialog):
            """让 Qt 文件框的确认按钮返回已选文件夹，而非把它当作“打开”。"""

            def __init__(self) -> None:
                super().__init__(None, "选择要切题的教材目录（可 Ctrl / Shift / 框选多个）", initial_dir)
                self.chosen_directories: list[str] = []

            def accept(self) -> None:
                selected: list[str] = []
                for view in self.findChildren(QListView) + self.findChildren(QTreeView):
                    model = view.model()
                    selection = view.selectionModel()
                    if selection is None or not hasattr(model, "filePath"):
                        continue
                    for index in selection.selectedRows(0):
                        path = model.filePath(index)
                        if path and Path(path).is_dir() and path not in selected:
                            selected.append(path)
                if selected:
                    self.chosen_directories = selected
                    self.done(QDialog.DialogCode.Accepted)
                    return
                # 未选中子目录时，允许把当前浏览位置本身作为一个输入目录。
                current = self.directory().absolutePath()
                if current and Path(current).is_dir():
                    self.chosen_directories = [current]
                    self.done(QDialog.DialogCode.Accepted)

        dialog = MultiDirectoryDialog()
        # Directory 使确认语义变成“选择文件夹”。扩展选择由下方视图单独开启。
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Drives)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setLabelText(QFileDialog.DialogLabel.Accept, "加入队列")
        for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if dialog.exec() != QFileDialog.Accepted:
            return []
        return dialog.chosen_directories

    def _add_input_dir_from_enter(self, event: object) -> str:
        """允许手动粘贴目录后直接按回车加入，不再需要额外点击添加。"""
        self.add_input_dir()
        return "break"

    def add_input_dir(self) -> bool:
        """将待添加目录放入顺序队列；重复路径只保留一次。"""
        source_text = self.input_var.get().strip()
        if not source_text:
            messagebox.showwarning(WINDOW_TITLE, "请先选择或填写要添加的输入目录。")
            return False
        source = Path(source_text)
        if not source.is_dir():
            messagebox.showwarning(WINDOW_TITLE, "输入目录不存在，请重新选择。")
            return False
        normalized = str(source)
        duplicate_index = next(
            (
                index for index, path in enumerate(self.input_dirs)
                if os.path.normcase(path) == os.path.normcase(normalized)
            ),
            None,
        )
        if duplicate_index is not None:
            self.input_list.selection_clear(0, "end")
            self.input_list.selection_set(duplicate_index)
            self.input_list.see(duplicate_index)
            return True
        self.input_dirs.append(normalized)
        self._refresh_input_list()
        self.input_list.selection_clear(0, "end")
        self.input_list.selection_set("end")
        self.input_list.see("end")
        self._update_result_preview()
        self._save_settings()
        return True

    def remove_selected_input_dirs(self) -> None:
        selected = list(self.input_list.curselection())
        if not selected:
            return
        for index in reversed(selected):
            del self.input_dirs[index]
        self._refresh_input_list()
        self._update_result_preview()
        self._save_settings()

    def clear_input_dirs(self) -> None:
        if not self.input_dirs:
            return
        self.input_dirs.clear()
        self._refresh_input_list()
        self._update_result_preview()
        self._save_settings()

    def _refresh_input_list(self) -> None:
        self.input_list.delete(0, "end")
        for path in self.input_dirs:
            self.input_list.insert("end", path)

    def choose_output(self) -> None:
        initial = self.output_var.get().strip() or str(Path.home())
        selected = filedialog.askdirectory(title="选择输出根目录", initialdir=initial)
        if selected:
            self.output_var.set(selected)
            self._update_result_preview()

    def _actual_output(self) -> Path:
        output_root = Path(self.output_var.get().strip())
        if self.mode_var.get() == "book" and len(self.input_dirs) == 1:
            return output_root / Path(self.input_dirs[0]).name
        return output_root

    def _parallel_setting(self) -> str | None:
        """返回用户选择的并行本数（数字字符串）；选择“自动”时返回 None。"""
        match = re.match(r"^(\d+)\s*本", self.parallel_var.get())
        return match.group(1) if match else None

    @staticmethod
    def _truncate_name(name: str) -> str:
        """截断过长的教材名，保留后段标识（ISBN/年份）更有区分度。"""
        if len(name) <= MAX_BOOK_NAME_LEN:
            return name
        return name[:MAX_BOOK_NAME_LEN - 2] + "…" + name[-3:]

    def start(self) -> None:
        if self.process is not None:
            return
        output_text = self.output_var.get().strip()
        if not CORE_SCRIPT.is_file():
            messagebox.showerror(WINDOW_TITLE, f"未找到核心脚本：\n{CORE_SCRIPT}")
            return
        if not output_text:
            messagebox.showwarning(WINDOW_TITLE, "请选择输出根目录。")
            return
        # 保留旧的“浏览一个目录后直接开始”使用习惯：队列为空时自动加入当前目录。
        if not self.input_dirs and self.input_var.get().strip() and not self.add_input_dir():
            return
        if not self.input_dirs:
            messagebox.showwarning(WINDOW_TITLE, "请至少添加一个输入目录。")
            return
        output_root = Path(output_text)
        if output_root.exists() and not output_root.is_dir():
            messagebox.showwarning(WINDOW_TITLE, "输出根目录不能是文件。")
            return

        self._save_settings()
        self.task_sources = [Path(path) for path in self.input_dirs]
        self.task_index = 0
        self.failed_sources = []
        self.stop_requested = False
        self.paused = False
        self.completed_book_count = 0
        self.total_minor_sum = 0
        self.total_matched_sum = 0
        self._reset_progress()
        self.run_started_at = time.monotonic()
        self.run_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="暂停")
        self.stop_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self._start_next_source()

    def _queue_label(self) -> str:
        total = max(1, len(self.task_sources))
        current = min(total, self.task_index + 1)
        return f"目录 {current}/{total}"

    def _start_next_source(self) -> None:
        """启动队列中当前目录；每项复用原有后台脚本，不改变裁切逻辑。"""
        if self.task_index >= len(self.task_sources):
            self._finish_queue()
            return
        source = self.task_sources[self.task_index]
        if not source.is_dir():
            self.failed_sources.append((source, -2))
            self._append_log(f"[错误] 跳过不存在的输入目录：{source}")
            self.task_index += 1
            self._start_next_source()
            return
        output_root = Path(self.output_var.get().strip())
        self.batch_book_current = 1
        self.batch_book_total = 1
        self.book_progress.clear()
        self.completed_book_count = 0
        self._reset_book_stage_progress()
        command = [
            sys.executable, "-B", "-u", str(CORE_SCRIPT), str(source),
            "--out", str(output_root), "--mode", self.mode_var.get(),
        ]
        if self.vlm_assist_var.get():
            command.append("--vlm-assist")
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        parallel = self._parallel_setting()
        if parallel:
            environment["PAPERCUTTER_BOOK_WORKERS"] = parallel
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._append_log("\n" + "=" * 72)
        self._append_log(f"开始 {self._queue_label()}：{source}")
        extra = f"（并行 {parallel} 本）" if parallel else "（并行按硬件自动）"
        self._append_log("执行：" + " ".join(f'"{part}"' if " " in part else part for part in command) + extra)
        self.status_var.set(f"正在处理{self._queue_label()}，请勿关闭窗口…")
        self.progress_text_var.set(f"当前：{self._queue_label()} · 等待后台启动")
        try:
            self.process = subprocess.Popen(
                command, cwd=str(APP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=environment, creationflags=creationflags,
            )
        except OSError as error:
            self.failed_sources.append((source, -1))
            self._append_log(f"[错误] 无法启动：{error}")
            self.task_index += 1
            self._start_next_source()
            return
        self._refresh_elapsed()
        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self.events.put(("log", line.rstrip("\r\n")))
        self.events.put(("finished", process.wait()))

    def stop(self) -> None:
        if self.process is None:
            return
        if not messagebox.askyesno(WINDOW_TITLE, "确定停止当前任务吗？已经写出的结果会保留。"):
            return
        self.stop_requested = True
        self.status_var.set("正在停止任务…")
        self.stop_button.configure(state="disabled")
        self._terminate_process_tree()

    def _terminate_process_tree(self) -> None:
        """结束切题主进程及其全部子进程（并行切题的 worker）。

        批量切题会在核心脚本内用多进程同时处理多本教材；只结束主进程会让
        worker 变成孤儿进程继续占用 CPU，也会让 GUI 读不到 stdout 的 EOF
        而卡在“正在停止任务…”。taskkill /T 按进程树递归结束全部子进程，
        失败时退回直接结束主进程。
        """
        process = self.process
        if process is None:
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            process.terminate()
        except OSError:
            pass

    def toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self.pause_button.configure(text="暂停")
            self.status_var.set("已恢复运行…")
            self._refresh_elapsed()
            # 暂停期间若当前书已切完，下一本被延迟启动；恢复时补上。
            if self.process is None and self.task_index < len(self.task_sources):
                self._start_next_source()
        else:
            self.paused = True
            self.pause_button.configure(text="继续")
            self.status_var.set("已暂停（后台仍在处理，界面暂不更新）")

    def _drain_events(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if self.paused:
                    if event == "finished":
                        self._on_finished(int(value))
                    # 暂停时仍消费管道避免阻塞子进程，但不更新 UI。
                    # _on_finished 可能启动了下一本书；继续丢弃其输出直到恢复。
                    continue
                if event == "log":
                    line = str(value)
                    if not self._consume_progress_line(line):
                        self._consume_batch_line(line)
                        self._consume_summary_line(line)
                        self._append_log(line)
                elif event == "finished":
                    self._on_finished(int(value))
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _on_finished(self, return_code: int) -> None:
        source = self.task_sources[self.task_index] if self.task_index < len(self.task_sources) else None
        self.process = None
        if return_code == 0:
            for key in self.stage_labels:
                if self.stage_total[key]:
                    self.stage_progress_vars[key].set(100)
                    self.stage_finished_at[key] = time.monotonic()
            self._refresh_stage_texts()
            self._append_log(f"{self._queue_label()} 处理完成。")
        else:
            if source is not None:
                self.failed_sources.append((source, return_code))
            self._append_log(f"{self._queue_label()} 结束，退出码：{return_code}")

        if self.stop_requested:
            self._finish_queue(stopped=True)
            return
        self.task_index += 1
        if self.paused:
            # 暂停时不启动下一本书，恢复时才启动，避免新进程的 stdout 事件被丢弃。
            return
        self._start_next_source()

    def _finish_queue(self, stopped: bool = False) -> None:
        """所有输入目录处理结束后恢复界面控件并显示完成总结。"""
        self.process = None
        self.paused = False
        elapsed = self._format_elapsed(time.monotonic() - self.run_started_at) if self.run_started_at else "-"
        self._refresh_elapsed()
        self.run_button.configure(state="normal")
        self.pause_button.configure(state="disabled", text="暂停")
        self.stop_button.configure(state="disabled")
        self.open_button.configure(state="normal")
        if stopped:
            self.status_var.set("任务已停止；已完成的结果会保留。")
            self.progress_text_var.set("当前：任务已停止")
            self._append_log("队列已停止，后续目录未启动。")
            return
        # 完成总结：把关键统计直接显示在状态栏，一目了然。
        summary_parts = [f"总用时 {elapsed}"]
        if self.total_minor_sum:
            summary_parts.append(f"切题 {self.total_minor_sum} 道")
        if self.total_matched_sum:
            summary_parts.append(f"配对答案 {self.total_matched_sum}")
        if self.failed_sources:
            summary_parts.append(f"失败 {len(self.failed_sources)} 项")
        summary = " · ".join(summary_parts)
        self.status_var.set(f"队列处理完成。{summary}")
        self.progress_text_var.set("当前：全部输入目录处理完成")
        self._append_log("队列处理完成。" + summary)
        if self.failed_sources:
            failures = "；".join(f"{path.name}（退出码 {code}）" for path, code in self.failed_sources)
            self._append_log("失败目录：" + failures)

    def _reset_progress(self) -> None:
        self.progress_phase = ""
        self.progress_total = 0
        self.progress_current = 0
        self.batch_book_current = 1
        self.batch_book_total = 1
        self.saw_answer_scan = False
        self.book_progress.clear()
        self.active_book = ""
        self.book_name_var.set("")
        self.completed_book_count = 0
        self._reset_book_stage_progress()
        self.progress_text_var.set("当前教材：正在启动后台…")
        self.elapsed_var.set("累计用时：00:00:00")

    def _reset_book_stage_progress(self) -> None:
        """重置当前教材的三段进度；批量模式每切换一本教材调用一次。"""
        for key in self.stage_labels:
            self.stage_progress_vars[key].set(0)
            self.stage_text_vars[key].set("等待开始")
            self.stage_started_at[key] = None
            self.stage_finished_at[key] = None
            self.stage_current[key] = 0
            self.stage_total[key] = 0
            self.stage_messages[key] = ""
        self.saw_answer_scan = False

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remain = divmod(total, 3600)
        minutes, seconds = divmod(remain, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _refresh_elapsed(self) -> None:
        if self.run_started_at is None:
            return
        self.elapsed_var.set(f"累计用时：{self._format_elapsed(time.monotonic() - self.run_started_at)}")
        self._refresh_stage_texts()
        if self.process is not None:
            self.root.after(250, self._refresh_elapsed)

    @staticmethod
    def _format_average(seconds: float) -> str:
        """平均耗时只展示已发生的平均值，不预测剩余时间。"""
        if seconds < 60:
            return f"{seconds:.1f} 秒"
        minutes, remain = divmod(int(seconds), 60)
        return f"{minutes}分{remain:02d}秒"

    def _refresh_stage_texts(self) -> None:
        now = time.monotonic()
        running = [n for n, e in self.book_progress.items() if not e.get("done")]
        for key in self.stage_labels:
            total = self.stage_total[key]
            current = self.stage_current[key]
            if total <= 0:
                continue
            ratio = min(1.0, current / total)
            # 并行多本书时，进度条显示的是最近更新那本书的数据；
            # 标注书名避免用户困惑"为什么单元数来回跳"。
            book_tag = ""
            if len(running) > 1 and self.active_book:
                book_tag = self._truncate_name(self.active_book) + " "
            suffix = f"{book_tag}{current}/{total} {self.stage_units[key]} · {ratio:.0%}"
            started_at = self.stage_started_at[key]
            if current > 0 and started_at is not None:
                finished_at = self.stage_finished_at[key] or now
                average = (finished_at - started_at) / current
                suffix += f" · 平均 {self._format_average(average)}/{self.stage_units[key]}"
            elif self.stage_messages[key]:
                suffix += " · 准备中"
            self.stage_text_vars[key].set(suffix)

    def _consume_batch_line(self, line: str) -> None:
        match = re.search(r'^教材\s*\[(\d+)\s*/\s*(\d+)\]', line.strip())
        if not match:
            return
        # 并行模式下这些标题行是主进程预先列出的全部教材，不代表当前正在处理哪本，
        # 因此只记录总数用于展示，不再在这里重置进度条。
        self.batch_book_total = max(1, int(match.group(2)))
        self.batch_book_current = max(1, int(match.group(1)))

    def _consume_summary_line(self, line: str) -> None:
        """从 process_book 的完成行提取统计，供队列完成总结使用。"""
        match = re.search(
            r'完成:\s*(\d+)道小题,\s*答案配对(\d+)\s*', line,
        )
        if match:
            self.total_minor_sum += int(match.group(1))
            self.total_matched_sum += int(match.group(2))

    def _consume_progress_line(self, line: str) -> bool:
        prefix = "@@PROGRESS@@"
        if not line.startswith(prefix):
            return False
        try:
            payload = json.loads(line[len(prefix):])
            phase = str(payload.get("phase") or "")
            current = max(0, int(payload.get("current") or 0))
            total = max(0, int(payload.get("total") or 0))
            message = str(payload.get("message") or "正在处理")
            book = str(payload.get("book") or "")
        except (ValueError, TypeError):
            return True

        self.progress_phase = phase
        self.progress_current = current
        self.progress_total = total
        stage_key = {
            "answer_scan": "answer_scan",
            "page_scan": "page_scan",
            "units": "unit",
            "unit": "unit",
        }.get(phase)

        # 按书记录各自进度，多本并行时主进度条跟随最近更新的一本。
        if book:
            now = time.monotonic()
            entry = self.book_progress.setdefault(
                book, {"done": False, "display": "", "stages": {}, "t": now,
                       "stage_started": {}, "stage_finished": {}},
            )
            entry["t"] = now
            entry["display"] = f"{message} {current}/{total}" if total else message
            if stage_key is not None:
                entry["stages"][stage_key] = {"current": current, "total": total}
                # 每本书记录各自的阶段计时，不跨书混算。
                if stage_key not in entry["stage_started"]:
                    entry["stage_started"][stage_key] = now
                entry["stage_finished"][stage_key] = now
            if book != self.active_book:
                self.active_book = book
                self.saw_answer_scan = False
                # 书名标签更新，解决“进度条看不出是哪本书”。
                self.book_name_var.set(self._truncate_name(book))
                # 切书时进度条跟随新书，但保留旧书的完成记录。
                for name in [n for n in self.book_progress if n != book]:
                    if self.book_progress[name].get("done"):
                        del self.book_progress[name]
            # 并行时平均耗时用当前书的独立计时，避免跨书混算。
            for key in self.stage_labels:
                stage_entry = entry.get("stage_started", {}).get(key)
                if stage_entry is not None:
                    self.stage_started_at[key] = stage_entry

        if phase in {"complete", "book_skip"}:
            if book:
                self.book_progress[book]["done"] = True
                self.completed_book_count += 1
                if self.active_book == book:
                    if phase == "book_skip":
                        for key in self.stage_labels:
                            self.stage_progress_vars[key].set(100)
                            self.stage_text_vars[key].set("已跳过（已有完整结果）")
                    self._sync_bars_to_active_book()
            else:
                for key in self.stage_labels:
                    if self.stage_total[key]:
                        self.stage_current[key] = self.stage_total[key]
                        self.stage_progress_vars[key].set(100)
                        self.stage_finished_at[key] = time.monotonic()
                self.completed_book_count = max(self.completed_book_count, 1)
            self._refresh_stage_texts()
            self.progress_text_var.set(self._build_progress_text())
            return True
        if stage_key is None:
            self.progress_text_var.set(self._build_progress_text())
            return True

        # 当前流程会优先使用“答案”目录；平铺教材则在页面识别时逐页判定答案页。
        # 旧的末尾预扫描未触发不代表教材没有答案，界面应说明实际的发现方式。
        if stage_key == "page_scan" and not self.saw_answer_scan:
            self.stage_progress_vars["answer_scan"].set(100)
            self.stage_text_vars["answer_scan"].set("由答案目录或页面识别自动确定")
        if stage_key == "answer_scan":
            self.saw_answer_scan = True

        self.stage_current[stage_key] = current
        self.stage_total[stage_key] = total
        self.stage_messages[stage_key] = message
        ratio = min(1.0, current / total) if total else 0.0
        self.stage_progress_vars[stage_key].set(ratio * 100.0)
        if total and current >= total:
            self.stage_finished_at[stage_key] = time.monotonic()
        self._refresh_stage_texts()
        self.progress_text_var.set(self._build_progress_text())
        return True

    def _build_progress_text(self) -> str:
        """多本并行时每本一行；书名截断避免窗口放不下。"""
        running = [
            (name, entry) for name, entry in self.book_progress.items()
            if not entry.get("done")
        ]
        done = [name for name, entry in self.book_progress.items() if entry.get("done")]
        total = max(self.batch_book_total, self.completed_book_count + len(running))

        parts = []
        for name, entry in self.book_progress.items():
            short = self._truncate_name(name)
            if entry.get("done"):
                continue
            if entry.get("display"):
                parts.append(f"{short}：{entry['display']}")
        if done:
            if len(done) <= 2:
                parts.append("已完成：" + "、".join(self._truncate_name(n) for n in done))
            else:
                parts.append(f"已完成 {len(done)} 本")
        if not parts:
            return f"当前：{self._queue_label()} · 等待进度"

        count_hint = ""
        if total > 1:
            count_hint = f"已完成 {self.completed_book_count}/{total} 本 · "
        return count_hint + "　".join(parts)

    def _sync_bars_to_active_book(self) -> None:
        """某本教材完成后，把主进度条和书名标签切到仍在运行的另一本。"""
        running = [
            name for name, entry in self.book_progress.items()
            if not entry.get("done")
        ]
        if not running:
            for key in self.stage_labels:
                if self.stage_total[key]:
                    self.stage_progress_vars[key].set(100)
            return
        self.active_book = running[0]
        self.book_name_var.set(self._truncate_name(running[0]))
        entry = self.book_progress[self.active_book]
        for key in self.stage_labels:
            data = entry.get("stages", {}).get(key)
            if data and data.get("total"):
                self.stage_current[key] = data["current"]
                self.stage_total[key] = data["total"]
                ratio = min(1.0, data["current"] / data["total"])
                self.stage_progress_vars[key].set(ratio * 100.0)

    def open_result(self) -> None:
        target = self._actual_output()
        if not target.exists():
            messagebox.showinfo(WINDOW_TITLE, f"结果目录尚不存在：\n{target}")
            return
        os.startfile(str(target))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        # 防止长时间批量运行时日志控件无限膨胀导致界面卡顿。
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            delete_to = f"{line_count - MAX_LOG_LINES}.0"
            self.log.delete("1.0", delete_to)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _load_settings(self) -> None:
        settings_path = SETTINGS_FILE if SETTINGS_FILE.is_file() else LEGACY_SETTINGS_FILE
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            self.input_var.set(str(data.get("input_dir") or ""))
            self.output_var.set(str(data.get("output_dir") or ""))
            mode = str(data.get("mode") or "book")
            self.mode_var.set(mode if mode in {"book", "year"} else "book")
            saved_dirs = data.get("input_dirs")
            if isinstance(saved_dirs, list):
                self.input_dirs = [
                    str(Path(path)) for path in saved_dirs
                    if isinstance(path, str) and path.strip()
                ]
            elif self.input_var.get().strip():
                # 兼容旧版只保存单个 input_dir 的设置。
                self.input_dirs = [self.input_var.get().strip()]
            parallel = str(data.get("parallel") or PARALLEL_OPTIONS[0])
            self.parallel_var.set(parallel if parallel in PARALLEL_OPTIONS else PARALLEL_OPTIONS[0])
            self.vlm_assist_var.set(bool(data.get("vlm_assist", False)))
        except (OSError, ValueError, TypeError):
            pass
        self._refresh_input_list()
        self._update_result_preview()

    def _save_settings(self) -> None:
        data = {
            "input_dir": self.input_var.get().strip(),
            "input_dirs": self.input_dirs,
            "output_dir": self.output_var.get().strip(),
            "mode": self.mode_var.get(),
            "parallel": self.parallel_var.get(),
            "vlm_assist": bool(self.vlm_assist_var.get()),
        }
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(WINDOW_TITLE, "任务仍在运行。确定停止并关闭吗？"):
                return
            self._terminate_process_tree()
        self.root.destroy()


def main() -> None:
    """Launch the Qt workspace; retain the legacy Tk window as a safe fallback.

    Some older copied deployments may not yet include PySide6.  They continue
    to work with the original interface instead of failing on startup.
    """
    try:
        from cut_questions_gui_qt import main as qt_main
        qt_main()
        return
    except Exception:
        # The VBS launcher is intentionally hidden for office users.  Preserve
        # a readable traceback and keep the known-good Tk interface available
        # when a new Qt DLL/font/plugin is unavailable on a copied deployment.
        error_log = APP_DIR / "logs" / "gui_startup_error.log"
        try:
            error_log.parent.mkdir(parents=True, exist_ok=True)
            error_log.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        root = tk.Tk()
        CutterGui(root)
        root.mainloop()


if __name__ == "__main__":
    main()
