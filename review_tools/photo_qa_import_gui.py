#!/usr/bin/env python3
"""Independent desktop launcher for long-running photographed-QA imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("photo_qa_worker.py")


class PhotoQaImportApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PaperFlow · 拍照 QA 后台导入")
        self.root.geometry("760x510")
        self.roots: list[str] = []
        self.process: subprocess.Popen | None = None
        self.job_dir: Path | None = None
        self.semantic = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="请选择三套已命名题目图片目录。")
        self._build()

    def _build(self) -> None:
        frame = tk.Frame(self.root, padx=18, pady=16); frame.pack(fill="both", expand=True)
        tk.Label(frame, text="拍照 QA 后台导入（不切题）", font=("Microsoft YaHei", 15, "bold")).pack(anchor="w")
        tk.Label(frame, text="题目与答案按名称形成候选关系，后台任务会保留原图并输出正式 JSON 与审核清单。", wraplength=700, justify="left").pack(anchor="w", pady=(7, 12))
        list_frame = tk.Frame(frame); list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_frame, height=12); self.listbox.pack(side="left", fill="both", expand=True)
        side = tk.Frame(list_frame); side.pack(side="left", fill="y", padx=(9, 0))
        tk.Button(side, text="添加题目目录", command=self.add_root).pack(fill="x", pady=(0, 6))
        tk.Button(side, text="移除选中项", command=self.remove_selected).pack(fill="x")
        tk.Checkbutton(frame, text="使用 Qwen 视觉模型做题答归属与中文解析（会发送所选题图）", variable=self.semantic).pack(anchor="w", pady=(12, 3))
        tk.Label(frame, textvariable=self.status, fg="#38506b", wraplength=700, justify="left").pack(anchor="w", pady=(4, 11))
        actions = tk.Frame(frame); actions.pack(fill="x")
        self.start_button = tk.Button(actions, text="启动后台导入", command=self.start, bg="#1677ff", fg="white")
        self.start_button.pack(side="left")
        tk.Button(actions, text="打开本次结果", command=self.open_results).pack(side="left", padx=8)

    def add_root(self) -> None:
        path = filedialog.askdirectory(title="选择一套已命名题目图片目录")
        if not path or path in self.roots:
            return
        self.roots.append(path); self.listbox.insert("end", path)
        self.status.set(f"已选择 {len(self.roots)} 套目录；建议选择 3 套。")

    def remove_selected(self) -> None:
        selected = self.listbox.curselection()
        if not selected:
            return
        index = selected[0]; self.listbox.delete(index); self.roots.pop(index)

    def start(self) -> None:
        if not self.roots:
            messagebox.showwarning("未选择目录", "请至少选择一套题目图片目录。")
            return
        if self.semantic.get() and not messagebox.askokcancel("确认发送图片", "将把题图发送给当前配置的 Qwen 服务，用于识别答案续页、解析配图和题答归属。是否继续？"):
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_dir = ROOT / "runtime" / "photo_qa_jobs" / stamp
        command = [sys.executable, str(WORKER), "--output", str(self.job_dir)]
        if self.semantic.get():
            command.append("--semantic-review")
        command.extend(self.roots)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(command, cwd=str(ROOT), creationflags=flags)
        self.start_button.configure(state="disabled")
        self.status.set("后台任务已启动；可关闭本窗口，任务仍会继续。")
        self.root.after(1000, self.poll)

    def poll(self) -> None:
        if not self.job_dir:
            return
        status_file = self.job_dir / "job_status.json"
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            message = str(status.get("message", "后台任务运行中…"))
            warning = str(status.get("warning", ""))
            self.status.set(message + ("\n" + warning if warning else ""))
            if status.get("state") in {"completed", "failed"}:
                self.start_button.configure(state="normal")
                return
        except (OSError, json.JSONDecodeError):
            pass
        self.root.after(1000, self.poll)

    def open_results(self) -> None:
        if self.job_dir and self.job_dir.exists():
            os.startfile(str(self.job_dir))
        else:
            messagebox.showinfo("尚无结果", "请先启动一次后台导入任务。")


def main() -> None:
    window = tk.Tk(); PhotoQaImportApp(window); window.mainloop()


if __name__ == "__main__":
    main()
