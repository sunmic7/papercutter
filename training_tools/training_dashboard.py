#!/usr/bin/env python3
"""Small local GUI for CPU question-detector training supervision."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:  # Supports both ``python training_dashboard.py`` and package imports.
    from . import training_control
except ImportError:
    import training_control


class TrainingDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("切题模型训练控制台")
        self.geometry("860x620")
        self.minsize(720, 480)
        self.status_var = tk.StringVar(value="正在读取训练状态…")
        self.metrics_var = tk.StringVar(value="尚无训练指标")
        self.epochs_var = tk.StringVar(value="5")
        self.imgsz_var = tk.StringVar(value="640")
        self.batch_var = tk.StringVar(value="2")
        self.kind_var = tk.StringVar(value="题目框（HiLEx）")
        self._build()
        self.refresh()
        self.after(2500, self._poll)

    def _build(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        ttk.Label(controls, text="训练目标").grid(row=0, column=0, sticky="w")
        ttk.Combobox(controls, width=18, textvariable=self.kind_var, values=("题目框（HiLEx）", "答案区域（HiLEx）", "答案框（审核样本）"), state="readonly").grid(row=0, column=1, padx=(4, 12))
        ttk.Label(controls, text="轮数").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, width=7, textvariable=self.epochs_var).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(controls, text="分辨率").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, width=7, textvariable=self.imgsz_var).grid(row=0, column=5, padx=(4, 12))
        ttk.Label(controls, text="Batch").grid(row=0, column=6, sticky="w")
        ttk.Entry(controls, width=7, textvariable=self.batch_var).grid(row=0, column=7, padx=(4, 12))
        ttk.Button(controls, text="启动训练", command=self.start).grid(row=0, column=8, padx=(8, 4))
        ttk.Button(controls, text="断点继续", command=self.resume).grid(row=0, column=9, padx=4)
        ttk.Button(controls, text="刷新", command=self.refresh).grid(row=0, column=10, padx=4)
        ttk.Button(controls, text="停止训练", command=self.stop).grid(row=0, column=11, padx=4)
        ttk.Label(controls, textvariable=self.status_var, foreground="#1f4f82").grid(
            row=1, column=0, columnspan=12, sticky="w", pady=(10, 2),
        )
        ttk.Label(controls, textvariable=self.metrics_var).grid(row=2, column=0, columnspan=12, sticky="w")

        ttk.Label(self, text="训练日志").pack(anchor="w", padx=12, pady=(4, 0))
        container = ttk.Frame(self, padding=(12, 4, 12, 12))
        container.pack(fill="both", expand=True)
        self.log = tk.Text(container, wrap="word", state="disabled", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(container, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _poll(self) -> None:
        self.refresh()
        self.after(2500, self._poll)

    def start(self) -> None:
        try:
            result = training_control.start(
                epochs=int(self.epochs_var.get()), imgsz=int(self.imgsz_var.get()),
                batch=int(self.batch_var.get()), workers=4,
                kind=(
                    "answer_hilex" if self.kind_var.get() == "答案区域（HiLEx）"
                    else "answer" if self.kind_var.get().startswith("答案")
                    else "question"
                ),
            )
            self._show_status(result)
        except Exception as exc:
            messagebox.showerror("无法启动训练", str(exc))

    def stop(self) -> None:
        current = training_control.status()
        if not current.get("active"):
            self.refresh()
            return
        if messagebox.askyesno("确认停止", "停止会丢失当前未保存轮次的进度，确定要停止吗？"):
            training_control.stop()
            self.refresh()

    def resume(self) -> None:
        current = training_control.status()
        if current.get("active"):
            messagebox.showinfo("训练仍在运行", "当前训练任务仍在运行，无需断点恢复。")
            return
        run_dir = Path(current.get("run_dir", ""))
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            messagebox.showerror("无法断点继续", "未找到可恢复的 weights\\last.pt。")
            return
        if not messagebox.askyesno(
            "确认断点继续",
            f"将从已保存的断点继续，而不是从头训练：\n{checkpoint}\n\n"
            "已完整结束的轮次会保留，当前未保存的轮次不会恢复。是否继续？",
        ):
            return
        try:
            result = training_control.start(
                epochs=int(self.epochs_var.get()), imgsz=int(self.imgsz_var.get()),
                batch=int(self.batch_var.get()), workers=4,
                kind=current.get("kind", "question"), resume=checkpoint,
            )
            self._show_status(result)
        except Exception as exc:
            messagebox.showerror("无法断点继续", str(exc))

    def _show_status(self, state: dict) -> None:
        active = state.get("active", False)
        pid = state.get("pid", "-")
        run_dir = state.get("run_dir", "")
        self.status_var.set(("训练中" if active else "未运行") + f"  |  PID: {pid}  |  {run_dir}")
        metrics = state.get("metrics") or {}
        if metrics:
            self.metrics_var.set(
                "Epoch {epoch}  |  Precision {precision}  |  Recall {recall}  |  mAP50 {map50}  |  mAP50-95 {map95}".format(
                    epoch=metrics.get("epoch", "-"), precision=metrics.get("metrics/precision(B)", "-"),
                    recall=metrics.get("metrics/recall(B)", "-"), map50=metrics.get("metrics/mAP50(B)", "-"),
                    map95=metrics.get("metrics/mAP50-95(B)", "-"),
                )
            )
        else:
            self.metrics_var.set("尚未写入首轮指标；CPU 首轮通常需要数分钟。")
        log_path = Path(state.get("log_path", "")) if state.get("log_path") else None
        text = ""
        if log_path and log_path.is_file():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            except OSError:
                pass
        elif run_dir:
            text = "此训练不是由控制台启动，无法回收历史标准输出；可查看实时指标和 results.csv。"
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("1.0", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh(self) -> None:
        self._show_status(training_control.status())


if __name__ == "__main__":
    TrainingDashboard().mainloop()
