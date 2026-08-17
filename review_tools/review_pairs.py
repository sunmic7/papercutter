#!/usr/bin/env python3
"""题目—答案并排审核工具，读写 review_queue.py 定义的统一协议。"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import review_queue


DECISION_LABELS = {
    'accept': '通过',
    'reject_pair': '配对错误',
    'question_bad': '题图有问题',
    'answer_bad': '答案图有问题',
    'needs_manual': '需进一步处理',
}

SHORTCUT_LABELS = {
    'accept': 'A', 'reject_pair': 'R', 'question_bad': 'Q',
    'answer_bad': 'W', 'needs_manual': 'M',
}


def _resolve(root, relative_path):
    if not relative_path:
        return None
    return os.path.join(root, relative_path.replace('/', os.sep))


def _photo_from_path(path, max_width, max_height):
    image = cv2.imread(path) if path else None
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        return None
    return tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'))


class BoxCorrectionDialog(tk.Toplevel):
    """在来源原页上重画题框或答案框，坐标直接写入审核决定。"""
    def __init__(self, parent, image_path, initial_box, title, on_save):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.grab_set()
        self.on_save = on_save
        self.image = cv2.imread(image_path) if image_path else None
        if self.image is None:
            messagebox.showerror(title, f'无法读取来源图片：\n{image_path}', parent=parent)
            self.destroy()
            return
        height, width = self.image.shape[:2]
        self.scale = min(900 / max(1, width), 650 / max(1, height), 1.0)
        display_width = max(1, int(width * self.scale))
        display_height = max(1, int(height * self.scale))
        shown = cv2.resize(
            self.image, (display_width, display_height), interpolation=cv2.INTER_AREA,
        ) if self.scale < 1.0 else self.image
        ok, encoded = cv2.imencode('.png', shown)
        if not ok:
            self.destroy()
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'))
        self.canvas = tk.Canvas(self, width=display_width, height=display_height, cursor='cross')
        self.canvas.pack(padx=8, pady=8)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.rectangle = None
        self.start = None
        self.box = list(initial_box or [])
        if len(self.box) == 4:
            self.rectangle = self.canvas.create_rectangle(
                *(value * self.scale for value in self.box),
                outline='#ff7a00', width=3,
            )
        self.canvas.bind('<Button-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<ButtonRelease-1>', self._release)
        footer = ttk.Frame(self)
        footer.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(footer, text='在原页拖出正确范围；橙框是原预测，绿框是新标注。').pack(side='left')
        ttk.Button(footer, text='保存坐标', command=self._save).pack(side='right')

    def _press(self, event):
        self.start = (event.x, event.y)
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline='#00a060', width=3,
        )

    def _drag(self, event):
        if self.start and self.rectangle is not None:
            self.canvas.coords(self.rectangle, self.start[0], self.start[1], event.x, event.y)

    def _release(self, event):
        if not self.start:
            return
        x0, x1 = sorted((self.start[0], event.x))
        y0, y1 = sorted((self.start[1], event.y))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            self.box = [
                int(round(x0 / self.scale)), int(round(y0 / self.scale)),
                int(round(x1 / self.scale)), int(round(y1 / self.scale)),
            ]
        self.start = None

    def _save(self):
        if len(self.box) != 4 or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            messagebox.showwarning(self.title(), '请先拖出有效矩形。', parent=self)
            return
        self.on_save(self.box)
        self.destroy()


class PairReviewApp:
    def __init__(self, root, queue_path):
        self.root = root
        self.root.title('题目—答案审核')
        self.root.geometry('1180x820')
        self.root.minsize(920, 650)
        self.queue_path = os.path.abspath(queue_path)
        self.unit_root = os.path.dirname(self.queue_path)
        self.decision_path = os.path.join(self.unit_root, 'review_decisions.json')
        self.queue = review_queue.load_json(self.queue_path)
        review_queue.validate_review_queue(self.queue)
        self.document = self._load_decisions()
        self.decisions = {
            record['review_id']: record for record in self.document.get('decisions', [])
        }
        self.corrections = {
            record['review_id']: {
                key: record[key] for key in ('corrected_question_box', 'corrected_answer_box')
                if key in record
            }
            for record in self.document.get('decisions', [])
        }
        self.index = 0
        self.question_photo = None
        self.answer_photo = None
        self.show_auto_var = tk.BooleanVar(value=False)
        self.note_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.items = []
        self._build_ui()
        self._refresh_items()
        self.root.bind('<Key-a>', lambda _event: self.record('accept'))
        self.root.bind('<Key-r>', lambda _event: self.record('reject_pair'))
        self.root.bind('<Key-q>', lambda _event: self.record('question_bad'))
        self.root.bind('<Key-w>', lambda _event: self.record('answer_bad'))
        self.root.bind('<Key-m>', lambda _event: self.record('needs_manual'))
        self.root.bind('<Right>', lambda _event: self.move(1))
        self.root.bind('<Left>', lambda _event: self.move(-1))

    def _load_decisions(self):
        if not os.path.isfile(self.decision_path):
            return review_queue.new_decision_document(self.queue, 'human', '本地审核')
        document = review_queue.load_json(self.decision_path)
        review_queue.validate_decisions(self.queue, document)
        return document

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill='both', expand=True)
        header = ttk.Frame(container)
        header.pack(fill='x')
        ttk.Label(header, text=f"单元：{self.queue.get('unit_name', '')}").pack(side='left')
        ttk.Checkbutton(
            header, text='显示已自动通过项', variable=self.show_auto_var,
            command=self._refresh_items,
        ).pack(side='left', padx=18)
        ttk.Label(header, textvariable=self.summary_var).pack(side='right')

        images = ttk.Frame(container)
        images.pack(fill='both', expand=True, pady=(10, 8))
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=1)
        images.rowconfigure(1, weight=1)
        ttk.Label(images, text='题目').grid(row=0, column=0)
        ttk.Label(images, text='答案候选').grid(row=0, column=1)
        self.question_label = ttk.Label(images, anchor='center', relief='solid')
        self.answer_label = ttk.Label(images, anchor='center', relief='solid')
        self.question_label.grid(row=1, column=0, sticky='nsew', padx=(0, 5))
        self.answer_label.grid(row=1, column=1, sticky='nsew', padx=(5, 0))

        ttk.Label(container, textvariable=self.status_var, wraplength=1120).pack(fill='x')
        note_row = ttk.Frame(container)
        note_row.pack(fill='x', pady=(8, 8))
        ttk.Label(note_row, text='备注：').pack(side='left')
        ttk.Entry(note_row, textvariable=self.note_var).pack(side='left', fill='x', expand=True)
        ttk.Button(note_row, text='重画题框', command=lambda: self.correct_box('question')).pack(side='left', padx=(8, 0))
        ttk.Button(note_row, text='重画答案框', command=lambda: self.correct_box('answer')).pack(side='left', padx=(6, 0))

        actions = ttk.Frame(container)
        actions.pack(fill='x')
        ttk.Button(actions, text='← 上一项', command=lambda: self.move(-1)).pack(side='left')
        ttk.Button(actions, text='下一项 →', command=lambda: self.move(1)).pack(side='left', padx=(6, 18))
        for decision, label in DECISION_LABELS.items():
            ttk.Button(
                actions, text=f'{label} ({SHORTCUT_LABELS[decision]})',
                command=lambda value=decision: self.record(value),
            ).pack(side='left', padx=3)
        ttk.Button(actions, text='应用审核结果', command=self.apply).pack(side='right')

    def _refresh_items(self):
        current_id = self.items[self.index]['review_id'] if self.items else None
        all_items = self.queue['items']
        if self.show_auto_var.get():
            self.items = list(all_items)
        else:
            self.items = [
                item for item in all_items
                if item.get('pairing', {}).get('decision_band') != 'auto'
                or item['review_id'] in self.decisions
            ]
            self.items.sort(key=lambda item: -int(item.get('review_priority', 0)))
        self.index = 0
        if current_id:
            for index, item in enumerate(self.items):
                if item['review_id'] == current_id:
                    self.index = index
                    break
        self.render()

    def render(self):
        total = len(self.items)
        reviewed = len(self.decisions)
        pending_total = self.queue.get('summary', {}).get('needs_review', 0)
        self.summary_var.set(f'待核重点 {pending_total} 项｜已记录 {reviewed} 项')
        if not self.items:
            self.question_label.configure(image='', text='没有需要审核的项目')
            self.answer_label.configure(image='', text='')
            self.status_var.set('可勾选“显示已自动通过项”进行抽检。')
            return
        self.index = max(0, min(self.index, total - 1))
        item = self.items[self.index]
        question_path = _resolve(self.unit_root, item.get('question', {}).get('image'))
        answer_path = _resolve(
            self.unit_root, (item.get('answer') or {}).get('proposed_image'),
        )
        self.question_photo = _photo_from_path(question_path, 540, 540)
        self.answer_photo = _photo_from_path(answer_path, 540, 540)
        self.question_label.configure(
            image=self.question_photo or '', text='' if self.question_photo else '题图读取失败',
        )
        self.answer_label.configure(
            image=self.answer_photo or '', text='' if self.answer_photo else '没有答案候选',
        )
        pairing = item.get('pairing', {})
        previous = self.decisions.get(item['review_id'])
        self.note_var.set((previous or {}).get('notes', ''))
        decision_text = DECISION_LABELS.get((previous or {}).get('decision'), '未审核')
        correction = self.corrections.get(item['review_id'], {})
        correction_text = ''
        if correction:
            correction_text = '　已重画：' + '、'.join(
                '题框' if key == 'corrected_question_box' else '答案框'
                for key in correction
            )
        reasons = '；'.join(pairing.get('reasons', []))
        self.status_var.set(
            f"{self.index + 1}/{total}　{item.get('major', {}).get('title', '')} "
            f"第{item.get('major', {}).get('sequence', '?')}题　"
            f"置信度 {pairing.get('score', 0):.2f}　当前：{decision_text}{correction_text}\n{reasons}"
        )

    def move(self, offset):
        if self.items:
            self.index = (self.index + offset) % len(self.items)
            self.render()

    def record(self, decision):
        if not self.items:
            return
        item = self.items[self.index]
        if decision == 'accept' and not item.get('answer'):
            messagebox.showinfo(
                '题目—答案审核',
                '这一项没有答案候选，不能标记为“通过”。请标记题图问题或需进一步处理。',
            )
            return
        record = {
            'review_id': item['review_id'],
            'decision': decision,
            'notes': self.note_var.get().strip(),
        }
        record.update(self.corrections.get(item['review_id'], {}))
        self.decisions[item['review_id']] = record
        self.document['decisions'] = list(self.decisions.values())
        self.document['reviewer'] = {'type': 'human', 'name': '本地审核'}
        review_queue.validate_decisions(self.queue, self.document)
        review_queue.write_json_atomic(self.decision_path, self.document)
        if self.index + 1 < len(self.items):
            self.index += 1
        self.render()

    def correct_box(self, target):
        if not self.items:
            return
        item = self.items[self.index]
        if target == 'question':
            data = item.get('question') or {}
            image_path = data.get('source_image')
            initial_box = data.get('source_box')
            field = 'corrected_question_box'
            title = '重画题目来源框'
        else:
            data = item.get('answer') or {}
            image_path = data.get('source_image')
            initial_box = data.get('source_box')
            field = 'corrected_answer_box'
            title = '重画答案来源框'
        if not image_path or not initial_box:
            messagebox.showinfo('题目—答案审核', '这一项没有可用的来源图片或原始框。')
            return

        def save(box):
            self.corrections.setdefault(item['review_id'], {})[field] = box
            self.render()

        BoxCorrectionDialog(self.root, image_path, initial_box, title, save)

    def apply(self):
        try:
            summary = review_queue.apply_decisions(
                self.unit_root, self.queue, self.document,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror('题目—答案审核', f'应用失败：\n{error}')
            return
        messagebox.showinfo(
            '题目—答案审核',
            f"已应用：通过 {summary['accepted']}，驳回 {summary['rejected']}，"
            f"标记 {summary['marked']}，缺失文件 {summary['missing_file']}。",
        )


def choose_queue():
    return filedialog.askopenfilename(
        title='选择 review_queue.json',
        filetypes=[('审核清单', 'review_queue.json'), ('JSON', '*.json')],
    )


def main():
    parser = argparse.ArgumentParser(description='题目—答案并排审核工具')
    parser.add_argument('queue', nargs='?', help='review_queue.json 路径')
    args = parser.parse_args()
    root = tk.Tk()
    queue_path = args.queue or choose_queue()
    if not queue_path:
        root.destroy()
        return
    try:
        PairReviewApp(root, queue_path)
    except (OSError, ValueError) as error:
        messagebox.showerror('题目—答案审核', str(error))
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    main()
