#!/usr/bin/env python3
"""原页题框标注工具：绘制、合并、删除题框并保存为真值清单。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import review_queue


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def _image_sort_key(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def create_annotation_manifest(image_dir, profile='general'):
    """扫描原教材图片目录，生成可直接在本工具打开的空标注清单。"""
    root = os.path.abspath(image_dir)
    pages = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(directory for directory in dirs if not directory.startswith('_'))
        for name in sorted(files, key=_image_sort_key):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(base, name)
            pages.append({
                'image': os.path.relpath(path, root).replace('\\', '/'),
                'profile': profile,
                'status': 'unlabeled',
                'questions': [],
            })
    return {
        'schema_version': 1,
        'image_root': root,
        'coordinate_format': 'xyxy_pixels',
        'pages': pages,
    }


def resolve_image_root(manifest, manifest_path):
    root = manifest.get('image_root') or os.path.dirname(os.path.abspath(manifest_path))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), root)
    return os.path.abspath(root)


def page_image_path(manifest, manifest_path, page):
    path = page.get('image', '')
    if os.path.isabs(path):
        return path
    return os.path.join(resolve_image_root(manifest, manifest_path), path)


def _normalize_path(path):
    return os.path.normcase(os.path.abspath(path))


def import_prediction_metadata(manifest, manifest_path, output_root):
    """把正式结果中的 metadata.json 题框导入为空白标注建议，不自动标记完成。"""
    by_source = {}
    invalid_files = 0
    for base, _, files in os.walk(output_root):
        if 'metadata.json' not in files:
            continue
        metadata_path = os.path.join(base, 'metadata.json')
        try:
            with open(metadata_path, 'r', encoding='utf-8') as file:
                metadata = json.load(file)
            source = metadata.get('source', {})
            source_image = source.get('image', '')
            box = source.get('question_box', [])
            if not source_image or not (
                isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]
            ):
                raise ValueError('来源图片或题框无效')
            by_source.setdefault(_normalize_path(source_image), []).append({
                'bbox': [int(round(value)) for value in box],
                'suggested': True,
                'source_metadata': metadata_path,
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid_files += 1

    imported_pages = 0
    imported_boxes = 0
    for page in manifest.get('pages', []):
        source_key = _normalize_path(page_image_path(manifest, manifest_path, page))
        suggestions = by_source.get(source_key, [])
        if not suggestions:
            continue
        existing = {
            tuple(question.get('bbox', []))
            for question in page.get('questions', [])
        }
        added = []
        for suggestion in suggestions:
            box_key = tuple(suggestion['bbox'])
            if box_key in existing:
                continue
            existing.add(box_key)
            added.append(suggestion)
        if added:
            page.setdefault('questions', []).extend(added)
            page['status'] = 'unlabeled'
            imported_pages += 1
            imported_boxes += len(added)
    return {
        'pages': imported_pages,
        'boxes': imported_boxes,
        'invalid_metadata_files': invalid_files,
    }


class AnnotationApp:
    CANVAS_WIDTH = 1050
    CANVAS_HEIGHT = 760

    def __init__(self, root, manifest_path=''):
        self.root = root
        self.root.title('切题精度标注工具')
        self.root.geometry('1360x900')
        self.root.minsize(1050, 720)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.manifest = None
        self.manifest_path = ''
        self.page_index = 0
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.image_origin = (0.0, 0.0)
        self.drag_start = None
        self.drag_item = None
        self.dirty = False

        self.mode_var = tk.StringVar(value='draw')
        self.status_var = tk.StringVar(value='请打开标注清单。')
        self.page_var = tk.StringVar(value='未打开')
        self.labeled_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.bind('<Control-s>', lambda _event: self.save_manifest())
        self.root.bind('<Left>', lambda _event: self.previous_page())
        self.root.bind('<Right>', lambda _event: self.next_page())
        self.root.bind('<Delete>', lambda _event: self.delete_selected())
        if manifest_path:
            self.open_manifest(manifest_path)

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='新建标注清单…', command=self.new_manifest).pack(side='left')
        ttk.Button(toolbar, text='打开标注清单…', command=self.choose_manifest).pack(side='left')
        ttk.Button(toolbar, text='保存', command=self.save_manifest).pack(side='left', padx=(6, 0))
        ttk.Button(toolbar, text='导入现有切题框…', command=self.import_predictions).pack(side='left', padx=(12, 0))
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=12)
        ttk.Radiobutton(toolbar, text='画题框', variable=self.mode_var, value='draw').pack(side='left')
        ttk.Radiobutton(toolbar, text='选择题框', variable=self.mode_var, value='select').pack(side='left', padx=(8, 0))
        ttk.Button(toolbar, text='合并选中', command=self.merge_selected).pack(side='left', padx=(12, 0))
        ttk.Button(toolbar, text='删除选中', command=self.delete_selected).pack(side='left', padx=(6, 0))
        ttk.Checkbutton(
            toolbar, text='本页已核验', variable=self.labeled_var,
            command=self.toggle_labeled,
        ).pack(side='right')

        body = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            body, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT,
            background='#3b3b3b', highlightthickness=0, cursor='crosshair',
        )
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<ButtonPress-1>', self.on_canvas_press)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)

        side = ttk.Frame(body, padding=(10, 0, 0, 0), width=250)
        side.grid(row=0, column=1, sticky='ns')
        ttk.Label(side, textvariable=self.page_var, wraplength=240).pack(fill='x')
        nav = ttk.Frame(side)
        nav.pack(fill='x', pady=(8, 10))
        ttk.Button(nav, text='← 上一页', command=self.previous_page).pack(side='left')
        ttk.Button(nav, text='下一页 →', command=self.next_page).pack(side='right')
        ttk.Label(side, text='题框（可多选后合并）：').pack(anchor='w')
        self.box_list = tk.Listbox(side, selectmode='extended', width=34, height=28)
        self.box_list.pack(fill='both', expand=True, pady=(5, 8))
        self.box_list.bind('<<ListboxSelect>>', lambda _event: self.redraw_overlays())
        ttk.Button(side, text='本页无题目并完成核验', command=self.mark_empty_labeled).pack(fill='x')
        ttk.Label(
            side,
            text='操作：拖动画框；选择多个框可合并。若一道题被多裁，可合并；若少裁，删除大框后重画。方向键翻页，Delete 删除。',
            wraplength=240, justify='left', foreground='#555555',
        ).pack(fill='x', pady=(10, 0))

        status = ttk.Label(self.root, textvariable=self.status_var, padding=(10, 4))
        status.pack(fill='x')

    def choose_manifest(self):
        path = filedialog.askopenfilename(
            title='选择 cut_annotations.json',
            filetypes=[('JSON 标注', '*.json'), ('所有文件', '*.*')],
        )
        if path:
            self.open_manifest(path)

    def new_manifest(self):
        """通过目录选择创建清单，免去首次使用时手工运行命令行。"""
        if self.dirty and not messagebox.askyesno(
            '未保存', '当前修改尚未保存，仍要新建标注清单吗？',
        ):
            return
        image_dir = filedialog.askdirectory(title='选择原教材图片目录')
        if not image_dir:
            return
        output_path = filedialog.asksaveasfilename(
            title='保存标注清单',
            initialdir=image_dir,
            initialfile='cut_annotations.json',
            defaultextension='.json',
            filetypes=[('JSON 标注', '*.json')],
        )
        if not output_path:
            return
        if os.path.exists(output_path) and not messagebox.askyesno(
            '覆盖确认', f'标注清单已存在，是否覆盖？\n{output_path}',
        ):
            return
        manifest = create_annotation_manifest(image_dir)
        if not manifest['pages']:
            messagebox.showwarning('无法创建', '所选目录及子目录中没有可标注的图片。')
            return
        self.manifest = manifest
        self.manifest_path = os.path.abspath(output_path)
        self.page_index = 0
        self.dirty = True
        if not self.save_manifest():
            return
        self.show_page()
        messagebox.showinfo(
            '标注清单已创建',
            f"已加入 {len(manifest['pages'])} 页原图。\n"
            '下一步可点击“导入现有切题框…”把自动切题结果作为橙色建议框。',
        )

    def open_manifest(self, path):
        if self.dirty and not messagebox.askyesno('未保存', '当前修改尚未保存，仍要打开其他清单吗？'):
            return
        try:
            with open(path, 'r', encoding='utf-8') as file:
                manifest = json.load(file)
            if manifest.get('schema_version') != 1 or not isinstance(manifest.get('pages'), list):
                raise ValueError('不是受支持的标注清单')
        except (OSError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror('打开失败', str(error))
            return
        self.manifest = manifest
        self.manifest_path = os.path.abspath(path)
        self.page_index = 0
        self.dirty = False
        self.show_page()

    def current_page(self):
        if not self.manifest or not self.manifest.get('pages'):
            return None
        return self.manifest['pages'][self.page_index]

    def show_page(self):
        page = self.current_page()
        self.canvas.delete('all')
        self.box_list.delete(0, 'end')
        if page is None:
            self.page_var.set('清单中没有页面')
            return
        path = page_image_path(self.manifest, self.manifest_path, page)
        data = np.fromfile(path, dtype=np.uint8) if os.path.isfile(path) else np.array([], dtype=np.uint8)
        self.image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if self.image is None:
            self.status_var.set(f'无法读取原页：{path}')
            return
        self.root.update_idletasks()
        canvas_width = max(300, self.canvas.winfo_width())
        canvas_height = max(300, self.canvas.winfo_height())
        height, width = self.image.shape[:2]
        self.scale = min(canvas_width / width, canvas_height / height)
        shown_width = max(1, int(round(width * self.scale)))
        shown_height = max(1, int(round(height * self.scale)))
        resized = cv2.resize(
            self.image, (shown_width, shown_height),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR,
        )
        ok, encoded = cv2.imencode('.png', resized)
        if not ok:
            self.status_var.set(f'无法显示原页：{path}')
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'))
        origin_x = (canvas_width - shown_width) / 2
        origin_y = (canvas_height - shown_height) / 2
        self.image_origin = (origin_x, origin_y)
        self.canvas.create_image(origin_x, origin_y, image=self.photo, anchor='nw', tags='page-image')
        questions = page.setdefault('questions', [])
        for index, question in enumerate(questions, 1):
            box = question.get('bbox', [])
            self.box_list.insert('end', f'{index:03d}  {box}')
        self.labeled_var.set(page.get('status') == 'labeled')
        labeled_count = sum(item.get('status') == 'labeled' for item in self.manifest['pages'])
        self.page_var.set(
            f"第 {self.page_index + 1}/{len(self.manifest['pages'])} 页\n"
            f"已核验 {labeled_count}/{len(self.manifest['pages'])}\n{page.get('image', '')}"
        )
        self.status_var.set('拖动鼠标绘制一道题的完整范围。')
        self.redraw_overlays()

    def _canvas_to_image(self, x, y):
        if self.image is None:
            return None
        origin_x, origin_y = self.image_origin
        height, width = self.image.shape[:2]
        image_x = int(round((x - origin_x) / self.scale))
        image_y = int(round((y - origin_y) / self.scale))
        if not (0 <= image_x <= width and 0 <= image_y <= height):
            return None
        return image_x, image_y

    def _box_to_canvas(self, box):
        origin_x, origin_y = self.image_origin
        return [
            origin_x + box[0] * self.scale,
            origin_y + box[1] * self.scale,
            origin_x + box[2] * self.scale,
            origin_y + box[3] * self.scale,
        ]

    def redraw_overlays(self):
        self.canvas.delete('question-box')
        page = self.current_page()
        if page is None or self.image is None:
            return
        selected = set(self.box_list.curselection())
        for index, question in enumerate(page.get('questions', [])):
            box = question.get('bbox', [])
            if len(box) != 4:
                continue
            color = '#ff3b30' if index in selected else ('#ff9500' if question.get('suggested') else '#00b7ff')
            width = 4 if index in selected else 2
            coords = self._box_to_canvas(box)
            self.canvas.create_rectangle(*coords, outline=color, width=width, tags='question-box')
            self.canvas.create_text(
                coords[0] + 4, coords[1] + 3, text=str(index + 1),
                fill=color, anchor='nw', font=('Arial', 11, 'bold'), tags='question-box',
            )

    def on_canvas_press(self, event):
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        if self.mode_var.get() == 'select':
            self.select_box_at(point)
            return
        self.drag_start = point
        self.drag_item = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline='#34c759', width=3, tags='drag-box',
        )

    def on_canvas_drag(self, event):
        if self.drag_start is None or self.drag_item is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        start_canvas = self._box_to_canvas([
            self.drag_start[0], self.drag_start[1], point[0], point[1]
        ])
        self.canvas.coords(self.drag_item, *start_canvas)

    def on_canvas_release(self, event):
        if self.drag_start is None:
            return
        end = self._canvas_to_image(event.x, event.y)
        start = self.drag_start
        self.drag_start = None
        if self.drag_item is not None:
            self.canvas.delete(self.drag_item)
            self.drag_item = None
        if end is None:
            return
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return
        page = self.current_page()
        page.setdefault('questions', []).append({'bbox': [x0, y0, x1, y1]})
        page['status'] = 'unlabeled'
        self.dirty = True
        self.show_page()

    def select_box_at(self, point):
        page = self.current_page()
        candidates = []
        for index, question in enumerate(page.get('questions', [])):
            box = question.get('bbox', [])
            if len(box) == 4 and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]:
                candidates.append(((box[2] - box[0]) * (box[3] - box[1]), index))
        if candidates:
            index = min(candidates)[1]
            self.box_list.selection_clear(0, 'end')
            self.box_list.selection_set(index)
            self.box_list.see(index)
            self.redraw_overlays()

    def delete_selected(self):
        page = self.current_page()
        if page is None:
            return
        selected = sorted(self.box_list.curselection(), reverse=True)
        if not selected:
            return
        for index in selected:
            del page['questions'][index]
        page['status'] = 'unlabeled'
        self.dirty = True
        self.show_page()

    def merge_selected(self):
        page = self.current_page()
        selected = sorted(self.box_list.curselection())
        if page is None or len(selected) < 2:
            messagebox.showinfo('合并题框', '请先选择至少两个题框。')
            return
        boxes = [page['questions'][index]['bbox'] for index in selected]
        merged = {
            'bbox': [
                min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes),
            ]
        }
        for index in reversed(selected):
            del page['questions'][index]
        page['questions'].insert(selected[0], merged)
        page['status'] = 'unlabeled'
        self.dirty = True
        self.show_page()

    def toggle_labeled(self):
        page = self.current_page()
        if page is None:
            return
        page['status'] = 'labeled' if self.labeled_var.get() else 'unlabeled'
        self.dirty = True
        self.show_page()

    def mark_empty_labeled(self):
        page = self.current_page()
        if page is None:
            return
        if page.get('questions') and not messagebox.askyesno('清空题框', '本页已有题框，确定全部删除并标记为无题目吗？'):
            return
        page['questions'] = []
        page['status'] = 'labeled'
        self.dirty = True
        self.show_page()

    def previous_page(self):
        if self.manifest and self.page_index > 0:
            self.page_index -= 1
            self.show_page()

    def next_page(self):
        if self.manifest and self.page_index + 1 < len(self.manifest.get('pages', [])):
            self.page_index += 1
            self.show_page()

    def import_predictions(self):
        if not self.manifest:
            messagebox.showinfo('导入切题框', '请先打开标注清单。')
            return
        output_root = filedialog.askdirectory(title='选择现有切题结果根目录')
        if not output_root:
            return
        result = import_prediction_metadata(
            self.manifest, self.manifest_path, output_root,
        )
        self.dirty = self.dirty or bool(result['boxes'])
        self.show_page()
        messagebox.showinfo(
            '导入完成',
            f"导入 {result['pages']} 页、{result['boxes']} 个建议题框；"
            f"无效元数据 {result['invalid_metadata_files']} 个。\n"
            '橙色框是程序建议，必须人工核验后再勾选“本页已核验”。',
        )

    def save_manifest(self):
        if not self.manifest or not self.manifest_path:
            return False
        temp_path = self.manifest_path + '.tmp'
        try:
            with open(temp_path, 'w', encoding='utf-8') as file:
                json.dump(review_queue.json_safe(self.manifest), file, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.manifest_path)
            self.dirty = False
            self.status_var.set(f'已保存：{self.manifest_path}')
            return True
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror('保存失败', str(error))
            return False

    def on_close(self):
        if self.dirty:
            decision = messagebox.askyesnocancel('保存标注', '关闭前是否保存修改？')
            if decision is None:
                return
            if decision and not self.save_manifest():
                return
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='可视化切题真值标注工具')
    parser.add_argument('manifest', nargs='?', help='cut_annotations.json 路径')
    args = parser.parse_args()
    root = tk.Tk()
    AnnotationApp(root, args.manifest or '')
    root.mainloop()


if __name__ == '__main__':
    main()
