# -*- coding: utf-8 -*-
"""Q8：真实页跑 题图候选（image_regions + collect），并把候选裁片落盘供人眼判断。

不造合成图。用 reports/audit_baseline_v2_fixed.jsonl 里同一批真实扫描页，
先跑生产切题拿到**真实题框**，再跑题图候选层，看：
  - 本页候选数
  - 硬拒分布（page_like / formula_only / number_label / too_small / 通过）
  - 归属判定分布（contain / center / nearest / cross_column / orphan）
  - 是否有"面积接近整页却未被硬拒"的候选（甲方明令禁止的整页图混入）
裁片存 reports/qa_illustration/<页名>_<候选id>_<原因>.png

跑法：
    python reports/qa_verify_illustrations.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, 'reports', 'qa_illustration')
BASELINE = os.path.join(ROOT, 'reports', 'audit_baseline_v2_fixed.jsonl')
TARGETS = ['20210126154119260707.jpg', 'image0002263B.jpg', 'Scan_20-06-2022_1634_0039.jpg']
# 每轮裁片带日期前缀：同名文件跨轮并存会让人眼抽查看错版本（QA 自己踩过）
RUN_TAG = '20261020r4'


def clean_stale_crops():
    """只删本脚本产出的裁片（`*.png`），保留 manifest 与人工放的其它文件。"""
    removed = []
    for entry in os.listdir(OUT_DIR):
        if entry.lower().endswith('.png'):
            os.remove(os.path.join(OUT_DIR, entry))
            removed.append(entry)
    return removed


def real_question_records(core, image, image_path, profile):
    """用生产口径切出的真实大题块当"题记录"，给归属判定当输入。"""
    items = core.ocr_page_structure(image_path, image, profile=profile)
    blocks = core.split_page_into_majors(image, items, profile=profile)
    records = []
    for index, block in enumerate(blocks):
        crop = block.get('crop_img')
        if crop is None:
            continue
        height, width = crop.shape[:2]
        x0 = int(block.get('_x_start', 0) or 0)
        y0 = int(block.get('_y_start', 0) or 0)
        records.append({
            'id': f'q{index + 1}',
            'region_role': 'question',
            'bbox': [x0, y0, x0 + width, y0 + height],
            'text': str(block.get('title') or ''),
        })
    return records, blocks


def main():
    import cv2
    import numpy as np

    import layout_arbiter as arbiter
    import review_tools.cut_quality_audit as audit
    from review_tools import illustration_candidates as ic

    core = audit._load_core()
    os.makedirs(OUT_DIR, exist_ok=True)
    removed = clean_stale_crops()
    if removed:
        print(f'清理上一轮裁片 {len(removed)} 张（避免新旧同名混淆）')
    baseline = [json.loads(line) for line in open(BASELINE, encoding='utf-8') if line.strip()]
    manifest = []
    report = []

    print(f'layout_arbiter.available()={arbiter.available()}  REVISION={arbiter.REVISION}')
    if not arbiter.available():
        print('[WARN] 版面模型不可用：image_regions() 会返回 None（"没算"≠"没图"）')

    concerns = []
    for name in TARGETS:
        entry = [item for item in baseline if item['file'] == name][0]
        image = core.cv2_imread(entry['path'])
        if image is None:
            print(f'[ERR] 读不到 {entry["path"]}')
            continue
        height, width = image.shape[:2]
        page_area = float(width * height)

        regions = arbiter.image_regions(image)
        layout = arbiter.detect(image, with_regions=True)
        records, blocks = real_question_records(core, image, entry['path'], entry.get('profile') or 'general')
        candidates = ic.collect(image, 0, records, layout=layout)

        reasons = collections.Counter(candidate['reject_reason'] or '通过' for candidate in candidates)
        owners = collections.Counter(candidate['owner_method'] for candidate in candidates)
        reviews = collections.Counter(candidate.get('needs_review') for candidate in candidates)
        print(f"\n=== {name}  {width}x{height}  真实题块 {len(records)}  候选 {len(candidates)} ===")
        print(f'  image_regions() -> {len(regions) if regions is not None else None} 个区域')
        print(f'  硬拒分布 {dict(reasons)}   归属分布 {dict(owners)}   needs_review {dict(reviews)}')
        hw = layout.get('columns') if layout else None
        print(f'  detect() columns={hw}  keys={sorted(layout.keys()) if layout else None}')

        picked = 0
        for candidate in candidates:
            x0, y0, x1, y1 = candidate['bbox']
            box_area_ratio = candidate.get('page_area_ratio')
            if box_area_ratio is None:
                box_area_ratio = round(((x1 - x0) * (y1 - y0)) / page_area, 4)
            tag = candidate['reject_reason'] or 'pass'
            # 文件名带日期后缀 + needs_review 标记，避免与历史轮次同名裁片混淆
            review_tag = 'review' if candidate.get('needs_review') else 'auto'
            print(f"    {candidate['id']:>9} label={candidate['label']:<8} "
                  f"score={candidate['layout_score']:.2f} area={box_area_ratio:.3f} "
                  f"hamming={candidate['page_hamming']} reason={tag:<12} "
                  f"needs_review={str(candidate.get('needs_review')):<5} "
                  f"owner={candidate['owner_id']}/{candidate['owner_method']}")
            # 甲方红线：面积接近整页却未被 page_like 拦下
            if box_area_ratio >= 0.85 and tag != 'page_like':
                concerns.append((name, candidate['id'], box_area_ratio, tag))
            if picked < 4:
                crop = image[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
                if crop.size:
                    filename = (f'{RUN_TAG}_{os.path.splitext(name)[0]}_'
                                f'{candidate["id"]}_{tag}_{review_tag}.png')
                    path = os.path.join(OUT_DIR, filename)
                    cv2.imencode('.png', crop)[1].tofile(path)
                    manifest.append({
                        'page': name, 'candidate': candidate['id'], 'file': path,
                        'bbox': candidate['bbox'], 'label': candidate['label'],
                        'layout_score': candidate['layout_score'],
                        'area_ratio': box_area_ratio,
                        'page_area_ratio': candidate.get('page_area_ratio'),
                        'reject_reason': candidate['reject_reason'],
                        'needs_review': bool(candidate.get('needs_review')),
                        'similarity_to_page': candidate.get('similarity_to_page'),
                        'page_hamming': candidate.get('page_hamming'),
                        'owner': f"{candidate['owner_id']}/{candidate['owner_method']}",
                        'crop_sha1': candidate['crop_sha1'],
                    })
                    picked += 1
        report.append({
            'page': name, 'size': [width, height], 'records': len(records),
            'regions': len(regions) if regions is not None else None,
            'candidates': len(candidates),
            'reject_reasons': dict(reasons), 'owner_methods': dict(owners),
        })

    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w', encoding='utf-8') as file:
        json.dump({'pages': report, 'crops': manifest, 'concerns': concerns},
                  file, ensure_ascii=False, indent=2)

    print(f'\n落盘 {len(manifest)} 张裁片 -> {OUT_DIR}')
    print(f'整页未拦（面积≥0.85 且 reason≠page_like）：{concerns if concerns else "无"}')
    return 0 if not concerns else 1


if __name__ == '__main__':
    sys.exit(main())
