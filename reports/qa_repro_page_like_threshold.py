# -*- coding: utf-8 -*-
"""最小复现：page_like（整页缩略图）硬拒在真实页上只在"像素级相同"时生效。

现象
----
`illustration_candidates.hard_reject()` 的 page_like 用 32×32 phash 的汉明距离，
阈值 `runtime/enrich/candidate_rules.json: hard_reject.page_like.hamming_max = 6`
（共 1024 bit）。真实扫描页内缩 **5 像素**（仍是 99.5% 页面积，肉眼就是整页）
后汉明距离即到 8 > 6 → 不再判 page_like，`reject_reason` 变成空串，
和正常题图一样进入"可采用"候选。

验法
----
    python reports/qa_repro_page_like_threshold.py

期望（甲方口径：整页图不得进交付）
    面积比 ≥0.9 的框 → reason == 'page_like'
实际
    inset ≥5px（面积比 0.995）→ reason == ''
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAGE = ('F:/2008-2015/2012/金版卷王名师面对面大试卷数学一年级下册人教版/'
        '20210126154119260707.jpg')


def main():
    import review_tools.cut_quality_audit as audit
    from review_tools import illustration_candidates as ic

    core = audit._load_core()
    image = core.cv2_imread(PAGE)
    if image is None:
        print(f'[SKIP] 缺测试页：{PAGE}')
        return 2
    height, width = image.shape[:2]
    page_area = float(width * height)
    limit = int(ic.RULES['hard_reject']['page_like']['hamming_max'])
    print(f'页 {width}x{height}  hamming_max={limit}（共 32*32=1024 bit）\n')

    print(f'{"内缩px":>7} {"面积比":>8} {"hamming":>8} {"similarity":>11}  reason')
    leaks = []
    for offset in (0, 1, 2, 3, 5, 8, 12, 20, 40):
        box = [float(offset), float(offset), float(width - offset), float(height - offset)]
        crop = ic._crop_of(image, box)
        verdict = ic.hard_reject(crop, box, (height, width), 'image', page=image)
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / page_area
        reason = verdict['reason']
        print(f'{offset:>7} {area_ratio:>8.4f} {str(verdict["page_hamming"]):>8} '
              f'{verdict["similarity_to_page"]:>11} {reason or "（未拦）"}')
        if area_ratio >= 0.9 and reason != 'page_like':
            leaks.append((offset, round(area_ratio, 4), verdict['page_hamming']))

    # 参照系：正常题图与整页的汉明距离有多远（说明提高阈值不会误杀）
    region = [float(width * 0.05), float(height * 0.35),
              float(width * 0.45), float(height * 0.65)]
    crop = ic._crop_of(image, region)
    normal = ic.hard_reject(crop, region, (height, width), 'image', page=image)
    print(f'\n正常题图（面积比 ~0.12）hamming={normal["page_hamming"]} '
          f'similarity={normal["similarity_to_page"]} reason={normal["reason"] or "（未拦）"}')
    print('真图 hamming≈400 量级 → 阈值取 64 仍有 6 倍余量，不会误杀真题图。')

    print(f'\n结论：面积比 ≥0.9 却未被 page_like 拦下的框 {len(leaks)} 个 → {leaks}')
    if leaks:
        print('REPRO OK：缺陷成立。建议 runtime/enrich/candidate_rules.json 的 '
              'hamming_max 由 6 调到 64（或补一条面积比 ≥0.9 的兜底判定）。')
        return 1
    print('未复现。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
