# -*- coding: utf-8 -*-
"""Q7：detect() 行为等价回归 —— 当前代码重跑切题主流程，与改造前快照逐块比对。

快照：reports/audit_baseline_v2_fixed.jsonl（layout_arbiter v2 时代跑的 60 页）
比对：block_count、每块 title / cut_numbers（附带 bounds / y_start / col）
允许差异：无

跑法：
    python reports/qa_verify_detect_equivalence.py            # 抽 3 页（对开/单栏/两栏）
    python reports/qa_verify_detect_equivalence.py --all      # 全 60 页
    python reports/qa_verify_detect_equivalence.py --all --keys 3   # 额外校准：≥3 栏候选页
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASELINE = os.path.join(ROOT, 'reports', 'audit_baseline_v2_fixed.jsonl')
TARGETS = ['20210126154119260707.jpg', 'image0002263B.jpg', 'Scan_20-06-2022_1634_0039.jpg']


def load_baseline():
    with open(BASELINE, encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def block_fingerprint(block):
    return {
        'title': str(block.get('title') or ''),
        'cut_numbers': list(block.get('cut_numbers') or []),
        'bounds': list(block.get('bounds') or []),
        'y_start': block.get('y_start'),
        'col': block.get('col'),
        'position': block.get('position'),
    }


def compare(entry, fresh):
    diff = []
    baseline_blocks = [block_fingerprint(item) for item in entry.get('blocks') or []]
    fresh_blocks = [block_fingerprint(item) for item in fresh.get('blocks') or []]
    if len(baseline_blocks) != len(fresh_blocks):
        diff.append(f"block_count {len(baseline_blocks)} -> {len(fresh_blocks)}")
    for index in range(min(len(baseline_blocks), len(fresh_blocks))):
        old, new = baseline_blocks[index], fresh_blocks[index]
        for key in ('title', 'cut_numbers', 'bounds', 'y_start', 'col'):
            if old[key] != new[key]:
                diff.append(f'块[{index}].{key}: {old[key]!r} -> {new[key]!r}')
    return baseline_blocks, fresh_blocks, diff


def risk_pages(baseline):
    """找出可能真正触发版面裁判（现有分栏 ≥3）的页：读 v2 缓存的 gutters 数量。"""
    import layout_arbiter as arbiter
    import numpy as np
    import cv2
    picked = []
    for entry in baseline:
        path = entry['path']
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            continue
        if image is None:
            continue
        key = arbiter._image_key(image)
        cache_path = os.path.join(arbiter._cache_dir(), f'{key}.json')
        try:
            payload = json.load(open(cache_path, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        # v2 缓存 = 改造前仲裁后的最终栏位；≥3 栏说明裁判真的介入过
        if len(payload.get('columns') or []) >= 3:
            picked.append((entry['file'], payload.get('revision'), len(payload['columns'])))
    return picked


def main():
    import review_tools.cut_quality_audit as audit

    run_all = '--all' in sys.argv
    baseline = load_baseline()
    if run_all:
        entries = baseline
    else:
        entries = [item for item in baseline if item['file'] in TARGETS]
        missing = [name for name in TARGETS if name not in {item['file'] for item in entries}]
        if missing:
            print(f'[WARN] 快照里没有这些页：{missing}')

    print(f'待验证 {len(entries)} 页；快照={os.path.basename(BASELINE)}')
    failures = []
    total = 0
    for entry in entries:
        total += 1
        started = time.perf_counter()
        fresh = audit.audit_page(entry['path'], profile=entry.get('profile') or 'general')
        elapsed = round(time.perf_counter() - started, 2)
        if fresh.get('error'):
            failures.append((entry['file'], [f"error={fresh['error']}"]))
            print(f"  [ERR ] {entry['file']} {fresh['error']}")
            continue
        old_blocks, new_blocks, diff = compare(entry, fresh)
        flag = 'PASS' if not diff else 'FAIL'
        print(f"  [{flag}] {entry['file']} 块 {len(old_blocks)}/{len(new_blocks)} 用时 {elapsed}s"
              f" 栏位 {entry.get('columns')}->{fresh.get('columns')}")
        for item in diff:
            print(f'         {item}')
        if diff:
            failures.append((entry['file'], diff))

    if '--keys' in sys.argv:
        print('\n== 额外校准：改造前缓存里 ≥3 栏（版面裁判真正介入过）的页 ==')
        risky = risk_pages(baseline)
        print(f'  命中 {len(risky)} 页：{risky[:8]}')
        for name, revision, columns in risky[:3]:
            entry = [item for item in baseline if item['file'] == name][0]
            fresh = audit.audit_page(entry['path'], profile=entry.get('profile') or 'general')
            old_blocks, new_blocks, diff = compare(entry, fresh)
            print(f"  [{'PASS' if not diff else 'FAIL'}] {name}（改造前 {revision}，{columns} 栏）"
                  f" 块 {len(old_blocks)}/{len(new_blocks)}")
            for item in diff:
                print(f'         {item}')
            if diff:
                failures.append((name, diff))

    print(f'\n==== 汇总：{total} 页，失败 {len(failures)} 页 ====')
    for name, diff in failures:
        print('  FAIL', name, diff[:6])
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
