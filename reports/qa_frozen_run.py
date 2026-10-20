# -*- coding: utf-8 -*-
"""冻结树跑测试：报数前后各取一次快照，证明这棵树在测试窗口内没有被人改动。

背景：共享主干上多人同时落盘，会跑出"重跑即过"的假失败（本仓库今天已发生 2 次）。
team-lead 定的判读规则是——报数前后各取一次 mtime，两次一致才算"静止树"，
此时的 `Ran X tests` 才具备证据力。本脚本把这条规则固化成可复跑的检查。

用法：
    python reports/qa_frozen_run.py                # 全量
    python reports/qa_frozen_run.py --pattern test_illustration_candidates.py

退出码：
    0  静止树且测试通过
    1  测试失败（静止树下，即真实失败）
    2  树在测试窗口内发生了变动 → 本次数字不可用作验收证据，请重跑
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
# 只跟踪"会改变测试结果"的文件；缓存/产物/报告不参与（它们每次跑都会变）
SKIP_DIRS = {'.git', '__pycache__', '.workbuddy', '.claude', 'reports', 'logs', 'runs',
             'release', 'online_review_beta', 'result16', 'vlm_output', 'vlm_source',
             '_experiments'}
SKIP_SUFFIX = ('.pyc', '.png', '.jpg', '.jpeg', '.onnx', '.pt', '.jsonl', '.docx', '.xlsx')


def snapshot():
    """(path -> sha1, mtime_ns) 全量快照。"""
    state = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(SKIP_SUFFIX):
                continue
            if not (name.endswith('.py') or name.endswith('.json')):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, ROOT)
            try:
                stat = os.stat(path)
                with open(path, 'rb') as file:
                    digest = hashlib.sha1(file.read()).hexdigest()
            except OSError:
                continue
            state[rel] = (digest, stat.st_mtime_ns)
    return state


def diff(before, after):
    changed = []
    for rel in sorted(set(before) | set(after)):
        if before.get(rel) != after.get(rel):
            changed.append(rel)
    return changed


def main():
    pattern = None
    if '--pattern' in sys.argv:
        pattern = sys.argv[sys.argv.index('--pattern') + 1]

    before = snapshot()
    command = [PYTHON, '-m', 'unittest', 'discover', '-s', 'tests']
    if pattern:
        command += ['-p', pattern]
    command.append('-q')

    started = time.time()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    elapsed = round(time.time() - started, 1)
    after = snapshot()

    output = (proc.stdout or '') + (proc.stderr or '')
    ran = re.findall(r'^Ran (\d+) tests? in ', output, re.MULTILINE)
    verdict = 'OK' if re.search(r'^OK', output, re.MULTILINE) else 'FAILED'

    changed = diff(before, after)
    stable = not changed

    print(f'窗口 {elapsed}s  跟踪文件 {len(before)} 个')
    print(f'结果: Ran {ran[-1] if ran else "?"} tests → {verdict}')
    if stable:
        print('树状态: 静止（测试窗口内无任何 .py/.json 变动）→ 本数字可作验收证据')
    else:
        print(f'树状态: 变动 {len(changed)} 个文件 → 本数字不可作验收证据，请重跑')
        for rel in changed[:15]:
            old = before.get(rel, ('(无)', 0))[0][:12]
            new = after.get(rel, ('(无)', 0))[0][:12]
            print(f'  {rel}  {old} -> {new}')

    if not stable:
        return 2
    return 0 if verdict == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
