# -*- coding: utf-8 -*-
"""最小复现：知识点词表里同一别名挂在两个年级节点上，归一结果静默取文档序第一条。

现象
----
`KnowledgeTaxonomy._build_index()` 用 `_by_alias.setdefault(text, code)` 建索引，
既不校验跨条目重复别名，也不报歧义。当前词表有 5 组这样的别名，例如
``笔算乘法`` 同时是

    M-G3-13 两位数乘两位数（三年级）
    M-G4-04 三位数乘两位数（四年级）

的别名。四年级的题面写"笔算乘法"时，`resolve()` 一律返回三年级那条，
`validate()` 把它当"已在词表内"接受 → knowledge_points 写成三年级知识点，
既不 rejected 也不升 L2。这与模块声明的"宁可拒收也不静默接受一个不在词表
里的说法"相反。

验法
----
    python reports/qa_repro_taxonomy_alias_ambiguity.py

期望
    歧义别名 → resolve() 返回 None（走 rejected + L2），或加载时抛错
实际
    resolve() 返回文档序第一条，且没有任何 warning
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VOCAB = os.path.join(ROOT, 'runtime', 'taxonomy', 'knowledge_points.math.2026.10.json')


def main():
    import json
    from review_tools import knowledge_taxonomy as kt

    payload = json.load(open(VOCAB, encoding='utf-8'))
    nodes = payload['nodes']
    owners = {}
    for node in nodes:
        for alias in node.get('aliases') or []:
            text = str(alias).strip()
            if text:
                owners.setdefault(text, []).append(node)

    ambiguous = {alias: items for alias, items in owners.items() if len(items) > 1}
    print(f'词表 {VOCAB}')
    print(f'节点 {len(nodes)}，跨条目重复别名 {len(ambiguous)} 组：\n')
    for alias, items in ambiguous.items():
        print(f'  {alias!r}')
        for node in items:
            print(f"      {node['code']:<9} {node['name']:<12} grades={node['grades']}")

    kt.clear_cache()
    taxonomy = kt.load('数学')
    leaks = []
    print('\nresolve() 实际行为：')
    for alias, items in ambiguous.items():
        code = taxonomy.resolve(alias)
        node = taxonomy.node(code)
        accepted, rejected = taxonomy.validate([alias])
        print(f"  resolve({alias!r}) -> {code} / {node['name']} / grades={node['grades']}"
              f"   validate -> accepted={accepted} rejected={rejected}")
        if code not in (None, '') and len(items) > 1:
            leaks.append((alias, code, [item['code'] for item in items]))

    print('\n命中歧义别名却仍被 accepted 的：')
    for alias, code, codes in leaks:
        print(f'  {alias!r} -> {code}（候选：{codes}）')

    if leaks:
        print('\nREPRO OK：缺陷成立。建议二选一：')
        print('  A. 加载校验加"跨条目重复别名 → 抛 TaxonomyInvalid"（与 name 唯一同口径）')
        print('  B. resolve() 命中歧义时返回 None（走 rejected + L2），并在 stats() 里报歧义数')
        print('  C. 词表运营侧删掉其中一条别名（数据修正，最小改动）')
        return 1
    print('未复现。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
