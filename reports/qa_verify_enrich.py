# -*- coding: utf-8 -*-
"""QA 独立验证脚本（不依赖工程师测试）：题 3 / 4 / 5 / 6 的机器可复现证据。

跑法：
    C:/Users/A001/AppData/Local/Programs/Python/Python313/python.exe reports/qa_verify_enrich.py

覆盖：
  Q3  题图候选 page_like 通道：collect() 所有硬拒调用是否必然带 page
  Q4  配对判定 4 组反例 + display_confidence 是否参与判定（源码级 grep）
  Q5  解析复算执行注入：QA 自写的 6 条攻击串（不用工程师那几条）
  Q6  词表封闭性：code 唯一 / stage+grades 合法 / alias 与 name 冲突 / 课标外概念
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = []


def check(item, name, ok, detail):
    RESULTS.append({'item': item, 'name': name, 'ok': ok, 'detail': detail})
    print(('  [PASS] ' if ok else '  [FAIL] ') + f'{item} {name} :: {detail}')


# ------------------------------------------------------------------ Q3
def q3_hard_reject_page_argument():
    print('\n== Q3 题图候选：page_like 通道是否必然带 page ==')
    import numpy as np
    from review_tools import illustration_candidates as ic

    source_path = os.path.join(ROOT, 'review_tools', 'illustration_candidates.py')
    tree = ast.parse(open(source_path, encoding='utf-8').read())

    call_sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, 'id', None) or getattr(func, 'attr', None)
            if name == 'hard_reject':
                keywords = {kw.arg for kw in node.keywords}
                call_sites.append({'line': node.lineno, 'keywords': sorted(keywords)})
    check('Q3', 'hard_reject 调用点数量', len(call_sites) == 1, f'call_sites={call_sites}')
    check('Q3', '唯一调用点显式传 page=',
          bool(call_sites) and 'page' in call_sites[0]['keywords'],
          f"第 {call_sites[0]['line'] if call_sites else '?'} 行 keywords={call_sites[0]['keywords'] if call_sites else None}")

    # 运行时证明：monkeypatch hard_reject 记录实参，确认 page 一定非 None
    captured = []
    original = ic.hard_reject

    def spy(crop, image_box, page_shape, label, page=None):
        captured.append(page is not None)
        return original(crop, image_box, page_shape, label, page=page)

    ic.hard_reject = spy
    try:
        page_img = np.full((1600, 1200, 3), 235, dtype=np.uint8)
        layout = {
            'columns': [[0.0, 600.0], [600.0, 1200.0]],
            'regions_detail': [
                {'label': 'image', 'score': 0.93, 'box': [0.0, 0.0, 1200.0, 1600.0]},   # 整页
                {'label': 'image', 'score': 0.93, 'box': [40.0, 300.0, 500.0, 700.0]},
                {'label': 'text', 'score': 0.99, 'box': [10.0, 10.0, 300.0, 60.0]},
            ],
        }
        candidates = ic.collect(page_img, 0, [], layout=layout)
    finally:
        ic.hard_reject = original
    check('Q3', 'collect() 每次硬拒都传了 page', bool(captured) and all(captured),
          f'调用 {len(captured)} 次，带 page {sum(captured)} 次')

    page_like_none = [c for c in candidates if c['page_hamming'] is None]
    check('Q3', '无 page_hamming=None 的候选（不可能静默跳过整页判定）',
          not page_like_none,
          f"candidates={[(c['bbox'], c['reject_reason'], c['page_hamming']) for c in candidates]}")

    whole = [c for c in candidates if c['bbox'] == [0, 0, 1200, 1600]]
    check('Q3', '整页区域被硬拒为 page_like', bool(whole) and whole[0]['reject_reason'] == 'page_like',
          f"整页候选 reject_reason={whole[0]['reject_reason'] if whole else '未产出'}")

    # 反向：不传 page 时也必须拦住整页框（Round 2 起由"面积闸"兜底）
    blank = np.full((1600, 1200, 3), 235, dtype=np.uint8)
    verdict_no_page = original(blank, [0, 0, 1200, 1600], (1600, 1200), 'image')
    verdict_with_page = original(blank, [0, 0, 1200, 1600], (1600, 1200), 'image', page=blank)
    check('Q3', '不传 page 时整页框仍被拦（面积闸，不依赖 phash）',
          verdict_no_page['reason'] == 'page_like' and verdict_with_page['reason'] == 'page_like',
          f"no_page={verdict_no_page['reason']!r}(area={verdict_no_page.get('page_area_ratio')}) "
          f"with_page={verdict_with_page['reason']!r}")

    # 合成纯色页：只能证明"判定链路通"，**不能**代表真实页灵敏度
    # （纯色无纹理，任何内缩后 phash 仍与原页相同 → 恒 hamming=0，见 Q3-SYNTH 标注）
    for inset in (0.01, 0.02, 0.05, 0.10):
        h, w = blank.shape[:2]
        box = [w * inset, h * inset, w * (1 - inset), h * (1 - inset)]
        verdict = original(blank, box, (h, w), 'image', page=blank)
        check('Q3-SYNTH', f'纯色页内缩 {inset:.0%} 判 page_like（仅链路验证，非灵敏度）',
              verdict['reason'] == 'page_like',
              f"reason={verdict['reason']!r} hamming={verdict['page_hamming']}")

    # 真实页（有纹理、非纯色）上的 same 判定：整页框 / 内缩框
    try:
        import cv2
        import review_tools.cut_quality_audit as audit
        core = audit._load_core()
        image = core.cv2_imread(
            'F:/2008-2015/2012/金版卷王名师面对面大试卷数学一年级下册人教版/20210126154119260707.jpg')
    except Exception as error:  # noqa: BLE001
        image = None
        print(f'  [SKIP] 真实页不可用：{error}')
    if image is not None:
        height, width = image.shape[:2]
        layout_real = {
            'columns': [[0.0, width / 2.0], [width / 2.0, float(width)]],
            'regions_detail': [
                {'label': 'image', 'score': 0.95, 'box': [0.0, 0.0, float(width), float(height)]},
                {'label': 'image', 'score': 0.95,
                 'box': [20.0, 20.0, float(width - 20), float(height - 20)]},
                {'label': 'image', 'score': 0.95,
                 'box': [float(width * 0.02), float(height * 0.02),
                         float(width * 0.98), float(height * 0.98)]},
            ],
        }
        real_candidates = ic.collect(image, 0, [], layout=layout_real)
        detail = [(c['bbox'], c['reject_reason'], c['page_hamming']) for c in real_candidates]
        whole = [c for c in real_candidates if c['page_hamming'] == 0]
        check('Q3', '真实页整页框被判 page_like', bool(whole) and whole[0]['reject_reason'] == 'page_like',
              f'{detail}')
        leaked = [c for c in real_candidates if c['reject_reason'] == ''
                  and (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1])
                  >= 0.85 * width * height]
        check('Q3-EXTRA', '真实页无 ≥85% 页面积却未拦的候选（集成外红线）', not leaked,
              '; '.join(f"{c['id']} 面积比≈{round((c['bbox'][2]-c['bbox'][0])*(c['bbox'][3]-c['bbox'][1])/(width*height),4)}"
                        f" hamming={c['page_hamming']} reason={c['reject_reason']!r}" for c in leaked)
              or '无')

        # 真实页上的灵敏度扫描：整页框内缩几像素后 page_like 失效
        scan = []
        for offset in (0, 1, 2, 3, 5, 8, 12, 20, 40):
            box = [float(offset), float(offset), float(width - offset), float(height - offset)]
            crop = ic._crop_of(image, box)
            verdict = ic.hard_reject(crop, box, (height, width), 'image', page=image)
            scan.append((offset, verdict['page_hamming'], verdict['reason'] or '未拦'))
        first_miss = next((item for item in scan if item[2] == '未拦'), None)
        check('Q3-EXTRA', 'page_like 在 0~40px 内缩区间内无失效点',
              first_miss is None,
              f'首个漏拦 {first_miss}；扫描={scan}')


# ------------------------------------------------------------------ Q4
def q4_pairing_counterexamples():
    print('\n== Q4 配对判定 4 组反例 ==')
    from review_tools import review_levels as levels

    def page(**kwargs):
        base = {'illustration_ready': False, 'questions': [], 'page_consistency': {}}
        base.update(kwargs)
        return base

    def rec(pair_value, **kwargs):
        base = {'id': 'q1', 'region_role': 'question', 'pair': pair_value,
                'illustrations': [], 'question_type': '计算', 'text': '3+5=?'}
        base.update(kwargs)
        return base

    level, reasons = levels.compute_review_level(
        rec({'path_match': False, 'score': 400.0, 'margin': 200.0}), page())
    check('Q4', 'path_match=False score=400 margin=200 不是 L0',
          level != 'L0' and 'pair_fail' in reasons, f'{level} {reasons}')

    level, reasons = levels.compute_review_level(
        rec({'path_match': True, 'score': 180.0, 'margin': 79.0}), page())
    check('Q4', 'path_match=True score=180 margin=79 不是 L0',
          level != 'L0' and 'pair_fail' in reasons, f'{level} {reasons}')

    level, reasons = levels.compute_review_level(
        rec({'path_match': True, 'score': 180.0, 'margin': 120.0}), page())
    check('Q4', 'path_match=True score=180 margin=120 是 L0',
          level == 'L0', f'{level} {reasons}')

    level, reasons = levels.compute_review_level(
        rec({}, score=400.0, display_confidence=1.0), page())
    check('Q4', 'pair={} 但记录自带 score/display_confidence 高 不得进 L0',
          level != 'L0' and 'pair_fail' in reasons, f'{level} {reasons}')

    # pair 里塞 display_confidence 试图单阈值放行
    level, reasons = levels.compute_review_level(
        rec({'path_match': False, 'score': 300.0, 'margin': 300.0, 'display_confidence': 1.0}),
        page())
    check('Q4', 'pair.display_confidence=1.0 且 path_match=False 不得放行',
          level != 'L0', f'{level} {reasons}')

    # display_confidence 是否参与判定：源码级
    source = open(os.path.join(ROOT, 'review_tools', 'review_levels.py'), encoding='utf-8').read()
    hits = [line.strip() for line in source.splitlines()
            if 'display_confidence' in line]
    decision_reads = [h for h in hits
                      if not h.startswith('#') and 'pair_confidence' not in h
                      and "THRESHOLDS = {" not in h
                      and 'display_confidence_scale' not in h]
    check('Q4', 'display_confidence 未被任何判定读取（仅展示/阈值表）',
          all(('"""' in h) or h.startswith('#') or '只供展示' in h or '禁止参与判定' in h
              or '任何判定不得读它' in h or '只反映' in h
              for h in decision_reads),
          f'hits={len(hits)} 可疑={decision_reads}')

    calls = re.findall(r"record(?:\.get)?\(" , source)
    reads_record_dc = re.search(r"get\(\s*['\"]display_confidence['\"]", source)
    check('Q4', "源码中不存在 record.get('display_confidence') 的读取",
          reads_record_dc is None, f'match={reads_record_dc.group(0) if reads_record_dc else None}')

    # 反向自检：确认我的断言真能区分（把阈值改成单阈值实现就该 FAIL）
    original_pair_ok = levels.pair_ok
    try:
        levels.pair_ok = lambda record: float(
            (record.get('pair') or {}).get('display_confidence', 0.0) or 0.0) >= 0.5
        broken_level, _ = levels.compute_review_level(
            rec({'path_match': False, 'score': 300.0, 'margin': 300.0, 'display_confidence': 1.0}),
            page())
    finally:
        levels.pair_ok = original_pair_ok
    check('Q4', '断言具备鉴别力（换成单阈值实现后该记录会变 L0）',
          broken_level == 'L0', f'注入单阈值后 level={broken_level}')


# ------------------------------------------------------------------ Q5
def q5_injection():
    print('\n== Q5 解析复算执行注入（QA 自写攻击串）==')
    from review_tools import analysis_verifier as av

    attacks = [
        '__import__("subprocess").run(["cmd"])',
        '().__class__.__bases__[0].__subclasses__()',
        'open("x","w")',
        '"a"*10**9',
        '1e999',
        '2**9999',
    ]
    marker = os.path.join(tempfile.mkdtemp(prefix='qa_inject_'), 'x')
    marker_dir = os.path.dirname(marker)
    before = set(os.listdir(marker_dir))

    errors = []
    for attack in attacks:
        try:
            value = av.safe_eval_expression(attack)
        except BaseException as error:  # noqa: BLE001 - 就是要抓一切
            errors.append((attack, f'{type(error).__name__}: {error}'))
            continue
        check('Q5', f'safe_eval_expression({attack!r}) 返回 None/有限值',
              value is None or abs(float(value)) < 10 ** 9, f'-> {value!r}')

    check('Q5', '攻击串均未抛出未捕获异常', not errors, f'errors={errors}')
    after = set(os.listdir(marker_dir))
    check('Q5', '攻击串未写文件', before == after, f'new={sorted(after - before)}')
    check('Q5', 'open() 未被创建于 cwd', not os.path.exists(os.path.join(os.getcwd(), 'x')),
          f"cwd={os.getcwd()}")

    # 端到端：把攻击串塞进题干，走 verify()
    statuses = []
    for attack in attacks:
        record = {
            'subject': '数学', 'question_type': '计算',
            'text': f'计算：{attack}',
            'answer': '1',
            'analysis': '先算括号里的加法，再把结果乘以二，最后得到一这个结果。',
        }
        try:
            payload = av.verify(record, {})
        except BaseException as error:  # noqa: BLE001
            statuses.append((attack, f'RAISED {type(error).__name__}: {error}'))
            continue
        statuses.append((attack, payload['status']))
    check('Q5', 'verify() 对攻击串只返回 pass/fail/not_checkable',
          all(status in ('pass', 'fail', 'not_checkable') for _a, status in statuses),
          f'{statuses}')

    # 危险结果上限：10**9 边界
    check('Q5', '3**18（超大）被结果上限拦住', av.safe_eval_expression('3**18') is None,
          f"3**18 -> {av.safe_eval_expression('3**18')}")
    check('Q5', '正常题 12*8+5 仍可复算', str(av.safe_eval_expression('12*8+5')) == '101',
          f"-> {av.safe_eval_expression('12*8+5')}")


# ------------------------------------------------------------------ Q6
def q6_taxonomy():
    print('\n== Q6 词表封闭性 ==')
    path = os.path.join(ROOT, 'runtime', 'taxonomy', 'knowledge_points.math.2026.10.json')
    payload = json.load(open(path, encoding='utf-8'))
    nodes = payload['nodes']

    codes = [str(node.get('code')) for node in nodes]
    check('Q6', 'code 无重复', len(codes) == len(set(codes)),
          f'节点 {len(codes)}，唯一 {len(set(codes))}')

    bad_grade, bad_stage = [], []
    for node in nodes:
        for grade in node.get('grades') or []:
            if isinstance(grade, bool) or not isinstance(grade, int) or not 1 <= grade <= 6:
                bad_grade.append((node['code'], grade))
        if str(node.get('stage') or '') != '小学':
            bad_stage.append((node['code'], node.get('stage')))
    check('Q6', 'grades 全在 1~6', not bad_grade, f'bad={bad_grade[:5]}')
    check('Q6', 'stage 全为小学', not bad_stage, f'bad={bad_stage[:5]}')

    names = {str(node['name']).strip(): node['code'] for node in nodes}
    conflicts = []
    for node in nodes:
        for alias in node.get('aliases') or []:
            text = str(alias).strip()
            if not text:
                continue
            if text in names and names[text] != node['code']:
                conflicts.append((node['code'], text, names[text]))
    check('Q6', 'alias 不与其他条目 name 冲突', not conflicts, f'conflicts={conflicts[:5]}')

    from review_tools import knowledge_taxonomy as kt
    kt.clear_cache()
    taxonomy = kt.load('数学')
    accepted, rejected = taxonomy.validate(['泰勒展开', '矩阵秩', '洛必达法则'])
    check('Q6', '课标外概念全部进 rejected', not accepted and len(rejected) == 3,
          f'accepted={accepted} rejected={rejected}')

    accepted2, rejected2 = taxonomy.validate(['分数加减法', '圆的认识（一）', '不存在的知识点XYZ'])
    check('Q6', '词表内词可归一 + 词表外词被拒',
          len(accepted2) == 2 and rejected2 == ['不存在的知识点XYZ'],
          f'accepted={accepted2} rejected={rejected2}')

    # Q6-EXTRA（文档未约束，记为 SUSPECT）：同一别名挂在两个年级节点上时，
    # _by_alias.setdefault 会静默取文档序第一条，没有歧义检测。
    alias_owner = {}
    alias_dup = []
    for node in nodes:
        for alias in node.get('aliases') or []:
            text = str(alias).strip()
            if not text:
                continue
            if text in alias_owner and alias_owner[text][0] != node['code']:
                alias_dup.append((text, alias_owner[text], (node['code'], node['name'], node.get('grades'))))
            alias_owner.setdefault(text, (node['code'], node['name'], node.get('grades')))
    check('Q6-EXTRA', 'alias 之间无跨条目重复（歧义归一风险）', not alias_dup,
          f'跨条目重复别名 {len(alias_dup)} 组：{alias_dup}')
    if alias_dup:
        sample = alias_dup[0][0]
        resolved = taxonomy.resolve(sample)
        node = taxonomy.node(resolved)
        check('Q6-EXTRA', f'歧义别名 {sample!r} 的归一结果可解释',
              False,
              f"resolve({sample!r}) -> {resolved} / {node['name']} / grades={node['grades']}"
              f"（另一节点 {alias_dup[0][2][0]} {alias_dup[0][2][1]} 被静默忽略）")

    # 归一化不得制造"就近挂靠"：去掉数字/括号后仍应命中或明确拒绝
    probe = ['分数加法', '三角形的面积', '泰勒展开式']
    outcome = [(item, taxonomy.resolve(item)) for item in probe]
    check('Q6', '归一化不放宽到"近似即命中"',
          outcome[2][1] is None, f'{outcome}')

    # Q6-QUAL：年级限定别名必须归到正确年级；不带限定必须拒收并记 warning
    qualified = []
    for node in nodes:
        for alias in node.get('aliases') or []:
            base, grade = kt.split_grade_qualifier(str(alias))
            if grade:
                qualified.append((alias, node['code'], grade,
                                  [kt.grade_index(item) for item in node.get('grades') or []]))
    wrong = [(alias, taxonomy.resolve(alias), code, grade, grades)
             for alias, code, grade, grades in qualified
             if taxonomy.resolve(alias) != code or grade not in grades]
    check('Q6-QUAL', '年级限定别名全部归到正确节点且年级自洽',
          not wrong, f'{len(qualified)} 条限定别名，异常={wrong}')

    declared = payload.get('ambiguous_terms') or {}
    check('Q6-QUAL', 'ambiguous_terms 注册表存在且每条至少 2 个 code',
          isinstance(declared, dict) and bool(declared)
          and all(isinstance(value, list) and len(value) >= 2 for value in declared.values()),
          f'{len(declared) if isinstance(declared, dict) else declared} 条：'
          f'{list(declared) if isinstance(declared, dict) else None}')

    silent, no_warn = [], []
    for term in (declared if isinstance(declared, dict) else {}):
        warnings = []
        code = taxonomy.resolve(term, warnings)
        if code is not None:
            silent.append((term, code))
        if not warnings:
            no_warn.append(term)
    check('Q6-QUAL', '歧义写法 resolve 一律 None（不猜年级）', not silent, f'静默命中={silent}')
    check('Q6-QUAL', '歧义写法必须记 warning（供升 L2 用）', not no_warn, f'缺 warning={no_warn}')

    wrong_grade = []
    for term in (declared if isinstance(declared, dict) else {}):
        for bad_grade in ('一年级', '六年级'):
            candidate = f'{term}（{bad_grade}）'
            resolved = taxonomy.resolve(candidate)
            if resolved is not None:
                wrong_grade.append((candidate, resolved))
    check('Q6-QUAL', '限定到不存在的年级不得静默命中', not wrong_grade, f'异常={wrong_grade}')


def main():
    print(f'python={sys.version.split()[0]}  root={ROOT}')
    for func in (q3_hard_reject_page_argument, q4_pairing_counterexamples,
                 q5_injection, q6_taxonomy):
        try:
            func()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            RESULTS.append({'item': func.__name__, 'name': 'EXCEPTION', 'ok': False,
                            'detail': traceback.format_exc().splitlines()[-1]})

    failed = [item for item in RESULTS if not item['ok']]
    print('\n==== 汇总 ====')
    print(f'断言 {len(RESULTS)} 条，失败 {len(failed)} 条')
    for item in failed:
        print('  FAIL', item['item'], item['name'], '::', item['detail'])
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
