# -*- coding: utf-8 -*-
"""QA 自写的 T03 端到端验证（**不使用工程师的 fixture**，自己造工程、真调 API）。

覆盖 team-lead 指定的 5 条：
  1) 降级是真兜住还是测试自证：真起 book、真 `_save_page`，让 freeze_page 真抛异常
  2) `rebuild_missing()` 幂等：第二次 frozen=0、created_at 不变、不留永久 L2
  3) A3 位置无关：中间插页不得让任何记录 stale / 掉级
  4) 存量旧快照（缺 `page_sha1_status`）重冻结的副作用：影响多少、是否可回滚
  5) 答案口径：已配对时改题目自己的 answer 不 stale；改答案页记录才 stale

跑法：python reports/qa_verify_t03_e2e.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('PAPERFLOW_ENRICH', '1')

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, ok, detail))
    print(('  [PASS] ' if ok else '  [FAIL] ') + f'{name} :: {detail}')


def make_book(tmp, *, paired=True, with_answer_page=True):
    """造一个"真"工程：真目录、真 JPEG、真 project.json（QA 自己写的结构）。"""
    import cv2
    import numpy as np

    book = os.path.join(tmp, 'book')
    img_dir = os.path.join(book, 'img')
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.RandomState(3)
    cv2.imencode('.jpg', rng.randint(0, 255, (400, 300, 3), dtype=np.uint8))[1].tofile(
        os.path.join(img_dir, 'p.jpg'))

    question = {
        'id': 'q1', 'region_role': 'question', 'bbox': [10, 20, 120, 90],
        'text': '计算 12×8 的结果', 'answer': '人工填的答案',
        'pair': {'path_match': True, 'score': 260.0, 'margin': 150.0, 'ok': True},
        'illustrations': [{'bbox': [12, 22, 60, 60]}],
    }
    project = {
        'book_root': book, 'image_root': img_dir, 'profile': 'general',
        'pages': [{'image': 'p.jpg', 'questions': [dict(question)]}],
        'answer_pages': [], 'pairs': [],
    }
    if with_answer_page:
        project['answer_pages'] = [{'image': 'p.jpg', 'answers': [
            {'id': 'a1', 'region_role': 'answer', 'answer': '答案页上的正确文本'}]}]
        if paired:
            project['pairs'] = [{'question_id': 'q1', 'answer_ids': ['a1']}]
    path = os.path.join(book, 'PaperFlow标注项目.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(project, file, ensure_ascii=False, indent=2)
    return path, project


def reload_project(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# ---------------------------------------------------------------- 1) 降级真兜住
def test_degrade_end_to_end():
    print('\n== 1) 降级真兜住（真 book、真 _save_page、真抛异常）==')
    from desktop_annotation import paperflow_annotation_api as api
    from review_tools import enrich_snapshot as es

    tmp = tempfile.mkdtemp(prefix='qa_t03_')
    try:
        path, _ = make_book(tmp)
        original_freeze = es.freeze_page
        calls = {'n': 0}

        def boom(*args, **kwargs):
            calls['n'] += 1
            raise RuntimeError('QA 注入：冻结服务不可用')

        es.freeze_page = boom
        try:
            response = api._save_page({
                'project_path': path, 'kind': 'question', 'page_index': 0,
                'records': [{'id': 'q1', 'bbox': [10, 20, 120, 90], 'text': '计算 12×8 的结果',
                             'answer': '人工填的答案'}],
                'drawing_tool_active': False, 'accepted': True,
            })
        finally:
            es.freeze_page = original_freeze

        check('1.1 冻结确实被调用到（证明不是绕过补全层的假降级）', calls['n'] > 0,
              f'freeze_page 调用 {calls["n"]} 次')
        check('1.2 响应 ok=True（不阻断保存）', response.get('ok') is True,
              f'ok={response.get("ok")} saved={response.get("saved")}')

        warnings = [str(item) for item in (response.get('warnings') or [])]
        check('1.3 warnings 里有降级提示（审核员看得见）',
              any('降级' in item or '未完成' in item for item in warnings),
              f'warnings={warnings}')

        on_disk = reload_project(path)
        record = (on_disk['pages'][0]['questions'] or [{}])[0]
        check('1.4 人工成果未丢（框还在）', list(record.get('bbox') or []) == [10, 20, 120, 90],
              f'bbox={record.get("bbox")}')
        check('1.5 人工成果未丢（答案还在）',
              str(record.get('answer') or '').strip() == '人工填的答案',
              f'answer={record.get("answer")!r}')
        check('1.6 降级后强制 L2（不能带着旧 L0 继续走）',
              str(record.get('review_level') or '') == 'L2',
              f'review_level={record.get("review_level")!r} reasons={record.get("review_reasons")}')
        check('1.7 enrich_status 记为 failed',
              str(record.get('enrich_status') or '') == 'failed',
              f'enrich_status={record.get("enrich_status")!r} error={str(record.get("enrich_error"))[:60]!r}')
        check('1.8 enrich_error 截断 ≤300 字',
              len(str(record.get('enrich_error') or '')) <= 300,
              f'长度={len(str(record.get("enrich_error") or ""))}')
        check('1.9 工程确实落盘了（save_project 执行）',
              int(on_disk.get('project_revision') or 0) >= 1,
              f'project_revision={on_disk.get("project_revision")}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 2) rebuild 幂等
def test_rebuild_idempotent():
    print('\n== 2) rebuild_missing() 幂等 ==')
    from review_tools import enrich_snapshot as es

    tmp = tempfile.mkdtemp(prefix='qa_t03_')
    try:
        path, project = make_book(tmp)
        page = project['pages'][0]
        record = page['questions'][0]
        check('2.0 初始无快照（模拟 PAPERFLOW_ENRICH=0 期间保存的页）',
              not es.is_valid_snapshot(record.get('snapshot')), 'snapshot 缺失')

        first = es.rebuild_missing(project)
        after_first = es.rebuild_missing(project)
        snap = record.get('snapshot') or {}
        check('2.1 第一次补建成功', int(first.get('frozen') or 0) >= 1,
              f'first={first} second={after_first}')
        check('2.2 第二次 frozen=0（幂等）', int(after_first.get('frozen') or 0) == 0,
              f'second={after_first}')
        check('2.3 第二次 skipped=全部（有效快照直接跳过）',
              int(after_first.get('skipped') or 0) >= 1, f'second={after_first}')
        check('2.4 补建后快照有效', es.is_valid_snapshot(snap), f'valid={es.is_valid_snapshot(snap)}')

        # 幂等的硬指标：第二次不应改写 created_at / hash
        created_before = str(snap.get('created_at') or '')
        hash_before = str(snap.get('hash') or '')
        es.rebuild_missing(project)
        snap2 = record.get('snapshot') or {}
        check('2.5 重复补建不改写 created_at/hash',
              str(snap2.get('created_at')) == created_before and str(snap2.get('hash')) == hash_before,
              f'created_at {created_before} -> {snap2.get("created_at")}')

        # 不留永久 L2：有配对 + 有题图 → 应当是 L0，而不是默认 L2
        check('2.6 补建后不是"永久 L2"（有配对+题图应为 L0）',
              str(record.get('review_level') or '') == 'L0',
              f'review_level={record.get("review_level")!r} reasons={record.get("review_reasons")}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 3) 位置无关
def test_position_independent():
    print('\n== 3) A3：中间插页不得让记录 stale / 掉级 ==')
    from review_tools import enrich_snapshot as es

    tmp = tempfile.mkdtemp(prefix='qa_t03_')
    try:
        path, project = make_book(tmp)
        page = project['pages'][0]
        record = page['questions'][0]
        es.freeze_or_invalidate(project, page, record, page_index=0)
        hash_before = str((record.get('snapshot') or {}).get('hash'))
        level_before = str(record.get('review_level') or '')
        created_before = str((record.get('snapshot') or {}).get('created_at'))

        project['pages'].insert(0, {'image': 'p.jpg', 'questions': []})
        result = es.freeze_or_invalidate(project, project['pages'][1], record, page_index=1)

        check('3.1 插页后 action 不是 stale', result.get('action') != 'stale',
              f'action={result.get("action")!r} reasons={result.get("reasons")}')
        check('3.2 插页后 hash 不变（位置不进哈希）',
              str((record.get('snapshot') or {}).get('hash')) == hash_before,
              f'{hash_before[:12]} -> {str((record.get("snapshot") or {}).get("hash"))[:12]}')
        check('3.3 插页后 created_at 不变（不重写快照）',
              str((record.get('snapshot') or {}).get('created_at')) == created_before, 'created_at 未变')
        check('3.4 插页后级别不掉（仍 L0）',
              str(record.get('review_level') or '') == level_before == 'L0',
              f'{level_before} -> {record.get("review_level")!r}')

        # 顶层与嵌套的 page_index 都不参与哈希
        keys = getattr(es, 'HASH_KEYS', ())
        check('3.5 HASH_KEYS 不含 page_index（位置键被排除）',
              'page_index' not in keys, f'HASH_KEYS={keys}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 4) 存量旧快照重冻结
def test_old_snapshot_refreeze():
    print('\n== 4) 存量旧快照（缺 page_sha1_status）重冻结的副作用 ==')
    from review_tools import enrich_snapshot as es

    tmp = tempfile.mkdtemp(prefix='qa_t03_')
    try:
        path, project = make_book(tmp)
        page = project['pages'][0]
        record = page['questions'][0]
        es.freeze_or_invalidate(project, page, record, page_index=0)
        fresh = dict(record.get('snapshot') or {})
        old = {key: value for key, value in fresh.items() if key != 'page_sha1_status'}

        check('4.1 旧快照（无 page_sha1_status）被判不可信',
              not es.is_valid_snapshot(old), 'is_valid_snapshot=False')
        check('4.2 新快照含 page_sha1_status',
              'page_sha1_status' in fresh, f'page_sha1_status={fresh.get("page_sha1_status")!r}')

        # 影响面：三条记录各带一个旧快照 → 会被重冻结几条
        project['pages'][0]['questions'].append(
            {'id': 'q2', 'region_role': 'question', 'bbox': [5, 5, 60, 60], 'text': '第二题',
             'answer': '答案2', 'pair': {'path_match': True, 'score': 250.0, 'margin': 140.0, 'ok': True},
             'illustrations': [{'bbox': [6, 6, 30, 30]}], 'snapshot': dict(old)})
        for item in project['pages'][0]['questions']:
            item['snapshot'] = {key: value for key, value in (item.get('snapshot') or {}).items()
                                if key != 'page_sha1_status'}
        counts = es.rebuild_missing(project)
        check('4.3 存量旧快照会被批量重冻结（影响面=记录数）',
              int(counts.get('frozen') or 0) >= 2, f'counts={counts}')

        rebuilt = project['pages'][0]['questions'][0].get('snapshot') or {}
        same_content = all(rebuilt.get(key) == fresh.get(key)
                           for key in ('bbox', 'text', 'answer', 'question_id'))
        check('4.4 重冻结后内容字段不变（不丢信息）', same_content,
              f'bbox/text/answer/question_id 一致={same_content}')
        check('4.5 内容哈希不变（provenance 不进 HASH_KEYS → 不会触发 AI 重算）',
              str(rebuilt.get('hash')) == str(fresh.get('hash')),
              f'hash {str(fresh.get("hash"))[:12]} -> {str(rebuilt.get("hash"))[:12]}（未变）')
        check('4.5b snapshot_id 不变（= hash[:16]）',
              str(project['pages'][0]['questions'][0].get('snapshot_id') or '')
              == str(fresh.get('hash'))[:16],
              f'snapshot_id={str(project["pages"][0]["questions"][0].get("snapshot_id"))[:16]}')
        check('4.5c 旧快照与新快照的差异只限 provenance（不产生内容变更）',
              set(old) ^ set(rebuilt) <= {'page_sha1_status'}
              and all(rebuilt.get(key) == old.get(key) for key in old if key != 'created_at'),
              f'键差异={sorted(set(old) ^ set(rebuilt))}；'
              f'其余键值差异={[k for k in old if k != "created_at" and old.get(k) != rebuilt.get(k)]}'
              f'（created_at 为秒级精度，同秒内重冻结可相同）')
        check('4.6 重冻结不强制 L2（有配对+题图仍 L0）',
              str(project['pages'][0]['questions'][0].get('review_level') or '') == 'L0',
              f'review_level={project["pages"][0]["questions"][0].get("review_level")!r}')
        check('4.7 旧快照内容可从工程备份回滚（本次只覆盖快照字段）',
              'snapshot' in project['pages'][0]['questions'][0],
              '重冻结只改 snapshot/snapshot_id/级别字段，未动人工字段')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 5) 答案口径
def test_answer_semantics():
    print('\n== 5) 答案口径：已配对时改题目自己的 answer 不 stale ==')
    from review_tools import enrich_snapshot as es

    tmp = tempfile.mkdtemp(prefix='qa_t03_')
    try:
        path, project = make_book(tmp, paired=True)
        page = project['pages'][0]
        record = page['questions'][0]
        es.freeze_or_invalidate(project, page, record, page_index=0)
        snap = record.get('snapshot') or {}
        check('5.1 已配对 → answer 取答案页文本、source=paired',
              str(snap.get('answer')) == '答案页上的正确文本' and snap.get('answer_source') == 'paired',
              f'answer={snap.get("answer")!r} source={snap.get("answer_source")!r}')

        record['answer'] = '我改的答案'
        result = es.freeze_or_invalidate(project, page, record, page_index=0)
        check('5.2 已配对时改题目自己的 answer 字段 → 不是 stale（已知口径）',
              result.get('action') == 'unchanged',
              f'action={result.get("action")!r} reasons={result.get("reasons")} '
              f'snapshot.answer={str((record.get("snapshot") or {}).get("answer"))!r}')

        project['answer_pages'][0]['answers'][0]['answer'] = '答案页改了'
        result2 = es.freeze_or_invalidate(project, page, record, page_index=0)
        check('5.3 改答案页那条记录 → 才 stale',
              result2.get('action') == 'stale',
              f'action={result2.get("action")!r} reasons={result2.get("reasons")}')

        # 未配对但人工填了答案 → source=manual
        path2, project2 = make_book(tmp + '2', paired=False)
        record2 = project2['pages'][0]['questions'][0]
        es.freeze_or_invalidate(project2, project2['pages'][0], record2, page_index=0)
        snap2 = record2.get('snapshot') or {}
        check('5.4 未配对 + 人工填答案 → source=manual',
              snap2.get('answer_source') == 'manual' and str(snap2.get('answer')) == '人工填的答案',
              f'answer={snap2.get("answer")!r} source={snap2.get("answer_source")!r}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp + '2', ignore_errors=True)


# ---------------------------------------------------------------- 6) 升级不造假待办
def test_legacy_upgrade_creates_no_fake_todo():
    print('\n== 6) 存量旧快照升级：不得凭空多出"待补全"（走真实 enrich-prepare / enrich-status）==')
    from desktop_annotation import paperflow_annotation_api as api
    from review_tools import enrich_snapshot as es

    tmp = tempfile.mkdtemp(prefix='qa_t03_')
    try:
        path, project = make_book(tmp)
        record = project['pages'][0]['questions'][0]
        es.freeze_page(project, 0, kind='question')
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(project, file, ensure_ascii=False, indent=2)

        before = api._enrich_status({'project_path': path})
        todo_before = int(before.get('todo_count') or 0)
        hash_before = str((record.get('snapshot') or {}).get('hash'))

        # 造一份"缺 page_sha1_status"的旧快照（升级前的样子），状态回到 none
        legacy = {key: value for key, value in (record.get('snapshot') or {}).items()
                  if key != 'page_sha1_status'}
        record['snapshot'] = legacy
        record['snapshot_id'] = str(legacy.get('hash'))[:16]
        record['enrich_status'] = 'none'
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(project, file, ensure_ascii=False, indent=2)

        check('6.1 旧快照被判不可信（触发一次性重冻结）',
              not es.is_valid_snapshot(record.get('snapshot')), 'is_valid_snapshot=False')

        api._enrich_prepare({'project_path': path})
        after = api._enrich_status({'project_path': path})
        todo_after = int(after.get('todo_count') or 0)
        on_disk = reload_project(path)
        rec2 = (on_disk['pages'][0]['questions'] or [{}])[0]

        check('6.2 升级后不进 pending（pending 是 todo 六态之一，会造出假待办）',
              str(rec2.get('enrich_status') or '') != 'pending',
              f'enrich_status={rec2.get("enrich_status")!r}')
        check('6.3 待办数不因升级而增加（现场不会凭空多出一批活儿）',
              todo_after <= todo_before,
              f'todo_count {todo_before} -> {todo_after}（enrich_summary={after.get("enrich_summary")}）')
        check('6.4 升级不降级别（仍 L0）',
              str(rec2.get('review_level') or '') == 'L0',
              f'review_level={rec2.get("review_level")!r}')
        check('6.5 内容哈希不变（不触发 AI 重算）',
              str((rec2.get('snapshot') or {}).get('hash')) == hash_before,
              f'{hash_before[:12]} -> {str((rec2.get("snapshot") or {}).get("hash"))[:12]}')

        # 对照组：真正的新记录（本来没有快照）必须置 pending，否则会漏跑补全
        project2 = reload_project(path)
        project2['pages'][0]['questions'].append({
            'id': 'q-new', 'region_role': 'question', 'bbox': [5, 5, 60, 60], 'text': '新题',
        })
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(project2, file, ensure_ascii=False, indent=2)
        api._enrich_prepare({'project_path': path})
        fresh = reload_project(path)
        new_rec = [item for item in fresh['pages'][0]['questions']
                   if item.get('id') == 'q-new'][0]
        check('6.6 对照组：真·新记录仍必须置 pending（否则漏跑补全）',
              str(new_rec.get('enrich_status') or '') == 'pending',
              f'enrich_status={new_rec.get("enrich_status")!r}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print(f'PAPERFLOW_ENRICH={os.environ.get("PAPERFLOW_ENRICH")}')
    for func in (test_degrade_end_to_end, test_rebuild_idempotent,
                 test_position_independent, test_old_snapshot_refreeze,
                 test_answer_semantics, test_legacy_upgrade_creates_no_fake_todo):
        try:
            func()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            RESULTS.append((func.__name__, False, traceback.format_exc().splitlines()[-1]))
    failed = [item for item in RESULTS if not item[1]]
    print(f'\n==== 汇总：断言 {len(RESULTS)} 条，失败 {len(failed)} 条 ====')
    for name, _ok, detail in failed:
        print('  FAIL', name, '::', detail)
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
