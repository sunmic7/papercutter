# -*- coding: utf-8 -*-
"""快照行为锁定测试（T03 缺口补测，只新增不改既有）。

背景：`tests/test_enrich_snapshot.py` 的 19 项覆盖了 §3.5 的五条失效条件与
§3.3 的哈希稳定性，但 QA 复核出 5 个**没有断言**的关键行为。本文件只补这些缺口，
每一条都对应一个"会让 R2 事故复发"或"会让脏数据静默进交付"的具体风险：

G1 冻结只读：不落盘、业务内容一字不动、记录上不出现预期外的新键
   —— 这是 R2（渲染层整工程 autosave 抹掉 AI 字段）的唯一防线的另一半：
      冻结过程本身绝不能改业务字段，也绝不能自己去写盘。
G2 回填幂等 且 不覆盖 ``field_source == 'human'``
   —— 自动回填重复执行不能把人工改过的字段标成 stale。
G3 答案后到：先确认题页、后确认答案页时，``snapshot.answer`` 必须被回填，
   ``answer_source`` 变 ``paired``，并标 stale 等 T04 重跑。
   —— 不回填的话，快照会永久缺答案，AI 解析只能瞎猜（§3.4）。
G4 快照可 JSON 序列化且体积有上限 —— 快照内嵌在 project.json 里，体积失控会拖死工程。
G5 元测试：``REQUIRED_KEYS ⊆ build_snapshot()``
   —— 只断言子集，**不写相等**：快照允许多带诊断键；而缺键会让所有快照一律判旧。
G6 已知缺口（expectedFailure）：冻结抛异常时不应中断整页保存。

G6 的现状（2026-10-20 实测）：`paperflow_annotation_api._save_page_enrich()` 里
没有任何 try/except，而 `_save_page()` 是在它**之后**才 `save_project()`。所以冻结
一旦抛异常，整页保存会连同人工刚画的框一起丢失。用 ``expectedFailure`` 标记：
将来补上降级后本条会变成 "unexpected success"，届时请把它改成普通断言。
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from review_tools import enrich_snapshot as snapshot
from review_tools import manual_annotation_data as annotation_data

# 业务内容键：冻结**不允许**改动其中任何一个
CONTENT_KEYS = (
    'id', 'bbox', 'text', 'options', 'illustrations', 'answer', 'answer_ids',
    'knowledge_points', 'analysis', 'difficulty', 'question_type', 'pair',
    'region_role', 'ocr',
)
# 冻结**允许**写入的键（其余新键一律视为越界）
SNAPSHOT_WRITE_KEYS = {
    'snapshot', 'snapshot_id', 'review_level', 'review_reasons',
    'review_trace', 'enrich_status', 'enrich_error', 'field_source',
}
PAIRED = {'path_match': True, 'score': 300.0, 'margin': 200.0,
          'ok': True, 'display_confidence': 1.0}


def _record(record_id: str, box, text: str = '') -> dict:
    return {
        'id': record_id,
        'bbox': list(box),
        'text': text,
        'region_role': 'question',
        'options': [],
        'illustrations': [],
    }


class _Fixture(unittest.TestCase):
    """与既有测试同构的最小工程：一页两题、一张答案页。"""

    def setUp(self) -> None:
        snapshot.clear_page_hash_cache()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (300, 100), (0, 0, 0), 2)
        cv2.rectangle(image, (10, 130), (300, 220), (0, 0, 0), 2)
        cv2.imwrite(str(self.root / '1.png'), image)

        self.question_one = {**_record('q-1', [10, 10, 300, 100], '1. 计算 12×4'),
                             'pair': dict(PAIRED)}
        self.question_two = {**_record('q-2', [10, 130, 300, 220], '2. 计算 3+5'),
                             'pair': dict(PAIRED)}
        self.answer_one = {'id': 'a-1', 'bbox': [10, 10, 300, 100], 'answer': '48',
                           'region_role': 'question', 'options': [], 'illustrations': []}
        self.project = {
            'image_root': str(self.root),
            'answer_image_root': str(self.root),
            'language': 'zh', 'stage': '小学', 'grade': '四年级', 'subject': '数学',
            'profile': 'exam', 'difficulty_scheme': '1-5',
            'enrichment': {'taxonomy_version': 'math-2026.10',
                           'prompt_version': 'enrich-v1'},
            'pairs': [{'question_id': 'q-1', 'answer_ids': ['a-1'], 'status': 'accepted'}],
            'pages': [{'image': '1.png', 'status': 'accepted',
                       'questions': [self.question_one, self.question_two]}],
            'answer_pages': [{'image': '1.png', 'status': 'accepted',
                              'answers': [self.answer_one]}],
        }
        self.page = self.project['pages'][0]

    def tearDown(self) -> None:
        snapshot.clear_page_hash_cache()
        self.temp.cleanup()


class ReadOnlyFreezeTests(_Fixture):
    """G1：冻结只读——不落盘、不改业务内容、不新增预期外的键。"""

    def test_freeze_never_persists(self) -> None:
        with mock.patch.object(annotation_data, 'save_project') as saver:
            snapshot.freeze_page(self.project, 0, kind='question')
        self.assertFalse(saver.called,
                         '冻结不允许自己写盘：一次保存只应由 _save_page 写一次')

    def test_freeze_keeps_every_business_field(self) -> None:
        records = [self.question_one, self.question_two]
        before = {record['id']: copy.deepcopy({key: record.get(key) for key in CONTENT_KEYS})
                  for record in records}
        page_before = {'image': self.page['image'], 'status': self.page['status']}

        snapshot.freeze_page(self.project, 0, kind='question')

        for record in records:
            for key in CONTENT_KEYS:
                with self.subTest(record=record['id'], key=key):
                    self.assertEqual(record.get(key), before[record['id']][key],
                                     f'冻结改动了业务字段 {key}')
        self.assertEqual(self.page['image'], page_before['image'])
        self.assertEqual(self.page['status'], page_before['status'])

    def test_freeze_does_not_invent_unexpected_record_keys(self) -> None:
        before_keys = {record['id']: set(record)
                       for record in (self.question_one, self.question_two)}
        snapshot.freeze_page(self.project, 0, kind='question')
        for record in (self.question_one, self.question_two):
            with self.subTest(record=record['id']):
                added = set(record) - before_keys[record['id']]
                self.assertLessEqual(
                    added, SNAPSHOT_WRITE_KEYS,
                    f'冻结写入了预期外的键：{sorted(added - SNAPSHOT_WRITE_KEYS)}',
                )
                self.assertIn('snapshot', record)


class BackfillIdempotencyTests(_Fixture):
    """G2：回填幂等，且绝不覆盖 ``field_source == 'human'``。"""

    def test_repeated_backfill_is_idempotent(self) -> None:
        snapshot.freeze_page(self.project, 0, kind='question')
        first = snapshot.refresh_answer_segment(self.project, ['a-1'])
        hash_after_first = self.question_one['snapshot']['hash']
        snapshot_id_after_first = self.question_one['snapshot_id']

        second = snapshot.refresh_answer_segment(self.project, ['a-1'])

        self.assertEqual(self.question_one['snapshot']['hash'], hash_after_first)
        self.assertEqual(self.question_one['snapshot_id'], snapshot_id_after_first)
        self.assertEqual(second.get('updated', 0), 0,
                         '第二次回填不应再判定为"已更新"')
        self.assertEqual(second.get('question_ids', []), [])
        self.assertLessEqual(first.get('updated', 0), 1)

    def test_stale_marking_skips_human_sourced_fields(self) -> None:
        snapshot.freeze_page(self.project, 0, kind='question')
        record = self.question_one
        record['field_source'] = {'analysis': 'human', 'knowledge_points': 'ai'}

        record['text'] = '1. 计算 12×5'          # 人改了题面 → 触发 stale
        result = snapshot.freeze_page(self.project, 0, kind='question')

        self.assertGreaterEqual(result['stale'], 1)
        self.assertEqual(record['field_source']['analysis'], 'human',
                         '人工来源的字段不允许被自动标记覆盖')
        self.assertEqual(record['field_source']['knowledge_points'], 'stale')

    def test_stale_marking_does_not_create_sources_for_unset_fields(self) -> None:
        snapshot.freeze_page(self.project, 0, kind='question')
        record = self.question_one
        record['field_source'] = {}
        record['text'] = '1. 计算 12×6'
        snapshot.freeze_page(self.project, 0, kind='question')
        # 没有来源的字段不凭空造一条记录（否则来源面板会出现幽灵行）
        self.assertEqual(record['field_source'], {})


class AnswerArrivesLateTests(_Fixture):
    """G3：答案后到——先冻结题页，答案页确认后必须回填进快照。"""

    def test_backfill_fills_answer_and_marks_paired(self) -> None:
        self.project['pairs'] = []              # 确认题页时还没有配对答案
        snapshot.freeze_page(self.project, 0, kind='question')
        self.assertEqual(self.question_one['snapshot']['answer'], '')
        self.assertEqual(self.question_one['snapshot']['answer_source'], 'missing')

        # 答案页后来才确认 → 配对建立 → 回填
        self.project['pairs'] = [
            {'question_id': 'q-1', 'answer_ids': ['a-1'], 'status': 'accepted'},
        ]
        result = snapshot.refresh_answer_segment(self.project, ['a-1'])

        self.assertEqual(self.question_one['snapshot']['answer'], '48')
        self.assertEqual(self.question_one['snapshot']['answer_source'], 'paired')
        self.assertIn('q-1', result['question_ids'])
        self.assertEqual(result['actions'].get('q-1'), snapshot.ACTION_STALE)

    def test_backfill_is_noop_without_matching_answer_ids(self) -> None:
        snapshot.freeze_page(self.project, 0, kind='question')
        before = self.question_one['snapshot']['hash']
        result = snapshot.refresh_answer_segment(self.project, ['a-unknown'])
        self.assertEqual(result.get('updated', 0), 0)
        self.assertEqual(self.question_one['snapshot']['hash'], before)

    def test_manual_answer_is_not_labelled_paired(self) -> None:
        self.project['pairs'] = []
        self.question_one['answer'] = '48'      # 人工直接填的答案
        snapshot.freeze_page(self.project, 0, kind='question')
        self.assertEqual(self.question_one['snapshot']['answer_source'], 'manual')


class SnapshotShapeTests(_Fixture):
    """G4 / G5：体积与可序列化；必需键只能是子集关系。"""

    def test_snapshot_is_json_serializable_within_size_budget(self) -> None:
        snap = snapshot.build_snapshot(self.project, self.page, self.question_one,
                                       page_index=0)
        payload = json.dumps(snap, ensure_ascii=False)
        self.assertEqual(json.loads(payload)['version'], snapshot.SNAPSHOT_VERSION)
        size = len(payload.encode('utf-8'))
        # 快照内嵌在 project.json 里：单题超过 2KB 会让整本工程迅速膨胀
        self.assertLessEqual(size, 2048, f'单题快照 {size} 字节，超过 2KB 预算')

    def test_required_keys_are_a_subset_of_the_built_snapshot(self) -> None:
        snap = snapshot.build_snapshot(self.project, self.page, self.question_one,
                                       page_index=0)
        missing = sorted(set(snapshot.REQUIRED_KEYS) - set(snap))
        # 反向（快照多带诊断键）是设计，**不做相等断言**
        self.assertEqual(missing, [],
                         f'snapshot 缺必需键 {missing}：缺键会让所有快照一律判旧')


def _load_api():
    """按 tests/test_annotation_api.py 的方式加载桥服务模块（desktop_annotation 不是包）。"""
    root = Path(__file__).resolve().parents[1]
    path = root / 'desktop_annotation' / 'paperflow_annotation_api.py'
    os.environ.setdefault('PAPERFLOW_DISABLE_PROFILE_DETECT', '1')
    spec = importlib.util.spec_from_file_location('paperflow_annotation_api_gaps', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PositionIndependentHashTests(_Fixture):
    """A3：位置不进内容哈希 —— 插一页不该让全书重算（A3 决议口径 ②）。

    锁三件事：①在中间插一页，原记录 hash 不变、快照仍有效、不产生 stale；
    ②嵌套的 ``shared_context_segments[*].page_index`` 变化同样不影响 hash
    （**这是最容易漏的一条**：只把顶层 page_index 排除掉是不够的）；
    ③正文/选项/答案任一真变化仍必须 stale —— 不能为了"不误判"把真失效也吃掉。
    """

    def _insert_page_before(self, project: dict) -> None:
        project['pages'].insert(0, {'image': '0.png', 'status': 'unlabeled',
                                    'questions': []})

    def test_inserting_a_page_keeps_existing_snapshot_valid(self) -> None:
        snapshot.freeze_page(self.project, 0, kind='question')
        before_hash = self.question_one['snapshot']['hash']
        before_page_index = self.question_one['snapshot'].get('page_index')

        self._insert_page_before(self.project)
        result = snapshot.freeze_page(self.project, 1, kind='question')

        self.assertEqual(result['stale'], 0, '插一页不应让原有记录判 stale')
        self.assertEqual(result['frozen'], 0, '内容没变就不该重新冻结')
        self.assertEqual(self.question_one['snapshot']['hash'], before_hash)
        self.assertTrue(snapshot.is_valid_snapshot(self.question_one['snapshot']))
        # 位置本身仍会更新（快照要能说明"当时在第几页"），只是不参与哈希
        self.assertNotEqual(self.question_one['snapshot'].get('page_index'),
                            before_page_index)

    def test_nested_shared_context_page_index_does_not_affect_hash(self) -> None:
        self.question_one['shared_context_segments'] = [
            {'page_index': 0, 'bbox': [0, 0, 10, 10]},
        ]
        snapshot.freeze_page(self.project, 0, kind='question')
        before_hash = self.question_one['snapshot']['hash']

        self.question_one['shared_context_segments'] = [
            {'page_index': 7, 'bbox': [0, 0, 10, 10]},
        ]
        result = snapshot.freeze_page(self.project, 0, kind='question')

        self.assertEqual(result['stale'], 0)
        self.assertEqual(self.question_one['snapshot']['hash'], before_hash)

    def test_real_content_changes_still_go_stale(self) -> None:
        # 注意：快照里的 answer 段来自**配对的答案记录**（answer_segment 优先读
        # answer_ids），所以改"答案"必须改答案页那条记录，改题目自己的 answer 字段
        # 在已配对时是无效输入 —— 那条由 test_manual_answer_* 覆盖。
        cases = (
            ('text', lambda rec: rec.__setitem__('text', '1. 计算 12×5')),
            ('options', lambda rec: rec.__setitem__('options', ['A. 1', 'B. 2'])),
            ('paired answer', lambda _rec: self.answer_one.__setitem__('answer', '49')),
        )
        for field, apply_change in cases:
            with self.subTest(field=field):
                record = dict(self.question_one)          # 每轮从干净起点开始
                self.project['pages'][0]['questions'] = [record]
                self.answer_one['answer'] = '48'          # 复位配对答案
                snapshot.freeze_page(self.project, 0, kind='question')
                before_hash = record['snapshot']['hash']

                apply_change(record)
                result = snapshot.freeze_page(self.project, 0, kind='question')

                self.assertEqual(result['stale'], 1,
                                 f'改 {field} 必须判 stale，不能被"位置无关"吃掉')
                self.assertNotEqual(record['snapshot']['hash'], before_hash)

    def test_hash_keys_exclude_position(self) -> None:
        # 元测试：写死 A3 决议，防止以后有人把 page_index 加回哈希
        self.assertNotIn('page_index', snapshot.HASH_KEYS)
        for key in snapshot.HASH_KEYS:
            with self.subTest(key=key):
                self.assertNotIn('page_index', key)


class PageFingerprintStatusTests(_Fixture):
    """C1：``page_sha1_status`` 区分"有指纹"与"这会儿读不到图"。"""

    def test_status_is_ok_when_source_image_is_readable(self) -> None:
        built = snapshot.build_snapshot(self.project, self.page, self.question_one,
                                        page_index=0)
        self.assertEqual(built['page_sha1_status'], 'ok')
        self.assertTrue(built['page_sha1'])

    def test_status_is_unavailable_when_image_missing(self) -> None:
        page = {'image': 'not-there.png', 'status': 'accepted', 'questions': []}
        built = snapshot.build_snapshot(self.project, page, self.question_one,
                                        page_index=0)
        self.assertEqual(built['page_sha1_status'], 'unavailable')
        self.assertEqual(built['page_sha1'], '')

    def test_missing_status_field_is_treated_as_untrusted(self) -> None:
        built = snapshot.build_snapshot(self.project, self.page, self.question_one,
                                        page_index=0)
        legacy = {key: value for key, value in built.items()
                  if key != 'page_sha1_status'}
        self.assertNotIn('page_sha1_status', legacy)
        # 旧快照缺这个字段 → is_valid 必须判不可信（否则门禁分不清"没图"和"图没变"）
        self.assertFalse(snapshot.is_valid_snapshot(legacy))


class RebuildMissingTests(_Fixture):
    """开关 0→1 的活锁：关闭期间保存的页面必须能补建快照。"""

    def test_rebuild_creates_snapshots_for_records_without_one(self) -> None:
        for record in (self.question_one, self.question_two):
            record.pop('snapshot', None)
            record.pop('snapshot_id', None)
        counts = snapshot.rebuild_missing(self.project)
        self.assertEqual(counts['frozen'], 2)
        self.assertEqual(counts['pages'], 1)
        for record in (self.question_one, self.question_two):
            self.assertTrue(snapshot.is_valid_snapshot(record.get('snapshot')))

    def test_rebuild_is_idempotent(self) -> None:
        first = snapshot.rebuild_missing(self.project)
        hashes = [record['snapshot']['hash'] for record in
                  (self.question_one, self.question_two)]
        second = snapshot.rebuild_missing(self.project)
        self.assertEqual(first['frozen'], 2)
        self.assertEqual(second['frozen'], 0)
        self.assertEqual(second['skipped'], 2)
        self.assertEqual([record['snapshot']['hash'] for record in
                          (self.question_one, self.question_two)], hashes)

    def test_rebuild_refreshes_levels_so_records_leave_permanent_l2(self) -> None:
        self.question_one['review_level'] = 'L2'
        snapshot.rebuild_missing(self.project)
        self.assertNotEqual(self.question_one.get('review_level'), 'L2',
                            '补建后应重算级别，不能停在默认 L2')


class PerceptualFingerprintCacheTests(_Fixture):
    """负缓存：读不到图时不许把空指纹缓存下来。"""

    def test_empty_fingerprint_is_not_cached(self) -> None:
        snapshot.clear_page_hash_cache()
        missing = str(self.root / 'nope.png')
        self.assertEqual(snapshot.page_sha1(missing), '')
        self.assertEqual(
            sum(1 for key in snapshot._PAGE_HASH_CACHE if key[0].endswith('nope.png')), 0,
            '空结果不得进缓存：否则图补齐后仍判不出指纹（负缓存只能等进程重启）',
        )

    def test_positive_fingerprint_is_cached(self) -> None:
        snapshot.clear_page_hash_cache()
        source = str(self.root / '1.png')
        first = snapshot.page_sha1(source)
        self.assertTrue(first)
        self.assertEqual(snapshot.page_sha1(source), first)


class LegacySnapshotUpgradeTests(_Fixture):
    """升级路径：旧快照缺新字段会被判不可信并重冻结，但不能凭空多出待办。"""

    def _legacy_snapshot(self):
        """先正常冻结一次，再造一份"缺 page_sha1_status"的旧快照。"""
        snapshot.freeze_page(self.project, 0, kind='question')
        full = dict(self.question_one['snapshot'])
        legacy = {key: value for key, value in full.items()
                  if key != 'page_sha1_status'}
        self.question_one['snapshot'] = legacy
        self.question_one['snapshot_id'] = str(legacy.get('hash'))[:16]
        self.question_one['enrich_status'] = 'none'
        return full

    def test_legacy_snapshot_is_invalid_and_refrozen(self) -> None:
        full = self._legacy_snapshot()
        self.assertFalse(snapshot.is_valid_snapshot(self.question_one['snapshot']))
        result = snapshot.freeze_page(self.project, 0, kind='question')
        self.assertEqual(result['frozen'], 1, '缺字段的旧快照要重建')
        self.assertEqual(self.question_one['snapshot']['hash'], full['hash'],
                         '内容没变 → 哈希不应变（否则会触发无谓的 AI 重算）')
        self.assertEqual(self.question_one['snapshot'].get('page_sha1_status'), 'ok')

    def test_legacy_refreeze_does_not_create_pending_todo(self) -> None:
        # pending 是 todo 的六种状态之一；升级重冻结若置 pending，现场会凭空多出
        # 一批"待补全"，而 AI 结果（prompt/taxonomy/哈希都没变）本来仍然有效。
        self._legacy_snapshot()
        snapshot.freeze_page(self.project, 0, kind='question')
        self.assertNotEqual(self.question_one.get('enrich_status'), 'pending')
        self.assertEqual(self.question_one.get('enrich_status'), 'none')

    def test_legacy_refreeze_does_not_downgrade_level(self) -> None:
        self._legacy_snapshot()
        snapshot.freeze_page(self.project, 0, kind='question')
        self.assertEqual(self.question_one.get('review_level'), 'L0')

    def test_genuinely_new_record_still_becomes_pending(self) -> None:
        # 对照组：真·新记录（本来就没有快照）必须置 pending，否则会漏跑补全。
        record = _record('q-new', [10, 10, 300, 100], '新题')
        self.project['pages'][0]['questions'].append(record)
        snapshot.freeze_page(self.project, 0, kind='question')
        self.assertEqual(record.get('enrich_status'), 'pending')


class SnapshotFailureDegradationTests(_Fixture):
    """G6：冻结抛异常时必须降级，且整页保存不能丢。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = _load_api()

    def test_snapshot_failure_is_downgraded_not_raised(self) -> None:
        """冻结异常必须降级成返回值，不能冒泡到 `_save_page`。

        2026-10-20 实测的缺陷：`_save_page_enrich()` 没有 try/except，而
        `_save_page()` 在其后才 `save_project()` —— 冻结一抛异常，整页保存连同
        人工刚画的框全部丢失。现已加降级兜底，本条由 expectedFailure 转正。
        """
        with mock.patch.object(self.api.enrich_snapshot, 'freeze_page',
                               side_effect=RuntimeError('模拟冻结失败')):
            result = self.api._save_page_enrich(self.project, 'question', 0, accepted=True)
        self.assertTrue(result.get('degraded'))
        self.assertIn('模拟冻结失败', str(result.get('error')))
        for record in (self.question_one, self.question_two):
            self.assertEqual(record.get('enrich_status'), 'failed')
            self.assertIn('模拟冻结失败', str(record.get('enrich_error')))

    def test_enrich_error_text_is_truncated_to_300_chars(self) -> None:
        text = self.api._enrich_error_text(RuntimeError('x' * 1000))
        self.assertEqual(len(text), 300)
        self.assertEqual(self.api._enrich_error_text(RuntimeError('')), 'RuntimeError')

    def test_page_save_still_persists_when_enrich_explodes(self) -> None:
        """端到端守住"补全不能阻断主流程"：冻结炸了，页面照样落盘、响应照样 ok。"""
        book = Path(self.temp.name) / 'book'
        (book / 'answers').mkdir(parents=True)
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(book / 'question.png'), image)
        opened = self.api._open_book({'book_root': str(book)})
        project_path = Path(opened['project_path'])
        page_index = next(
            index for index, page in enumerate(opened['project']['pages'])
            if page['image'].replace('\\', '/') == 'question.png'
        )
        payload = {
            'project_path': str(project_path), 'kind': 'question',
            'page_index': page_index, 'accepted': True,
            'records': [{'id': 'q-degrade', 'bbox': [10, 10, 80, 80],
                         'answer': '48'}],
        }
        with mock.patch.object(self.api.enrich_snapshot, 'freeze_page',
                               side_effect=RuntimeError('模拟冻结失败')):
            response = self.api._save_page(payload)

        self.assertTrue(response.get('ok'))
        self.assertTrue(response['enrich'].get('degraded'))
        self.assertTrue(any('降级' in str(item) for item in response.get('warnings') or []))
        saved = json.loads(project_path.read_text(encoding='utf-8'))
        page = next(item for item in saved['pages']
                    if item['image'].replace('\\', '/') == 'question.png')
        self.assertEqual([record['id'] for record in page['questions']], ['q-degrade'])
        self.assertEqual(page['questions'][0]['answer'], '48')
        self.assertEqual(page['questions'][0]['enrich_status'], 'failed')
        # 失败标记必须把该题顶到 L2（enrich_failed 规则），不能留着上次的 L0
        self.assertEqual(page['questions'][0]['review_level'], 'L2')


if __name__ == '__main__':
    unittest.main()
