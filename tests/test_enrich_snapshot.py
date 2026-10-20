# -*- coding: utf-8 -*-
"""快照冻结/失效（T03）的回归测试。

覆盖架构文档 §3.5 的五条失效条件、§3.6 的 record 级失效粒度、§3.4 的答案
后到回写，以及 §3.3 的内容哈希稳定性。全部只依赖纯逻辑 + 临时图片，
不加载 OCR / 版面模型。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import enrich_snapshot as snapshot


def _record(record_id: str, box, text: str = '', *, role: str = 'question') -> dict:
    return {
        'id': record_id,
        'bbox': list(box),
        'text': text,
        'region_role': role,
        'options': [],
        'illustrations': [],
    }


class EnrichSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        snapshot.clear_page_hash_cache()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.page_path = self.root / '1.png'
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (300, 100), (0, 0, 0), 2)
        cv2.rectangle(image, (10, 130), (300, 220), (0, 0, 0), 2)
        cv2.imwrite(str(self.page_path), image)

        paired = {'path_match': True, 'score': 300.0, 'margin': 200.0,
                  'ok': True, 'display_confidence': 1.0}
        self.question_one = {**_record('q-1', [10, 10, 300, 100], '1. 计算 12×4'),
                             'pair': dict(paired)}
        self.question_two = {**_record('q-2', [10, 130, 300, 220], '2. 计算 3+5'),
                             'pair': dict(paired)}
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

    def _freeze(self, record=None, **kwargs):
        record = self.question_one if record is None else record
        return snapshot.freeze_or_invalidate(
            self.project, self.page, record, page_index=0, **kwargs,
        )

    # ---------------------------------------------------------------- 哈希
    def test_identical_input_freezes_to_the_same_hash_twice(self) -> None:
        first = self._freeze()
        first_hash = self.question_one['snapshot']['hash']
        created_at = self.question_one['snapshot']['created_at']
        second = self._freeze()
        self.assertEqual(first['action'], snapshot.ACTION_FROZEN)
        self.assertEqual(second['action'], snapshot.ACTION_UNCHANGED)
        self.assertEqual(self.question_one['snapshot']['hash'], first_hash)
        self.assertEqual(self.question_one['snapshot']['created_at'], created_at,
                         '同输入再冻结必须保留原快照，不能刷新 created_at')
        self.assertEqual(self.question_one['snapshot_id'], first_hash[:16])

    def test_diagnostics_and_batch_id_do_not_change_the_content_hash(self) -> None:
        self._freeze()
        first_hash = self.question_one['snapshot']['hash']
        # 开新批次 / 重跑 OCR 的置信度抖动都不该让缓存全部失效
        self.project['enrichment']['batch_id'] = 'B0001-20261020-2'
        self.question_one['ocr'] = {'engine': 'paddleocr_local', 'confidence': 0.77}
        result = self._freeze()
        self.assertEqual(result['action'], snapshot.ACTION_UNCHANGED)
        self.assertEqual(self.question_one['snapshot']['hash'], first_hash)

    def test_freeze_stamps_snapshot_id_and_marks_the_question_pending(self) -> None:
        result = self._freeze()
        snap = self.question_one['snapshot']
        self.assertTrue(snap['hash'])
        self.assertEqual(snap['version'], snapshot.SNAPSHOT_VERSION)
        self.assertEqual(snap['page_index'], 0)
        self.assertEqual(snap['question_id'], 'q-1')
        self.assertEqual(snap['answer'], '48')
        self.assertEqual(snap['answer_ids'], ['a-1'])
        self.assertEqual(snap['answer_source'], 'paired')
        self.assertEqual(snap['taxonomy_version'], 'math-2026.10')
        self.assertEqual(self.question_one['enrich_status'], 'pending')
        self.assertEqual(result['review_level'], self.question_one['review_level'])
        self.assertTrue(snapshot.is_valid(self.question_one))

    # ---------------------------------------------------------------- 失效
    def test_bbox_change_invalidates_only_that_record(self) -> None:
        self._freeze(self.question_one)
        self._freeze(self.question_two)
        self.assertEqual(self.question_one['review_level'], 'L0')
        self.assertEqual(self.question_two['review_level'], 'L0')
        untouched_hash = self.question_two['snapshot']['hash']

        self.question_one['bbox'] = [12, 12, 300, 102]
        result = self._freeze(self.question_one)

        self.assertEqual(result['action'], snapshot.ACTION_STALE)
        self.assertEqual(self.question_one['enrich_status'], 'stale')
        self.assertEqual(self.question_one['review_level'], 'L2',
                         '人改了题框 → 该题强制 L2')
        self.assertIn('enrich_failed', self.question_one['review_reasons'])
        # 同页其它记录不受影响（§3.6 的硬要求）
        self.assertEqual(self.question_two['review_level'], 'L0')
        self.assertEqual(self.question_two['enrich_status'], 'pending')
        self.assertEqual(self.question_two['snapshot']['hash'], untouched_hash)

    def test_text_change_invalidates_and_marks_ai_fields_stale(self) -> None:
        self.question_one['field_source'] = {'knowledge_points': 'ai', 'analysis': 'ai'}
        self._freeze()
        self.question_one['text'] = '1. 计算 12×5'
        result = self._freeze()
        self.assertEqual(result['action'], snapshot.ACTION_STALE)
        self.assertIn('text_changed', result['reasons'])
        self.assertEqual(self.question_one['review_level'], 'L2')
        self.assertEqual(self.question_one['field_source']['knowledge_points'], 'stale')
        self.assertEqual(self.question_one['field_source']['analysis'], 'stale')

    def test_options_change_invalidates(self) -> None:
        self._freeze()
        self.question_one['options'] = ['A. 48', 'B. 46']
        result = self._freeze()
        self.assertEqual(result['action'], snapshot.ACTION_STALE)
        self.assertIn('options_changed', result['reasons'])

    def test_answer_change_invalidates(self) -> None:
        self._freeze()
        self.assertEqual(self.question_one['snapshot']['answer'], '48')
        self.answer_one['answer'] = '49'
        result = snapshot.refresh_answer_segment(self.project, ['a-1'])
        self.assertEqual(result['updated'], 1)
        self.assertEqual(self.question_one['snapshot']['answer'], '49')
        self.assertEqual(self.question_one['enrich_status'], 'stale')
        self.assertEqual(self.question_one['review_level'], 'L2')

    def test_page_image_change_invalidates_through_page_sha1(self) -> None:
        self._freeze()
        before = self.question_one['snapshot']['page_sha1']
        self.assertTrue(before)
        replacement = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.circle(replacement, (160, 120), 60, (0, 0, 0), 3)
        cv2.imwrite(str(self.page_path), replacement)
        snapshot.clear_page_hash_cache()
        result = self._freeze()
        self.assertNotEqual(self.question_one['snapshot']['page_sha1'], before)
        self.assertEqual(result['action'], snapshot.ACTION_STALE)
        self.assertIn('page_sha1_changed', result['reasons'])
        self.assertEqual(self.question_one['review_level'], 'L2')

    def test_taxonomy_version_change_marks_kp_stale_without_downgrading(self) -> None:
        self._freeze()
        self.assertEqual(self.question_one['review_level'], 'L0')
        self.project['enrichment']['taxonomy_version'] = 'math-2027.01'
        result = self._freeze()
        self.assertEqual(result['action'], snapshot.ACTION_KP_STALE)
        self.assertEqual(self.question_one['enrich_status'], 'kp_stale')
        self.assertEqual(self.question_one['review_level'], 'L0',
                         '词表升版不该让人重看题框')
        self.assertEqual(self.question_one['snapshot']['taxonomy_version'], 'math-2027.01')

    def test_prompt_version_change_marks_analysis_stale_without_downgrading(self) -> None:
        self._freeze()
        self.project['enrichment']['prompt_version'] = 'enrich-v2'
        result = self._freeze()
        self.assertEqual(result['action'], snapshot.ACTION_ANALYSIS_STALE)
        self.assertEqual(self.question_one['enrich_status'], 'analysis_stale')
        self.assertEqual(self.question_one['review_level'], 'L0')

    def test_book_meta_rev_change_invalidates_and_downgrades(self) -> None:
        """A3 决议：书级四字段变更 → stale + **强制 L2**（原"D6 不降级"已撤回）。"""
        self._freeze()
        before_rev = self.question_one['snapshot']['book_meta_rev']
        self.project['grade'] = '五年级'
        result = self._freeze()
        self.assertNotEqual(self.question_one['snapshot']['book_meta_rev'], before_rev)
        self.assertEqual(result['action'], snapshot.ACTION_BOOK_META_STALE)
        self.assertEqual(self.question_one['enrich_status'], 'stale')
        self.assertEqual(self.question_one['review_level'], 'L2',
                         '书级字段一错污染全书：宁可重看框，也不让旧学科的知识点混进交付')

    # ---------------------------------------------------------------- 有效性
    def test_is_valid_rejects_snapshots_with_missing_fields(self) -> None:
        self._freeze()
        broken = json.loads(json.dumps(self.question_one))
        broken['snapshot'].pop('answer')
        self.assertFalse(snapshot.is_valid(broken))
        self.assertFalse(snapshot.is_valid_snapshot(broken['snapshot']))

    def test_is_valid_rejects_an_old_snapshot_version(self) -> None:
        self._freeze()
        legacy = json.loads(json.dumps(self.question_one))
        legacy['snapshot']['version'] = 'snapshot-v0'
        self.assertFalse(snapshot.is_valid(legacy))
        # 版本不符的旧快照被当成"没有快照"：重新冻结，而不是判失效
        self.question_one['snapshot'] = legacy['snapshot']
        result = self._freeze()
        self.assertEqual(result['action'], snapshot.ACTION_FROZEN)

    def test_is_valid_rejects_a_tampered_hash(self) -> None:
        self._freeze()
        tampered = json.loads(json.dumps(self.question_one))
        tampered['snapshot']['text'] = '被改过的题面'
        self.assertFalse(snapshot.is_valid(tampered))
        tampered['snapshot_id'] = 'deadbeefdeadbeef'
        self.assertFalse(snapshot.is_valid(self.question_one | {'snapshot_id': 'deadbeefdeadbeef'}))

    # ---------------------------------------------------------------- 其它
    def test_answer_segment_is_empty_when_the_question_has_no_pair(self) -> None:
        text, ids = snapshot.answer_segment(self.project, self.question_two)
        self.assertEqual(text, '')
        self.assertEqual(ids, [])
        self._freeze(self.question_two)
        self.assertEqual(self.question_two['snapshot']['answer_source'], 'missing')

    def test_illustration_boxes_ignore_ai_candidates(self) -> None:
        self.question_one['illustrations'] = [{'bbox': [40, 40, 90, 90],
                                               'image_kind': 'question_image'}]
        self.question_one['illustration_candidates'] = [{'bbox': [1, 2, 3, 4],
                                                        'owner_id': 'q-1'}]
        self._freeze()
        self.assertEqual(self.question_one['snapshot']['illustration_boxes'],
                         [[40, 40, 90, 90]])
        self.assertNotIn('illustration_candidates', self.question_one['snapshot'])

    def test_continuation_segments_enter_the_snapshot(self) -> None:
        self.page['questions'].append({
            'id': 'q-1c', 'bbox': [10, 230, 300, 240], 'text': '续页',
            'region_role': 'continuation', 'continuation_of': 'q-1',
        })
        self._freeze()
        self.assertEqual(self.question_one['snapshot']['continuations'],
                         [{'page_index': 0, 'bbox': [10, 230, 300, 240]}])

    def test_freeze_page_skips_non_question_roles_and_writes_review_summary(self) -> None:
        self.page['questions'].append({
            'id': 'q-group', 'bbox': [0, 0, 320, 8], 'region_role': 'group',
        })
        result = snapshot.freeze_page(self.project, 0)
        self.assertEqual(result['frozen'], 2)
        self.assertNotIn('q-group', result['decisions'])
        self.assertNotIn('snapshot', self.page['questions'][2])
        self.assertEqual(self.page['review_summary'], {'L0': 2, 'L1': 0, 'L2': 0})

    def test_page_sha1_is_stable_for_the_same_file_and_differs_for_new_pixels(self) -> None:
        first = snapshot.page_sha1(self.page_path)
        self.assertTrue(first)
        self.assertEqual(snapshot.page_sha1(self.page_path), first)
        other = self.root / 'other.png'
        cv2.imwrite(str(other), np.full((240, 320, 3), 0, dtype=np.uint8))
        self.assertNotEqual(snapshot.page_sha1(other), first)
        self.assertEqual(snapshot.page_sha1(self.root / 'missing.png'), '')


if __name__ == '__main__':
    unittest.main()
