# -*- coding: utf-8 -*-
"""分级策略单测（review-levels-v1）。

重点是三处最容易写错的口径（架构文档 §8.2 / §12.3）：
- 配对必须三条件合取，路径不一致但 score=300 不得放行
- L0 必须显式要求 margin ≥ 120，未配对记录不得落 L0
- illustration_ready=False 时不得命中 no_image_candidate
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import review_levels as levels


def pair(path_match=True, score=200.0, margin=130.0):
    return {
        'path_match': path_match,
        'score': score,
        'margin': margin,
        'ok': True,
        'display_confidence': max(0.0, min(1.0, score / 300.0)),
    }


def record(**overrides):
    base = {
        'id': 'q1',
        'region_role': 'question',
        'pair': pair(),
        'illustrations': [],
    }
    base.update(overrides)
    return base


def page(**overrides):
    base = {'illustration_ready': False, 'questions': [], 'page_consistency': {}}
    base.update(overrides)
    return base


def level_of(record_obj, page_obj=None, **kwargs):
    return levels.compute_review_level(record_obj, page_obj or page(), **kwargs)


class PairRuleTests(unittest.TestCase):
    def test_path_mismatch_with_full_score_is_rejected(self):
        # D1 的核心反例：单阈值实现会把这条误放行
        broken = record(pair=pair(path_match=False, score=300.0, margin=300.0))
        self.assertFalse(levels.pair_ok(broken))
        level, reasons = level_of(broken)
        self.assertEqual(level, 'L2')
        self.assertIn('pair_fail', reasons)

    def test_margin_below_eighty_is_rejected(self):
        borderline = record(pair=pair(score=180.0, margin=79.0))
        self.assertFalse(levels.pair_ok(borderline))
        self.assertEqual(level_of(borderline)[0], 'L2')

    def test_score_below_180_is_rejected(self):
        borderline = record(pair=pair(score=179.9, margin=200.0))
        self.assertFalse(levels.pair_ok(borderline))

    def test_thresholds_are_inclusive(self):
        exact = record(pair=pair(score=180.0, margin=80.0))
        self.assertTrue(levels.pair_ok(exact))

    def test_empty_pair_is_never_l0(self):
        # pair == {} 的 margin 读出来是 0，若靠"没命中 L1 就自动 L0"会误判免检
        empty = record(pair={})
        self.assertFalse(levels.pair_ok(empty))
        self.assertEqual(levels.pair_margin(empty), 0.0)
        level, reasons = level_of(empty)
        self.assertEqual(level, 'L2')
        self.assertIn('pair_fail', reasons)

    def test_display_confidence_never_used_for_decision(self):
        # display_confidence=1.0 但 margin 只有 50 → 必须 L2
        tricky = record(pair={'path_match': True, 'score': 300.0, 'margin': 50.0,
                              'ok': True, 'display_confidence': 1.0})
        self.assertFalse(levels.pair_ok(tricky))
        level, reasons = level_of(tricky)
        self.assertEqual(level, 'L2')
        self.assertIn('pair_fail', reasons)

    def test_path_match_must_be_real_boolean(self):
        # 字符串 'false' / '0' 都是 truthy：用 bool() 会把"路径不一致"放行成 L0，
        # 这是最危险的方向（该看的没看）。只有布尔真值本身才算通过。
        for value in ('false', 'False', '0', 0, 1, 'true', None):
            with self.subTest(value=value):
                tricky = record(pair={'path_match': value, 'score': 300.0, 'margin': 300.0})
                self.assertFalse(levels.pair_ok(tricky))
                self.assertEqual(level_of(tricky)[0], 'L2')

    def test_pair_confidence_is_display_only_scale(self):
        self.assertAlmostEqual(levels.pair_confidence(180.0), 0.6, places=6)
        self.assertAlmostEqual(levels.pair_confidence(650.0), 1.0, places=6)


class LevelBoundaryTests(unittest.TestCase):
    def test_margin_120_reaches_l0(self):
        level, reasons = level_of(record(pair=pair(score=220.0, margin=120.0)))
        self.assertEqual(level, 'L0')
        self.assertEqual(reasons, ['default'])

    def test_margin_119_stays_l1(self):
        level, reasons = level_of(record(pair=pair(score=220.0, margin=119.0)))
        self.assertEqual(level, 'L1')
        self.assertIn('pair_margin_low', reasons)


class IllustrationSignalTests(unittest.TestCase):
    def test_not_ready_never_hits_no_image_candidate(self):
        ready_false = page(illustration_ready=False)
        level, reasons = level_of(record(), ready_false)
        self.assertEqual(level, 'L0')
        self.assertNotIn('no_image_candidate', reasons)

    def test_ready_without_candidate_is_l2(self):
        ready = page(illustration_ready=True)
        level, reasons = level_of(record(), ready)
        self.assertEqual(level, 'L2')
        self.assertIn('no_image_candidate', reasons)

    def test_ready_with_owned_candidate_passes(self):
        ready = page(illustration_ready=True)
        owned = record(illustration_candidates=[
            {'id': 'ic-0-1', 'owner_id': 'q1', 'owner_method': 'contain',
             'owner_confidence': 0.95, 'reject_reason': ''},
        ])
        level, reasons = level_of(owned, ready)
        self.assertEqual(level, 'L0')
        self.assertNotIn('no_image_candidate', reasons)

    def test_confirmed_illustration_satisfies_candidate_signal(self):
        ready = page(illustration_ready=True)
        confirmed = record(illustrations=[{'bbox': [10, 10, 60, 60], 'sha1': 'abc'}])
        level, _reasons = level_of(confirmed, ready)
        self.assertEqual(level, 'L0')

    def test_ambiguous_owner_is_l2(self):
        ambiguous = record(illustration_candidates=[
            {'id': 'ic-0-1', 'owner_id': '', 'owner_method': 'cross_column',
             'owner_confidence': 0.0},
        ])
        level, reasons = level_of(ambiguous, page(illustration_ready=True))
        self.assertEqual(level, 'L2')
        self.assertIn('image_owner_ambiguous', reasons)

    def test_nearest_owner_is_l1(self):
        nearest = record(illustration_candidates=[
            {'id': 'ic-0-1', 'owner_id': 'q1', 'owner_method': 'nearest',
             'owner_confidence': 0.5, 'confirmed': True},
        ])
        level, reasons = level_of(nearest, page(illustration_ready=True))
        self.assertEqual(level, 'L1')
        self.assertIn('image_owner_nearest', reasons)


class StructuralSignalTests(unittest.TestCase):
    def test_minor_number_inferred_is_l2(self):
        level, reasons = level_of(record(minor_number_inferred=True))
        self.assertEqual(level, 'L2')
        self.assertIn('minor_inferred', reasons)

    def test_group_role_is_l2(self):
        level, reasons = level_of(record(region_role='group'))
        self.assertEqual(level, 'L2')
        self.assertIn('region_group', reasons)

    def test_continuation_of_is_l2(self):
        level, reasons = level_of(record(continuation_of='q0'))
        self.assertEqual(level, 'L2')
        self.assertIn('region_continuation', reasons)

    def test_cross_column_with_columns_is_l2(self):
        crossing = record(bbox=[500, 100, 700, 200])
        columns = [[0.0, 600.0], [600.0, 1200.0]]
        level, reasons = level_of(crossing, columns=columns)
        self.assertEqual(level, 'L2')
        self.assertIn('cross_column', reasons)

    def test_missing_columns_skips_rule_and_warns(self):
        inside_one_column = record(bbox=[100, 100, 300, 200])
        detail = levels.evaluate(inside_one_column, page(), columns=None)
        self.assertEqual(detail['level'], 'L0')
        self.assertNotIn('cross_column', detail['reasons'])
        self.assertTrue(any('cross_column_skipped' in item for item in detail['warnings']))

    def test_enrich_failed_is_l2(self):
        for status in ('failed', 'rejected', 'stale'):
            level, reasons = level_of(record(enrich_status=status))
            self.assertEqual(level, 'L2')
            self.assertIn('enrich_failed', reasons)

    def test_analysis_fail_is_l2(self):
        level, reasons = level_of(record(analysis_check={'status': 'fail'}))
        self.assertEqual(level, 'L2')
        self.assertIn('analysis_fail', reasons)

    def test_analysis_unverifiable_is_l1(self):
        level, reasons = level_of(record(analysis_check={'status': 'not_checkable'}))
        self.assertEqual(level, 'L1')
        self.assertIn('analysis_unverifiable', reasons)

    def test_new_book_front_is_l2(self):
        level, reasons = level_of(record(), page_index=3, is_new_book=True)
        self.assertEqual(level, 'L2')
        self.assertIn('new_book_front', reasons)

    def test_new_book_back_pages_are_not_flagged(self):
        level, _reasons = level_of(record(), page_index=25, is_new_book=True)
        self.assertEqual(level, 'L0')

    def test_batch_degraded_is_l2(self):
        enrichment = {
            'batch_id': 'B0001-20261019-1',
            'stratum_id': 'stratum-math-7',
            'degraded': {'B0001-20261019-1': {'level': 'batch', 'reason': 'sampling_defect'}},
        }
        level, reasons = level_of(record(), enrichment=enrichment)
        self.assertEqual(level, 'L2')
        self.assertIn('batch_degraded', reasons)

    def test_page_consistency_failures_map_to_page_codes(self):
        bad_page = page(page_consistency={'ok': False, 'checks': [
            {'name': 'major_sequence', 'ok': False, 'detail': '跳号'},
            {'name': 'question_density', 'ok': False, 'detail': '2 题'},
        ]})
        level, reasons = level_of(record(), bad_page)
        self.assertEqual(level, 'L2')
        self.assertIn('major_sequence_gap', reasons)
        self.assertIn('page_density_outlier', reasons)


class L1SignalTests(unittest.TestCase):
    def test_many_subquestions_from_sibling_count(self):
        level, reasons = level_of(record(), sibling_count=6)
        self.assertEqual(level, 'L1')
        self.assertIn('many_subquestions', reasons)

    def test_many_subquestions_from_page_records(self):
        crowded = page(questions=[{'id': f'q{index}', 'region_role': 'question'}
                                  for index in range(7)])
        level, reasons = level_of(record(), crowded)
        self.assertEqual(level, 'L1')
        self.assertIn('many_subquestions', reasons)

    def test_multi_part_answer_is_l1(self):
        multi = record(question_type='解答', text='（1）求面积（2）求周长')
        level, reasons = level_of(multi)
        self.assertEqual(level, 'L1')
        self.assertIn('multi_part_answer', reasons)

    def test_single_part_answer_is_not_flagged(self):
        single = record(question_type='解答', text='求下面图形的面积。')
        level, reasons = level_of(single)
        self.assertEqual(level, 'L0')
        self.assertNotIn('multi_part_answer', reasons)


class BatchStatsTests(unittest.TestCase):
    def test_ratios_and_reason_counts(self):
        records = [
            {'review_level': 'L0', 'review_reasons': ['default']},
            {'review_level': 'L0', 'review_reasons': ['default']},
            {'review_level': 'L0', 'review_reasons': ['default']},
            {'review_level': 'L1', 'review_reasons': ['pair_margin_low']},
            {'review_level': 'L2', 'review_reasons': ['analysis_fail', 'new_book_front']},
        ]
        stats = levels.batch_stats(records)
        self.assertEqual(stats['total'], 5)
        self.assertEqual(stats['L0'], 3)
        self.assertEqual(stats['L2'], 1)
        self.assertAlmostEqual(stats['l2_ratio'], 0.2, places=4)
        self.assertFalse(stats['l2_within_target'])
        self.assertEqual(stats['new_book_front'], 1)
        self.assertEqual(stats['by_reason']['default'], 3)

    def test_missing_level_defaults_to_l2(self):
        stats = levels.batch_stats([{'review_reasons': []}])
        self.assertEqual(stats['L2'], 1)

    def test_empty_batch(self):
        stats = levels.batch_stats([])
        self.assertEqual(stats['total'], 0)
        self.assertEqual(stats['l2_ratio'], 0.0)


class DifficultyDistributionTests(unittest.TestCase):
    def test_single_bucket_above_60_percent_collapses(self):
        records = [{'difficulty': 3} for _ in range(7)] + [{'difficulty': 2} for _ in range(3)]
        result = levels.check_difficulty_distribution(records)
        self.assertEqual(result['scheme'], '1-5')
        self.assertEqual(result['buckets']['3'], 7)
        self.assertAlmostEqual(result['dominant_ratio'], 0.7, places=4)
        self.assertTrue(result['collapsed'])
        self.assertEqual(result['action'], 'rerun')

    def test_balanced_distribution_is_healthy(self):
        records = [{'difficulty': 1}, {'difficulty': 2}, {'difficulty': 3},
                   {'difficulty': 4}, {'difficulty': 5}]
        result = levels.check_difficulty_distribution(records)
        self.assertFalse(result['collapsed'])
        self.assertEqual(result['action'], 'ok')

    def test_out_of_range_values_are_counted_separately(self):
        records = [{'difficulty': 1}, {'difficulty': 2}, {'difficulty': 2},
                   {'difficulty': 3}, {'difficulty': 3},
                   {'difficulty': '中'}, {'difficulty': None}, {'difficulty': 7}]
        result = levels.check_difficulty_distribution(records)
        self.assertEqual(result['total'], 5)
        self.assertEqual(result['invalid_count'], 3)
        self.assertEqual(result['buckets']['3'], 2)
        self.assertFalse(result['collapsed'])

    def test_string_digits_are_accepted(self):
        result = levels.check_difficulty_distribution([{'difficulty': '4'}])
        self.assertEqual(result['buckets']['4'], 1)


if __name__ == '__main__':
    unittest.main()
