# -*- coding: utf-8 -*-
"""题图候选：归属四条路径 + 硬拒四条规则。

归属决定"这张图属于哪道题"，交付后无法挽回，所以每条路径都必须有确定的行为：
contain / center / nearest / cross_column（否决）。
"""
import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import illustration_candidates as candidates

COLUMNS = [[0.0, 600.0], [600.0, 1200.0]]
PAGE_SHAPE = (1600, 1200)


def question(record_id, bbox):
    return {'id': record_id, 'region_role': 'question', 'bbox': list(bbox)}


def noise_page(height=1600, width=1200, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def textured_page(height=1400, width=1000, seed=3):
    """有纹理的合成页面：随机文字行 + 矩形块（低频主体，接近真实扫描页）。

    为什么不用纯色页或纯噪声页：纯色页感知哈希无鉴别力（全是同一个 bit），
    纯噪声页哈希又不稳定（内缩 5px 汉明距离就冲到 141）。真实扫描页介于两者之间，
    所以这里用"低频矩形块 + 随机版式"来逼近它 —— QA 实测真实页内缩
    0/3/5/20/40px 的汉明距离是 0/5/8/38/80，本页同一量级。
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), 246, dtype=np.uint8)
    top = 60
    while top < height - 80:
        line_height = int(rng.integers(18, 40))
        left = int(rng.integers(40, 120))
        right = int(rng.integers(width - 200, width - 40))
        page[top:top + line_height, left:right] = int(rng.integers(20, 120))
        if rng.random() < 0.35:
            gap = int(rng.integers(left + 40, right - 40))
            page[top:top + line_height, gap:gap + int(rng.integers(20, 60))] = 246
        top += line_height + int(rng.integers(10, 30))
    page[520:900, 120:820] = int(rng.integers(120, 180))
    return page


class OwnerTests(unittest.TestCase):
    def test_contain_path(self):
        records = [question('q1', [50, 250, 550, 620])]
        owner = candidates.assign_owner([100, 300, 300, 450], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'contain')
        self.assertEqual(owner['owner_id'], 'q1')
        self.assertAlmostEqual(owner['owner_confidence'], 0.95, places=2)

    def test_center_path(self):
        # 只包住图框中心，面积覆盖不足 85%
        records = [question('q1', [80, 350, 320, 500])]
        owner = candidates.assign_owner([100, 300, 300, 450], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'center')
        self.assertEqual(owner['owner_id'], 'q1')
        self.assertAlmostEqual(owner['owner_confidence'], 0.8, places=2)

    def test_nearest_path(self):
        # 图在题框正下方，间隙 20px ≤ 1.2×题框高
        records = [question('q1', [60, 100, 560, 280])]
        owner = candidates.assign_owner([100, 300, 300, 450], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'nearest')
        self.assertEqual(owner['owner_id'], 'q1')
        self.assertAlmostEqual(owner['owner_confidence'], 0.5, places=2)

    def test_nearest_picks_the_lowest_record_above(self):
        records = [
            question('q1', [60, 60, 560, 200]),
            question('q2', [60, 240, 560, 290]),
        ]
        owner = candidates.assign_owner([100, 300, 300, 450], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'nearest')
        self.assertEqual(owner['owner_id'], 'q2')

    def test_nearest_gap_too_large_falls_back(self):
        # 题框高 80，间隙 420 > 1.2×80 → 不算邻接
        records = [question('q1', [60, 100, 560, 180])]
        owner = candidates.assign_owner([100, 600, 300, 750], records, COLUMNS)
        self.assertNotEqual(owner['owner_method'], 'nearest')

    def test_cross_column_path(self):
        # 图框宽 200px，跨过 x=600 的栏缝，两侧各占 100px（50% ≥ 20%）
        records = [question('q1', [60, 1000, 560, 1200])]
        owner = candidates.assign_owner([500, 300, 700, 450], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'cross_column')
        self.assertEqual(owner['owner_id'], '')
        self.assertEqual(owner['owner_confidence'], 0.0)

    def test_orphan_path(self):
        records = [question('q1', [60, 1400, 560, 1580])]
        owner = candidates.assign_owner([1000, 100, 1100, 200], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'orphan')
        self.assertEqual(owner['owner_id'], '')

    def test_no_question_records_is_orphan(self):
        owner = candidates.assign_owner([100, 300, 300, 450], [], COLUMNS)
        self.assertEqual(owner['owner_method'], 'orphan')

    def test_non_question_roles_are_ignored(self):
        records = [{'id': 'g1', 'region_role': 'group', 'bbox': [50, 250, 550, 620]}]
        owner = candidates.assign_owner([100, 300, 300, 450], records, COLUMNS)
        self.assertEqual(owner['owner_method'], 'orphan')


class HardRejectTests(unittest.TestCase):
    """硬拒规则：page_like 三条判据 + 合同级 table / formula 规则（F1 / F4）。"""

    def setUp(self):
        self.page = textured_page()
        self.shape = self.page.shape[:2]        # (1400, 1000)

    def _inset(self, inset):
        height, width = self.page.shape[:2]
        box = [inset, inset, width - inset, height - inset]
        return box, self.page[inset:height - inset, inset:width - inset]

    def test_page_like_full_page_is_rejected(self):
        box, crop = self._inset(0)
        verdict = candidates.hard_reject(crop, box, self.shape, 'image', page=self.page)
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'page_like')
        self.assertEqual(verdict['page_area_ratio'], 1.0)

    def test_page_like_for_small_insets(self):
        # 版面模型给的整页框常态内缩几个像素：0/5/20/40px 都必须是 page_like
        for inset in (0, 5, 20, 40):
            with self.subTest(inset=inset):
                box, crop = self._inset(inset)
                verdict = candidates.hard_reject(crop, box, self.shape, 'image',
                                                 page=self.page)
                self.assertTrue(verdict['rejected'])
                self.assertEqual(verdict['reason'], 'page_like')

    def test_page_like_margin_rule_is_bounded_by_the_short_side(self):
        """③ 的边界由**短边**决定，不按像素数。

        同一条 40px 内缩：1000×1400（高 margin 2.86%）→ 拦；1000×686（高 margin 5.83%）→ 放行。
        写这条是为了锁住"阈值按方向归一"这个语义 —— 一旦有人把它改成按长边/绝对像素，
        这个用例会立刻失败。当前有效护栏约"面积 ≥ 0.83~0.87（随纵横比浮动）"。
        """
        tall = textured_page(1400, 1000)
        wide = textured_page(686, 1000)          # 对开页缩到 1000 宽时的纵横比
        for page, expect_rejected in ((tall, True), (wide, False)):
            height, width = page.shape[:2]
            box = [40, 40, width - 40, height - 40]
            crop = page[40:height - 40, 40:width - 40]
            verdict = candidates.hard_reject(crop, box, (height, width), 'image', page=page)
            with self.subTest(ratio=round(width / height, 3)):
                self.assertEqual(verdict['rejected'], expect_rejected,
                                 f'{width}x{height} 上 40px 内缩的判定与短边语义不符')

    def test_page_like_margin_rule_works_without_page_pixels(self):
        # ③ 是纯几何判据：没有整页像素也要拦住"框就是整页"
        box, _crop = self._inset(20)
        verdict = candidates.hard_reject(None, box, self.shape, 'image')
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'page_like')

    def test_page_like_area_rule_independent_of_hash(self):
        # ② 面积+相似度：即使哈希被判据①放过，面积判据也要兜住
        box = [0, 0, 1000, 1370]
        crop = self.page[0:1370, 0:1000]
        verdict = candidates.hard_reject(crop, box, self.shape, 'image', page=self.page)
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'page_like')
        self.assertGreaterEqual(verdict['page_area_ratio'], 0.9)

    def test_page_thumbnail_pasted_as_small_region_is_rejected(self):
        # 小区域里装一张整页缩略图：面积/页边判据都不触发，只能靠感知哈希
        thumbnail = cv2.resize(self.page, (300, 420), interpolation=cv2.INTER_AREA)
        verdict = candidates.hard_reject(thumbnail, [700, 900, 1000, 1320], self.shape,
                                         'image', page=self.page)
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'page_like')
        self.assertLessEqual(verdict['page_hamming'], 64)

    def test_local_figure_is_not_page_like(self):
        # 占页 8% 的局部插图：三条 page_like 判据都不该命中
        box = [120, 520, 600, 760]
        crop = self.page[520:760, 120:600]
        verdict = candidates.hard_reject(crop, box, self.shape, 'image', page=self.page)
        self.assertNotEqual(verdict['reason'], 'page_like')
        self.assertLess(verdict['page_area_ratio'], 0.2)
        self.assertEqual(verdict['reason'], '')

    def test_table_region_is_downgraded_to_needs_review_by_default(self):
        # B 方案（2026-10-20 裁决）：需求允许图表/统计图作题图，所以默认不硬拒，
        # 改为 needs_review=True 交人判断（统计数据表可能正是该题的题图）
        verdict = candidates.hard_reject(None, [100, 200, 900, 500], self.shape, 'table')
        self.assertFalse(verdict['rejected'])
        self.assertEqual(verdict['reason'], '')
        self.assertTrue(verdict['needs_review'])
        self.assertFalse(candidates.RULES['hard_reject']['table_region']['enabled'])

    def test_table_region_switch_is_live(self):
        # 临时把开关置 true → 必须恢复硬拒（证明它是活配置，不是死键）
        with mock.patch.dict(candidates.RULES['hard_reject']['table_region'],
                             {'enabled': True}):
            verdict = candidates.hard_reject(None, [100, 200, 900, 500], self.shape, 'table')
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'table_region')

    def test_thin_formula_line_is_rejected(self):
        # QA 的 p3-ic-0-4：`78×199≈16000( )` 细条算式行，面积占比 1%
        verdict = candidates.hard_reject(None, [80, 640, 920, 700], self.shape, 'formula')
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'formula_only')

    def test_long_formula_mixed_region_keeps_needs_review(self):
        # 公式与题干混排的大块：不硬拒，但必须标 needs_review 让人看
        verdict = candidates.hard_reject(None, [100, 300, 700, 800], self.shape, 'formula')
        self.assertFalse(verdict['rejected'])
        self.assertTrue(verdict['needs_review'])

    def test_normal_image_does_not_need_review(self):
        verdict = candidates.hard_reject(None, [100, 300, 500, 700], self.shape, 'image')
        self.assertFalse(verdict['rejected'])
        self.assertFalse(verdict['needs_review'])

    def test_formula_strip_is_rejected(self):
        # 长条公式：高/宽 > 3
        verdict = candidates.hard_reject(None, [100, 100, 400, 1300], (1600, 1200), 'formula')
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'formula_only')

    def test_number_label_is_rejected(self):
        verdict = candidates.hard_reject(None, [100, 100, 110, 120], (1600, 1200), 'image')
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'number_label')

    def test_too_small_is_rejected(self):
        verdict = candidates.hard_reject(None, [100, 100, 200, 200], (1600, 1200), 'image')
        self.assertTrue(verdict['rejected'])
        self.assertEqual(verdict['reason'], 'too_small')

    def test_rules_come_from_json(self):
        self.assertIn('hard_reject', candidates.RULES)
        page_like = candidates.RULES['hard_reject']['page_like']
        self.assertEqual(page_like['hamming_max'], 64)
        self.assertLessEqual(page_like['page_area_ratio_min'], 0.95)
        self.assertLessEqual(page_like['page_similarity_min'], 0.95)
        self.assertLessEqual(page_like['page_margin_ratio_max'], 0.1)
        self.assertIn('table_region', candidates.RULES['hard_reject'])
        formula = candidates.RULES['hard_reject']['formula_only']
        self.assertLessEqual(formula['thin_area_ratio_max'], 0.05)
        # quality_gate 只存在于 runtime/enrich/candidate_rules.json，
        # 出现它就证明阈值是从配置文件读的，而不是内置默认值
        self.assertIn('quality_gate', candidates.RULES)
        self.assertEqual(candidates.RULES['hard_reject']['too_small']['side_min'], 128)
        loaded = candidates.load_rules()
        self.assertEqual(loaded['owner']['confidences']['contain'], 0.95)


class CollectTests(unittest.TestCase):
    def _layout(self):
        return {
            'columns': COLUMNS,
            'width': 1200,
            'height': 1600,
            'regions_detail': [
                {'label': 'image', 'score': 0.93, 'box': [100, 300, 500, 700]},
                {'label': 'text', 'score': 0.9, 'box': [50, 100, 550, 280]},
                {'label': 'image', 'score': 0.4, 'box': [700, 300, 900, 500]},
            ],
        }

    def test_collect_builds_candidates(self):
        page = noise_page()
        records = [question('q1', [50, 250, 550, 760])]
        found = candidates.collect(page, 3, records, self._layout())
        self.assertEqual(len(found), 1)          # 低分 text/低分 image 都被过滤
        candidate = found[0]
        self.assertEqual(candidate['id'], 'ic-3-1')
        self.assertEqual(candidate['label'], 'image')
        self.assertEqual(candidate['owner_id'], 'q1')
        self.assertEqual(candidate['owner_method'], 'contain')
        self.assertEqual(candidate['reject_reason'], '')
        self.assertFalse(candidate['confirmed'])
        self.assertIn('needs_review', candidate)
        self.assertTrue(candidate['crop_sha1'])
        self.assertIn('similarity_to_page', candidate)

    def test_collect_returns_empty_without_layout(self):
        self.assertEqual(candidates.collect(None, 0, [], None), [])

    def test_collect_keeps_rejected_candidates_visible(self):
        layout = self._layout()
        layout['regions_detail'] = [{'label': 'image', 'score': 0.9,
                                     'box': [100, 300, 500, 700]}]
        page = noise_page()
        page[300:700, 100:500] = page.copy()[0:400, 0:400]  # 与左上角同源，制造 page_like 嫌疑
        found = candidates.collect(page, 0, [question('q1', [50, 250, 550, 760])], layout)
        self.assertEqual(len(found), 1)
        # 被拒也必须下发：人要能看到"系统为什么认为这是坏图"
        self.assertIsInstance(found[0]['reject_reason'], str)


class CollectNeedsReviewTests(unittest.TestCase):
    def test_collect_propagates_needs_review_for_table(self):
        page = textured_page()
        layout = {
            'columns': [[0.0, 500.0], [500.0, 1000.0]],
            'width': 1000, 'height': 1400,
            'regions_detail': [
                {'label': 'table', 'score': 0.9, 'box': [120.0, 200.0, 880.0, 420.0]},
            ],
        }
        records = [question('q1', [60, 60, 940, 1390])]
        found = candidates.collect(page, 4, records, layout)
        self.assertEqual(len(found), 1)
        self.assertIn('needs_review', found[0])
        self.assertTrue(found[0]['needs_review'])
        self.assertEqual(found[0]['reject_reason'], '')

    def test_collect_propagates_need_review_flag_on_every_candidate(self):
        page = textured_page()
        layout = {
            'columns': [[0.0, 500.0], [500.0, 1000.0]],
            'width': 1000, 'height': 1400,
            'regions_detail': [
                {'label': 'image', 'score': 0.9, 'box': [120.0, 200.0, 600.0, 420.0]},
                {'label': 'formula', 'score': 0.9, 'box': [80.0, 900.0, 920.0, 1300.0]},
            ],
        }
        records = [question('q1', [60, 60, 940, 1390])]
        found = candidates.collect(page, 5, records, layout)
        self.assertEqual(len(found), 2)
        for item in found:
            self.assertIn('needs_review', item)
            self.assertIsInstance(item['needs_review'], bool)


class PerceptualHashTests(unittest.TestCase):
    def test_hash_is_stable_and_detects_identity(self):
        page = noise_page(seed=11)
        self.assertEqual(
            int(np.count_nonzero(candidates.perceptual_hash(page)
                                 != candidates.perceptual_hash(page))), 0,
        )

    def test_different_images_have_different_hash(self):
        left = candidates.perceptual_hash(noise_page(seed=1))
        right = candidates.perceptual_hash(noise_page(seed=2))
        self.assertGreater(candidates.hamming_distance(left, right), 6)


if __name__ == '__main__':
    unittest.main()
