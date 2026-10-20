# -*- coding: utf-8 -*-
"""layout_arbiter 扩展的不变量测试（不需要版面模型）。

最关键的一条：``detect(img)`` 不传新参时返回的**键集合必须与改造前完全相同**，
这样 ``cut_questions_v4._apply_layout_arbiter()`` 与主切题流程才真的零影响。
用假的推理管道（FakePipe）跑通整条分支，因此本文件不依赖 125MB 模型。
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import layout_arbiter as arbiter

# 改造前 detect() 的键集合（布局裁判 v2 的契约）
LEGACY_KEYS = {'revision', 'width', 'height', 'regions', 'body_regions', 'columns', 'gutters'}
NEW_KEYS = {'regions_detail', 'image_boxes'}


class FakeResult:
    def __init__(self, payload):
        self.json = payload


class FakePipe:
    """模拟 PP-DocLayout 的输出结构，只保留 detect() 真正读的字段。"""

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, img, batch_size=1):  # noqa: ARG002 - 结构对齐
        return [FakeResult({'res': {'boxes': self.boxes}})]


def box(label, score, coordinate):
    return {'label': label, 'score': score, 'coordinate': coordinate}


def sample_boxes():
    """2 栏正文 + 1 张大图 + 1 条窄公式 + 1 个序号级小图。"""
    boxes = []
    for index in range(4):
        top = 100 + index * 200
        boxes.append(box('text', 0.95, [50.0, float(top), 550.0, float(top + 180)]))
        boxes.append(box('text', 0.95, [650.0, float(top), 1150.0, float(top + 180)]))
    boxes.append(box('image', 0.93, [200.0, 1000.0, 500.0, 1200.0]))
    boxes.append(box('formula', 0.88, [60.0, 1250.0, 1150.0, 1300.0]))
    boxes.append(box('image', 0.90, [60.0, 1350.0, 120.0, 1400.0]))
    return boxes


def blank_page(height=1600, width=1200):
    return np.full((height, width, 3), 255, dtype=np.uint8)


class LayoutArbiterImageTests(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        arbiter._MEMORY_CACHE.clear()
        patcher = mock.patch.object(arbiter, '_cache_dir', lambda: self.cache_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        pipe_patcher = mock.patch.object(arbiter, '_get_pipe', lambda: FakePipe(sample_boxes()))
        pipe_patcher.start()
        self.addCleanup(pipe_patcher.stop)
        self.addCleanup(shutil.rmtree, self.cache_dir, True)
        self.addCleanup(arbiter._MEMORY_CACHE.clear)
        self.image = blank_page()

    def test_default_call_returns_legacy_key_set(self):
        payload = arbiter.detect(self.image)
        self.assertEqual(set(payload), LEGACY_KEYS)
        self.assertFalse(NEW_KEYS & set(payload))

    def test_default_call_key_set_is_stable_on_cache_hit(self):
        first = arbiter.detect(self.image)
        second = arbiter.detect(self.image)
        self.assertEqual(set(second), LEGACY_KEYS)
        self.assertEqual(first, second)

    def test_default_argument_is_false(self):
        self.assertEqual(arbiter.detect.__defaults__, (False,))

    def test_with_regions_adds_exactly_two_keys(self):
        payload = arbiter.detect(self.image, with_regions=True)
        self.assertEqual(set(payload), LEGACY_KEYS | NEW_KEYS)
        self.assertEqual(payload['revision'], arbiter.REVISION)
        self.assertEqual(len(payload['regions_detail']), len(sample_boxes()))

    def test_regions_detail_shape(self):
        payload = arbiter.detect(self.image, with_regions=True)
        detail = payload['regions_detail'][0]
        self.assertEqual(set(detail), {'label', 'score', 'box'})
        self.assertEqual(len(detail['box']), 4)

    def test_image_boxes_only_contains_image_labels(self):
        payload = arbiter.detect(self.image, with_regions=True)
        self.assertEqual(len(payload['image_boxes']), 3)   # 2 张 image + 1 条 formula
        text_boxes = [item['box'] for item in payload['regions_detail']
                      if item['label'] == 'text']
        for image_box in payload['image_boxes']:
            self.assertNotIn(image_box, text_boxes)

    def test_existing_keys_are_untouched_by_with_regions(self):
        legacy = arbiter.detect(self.image)
        detailed = arbiter.detect(self.image, with_regions=True)
        for key in LEGACY_KEYS:
            self.assertEqual(legacy[key], detailed[key])

    def test_revision_bump_invalidates_old_cache(self):
        # 旧 revision 的缓存文件必须被拒绝（这是唯一的缓存迁移动作）
        key = arbiter._image_key(self.image)
        stale = {'revision': 'layout-arbiter-v2-20261019', 'columns': [[0.0, 1.0]]}
        with open(os.path.join(self.cache_dir, f'{key}.json'), 'w', encoding='utf-8') as file:
            json.dump(stale, file)
        arbiter._MEMORY_CACHE.clear()
        payload = arbiter.detect(self.image)
        self.assertEqual(payload['revision'], arbiter.REVISION)
        self.assertNotEqual(payload['columns'], [[0.0, 1.0]])

    def test_image_regions_returns_none_when_unavailable(self):
        with mock.patch.object(arbiter, 'available', lambda: False):
            self.assertIsNone(arbiter.image_regions(self.image))

    def test_image_regions_returns_boxes_when_available(self):
        with mock.patch.object(arbiter, 'available', lambda: True):
            found = arbiter.image_regions(self.image)
        # 1000~1200 的大图通过；窄公式（高 50）与小图（60×50）被最短边 64 过滤
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['label'], 'image')
        self.assertEqual(found[0]['box'], [200.0, 1000.0, 500.0, 1200.0])

    def test_image_regions_returns_none_without_pipe(self):
        with mock.patch.object(arbiter, 'available', lambda: True), \
                mock.patch.object(arbiter, '_get_pipe', lambda: None):
            self.assertIsNone(arbiter.image_regions(self.image))

    def test_rules_are_externalised(self):
        rules = arbiter.image_rules()
        self.assertIn('score_min', rules)
        self.assertIn('labels', rules)
        self.assertEqual(rules['score_min'], 0.5)


class RealModelTests(unittest.TestCase):
    @unittest.skipUnless(arbiter.available(), '版面模型不可用，跳过真实推理验证')
    def test_real_page_returns_sorted_regions(self):  # pragma: no cover - 依赖模型
        image = np.full((1600, 1200, 3), 255, dtype=np.uint8)
        found = arbiter.image_regions(image)
        # 空白页可能没有任何够格区域；这里只验证"返回值语义与排序"
        self.assertTrue(found is None or isinstance(found, list))
        if found:
            tops = [(item['box'][1], item['box'][0]) for item in found]
            self.assertEqual(tops, sorted(tops))


if __name__ == '__main__':
    unittest.main()
