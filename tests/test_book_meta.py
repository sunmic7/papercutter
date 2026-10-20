# -*- coding: utf-8 -*-
"""书名解析器（G0b）与书级信息版本校验（D6）单测。

核心口径：**猜不出来就留空 + warnings**。猜错的代价是整本书的元数据一致地错，
比"留空等人填"贵得多，所以 "专项训练" 的 grade 必须是空字符串。
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import book_meta


class GuessTests(unittest.TestCase):
    def test_full_title(self):
        meta = book_meta.guess_book_metadata('七年级上册数学')
        self.assertEqual(meta['grade'], '七年级')
        self.assertEqual(meta['stage'], '初中')
        self.assertEqual(meta['subject'], '数学')
        self.assertEqual(meta['semester'], '上册')
        self.assertEqual(meta['grade_index'], 7)

    def test_primary_school_title_with_edition(self):
        meta = book_meta.guess_book_metadata('人教版三年级下册数学')
        self.assertEqual(meta['grade'], '三年级')
        self.assertEqual(meta['stage'], '小学')
        self.assertEqual(meta['edition'], '人教版')
        self.assertEqual(meta['semester'], '下册')

    def test_senior_high_title(self):
        meta = book_meta.guess_book_metadata('高一数学必修一')
        self.assertEqual(meta['stage'], '高中')
        self.assertEqual(meta['subject'], '数学')

    def test_arabic_grade_is_accepted(self):
        meta = book_meta.guess_book_metadata('5年级数学上册')
        self.assertEqual(meta['grade'], '五年级')
        self.assertEqual(meta['grade_index'], 5)

    def test_unknown_grade_is_left_empty_with_warnings(self):
        # 不许猜：专项训练没有年级信息，grade 必须为空且必须留告警
        meta = book_meta.guess_book_metadata('专项训练')
        self.assertEqual(meta['grade'], '')
        self.assertTrue(meta['warnings'])
        self.assertEqual(meta['book_type'], '专项训练')
        self.assertTrue(any('年级' in item for item in meta['warnings']))

    def test_unknown_subject_is_left_empty_with_warnings(self):
        meta = book_meta.guess_book_metadata('第一单元测试卷')
        self.assertEqual(meta['subject'], '')
        self.assertTrue(any('学科' in item for item in meta['warnings']))

    def test_confidence_is_capped_when_grade_missing(self):
        meta = book_meta.guess_book_metadata('人教版数学专项训练')
        self.assertLessEqual(meta['confidence'], 0.5)

    def test_directory_path_is_cleaned(self):
        meta = book_meta.guess_book_metadata('D:\\\\corpus\\\\六年级上册数学')
        self.assertEqual(meta['grade'], '六年级')
        self.assertEqual(meta['stage'], '小学')

    def test_difficulty_scheme_defaults_to_five_levels(self):
        self.assertEqual(book_meta.guess_book_metadata('七年级上册数学')['difficulty_scheme'], '1-5')


class RevisionTests(unittest.TestCase):
    def _meta(self, **overrides):
        meta = {
            'language': 'zh',
            'stage': '初中',
            'grade': '七年级',
            'subject': '数学',
            'difficulty_scheme': '1-5',
        }
        meta.update(overrides)
        return meta

    def test_revision_is_stable_for_same_input(self):
        self.assertEqual(book_meta.book_meta_rev(self._meta()),
                         book_meta.book_meta_rev(self._meta()))

    def test_revision_changes_with_any_key_field(self):
        base = book_meta.book_meta_rev(self._meta())
        for key, value in (('subject', '语文'), ('stage', '小学'), ('grade', '八年级'),
                           ('language', 'en'), ('difficulty_scheme', '1-3')):
            with self.subTest(key=key):
                self.assertNotEqual(base, book_meta.book_meta_rev(self._meta(**{key: value})))

    def test_missing_language_is_a_problem(self):
        result = book_meta.validate_book_metadata_revision(self._meta(language=''))
        self.assertFalse(result['ok'])
        self.assertTrue(any('language' in item for item in result['problems']))

    def test_dirty_when_revision_differs(self):
        result = book_meta.validate_book_metadata_revision(self._meta(), expected_rev='deadbeef')
        self.assertTrue(result['dirty'])
        self.assertTrue(result['ok'])          # 变了不是错，但必须能看出脏了
        self.assertTrue(result['warnings'])

    def test_clean_when_revision_matches(self):
        current = book_meta.book_meta_rev(self._meta())
        result = book_meta.validate_book_metadata_revision(self._meta(), expected_rev=current)
        self.assertFalse(result['dirty'])

    def test_locked_change_requires_unlock_reason(self):
        rejected = book_meta.validate_book_metadata_revision(
            self._meta(), expected_rev='deadbeef', locked=True,
        )
        self.assertFalse(rejected['ok'])
        allowed = book_meta.validate_book_metadata_revision(
            self._meta(), expected_rev='deadbeef', locked=True, unlock_reason='甲方换版',
        )
        self.assertTrue(allowed['ok'])

    def test_grade_stage_mismatch_is_a_problem(self):
        result = book_meta.validate_book_metadata_revision(self._meta(stage='小学', grade='七年级'))
        self.assertFalse(result['ok'])

    def test_invalid_difficulty_scheme_is_a_problem(self):
        result = book_meta.validate_book_metadata_revision(self._meta(difficulty_scheme='3档'))
        self.assertFalse(result['ok'])


if __name__ == '__main__':
    unittest.main()
