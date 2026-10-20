# -*- coding: utf-8 -*-
"""知识点词表单测：三步归一、越界拒收、加载校验、版本选择。

词表必须是**封闭**的：越界词进 rejected，不得静默接受（架构文档 §6.3）。
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import knowledge_taxonomy as taxonomy


def write_vocabulary(directory, slug, version, payload):
    path = os.path.join(directory, f'knowledge_points.{slug}.{version}.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False)
    return path


def minimal_payload(**overrides):
    payload = {
        'version': 'math-2026.10',
        'subject': '数学',
        'stages': ['小学'],
        'domains': ['数与代数'],
        'nodes': [
            {'code': 'M-G4', 'name': '四年级', 'parent': '', 'grades': [4],
             'stage': '小学', 'domain': ''},
            {'code': 'M-G4-07', 'name': '三角形面积', 'parent': 'M-G4',
             'grades': [4, 5], 'stage': '小学', 'domain': '数与代数',
             'aliases': ['三角形的面积']},
        ],
    }
    payload.update(overrides)
    return payload


class ShippedVocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = taxonomy.load('数学')

    def test_loads_latest_version(self):
        self.assertEqual(self.taxonomy.version, 'math-2026.10')
        self.assertEqual(self.taxonomy.subject, '数学')

    def test_vocabulary_is_large_enough_to_be_useful(self):
        stats = self.taxonomy.stats()
        self.assertGreaterEqual(stats['leaves'], 60)
        self.assertLessEqual(stats['leaves'], 120)

    def test_covers_four_domains(self):
        domains = self.taxonomy.stats()['by_domain']
        for name in ('数与代数', '图形与几何', '统计与概率', '综合与实践'):
            self.assertIn(name, domains)
            self.assertGreater(domains[name], 0)

    def test_exact_name_resolves(self):
        self.assertEqual(self.taxonomy.resolve('三角形面积'), 'M-G4-07')

    def test_alias_resolves(self):
        self.assertEqual(self.taxonomy.resolve('三角形的面积'), 'M-G4-07')

    def test_normalized_bracket_suffix_resolves(self):
        # "三角形面积（1）" → 去括号补充说明后命中
        self.assertEqual(self.taxonomy.resolve('三角形面积（1）'), 'M-G4-07')

    def test_out_of_vocabulary_returns_none(self):
        self.assertIsNone(self.taxonomy.resolve('量子纠缠'))
        self.assertIsNone(self.taxonomy.resolve(''))

    def test_validate_splits_accepted_and_rejected(self):
        accepted, rejected = self.taxonomy.validate(
            ['三角形的面积', '量子纠缠', '三角形面积', '声波干涉']
        )
        self.assertEqual(accepted, [('三角形面积', 'M-G4-07')])
        self.assertEqual(rejected, ['量子纠缠', '声波干涉'])

    def test_validate_uses_canonical_name_not_the_input_writing(self):
        accepted, _rejected = self.taxonomy.validate(['三角形的面积'])
        self.assertEqual(accepted[0][0], '三角形面积')

    def test_prompt_vocabulary_filters_by_grade(self):
        names = self.taxonomy.prompt_vocabulary('小学', '四年级')
        self.assertIn('三角形面积', names)
        self.assertNotIn('20以内数的认识', names)

    def test_prompt_vocabulary_respects_limit(self):
        names = self.taxonomy.prompt_vocabulary('小学', '', limit=5)
        self.assertEqual(len(names), 5)

    def test_prompt_vocabulary_excludes_group_nodes(self):
        names = self.taxonomy.prompt_vocabulary('小学', '四年级')
        self.assertNotIn('四年级', names)

    def test_cache_returns_same_instance(self):
        self.assertIs(taxonomy.load('数学'), self.taxonomy)

    def test_accepts_slug_as_subject(self):
        self.assertEqual(taxonomy.load('math').version, 'math-2026.10')


class AmbiguousAliasTests(unittest.TestCase):
    """跨条目重复别名必须拒绝加载（QA 实测缺陷 F2）。

    同一写法挂在两个年级节点上时，旧的 setdefault 会静默选中文档序在前的那个，
    于是四年级题面被判成三年级知识点、validate() 还判 accepted —— 既不拒收
    也不升 L2，整本书一致地错。这与"宁可拒收也不静默接受"直接冲突。
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)
        taxonomy.clear_cache()

    def _payload(self, left_alias, right_alias, left_name='两位数乘两位数',
                 right_name='三位数乘两位数'):
        return {
            'version': 'math-2026.10',
            'subject': '数学',
            'stages': ['小学'],
            'domains': [],
            'nodes': [
                {'code': 'M-G3-13', 'name': left_name, 'parent': '', 'grades': [3],
                 'stage': '小学', 'aliases': [left_alias]},
                {'code': 'M-G4-04', 'name': right_name, 'parent': '', 'grades': [4],
                 'stage': '小学', 'aliases': [right_alias]},
            ],
        }

    def test_shared_alias_across_nodes_raises(self):
        write_vocabulary(self.directory, 'math', '2026.10',
                         self._payload('笔算乘法', '笔算乘法'))
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_alias_colliding_with_another_node_name_raises(self):
        write_vocabulary(self.directory, 'math', '2026.10',
                         self._payload('图形的运动', '平移', right_name='图形的运动'))
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_name_collision_after_normalization_raises(self):
        # "图形的运动（旋转）" 规范化后等于 "图形的运动"，同名不同码也必须拒绝
        write_vocabulary(self.directory, 'math', '2026.10',
                         self._payload('', '', left_name='图形的运动',
                                       right_name='图形的运动（旋转）'))
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_unambiguous_aliases_still_load(self):
        write_vocabulary(self.directory, 'math', '2026.10',
                         self._payload('两位数乘法', '三位数乘法'))
        loaded = taxonomy.load('数学', root=self.directory)
        self.assertEqual(loaded.resolve('两位数乘法'), 'M-G3-13')


# 课标里确实跨年级的 5 组写法（人教版目录），已改成"带年级限定才收录"
AMBIGUOUS_CASES = (
    # 写法, 低年级节点, 低年级, 高年级节点, 高年级
    ('笔算乘法', 'M-G3-13', '三年级', 'M-G4-04', '四年级'),
    ('笔算除法', 'M-G3-12', '三年级', 'M-G4-06', '四年级'),
    ('统计表', 'M-G2-14', '二年级', 'M-G3-18', '三年级'),
    ('运算顺序', 'M-G2-11', '二年级', 'M-G4-10', '四年级'),
    ('旋转', 'M-G2-13', '二年级', 'M-G5-11', '五年级'),
)


class ShippedVocabularyAmbiguityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = taxonomy.load('数学')

    def test_no_alias_conflicts(self):
        self.assertEqual(self.taxonomy.stats()['alias_conflicts'], 0)

    def test_registry_is_declared_and_consistent(self):
        self.assertEqual(len(self.taxonomy.ambiguous_terms), len(AMBIGUOUS_CASES))
        codes = set(self.taxonomy.codes())
        for term, *declared in ((item[0], item[1], item[3]) for item in AMBIGUOUS_CASES):
            with self.subTest(term=term):
                owners = self.taxonomy.ambiguous_terms[term]
                self.assertEqual(sorted(owners), sorted(declared))
                self.assertTrue(set(owners) <= codes)

    def test_grade_qualified_aliases_are_used(self):
        self.assertEqual(self.taxonomy.stats()['grade_qualified_aliases'], 10)

    def test_unqualified_ambiguous_terms_are_never_guessed(self):
        for term, *_rest in AMBIGUOUS_CASES:
            with self.subTest(term=term):
                warnings: list = []
                self.assertIsNone(self.taxonomy.resolve(term, warnings))
                self.assertTrue(warnings)

    def test_grade_qualified_terms_resolve_to_the_right_node(self):
        for term, low_code, low_grade, high_code, high_grade in AMBIGUOUS_CASES:
            with self.subTest(term=term):
                self.assertEqual(self.taxonomy.resolve(f'{term}（{low_grade}）'), low_code)
                self.assertEqual(self.taxonomy.resolve(f'{term}（{high_grade}）'), high_code)

    def test_grade_qualifier_outside_node_grades_is_not_guessed(self):
        # 限定到该节点不覆盖的年级（笔算乘法只在三/四年级）→ 必须 None + warning，不就近挂靠
        for text_value in ('笔算乘法（五年级）', '笔算乘法（一年级）', '旋转（四年级）'):
            with self.subTest(value=text_value):
                warnings: list = []
                self.assertIsNone(self.taxonomy.resolve(text_value, warnings))
                self.assertTrue(warnings)

    def test_halfwidth_brackets_resolve_the_same_way(self):
        self.assertEqual(self.taxonomy.resolve('笔算乘法(四年级)'), 'M-G4-04')
        self.assertEqual(self.taxonomy.resolve('笔算乘法(三年级)'), 'M-G3-13')

    def test_ambiguous_terms_land_in_rejected(self):
        accepted, rejected = self.taxonomy.validate(['笔算乘法', '三角形的面积'])
        self.assertEqual(accepted, [('三角形面积', 'M-G4-07')])
        self.assertEqual(rejected, ['笔算乘法'])

    def test_validate_reports_warning_for_ambiguous_terms(self):
        warnings: list = []
        self.taxonomy.validate(['笔算乘法'], warnings)
        self.assertTrue(warnings)
        self.assertIn('笔算乘法', warnings[0])

    def test_disambiguated_names_still_resolve(self):
        self.assertEqual(self.taxonomy.resolve('图形的运动'), 'M-G2-13')
        self.assertEqual(self.taxonomy.resolve('图形的旋转'), 'M-G5-11')

    def test_declaring_an_already_resolvable_term_is_a_contradiction(self):
        # ambiguous_terms 与别名索引不能自相矛盾（otherwise 静默二选一）
        payload = minimal_payload()
        payload['ambiguous_terms'] = {'三角形面积': ['M-G4-07', 'M-G4']}
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        self.addCleanup(taxonomy.clear_cache)
        write_vocabulary(directory, 'math', '2026.10', payload)
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=directory)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)
        taxonomy.clear_cache()

    def test_duplicate_code_raises(self):
        payload = minimal_payload()
        payload['nodes'][1]['code'] = 'M-G4'
        write_vocabulary(self.directory, 'math', '2026.10', payload)
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_duplicate_name_raises(self):
        payload = minimal_payload()
        payload['nodes'][1]['name'] = '四年级'
        write_vocabulary(self.directory, 'math', '2026.10', payload)
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_dangling_parent_raises(self):
        payload = minimal_payload()
        payload['nodes'][1]['parent'] = 'M-G9'
        write_vocabulary(self.directory, 'math', '2026.10', payload)
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_grades_out_of_range_raises(self):
        payload = minimal_payload()
        payload['nodes'][1]['grades'] = [13]
        write_vocabulary(self.directory, 'math', '2026.10', payload)
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_version_mismatch_raises(self):
        payload = minimal_payload(version='math-2025.01')
        write_vocabulary(self.directory, 'math', '2026.10', payload)
        with self.assertRaises(ValueError):
            taxonomy.load('数学', root=self.directory)

    def test_missing_subject_raises_value_error(self):
        with self.assertRaises(ValueError):
            taxonomy.load('化学', root=self.directory)

    def test_unknown_version_raises(self):
        write_vocabulary(self.directory, 'math', '2026.10', minimal_payload())
        with self.assertRaises(ValueError):
            taxonomy.load('数学', version='2026.09', root=self.directory)

    def test_picks_highest_version_by_default(self):
        write_vocabulary(self.directory, 'math', '2026.09',
                         minimal_payload(version='math-2026.09'))
        write_vocabulary(self.directory, 'math', '2026.10',
                         minimal_payload(version='math-2026.10'))
        self.assertEqual(taxonomy.load('数学', root=self.directory).version, 'math-2026.10')
        self.assertEqual(
            taxonomy.load('数学', version='2026.09', root=self.directory).version,
            'math-2026.09',
        )


class NormalizationTests(unittest.TestCase):
    def test_fullwidth_and_spaces_are_normalized(self):
        self.assertEqual(taxonomy.normalize_name('三角形 面积（1）'), '三角形面积')

    def test_traditional_chinese_is_mapped(self):
        self.assertEqual(taxonomy.normalize_name('三角形面積'), '三角形面积')

    def test_quantity_prefix_is_not_stripped_as_index(self):
        # 只有带分隔符的行首数字才是序号；"20以内数的认识" 的 20 是数量
        self.assertEqual(taxonomy.normalize_name('20以内数的认识'), '20以内数的认识')
        self.assertNotEqual(taxonomy.normalize_name('20以内数的认识'),
                            taxonomy.normalize_name('100以内数的认识'))
        self.assertEqual(taxonomy.normalize_name('1.三角形面积'), '三角形面积')

    def test_grade_index_accepts_several_forms(self):
        self.assertEqual(taxonomy.grade_index('三年级'), 3)
        self.assertEqual(taxonomy.grade_index('七年级上册'), 7)
        self.assertEqual(taxonomy.grade_index('3'), 3)
        self.assertEqual(taxonomy.grade_index(4), 4)
        self.assertEqual(taxonomy.grade_index('初一'), 7)
        self.assertIsNone(taxonomy.grade_index('未知'))


if __name__ == '__main__':
    unittest.main()
