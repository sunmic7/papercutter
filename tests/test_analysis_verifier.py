# -*- coding: utf-8 -*-
"""解析机检单测（analysis-verify-v1）。

三件事必须成立：
1. 能复算的算错就必须判 fail（``12×4+8`` vs 56 → pass，vs 54 → fail）
2. 不可信的表达式串（``__import__('os').system(...)``）判 not_checkable 且**绝不执行**
3. 解析提到图形但题图缺失 → fail（幻觉的高发模式）
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from review_tools import analysis_verifier as verifier


def make_record(text='', answer='', analysis='', question_type='计算', subject='数学', **over):
    record = {
        'id': 'q1',
        'subject': subject,
        'question_type': question_type,
        'text': text,
        'answer': answer,
        'analysis': analysis,
    }
    record.update(over)
    return record


def make_snapshot(text='', answer='', illustration_boxes=None, **over):
    snapshot = {'text': text, 'answer': answer}
    if illustration_boxes is not None:
        snapshot['illustration_boxes'] = illustration_boxes
    snapshot.update(over)
    return snapshot


def check_named(result, name):
    return next((check for check in result['checks'] if check['name'] == name), {})


class StructureChainTests(unittest.TestCase):
    def test_short_analysis_fails(self):
        record = make_record('计算 3+4。', '7', '等于7')
        result = verifier.verify(record, make_snapshot('计算 3+4。', '7'))
        self.assertEqual(result['status'], 'fail')
        self.assertFalse(check_named(result, 'non_empty')['ok'])

    def test_placeholder_analysis_fails(self):
        record = make_record('计算 3+4。', '7', '略')
        result = verifier.verify(record, make_snapshot('计算 3+4。', '7'))
        self.assertEqual(result['status'], 'fail')
        self.assertFalse(check_named(result, 'not_placeholder')['ok'])

    def test_answer_copy_fails(self):
        analysis = '三加四等于七，所以答案是7。7'
        record = make_record('计算 3+4。', analysis, analysis)
        result = verifier.verify(record, make_snapshot('计算 3+4。', analysis))
        self.assertEqual(result['status'], 'fail')
        self.assertFalse(check_named(result, 'not_answer_copy')['ok'])

    def test_phantom_figure_fails_without_illustration(self):
        record = make_record(
            '求阴影部分的面积。', '24',
            '如图所示，阴影部分的面积等于长方形面积的一半，即 24 平方厘米。',
            question_type='解答',
        )
        result = verifier.verify(record, make_snapshot('求阴影部分的面积。', '24'))
        self.assertFalse(check_named(result, 'no_phantom_figure')['ok'])
        self.assertEqual(result['status'], 'fail')

    def test_phantom_figure_passes_when_boxes_present(self):
        record = make_record(
            '求阴影部分的面积。', '24',
            '如图所示，阴影部分的面积等于长方形面积的一半，即 24 平方厘米。',
            question_type='解答',
        )
        result = verifier.verify(
            record,
            make_snapshot('求阴影部分的面积。', '24',
                          illustration_boxes=[{'bbox': [10, 10, 80, 90]}]),
        )
        self.assertTrue(check_named(result, 'no_phantom_figure')['ok'])

    def test_refer_question_is_recorded_but_never_blocking(self):
        record = make_record('计算 3+4。', '7', '把两个加数合并起来就得到结果，这个结果是一个一位数。')
        result = verifier.verify(record, make_snapshot('计算 3+4。', '7'))
        refer = check_named(result, 'refer_question')
        self.assertFalse(refer['hard'])
        self.assertNotEqual(result['status'], 'fail')


class NumericRecomputeTests(unittest.TestCase):
    def test_correct_arithmetic_passes(self):
        record = make_record(
            '计算：12×4+8', '56', '先算乘法：12×4=48，再加上 8，得到 56。',
        )
        result = verifier.verify(record, make_snapshot('计算：12×4+8', '56'))
        self.assertEqual(result['status'], 'pass')
        self.assertTrue(check_named(result, 'numeric_recompute')['ok'])

    def test_wrong_arithmetic_fails_with_expected_and_got(self):
        record = make_record(
            '计算：12×4+8', '54', '先算乘法：12×4=48，再加上 8，得到 56。',
        )
        result = verifier.verify(record, make_snapshot('计算：12×4+8', '54'))
        numeric = check_named(result, 'numeric_recompute')
        self.assertEqual(result['status'], 'fail')
        self.assertFalse(numeric['ok'])
        self.assertIn('expected=54', numeric['detail'])
        self.assertIn('got=56', numeric['detail'])

    def test_fraction_arithmetic_is_exact(self):
        record = make_record(
            '计算：1/2+1/3', '5/6', '先通分，二分之一是六分之三，三分之一是六分之二，合起来是六分之五。',
            question_type='填空',
        )
        result = verifier.verify(record, make_snapshot('计算：1/2+1/3', '5/6'))
        self.assertTrue(check_named(result, 'numeric_recompute')['ok'])

    def test_square_root_is_exact_when_perfect_square(self):
        record = make_record(
            '计算：√9+1', '4', '九的算术平方根是三，再加上一，结果是四。',
        )
        result = verifier.verify(record, make_snapshot('计算：√9+1', '4'))
        self.assertTrue(check_named(result, 'numeric_recompute')['ok'])

    def test_multi_part_answers_are_compared_one_by_one(self):
        record = make_record(
            '计算：（1）12×4+8（2）3+4', '56 7',
            '第一问先算乘法再算加法，第二问直接相加，两问分别得到两个结果。',
        )
        good = verifier.verify(record, make_snapshot('计算：（1）12×4+8（2）3+4', '56 7'))
        bad = verifier.verify(record, make_snapshot('计算：（1）12×4+8（2）3+4', '56 9'))
        self.assertTrue(check_named(good, 'numeric_recompute')['ok'])
        self.assertFalse(check_named(bad, 'numeric_recompute')['ok'])
        self.assertEqual(bad['status'], 'fail')

    def test_unsupported_question_type_is_not_applicable(self):
        record = make_record('证明三角形内角和为180度。', '略', '过顶点作平行线，用内错角相等完成证明。',
                             question_type='证明')
        result = verifier.verify(record, make_snapshot('证明三角形内角和为180度。', '180'))
        self.assertIsNone(check_named(result, 'numeric_recompute')['ok'])

    def test_non_math_subject_is_not_applicable(self):
        record = make_record('读拼音写词语 12×4。', '56', '根据拼音写出对应的词语，注意字形结构。',
                             subject='语文')
        result = verifier.verify(record, make_snapshot('读拼音写词语 12×4。', '56'))
        self.assertIsNone(check_named(result, 'numeric_recompute')['ok'])


class InjectionSafetyTests(unittest.TestCase):
    def test_import_call_is_not_evaluated(self):
        payload = "__import__('os').system('touch pwned')"
        self.assertIsNone(verifier.safe_eval_expression(payload))
        self.assertIsNone(verifier.safe_eval_expression("open('/etc/passwd').read()"))
        self.assertIsNone(verifier.safe_eval_expression('os.system("ls")'))
        self.assertIsNone(verifier.safe_eval_expression('__import__("os")'))

    def test_whitelist_rejects_unknown_nodes(self):
        for expression in ('max(1,2)', '2**999', '1/0', '(1,2)', '[1,2]', '1 if 1 else 2',
                           'abs(-1)+sqrt(4)'):
            with self.subTest(expression=expression):
                value = verifier.safe_eval_expression(expression)
                if expression == 'abs(-1)+sqrt(4)':
                    self.assertEqual(value, 3)
                else:
                    self.assertIsNone(value)

    def test_injection_never_creates_side_effect_file(self):
        directory = tempfile.mkdtemp()
        marker = os.path.join(directory, 'pwned').replace('\\', '/')
        payload = f"__import__('os').system('touch {marker}')"
        record = make_record(
            f'计算：{payload}', '56',
            '这道题需要按运算顺序计算，题目文字本身不含任何可执行的表达式。',
        )
        result = verifier.verify(record, make_snapshot(f'计算：{payload}', '56'))
        self.assertNotEqual(result['status'], 'pass')
        self.assertFalse(os.path.exists(marker))
        self.assertIsNone(verifier.safe_eval_expression(payload))


class DegradedChainTests(unittest.TestCase):
    def _record(self):
        return make_record(
            '买3支笔，每支2元，一共（  ）元。', '6元',
            '每支2元，3支就是 3×2=6，所以一共 6 元。',
            question_type='解答',
        )

    def test_all_degraded_checks_pass_marks_status_pass(self):
        record = self._record()
        result = verifier.verify(record, make_snapshot('买3支笔，每支2元，一共（  ）元。', '6元'))
        self.assertEqual(result['status'], 'pass')
        self.assertTrue(result['degraded'])

    def test_phantom_number_in_analysis_fails(self):
        record = make_record(
            '求阴影部分的面积。', '24',
            '先把长方形面积算出来是 48，再取一半得到 24，所以阴影面积是 24 平方厘米。',
            question_type='解答',
        )
        snapshot = make_snapshot('求阴影部分的面积。', '24',
                                 illustration_boxes=[{'bbox': [0, 0, 10, 10]}])
        result = verifier.verify(record, snapshot)
        self.assertFalse(check_named(result, 'symbol_subset')['ok'])
        self.assertEqual(result['status'], 'fail')

    def test_blank_answer_missing_from_analysis_fails(self):
        record = self._record()
        record['analysis'] = '根据乘法的意义，逐支相加即可得到总数，注意单位要写完整。'
        result = verifier.verify(record, make_snapshot('买3支笔，每支2元，一共（  ）元。', '6元'))
        self.assertFalse(check_named(result, 'blank_fill')['ok'])
        self.assertEqual(result['status'], 'fail')

    def test_unexpected_unit_fails(self):
        record = make_record(
            '边长3米的正方形面积是多少？', '9',
            '正方形的面积等于边长乘边长，所以面积是 9 平方米。',
            question_type='解答',
        )
        result = verifier.verify(record, make_snapshot('边长3米的正方形面积是多少？', '9'))
        self.assertFalse(check_named(result, 'unit_consistency')['ok'])
        self.assertEqual(result['status'], 'fail')

    def test_nothing_checkable_is_not_checkable(self):
        record = make_record(
            '作一个半径为2厘米的圆，并标出圆心。', '略',
            '用圆规量出半径后旋转一周即可完成作图，注意保留作图痕迹。',
            question_type='作图',
        )
        record['analysis'] = '用圆规量出半径后旋转一周即可完成作图，注意保留作图痕迹。'
        result = verifier.verify(record, make_snapshot('作一个半径为2厘米的圆，并标出圆心。', ''))
        self.assertEqual(result['status'], 'not_checkable')


class ContractTests(unittest.TestCase):
    def test_result_carries_version_and_all_check_names(self):
        record = make_record('计算：12×4+8', '56', '先算乘法：12×4=48，再加上 8，得到 56。')
        result = verifier.verify(record, make_snapshot('计算：12×4+8', '56'))
        self.assertEqual(result['verifier_version'], verifier.VERIFIER_VERSION)
        names = {check['name'] for check in result['checks']}
        self.assertTrue({
            'non_empty', 'not_placeholder', 'not_answer_copy', 'no_phantom_figure',
            'refer_question', 'numeric_recompute', 'symbol_subset', 'blank_fill',
            'unit_consistency',
        } <= names)

    def test_verify_tolerates_missing_snapshot(self):
        record = make_record('计算：12×4+8', '56', '先算乘法：12×4=48，再加上 8，得到 56。')
        result = verifier.verify(record, None)
        self.assertIn(result['status'], {'pass', 'fail', 'not_checkable'})


if __name__ == '__main__':
    unittest.main()
