"""标注工作台的页码口径测试。

工作台里同时存在三处「第 N 页」：左侧缩略图/顶部 1/N、题目-答案匹配面板、
匹配分组面板。这三处必须指同一个数字，否则同一张图在三个地方显示三个页码。

历史 bug（2026-11-06）：匹配面板按「题目页 + 答案页 + 忽略页合并排序」算全书序号，
这本教材开头 4 张封面被归到忽略页，于是题目页第 1 页显示成「第 5 页」、
答案页第 1 页显示成「第 41 页」，而教材界面显示的是「第 1 页」。

这里直接从 main.js 里抽出这两个方法在 node 里真跑，而不是只做文本断言。
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

APP = pathlib.Path(__file__).resolve().parents[1] / 'desktop_annotation' / 'PaperFlow-Annotation' / 'resources' / 'app'
MAIN_JS = APP / 'js' / 'main.js'

NODE = shutil.which('node')


def extract_method(source: str, name: str) -> str:
    """按大括号配平抽取一个类方法（模板字符串里的 ${} 是配对的，不影响计数）。"""
    marker = f'    {name}('
    start = source.index(marker)
    cursor = source.index('{', start)
    depth = 0
    while cursor < len(source):
        char = source[cursor]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return source[start:cursor + 1]
        cursor += 1
    raise AssertionError(f'{name} 方法没有闭合')


def build_project():
    """按出事那本教材的真实结构造数据：4 张封面被忽略，36+2 张题目页，2 张答案页。"""
    def page(name):
        return {'image': f'J:/PaperFlowData/Public/SourceImages/BADA/{name}.jpg',
                'source_path': f'J:/PaperFlowData/Public/SourceImages/BADA/{name}.jpg'}

    ignored = [page(f'20210126153257260{n}') for n in (649, 650, 651, 652)]
    questions = [page(f'20210126153257260{n}') for n in list(range(653, 689)) + [691, 692]]
    answers = [page('20210126153257260 689'.replace(' ', '')), page('20210126153257260 690'.replace(' ', ''))]
    return {'pages': questions, 'answer_pages': answers, 'ignored_pages': ignored}


@unittest.skipUnless(NODE, '本机没有 node，跳过前端逻辑测试')
class PageNumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = MAIN_JS.read_text(encoding='utf-8')
        cls.source = source
        cls.number_fn = extract_method(source, 'paperFlowSourcePageNumber')
        cls.label_fn = extract_method(source, 'paperFlowMatchRecordLabel')

    def run_js(self, body: str, project=None):
        script = f"""
const project = {json.dumps(project or build_project(), ensure_ascii=False)};
const ctx = {{
  paperFlowKind: 'question',
  paperFlowProject: project,
  paperFlowDisplayNumber: record => String(record?.number ?? ''),
  paperFlowHeadingWithoutNumber: () => ''
}};
// 抽出来的是「方法简写」形式，放进对象字面量再 bind 到假实例上。
const holder = eval('({{' + {json.dumps(self.number_fn)} + ',' + {json.dumps(self.label_fn)} + '}})');
for (const [key, value] of Object.entries(holder)) ctx[key] = value.bind(ctx);
const out = (() => {{ {body} }})();
process.stdout.write(JSON.stringify(out));
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'probe.js'
            path.write_text(script, encoding='utf-8')
            result = subprocess.run([NODE, str(path)], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_first_question_page_is_page_one(self):
        """题目页第 1 张必须是第 1 页，不能因为开头有 4 张忽略页就变成第 5 页。"""
        value = self.run_js('return ctx.paperFlowSourcePageNumber(project.pages[0], 0, "question");')
        self.assertEqual(value, 1)

    def test_first_answer_page_is_page_one(self):
        """答案页是独立页签，第 1 张必须是第 1 页，不是全书第 41 页。"""
        value = self.run_js('return ctx.paperFlowSourcePageNumber(project.answer_pages[0], 0, "answer");')
        self.assertEqual(value, 1)

    def test_answer_page_never_counts_question_pages(self):
        value = self.run_js('return ctx.paperFlowSourcePageNumber(project.answer_pages[1], 1, "answer");')
        self.assertEqual(value, 2)

    def test_label_carries_page_kind(self):
        """题目和答案各有一套 1..N，标签必须写清是题目页还是答案页的第几页。"""
        value = self.run_js("""
return [
  ctx.paperFlowMatchRecordLabel({pageIndex: 0, page: project.pages[0], question: {number: '2-1'}}, false),
  ctx.paperFlowMatchRecordLabel({pageIndex: 0, page: project.answer_pages[0], answer: {number: '2-1'}}, true)
];
""")
        self.assertEqual(value[0], '题目页 第1页 · 2-1')
        self.assertEqual(value[1], '答案页 第1页 · 2-1')

    def test_falls_back_to_image_lookup(self):
        """pageIndex 缺失时按图片路径反查，仍然给页签内序号。"""
        value = self.run_js('return ctx.paperFlowSourcePageNumber(project.pages[4], undefined, "question");')
        self.assertEqual(value, 5)

    def test_unknown_page_does_not_invent_a_number(self):
        value = self.run_js('return ctx.paperFlowSourcePageNumber({image: "nope.jpg"}, undefined, "question");')
        self.assertEqual(value, 0)
        label = self.run_js('return ctx.paperFlowMatchRecordLabel({pageIndex: undefined, page: {image: "nope.jpg"}, question: {number: "1-1"}}, false);')
        self.assertEqual(label, '题目页 · 1-1')

    def test_source_page_number_is_not_merged_book_order(self):
        """回归护栏：页码函数里不能再出现合并排序/忽略页/文件名前缀取值。"""
        self.assertNotIn('ignored_pages', self.number_fn)
        self.assertNotIn('localeCompare', self.number_fn)
        self.assertNotIn('source_page_number', self.number_fn)

    def test_thumbnail_counter_uses_same_base(self):
        """顶部 1/N 必须仍是当前页签内的 currentIndex+1，和上面保持同一套编号。"""
        source = (APP / 'js' / 'image-manager.js').read_text(encoding='utf-8')
        self.assertIn('currentImageIndex.textContent = this.hasImages() ? this.currentIndex + 1 : 0', source)
        self.assertIn('totalImages.textContent = this.images.length', source)


if __name__ == '__main__':
    unittest.main()
