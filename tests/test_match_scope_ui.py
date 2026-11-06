"""匹配分组面板的交互约束。

两个需求（2026-11-06）：
1. 「当前分组」默认落在上一次操作过的分组上，并且没有「请选择」这一项；
2. 已经归过分组的分组，在下拉选项右边标一个绿点。

另外锁住「页面分类」不再加载原图缩略图 —— 一本教材几百页，弹窗一开
就要读几百张扫描图，读卡器上要等十几秒。
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
    return {
        'match_scopes': [{'id': 'scope-1', 'name': '1'}, {'id': 'scope-2', 'name': '2'},
                         {'id': 'scope-3', 'name': '3'}],
        'pages': [
            {'image': 'p1.jpg', 'match_scope_id': 'scope-1'},
            {'image': 'p2.jpg', 'match_scope_regions': [{'scope_id': 'scope-2', 'bbox': [0, 0, 1, 1]}],
             'questions': [{'id': 'q1', 'match_scope_id': 'scope-3'}]},
            {'image': 'p3.jpg'},
        ],
        'answer_pages': [
            {'image': 'a1.jpg', 'answers': [{'id': 'a1', 'match_scope_id': 'scope-1'}]},
        ],
    }


@unittest.skipUnless(NODE, '本机没有 node，跳过前端逻辑测试')
class MatchScopeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN_JS.read_text(encoding='utf-8')

    def run_js(self, body: str, prelude: str = '') -> object:
        methods = ','.join(extract_method(self.source, name) for name in (
            'paperFlowUsedScopeIds', 'paperFlowActiveScopeId', 'paperFlowRememberScope',
            'paperFlowRecalledScope', 'paperFlowScopeMemoryKey'))
        script = f"""
{prelude}
const project = {json.dumps(build_project(), ensure_ascii=False)};
const ctx = {{
  paperFlowKind: 'question',
  paperFlowProject: project,
  paperFlowProjectPath: 'J:/PaperFlowData/Public/SourceImages/BOOK',
  paperFlowSelectedScopeId: ''
}};
const holder = eval('({{' + {json.dumps(methods)} + '}})');
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

    def test_used_scopes_cover_pages_regions_and_records(self):
        """整页分组、框选区域、题目/答案各自的 match_scope_id 都算「用过」。"""
        used = self.run_js('return [...ctx.paperFlowUsedScopeIds()].sort();')
        self.assertEqual(used, ['scope-1', 'scope-2', 'scope-3'])

    def test_unused_scope_is_not_green(self):
        project = build_project()
        project['match_scopes'].append({'id': 'scope-9', 'name': '9'})
        used = self.run_js('return [...ctx.paperFlowUsedScopeIds()].sort();')
        self.assertNotIn('scope-9', used)

    def test_default_scope_keeps_previous_choice(self):
        """本次会话选过就沿用它，不能每次都跳回第一个。"""
        value = self.run_js('ctx.paperFlowSelectedScopeId = "scope-3"; return ctx.paperFlowActiveScopeId(project.match_scopes);')
        self.assertEqual(value, 'scope-3')

    def test_default_scope_uses_remembered_choice(self):
        """新开面板时用本机记住的上一次分组。"""
        prelude = "global.localStorage = { getItem: () => 'scope-2', setItem: () => {} };"
        value = self.run_js('return ctx.paperFlowActiveScopeId(project.match_scopes);', prelude)
        self.assertEqual(value, 'scope-2')

    def test_default_scope_falls_back_to_first_when_missing(self):
        """记住的分组被删了也不能卡住，退回第一个。"""
        value = self.run_js('ctx.paperFlowSelectedScopeId = "scope-gone"; return ctx.paperFlowActiveScopeId(project.match_scopes);')
        self.assertEqual(value, 'scope-1')

    def test_default_scope_is_empty_without_scopes(self):
        value = self.run_js('return ctx.paperFlowActiveScopeId([]);')
        self.assertEqual(value, '')

    def test_scope_select_has_no_placeholder_option(self):
        self.assertNotIn('<option value="">请选择</option>', self.source)
        self.assertIn('暂无分组', self.source)

    def test_scope_select_marks_used_scopes(self):
        block = self.source[self.source.index('id="pfScopeSelect"'):]
        block = block[:block.index('</select>')]
        self.assertIn('usedScopeIds.has(String(item.id))', block)
        self.assertIn('🟢', block)

    def test_page_manager_card_has_no_thumbnail(self):
        """页面分类卡片只给序号和状态，不再塞原图。"""
        card = extract_method(self.source, 'paperFlowPageManagerCardHtml')
        self.assertNotIn('<img', card)
        self.assertIn('序号 ${index+1}', card)


if __name__ == '__main__':
    unittest.main()
