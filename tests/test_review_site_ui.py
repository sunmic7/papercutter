"""审核网站前端的交互护栏（静态断言）。

这些是纯前端行为，后端测试碰不到：审核任务顶部的状态卡片曾经一点就进筛选、
再也回不到"全部当前任务"，用户只能刷新页面；诊断的示例列也曾被塞进
"第N页 题号"，而那个页号是扫描页序，和书上的印刷页码对不上。
"""
from pathlib import Path
import unittest


STATIC = Path(__file__).resolve().parents[1] / "online_review_beta" / "static"


class TaskStatusCardTests(unittest.TestCase):
    def test_clicking_the_active_status_card_clears_the_filter(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("taskState.filter === next ? 'active' : next", source,
                      "状态卡片必须是开关：再点一次回到「全部当前任务」")


class DiagnosisExampleTests(unittest.TestCase):
    def test_example_column_header_is_just_the_question_number(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("<th>示例题号</th>", source)
        self.assertNotIn("示例（页码 · 题号）", source)


if __name__ == "__main__":
    unittest.main()
