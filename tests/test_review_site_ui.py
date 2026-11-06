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


class TaskListGroupingTests(unittest.TestCase):
    """同一本教材提交过多次时，列表只给最新一条；要看全部走「历史版本」。"""

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_only_the_newest_submission_per_book_is_listed(self):
        self.assertIn("function latestTaskPerBook(rows)", self.source)
        self.assertIn("const grouped = !['all','resubmitted'].includes(taskState.filter)", self.source)
        self.assertIn("grouped ? latestTaskPerBook(taskState.rows) : taskState.rows", self.source)

    def test_kpi_counts_use_the_grouped_rows(self):
        self.assertIn("const count = filter === 'pending_pages' ? baseRows.reduce", self.source)


class TaskBatchManagementTests(unittest.TestCase):
    """管理员在审核任务页要有批量管理，而不是只能一条条点进去。"""

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_batch_bar_exposes_assign_delete_and_restore(self):
        self.assertIn("id=\"taskBatchBar\"", self.source)
        self.assertIn("'/api/submissions/batch/assign'", self.source)
        self.assertIn("'/api/deliveries/tasks/batch'", self.source)
        self.assertIn("batchRestoreTasks", self.source)

    def test_batch_management_is_offered_to_admins(self):
        self.assertIn("button(manage ? '退出批量管理' : '批量管理','toggleTaskManage')", self.source)
        self.assertIn("const admin = canManageUsers(), manage = taskState.manage;", self.source)


class ReviewQuickDecisionTests(unittest.TestCase):
    """逐页点单选+保存太慢，页面上要能一键出结论。"""

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_one_click_conclusion_buttons_exist(self):
        for value in ("data-quick=\"approved\"", "data-quick=\"needs_fix\"", "data-quick=\"returned\""):
            self.assertIn(value, self.source)
        self.assertIn("form.elements.status.value = node.dataset.quick", self.source)

    def test_whole_book_approval_uses_the_bulk_endpoint(self):
        self.assertIn("id=\"approveAllPages\"", self.source)
        self.assertIn("/pages/decide`, {status:'approved'}", self.source)

    def test_outdated_notice_explains_why_it_happened(self):
        self.assertIn("「待重新提交」＝提交之后工作台里这本书的标注又被改过", self.source)


class SourceFolderMenuTests(unittest.TestCase):
    """来源文件夹从"扫描时顺手记一笔"升级成独立子菜单，可编辑。"""

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_library_page_has_a_source_folder_tab(self):
        self.assertIn("['sources','来源文件夹']", self.source)
        self.assertIn("async function showSourceFolders()", self.source)
        self.assertIn("'/api/library/sources'", self.source)

    def test_old_add_folder_button_is_gone(self):
        self.assertNotIn("button('添加教材文件夹','addLibraryFolder')", self.source)

    def test_editing_a_source_folder_can_change_its_path(self):
        self.assertIn("新路径", self.source)
        self.assertIn("new_path", self.source)


if __name__ == "__main__":
    unittest.main()
