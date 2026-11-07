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
    """汇总表只说"缺什么、各多少题"；具体是哪几道题，看下面的逐题明细。

    这张表曾经挂过「示例题号」一列（先是裸题号，后来是"第N页 2-1"），
    两种写法都被操作员判为看不懂，而且和逐题明细重复，已整列去掉。
    """

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_summary_table_has_no_example_column(self):
        self.assertNotIn("<th>示例题号</th>", self.source)
        self.assertNotIn("示例（页码 · 题号）", self.source)

    def test_summary_table_keeps_reason_and_count(self):
        self.assertIn("<th>缺少项</th><th>题数</th>", self.source)


class DeliveryCheckEntryTests(unittest.TestCase):
    """「检查缺什么」和「页面明细」合成一个入口，并改成整页 + 左侧菜单。

    题目一多，逐题明细就是几百行；原来的做法是把所有块堆在一个弹窗里一路往下拉，
    操作员要看最后一页得滚很久。现在左边点菜单、右边只看这一块，明细自带分页。
    """

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_one_merged_entry_replaces_two_buttons(self):
        self.assertIn("检查与明细", self.source)
        self.assertIn("async function showDeliveryCheck(", self.source)
        self.assertIn("function checkShellHtml(", self.source)
        self.assertNotIn("diagnose-delivery", self.source)
        self.assertNotIn("delivery-pages'", self.source)

    def test_live_sample_list_is_gone(self):
        # 「实时题单」与「预览内容」同源，且服务器端早已不产草稿目录，入口已删。
        self.assertNotIn("实时题单", self.source)
        self.assertNotIn("preview-live-samples", self.source)
        self.assertNotIn("openLiveSamples", self.source)

    def test_panel_is_a_full_page_with_a_menu(self):
        # 不再是 showPanel 弹窗，而是和「预览内容」一样的整页视图。
        self.assertIn("'checkView'", self.source)
        self.assertIn("class=\"chk-body\"", self.source)
        self.assertIn("class=\"chk-menu\"", self.source)
        for key, label in (("overview", "概览"), ("missing", "缺什么"),
                           ("items", "逐题明细"), ("pages", "逐页收录")):
            self.assertIn(f"['{key}','{label}']", self.source)

    def test_item_detail_is_paginated(self):
        # 逐题明细必须分页，不能一次把几百行全铺出来；分页条还要在表格上方，
        # 否则一页 30 条也得先滚到底才能翻页。
        self.assertIn("CHECK_PAGE_SIZE = 30", self.source)
        self.assertIn("paginate(filtered,checkState.page,CHECK_PAGE_SIZE)", self.source)
        self.assertIn("bindPager(panel,checkState.lastPage", self.source)
        self.assertIn('class="chk-listhead"', self.source)

    def test_missing_reason_row_jumps_to_filtered_items(self):
        self.assertIn("data-reason-jump", self.source)
        self.assertIn("checkState.reason=node.dataset.reasonJump", self.source)
        self.assertIn("data-clear-filter", self.source)

    def test_hints_and_sample_json_are_no_longer_shown_here(self):
        # 两个碍眼的块都撤了：题图提醒没有行动价值，JSON 说明搬到「预览内容」。
        self.assertNotIn("只是提醒，不影响生成", self.source)
        self.assertNotIn("可交付样本长什么样", self.source)

    def test_delivery_json_is_explained_in_the_preview(self):
        self.assertIn("交付 JSON（这条题写进 dataset.jsonl 的原文）", self.source)
        self.assertIn("这就是最终交付出去的内容", self.source)

    def test_page_coverage_is_hidden_from_annotators(self):
        # 逐页明细接口只对审核员/管理员开放，标注员点开时降级成一句说明。
        self.assertIn("逐页收录明细只对有审核权限的账号开放", self.source)
        server = (STATIC.parent / "central_library.py").read_text(encoding="utf-8")
        self.assertIn("只有审核员或管理员可以查看交付页面明细", server)

    def test_page_coverage_is_hidden_from_annotators(self):
        # 逐页明细接口只对审核员/管理员开放，标注员点开时降级成一句说明。
        self.assertIn("逐页收录明细只对有审核权限的账号开放", self.source)
        server = (STATIC.parent / "central_library.py").read_text(encoding="utf-8")
        self.assertIn("只有审核员或管理员可以查看交付页面明细", server)


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


class WordingGuardTests(unittest.TestCase):
    """界面上不再出现"甲方"这种对外称呼，一律用"交付"口径。"""

    def test_no_supplier_wording_on_any_ui_surface(self):
        root = STATIC.parents[1]
        workbench = root / "desktop_annotation" / "PaperFlow-Annotation" / "resources" / "app" / "js"
        surfaces = [STATIC / "app.js", STATIC / "index.html", STATIC / "styles.css", STATIC / "review-ui.js"]
        surfaces.extend(path for path in sorted(workbench.glob("*.js")) if path.is_file())
        for path in surfaces:
            if not path.is_file():
                continue
            self.assertNotIn("甲方", path.read_text(encoding="utf-8"), f"{path.name} 里还有“甲方”字样")

    def test_answer_and_analysis_share_one_block(self):
        source = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("<h4>答案与解析</h4>", source)
        # 解析已经并进答案，单独的「详细解析」块不再常驻；旧数据才补一块「补充解析」。
        self.assertNotIn("<h4>详细解析</h4>", source)
