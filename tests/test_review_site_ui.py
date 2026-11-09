"""审核网站前端的交互护栏（静态断言）。

这些是纯前端行为，后端测试碰不到：审核任务顶部的状态卡片曾经一点就进筛选、
再也回不到"全部当前任务"，用户只能刷新页面；诊断的示例列也曾被塞进
"第N页 题号"，而那个页号是扫描页序，和书上的印刷页码对不上。
"""
from pathlib import Path
import re
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
        for key, label in (("items", "逐题明细"), ("missing", "缺什么"), ("coverage", "生成概览")):
            self.assertIn(f"['{key}','{label}']", self.source)

    def test_item_detail_comes_first_and_is_the_default(self):
        # 点进来最想看的就是"哪道题不行"，逐题明细排第一且默认选中，别再多点一次。
        self.assertIn("['items','逐题明细'],", self.source)
        self.assertIn("menu:'items'", self.source)
        self.assertIn("checkState.menu='items';", self.source)
        # 旧的「概览」「逐页收录」两个菜单已经合并成一个。
        self.assertNotIn("['overview','概览']", self.source)
        self.assertNotIn("['pages','逐页收录']", self.source)

    def test_overview_and_page_coverage_are_one_menu(self):
        self.assertIn("function checkCoverageHtml()", self.source)
        self.assertIn("chk-divider", self.source)
        self.assertIn("${checkPagesHtml()}", self.source)
        self.assertNotIn("checkOverviewHtml", self.source)

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
        self.assertIn("逐页生成明细只对有审核权限的账号开放", self.source)
        server = (STATIC.parent / "central_library.py").read_text(encoding="utf-8")
        self.assertIn("只有审核员或管理员可以查看交付页面明细", server)


class RegenerateSucceededDeliveryTests(unittest.TestCase):
    """生成成功之后还要能再来一次：后面补了新题、改了标注，全靠这个按钮继续出包。"""

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_succeeded_rows_still_offer_regenerate(self):
        self.assertIn("const rerun=state==='succeeded';", self.source)
        self.assertIn("['failed','not_generated','awaiting_review'].includes(state)||rerun", self.source)
        self.assertIn("rerun?'重新生成'", self.source)

    def test_regenerate_is_forced(self):
        # 不带 force 时后端看到已有成功任务会直接返回 already_queued，按钮就成摆设。
        self.assertIn("data-force=\"${rerun?'1':''}\"", self.source)
        self.assertIn("{force:Boolean(force),only_reviewed:Boolean(values.only_reviewed)}", self.source)

    def test_regenerate_dialog_says_the_old_version_is_replaced(self):
        self.assertIn("生成成功后新版本替换上一版，旧的那份 JSON 一并删除", self.source)


class PreviewRenderSafetyTests(unittest.TestCase):
    """交付预览整页崩过一次：``map(text=>`` 被写成 ``map(text=``，
    strict mode 下每次迭代都抛 ``text is not defined``，页面只剩一句报错。"""

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_no_assignment_inside_callback_parameters(self):
        broken = re.compile(r"\.(?:map|filter|forEach|find|some|every)\(\s*[A-Za-z_$][\w$]*\s*=(?![=>])")
        self.assertEqual(broken.findall(self.source), [],
                         "回调参数写成了赋值（漏了 =>），运行时会抛 ReferenceError")

    def test_analysis_only_shows_when_not_already_merged(self):
        # 答案与解析已经合并；analysis 只有在没并进 answer 时才单独补一块。
        self.assertIn("const hasExtraAnalysis = extraAnalysis && !String(body.answer || '').includes(extraAnalysis);", self.source)
        self.assertIn("${hasExtraAnalysis ?", self.source)

    def test_preview_keeps_answer_and_analysis_in_one_block(self):
        self.assertIn("<h4>答案与解析</h4>", self.source)
        self.assertNotIn("<h4>详细解析</h4>", self.source)


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


class DeliveryScopeBadgeTests(unittest.TestCase):
    """生成成功 ≠ 整本都生成了：列表要写明这一版生成的页数/题数，部分生成单独标。

    操作员反馈：看到"最终 JSON 与裁剪图已发布"还以为是整本教材都出完了，
    实际上被退回的页、字段不全的题都没生成。另外口径统一用"生成"，不用"收录"。
    """

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_row_shows_real_question_counts(self):
        self.assertIn("function deliveryCoverageView(", self.source)
        self.assertIn("row.delivery_coverage||{}", self.source)
        self.assertIn("页全部生成", self.source)
        self.assertIn("生成 ${num(cov.inQuestions)}/${num(total)} 题", self.source)

    def test_every_missing_case_has_its_own_phrase(self):
        # 每种没生成的情况各有一句话，有几种列几种，不允许只留一句笼统的。
        self.assertIn("'未审核放行':'被审核退回'}，没有生成", self.source)
        self.assertIn("页的题目全都没通过交付校验", self.source)
        self.assertIn("页只有部分题目通过校验", self.source)
        self.assertIn("题没通过交付校验", self.source)

    def test_legacy_summary_without_question_counts_stays_honest(self):
        # 老汇总没有题数：不硬凑数字，只说"部分生成"，页数从略。
        self.assertIn("已发布（部分生成）", self.source)
        self.assertIn("cov.hasQuestions", self.source)

    def test_partial_rows_get_a_distinct_badge(self):
        self.assertIn("'部分生成'", self.source)
        self.assertIn("const badgeTone=", self.source)

    def test_kpi_card_counts_partial_successes(self):
        self.assertIn("其中 ${num(partialCount)} 项部分生成", self.source)

    def test_old_rows_without_coverage_keep_a_honest_fallback(self):
        # 升级前的老任务没有覆盖汇总，只能退回一句话，且要承认"看不到具体页数"。
        self.assertIn("cov.known", self.source)
        self.assertIn("最终 JSON 与裁剪图已发布", self.source)

    def test_status_legend_explains_the_numbers(self):
        self.assertIn("标“部分生成”说明还有页面或题目没生成", self.source)

    def test_the_word_shou_lu_is_gone_from_user_visible_text(self):
        for path in ("app.js", "styles.css", "index.html", "review-ui.js"):
            text = (STATIC / path).read_text(encoding="utf-8")
            self.assertNotIn("收录", text, f"{path} 里还有“收录”字样")
        for name in ("central_library.py", "server.py"):
            text = (STATIC.parent / name).read_text(encoding="utf-8")
            self.assertNotIn("收录", text, f"{name} 里还有“收录”字样")


class ContinuationBoxReviewTests(unittest.TestCase):
    """审核页要能看到续框：跨页题目在网站上只剩半截，审核员没法下结论。

    提交快照以前只带 region_role=question 的框，续框（跨页续题片段）整个消失；
    现在快照带上 role=continuation，前端用橙色虚线框单独画出来并在计数里注明。
    """

    def setUp(self):
        self.source = (STATIC / "app.js").read_text(encoding="utf-8")
        self.styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    def test_queue_shows_continuation_count(self):
        self.assertIn("个框${contCount ? ` · ${contCount} 续框` : ''}", self.source)

    def test_overlay_draws_continuation_boxes_with_a_distinct_style(self):
        self.assertIn('String(region.role || \'question\') === \'continuation\'', self.source)
        self.assertIn('class="cont"', self.source)
        self.assertIn("续框", self.source)
        self.assertIn(".image-board rect.cont{stroke:#f79009", self.styles)
        self.assertIn(".image-board text.cont{fill:#b45309}", self.styles)


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
