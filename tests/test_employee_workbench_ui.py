from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "desktop_annotation" / "PaperFlow-Annotation" / "resources" / "app" / "js"


class EmployeeWorkbenchInteractionTests(unittest.TestCase):
    def test_logged_out_staff_must_login_before_opening_any_textbook(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        review = (APP_JS / "company-review.js").read_text(encoding="utf-8")
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")

        open_start = main.index("async openPaperFlowBook(")
        open_end = main.index("redrawCurrentPaperFlowPageAfterLayout", open_start)
        open_method = main[open_start:open_end]
        self.assertIn("this.companyReview?.session?.logged_in", open_method)
        self.assertNotIn("window.electronAPI.selectWorkspaceFolder()", open_method)
        self.assertLess(open_method.index("this.companyReview?.session?.logged_in"), open_method.index("if (!bookRoot)"))
        self.assertIn('id="companyLoginGate"', html)
        self.assertIn('this.loginReady?.()', review)
        self.assertIn("请先点击上方“登录公司账号”，登录后即可使用审核功能。", review)
        self.assertLess(html.index('class="workflow-steps"'), html.index('class="company-account-controls"'))

    def test_startup_does_not_try_to_restore_a_server_book_before_login(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        init_start = source.index("async init()")
        init_end = source.index("setPaperFlowInteractionLocked", init_start)
        init = source[init_start:init_end]

        self.assertIn("await this.companyReview.init()", init)
        self.assertIn("await this.restorePaperFlowAccountBook();", init)
        self.assertIn("await this.companyReview.requireLogin()", init)
        restore = source[source.index("async restorePaperFlowAccountBook()"):source.index("async paperFlowRequest(")]
        self.assertIn("const key = this.paperFlowLastBookKey();", restore)
        self.assertIn("if (!key) return;", restore)
        self.assertNotIn("getItem('paperflowLastBookRoot')", restore)

    def test_edit_menu_rotates_only_an_empty_paperflow_page(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        start = source.index("async rotateCurrentPaperFlowPage()")
        end = source.index("async showPaperFlowDeliveryOutput()", start)
        method = source[start:end]

        self.assertIn("顺时针旋转当前页面 90°", electron)
        self.assertIn("menu-rotate-current-page", electron)
        self.assertIn("onMenuRotateCurrentPage", preload)
        self.assertIn("currentAnnotations.length || storedRecords.length", method)
        self.assertIn("'/api/project/rotate-page'", method)
        self.assertLess(method.index("await this.persistCurrentPaperFlowCache()"),
                        method.index("'/api/project/rotate-page'"))

    def test_canvas_context_menu_copies_a_frame_and_pastes_paperflow_continuation(self):
        source = (APP_JS / "annotation-manager.js").read_text(encoding="utf-8")
        start = source.index("onContextMenu(event)")
        end = source.index("/**\n     * 添加回调函数", start)
        method = source[start:end]
        paste_start = source.index("pasteAnnotationAt(imageCoords, mode = 'continuation')")
        paste_end = source.index("/**\n     * 添加回调函数", paste_start)
        paste = source[paste_start:paste_end]

        self.assertIn("canvas.addEventListener('contextmenu', (e) => this.onContextMenu(e))", source)
        self.assertIn("label: '复制此框'", method)
        self.assertIn("粘贴为${this.annotationClipboard.displayName}的续框", method)
        self.assertIn("copyAnnotationForContinuation(annotation)", source)
        self.assertIn("attributes.paperflowRegionRole = 'continuation';", paste)
        self.assertIn("attributes.paperflowContinuationOf = clipboard.sourceId;", paste)
        self.assertIn("this.resetCopiedPaperFlowContent(attributes);", paste)
        self.assertIn("this.createAnnotation(clipboard.type", paste)

    def test_right_button_pan_does_not_open_the_annotation_copy_menu(self):
        source = (APP_JS / "annotation-manager.js").read_text(encoding="utf-8")
        image_manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        context_start = source.index("    onContextMenu(event) {")
        context_end = source.index("    openAnnotationContextMenu", context_start)
        context = source[context_start:context_end]
        mouse_down_start = source.index("    onMouseDown(e) {")
        mouse_down_end = source.index("    onMouseMove(e) {", mouse_down_start)
        mouse_down = source[mouse_down_start:mouse_down_end]
        self.assertIn("this.rightPointerGesture", source)
        self.assertIn("const wasRightDrag = Boolean(this.rightPointerGesture?.moved);", context)
        self.assertIn("if (wasRightDrag)", context)
        self.assertIn("e.button === 2", mouse_down)
        self.assertIn("Math.hypot(deltaX, deltaY) >= 50", source)
        self.assertIn("this.rightPanGesture", image_manager)
        self.assertIn("this.rightPanThreshold = 50", image_manager)
        self.assertIn("distance >= this.rightPanThreshold", image_manager)
        self.assertIn("this.rightPanGesture.active = true;", image_manager)
        self.assertNotIn("document.addEventListener('scroll', () => this.closeAnnotationContextMenu(), true);", source)
        self.assertIn("event.button === 0", source)

    def test_canvas_copy_can_paste_the_next_hierarchy_number_without_parentheses(self):
        annotation_manager = (APP_JS / "annotation-manager.js").read_text(encoding="utf-8")
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")

        self.assertIn("粘贴为下一个框（${nextPath.join('-')}）", annotation_manager)
        self.assertIn("nextPaperFlowClipboardNumberPath()", annotation_manager)
        self.assertIn("incrementPaperFlowNumberToken(value)", annotation_manager)
        self.assertIn("applyNextPaperFlowNumber(attributes, nextPath);", annotation_manager)
        self.assertIn("paperflowRegionRole = 'question'", annotation_manager)
        self.assertIn("return parts.join('-') || '?';", ui)
        self.assertIn("if (path.length) return path.map(value => this.paperFlowCanonicalNumberPart(value)).join('-');", main)

    def test_latex_preview_treats_ocr_content_as_text(self):
        source = (APP_JS / "latex-renderer.js").read_text(encoding="utf-8")
        start = source.index("render(content, container)")
        end = source.index("    preprocessContent(content)", start + 30)
        method = source[start:end]

        self.assertIn("container.replaceChildren()", method)
        self.assertIn("document.createTextNode(line)", method)
        self.assertNotIn("container.innerHTML", method)

        ui_source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        preview_start = ui_source.index("updatePreview(content, previewArea)")
        preview = ui_source[preview_start:
                            ui_source.index("    processContentWithLineBreaks(content)", preview_start)]
        self.assertIn("previewContent.textContent = safeContent", preview)
        self.assertNotIn("previewContent.innerHTML = processedContent", preview)

    def test_electron_renderer_is_sandboxed_and_blocks_untrusted_navigation(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        self.assertIn("sandbox: true", electron)
        self.assertIn("webContents.on('will-navigate'", electron)
        self.assertIn("setWindowOpenHandler", electron)
        self.assertIn("只允许打开 HTTP(S) 链接", electron)

    def test_large_answer_pages_render_cards_in_bounded_batches(self):
        source = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertIn("this.cardBatchSize = 40", source)
        self.assertIn("annotations.slice(0, Math.max(this.cardBatchSize, this.visibleCardCount))", source)
        self.assertIn("data-load-more-cards", source)
        self.assertIn("visibleAnnotations.forEach", source)

    def test_card_order_prefers_unit_and_question_hierarchy_when_most_numbers_are_known(self):
        source = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertIn("paperFlowOrdinal(value)", source)
        self.assertIn("paperFlowHierarchySortPath(annotation)", source)
        self.assertIn("sortAnnotationsByHierarchyWhenAvailable(readingOrder)", source)
        self.assertIn("[record.major_number, record.subsection_number, record.minor_number]", source)
        self.assertIn("knownCount < Math.max(2, Math.ceil(items.length / 2))", source)
        self.assertIn("return this.sortAnnotationsByHierarchyWhenAvailable(readingOrder);", source)

    def test_canvas_and_card_selection_are_linked_and_selected_card_is_materialized(self):
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        image_manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")

        selection_start = workbench.index("    selectAnnotation(id) {")
        selection_end = workbench.index("    refreshCardMeta", selection_start)
        selection = workbench[selection_start:selection_end]
        self.assertIn("this.tool.annotationManager.selectAnnotation(annotation);", selection)
        self.assertIn("this.tool.imageManager?.focusAnnotation(annotation);", selection)
        self.assertIn("ensureSelectedCardVisible()", selection)
        self.assertIn("this.visibleCardCount = Math.ceil((index + 1) / this.cardBatchSize) * this.cardBatchSize;", selection)
        self.assertIn("this.renderCards();", selection)
        self.assertIn("focusAnnotation(annotation)", image_manager)
        self.assertIn("container.scrollTo({ left: targetLeft, top: targetTop, behavior: 'smooth' });", image_manager)
        self.assertIn("this.employeeWorkbench?.updateSelection(annotation);", main)

    def test_top_bar_delete_skips_confirmation_but_other_delete_paths_keep_it(self):
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        start = source.index("    deleteAnnotation(requireConfirmation = true) {")
        end = source.index("    /**", start + 5)
        method = source[start:end]
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")

        self.assertIn("if (requireConfirmation)", method)
        self.assertIn("Utils.showConfirm(", method)
        self.assertIn("'确定要删除这个框吗？'", method)
        self.assertIn("annotationDelete", method)
        self.assertIn("this.tool.uiManager.deleteAnnotation(false);", workbench)

    def test_workspace_has_a_non_broken_empty_image_state(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        self.assertIn('id="imageEmptyState"', html)
        self.assertIn('id="currentImage" alt="" hidden', html)
        self.assertIn("this.emptyStateElement.hidden = hasImages", manager)

        update_start = manager.index("    updateUI() {")
        update_end = manager.index("\n    getCurrentImage()", update_start)
        update_method = manager[update_start:update_end]
        self.assertNotIn("navigationGeneration", update_method)

    def test_dense_answer_prelabels_surface_backend_review_warnings(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        method = source[source.index("async generatePaperFlowPrelabels()"):
                        source.index("applyPaperFlowPrelabels(image, records)")]
        self.assertIn("payload.diagnostics?.requires_review", method)
        self.assertIn("payload.diagnostics.warnings", method)
        self.assertIn("本页预标注需重点复核", method)

    def test_detector_boundary_provenance_survives_frontend_roundtrip(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("paperflowBoundarySource: String(record.boundary_source", source)
        self.assertIn("paperflowDetectorAssisted: Boolean(record.detector_assisted)", source)
        self.assertIn("boundary_source: annotation.attributes?.paperflowBoundarySource", source)
        self.assertIn("detector_confidence: Number(annotation.attributes?.paperflowDetectorConfidence", source)

    def test_empty_question_page_cannot_be_confirmed_accidentally(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        method = source[source.index("async savePaperFlowPage("):
                        source.index("paperFlowAcceptedQuestions()")]
        self.assertIn("image.paperFlowKind !== 'answer' && records.length === 0", method)
        self.assertIn("本页没有题框，不能直接确认", method)

    def test_answer_page_confirmation_allows_blank_answer_text_and_surfaces_server_warning(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        method = source[source.index("async savePaperFlowPage("):
                        source.index("paperFlowAcceptedQuestions()")]
        self.assertNotIn("if (accepted && image.paperFlowKind === 'answer')", method)
        self.assertIn("const warningText = (result.warnings || []).join", method)
        self.assertIn("warningText ? 'warning' : 'success'", method)

    def test_editor_clicks_do_not_reselect_and_rebuild_the_card(self):
        source = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        guard = "if (event.target.closest('.employee-card-editor')) return;"
        select = "this.selectAnnotation(id);"

        self.assertIn(guard, source)
        self.assertLess(source.index(guard), source.index(select))

    def test_live_form_updates_do_not_rebuild_employee_cards(self):
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        start = source.index("updateAnnotationInMemory(annotation)")
        end = source.index("createMissingAnswerAreas", start)
        method = source[start:end]

        self.assertIn("this.updateQuestionNavigation(annotations);", method)
        self.assertNotIn("window.ocrTool.updateAnnotationLists();", method)

    def test_silent_autosave_preserves_the_live_editor(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async savePaperFlowPage(")
        end = source.index("paperFlowAcceptedQuestions()", start)
        method = source[start:end]

        self.assertIn("if (!silent && stillOnSavedPage) this.employeeWorkbench?.render();", method)
        self.assertNotIn("\n            this.employeeWorkbench?.render();", method)

    def test_continuation_candidates_only_use_current_and_previous_page(self):
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        self.assertIn("const firstCandidatePageIndex = Math.max(0, currentPageIndex - 1);", source)
        self.assertIn(
            "if (pageIndex < firstCandidatePageIndex || pageIndex > currentPageIndex) continue;",
            source,
        )
        self.assertIn("当前页（未保存）", source)
        self.assertIn("continuationCandidateIds", source)

    def test_book_level_metadata_is_not_repeated_in_question_editor(self):
        ui_source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        main_source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertNotIn('id="editLanguage"', ui_source)
        self.assertNotIn('id="editStage"', ui_source)
        self.assertNotIn('id="editGrade"', ui_source)
        self.assertNotIn('id="editSubject"', ui_source)
        self.assertIn("async editPaperFlowBookMetadata()", main_source)
        self.assertIn("async editPaperFlowBookMetadata()", main_source)

    def test_book_metadata_button_is_next_to_open_book(self):
        # 「教材信息」现在收在「教材设置」下拉里，与「打开教材」同属教材组，
        # 但仍在第一个操作组内，不会跑到页面工具或交付组里去。
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        open_button = html.index('id="openWorkspace"')
        metadata_button = html.index('id="editBookMetadata"')
        draw_button = html.index('id="generatePrelabels"')
        self.assertLess(open_button, metadata_button)
        self.assertLess(metadata_button, draw_button)
        book_group = html.index('class="pf-action-group" aria-label="教材"')
        self.assertLess(book_group, metadata_button)

    def test_help_menu_opens_the_complete_operation_guide(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")

        self.assertIn("教材信息", html)
        self.assertIn("label: '操作说明'", electron)
        self.assertIn("menu-show-operation-help", electron)
        self.assertIn("onMenuShowOperationHelp", preload)
        self.assertIn("showPaperFlowOperationHelp()", main)
        self.assertIn("window.ocrTool?.showPaperFlowOperationHelp?.()", (APP_JS / "ui-manager.js").read_text(encoding="utf-8"))
        self.assertIn('style="width:min(900px,96vw)', main)
        self.assertIn("onlyPendingFrames", main)
        self.assertIn("data-operation-help-close", main)

    def test_question_options_are_derived_only_during_formal_export(self):
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        delivery = (ROOT / "review_tools" / "delivery_workspace.py").read_text(encoding="utf-8")
        self.assertNotIn('id="questionOptionsField"', source)
        self.assertNotIn('id="editOptionsText"', source)
        self.assertNotIn("extractPaperFlowOptions", main)
        self.assertIn("options: []", main)
        self.assertIn("def _split_ocr_choices", delivery)

    def test_print_handwriting_attribute_is_removed_globally(self):
        targets = [
            APP_JS / "annotation-manager.js",
            APP_JS / "ui-manager.js",
            APP_JS / "data-manager.js",
            APP_JS / "quality-check.js",
            APP_JS.parent / "css" / "components.css",
        ]
        forbidden = ["printWriteAttribute", "印刷手写属性", "题干印刷手写属性",
                     "答题区域印刷手写属性", "print-attribute"]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in targets)
        for value in forbidden:
            self.assertNotIn(value, combined)

    def test_confirm_page_flushes_editor_before_persisting(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async savePaperFlowPage(")
        end = source.index("paperFlowAcceptedQuestions()", start)
        method = source[start:end]
        self.assertIn("if (accepted)", method)
        self.assertIn("this.commitCurrentPaperFlowPageToMemory", method)
        self.assertIn("page.status = result.status || page.status;", method)
        self.assertIn("page.status = previousStatus;", method)

    def test_exit_flushes_current_page_before_closing(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async savePaperFlowBeforeExit()")
        end = source.index("async paperFlowRequest", start)
        method = source[start:end]
        self.assertIn("this.uiManager.updateAnnotationInMemory(selected);", method)
        self.assertIn("saveProjectCacheOnExit", method)
        self.assertIn("startsWith('server://') ? 60000 : 3000", method)
        self.assertIn("startsWith('server://')) throw serviceSaveError", method)
        self.assertIn("project: this.paperFlowProject", method)
        self.assertLess(
            method.index("paperFlowRequest('/api/project/save-cache'"),
            method.index("saveProjectCacheOnExit(payload)"),
        )
        self.assertNotIn("cancelAppClose", method)
        self.assertIn("confirmAppClose", method)

    def test_main_process_has_a_bounded_close_fallback(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        self.assertIn("this.closeSaveTimeoutMs = 5000;", electron)
        self.assertIn("this.finishAppClose('save-timeout');", electron)
        self.assertIn("save-project-cache-on-exit", electron)
        self.assertIn("saveProjectCacheOnExit", preload)

    def test_long_image_name_is_ellipsized_without_covering_toolbar_controls(self):
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        self.assertIn(".toolbar-center { min-width: 0; overflow: hidden; }", css)
        self.assertIn(".toolbar-center #currentImageName", css)
        self.assertIn("text-overflow: ellipsis", css)

    def test_server_typing_saves_automatically_while_legacy_local_typing_stays_in_memory(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        # 锚定方法定义本身；调用点（如配对画框）会出现在定义之前。
        start = source.index("schedulePaperFlowSave() {")
        end = source.index("commitCurrentPaperFlowPageToMemory", start)
        method = source[start:end]
        self.assertIn("退出时保存", method)
        self.assertIn("this.isPaperFlowServerProject()", method)
        self.assertIn("this.canSyncPaperFlowServer()", method)
        self.assertIn("this.persistCurrentPaperFlowCache()", method)
        self.assertIn("setTimeout", method)
        self.assertNotIn("savePaperFlowPage", method)

    def test_page_navigation_only_commits_to_memory(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("this.commitCurrentPaperFlowPageToMemory();", source)

    def test_file_menu_can_show_current_delivery_location(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("查看当前输出位置", electron)
        self.assertIn("/api/project/delivery-location", source)

    def test_native_menus_only_keep_staff_workflow_actions(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("label: '打开教材…'", electron)
        # 最终 JSON 统一由服务器在逐页审核通过后生成，客户端不再保留导出入口。
        self.assertNotIn("本地导出", electron)
        # 「页面」菜单已删除，旋转并入「视图」；「模式」与文件/视图并列。
        self.assertNotIn("label: '页面'", electron)
        self.assertIn("label: '模式'", electron)
        self.assertIn("顺时针旋转当前页面 90°", electron)
        self.assertIn("还原当前页大小和位置", electron)
        self.assertNotIn("label: '编辑'", electron)
        self.assertNotIn("label: '工具'", electron)
        for role in ("role: 'undo'", "role: 'redo'", "role: 'cut'", "role: 'copy'", "role: 'paste'", "role: 'selectall'", "role: 'zoomIn'", "role: 'zoomOut'", "role: 'resetZoom'"):
            self.assertNotIn(role, electron)
        self.assertNotIn("打开题库管理", electron)
        self.assertIn("onMenuResetPageZoom", preload)
        self.assertIn("onMenuResetPageZoom?.(() =>", main)
        self.assertIn("this.imageManager.resetZoom();", main)

    def test_answer_workflow_has_page_classification_and_dedicated_fields(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        self.assertIn('id="managePaperFlowPages"', html)
        self.assertIn("openPaperFlowPageManager()", main)
        self.assertIn("isPaperFlowAnswer", ui)
        self.assertIn("答案页 OCR 原文", ui)

    def test_answer_editor_uses_one_ocr_field_and_keeps_continuation_at_bottom(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        start = ui.index("updateSelectedAnnotationInfo(annotation)")
        end = ui.index("setupAutoSaveListeners(annotation)", start)
        method = ui[start:end]

        self.assertNotIn("<label>栏目标题：</label>", method)
        self.assertIn("${isPaperFlowAnswer ? `\n                    <div class=\"form-group\">", method)
        self.assertIn("${!isPaperFlowAnswer && currentMaterialType === 'answer' ?", method)
        self.assertLess(method.index("'答案页 OCR 原文'"), method.index("${!isPaperFlowAnswer ? `<div class=\"form-group\">"))
        continuation_index = method.rindex("续接到上一答案：")
        ocr_index = method.index("'答案页 OCR 原文'")
        self.assertGreater(continuation_index, ocr_index)

    def test_question_continuation_control_is_also_at_the_form_bottom(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        start = ui.index("updateSelectedAnnotationInfo(annotation)")
        end = ui.index("setupAutoSaveListeners(annotation)", start)
        method = ui[start:end]

        self.assertIn("${isPaperFlow ? `", method)
        self.assertGreater(method.rindex("续接到上一题："), method.index("'题干内容'"))

    def test_wheel_zoom_reveals_a_restore_page_size_button(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        image_manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")

        self.assertIn('id="restorePageZoom"', html)
        self.assertIn("onWheelZoomChange: []", image_manager)
        self.assertIn("this.trigger('onWheelZoomChange', this.zoomLevel);", image_manager)
        self.assertIn("this.imageManager.on('onWheelZoomChange'", main)
        self.assertIn("this.setRestorePageZoomButtonVisible(true);", main)
        self.assertIn("this.imageManager.resetZoom();", main)
        self.assertIn("setRestorePageZoomButtonVisible(visible)", main)

    def test_simple_middle_click_restores_zoom_while_middle_drag_keeps_panning(self):
        image_manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        self.assertIn("this.middlePointerGesture", image_manager)
        self.assertIn("e.button === 1", image_manager)
        self.assertIn("Math.hypot(", image_manager)
        self.assertIn("if (distance >= 6) this.middlePointerGesture.moved = true;", image_manager)
        self.assertIn("if (middleClick && this.zoomLevel !== 1) this.resetZoom();", image_manager)
        self.assertIn("this.canvasElement.addEventListener('auxclick'", image_manager)

    def test_reset_zoom_applies_final_view_before_annotation_redraw(self):
        image_manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        start = image_manager.index("resetZoom()")
        end = image_manager.index("/**\n     * 设置缩放级别", start)
        method = image_manager[start:end]

        self.assertNotIn("this.setZoom(1);", method)
        self.assertNotIn("this.resetPan();", method)
        self.assertIn("this.zoomLevel = 1;", method)
        self.assertIn("this.panX = 0;", method)
        self.assertIn("this.panY = 0;", method)
        self.assertLess(method.index("this.applyTransform();"), method.index("this.trigger('onZoomChange'"))

    def test_question_basics_and_coordinates_are_visually_deemphasized(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")

        self.assertIn('class="paperflow-question-basics"', ui)
        self.assertIn("employee-card-coordinates", workbench)
        self.assertIn("label: `[${number}] ${materialType}`", workbench)
        self.assertIn(".paperflow-question-basics { display: grid", css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", css)
        self.assertIn(".employee-card-coordinates", css)
        self.assertIn("opacity: .48", css)

    def test_answer_pages_can_be_found_and_classified_manually(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        image_manager = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        workbench_css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        self.assertIn('id="markCurrentPaperFlowPage"', html)
        self.assertIn("async moveCurrentPaperFlowPage(targetKind)", main)
        self.assertIn("async startPaperFlowManualAnswerSearch()", main)
        self.assertIn("请继续浏览教材，找到答案后点击", main)
        self.assertIn("async loadImageObjects(imageObjects, initialIndex = 0)", image_manager)
        self.assertIn("[hidden] { display: none !important; }", workbench_css)

    def test_page_category_apply_runs_without_a_second_confirmation(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("body.querySelector('#pfMovePages')?.addEventListener")
        apply_handler = source[start:start + 2200]
        self.assertNotIn("window.confirm", apply_handler)
        self.assertIn("/api/project/reclassify-pages", apply_handler)

    def test_matching_modal_has_reliable_exit_controls(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn('id="pfCloseMatching"', source)
        self.assertIn('id="pfFindAnswerPages"', source)
        self.assertIn("event.key === 'Escape'", source)
        self.assertIn("closePaperFlowModal()", source)

    def test_matching_has_one_primary_entry_and_nonduplicated_number_labels(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertNotIn('id="openPaperFlowMatching"', html)
        self.assertEqual(html.count('data-workflow="matching"'), 1)
        self.assertIn("paperFlowHeadingWithoutNumber(record)", source)
        self.assertIn("paperFlowMatchRecordLabel(item, answer = false)", source)
        self.assertIn("withoutPrintedNumber", source)

    def test_matching_uses_searchable_cards_and_ignores_stale_candidate_requests(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        self.assertIn('id="pfMatchQuestionList"', source)
        self.assertIn('id="pfMatchSearch"', source)
        self.assertIn("paperFlowMatchingRequestToken", source)
        self.assertIn("requestToken !== this.paperFlowMatchingRequestToken", source)
        # 页码按「所在页签内的序号」算，不再用合并排序的全书序号（见 test_annotation_page_numbers）
        self.assertIn("paperFlowSourcePageNumber(page, pageIndex, kind = this.paperFlowKind)", source)
        self.assertIn("paperflow-answer-candidate", css)

    def test_matching_filters_unfinished_work_and_supports_safe_batch_accept(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn('id="pfMatchFilter"', source)
        self.assertIn("paperFlowMatchStats", source)
        self.assertIn("/api/project/accept-confident-matches", source)

    def test_answer_ocr_is_conservatively_split_without_overwriting_reviewed_text(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("splitPaperFlowAnswerText(value)", source)
        self.assertIn("parts.answer", source)
        self.assertIn("parts.answer", source)
        self.assertIn("paperflowText", source)

    def test_ocr_migration_clears_legacy_local_default_but_preserves_new_user_choices(self):
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertIn("vision-free-v5", workbench)
        self.assertIn("paperflowOcrModelUserSelected", workbench)
        self.assertIn("? saved\n            : 'qwen3.8-flash'", workbench)

    def test_ocr_model_selectors_only_offer_cloud_models_and_one_local_emergency_fallback(self):
        service = (APP_JS / "ocr-service.js").read_text(encoding="utf-8")
        settings = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        for source in (service, settings):
            for model in ('qwen3.8-flash', 'qwen3.8-max-0902', 'qwen3.8-omni-flash'):
                self.assertIn(model, source)
            self.assertNotIn('qwen3-vl-plus', source)
            self.assertNotIn('qwen-vl-ocr', source)
            self.assertIn("local-rapidocr", source)
            self.assertNotIn("local-pix2text", source)
            self.assertNotIn("doubao-seed", source)
            # Removed legacy providers must not return through either selector.
            self.assertNotIn("qwen3-vl-flash", source)

    def test_preview_uses_the_offline_wysiwyg_formula_renderer(self):
        renderer = (APP_JS / "latex-renderer.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "components.css").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertIn("this.renderWysiwyg(safeContent, container);", renderer)
        self.assertIn("command === 'frac'", renderer)
        self.assertIn("command === 'sqrt'", renderer)
        self.assertIn("paperflow-fraction", css)
        self.assertIn("paperflow-radicand", css)
        self.assertIn("this.showRenderedOcrPreview();", main)
        self.assertNotIn('data-ocr-summary=', workbench)
        self.assertNotIn("window.latexRenderer.render(text, summary)", workbench)
        preview_start = main.index("showRenderedOcrPreview() {")
        preview_end = main.index("    /**", preview_start + 20)
        preview_method = main[preview_start:preview_end]
        self.assertIn("header.textContent = '识别结果（可直接修改）'", preview_method)
        self.assertNotIn("editor-buttons')?.replaceChildren", preview_method)
        self.assertIn("this.uiManager.enableRenderedOcrEditing(preview, textarea);", preview_method)
        self.assertIn("textarea.style.display = 'none'", preview_method)
        self.assertNotIn("编辑源码", preview_method)

    def test_formula_conversion_failure_is_visible_and_ocr_source_remains_editable(self):
        renderer = (APP_JS / "latex-renderer.js").read_text(encoding="utf-8")
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "components.css").read_text(encoding="utf-8")
        self.assertIn("validateFormula(source)", renderer)
        self.assertIn("appendFailure(parent, rawSource, reason)", renderer)
        self.assertIn("paperflow-math-conversion-failed", renderer)
        self.assertIn("ocr-formula-conversion-notice", ui)
        self.assertIn("原字符已标红", ui)
        self.assertIn(".paperflow-math-conversion-failed", css)

    def test_converted_ocr_result_is_editable_and_raw_source_is_explicitly_toggled(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        renderer = (APP_JS / "latex-renderer.js").read_text(encoding="utf-8")
        self.assertIn("toggleOcrRawSource(event)", ui)
        self.assertIn("查看原文", ui)
        self.assertIn("serializeRenderedOcrPreview(previewContent)", ui)
        self.assertIn("previewContent.contentEditable = 'true'", ui)
        self.assertIn("formula.dataset.rawSource", renderer)
        self.assertIn("formula.dataset.renderedText", renderer)

    def test_rendered_ocr_keeps_toolbar_tools_and_switches_to_source_safely(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        self.assertNotIn("ocr-final-result-toolbar .editor-buttons { display: none; }", css)
        self.assertIn("white-space: nowrap", css)
        self.assertNotIn("toolbar.querySelector('.editor-buttons')?.replaceChildren()", main)
        self.assertIn("prepareOcrSourceForToolbar(textareaId)", ui)
        self.assertIn("已切换到 OCR 原文编辑", ui)
        self.assertIn("if (this.prepareOcrSourceForToolbar(textareaId)) return;", ui)

    def test_ocr_toolbar_tools_are_left_aligned_and_collapsed_until_requested(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        self.assertIn('class="editor-inline-tools"', ui)
        self.assertIn("toggleOcrToolbarTools(event)", ui)
        self.assertIn("编辑工具", ui)
        self.assertIn("ocr-editor-toolbar .editor-buttons { display: none", css)
        self.assertIn("ocr-editor-toolbar.show-editor-tools .editor-buttons { display: flex", css)
        self.assertIn("ocr-editor-toolbar { justify-content: flex-start", css)

    def test_page_ocr_button_sits_beside_the_model_and_not_in_the_top_toolbar(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        ocr_button = html.index('id="pageOcr"')
        model_control = html.index('class="employee-ocr-control"')
        top_toolbar_end = html.index('<div class="toolbar-center">')
        self.assertGreater(ocr_button, model_control)
        self.assertGreater(ocr_button, top_toolbar_end)
        self.assertIn("grid-template-columns: auto auto minmax(96px,1fr)", css)
        self.assertIn(".employee-page-ocr { grid-row: 1;", css)
        self.assertNotIn('id="showSettings"', html)
        self.assertIn("label: '通用设置'", electron)
        self.assertIn("onMenuShowOcrSettings", preload)
        self.assertIn("onMenuShowOcrSettings?.(() => this.uiManager?.showSettings())", main)

    def test_page_confirmation_only_starts_ocr_for_pending_frames(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn('id="settingsAutoPageOcrOnConfirm" checked disabled', html)
        self.assertIn("autoPageOcrOnConfirm: autoPageOcrOnConfirm", html)
        self.assertIn("paperFlowAutoPageOcrOnConfirm()", main)
        self.assertIn("return true;", main)
        self.assertIn("paperflowOcrNeedsRefresh", main)
        self.assertIn("markPaperFlowFrameOcrDirty(annotation)", main)
        self.assertIn("await this.recognizeCurrentPage({autoAfterConfirm: true, onlyPendingFrames: true});", main)
        self.assertIn("/api/project/set-ocr-settings", main)

    def test_page_ocr_uses_bounded_cloud_parallelism_and_smaller_jpeg_uploads(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        service = (APP_JS / "ocr-service.js").read_text(encoding="utf-8")
        self.assertIn("const concurrency = provider === 'qwen' ? 2 : 1;", main)
        self.assertIn("Promise.all(Array.from({ length: Math.min(concurrency, targets.length) }, worker))", main)
        self.assertIn("toDataURL('image/jpeg', this.imageConfig.jpegQuality)", service)
        self.assertIn("toImageDataUrl(base64Image)", service)

    def test_question_image_marking_is_contextual_and_option_images_are_not_manual(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        manager = (APP_JS / "annotation-manager.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        self.assertNotIn('id="employeeDrawQuestionImage"', html)
        self.assertNotIn('id="employeeDrawOptionImage"', html)
        self.assertIn('id="drawPaperFlowQuestionImage"', ui)
        self.assertIn("setupPaperFlowQuestionImageControl(annotation)", ui)
        self.assertIn("startQuestionImageDrawing(annotation.id)", ui)
        self.assertIn("if (checkbox.checked) workbench?.startQuestionImageDrawing(annotation.id);", ui)
        self.assertIn("employeeWorkbench?.toggleQuestionImageDrawing(annotation.id);", ui)
        self.assertIn("startQuestionImageDrawing(questionId)", workbench)
        self.assertIn("toggleQuestionImageDrawing(questionId)", workbench)
        self.assertIn("stopQuestionImageDrawing(questionId = '')", workbench)
        self.assertIn("button.textContent = active ? '标注题图中' : '标注题图';", workbench)
        self.assertIn("finishQuestionImageDrawing(annotation.parentId)", main)
        self.assertNotIn("toggleImageDrawing(", workbench)
        self.assertIn("this.selectedAnnotation?.type === 'main-question'", manager)
        self.assertIn("annotation.attributes.imageKind = 'question_image';", main)

    def test_marked_question_images_are_shown_between_image_switch_and_continuation(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        css = (APP_JS.parent / "css" / "employee-workbench.css").read_text(encoding="utf-8")
        image_switch = ui.index('id="editHasImage"')
        preview = ui.index('${questionImagePreview}', image_switch)
        continuation = ui.index('id="editPaperflowContinuationOf"')
        self.assertGreater(preview, image_switch)
        self.assertLess(preview, continuation)
        self.assertIn('data-question-image-preview=', ui)
        self.assertIn('data-question-image-delete=', ui)
        self.assertIn("renderPaperFlowCrop?.(canvas, sourcePath", ui)
        self.assertIn("deletePaperFlowQuestionImage(questionId, imageId)", ui)
        self.assertIn("manager.deleteAnnotation(image.id);", ui)
        self.assertIn(".paperflow-question-image-list", css)
        self.assertIn(".btn-compact.is-drawing", css)

    def test_column_title_is_not_a_staff_editable_or_customer_delivery_field(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        delivery = (ROOT / "review_tools" / "delivery_workspace.py").read_text(encoding="utf-8")
        editor = ui[ui.index("updateSelectedAnnotationInfo(annotation)"):ui.index("setupAutoSaveListeners(annotation)")]
        self.assertNotIn('<label>栏目标题：</label>', editor)
        self.assertNotIn('id="editPaperflowMajor"', editor)
        self.assertNotIn('"major_title"', delivery[delivery.index('"metainfo": {'):delivery.index('"source": {')])

    def test_clicking_a_figure_opens_its_parent_question_card(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = main.index("    onAnnotationSelect(annotation) {")
        end = main.index("    /**", start + 20)
        method = main[start:end]
        self.assertIn("annotation?.type === 'image-area'", method)
        self.assertIn("this.annotationManager.getAnnotationById(annotation.parentId)", method)
        self.assertIn("this.uiManager.updateSelectedAnnotationInfo(cardAnnotation || annotation)", method)
        self.assertIn("this.employeeWorkbench?.updateSelection(cardAnnotation || annotation)", method)

    def test_ocr_splits_multiple_choice_options_and_masks_marked_figures(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        service = (APP_JS / "ocr-service.js").read_text(encoding="utf-8")
        delivery = (ROOT / "review_tools" / "delivery_workspace.py").read_text(encoding="utf-8")
        self.assertNotIn("extractPaperFlowOptions", main)
        self.assertIn("updates.content = normalizedText;", main)
        self.assertIn("def _split_ocr_choices", delivery)
        self.assertIn("paperflowRawText", main)
        self.assertIn("excludedCoordinates", main)
        self.assertIn("cropImageFromElement(imageElement, coordinates, excludedCoordinates = [])", service)
        self.assertIn("tempCtx.fillStyle = '#ffffff';", service)
        self.assertIn("配图区域作为导出图片保留，明确不加入 OCR 队列", service)

    def test_ocr_json_wrapper_and_escaped_newlines_are_unwrapped_before_display(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        service = (APP_JS / "ocr-service.js").read_text(encoding="utf-8")
        self.assertIn("unwrapPaperFlowOcrText(value)", main)
        self.assertIn("const transcription = parsed.transcription ?? parsed.text;", main)
        self.assertIn(".replace(/\\\\n(?=$|[^A-Za-z]|[A-D](?:[.．、]|$))/g, '\\n')", main)
        self.assertIn("extractStructuredOcrText(content)", service)
        self.assertIn("for (let attempt = 0; attempt < 2 && raw; attempt += 1)", service)
        self.assertIn('直接输出转写正文，不要 JSON、字段名、代码块或任何说明。', service)
        self.assertNotIn("response_format: { type: 'json_object' }", service)
        self.assertIn('const legacyPrefix = raw.match(/^\\s*\\{\\s*"transcription"', main)
        self.assertIn('const legacyPrefix = raw.match(/^\\s*\\{\\s*"transcription"', service)

    def test_ocr_service_unwraps_valid_and_malformed_legacy_transcription_envelopes(self):
        script = r'''
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8')
    + '\nglobalThis.TestOCRService = OCRService;';
const context = {};
vm.runInNewContext(source, context);
const extract = context.TestOCRService.prototype.extractStructuredOcrText;
const cases = [
  ['{"transcription":"甲\\n乙"}', '甲\n乙'],
  [`{"transcription":"甲（用"<"连接）
乙"}`, '甲\n乙'.replace('甲', '甲（用"<"连接）')],
  [`{"transcription":"甲（用"<"连接）
乙`, '甲\n乙'.replace('甲', '甲（用"<"连接）')]
];
for (const [input, expected] of cases) {
  const actual = extract.call({}, input);
  if (actual !== expected) throw new Error(JSON.stringify({input, expected, actual}));
}
'''
        subprocess.run(
            ["node", "-e", script, str(APP_JS / "ocr-service.js")],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_ocr_text_normalisation_never_breaks_latex_n_commands(self):
        script = r'''
const fs = require('fs');
const vm = require('vm');
let source = fs.readFileSync(process.argv[1], 'utf8');
source = source.replace(
  "document.addEventListener('DOMContentLoaded', () => {\n    window.ocrTool = new OCRAnnotationTool();\n});",
  ''
) + '\nglobalThis.TestOCRAnnotationTool = OCRAnnotationTool;';
const context = { window: {}, document: { addEventListener() {} } };
vm.runInNewContext(source, context);
const normalize = context.TestOCRAnnotationTool.prototype.normalizePaperFlowOcrText;
const run = (value) => normalize.call({ unwrapPaperFlowOcrText: (text) => text }, value);
const formula = String.raw`若 $a \ne 0$，则 $\frac{|a|}{a}$ 的值是（ ）。`;
const commands = String.raw`$\neq$，$\notin$，$\nabla$，$\nu$`;
const legacyBreak = String.raw`第一行\n第二行`;
if (run(formula) !== formula) throw new Error('LaTex \\ne was modified');
if (run(commands) !== commands) throw new Error('LaTex n-command was modified');
if (run(legacyBreak) !== '第一行\n第二行') throw new Error('literal legacy newline was not decoded');
'''
        subprocess.run(
            ["node", "-e", script, str(APP_JS / "main.js")],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_saved_illustrations_restore_the_question_image_checkbox(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("hasImage: Array.isArray(record.illustrations) && record.illustrations.length > 0", main)

    def test_thin_duplicate_frame_is_removed_only_when_it_abuts_the_same_numbered_question(self):
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("pruneAccidentalDuplicatePaperFlowFrames(records)", main)
        self.assertIn("height > 44 || width < height * 10", main)
        self.assertIn("this.paperFlowRecordNumberKey(full) !== number", main)
        self.assertIn("verticalGap >= -8 && verticalGap <= 30", main)

    def test_client_no_longer_reports_a_local_delivery_path(self):
        """交付路径只在服务器侧产生；工作台不再展示本地生成结果。"""
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        api = (ROOT / "desktop_annotation" / "paperflow_annotation_api.py").read_text(encoding="utf-8")
        self.assertNotIn("result.k12_output_path", main)
        self.assertNotIn("/api/review/generate-delivery", api)
        # 服务器生成仍然走同一套交付契约。
        self.assertIn("sync_delivery_workspace(project_path", api)

    def test_ocr_result_records_engine_and_elapsed_time_without_logging_text(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("duration_ms", source)
        self.assertIn("[paperflow-ocr] finish model=", source)
        self.assertIn("chars=${characterCount}", source)

    def test_answer_mode_uses_answer_specific_labels(self):
        source = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertIn("'绘制答案框'", source)
        self.assertIn("'自动框答案'", source)
        self.assertIn("'个答案'", source)

    def test_draw_button_is_a_real_start_stop_toggle_and_clears_partial_draw_state(self):
        workbench = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        manager = (APP_JS / "annotation-manager.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")

        self.assertIn("drawButton?.addEventListener('click', () => this.toggleMainFrameDrawing());", workbench)
        self.assertIn("return this.tool.paperFlowKind === 'answer' ? '绘制答案框' : '绘制题目框';", workbench)
        self.assertIn("active ? '停止绘制' : this.mainFrameDrawLabel()", workbench)
        self.assertIn("if (manager.currentTool === 'main-question') manager.stopDrawing();", workbench)
        self.assertIn("else manager.setCurrentTool('main-question');", workbench)
        self.assertIn("stopDrawing() {", manager)
        self.assertIn("this.setCurrentTool('');", manager)
        set_tool_start = manager.index("    setCurrentTool(tool) {")
        set_tool_end = manager.index("    toggleDrawMode()", set_tool_start)
        set_tool = manager[set_tool_start:set_tool_end]
        self.assertIn("this.drawingStart = null;", set_tool)
        self.assertIn("this.drawingCurrent = null;", set_tool)
        self.assertIn("drawing_tool_active: this.annotationManager?.currentTool === 'main-question'", main)

    def test_new_manual_answer_frame_renders_its_editor_before_selection_is_attached(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("    onAnnotationCreate(annotation) {")
        end = source.index("    /**", start + 5)
        method = source[start:end]

        self.assertIn("if (this.paperFlowProject && annotation.type === 'main-question')", method)
        self.assertIn("this.employeeWorkbench?.render();", method)
        self.assertIn("this.employeeWorkbench?.updateSelection(cardAnnotation || annotation);", method)
        self.assertLess(method.index("this.employeeWorkbench?.render();"),
                        method.index("this.employeeWorkbench?.updateSelection(cardAnnotation || annotation);"))

    def test_switching_books_persists_the_current_project_before_replacement(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async openPaperFlowBook(")
        end = source.index("async choosePaperFlowDeliveryOutput()", start)
        method = source[start:end]

        self.assertIn("async persistCurrentPaperFlowCache()", source)
        self.assertIn("flushPendingPaperFlowPrelabelsToProject()", source)
        self.assertIn("await this.persistCurrentPaperFlowCache();", method)
        self.assertIn("paperFlowOpenInProgress", method)
        self.assertIn("localStorage.setItem(accountBookKey", method)
        self.assertIn("openPaperFlowBook(root, null, {restoring:true})", source)
        self.assertIn("const mainContent = document.querySelector('.main-content');", source)
        self.assertIn("mainContent.style.pointerEvents = 'none';", source)
        self.assertLess(
            method.index("await this.persistCurrentPaperFlowCache();"),
            method.index("this.paperFlowProject = result.project")
        )

    def test_first_open_remeasures_and_redraws_the_restored_first_page(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async openPaperFlowBook(")
        refresh_start = source.index("    redrawCurrentPaperFlowPageAfterLayout() {")
        open_method = source[start:refresh_start]
        refresh_end = source.index("async choosePaperFlowDeliveryOutput()", refresh_start)
        refresh = source[refresh_start:refresh_end]

        self.assertIn("this.employeeWorkbench?.render();", open_method)
        self.assertIn("this.redrawCurrentPaperFlowPageAfterLayout();", open_method)
        self.assertIn("requestAnimationFrame(() => requestAnimationFrame", refresh)
        self.assertIn("updateImageInfo", refresh)
        self.assertIn("this.annotationManager?.redraw?.();", refresh)

    def test_image_loading_awaits_the_requested_generation_and_handles_errors(self):
        source = (APP_JS / "image-manager.js").read_text(encoding="utf-8")
        load_objects = source[source.index("async loadImageObjects("):source.index("async loadCurrentImage()")]
        load_start = source.index("async loadCurrentImage()")
        load_current = source[load_start:source.index("\n    preloadNearbyImages()", load_start)]

        self.assertIn("this.imageLoadGeneration = 0;", source)
        self.assertIn("this.invalidateImageLoads();", load_objects)
        self.assertIn("const previousImages = this.images;", load_objects)
        self.assertIn("this.images = previousImages;", load_objects)
        self.assertIn("this.imageElement.src = fallbackImage.url;", load_objects)
        self.assertIn("await this.waitForImageResource", load_current)
        self.assertIn("requestToken !== this.imageLoadToken", load_current)
        self.assertIn("图片加载失败:", load_current)
        self.assertIn("this.navigationInProgress = false;", source)
        self.assertGreaterEqual(source.count("if (this.navigationInProgress) return false;"), 3)
        self.assertGreaterEqual(source.count("generation === this.imageLoadGeneration"), 3)
        self.assertIn("setNavigationInteractionLocked(true)", load_current)
        self.assertIn("setNavigationInteractionLocked(false)", load_current)
        loaded_start = source.index("    onImageLoaded(")
        loaded_method = source[loaded_start:source.index("    showLoadingState()", loaded_start)]
        self.assertIn("this.navigationInProgress = false;", loaded_method)

    def test_paperflow_kind_switch_does_not_claim_success_when_the_image_failed(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async loadPaperFlowKind(")
        end = source.index("\n    updatePaperFlowPageClassificationControls()", start)
        method = source[start:end]

        self.assertIn("const imageLoaded = await this.imageManager.loadImageObjects", method)
        self.assertIn("if (!imageLoaded)", method)
        self.assertIn("this.paperFlowKind = previousKind", method)
        self.assertIn("const previousAnnotationData = new Map(this.annotationData);", method)

    def test_prelabel_results_are_scoped_to_project_page_and_have_a_timeout(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async generatePaperFlowPrelabels()")
        end = source.index("applyPaperFlowPrelabels(", start)
        method = source[start:end]

        self.assertIn("paperFlowProjectSession", method)
        self.assertIn("paperFlowPrelabelKey(image", source)
        self.assertIn("const controller = new AbortController();", method)
        self.assertIn("setTimeout(() => controller.abort()", method)
        self.assertIn("taskIdentity.projectPath !== this.paperFlowProjectPath", method)
        self.assertIn("this.paperFlowPrelabelControllers.add(controller);", method)
        self.assertIn("controller.abort();", source[source.index("async openPaperFlowBook("):source.index("async choosePaperFlowDeliveryOutput()")])
        # 首次生成必须复用 OCR 缓存，只有本页已有框（用户确认"重新生成"）才强制重跑，
        # 否则每点一次都要重新识别整页（实测 40~110 秒）。
        self.assertIn("const hasExistingBoxes = existing.length > 0;", method)
        self.assertIn("force_refresh: hasExistingBoxes", method)
        self.assertIn("refresh_token: hasExistingBoxes ?", method)
        # 首次识别实测可达 110 秒以上，超时不得低于 5 分钟，否则会掐断已算到一半的任务。
        self.assertIn("const PRELABEL_TIMEOUT_MS = 300000;", method)
        self.assertIn("setTimeout(() => controller.abort(), PRELABEL_TIMEOUT_MS)", method)

    def test_desktop_replaces_only_an_outdated_paperflow_bridge(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        self.assertIn("PAPERFLOW_BRIDGE_REVISION", electron)
        self.assertIn("paperFlowBridgeHealth()", electron)
        self.assertIn("async stopOutdatedPaperFlowBridge()", electron)
        self.assertIn("health?.service !== 'paperflow-local'", electron)
        self.assertIn("health.bridge_revision === PAPERFLOW_BRIDGE_REVISION", electron)

    def test_final_delivery_uses_the_authenticated_private_upload_flow(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        page = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        # Company delivery is generated automatically after final approval.
        self.assertNotIn('id="uploadPaperFlowDelivery"', page)
        self.assertNotIn('id="exportPaperFlowProject"', page)
        self.assertIn('id="companyAccount"', page)
        self.assertIn("服务器自动生成最终交付", source)

    def test_review_centre_lists_delivery_jobs_and_allows_admin_retry(self):
        source = (ROOT / "online_review_beta" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("/api/deliveries/${encodeURIComponent(bookId)}", source)
        self.assertIn("重新生成交付", source)
        self.assertIn("only_reviewed", source)
        # 生成不再要求整本审完：按交付字段判定，字段齐全就能出包。
        self.assertIn("按交付字段判定", source)
        self.assertIn("只生成审核已通过的页面", source)
        self.assertNotIn("<th>交付生成</th>", source)
        self.assertIn("生成JSON", (ROOT / "online_review_beta" / "static" / "index.html").read_text(encoding="utf-8"))
        self.assertIn("适应整页", source)
        for status in ("等待生成", "正在生成", "生成成功", "生成失败"):
            self.assertIn(status, source)

    def test_failed_confirmation_restores_the_status_from_before_commit(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async savePaperFlowPage(")
        end = source.index("paperFlowAcceptedQuestions()", start)
        method = source[start:end]

        self.assertLess(
            method.index("const previousStatus"),
            method.index("this.commitCurrentPaperFlowPageToMemory")
        )
        self.assertLess(
            method.index("const wasDirty"),
            method.index("this.commitCurrentPaperFlowPageToMemory")
        )
        self.assertIn("page.status = previousStatus;", method)

        image_change = source[source.index("async onImageChange(image)"):
                              source.index("async loadAnnotationsFromWorkspace", source.index("async onImageChange(image)"))]
        self.assertIn("this.paperFlowDirty = Boolean(pending);", image_change)

    def test_production_menu_guards_reload_and_devtools_items(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        start = electron.index("    setupMenu() {")
        end = electron.index("    setupIPC()", start)
        method = electron[start:end]

        self.assertIn("const developmentViewItems = this.isDev ? [", method)
        self.assertIn("...developmentViewItems", method)
        self.assertIn("{ role: 'reload'", method)
        self.assertIn("{ role: 'forceReload'", method)

    def test_shortcuts_save_paperflow_and_never_delete_from_focused_controls(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("handleShortcut(shortcut)")
        end = source.index("handleGetChildAnnotations", start)
        method = source[start:end]

        self.assertIn("isShortcutEditingTarget", source)
        self.assertIn("shortcut.action === 'deleteSelected' && this.isShortcutEditingTarget()", method)
        self.assertIn("this.uiManager.deleteAnnotation();", method)
        self.assertIn("this.saveCurrentPaperFlowProject()", method)
        self.assertNotIn("this.annotationManager.deleteAnnotation(selected.id)", method)

    def test_file_menu_saves_the_complete_project_and_open_metadata_draft(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        start = source.index("async saveCurrentPaperFlowProject()")
        end = source.index("flushPendingPaperFlowPrelabelsToProject()", start)
        method = source[start:end]

        self.assertIn("label: '保存'", electron)
        self.assertIn("accelerator: 'CmdOrCtrl+S'", electron)
        self.assertIn("menu-save-current", electron)
        self.assertIn("onMenuSaveCurrent", preload)
        self.assertIn("onMenuSaveCurrent?.(() => this.saveCurrentPaperFlowProject())", source)
        self.assertIn("await this.persistCurrentPaperFlowCache();", method)
        self.assertIn("paperFlowBookMetadataDialog", method)
        self.assertIn("'/api/project/set-book-metadata'", method)
        self.assertIn("paperFlowManualSaveInProgress", method)

    def test_client_no_longer_generates_or_uploads_delivery(self):
        """生成 JSON 只有服务器一条路；客户端不再保留导出/上传入口。"""
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        company = (APP_JS / "company-review.js").read_text(encoding="utf-8")
        self.assertNotIn("本地导出", electron)
        self.assertNotIn("menu-generate-group-delivery", electron)
        self.assertNotIn("onMenuGenerateGroupDelivery", preload)
        self.assertNotIn("exportPaperFlowProject", source)
        self.assertNotIn("exportPaperFlowProject", company)
        # 工作台只负责提交审核快照，生成与上传都在服务器完成。
        self.assertNotIn("generate-delivery", source)
        self.assertNotIn("async generate()", company)
        self.assertNotIn('<span>生成并上传交付</span>', html)
        # 保存页仍然只落盘、不触发任何交付同步。
        self.assertIn("sync_delivery: false", source)
        self.assertIn("paperflowPageAccepted", source)
        self.assertNotIn('id="paperFlowPageKind"', html)
        self.assertIn("提交审核", source)
        # 顶部工具栏改成「教材 / 当前页 / 交付」三段分组后，跨组顺序不再固定；
        # 但仍需锁定每一组内部的先后，防止组内的操作步骤被打乱。
        groups = [
            ['id="managePaperFlowPages"', 'id="manageMatchScopes"'],
            ['id="generatePrelabels"', 'id="employeeDrawQuestion"',
             'id="employeeDeleteSelected"', 'id="confirmPaperFlowPage"'],
            ['id="submitPaperFlowReview"'],
        ]
        for group in groups:
            positions = [html.index(value) for value in group]
            self.assertEqual(positions, sorted(positions), group)
        # 教材信息 / 页面分类 / 匹配分组三个高频按钮平铺在主界面（原「教材设置」下拉已删），
        # 顺序：教材信息 → 页面分类 → 匹配分组 → 生成预标注。
        self.assertNotIn('id="pfMenuBook"', html)
        self.assertNotIn('data-pf-menu-target="pfMenuBook"', html)
        book_order = [
            'id="editBookMetadata"',
            'id="managePaperFlowPages"',
            'id="manageMatchScopes"',
            'id="generatePrelabels"',
        ]
        positions = [html.index(value) for value in book_order]
        self.assertEqual(positions, sorted(positions), book_order)
        # 远端审核动作仍收在下拉菜单里。
        self.assertIn('class="pf-menu" id="pfMenuReview"', html)
        self.assertIn('data-pf-menu-target="pfMenuReview"', html)

    def test_scope_auto_match_button_uses_the_requested_wording(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("执行一次自动的题目和答案分组匹配", source)
        self.assertNotIn(">执行安全自动匹配<", source)

    def test_number_level_ui_uses_append_semantics(self):
        """题号层级：大题号+小题号之外只在末尾追加，不再有「中间层级」。"""
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        self.assertIn("＋ 题号层级", source)
        self.assertNotIn("中间层级", source)
        self.assertIn("paperflowExtraLevels", source)
        self.assertNotIn("paperflowMiddleLevels", source)
        # 追加层从第 3 层开始编号（第 1 层大题号、第 2 层小题号）。
        self.assertIn("第${index + 3}层", source)
        # 保存时路径 = [大题号, 小题号, ...追加层]。
        self.assertIn(
            "effectiveMajorNumber, sourceQuestionNumber.value.trim(), ...extras",
            source,
        )

    def test_question_card_shows_matched_answer_text(self):
        """匹配完成后答案文字显示在题干内容下方，点击可跳到答案所在页。"""
        ui_source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        # 答案块插在题干内容 form-group 之后。
        content_group_end = ui_source.index("generatePreviewArea('previewContent')")
        insert_at = ui_source.index("${matchedAnswerBlock}", content_group_end)
        self.assertLess(content_group_end, insert_at)
        self.assertIn("buildMatchedAnswerBlock", ui_source)
        self.assertIn("pf-card-answer-jump", ui_source)
        self.assertIn("data-answer-locations", ui_source)
        # 答案块不再出现在卡片摘要区。
        workbench_source = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertNotIn("employee-card-answer", workbench_source)
        # 主程序有跳转实现：切答案页 + 选中并聚焦答案框。
        main_source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("async jumpToPaperFlowAnswer(", main_source)
        self.assertIn("loadPaperFlowKind('answer', page.image)", main_source)
        self.assertIn("selectAnnotation(first.answerId)", main_source)

    def test_matched_answer_preview_uses_delivery_cleaning_chain(self):
        """答案卡预览必须与交付同口径：answer → text → analysis，剥印刷小题号。

        之前卡片直接显示 record.answer 原文——旧数据里残留的「、1. 12.56 …」
        和裸 LaTeX（`$4\\frac{1}{2}$`）原样亮给员工，交付修好后界面也"还是没变"。
        """
        ui_source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        self.assertIn("paperFlowAnswerPreviewText(record, preferText = false)", ui_source)
        self.assertIn("part !== record", ui_source)
        # 候选顺序与交付一致：answer → text → analysis；有续框的主框改为
        # text → answer → analysis（answer 是拆分前的旧合并文本）。
        self.assertIn("preferText ? ['text', 'answer', 'analysis'] : ['answer', 'text', 'analysis']", ui_source)
        # 答案链要包含续框片段（continuation_of），与交付 answer_chain 同口径。
        self.assertIn("childrenOf", ui_source)
        self.assertIn("continuation_of", ui_source)
        # 每个候选剥印刷小题号（复用主程序的剥号逻辑）。
        self.assertIn("window.ocrTool?.stripPaperFlowPrintedNumber", ui_source)
        # analysis 里旧 OCR 残留的纯数字碎片会被丢掉。
        self.assertIn("isPaperFlowNumericFragment(stripped, firstBody)", ui_source)
        self.assertIn("isPaperFlowNumericFragment(value, body)", ui_source)
        # 含公式的答案交给渲染器，员工看渲染结果而不是 `$...$` 源码。
        self.assertIn("pf-answer-preview-text", ui_source)
        self.assertIn("window.latexRenderer.render(rawText, previewNode)", ui_source)

    def test_strip_printed_number_handles_punctuation_variants(self):
        """剥号逻辑要认「、1. 12.56 …」这类前导标点变体（与交付层一致）。"""
        main_source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("/^[、，,.\\s]*(\\d{1,3})\\s*[.、．](\\s*)(\\S[\\s\\S]*)$/", main_source)
        # 点号后有空格（印刷答案行 "3. 15立方分米"）也要剥；真小数 "3.14" 不动。
        self.assertIn("if (/^\\d/.test(rest) && !match[2] && !/[.、．]/.test(rest)) return source;", main_source)

    def test_pair_draw_mode_wiring(self):
        """配对画框：入口在顶部「模式」菜单，画题框记配对源，答案页画答案框自动配对。"""
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        # 主界面不再放按钮，只保留状态条。
        self.assertNotIn('id="employeePairDraw"', html)
        self.assertIn('id="pairDrawStatus"', html)
        # Electron：顶层「模式」菜单与文件/视图并列，页面菜单已删、旋转并入视图。
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        self.assertNotIn("label: '页面'", electron)
        self.assertIn("label: '模式'", electron)
        view_pos = electron.index("label: '视图'")
        mode_pos = electron.index("label: '模式'")
        self.assertLess(view_pos, mode_pos)
        rotate_pos = electron.index("menu-rotate-current-page")
        self.assertGreater(rotate_pos, view_pos, "旋转应并入视图菜单之后发出")
        self.assertIn("menu-toggle-pair-draw", electron)
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        self.assertIn("onMenuTogglePairDraw", preload)
        main_source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("onMenuTogglePairDraw?.(() => this.togglePaperFlowPairMode())", main_source)
        self.assertIn("async onPaperFlowPairFrameDrawn(", main_source)
        self.assertIn("async flushPaperFlowPairAnswers(", main_source)
        # 绘制与选中两个钩子都要接上。
        self.assertIn("void this.onPaperFlowPairFrameDrawn(annotation)", main_source)
        self.assertIn("setPaperFlowPairQuestion(cardAnnotation.id)", main_source)
        # 答案页点选答案框也要配对（画/点选统一走 pairPaperFlowAnswerFrame）。
        select_pos = main_source.index("onAnnotationSelect(annotation)")
        self.assertIn(
            "pairPaperFlowAnswerFrame(cardAnnotation)",
            main_source[select_pos:],
        )
        # 配对时把配对源题框的题号同步到答案框上；配对源跨页后从项目内存取题号。
        self.assertIn("paperFlowPairQuestionNumberPath", main_source)
        self.assertIn("attrs.paperflowNumberPath = questionPath.slice()", main_source)
        self.assertIn("attrs.paperflowNumberInferred = false", main_source)
        # 确认本页后自动补配挂起的答案框。
        self.assertIn("await this.flushPaperFlowPairAnswers()", main_source)
        # 配对复用既有 set-pair 端点，后端零改动。
        flush_start = main_source.index("async flushPaperFlowPairAnswers(")
        self.assertIn(
            "'/api/project/set-pair'",
            main_source[flush_start:flush_start + 3000],
        )

    def test_confirmed_match_keeps_the_stable_pair_identity_from_backend(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async confirmPaperFlowMatch(noAnswer)")
        end = source.index("async generatePaperFlowPrelabels()", start)
        method = source[start:end]

        self.assertIn("const result = await this.paperFlowRequest('/api/project/set-pair'", method)
        self.assertIn("this.paperFlowProject.pairs.push(result.pair);", method)
        self.assertNotIn("pairs.push({ question_id", method)

    def test_delivery_output_change_keeps_the_backend_revision(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async choosePaperFlowDeliveryOutput()")
        end = source.index("async showPaperFlowDeliveryOutput()", start)
        method = source[start:end]

        self.assertIn("const wasDirty = this.paperFlowDirty;", method)
        self.assertIn("result.project?.delivery_output_root", method)
        self.assertIn("result.project?.project_revision", method)
        self.assertIn("this.paperFlowDirty = wasDirty;", method)

    def test_revision_is_kept_across_reclassification_and_recovery_paths(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        move_start = source.index("async moveCurrentPaperFlowPage(")
        move_end = source.index("openPaperFlowPageManager()", move_start)
        move_method = source[move_start:move_end]
        manager_start = move_end
        manager_end = source.index("\n    paperFlowPageForImage(", manager_start)
        manager_method = source[manager_start:manager_end]
        persist_start = source.index("async persistCurrentPaperFlowCache()")
        persist_end = source.index("\n    flushPendingPaperFlowPrelabelsToProject()", persist_start)
        persist_method = source[persist_start:persist_end]

        self.assertIn("const cacheResult = await this.paperFlowRequest('/api/project/save-cache'", move_method)
        self.assertIn("cacheResult.project_revision", move_method)
        self.assertIn("const cacheResult = await this.paperFlowRequest('/api/project/save-cache'", manager_method)
        self.assertIn("cacheResult.project_revision", manager_method)
        self.assertIn("recoveryResult?.project_revision", persist_method)

    def test_metadata_flushes_live_page_before_replacing_the_project(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async editPaperFlowBookMetadata()")
        end = source.index("async loadPaperFlowKind(", start)
        method = source[start:end]
        self.assertLess(
            method.index("await this.persistCurrentPaperFlowCache();"),
            method.index("'/api/project/set-book-metadata'")
        )

    def test_bulk_match_uses_authoritative_pair_from_backend(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("document.getElementById('pfAcceptConfident').addEventListener")
        end = source.index("document.getElementById('pfConfirmMatch')", start)
        method = source[start:end]
        self.assertIn("this.paperFlowProject.pairs.push(match.pair || {", method)

    def test_precise_match_scope_workflow_is_available_in_the_employee_ui(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        annotations = (APP_JS / "annotation-manager.js").read_text(encoding="utf-8")

        self.assertIn('id="manageMatchScopes"', html)
        self.assertIn("openPaperFlowMatchScopeManager()", source)
        self.assertIn("'/api/project/upsert-match-scope'", source)
        self.assertIn("'/api/project/assign-match-scope'", source)
        self.assertIn("'/api/project/auto-match-scopes'", source)
        self.assertIn("paperflowMatchScopeId", source)
        self.assertIn("paperFlowCanonicalNumberPart", source)
        self.assertIn("'match-scope'", annotations)
        self.assertIn("onScopeRegionSelect", annotations)
        self.assertNotIn("同一套试卷", source)
        self.assertIn('placeholder="新分组名称，如：1"', source)
        self.assertIn("paperFlowMatchScopeCoverageHtml(scopeId)", source)
        self.assertIn("框选区域 ${regions.length} 处", source)
        self.assertIn("题目页", source)
        self.assertIn("答案页", source)
        self.assertIn("data-scope-remove-page", source)
        self.assertIn("'/api/project/remove-match-scope-pages'", source)
        self.assertIn("data-scope-preview-page", source)
        self.assertIn("previewPaperFlowMatchScopePage(kind, pageIndex, scopeId)", source)
        self.assertIn("confirmCurrentPaperFlowPage()", source)
        self.assertIn("pfConfirmAllScopeRegions", source)
        self.assertIn("pfConfirmAllAnswerFrames", source)
        self.assertIn("data-confirm-scope-region", source)
        self.assertIn("confirmed_match_scope_region_ids", source)
        self.assertIn("confirmed_match_record_ids", source)
        self.assertIn("clearScopeReviewOverlays", annotations)
        self.assertIn("setScopeReviewOverlays", annotations)
        self.assertIn("drawScopeReviewOverlays", annotations)
        self.assertIn("只移除分组关系，不删除教材页或框", source)
        self.assertIn("renderPageSelectionSection('question', '题目页'", source)
        self.assertIn("renderPageSelectionSection('answer', '答案页'", source)
        self.assertIn('data-scope-page-list="${sectionKind}"', source)
    def test_logged_out_session_never_attempts_repeated_server_autosaves(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("    schedulePaperFlowSave() {")
        end = source.index("    commitCurrentPaperFlowPageToMemory", start)
        method = source[start:end]

        self.assertIn("isPaperFlowServerProject()", source)
        self.assertIn("canSyncPaperFlowServer()", source)
        self.assertIn("this.isPaperFlowServerProject() && !this.canSyncPaperFlowServer()", method)
        self.assertIn("本地修改已保留 · 登录后再同步", method)
        self.assertIn("if (this.canSyncPaperFlowServer())", method)
        self.assertLess(method.index("!this.canSyncPaperFlowServer()"),
                        method.index("await this.persistCurrentPaperFlowCache()"))

    def test_selected_question_frame_can_be_numbered_with_digits_and_space(self):
        ui = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        shortcut_start = ui.index("    onKeyDown(e) {")
        shortcut_end = ui.index("    /**", shortcut_start)
        shortcut = ui[shortcut_start:shortcut_end]
        handler_start = main.index("    handlePaperFlowQuickNumberKey(key) {")
        handler_end = main.index("    handleShortcut(shortcut)", handler_start)
        handler = main[handler_start:handler_end]

        self.assertIn("'1': 'quickNumberDigit'", shortcut)
        self.assertIn("' ': 'quickNumberNextLevel'", shortcut)
        self.assertIn("quickNumberZeroOrReset", shortcut)
        self.assertIn("e.target.tagName === 'SELECT'", shortcut)
        self.assertIn("entry.level = 'minor'", handler)
        self.assertIn("entry.major += key", handler)
        self.assertIn("entry.minor += key", handler)
        self.assertIn("annotation.attributes.paperflowNumberPath = path", main)
        self.assertIn("按空格后再输入小题号，如 2，题号即为 3-2", main)
        self.assertNotIn("switchQuestionPage", main)
        self.assertNotIn("switchAnswerPage", main)
        self.assertNotIn("openAnswerMatching", main)


if __name__ == "__main__":
    unittest.main()
