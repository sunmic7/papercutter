from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "desktop_annotation" / "PaperFlow-Annotation" / "resources" / "app" / "js"


class EmployeeWorkbenchInteractionTests(unittest.TestCase):
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

    def test_deleting_the_selected_frame_requires_the_requested_confirmation(self):
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        start = source.index("    deleteAnnotation() {")
        end = source.index("    /**", start + 5)
        method = source[start:end]

        self.assertIn("Utils.showConfirm(", method)
        self.assertIn("'确定要删除这个框吗？'", method)
        self.assertIn("annotationDelete", method)

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
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        open_button = html.index('id="openWorkspace"')
        metadata_button = html.index('id="editBookMetadata"')
        draw_tools = html.index('class="employee-draw-tools"')
        self.assertLess(open_button, metadata_button)
        self.assertLess(metadata_button, draw_tools)

    def test_help_menu_opens_the_complete_operation_guide(self):
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        main = (APP_JS / "main.js").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")

        self.assertIn("设置教材信息", html)
        self.assertIn("label: '操作说明'", electron)
        self.assertIn("menu-show-operation-help", electron)
        self.assertIn("onMenuShowOperationHelp", preload)
        self.assertIn("showPaperFlowOperationHelp()", main)
        for section in ["题目页切分与复核", "答案页切分", "续框、旋转与页面分类", "保存、匹配和生成"]:
            self.assertIn(section, main)

    def test_question_options_are_only_visible_for_multiple_choice(self):
        source = (APP_JS / "ui-manager.js").read_text(encoding="utf-8")
        self.assertIn('id="questionOptionsField"', source)
        self.assertIn("questionType === '选择题' ? '' : 'display:none'", source)
        self.assertIn("this.updateQuestionTypeFields(input.value);", source)
        self.assertIn("questionType === '选择题' ? '' : 'none'", source)

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
        self.assertIn("timeoutMs: 3000", method)
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

    def test_typing_only_marks_memory_dirty_without_disk_autosave(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("schedulePaperFlowSave()")
        end = source.index("commitCurrentPaperFlowPageToMemory", start)
        method = source[start:end]
        self.assertIn("退出时保存", method)
        self.assertNotIn("setTimeout", method)
        self.assertNotIn("savePaperFlowPage", method)

    def test_page_navigation_only_commits_to_memory(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("this.commitCurrentPaperFlowPageToMemory();", source)

    def test_file_menu_can_show_current_delivery_location(self):
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("查看当前交付输出位置", electron)
        self.assertIn("/api/project/delivery-location", source)

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

        self.assertIn("${isPaperFlow && !isPaperFlowAnswer ?", method)
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
        self.assertIn("paperFlowPageOrderCache", source)
        self.assertIn("paperflow-answer-candidate", css)

    def test_matching_filters_unfinished_work_and_supports_safe_batch_accept(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn('id="pfMatchFilter"', source)
        self.assertIn("paperFlowMatchStats", source)
        self.assertIn("/api/project/accept-confident-matches", source)

    def test_answer_ocr_is_conservatively_split_without_overwriting_reviewed_text(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        self.assertIn("splitPaperFlowAnswerText(value)", source)
        self.assertIn("parts.status", source)
        self.assertIn("if (!String(annotation.attributes.answerText || '').trim()", source)
        self.assertIn("paperflowAnswerSplitStatus", source)

    def test_answer_mode_uses_answer_specific_labels(self):
        source = (APP_JS / "employee-workbench.js").read_text(encoding="utf-8")
        self.assertIn("'绘制答案框'", source)
        self.assertIn("'生成答案框'", source)
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
        self.assertIn("this.employeeWorkbench?.updateSelection(annotation);", method)
        self.assertLess(method.index("this.employeeWorkbench?.render();"),
                        method.index("this.employeeWorkbench?.updateSelection(annotation);"))

    def test_switching_books_persists_the_current_project_before_replacement(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async openPaperFlowBook()")
        end = source.index("async choosePaperFlowDeliveryOutput()", start)
        method = source[start:end]

        self.assertIn("async persistCurrentPaperFlowCache()", source)
        self.assertIn("flushPendingPaperFlowPrelabelsToProject()", source)
        self.assertIn("await this.persistCurrentPaperFlowCache();", method)
        self.assertIn("paperFlowOpenInProgress", method)
        self.assertIn("const mainContent = document.querySelector('.main-content');", source)
        self.assertIn("mainContent.style.pointerEvents = 'none';", source)
        self.assertLess(
            method.index("await this.persistCurrentPaperFlowCache();"),
            method.index("this.paperFlowProject = result.project")
        )

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
        self.assertIn("controller.abort();", source[source.index("async openPaperFlowBook()"):source.index("async choosePaperFlowDeliveryOutput()")])

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

        self.assertIn("label: '保存教材信息和标注框'", electron)
        self.assertIn("accelerator: 'CmdOrCtrl+S'", electron)
        self.assertIn("menu-save-current", electron)
        self.assertIn("onMenuSaveCurrent", preload)
        self.assertIn("onMenuSaveCurrent?.(() => this.saveCurrentPaperFlowProject())", source)
        self.assertIn("await this.persistCurrentPaperFlowCache();", method)
        self.assertIn("paperFlowBookMetadataDialog", method)
        self.assertIn("'/api/project/set-book-metadata'", method)
        self.assertIn("paperFlowManualSaveInProgress", method)

    def test_group_generation_is_an_explicit_file_menu_action(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        electron = (APP_JS.parent / "electron" / "main.js").read_text(encoding="utf-8")
        preload = (APP_JS.parent / "electron" / "preload.js").read_text(encoding="utf-8")
        self.assertIn("label: '生成裁剪图片和 JSON…'", electron)
        self.assertIn("menu-generate-group-delivery", electron)
        self.assertIn("onMenuGenerateGroupDelivery", preload)
        self.assertIn("'/api/project/group-export-preflight'", source)
        self.assertIn("'/api/project/export-match-scope'", source)
        save_start = source.index("async savePaperFlowPage(")
        save_end = source.index("async exportPaperFlowProject", save_start)
        self.assertIn("sync_delivery: false", source[save_start:save_end])
        self.assertIn("paperflowPageAccepted", source)
        self.assertNotIn('id="paperFlowPageKind"', html)
        self.assertIn('<span>生成裁剪图片和 JSON</span>', html)
        ordered_ids = [
            'id="managePaperFlowPages"', 'id="manageMatchScopes"',
            'id="exportPaperFlowProject"', 'id="employeeDrawQuestion"',
            'id="employeeDeleteSelected"',
        ]
        positions = [html.index(value) for value in ordered_ids]
        self.assertEqual(positions, sorted(positions))

    def test_confirmed_match_keeps_the_stable_pair_identity_from_backend(self):
        source = (APP_JS / "main.js").read_text(encoding="utf-8")
        start = source.index("async confirmPaperFlowMatch(noAnswer)")
        end = source.index("async exportPaperFlowProject()", start)
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
        manager_end = source.index("paperFlowPageForImage(", manager_start)
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
        self.assertIn("只移除分组关系，不删除教材页或框", source)
        self.assertIn("renderPageSelectionSection('question', '题目页'", source)
        self.assertIn("renderPageSelectionSection('answer', '答案页'", source)
        self.assertIn('data-scope-page-list="${sectionKind}"', source)


if __name__ == "__main__":
    unittest.main()
