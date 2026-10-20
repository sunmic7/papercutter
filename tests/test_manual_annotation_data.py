import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from review_tools import manual_annotation_data as data
from review_tools.export_manual_annotations import _stack_segments, export_project


class ManualAnnotationDataTests(unittest.TestCase):
    def test_unmatched_accepted_question_is_exported_as_pending_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_root = root / "questions"
            question_root.mkdir()
            image = np.full((80, 120, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            project = data.create_project(str(question_root))
            question = data.new_record([0, 0, 120, 80], number="1")
            project["pages"][0].update(status="accepted", questions=[question])
            project_path = root / "project.json"
            data.save_project(project, project_path)

            report = export_project(str(project_path), str(root / "export"))

            self.assertEqual(report["exported"], 1)
            self.assertEqual(report["samples"][0]["answer_status"], "pending_generation")
            sample = root / "export" / report["samples"][0]["path"]
            metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["answer_status"], "pending_generation")
            self.assertFalse((sample / "answer_1.jpg").exists())

        top = np.full((3, 5, 3), 40, dtype=np.uint8)
        bottom = np.full((2, 5, 3), 180, dtype=np.uint8)
        composed = _stack_segments(top, bottom)
        self.assertEqual(composed.shape, (5, 5, 3))
        self.assertTrue(np.all(composed[2] == 40))
        self.assertTrue(np.all(composed[3] == 180))

    def test_canonical_number_path_normalizes_chinese_wrappers_and_fullwidth_digits(self):
        self.assertEqual(
            data.canonical_number_path({"number_path": ["七", "1", "（1）"]}),
            ("7", "1", "1"),
        )
        self.assertEqual(
            data.canonical_number_path({"number_path": ["第八题", "（二）"]}),
            ("8", "2"),
        )
        self.assertEqual(data._canonical_number("００７"), "7")

    def test_new_projects_initialize_match_scopes_and_page_scope_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1.jpg").write_bytes(b"")
            project = data.create_book_project(str(root))
            self.assertEqual(project["match_scopes"], [])
            path = root / "project.json"
            data.save_project(project, path)
            loaded = data.load_project(path)
            self.assertEqual(loaded["pages"][0]["match_scope_id"], "")
            self.assertEqual(loaded["pages"][0]["match_scope_regions"], [])

    def test_batch_preannotation_migrates_page_scope_to_existing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1.jpg").write_bytes(b"")
            project = data.create_book_project(str(root))
            project["preannotation_mode"] = "paperflow"
            project["match_scopes"] = [{"id": "scope-1", "name": "第一单元"}]
            project["pages"][0].update({
                "match_scope_id": "scope-1",
                "questions": [{"id": "q-1", "bbox": [2, 3, 40, 50], "match_scope_id": "scope-1"}],
            })
            path = root / "PaperFlow标注项目.json"
            data.save_project(project, path)

            loaded = data.load_project(path)

            page = loaded["pages"][0]
            self.assertEqual(page["match_scope_id"], "")
            self.assertEqual(page["questions"][0]["match_scope_id"], "scope-1")
            self.assertEqual(page["match_scope_regions"], [{
                "id": "region-prelabel-q-1", "scope_id": "scope-1", "bbox": [2, 3, 40, 50],
            }])
            self.assertTrue(loaded["_paperflow_scope_migration_pending"])

    def test_new_book_keeps_all_pages_browsable_until_staff_marks_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "answers"
            answers.mkdir()
            (root / "1.jpg").write_bytes(b"")
            (answers / "2.jpg").write_bytes(b"")

            project = data.create_book_project(str(root))

            self.assertEqual(project["answer_pages"], [])
            self.assertEqual(
                [page["image"].replace("\\", "/") for page in project["pages"]],
                ["1.jpg", "answers/2.jpg"],
            )

    def test_book_scan_ignores_its_own_delivery_images_and_reconciles_old_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((30, 40, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", source)[1].tofile(str(root / "1.jpg"))
            project = data.create_book_project(str(root))
            project["pages"][0]["questions"] = [data.new_record([1, 1, 20, 20], major_number="一")]
            delivery = root / "B123"
            crop_dir = delivery / "images" / "questions"
            crop_dir.mkdir(parents=True)
            (delivery / "manifest.json").write_text("{}", encoding="utf-8")
            cv2.imencode(".jpg", source)[1].tofile(str(crop_dir / "Q123.jpg"))
            project["pages"].append({"image": "B123/images/questions/Q123.jpg", "questions": []})

            changed = data.reconcile_book_pages(project, str(root))
            self.assertTrue(changed)
            self.assertEqual([page["image"] for page in project["pages"]], ["1.jpg"])
            self.assertEqual(len(project["pages"][0]["questions"]), 1)

    def test_reconcile_preserves_manually_classified_answer_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((30, 40, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", source)[1].tofile(str(root / "1.jpg"))
            cv2.imencode(".jpg", source)[1].tofile(str(root / "2.jpg"))
            project = data.create_book_project(str(root))
            data.reclassify_pages(project, "question", [1], "answer")

            data.reconcile_book_pages(project, str(root))

            self.assertEqual([page["image"] for page in project["pages"]], ["1.jpg"])
            self.assertEqual([page["image"] for page in project["answer_pages"]], ["2.jpg"])

    def test_legitimate_images_directory_is_still_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            source = np.full((30, 40, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", source)[1].tofile(str(images / "1.jpg"))
            project = data.create_book_project(str(root))
            self.assertEqual(project["pages"][0]["image"], "images/1.jpg")

    def test_hierarchical_number_is_migrated_and_displayed(self):
        record = data.new_record([1, 2, 30, 40], number="2", major="四、按要求完成句子")
        self.assertEqual(record["major_number"], "四")
        self.assertEqual(record["minor_number"], "2")
        self.assertEqual(record["display_number"], "四-2")

    def test_three_level_number_and_unnumbered_major_are_displayed_faithfully(self):
        grouped = data.new_record(
            [1, 2, 30, 40], number="3", major="七、阅读",
            subsection_title="（二）课外阅读",
        )
        composition = data.new_record([1, 2, 30, 40], number="", major="八、习作")
        self.assertEqual(grouped["subsection_number"], "二")
        self.assertEqual(grouped["display_number"], "七-二-3")
        self.assertEqual(composition["display_number"], "八")

    def test_arbitrary_depth_number_path_migrates_legacy_fields(self):
        record = data.new_record(
            [1, 2, 30, 40], number_path=["七", "二", "甲", "1"],
        )
        self.assertEqual(record["display_number"], "七-二-甲-1")
        self.assertEqual(record["major_number"], "七")
        self.assertEqual(record["minor_number"], "1")

    def test_shared_material_is_migrated_to_group_parent(self):
        record = data.new_record(
            [1, 2, 30, 40], major_number="七", subsection_number="二",
            region_role="shared_material",
        )
        self.assertEqual(record["region_role"], "group")
        self.assertEqual(record["number_path"], ["七", "二"])

    def test_legacy_shared_material_becomes_parent_of_matching_questions(self):
        group = data.new_record(
            [0, 0, 100, 100], major_number="七", subsection_number="二",
            region_role="shared_material",
        )
        question = data.new_record(
            [100, 0, 200, 100], number="1", major_number="七", subsection_number="二",
        )
        project = {"pages": [{"questions": [group, question]}], "answer_pages": []}
        data.ensure_ids(project)
        self.assertEqual(question["parent_id"], group["id"])

    def test_support_regions_do_not_trigger_duplicate_question_warning(self):
        records = [
            data.new_record([1, 1, 10, 10], number="1", major="六、填空"),
            data.new_record(
                [1, 11, 10, 20], number="1", major="六、填空",
                region_role="continuation", continuation_of="q-previous",
            ),
            data.new_record(
                [1, 21, 10, 30], number="", major="七、阅读",
                subsection_title="（二）课外阅读", region_role="shared_material",
            ),
        ]
        self.assertEqual(data.hierarchy_warnings(records), [])

    def test_major_number_without_minor_number_is_valid_for_confirmation(self):
        record = data.new_record(
            [1, 1, 10, 10], major_number="一", number_path=["一"],
        )

        self.assertEqual(record["minor_number"], "")
        self.assertEqual(record["display_number"], "一")
        self.assertEqual(data.hierarchy_warnings([record]), [])

    def test_reclassify_pages_preserves_ignored_annotations_and_cleans_pairs(self):
        question = data.new_record([1, 1, 10, 10], number="1", major_number="一")
        answer = data.new_record([1, 1, 10, 10], answer=True, number="1", major_number="一")
        project = {
            "image_root": "C:/book", "book_root": "C:/book", "answer_image_root": "C:/book",
            "pages": [{"image": "1.jpg", "status": "accepted", "questions": [question]}],
            "answer_pages": [{"image": "answers/1.jpg", "status": "accepted", "answers": [answer]}],
            "ignored_pages": [], "pairs": [],
        }
        data.set_pair(project, question["id"], [answer["id"]])

        result = data.reclassify_pages(project, "answer", [0], "ignored")

        self.assertEqual(result["removed_pairs"], 1)
        self.assertEqual(project["answer_pages"], [])
        self.assertEqual(project["ignored_pages"][0]["answers"][0]["id"], answer["id"])
        data.reclassify_pages(project, "ignored", [0], "answer")
        self.assertEqual(project["answer_pages"][0]["answers"][0]["id"], answer["id"])

    def test_reclassify_one_part_of_a_multi_answer_pair_keeps_the_other_part(self):
        question = data.new_record([1, 1, 10, 10], number="1")
        first = data.new_record([1, 1, 10, 10], answer=True, number="1")
        second = data.new_record([1, 1, 10, 10], answer=True, number="1")
        project = {
            "image_root": "C:/book", "book_root": "C:/book", "answer_image_root": "C:/book",
            "pages": [{"image": "q.jpg", "questions": [question]}],
            "answer_pages": [
                {"image": "a1.jpg", "answers": [first]},
                {"image": "a2.jpg", "answers": [second]},
            ],
            "ignored_pages": [], "pairs": [],
        }
        data.set_pair(project, question["id"], [first["id"], second["id"]])

        result = data.reclassify_pages(project, "answer", [0], "ignored")

        self.assertEqual(result["removed_pairs"], 0)
        self.assertEqual(project["pairs"][0]["answer_ids"], [second["id"]])

    def test_reclassify_remaps_shared_context_page_indexes(self):
        question = data.new_record(
            [1, 1, 10, 10], number="1",
            shared_context_segments=[{"page_index": 1, "bbox": [0, 0, 20, 20]}],
        )
        project = {
            "image_root": "C:/book", "book_root": "C:/book",
            "pages": [
                {"image": "1.jpg", "questions": []},
                {"image": "2.jpg", "questions": []},
                {"image": "3.jpg", "questions": [question]},
            ],
            "answer_pages": [], "ignored_pages": [], "pairs": [],
        }

        result = data.reclassify_pages(project, "question", [0], "answer")

        self.assertEqual(result["removed_context_segments"], 0)
        self.assertEqual(
            project["pages"][1]["questions"][0]["shared_context_segments"][0]["page_index"],
            0,
        )

    def test_reclassify_drops_context_when_source_page_leaves_questions(self):
        question = data.new_record(
            [1, 1, 10, 10], number="1",
            shared_context_segments=[{"page_index": 0, "bbox": [0, 0, 20, 20]}],
        )
        project = {
            "image_root": "C:/book", "book_root": "C:/book",
            "pages": [
                {"image": "1.jpg", "questions": []},
                {"image": "2.jpg", "questions": [question]},
            ],
            "answer_pages": [], "ignored_pages": [], "pairs": [],
        }

        result = data.reclassify_pages(project, "question", [0], "ignored")

        self.assertEqual(result["removed_context_segments"], 1)
        self.assertEqual(project["pages"][0]["questions"][0]["shared_context_segments"], [])

    def test_candidate_filter_can_require_confirmed_answer_pages(self):
        question = data.new_record([1, 1, 10, 10], number="1", major_number="一")
        draft = data.new_record([1, 1, 10, 10], answer=True, number="1", major_number="一")
        accepted = data.new_record([1, 1, 10, 10], answer=True, number="2", major_number="一")
        project = {"answer_pages": [
            {"image": "draft.jpg", "status": "draft", "answers": [draft]},
            {"image": "accepted.jpg", "status": "accepted", "answers": [accepted]},
        ]}

        candidates = data.candidates(project, question, accepted_only=True)

        self.assertEqual([item[3]["id"] for item in candidates], [accepted["id"]])

    def test_answer_candidate_prefers_exact_major_and_minor(self):
        question = data.new_record([1, 1, 20, 20], number="1", major="四、句子练习")
        wrong_major = data.new_record(
            [1, 1, 20, 20], answer=True, number="1", major="五、句子练习", text="相同文本",
        )
        exact = data.new_record(
            [1, 1, 20, 20], answer=True, number="1", major="四、其他标题", text="",
        )
        project = {
            "answer_pages": [{"image": "a.jpg", "answers": [wrong_major, exact]}],
        }
        self.assertEqual(data.candidates(project, question)[0][3]["id"], exact["id"])

    def test_complete_number_path_is_required_for_unique_recommendation(self):
        question = data.new_record(
            [1, 1, 20, 20], number_path=["七", "二", "1"], major="七、阅读",
        )
        exact = data.new_record(
            [1, 1, 20, 20], answer=True, number_path=["七", "二", "1"], major="七、阅读",
        )
        other_group = data.new_record(
            [1, 1, 20, 20], answer=True, number_path=["七", "一", "1"], major="七、阅读",
        )
        self.assertTrue(data.number_path_matches(question, exact))
        self.assertFalse(data.number_path_matches(question, other_group))
        self.assertEqual(data.candidate_reason(question, exact), "完整题号层级一致")

    def test_answer_candidate_prefers_exact_subsection(self):
        question = data.new_record(
            [1, 1, 20, 20], number="1", major="七、阅读",
            subsection_title="（二）课外阅读",
        )
        wrong_group = data.new_record(
            [1, 1, 20, 20], answer=True, number="1", major="七、阅读",
            subsection_title="（一）课内阅读",
        )
        exact = data.new_record(
            [1, 1, 20, 20], answer=True, number="1", major="七、阅读",
            subsection_title="（二）课外阅读",
        )
        project = {"answer_pages": [{"image": "a.jpg", "answers": [wrong_group, exact]}]}
        self.assertEqual(data.candidates(project, question)[0][3]["id"], exact["id"])

    def test_duplicate_hierarchy_number_produces_warning(self):
        records = [
            data.new_record([1, 1, 10, 10], number="1", major="一、栏目"),
            data.new_record([1, 11, 10, 20], number="1", major="一、栏目"),
        ]
        self.assertIn("一-1", data.hierarchy_warnings(records)[0])

    def test_project_keeps_legacy_question_shape_and_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "question"
            root.mkdir()
            (root / "2.png").write_bytes(b"")
            (root / "1.jpg").write_bytes(b"")
            project = data.create_project(str(root))
            self.assertEqual([page["image"] for page in project["pages"]], ["1.jpg", "2.png"])
            question = data.new_record([1, 2, 30, 40], number="3")
            project["pages"][0]["questions"].append(question)
            answer = data.new_record([5, 6, 20, 30], answer=True, number="3", text="答案")
            project["answer_pages"] = [{"image": "a.png", "answers": [answer]}]
            data.set_pair(project, question["id"], [answer["id"]])
            self.assertEqual(data.paired_answer_ids(project, question["id"]), [answer["id"]])
            self.assertEqual(data.candidates(project, question)[0][3]["id"], answer["id"])

    def test_export_writes_question_answer_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); question_root, answer_root = base / "q", base / "a"
            question_root.mkdir(); answer_root.mkdir()
            cv2.imencode(".jpg", np.full((30, 40, 3), 255, dtype=np.uint8))[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", np.full((30, 40, 3), 240, dtype=np.uint8))[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            question = data.new_record([1, 1, 20, 20], number="1"); answer = data.new_record([2, 2, 22, 22], answer=True, number="1")
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            path = base / "project.json"; data.save_project(project, path)
            report = export_project(str(path), str(base / "out"))
            self.assertEqual(report["exported"], 1)
            self.assertTrue((base / "out" / report["samples"][0]["path"] / "answer_1.jpg").is_file())

    def test_export_composes_shared_instruction_for_later_question(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); question_root, answer_root = base / "q", base / "a"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((100, 60, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            question = data.new_record(
                [0, 50, 60, 80], number="2", major="一、栏目",
                shared_context_bbox=[0, 0, 60, 20], use_shared_context=True,
            )
            answer = data.new_record([0, 10, 60, 30], answer=True, number="2", major="一、栏目")
            project["pages"][0].update(status="accepted", questions=[question])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            path = base / "project.json"; data.save_project(project, path)
            report = export_project(str(path), str(base / "out"))
            output = cv2.imread(str(base / "out" / report["samples"][0]["path"] / "question.jpg"))
            self.assertEqual(output.shape[0], 50)
            self.assertEqual(report["samples"][0]["display_number"], "一-2")

    def test_export_appends_cross_page_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); question_root, answer_root = base / "q", base / "a"
            question_root.mkdir(); answer_root.mkdir()
            image = np.full((40, 60, 3), 255, dtype=np.uint8)
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "1.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(question_root / "2.jpg"))
            cv2.imencode(".jpg", image)[1].tofile(str(answer_root / "1.jpg"))
            project = data.create_project(str(question_root), str(answer_root))
            question = data.new_record([0, 0, 60, 20], number="1", major="六、填空")
            continuation = data.new_record(
                [0, 0, 60, 10], number="1", major="六、填空",
                region_role="continuation", continuation_of=question["id"],
            )
            answer = data.new_record([0, 0, 60, 20], answer=True, number="1", major="六、填空")
            project["pages"][0].update(status="accepted", questions=[question])
            project["pages"][1].update(status="accepted", questions=[continuation])
            project["answer_pages"][0].update(status="accepted", answers=[answer])
            data.set_pair(project, question["id"], [answer["id"]])
            path = base / "project.json"; data.save_project(project, path)
            report = export_project(str(path), str(base / "out"))
            output = cv2.imread(str(base / "out" / report["samples"][0]["path"] / "question.jpg"))
            self.assertEqual(output.shape[0], 30)

    def test_create_book_project_leaves_nested_answer_folder_for_manual_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "第一单元" / "题目").mkdir(parents=True); (root / "第一单元" / "答案").mkdir()
            (root / "第一单元" / "题目" / "1.jpg").write_bytes(b"")
            (root / "第一单元" / "答案" / "1.jpg").write_bytes(b"")
            project = data.create_book_project(str(root))
            self.assertEqual(len(project["pages"]), 2)
            self.assertEqual(len(project["answer_pages"]), 0)
            path = data.default_book_project_path(str(root), str(root / "runtime"))
            self.assertEqual(path, root.resolve() / "PaperFlow标注项目.json")
            legacy = data.legacy_book_project_path(str(root), str(root / "runtime"))
            self.assertEqual(legacy.parent.name, "manual_annotation_projects")

    def test_source_path_cannot_escape_the_textbook_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "book"
            root.mkdir()
            project = {"image_root": str(root), "answer_image_root": str(root)}
            with self.assertRaisesRegex(ValueError, "超出"):
                data.source_path(project, {"image": "../outside.jpg"})
            with self.assertRaisesRegex(ValueError, "相对路径"):
                data.source_path(project, {"image": str((root.parent / "outside.jpg").resolve())})
