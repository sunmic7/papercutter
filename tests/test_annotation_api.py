from __future__ import annotations

import importlib.util
import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "desktop_annotation" / "paperflow_annotation_api.py"
SPEC = importlib.util.spec_from_file_location("paperflow_annotation_api", API_PATH)
api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(api)


class AnnotationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.book = Path(self.temp.name)
        (self.book / "answers").mkdir()
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (280, 180), (0, 0, 0), 2)
        cv2.imwrite(str(self.book / "question.png"), image)
        cv2.imwrite(str(self.book / "answers" / "answer.png"), image)
        opened = api._open_book({"book_root": str(self.book)})
        self.project_path = Path(opened["project_path"])
        answer_index = next(
            index for index, page in enumerate(opened["project"]["pages"])
            if page["image"].replace("\\", "/") == "answers/answer.png"
        )
        api._reclassify_pages({
            "project_path": str(self.project_path), "source_kind": "question",
            "page_indices": [answer_index], "target_kind": "answer",
        })
        self.export_root = ROOT / "runtime" / "manual_annotation_exports" / self.project_path.stem
        opened_project = api.annotation_data.load_project(self.project_path)
        self.delivery_root = ROOT / "runtime" / "delivery_packages" / opened_project["delivery_book_id"]

    def test_complete_project_cache_request_limit_is_not_64kb(self) -> None:
        self.assertGreaterEqual(api.MAX_REQUEST_BODY_BYTES, 64 * 1024 * 1024)

    def test_prelabel_rejects_an_image_that_is_not_the_requested_project_page(self) -> None:
        outside = self.book / "outside.png"
        cv2.imwrite(str(outside), np.full((20, 20, 3), 255, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "不匹配"):
            api._prelabel({
                "project_path": str(self.project_path), "kind": "question",
                "page_index": 0, "image_path": str(outside), "profile": "auto",
            })

    def test_prelabel_auto_profile_uses_the_textbook_directory(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        expected_image = api.annotation_data.source_path(project, project["pages"][0])
        with (
            mock.patch.object(api, "infer_book_profile", return_value="exam") as infer,
            mock.patch.object(api, "_question_suggestions", return_value=[]) as suggest,
        ):
            api._prelabel({
                "project_path": str(self.project_path), "kind": "question",
                "page_index": 0, "image_path": expected_image, "profile": "auto",
            })
        infer.assert_called_once_with(str(self.book.resolve()))
        self.assertEqual(suggest.call_args.args[1], "exam")

    def test_save_page_preserves_detector_provenance(self) -> None:
        api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [{
                "id": "q-detector", "bbox": [10, 10, 100, 100],
                "major_number": "一", "minor_number": "1",
                "boundary_source": "detector_assisted",
                "detector_assisted": True, "detector_confidence": 0.91,
            }],
        })
        saved = api.annotation_data.load_project(self.project_path)["pages"][0]["questions"][0]
        self.assertEqual(saved["boundary_source"], "detector_assisted")
        self.assertTrue(saved["detector_assisted"])
        self.assertAlmostEqual(saved["detector_confidence"], 0.91)

    def test_save_page_keeps_the_draw_button_state_transient(self) -> None:
        result = api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "drawing_tool_active": True,
            "records": [{"id": "q-draw-state", "bbox": [10, 10, 100, 100]}],
        })
        self.assertTrue(result["transient_drawing_tool_active"])
        saved_page = api.annotation_data.load_project(self.project_path)["pages"][0]
        self.assertNotIn("drawing_tool_active", saved_page)

    def test_answer_prelabel_auto_keeps_the_page_profile(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        answer_page = project["answer_pages"][0]
        answer_page["profile"] = "general"
        api.annotation_data.save_project(project, self.project_path)
        expected_image = api.annotation_data.source_path(project, answer_page, True)
        with (
            mock.patch.object(api, "infer_book_profile", return_value="exam") as infer,
            mock.patch.object(api, "_answer_suggestions", return_value=[]) as suggest,
        ):
            api._prelabel({
                "project_path": str(self.project_path), "kind": "answer",
                "page_index": 0, "image_path": expected_image, "profile": "auto",
            })
        infer.assert_not_called()
        self.assertEqual(suggest.call_args.args[1], "general")

    def test_empty_page_rotation_is_persistent_and_returns_to_original(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        page = project["pages"][0]
        original_path = Path(api.annotation_data.original_source_path(project, page))
        original = cv2.imdecode(np.fromfile(str(original_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(original.shape[:2], (200, 300))

        rotations = []
        for expected in (90, 180, 270, 0):
            result = api._rotate_page({
                "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            })
            rotations.append(result["rotation"])
            saved = api.annotation_data.load_project(self.project_path)
            saved_page = saved["pages"][0]
            self.assertEqual(int(saved_page.get("rotation", 0)), expected)
            display_path = Path(api.annotation_data.source_path(saved, saved_page))
            if expected:
                self.assertTrue(display_path.is_file())
                displayed = cv2.imdecode(
                    np.fromfile(str(display_path), dtype=np.uint8), cv2.IMREAD_COLOR,
                )
                expected_shape = (300, 200) if expected in {90, 270} else (200, 300)
                self.assertEqual(displayed.shape[:2], expected_shape)
                self.assertEqual(Path(result["project"]["pages"][0]["source_path"]), display_path)
            else:
                self.assertEqual(display_path, original_path)
                self.assertNotIn("rotated_image", saved_page)
        self.assertEqual(rotations, [90, 180, 270, 0])
        self.assertTrue(original_path.is_file())

    def test_page_with_any_record_cannot_rotate(self) -> None:
        api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [{"id": "q-no-rotate", "bbox": [10, 10, 100, 100]}],
        })

        with self.assertRaisesRegex(ValueError, "已有标注框"):
            api._rotate_page({
                "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            })

    def test_box_created_after_rotation_uses_rotated_image_coordinates(self) -> None:
        result = api._rotate_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
        })
        self.assertEqual(result["rotation"], 90)
        saved = api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [{"id": "q-after-rotate", "bbox": [20, 30, 150, 250]}],
            "accepted": True, "sync_delivery": True,
        })
        self.assertEqual(saved["saved"], 1)
        project = api.annotation_data.load_project(self.project_path)
        page = project["pages"][0]
        displayed = cv2.imdecode(
            np.fromfile(api.annotation_data.source_path(project, page), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(displayed.shape[:2], (300, 200))
        self.assertEqual(page["questions"][0]["bbox"], [20, 30, 150, 250])
        # Page confirmation now persists review state only; delivery generation
        # is an explicit File-menu action.
        self.assertEqual(list(self.delivery_root.glob("images/questions/P*/Q*.jpg")), [])
        api._sync_delivery({"project_path": str(self.project_path)})
        delivered_crops = list(self.delivery_root.glob("images/questions/P*/Q*.jpg"))
        self.assertEqual(len(delivered_crops), 1)
        delivered = cv2.imdecode(
            np.fromfile(str(delivered_crops[0]), dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        self.assertEqual(delivered.shape[:2], (220, 130))

    def tearDown(self) -> None:
        if self.project_path.is_file():
            self.project_path.unlink()
        workspace_db = api.database_path(self.project_path, ROOT / "runtime")
        if workspace_db.is_file():
            workspace_db.unlink()
        if self.export_root.is_dir():
            shutil.rmtree(self.export_root)
        if self.delivery_root.is_dir():
            shutil.rmtree(self.delivery_root)
        self.temp.cleanup()

    def test_project_save_match_and_export_roundtrip(self) -> None:
        common = {"project_path": str(self.project_path), "page_index": 0, "accepted": True}
        api._save_page({
            **common, "kind": "question",
            "records": [{
                "id": "q-test", "bbox": [10, 10, 290, 190],
                "number": "1", "minor_number": "1", "major_number": "四",
                "major_title": "四、填空题", "subsection_number": "二",
                "subsection_title": "（二）综合练习", "region_role": "question",
                "number_path": ["四", "二", "甲", "1"],
                "parent_id": "q-parent",
                "shared_context_bbox": [10, 10, 290, 30],
                "shared_context_segments": [{"page_index": 0, "bbox": [10, 10, 290, 30]}],
                "use_shared_context": False,
            }],
        })
        api._save_page({
            **common, "kind": "answer",
            "records": [{
                "id": "a-test", "bbox": [10, 10, 290, 190],
                "number": "1", "minor_number": "1", "major_number": "四",
                "major_title": "四、填空题", "subsection_number": "二",
                "subsection_title": "（二）综合练习",
            }],
        })
        candidates = api._match_candidates({
            "project_path": str(self.project_path), "question_id": "q-test",
        })
        self.assertEqual(candidates["candidates"][0]["answer"]["id"], "a-test")
        self.assertEqual(candidates["candidates"][0]["match_reason"], "大题号一致、小题号一致")
        self.assertFalse(candidates["candidates"][0]["recommended"], "答案缺少题目的中间层级时不能自动推荐")
        self.assertEqual(candidates["question"]["display_number"], "四-二-甲-1")
        self.assertEqual(candidates["question"]["parent_id"], "q-parent")
        self.assertEqual(candidates["question"]["shared_context_segments"][0]["page_index"], 0)
        api._set_pair({
            "project_path": str(self.project_path), "question_id": "q-test", "answer_ids": ["a-test"],
        })
        result = api._export_project({"project_path": str(self.project_path)})
        self.assertEqual(result["report"]["exported"], 1)
        sample = next(Path(result["output_path"]).glob("*/question.jpg"))
        self.assertTrue(sample.is_file())
        delivery = api._sync_delivery({"project_path": str(self.project_path)})
        self.assertEqual(delivery["ready_question_count"], 1)
        self.assertTrue((Path(delivery["output_path"]) / "database_import.json").is_file())

    def test_whole_project_cache_preserves_backend_delivery_ids_and_relation(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        question = api.annotation_data.new_record(
            [10, 10, 290, 190], number="1", major_number="一",
        )
        answer = api.annotation_data.new_record(
            [10, 10, 290, 190], answer=True, number="1", major_number="一",
        )
        project["pages"][0].update(status="accepted", questions=[question])
        project["answer_pages"][0].update(status="accepted", answers=[answer])
        api.annotation_data.set_pair(project, question["id"], [answer["id"]])
        api.annotation_data.save_project(project, self.project_path)
        authoritative = api.annotation_data.load_project(self.project_path)
        expected_book_id = authoritative["delivery_book_id"]
        expected_question_id = authoritative["pages"][0]["questions"][0]["delivery_id"]
        expected_answer_id = authoritative["answer_pages"][0]["answers"][0]["delivery_id"]
        expected_pair = copy.deepcopy(authoritative["pairs"][0])

        stale_renderer = copy.deepcopy(authoritative)
        stale_renderer["pages"][0]["questions"][0].pop("delivery_id")
        stale_renderer["answer_pages"][0]["answers"][0]["delivery_id"] = "AWRONG"
        stale_renderer["delivery_book_id"] = "BWRONG"
        stale_renderer["image_root"] = str(self.book.parent / "outside")
        stale_renderer["answer_image_root"] = str(self.book.parent / "outside-answers")
        stale_renderer["pairs"] = []
        api._save_project_cache({
            "project_path": str(self.project_path), "project": stale_renderer,
        })

        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(saved["delivery_book_id"], expected_book_id)
        self.assertEqual(saved["image_root"], authoritative["image_root"])
        self.assertEqual(saved["answer_image_root"], authoritative["answer_image_root"])
        self.assertEqual(saved["pages"][0]["questions"][0]["delivery_id"], expected_question_id)
        self.assertEqual(saved["answer_pages"][0]["answers"][0]["delivery_id"], expected_answer_id)
        self.assertEqual(saved["pairs"][0], expected_pair)

    def test_stale_whole_project_cache_cannot_delete_a_newer_backend_record(self) -> None:
        stale = api.annotation_data.load_project(self.project_path)
        newer = copy.deepcopy(stale)
        new_record = api.annotation_data.new_record([10, 10, 290, 190], number="new")
        newer["pages"][0]["questions"] = [new_record]
        api.annotation_data.save_project(newer, self.project_path)

        with self.assertRaisesRegex(ValueError, "缓存已在其他操作中更新"):
            api._save_project_cache({
                "project_path": str(self.project_path), "project": stale,
                "base_revision": stale["project_revision"],
            })

        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(len(saved["pages"][0]["questions"]), 1)
        self.assertEqual(saved["pages"][0]["questions"][0]["id"], new_record["id"])

    def test_save_page_roundtrips_answer_review_status_and_cleans_dangling_pair(self) -> None:
        question = api.annotation_data.new_record([10, 10, 290, 190], number="1")
        answer = api.annotation_data.new_record([10, 10, 290, 190], answer=True, number="1")
        project = api.annotation_data.load_project(self.project_path)
        project["pages"][0].update(status="accepted", questions=[question])
        project["answer_pages"][0].update(status="accepted", answers=[answer])
        api.annotation_data.set_pair(project, question["id"], [answer["id"]])
        api.annotation_data.save_project(project, self.project_path)

        api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "records": [{
                "id": answer["id"], "bbox": [10, 10, 290, 190],
                "answer_split_status": "needs_review",
            }],
        })
        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(
            saved["answer_pages"][0]["answers"][0]["answer_split_status"],
            "needs_review",
        )

        result = api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [],
        })
        self.assertEqual(result["removed_pairs"], 1)
        self.assertEqual(api.annotation_data.load_project(self.project_path)["pairs"], [])

    def test_backend_confirms_blank_answer_boxes_with_a_follow_up_warning(self) -> None:
        result = api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "accepted": True,
            "records": [{"id": "a-blank", "bbox": [10, 10, 100, 100]}],
        })
        self.assertEqual(result["status"], "accepted")
        self.assertIn("1 个答案框尚无答案文字", "；".join(result["warnings"]))
        saved = api.annotation_data.load_project(self.project_path)["answer_pages"][0]
        self.assertEqual(saved["status"], "accepted")

    def test_backend_refuses_to_accept_an_empty_question_page(self) -> None:
        with self.assertRaisesRegex(ValueError, "题框"):
            api._save_page({
                "project_path": str(self.project_path), "kind": "question",
                "page_index": 0, "records": [], "accepted": True,
            })

    def test_invalid_record_does_not_partially_replace_the_saved_page(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        question = api.annotation_data.new_record([10, 10, 290, 190], number="1")
        project["pages"][0]["questions"] = [question]
        api.annotation_data.save_project(project, self.project_path)
        for malformed in (None, {"id": question["id"], "bbox": [0, 0, 10]}):
            with self.assertRaisesRegex(ValueError, "页面未保存"):
                api._save_page({
                    "project_path": str(self.project_path), "kind": "question",
                    "page_index": 0, "records": [malformed],
                })
            saved = api.annotation_data.load_project(self.project_path)
            self.assertEqual(saved["pages"][0]["questions"][0]["id"], question["id"])

    def test_duplicate_internal_ids_are_rejected_atomically(self) -> None:
        duplicate = {"id": "q-duplicate", "bbox": [10, 10, 100, 100]}
        with self.assertRaisesRegex(ValueError, "ID.*重复"):
            api._save_page({
                "project_path": str(self.project_path), "kind": "question", "page_index": 0,
                "records": [duplicate, {**duplicate, "bbox": [110, 10, 200, 100]}],
            })
        self.assertEqual(api.annotation_data.load_project(self.project_path)["pages"][0]["questions"], [])

    def test_parent_question_boxes_are_removed_on_open_and_save(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        project["pages"][0]["questions"] = [
            {"id": "old-parent", "bbox": [0, 0, 300, 80], "region_role": "group"},
            {"id": "real-question", "bbox": [0, 80, 300, 190], "region_role": "question", "parent_id": "old-parent"},
        ]
        api.annotation_data.save_project(project, self.project_path)
        reopened = api._open_book({"book_root": str(self.book)})["project"]
        self.assertEqual([item["id"] for item in reopened["pages"][0]["questions"]], ["real-question"])
        self.assertEqual(reopened["pages"][0]["questions"][0]["parent_id"], "")

        result = api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [{"id": "ignored-parent", "bbox": [0, 0, 300, 80], "region_role": "group"}],
        })
        self.assertEqual(result["saved"], 0)

    def test_reopen_reconciles_generated_delivery_images_without_losing_questions(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        original = api.annotation_data.new_record(
            [10, 10, 290, 190], number="1", major_number="一",
        )
        project["pages"][0]["questions"] = [original]
        generated = self.book / "B123" / "images" / "questions"
        generated.mkdir(parents=True)
        (generated.parents[1] / "manifest.json").write_text("{}", encoding="utf-8")
        cv2.imwrite(str(generated / "Q123.jpg"), np.full((20, 20, 3), 255, dtype=np.uint8))
        project["pages"].append({
            "image": "B123/images/questions/Q123.jpg",
            "status": "draft",
            "questions": [],
        })
        api.annotation_data.save_project(project, self.project_path)

        reopened = api._open_book({"book_root": str(self.book)})["project"]

        self.assertNotIn(
            "B123/images/questions/Q123.jpg",
            [page["image"].replace("\\", "/") for page in reopened["pages"]],
        )
        question_page = next(page for page in reopened["pages"] if page["image"] == "question.png")
        self.assertEqual(question_page["questions"][0]["id"], original["id"])

    def test_delivery_output_can_be_changed_for_the_current_book(self) -> None:
        output_root = self.book / "chosen_delivery_output"
        output_root.mkdir()

        result = api._set_delivery_output({
            "project_path": str(self.project_path),
            "output_root": str(output_root),
        })

        expected = output_root / result["delivery"]["book_id"]
        self.assertEqual(Path(result["output_path"]), expected)
        self.assertFalse((expected / "manifest.json").is_file(), "changing location must not eagerly export pages")
        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(Path(saved["delivery_output_root"]), output_root.resolve())

    def test_page_reclassification_and_confirmed_answer_matching(self) -> None:
        result = api._reclassify_pages({
            "project_path": str(self.project_path), "source_kind": "question",
            "page_indices": [0], "target_kind": "answer",
        })
        self.assertEqual(result["moved"], 1)
        self.assertEqual(len(result["project"]["pages"]), 0)
        self.assertEqual(len(result["project"]["answer_pages"]), 2)

        project = api.annotation_data.load_project(self.project_path)
        project["answer_pages"][0]["status"] = "draft"
        project["answer_pages"][0]["answers"] = [
            api.annotation_data.new_record([1, 1, 20, 20], answer=True, major_number="一")
        ]
        api.annotation_data.save_project(project, self.project_path)
        with self.assertRaisesRegex(ValueError, "已经确认"):
            api._set_pair({
                "project_path": str(self.project_path), "question_id": "missing",
                "answer_ids": [project["answer_pages"][0]["answers"][0]["id"]],
            })

    def test_batch_accept_only_uses_unambiguous_confirmed_answers(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        question = api.annotation_data.new_record(
            [1, 1, 20, 20], number="1", major_number="一", number_path=["一", "1"],
        )
        answer = api.annotation_data.new_record(
            [1, 1, 20, 20], answer=True, number="1", major_number="一",
            number_path=["一", "1"], text="A", analysis="解析",
        )
        answer["answer"] = "A"
        project["pages"][0].update(status="accepted", questions=[question])
        project["answer_pages"][0].update(status="accepted", answers=[answer])
        api.annotation_data.save_project(project, self.project_path)

        candidate_result = api._match_candidates({
            "project_path": str(self.project_path), "question_id": question["id"],
        })
        self.assertTrue(candidate_result["candidates"][0]["recommended"])
        self.assertEqual(candidate_result["candidates"][0]["match_reason"], "完整题号层级一致")

        result = api._accept_confident_matches({"project_path": str(self.project_path)})

        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["matches"][0]["answer_ids"], [answer["id"]])
        self.assertEqual(result["matches"][0]["pair"]["question_id"], question["id"])
        self.assertEqual(result["matches"][0]["pair"]["answer_ids"], [answer["id"]])
        self.assertTrue(result["matches"][0]["pair"].get("delivery_id"))

    def test_scope_region_assignment_is_persisted_and_inherited_on_save(self) -> None:
        api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [
                {"id": "q-inside", "bbox": [10, 10, 80, 80], "number_path": ["七", "1"]},
                {"id": "q-outside", "bbox": [150, 100, 250, 180], "number_path": ["七", "2"]},
            ],
        })
        scope = api._upsert_match_scope({
            "project_path": str(self.project_path), "name": "第一单元",
        })["scope"]
        result = api._assign_match_scope({
            "project_path": str(self.project_path), "kind": "question",
            "scope_id": scope["id"], "page_indices": [0],
            "region_bbox": [0, 0, 100, 100],
        })
        self.assertEqual(result["assigned_count"], 1)
        records = result["project"]["pages"][0]["questions"]
        self.assertEqual(records[0]["match_scope_id"], scope["id"])
        self.assertEqual(records[1]["match_scope_id"], "")

        api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [{"id": "q-new", "bbox": [20, 20, 90, 90], "number_path": ["八"]}],
        })
        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(saved["pages"][0]["questions"][0]["match_scope_id"], scope["id"])

    def test_scope_auto_match_accepts_only_unique_confirmed_canonical_numbers(self) -> None:
        scope = api._upsert_match_scope({
            "project_path": str(self.project_path), "name": "第二单元",
        })["scope"]
        api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "accepted": True,
            "records": [{
                "id": "q-scoped", "bbox": [10, 10, 100, 100],
                "number_path": ["七", "1", "（1）"], "match_scope_id": scope["id"],
            }],
        })
        api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "accepted": True,
            "records": [{
                "id": "a-scoped", "bbox": [10, 10, 100, 100],
                "number_path": ["7", "1", "1"], "match_scope_id": scope["id"],
            }],
        })
        result = api._auto_match_scopes({"project_path": str(self.project_path)})
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["pairs"][0]["source"], "scope_number_auto")
        self.assertEqual(result["pairs"][0]["answer_ids"], ["a-scoped"])

    def test_scope_auto_match_abstains_when_a_number_is_ambiguous(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        scope_id = "scope-duplicate"
        project["match_scopes"] = [{"id": scope_id, "name": "重复题号试卷"}]
        project["pages"][0].update(status="accepted", questions=[
            api.annotation_data.new_record([1, 1, 20, 20], number_path=["八", "2"], match_scope_id=scope_id),
            api.annotation_data.new_record([30, 1, 50, 20], number_path=["8", "（二）"], match_scope_id=scope_id),
        ])
        project["answer_pages"][0].update(status="accepted", answers=[
            api.annotation_data.new_record([1, 1, 20, 20], answer=True, number_path=["8", "2"], match_scope_id=scope_id),
        ])
        api.annotation_data.save_project(project, self.project_path)
        result = api._auto_match_scopes({"project_path": str(self.project_path)})
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["conflict_count"], 1)

    def test_selected_question_and_answer_pages_can_be_removed_from_a_scope(self) -> None:
        scope = api._upsert_match_scope({
            "project_path": str(self.project_path), "name": "可移除分组",
        })["scope"]
        api._save_page({
            "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            "records": [{"id": "q-remove-scope", "bbox": [10, 10, 80, 80]}],
        })
        api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "records": [{"id": "a-remove-scope", "bbox": [10, 10, 80, 80]}],
        })
        api._assign_match_scope({
            "project_path": str(self.project_path), "kind": "question",
            "scope_id": scope["id"], "page_indices": [0], "set_page_default": True,
        })
        api._assign_match_scope({
            "project_path": str(self.project_path), "kind": "answer",
            "scope_id": scope["id"], "page_indices": [0], "region_bbox": [0, 0, 100, 100],
        })

        result = api._remove_match_scope_pages({
            "project_path": str(self.project_path), "scope_id": scope["id"],
            "pages": [
                {"kind": "question", "page_index": 0},
                {"kind": "answer", "page_index": 0},
            ],
        })

        self.assertEqual(result["removed_page_count"], 2)
        self.assertEqual(result["removed_region_count"], 1)
        self.assertEqual(result["cleared_record_count"], 2)
        self.assertEqual(len(result["project"]["pages"]), 1)
        self.assertEqual(len(result["project"]["answer_pages"]), 1)
        self.assertEqual(result["project"]["pages"][0]["questions"][0]["match_scope_id"], "")
        self.assertEqual(result["project"]["answer_pages"][0]["answers"][0]["match_scope_id"], "")
        self.assertEqual(result["project"]["answer_pages"][0]["match_scope_regions"], [])


if __name__ == "__main__":
    unittest.main()
