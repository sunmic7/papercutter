from __future__ import annotations

import importlib.util
import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
# 打开教材会在后台抽样正文判定版式；单元测试不加载 OCR 模型。
os.environ["PAPERFLOW_DISABLE_PROFILE_DETECT"] = "1"
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
        self.delivery_root = self.book / opened_project["delivery_book_id"]

    def test_complete_project_cache_request_limit_is_not_64kb(self) -> None:
        self.assertGreaterEqual(api.MAX_REQUEST_BODY_BYTES, 64 * 1024 * 1024)

    def test_removed_local_formula_ocr_route_is_not_exposed(self) -> None:
        self.assertNotIn("/api/mixed-ocr", api.ROUTES)

    def test_auto_page_ocr_setting_defaults_to_enabled_and_is_persisted_per_book(self) -> None:
        opened = api._open_book({"book_root": str(self.book)})
        self.assertTrue(opened["project"]["ocr_settings"]["auto_page_ocr_on_confirm"])
        result = api._set_ocr_settings({
            "project_path": str(self.project_path),
            "auto_page_ocr_on_confirm": False,
        })
        self.assertFalse(result["ocr_settings"]["auto_page_ocr_on_confirm"])
        saved = api.annotation_data.load_project(self.project_path)
        self.assertFalse(saved["ocr_settings"]["auto_page_ocr_on_confirm"])

    def test_open_book_repairs_stale_roots_of_a_moved_book_project(self) -> None:
        """教材文件夹整体复制/搬移后，随行项目文件的根字段仍指向旧目录。

        打开教材时应以实际打开目录为准自愈，否则后续所有走 _load_project
        的接口（保存缓存、页面分类等）都会报"标注项目路径与教材目录不匹配"。
        """
        moved = Path(self.temp.name) / "教材副本"
        (moved / "answers").mkdir(parents=True)
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(moved / "question.png"), image)
        cv2.imwrite(str(moved / "answers" / "answer.png"), image)
        project = {
            "book_root": str(self.book),
            "image_root": str(self.book),
            "answer_image_root": str(self.book),
            "pages": [{"image": "question.png"}],
            "answer_pages": [{"image": "answers/answer.png"}],
        }
        moved_project_path = moved / "PaperFlow标注项目.json"
        api.annotation_data.save_project(project, moved_project_path)

        opened = api._open_book({"book_root": str(moved)})
        self.assertEqual(Path(opened["project"]["book_root"]), moved.resolve())
        self.assertEqual(Path(opened["project"]["image_root"]), moved.resolve())
        self.assertEqual(Path(opened["project"]["answer_image_root"]), moved.resolve())

        # 自愈后 _load_project 校验通过，页面分类等接口可以正常工作。
        result = api._reclassify_pages({
            "project_path": str(moved_project_path), "source_kind": "answer",
            "page_indices": [0], "target_kind": "question",
        })
        self.assertEqual(result["moved"], 1)

    def test_prelabel_rejects_an_image_that_is_not_the_requested_project_page(self) -> None:
        outside = self.book / "outside.png"
        cv2.imwrite(str(outside), np.full((20, 20, 3), 255, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "不匹配"):
            api._prelabel({
                "project_path": str(self.project_path), "kind": "question",
                "page_index": 0, "image_path": str(outside), "profile": "auto",
            })

    def test_prelabel_auto_profile_uses_the_persisted_book_profile(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        project["profile"] = "exam"
        project["profile_locked"] = True
        api.annotation_data.save_project(project, self.project_path)
        expected_image = api.annotation_data.source_path(project, project["pages"][0])
        with (
            mock.patch.object(api, "infer_book_profile", return_value="workbook") as infer,
            mock.patch.object(api, "_question_suggestions", return_value=[]) as suggest,
        ):
            result = api._prelabel({
                "project_path": str(self.project_path), "kind": "question",
                "page_index": 0, "image_path": expected_image, "profile": "auto",
            })
        # 版式已经在打开教材时落盘，预标注不再每次按目录名重猜。
        infer.assert_not_called()
        self.assertEqual(result["profile"], "exam")
        self.assertEqual(suggest.call_args.args[1], "exam")

    def test_prelabel_falls_back_to_directory_inference_without_a_stored_profile(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        project.pop("profile", None)
        project.pop("profile_locked", None)
        api.annotation_data.save_project(project, self.project_path)
        expected_image = api.annotation_data.source_path(project, project["pages"][0])
        with (
            mock.patch.object(api, "infer_book_profile", return_value="exam") as infer,
            mock.patch.object(api, "read_book_profile_cache", return_value=None),
            mock.patch.object(api, "_question_suggestions", return_value=[]) as suggest,
        ):
            api._prelabel({
                "project_path": str(self.project_path), "kind": "question",
                "page_index": 0, "image_path": expected_image, "profile": "auto",
            })
        infer.assert_called_once_with(str(self.book.resolve()))
        self.assertEqual(suggest.call_args.args[1], "exam")

    def test_open_book_persists_the_profile_and_syncs_page_profiles(self) -> None:
        book = Path(self.temp.name) / "第一单元达标检测卷"
        book.mkdir()
        # cv2.imwrite 在中文路径上会静默失败，这里直接写编码后的字节。
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        (book / "p1.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
        opened = api._open_book({"book_root": str(book)})
        project = opened["project"]
        self.assertEqual(project["profile"], "exam")
        self.assertEqual(project["profile_source"], "auto_name")
        self.assertIn("书名含", project["profile_reason"])
        self.assertEqual([page["profile"] for page in project["pages"]], ["exam"])
        self.assertIn("exam", {item["value"] for item in opened["book_profiles"]})
        # 界面选版式时要能看到该版式启用了哪些特殊处理。
        notes = opened["book_profile_notes"]
        self.assertEqual(set(notes), set(api.BOOK_PROFILE_CHOICES))
        self.assertTrue(any("28%" in note for note in notes["exam"]))
        self.assertEqual(notes["general"], [])
        # 重新打开时沿用落盘值，不会因为目录名判不出而退回 general。
        self.assertEqual(api._open_book({"book_root": str(book)})["project"]["profile"], "exam")

    def test_book_metadata_can_pin_and_release_the_layout_profile(self) -> None:
        metadata = {
            "language": "zh", "stage": "小学", "grade": "四年级",
            "subject": "数学", "answer_mode": "textbook", "profile": "drill",
        }
        result = api._set_book_metadata({
            "project_path": str(self.project_path), "metadata": metadata,
        })
        self.assertEqual(result["project"]["profile"], "drill")
        self.assertTrue(result["project"]["profile_locked"])
        self.assertEqual(
            {page["profile"] for page in result["project"]["pages"]}, {"drill"},
        )
        with self.assertRaisesRegex(ValueError, "版式"):
            api._set_book_metadata({
                "project_path": str(self.project_path),
                "metadata": {**metadata, "profile": "nonsense"},
            })
        released = api._set_book_metadata({
            "project_path": str(self.project_path),
            "metadata": {**metadata, "profile": "auto"},
        })
        self.assertNotIn("profile_locked", released["project"])
        self.assertEqual(released["project"]["profile"], "general")
        self.assertEqual(
            {page["profile"] for page in released["project"]["pages"]}, {"general"},
        )

    def test_background_profile_detection_saves_without_bumping_the_revision(self) -> None:
        before = api.annotation_data.load_project(self.project_path)["project_revision"]
        detected = {"profile": "exam", "source": "text", "reason": "正文含单元测试卷名"}
        with (
            mock.patch.object(api, "PROFILE_DETECT_DELAY_SECONDS", 0),
            mock.patch.object(api, "detect_book_profile", return_value=detected),
        ):
            api._profile_detect_job(self.book)
        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(saved["profile"], "exam")
        self.assertEqual(saved["profile_source"], "auto_text")
        # 后台推断不能推进修订号，否则前端下一次整本缓存保存会被判成冲突。
        self.assertEqual(saved["project_revision"], before)
        self.assertEqual({page["profile"] for page in saved["pages"]}, {"exam"})


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

    def test_force_refresh_prelabel_forwards_the_request_to_answer_ocr(self) -> None:
        project = api.annotation_data.load_project(self.project_path)
        answer_page = project["answer_pages"][0]
        expected_image = api.annotation_data.source_path(project, answer_page, True)
        with mock.patch.object(api, "_answer_suggestions", return_value=[]) as suggest:
            result = api._prelabel({
                "project_path": str(self.project_path), "kind": "answer",
                "page_index": 0, "image_path": expected_image, "profile": "auto",
                "force_refresh": True, "refresh_token": "manual-refresh-1",
            })

        self.assertTrue(result["force_refresh"])
        self.assertEqual(result["bridge_revision"], api.PAPERFLOW_BRIDGE_REVISION)
        self.assertTrue(suggest.call_args.kwargs["force_refresh"])

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
        self.assertTrue((expected / "PaperFlow标注项目.json").is_file(), "shared progress snapshot should be published on save")
        saved = api.annotation_data.load_project(self.project_path)
        self.assertEqual(Path(saved["delivery_output_root"]), output_root.resolve())

    def test_company_public_share_cannot_be_used_for_final_delivery(self) -> None:
        with self.assertRaisesRegex(ValueError, "不再用于最终交付"):
            api._set_delivery_output({
                "project_path": str(self.project_path),
                "output_root": r"\\192.168.1.204\PaperFlowTest\Projects",
            })

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

    def test_region_assignment_replaces_an_earlier_whole_page_scope(self) -> None:
        api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "records": [
                {"id": "a-inside", "bbox": [10, 10, 80, 80]},
                {"id": "a-outside", "bbox": [150, 10, 240, 80]},
            ],
        })
        scope = api._upsert_match_scope({
            "project_path": str(self.project_path), "name": "框选优先",
        })["scope"]
        api._assign_match_scope({
            "project_path": str(self.project_path), "kind": "answer",
            "scope_id": scope["id"], "page_indices": [0], "set_page_default": True,
        })
        result = api._assign_match_scope({
            "project_path": str(self.project_path), "kind": "answer",
            "scope_id": scope["id"], "page_indices": [0],
            "region_bbox": [0, 0, 100, 100],
        })
        page = result["project"]["answer_pages"][0]
        self.assertEqual(page["match_scope_id"], "")
        self.assertEqual(len(page["match_scope_regions"]), 1)
        self.assertEqual(page["answers"][0]["match_scope_id"], scope["id"])
        self.assertEqual(page["answers"][1]["match_scope_id"], "")

    def test_confirming_selected_answer_regions_only_exposes_those_answers_to_matching(self) -> None:
        first_scope = api._upsert_match_scope({
            "project_path": str(self.project_path), "name": "第一部分",
        })["scope"]
        second_scope = api._upsert_match_scope({
            "project_path": str(self.project_path), "name": "第二部分",
        })["scope"]
        project = api.annotation_data.load_project(self.project_path)
        question = api.annotation_data.new_record(
            [10, 10, 90, 90], number_path=["一", "1"], match_scope_id=first_scope["id"],
        )
        first_answer = api.annotation_data.new_record(
            [10, 10, 90, 90], answer=True, number_path=["一", "1"], match_scope_id=first_scope["id"],
        )
        second_answer = api.annotation_data.new_record(
            [150, 10, 230, 90], answer=True, number_path=["一", "1"], match_scope_id=second_scope["id"],
        )
        ungrouped_answer = api.annotation_data.new_record(
            [265, 10, 295, 90], answer=True, number_path=["一", "1"],
        )
        project["pages"][0].update(status="accepted", questions=[question])
        project["answer_pages"][0].update(
            status="draft", answers=[first_answer, second_answer, ungrouped_answer],
            match_scope_regions=[
                {"id": "answer-region-1", "scope_id": first_scope["id"], "bbox": [0, 0, 100, 100]},
                {"id": "answer-region-2", "scope_id": second_scope["id"], "bbox": [120, 0, 260, 100]},
            ],
        )
        api.annotation_data.save_project(project, self.project_path)

        result = api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "accepted": True, "confirmed_match_scope_region_ids": ["answer-region-1"],
            "records": [
                {"id": first_answer["id"], "bbox": [10, 10, 90, 90], "number_path": ["一", "1"]},
                {"id": second_answer["id"], "bbox": [150, 10, 230, 90], "number_path": ["一", "1"]},
                {"id": ungrouped_answer["id"], "bbox": [265, 10, 295, 90], "number_path": ["一", "1"]},
            ],
        })
        self.assertEqual(result["confirmed_match_scope_region_ids"], ["answer-region-1"])
        candidates = api._match_candidates({
            "project_path": str(self.project_path), "question_id": question["id"],
        })["candidates"]
        self.assertEqual([item["answer"]["id"] for item in candidates], [first_answer["id"]])
        with self.assertRaisesRegex(ValueError, "确认"):
            api._set_pair({
                "project_path": str(self.project_path), "question_id": question["id"],
                "answer_ids": [second_answer["id"]],
            })

        api._save_page({
            "project_path": str(self.project_path), "kind": "answer", "page_index": 0,
            "accepted": True, "confirmed_match_scope_region_ids": ["answer-region-1"],
            "confirm_all_answer_records": True,
            "records": [
                {"id": first_answer["id"], "bbox": [10, 10, 90, 90], "number_path": ["一", "1"]},
                {"id": second_answer["id"], "bbox": [150, 10, 230, 90], "number_path": ["一", "1"]},
                {"id": ungrouped_answer["id"], "bbox": [265, 10, 295, 90], "number_path": ["一", "1"]},
            ],
        })
        saved_page = api.annotation_data.load_project(self.project_path)["answer_pages"][0]
        self.assertEqual(set(saved_page["confirmed_match_record_ids"]), {
            first_answer["id"], second_answer["id"], ungrouped_answer["id"],
        })

    def test_scope_auto_match_pairs_same_scope_and_canonical_number(self) -> None:
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
                "number_path": ["七", "1", "1"], "match_scope_id": scope["id"],
            }],
        })
        result = api._auto_match_scopes({"project_path": str(self.project_path)})
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["pairs"][0]["source"], "scope_number_auto")
        self.assertEqual(result["pairs"][0]["answer_ids"], ["a-scoped"])

    def test_scope_auto_match_pairs_duplicate_numbers_to_the_same_answer(self) -> None:
        """新规则：同分组+同题号即自动配对；同号多题共用同一答案。

        （2）与 2 只是括号排版差异，仍视为同一个号；中文「八」与阿拉伯 8
        是不同编号体系，互不匹配。
        """
        project = api.annotation_data.load_project(self.project_path)
        scope_id = "scope-duplicate"
        project["match_scopes"] = [{"id": scope_id, "name": "重复题号试卷"}]
        project["pages"][0].update(status="accepted", questions=[
            api.annotation_data.new_record([1, 1, 20, 20], number_path=["八", "2"], match_scope_id=scope_id),
            api.annotation_data.new_record([30, 1, 50, 20], number_path=["八", "（2）"], match_scope_id=scope_id),
        ])
        project["answer_pages"][0].update(status="accepted", answers=[
            api.annotation_data.new_record([1, 1, 20, 20], answer=True, number_path=["八", "2"], match_scope_id=scope_id),
        ])
        api.annotation_data.save_project(project, self.project_path)
        result = api._auto_match_scopes({"project_path": str(self.project_path)})
        self.assertEqual(result["accepted_count"], 2)
        self.assertEqual(result["conflict_count"], 0)
        answer_id = project["answer_pages"][0]["answers"][0]["id"]
        paired_answers = {tuple(pair["answer_ids"]) for pair in result["pairs"]}
        self.assertEqual(paired_answers, {(answer_id,)})

    def test_scope_auto_match_accepts_a_unique_recommendation(self) -> None:
        """唯一推荐自动匹配：同号配对没命中的题，若推荐答案唯一且分差足够，直接配对。

        题目没有归入分组，同号配对不会命中；但题号层级与答案完全一致且
        第二候选分差很大，属于唯一推荐，应自动匹配，无需人工确认。
        """
        project = api.annotation_data.load_project(self.project_path)
        project["pages"][0].update(status="accepted", questions=[
            api.annotation_data.new_record([1, 1, 20, 20], number_path=["九", "3"]),
        ])
        project["answer_pages"][0].update(status="accepted", answers=[
            api.annotation_data.new_record([1, 1, 20, 20], answer=True, number_path=["九", "3"]),
            api.annotation_data.new_record([30, 1, 50, 20], answer=True, number_path=["九", "4"]),
        ])
        api.annotation_data.save_project(project, self.project_path)
        result = api._auto_match_scopes({"project_path": str(self.project_path)})
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["confident_count"], 1)
        self.assertEqual(result["conflict_count"], 0)
        question_id = project["pages"][0]["questions"][0]["id"]
        pair = next(item for item in result["pairs"] if item["question_id"] == question_id)
        self.assertEqual(pair["source"], "confident_auto")
        self.assertEqual(
            pair["answer_ids"], [project["answer_pages"][0]["answers"][0]["id"]]
        )

    def test_scope_auto_match_keeps_ambiguous_recommendations_for_manual_review(self) -> None:
        """两个候选都能对上题号时不算唯一推荐，保持待处理交人工核对。"""
        project = api.annotation_data.load_project(self.project_path)
        project["pages"][0].update(status="accepted", questions=[
            api.annotation_data.new_record([1, 1, 20, 20], number_path=["九", "3"]),
        ])
        project["answer_pages"][0].update(status="accepted", answers=[
            api.annotation_data.new_record([1, 1, 20, 20], answer=True, number_path=["九", "3"]),
            api.annotation_data.new_record([30, 1, 50, 20], answer=True, number_path=["九", "3"]),
        ])
        api.annotation_data.save_project(project, self.project_path)
        result = api._auto_match_scopes({"project_path": str(self.project_path)})
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["confident_count"], 0)
        self.assertEqual(result["pairs"], [])

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


class EnrichBridgeTests(unittest.TestCase):
    """T03：快照冻结 / 单页分级 / pair 字典 / 书级锁 / 总开关。

    与 ``AnnotationApiTests`` 用同一套临时教材，但**只新增**断言，
    不改既有用例（既有的 41 项必须原样通过）。
    """

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
        opened_project = api.annotation_data.load_project(self.project_path)
        self.delivery_root = self.book / opened_project["delivery_book_id"]
        self._saved_enrich_flag = os.environ.get("PAPERFLOW_ENRICH")
        os.environ.pop("PAPERFLOW_ENRICH", None)

    def tearDown(self) -> None:
        if self._saved_enrich_flag is None:
            os.environ.pop("PAPERFLOW_ENRICH", None)
        else:
            os.environ["PAPERFLOW_ENRICH"] = self._saved_enrich_flag
        if self.project_path.is_file():
            self.project_path.unlink()
        workspace_db = api.database_path(self.project_path, ROOT / "runtime")
        if workspace_db.is_file():
            workspace_db.unlink()
        if self.delivery_root.is_dir():
            shutil.rmtree(self.delivery_root)
        self.temp.cleanup()

    def _confirm_question_page(self, records=None) -> dict:
        records = records if records is not None else [
            {"id": "q-enrich", "bbox": [10, 10, 290, 190],
             "number_path": ["一", "1"], "text": "1. 计算 12×4"},
        ]
        return api._save_page({
            "project_path": str(self.project_path), "kind": "question",
            "page_index": 0, "accepted": True, "records": records,
        })

    def _question(self, index: int = 0) -> dict:
        project = api.annotation_data.load_project(self.project_path)
        return project["pages"][0]["questions"][index]

    def test_save_page_freezes_a_snapshot_and_writes_a_review_level(self) -> None:
        result = self._confirm_question_page()
        self.assertTrue(result["enrich"]["enabled"])
        self.assertEqual(result["enrich"]["frozen"], 1)
        record = self._question()
        self.assertIsInstance(record["snapshot"], dict)
        self.assertTrue(record["snapshot"]["hash"])
        self.assertEqual(record["snapshot_id"], record["snapshot"]["hash"][:16])
        self.assertIn(record["review_level"], {"L0", "L1", "L2"})
        self.assertEqual(record["enrich_status"], "pending")
        project = api.annotation_data.load_project(self.project_path)
        self.assertEqual(project["pages"][0]["review_summary"],
                         {"L0": 0, "L1": 0, "L2": 1})

    def test_editing_one_box_marks_only_that_record_stale(self) -> None:
        self._confirm_question_page([
            {"id": "q-a", "bbox": [10, 10, 290, 90], "number_path": ["一", "1"]},
            {"id": "q-b", "bbox": [10, 100, 290, 190], "number_path": ["一", "2"]},
        ])
        untouched = copy.deepcopy(self._question(1))
        self._confirm_question_page([
            {"id": "q-a", "bbox": [20, 20, 290, 100], "number_path": ["一", "1"]},
            {"id": "q-b", "bbox": [10, 100, 290, 190], "number_path": ["一", "2"]},
        ])
        changed = self._question(0)
        kept = self._question(1)
        self.assertEqual(changed["enrich_status"], "stale")
        self.assertEqual(changed["review_level"], "L2")
        self.assertEqual(kept["enrich_status"], untouched["enrich_status"])
        self.assertEqual(kept["snapshot"]["hash"], untouched["snapshot"]["hash"])
        self.assertEqual(kept["review_level"], untouched["review_level"])

    def test_whole_project_cache_save_keeps_enrich_fields(self) -> None:
        """R2 回归：旧版渲染层整工程 autosave 不能抹掉 AI 补全结果。"""
        self._confirm_question_page()
        project = api.annotation_data.load_project(self.project_path)
        record = project["pages"][0]["questions"][0]
        record["field_confidence"] = {"difficulty": 0.91}
        record["enrich_status"] = "done"
        record["kp_codes"] = ["M-001"]
        record["analysis_check"] = {"status": "pass"}
        project["pages"][0]["illustration_ready"] = True
        project["enrichment"] = {"taxonomy_version": "math-2026.10",
                                 "prompt_version": "enrich-v1"}
        api.annotation_data.save_project(project, self.project_path, increment_revision=False)
        authoritative = api.annotation_data.load_project(self.project_path)
        expected_snapshot = authoritative["pages"][0]["questions"][0]["snapshot"]

        stale = copy.deepcopy(authoritative)
        stale_record = stale["pages"][0]["questions"][0]
        for key in api.annotation_data.ENRICH_PERSISTENT_KEYS:
            stale_record.pop(key, None)
        stale["pages"][0].pop("review_summary", None)
        stale["pages"][0].pop("illustration_ready", None)
        stale.pop("enrichment", None)
        api._save_project_cache({
            "project_path": str(self.project_path), "project": stale,
            "base_revision": authoritative["project_revision"],
        })

        saved = api.annotation_data.load_project(self.project_path)
        saved_record = saved["pages"][0]["questions"][0]
        self.assertEqual(saved_record["snapshot"], expected_snapshot)
        self.assertEqual(saved_record["enrich_status"], "done")
        self.assertEqual(saved_record["field_confidence"], {"difficulty": 0.91})
        self.assertEqual(saved_record["kp_codes"], ["M-001"])
        self.assertEqual(saved_record["analysis_check"], {"status": "pass"})
        self.assertEqual(saved_record["review_level"], authoritative["pages"][0]["questions"][0]["review_level"])
        self.assertIn("review_summary", saved["pages"][0])
        self.assertTrue(saved["pages"][0]["illustration_ready"])
        self.assertEqual(saved["enrichment"]["taxonomy_version"], "math-2026.10")

    def test_enrich_switch_off_restores_the_previous_behaviour(self) -> None:
        with mock.patch.dict(os.environ, {"PAPERFLOW_ENRICH": "0"}):
            result = self._confirm_question_page()
            self.assertEqual(result["enrich"], {"enabled": False})
            record = self._question()
            self.assertIsNone(record.get("snapshot"))
            self.assertNotIn("review_level", record)
            self.assertNotIn("review_summary", api.annotation_data.load_project(
                self.project_path)["pages"][0])
            for route in ("enrich-prepare", "enrich-status", "book-meta-suggest"):
                payload = api.ROUTES[f"/api/project/{route}"]({
                    "project_path": str(self.project_path), "page_index": 0,
                })
                self.assertFalse(payload["ok"])
                self.assertTrue(payload["disabled"])

    def test_set_pair_writes_a_pair_dictionary_with_five_keys(self) -> None:
        self._confirm_question_page()
        api._save_page({
            "project_path": str(self.project_path), "kind": "answer",
            "page_index": 0, "accepted": True,
            "records": [{"id": "a-enrich", "bbox": [10, 10, 290, 190],
                         "number_path": ["一", "1"], "answer": "48"}],
        })
        result = api._set_pair({
            "project_path": str(self.project_path),
            "question_id": "q-enrich", "answer_ids": ["a-enrich"],
        })
        pair = result["pair_metrics"]
        self.assertEqual(set(pair), {"score", "margin", "path_match", "ok", "display_confidence"})
        self.assertTrue(pair["path_match"])
        self.assertTrue(pair["ok"])
        self.assertGreaterEqual(pair["score"], 180)
        self.assertGreaterEqual(pair["margin"], 80)
        self.assertAlmostEqual(pair["display_confidence"], min(1.0, pair["score"] / 300.0), places=4)
        record = self._question()
        self.assertEqual(record["pair"], pair)
        self.assertNotIn("pair_confidence", record)
        # 答案后到：快照里的 answer 段必须补上（§3.4）
        self.assertEqual(record["snapshot"]["answer"], "48")
        self.assertEqual(record["snapshot"]["answer_source"], "paired")

    def test_locked_book_metadata_requires_an_unlock_reason(self) -> None:
        metadata = {"language": "zh", "stage": "小学", "grade": "四年级",
                    "subject": "数学", "answer_mode": "textbook"}
        api._set_book_metadata({
            "project_path": str(self.project_path), "metadata": metadata,
        })
        saved = api.annotation_data.load_project(self.project_path)
        self.assertTrue(saved["book_meta_locked"])
        self.assertEqual(saved["book_meta_rev"], api.book_meta.book_meta_rev(saved))
        with self.assertRaisesRegex(ValueError, "unlock_reason"):
            api._set_book_metadata({
                "project_path": str(self.project_path),
                "metadata": {**metadata, "grade": "五年级"},
            })
        self.assertEqual(
            api.annotation_data.load_project(self.project_path)["grade"], "四年级",
        )
        unlocked = api._set_book_metadata({
            "project_path": str(self.project_path),
            "metadata": {**metadata, "grade": "五年级"},
            "unlock_reason": "教材实际为五年级上册",
        })
        self.assertEqual(unlocked["project"]["grade"], "五年级")
        self.assertTrue(unlocked["book_meta_locked"])
        self.assertNotEqual(
            api.annotation_data.load_project(self.project_path)["book_meta_rev"],
            saved["book_meta_rev"],
        )

    def test_page_ocr_keeps_the_old_text_when_recognition_matches(self) -> None:
        self._confirm_question_page()
        project = api.annotation_data.load_project(self.project_path)
        ocr_items = [{"bbox": [30, 30, 200, 60], "text": "1. 计算 12×4", "conf": 0.9}]
        with mock.patch.object(api, "ocr_image", return_value=ocr_items):
            result = api._page_ocr({
                "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            })
        payload = result["results"]["q-enrich"]
        self.assertFalse(payload["changed"])
        self.assertEqual(payload["text"], "1. 计算 12×4")
        self.assertEqual(result["unchanged"], ["q-enrich"])

        with mock.patch.object(api, "ocr_image",
                               return_value=[{"bbox": [30, 30, 200, 60],
                                              "text": "1. 计算 12×5", "conf": 0.9}]):
            changed = api._page_ocr({
                "project_path": str(self.project_path), "kind": "question", "page_index": 0,
            })
        self.assertTrue(changed["results"]["q-enrich"]["changed"])
        # _normalize_ocr_text 会去掉中文与数字之间的空格
        self.assertEqual(changed["results"]["q-enrich"]["text"], "1. 计算12×5")
        self.assertEqual(changed["unchanged"], [])

    def test_enrich_prepare_publishes_candidates_without_confirming_them(self) -> None:
        self._confirm_question_page()
        candidate = {
            "id": "ic-0-1", "bbox": [20, 20, 120, 120], "label": "image",
            "owner_id": "q-enrich", "owner_method": "contain", "owner_confidence": 0.95,
            "reject_reason": "", "confirmed": False,
        }
        with mock.patch("review_tools.illustration_candidates.collect",
                        return_value=[candidate]), \
                mock.patch.dict(os.environ, {"PAPERFLOW_LAYOUT_ARBITER": "0"}):
            result = api._enrich_prepare({
                "project_path": str(self.project_path), "page_index": 0,
            })
        self.assertEqual(result["candidates"], 1)
        saved = api.annotation_data.load_project(self.project_path)
        record = saved["pages"][0]["questions"][0]
        self.assertEqual([item["id"] for item in record["illustration_candidates"]],
                         ["ic-0-1"])
        self.assertEqual(record["illustrations"], [],
                         "候选永远不能自动变成正式题图")
        # 版面裁判不可用 = "没算"，不是"没图"：illustration_ready 必须保持 False，
        # 否则 no_image_candidate 会把整页题目误判 L2。
        self.assertFalse(saved["pages"][0]["illustration_ready"])
        self.assertIn("review_summary", saved["pages"][0])

    def test_enrich_status_reports_levels_and_todo_items(self) -> None:
        self._confirm_question_page()
        status = api._enrich_status({"project_path": str(self.project_path)})
        self.assertTrue(status["ok"])
        self.assertEqual(status["level_stats"]["total"], 1)
        self.assertEqual(status["review_summary"]["L2"], 1)
        self.assertEqual(status["todo_count"], 1)
        self.assertEqual(status["todo"][0]["question_id"], "q-enrich")
        self.assertIn("pair_fail", status["todo"][0]["review_reasons"])
        self.assertEqual(status["enrich_summary"], {"pending": 1})

    def test_enrich_status_counts_readable_page_hash_as_ok(self) -> None:
        """C1：源图读得到时，指纹状态记 ok，不产生"核不了"的告警数。"""
        self._confirm_question_page()
        status = api._enrich_status({"project_path": str(self.project_path)})
        self.assertEqual(
            status["snapshot_health"],
            {"total": 1, "page_sha1_ok": 1, "page_sha1_unavailable": 0, "no_snapshot": 0},
        )
        self.assertEqual(
            self._question()["snapshot"]["page_sha1_status"], "ok",
            "夹具里的 question.png 真实存在，指纹必须算得出来",
        )

    def test_enrich_status_flags_records_whose_page_hash_is_unreadable(self) -> None:
        """C1：源图没了（换机/拔盘/目录改名）时，必须能看见，而不是安静地当正常。"""
        (self.book / "question.png").unlink()
        api.enrich_snapshot.clear_page_hash_cache()
        self._confirm_question_page()
        status = api._enrich_status({"project_path": str(self.project_path)})
        self.assertEqual(
            status["snapshot_health"],
            {"total": 1, "page_sha1_ok": 0, "page_sha1_unavailable": 1, "no_snapshot": 0},
        )
        record = self._question()
        self.assertEqual(record["snapshot"]["page_sha1_status"], "unavailable")
        # 只读诊断：图读不到**不能**把快照判成无效，否则离机审查会整本失效
        self.assertTrue(api.enrich_snapshot.is_valid_snapshot(record["snapshot"]))
        self.assertEqual(status["todo_count"], 1)

    def test_book_meta_suggest_never_writes_the_project(self) -> None:
        before = self.project_path.read_bytes()
        suggestion = api._book_meta_suggest({
            "project_path": str(self.project_path), "title": "人教版四年级上册数学",
        })
        self.assertTrue(suggestion["ok"])
        self.assertEqual(suggestion["candidates"]["subject"], "数学")
        self.assertEqual(suggestion["candidates"]["grade"], "四年级")
        self.assertTrue(suggestion["validation"]["ok"])
        self.assertEqual(self.project_path.read_bytes(), before)


class PrewarmTests(unittest.TestCase):
    """启动预热是纯性能优化：必须后台完成，且失败也不能影响点「自动框题」。"""

    def setUp(self) -> None:
        self._saved_state = dict(api.PREWARM_STATE)

    def tearDown(self) -> None:
        with api.PREWARM_STATE_LOCK:
            api.PREWARM_STATE.clear()
            api.PREWARM_STATE.update(self._saved_state)

    def test_prewarm_loads_both_engines_while_holding_the_inference_lock(self) -> None:
        observed: list[tuple[str, bool]] = []

        def record(name: str):
            def side_effect(*_args, **_kwargs):
                observed.append((name, api.PRELABEL_INFERENCE_LOCK.locked()))
                return None
            return side_effect

        import cut_questions_v4 as cutter

        with mock.patch.object(cutter, "get_fast_ocr_engine", side_effect=record("fast")), \
                mock.patch.object(cutter, "get_ocr_engine", side_effect=record("medium")), \
                mock.patch.object(cutter, "_run_fast_ocr", lambda image: ([], True)), \
                mock.patch.object(cutter, "_run_ocr", lambda image: []):
            api._prewarm_ocr_models()

        self.assertEqual([name for name, _ in observed], ["fast", "medium"])
        self.assertTrue(
            all(locked for _, locked in observed),
            "预热必须与真实预标注共用同一把推理锁，否则 Paddle predictor 会被两线程同时使用",
        )
        self.assertEqual(api.prewarm_state()["status"], "ready")

    def test_prewarm_failure_degrades_instead_of_raising(self) -> None:
        import cut_questions_v4 as cutter

        with mock.patch.object(cutter, "get_fast_ocr_engine", return_value=None), \
                mock.patch.object(cutter, "get_ocr_engine", side_effect=RuntimeError("模型缺失")), \
                mock.patch.object(api.traceback, "print_exc", lambda: None):
            api._prewarm_ocr_models()  # 不允许抛异常：预热失败仍要能正常服务

        state = api.prewarm_state()
        self.assertEqual(state["status"], "failed")
        self.assertIn("模型缺失", state["detail"])

    def test_prewarm_can_be_disabled_by_environment(self) -> None:
        with mock.patch.dict("os.environ", {"PAPERFLOW_DISABLE_PREWARM": "1"}):
            api._start_prewarm_thread()
        self.assertEqual(api.prewarm_state()["status"], "disabled")

    def test_health_endpoint_exposes_prewarm_progress(self) -> None:
        api._set_prewarm_state("loading")
        handler = api.PaperFlowHandler.__new__(api.PaperFlowHandler)
        handler.path = "/api/health"
        with mock.patch.object(api.PaperFlowHandler, "_respond") as respond:
            handler.do_GET()

        payload = respond.call_args.args[1]
        self.assertEqual(payload["bridge_revision"], api.PAPERFLOW_BRIDGE_REVISION)
        self.assertEqual(payload["prewarm"]["status"], "loading")

    def test_server_start_launches_prewarm_on_a_daemon_thread(self) -> None:
        source = API_PATH.read_text(encoding="utf-8")
        main_body = source.split("def main() -> None:", 1)[1]
        self.assertIn("_start_prewarm_thread()", main_body)
        self.assertIn("daemon=True", source)


if __name__ == "__main__":
    unittest.main()
