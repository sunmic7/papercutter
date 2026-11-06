"""交付预览接口：分页、筛选、图片越界与标记权限。

预览读的是已发布版本里的 dataset.jsonl，所以这里直接搭一个真实的
``current.json`` + ``versions/<v>/k12_dataset`` 目录，不走生成链路。
"""
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from online_review_beta.server import ReviewStore


BOOK = "BOOKPREVIEW0001"
VERSION = "20261106-120000-abcd"


def make_sample(index: int, *, question_type: str = "选择", difficulty: str = "基础",
                question_page: int = 1, number: str = "") -> dict:
    number = number or str(index)
    return {
        "id": f"{index:07d}",
        "metainfo": {
            "language": "zh", "stage": "小学", "grade": "六年级", "subject": "数学",
            "question_type": question_type, "difficulty": difficulty,
            "knowledge_points": ["分数的加法"],
        },
        "data": {
            "question_images": [f"images/{index:07d}_img_01.jpg"],
            "option_images": [f"images/{index:07d}_option_A_01.jpg"] if question_type == "选择" else [],
            "question": f"第{number}题：计算 1/2 + 1/3 的值。",
            "options": ["5/6", "2/5", "1/6"] if question_type == "选择" else [],
            "answer": "5/6",
            "analysis": "先通分再相加。",
        },
        "source": {
            "book_id": BOOK, "question_page": question_page, "page_order": 1,
            "question_number": number, "internal_question_id": f"q-{index}",
            "internal_answer_ids": [f"a-{index}"],
        },
    }


class DeliveryPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = ReviewStore(root / "data", staff_data_root=root / "staff")
        self.super_admin = self.store.create_user("superadmin", "very-secret-password", "super_admin")
        self.admin = self.store.create_managed_user(self.super_admin, "admin", "very-secret-password", "admin")
        self.reviewer = self.store.create_managed_user(self.admin, "reviewer", "very-secret-password", "reviewer")
        self.other_reviewer = self.store.create_managed_user(self.admin, "reviewer2", "very-secret-password", "reviewer")
        self.annotator = self.store.create_managed_user(self.admin, "marker", "very-secret-password", "annotator")
        with closing(self.store._connect()) as db:
            db.execute(
                "INSERT INTO project_assignments (book_id,book_name,team_owner_id,annotator_id,"
                "assigned_by_id,assigned_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (BOOK, "六年级数学（上）", self.admin["id"], self.annotator["id"],
                 self.admin["id"], "2026-11-06T10:00:00+00:00", "2026-11-06T10:00:00+00:00"))
            db.commit()
        self.root = self.store._delivery_root({"team_owner_id": self.admin["id"],
                                               "annotator_id": self.annotator["id"]}, BOOK)
        self.publish(VERSION, [make_sample(1), make_sample(2, question_type="填空"),
                               make_sample(3, difficulty="困难", question_page=2)])

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, version: str, samples: list[dict], *, pointer: bool = True,
                rejected: int = 0) -> Path:
        dataset_dir = self.root / "versions" / version / "k12_dataset"
        (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
        with (dataset_dir / "dataset.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        (dataset_dir / "quality_report.json").write_text(json.dumps({
            "schema_version": 1, "total_questions": len(samples) + rejected,
            "eligible_count": len(samples), "rejected_count": rejected,
            "reason_summary": ([{"reason": "缺少知识点", "count": rejected, "questions": ["9"]}]
                               if rejected else []),
            "items": [],
        }, ensure_ascii=False), encoding="utf-8")
        body = b"\xff\xd8\xff\xe0preview-crop"
        (dataset_dir / "images" / "0000001_img_01.jpg").write_bytes(body)
        (dataset_dir / "images" / "0000001_option_A_01.jpg").write_bytes(body)
        if pointer:
            (self.root / "current.json").write_text(json.dumps({
                "version": version, "path": f"versions/{version}",
                "published_at": "2026-11-06T12:00:00+00:00",
            }), encoding="utf-8")
        return dataset_dir

    # -- 数据读取 -------------------------------------------------------
    def test_samples_expose_delivered_fields_without_rewriting_them(self):
        data = self.store.delivery_samples(self.reviewer, BOOK, offset=0, limit=2)
        self.assertTrue(data["ok"])
        self.assertEqual(data["book_name"], "六年级数学（上）")
        self.assertEqual(data["version"], VERSION)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["matched"], 3)
        self.assertEqual(str(data["dataset_path"]), f"versions/{VERSION}/k12_dataset/dataset.jsonl")
        self.assertEqual(len(data["samples"]), 2)
        first = data["samples"][0]
        self.assertEqual(first["index"], 0)
        # 原始样本原样返回，右侧「原始 JSON」才是交付包原文
        self.assertEqual(first["sample"], make_sample(1))
        self.assertTrue(first["question_image_urls"][0].startswith(
            f"/api/deliveries/{BOOK}/sample-image?version={VERSION}&path=images%2F"))
        self.assertEqual(data["filters"]["question_types"], ["填空", "选择"])
        self.assertIn("基础", data["filters"]["difficulties"])

    def test_pagination_and_second_page(self):
        data = self.store.delivery_samples(self.reviewer, BOOK, offset=2, limit=2)
        self.assertEqual(len(data["samples"]), 1)
        self.assertEqual(data["samples"][0]["index"], 2)
        self.assertEqual(data["samples"][0]["sample"]["source"]["question_page"], 2)

    def test_filter_by_type_difficulty_and_query(self):
        only_fill = self.store.delivery_samples(self.reviewer, BOOK, question_type="填空")
        self.assertEqual(only_fill["matched"], 1)
        self.assertEqual(only_fill["samples"][0]["sample"]["id"], "0000002")
        hard = self.store.delivery_samples(self.reviewer, BOOK, difficulty="困难")
        self.assertEqual(hard["matched"], 1)
        searched = self.store.delivery_samples(self.reviewer, BOOK, query="第3题")
        self.assertEqual(searched["matched"], 1)
        self.assertEqual(searched["samples"][0]["sample"]["id"], "0000003")

    def test_rejected_reasons_travel_with_the_preview(self):
        self.publish("20261106-130000-efgh", [make_sample(1)], rejected=2)
        data = self.store.delivery_samples(self.reviewer, BOOK)
        self.assertEqual(data["rejected_count"], 2)
        self.assertEqual(data["reason_summary"][0]["reason"], "缺少知识点")
        self.assertEqual(len(data["versions"]), 2)
        self.assertEqual(data["versions"][0]["version"], "20261106-130000-efgh")
        self.assertTrue(data["versions"][0]["is_current"])

    def test_pinned_old_version_still_reads_history(self):
        older = "20261106-090000-0000"
        self.publish(older, [make_sample(7, number="7")])
        data = self.store.delivery_samples(self.reviewer, BOOK, version=older)
        self.assertEqual(data["version"], older)
        self.assertEqual(data["samples"][0]["sample"]["source"]["question_number"], "7")
        with self.assertRaises(LookupError):
            self.store.delivery_samples(self.reviewer, BOOK, version="20990101-000000-nope")

    def test_missing_version_speaks_plainly(self):
        (self.root / "current.json").unlink()
        with self.assertRaises(LookupError) as caught:
            self.store.delivery_samples(self.reviewer, BOOK)
        self.assertIn("还没有已发布的交付版本", str(caught.exception))

    def test_annotator_cannot_open_the_delivery_preview(self):
        with self.assertRaises(PermissionError):
            self.store.delivery_samples(self.annotator, BOOK)

    def test_unknown_book_is_not_found(self):
        with self.assertRaises(LookupError):
            self.store.delivery_samples(self.reviewer, "BOOKDOESNOTEXIST")

    # -- 图片 -----------------------------------------------------------
    def test_image_is_served_from_the_published_version(self):
        target = self.store.delivery_sample_image(
            self.reviewer, BOOK, "images/0000001_img_01.jpg", VERSION)
        self.assertTrue(target.is_file())
        self.assertEqual(target.parent, (self.root / "versions" / VERSION / "k12_dataset" / "images"))

    def test_image_path_cannot_escape_the_dataset(self):
        for bad in ("../../current.json", "/etc/passwd", "images/../../current.json",
                    "k12_dataset/dataset.jsonl", "images/note.txt", ""):
            with self.subTest(path=bad), self.assertRaises((ValueError, LookupError)):
                self.store.delivery_sample_image(self.reviewer, BOOK, bad, VERSION)

    def test_missing_image_is_reported_not_500(self):
        with self.assertRaises(LookupError):
            self.store.delivery_sample_image(self.reviewer, BOOK, "images/nope.jpg", VERSION)

    # -- 标记 -----------------------------------------------------------
    def test_note_lifecycle_and_ownership(self):
        note = self.store.add_delivery_note(
            self.reviewer, BOOK, {"version": VERSION, "sample_id": "0000001",
                                  "body": "第 1 张题图裁进了下一题"})
        self.assertEqual(note["version"], VERSION)
        listed = self.store.delivery_samples(self.reviewer, BOOK)
        self.assertEqual([item["id"] for item in listed["notes"]], [note["id"]])
        self.assertEqual(listed["notes"][0]["sample_id"], "0000001")
        with self.assertRaises(PermissionError):
            self.store.delete_delivery_note(self.other_reviewer, BOOK, note["id"])
        self.store.delete_delivery_note(self.admin, BOOK, note["id"])
        self.assertEqual(self.store.delivery_samples(self.reviewer, BOOK)["notes"], [])

    def test_note_requires_body_and_known_version(self):
        with self.assertRaises(ValueError):
            self.store.add_delivery_note(self.reviewer, BOOK,
                                         {"version": VERSION, "sample_id": "0000001", "body": "   "})
        with self.assertRaises(ValueError):
            self.store.add_delivery_note(self.reviewer, BOOK,
                                         {"version": VERSION, "sample_id": "", "body": "缺样本号"})
        with self.assertRaises(LookupError):
            self.store.add_delivery_note(self.reviewer, BOOK,
                                         {"version": "20990101-000000-nope",
                                          "sample_id": "0000001", "body": "版本不存在"})

    def test_annotator_cannot_write_notes(self):
        with self.assertRaises(PermissionError):
            self.store.add_delivery_note(self.annotator, BOOK,
                                         {"version": VERSION, "sample_id": "0000001", "body": "越权"})

    def test_deleting_unknown_note_is_not_found(self):
        with self.assertRaises(LookupError):
            self.store.delete_delivery_note(self.reviewer, BOOK, 99999)


if __name__ == "__main__":
    unittest.main()
