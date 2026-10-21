import base64
import hashlib
import json
import os
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from online_review_beta.server import ConflictError, ReviewHandler, ReviewStore


def snapshot() -> dict:
    preview = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8" + b"review-preview" * 20).decode("ascii")
    return {
        "schema_version": 1,
        "book": {"id": "book-test", "name": "测试教材"},
        "project_revision": 4,
        "snapshot_fingerprint": "a" * 64,
        "pages": [{
            "page_key": "question-0", "kind": "question", "page_index": 0,
            "image_name": "001.jpg", "width": 1200, "height": 1800,
            "regions": [{"id": "q-1", "display_number": "1", "bbox": [20, 30, 400, 500]}],
            "preview": preview,
        }],
    }


class OnlineReviewBetaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self.temp.name))
        self.super_admin = self.store.create_user("superadmin", "very-secret-password", "super_admin")
        self.admin = self.store.create_managed_user(self.super_admin, "admin", "very-secret-password", "admin")
        self.annotator = self.store.create_managed_user(self.admin, "marker", "very-secret-password", "annotator")
        self.reviewer = self.store.create_managed_user(self.admin, "reviewer", "very-secret-password", "reviewer")

    def tearDown(self):
        self.temp.cleanup()

    def test_submission_keeps_only_review_snapshot_and_returns_decisions(self):
        created = self.store.create_submission(self.annotator, snapshot())
        self.assertEqual(created["page_count"], 1)
        detail = self.store.submission(self.reviewer, created["id"])
        self.assertEqual(detail["pages"][0]["regions"][0]["id"], "q-1")
        self.assertTrue(self.store.preview_path(self.reviewer, created["id"], "question-0").is_file())

        result = self.store.decide_page(self.reviewer, created["id"], "question-0", "needs_fix", "题框需要收窄")
        self.assertEqual(result["submission_status"], "needs_fix")
        decisions = self.store.decisions(self.annotator, created["id"])
        self.assertEqual(decisions["snapshot_fingerprint"], "a" * 64)
        self.assertEqual(decisions["decisions"][0]["review_status"], "needs_fix")
        self.assertEqual(decisions["unresolved_page_count"], 1)

    def test_assignment_locks_reviewer_and_resubmission_links_the_previous_task(self):
        second_reviewer = self.store.create_managed_user(
            self.admin, "reviewer2", "very-secret-password", "reviewer"
        )
        first = self.store.create_submission(self.annotator, snapshot())
        assigned = self.store.assign_submission(self.admin, first["id"], self.reviewer["id"], "high", "2026-08-30T12:00")
        self.assertEqual(assigned["status"], "assigned")
        self.assertEqual(assigned["assigned_reviewer_id"], self.reviewer["id"])
        with self.assertRaises(PermissionError):
            self.store.decide_page(second_reviewer, first["id"], "question-0", "approved", "")
        self.store.decide_page(self.reviewer, first["id"], "question-0", "returned", "please redraw")
        second = self.store.create_submission(self.annotator, snapshot())
        self.assertEqual(second["supersedes_submission_id"], first["id"])
        self.assertEqual(self.store.submission(self.admin, first["id"])["status"], "resubmitted")

    def test_user_can_change_own_password_without_changing_any_role(self):
        self.store.change_own_password(self.super_admin, "very-secret-password", "a123")
        _token, account = self.store.login("superadmin", "a123")
        self.assertEqual(account["role"], "super_admin")
        self.store.manage_user(self.super_admin, self.super_admin["id"], {"password": "self-reset"})
        _token, account = self.store.login("superadmin", "self-reset")
        self.assertEqual(account["role"], "super_admin")
        # An administrator is itself responsible to the super admin, so it
        # must still be able to use the compact self-reset action.
        self.store.manage_user(self.admin, self.admin["id"], {"password": "admin-self-reset"})
        _token, account = self.store.login("admin", "admin-self-reset")
        self.assertEqual(account["role"], "admin")
        with self.assertRaises(ValueError):
            self.store.manage_user(self.super_admin, self.super_admin["id"], {"active": False})
        with self.assertRaises(ValueError):
            self.store.change_own_password(self.super_admin, "wrong-password", "another")

    def test_manager_can_disable_and_reset_direct_report_only(self):
        updated = self.store.manage_user(self.admin, self.reviewer["id"], {"active": False})
        self.assertFalse(updated["active"])
        with self.assertRaises(ValueError):
            self.store.login("reviewer", "very-secret-password")
        self.store.manage_user(self.admin, self.reviewer["id"], {"active": True, "password": "replacement-password"})
        _token, account = self.store.login("reviewer", "replacement-password")
        self.assertEqual(account["role"], "reviewer")
        with self.assertRaises(LookupError):
            self.store.manage_user(self.admin, self.super_admin["id"], {"active": False})

    def test_annotator_cannot_review_and_rejection_requires_note(self):
        created = self.store.create_submission(self.annotator, snapshot())
        with self.assertRaises(PermissionError):
            self.store.decide_page(self.annotator, created["id"], "question-0", "approved", "")
        with self.assertRaises(ValueError):
            self.store.decide_page(self.reviewer, created["id"], "question-0", "returned", "")

    def test_submission_list_and_stats_count_frames(self):
        self.store.create_submission(self.annotator, snapshot())
        submissions = self.store.list_submissions(self.reviewer)
        self.assertEqual(submissions[0]["region_count"], 1)
        stats = self.store.stats(self.admin)
        self.assertEqual(stats["totals"]["regions"], 1)

    def test_progress_reads_shared_paperflow_project_snapshot(self):
        projects_root = Path(self.temp.name) / "Projects"
        book_dir = projects_root / "BDEMO"
        book_dir.mkdir(parents=True)
        (book_dir / "PaperFlow标注项目.json").write_text(json.dumps({
            "book_root": r"G:\教材\示例数学",
            "project_revision": 7,
            "language": "zh", "stage": "小学", "grade": "二年级", "subject": "数学",
            "answer_mode": "textbook",
            "pages": [{
                "image": "question-01.jpg", "status": "accepted",
                "questions": [{"id": "q-1", "number_path": ["1"], "bbox": [1, 2, 30, 40], "text": "题干"}],
            }],
            "answer_pages": [{
                "image": "answer-01.jpg", "status": "accepted",
                "answers": [{"id": "a-1", "number_path": ["1"], "bbox": [1, 2, 30, 40], "text": "答案"}],
            }],
            "ignored_pages": [],
            "match_scopes": [{"id": "scope-1"}],
            "pairs": [{"id": "pair-1", "status": "accepted", "question_id": "q-1"}],
        }, ensure_ascii=False), encoding="utf-8")
        self.store.projects_root = projects_root.resolve()
        progress_store = self.store

        rows = progress_store.list_projects(self.admin)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["book_name"], "示例数学")
        self.assertEqual(rows[0]["question_regions"], 1)
        self.assertEqual(rows[0]["answer_regions"], 1)
        assignment = progress_store.assign_project(self.admin, "BDEMO", self.annotator["id"])
        self.assertEqual(assignment["annotator_id"], self.annotator["id"])
        self.assertEqual(progress_store.list_projects(self.annotator)[0]["book_id"], "BDEMO")
        other = progress_store.create_managed_user(self.admin, "marker-two", "very-secret-password", "annotator")
        self.assertEqual(progress_store.list_projects(other), [])
        self.assertEqual(progress_store.list_projects(self.reviewer)[0]["book_id"], "BDEMO")
        detail = progress_store.project_detail(self.admin, "BDEMO")
        self.assertTrue(detail["pages"][0]["regions"][0]["has_ocr"])
        self.assertEqual(detail["answer_pages"][0]["filled_count"], 1)
        with self.assertRaises(LookupError):
            progress_store.project_detail(other, "BDEMO")

    def test_annotator_sync_creates_an_account_owned_path_and_isolates_it(self):
        projects_root = Path(self.temp.name) / "Projects"
        self.store.projects_root = projects_root.resolve()
        self.store.staff_data_root = projects_root / "StaffProjects"
        document = {
            "book": {"id": "BSTAFF", "name": "账号归档教材"},
            "project": {
                "delivery_book_id": "BSTAFF", "project_revision": 9,
                "pages": [{"status": "accepted", "questions": [{"id": "q-1", "bbox": [1, 2, 3, 4]}]}],
                "answer_pages": [], "pairs": [],
            },
        }
        synced = self.store.sync_project_snapshot(self.annotator, document)
        self.assertEqual(synced["server_relative_path"], "private/team-2/annotator-3/BSTAFF/PaperFlow标注项目.json")
        self.assertTrue((self.store.staff_data_root / "team-2" / "annotator-3" / "BSTAFF" / "PaperFlow标注项目.json").is_file())
        self.assertEqual(self.store.list_projects(self.annotator)[0]["book_id"], "BSTAFF")
        other = self.store.create_managed_user(self.admin, "marker-two", "very-secret-password", "annotator")
        self.assertEqual(self.store.list_projects(other), [])
        with self.assertRaises(PermissionError):
            self.store.sync_project_snapshot(other, document)

    def test_annotator_can_upload_only_own_final_delivery_file(self):
        # 该用例验证“归档隔离”本身，与审核放行门禁正交，因此临时关门禁。
        previous = os.environ.get("PAPERFLOW_DELIVERY_REQUIRE_APPROVAL")
        os.environ["PAPERFLOW_DELIVERY_REQUIRE_APPROVAL"] = "0"
        try:
            self._annotator_delivery_archive_checks()
        finally:
            if previous is None:
                os.environ.pop("PAPERFLOW_DELIVERY_REQUIRE_APPROVAL", None)
            else:
                os.environ["PAPERFLOW_DELIVERY_REQUIRE_APPROVAL"] = previous

    def _annotator_delivery_archive_checks(self):
        self.store.staff_data_root = Path(self.temp.name) / "private" / "staff_projects"
        self.store.sync_project_snapshot(self.annotator, {"book": {"id": "BFINAL", "name": "交付教材"}, "project": {"delivery_book_id": "BFINAL"}})
        content = b'{"ok":true}'
        uploaded = self.store.upload_delivery_file(self.annotator, {
            "book": {"id": "BFINAL"}, "relative_path": "k12_dataset/dataset.jsonl",
            "content_base64": base64.b64encode(content).decode("ascii"), "sha256": hashlib.sha256(content).hexdigest(),
        })
        self.assertEqual(uploaded["bytes"], len(content))
        self.assertTrue((self.store.staff_data_root.parent / "final_deliveries" / "team-2" / "annotator-3" / "BFINAL" / "k12_dataset" / "dataset.jsonl").is_file())
        other = self.store.create_managed_user(self.admin, "marker-final", "very-secret-password", "annotator")
        with self.assertRaises(PermissionError):
            self.store.upload_delivery_file(other, {"book": {"id": "BFINAL"}, "relative_path": "x.json", "content_base64": "e30=", "sha256": hashlib.sha256(b"{}").hexdigest()})
        listed = self.store.delivery_files(self.reviewer, "BFINAL")
        self.assertEqual(listed["files"][0]["path"], "k12_dataset/dataset.jsonl")
        replacement = b'{"reviewed":true}'
        revised = self.store.upload_delivery_file(self.reviewer, {
            "book": {"id": "BFINAL"}, "relative_path": "k12_dataset/dataset.jsonl",
            "content_base64": base64.b64encode(replacement).decode("ascii"), "sha256": hashlib.sha256(replacement).hexdigest(),
        })
        self.assertEqual(revised["bytes"], len(replacement))
        self.assertEqual((self.store.staff_data_root.parent / "final_deliveries" / "team-2" / "annotator-3" / "BFINAL" / "k12_dataset" / "dataset.jsonl").read_bytes(), replacement)
        manager_upload = self.store.upload_delivery_file(self.admin, {
            "book": {"id": "BFINAL"}, "relative_path": "manager-note.json",
            "content_base64": base64.b64encode(b"{}").decode("ascii"),
        })
        self.assertEqual(manager_upload["sha256"], hashlib.sha256(b"{}").hexdigest())

    def test_final_delivery_requires_full_review_approval(self):
        self.store.staff_data_root = Path(self.temp.name) / "private" / "staff_projects"
        self.store.sync_project_snapshot(self.annotator, {"book": {"id": "BGATE", "name": "门禁教材"}, "project": {"delivery_book_id": "BGATE"}})
        content = b'{"k12":true}'
        payload = {
            "book": {"id": "BGATE"}, "relative_path": "k12_dataset/dataset.jsonl",
            "content_base64": base64.b64encode(content).decode("ascii"), "sha256": hashlib.sha256(content).hexdigest(),
        }
        # 1) 从未提交审核：拒绝。
        with self.assertRaises(PermissionError):
            self.store.upload_delivery_file(self.annotator, payload)
        # 2) 已提交但审核未放行：拒绝。
        document = snapshot()
        document["book"] = {"id": "BGATE", "name": "门禁教材"}
        created = self.store.create_submission(self.annotator, document)
        with self.assertRaises(PermissionError):
            self.store.upload_delivery_file(self.annotator, payload)
        # 3) 审核员逐页放行后：允许。
        self.store.decide_page(self.reviewer, created["id"], "question-0", "approved", "")
        uploaded = self.store.upload_delivery_file(self.annotator, payload)
        self.assertEqual(uploaded["bytes"], len(content))
        # 4) 退回新页面后再次上传：重新被拒（最新提交必须全放行）。
        second = self.store.create_submission(self.annotator, document)
        with self.assertRaises(PermissionError):
            self.store.upload_delivery_file(self.annotator, payload)
        self.store.decide_page(self.reviewer, second["id"], "question-0", "approved", "")
        self.store.upload_delivery_file(self.annotator, payload)

    def test_reviewer_can_pull_snapshot_and_push_back_with_conflict_guard(self):
        self.store.staff_data_root = Path(self.temp.name) / "private" / "staff_projects"
        book = {"book": {"id": "BLOOP", "name": "循环教材"}}
        self.store.sync_project_snapshot(self.annotator, {**book, "project": {"delivery_book_id": "BLOOP", "project_revision": 5}})
        # 审核员可以完整下载员工的工程快照。
        snapshot_downloaded = self.store.project_snapshot(self.reviewer, "BLOOP")
        self.assertEqual(snapshot_downloaded["project_revision"], 5)
        self.assertEqual(snapshot_downloaded["assigned_annotator_id"], self.annotator["id"])
        # 基于旧版本的回传必须被 409 拒绝。
        with self.assertRaises(ConflictError):
            self.store.sync_project_snapshot(self.reviewer, {**book, "project": {"delivery_book_id": "BLOOP", "project_revision": 5}, "base_revision": 4})
        # 基于正确版本的回传成为下一版。
        pushed = self.store.sync_project_snapshot(self.reviewer, {**book, "project": {"delivery_book_id": "BLOOP", "project_revision": 5}, "base_revision": 5})
        self.assertEqual(pushed["project_revision"], 6)
        # 标注员拿旧版本覆盖会被拒；导入新版（revision 6）后可以继续同步。
        with self.assertRaises(ConflictError):
            self.store.sync_project_snapshot(self.annotator, {**book, "project": {"delivery_book_id": "BLOOP", "project_revision": 5}})
        merged = self.store.sync_project_snapshot(self.annotator, {**book, "project": {"delivery_book_id": "BLOOP", "project_revision": 6}})
        self.assertEqual(merged["project_revision"], 6)

    def test_annotator_cannot_push_without_base_revision_for_reviewer_role(self):
        # 审核员必须带 base_revision；缺失时按参数错误拒绝而不是覆盖。
        self.store.staff_data_root = Path(self.temp.name) / "private" / "staff_projects"
        book = {"book": {"id": "BNOREV", "name": "无版本教材"}}
        self.store.sync_project_snapshot(self.annotator, {**book, "project": {"delivery_book_id": "BNOREV", "project_revision": 1}})
        with self.assertRaises(ValueError):
            self.store.sync_project_snapshot(self.reviewer, {**book, "project": {"delivery_book_id": "BNOREV", "project_revision": 1}})

    def _seed_source_book(self, book_id: str = "BSRC") -> str:
        self.store.staff_data_root = Path(self.temp.name) / "private" / "staff_projects"
        self.store.projects_root = (Path(self.temp.name) / "Projects").resolve()
        self.store.projects_root.mkdir(parents=True, exist_ok=True)
        self.store.sync_project_snapshot(self.annotator, {
            "book": {"id": book_id, "name": "原书图片教材"},
            "project": {"delivery_book_id": book_id, "project_revision": 1},
        })
        return book_id

    def test_source_images_travel_with_the_snapshot_for_reviewers(self):
        book_id = self._seed_source_book()
        content = b"\xff\xd8" + b"page-scan" * 40
        uploaded = self.store.upload_source_file(self.annotator, book_id, {
            "relative_path": "教材/1/题目/1.jpg",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
        self.assertEqual(uploaded["path"], "教材/1/题目/1.jpg")
        self.assertEqual(uploaded["bytes"], len(content))
        # 审核员能列出并逐张取回，导入本地工作台时才能画框。
        listed = self.store.source_files(self.reviewer, book_id)
        self.assertEqual(len(listed["files"]), 1)
        stored = self.store.source_file_path(self.reviewer, book_id, "教材/1/题目/1.jpg")
        self.assertEqual(stored.read_bytes(), content)

    def test_source_image_upload_rejects_unsafe_paths_types_and_foreign_books(self):
        book_id = self._seed_source_book("BSAFE")
        content = b"\xff\xd8" + b"x" * 140
        encoded = base64.b64encode(content).decode("ascii")
        for bad_path in ("../outside.jpg", "/absolute.jpg", "notes.txt"):
            with self.assertRaises(ValueError):
                self.store.upload_source_file(self.annotator, book_id, {"relative_path": bad_path, "content_base64": encoded})
        with self.assertRaises(ValueError):
            self.store.upload_source_file(self.annotator, book_id, {
                "relative_path": "1.jpg", "content_base64": encoded, "sha256": "0" * 64,
            })
        # 未分配的教材、以及别组标注员，都看不到也传不了。
        with self.assertRaises(LookupError):
            self.store.upload_source_file(self.annotator, "BOTHER", {"relative_path": "1.jpg", "content_base64": encoded})
        other_admin = self.store.create_managed_user(self.super_admin, "other-admin", "very-secret-password", "admin")
        other_annotator = self.store.create_managed_user(other_admin, "other-marker", "very-secret-password", "annotator")
        with self.assertRaises(LookupError):
            self.store.source_files(other_annotator, book_id)

    def test_account_hierarchy_keeps_super_admin_separate(self):
        created_admin = self.store.create_managed_user(self.super_admin, "owner2", "very-secret-password", "admin")
        self.assertEqual(created_admin["role"], "admin")
        created_reviewer = self.store.create_managed_user(self.admin, "reviewer2", "very-secret-password", "reviewer")
        self.assertEqual(created_reviewer["role"], "reviewer")
        with self.assertRaises(PermissionError):
            self.store.create_managed_user(self.admin, "owner3", "very-secret-password", "admin")
        with self.assertRaises(PermissionError):
            self.store.create_managed_user(self.reviewer, "marker2", "very-secret-password", "annotator")

    def test_admin_can_edit_and_logically_delete_direct_accounts(self):
        updated = self.store.edit_user(self.admin, self.reviewer["id"], "reviewer-renamed", "annotator")
        self.assertEqual(updated["username"], "reviewer-renamed")
        self.assertEqual(updated["role"], "annotator")
        with self.assertRaises(ValueError):
            self.store.login("reviewer", "very-secret-password")
        _token, account = self.store.login("reviewer-renamed", "very-secret-password")
        self.assertEqual(account["role"], "annotator")
        self.store.delete_user(self.admin, self.reviewer["id"])
        self.assertNotIn(self.reviewer["id"], {item["id"] for item in self.store.list_users(self.admin)})
        with self.assertRaises(ValueError):
            self.store.login("reviewer-renamed", "very-secret-password")
        with self.assertRaises(LookupError):
            self.store.delete_user(self.admin, self.admin["id"])

    def test_legacy_boss_role_is_migrated_to_administrator(self):
        with closing(self.store._connect()) as connection:
            connection.execute("UPDATE users SET role='boss' WHERE id=?", (self.admin["id"],))
            connection.commit()
        migrated = ReviewStore(Path(self.temp.name))
        _token, account = migrated.login("admin", "very-secret-password")
        self.assertEqual(account["role"], "admin")

    def test_super_admin_can_set_and_clear_account_responsible_person(self):
        other_admin = self.store.create_managed_user(
            self.super_admin, "admin2", "very-secret-password", "admin"
        )
        updated = self.store.set_responsible_person(self.super_admin, self.annotator["id"], other_admin["id"])
        self.assertEqual(updated["manager_id"], other_admin["id"])
        self.assertEqual(updated["manager_name"], "admin2")
        cleared = self.store.set_responsible_person(self.super_admin, self.annotator["id"], None)
        self.assertIsNone(cleared["manager_id"])
        with self.assertRaises(ValueError):
            self.store.set_responsible_person(self.super_admin, other_admin["id"], self.admin["id"])
        with self.assertRaises(PermissionError):
            self.store.set_responsible_person(self.admin, self.annotator["id"], self.admin["id"])

    def test_teams_are_isolated_for_tasks_reviews_and_statistics(self):
        other_admin = self.store.create_managed_user(
            self.super_admin, "admin2", "very-secret-password", "admin"
        )
        other_marker = self.store.create_managed_user(
            other_admin, "marker2", "very-secret-password", "annotator"
        )
        other_reviewer = self.store.create_managed_user(
            other_admin, "reviewer2", "very-secret-password", "reviewer"
        )
        first_submission = self.store.create_submission(self.annotator, snapshot())
        second_submission = self.store.create_submission(other_marker, snapshot())

        self.assertEqual([row["id"] for row in self.store.list_submissions(self.admin)], [first_submission["id"]])
        self.assertEqual([row["id"] for row in self.store.list_submissions(other_reviewer)], [second_submission["id"]])
        with self.assertRaises(LookupError):
            self.store.submission(other_admin, first_submission["id"])
        with self.assertRaises(LookupError):
            self.store.decide_page(other_reviewer, first_submission["id"], "question-0", "approved", "")

        stats = self.store.stats(self.admin)
        self.assertEqual(stats["totals"]["submissions"], 1)
        self.assertEqual({row["username"] for row in stats["people"]}, {"admin", "marker", "reviewer"})

    def test_http_login_submit_preview_and_review(self):
        ReviewHandler.store = self.store
        ReviewHandler.static_root = Path(__file__).resolve().parents[1] / "online_review_beta" / "static"
        server = ThreadingHTTPServer(("127.0.0.1", 0), ReviewHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, method="GET", payload=None, token=""):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            data = json.dumps(payload).encode("utf-8") if payload is not None else None
            with urlopen(Request(base + path, data=data, method=method, headers=headers), timeout=4) as response:
                return response.read(), response.headers.get_content_type()

        try:
            raw, _ = request("/api/login", "POST", {"username": "marker", "password": "very-secret-password"})
            marker_token = json.loads(raw)["token"]
            raw, _ = request("/api/submissions", "POST", snapshot(), marker_token)
            submission_id = json.loads(raw)["submission"]["id"]
            raw, _ = request("/api/login", "POST", {"username": "reviewer", "password": "very-secret-password"})
            reviewer_token = json.loads(raw)["token"]
            preview, content_type = request(f"/api/submissions/{submission_id}/pages/question-0/preview", token=reviewer_token)
            self.assertEqual(content_type, "image/jpeg")
            self.assertTrue(preview.startswith(b"\xff\xd8"))
            raw, _ = request(f"/api/submissions/{submission_id}/pages/question-0/decision", "POST", {"status": "approved", "note": ""}, reviewer_token)
            self.assertEqual(json.loads(raw)["result"]["status"], "approved")
        finally:
            server.shutdown()
            server.server_close()
