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
from unittest.mock import patch

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
    def test_real_names_are_independent_of_login_and_keep_sessions_and_ownership(self):
        for actor in (self.annotator,self.reviewer,self.admin,self.super_admin):
            token,_=self.store.login(actor['username'],'very-secret-password')
            result=self.store.set_real_name(actor,actor['id'],'  同一个姓名  ')
            self.assertEqual(result['username'],actor['username'])
            self.assertEqual(result['real_name'],'同一个姓名')
            self.assertEqual(result['role'],actor['role'])
            self.assertEqual(result['manager_id'],actor.get('manager_id'))
            self.assertEqual(self.store.user_for_token(token)['real_name'],'同一个姓名')
            self.assertEqual(self.store.login(actor['username'],'very-secret-password')[1]['real_name'],'同一个姓名')

    def test_real_name_rank_matrix_includes_peers_and_foreign_team_staff(self):
        peers=[self.store.create_user('peer-'+role,'test-password',role) for role in ('annotator','reviewer','admin','super_admin')]
        ranks={'annotator':0,'reviewer':1,'admin':2,'super_admin':3}
        for actor in (self.annotator,self.reviewer,self.admin,self.super_admin):
            for target in peers:
                allowed=actor['role']!='annotator' and ranks[actor['role']]>=ranks[target['role']]
                if allowed:self.assertEqual(self.store.set_real_name(actor,target['id'],'姓名测试')['real_name'],'姓名测试')
                else:
                    with self.assertRaises(PermissionError):self.store.set_real_name(actor,target['id'],'禁止修改')
        visible={row['id'] for row in self.store.list_users(self.reviewer)}
        self.assertIn(peers[0]['id'],visible)
        self.assertIn(peers[1]['id'],visible)
        self.assertNotIn(peers[3]['id'],visible)

    def test_real_name_validation_clear_and_deleted_account(self):
        for value in (None,123,'甲'*65,'张\n三','张\x00三'):
            with self.assertRaises(ValueError):self.store.set_real_name(self.annotator,self.annotator['id'],value)
        self.store.set_real_name(self.annotator,self.annotator['id'],'张三')
        self.assertEqual(self.store.set_real_name(self.annotator,self.annotator['id'],'')['real_name'],'')
        other=self.store.create_managed_user(self.admin,'deleted-name-test','test-password','annotator')
        self.store.delete_user(self.admin,other['id'])
        with self.assertRaises(PermissionError):self.store.set_real_name(self.admin,other['id'],'不可恢复')

    def test_real_names_follow_author_and_responsible_display_without_renaming_accounts(self):
        self.store.set_real_name(self.annotator,self.annotator['id'],'张三')
        self.store.set_real_name(self.reviewer,self.reviewer['id'],'李四')
        task=self.store.create_submission(self.annotator,snapshot())
        self.assertEqual(self.store.list_submissions(self.admin)[0]['author_name'],'marker（张三）')
        self.assertEqual(self.store.submission(self.admin,task['id'])['author_name'],'marker（张三）')
        row=next(r for r in self.store.list_users(self.admin) if r['id']==self.annotator['id'])
        self.assertEqual(row['username'],'marker')
        self.assertEqual(row['responsible_name'],'reviewer（李四）')
        self.store.set_real_name(self.super_admin,self.super_admin['id'],'隐藏的姓名')
        self.assertNotIn('隐藏的姓名',json.dumps(self.store.list_users(self.reviewer),ensure_ascii=False))

    def test_legacy_database_real_name_column_is_added_without_changing_existing_accounts(self):
        with closing(self.store._connect()) as db:
            db.execute('ALTER TABLE users DROP COLUMN real_name');db.commit()
        migrated=ReviewStore(Path(self.temp.name))
        self.assertEqual(migrated.login('marker','very-secret-password')[1]['real_name'],'')
        self.assertEqual(migrated.login('reviewer','very-secret-password')[1]['id'],self.reviewer['id'])
    def test_preview_checks_current_access_without_loading_full_submission(self):
        task = self.store.create_submission(self.annotator, snapshot())
        with patch.object(self.store, "submission", side_effect=AssertionError("full detail read")):
            for actor in (self.annotator, self.reviewer, self.admin):
                self.assertTrue(self.store.preview_path(actor, task["id"], "question-0").is_file())
            with self.assertRaises(LookupError):
                self.store.preview_path(self.reviewer, task["id"], "missing")
            other = self.store.create_user("foreign-admin-preview", "password", "admin")
            self.assertTrue(self.store.preview_path(other,task['id'],'question-0').is_file())
            outsider=self.store.create_user('foreign-reviewer-preview','password','reviewer')
            self.assertTrue(self.store.preview_path(outsider,task['id'],'question-0').is_file())

    def test_book_filtered_submissions_keep_role_scope(self):
        task = self.store.create_submission(self.annotator, snapshot())
        for actor in (self.annotator, self.reviewer, self.admin):
            self.assertEqual([row["id"] for row in self.store.list_submissions(actor, book_id="book-test")], [task["id"]])
            self.assertEqual(self.store.list_submissions(actor, book_id="not-this-book"), [])
        other = self.store.create_user("foreign-admin-filter", "password", "admin")
        self.assertEqual([r['id'] for r in self.store.list_submissions(other,book_id='book-test')],[task['id']])
        outsider=self.store.create_user('foreign-reviewer-filter','password','reviewer')
        self.assertEqual(self.store.list_submissions(outsider,book_id='book-test'),[])

    def test_admin_can_comment_on_unsubmitted_annotations_and_staff_can_read(self):
        book_id = self._seed_source_book("BOPINION")
        result = self.store.add_annotation_opinion(self.admin, book_id, {
            "body": "请核对第3题的题图边界", "kind": "question", "page_index": 0,
            "target_label": "3-2", "project_revision": 1})
        self.assertGreater(result["id"], 0)
        for actor in (self.annotator, self.reviewer, self.admin):
            data = self.store.annotation_opinions(actor, book_id)
            self.assertEqual(data["opinions"][0]["body"], "请核对第3题的题图边界")
            self.assertEqual(data["can_comment"], actor["role"] == "admin")
        with self.assertRaises(PermissionError):
            self.store.add_annotation_opinion(self.reviewer, book_id, {"body": "无权发表"})

    def test_opinions_do_not_change_review_decisions_and_reject_foreign_books(self):
        document = snapshot()
        task = self.store.create_submission(self.annotator, document)
        self.store.add_annotation_opinion(self.admin, document["book"]["id"], {
            "body": "请核对文字", "submission_id": task["id"], "kind": "question", "page_index": 0})
        detail = self.store.submission(self.reviewer, task["id"])
        self.assertEqual(detail["status"], "pending")
        self.assertEqual(detail["pages"][0]["review_status"], "pending")
        other_admin = self.store.create_managed_user(self.super_admin, "foreign-admin", "test-password", "admin")
        self.assertTrue(self.store.annotation_opinions(other_admin, document['book']['id'])['can_comment'])
        self.store.add_annotation_opinion(other_admin, document['book']['id'], {'body':'管理员跨团队复核意见'})
        with self.assertRaises(ValueError):
            self.store.add_annotation_opinion(self.admin, document["book"]["id"], {"body": " "})
        with self.assertRaises(ValueError):
            self.store.add_annotation_opinion(self.admin, document["book"]["id"], {"body": "页码错误", "kind": "question", "page_index": -1})

    def test_review_submission_uses_original_library_title_instead_of_storage_id(self):
        book_id = "B0123456789ABCDEF0123456789ABCDEF"
        original = "金版卷王数学六年级下册北师大版"
        with closing(self.store._connect()) as connection:
            connection.execute("INSERT INTO textbook_library VALUES(?,?,?,?,?)", (book_id, original, original, 1, 100))
            connection.commit()
        document = snapshot()
        document["book"] = {"id": book_id, "name": book_id}
        task = self.store.create_submission(self.annotator, document)
        self.assertEqual(self.store.submission(self.reviewer, task["id"])["book_name"], original)
        with closing(self.store._connect()) as connection:
            connection.execute("UPDATE submissions SET book_name=? WHERE id=?", (book_id, task["id"]))
            connection.commit()
        self.assertEqual(self.store.repair_book_names(), 1)
        self.assertEqual(self.store.submission(self.reviewer, task["id"])["book_name"], original)
        self.assertEqual(self.store.repair_book_names(), 0)

    def test_old_submission_title_recovers_a_new_id_title_for_an_uploaded_book(self):
        document = snapshot()
        document["book"] = {"id": "BUPLOADEDTITLE", "name": "原书名（第一版）"}
        self.store.create_submission(self.annotator, document)
        document["book"]["name"] = "BUPLOADEDTITLE"
        newer = self.store.create_submission(self.annotator, document)
        self.assertEqual(self.store.submission(self.reviewer, newer["id"])["book_name"], "原书名（第一版）")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ReviewStore(Path(self.temp.name))
        self.super_admin = self.store.create_user("superadmin", "very-secret-password", "super_admin")
        self.admin = self.store.create_managed_user(self.super_admin, "admin", "very-secret-password", "admin")
        self.annotator = self.store.create_managed_user(self.admin, "marker", "very-secret-password", "annotator")
        self.reviewer = self.store.create_managed_user(self.admin, "reviewer", "very-secret-password", "reviewer")
        self.store.set_responsible_person(self.super_admin, self.annotator["id"], self.reviewer["id"])
        # Legacy cases submit bare review snapshots.  Production always freezes
        # the annotation project alongside a submission, so give those cases a
        # minimal frozen document that tracks whatever revision they use.
        raw_create = self.store.create_submission
        def create_with_freeze(actor, document, **kwargs):
            if isinstance(document, dict) and kwargs.get("project_document") is None and not document.get("project_document"):
                document = {**document, "project_document": {
                    "project_revision": document.get("project_revision", 0), "pages": []}}
            return raw_create(actor, document, **kwargs)
        self.store.create_submission = create_with_freeze

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
        with self.assertRaises((PermissionError, LookupError)):
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

    def test_all_roles_change_own_password_and_revoke_existing_sessions(self):
        for actor in (self.annotator, self.reviewer, self.admin, self.super_admin):
            with self.subTest(role=actor['role']):
                old_token, _ = self.store.login(actor['username'], 'very-secret-password')
                with self.assertRaises(ValueError):
                    self.store.change_own_password(actor, 'wrong-current', 'new-secret-password')
                self.store.change_own_password(actor, 'very-secret-password', 'new-secret-password')
                with self.assertRaises(ValueError):
                    self.store.login(actor['username'], 'very-secret-password')
                _, account = self.store.login(actor['username'], 'new-secret-password')
                self.assertEqual(account['role'], actor['role'])
                self.assertIsNone(self.store.user_for_token(old_token))

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

    def test_inactive_accounts_are_listed_after_active_accounts(self):
        self.store.manage_user(self.admin, self.reviewer["id"], {"active": False})
        accounts = self.store.list_users(self.admin)
        active_flags = [bool(account["active"]) for account in accounts]
        self.assertEqual(active_flags, sorted(active_flags, reverse=True))

    def test_annotator_cannot_review_and_negative_decisions_allow_empty_notes(self):
        created = self.store.create_submission(self.annotator, snapshot())
        with self.assertRaises(PermissionError):
            self.store.decide_page(self.annotator, created["id"], "question-0", "approved", "")
        for status in ("returned", "needs_fix"):
            result = self.store.decide_page(self.reviewer, created["id"], "question-0", status, "")
            self.assertEqual(result["submission_status"], status)
            self.assertEqual(self.store.decisions(self.annotator, created["id"])["decisions"][0]["review_note"], "")

    def test_partial_review_stays_in_review_until_all_pages_have_decisions(self):
        document = snapshot()
        document["pages"].append({**document["pages"][0], "page_key": "question-1", "page_index": 1})
        created = self.store.create_submission(self.annotator, document)
        task_id = created["id"]
        self.assertEqual(self.store.list_submissions(self.reviewer)[0]["pending_pages"], 2)
        for negative in ("needs_fix", "returned"):
            self.store.decide_page(self.reviewer, task_id, "question-0", negative, "")
            self.assertEqual(self.store.submission(self.reviewer, task_id)["status"], "in_review")
        self.assertEqual(self.store.list_submissions(self.reviewer)[0]["pending_pages"], 1)
        self.store.decide_page(self.reviewer, task_id, "question-1", "needs_fix", "")
        self.assertEqual(self.store.submission(self.reviewer, task_id)["status"], "returned")
        self.store.decide_page(self.reviewer, task_id, "question-0", "approved", "")
        self.assertEqual(self.store.submission(self.reviewer, task_id)["status"], "needs_fix")
        self.store.decide_page(self.reviewer, task_id, "question-1", "approved", "")
        self.assertEqual(self.store.submission(self.reviewer, task_id)["status"], "reviewed")

    def test_new_submission_archives_pending_snapshot_and_blocks_old_decisions(self):
        first = self.store.create_submission(self.annotator, snapshot())
        second = self.store.create_submission(self.annotator, snapshot())
        self.assertEqual(second["supersedes_submission_id"], first["id"])
        self.assertEqual(self.store.submission(self.reviewer, first["id"])["status"], "resubmitted")
        with self.assertRaises(ConflictError):
            self.store.decide_page(self.reviewer, first["id"], "question-0", "approved", "")
        self.assertEqual(self.store.stats(self.admin)["page_statuses"]["pending"], 1)

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

    def test_reviewer_can_remove_progress_without_losing_project_history(self):
        projects_root = Path(self.temp.name) / "Projects"
        self.store.projects_root = projects_root.resolve()
        self.store.staff_data_root = projects_root / "StaffProjects"
        document = {
            "book": {"id": "BHISTORY", "name": "历史教材"},
            "project": {
                "delivery_book_id": "BHISTORY", "project_revision": 1,
                "pages": [{"status": "accepted", "questions": [{"id": "q-1", "bbox": [1, 2, 3, 4]}]}],
                "answer_pages": [], "pairs": [],
            },
        }
        self.store.sync_project_snapshot(self.annotator, document)
        self.assertEqual([row["book_id"] for row in self.store.list_projects(self.reviewer)], ["BHISTORY"])

        removed = self.store.remove_project_progress(self.reviewer, "BHISTORY")
        self.assertEqual(removed["book_id"], "BHISTORY")
        self.assertEqual(self.store.list_projects(self.reviewer), [])
        history = self.store.list_project_history(self.reviewer)
        self.assertEqual([row["book_id"] for row in history], ["BHISTORY"])
        self.assertTrue(history[0]["progress_deleted_at"])
        self.assertEqual(self.store.project_detail(self.reviewer, "BHISTORY")["book_id"], "BHISTORY")
        with self.assertRaises(PermissionError):
            self.store.remove_project_progress(self.annotator, "BHISTORY")

        # A later worker sync is active work again, so it returns to the board
        # while its continuous project history remains intact.
        self.store.sync_project_snapshot(self.annotator, document)
        self.assertEqual([row["book_id"] for row in self.store.list_projects(self.reviewer)], ["BHISTORY"])
        self.assertIsNone(self.store.list_project_history(self.reviewer)[0]["progress_deleted_at"])

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

    def test_final_location_config_survives_restart_and_preserves_access_checks(self):
        final = Path(self.temp.name) / '最终交付'
        (Path(self.temp.name) / 'storage_config.json').write_text(
            json.dumps({'final_delivery_root': str(final)}, ensure_ascii=False), encoding='utf-8-sig')
        self.store = ReviewStore(Path(self.temp.name))
        self.assertEqual(self.store.delivery_root, final)
        self.test_annotator_can_upload_only_own_final_delivery_file()
        reopened = ReviewStore(Path(self.temp.name), staff_data_root=self.store.staff_data_root)
        self.assertEqual(reopened.delivery_root, final)
        self.assertTrue(reopened.delivery_files(self.reviewer, 'BFINAL')['files'])
        self.assertFalse((Path(self.temp.name) / 'private/final_deliveries').exists())

    def _annotator_delivery_archive_checks(self):
        self.store.staff_data_root = Path(self.temp.name) / "private" / "staff_projects"
        self.store.sync_project_snapshot(self.annotator, {"book": {"id": "BFINAL", "name": "交付教材"}, "project": {"delivery_book_id": "BFINAL"}})
        content = b'{"ok":true}'
        uploaded = self.store.upload_delivery_file(self.annotator, {
            "book": {"id": "BFINAL"}, "relative_path": "k12_dataset/dataset.jsonl",
            "content_base64": base64.b64encode(content).decode("ascii"), "sha256": hashlib.sha256(content).hexdigest(),
        })
        self.assertEqual(uploaded["bytes"], len(content))
        expected = ((self.store.delivery_root / 'BFINAL') if self.store.delivery_root else
                    self.store.staff_data_root.parent / 'final_deliveries' / 'team-2' / 'annotator-3' / 'BFINAL')
        self.assertTrue((expected / 'k12_dataset/dataset.jsonl').is_file())
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
        self.assertEqual((expected / 'k12_dataset/dataset.jsonl').read_bytes(), replacement)
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

    def test_reviewer_push_keeps_old_decisions_but_requires_new_review_snapshot(self):
        book_id = self._seed_source_book("BLIFECYCLE")
        document = snapshot()
        document["book"]["id"] = book_id
        document["project_revision"] = 1
        first = self.store.create_submission(self.annotator, document)
        self.store.decide_page(self.reviewer, first["id"], "question-0", "approved", "已核对")
        self.store.sync_project_snapshot(self.reviewer, {
            "book": document["book"], "project": {"delivery_book_id": book_id, "project_revision": 1},
            "base_revision": 1,
        })
        old = self.store.submission(self.reviewer, first["id"])
        self.assertEqual(old["status"], "outdated")
        self.assertFalse(old["can_review"])
        self.assertEqual(old["pages"][0]["review_status"], "approved")
        self.assertEqual(old["pages"][0]["review_note"], "已核对")
        with self.assertRaises(ConflictError):
            self.store.decide_page(self.reviewer, first["id"], "question-0", "approved", "")
        self.assertNotIn("approved", self.store.stats(self.admin)["page_statuses"])
        document["project_revision"] = 2
        newer = self.store.create_submission(self.reviewer, document, reviewer_refresh=True)
        self.assertEqual(newer["supersedes_submission_id"], first["id"])
        self.assertIn(self.store.submission(self.reviewer, first["id"])["status"], {"resubmitted", "outdated"})
        self.assertEqual(self.store.submission(self.reviewer, newer["id"])["status"], "pending")

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
        self.assertEqual(self.store.create_managed_user(self.admin,'owner3','very-secret-password','admin')['role'],'admin')
        with self.assertRaises(PermissionError):self.store.create_managed_user(self.admin,'forbidden-super','very-secret-password','super_admin')
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
        self.assertEqual(updated["manager_id"], self.admin["id"])
        self.assertEqual(updated["responsible_id"], other_admin["id"])
        self.assertEqual(updated["responsible_name"], "admin2")
        cleared = self.store.set_responsible_person(self.super_admin, self.annotator["id"], None)
        self.assertIsNone(cleared["responsible_id"])
        with self.assertRaises(ValueError):
            self.store.set_responsible_person(self.super_admin, self.annotator["id"], self.super_admin["id"])
        updated = self.store.set_responsible_person(self.admin, self.annotator["id"], self.admin["id"])
        self.assertEqual(updated["responsible_id"], self.admin["id"])
        self.assertEqual(self.store.set_responsible_person(other_admin,self.annotator['id'],other_admin['id'])['responsible_id'],other_admin['id'])

    def test_concurrent_page_review_rejects_stale_screen(self):
        task = self.store.create_submission(self.annotator, snapshot())
        self.store.decide_page(self.reviewer, task["id"], "question-0", "needs_fix", "", expected_reviewed_at="")
        with self.assertRaises(ConflictError):
            self.store.decide_page(self.reviewer, task["id"], "question-0", "approved", "", expected_reviewed_at="")
        page = self.store.submission(self.reviewer, task["id"])["pages"][0]
        self.store.decide_page(self.reviewer, task["id"], "question-0", "approved", "", expected_reviewed_at=page["reviewed_at"])
        self.assertEqual(self.store.submission(self.reviewer, task["id"])["status"], "reviewed")

    def test_history_migration_and_partial_review_repair_preserve_page_decisions(self):
        first = self.store.create_submission(self.annotator, snapshot())
        second = self.store.create_submission(self.annotator, snapshot())
        with closing(self.store._connect()) as connection:
            connection.execute("UPDATE submissions SET status='pending' WHERE id=?", (first["id"],))
            connection.commit()
        migrated = ReviewStore(Path(self.temp.name))
        self.assertEqual(migrated.submission(self.reviewer, first["id"])["status"], "resubmitted")
        self.assertEqual(migrated.submission(self.reviewer, second["id"])["status"], "pending")
        self.assertEqual(migrated.stats(self.admin)["page_statuses"]["pending"], 1)

    def test_reviewer_can_change_own_team_annotator_responsible_without_moving_team(self):
        other_admin = self.store.create_managed_user(
            self.super_admin, "admin2", "very-secret-password", "admin"
        )
        other_annotator = self.store.create_managed_user(
            other_admin, "marker2", "very-secret-password", "annotator"
        )
        other_reviewer = self.store.create_managed_user(
            self.admin, "reviewer2", "very-secret-password", "reviewer"
        )
        visible = {row["id"] for row in self.store.list_users(self.reviewer)}
        self.assertIn(self.annotator["id"], visible)
        self.assertIn(self.admin["id"], visible)
        self.assertIn(other_admin["id"], visible)
        self.assertIn(other_reviewer["id"], visible)
        self.assertNotIn(self.super_admin["id"], visible)
        with self.assertRaises(ValueError):
            self.store.set_responsible_person(self.reviewer, self.annotator["id"], None)
        updated = self.store.set_responsible_person(self.reviewer, self.annotator["id"], other_reviewer["id"])
        self.assertEqual(updated["manager_id"], self.admin["id"])
        self.assertEqual(updated["responsible_id"], other_reviewer["id"])
        with self.assertRaises(LookupError):
            self.store.set_responsible_person(self.reviewer, other_annotator["id"], self.admin["id"])
        with self.assertRaises(PermissionError):
            self.store.manage_user(self.reviewer, self.annotator["id"], {"active": False})

    def test_administrators_have_global_visibility_but_reviewers_remain_scoped(self):
        other_admin = self.store.create_managed_user(
            self.super_admin, "admin2", "very-secret-password", "admin"
        )
        other_marker = self.store.create_managed_user(
            other_admin, "marker2", "very-secret-password", "annotator"
        )
        other_reviewer = self.store.create_managed_user(
            other_admin, "reviewer2", "very-secret-password", "reviewer"
        )
        self.store.set_responsible_person(self.super_admin, other_marker["id"], other_reviewer["id"])
        first_submission = self.store.create_submission(self.annotator, snapshot())
        # A book has exactly one live task, so a second concurrent worker must
        # submit a different book (otherwise it would correctly retire first).
        second_document = snapshot()
        second_document["book"] = {"id": "book-other-team", "name": "另一本教材"}
        second_submission = self.store.create_submission(other_marker, second_document)

        self.assertEqual({row['id'] for row in self.store.list_submissions(self.admin)}, {first_submission['id'],second_submission['id']})
        self.assertEqual([row["id"] for row in self.store.list_submissions(other_reviewer)], [second_submission["id"]])
        self.assertEqual(self.store.submission(other_admin,first_submission['id'])['id'],first_submission['id'])
        with self.assertRaises(PermissionError):
            self.store.decide_page(other_reviewer, first_submission["id"], "question-0", "approved", "")

        stats = self.store.stats(self.admin)
        self.assertEqual(stats['totals']['submissions'],2)
        self.assertEqual({row['username'] for row in stats['people']},{'admin','marker','reviewer','admin2','marker2','reviewer2'})

    def test_admin_manages_foreign_reviewers_without_reparenting_or_super_admin_access(self):
        outsider=self.store.create_managed_user(self.super_admin,'gl1-example','test-password','admin')
        self.assertIn(self.reviewer['id'],{row['id'] for row in self.store.list_users(outsider)})
        self.assertNotIn(self.super_admin['id'],{row['id'] for row in self.store.list_users(outsider)})
        changed=self.store.manage_user(outsider,self.reviewer['id'],{'active':False})
        self.assertFalse(changed['active'])
        changed=self.store.manage_user(outsider,self.reviewer['id'],{'active':True,'password':'replacement'})
        self.assertEqual(changed['manager_id'],self.admin['id'])
        renamed=self.store.edit_user(outsider,self.reviewer['id'],'reviewer-updated','reviewer')
        self.assertEqual(renamed['manager_id'],self.admin['id'])
        with self.assertRaises(LookupError):self.store.manage_user(outsider,self.super_admin['id'],{'active':False})
        self.assertFalse(self.store.manage_user(outsider,self.admin['id'],{'active':False})['active'])
        self.store.delete_user(outsider,self.reviewer['id'])
        self.assertNotIn(self.reviewer['id'],{row['id'] for row in self.store.list_users(outsider)})

    def test_reviewer_lists_only_annotators_for_which_it_is_responsible(self):
        other_marker = self.store.create_managed_user(
            self.admin, "marker-with-other-owner", "very-secret-password", "annotator"
        )
        first = self.store.create_submission(self.annotator, snapshot())
        second = self.store.create_submission(other_marker, snapshot())
        self.assertEqual([row["id"] for row in self.store.list_submissions(self.reviewer)], [first["id"]])
        self.assertFalse(self.store.submission(self.reviewer,second['id'])['can_review'])

    def test_admin_has_all_account_operations_except_super_admin_targets(self):
        new_admin=self.store.create_managed_user(self.admin,'peer-admin','test-password','admin')
        self.assertEqual(self.store.edit_user(self.admin,new_admin['id'],'peer-renamed','admin')['username'],'peer-renamed')
        self.assertFalse(self.store.manage_user(self.admin,new_admin['id'],{'active':False})['active'])
        self.store.manage_user(self.admin,new_admin['id'],{'active':True,'password':'new-password'})
        self.store.delete_user(self.admin,new_admin['id'])
        for operation in (lambda:self.store.manage_user(self.admin,self.super_admin['id'],{'active':False}),
                          lambda:self.store.edit_user(self.admin,self.super_admin['id'],'forbidden','admin'),
                          lambda:self.store.delete_user(self.admin,self.super_admin['id'])):
            with self.assertRaises((LookupError,PermissionError)):operation()
        with self.assertRaises(PermissionError):self.store.manage_user(self.admin,self.reviewer['id'],{'role':'super_admin'})
        self.assertFalse(self.store.manage_user(self.super_admin,self.admin['id'],{'active':False})['active'])

    def test_reviewer_scope_defaults_to_mine_but_allows_all_read_only(self):
        other_admin=self.store.create_managed_user(self.super_admin,'scope-admin','test-password','admin')
        other_marker=self.store.create_managed_user(other_admin,'scope-marker','test-password','annotator')
        first=self.store.create_submission(self.annotator,snapshot())
        second=self.store.create_submission(other_marker,snapshot())
        self.assertEqual([r['id'] for r in self.store.list_submissions(self.reviewer)],[first['id']])
        self.assertEqual({r['id'] for r in self.store.list_submissions(self.reviewer,scope='all')},{first['id'],second['id']})
        self.assertFalse(self.store.submission(self.reviewer,second['id'])['can_review'])
        self.assertTrue(self.store.preview_path(self.reviewer,second['id'],'question-0').is_file())
        with self.assertRaises(PermissionError):self.store.decide_page(self.reviewer,second['id'],'question-0','approved','')
        self.assertEqual([r['id'] for r in self.store.list_submissions(self.annotator,scope='all')],[first['id']])
        with self.assertRaises(ValueError):self.store.list_submissions(self.reviewer,scope='invalid')

    def test_reviewer_global_project_read_does_not_allow_other_project_writes(self):
        book_id=self._seed_source_book('BREADALL')
        other_reviewer=self.store.create_managed_user(self.admin,'scope-reviewer','test-password','reviewer')
        self.store.set_responsible_person(self.super_admin,self.annotator['id'],other_reviewer['id'])
        self.assertFalse(any(r['book_id']==book_id for r in self.store.list_projects(self.reviewer)))
        self.assertTrue(any(r['book_id']==book_id for r in self.store.list_projects(self.reviewer,scope='all')))
        self.assertFalse(self.store.project_detail(self.reviewer,book_id)['can_manage'])
        self.assertEqual(self.store.project_snapshot(self.reviewer,book_id)['book_id'],book_id)
        self.store.source_files(self.reviewer,book_id)
        with self.assertRaises(PermissionError):self.store.remove_project_progress(self.reviewer,book_id)

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
