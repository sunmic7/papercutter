"""Regression tests for concurrent submits, repeated submits, reviewer pushes,
administrator retries and the reviewer's full ownership of the JSON workspace.

These tests intentionally call the real ReviewStore (no patched create_submission),
so a missing frozen project is caught here rather than only in production.
"""
import base64
import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

from online_review_beta.central_library import MANAGERS
from online_review_beta.server import (
    ACTIVE_SUBMISSION_STATUSES, ConflictError, DELIVERY_ROLES, ReviewStore,
)


PREVIEW = "data:image/jpeg;base64," + base64.b64encode(
    b"\xff\xd8" + b"concurrency-preview" * 20).decode("ascii")


def review_document(book_id: str, revision: int, fingerprint: str = "") -> tuple[dict, dict]:
    fingerprint = fingerprint or hashlib.sha256(f"{book_id}:{revision}".encode()).hexdigest()
    document = {
        "schema_version": 1,
        "book": {"id": book_id, "name": f"教材 {book_id}"},
        "project_revision": revision,
        "snapshot_fingerprint": fingerprint,
        "pages": [{
            "page_key": "question-0", "kind": "question", "page_index": 0,
            "image_name": "001.jpg", "width": 1200, "height": 1800,
            "regions": [{"id": "q-1", "display_number": "1", "bbox": [20, 30, 400, 500]}],
            "preview": PREVIEW,
        }],
    }
    frozen = {
        "project_revision": revision,
        "delivery_book_id": book_id,
        "pages": [], "answer_pages": [], "pairs": [],
    }
    return document, frozen


class ReviewConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = ReviewStore(root / "data", staff_data_root=root / "staff")
        self.super_admin = self.store.create_user("superadmin", "very-secret-password", "super_admin")
        self.admin = self.store.create_managed_user(self.super_admin, "admin", "very-secret-password", "admin")
        self.annotator = self.store.create_managed_user(self.admin, "marker", "very-secret-password", "annotator")
        self.reviewer = self.store.create_managed_user(self.admin, "reviewer", "very-secret-password", "reviewer")
        self.store.set_responsible_person(self.super_admin, self.annotator["id"], self.reviewer["id"])

    def tearDown(self):
        self.temp.cleanup()

    def _submit(self, book_id="BOOK-A", revision=1, actor=None, fingerprint=""):
        document, frozen = review_document(book_id, revision, fingerprint)
        return self.store.create_submission(actor or self.annotator, document,
                                            project_document=frozen)

    def _rows(self, book_id, extra=""):
        with closing(self.store._connect()) as db:
            return [dict(row) for row in db.execute(
                "SELECT id,status,project_revision,snapshot_fingerprint FROM submissions WHERE book_id=? " + extra,
                (book_id,))]

    def _job(self, submission_id):
        with closing(self.store._connect()) as db:
            row = db.execute("SELECT * FROM delivery_jobs WHERE submission_id=?",
                             (submission_id,)).fetchone()
            return dict(row) if row else None

    def _status(self, submission_id):
        with closing(self.store._connect()) as db:
            return db.execute("SELECT status FROM submissions WHERE id=?",
                              (submission_id,)).fetchone()["status"]

    def _live(self, book_id):
        return [row for row in self._rows(book_id)
                if row["status"] not in {"resubmitted", "outdated"}]

    def test_every_submission_freezes_its_project(self):
        first = self._submit(revision=7)
        frozen = self.store.preview_root / first["id"] / "project.json"
        self.assertTrue(frozen.is_file())
        self.assertEqual(json.loads(frozen.read_text(encoding="utf-8"))["project_revision"], 7)
        self.assertEqual(json.loads(frozen.read_text(encoding="utf-8"))["delivery_book_id"], "BOOK-A")

    def test_submission_without_frozen_project_is_rejected(self):
        document, _frozen = review_document("BOOK-A", 1)
        with self.assertRaises(ValueError) as caught:
            self.store.create_submission(self.annotator, document)
        self.assertIn("冻结", str(caught.exception))

    def test_parallel_submissions_converge_to_one_live_task(self):
        results, errors = [], []

        def submit(index):
            try:
                results.append(self._submit(revision=index + 1,
                                            fingerprint=hashlib.sha256(str(index).encode()).hexdigest()))
            except Exception as error:  # pragma: no cover - surfaced below
                errors.append(error)

        threads = [threading.Thread(target=submit, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        self.assertEqual(len(self._live("BOOK-A")), 1)
        self.assertEqual(len(self._rows("BOOK-A")), 8)
        for result in results:
            self.assertTrue((self.store.preview_root / result["id"] / "project.json").is_file())

    def test_different_books_do_not_interfere(self):
        for book_id in ("BOOK-A", "BOOK-B", "BOOK-C"):
            self._submit(book_id=book_id)
        for book_id in ("BOOK-A", "BOOK-B", "BOOK-C"):
            self.assertEqual(len(self._live(book_id)), 1)

    def test_resubmission_cancels_unfinished_generation_job(self):
        first = self._submit(revision=1)
        self.store.decide_page(self.reviewer, first["id"], "question-0", "approved", "")
        self.assertEqual(self._job(first["id"])["status"], "queued")
        second = self._submit(revision=2)
        self.assertEqual(self._job(first["id"])["status"], "obsolete")
        self.assertIn(self._status(first["id"]), {"resubmitted", "outdated"})
        self.assertEqual(len(self._live("BOOK-A")), 1)
        self.assertEqual(self._live("BOOK-A")[0]["id"], second["id"])

    def test_reviewer_and_admin_can_retry_and_retry_is_idempotent(self):
        self.assertIn("reviewer", DELIVERY_ROLES)
        self.assertIn("reviewer", MANAGERS)
        first = self._submit(revision=1)
        self.store.decide_page(self.reviewer, first["id"], "question-0", "approved", "")
        with self.assertRaises(PermissionError):
            self.store.retry_delivery(self.annotator, "BOOK-A")
        job = self._job(first["id"])
        first_retry = self.store.retry_delivery(self.reviewer, "BOOK-A")
        second_retry = self.store.retry_delivery(self.admin, "BOOK-A")
        self.assertEqual(first_retry["id"], job["id"])
        self.assertEqual(second_retry["id"], job["id"])
        self.assertTrue(first_retry["already_queued"])
        self.assertEqual(self._job(first["id"])["attempts"], job["attempts"])
        self.assertTrue(self.store.delivery_preflight(first["id"])["ok"])

    def test_retry_preflight_rejects_unfreezable_history(self):
        first = self._submit(revision=1)
        (self.store.preview_root / first["id"] / "project.json").unlink()
        with closing(self.store._connect()) as db:
            db.execute("UPDATE submissions SET status='reviewed' WHERE id=?", (first["id"],))
            db.execute("UPDATE review_pages SET review_status='approved' WHERE submission_id=?",
                       (first["id"],))
            db.commit()
        preflight = self.store.delivery_preflight(first["id"])
        self.assertFalse(preflight["ok"])
        self.assertIn("冻结", preflight["reason"])
        with self.assertRaises(ConflictError) as caught:
            self.store.retry_delivery(self.admin, "BOOK-A")
        self.assertIn("冻结", str(caught.exception))
        self.assertIsNone(self._job(first["id"]))

    def test_project_edit_retires_stale_review_task(self):
        first = self._submit(revision=1)
        self.store.decide_page(self.reviewer, first["id"], "question-0", "approved", "")
        with closing(self.store._connect()) as db:
            retired = self.store.note_project_saved(db, "BOOK-A", 2, self.admin["id"])
            db.commit()
        self.assertEqual(retired, 1)
        self.assertEqual(self._status(first["id"]), "outdated")
        self.assertEqual(self._job(first["id"])["status"], "obsolete")

    def test_reviewer_push_creates_one_frozen_task_and_replay_is_idempotent(self):
        self.store.sync_project_snapshot(self.annotator, {
            "book": {"id": "BOOK-A", "name": "教材 BOOK-A"},
            "project": {"delivery_book_id": "BOOK-A", "project_revision": 1},
        })
        document, _frozen = review_document("BOOK-A", 1,
                                            hashlib.sha256(b"reviewer-push").hexdigest())
        payload = {
            "book": {"id": "BOOK-A", "name": "教材 BOOK-A"},
            "project": {"delivery_book_id": "BOOK-A", "project_revision": 1},
            "base_revision": 1,
            "review_snapshot": document,
        }
        first = self.store.sync_project_snapshot(self.reviewer, payload)
        self.assertTrue(first.get("submission"))
        self.assertEqual(first["project_revision"], 2)
        self.assertEqual(len(self._live("BOOK-A")), 1)
        submission_id = first["submission"]["id"]
        self.assertTrue((self.store.preview_root / submission_id / "project.json").is_file())
        # A lost response makes the client repeat the identical request.
        replay = self.store.sync_project_snapshot(self.reviewer, payload)
        self.assertTrue(replay["idempotent"])
        self.assertEqual(replay["submission"]["id"], submission_id)
        self.assertEqual(len(self._rows("BOOK-A")), 1)
        self.assertEqual(len(self._live("BOOK-A")), 1)

    def test_legacy_reviewer_push_without_snapshot_still_retires_tasks(self):
        self.store.sync_project_snapshot(self.annotator, {
            "book": {"id": "BOOK-A", "name": "教材 BOOK-A"},
            "project": {"delivery_book_id": "BOOK-A", "project_revision": 1},
        })
        created = self._submit(book_id="BOOK-A", revision=1)
        self.store.sync_project_snapshot(self.reviewer, {
            "book": {"id": "BOOK-A", "name": "教材 BOOK-A"},
            "project": {"delivery_book_id": "BOOK-A", "project_revision": 1},
            "base_revision": 1,
        })
        self.assertEqual(len(self._live("BOOK-A")), 0)
        self.assertEqual(self._status(created["id"]), "outdated")

    def test_active_status_constant_covers_every_review_state(self):
        self.assertEqual(set(ACTIVE_SUBMISSION_STATUSES),
                         {"pending", "assigned", "in_review", "reviewed",
                          "returned", "needs_fix"})

    def test_reviewer_delivery_workspace_matches_json_menu_scope(self):
        """A book listed by ``scope=all`` must also open its delivery detail.

        The reviewer is not accountable for ``marker2``; previously the row was
        visible in the JSON menu but the artifact/batch endpoints returned 404.
        """
        outsider = self.store.create_managed_user(
            self.admin, "marker2", "very-secret-password", "annotator")
        self.store.sync_project_snapshot(outsider, {
            "book": {"id": "BOOK-B", "name": "教材 BOOK-B"},
            "project": {"delivery_book_id": "BOOK-B", "project_revision": 1},
        })
        listing = self.store.delivery_files(self.reviewer, "BOOK-B")
        self.assertEqual(listing["book_id"], "BOOK-B")
        self.assertEqual(listing["files"], [])
        # Annotators still cannot inspect somebody else's private delivery.
        with self.assertRaises((PermissionError, LookupError)):
            self.store.delivery_files(self.annotator, "BOOK-B")
        # The reviewer passes the role check and is stopped only by the review
        # approval gate, which must stay intact for every delivery write.
        with self.assertRaises(PermissionError) as caught:
            self.store.begin_delivery_batch(self.reviewer, {
                "book": {"id": "BOOK-B"},
                "files": [{"relative_path": "images/a.json", "bytes": 2,
                           "sha256": hashlib.sha256(b"{}").hexdigest()}],
            })
        self.assertIn("审核", str(caught.exception))
        self.assertNotIn("无权上传最终交付", str(caught.exception))
