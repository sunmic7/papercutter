"""传输层提速的护栏：二进制上传、gzip、长连接、预览中转区。

这些改动都在"员工保存/提交审核"的必经之路上，一旦回归就会表现为
"提交审核很慢"或"点了没反应"，所以用真服务器 + 真 socket 覆盖。
"""
import gzip
import hashlib
import http.client
import io
import json
import os
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from online_review_beta.server import ReviewStore, ReviewHandler, json_response
from review_tools import manual_annotation_data as data

BOOK_ID = "BTRANSFERTEST0001"


def _jpeg(seed: int = 0, size: int = 60) -> bytes:
    image = np.full((size, size, 3), seed % 255, dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


class _StubHandler:
    """Minimal stand-in for BaseHTTPRequestHandler used by json_response."""

    def __init__(self, accept_encoding: str = ""):
        self.headers = {"Accept-Encoding": accept_encoding}
        self.wfile = io.BytesIO()
        self.status = None
        self.sent = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.sent[key] = value

    def end_headers(self):
        pass


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ready_samples = os.environ.get("PAPERFLOW_DISABLE_READY_SAMPLES")
        os.environ["PAPERFLOW_DISABLE_READY_SAMPLES"] = "1"
        self.store = ReviewStore(
            self.root / "data",
            projects_root=self.root / "projects" / "library",
            staff_data_root=self.root / "staff",
        )
        self.boss = self.store.create_user("boss", "password", "super_admin")
        self.admin = self.store.create_managed_user(self.boss, "admin", "password", "admin")
        self.marker = self.store.create_managed_user(self.admin, "marker", "password", "annotator")
        self.reviewer = self.store.create_managed_user(self.admin, "reviewer", "password", "reviewer")
        self.store.set_responsible_person(self.boss, self.marker["id"], self.reviewer["id"])
        # 真实路径：员工同步一次工程，服务器才建立教材归属。
        self.store.sync_project_snapshot(self.marker, {
            "book": {"id": BOOK_ID, "name": "传输测试教材"},
            "project": {"delivery_book_id": BOOK_ID, "project_revision": 1,
                        "pages": [{"image": "p001.jpg", "status": "accepted", "questions": [
                            {"id": "q-1", "bbox": [1, 2, 30, 40], "text": "1+1="}]}],
                        "answer_pages": [], "pairs": []},
        })
        handler = type("TransferHandler", (ReviewHandler,), {"store": self.store,
                                                             "static_root": self.root})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.token = self.store.login("marker", "password")[0]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()
        if self.ready_samples is None:
            os.environ.pop("PAPERFLOW_DISABLE_READY_SAMPLES", None)
        else:
            os.environ["PAPERFLOW_DISABLE_READY_SAMPLES"] = self.ready_samples

    # ---- 长连接 -------------------------------------------------------
    def test_keep_alive_reuses_one_connection(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request("GET", "/api/health")
            first = connection.getresponse()
            self.assertEqual(200, first.status)
            first.read()
            socket_after_first = connection.sock
            connection.request("GET", "/api/health")
            second = connection.getresponse()
            self.assertEqual(200, second.status)
            second.read()
            # HTTP/1.1 下两次请求应当复用同一个 socket，否则每张缩略图都要重新握手。
            self.assertIsNotNone(socket_after_first)
            self.assertIs(socket_after_first, connection.sock)
        finally:
            connection.close()

    # ---- gzip --------------------------------------------------------
    def test_json_response_is_gzipped_when_accepted(self):
        handler = _StubHandler("gzip, deflate")
        payload = {"ok": True, "rows": ["行" * 40] * 60}
        json_response(handler, 200, payload)
        self.assertEqual("gzip", handler.sent.get("Content-Encoding"))
        self.assertEqual(len(handler.wfile.getvalue()), int(handler.sent["Content-Length"]))
        self.assertEqual(payload, json.loads(gzip.decompress(handler.wfile.getvalue())))

    def test_json_response_stays_plain_without_accept_encoding(self):
        handler = _StubHandler("")
        payload = {"ok": True, "rows": ["行" * 40] * 60}
        json_response(handler, 200, payload)
        self.assertNotIn("Content-Encoding", handler.sent)
        self.assertEqual(payload, json.loads(handler.wfile.getvalue()))

    def test_gzipped_request_body_is_accepted(self):
        body = gzip.compress(json.dumps({"username": "marker", "password": "password"}).encode("utf-8"))
        request = Request(self.url + "/api/login", data=body, method="POST",
                          headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
        with urlopen(request, timeout=10) as response:
            document = json.loads(response.read())
        self.assertTrue(document["ok"])
        self.assertIn("token", document)

    def test_gzipped_response_round_trips_over_http(self):
        request = Request(self.url + "/api/library",
                          headers={"Authorization": f"Bearer {self.token}", "Accept-Encoding": "gzip"})
        with urlopen(request, timeout=10) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
        self.assertTrue(json.loads(raw)["ok"])

    # ---- 原书图片二进制上传 ------------------------------------------
    def test_binary_source_upload_stores_exact_bytes(self):
        content = _jpeg(11, 80)
        digest = hashlib.sha256(content).hexdigest()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request(
                "POST", f"/api/projects/{BOOK_ID}/source-file?path=p001.jpg", body=content,
                headers={"Content-Type": "application/octet-stream",
                         "Content-Length": str(len(content)),
                         "Authorization": f"Bearer {self.token}",
                         "X-Paperflow-Sha256": digest})
            response = connection.getresponse()
            document = json.loads(response.read())
        finally:
            connection.close()
        self.assertEqual(201, response.status)
        self.assertTrue(document["ok"])
        self.assertEqual(digest, document["file"]["sha256"])
        stored = self.root / "projects" / "SourceImages" / BOOK_ID / "p001.jpg"
        self.assertEqual(content, stored.read_bytes())

    def test_binary_source_upload_rejects_wrong_digest(self):
        content = _jpeg(12, 70)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request(
                "POST", f"/api/projects/{BOOK_ID}/source-file?path=p002.jpg", body=content,
                headers={"Content-Type": "application/octet-stream",
                         "Content-Length": str(len(content)),
                         "Authorization": f"Bearer {self.token}",
                         "X-Paperflow-Sha256": "0" * 64})
            response = connection.getresponse()
            document = json.loads(response.read())
        finally:
            connection.close()
        self.assertEqual(400, response.status)
        self.assertIn("校验和", document["error"])

    # ---- 预览中转区 --------------------------------------------------
    def _snapshot(self, previews: dict | None, inline: str = "") -> tuple[dict, dict]:
        preview = previews or {}
        project = {
            "delivery_book_id": BOOK_ID,
            "project_revision": 1,
            "pages": [{"image": "p001.jpg", "status": "accepted", "questions": [
                {"id": "q-1", "bbox": [1, 2, 30, 40], "text": "1+1=", "minor_number": "1"}]}],
            "answer_pages": [], "pairs": [],
        }
        page = {"page_key": "question-0", "kind": "question", "page_index": 0,
                "image_name": "p001.jpg", "width": 60, "height": 60,
                "regions": [{"id": "q-1", "display_number": "1", "bbox": [1, 2, 30, 40]}]}
        if inline:
            page["preview"] = inline
        document = {"schema_version": 1, "book": {"id": BOOK_ID, "name": "传输测试教材"},
                    "project_revision": 1, "snapshot_fingerprint": "f" * 64, "pages": [page]}
        if preview:
            document["page_previews"] = {key: hashlib.sha256(value).hexdigest()
                                         for key, value in preview.items()}
        return document, project

    def test_submission_uses_staged_preview(self):
        payload = _jpeg(21, 64)
        digest = self.store.store_staged_preview(self.marker, BOOK_ID, hashlib.sha256(payload).hexdigest(),
                                                 payload)["sha256"]
        self.assertEqual([digest], self.store.staged_preview_digests(self.marker, BOOK_ID)["sha256"])
        document, project = self._snapshot({"question-0": payload})
        submission = self.store.create_submission(self.marker, document, project_document=project)
        stored = list((self.store.preview_root / submission["id"]).glob("*.jpg"))
        self.assertEqual(1, len(stored))
        self.assertEqual(payload, stored[0].read_bytes())

    def test_missing_preview_is_a_clear_error(self):
        document, project = self._snapshot(None)
        with self.assertRaises(ValueError) as caught:
            self.store.create_submission(self.marker, document, project_document=project)
        self.assertIn("预览缺失", str(caught.exception))

    def test_inline_preview_still_works(self):
        payload = _jpeg(22, 64)
        document, project = self._snapshot({"question-0": payload})
        document["pages"][0].pop("preview_sha256", None)
        document.pop("page_previews", None)
        document["pages"][0]["preview"] = "data:image/jpeg;base64," + __import__("base64").b64encode(
            payload).decode("ascii")
        submission = self.store.create_submission(self.marker, document, project_document=project)
        self.assertEqual(1, submission["page_count"])

    def test_preview_response_carries_etag_and_revalidates(self):
        payload = _jpeg(23, 64)
        self.store.store_staged_preview(self.marker, BOOK_ID, hashlib.sha256(payload).hexdigest(), payload)
        document, project = self._snapshot({"question-0": payload})
        submission = self.store.create_submission(self.marker, document, project_document=project)
        url = f"{self.url}/api/submissions/{submission['id']}/pages/question-0/preview"
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        try:
            connection.request("GET", url, headers={"Authorization": f"Bearer {self.token}"})
            first = connection.getresponse()
            first.read()
            etag = first.getheader("ETag")
            self.assertEqual(200, first.status)
            self.assertTrue(etag)
            self.assertIn("max-age", first.getheader("Cache-Control"))
            connection.request("GET", url, headers={"Authorization": f"Bearer {self.token}",
                                                    "If-None-Match": etag})
            second = connection.getresponse()
            second.read()
        finally:
            connection.close()
        self.assertEqual(304, second.status)


class ClientTransferTests(unittest.TestCase):
    """客户端侧：gzip 兼容老服务器、预览缓存命中。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_client_retries_plain_when_server_rejects_gzip(self):
        from review_tools import online_review_sync as sync

        seen: list[bool] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                gzipped = "gzip" in str(self.headers.get("Content-Encoding") or "")
                seen.append(gzipped)
                if gzipped:
                    # 老服务器：读不懂压缩体，直接 400。
                    json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False,
                                                                 "error": "请求 JSON 必须是对象。"})
                    return
                json_response(self, HTTPStatus.OK, {"ok": True, "accepted": True})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            big = {"blob": "x" * (400 * 1024)}
            result = sync._request_json(f"http://127.0.0.1:{server.server_port}/api/projects/sync",
                                        "POST", big)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual({"ok": True, "accepted": True}, result)
        self.assertEqual([True, False], seen)

    def test_preview_cache_round_trip(self):
        from review_tools import online_review_sync as sync

        cache = self.root / "cache"
        payload = _jpeg(31, 48)
        digest = hashlib.sha256(b"source-bytes").hexdigest()
        self.assertIsNone(sync._cached_preview(cache, digest, 1600))
        sync._store_preview(cache, digest, 1600, payload, 32, 24)
        cached = sync._cached_preview(cache, digest, 1600)
        self.assertIsNotNone(cached)
        self.assertEqual(payload, cached[0])
        self.assertEqual((32, 24), cached[1:])

    def test_staging_falls_back_for_old_server(self):
        from review_tools import online_review_sync as sync

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def do_GET(self):  # noqa: N802
                json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            snapshot = {"pages": [{"page_key": "question-0"}]}
            staged = sync.stage_submission_previews(f"http://127.0.0.1:{server.server_port}",
                                                    "token", BOOK_ID, snapshot,
                                                    {"question-0": _jpeg(32, 40)})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        # 老服务器没有中转区接口 —— 必须回退到内联 base64，而不是直接失败。
        self.assertFalse(staged)
        self.assertNotIn("page_previews", snapshot)


if __name__ == "__main__":
    unittest.main()
