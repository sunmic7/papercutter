import base64
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import numpy as np

from online_review_beta.server import ReviewStore, ReviewHandler, ConflictError
from review_tools import online_review_sync as sync
from review_tools import manual_annotation_data as data
from review_tools.review_session import ReviewSession, REVIEWERS
from desktop_annotation import paperflow_annotation_api as api


class CompanyReviewSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ReviewStore(self.root / 'data', staff_data_root=self.root / 'staff')
        self.boss = self.store.create_user('boss', 'password', 'super_admin')
        self.admin = self.store.create_managed_user(self.boss, 'admin', 'password', 'admin')
        self.marker = self.store.create_managed_user(self.admin, 'marker', 'password', 'annotator')
        self.reviewer = self.store.create_managed_user(self.admin, 'reviewer', 'password', 'reviewer')
        self.store.set_responsible_person(self.boss, self.marker['id'], self.reviewer['id'])
        handler = type('SessionTestHandler', (ReviewHandler,), {'store': self.store, 'log_message': lambda *args: None})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.session = ReviewSession()
        self.session_patch = patch.object(api, 'REVIEW_SESSION', self.session)
        self.session_patch.start()
        book = self.root / 'book'
        book.mkdir()
        cv2.imencode('.jpg', np.full((100, 100, 3), 255, dtype=np.uint8))[1].tofile(str(book / '001.jpg'))
        self.path = book / 'PaperFlow标注项目.json'
        self.project = {'delivery_book_id': 'BSESSIONTEST', 'book_root': str(book), 'image_root': str(book),
                        'pages': [{'image': '001.jpg', 'status': 'accepted', 'questions': [
                            {'id': 'q-1', 'bbox': [1, 2, 80, 90], 'text': '1+1=', 'minor_number': '1'}]}],
                        'answer_pages': [], 'pairs': []}
        data.save_project(self.project, self.path)

    def tearDown(self):
        self.session_patch.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def login(self, username='marker'):
        return self.session.login({'server_url': self.url, 'username': username, 'password': 'password'})

    def submit(self):
        return api.ROUTES['/api/review/submit']({'project_path': str(self.path), 'upload_source_images': False})

    def test_login_returns_identity_but_never_password_or_token(self):
        result = self.login()
        self.assertEqual(result['user']['role'], 'annotator')
        self.assertNotIn('token', json.dumps(result))
        self.assertNotIn('password', json.dumps(self.session.status()))

    def test_logout_revokes_only_its_own_session(self):
        self.login()
        token = self.session.credentials()['_review_token']
        other, _ = self.store.login('marker', 'password')
        self.session.logout()
        self.assertIsNone(self.store.user_for_token(token))
        self.assertIsNotNone(self.store.user_for_token(other))
        with self.assertRaises(PermissionError):
            self.session.credentials()

    def test_expired_session_clears_local_identity(self):
        self.login()
        self.store.logout(self.session.credentials()['_review_token'])
        with self.assertRaises(PermissionError):
            self.session.credentials()
        self.assertFalse(self.session.status()['logged_in'])

    def test_required_login_is_cached_but_revalidates_and_rejects_revoked_session(self):
        with self.assertRaises(PermissionError):self.session.require_login()

        self.login()
        with patch.object(self.session, 'credentials', wraps=self.session.credentials) as validate:
            self.session.require_login();self.session.require_login()
            validate.assert_not_called()
            self.session._validated_at = 0
            self.session.require_login()
            validate.assert_called_once()
        token=self.session.credentials()['_review_token']
        self.store.logout(token)
        self.session._validated_at=0
        self.assertFalse(self.session.validated_status()['logged_in'])
        with self.assertRaises(PermissionError):self.session.require_login()

    def test_http_editor_routes_reject_unsigned_requests_before_touching_files(self):
        bridge=ThreadingHTTPServer(('127.0.0.1',0),api.PaperFlowHandler)
        worker=threading.Thread(target=bridge.serve_forever,daemon=True);worker.start()
        def call(route):
            return urlopen(Request(f'http://127.0.0.1:{bridge.server_port}'+route,
                data=b'{}',headers={'Content-Type':'application/json'}),timeout=5)
        try:
            for route in ['/api/project/open-book','/api/project/save-page','/api/project/page-ocr']:
                with patch.dict(api.ROUTES,{route:lambda payload:self.fail('Unsigned operation executed')}):
                    with self.assertRaises(HTTPError) as caught:call(route)
                    self.assertEqual(caught.exception.code,401)
                    self.assertTrue(json.loads(caught.exception.read())['auth_required'])
            self.login()
            with patch.dict(api.ROUTES,{'/api/project/open-book':lambda payload:{'ok':True,'opened':True}}):
                with call('/api/project/open-book') as response:self.assertTrue(json.load(response)['opened'])
            self.session.logout()
            with self.assertRaises(HTTPError) as caught:call('/api/project/open-book')
            self.assertEqual(caught.exception.code,401)
        finally:
            bridge.shutdown();bridge.server_close();worker.join()

    def test_role_gate_and_spoofed_account_cannot_bypass_session(self):
        self.login()
        # 工作台只提交审核快照；客户端生成/上传交付的接口已下线，
        # 最终 JSON 一律由服务器在逐页审核通过后产出。
        self.assertNotIn('/api/review/generate-delivery', api.ROUTES)
        self.assertNotIn('/api/review/upload-delivery', api.ROUTES)
        run = api._authenticated_review_route(lambda payload: payload)
        result = run({'username': 'boss', 'server_url': 'http://wrong', '_review_token': 'forged'})
        self.assertEqual(result['username'], 'marker')
        self.assertEqual(result['server_url'], self.url)
        self.assertNotEqual(result['_review_token'], 'forged')

    def test_submit_reuses_login_and_keeps_remote_base_separate(self):
        self.login()
        with patch.object(sync, '_login', wraps=sync._login) as login:
            result = self.submit()
            self.assertTrue(login.call_args.kwargs['token'])
        project = data.load_project(self.path)
        self.assertEqual(project['online_review_base_revision'], result['project']['project_revision'])
        self.assertEqual(project['project_revision'], result['project_revision'])
        self.assertGreater(project['project_revision'], project['online_review_base_revision'])

    def test_status_automatically_pulls_matching_decisions(self):
        self.login()
        submitted = self.submit()
        self.store.decide_page(self.reviewer, submitted['submission']['id'], 'question-0', 'approved', '')
        result = api.ROUTES['/api/review/status']({'project_path': str(self.path)})
        self.assertTrue(result['can_deliver'])
        self.assertTrue(self.path.with_name(sync.DECISIONS_FILENAME).is_file())

    def test_reviewer_can_save_locally_and_push_twice_without_false_conflicts(self):
        self.login()
        self.submit()
        self.session.logout()
        self.login('reviewer')
        for _ in range(2):
            project = data.load_project(self.path)
            for _ in range(3):
                data.save_project(project, self.path)
            before = project['project_revision']
            result = api.ROUTES['/api/review/push-project']({'project_path': str(self.path)})
            saved = data.load_project(self.path)
            self.assertGreater(saved['project_revision'], before)
            self.assertEqual(result['project_revision'], saved['project_revision'])
            self.assertEqual(result['server_revision'], saved['online_review_base_revision'])

    def test_stale_marker_cannot_overwrite_reviewer_even_with_large_local_revision(self):
        self.login()
        self.submit()
        stale = data.load_project(self.path)
        document = sync.build_project_sync_snapshot(self.path)
        self.store.sync_project_snapshot(self.reviewer, document)
        stale['project_revision'] = 999
        data.save_project(stale, self.path)
        with self.assertRaises(sync.ReviewConflictError):
            self.submit()

    def test_renderer_save_preserves_server_base(self):
        incoming = {'pages': []}
        data.merge_persistent_delivery_state(incoming, {'online_review_base_revision': 7, 'pages': []})
        self.assertEqual(incoming['online_review_base_revision'], 7)

    def test_reviewer_edit_refreshes_preview_and_revokes_old_approval(self):
        self.login()
        old = self.submit()['submission']['id']
        self.store.decide_page(self.reviewer, old, 'question-0', 'approved', '')
        self.session.logout()
        self.login('reviewer')
        project = data.load_project(self.path)
        project['pages'][0]['questions'][0]['bbox'] = [5, 5, 90, 90]
        data.save_project(project, self.path)
        result = api.ROUTES['/api/review/push-project']({'project_path': str(self.path)})
        self.assertIn('网站审核预览已更新', result['notice'])
        status = api.ROUTES['/api/review/status']({'project_path': str(self.path)})
        self.assertTrue(status['matches_local'])
        self.assertFalse(status['can_deliver'])
        self.assertNotEqual(status['submission_id'], old)
        self.assertNotEqual(self.store.submission(self.reviewer, old)['status'], 'reviewed')
        self.store.decide_page(self.reviewer, status['submission_id'], 'question-0', 'approved', '')
        self.assertTrue(api.ROUTES['/api/review/status']({'project_path': str(self.path)})['can_deliver'])

    def test_delivery_batch_keeps_partial_files_invisible_until_commit(self):
        self.login()
        submitted = self.submit()
        self.store.decide_page(self.reviewer, submitted['submission']['id'], 'question-0', 'approved', '')
        content = b'{"schema_version":1}\n'
        digest = hashlib.sha256(content).hexdigest()
        batch = self.store.begin_delivery_batch(self.marker, {
            'book': {'id': 'BSESSIONTEST'},
            'files': [{'relative_path': 'k12_dataset/dataset.jsonl', 'bytes': len(content), 'sha256': digest}],
        })
        self.assertEqual(self.store.delivery_files(self.marker, 'BSESSIONTEST')['files'], [])
        with self.assertRaisesRegex(ValueError, '尚未上传完整'):
            self.store.commit_delivery_batch(self.marker, batch['batch_id'])
        self.store.upload_delivery_batch_file(self.marker, batch['batch_id'], {
            'relative_path': 'k12_dataset/dataset.jsonl',
            'content_base64': base64.b64encode(content).decode('ascii'),
        })
        committed = self.store.commit_delivery_batch(self.marker, batch['batch_id'])
        self.assertEqual(committed['file_count'], 1)
        self.assertEqual([row['path'] for row in self.store.delivery_files(self.marker, 'BSESSIONTEST')['files']],
                         ['k12_dataset/dataset.jsonl'])

    def test_only_reviewer_or_above_can_import_server_project(self):
        self.login()
        with self.assertRaises(PermissionError):
            api.ROUTES['/api/review/import-project']({'book_id': 'BSESSIONTEST', 'book_root': str(self.root / 'import')})

    def test_no_login_does_not_disable_local_project_loading(self):
        self.assertEqual(api._load_project({'project_path': str(self.path)})[1]['delivery_book_id'], 'BSESSIONTEST')
        with self.assertRaises(PermissionError):
            api.ROUTES['/api/review/list-projects']({})

    def test_approval_fingerprint_tracks_text_but_not_local_save_count(self):
        first = sync.build_review_snapshot(self.path)['snapshot_fingerprint']
        project = data.load_project(self.path)
        data.save_project(project, self.path)
        self.assertEqual(first, sync.build_review_snapshot(self.path)['snapshot_fingerprint'])
        project['pages'][0]['questions'][0]['text'] = '2+2='
        data.save_project(project, self.path)
        self.assertNotEqual(first, sync.build_review_snapshot(self.path)['snapshot_fingerprint'])

    def test_exit_forgets_identity_without_waiting_for_remote_logout(self):
        self.login()
        with patch('review_tools.review_session._request_json', side_effect=RuntimeError('offline')):
            result = self.session.forget_on_exit()
        self.assertFalse(result['logged_in'])
        self.assertFalse(self.session.status()['logged_in'])

    def test_submission_failure_after_sync_can_be_retried(self):
        self.login()
        request = sync._request_json
        def fail_submission(url, *args, **kwargs):
            if url.endswith('/api/submissions'):
                raise RuntimeError('temporary submission outage')
            return request(url, *args, **kwargs)
        with patch.object(sync, '_request_json', side_effect=fail_submission):
            with self.assertRaises(RuntimeError):
                self.submit()
        self.assertIn('online_review_base_revision', data.load_project(self.path))
        self.assertIn('submission', self.submit())
