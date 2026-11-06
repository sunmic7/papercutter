"""审核任务管理的服务端契约：内容判定、批量结论、批量指派、来源文件夹。

这些行为都是线上真跑出来的问题：
* 员工提交审核后又顺手存了一次（内容一个框都没动），任务立刻变成"待重新提交"，
  审核员根本来不及看——作废条件必须比"版本号变了"更严。
* 40 页的书逐页点单选+保存，审核员一晚上审不完一本书。
* 同一本教材被反复提交，任务列表里出现好几条同名任务，没人知道该点哪一条。
* 教材放在哪个磁盘目录原来只能扫描时顺手记一笔，搬盘以后没有地方改。
"""
import copy
import json
import os
import threading
from contextlib import closing
import tempfile
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from online_review_beta.server import ReviewHandler, ReviewStore
from review_tools import manual_annotation_data as data
from urllib.request import Request, urlopen


class ReviewTaskManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / 'originals'
        folder = self.sources / 'math-2015'
        folder.mkdir(parents=True)
        image = np.full((500, 500, 3), 255, np.uint8)
        cv2.rectangle(image, (20, 20), (220, 220), (0, 0, 0), 2)
        for name in ('1.jpg', '2.jpg'):
            cv2.imencode('.jpg', image)[1].tofile(str(folder / name))
        self.env = patch.dict(os.environ, {
            'PAPERFLOW_LIBRARY_ROOT': str(self.sources),
            'PAPERFLOW_ENGINE_ROOT': str(Path(__file__).resolve().parents[1]),
            'PAPERFLOW_ENRICH_ENABLED': '0',
            'PAPERFLOW_DISABLE_READY_SAMPLES': '1',
        })
        self.env.start()
        projects = self.root / 'projects'
        projects.mkdir()
        self.store = ReviewStore(self.root / 'database', projects_root=projects,
                                 staff_data_root=self.root / 'staff',
                                 delivery_root=self.root / 'final-output')
        self.library = self.store.library
        self.boss = self.store.create_user('boss', 'test-password', 'super_admin')
        self.admin = self.store.create_managed_user(self.boss, 'team', 'test-password', 'admin')
        self.marker = self.store.create_managed_user(self.admin, 'marker', 'test-password', 'annotator')
        self.reviewer = self.store.create_managed_user(self.admin, 'reviewer', 'test-password', 'reviewer')
        self.store.set_responsible_person(self.boss, self.marker['id'], self.reviewer['id'])
        self.library.scan(self.boss)
        self.book_id = self.library.listing(self.boss, catalog=True)[0]['id']
        self.library.assign(self.reviewer, {'book_id': self.book_id, 'annotator_id': self.marker['id']})

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    # ------------------------------------------------------------------
    def operate(self, user, operation, payload=None, client='marker-desk'):
        project = data.load_project(self.library.project_file(self.book_id))
        return self.library.operate(user, {'book_id': self.book_id, 'operation': operation,
            'payload': payload or {}, 'client_id': client,
            'base_revision': project['project_revision']})

    def working_project(self):
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        project = copy.deepcopy(opened['project'])
        question = data.new_record([0, 0, 500, 240], major_number='一', text='实验题',
            question_type='填空题', difficulty='基础', knowledge_points=['几何'],
            illustrations=[{'bbox': [20, 20, 220, 220], 'image_kind': 'question_image'}])
        answer = data.new_record([0, 0, 500, 240], answer=True, major_number='一', analysis='实验解析')
        answer['answer'] = '42'
        project.update(language='zh', stage='小学', grade='六年级', subject='数学')
        answer_page = project['pages'].pop()
        answer_page.update(status='accepted', answers=[answer])
        answer_page.pop('questions', None)
        project['answer_pages'] = [answer_page]
        project['pages'][0].update(status='accepted', questions=[question])
        data.set_pair(project, question['id'], [answer['id']])
        return project

    def submit(self):
        project = self.working_project()
        self.operate(self.marker, '/api/project/save-cache', {'project': project})
        self.operate(self.marker, '/api/review/submit')
        return self.store.list_submissions(self.reviewer)[0]

    def status_of(self, submission_id):
        return self.store.submission(self.reviewer, submission_id)['status']

    # ------------------------------------------------------------------
    def test_saving_unchanged_content_keeps_the_review_task_alive(self):
        """存一次同样的框，不该把刚提交的审核任务作废。"""
        submission = self.submit()
        self.assertEqual(submission['status'], 'pending')
        # 自动保存：内容一个字节都没变，只是修订号会涨。
        unchanged = data.load_project(self.library.project_file(self.book_id))
        self.operate(self.marker, '/api/project/save-cache', {'project': unchanged})
        self.assertEqual(self.status_of(submission['id']), 'pending')

    def test_changing_the_annotation_content_retires_the_snapshot(self):
        submission = self.submit()
        changed = data.load_project(self.library.project_file(self.book_id))
        changed['pages'][0]['questions'][0]['text'] = '换了一道题'
        self.operate(self.marker, '/api/project/save-cache', {'project': changed})
        self.assertEqual(self.status_of(submission['id']), 'outdated')

    def test_bulk_approve_only_touches_pending_pages(self):
        submission = self.submit()
        detail = self.store.submission(self.reviewer, submission['id'])
        first = detail['pages'][0]['page_key']
        self.store.decide_page(self.reviewer, submission['id'], first, 'returned', '这页要重标')
        result = self.store.decide_pages(self.reviewer, submission['id'], 'approved')
        self.assertEqual(result['decided'], len(detail['pages']) - 1)
        after = self.store.submission(self.reviewer, submission['id'])
        statuses = {page['page_key']: page['review_status'] for page in after['pages']}
        self.assertEqual(statuses[first], 'returned', '批量通过不能覆盖别人已经点过的退回重标')
        self.assertEqual(after['status'], 'returned')

    def test_bulk_approve_completes_the_book_and_queues_delivery(self):
        submission = self.submit()
        result = self.store.decide_pages(self.reviewer, submission['id'], 'approved')
        self.assertEqual(result['submission_status'], 'reviewed')
        self.assertGreater(result['decided'], 0)
        with closing(self.store._connect()) as db:
            job = db.execute('SELECT status FROM delivery_jobs WHERE submission_id=?',
                             (submission['id'],)).fetchone()
        self.assertIsNotNone(job, '整本放行后必须自动排队生成')

    def test_bulk_assign_reports_each_row(self):
        submission = self.submit()
        result = self.store.assign_submissions(self.admin, {
            'submission_ids': [submission['id'], 'sub-does-not-exist'],
            'reviewer_id': self.reviewer['id'], 'priority': 'high'})
        self.assertEqual((result['succeeded'], result['failed']), (1, 1))
        failed = next(item for item in result['results'] if not item['ok'])
        self.assertIn('不存在', failed['error'])
        detail = self.store.submission(self.reviewer, submission['id'])
        self.assertEqual(detail['assigned_reviewer_id'], self.reviewer['id'])

    def test_bulk_assign_is_admin_only(self):
        submission = self.submit()
        with self.assertRaises(PermissionError):
            self.store.assign_submissions(self.marker, {
                'submission_ids': [submission['id']], 'reviewer_id': self.reviewer['id']})

    # ------------------------------------------------------------------
    def test_source_folders_can_be_listed_edited_repointed_and_removed(self):
        year = self.sources.parent / '2016'
        image = (self.sources / 'math-2015' / '1.jpg').read_bytes()
        book_folder = year / '新教材甲' / '1' / '题目'
        book_folder.mkdir(parents=True)
        (book_folder / '1.jpg').write_bytes(image)
        self.library.scan(self.reviewer, {'folder': str(year)})
        listed = {item['path']: item for item in self.library.source_folders(self.reviewer)['folders']}
        self.assertIn(str(year), listed)
        self.assertEqual(listed[str(year)]['book_count'], 1)
        self.assertTrue(listed[str(self.sources)]['builtin'])

        self.library.save_source_folder(self.reviewer, {
            'path': str(year), 'label': '2016 小学数学', 'note': '第一批扫描件'})
        listed = {item['path']: item for item in self.library.source_folders(self.reviewer)['folders']}
        self.assertEqual(listed[str(year)]['label'], '2016 小学数学')

        # 整个目录搬到新磁盘：只改登记，教材记录跟着走。
        moved = self.sources.parent / '2016-moved'
        os.replace(year, moved)
        self.library.save_source_folder(self.reviewer, {'path': str(year), 'new_path': str(moved),
                                                        'label': '2016 小学数学', 'note': ''})
        with closing(self.store._connect()) as db:
            row = db.execute('SELECT path FROM library_book_sources').fetchall()
        paths = [item['path'] for item in row]
        self.assertTrue(any(value.startswith(str(moved)) for value in paths), paths)
        self.assertFalse(any(value.startswith(str(year) + os.sep) for value in paths), paths)

        self.library.remove_source_folder(self.reviewer, {'path': str(moved)})
        remaining = {item['path'] for item in self.library.source_folders(self.reviewer)['folders']}
        self.assertNotIn(str(moved), remaining)
        # 移除登记不删教材记录。
        names = [item['name'] for item in self.library.listing(self.boss, catalog=True)]
        self.assertIn('新教材甲', names)

    def test_source_folder_management_is_manager_only(self):
        with self.assertRaises(PermissionError):
            self.library.source_folders(self.marker)
        with self.assertRaises(PermissionError):
            self.library.save_source_folder(self.marker, {'path': str(self.sources)})

    def test_builtin_root_cannot_be_removed_or_renamed_into_itself(self):
        with self.assertRaises(ValueError):
            self.library.save_source_folder(self.reviewer, {'path': str(self.sources)})
        with self.assertRaises(ValueError):
            self.library.remove_source_folder(self.reviewer, {'path': str(self.sources)})

    # ------------------------------------------------------------------
    def test_new_http_routes_are_wired(self):
        """网页真正调的是 HTTP 路由；store 方法测过了不代表路径没写错。"""
        submission = self.submit()

        def call(method, path, payload=None, account='reviewer'):
            token, _ = self.store.login(account, 'test-password')
            body = json.dumps(payload).encode('utf-8') if payload is not None else None
            request = Request(f'http://127.0.0.1:{server.server_port}{path}', data=body, method=method,
                              headers={'Authorization': f'Bearer {token}',
                                       'Content-Type': 'application/json; charset=utf-8'})
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))

        handler = type('TaskManagementReview', (ReviewHandler,),
                       {'store': self.store, 'log_message': lambda *args: None})
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            decided = call('POST', f'/api/submissions/{submission["id"]}/pages/decide', {'status': 'approved'})
            self.assertGreater(decided['result']['decided'], 0)

            assigned = call('POST', '/api/submissions/batch/assign', {
                'submission_ids': [submission['id']], 'reviewer_id': self.reviewer['id']}, 'team')
            self.assertEqual(assigned['succeeded'], 1)

            listed = call('GET', '/api/library/sources')
            self.assertTrue(listed['folders'][0]['builtin'])

            year = self.sources.parent / '2017'
            (year / '教材乙').mkdir(parents=True)
            (year / '教材乙' / '1.jpg').write_bytes((self.sources / 'math-2015' / '1.jpg').read_bytes())
            saved = call('POST', '/api/library/sources',
                         {'path': str(year), 'label': '2017 春', 'note': '新到的'})
            self.assertEqual(saved['folder']['label'], '2017 春')
            scanned = call('POST', '/api/library/sources/scan', {'path': str(year)})
            self.assertEqual(scanned['added'], 1)
            call('POST', '/api/library/sources/remove', {'path': str(year)})
            left = {item['path'] for item in call('GET', '/api/library/sources')['folders']}
            self.assertNotIn(str(year), left)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
