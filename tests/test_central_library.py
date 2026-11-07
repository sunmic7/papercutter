"""Real-image server-workspace acceptance tests, with isolated users/data."""
import copy
from contextlib import closing
import base64
import hashlib
import json
import os
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

from online_review_beta.server import ReviewStore, ReviewHandler
from review_tools import manual_annotation_data as data


class CentralLibraryTests(unittest.TestCase):
    def session_actor(self, actor):
        token, _ = self.store.login(actor['username'], 'test-password')
        return {**actor, '_lease_session_hash':hashlib.sha256(token.encode()).hexdigest()}, token

    def test_workstation_identity_reopens_after_restart_but_other_desks_stay_blocked(self):
        first, _ = self.session_actor(self.marker)
        second, _ = self.session_actor(self.marker)
        self.library.acquire(first,self.book_id,'stable-workstation')
        self.library.acquire(second,self.book_id,'stable-workstation')
        with self.assertRaises(ValueError):self.library.acquire(second,self.book_id,'other-computer')
        with self.assertRaises(ValueError):self.library.acquire(self.reviewer,self.book_id,'stable-workstation')

    def test_logout_releases_only_leases_of_that_session(self):
        first, token = self.session_actor(self.marker)
        second, other_token = self.session_actor(self.marker)
        self.library.acquire(first,self.book_id,'first-computer')
        self.store.logout(other_token)
        self.assertIn(self.book_id,self.library.leases)
        self.store.logout(token)
        self.assertNotIn(self.book_id,self.library.leases)
        third, _ = self.session_actor(self.marker)
        self.library.acquire(third,self.book_id,'new-computer')

    def test_password_rotation_does_not_leave_an_orphaned_lease(self):
        first, _ = self.session_actor(self.marker)
        self.library.acquire(first,self.book_id,'old-computer')
        self.store.change_own_password(self.marker,'test-password','new-password')
        token, _ = self.store.login(self.marker['username'],'new-password')
        fresh={**self.marker,'_lease_session_hash':hashlib.sha256(token.encode()).hexdigest()}
        self.library.acquire(fresh,self.book_id,'new-computer')

    def test_discovery_counts_every_scan_under_the_book_folder_in_one_pass(self):
        """扫描只走一遍文件系统，统计仍要把题目/答案/数字子目录都算进来。"""
        image = (self.sources / 'math-2015' / '1.jpg').read_bytes()
        book = self.sources.parent / '2020' / '合集教材'
        for parts, name in ((('题目',), 'a.jpg'), (('答案',), 'b.jpg'), (('1', '题目'), 'c.jpg')):
            folder = book.joinpath(*parts)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(image)
        discovered = self.library._discover_books(self.sources.parent / '2020')
        self.assertEqual(1, len(discovered))
        folder, count, size = discovered[0]
        self.assertEqual(book, folder)
        self.assertEqual(3, count)
        self.assertEqual(len(image) * 3, size)

    def test_folder_registration_recurses_and_survives_restart_without_duplicate_books(self):
        from online_review_beta.central_library import CentralLibrary
        year = self.sources.parent / '2016'
        image = (self.sources / 'math-2015' / '1.jpg').read_bytes()
        for name in ('新教材甲', '新教材乙'):
            folder = year / name / '1' / '题目'
            folder.mkdir(parents=True)
            (folder / '1.jpg').write_bytes(image)
        choices = self.library.folders(self.reviewer,str(self.sources.parent))
        self.assertIn('2016', [x['name'] for x in choices['folders']])
        result = self.library.scan(self.reviewer, {'folder':'2016'})
        self.assertEqual((result['count'], result['added']), (2,2))
        rows = self.library.listing(self.boss, catalog=True)
        new = next(x for x in rows if x['name']=='新教材甲')
        self.library.assign(self.reviewer, {'book_id':new['id'], 'annotator_id':self.marker['id']})
        self.assertEqual(self.library.book(self.marker,new['id'])[2],year/'新教材甲')
        again = CentralLibrary(self.store)
        self.assertEqual(again.scan(self.reviewer)['added'],0)
        self.assertEqual(len(again.listing(self.boss,catalog=True)),3)
        self.assertEqual(next(x for x in again.listing(self.boss,catalog=True) if x['name']=='新教材甲')['id'],new['id'])
        self.assertEqual(again.scan(self.reviewer,{'folder':'2016/新教材甲'})['added'],0)

    def test_folder_scan_permissions_boundaries_and_empty_folder(self):
        with self.assertRaises(PermissionError):self.library.folders(self.marker)
        with self.assertRaises(PermissionError):self.library.scan(self.marker,{'folder':'.'})
        with self.assertRaises(ValueError):self.library.folders(self.reviewer,str(self.sources/'missing'))
        with self.assertRaises(ValueError):self.library.scan(self.reviewer,{'folder':''})
        empty = self.sources.parent/'empty';empty.mkdir()
        with self.assertRaises(ValueError):self.library.scan(self.reviewer,{'folder':'empty'})
        old_id=self.book_id
        self.assertEqual(self.library.scan(self.reviewer)['added'],0)
        self.assertEqual(self.library.listing(self.boss,catalog=True)[0]['id'],old_id)

    def test_any_server_folder_and_drive_picker(self):
        roots=self.library.folders(self.reviewer)
        self.assertFalse(roots['selectable'])
        self.assertTrue(roots['folders'])
        self.assertTrue(all(Path(item['path']).is_absolute() for item in roots['folders']))
        with tempfile.TemporaryDirectory() as outside:
            folder=Path(outside)/'全新教材';folder.mkdir()
            (folder/'1.jpg').write_bytes((self.sources/'math-2015'/'1.jpg').read_bytes())
            self.assertTrue(self.library.folders(self.reviewer,str(folder))['selectable'])
            self.assertEqual(self.library.scan(self.reviewer,{'folder':str(folder)})['added'],1)
            book=next(r for r in self.library.listing(self.boss,catalog=True) if r['name']=='全新教材')
            self.library.assign(self.reviewer,{'book_id':book['id'],'annotator_id':self.marker['id']})
            self.assertEqual(self.library.book(self.marker,book['id'])[2],folder.resolve())
            self.assertEqual(book['source_folder'],str(folder.parent))
            self.assertIn(str(folder.parent),book['source_folders'])

    def test_hidden_books_survive_refresh_and_restart_then_restore_without_data_loss(self):
        from online_review_beta.central_library import CentralLibrary
        self.library.open(self.marker,{'book_id':self.book_id,'client_id':'marker-desk'})
        before=self.library.project_file(self.book_id).read_bytes()
        self.library.set_visibility(self.reviewer,{'book_ids':[self.book_id],'hidden':True})
        self.assertEqual(self.library.listing(self.marker),[])
        self.assertEqual(self.library.listing(self.reviewer,catalog=True),[])
        self.assertTrue(self.library.listing(self.reviewer,catalog=True,include_hidden=True)[0]['hidden'])
        with self.assertRaises(ValueError):self.library.assign(self.reviewer,{'book_id':self.book_id,'annotator_id':self.marker['id']})
        self.library.scan(self.reviewer)
        again=CentralLibrary(self.store)
        self.assertEqual(again.listing(self.marker),[])
        again.set_visibility(self.reviewer,{'book_ids':[self.book_id],'hidden':False})
        self.assertEqual(again.listing(self.marker)[0]['id'],self.book_id)
        self.assertEqual(again.project_file(self.book_id).read_bytes(),before)
        with closing(self.store._connect()) as db:
            self.assertEqual(db.execute('SELECT annotator_id FROM project_assignments WHERE book_id=?',(self.book_id,)).fetchone()[0],self.marker['id'])

    def test_visibility_permissions_and_atomic_batch(self):
        with self.assertRaises(PermissionError):self.library.set_visibility(self.marker,{'book_ids':[self.book_id],'hidden':True})
        with self.assertRaises(PermissionError):self.library.listing(self.marker,include_hidden=True)
        with self.assertRaises(PermissionError):self.library.set_visibility(self.reviewer,{'book_ids':[self.book_id,'BOTHER'],'hidden':True})
        self.assertFalse(self.library.listing(self.reviewer,catalog=True)[0]['hidden'])
        with self.assertRaises(ValueError):self.library.set_visibility(self.reviewer,{'book_ids':[self.book_id],'hidden':'false'})
    def test_unassigned_books_are_only_visible_in_manager_catalog(self):
        with closing(self.store._connect()) as db:
            db.execute('DELETE FROM project_assignments WHERE book_id=?', (self.book_id,))
            db.commit()
        self.assertEqual(self.library.listing(self.marker), [])
        self.assertEqual(self.library.listing(self.reviewer), [])
        self.assertEqual([r['id'] for r in self.library.listing(self.reviewer, catalog=True)], [self.book_id])
        with self.assertRaises(PermissionError):
            self.library.listing(self.marker, catalog=True)
        with self.assertRaises(PermissionError):
            self.library.open(self.marker, {'book_id': self.book_id, 'client_id':'a'})
        with self.assertRaises(PermissionError):
            self.library.assign(self.marker, {'book_id':self.book_id,'annotator_id':self.marker['id']})

    def test_reassignment_removes_old_access_and_keeps_exact_central_frames(self):
        opened = self.library.open(self.marker, {'book_id':self.book_id,'client_id':'marker-desk'})
        project = opened['project']
        record = data.new_record([10,20,120,130], major_number='六', number='2', text='保留原标注')
        project['pages'][0]['questions'] = [record]
        self.operate(self.marker, '/api/project/save-cache', {'project':project})
        expected = data.load_project(self.library.project_file(self.book_id))['pages'][0]['questions'][0]
        other = self.store.create_managed_user(self.admin, 'next-marker', 'test-password', 'annotator')
        self.store.set_responsible_person(self.boss, other['id'], self.reviewer['id'])
        with self.assertRaises(ValueError):
            self.library.assign(self.reviewer, {'book_id':self.book_id,'annotator_id':other['id']})
        self.operate(self.marker, '/api/library/release')
        self.library.assign(self.reviewer, {'book_id':self.book_id,'annotator_id':other['id']})
        self.assertEqual(self.library.listing(self.marker), [])
        with self.assertRaises(PermissionError):
            self.library.image(self.marker, self.book_id, 'pages', 0)
        with self.assertRaises(PermissionError):
            self.library.open(self.marker, {'book_id':self.book_id,'client_id':'old'})
        reopened = self.library.open(other, {'book_id':self.book_id,'client_id':'new'})
        self.assertEqual(reopened['project']['pages'][0]['questions'][0], expected)
        row = self.library.listing(other)[0]
        self.assertEqual(row['annotation_status'], 'in_progress')
        self.assertEqual(row['question_regions'], 1)

    def test_assignment_and_status_require_valid_team_and_confirmed_pages(self):
        opened = self.library.open(self.marker, {'book_id':self.book_id,'client_id':'marker-desk'})
        self.assertEqual(self.library.listing(self.marker)[0]['annotation_status'], 'not_started')
        project = opened['project']
        for page in project['pages']:
            page.update(status='accepted', questions=[data.new_record([0,0,100,100])])
        self.operate(self.marker, '/api/project/save-cache', {'project':project})
        row = self.library.listing(self.marker)[0]
        self.assertEqual((row['annotation_status'],row['confirmed_pages']), ('completed',2))
        foreign = self.store.create_managed_user(self.boss, 'foreign-admin', 'test-password', 'admin')
        outsider = self.store.create_managed_user(foreign, 'foreign-marker', 'test-password', 'annotator')
        with self.assertRaises(PermissionError):
            self.library.assign(self.reviewer, {'book_id':self.book_id,'annotator_id':outsider['id']})
    def test_save_reuses_final_project_read_and_lazy_image_mapping(self):
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        revision = opened['project']['project_revision']
        with patch.object(data, 'load_project', wraps=data.load_project) as loads:
            saved = self.operate(self.marker, '/api/project/save-cache', {'project': opened['project']},
                                 revision=opened['project']['project_revision'])
        self.assertEqual(loads.call_count, 3)
        self.assertEqual(saved['project_revision'], revision + 1)
        from online_review_beta.central_library import CentralLibrary
        with patch('online_review_beta.central_library.Path', side_effect=AssertionError('unnecessary image paths')):
            self.assertEqual(CentralLibrary.response({'ok': True}, self.book_id, opened['project']), {'ok': True})

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
        self.before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        self.env = patch.dict(os.environ, {
            'PAPERFLOW_LIBRARY_ROOT': str(self.sources),
            'PAPERFLOW_ENGINE_ROOT': str(Path(__file__).resolve().parents[1]),
            'PAPERFLOW_ENRICH_ENABLED': '0',
            # 后台实时样例题库线程有 20 秒去抖，测试里关掉，避免用例之间互相踩盘。
            'PAPERFLOW_DISABLE_READY_SAMPLES': '1',
        })
        self.env.start()
        projects = self.root / 'projects'
        projects.mkdir()
        self.store = ReviewStore(self.root / 'database', projects_root=projects, staff_data_root=self.root / 'staff',
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

    def operate(self, user, operation, payload=None, client='marker-desk', revision=None):
        if revision is None:
            revision = data.load_project(self.library.project_file(self.book_id))['project_revision']
        return self.library.operate(user, {'book_id': self.book_id, 'operation': operation,
            'payload': payload or {}, 'client_id': client, 'base_revision': revision})

    def test_diagnose_before_any_project_explains_instead_of_errno(self):
        """刚分配、还没提交标注的教材，点「检查缺什么」要看到人话而不是 Errno 2。"""
        project_file = self.library.project_file(self.book_id)
        if project_file.is_file():
            project_file.unlink()
        with self.assertRaises(ValueError) as caught:
            self.library.diagnose_delivery(self.reviewer, self.book_id)
        message = str(caught.exception)
        self.assertIn('标注工程', message)
        self.assertNotIn('Errno', message)

    def test_full_annotation_review_delivery_cycle_preserves_originals(self):
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        self.assertEqual(opened['project_path'], 'server://' + self.book_id)
        self.assertTrue(opened['project']['pages'][0]['source_path'].startswith('http://127.0.0.1:8601/'))
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
        self.operate(self.marker, '/api/project/save-cache', {'project': project})
        source = self.library.image(self.marker, self.book_id, 'pages', 0)
        self.assertTrue(source.is_file())
        self.operate(self.marker, '/api/review/submit')
        with self.assertRaises(ValueError):
            self.library.open(self.reviewer, {'book_id': self.book_id, 'client_id': 'review-desk'})
        self.operate(self.marker, '/api/library/release')
        review = self.library.open(self.reviewer, {'book_id': self.book_id, 'client_id': 'review-desk'})
        self.assertEqual(review['project']['pages'][0]['questions'][0]['text'], '实验题')
        # 默认按甲方字段判定：还没放行也能出包，这里不再被拦。
        # 严格模式（only_reviewed）仍要求整页放行，见 test_partial_approval...。
        submission = self.store.list_submissions(self.reviewer)[0]
        frozen_project = self.store.preview_root / submission['id'] / 'project.json'
        self.assertTrue(frozen_project.is_file())
        detail = self.store.submission(self.reviewer, submission['id'])
        for page in detail['pages']:
            self.store.decide_page(self.reviewer, submission['id'], page['page_key'], 'approved', '')
        status = self.library.status(self.reviewer, self.book_id, self.library.project_file(self.book_id))
        self.assertTrue(status['can_deliver'], status)
        with closing(self.store._connect()) as db:
            queued = db.execute("SELECT status FROM delivery_jobs WHERE submission_id=?", (submission['id'],)).fetchone()
        self.assertEqual(queued['status'], 'queued')
        listed = next(row for row in self.store.list_submissions(self.reviewer) if row['id'] == submission['id'])
        self.assertEqual(listed['delivery_status'], 'queued')
        # Later workspace edits must not change the version that reviewers approved.
        live_path = self.library.project_file(self.book_id)
        live_project = data.load_project(live_path)
        live_project['subject'] = '审核后发生变化的当前工程'
        data.save_project(live_project, live_path, increment_revision=True)
        result = self.store.run_next_delivery_job()
        self.assertEqual(result['status'], 'succeeded')
        output = Path(result['output_path'])
        self.assertEqual(output.parent.parent, self.root / 'final-output' / self.book_id)
        self.assertTrue((output / 'k12_dataset/dataset.jsonl').is_file())
        self.assertEqual(result['k12']['eligible_count'], 1)
        self.assertTrue(list(output.rglob('*.jpg')))
        provenance = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['annotator']['account'], 'marker')
        self.assertEqual(provenance['page_reviews'][0]['reviewer_username'], 'reviewer')
        self.assertTrue(provenance['artifacts'])
        generation = json.loads((output / 'generation.json').read_text(encoding='utf-8'))
        self.assertEqual(generation['source_binding']['mode'], 'server_library')
        current = json.loads((self.root / 'final-output' / self.book_id / 'current.json').read_text(encoding='utf-8'))
        self.assertEqual(current['version'], result['version'])
        self.assertTrue(self.store.delivery_files(self.reviewer, self.book_id)['files'])
        self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.parent.iterdir()}, self.before)
        # Editing after approval never leaks the changed frame into the reviewed package:
        # regeneration still uses the immutable submitted project snapshot.
        changed = data.load_project(self.library.project_file(self.book_id))
        changed['pages'][0]['questions'][0]['text'] = '修改后的实验题'
        self.operate(self.reviewer, '/api/project/save-cache', {'project': changed}, client='review-desk')
        regenerated = self.operate(self.reviewer, '/api/review/generate-delivery', client='review-desk')
        self.assertEqual(regenerated['submission_id'], submission['id'])

    # ------------------------------------------------------------------
    # 部分放行：没审完也能出包，但没放行的页面绝不进包
    # ------------------------------------------------------------------
    def _two_page_submission(self):
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        project = copy.deepcopy(opened['project'])
        project.update(language='zh', stage='小学', grade='六年级', subject='数学')
        pages = project['pages']
        self.assertGreaterEqual(len(pages), 2)
        answer_page = copy.deepcopy(pages[1])
        answer_page.pop('questions', None)
        questions, answers = [], []
        for index, page in enumerate(pages[:2]):
            question = data.new_record([0, 0, 500, 240], major_number='一', text=f'第{index + 1}题',
                                       question_type='填空题', difficulty='基础', knowledge_points=['几何'],
                                       illustrations=[{'bbox': [20, 20, 220, 220],
                                                       'image_kind': 'question_image'}])
            answer = data.new_record([0, 0, 500, 240], answer=True, major_number='一', analysis='解析')
            answer['answer'] = str(index + 1)
            page.update(status='accepted', questions=[question])
            questions.append(question)
            answers.append(answer)
        answer_page.update(status='accepted', answers=answers)
        project['answer_pages'] = [answer_page]
        for question, answer in zip(questions, answers):
            data.set_pair(project, question['id'], [answer['id']])
        # 直接落盘：这两页测试要复用第 2 张扫描件做答案页，走 save-cache 会被
        # "原教材页面清单不能由客户端更换" 挡下来（那是防止客户端偷换原图）。
        data.save_project(project, self.library.project_file(self.book_id))
        self.operate(self.marker, '/api/review/submit')
        return self.store.list_submissions(self.reviewer)[0]

    def test_partial_approval_ships_only_approved_pages_and_overwrites_on_regenerate(self):
        submission = self._two_page_submission()
        detail = self.store.submission(self.reviewer, submission['id'])
        for page in detail['pages']:
            if page['kind'] == 'answer' or page['page_index'] == 0:
                self.store.decide_page(self.reviewer, submission['id'], page['page_key'], 'approved', '')
        # 只放行第 1 页，仍然可以生成——这正是"不必等整本审完"。
        # 严格模式（only_reviewed）：只收录放行页，版本标 partial。
        self.store.retry_delivery(self.reviewer, self.book_id, force=True, only_reviewed=True)
        first = self.store.run_next_delivery_job()
        self.assertEqual(first['status'], 'succeeded', first.get('error'))
        self.assertTrue(first['partial'])
        self.assertEqual((first['approved_pages'], first['total_pages']), (1, 2))
        output = Path(first['output_path'])
        rows = [json.loads(line) for line in
                (output / 'k12_dataset' / 'dataset.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual([row['data']['question'] for row in rows], ['第1题'])
        generation = json.loads((output / 'generation.json').read_text(encoding='utf-8'))
        self.assertTrue(generation['partial'])
        # 逐页结果必须和包里的实际内容对得上：第 1 页收录，第 2 页没放行。
        coverage = json.loads((output / 'page_coverage.json').read_text(encoding='utf-8'))
        self.assertEqual(coverage['summary']['question_pages'], 2)
        self.assertEqual(coverage['summary']['delivered_pages'], 1)
        self.assertEqual(coverage['summary']['skipped_pages'], 1)
        self.assertEqual(coverage['summary']['answer_skipped_pages'], 0)
        question_rows = {row['page']: row for row in coverage['pages'] if row['kind'] == 'question'}
        self.assertEqual(question_rows[1]['status'], 'delivered')
        self.assertEqual(question_rows[1]['delivered'], 1)
        self.assertEqual(question_rows[2]['status'], 'skipped')
        self.assertIn('未审核放行', ' '.join(question_rows[2]['reasons']))
        served = self.library.delivery_page_coverage(self.reviewer, self.book_id)
        self.assertTrue(served['available'])
        self.assertEqual(served['summary']['delivered_pages'], 1)
        first_ids = {row['source']['internal_question_id']: row['id'] for row in rows}

        # 整本放行后自动补一版完整的；同一道题沿用同一个 sample id，旧版本目录被清掉。
        for page in detail['pages']:
            if page['review_status'] != 'approved':
                self.store.decide_page(self.reviewer, submission['id'], page['page_key'], 'approved', '')
        second = self.store.run_next_delivery_job()
        self.assertEqual(second['status'], 'succeeded', second.get('error'))
        self.assertFalse(second['partial'])
        self.assertEqual(second['approved_pages'], 2)
        final = Path(second['output_path'])
        rows = [json.loads(line) for line in
                (final / 'k12_dataset' / 'dataset.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual([row['data']['question'] for row in rows], ['第1题', '第2题'])
        self.assertEqual(rows[0]['id'], first_ids[rows[0]['source']['internal_question_id']])
        versions = self.root / 'final-output' / self.book_id / 'versions'
        self.assertEqual([path.name for path in versions.iterdir() if path.is_dir()], [second['version']])
        current = json.loads((self.root / 'final-output' / self.book_id / 'current.json').read_text(encoding='utf-8'))
        self.assertFalse(current['partial'])

        # 网页上的「页面明细」走的是这个 HTTP 路由，不能只验 store 方法。
        handler = type('PageCoverageReview', (ReviewHandler,),
                       {'store': self.store, 'log_message': lambda *args: None})
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            _actor, token = self.session_actor(self.reviewer)
            url = (f"http://127.0.0.1:{server.server_port}"
                   f"/api/deliveries/{self.book_id}/pages")
            with urlopen(Request(url, headers={'Authorization': f'Bearer {token}'}), timeout=5) as response:
                payload = json.loads(response.read().decode('utf-8'))
            self.assertTrue(payload['available'])
            self.assertEqual(payload['summary']['delivered_pages'], 2)
            _marker, marker_token = self.session_actor(self.marker)
            with self.assertRaises(HTTPError) as denied:
                urlopen(Request(url, headers={'Authorization': f'Bearer {marker_token}'}), timeout=5)
            self.assertEqual(denied.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()

    # ------------------------------------------------------------------
    # 实时样例题库：单题够格就产出，不需要整本审核
    # ------------------------------------------------------------------
    def _ready_project(self, difficulty='基础'):
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        project = copy.deepcopy(opened['project'])
        question = data.new_record([0, 0, 500, 240], major_number='一', text='实验题',
            question_type='填空题', difficulty=difficulty, knowledge_points=['几何'],
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
        self.operate(self.marker, '/api/project/save-cache', {'project': project})
        return project

    def test_ready_samples_emit_one_question_without_any_review(self):
        """一题填齐就出一题的 JSON：不提交审核、不放行，也能拿到这一题。"""
        self._ready_project()
        result = self.library.sync_ready_samples(self.marker, self.book_id)
        self.assertEqual(result['ready_count'], 1, result)
        book_root = self.root / 'final-output' / self.book_id
        dataset = book_root / 'ready_samples' / 'k12_dataset' / 'dataset.jsonl'
        self.assertTrue(dataset.is_file())
        rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['metainfo']['difficulty'], '基础')
        self.assertEqual(rows[0]['data']['answer'], '42')
        self.assertTrue((book_root / 'ready_samples' / 'k12_dataset' / 'images').is_dir())
        # 只有审核后生成才允许写 current.json / versions：草稿绝不能冒充正式交付。
        self.assertFalse((book_root / 'current.json').exists())
        self.assertFalse((book_root / 'versions').exists())
        # 同一内容重复重算必须复用同一个 id，否则正式版会变成另一条样本。
        self.library.sync_ready_samples(self.marker, self.book_id)
        again = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual(again[0]['id'], rows[0]['id'])
        # 标注员自己就能看（"填好一题就能看到"的那条路）。
        preview = self.store.delivery_samples(self.marker, self.book_id, version='live')
        self.assertEqual(preview['total'], 1)
        self.assertTrue(preview['versions'][0]['live'])
        self.assertEqual(preview['versions'][0]['ready_count'], 1)

    def test_ready_samples_drop_a_question_that_loses_eligibility(self):
        self._ready_project()
        self.library.sync_ready_samples(self.marker, self.book_id)
        project = data.load_project(self.library.project_file(self.book_id))
        project['pages'][0]['questions'][0]['difficulty'] = '简单'
        data.save_project(project, self.library.project_file(self.book_id), increment_revision=True)
        result = self.library.sync_ready_samples(self.marker, self.book_id)
        self.assertEqual(result['ready_count'], 0)
        reasons = {row['reason'] for row in result['reason_summary']}
        self.assertTrue(any('难度' in reason for reason in reasons), reasons)
        dataset = (self.root / 'final-output' / self.book_id / 'ready_samples'
                   / 'k12_dataset' / 'dataset.jsonl')
        self.assertEqual(dataset.read_text(encoding='utf-8').strip(), '')

    def test_ready_samples_are_private_to_the_books_own_annotator(self):
        self._ready_project()
        self.library.sync_ready_samples(self.marker, self.book_id)
        other = self.store.create_managed_user(self.admin, 'marker2', 'test-password', 'annotator')
        self.store.set_responsible_person(self.boss, other['id'], self.reviewer['id'])
        with self.assertRaises(LookupError):
            self.store.delivery_samples(other, self.book_id, version='live')
        # 已发布交付版本的鉴权没有被顺带放开：标注员仍然进不去。
        with self.assertRaises(PermissionError):
            self.store._delivery_preview_target(self.marker, self.book_id)

    def test_ready_samples_need_a_project_first(self):
        project_file = self.library.project_file(self.book_id)
        if project_file.is_file():
            project_file.unlink()
        with self.assertRaises(ValueError) as caught:
            self.library.sync_ready_samples(self.marker, self.book_id)
        self.assertIn('标注工程', str(caught.exception))

    def test_conflicts_paths_and_cross_team_access_are_rejected(self):
        self.library.scan(self.boss)
        self.assertEqual(self.library.listing(self.marker)[0]['id'], self.book_id)
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        project = opened['project']
        with self.assertRaises(ValueError):
            self.operate(self.marker, '/api/project/save-cache', {'project': project}, revision=0)
        malicious = copy.deepcopy(project)
        malicious['pages'][0]['image'] = '../secret.jpg'
        with self.assertRaises(ValueError):
            self.operate(self.marker, '/api/project/save-cache', {'project': malicious})
        with self.assertRaises(ValueError):
            self.operate(self.marker, '/api/project/rotate-page')
        other = self.store.create_managed_user(self.boss, 'other-team', 'test-password', 'admin')
        self.assertEqual(self.library.listing(other), [])
        self.assertTrue(self.library.image(other,self.book_id,'pages',0).is_file())
        other_reviewer = self.store.create_managed_user(self.admin, 'other-reviewer', 'test-password', 'reviewer')
        self.assertEqual(self.library.listing(other_reviewer), [])
        with self.assertRaises(PermissionError):
            self.library.image(other_reviewer, self.book_id, 'pages', 0)

    def test_reviewer_library_only_lists_uploaded_annotations_from_own_markers(self):
        self.assertEqual(self.library.listing(self.reviewer), [])
        opened = self.library.open(self.marker, {'book_id': self.book_id, 'client_id': 'marker-desk'})
        self.assertEqual(self.library.listing(self.reviewer), [], 'Claiming an empty textbook is not an upload')
        project = opened['project']
        project['pages'][0]['questions'] = [data.new_record([0, 0, 100, 100], text='标注员已保存的框')]
        self.operate(self.marker, '/api/project/save-cache', {'project': project})
        books = self.library.listing(self.reviewer)
        self.assertEqual([book['id'] for book in books], [self.book_id])
        self.assertEqual(books[0]['annotator_name'], 'marker')
        self.assertEqual(books[0]['question_regions'], 1)
        self.assertEqual(books[0]['review_status'], 'not_submitted')
        other = self.store.create_managed_user(self.admin, 'another-reviewer', 'test-password', 'reviewer')
        self.assertEqual(self.library.listing(other), [])
        self.store.remove_project_progress(self.reviewer, self.book_id)
        self.assertEqual(self.library.listing(self.reviewer), [])

    def test_uploaded_legacy_project_opens_exact_frames_without_client_downloads(self):
        legacy_id = 'B12345678901234567890123456789012'
        project = data.create_book_project(str(self.sources / 'math-2015'))
        question = data.new_record([10, 10, 100, 100], text='旧工程中的原标注')
        project['pages'][0]['questions'] = [question]
        project.update(delivery_book_id=legacy_id, project_revision=7)
        document = {'book': {'id': legacy_id, 'name': '已上传教材'}, 'project': project}
        self.store.sync_project_snapshot(self.marker, document)
        self.assertEqual(self.library.listing(self.reviewer)[0]['id'], legacy_id)
        # Missing originals must not silently create an empty replacement.
        with self.assertRaisesRegex(ValueError, '原图尚未上传完整'):
            self.library.open(self.reviewer, {'book_id': legacy_id, 'client_id': 'review-desk'})
        self.assertFalse(self.library.project_file(legacy_id).exists())
        for page in project['pages']:
            content = (self.sources / 'math-2015' / page['image']).read_bytes()
            self.store.upload_source_file(self.marker, legacy_id, {
                'relative_path': page['image'], 'content_base64': base64.b64encode(content).decode()})
        opened = self.library.open(self.reviewer, {'book_id': legacy_id, 'client_id': 'review-desk'})
        self.assertEqual(opened['project_path'], 'server://' + legacy_id)
        record = opened['project']['pages'][0]['questions'][0]
        self.assertEqual(record['id'], question['id'])
        self.assertEqual(record['text'], '旧工程中的原标注')
        self.assertEqual(record['bbox'], question['bbox'])
        self.assertEqual(opened['project']['project_revision'], 7)
        self.assertTrue(self.library.image(self.reviewer, legacy_id, 'pages', 0).is_file())
        with self.assertRaisesRegex(ValueError, '服务器统一管理'):
            self.store.sync_project_snapshot(self.marker, document)
        other = self.store.create_managed_user(self.admin, 'other-legacy-reviewer', 'test-password', 'reviewer')
        self.assertEqual(self.library.listing(other), [])
        with self.assertRaises(PermissionError):
            self.library.open(other, {'book_id': legacy_id, 'client_id': 'other-desk'})

    def test_local_gateway_reads_images_saves_and_recovers_an_offline_draft(self):
        from desktop_annotation import paperflow_annotation_api as api
        from review_tools.online_review_sync import _request_json
        from review_tools.review_session import ReviewSession
        handler = type('GatewayReview', (ReviewHandler,), {'store':self.store, 'log_message':lambda *args:None})
        review = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        bridge = ThreadingHTTPServer(('127.0.0.1', 0), api.PaperFlowHandler)
        for server in (review, bridge):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        session = ReviewSession()
        session.login({'server_url':f'http://127.0.0.1:{review.server_port}',
                       'username':self.marker['username'], 'password':'test-password'})
        base = f'http://127.0.0.1:{bridge.server_port}'
        try:
            with patch.object(api, 'REVIEW_SESSION', session), patch.object(api, 'ROOT', self.root):
                opened = _request_json(base + '/api/library', 'POST', {'action':'open','book_id':self.book_id})
                project = opened['project']
                with urlopen(base + f'/api/library/image?book_id={self.book_id}&kind=pages&page_index=0') as response:
                    self.assertGreater(len(response.read()), 100)
                    self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'null')
                project['pages'][0]['questions'] = [data.new_record([1,2,30,40], text='draft')]
                with patch.object(session, 'credentials', side_effect=RuntimeError('offline')):
                    with self.assertRaises(RuntimeError):
                        api._library_request({'action':'operate','operation':'/api/project/save-cache',
                            'book_id':self.book_id,'base_revision':project['project_revision'], 'payload':{'project':project}})
                reopened = _request_json(base + '/api/library', 'POST', {'action':'open','book_id':self.book_id})
                self.assertEqual(reopened['recovery_project']['pages'][0]['questions'][0]['text'], 'draft')
                saved = _request_json(base + '/api/project/save-cache', 'POST', {
                    'project_path':'server://' + self.book_id,'project':project,
                    '_server_revision':project['project_revision']})
                self.assertGreater(saved['project_revision'], project['project_revision'])
                reopened = _request_json(base + '/api/library', 'POST', {'action':'open','book_id':self.book_id})
                self.assertNotIn('recovery_project', reopened)
        finally:
            for server in (bridge, review):
                server.shutdown()
                server.server_close()


class DistributedServerCroppingTests(unittest.TestCase):
    """Employee PC annotates and uploads scans; the server crops and exports.

    These cases reuse the real-image fixture from CentralLibraryTests but
    deliberately strip the local ``image_root`` from the frozen project, which
    is exactly what ``build_project_sync_snapshot`` does for an employee
    workstation.  The only scans available to the server are then the uploaded
    ``SourceImages/<book_id>`` copies.
    """

    def setUp(self):
        CentralLibraryTests.setUp(self)

    def tearDown(self):
        CentralLibraryTests.tearDown(self)

    def _approved_submission(self):
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
        CentralLibraryTests.operate(self, self.marker, '/api/project/save-cache', {'project': project})
        CentralLibraryTests.operate(self, self.marker, '/api/review/submit')
        submission = self.store.list_submissions(self.reviewer)[0]
        detail = self.store.submission(self.reviewer, submission['id'])
        for page in detail['pages']:
            self.store.decide_page(self.reviewer, submission['id'], page['page_key'], 'approved', '')
        return submission

    def _strip_roots_and_upload_scans(self, submission_id):
        frozen = self.store.preview_root / submission_id / 'project.json'
        project = json.loads(frozen.read_text(encoding='utf-8'))
        for key in ('image_root', 'book_root', 'answer_image_root'):
            project.pop(key, None)
        frozen.write_text(json.dumps(project, ensure_ascii=False), encoding='utf-8')
        target = self.store._source_root(self.book_id)
        originals = {p.name: p for p in (self.sources / 'math-2015').iterdir() if p.is_file()}
        for pages_key in ('pages', 'answer_pages', 'ignored_pages'):
            for page in project.get(pages_key) or []:
                for field in ('image', 'rotated_image'):
                    relative = str((page or {}).get(field) or '').strip().replace('\\', '/').lstrip('/')
                    if not relative:
                        continue
                    source = originals.get(Path(relative).name)
                    if source is None:
                        continue
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())
        return frozen, target

    def test_server_crops_from_uploaded_employee_scans(self):
        submission = self._approved_submission()
        frozen, _target = self._strip_roots_and_upload_scans(submission['id'])
        result = self.store.run_next_delivery_job()
        self.assertEqual(result['status'], 'succeeded', result.get('error'))
        output = Path(result['output_path'])
        self.assertEqual(result['k12']['eligible_count'], 1)
        self.assertGreater((output / 'k12_dataset' / 'dataset.jsonl').stat().st_size, 0)
        self.assertTrue(list(output.rglob('*.jpg')), '服务器没有裁出任何题图')
        generation = json.loads((output / 'generation.json').read_text(encoding='utf-8'))
        self.assertEqual(generation['source_binding']['mode'], 'uploaded_scans')
        self.assertEqual(generation['source_binding']['missing'], 0)
        self.assertNotIn('image_root', json.loads(frozen.read_text(encoding='utf-8')),
                         '冻结工程原件不应被服务器改写')

    def test_missing_uploaded_scans_fails_with_actionable_message(self):
        submission = self._approved_submission()
        _frozen, target = self._strip_roots_and_upload_scans(submission['id'])
        for image in target.rglob('*.jpg'):
            image.unlink()
        with self.assertRaises(ValueError) as caught:
            self.library.generate_delivery(self.reviewer, self.book_id,
                                           expected_submission_id=submission['id'])
        self.assertIn('原书页面图不完整', str(caught.exception))
        self.assertIn('直接重试不会成功', str(caught.exception))

    def test_diagnosis_names_blockers_before_any_generation(self):
        """排队前就能看到缺什么，而且体检本身不产生交付版本。"""
        submission = self._approved_submission()
        self._strip_roots_and_upload_scans(submission['id'])
        report = self.library.diagnose_delivery(self.reviewer, self.book_id)
        self.assertEqual(report['project_source'], 'frozen')
        self.assertEqual(report['review']['pending_count'], 0)
        self.assertEqual(report['review']['approved_count'], report['review']['page_count'])
        self.assertTrue(report['deliverable'], report['reason_summary'])
        self.assertEqual(report['eligible_count'], 1)
        _book, assignment, _root = self.library.book(self.reviewer, self.book_id)
        final_root = self.store._delivery_root(assignment, self.book_id)
        self.assertFalse((final_root / 'current.json').exists(), '体检不得发布交付版本')
        result = self.store.run_next_delivery_job()
        self.assertEqual(result['status'], 'succeeded', result.get('error'))
        self.assertEqual(result['k12']['eligible_count'], report['eligible_count'])

    def test_diagnosis_reports_what_is_still_missing(self):
        """缺字段时给出逐题原因，而不是泛泛的"没有符合规范的题目"。"""
        submission = self._approved_submission()
        frozen, _target = self._strip_roots_and_upload_scans(submission['id'])
        project = json.loads(frozen.read_text(encoding='utf-8'))
        for page in project.get('pages') or []:
            for question in page.get('questions') or []:
                question['difficulty'] = ''
                question['knowledge_points'] = []
        frozen.write_text(json.dumps(project, ensure_ascii=False), encoding='utf-8')
        report = self.library.diagnose_delivery(self.reviewer, self.book_id)
        self.assertFalse(report['deliverable'])
        reasons = {row['reason'] for row in report['reason_summary']}
        self.assertIn('缺少难度', reasons)
        self.assertIn('缺少知识点', reasons)
        self.assertTrue(report['items'])


class PageCoverageTests(unittest.TestCase):
    """生成后的逐页结果：一页到底进包了没有，必须能一眼看懂。"""

    def test_page_status_classification_and_reason_cleanup(self):
        from online_review_beta.central_library import build_page_coverage
        decisions = {("question", 0): "approved", ("question", 1): "approved",
                     ("question", 2): "approved", ("question", 3): "pending",
                     ("question", 4): "approved",
                     ("answer", 0): "approved", ("answer", 1): "pending"}
        items = [
            {"page": 1, "eligible": True, "errors": []},
            {"page": 1, "eligible": True, "errors": []},
            {"page": 2, "eligible": True, "errors": []},
            {"page": 2, "eligible": False, "errors": ["缺少标准答案"]},
            {"page": 3, "eligible": False, "errors": ["缺少题目文本", "缺少有效题图"]},
            {"page": 4, "eligible": False, "errors": ["该题所在页面尚未审核放行"]},
        ]
        report = build_page_coverage(book_id='b1', version='v1', partial=True,
                                     decisions=decisions, items=items, generated_at='now',
                                     only_reviewed=True)
        summary = report['summary']
        self.assertEqual(summary['question_pages'], 5)
        self.assertEqual(summary['delivered_pages'], 1)
        self.assertEqual(summary['partial_pages'], 1)
        self.assertEqual(summary['failed_pages'], 1)
        self.assertEqual(summary['skipped_pages'], 1)
        self.assertEqual(summary['no_question_pages'], 1)
        self.assertEqual(summary['answer_pages'], 2)
        self.assertEqual(summary['answer_skipped_pages'], 1)
        rows = {row['page']: row for row in report['pages'] if row['kind'] == 'question'}
        self.assertEqual(rows[2]['status'], 'partial')
        self.assertEqual(rows[2]['reasons'], ['缺少标准答案'])
        self.assertEqual(rows[3]['status'], 'failed')
        self.assertEqual(len(rows[3]['reasons']), 2)
        # 页级已经说明"未放行"了，题级那句重复原因不该再出现。
        self.assertEqual(rows[4]['reasons'], ['页面未审核放行，整页未进包'])
        self.assertEqual(rows[5]['status'], 'no_question')
        answer = [row for row in report['pages'] if row['kind'] == 'answer']
        self.assertEqual([row['status'] for row in answer], ['delivered', 'skipped'])
        self.assertTrue(report['partial'])
        self.assertEqual(report['generated_at'], 'now')


if __name__ == '__main__':
    unittest.main()
