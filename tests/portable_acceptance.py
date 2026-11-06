"""Run the copied portable distribution using only its bundled interpreter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time


def worker(root, report):
    sys.path.insert(0, str(root))
    import cv2
    import numpy as np
    import paddle
    import paddleocr
    import onnxruntime
    import PySide6
    from PIL import Image
    from desktop_annotation import paperflow_annotation_api as api
    from review_tools import manual_annotation_data as data
    import cut_questions_v4 as core
    import layout_arbiter
    from question_detector import QuestionDetector
    from answer_region_detector import AnswerRegionDetector
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
        print(name, flush=True)
    check('bundled_python', pathlib.Path(sys.executable).is_relative_to(root))
    for module in (cv2, np, paddle, paddleocr, onnxruntime, PySide6, Image, api, data):
        check('bundled_module_' + module.__name__, pathlib.Path(module.__file__).is_relative_to(root))
    # 桥版本两端必须一致：Electron 只按这个字符串决定复用还是重启本地服务。
    # 直接和 main.js 比对，升级桥时不用再改这行。
    electron_main = (pathlib.Path(root) / 'desktop_annotation' / 'PaperFlow-Annotation'
                     / 'resources' / 'app' / 'electron' / 'main.js')
    electron_revision = re.search(r"PAPERFLOW_BRIDGE_REVISION = '([^']+)'",
                                  electron_main.read_text(encoding='utf-8'))
    check('current_bridge', bool(electron_revision)
          and electron_revision.group(1) == api.PAPERFLOW_BRIDGE_REVISION)
    with tempfile.TemporaryDirectory(prefix='portable-book-', dir=report.parent) as directory:
        book = pathlib.Path(directory) / '\u590d\u5236\u540e\u7684\u6559\u6750'
        book.mkdir()
        image = np.full((600, 1000, 3), 255, np.uint8)
        cv2.putText(image, '1. 12 + 8 = 20', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
        cv2.putText(image, '2. 6 x 5 = 30', (70, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
        image_path = book / '1.jpg'
        cv2.imencode('.jpg', image)[1].tofile(str(image_path))
        api.ROOT = pathlib.Path(directory)
        opened = api._open_book({'book_root': str(book)})
        # Internal function test only: the public workbench still requires login.
        check('internal_local_book_pipeline', len(opened['project']['pages']) == 1 and not api.REVIEW_SESSION.status()['logged_in'])
        project = opened['project']
        record = data.new_record([20, 30, 900, 220], major_number='3', minor_number='2', text='12 + 8 = 20')
        project['pages'][0]['questions'] = [record]
        saved = api._save_project_cache({'project_path': opened['project_path'], 'project': project,
                                       'base_revision': project['project_revision']})
        check('local_save_and_reopen', data.load_project(opened['project_path'])['pages'][0]['questions'][0]['text'] == record['text'])
        check('local_revision', saved['project_revision'] > 0)
        for name, detector in [('question_detector', QuestionDetector()), ('answer_region_detector', AnswerRegionDetector())]:
            check(name + '_load', detector._load())
            check(name + '_inference', isinstance(detector.detect(image), list))
        check('offline_fast_ocr_load', core.get_fast_ocr_engine() is not None)
        fast = core._run_fast_ocr(image)
        check('offline_fast_ocr_inference', bool(fast))
        check('offline_high_ocr_load', core.get_ocr_engine() is not None)
        high = core._run_ocr(image)
        check('offline_high_ocr_inference', bool(high))
        prelabel = api._prelabel({'project_path': opened['project_path'], 'image_path': str(image_path),
                                 'page_index': 0, 'kind': 'question', 'profile': 'exam'})
        check('offline_auto_frame_pipeline', prelabel.get('ok') and bool(prelabel.get('records')))
        check('bundled_layout_model', pathlib.Path(layout_arbiter._model_dir()).is_relative_to(root))
        check('offline_layout_load', layout_arbiter._get_pipe() is not None)
    result = {'ok': True, 'checks': checks, 'python': sys.executable, 'offline': True}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def parent(root, report):
    report.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise AssertionError('Manifest mismatch: ' + item['path'])
    print('Manifest verified:', len(manifest['files']), flush=True)
    launcher = subprocess.run([str(root / '启动切题程序.exe'), '--check'], cwd=report.parent, timeout=20)
    if launcher.returncode:
        raise AssertionError('Portable EXE launcher validation failed')
    with tempfile.TemporaryDirectory(prefix='portable-clean-home-', dir=report.parent) as directory:
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith(('PYTHON', 'PADDLE', 'PAPERCUTTER', 'PAPERFLOW'))}
        env.update({'PATH': str(root / 'runtime/python') + os.pathsep + str(pathlib.Path(os.environ['WINDIR']) / 'System32'),
            'USERPROFILE': directory, 'HOME': directory, 'APPDATA': directory, 'LOCALAPPDATA': directory,
            'PADDLE_PDX_CACHE_HOME': str(pathlib.Path(directory) / 'paddlex'),
            'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK': 'True', 'PYTHONUTF8': '1',
            'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
            'PAPERFLOW_DISABLE_PROFILE_DETECT': '1', 'PAPERFLOW_DISABLE_PREWARM': '1',
            'PAPERFLOW_MODEL_CACHE': str(report.parent / 'portable-model-cache-20261023'),
            'HTTP_PROXY': 'http://127.0.0.1:9', 'HTTPS_PROXY': 'http://127.0.0.1:9'})
        log = report.with_suffix('.log')
        with log.open('w', encoding='utf-8') as output:
            result = subprocess.run([str(root / 'runtime/python/python.exe'), '-B', str(pathlib.Path(__file__).resolve()),
                str(root), '--report', str(report), '--worker'], cwd=directory, env=env,
                stdout=output, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            print(log.read_text(encoding='utf-8')[-7000:])
            raise AssertionError('Bundled runtime/model acceptance failed; see ' + str(log))
    print(json.dumps({'ok': True, 'files': len(manifest['files']), 'checks': len(json.loads(report.read_text(encoding='utf-8'))['checks'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=pathlib.Path)
    parser.add_argument('--report', type=pathlib.Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    (worker if args.worker else parent)(args.root.resolve(), args.report.resolve())
