"""本地恢复草稿不能把"保存成功"变成"保存失败"。

背景：草稿写入曾经用固定的 <book_id>.tmp 做 os.replace。自动保存每秒触发两次，
两个请求同时写同一个 .tmp 时，Windows 会让后到的替换报 [WinError 5] 拒绝访问，
异常一路冒到界面。服务器那份其实已经写成功了，员工看到的却是"保存失败"。
"""
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from desktop_annotation import paperflow_annotation_api as api


class RecoveryDraftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def _tmp_leftovers(self):
        return [p.name for p in self.dir.iterdir() if p.name.endswith('.tmp')]

    def test_writes_draft_and_leaves_no_temp_file(self):
        target = (self.dir / ('B' + 'A' * 32)).with_suffix('.json')
        api._write_recovery_draft(target, {'book_id': 'x', 'synced': False})
        self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['book_id'], 'x')
        self.assertEqual(self._tmp_leftovers(), [])

    def test_unwritable_target_is_skipped_not_raised(self):
        """目标只读时 Windows 会报 WinError 5，这里只要求不抛、不留垃圾。"""
        target = (self.dir / ('B' + 'A' * 32)).with_suffix('.json')
        target.write_text('{"old": true}', encoding='utf-8')
        os.chmod(target, stat.S_IREAD)
        self.addCleanup(lambda: os.chmod(target, stat.S_IWRITE))
        try:
            api._write_recovery_draft(target, {'book_id': 'new'})
        finally:
            os.chmod(target, stat.S_IWRITE)
        # 旧内容仍在，说明这次写入被跳过；关键是没有异常、没有残留临时文件。
        self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'old': True})
        self.assertEqual(self._tmp_leftovers(), [])

    def test_concurrent_writes_never_raise_and_leave_valid_json(self):
        target = (self.dir / ('B' + 'A' * 32)).with_suffix('.json')
        errors = []

        def worker(index):
            try:
                for round_index in range(30):
                    api._write_recovery_draft(target, {'book_id': 'b', 'round': index,
                                                       'n': round_index})
            except BaseException as error:  # noqa: BLE001 - 测试要抓全部
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertIn('book_id', json.loads(target.read_text(encoding='utf-8')))
        self.assertEqual(self._tmp_leftovers(), [])


if __name__ == '__main__':
    unittest.main()
