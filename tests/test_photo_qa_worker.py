from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from review_tools import photo_qa_worker as worker


class PhotoQaWorkerTests(unittest.TestCase):
    def test_worker_persists_completed_status_without_touching_source_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"
            batch = {"batch": "大学物理10.24", "items": [], "dataset": [], "review_queue": []}
            report = {"batch": "大学物理10.24", "eligible_count": 0, "review_count": 1}
            with patch.object(worker, "build_batch", return_value=batch) as build, patch.object(worker, "write_batch", return_value=report):
                code = worker.run(["F:/source/大学物理10.24"], str(output), semantic_review=False)
            self.assertEqual(code, 0)
            build.assert_called_once()
            status = json.loads((output / "job_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["reports"], [report])


if __name__ == "__main__":
    unittest.main()
