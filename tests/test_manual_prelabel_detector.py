from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_tools import manual_annotation_workspace as workspace


class ManualPrelabelDetectorTests(unittest.TestCase):
    def test_detector_is_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPERCUTTER_MANUAL_PRELABEL_DETECTOR", None)
            self.assertFalse(workspace._manual_prelabel_detector_enabled())

    def test_detector_failure_falls_back_to_rule_path(self) -> None:
        cutter = SimpleNamespace(
            detect_question_blocks_with_model=mock.Mock(side_effect=RuntimeError("bad model"))
        )
        with mock.patch.dict(
            os.environ, {"PAPERCUTTER_MANUAL_PRELABEL_DETECTOR": "1"}, clear=False,
        ):
            self.assertEqual(workspace._optional_question_detections(cutter, np.zeros((2, 2, 3))), [])

    def test_fusion_delegates_only_to_conservative_refinement(self) -> None:
        minors = [{"_source_box": [0, 0, 20, 20]}]
        detections = [object()]
        refined = [{"_source_box": [1, 1, 19, 19], "_boundary_source": "detector_assisted"}]
        refine = mock.Mock(return_value=(refined, {"refined": 1}))
        cutter = SimpleNamespace(
            QUESTION_DETECTOR_ASSIST_IOU=0.55,
            question_detector=SimpleNamespace(refine_legacy_cuts=refine),
        )

        result = workspace._refine_manual_prelabel_minors(
            cutter, minors, np.zeros((20, 20, 3), dtype=np.uint8), detections,
        )

        self.assertIs(result, refined)
        refine.assert_called_once()
        self.assertEqual(refine.call_args.kwargs["min_iou"], 0.55)


if __name__ == "__main__":
    unittest.main()
