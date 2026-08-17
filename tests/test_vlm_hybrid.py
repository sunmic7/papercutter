import unittest

import vlm_hybrid


class VlmHybridContractTests(unittest.TestCase):
    def test_normalize_keeps_primary_groups_and_answer_card_regions(self):
        result = vlm_hybrid.normalize_payload({
            "page_type": "question",
            "page_confidence": 0.91,
            "groups": [{"id": "g1", "bbox": [10, 20, 500, 900], "primary_numbers": [1, 2]}],
            "ignore_regions": [{"id": "a1", "type": "answer_card", "bbox": [600, 20, 980, 900]}],
        }, 2000, 1000)
        self.assertEqual(result["groups"][0]["bbox"], [20, 20, 1000, 900])
        self.assertEqual(result["ignore_regions"][0]["bbox"], [1200, 20, 1960, 900])

    def test_normalize_keeps_a_short_paper_title(self):
        result = vlm_hybrid.normalize_payload({
            "page_type": "question", "paper_title": "第一单元检测卷",
            "groups": [], "ignore_regions": [],
        }, 1000, 1000)
        self.assertEqual(result["paper_title"], "第一单元检测卷")

    def test_invalid_model_box_is_dropped_not_raised(self):
        result = vlm_hybrid.normalize_payload({
            "page_type": "question", "page_confidence": 1,
            "groups": [{"id": "bad", "bbox": [100, 100, 100, 200]}],
            "ignore_regions": [],
        }, 1000, 1000)
        self.assertEqual(result["groups"], [])

    def test_answer_sheet_skip_needs_high_confidence_and_no_groups(self):
        self.assertTrue(vlm_hybrid.should_skip_page(
            {"page_type": "answer_sheet", "page_confidence": 0.95, "groups": []}, "question",
        ))
        self.assertFalse(vlm_hybrid.should_skip_page(
            {"page_type": "answer_sheet", "page_confidence": 0.80, "groups": []}, "question",
        ))
        self.assertFalse(vlm_hybrid.should_skip_page(
            {"page_type": "answer_sheet", "page_confidence": 0.99, "groups": [{"id": "g"}]}, "question",
        ))

    def test_missing_confidence_is_only_promoted_for_a_group_free_non_question_page(self):
        answer_key = vlm_hybrid.normalize_payload(
            {"page_type": "answer_key", "groups": [], "ignore_regions": []}, 1000, 1000,
        )
        question = vlm_hybrid.normalize_payload(
            {"page_type": "question", "groups": [], "ignore_regions": []}, 1000, 1000,
        )
        self.assertGreaterEqual(answer_key["page_confidence"], 0.93)
        self.assertLess(question["page_confidence"], 0.93)

    def test_only_crop_almost_entirely_in_answer_card_is_suppressed(self):
        cuts = [
            {"_source_box": [10, 10, 90, 90]},
            {"_source_box": [110, 10, 190, 90]},
        ]
        result, stats = vlm_hybrid.apply_to_rule_cuts(cuts, {
            "groups": [{"id": "q1", "bbox": [0, 0, 100, 100]}],
            "ignore_regions": [{"id": "card", "type": "answer_card", "bbox": [105, 0, 200, 100]}],
        })
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_vlm_group_id"], "q1")
        self.assertEqual(stats["answer_card_suppressed"], 1)

    def test_fenced_json_is_accepted(self):
        payload = vlm_hybrid._json_object("```json\n{\"page_type\": \"question\"}\n```")
        self.assertEqual(payload["page_type"], "question")

    def test_answer_payload_requires_numbered_valid_boxes(self):
        result = vlm_hybrid.normalize_answer_payload({
            "page_type": "answer_key",
            "entries": [
                {"id": "a1", "major_number": 2, "question_number": 3, "bbox": [10, 20, 500, 80]},
                {"id": "bad", "major_number": 2, "question_number": 0, "bbox": [10, 20, 500, 80]},
            ],
        }, 1000, 1000)
        self.assertEqual(result["entries"], [{"id": "a1", "major_number": 2, "question_number": 3, "bbox": [10, 20, 500, 80]}])

    def test_answer_section_payload_keeps_only_valid_major_boxes(self):
        result = vlm_hybrid.normalize_answer_section_payload({
            "page_type": "answer_key",
            "sections": [
                {"id": "a", "major_number": 1, "bbox": [0, 0, 500, 500]},
                {"id": "bad", "major_number": 0, "bbox": [0, 0, 500, 500]},
            ],
        }, 200, 100)
        self.assertEqual(result["page_type"], "answer_key")
        self.assertEqual(result["sections"], [{
            "id": "a", "major_number": 1, "bbox": [0, 0, 100, 50],
        }])


if __name__ == "__main__":
    unittest.main()
