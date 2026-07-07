import unittest

from tools.tool_capability_registry import ToolCapabilityRegistry
from tools.tool_planner.schema import ToolNeed


class ToolCapabilityRegistryTests(unittest.TestCase):
    def test_matches_youtube_transcript_need(self) -> None:
        registry = ToolCapabilityRegistry()
        steps = registry.match_steps(
            needs=[
                ToolNeed(
                    need_type="video",
                    required_capabilities=["youtube_url", "transcript"],
                    input_refs=["question.youtube_url"],
                )
            ],
            candidate_tool_names=["video_transcript", "search"],
            available_inputs=["question", "question.youtube_url", "youtube_url"],
        )

        self.assertEqual([step.tool_name for step in steps], ["video_transcript"])

    def test_infers_video_need_from_question_url(self) -> None:
        registry = ToolCapabilityRegistry()
        needs = registry.infer_needs(
            question="In https://www.youtube.com/watch?v=L1vXCYZAYYM, what is said?",
            attachment={},
            routing={},
        )

        self.assertEqual(needs[0].need_type, "video")
        self.assertIn("transcript", needs[0].required_capabilities)

    def test_attachment_table_need_matches_attachment_reader(self) -> None:
        registry = ToolCapabilityRegistry()
        needs = registry.infer_needs(
            question="Analyze the spreadsheet.",
            attachment={"file_path": "scores.xlsx", "extension": ".xlsx"},
            routing={},
        )
        steps = registry.match_steps(
            needs=needs,
            candidate_tool_names=["attachment_reader", "search"],
            available_inputs=registry.available_inputs(
                question="Analyze the spreadsheet.",
                attachment={"file_path": "scores.xlsx", "extension": ".xlsx"},
            ),
        )

        self.assertEqual([step.tool_name for step in steps], ["attachment_reader"])


if __name__ == "__main__":
    unittest.main()
