"""Pin that an audio transcript reaches fact extraction as segments, not a blob.

`AttachmentFactExtractor` builds one semantic unit per text block. A PowerPoint
arrives as one block per slide and qwen3:4b returns a grounded fact for each --
`input_count: 8, fact_count: 8` on level1_final_16 task 025. Audio used to
arrive as a single block holding the whole recording, and the same model
returned nothing usable: `input_count: 1, fact_count: 0` on task 031 and
`fact_count: 1, grounded_count: 0` on task 045. Both then fell back to an Agent
reading raw transcript text and dropped list items -- 031 lost one ingredient of
five, 045 returned two page numbers of five -- with every expected item present
in the transcript, so nothing was lost in transcription.

Whisper already emits the segmentation; it was simply flattened into one string.
These tests hold the split to those timestamps, keep model and language headers
out of the content blocks, and keep the old single-block behaviour for anything
without timestamps.
"""

from __future__ import annotations

import unittest

from tools.attachment_reader.models import ParsedAttachmentPayload
from tools.attachment_reader.payload_builder import AttachmentPayloadBuilder

TRANSCRIPT = (
    "Audio transcription:\n"
    "- faster_whisper_model: base\n"
    "- device: cuda\n"
    "- detected_language: en confidence=1.00\n"
    "- question_focus: list the ingredients\n"
    "Transcript:\n"
    "[0.00-5.84] You will need ripe strawberries and granulated sugar.\n"
    "[5.84-11.12] Add cornstarch to thicken it up.\n"
    "[11.12-16.72] Then pure vanilla extract.\n"
    "[16.72-21.20] And freshly squeezed lemon juice.\n"
)


def _blocks(content: str) -> list:
    payload = ParsedAttachmentPayload(provenance={})
    AttachmentPayloadBuilder()._read_transcript_segments(content, payload)
    return payload.text_blocks


class TranscriptSegmentBlockTest(unittest.TestCase):
    def test_each_timestamped_segment_becomes_its_own_block(self) -> None:
        blocks = _blocks(TRANSCRIPT)

        self.assertEqual(len(blocks), 4)
        self.assertEqual([block.page for block in blocks], [1, 2, 3, 4])
        self.assertTrue(all(block.block_type == "transcript_segment" for block in blocks))

    def test_every_spoken_item_survives_the_split(self) -> None:
        """The failure was losing list items, so check they all still arrive."""

        joined = " ".join(block.text for block in _blocks(TRANSCRIPT)).casefold()

        for item in (
            "ripe strawberries",
            "granulated sugar",
            "cornstarch",
            "pure vanilla extract",
            "freshly squeezed lemon juice",
        ):
            self.assertIn(item, joined)

    def test_header_metadata_does_not_become_a_content_block(self) -> None:
        blocks = _blocks(TRANSCRIPT)

        self.assertTrue(all("faster_whisper" not in block.text for block in blocks))
        self.assertTrue(all("question_focus" not in block.text for block in blocks))

    def test_a_transcript_without_timestamps_stays_one_block(self) -> None:
        blocks = _blocks("Audio transcription:\n- model: base\nTranscript:\n(empty transcription)")

        self.assertEqual(len(blocks), 1)

    def test_empty_content_produces_nothing(self) -> None:
        self.assertEqual(_blocks(""), [])

    def test_audio_extensions_route_to_the_splitter(self) -> None:
        builder = AttachmentPayloadBuilder()
        for extension in (".mp3", ".wav", ".mp4"):
            with self.subTest(extension=extension):
                payload = ParsedAttachmentPayload(provenance={})
                builder._read_transcript_segments(TRANSCRIPT, payload)
                self.assertEqual(len(payload.text_blocks), 4)


class TranscriptSurvivesTheAttachmentBudgetTest(unittest.TestCase):
    """The segments were reaching fact extraction and still losing their tail.

    Splitting the transcript fixed how it was chunked, not whether it arrived.
    `max_attachment_chars` is 1200 and the header carried the whole question
    back as `question_focus`, which ran to 1052 characters on level1_final_20
    task 031 -- 88% of the budget spent restating text the Agent already had,
    leaving 152 characters of transcript. The cut landed on `and cornstarch.
    Cook t ...` and took `pure vanilla extract` with it, the single ingredient
    the answer omitted. Task 045 lost `132, 133, 134` from its closing segment
    and answered `197, 245`, exactly what survived.

    Neither transcript was too long: 455 and 660 characters against a budget of
    1200. Only the header made them not fit.
    """

    # The real header, minus the question, at its real length.
    HEADER = (
        "Audio transcription:\n"
        "- faster_whisper_model: base\n"
        "- device: cuda\n"
        "- compute_type: float16\n"
        "- detected_language: en confidence=1.00\n"
        "Transcript:\n"
    )
    # Task 031's question verbatim, 885 characters, which is the point: the
    # header it produced ran to 1052 against a 1200 budget. A paraphrase is not
    # interchangeable here -- a shorter one leaves room and the cut moves.
    QUESTION = (
        "Hi, I'm making a pie but I could use some help with my shopping list. I have "
        "everything I need for the crust, but I'm not sure about the filling. I got the "
        "recipe from my friend Aditi, but she left it as a voice memo and the speaker on "
        "my phone is buzzing so I can't quite make out what she's saying. Could you "
        "please listen to the recipe and list all of the ingredients that my friend "
        "described? I only want the ingredients for the filling, as I have everything I "
        "need to make my favorite pie crust. I've attached the recipe as Strawberry "
        "pie.mp3.\n\nIn your response, please only list the ingredients, not any "
        "measurements. So if the recipe calls for \"a pinch of salt\" or \"two cups of "
        "ripe strawberries\" the ingredients on the list would be \"salt\" and \"ripe "
        "strawberries\".\n\nPlease format your response as a comma separated list of "
        "ingredients. Also, please alphabetize the ingredients."
    )
    BODY = (
        "[0.00-5.84] In a saucepan, combine ripe strawberries, granulated sugar, "
        "freshly squeezed lemon juice\n"
        "[5.84-11.12] and cornstarch. Cook the mixture over medium heat, stirring "
        "constantly until it thickens to\n"
        "[11.12-16.72] a smooth consistency. Remove from heat and stir in a dash of "
        "pure vanilla extract.\n"
        "[16.72-21.20] Allow the strawberry pie filling to cool before using it as a "
        "delicious and fruity filling for\n"
        "[21.20-31.20] your pie crust."
    )
    INGREDIENTS = (
        "ripe strawberries",
        "granulated sugar",
        "freshly squeezed lemon juice",
        "cornstarch",
        "pure vanilla extract",
    )

    def _fit(self, content: str) -> str:
        from context.context_budget import ContextBudget, ContextBudgetManager

        return ContextBudgetManager()._truncate(
            content, ContextBudget().max_attachment_chars
        )

    def test_every_ingredient_survives_the_budget(self) -> None:
        kept = self._fit(self.HEADER + self.BODY).casefold()

        for item in self.INGREDIENTS:
            with self.subTest(item=item):
                self.assertIn(item, kept)

    def test_the_question_in_the_header_is_what_used_to_cut_it(self) -> None:
        """Guard the regression direction, and the size of the effect."""

        with_question = (
            "Audio transcription:\n"
            "- faster_whisper_model: base\n"
            "- device: cuda\n"
            "- compute_type: float16\n"
            "- detected_language: en confidence=1.00\n"
            f"- question_focus: {self.QUESTION}\n"
            "Transcript:\n"
        ) + self.BODY
        kept = self._fit(with_question).casefold()

        self.assertNotIn("pure vanilla extract", kept)
        self.assertIn("ripe strawberries", kept)


if __name__ == "__main__":
    unittest.main()
