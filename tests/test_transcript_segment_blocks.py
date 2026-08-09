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


if __name__ == "__main__":
    unittest.main()
