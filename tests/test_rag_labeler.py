from __future__ import annotations

import unittest
from pathlib import Path

import torch
from transformers import DebertaV2Tokenizer

from tools.search_result_builder.source_analyze.rag_labeler import (
    CONTINUE_TAG,
    EfficientRAGLabelerAdapter,
    PROJECT_LABELER_CHECKPOINT,
)


class RAGLabelerTests(unittest.TestCase):
    def test_default_checkpoint_uses_project_mixed_labeler_directory(self):
        labeler = EfficientRAGLabelerAdapter()

        self.assertEqual(
            Path(labeler.labeler_checkpoint),
            PROJECT_LABELER_CHECKPOINT,
        )
        self.assertTrue(
            Path(labeler.labeler_checkpoint, "config.json").exists()
        )

    def test_missing_checkpoint_uses_explicit_fallback(self):
        labeler = EfficientRAGLabelerAdapter(
            labeler_checkpoint="missing-labeler-checkpoint",
        )

        result = labeler.label_text(
            question="Moon minimum perigee distance",
            text="The Moon has a minimum perigee distance.",
        )

        self.assertEqual(result.label, "useful")
        self.assertEqual(
            result.metadata["method"],
            "efficientrag_labeler_fallback",
        )
        self.assertIn("FileNotFoundError", result.metadata["model_error"])

    def test_aligns_exact_word_and_multiword_span(self):
        labeler = EfficientRAGLabelerAdapter()

        word, word_repaired = labeler._align_span_to_text(
            decoded_span="perigee",
            text="The minimum perigee distance is listed here.",
        )
        phrase, phrase_repaired = labeler._align_span_to_text(
            decoded_span="United States",
            text="She served as Chief of Protocol of the United States.",
        )

        self.assertEqual(word, "perigee")
        self.assertFalse(word_repaired)
        self.assertEqual(phrase, "United States")
        self.assertFalse(phrase_repaired)

    def test_repairs_unique_sentencepiece_prefix_to_full_word(self):
        labeler = EfficientRAGLabelerAdapter()

        aligned, repaired = labeler._align_span_to_text(
            decoded_span="Kipcho",
            text="Eliud Kipchoge maintained a record marathon pace.",
        )

        self.assertEqual(aligned, "Kipchoge")
        self.assertTrue(repaired)

    def test_rejects_unmatched_or_ambiguous_prefix(self):
        labeler = EfficientRAGLabelerAdapter()

        unmatched = labeler._align_span_to_text(
            decoded_span="perigeecho",
            text="The minimum perigee distance is listed here.",
        )
        ambiguous = labeler._align_span_to_text(
            decoded_span="Kipcho",
            text="Kipchoge and Kipchomai are both present.",
        )

        self.assertEqual(unmatched, ("", False))
        self.assertEqual(ambiguous, ("", False))

    def test_extracts_only_positive_tokens_from_passage_region(self):
        labeler = EfficientRAGLabelerAdapter()
        labeler._tokenizer = DebertaV2Tokenizer.from_pretrained(
            PROJECT_LABELER_CHECKPOINT
        )
        tokenized = labeler._build_inputs(
            question="Who is Kipchoge?",
            texts=["Eliud Kipchoge maintained a record marathon pace."],
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = torch.zeros_like(input_ids)
        passage_start, passage_end = labeler._passage_token_bounds(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tokens = labeler._tokenizer.convert_ids_to_tokens(
            input_ids[passage_start:passage_end].tolist()
        )
        kip_start = tokens.index("▁Kip")
        labels[passage_start + kip_start] = 1
        labels[passage_start + kip_start + 1] = 1
        labels[1] = 1

        spans, metadata = labeler._extract_useful_spans(
            text="Eliud Kipchoge maintained a record marathon pace.",
            input_ids=input_ids,
            token_labels=labels,
            attention_mask=attention_mask,
        )

        self.assertEqual(spans, ["kipchoge"])
        self.assertEqual(
            metadata["repaired_useful_spans"],
            ["Kipcho -> Kipchoge"],
        )


if __name__ == "__main__":
    unittest.main()
