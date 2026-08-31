"""Acceptance checks the annotation set has to pass before anyone labels it.

Four rounds of this set went out with a summary that did not match the files:
key fields the report described and the writer never emitted, a 68,000-character
block nobody could read, new columns with no permitted values, and a pipeline
label still sitting in the text. Each was caught by hand.

So the populations are asserted rather than reported. Three of them exist -- 231
source blocks, 225 that reached the compressor, 177 delivered -- and every rate
computed later depends on not mixing them, which a summary line cannot enforce
and these can.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from collections import Counter

OUT = "c:/SCP/outputs/block_role_annotation"
BLIND = f"{OUT}/block_role_blind.jsonl"
KEY = f"{OUT}/_disposition_key.jsonl"
SCHEMA = f"{OUT}/annotation_schema.json"


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


@unittest.skipUnless(os.path.exists(BLIND), "annotation set not built")
class PopulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blind = load(BLIND)
        cls.key = load(KEY)
        cls.parents = [k for k in cls.key if k["system_parent"]]

    def test_the_three_populations_are_the_expected_sizes(self) -> None:
        self.assertEqual(sum(1 for k in self.key if k["system_parent"]), 231)
        self.assertEqual(sum(1 for k in self.key if k["source_role_eligible"]), 231)
        self.assertEqual(
            sum(1 for k in self.parents if k["post_compressor_eligible"]), 225
        )
        self.assertEqual(sum(1 for k in self.parents if k["rendered_to_agent"]), 177)

    def test_parent_dispositions_match_the_system(self) -> None:
        counts = Counter(k["final_disposition"] for k in self.parents)

        self.assertEqual(counts["kept"], 177)
        self.assertEqual(counts["dropped_by_budget"], 48)
        self.assertEqual(counts["dropped_by_compressor"], 6)

    def test_one_cluster_per_system_block(self) -> None:
        self.assertEqual(len({k["statistical_cluster_id"] for k in self.parents}), 231)

    def test_only_parents_count_toward_the_drop_rate(self) -> None:
        """Units this script invented must not appear in a system rate."""

        for row in self.key:
            with self.subTest(row=row["annotation_id"]):
                self.assertEqual(
                    row["include_in_system_drop_rate"], row["system_parent"]
                )

    def test_representations_are_labelled(self) -> None:
        counts = Counter(k["representation"] for k in self.key)

        self.assertEqual(counts["raw"], 231)
        self.assertEqual(counts["compressed"], 2)


@unittest.skipUnless(os.path.exists(BLIND), "annotation set not built")
class BlindFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blind = load(BLIND)
        cls.key = {k["annotation_id"]: k for k in load(KEY)}

    def test_ids_are_unique_and_every_one_joins_back(self) -> None:
        """The key is the larger set: it also carries the four containers, which
        are derived from their children rather than labelled."""

        ids = [b["annotation_id"] for b in self.blind]

        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(ids) - set(self.key), set())
        self.assertEqual(len(set(self.key) - set(ids)), 4)

    def test_no_identity_leaks_into_the_blind_file(self) -> None:
        forbidden = {
            "task_id", "block_id", "variant", "final_disposition",
            "raw_block_hash", "block_hash", "system_parent",
        }
        for row in self.blind:
            with self.subTest(row=row["annotation_id"]):
                self.assertEqual(forbidden & set(row), set())

    def test_no_pipeline_wrapper_survives_in_the_text(self) -> None:
        patterns = (
            re.compile(r"(?m)^\s*Evidence:"),
            re.compile(r"NOT verified answer support"),
            re.compile(r"(?i)Candidate answer"),
            re.compile(r"(?m)^\s*Unverified References:\s*$"),
            re.compile(r"(?m)^\s*Grounded Evidence:\s*$"),
        )
        for pattern in patterns:
            hits = [b["annotation_id"] for b in self.blind if pattern.search(b["block_text"])]
            with self.subTest(pattern=pattern.pattern):
                self.assertEqual(hits, [])

    def test_every_row_has_something_to_read(self) -> None:
        """Four blank container rows reached an earlier build.

        A split parent holds no text of its own, and leaving it in the file
        offered an annotator nothing to judge while still inviting a label --
        which would then be counted as a real one.
        """

        blank = [b["annotation_id"] for b in self.blind if not b["block_text"].strip()]

        self.assertEqual(blank, [])

    def test_the_labelable_population_is_272(self) -> None:
        self.assertEqual(len(self.blind), 272)
        by_variant = Counter(self.key[b["annotation_id"]]["variant"] for b in self.blind)
        self.assertEqual(by_variant["raw"], 228)
        self.assertEqual(by_variant["compressed"], 1)
        self.assertEqual(by_variant["child"], 39)
        self.assertEqual(by_variant["compressed_child"], 4)

    def test_containers_are_derived_not_annotated(self) -> None:
        containers = [k for k in self.key.values() if k["container_only"]]

        self.assertEqual(len(containers), 4)
        for row in containers:
            with self.subTest(row=row["annotation_id"]):
                self.assertFalse(row["annotation_required"])
                self.assertEqual(row["role_aggregation"], "union_children")
                self.assertEqual(row["label_source"], "derived_from_children")
                self.assertNotIn(row["annotation_id"], {b["annotation_id"] for b in self.blind})

    def test_labelled_rows_declare_themselves_direct(self) -> None:
        for row in self.blind:
            key = self.key[row["annotation_id"]]
            with self.subTest(row=row["annotation_id"]):
                self.assertTrue(key["annotation_required"])
                self.assertEqual(key["label_source"], "direct")
                self.assertEqual(key["role_aggregation"], "self")

    def test_no_unit_is_too_long_to_read(self) -> None:
        """The 68,000-character block is what the segmenter exists for."""

        oversized = [
            b["annotation_id"] for b in self.blind if len(b["block_text"]) > 6000
        ]

        self.assertEqual(oversized, [])

    def test_a_segmented_parent_carries_no_text_of_its_own(self) -> None:
        """Otherwise its content is annotated twice, once whole and once in parts."""

        for row in self.blind:
            key = self.key[row["annotation_id"]]
            if key.get("child_index"):
                continue
            if key["variant"] == "raw" and not row["block_text"]:
                self.assertIsNotNone(key.get("statistical_cluster_id"))


@unittest.skipUnless(os.path.exists(SCHEMA), "schema not built")
class SchemaTest(unittest.TestCase):
    def test_every_annotated_column_has_permitted_values(self) -> None:
        schema = json.load(open(SCHEMA, encoding="utf-8"))
        blind = load(BLIND)[0]
        annotated = {
            k for k, v in blind.items()
            if v in ("", []) and k not in ("notes",)
        }
        described = set(schema) | {"representation_role"}
        for column in annotated:
            with self.subTest(column=column):
                self.assertIn(
                    column.replace("representation_role", "roles"),
                    described | {"roles"},
                )


if __name__ == "__main__":
    unittest.main()
