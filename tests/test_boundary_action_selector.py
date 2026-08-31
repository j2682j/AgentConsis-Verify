"""The validator is what makes an unreliable selector safe to run.

A 4b model asked to copy a span out of a sentence will sometimes paraphrase it,
sometimes return a boundary no generator proposed, and sometimes answer KEEP
while handing back different text. None of those may reach the pipeline: the
standing rule is that a task which is already answered correctly must not
regress, and a wrong boundary repair is exactly how that happens.

So every one of those failures has to land on DEFER, and DEFER has to leave the
original span untouched. These tests are written from the failure side -- what
the model has to do wrong for the span to survive unchanged.
"""

from __future__ import annotations

import unittest

from score.boundary_action_selector import (
    SelectorInput,
    apply,
    build_messages,
    parse,
    validate,
)

CONTEXT = "Who are the pitchers with the number before and after Taisho Tamai's number?"
SPAN = (CONTEXT.index("pitchers"), CONTEXT.index("pitchers") + len("pitchers"))
WIDER = (CONTEXT.index("the pitchers"), CONTEXT.index("number?") + len("number"))

ITEM = SelectorInput(
    annotation_id="T001",
    context=CONTEXT,
    span=SPAN,
    question_role="a list of two names",
    answer_target="pitcher names",
)
ALLOWED = {SPAN, WIDER}


def reply(action: str, selected: str = "") -> str:
    return f'{{"action": "{action}", "selected_text": "{selected}"}}'


class PromptTest(unittest.TestCase):
    def test_the_span_is_marked_in_place(self) -> None:
        marked = ITEM.marked_context()

        self.assertIn("[[pitchers]]", marked)
        self.assertEqual(marked.replace("[[", "").replace("]]", ""), CONTEXT)

    def test_the_candidate_list_is_not_in_the_prompt(self) -> None:
        """261 offsets in a prompt is the thing this design exists to avoid.

        The ceiling is loose on purpose. It is not a token budget -- it is a
        tripwire for the lattice leaking into the prompt, and the smallest
        lattice here would blow straight past it.
        """

        body = " ".join(m["content"] for m in build_messages(ITEM))

        self.assertNotIn("candidate", body.casefold())
        self.assertLess(len(body), 2500)


class ParseTest(unittest.TestCase):
    def test_a_fenced_object_is_read(self) -> None:
        self.assertEqual(
            parse('```json\n{"action": "DROP", "selected_text": ""}\n```'),
            ("DROP", ""),
        )

    def test_thinking_is_discarded_before_parsing(self) -> None:
        raw = '<think>hmm, the span looks short</think>{"action":"DEFER","selected_text":""}'

        self.assertEqual(parse(raw), ("DEFER", ""))

    def test_an_object_buried_in_prose_is_recovered(self) -> None:
        raw = 'Here is my answer:\n{"action": "KEEP", "selected_text": "pitchers"}\nHope that helps.'

        self.assertEqual(parse(raw), ("KEEP", "pitchers"))

    def test_output_with_no_object_is_refused(self) -> None:
        self.assertIsNone(parse("I think the span should stay as it is."))


class ValidationTest(unittest.TestCase):
    def test_a_replacement_from_the_lattice_is_accepted(self) -> None:
        decision = validate(reply("REPLACE", CONTEXT[WIDER[0] : WIDER[1]]), ITEM, ALLOWED)

        self.assertEqual(decision.action, "REPLACE")
        self.assertEqual((decision.start, decision.end), WIDER)

    def test_a_paraphrase_defers(self) -> None:
        """`the pitcher` is not in the text; the model rewrote it."""

        decision = validate(reply("REPLACE", "the pitcher with the number"), ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "not_in_context")

    def test_a_boundary_outside_the_frozen_set_defers(self) -> None:
        """Verbatim, but no generator proposed it, so it cannot be checked."""

        decision = validate(reply("REPLACE", "number before and after"), ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "not_in_candidate_set")

    def test_a_selection_occurring_twice_defers(self) -> None:
        context = "the number before and after Tamai's number"
        item = SelectorInput("T002", context, (0, 10))
        allowed = {(4, 10), (36, 42)}

        decision = validate(reply("REPLACE", "number"), item, allowed)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "ambiguous_occurrence")

    def test_keep_must_hand_back_the_same_span(self) -> None:
        decision = validate(reply("KEEP", CONTEXT[WIDER[0] : WIDER[1]]), ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "keep_must_equal_span")

    def test_replace_with_the_original_span_is_recorded_as_keep(self) -> None:
        """Otherwise a no-op inflates the mutation rate the controls measure."""

        decision = validate(reply("REPLACE", "pitchers"), ITEM, ALLOWED)

        self.assertEqual(decision.action, "KEEP")
        self.assertIn("replace_to_keep", decision.normalisations)

    def test_drop_is_no_longer_in_the_contract(self) -> None:
        """It was, and it deleted 47 of 90 already-correct spans.

        A model that still emits DROP is answering a question this module does
        not ask any more, so the span survives rather than being deleted on the
        strength of an answer to a different question.
        """

        decision = validate(reply("DROP"), ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "action_not_in_contract")
        self.assertEqual(apply(decision, ITEM), SPAN)

    def test_the_prompt_offers_no_way_to_remove_a_span(self) -> None:
        body = " ".join(m["content"] for m in build_messages(ITEM))

        self.assertNotIn("DROP", body)
        self.assertIn("KEEP|REPLACE|DEFER", body)

    def test_an_unknown_action_defers(self) -> None:
        decision = validate(reply("EXPAND", "pitchers"), ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "unknown_action")

    def test_unparseable_output_defers(self) -> None:
        decision = validate("no idea", ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "malformed_output")

    def test_replace_with_nothing_defers(self) -> None:
        decision = validate(reply("REPLACE"), ITEM, ALLOWED)

        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(decision.defer_reason, "empty_selection")


class MarkerTest(unittest.TestCase):
    """The brackets are injected by this module, so copying them back is not
    the model's mistake to be punished for.

    In the first shadow run 15 of 17 grounding failures were exactly this --
    `"[[Moon]]"` returned against a context that contains `Moon`.
    """

    def test_the_exact_injected_marking_is_undone(self) -> None:
        """`[[pitchers]]` is what this module wrote into the context."""

        marked = CONTEXT[WIDER[0] : WIDER[1]].replace("pitchers", "[[pitchers]]")
        decision = validate(reply("REPLACE", marked), ITEM, ALLOWED)

        self.assertEqual(decision.action, "REPLACE")
        self.assertEqual((decision.start, decision.end), WIDER)

    def test_brackets_that_were_not_injected_are_left_alone(self) -> None:
        """A blanket bracket strip would edit the model's answer, silently.

        The question here really contains `[[Doctor Who]]`, and the span is
        something else, so nothing was injected around it. Rewriting it would
        be this module repairing an answer rather than undoing its own edit --
        so the selection stands or falls on its own, as ungrounded.
        """

        context = "Which episode of [[Doctor Who]] aired first?"
        item = SelectorInput("T003", context, (0, 5))

        decision = validate(reply("REPLACE", "[[Doctor Who]]"), item, set())

        self.assertFalse(decision.marker_stripped)
        self.assertEqual(decision.raw_model_text, "[[Doctor Who]]")
        self.assertEqual(decision.defer_reason, "not_in_candidate_set")

    def test_both_texts_are_kept_side_by_side(self) -> None:
        marked = CONTEXT[WIDER[0] : WIDER[1]].replace("pitchers", "[[pitchers]]")
        decision = validate(reply("REPLACE", marked), ITEM, ALLOWED)

        self.assertTrue(decision.marker_stripped)
        self.assertEqual(decision.raw_model_text, marked)
        self.assertEqual(decision.marker_stripped_text, CONTEXT[WIDER[0] : WIDER[1]])
        self.assertIn("marker_stripped", decision.normalisations)

    def test_an_untouched_selection_records_nothing(self) -> None:
        decision = validate(reply("REPLACE", CONTEXT[WIDER[0] : WIDER[1]]), ITEM, ALLOWED)

        self.assertEqual(decision.normalisations, ())
        self.assertFalse(decision.marker_stripped)
        self.assertEqual(decision.raw_model_text, decision.marker_stripped_text)

    def test_the_prompt_tells_the_model_to_leave_the_markers_out(self) -> None:
        body = " ".join(m["content"] for m in build_messages(ITEM))

        self.assertIn("never include [[ or ]]", body)


class ApplicationTest(unittest.TestCase):
    def test_every_defer_reason_leaves_the_span_alone(self) -> None:
        for raw in (
            "no idea",
            reply("EXPAND", "pitchers"),
            reply("DEFER"),
            reply("REPLACE"),
            reply("REPLACE", "the pitcher with the number"),
            reply("REPLACE", "number before and after"),
            reply("KEEP", CONTEXT[WIDER[0] : WIDER[1]]),
        ):
            with self.subTest(raw=raw[:40]):
                decision = validate(raw, ITEM, ALLOWED)

                self.assertEqual(decision.action, "DEFER")
                self.assertEqual(apply(decision, ITEM), SPAN)

    def test_an_accepted_replacement_moves_the_span(self) -> None:
        decision = validate(reply("REPLACE", CONTEXT[WIDER[0] : WIDER[1]]), ITEM, ALLOWED)

        self.assertEqual(apply(decision, ITEM), WIDER)

    def test_no_reply_can_make_a_span_disappear(self) -> None:
        """The module has no delete path, which is the whole point of v3."""

        for raw in (
            reply("DROP"),
            reply("DROP", "pitchers"),
            reply("DEFER"),
            reply("KEEP", "pitchers"),
            reply("REPLACE", CONTEXT[WIDER[0] : WIDER[1]]),
            "garbage",
        ):
            with self.subTest(raw=raw[:40]):
                self.assertIsNotNone(apply(validate(raw, ITEM, ALLOWED), ITEM))

    def test_an_applied_replacement_is_still_the_context_text(self) -> None:
        decision = validate(reply("REPLACE", CONTEXT[WIDER[0] : WIDER[1]]), ITEM, ALLOWED)
        start, end = apply(decision, ITEM)

        self.assertIn(CONTEXT[start:end], CONTEXT)


if __name__ == "__main__":
    unittest.main()


class DeferClassTest(unittest.TestCase):
    """A DEFER total is four different things added together.

    Across four design runs the model chose DEFER zero times out of 133, while
    36 refusals came from replies that failed to parse. Summing those into one
    "abstention rate" would report a parser accident as a calibrated model.
    """

    def test_every_reason_has_exactly_one_class(self) -> None:
        from score.boundary_action_selector import DEFER_CLASSES, DEFER_REASONS

        for reason in DEFER_REASONS:
            with self.subTest(reason=reason):
                owners = [n for n, rs in DEFER_CLASSES.items() if reason in rs]
                self.assertEqual(len(owners), 1, f"{reason} -> {owners}")

    def test_only_a_deliberate_refusal_counts_as_explicit(self) -> None:
        from score.boundary_action_selector import defer_class

        self.assertEqual(defer_class("model_deferred"), "explicit_defer")
        for reason in ("malformed_output", "unknown_action", "not_in_context"):
            with self.subTest(reason=reason):
                self.assertNotEqual(defer_class(reason), "explicit_defer")

    def test_a_broken_keep_contract_is_not_a_grounding_failure(self) -> None:
        """The text was found; it just was not the span KEEP promised."""

        from score.boundary_action_selector import defer_class

        self.assertEqual(defer_class("keep_must_equal_span"), "invalid_action_fallback")

    def test_a_decision_that_was_not_deferred_has_no_class(self) -> None:
        from score.boundary_action_selector import defer_class

        self.assertEqual(defer_class(""), "")

    def test_the_frozen_run_records_classify_without_a_leftover(self) -> None:
        """The split has to work on the traces already on disk, or it is a
        change to the contract rather than a reading of it."""

        import json
        import os

        from score.boundary_action_selector import defer_class

        base = "c:/SCP/outputs/query_span_analysis"
        for version in ("v1", "v2", "v3", "v4"):
            path = f"{base}/boundary_selector_shadow_{version}.jsonl"
            if not os.path.exists(path):
                continue
            with self.subTest(version=version):
                for line in open(path, encoding="utf-8"):
                    record = json.loads(line)
                    if record["defer_reason"]:
                        self.assertTrue(defer_class(record["defer_reason"]))
