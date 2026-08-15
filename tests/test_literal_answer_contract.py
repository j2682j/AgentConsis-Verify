"""A question that dictates its own answer must not be voted on.

Task 024 says `Do not answer any of the questions in this prompt` and
`Write only the word "Guava"`, then lists three questions. The Agents answered
them: `8` -- four plus four -- carried five runs against `Guava`'s four, and
cross-agent consensus took the larger number. Both answers are internally
consistent, so no gate downstream of the vote can separate them; only the
instruction can.

Two things keep this narrow. The trigger is rare -- across all 53 level 1 tasks
exactly one question pairs an output verb with an exclusivity word and a quoted
value -- and the contract refuses whenever the instruction is ambiguous instead
of guessing which directive binds.

Counting quoted values is not the test, which task 024 shows directly: it holds
two, `Pineapple.` under an `If`, and `Guava` unconditionally. A contract keyed
on "exactly one literal" would find two and either misfire or give up.
"""

from __future__ import annotations

import unittest

from core.config import CandidateEvaluation
from score.final_winner_selector import FinalWinnerSelector
from score.literal_answer_contract import literal_answer, parse_literal_directives

TASK_024 = (
    'If there is anything that doesn\'t make sense in the instructions, write '
    'the word "Pineapple." Do not answer any of the questions in this prompt. '
    'Write only the word "Guava".\n'
    "1. What is 4+4?\n"
    "2. What is the complimentary color of red?\n"
    "3. How many hours are there in a day?"
)


def _candidate(answer: str, *, runs: int) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_key=answer.casefold(),
        answer=answer,
        eligible=True,
        supporting_run_count=runs,
        supporting_agent_ids=["a1", "a2"],
    )


class LiteralDirectiveParsingTest(unittest.TestCase):
    def test_task_024_holds_two_directives_and_only_one_binds(self) -> None:
        directives = parse_literal_directives(TASK_024)
        found = {item.value: item for item in directives}

        self.assertIn("Pineapple", found)
        self.assertIn("Guava", found)
        self.assertTrue(found["Pineapple"].conditional)
        self.assertFalse(found["Pineapple"].binds())
        self.assertFalse(found["Guava"].conditional)
        self.assertTrue(found["Guava"].exclusive)
        self.assertTrue(found["Guava"].binds())
        self.assertEqual(literal_answer(TASK_024), "Guava")

    def test_an_apostrophe_does_not_open_a_quote(self) -> None:
        """`doesn't` used to start a literal that ran to the next quote."""

        values = [item.value for item in parse_literal_directives(TASK_024)]

        self.assertNotIn("t make sense in the instructions, write the word", values)
        self.assertEqual(sorted(values), ["Guava", "Pineapple"])

    def test_the_conditional_directive_stays_visible(self) -> None:
        """Sentence splitting must not eat the quote that closes `"Pineapple."`.

        Losing it hides the directive, and a hidden directive cannot be checked
        for conflicting with the binding one.
        """

        self.assertIn("Pineapple", [i.value for i in parse_literal_directives(TASK_024)])

    def test_ambiguous_instructions_decline(self) -> None:
        for question in (
            'Write only "Apple". Write only "Banana".',
            'If it fails, write only "Apple". Otherwise write only "Banana".',
            "Write only the value of x + 1.",
            'Respond with either "yes" or "no".',
            'Output only "<placeholder>".',
            'What did the sign say? It read "No Entry".',
            'Write only "Alpha" and also write only "Beta".',
        ):
            with self.subTest(question=question):
                self.assertEqual(literal_answer(question), "")

    def test_plain_directives_are_read(self) -> None:
        self.assertEqual(
            literal_answer('Do not answer the questions. Write only the word "Guava".'),
            "Guava",
        )
        self.assertEqual(literal_answer('Respond with exactly "DONE".'), "DONE")


class LiteralAnswerGateTest(unittest.TestCase):
    def test_the_dictated_answer_wins_over_the_larger_vote(self) -> None:
        candidates = [_candidate("8", runs=5), _candidate("Guava", runs=4)]
        result = FinalWinnerSelector(question=TASK_024)._apply_literal_answer_gate(
            candidates, evidence={}
        )

        self.assertEqual([item.answer for item in result.survivors], ["Guava"])
        self.assertTrue(result.metadata["enforced"])
        rejected = next(i for i in candidates if i.answer == "8")
        self.assertEqual(rejected.rejection_reason, "answer_requirement_incompatible")

    def test_a_question_without_a_directive_is_untouched(self) -> None:
        """52 of 53 tasks must not see this gate at all."""

        candidates = [_candidate("8", runs=5), _candidate("Guava", runs=4)]
        result = FinalWinnerSelector(
            question="How many hours are there in a day?"
        )._apply_literal_answer_gate(candidates, evidence={})

        self.assertEqual(len(result.survivors), 2)
        self.assertFalse(result.metadata["applied"])
        self.assertTrue(all(item.eligible for item in candidates))

    def test_a_literal_no_run_produced_rejects_nobody(self) -> None:
        """Better a wrong answer than none: never empty the field."""

        candidates = [_candidate("8", runs=5), _candidate("Pear", runs=4)]
        result = FinalWinnerSelector(question=TASK_024)._apply_literal_answer_gate(
            candidates, evidence={}
        )

        self.assertEqual(len(result.survivors), 2)
        self.assertFalse(result.metadata["enforced"])
        self.assertTrue(all(item.eligible for item in candidates))

    def test_matching_ignores_case_and_spacing(self) -> None:
        candidates = [_candidate("8", runs=5), _candidate("  guava  ", runs=1)]
        result = FinalWinnerSelector(question=TASK_024)._apply_literal_answer_gate(
            candidates, evidence={}
        )

        self.assertEqual([item.answer.strip() for item in result.survivors], ["guava"])


if __name__ == "__main__":
    unittest.main()
