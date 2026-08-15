"""Read an instruction that names its own answer.

Task 024 asks nothing that needs solving:

    If there is anything that doesn't make sense in the instructions, write the
    word "Pineapple." Do not answer any of the questions in this prompt. Write
    only the word "Guava".
    1. What is 4+4?  2. What is the complimentary color of red?  ...

The answer is stated in the question. The Agents solved `4+4` instead: `8` took
five runs, `Guava` four, and consensus counted the wrong one higher. Nothing
downstream can recover that, because both answers are internally consistent --
one just answers the question the prompt forbids.

Extracting the literal is deterministic, so nothing here calls a model. What
makes it safe is how narrow the trigger is: over all 53 level 1 tasks exactly
one question carries an output verb, an exclusivity word and a quoted value
together, so on the other 52 this module returns nothing at all.

The two literals in task 024 are why "exactly one quoted value" is not the test.
`Pineapple.` is conditional -- it applies only *if* something does not make
sense -- while `Guava` is unconditional and exclusive. A contract keyed on
counting quotes would find two and either misfire or give up. It has to read
which directive actually binds.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from utils.network_utils import normalize_text

# `write`, `respond`, `reply`, `output`, `return`, `answer with`.
_OUTPUT_VERB = re.compile(
    r"\b(?:writ(?:e|ing)|respond|reply|output|return|answer\s+with)\b", re.IGNORECASE
)
_CONDITIONAL = re.compile(
    r"\b(?:if|unless|when|whenever|in\s+case|should\s+there|otherwise|else)\b",
    re.IGNORECASE,
)
_EXCLUSIVE = re.compile(r"\b(?:only|exactly|nothing\s+else|instead\s+of)\b", re.IGNORECASE)
# Straight single quotes only count when they are not a word's apostrophe.
# Without that guard the `'` in `doesn't` opens a quote that runs to the next
# one, and task 024 yields a directive reading
# `t make sense in the instructions, write the word`.
_QUOTED = re.compile(
    r"[\"“]([^\"“”]{1,60})[\"”]"
    r"|[‘]([^‘’]{1,60})[’]"
    r"|(?<![A-Za-z])'([^']{1,60})'(?![A-Za-z])"
)
# A literal the pipeline must not treat as an answer: an expression, a
# placeholder, or a list of alternatives.
_NOT_LITERAL = re.compile(r"[<>{}\[\]=+*/\\]|\b(?:or|either|each|every)\b|\d\s*[-+*/]\s*\d", re.IGNORECASE)
# `"Pineapple."` closes its sentence inside the quotes, so the boundary has to
# keep the closing quote on the left. Splitting on the match would eat it and
# leave the sentence with an unterminated quote, which hides the directive
# entirely -- and hiding a directive defeats the check that no second
# unconditional one exists.
_SENTENCE_END = re.compile(r"[.!?][\"”'’]?\s+(?=[A-Z(])")


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        parts.append(text[start : match.end()])
        start = match.end()
    parts.append(text[start:])
    return [part for part in parts if part.strip()]


@dataclass(frozen=True)
class LiteralDirective:
    """One "write X" instruction and the conditions attached to it."""

    value: str
    conditional: bool
    exclusive: bool
    sentence: str

    def binds(self) -> bool:
        return self.exclusive and not self.conditional


def parse_literal_directives(question: str) -> list[LiteralDirective]:
    """Every quoted value that an output verb in the same sentence asks for."""

    text = str(question or "")
    directives: list[LiteralDirective] = []
    for sentence in _sentences(text):
        verb = _OUTPUT_VERB.search(sentence)
        if not verb:
            continue
        for match in _QUOTED.finditer(sentence):
            captured = next((group for group in match.groups() if group), "")
            value = normalize_text(captured).strip().strip(".,;:")
            if not value or _NOT_LITERAL.search(value):
                continue
            # Only text before the value can qualify the instruction; a
            # condition stated afterwards belongs to the next clause.
            head = sentence[: match.start()]
            directives.append(
                LiteralDirective(
                    value=value,
                    conditional=bool(_CONDITIONAL.search(head)),
                    exclusive=bool(_EXCLUSIVE.search(head)),
                    sentence=sentence.strip(),
                )
            )
    return directives


def literal_answer(question: str) -> str:
    """The answer the question dictates, or "" when it dictates none.

    Four conditions, all required. Exactly one directive is unconditional and
    exclusive; every other directive is conditional; no two unconditional
    directives disagree; and the value is a plain literal rather than an
    expression or a list. Anything else returns "" and leaves the normal
    pipeline alone -- a question holding two competing `write only` instructions
    is ambiguous, and taking the last one would be a guess.
    """

    directives = parse_literal_directives(question)
    if not directives:
        return ""

    binding = [item for item in directives if item.binds()]
    if len(binding) != 1:
        return ""

    unconditional = {
        item.value.casefold() for item in directives if not item.conditional
    }
    if len(unconditional) > 1:
        return ""

    others = [item for item in directives if item is not binding[0]]
    if any(not item.conditional for item in others):
        return ""
    return binding[0].value


__all__ = ["LiteralDirective", "literal_answer", "parse_literal_directives"]
