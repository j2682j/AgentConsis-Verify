from __future__ import annotations

"""Detect candidate answers that merely echo text already given in the question.

A candidate that is a long verbatim fragment of the question (for example a
quoted paper title, or a slice of a reversed-text puzzle) will always be
"found" inside retrieved evidence, because search engines and fetched pages
echo the query. Such matches are tautological and must not be promoted to
evidence support. Short answers (numbers, yes/no, single words) are exempt:
legitimately repeating one question token is common and harmless.
"""

import re

from utils.network_utils import normalize_text

_QUOTED_SPAN_RE = re.compile(r"\"([^\"]{3,200})\"|“([^”]{3,200})”")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _echo_key(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", normalize_text(value).casefold()).strip()


def is_question_echo(answer: str, question: str, *, min_tokens: int = 3) -> bool:
    """Return True when the answer is a long verbatim fragment of the question."""

    answer_key = _echo_key(answer)
    question_key = _echo_key(question)
    if not answer_key or not question_key:
        return False
    token_count = len(answer_key.split())
    for match in _QUOTED_SPAN_RE.finditer(str(question or "")):
        quoted = match.group(1) or match.group(2) or ""
        if token_count >= 2 and answer_key == _echo_key(quoted):
            return True
    if token_count < max(1, min_tokens):
        return False
    return f" {answer_key} " in f" {question_key} "


__all__ = ["is_question_echo"]
