"""Choose an action for a query span, and refuse anything that cannot be checked.

The candidate lattice reaches 37 of 38 gold boundaries, so the remaining problem
is selection. Putting the lattice in the prompt would be the obvious way to do
that and the wrong one: a case carries 99 candidates at the median and 261 at
the worst, almost all of them from `expansion`, and they are not 99 distinct
ideas -- they are combinations of two edges over one short context. A 4b model
asked to rank them is being asked to read the same sentence a hundred times.

So the model sees the context once, with the current span marked, and answers
what to do with it. The lattice moves to the other side of the call, where it
becomes the set of boundaries an answer is *allowed* to name. That keeps the
prompt short and keeps the guarantee: a span this module returns is a boundary
generation already produced, not a boundary the model invented.

Everything the model returns is checked against the frozen offsets, and anything
that fails a check becomes DEFER. Not a guess, not a partial repair -- the
original span survives untouched. A wrong repair costs a task; a DEFER costs
nothing that was not already lost.

This module moves boundaries and does nothing else. It has no way to remove a
span, deliberately: the first version could, and deleted 47 of 90 spans that
were already correct. Deciding whether a span belongs in the query needs the
other spans, their roles and what the query already covers, none of which is
visible from one marked phrase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

#: DROP is deliberately absent. It was in the first contract and it cost 47 of
#: 90 already-correct spans, because "is this span the answer?" and "is this
#: span worth searching for?" are different questions and a boundary module is
#: only competent at neither. `Eliud Kipchoge` is not the answer and is exactly
#: what you search for. Whether a span enters the query at all belongs to query
#: admission, which sees the role and the whole span set; this module decides
#: where a span ends and nothing else.
ACTIONS = ("KEEP", "REPLACE", "DEFER")

#: Why an answer was not usable. Kept separate from the action so a shadow run
#: can tell a model that refuses from a model that answers ungroundably -- the
#: first needs a better prompt, the second needs a tighter contract.
DEFER_REASONS = (
    "model_deferred",
    "malformed_output",
    "unknown_action",
    "empty_selection",
    "action_not_in_contract",
    "not_in_context",
    "not_in_candidate_set",
    "ambiguous_occurrence",
    "keep_must_equal_span",
)

#: Decoding is part of the contract, not a caller's choice. The same prompt at a
#: different temperature is a different system, and a holdout that cannot say
#: which one it tested is not a holdout. `seed` is fixed so a disputed result can
#: be reproduced rather than argued about; the value itself is arbitrary.
#: `enable_thinking` was False here, chosen without evidence and frozen that
#: way. It was also silently ignored on the OpenAI-compatible path, so the runs
#: that produced the good numbers had reasoning switched on while the manifest
#: said otherwise. Measured directly, on one path, with everything else held:
#: fragmented recovery 6/37 off against 15/37 on.
#:
#: Kept on, with the cost stated rather than buried: it also raises complete-span
#: mutation, 5 of 40 against 8 of 40, which is the metric that decides whether
#: any of this ships. Off is not the safer choice, it is the choice that recovers
#: too little to be worth the risk either way.
DECODE_SETTINGS = {
    "temperature": 0.0,
    "seed": 20260819,
    "enable_thinking": True,
}


#: A DEFER total is not one number. A model that says "I cannot tell" is
#: calibrated; a model whose reply failed to parse is not, and reading the
#: second as the first would credit an accident as a safety property -- which is
#: exactly what happened in the unconstrained run, where 36 DEFERs came entirely
#: from prose replies and the model chose DEFER zero times out of 133.
#:
#: Two placements are worth stating. `keep_must_equal_span` is a contract
#: violation rather than a grounding failure: the text was found, it simply was
#: not the span KEEP promised. `empty_selection` is the same shape -- an action
#: was named and its payload withheld.
DEFER_CLASSES = {
    "explicit_defer": ("model_deferred",),
    "parse_failure_fallback": ("malformed_output",),
    "invalid_action_fallback": (
        "unknown_action",
        "action_not_in_contract",
        "empty_selection",
        "keep_must_equal_span",
    ),
    "grounding_failure_fallback": (
        "not_in_context",
        "not_in_candidate_set",
        "ambiguous_occurrence",
    ),
}


def defer_class(defer_reason: str) -> str:
    """Which kind of refusal this was. Empty for decisions that were not DEFER."""

    for name, reasons in DEFER_CLASSES.items():
        if defer_reason in reasons:
            return name
    return ""


SPAN_OPEN = "[["
SPAN_CLOSE = "]]"

#: Constrained decoding, because 28 of 133 replies in the unconstrained run were
#: prose -- and 21 of those were `complete` spans, where the model explains that
#: the span is already fine instead of saying so in JSON. That is not a boundary
#: error but it lands as one, since an unparseable reply becomes DEFER, and the
#: bias runs one way: the spans most likely to be explained are the ones that
#: needed no change.
#:
#: The alternative was a more forgiving parser, which would mean reading an
#: action out of prose -- guessing at the model's answer, exactly what the DEFER
#: contract exists to prevent. Constraining the decoder removes the failure
#: instead of interpreting it. Under constraint S044 returned `Legume Wikipedia
#: page`, its gold: the prose was hiding real decisions, not covering for absent
#: ones.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "selected_text": {"type": "string"},
    },
    "required": ["action", "selected_text"],
}


def response_format() -> dict:
    """The `response_format` argument that pins replies to `RESPONSE_SCHEMA`."""

    return {
        "type": "json_schema",
        "json_schema": {"name": "boundary_decision", "schema": RESPONSE_SCHEMA},
    }


def call_model(client, model: str, messages: list[dict[str, str]]) -> str:
    """Ask the model, on the path where `DECODE_SETTINGS` actually applies.

    The OpenAI-compatible endpoint accepts `seed` and `enable_thinking` and
    honours neither against Ollama: `seed` never reaches `options.seed`, and
    `enable_thinking` is applied only when the provider is vLLM. A holdout run
    under settings that are silently discarded is a run whose manifest describes
    a different system, so the same span came back KEEP once and REPLACE five
    times with the seed supposedly fixed.

    Ollama's native endpoint takes both, and on it the decision is stable. The
    JSON still arrives pretty-printed sometimes and compact other times, which
    is why determinism is judged on the parsed decision rather than the bytes.
    """

    if getattr(client, "provider", "") == "ollama":
        result = client.ollama_native_chat(
            model=model,
            messages=messages,
            temperature=DECODE_SETTINGS["temperature"],
            think=DECODE_SETTINGS["enable_thinking"],
            json_format=RESPONSE_SCHEMA,
            seed=DECODE_SETTINGS["seed"],
        )
    else:
        result = client.chat(
            model=model,
            messages=messages,
            response_format=response_format(),
            **DECODE_SETTINGS,
        )
    return result.content or result.reasoning

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)

#: Two earlier versions are worth keeping in view, because both failures were in
#: the question being asked rather than in the model's answer.
#:
#: The first asked whether the span was "part of what is being asked for". The
#: model answered that correctly and dropped 47 of 90 already-correct spans, 28
#: of them named entities -- `Eliud Kipchoge` is not what the question asks for.
#:
#: The second kept DROP but redefined it around retrieval usefulness. That still
#: asks one module two questions, and the second one needs what this module does
#: not have: the other spans, the roles, and what the query already covers. So
#: the prompt now only ever asks where the unit ends. A span that should not be
#: searched for is still returned, with its boundary fixed, for query admission
#: to exclude on its own evidence.
SYSTEM_PROMPT = """A span has been pulled out of a question. Your only job is to
decide whether the span covers the whole phrase it belongs to.

You are not deciding whether the span is useful, whether it is the answer, or
whether it should be searched for. Something else decides that. Decide only
where the phrase starts and ends.

Reply with one JSON object and nothing else:

{"action": "KEEP|REPLACE|DEFER", "selected_text": "..."}

KEEP     the span is already the whole phrase. Repeat it exactly.
REPLACE  the span is cut off mid-phrase, or has picked up words that belong to
         a neighbouring phrase. selected_text is the whole phrase, copied from
         the question.
DEFER    you cannot tell where the phrase ends. selected_text is "".

The span is shown between [[ and ]] so you can see where it sits. Those brackets
are not part of the text: never include [[ or ]] in selected_text.

selected_text must be copied verbatim from the question. Do not paraphrase, do
not fix spelling, do not add or remove words. If the phrase you want cannot be
copied exactly, answer DEFER.

Example question: Who are the [[pitchers]] with the number before and after
Taisho Tamai's number?
Example reply: {"action": "REPLACE", "selected_text": "pitchers with the number before and after Taisho Tamai's number"}

Example question: How many albums were published by [[Mercedes Sosa]] between 2000 and 2009?
Example reply: {"action": "KEEP", "selected_text": "Mercedes Sosa"}"""


@dataclass(frozen=True)
class SelectorInput:
    """One span to judge, with the little context the decision needs."""

    annotation_id: str
    context: str
    span: tuple[int, int]
    question_role: str = ""
    answer_target: str = ""

    @property
    def span_text(self) -> str:
        return self.context[self.span[0] : self.span[1]]

    def marked_context(self) -> str:
        start, end = self.span
        return (
            self.context[:start]
            + SPAN_OPEN
            + self.context[start:end]
            + SPAN_CLOSE
            + self.context[end:]
        )


@dataclass(frozen=True)
class SelectorDecision:
    """What to do, and -- when the answer was refused -- why.

    `start`/`end` are offsets into the same context the input carried, so an
    applied decision cannot drift from the text it was checked against.
    """

    action: str
    start: int | None = None
    end: int | None = None
    defer_reason: str = ""
    normalisations: tuple[str, ...] = ()
    raw: str = ""
    #: The model's `selected_text` exactly as it arrived, kept beside the
    #: version that was grounded. Marker removal is the one place this module
    #: alters what the model said, and a repair that cannot be inspected
    #: afterwards is a repair nobody can audit.
    raw_model_text: str = ""
    marker_stripped_text: str = ""
    marker_stripped: bool = False

    @property
    def rewrites(self) -> bool:
        return self.action == "REPLACE"

    def text(self, context: str) -> str:
        if self.start is None or self.end is None:
            return ""
        return context[self.start : self.end]


def build_messages(item: SelectorInput) -> list[dict[str, str]]:
    """The prompt. Four short fields, and no candidate list."""

    lines = []
    if item.question_role:
        lines.append(f"What the question asks for: {item.question_role}")
    if item.answer_target:
        lines.append(f"The answer should be: {item.answer_target}")
    lines.append(f"Question text: {item.marked_context()}")
    lines.append(f"Marked span: {item.span_text}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse(raw: str) -> tuple[str, str] | None:
    """Pull one action and one selection out of whatever the model emitted."""

    body = _THINK.sub("", raw or "").strip()
    body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.MULTILINE).strip()
    for match in [body] + [m.group(0) for m in _OBJECT.finditer(body)]:
        try:
            payload = json.loads(match)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        action = str(payload.get("action", "")).strip().upper()
        selected = payload.get("selected_text")
        return action, "" if selected is None else str(selected)
    return None


def _undo_markers(selected: str, span_text: str) -> tuple[str, bool]:
    """Undo this module's own injection, and only that.

    15 of the first run's 17 grounding failures were the model returning
    `"[[Moon]]"` -- it copied verbatim from the text it was shown, and the text
    it was shown was the one this module had wrapped in brackets. Undoing an
    injection of our own is not the same as repairing the model's answer.

    Which is why the exact injected string is what gets undone, not brackets in
    general. A blanket `.strip("[]")` would also rewrite a question that
    genuinely contains brackets, and then this module would be silently editing
    the model's answer -- the thing the DEFER contract exists to prevent.
    """

    injected = SPAN_OPEN + span_text + SPAN_CLOSE
    if injected not in selected:
        return selected, False
    return selected.replace(injected, span_text).strip(), True


def _occurrences(context: str, needle: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    at = context.casefold().find(needle.casefold())
    while at >= 0:
        found.append((at, at + len(needle)))
        at = context.casefold().find(needle.casefold(), at + 1)
    return found


def validate(
    raw: str,
    item: SelectorInput,
    allowed: set[tuple[int, int]],
) -> SelectorDecision:
    """Turn a model reply into an action, or into a DEFER that says why.

    The order matters. Grounding is checked before lattice membership, so a
    hallucinated span is reported as ungrounded rather than as an unlucky
    boundary -- those are different failures and only one of them is fixable by
    widening generation.
    """

    parsed = parse(raw)
    if parsed is None:
        return SelectorDecision("DEFER", defer_reason="malformed_output", raw=raw)
    action, selected = parsed
    if action == "DROP":
        # DROP was removed from the contract, not merely from the prompt. A
        # model that still emits it is answering a question this module no
        # longer asks, so the span is left alone rather than deleted.
        return SelectorDecision(
            "DEFER", defer_reason="action_not_in_contract",
            raw=raw, raw_model_text=selected,
        )
    if action not in ACTIONS:
        return SelectorDecision(
            "DEFER", defer_reason="unknown_action", raw=raw, raw_model_text=selected
        )
    if action == "DEFER":
        return SelectorDecision(
            "DEFER", defer_reason="model_deferred", raw=raw, raw_model_text=selected
        )

    raw_selection = selected.strip()
    selected, stripped = _undo_markers(raw_selection, item.span_text)
    notes = ("marker_stripped",) if stripped else ()
    trace = {
        "raw": raw,
        "raw_model_text": raw_selection,
        "marker_stripped_text": selected,
        "marker_stripped": stripped,
        "normalisations": notes,
    }
    if not selected:
        return SelectorDecision("DEFER", defer_reason="empty_selection", **trace)

    hits = _occurrences(item.context, selected)
    if not hits:
        return SelectorDecision("DEFER", defer_reason="not_in_context", **trace)

    permitted = [h for h in hits if h in allowed]
    if not permitted:
        return SelectorDecision("DEFER", defer_reason="not_in_candidate_set", **trace)
    if len(permitted) > 1:
        return SelectorDecision("DEFER", defer_reason="ambiguous_occurrence", **trace)

    start, end = permitted[0]
    if action == "KEEP" and (start, end) != item.span:
        return SelectorDecision("DEFER", defer_reason="keep_must_equal_span", **trace)
    if action == "REPLACE" and (start, end) == item.span:
        # Same text under a different label. The effect is KEEP, and calling it
        # REPLACE would count a no-op as a rewrite in the mutation rate.
        trace["normalisations"] = notes + ("replace_to_keep",)
        return SelectorDecision("KEEP", start, end, **trace)
    return SelectorDecision(action, start, end, **trace)


def apply(decision: SelectorDecision, item: SelectorInput) -> tuple[int, int]:
    """The span after the decision. Every span survives; only its edges move.

    There is no return path that deletes a span. KEEP, DEFER and every
    validation failure land on the original offsets, and REPLACE lands on a
    boundary the frozen lattice already contained. Whether the span belongs in
    the query at all is decided elsewhere, on evidence this module cannot see.
    """

    if decision.action == "REPLACE" and decision.start is not None:
        return (decision.start, decision.end)
    return item.span


__all__ = [
    "ACTIONS",
    "DECODE_SETTINGS",
    "DEFER_CLASSES",
    "RESPONSE_SCHEMA",
    "DEFER_REASONS",
    "SYSTEM_PROMPT",
    "SelectorDecision",
    "SelectorInput",
    "apply",
    "build_messages",
    "call_model",
    "defer_class",
    "parse",
    "response_format",
    "validate",
]
