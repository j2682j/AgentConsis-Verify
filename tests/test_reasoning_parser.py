from parsers.reasoning_parser import (
    ReasoningParseQuality,
    extract_reasoning_steps,
    format_reasoning_steps,
    prepare_reasoning_for_verifier,
)


def test_extracts_standard_step_markers() -> None:
    text = "step 1. Find the distance.\nstep 2. Compute the speed.\nstep 3. Divide."

    assert extract_reasoning_steps(text) == [
        (1, "Find the distance."),
        (2, "Compute the speed."),
        (3, "Divide."),
    ]


def test_extracts_inline_markdown_step_markers() -> None:
    text = (
        "step 1. Understand the problem. "
        "### 🔹 Step 2: Gather the perigee distance. "
        "### 🔹 Step 3: Compute the running speed. "
        "### 🔹 Step 4: Divide distance by speed."
    )

    steps = extract_reasoning_steps(text)

    assert steps == [
        (1, "Understand the problem."),
        (2, "Gather the perigee distance."),
        (3, "Compute the running speed."),
        (4, "Divide distance by speed."),
    ]


def test_extracts_numbered_steps_only_from_line_boundaries() -> None:
    text = "1. Find the distance.\n2. Compute the speed.\n3. Divide."

    assert extract_reasoning_steps(text) == [
        (1, "Find the distance."),
        (2, "Compute the speed."),
        (3, "Divide."),
    ]


def test_does_not_split_decimal_numbers_as_numbered_steps() -> None:
    text = (
        "The marathon time is 1.994444 hours. "
        "The distance is 42.195 km. "
        "No explicit reasoning step appears here."
    )

    assert extract_reasoning_steps(text) == []


def test_prefers_explicit_step_markers_over_numbered_content() -> None:
    text = (
        "step 1. Use 1.994444 hours and 42.195 km to compute speed. "
        "Step 2: Divide the perigee distance by speed."
    )

    assert extract_reasoning_steps(text) == [
        (1, "Use 1.994444 hours and 42.195 km to compute speed."),
        (2, "Divide the perigee distance by speed."),
    ]


def test_format_reasoning_steps_normalizes_to_versa_friendly_lines() -> None:
    text = "### Step 1: A. ### Step 2: B."

    assert format_reasoning_steps(text) == "step 1. A.\nstep 2. B."


def test_strips_final_answer_tail_from_last_step_only() -> None:
    text = (
        "step 1. Identify the value. "
        "step 2. Round the result. --- ### Final Answer: 17"
    )

    assert extract_reasoning_steps(text) == [
        (1, "Identify the value."),
        (2, "Round the result."),
    ]


def test_strips_final_answer_with_underscore_from_last_step() -> None:
    text = "step 1. Round hours. FINAL_ANSWER: 17"

    assert extract_reasoning_steps(text) == [(1, "Round hours.")]


def test_strips_markdown_and_emoji_before_final_answer_marker() -> None:
    text = "step 1. Round hours. --- ### \u2705 Final Answer: 17"

    assert extract_reasoning_steps(text) == [(1, "Round hours.")]


def test_keeps_answer_formatting_action_without_final_answer_marker() -> None:
    text = "step 1. Format the computed value according to the question."

    assert extract_reasoning_steps(text) == [
        (1, "Format the computed value according to the question.")
    ]


def test_structured_steps_strip_final_answer_before_versa() -> None:
    result = prepare_reasoning_for_verifier(
        "",
        final_answer="17",
        structured_steps=[
            "step 1. Read the value.",
            "step 2. Round the value. Final Answer: 17",
        ],
    )

    assert result.steps == [(1, "Read the value."), (2, "Round the value.")]
    assert result.quality_status == ReasoningParseQuality.REPAIRED
    assert result.versa_eligible
    assert result.diagnostics.final_answer_removed


def test_atomizes_clear_multiple_sentences() -> None:
    result = prepare_reasoning_for_verifier(
        "step 1. Read E1. Compare it with E2. Compute the difference."
    )

    assert result.steps == [
        (1, "Read E1."),
        (2, "Compare it with E2."),
        (3, "Compute the difference."),
    ]
    assert result.quality_status == ReasoningParseQuality.REPAIRED


def test_does_not_split_nominal_and_or_numeric_range() -> None:
    result = prepare_reasoning_for_verifier(
        "step 1. Compare France and Germany between 2010 and 2020."
    )

    assert result.steps == [
        (1, "Compare France and Germany between 2010 and 2020.")
    ]
    assert result.quality_status == ReasoningParseQuality.VALID


def test_only_final_answer_is_unreliable_but_answer_is_preserved() -> None:
    result = prepare_reasoning_for_verifier("Final Answer: Tokyo")

    assert result.steps == []
    assert result.extracted_final_answer == "Tokyo"
    assert result.quality_status == ReasoningParseQuality.UNRELIABLE
    assert not result.versa_eligible
