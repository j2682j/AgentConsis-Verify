from parsers.reasoning_step_atomizer import ReasoningStepAtomizer


def test_splits_clear_imperative_action_list() -> None:
    result = ReasoningStepAtomizer().atomize(
        [(1, "Compare the values, calculate the difference, and conclude the result.")]
    )

    assert result.steps == [
        (1, "Compare the values"),
        (2, "calculate the difference"),
        (3, "conclude the result."),
    ]
    assert result.atomized_step_indices == [1]


def test_keeps_single_calculation_with_and() -> None:
    result = ReasoningStepAtomizer().atomize([(1, "Add A and B.")])

    assert result.steps == [(1, "Add A and B.")]
    assert not result.compound_step_indices
