from tools.deterministic_handlers.registry import default_deterministic_registry
from tools.deterministic_handlers.router import DeterministicHandlerRouter
from tools.deterministic_handlers.schema import HandlerIOContract


def test_registered_handlers_expose_contracts():
    registry = default_deterministic_registry()
    handlers = registry.list_handlers()
    assert handlers

    for handler in handlers:
        assert isinstance(handler.input_schema, HandlerIOContract), handler.name
        assert isinstance(handler.output_schema, HandlerIOContract), handler.name
        assert handler.input_schema.required_input_names(), handler.name
        assert "answer" in handler.output_schema.required_output_names(), handler.name


def test_match_diagnostics_include_required_inputs():
    router = DeterministicHandlerRouter(threshold=0.99, similarity_fn=lambda _left, _right: 0.0)
    result = router.run(question="How many rows are in this table?")

    assert result.status == "no_match"
    matches = result.structured_result["matches"]
    table_match = next(item for item in matches if item["handler_name"] == "table_exact_operations")
    assert "table_rows" in table_match["missing_inputs"]
    assert "rows" in table_match["required_inputs"]
    assert table_match["schema_version"] == "deterministic-handler-v1"


def test_success_result_contains_contract_metadata():
    router = DeterministicHandlerRouter(threshold=0.1, similarity_fn=lambda _left, _right: 0.0)
    result = router.run(question="What is the difference between 9 and 4?")

    assert result.ok
    assert result.answer == "5"
    assert result.input_summary
    assert result.structured_result["input_summary"] == result.input_summary
    assert result.structured_result["task_type"]
    assert "operation" in result.structured_result
    assert "calculation_trace" in result.structured_result
    assert result.structured_result["input_contract"]["schema_version"] == "deterministic-handler-v1"
    assert "answer" in result.structured_result["output_contract"]["required_outputs"]
