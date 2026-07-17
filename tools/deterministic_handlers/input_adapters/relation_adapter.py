from __future__ import annotations

from ..base import HandlerInput
from .base import AdapterResult, parsed_payload, payload_provenance


class RelationInputAdapter:
    handler_names = {"graph_shortest_path"}

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        del handler_name
        payload = parsed_payload(handler_input)
        relations = [
            relation
            for relation in list(payload.get("relations") or [])
            if isinstance(relation, dict)
            and str(relation.get("source") or "").strip()
            and str(relation.get("target") or "").strip()
        ]
        edges = [
            (str(item["source"]), str(item["target"]), 1.0)
            for item in relations
        ]
        if not edges:
            return AdapterResult(
                status="missing_inputs",
                missing_inputs=["edges"],
                input_provenance=payload_provenance(handler_input),
                reason="typed_relations_missing",
            )
        return AdapterResult(
            status="ready",
            inputs={
                "edges": edges,
                "relations": relations,
                "directed": True,
                "weighted": False,
            },
            input_provenance=payload_provenance(handler_input),
            reason="typed_relation_payload",
        )


__all__ = ["RelationInputAdapter"]
