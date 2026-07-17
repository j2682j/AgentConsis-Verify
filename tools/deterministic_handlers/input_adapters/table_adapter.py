from __future__ import annotations

from typing import Any

from ..base import HandlerInput
from .base import AdapterResult, parsed_payload, payload_provenance


class TableInputAdapter:
    handler_names = {"table_exact_operations", "table_aggregation", "list_operations"}

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        payload = parsed_payload(handler_input)
        if handler_name == "list_operations":
            items: list[str] = []
            for block in list(payload.get("lists") or []):
                if isinstance(block, dict):
                    items.extend(str(item) for item in block.get("items") or [] if str(item).strip())
            return AdapterResult(
                status="ready" if items else "missing_inputs",
                inputs={"list_items": items} if items else {},
                missing_inputs=[] if items else ["list_items"],
                input_provenance=payload_provenance(handler_input),
                reason="typed_list_payload" if items else "typed_list_payload_missing",
            )

        tables = [table for table in list(payload.get("tables") or []) if isinstance(table, dict)]
        rows: list[list[str]] = []
        table_name = ""
        for table in tables:
            columns = [str(value) for value in table.get("columns") or []]
            data_rows = [
                [str(value) for value in row]
                for row in table.get("rows") or []
                if isinstance(row, list)
            ]
            if columns and data_rows:
                rows = [columns, *data_rows]
                table_name = str(table.get("name") or "")
                break
        if not rows:
            return AdapterResult(
                status="missing_inputs",
                missing_inputs=["rows"],
                input_provenance=payload_provenance(handler_input),
                reason="typed_table_rows_missing",
            )
        return AdapterResult(
            status="ready",
            inputs={"rows": rows, "table_name": table_name},
            input_provenance=payload_provenance(handler_input),
            reason="typed_table_payload",
        )


__all__ = ["TableInputAdapter"]
