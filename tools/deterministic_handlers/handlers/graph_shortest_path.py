from __future__ import annotations

import heapq
import re
from typing import Any

from ..base import HandlerInput, HandlerMatch, HandlerResult
from ..contracts import default_outputs, input_field, io_contract, output_field
from .common import (
    attachment_path,
    extract_quoted_or_word_pair,
    normalize_text,
    parse_inline_delimited_rows,
    read_delimited_rows,
)


class GraphShortestPathRouterHandler:
    name = "graph_shortest_path"
    capability_description = (
        "Find a shortest path, route, station count, stop count, hop count, or graph distance "
        "from an edge list in the question, attachment, or evidence."
    )
    supported_attachment_types: set[str] = {".csv", ".tsv", ".txt"}
    routing_terms = {"graph", "shortest", "path", "route", "station", "stations", "stops", "hops", "edges"}
    input_schema = io_contract(
        name,
        [
            input_field("edges", "list[tuple[str,str,float]]", True, "Graph edges.", "question|attachment"),
            input_field("start", "str", True, "Start node.", "question"),
            input_field("end", "str", True, "End node.", "question"),
            input_field("directed", "bool", False, "Whether edges are directed.", "question|attachment"),
            input_field("weighted", "bool", False, "Whether edges include weights.", "question|attachment"),
        ],
        [
            *default_outputs(),
            output_field("path", "list[str]", True, "Shortest path."),
            output_field("total_weight", "float", False, "Total path weight."),
        ],
        supported_attachment_types=supported_attachment_types,
    )
    output_schema = input_schema

    EDGE_RE = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_.-]*)\s*(?:->|-)\s*([A-Za-z][A-Za-z0-9_.-]*)\b"
    )
    WEIGHTED_EDGE_RE = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_.-]*)\s*(?:->|-)\s*([A-Za-z][A-Za-z0-9_.-]*)"
        r"\s*(?:[:=,]\s*|\s+weight\s+|\s+cost\s+|\s+distance\s+)"
        r"([-+]?\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    def match_input(self, handler_input: HandlerInput) -> HandlerMatch:
        inputs = self.build_input(handler_input)
        missing = []
        if not inputs.get("edges"):
            missing.append("edges")
        if not inputs.get("start"):
            missing.append("start_node")
        if not inputs.get("end"):
            missing.append("end_node")
        return HandlerMatch(
            handler_name=self.name,
            matched=not missing,
            confidence=0.94 if not missing else 0.4,
            reason="graph_edges_start_end_readiness",
            missing_inputs=missing,
        )

    def build_input(self, handler_input: HandlerInput) -> dict[str, Any]:
        rows = self._attachment_rows(handler_input) or parse_inline_delimited_rows(handler_input.combined_text())
        row_edges = self._edges_from_rows(rows) if len(rows) > 1 else []
        edges = row_edges or self._edges_from_text(handler_input.combined_text())
        start, end = extract_quoted_or_word_pair(handler_input.question)
        directed = "->" in handler_input.combined_text()
        weighted = any(len(edge) >= 3 for edge in edges)
        answer_mode = self._answer_mode(handler_input.question)
        return {
            "edges": edges,
            "start": start,
            "end": end,
            "directed": directed,
            "weighted": weighted,
            "answer_mode": answer_mode,
            "asks_count": answer_mode in {"edge_count", "node_count"},
        }

    def run(self, inputs: dict[str, Any]) -> HandlerResult:
        missing = []
        edges = list(inputs.get("edges") or [])
        start = normalize_text(inputs.get("start", ""))
        end = normalize_text(inputs.get("end", ""))
        if not edges:
            missing.append("edges")
        if not start:
            missing.append("start_node")
        if not end:
            missing.append("end_node")
        if missing:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=missing,
                structured_result={"edges": edges[:10], "start": start, "end": end},
            )
        path, total_weight = self._weighted_shortest_path(edges, start, end, directed=bool(inputs.get("directed")))
        if not path:
            return HandlerResult.missing(
                handler_name=self.name,
                missing_inputs=["connected_path"],
                structured_result={"edges": edges[:20], "start": start, "end": end},
                next_action_hint="Provide a connected edge list containing both requested nodes.",
            )
        answer = self._format_answer(path, total_weight, str(inputs.get("answer_mode") or "path"))
        task_type = (
            "graph_weighted_shortest_path"
            if inputs.get("weighted")
            else "graph_shortest_path"
        )
        return HandlerResult(
            handler_name=self.name,
            status="ok",
            answer=answer,
            evidence_text=(
                "Deterministic handler evidence:\n"
                f"Handler: {self.name}\n"
                f"Task: {task_type}\n"
                f"Start: {start}\n"
                f"End: {end}\n"
                f"Path: {' -> '.join(path)}\n"
                f"Total weight: {total_weight}\n"
                f"Answer: {answer}\n"
                "Instruction: prefer this exact deterministic result for closed-world graph tasks."
            ),
            structured_result={
                "task_type": task_type,
                "path": path,
                "edge_count": len(edges),
                "directed": bool(inputs.get("directed")),
                "weighted": bool(inputs.get("weighted")),
                "total_weight": total_weight,
                "answer_mode": inputs.get("answer_mode"),
                "answer_is_hop_count": bool(inputs.get("asks_count")),
            },
            confidence=0.96,
        )

    def _attachment_rows(self, handler_input: HandlerInput) -> list[list[str]]:
        path = attachment_path(handler_input.attachment)
        if path and path.suffix.lower() in {".csv", ".tsv"}:
            return read_delimited_rows(path)
        return []

    def _edges_from_rows(self, rows: list[list[str]]) -> list[tuple[str, str, float]]:
        if not rows:
            return []
        data_rows = rows[1:] if len(rows) > 1 and self._looks_like_header(rows[0]) else rows
        edges: list[tuple[str, str, float]] = []
        for row in data_rows:
            if len(row) < 2:
                continue
            left = normalize_text(row[0])
            right = normalize_text(row[1])
            weight = self._coerce_weight(row[2]) if len(row) >= 3 else 1.0
            if left and right:
                edges.append((left, right, weight))
        return edges

    def _looks_like_header(self, row: list[str]) -> bool:
        lowered = [cell.lower() for cell in row]
        return any(cell in {"source", "from", "start", "node1"} for cell in lowered) and any(
            cell in {"target", "to", "end", "node2"} for cell in lowered
        )

    def _edges_from_text(self, text: str) -> list[tuple[str, str, float]]:
        weighted = [
            (left, right, float(weight))
            for left, right, weight in self.WEIGHTED_EDGE_RE.findall(text or "")
        ]
        if weighted:
            return weighted
        return [(left, right, 1.0) for left, right in self.EDGE_RE.findall(text or "")]

    def _weighted_shortest_path(
        self,
        edges: list[tuple[str, str, float]],
        start: str,
        end: str,
        *,
        directed: bool,
    ) -> tuple[list[str], float]:
        adjacency: dict[str, list[tuple[str, float]]] = {}
        canonical: dict[str, str] = {}
        for edge in edges:
            left, right, weight = edge[0], edge[1], float(edge[2] if len(edge) >= 3 else 1.0)
            canonical[left.casefold()] = left
            canonical[right.casefold()] = right
            adjacency.setdefault(left.casefold(), []).append((right.casefold(), weight))
            if not directed:
                adjacency.setdefault(right.casefold(), []).append((left.casefold(), weight))
        start_key = start.casefold()
        end_key = end.casefold()
        queue: list[tuple[float, str, list[str]]] = [(0.0, start_key, [start_key])]
        best_distance = {start_key: 0.0}
        while queue:
            distance, node, path = heapq.heappop(queue)
            if node == end_key:
                return [canonical.get(item, item) for item in path], distance
            if distance > best_distance.get(node, float("inf")):
                continue
            for neighbor, weight in adjacency.get(node, []):
                new_distance = distance + weight
                if new_distance >= best_distance.get(neighbor, float("inf")):
                    continue
                best_distance[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor, [*path, neighbor]))
        return [], 0.0

    def _answer_mode(self, question: str) -> str:
        lowered = str(question or "").lower()
        if re.search(r"\b(exist|exists|reachable|can .* reach|is there a path)\b", lowered):
            return "exists"
        if re.search(r"\b(nodes?|stations?)\b", lowered) and re.search(r"\bhow many|number of|count\b", lowered):
            return "node_count"
        if re.search(r"\b(stops?|hops?|edges?)\b", lowered) and re.search(r"\bhow many|number of|fewest|minimum\b", lowered):
            return "edge_count"
        if re.search(r"\b(distance|cost|weight|length|total)\b", lowered):
            return "total_weight"
        return "path"

    def _format_answer(self, path: list[str], total_weight: float, mode: str) -> str:
        if mode == "exists":
            return "yes"
        if mode == "node_count":
            return str(len(path))
        if mode == "edge_count":
            return str(max(0, len(path) - 1))
        if mode == "total_weight":
            return self._format_number(total_weight)
        return " -> ".join(path)

    def _coerce_weight(self, value: Any) -> float:
        try:
            return float(str(value).replace(",", "").strip())
        except Exception:
            return 1.0

    def _format_number(self, value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"


__all__ = ["GraphShortestPathRouterHandler"]
