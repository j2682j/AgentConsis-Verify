from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

from ..schemas import DeterministicSolverResult
from .common import clean_text, lower_text


class GraphHandler:
    """Solve closed-world graph traversal and shortest-path tasks."""

    def solve(self, question: str, **_: Any) -> DeterministicSolverResult:
        lowered = lower_text(question)
        if not any(
            term in lowered
            for term in ("graph", "shortest path", "shortest route", "stations", "stops", "edges")
        ):
            return DeterministicSolverResult.miss("graph")

        edges, start, end, directed = self._parse_graph(question)
        if not edges or not start or not end:
            return DeterministicSolverResult.miss("graph", "graph edges or endpoints are missing")
        path = self._shortest_path(edges, start, end, directed=directed)
        if not path:
            return DeterministicSolverResult.miss("graph", f"no path from {start} to {end}")

        asks_count = any(term in lowered for term in ("how many", "number of", "fewest"))
        if asks_count or any(term in lowered for term in ("stations", "stops", "hops")):
            answer = str(len(path) - 1)
            task_type = "graph_hop_count"
        else:
            answer = " -> ".join(path)
            task_type = "graph_shortest_path"
        return DeterministicSolverResult(
            used_deterministic_solver=True,
            task_type=task_type,
            answer=answer,
            answer_text=answer,
            confidence=0.95,
            evidence={"path": path, "edges": edges, "directed": directed},
        )

    def _parse_graph(self, question: str) -> tuple[list[list[str]], str, str, bool]:
        payload = self._json_payload(question)
        if payload:
            raw_edges = payload.get("edges") or []
            edges = [
                [clean_text(edge[0]), clean_text(edge[1])]
                for edge in raw_edges
                if isinstance(edge, (list, tuple)) and len(edge) >= 2
            ]
            return (
                edges,
                clean_text(payload.get("start")),
                clean_text(payload.get("end")),
                bool(payload.get("directed", False)),
            )

        edges = [
            [left, right]
            for left, right in re.findall(
                r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:->|-)\s*([A-Za-z][A-Za-z0-9_]*)\b",
                question,
            )
        ]
        endpoint = re.search(
            r"\bfrom\s+([A-Za-z][A-Za-z0-9_]*)\s+to\s+([A-Za-z][A-Za-z0-9_]*)\b",
            question,
            flags=re.IGNORECASE,
        )
        start, end = endpoint.groups() if endpoint else ("", "")
        return edges, start, end, "->" in question

    def _shortest_path(
        self,
        edges: list[list[str]],
        start: str,
        end: str,
        *,
        directed: bool,
    ) -> list[str]:
        adjacency: dict[str, list[str]] = {}
        for left, right in edges:
            adjacency.setdefault(left, []).append(right)
            if not directed:
                adjacency.setdefault(right, []).append(left)
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path
            for neighbor in adjacency.get(node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append([*path, neighbor])
        return []

    def _json_payload(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


__all__ = ["GraphHandler"]
