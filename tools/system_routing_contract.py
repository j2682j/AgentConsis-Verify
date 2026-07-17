from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _clean(value: Any) -> str:
    """
    將任意值轉成壓縮空白後的文字。

    Args:
        - value: 要清理的任意值。

    Returns:
        - str: 清理後文字。
    """
    return " ".join(str(value or "").split()).strip()


def _has_word(text: str, words: set[str]) -> list[str]:
    """
    在文字中尋找完整詞命中的關鍵字。

    Args:
        - text: 要搜尋的文字。
        - words: 候選關鍵字集合。

    Returns:
        - list[str]: 命中的關鍵字清單。
    """
    hits: list[str] = []
    for word in sorted(words):
        if re.search(rf"\b{re.escape(word)}\b", text):
            hits.append(word)
    return hits


@dataclass
class SystemRoutingDecision:
    """
    保存系統層工具 routing 判斷結果。

    Args:
        - use_search: 是否需要 search evidence。
        - use_deterministic_solver: 是否需要 deterministic solver。
        - use_python_solver: 是否需要 Python 類解題能力。
        - use_attachment: 是否需要 attachment evidence。
        - use_calculator: 是否需要 calculator。
        - task_type: 目前判定的任務類型。
        - trigger_terms: 觸發 routing 的關鍵字。
        - routing_reasons: routing 判定原因。
        - tool_policy: prefer、optional、avoid 工具政策。

    Returns:
        - SystemRoutingDecision: 單次 routing 判斷結果。
    """

    use_search: bool = False
    use_deterministic_solver: bool = False
    use_python_solver: bool = False
    use_attachment: bool = False
    use_calculator: bool = False
    initial_route: str = "agent_direct"
    search_allowed: bool = False
    has_attachment: bool = False
    attachment_type: str | None = None
    task_type: str = "system_contract"
    trigger_terms: list[str] = field(default_factory=list)
    routing_reasons: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)

    @property
    def needs_routed_tool(self) -> bool:
        """
        判斷此 routing decision 是否需要任一工具。

        Args:
            - 無。

        Returns:
            - bool: 若需要 search、solver 或 calculator 則回傳 True。
        """
        return bool(
            self.use_search
            or self.use_deterministic_solver
            or self.use_python_solver
            or self.use_attachment
            or self.use_calculator
        )

    def to_dict(self) -> dict[str, Any]:
        """
        將 routing decision 轉成可序列化 dict。

        Args:
            - 無。

        Returns:
            - dict[str, Any]: routing metadata。
        """
        return {
            "use_calculator": self.use_calculator,
            "use_search": self.use_search,
            "use_deterministic_solver": self.use_deterministic_solver,
            "use_python_solver": self.use_python_solver,
            "use_attachment": self.use_attachment,
            "calculator_expression": None,
            "initial_route": self.initial_route,
            "search_allowed": self.search_allowed,
            "has_attachment": self.has_attachment,
            "attachment_type": self.attachment_type,
            "task_type": self.task_type,
            "trigger_terms": list(self.trigger_terms),
            "tool_policy": {
                "prefer": list(self.tool_policy.get("prefer", [])),
                "optional": list(self.tool_policy.get("optional", [])),
                "avoid": list(self.tool_policy.get("avoid", [])),
            },
            "routing_reasons": list(self.routing_reasons),
            "routing_source": "system_routing_contract",
        }


class SystemRoutingContract:
    """
    使用規則式關鍵字判斷 Stage1 是否需要提前準備 search、attachment 或 solver evidence。

    Args:
        - 無。

    Returns:
        - SystemRoutingContract: 可根據 question 產生 SystemRoutingDecision 的 routing 規則物件。
    """

    SEARCH_TERMS = {
        "added",
        "who",
        "when",
        "where",
        "which",
        "title",
        "date",
        "website",
        "wikipedia",
        "paper",
        "book",
        "video",
        "company",
        "institution",
        "github",
        "issue",
        "label",
        "repository",
        "repo",
        "commit",
        "release",
        "oldest",
        "closed",
        "code",
        "syntax",
        "output",
        "character",
        "language",
        "documentation",
        "library",
    }
    WEAK_SEARCH_TERMS = {
        "who",
        "when",
        "where",
        "which",
        "date",
    }
    SEARCH_PHRASES = {
        "according to github",
        "web site",
        "webpage",
        "journal article",
        "research paper",
        "pull request",
        "source code",
        "programming language",
        "youtube",
        "movie",
        "film",
        "author",
        "published",
        "publication",
        "university",
        "organization",
        "organisation",
        "agency",
        "official",
        "located",
    }
    CLOSED_WORLD_TERMS = {
        "above",
        "attached",
        "chess",
        "grid",
        "logic",
        "logical",
        "piston",
        "riddle",
        "spreadsheet",
    }
    CLOSED_WORLD_PHRASES = {
        "algebraic notation",
        "attached spreadsheet",
        "black's turn",
        "closed world",
        "fictional language",
        "game show",
        "logically equivalent",
        "no other plots",
        "provided in the image",
        "provided evidence",
        "reverse string",
        "the above",
        "this sentence",
        "truth table",
    }
    PUZZLE_TERMS = {
        "choose",
        "ejected",
        "equivalent",
        "odds",
        "puzzle",
        "ramp",
        "translate",
    }
    PUZZLE_PHRASES = {
        "maximize your odds",
        "which ball should you choose",
        "pick one of",
        "ping-pong",
        "direct object",
        "verb first",
    }
    PYTHON_TERMS = {
        "calculate",
        "compute",
        "sort",
        "sorting",
        "order",
        "rank",
        "string",
        "reverse",
        "convert",
        "conversion",
        "table",
        "spreadsheet",
        "excel",
        "csv",
        "count",
        "sum",
        "average",
        "mean",
        "median",
    }
    PYTHON_PHRASES = {
        "unit conversion",
        "unit convert",
        "how many",
        "how much",
        "data frame",
        "dataframe",
        "regular expression",
        "regex",
        "character count",
        "word count",
        "sort alphabetically",
    }
    CODE_DETERMINISTIC_TERMS = {
        "code",
        "syntax",
        "output",
        "character",
        "program",
        "parse",
        "pdb",
    }
    CODE_DETERMINISTIC_PHRASES = {
        "following code",
        "correct the following code",
        "what exact character",
        "what exact charcter",
        "what exact text",
        "needs to be added",
        "to output",
        "parse the pdb file",
        "using the biopython",
    }

    def route(
        self,
        *,
        question: str,
        stage: str,
        benchmark: str = "",
        has_attachment: bool = False,
        attachment_type: str | None = None,
    ) -> SystemRoutingDecision:
        """
        根據問題、階段與附檔狀態產生工具 routing decision。

        Args:
            - question: 使用者問題。
            - stage: 目前流程階段。
            - benchmark: benchmark 名稱，例如 BFCL。
            - has_attachment: 題目是否包含附檔。
            - attachment_type: 附檔類型。

        Returns:
            - SystemRoutingDecision: 工具 routing 判斷結果。
        """
        normalized = _clean(question).lower()
        decision = SystemRoutingDecision()
        stage_key = _clean(stage).lower()
        benchmark_key = _clean(benchmark).upper()
        attachment_type_clean = _clean(attachment_type).lower() or None
        decision.has_attachment = bool(has_attachment)
        decision.attachment_type = attachment_type_clean

        if has_attachment:
            decision.use_attachment = True
            decision.initial_route = "attachment_first"
            decision.search_allowed = False
            decision.task_type = "attachment_evidence"
            decision.trigger_terms.append(attachment_type_clean or "attachment")
            decision.routing_reasons.append("attachment is present; first-round attachment evidence is required")

        if benchmark_key == "BFCL":
            decision.task_type = "function_calling"
            decision.initial_route = "agent_direct"
            decision.search_allowed = False
            decision.tool_policy = {
                "prefer": [],
                "optional": [],
                "avoid": ["search", "deterministic_solver", "python_solver"],
            }
            decision.routing_reasons.append(
                "BFCL is a structured function-calling task; external tools are not enabled by system contract"
            )
            return decision

        python_hits = _has_word(normalized, self.PYTHON_TERMS)
        python_hits.extend(phrase for phrase in sorted(self.PYTHON_PHRASES) if phrase in normalized)
        if attachment_type_clean in {"xlsx", "xls", "csv", "tsv"}:
            python_hits.append(f"{attachment_type_clean}_table")

        closed_world_hits = self._closed_world_hits(normalized)
        puzzle_hits = self._puzzle_hits(normalized)
        code_hits = self._code_deterministic_hits(normalized)
        search_hits = self._search_hits(normalized)
        strong_search_hits = [
            hit
            for hit in search_hits
            if hit not in self.WEAK_SEARCH_TERMS
        ]

        if has_attachment:
            if python_hits or closed_world_hits or puzzle_hits or code_hits:
                decision.use_deterministic_solver = bool(python_hits or closed_world_hits or puzzle_hits or code_hits)
                decision.use_python_solver = bool(python_hits)
                decision.task_type = "attachment_deterministic_solver"
                decision.trigger_terms.extend((python_hits + closed_world_hits + puzzle_hits + code_hits)[:8])
                decision.routing_reasons.append(
                    "attachment metadata takes priority; deterministic/code signals are handled after attachment parsing"
                )
            elif strong_search_hits:
                decision.routing_reasons.append(
                    "factual lookup signals are deferred until attachment evidence is available: "
                    + ", ".join(strong_search_hits[:6])
                )
            self._write_tool_policy(decision)
            if stage_key == "stage1_round0" and decision.needs_routed_tool:
                decision.routing_reasons.append("stage1 round0 system contract enables early evidence gathering")
            return decision

        if closed_world_hits or puzzle_hits:
            decision.initial_route = "deterministic_first"
            decision.search_allowed = False
            decision.use_deterministic_solver = True
            decision.trigger_terms.extend((closed_world_hits + puzzle_hits)[:8])
            decision.routing_reasons.append(
                "question appears closed-world or puzzle-like; search is deferred"
            )
            if python_hits:
                decision.use_python_solver = True
                decision.trigger_terms.extend(python_hits[:8])
                decision.routing_reasons.append(
                    "closed-world task also contains deterministic computation/data-processing signals: "
                    + ", ".join(python_hits[:6])
                )
            decision.task_type = (
                "closed_world_attachment"
                if decision.use_attachment
                else "closed_world_puzzle"
            )
            self._write_tool_policy(decision)
            return decision

        if (python_hits or code_hits) and not strong_search_hits:
            decision.use_deterministic_solver = True
            decision.use_python_solver = bool(python_hits)
            decision.initial_route = "deterministic_first"
            decision.search_allowed = False
            decision.trigger_terms.extend((python_hits + code_hits)[:8])
            decision.routing_reasons.append(
                "question contains deterministic/code signals before factual search: "
                + ", ".join((python_hits + code_hits)[:6])
            )
            decision.task_type = (
                "attachment_deterministic_solver"
                if decision.use_attachment
                else "deterministic_solver"
            )
            self._write_tool_policy(decision)
            if stage_key == "stage1_round0" and decision.needs_routed_tool:
                decision.routing_reasons.append("stage1 round0 system contract enables early evidence gathering")
            return decision

        if code_hits and strong_search_hits:
            decision.use_deterministic_solver = True
            decision.initial_route = "deterministic_first"
            decision.search_allowed = False
            decision.trigger_terms.extend((code_hits + strong_search_hits)[:8])
            decision.routing_reasons.append(
                "question mixes code/deterministic and lookup signals; deterministic handling is tried before search"
            )
            decision.task_type = "deterministic_solver"
            self._write_tool_policy(decision)
            if stage_key == "stage1_round0" and decision.needs_routed_tool:
                decision.routing_reasons.append("stage1 round0 system contract enables early evidence gathering")
            return decision

        if strong_search_hits:
            decision.use_search = True
            decision.initial_route = "search_first"
            decision.search_allowed = True
            decision.trigger_terms.extend(strong_search_hits[:8])
            decision.routing_reasons.append(
                "question contains factual lookup signals: " + ", ".join(strong_search_hits[:6])
            )

        if python_hits:
            decision.use_deterministic_solver = True
            decision.use_python_solver = True
            decision.trigger_terms.extend(python_hits[:8])
            decision.routing_reasons.append(
                "question contains deterministic computation/data-processing signals: "
                + ", ".join(python_hits[:6])
            )

        if decision.use_search and decision.use_python_solver:
            decision.task_type = "hybrid_search_and_solver"
            decision.initial_route = "search_first"
            decision.search_allowed = True
        elif decision.use_search:
            decision.task_type = "factual_search"
            decision.initial_route = "search_first"
            decision.search_allowed = True
        elif decision.use_python_solver:
            decision.task_type = "deterministic_solver"
            decision.initial_route = "deterministic_first"
            decision.search_allowed = False
        elif decision.use_attachment:
            decision.task_type = "attachment_evidence"
            decision.initial_route = "attachment_first"
            decision.search_allowed = False

        self._write_tool_policy(decision)

        if stage_key == "stage1_round0" and decision.needs_routed_tool:
            decision.routing_reasons.append("stage1 round0 system contract enables early evidence gathering")

        return decision

    def _search_hits(self, normalized: str) -> list[str]:
        hits = _has_word(normalized, self.SEARCH_TERMS)
        hits.extend(phrase for phrase in sorted(self.SEARCH_PHRASES) if phrase in normalized)
        if re.search(r"\b(18|19|20)\d{2}\b", normalized):
            hits.append("year")
        return hits

    def _closed_world_hits(self, normalized: str) -> list[str]:
        hits = _has_word(normalized, self.CLOSED_WORLD_TERMS)
        hits.extend(phrase for phrase in sorted(self.CLOSED_WORLD_PHRASES) if phrase in normalized)
        return hits

    def _puzzle_hits(self, normalized: str) -> list[str]:
        hits = _has_word(normalized, self.PUZZLE_TERMS)
        hits.extend(phrase for phrase in sorted(self.PUZZLE_PHRASES) if phrase in normalized)
        return hits

    def _code_deterministic_hits(self, normalized: str) -> list[str]:
        hits = _has_word(normalized, self.CODE_DETERMINISTIC_TERMS)
        hits.extend(phrase for phrase in sorted(self.CODE_DETERMINISTIC_PHRASES) if phrase in normalized)
        if "`" in normalized:
            hits.append("inline_code")
        return hits

    def _write_tool_policy(self, decision: SystemRoutingDecision) -> None:
        prefer = []
        if decision.use_search:
            prefer.append("search")
        if decision.use_deterministic_solver:
            prefer.append("deterministic_solver")
        if decision.use_python_solver:
            prefer.append("python_solver")
        if decision.use_attachment:
            prefer.append("attachment_reader")
        decision.tool_policy = {"prefer": prefer, "optional": [], "avoid": []}
