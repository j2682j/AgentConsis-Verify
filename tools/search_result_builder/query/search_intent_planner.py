from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.llm_client import LLMClient
from core.model_registry import resolve_model_id
from utils.network_utils import normalize_text


@dataclass(frozen=True)
class SearchIntentPlan:
    """
    Store the minimal first-hop search intent used before query generation.

    Args:
        - search_needed: Whether web search should be used.
        - intent: Coarse first-hop search type.
        - target: Short description of what the first search should find.
        - must_include: Exact terms that should appear in generated queries.
        - avoid_terms: Noisy or later-hop terms that should not appear in queries.
        - preferred_domain: Optional authoritative domain for site: queries.

    Returns:
        - SearchIntentPlan: Compact intent plan for query generation.
    """

    search_needed: bool
    intent: str
    target: str
    must_include: list[str]
    avoid_terms: list[str]
    preferred_domain: str = ""
    state: str = "pending"
    completed_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SearchIntentPlan":
        data = dict(value or {})
        return cls(
            search_needed=bool(data.get("search_needed", True)),
            intent=normalize_text(str(data.get("intent") or "fact")).lower() or "fact",
            target=normalize_text(str(data.get("target") or "")),
            must_include=[
                normalize_text(str(item))
                for item in list(data.get("must_include") or [])
                if normalize_text(str(item))
            ],
            avoid_terms=[
                normalize_text(str(item))
                for item in list(data.get("avoid_terms") or [])
                if normalize_text(str(item))
            ],
            preferred_domain=normalize_text(str(data.get("preferred_domain") or "")),
            state=normalize_text(str(data.get("state") or "pending")) or "pending",
            completed_terms=[
                normalize_text(str(item))
                for item in list(data.get("completed_terms") or [])
                if normalize_text(str(item))
            ],
            missing_terms=[
                normalize_text(str(item))
                for item in list(data.get("missing_terms") or [])
                if normalize_text(str(item))
            ],
        )

    def replace(self, **changes: Any) -> "SearchIntentPlan":
        return dataclasses.replace(self, **changes)


class SearchIntentPlanner:
    """
    Ask a small language model for only the first-hop search intent.

    Args:
        - model_name: Model used for compact JSON intent planning.
        - llm_client: Provider-neutral LLM client.
        - max_terms: Maximum must_include and avoid_terms kept.

    Returns:
        - SearchIntentPlanner: Lightweight planner used before query generation.
    """

    ALLOWED_INTENTS = {
        "official_page",
        "paper",
        "definition",
        "fact",
        "media",
        "no_search",
    }
    LOCAL_LIST_NOISE = {
        "grocery",
        "list",
        "mom",
        "food",
        "foods",
        "item",
        "items",
    }
    KNOWN_SINGLE_TOKEN_DOMAINS = {
        "wikipedia",
        "youtube",
        "github",
        "imdb",
        "archive",
    }

    SYSTEM_PROMPT = """You plan the first web search.
Do not answer the question.
Return JSON only."""

    USER_TEMPLATE = """Question:
{question}

Choose:
- search_needed: true or false
- intent: official_page | paper | definition | fact | media | no_search
- target: what the first search should find

Terms:
- must_include: max 4 title/date/source/rule terms
- avoid_terms: max 4 noise/later-hop/local-list terms
- preferred_domain: one domain or ""

Rules:
- First search only; local lists search external rules only.
- Never avoid titles, dates, sources, or main entities.
- Keep later-hop requirements out of must_include.

Return JSON:
{{"search_needed": true, "intent": "fact", "target": "", "must_include": [], "avoid_terms": [], "preferred_domain": ""}}"""

    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "search_needed": {"type": "boolean"},
            "intent": {
                "type": "string",
                "enum": sorted(ALLOWED_INTENTS),
            },
            "target": {"type": "string"},
            "must_include": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "avoid_terms": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "preferred_domain": {"type": "string"},
        },
        "required": [
            "search_needed",
            "intent",
            "target",
            "must_include",
            "avoid_terms",
            "preferred_domain",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        model_name: str | None = None,
        llm_client: LLMClient | None = None,
        max_terms: int = 4,
    ) -> None:
        resolved_model = model_name or os.getenv(
            "TOOL_PLANNER_MODEL",
            os.getenv("QUERY_GENERATOR_MODEL", "qwen3:4b"),
        )
        self.model_name = resolve_model_id(resolved_model)
        self.llm_client = llm_client or LLMClient()
        self.max_terms = max(1, max_terms)
        self.last_diagnostics: dict[str, Any] = {}

    def plan(self, question: str) -> SearchIntentPlan:
        """
        Build a compact first-hop search intent plan.

        Args:
            - question: Original user task.

        Returns:
            - SearchIntentPlan: Parsed or fallback intent plan.
        """
        text = normalize_text(question)
        if not text:
            return self._fallback_plan("")

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_TEMPLATE.format(question=text)},
        ]
        try:
            raw_reply = self._invoke_model(messages)
            plan = self._parse_plan(raw_reply, question=text)
            self.last_diagnostics = {
                "model": self.model_name,
                "provider": self.llm_client.provider,
                "fallback_used": False,
                "plan": plan.to_dict(),
            }
            return plan
        except Exception as exc:
            plan = self._fallback_plan(text)
            self.last_diagnostics = {
                "model": self.model_name,
                "provider": self.llm_client.provider,
                "fallback_used": True,
                "error": f"{type(exc).__name__}: {exc}",
                "plan": plan.to_dict(),
            }
            return plan

    def _invoke_model(self, messages: list[dict[str, str]]) -> str:
        if self.llm_client.provider == "ollama":
            result = self.llm_client.ollama_native_chat(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=384,
                think=False,
                json_format=self.JSON_SCHEMA,
            )
        else:
            result = self.llm_client.chat(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=384,
                enable_thinking=False,
            )
        return result.content

    def _parse_plan(self, raw_reply: str, *, question: str) -> SearchIntentPlan:
        parsed = self._parse_json_object(raw_reply)
        if not isinstance(parsed, dict):
            raise ValueError("intent planner did not return a JSON object")

        intent = normalize_text(str(parsed.get("intent") or "fact")).lower()
        if intent not in self.ALLOWED_INTENTS:
            intent = "fact"
        search_needed = bool(parsed.get("search_needed"))
        if intent == "no_search":
            search_needed = False

        plan = SearchIntentPlan(
            search_needed=search_needed,
            intent=intent,
            target=self._clean_text(parsed.get("target"), max_chars=180),
            must_include=self._clean_terms(parsed.get("must_include")),
            avoid_terms=self._clean_terms(parsed.get("avoid_terms")),
            preferred_domain=self._clean_domain(parsed.get("preferred_domain")),
        )
        return self._sanitize_plan(plan, question=question)

    def _parse_json_object(self, raw_reply: str) -> Any:
        cleaned = str(raw_reply or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise

    def _fallback_plan(self, question: str) -> SearchIntentPlan:
        return SearchIntentPlan(
            search_needed=bool(normalize_text(question)),
            intent="fact",
            target=normalize_text(question)[:180],
            must_include=[],
            avoid_terms=[],
            preferred_domain="",
        )

    def _sanitize_plan(self, plan: SearchIntentPlan, *, question: str) -> SearchIntentPlan:
        must_include = self._clean_terms(plan.must_include)
        quoted_titles = self._quoted_titles(question)
        if self._looks_like_paper_lookup(question) and quoted_titles:
            must_include = self._paper_first_hop_terms(question, quoted_titles[0])

        if self._looks_like_local_list(question):
            must_include = [
                term for term in must_include if not self._has_local_list_noise(term)
            ]
            must_include = self._add_local_list_rule_terms(question, must_include)

        preferred_domain = plan.preferred_domain
        if preferred_domain and not self._domain_supported_by_question(
            preferred_domain,
            question,
        ):
            preferred_domain = ""

        avoid_terms: list[str] = []
        for term in self._clean_terms(plan.avoid_terms):
            if any(self._terms_overlap(term, must) for must in must_include):
                continue
            if preferred_domain and self._terms_overlap(term, preferred_domain):
                continue
            avoid_terms.append(term)
        intent = plan.intent
        target = plan.target
        if self._looks_like_paper_lookup(question) and quoted_titles:
            intent = "paper"
            target = f"Find the authors of the paper {quoted_titles[0]}."
        if self._looks_like_local_list(question) and self._has_terms(
            question,
            ["botanical", "fruit", "vegetable"],
        ):
            intent = "definition"
        return SearchIntentPlan(
            search_needed=plan.search_needed,
            intent=intent,
            target=target,
            must_include=must_include,
            avoid_terms=avoid_terms,
            preferred_domain=preferred_domain,
        )

    def _clean_terms(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            term = self._clean_text(item, max_chars=90)
            key = self._term_key(term)
            if not term or not key or key in seen:
                continue
            seen.add(key)
            result.append(term)
            if len(result) >= self.max_terms:
                break
        return result

    def _clean_text(self, value: Any, *, max_chars: int) -> str:
        cleaned = normalize_text(str(value or "")).strip().strip("\"'")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:max_chars].strip()

    def _clean_domain(self, value: Any) -> str:
        domain = normalize_text(str(value or "")).strip().lower()
        domain = re.sub(r"^https?://", "", domain)
        domain = domain.split("/")[0].strip()
        if not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", domain):
            return ""
        return domain

    def _term_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _terms_overlap(self, left: str, right: str) -> bool:
        left_tokens = set(self._term_key(left).split())
        right_tokens = set(self._term_key(right).split())
        if not left_tokens or not right_tokens:
            return False
        return left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)

    def _looks_like_local_list(self, question: str) -> bool:
        lowered = normalize_text(question).lower()
        return bool(
            re.search(r"\b(list|table|options?|items?)\b", lowered)
            and ("," in lowered or "\n" in lowered)
        )

    def _has_local_list_noise(self, term: str) -> bool:
        tokens = set(self._term_key(term).split())
        return bool(tokens & self.LOCAL_LIST_NOISE)

    def _add_local_list_rule_terms(self, question: str, terms: list[str]) -> list[str]:
        result = list(terms)
        lowered = normalize_text(question).lower()
        for term in ("botanical", "fruit", "vegetable"):
            if term in lowered and not any(self._terms_overlap(term, item) for item in result):
                result.append(term)
        return result[: self.max_terms]

    def _has_terms(self, question: str, terms: list[str]) -> bool:
        lowered = normalize_text(question).lower()
        return all(term in lowered for term in terms)

    def _domain_supported_by_question(self, domain: str, question: str) -> bool:
        domain_key = self._term_key(domain)
        question_key = self._term_key(question)
        if not domain_key or not question_key:
            return False
        if domain_key in question_key:
            return True
        stem = re.sub(r"^www\s+", "", domain_key)
        stem = re.sub(r"\s+(com|org|net|edu|gov|io|co|uk)$", "", stem)
        tokens = [
            token
            for token in stem.split()
            if token not in {"www", "com", "org", "net", "edu", "gov", "io", "co", "uk"}
        ]
        if not tokens:
            return False
        if len(tokens) == 1:
            return tokens[0] in self.KNOWN_SINGLE_TOKEN_DOMAINS and tokens[0] in question_key
        return all(token in question_key for token in tokens)

    def _looks_like_paper_lookup(self, question: str) -> bool:
        lowered = normalize_text(question).lower()
        return bool(
            re.search(r"\b(paper|article|journal|proceedings|publication)\b", lowered)
        )

    def _quoted_titles(self, question: str) -> list[str]:
        return [
            self._clean_text(match, max_chars=120)
            for match in re.findall(r"['\"]([^'\"]{5,120})['\"]", question)
        ]

    def _paper_first_hop_terms(self, question: str, title: str) -> list[str]:
        terms = [title]
        year_match = re.search(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", question)
        if year_match:
            terms.append(year_match.group(0))
        terms.append("authors")
        return self._clean_terms(terms)


__all__ = ["SearchIntentPlan", "SearchIntentPlanner"]
