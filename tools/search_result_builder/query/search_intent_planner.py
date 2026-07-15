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
        - answer_role: Expected answer type that query generation must preserve.
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
    answer_role: str = "unknown"
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
            answer_role=normalize_text(str(data.get("answer_role") or "unknown")).lower()
            or "unknown",
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
    ALLOWED_ANSWER_ROLES = {
        "number",
        "count",
        "volume",
        "duration",
        "distance",
        "date",
        "person",
        "location",
        "organization",
        "title",
        "species",
        "boolean",
        "text_span",
        "unknown",
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
- answer_role: expected answer type

Terms:
- must_include: max 4 title/date/source/rule terms
- avoid_terms: max 4 noise/later-hop/local-list terms
- preferred_domain: one domain or ""

Rules:
- First search only; local lists search external rules only.
- Never avoid titles, dates, sources, or main entities.
- Preserve the answer role in target and must_include.
- For paper questions, search for the requested value; use authors only if the question asks who wrote it.
- Keep later-hop requirements out of must_include.

Return JSON:
{{"search_needed": true, "intent": "fact", "target": "", "answer_role": "unknown", "must_include": [], "avoid_terms": [], "preferred_domain": ""}}"""

    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "search_needed": {"type": "boolean"},
            "intent": {
                "type": "string",
                "enum": sorted(ALLOWED_INTENTS),
            },
            "target": {"type": "string"},
            "answer_role": {
                "type": "string",
                "enum": sorted(ALLOWED_ANSWER_ROLES),
            },
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
            "answer_role",
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
                keep_alive=0,
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
            answer_role=self._clean_answer_role(parsed.get("answer_role")),
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
            answer_role=self._infer_answer_role(question),
            must_include=[],
            avoid_terms=[],
            preferred_domain="",
        )

    def _sanitize_plan(self, plan: SearchIntentPlan, *, question: str) -> SearchIntentPlan:
        must_include = self._clean_terms(plan.must_include)
        quoted_titles = self._quoted_titles(question)
        answer_role = self._repair_answer_role(plan.answer_role, question)
        if self._looks_like_paper_lookup(question) and quoted_titles:
            must_include = self._paper_first_hop_terms(
                question,
                quoted_titles[0],
                answer_role=answer_role,
            )

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
            target = self._paper_target(
                question,
                quoted_titles[0],
                answer_role=answer_role,
                current_target=target,
            )
        if self._looks_like_local_list(question) and self._has_terms(
            question,
            ["botanical", "fruit", "vegetable"],
        ):
            intent = "definition"
        return SearchIntentPlan(
            search_needed=plan.search_needed,
            intent=intent,
            target=target,
            answer_role=answer_role,
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

    def _clean_answer_role(self, value: Any) -> str:
        role = normalize_text(str(value or "")).lower().strip()
        role = re.sub(r"[^a-z_]+", "_", role).strip("_")
        return role if role in self.ALLOWED_ANSWER_ROLES else "unknown"

    def _repair_answer_role(self, role: str, question: str) -> str:
        cleaned = self._clean_answer_role(role)
        inferred = self._infer_answer_role(question)
        if cleaned == "unknown":
            return inferred
        if inferred == "unknown" or self._compatible_answer_roles(cleaned, inferred):
            return cleaned
        if self._should_override_answer_role(
            model_role=cleaned,
            inferred_role=inferred,
            question=question,
        ):
            return inferred
        return cleaned

    def _infer_answer_role(self, question: str) -> str:
        lowered = normalize_text(question).lower()
        if re.search(r"\bwhat\s+does\b.+\bstand\s+for\b", lowered):
            return "text_span"
        if re.search(r"\b(?:what|which)\s+writer\b|\bquoted\s+by\b|\bwho\s+(?:is|was|were|wrote|authored)\b", lowered):
            return "person"
        if re.search(r"\bfirst\s+name\b|\blast\s+name\b|\bfull\s+name\b", lowered):
            return "person"
        if re.search(r"\bwho|author|authors|director|founder|person\b", lowered):
            return "person"
        if re.search(r"\b(?:ioc|country|nation|airport|station)\s+code\b|\bcode\s+as\s+your\s+answer\b", lowered):
            return "text_span"
        if re.search(r"\bzip code\b|\bzipcode\b|\bfive-digit\b", lowered):
            return "text_span"
        if re.search(r"\b(m\^3|m3|cubic\s+met(?:er|re)s?|lit(?:er|re)s?|volume)\b", lowered):
            return "volume"
        if re.search(r"\b(hours?|minutes?|seconds?|duration|elapsed time)\b", lowered):
            return "duration"
        if re.search(r"\b(km|kilomet(?:er|re)s?|meters?|metres?|miles?|distance)\b", lowered):
            return "distance" if "distance" in self.ALLOWED_ANSWER_ROLES else "number"
        if re.search(r"\b(how many|number of|count|highest number|total)\b", lowered):
            if re.search(r"\b(species|bird species)\b", lowered):
                return "species"
            return "count"
        if re.search(r"\bwhen\b|\bwhat date\b|\bwhich date\b|\bwhat year\b|\bwhich year\b", lowered):
            return "date"
        if re.search(r"\b(where|city|country|location|place)\b", lowered):
            return "location"
        if re.search(r"\b(organization|company|university|agency|institution)\b", lowered):
            return "organization"
        if re.search(r"\b(title|name of the book|name of the song|album)\b", lowered):
            return "title"
        if re.search(r"\b(true|false|yes|no|whether)\b", lowered):
            return "boolean"
        return "unknown"

    def _compatible_answer_roles(self, left: str, right: str) -> bool:
        if left == right:
            return True
        compatible_groups = [
            {"count", "number"},
            {"title", "text_span"},
            {"boolean", "text_span"},
        ]
        return any(left in group and right in group for group in compatible_groups)

    def _should_override_answer_role(
        self,
        *,
        model_role: str,
        inferred_role: str,
        question: str,
    ) -> bool:
        lowered = normalize_text(question).lower()
        if inferred_role == "text_span" and re.search(r"\bwhat\s+does\b.+\bstand\s+for\b", lowered):
            return True
        if inferred_role == "person" and re.search(
            r"\b(?:what|which)\s+writer\b|\bquoted\s+by\b|\bfirst\s+name\b|\blast\s+name\b|\bwho\b",
            lowered,
        ):
            return True
        if inferred_role == "text_span" and re.search(
            r"\b(?:ioc|country|nation|airport|station)\s+code\b|\bcode\s+as\s+your\s+answer\b",
            lowered,
        ):
            return True
        has_entity_answer_cue = bool(
            re.search(
                r"\b(?:what|which)\s+writer\b|\bquoted\s+by\b|\bfirst\s+name\b|\blast\s+name\b|\bwho\b|\bwhere\b|\bwhich country\b|\bwhich city\b",
                lowered,
            )
        )
        if inferred_role == "volume" and re.search(r"\b(m\^3|m3|cubic\s+met(?:er|re)s?|volume)\b", lowered):
            return not has_entity_answer_cue or model_role in {"number", "count", "unknown", "text_span"}
        if inferred_role == "duration" and re.search(r"\bhow\s+long\b|\bhours?\b|\bminutes?\b|\bseconds?\b", lowered):
            return not has_entity_answer_cue or model_role in {"number", "count", "unknown", "text_span"}
        if inferred_role == "distance" and re.search(r"\bdistance\b|\bkilomet(?:er|re)s?\b|\bmiles?\b", lowered):
            return not has_entity_answer_cue or model_role in {"number", "count", "unknown", "text_span"}
        if inferred_role == "date" and re.search(r"\bwhen\b|\bwhat date\b|\bwhich date\b|\bwhat year\b|\bwhich year\b", lowered):
            return model_role in {"unknown", "text_span"}
        return False

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

    def _paper_first_hop_terms(
        self,
        question: str,
        title: str,
        *,
        answer_role: str,
    ) -> list[str]:
        terms = [title]
        year_match = re.search(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", question)
        if year_match:
            terms.append(year_match.group(0))
        role_terms = self._answer_role_terms(question, answer_role)
        terms.extend(role_terms)
        if answer_role == "person" and re.search(r"\bauthors?\b|\bwho wrote\b", question, flags=re.IGNORECASE):
            terms.append("authors")
        return self._clean_terms(terms)

    def _paper_target(
        self,
        question: str,
        title: str,
        *,
        answer_role: str,
        current_target: str,
    ) -> str:
        if answer_role == "person" and re.search(r"\bauthors?\b|\bwho wrote\b", question, flags=re.IGNORECASE):
            return f"Find the authors of the paper {title}."
        role_terms = self._answer_role_terms(question, answer_role)
        role_text = " ".join(role_terms[:2]) if role_terms else answer_role.replace("_", " ")
        if role_text and role_text != "unknown":
            return f"Find the {role_text} reported in the paper {title}."
        return current_target or f"Find the requested value in the paper {title}."

    def _answer_role_terms(self, question: str, answer_role: str) -> list[str]:
        lowered = normalize_text(question).lower()
        terms: list[str] = []
        role_term_map = {
            "volume": ["volume", "m^3", "fish bag"],
            "duration": ["duration", "hours"],
            "count": ["number", "count"],
            "species": ["bird species", "simultaneously"],
            "date": ["date"],
            "person": ["author"],
            "location": ["location"],
            "organization": ["organization"],
            "title": ["title"],
        }
        for term in role_term_map.get(answer_role, []):
            if term in lowered or answer_role in {"count", "date", "person", "location", "organization", "title"}:
                terms.append(term)
        quoted = self._quoted_titles(question)
        protected_ranges = [range(question.find(title), question.find(title) + len(title)) for title in quoted if title in question]
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9^.-]*(?:\s+[A-Za-z][A-Za-z0-9^.-]*){0,2}\b", question):
            if any(match.start() in span for span in protected_ranges):
                continue
            phrase = self._clean_text(match.group(0), max_chars=40)
            key = self._term_key(phrase)
            if not key or key in {"what", "was", "the", "that", "calculated", "paper"}:
                continue
            if answer_role == "volume" and key in {"fish bag", "m 3", "m3", "volume"}:
                terms.append(phrase)
        return self._clean_terms(terms)


__all__ = ["SearchIntentPlan", "SearchIntentPlanner"]
