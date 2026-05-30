from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

QWEN_MODEL = "qwen3:4b"


@dataclass
class QueryCandidate:
    """
    Store one model-generated search query candidate.
    """

    query: str
    purpose: str = ""
    must_include: list[str] = field(default_factory=list)
    expected_answer_type: str = "unknown"
    score_hint: float = 0.0


class ModelQueryCandidateGenerator:
    """
    Ask qwen3:4b to produce structured search query candidates from a question.

    Args:
        - model_name: Ollama model name. This tool only allows qwen3:4b.
        - max_tokens: Maximum completion tokens for query generation.
        - temperature: Sampling temperature.

    Returns:
        - ModelQueryCandidateGenerator: Query candidate generation helper.
    """

    SYSTEM_PROMPT = """You are acting only as a search query candidate generator.
    Return a JSON object only.
    The first output character must be { and the last output character must be }.
    Do not answer the question.
    Do not explain your reasoning."""

    USER_TEMPLATE = """Question:
{question}

Generate {num_candidates} search query candidates.

Rules:
- Each query should be short, precise, and usable in a web search engine.
- Preserve rare entities, dates, source names, titles, acronyms, and constraints.
- Include multi-hop subtasks as separate query candidates when needed.
- Avoid generic-only queries such as "paper article answer".
- Do not include the final answer unless it is explicitly in the question.

Return exactly this JSON shape:
{{
    "queries": [
        {{
            "query": "...",
            "purpose": "what this query is trying to find",
            "must_include": ["important term 1", "important term 2"],
            "expected_answer_type": "person|date|number|title|code|word|place|entity",
            "score_hint": 0.0
        }}
    ]
}}"""

    def __init__(
        self,
        *,
        model_name: str = QWEN_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        if model_name != QWEN_MODEL:
            raise ValueError(f"model_query_candidate only supports {QWEN_MODEL}.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, *, num_candidates: int = 6) -> list[QueryCandidate]:
        """
        Generate query candidates for a question.

        Args:
            - question: Original task question.
            - num_candidates: Number of candidate queries requested from the model.

        Returns:
            - list[QueryCandidate]: Parsed query candidates.
        """
        messages = self.build_messages(question, num_candidates=num_candidates)
        raw_reply = self.invoke_qwen(messages)
        return self.parse_candidates(raw_reply)[:num_candidates]

    def invoke_qwen(self, messages: list[dict[str, str]]) -> str:
        """
        Invoke qwen3:4b through Ollama native chat API and return assistant content.

        Args:
            - messages: Chat messages for query generation.

        Returns:
            - str: Assistant content.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        request = urllib.request.Request(
            self._ollama_chat_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds()) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"qwen3:4b query candidate generation failed: {exc}") from exc

        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        thinking = str(message.get("thinking") or "").strip()
        return content or self._extract_json_text(thinking)

    def build_messages(self, question: str, *, num_candidates: int) -> list[dict[str, str]]:
        """
        Build chat messages for query candidate generation.

        Args:
            - question: Original task question.
            - num_candidates: Number of candidate queries requested from the model.

        Returns:
            - list[dict[str, str]]: Chat messages for qwen3:4b.
        """
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(
                    question=question.strip(),
                    num_candidates=num_candidates,
                ),
            },
        ]

    def parse_candidates(self, raw_reply: str) -> list[QueryCandidate]:
        """
        Parse model output into QueryCandidate objects.

        Args:
            - raw_reply: Raw model response.

        Returns:
            - list[QueryCandidate]: Parsed candidates. Falls back to line parsing.
        """
        parsed = self._parse_json(raw_reply)
        if isinstance(parsed, dict):
            raw_queries = parsed.get("queries", [])
        elif isinstance(parsed, list):
            raw_queries = parsed
        else:
            raw_queries = []

        candidates: list[QueryCandidate] = []
        for item in raw_queries:
            candidate = self._candidate_from_item(item)
            if candidate and self._is_new_query(candidate.query, candidates):
                candidates.append(candidate)

        if candidates:
            return candidates

        return self._fallback_parse_lines(raw_reply)

    def _parse_json(self, text: str) -> Any | None:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _extract_json_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return cleaned[start : end + 1]
        return ""

    def _ollama_chat_url(self) -> str:
        base_url = (
            os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url.rstrip('/')}/api/chat"

    def _timeout_seconds(self) -> int:
        try:
            return int(os.getenv("OLLAMA_TIMEOUT", "180"))
        except ValueError:
            return 180

    def _candidate_from_item(self, item: Any) -> QueryCandidate | None:
        if isinstance(item, str):
            query = item.strip()
            return QueryCandidate(query=query) if query else None
        if not isinstance(item, dict):
            return None

        query = str(item.get("query", "") or "").strip()
        if not query:
            return None
        must_include = item.get("must_include", [])
        if not isinstance(must_include, list):
            must_include = [str(must_include)]

        try:
            score_hint = float(item.get("score_hint", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_hint = 0.0

        return QueryCandidate(
            query=query,
            purpose=str(item.get("purpose", "") or "").strip(),
            must_include=[str(value).strip() for value in must_include if str(value).strip()],
            expected_answer_type=str(item.get("expected_answer_type", "unknown") or "unknown").strip(),
            score_hint=score_hint,
        )

    def _fallback_parse_lines(self, text: str) -> list[QueryCandidate]:
        candidates: list[QueryCandidate] = []
        for pattern in [
            r'"query"\s*:\s*"([^"]+)"',
            r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*[\"“]([^\"”\n]{4,120})[\"”]",
            r"(?:query|search query)\s*[:：]\s*[\"“]?([^\"”\n]{4,120})[\"”]?",
        ]:
            for match in re.finditer(pattern, str(text or ""), flags=re.IGNORECASE):
                query = self._clean_fallback_query(match.group(1))
                if query and self._is_new_query(query, candidates):
                    candidates.append(QueryCandidate(query=query))
        if candidates:
            return candidates

        for line in str(text or "").splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            cleaned = self._clean_fallback_query(cleaned)
            if not cleaned:
                continue
            if cleaned.lower().startswith(("query", "purpose", "must_include")):
                continue
            if self._is_new_query(cleaned, candidates):
                candidates.append(QueryCandidate(query=cleaned))
        return candidates

    def _clean_fallback_query(self, value: str) -> str:
        cleaned = str(value or "").strip().strip('"').strip("'").strip()
        cleaned = re.sub(r"\s+[–-]\s+.*$", "", cleaned).strip()
        if not cleaned or len(cleaned) < 4 or len(cleaned) > 140:
            return ""
        lowered = cleaned.lower()
        blocked_prefixes = (
            "okay",
            "let's",
            "the user",
            "i need",
            "i should",
            "hmm",
            "wait",
            "another",
            "first query",
            "second",
            "third",
        )
        if lowered.startswith(blocked_prefixes):
            return ""
        return cleaned

    def _is_new_query(self, query: str, candidates: list[QueryCandidate]) -> bool:
        normalized = self._normalize_query(query)
        return normalized not in {self._normalize_query(candidate.query) for candidate in candidates}

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "").strip().lower())


def print_candidates(candidates: list[QueryCandidate]) -> None:
    print(json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SLM-based search query candidates.")
    parser.add_argument("--question", default="", help="Question text. If omitted, stdin or interactive input is used.")
    parser.add_argument("--question-file", default="", help="Read question text from a UTF-8 text file.")
    parser.add_argument("--model", default=QWEN_MODEL, choices=[QWEN_MODEL])
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args(argv)
