from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List

import requests
from dotenv import load_dotenv

from .base import Tool, ToolParameter

try:
    from ddgs import DDGS  # type: ignore
except Exception:
    DDGS = None  # type: ignore

try:
    from tavily import TavilyClient  # type: ignore
except Exception:
    TavilyClient = None  # type: ignore

try:
    from serpapi import GoogleSearch  # type: ignore
except Exception:
    GoogleSearch = None  # type: ignore

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_MAX_RESULTS = 5
SUPPORTED_BACKENDS = {
    "hybrid",
    "advanced",
    "tavily",
    "serpapi",
    "duckduckgo",
    "searxng",
    "perplexity",
}


def _normalized_result(
    *,
    title: str,
    url: str,
    content: str,
) -> Dict[str, str]:
    return {
        "title": str(title or url or "").strip(),
        "url": str(url or "").strip(),
        "content": str(content or "").strip(),
    }


def _structured_payload(
    results: Iterable[Dict[str, Any]],
    *,
    backend: str,
    answer: str | None = None,
    notices: Iterable[str] | None = None,
) -> Dict[str, Any]:
    return {
        "results": list(results),
        "backend": backend,
        "answer": answer,
        "notices": list(notices or []),
    }


class SearchTool(Tool):
    """
    呼叫搜尋後端並回傳 normalized raw search results。

    Args:
        - backend: 預設搜尋後端。
        - tavily_key: Tavily API key。
        - serpapi_key: SerpApi API key。
        - perplexity_key: Perplexity API key。

    Returns:
        - SearchTool: 搜尋 API adapter。
    """

    def __init__(
        self,
        backend: str | None = None,
        tavily_key: str | None = None,
        serpapi_key: str | None = None,
        perplexity_key: str | None = None,
    ) -> None:
        super().__init__(
            name="search",
            description="Web search backend adapter returning normalized raw search results.",
        )
        self.backend = (backend or os.getenv("SEARCH_BACKEND") or "hybrid").lower()
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.perplexity_key = perplexity_key or os.getenv("PERPLEXITY_API_KEY")
        self.searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")

        self.available_backends: list[str] = []
        self.tavily_client = None
        self._setup_backends()

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        """
        執行搜尋並回傳 structured payload。

        Args:
            - parameters: 包含 input/query、backend、max_results 的參數。

        Returns:
            - dict[str, Any]: results/backend/answer/notices。
        """
        query = str(parameters.get("input") or parameters.get("query") or "").strip()
        if not query:
            return _structured_payload([], backend=self.backend, notices=["empty_query"])

        backend = str(parameters.get("backend", self.backend) or "hybrid").lower()
        backend = backend if backend in SUPPORTED_BACKENDS else "hybrid"
        max_results = self._positive_int(parameters.get("max_results"), DEFAULT_MAX_RESULTS)
        loop_count = self._positive_int(parameters.get("loop_count"), 0)

        return self._structured_search(
            query=query,
            backend=backend,
            max_results=max_results,
            loop_count=loop_count,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Search query.",
                required=True,
            ),
        ]

    def _setup_backends(self) -> None:
        if self.tavily_key and TavilyClient is not None:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
            except Exception as exc:
                logger.warning("Tavily initialization failed: %s", exc)

        if self.serpapi_key and GoogleSearch is not None:
            self.available_backends.append("serpapi")

        if self.searxng_url:
            self.available_backends.append("searxng")

        if DDGS is not None:
            self.available_backends.append("duckduckgo")

        if self.perplexity_key:
            self.available_backends.append("perplexity")

        if self.backend not in SUPPORTED_BACKENDS:
            self.backend = "hybrid"

    def _structured_search(
        self,
        *,
        query: str,
        backend: str,
        max_results: int,
        loop_count: int,
    ) -> Dict[str, Any]:
        target_backend = "advanced" if backend == "hybrid" else backend

        if target_backend == "tavily":
            return self._search_tavily(query=query, max_results=max_results)
        if target_backend == "serpapi":
            return self._search_serpapi(query=query, max_results=max_results)
        if target_backend == "duckduckgo":
            return self._search_duckduckgo(query=query, max_results=max_results)
        if target_backend == "searxng":
            return self._search_searxng(query=query, max_results=max_results)
        if target_backend == "perplexity":
            return self._search_perplexity(
                query=query,
                max_results=max_results,
                loop_count=loop_count,
            )
        if target_backend == "advanced":
            return self._search_advanced(
                query=query,
                max_results=max_results,
                loop_count=loop_count,
            )
        raise ValueError(f"Unsupported search backend: {backend}")

    def _search_tavily(self, *, query: str, max_results: int) -> Dict[str, Any]:
        if not self.tavily_client:
            raise RuntimeError("TAVILY_API_KEY is not configured; Tavily search is unavailable.")

        response = self.tavily_client.search(  # type: ignore[call-arg]
            query=query,
            max_results=max_results,
        )
        results = [
            _normalized_result(
                title=item.get("title") or item.get("url", ""),
                url=item.get("url", ""),
                content=item.get("content") or "",
            )
            for item in response.get("results", [])[:max_results]
        ]
        return _structured_payload(results, backend="tavily", answer=response.get("answer"))

    def _search_serpapi(self, *, query: str, max_results: int) -> Dict[str, Any]:
        if not self.serpapi_key or GoogleSearch is None:
            raise RuntimeError("SerpApi search is unavailable.")

        response = GoogleSearch(
            {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "gl": "us",
                "hl": "en",
                "num": max_results,
            }
        ).get_dict()
        answer_box = response.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet")
        results = [
            _normalized_result(
                title=item.get("title") or item.get("link", ""),
                url=item.get("link", ""),
                content=item.get("snippet") or "",
            )
            for item in response.get("organic_results", [])[:max_results]
        ]
        return _structured_payload(results, backend="serpapi", answer=answer)

    def _search_duckduckgo(self, *, query: str, max_results: int) -> Dict[str, Any]:
        if DDGS is None:
            raise RuntimeError("DuckDuckGo search is unavailable.")

        with DDGS(timeout=10) as client:  # type: ignore[call-arg]
            search_results = client.text(query, max_results=max_results, backend="duckduckgo")

        results: list[Dict[str, str]] = []
        for entry in search_results:
            url = entry.get("href") or entry.get("url") or ""
            title = entry.get("title") or url
            content = entry.get("body") or entry.get("content") or ""
            if url and title:
                results.append(_normalized_result(title=title, url=url, content=content))
        return _structured_payload(results[:max_results], backend="duckduckgo")

    def _search_searxng(self, *, query: str, max_results: int) -> Dict[str, Any]:
        response = requests.get(
            f"{self.searxng_url}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": 1,
                "categories": "general",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

        results: list[Dict[str, str]] = []
        for entry in payload.get("results", [])[:max_results]:
            url = entry.get("url") or entry.get("link") or ""
            title = entry.get("title") or url
            content = entry.get("content") or entry.get("snippet") or ""
            if url and title:
                results.append(_normalized_result(title=title, url=url, content=content))
        return _structured_payload(results, backend="searxng")

    def _search_perplexity(
        self,
        *,
        query: str,
        max_results: int,
        loop_count: int,
    ) -> Dict[str, Any]:
        if not self.perplexity_key:
            raise RuntimeError("Perplexity search is unavailable.")

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.perplexity_key}",
            },
            json={
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "Search the web and provide factual information with sources.",
                    },
                    {"role": "user", "content": query},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", []) or ["https://perplexity.ai"]
        results = [
            _normalized_result(
                title=f"Perplexity Source {loop_count + 1}-{idx}",
                url=url,
                content=content if idx == 1 else "See main Perplexity response above.",
            )
            for idx, url in enumerate(citations[:max_results], start=1)
        ]
        return _structured_payload(results, backend="perplexity", answer=content)

    def _search_advanced(
        self,
        *,
        query: str,
        max_results: int,
        loop_count: int,
    ) -> Dict[str, Any]:
        notices: list[str] = []
        backend_order = ["searxng", "tavily", "duckduckgo", "perplexity"]
        if os.getenv("SEARCH_ENABLE_SERPAPI_FALLBACK", "").lower() in {"1", "true", "yes", "on"}:
            backend_order.insert(2, "serpapi")

        for backend in backend_order:
            if backend not in self.available_backends:
                continue
            try:
                payload = self._structured_search(
                    query=query,
                    backend=backend,
                    max_results=max_results,
                    loop_count=loop_count,
                )
            except Exception as exc:
                notices.append(f"{backend}_failed:{type(exc).__name__}")
                continue
            if payload.get("results"):
                payload["notices"] = notices + list(payload.get("notices") or [])
                return payload
            notices.append(f"{backend}_empty")

        return _structured_payload([], backend="advanced", notices=notices)

    def _positive_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return max(0, default)
        return max(0, parsed)


def search(query: str, backend: str | None = None) -> str:
    tool = SearchTool(backend=backend)
    parameters: dict[str, Any] = {"input": query}
    if backend:
        parameters["backend"] = backend
    return json.dumps(tool.run(parameters), ensure_ascii=False)


def search_tavily(query: str) -> str:
    return search(query, backend="tavily")


def search_serpapi(query: str) -> str:
    return search(query, backend="serpapi")


def search_searxng(query: str) -> str:
    return search(query, backend="searxng")


def search_hybrid(query: str) -> str:
    return search(query, backend="hybrid")
