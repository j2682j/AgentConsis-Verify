from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from utils.network_utils import normalize_text

from .config import SearchSourceCandidate
from .query import SearchQueryRequest


URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


@dataclass
class SourceAcquisitionTrace:
    """
    記錄單一 query 實際採用的來源取得方式與回退結果。

    Args:
     - query: Query Generator 產生的搜尋文字。
     - requested_source_kind: 模型指定的來源型態。
     - requested_access_mode: 模型指定的取得方式。
     - source_hint: 模型指定的平台、網站或 URL 提示。
     - actual_acquirer: 實際執行的搜尋或解析器。
     - result_count: 成功建立的來源候選數量。
     - source_ids: 對應的 SearchSourceCandidate IDs。
     - fallback_used: 是否因主要取得方式失敗而回退。
     - notices: 取得失敗或修復資訊。

    Returns:
     - SourceAcquisitionTrace: 可寫入實驗紀錄的來源取得 trace。
    """

    query: str
    requested_source_kind: str
    requested_access_mode: str
    source_hint: str = ""
    actual_acquirer: str = ""
    result_count: int = 0
    source_ids: list[str] = field(default_factory=list)
    fallback_used: bool = False
    notices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceAcquisitionRouter:
    """
    依 query 的來源需求選擇一般搜尋、直接頁面或影片證據工具。

    Args:
     - search_tool: 回傳結構化搜尋結果的 SearchTool。
     - video_tool: 可解析遠端影片 URL 的 VideoEvidenceTool。

    Returns:
     - SourceAcquisitionRouter: 統一輸出 SearchSourceCandidate 的來源路由器。
    """

    def __init__(self, *, search_tool: Any, video_tool: Any | None = None) -> None:
        self.search_tool = search_tool
        if video_tool is None:
            from ..video_evidence_tool import VideoEvidenceTool

            video_tool = VideoEvidenceTool()
        self.video_tool = video_tool

    def acquire_many(
        self,
        requests: list[SearchQueryRequest],
        *,
        question: str,
        max_results: int,
    ) -> tuple[list[SearchSourceCandidate], list[SourceAcquisitionTrace]]:
        sources: list[SearchSourceCandidate] = []
        traces: list[SourceAcquisitionTrace] = []
        acquired_direct_urls: set[str] = set()
        for query_index, request in enumerate(requests, start=1):
            direct_url = self._request_direct_url(request)
            direct_key = direct_url.casefold()
            if direct_key and direct_key in acquired_direct_urls:
                requirement = request.source_requirement
                traces.append(
                    SourceAcquisitionTrace(
                        query=request.query,
                        requested_source_kind=requirement.source_kind,
                        requested_access_mode=requirement.access_mode,
                        source_hint=requirement.source_hint,
                        actual_acquirer="duplicate_direct_source_skipped",
                        notices=["duplicate_direct_url"],
                    )
                )
                continue
            acquired, trace = self.acquire(
                request,
                question=question,
                query_id=f"Q{query_index}",
                max_results=max_results,
            )
            for source in acquired:
                source.source_id = f"S{len(sources) + 1}"
                sources.append(source)
                trace.source_ids.append(source.source_id)
            trace.result_count = len(acquired)
            traces.append(trace)
            if direct_key and acquired:
                acquired_direct_urls.add(direct_key)
        return sources, traces

    def acquire(
        self,
        request: SearchQueryRequest,
        *,
        question: str,
        query_id: str,
        max_results: int,
    ) -> tuple[list[SearchSourceCandidate], SourceAcquisitionTrace]:
        requirement = request.source_requirement
        trace = SourceAcquisitionTrace(
            query=request.query,
            requested_source_kind=requirement.source_kind,
            requested_access_mode=requirement.access_mode,
            source_hint=requirement.source_hint,
        )

        if requirement.source_kind == "video":
            return self._acquire_video(
                request,
                question=question,
                query_id=query_id,
                max_results=max_results,
                trace=trace,
            )

        direct_url = self._extract_url(
            " ".join([request.query, requirement.source_hint])
        )
        if requirement.access_mode in {"direct_fetch", "browser"} and direct_url:
            trace.actual_acquirer = (
                "playwright" if requirement.access_mode == "browser" else "direct_page"
            )
            return [
                self._direct_source(
                    request,
                    query_id=query_id,
                    url=direct_url,
                )
            ], trace

        if requirement.access_mode in {"direct_fetch", "browser"}:
            trace.fallback_used = True
            trace.notices.append("missing_direct_url_fallback_to_search")

        sources, backend, notices = self._search_sources(
            request,
            query_id=query_id,
            max_results=max_results,
        )
        trace.actual_acquirer = backend or "search"
        trace.notices.extend(notices)
        return sources, trace

    def _acquire_video(
        self,
        request: SearchQueryRequest,
        *,
        question: str,
        query_id: str,
        max_results: int,
        trace: SourceAcquisitionTrace,
    ) -> tuple[list[SearchSourceCandidate], SourceAcquisitionTrace]:
        direct_url = self._video_url(
            " ".join([request.query, request.source_requirement.source_hint])
        )
        search_sources: list[SearchSourceCandidate] = []
        if not direct_url:
            search_sources, backend, notices = self._search_sources(
                request,
                query_id=query_id,
                max_results=max_results,
            )
            trace.actual_acquirer = backend or "search"
            trace.notices.extend(notices)
            direct_url = next(
                (
                    self._video_url(source.url)
                    for source in search_sources
                    if self._video_url(source.url)
                ),
                "",
            )

        if direct_url:
            payload = self.video_tool.run(
                {
                    "url": direct_url,
                    "question": question,
                }
            )
            if bool(payload.get("ok")) and normalize_text(payload.get("output_text", "")):
                trace.actual_acquirer = "video_evidence"
                source = self._video_source(
                    request,
                    query_id=query_id,
                    url=direct_url,
                    payload=payload,
                )
                return [source], trace
            trace.fallback_used = True
            error = normalize_text(
                str(payload.get("error_message") or payload.get("error") or "video_evidence_failed")
            )
            trace.notices.append(error or "video_evidence_failed")

        if not search_sources:
            search_sources, backend, notices = self._search_sources(
                request,
                query_id=query_id,
                max_results=max_results,
            )
            trace.actual_acquirer = backend or "search"
            trace.notices.extend(notices)
            trace.fallback_used = True
        return search_sources, trace

    def _search_sources(
        self,
        request: SearchQueryRequest,
        *,
        query_id: str,
        max_results: int,
    ) -> tuple[list[SearchSourceCandidate], str, list[str]]:
        try:
            payload = self.search_tool.run(
                {
                    "input": request.query,
                    "mode": "structured",
                    "max_results": max_results,
                }
            )
        except Exception as exc:
            return [], "", [f"{type(exc).__name__}: {exc}"]

        requirement = request.source_requirement
        sources: list[SearchSourceCandidate] = []
        for rank, item in enumerate(list(payload.get("results") or []), start=1):
            if not isinstance(item, dict):
                continue
            url = normalize_text(str(item.get("url") or ""))
            if not url:
                continue
            sources.append(
                SearchSourceCandidate(
                    source_id="",
                    query_id=query_id,
                    title=normalize_text(str(item.get("title") or "")) or url,
                    url=url,
                    snippet=normalize_text(
                        str(item.get("content", item.get("snippet", "")) or "")
                    ),
                    rank=rank,
                    source_kind=requirement.source_kind,
                    access_mode=requirement.access_mode,
                    source_hint=requirement.source_hint,
                )
            )
        sources = self._prioritize_hint(sources, requirement.source_hint)
        return (
            sources,
            normalize_text(str(payload.get("backend") or "")),
            [str(item) for item in list(payload.get("notices") or [])],
        )

    def _direct_source(
        self,
        request: SearchQueryRequest,
        *,
        query_id: str,
        url: str,
    ) -> SearchSourceCandidate:
        requirement = request.source_requirement
        return SearchSourceCandidate(
            source_id="",
            query_id=query_id,
            title=requirement.source_hint or url,
            url=url,
            rank=1,
            should_fetch_full_page=True,
            source_kind=requirement.source_kind,
            access_mode=requirement.access_mode,
            source_hint=requirement.source_hint,
        )

    def _video_source(
        self,
        request: SearchQueryRequest,
        *,
        query_id: str,
        url: str,
        payload: dict[str, Any],
    ) -> SearchSourceCandidate:
        raw = payload.get("raw_result")
        raw = raw if isinstance(raw, dict) else {}
        content = normalize_text(str(payload.get("output_text") or ""))
        title = normalize_text(str(raw.get("title") or "")) or "Video evidence"
        requirement = request.source_requirement
        return SearchSourceCandidate(
            source_id="",
            query_id=query_id,
            title=title,
            url=url,
            snippet=content[:500],
            raw_content=content,
            rank=1,
            fetched=True,
            should_fetch_full_page=False,
            source_kind="video",
            access_mode=requirement.access_mode,
            source_hint=requirement.source_hint,
            filter_reasons=["source_acquirer:video_evidence"],
        )

    def _prioritize_hint(
        self,
        sources: list[SearchSourceCandidate],
        source_hint: str,
    ) -> list[SearchSourceCandidate]:
        hint = self._hint_key(source_hint)
        if not hint:
            return sources
        return sorted(
            sources,
            key=lambda source: (
                0 if hint in self._hint_key(source.url) else 1,
                source.rank,
            ),
        )

    def _video_url(self, value: str) -> str:
        extractor = getattr(self.video_tool, "extract_url", None)
        if callable(extractor):
            return normalize_text(str(extractor(value) or ""))
        return self._extract_url(value)

    def _request_direct_url(self, request: SearchQueryRequest) -> str:
        if request.source_requirement.access_mode not in {"direct_fetch", "browser"}:
            return ""
        value = " ".join(
            [request.query, request.source_requirement.source_hint]
        )
        if request.source_requirement.source_kind == "video":
            return self._video_url(value)
        return self._extract_url(value)

    def _extract_url(self, value: str) -> str:
        match = URL_RE.search(str(value or ""))
        return match.group(0).rstrip(".,)") if match else ""

    def _hint_key(self, value: str) -> str:
        text = normalize_text(value).lower()
        if not text:
            return ""
        parsed = urlparse(text if "://" in text else f"https://{text}")
        return (parsed.netloc or parsed.path).removeprefix("www.").strip("/")


__all__ = [
    "SourceAcquisitionRouter",
    "SourceAcquisitionTrace",
]
