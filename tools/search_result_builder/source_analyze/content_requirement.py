from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from utils.network_utils import normalize_text


@dataclass(frozen=True)
class ContentAcquisitionState:
    """
    分離傳輸成功、內容抽取與任務內容需求是否滿足。

    Args:
     - transport_ok: 來源是否成功連線或由專用工具讀取。
     - content_extracted: 是否取得可供後續處理的內容。
     - content_complete: 內容是否完整且未截斷。
     - requirement_met: 取得的內容是否符合 required_content。
     - failed: 來源取得是否失敗。
     - required_content: 本次要求的內容類型。
     - missing_content: 尚缺少的內容類型。
     - method: 實際使用的取得或解析方法。

    Returns:
     - ContentAcquisitionState: 可供 retrieval controller 使用的來源狀態。
    """

    transport_ok: bool
    content_extracted: bool
    content_complete: bool
    requirement_met: bool
    failed: bool
    required_content: str = "html_text"
    missing_content: list[str] = field(default_factory=list)
    method: str = ""

    @property
    def state(self) -> str:
        if self.failed:
            return "failed"
        if self.requirement_met:
            return "requirement_met"
        if self.content_extracted:
            return "content_extracted"
        if self.transport_ok:
            return "transport_ok"
        return "pending"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "state": self.state}


class ContentRequirementVerifier:
    """判斷取得內容是否符合 query 所宣告的 required_content。"""

    def verify(
        self,
        *,
        required_content: str,
        content: str,
        method: str,
        content_type: str = "",
        status_code: int = 0,
        content_complete: bool = False,
        source_kind: str = "web",
    ) -> ContentAcquisitionState:
        required = normalize_text(required_content).lower() or "html_text"
        text = normalize_text(content)
        method_key = normalize_text(method).lower()
        content_type_key = normalize_text(content_type).lower()
        transport_ok = bool(
            text
            or method_key
            or (status_code and status_code < 400)
        )
        extracted = bool(text)
        requirement_met = self._requirement_met(
            required=required,
            text=text,
            method=method_key,
            content_type=content_type_key,
            content_complete=content_complete,
            source_kind=normalize_text(source_kind).lower(),
        )
        return ContentAcquisitionState(
            transport_ok=transport_ok,
            content_extracted=extracted,
            content_complete=bool(content_complete),
            requirement_met=requirement_met,
            failed=not transport_ok or not extracted,
            required_content=required,
            missing_content=[] if requirement_met else [required],
            method=method_key,
        )

    def _requirement_met(
        self,
        *,
        required: str,
        text: str,
        method: str,
        content_type: str,
        content_complete: bool,
        source_kind: str,
    ) -> bool:
        if not text:
            return False
        if required == "html_text":
            return True
        if required == "full_page":
            return content_complete
        if required == "pdf_text":
            return "pdf" in method or "pdf" in content_type
        if required == "pdf_figure":
            return ("pdf" in method or "pdf" in content_type) and any(
                marker in text.casefold()
                for marker in ("figure", "fig.", "caption", "image")
            )
        if required == "transcript":
            return "transcript" in method or "video transcript source:" in text.casefold()
        if required == "temporal_video":
            return source_kind == "video" and any(
                marker in text.casefold()
                for marker in ("timestamp", "seconds", "frame", "[00:")
            )
        if required == "visual":
            return source_kind == "video" and any(
                marker in method or marker in text.casefold()
                for marker in ("vision", "frame", "visual")
            )
        if required == "collection_records":
            return any(
                marker in text.casefold()
                for marker in ("record type:", "tables:", "publication", "database row")
            )
        return True


__all__ = ["ContentAcquisitionState", "ContentRequirementVerifier"]
