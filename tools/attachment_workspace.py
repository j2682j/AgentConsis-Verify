from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class PreparedAttachmentArtifact:
    """Store one task attachment's parsed representation in memory."""

    fingerprint: str
    file_path: str
    extension: str
    context: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    parsed_payload: dict[str, Any] = field(default_factory=dict)
    eligible_capabilities: list[dict[str, Any]] = field(default_factory=list)
    reader_status: str = "not_prepared"
    parse_status: str = "not_prepared"
    strategy: dict[str, Any] = field(default_factory=dict)
    handler_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttachmentReuseDecision:
    action: str
    reason: str
    information_need: str
    cached_result: dict[str, Any] | None = None


class AttachmentWorkspace:
    """Maintain task-scoped attachment artifacts and Stage1 reuse diagnostics."""

    def __init__(self, attachment: dict[str, Any] | None = None) -> None:
        self.attachment = dict(attachment or {})
        self.fingerprint = self.build_fingerprint(self.attachment)
        self._artifact: PreparedAttachmentArtifact | None = None
        self._results: dict[str, dict[str, Any]] = {}
        self._failed_needs: set[str] = set()
        self._lock = Lock()
        self.reader_execution_count = 0
        self.vision_execution_count = 0
        self.stage1_request_count = 0
        self.payload_reuse_count = 0
        self.result_cache_hit_count = 0
        self.handler_execution_count = 0
        self.blocked_duplicate_count = 0
        self.selected_handlers: list[str] = []

    @staticmethod
    def build_fingerprint(attachment: dict[str, Any] | None) -> str:
        data = dict(attachment or {})
        raw_path = str(data.get("file_path") or data.get("path") or "").strip()
        if not raw_path:
            return ""
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve()
            stat = resolved.stat()
            source = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            source = str(path.absolute())
        return hashlib.sha256(source.encode("utf-8", errors="ignore")).hexdigest()

    def seed_from_read_result(
        self,
        read_result: dict[str, Any],
        *,
        reader_executed: bool = True,
    ) -> PreparedAttachmentArtifact:
        profile = dict(read_result.get("profile") or {})
        payload = dict(read_result.get("parsed_payload") or {})
        metadata = dict(read_result.get("metadata") or {})
        artifact = PreparedAttachmentArtifact(
            fingerprint=self.fingerprint,
            file_path=str(
                metadata.get("file_path")
                or self.attachment.get("file_path")
                or self.attachment.get("path")
                or ""
            ),
            extension=str(
                metadata.get("file_type")
                or self.attachment.get("extension")
                or ""
            ).lower(),
            context=str(read_result.get("context") or ""),
            profile=profile,
            parsed_payload=payload,
            reader_status=("success" if read_result.get("used") else "failed"),
            parse_status=str(profile.get("parse_status") or "failed"),
        )
        with self._lock:
            self._artifact = artifact
            if reader_executed:
                self.reader_execution_count += 1
                reader = str(metadata.get("reader") or profile.get("reader") or "").lower()
                if any(token in reader for token in ("vision", "video", "audio", "whisper")):
                    self.vision_execution_count += 1
        return artifact

    def seed_from_strategy_result(
        self,
        strategy_result: Any,
        *,
        reader_executed: bool = True,
    ) -> PreparedAttachmentArtifact:
        profile = dict(getattr(strategy_result, "attachment_profile", {}) or {})
        payload = dict(getattr(strategy_result, "parsed_payload", {}) or {})
        metadata = dict(getattr(strategy_result, "metadata", {}) or {})
        strategy = getattr(strategy_result, "strategy", None)
        artifact = PreparedAttachmentArtifact(
            fingerprint=self.fingerprint,
            file_path=str(
                (payload.get("provenance") or {}).get("file_path")
                or self.attachment.get("file_path")
                or self.attachment.get("path")
                or ""
            ),
            extension=str(
                profile.get("extension")
                or self.attachment.get("extension")
                or ""
            ).lower(),
            context=str(getattr(strategy_result, "attachment_context", "") or ""),
            profile=profile,
            parsed_payload=payload,
            eligible_capabilities=list(metadata.get("handler_capabilities") or []),
            reader_status=str(getattr(strategy_result, "reader_status", "") or "failed"),
            parse_status=str(profile.get("parse_status") or "failed"),
            strategy=(strategy.to_dict() if hasattr(strategy, "to_dict") else {}),
            handler_result={
                "solver_context": str(getattr(strategy_result, "solver_context", "") or ""),
                "handler_status": str(getattr(strategy_result, "handler_status", "") or ""),
            },
        )
        with self._lock:
            previous = self._artifact
            self._artifact = artifact
            if reader_executed and previous is None:
                self.reader_execution_count += 1
                reader = str(profile.get("reader") or "").lower()
                if any(token in reader for token in ("vision", "video", "audio", "whisper")):
                    self.vision_execution_count += 1
        return artifact

    def artifact(self) -> PreparedAttachmentArtifact | None:
        with self._lock:
            return self._artifact

    def decide(self, information_need: str) -> AttachmentReuseDecision:
        need = str(information_need or "").strip()
        key = self.normalize_need(need)
        with self._lock:
            self.stage1_request_count += 1
            cached = self._results.get(key)
            if cached is not None:
                self.result_cache_hit_count += 1
                self.blocked_duplicate_count += 1
                result = dict(cached)
                result.update(
                    {
                        "status": "already_available",
                        "cache_hit": True,
                        "duplicate_request": True,
                    }
                )
                return AttachmentReuseDecision("reuse_cached_result", "same_information_need", need, result)
            if key in self._failed_needs:
                self.blocked_duplicate_count += 1
                return AttachmentReuseDecision("blocked_duplicate", "same_failed_request", need)
            if self._artifact is not None:
                self.payload_reuse_count += 1
                return AttachmentReuseDecision("reuse_prepared_payload", "prepared_payload_available", need)
            return AttachmentReuseDecision("reader_required", "attachment_not_prepared", need)

    def record_result(
        self,
        information_need: str,
        result: dict[str, Any],
        *,
        handler_name: str = "",
        handler_executed: bool = False,
    ) -> None:
        key = self.normalize_need(information_need)
        with self._lock:
            if result.get("evidence_valid") or result.get("ok"):
                self._results[key] = dict(result)
                self._failed_needs.discard(key)
            else:
                self._failed_needs.add(key)
            if handler_executed:
                self.handler_execution_count += 1
            if handler_name and handler_name not in self.selected_handlers:
                self.selected_handlers.append(handler_name)

    def record_tool_cache_hit(self) -> None:
        """Account for attachment requests answered before the coordinator is entered."""

        with self._lock:
            self.stage1_request_count += 1
            self.result_cache_hit_count += 1
            self.blocked_duplicate_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact
            return {
                "fingerprint": self.fingerprint,
                "prepared_available": artifact is not None,
                "reader_status": artifact.reader_status if artifact else "not_prepared",
                "parse_status": artifact.parse_status if artifact else "not_prepared",
                "available_inputs": list((artifact.profile if artifact else {}).get("available_inputs") or []),
                "eligible_handlers": [
                    str(item.get("handler_name") or "")
                    for item in list((artifact.eligible_capabilities if artifact else []) or [])
                    if isinstance(item, dict) and item.get("handler_name")
                ],
                "reader_execution_count": self.reader_execution_count,
                "vision_execution_count": self.vision_execution_count,
                "stage1_request_count": self.stage1_request_count,
                "payload_reuse_count": self.payload_reuse_count,
                "result_cache_hit_count": self.result_cache_hit_count,
                "handler_execution_count": self.handler_execution_count,
                "blocked_duplicate_count": self.blocked_duplicate_count,
                "selected_handlers": list(self.selected_handlers),
            }

    @staticmethod
    def normalize_need(value: str) -> str:
        return " ".join(str(value or "").casefold().split()) or "__default__"


__all__ = [
    "AttachmentReuseDecision",
    "AttachmentWorkspace",
    "PreparedAttachmentArtifact",
]
