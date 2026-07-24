from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..base import HandlerInput
from .base import AdapterResult, payload_provenance


@lru_cache(maxsize=1)
def specialized_attachment_handler_names() -> frozenset[str]:
    """Handlers that parse the attachment file themselves.

    Derived from each handler's own ``uses_specialized_attachment_parser``
    declaration rather than a hand-maintained list. A handler without an
    input adapter is reported as not attachment-bound, which silently drops
    it from the eligible-capability set before the strategy planner ever
    sees it — so a missing name here disables the handler with no error.
    Deriving the set keeps that from happening when a handler is added.

    Imported lazily: the handler registry pulls in every handler module, and
    this adapter is constructed while that package is still initializing.
    """
    from ..registry import default_deterministic_registry

    return frozenset(
        handler.name
        for handler in default_deterministic_registry().list_handlers()
        if getattr(handler, "uses_specialized_attachment_parser", False)
    )


class AttachmentFileInputAdapter:
    """Bind handlers that read the attachment file directly to its parsed path."""

    def __init__(self, handler_names: frozenset[str] | set[str] | None = None) -> None:
        self.handler_names = (
            frozenset(handler_names)
            if handler_names is not None
            else specialized_attachment_handler_names()
        )

    def adapt(self, handler_name: str, handler_input: HandlerInput) -> AdapterResult:
        del handler_name
        attachment = handler_input.attachment if isinstance(handler_input.attachment, dict) else {}
        file_path = str(attachment.get("file_path") or attachment.get("path") or "").strip()
        exists = bool(file_path and Path(file_path).is_file())
        return AdapterResult(
            status="ready" if exists else "missing_inputs",
            inputs={"file_path": file_path} if exists else {},
            missing_inputs=[] if exists else ["file_path"],
            input_provenance=(
                {
                    "source": "specialized_attachment_input",
                    "file_path": file_path,
                    "parse_status": "success",
                }
                if exists
                else payload_provenance(handler_input)
            ),
            reason="attachment_file_path" if exists else "attachment_file_path_missing",
        )


__all__ = ["AttachmentFileInputAdapter", "specialized_attachment_handler_names"]
